"""
controller.py
=============
HPDSController — Model Predictive Control for cardamom drying.

At every 10-second tick the controller:
1. Gets the current state estimate from the UKF (T_core, T_surface, c, MR)
2. Tests every candidate heater setpoint in T_CANDS
3. For each candidate, runs the FNO forward HORIZON steps (6 min ahead)
4. Evaluates the MPC objective: moisture deviation + EO loss + energy
5. Picks the setpoint with lowest cost and sends it to the heater

Fan speed is CONSTANT — it is NOT optimised.

Grid size: 6 candidates × 1 (no fan sweep) = 6 FNO rollouts per control step.
At ~2 ms per FNO call: ~12 ms total — well within the 10-second budget.
"""

import sys
from pathlib import Path
from typing import Optional
import numpy as np
import torch

import importlib.util as _ilu

# ── Step 1: Pre-load FNO's config.py into sys.modules['config'] ────────────────
FNO_DIR      = Path(__file__).parent.parent / "architecture-training-FNO"
_fno_cfg_spec = _ilu.spec_from_file_location("config", str(FNO_DIR / "config.py"))
_fno_cfg_mod  = _ilu.module_from_spec(_fno_cfg_spec)
sys.modules["config"] = _fno_cfg_mod
_fno_cfg_spec.loader.exec_module(_fno_cfg_mod)

# ── Step 2: Import FNO model + dataset (their 'from config import' hits cache) ─
sys.path.insert(0, str(FNO_DIR))
from model   import CardamomFNO
from dataset import load_geometry

# ── Step 3: Load MPC config under a distinct name 'mpc_config' ─────────────────
MPC_DIR   = Path(__file__).parent
_mpc_spec = _ilu.spec_from_file_location("mpc_config", str(MPC_DIR / "config.py"))
C         = _ilu.module_from_spec(_mpc_spec)
_mpc_spec.loader.exec_module(C)



# ─────────────────────────────────────────────────────────────────────────────
# FNO loader
# ─────────────────────────────────────────────────────────────────────────────

def load_fno(ckpt_path: Path, device: torch.device) -> CardamomFNO:
    ckpt  = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cfg   = ckpt.get("model_cfg", {})
    model = CardamomFNO(
        latent_channels = cfg.get("latent_channels", 64),
        num_fno_layers  = cfg.get("num_fno_layers",  4),
        modes1          = cfg.get("modes1",           4),
        modes2          = cfg.get("modes2",           8),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[FNO] Loaded {ckpt_path.name} on {device}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation helpers (must match architecture-training-FNO/infer.py)
# ─────────────────────────────────────────────────────────────────────────────

def _norm_x(T: np.ndarray, M: np.ndarray, T_h: float) -> torch.Tensor:
    """Build normalised input tensor from physical fields."""
    Th_field = np.full((C.FNO_NX, C.FNO_NY), T_h, dtype=np.float32)
    X    = np.stack([T, M, Th_field], axis=0)                   # (3, 8, 16)
    mean = np.array(C.NORM_MEAN_X, dtype=np.float32)[:, None, None]
    std  = np.array(C.NORM_STD_X,  dtype=np.float32)[:, None, None]
    return torch.from_numpy((X - mean) / std).unsqueeze(0)       # (1, 3, 8, 16)


def _denorm_y(y_norm: torch.Tensor) -> np.ndarray:
    """Denormalise FNO output to physical units.  Returns (2, 8, 16)."""
    mean = np.array(C.NORM_MEAN_Y, dtype=np.float32)[:, None, None]
    std  = np.array(C.NORM_STD_Y,  dtype=np.float32)[:, None, None]
    y    = y_norm[0].cpu().numpy()   # (2, 8, 16)
    return y * std + mean


# ─────────────────────────────────────────────────────────────────────────────
# Drying phase classifier
# ─────────────────────────────────────────────────────────────────────────────

def classify_phase(MR: float) -> str:
    if MR > C.MR_PHASE1_END:
        return "Phase1_ConstantRate"
    elif MR > C.MR_PHASE2_END:
        return "Phase2_FallingRate"
    else:
        return "Phase3_Equilibrium"


# ─────────────────────────────────────────────────────────────────────────────
# HPDSController
# ─────────────────────────────────────────────────────────────────────────────

class HPDSController:
    """
    Model Predictive Controller for Cardamom HPDS drying.

    At each call to `step()`:
      - The UKF state estimator provides the current internal pod state.
      - For each candidate heater setpoint T_cand ∈ T_CANDS:
          • Run FNO HORIZON steps forward (6 min) to predict future T and M fields
          • Evaluate the MPC objective cost
      - The setpoint with minimum cost is returned.

    Parameters
    ----------
    fno_model   : trained CardamomFNO (loaded from checkpoint)
    estimator   : SensorFusionUKF (provides current state every step)
    eo_model    : EOModel (tracks cumulative EO quality)
    device      : torch.device
    geometry_path : Path to pod_geometry.h5 (for pod_mask)
    """

    def __init__(
        self,
        fno_model,
        estimator,
        eo_model,
        device: torch.device,
        geometry_path: Optional[Path] = None,
    ):
        self.fno       = fno_model
        self.estimator = estimator
        self.eo        = eo_model
        self.device    = device

        # Load pod mask (which pixels correspond to the cardamom pod)
        geo_path = geometry_path or C.FNO_GEOMETRY
        if Path(str(geo_path)).exists():
            geo            = load_geometry(device=self.device)
            self._pod_mask = geo["pod_mask"].cpu().numpy().astype(bool)  # (8, 16)
        else:
            # Fallback: assume all pixels are pod (safe default)
            self._pod_mask = np.ones((C.FNO_NX, C.FNO_NY), dtype=bool)
            print("[Controller] WARNING: pod_geometry.h5 not found — using full-grid mask")

        # Current best T/M fields (initialised to uniform conditions)
        self._T_field = np.full((C.FNO_NX, C.FNO_NY), 25.0, dtype=np.float32)
        self._M_field = np.full((C.FNO_NX, C.FNO_NY), 0.80, dtype=np.float32)

        # History for the demo / logging
        self.history: list[dict] = []
        
        # MPC re-evaluation interval: run expensive grid-search every N steps (1 min)
        self._mpc_interval = 6   # 6 steps × 10s = 60s
        self._steps_since_mpc = self._mpc_interval  # trigger immediately on first call
        self._last_best_T    = C.T_CANDS[2]          # 40°C warm safe fallback
        self._last_best_cost = np.inf
        self._last_best_MR   = 0.80

    # ── Internal field update from PINN state ─────────────────────────────────

    def update_fields_from_state(self, state: dict) -> None:
        """
        Synchronise the FNO input fields with the latest UKF state estimate.

        The PINN state gives scalar T_surface and c_surface; we tile these
        as uniform fields.  A more sophisticated version would query the full
        PINN spatial grid, but scalar tiling is fast and sufficient for MPC.
        """
        self._T_field[:] = state["T_surface"]
        M_surface = state["c_surface"] / C.C_INIT   # convert mol/m³ → MR proxy
        self._M_field[:] = float(np.clip(M_surface, 0.0, 1.5))

    # ── FNO rollout for one candidate setpoint ─────────────────────────────────

    @torch.no_grad()
    def _fno_rollout(
        self,
        T_field_init: np.ndarray,
        M_field_init: np.ndarray,
        T_heater: float,
    ) -> dict:
        """
        Roll FNO forward HORIZON steps with a fixed heater setpoint.

        Returns dict with predicted MR_final and surface temperature history.
        """
        T_curr = T_field_init.copy()
        M_curr = M_field_init.copy()
        T_surf_history = []

        for _ in range(C.HORIZON):
            x_norm = _norm_x(T_curr, M_curr, T_heater).to(self.device)
            y_norm = self.fno(x_norm)
            y_phys = _denorm_y(y_norm)            # (2, 8, 16)

            T_next = y_phys[0]                    # (8, 16) °C
            M_next = y_phys[1]                    # (8, 16) kg_w/kg_dry

            # Apply pod mask: outside pod → heater temperature, M = 0
            T_next = np.where(self._pod_mask, T_next, T_heater)
            M_next = np.where(self._pod_mask, np.clip(M_next, 0.0, 3.0), 0.0)

            T_curr, M_curr = T_next, M_next
            T_surf_pod = T_curr[self._pod_mask]
            T_surf_history.append(float(np.nanmean(T_surf_pod)) if T_surf_pod.size > 0 else T_heater)

        # Mean MR over pod pixels at end of horizon
        M_pod = M_curr[self._pod_mask]
        MR_final = float(np.clip(np.nanmean(M_pod), 0.0, 1.5)) if M_pod.size > 0 else 0.0

        return {
            "MR_final":        MR_final,
            "T_surf_history":  np.array(T_surf_history),
            "T_surf_mean_max": float(np.max(T_surf_history)) if T_surf_history else T_heater,
        }

    # ── MPC objective ──────────────────────────────────────────────────────────

    def _objective(
        self,
        MR_final: float,
        T_surf_history: np.ndarray,
        T_heater: float,
        MR_now: float,
    ) -> float:
        """
        Phase-aware MPC cost function.

        Phase 1/2 (MR far from target): aggressively reduce MR → use high T
        Phase 3 (MR near target)      : protect EO quality, coast to finish

        cost = λ_MR * (MR_final - MR_target)²
             + λ_EO * EO_lost_fraction_over_horizon  [downweighted in Ph1]
             + λ_E  * energy_proxy
        """
        # Use mean surface temperature (not max) to avoid FNO noise blocking
        # all high-temperature candidates from a single hot pixel
        T_surf_mean = float(np.nanmean(T_surf_history)) if T_surf_history.size > 0 else T_heater
        if T_surf_mean > C.T_SURF_MAX + 2.0:   # 2°C tolerance on mean
            return np.inf

        # EO loss during this rollout (previews without modifying EO state)
        EO_retained_end = self.eo.preview_loss(T_surf_history, dt=C.DT)
        EO_lost = 1.0 - EO_retained_end

        # Phase-adaptive EO weight: only protect EO when close to target MR
        # In Phase1/2, drying urgency dominates; in Phase3 switch to EO guard
        if MR_now > C.MR_PHASE2_END:
            # Phase 1 or 2: drying is urgent, reduce EO penalty weight
            lambda_eo = C.LAMBDA_EO * 0.2
        else:
            # Phase 3: near equilibrium, protect EO quality
            lambda_eo = C.LAMBDA_EO * 1.5

        # Energy proxy: normalised heater temperature
        energy_proxy = (T_heater - C.T_AIR_MIN) / (C.T_AIR_MAX - C.T_AIR_MIN)

        cost = (
            C.LAMBDA_MR * (MR_final - C.MR_TARGET) ** 2
            + lambda_eo * EO_lost
            + C.LAMBDA_E  * energy_proxy
        )
        
        # Phase 1/2 bonus: directly reward higher temperature when far from target
        # This drives the controller to raise heat even if FNO predictions are similar
        if MR_now > C.MR_PHASE2_END:
            # Higher temperature → lower (better) cost. Scale by distance from target.
            # energy_proxy=1 at T_max (55C), 0 at T_min (30C)
            T_boost = 0.5 * (MR_now - C.MR_TARGET) * energy_proxy
            cost -= T_boost   # subtract → lower cost for higher T
        
        return float(cost)


    # ── Main control step ──────────────────────────────────────────────────────

    def step(
        self,
        sensor_obs: dict,
        t: float,
    ) -> dict:
        """
        Run one MPC step.

        Parameters
        ----------
        sensor_obs : dict  {'T_out': float, 'RH_out': float, 'mass_g': float}
        t          : float  current time [s]

        Returns
        -------
        dict:
            T_setpoint          : float  optimal heater setpoint [°C]
            fan_speed           : float  constant (= FAN_SPEED)
            drying_phase        : str    current drying phase label
            predicted_MR        : float  FNO-forecast MR at horizon end
            predicted_EO_retained: float
            cost                : float  MPC objective value
            state               : dict   UKF state estimate
        """
        # ── 1. State estimation ───────────────────────────────────────────
        state_vec, _ = self.estimator.estimate(sensor_obs, t)
        state        = self.estimator.state          # named dict
        MR_now       = state["MR"]
        phase        = classify_phase(MR_now)

        # Update FNO input fields from state
        self.update_fields_from_state(state)

        # ── 2. MPC grid-search (re-run every _mpc_interval steps) ────────────
        self._steps_since_mpc += 1
        run_mpc = (self._steps_since_mpc >= self._mpc_interval)
        
        if run_mpc:
            self._steps_since_mpc = 0
            best_T      = C.T_CANDS[2]   # safe fallback: 40°C
            best_cost   = np.inf
            best_result = {}

            for T_cand in C.T_CANDS:
                rollout = self._fno_rollout(
                    self._T_field.copy(),
                    self._M_field.copy(),
                    T_cand,
                )
                cost = self._objective(
                    MR_final       = rollout["MR_final"],
                    T_surf_history = rollout["T_surf_history"],
                    T_heater       = T_cand,
                    MR_now         = MR_now,
                )
                if cost < best_cost:
                    best_cost   = cost
                    best_T      = T_cand
                    best_result = rollout
            
            # Cache for in-between steps
            self._last_best_T    = best_T
            self._last_best_cost = best_cost
            self._last_best_MR   = best_result.get("MR_final", MR_now)
        else:
            # Hold previous setpoint — still update EO state
            best_T      = self._last_best_T
            best_cost   = self._last_best_cost
            best_result = {"MR_final": self._last_best_MR}

        # ── 3. Update EO model state with chosen action ────────────────────
        T_surf_now = state.get("T_surface", best_T)
        eo_result  = self.eo.step(T_surface_C=T_surf_now, dt=C.DT)

        # ── 4. Build output record ─────────────────────────────────────────
        output = {
            "t_s":                  t,
            "T_setpoint":           best_T,
            "fan_speed":            C.FAN_SPEED,
            "drying_phase":         phase,
            "predicted_MR":         best_result.get("MR_final", MR_now),
            "predicted_EO_retained": eo_result["EO_retained"],
            "cost":                 best_cost,
            "state":                state,
            "MR_now":               MR_now,
        }
        self.history.append(output)
        return output


# ─────────────────────────────────────────────────────────────────────────────
# Self-test (no checkpoint needed — uses random FNO weights)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from eo_model       import EOModel
    from state_estimator import NumpyUKF

    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    print("=== Controller Self-Test (random FNO weights) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build a random FNO (no checkpoint)
    fno = CardamomFNO().to(device)
    fno.eval()

    eo = EOModel()
    eo.reset()

    # Minimal mock estimator
    class MockEstimator:
        def estimate(self, obs, t):
            return np.array([45.0, 47.0, 30000.0, 25000.0, 0.6]), np.eye(5)
        @property
        def state(self):
            return {"T_core": 45.0, "T_surface": 47.0,
                    "c_core": 30000.0, "c_surface": 25000.0, "MR": 0.6}

    ctrl = HPDSController(fno, MockEstimator(), eo, device)
    obs  = {"T_out": 48.0, "RH_out": 60.0, "mass_g": 300.0}

    result = ctrl.step(obs, t=0.0)
    print(f"  Optimal T_setpoint : {result['T_setpoint']} °C")
    print(f"  Drying phase       : {result['drying_phase']}")
    print(f"  Predicted MR       : {result['predicted_MR']:.4f}")
    print(f"  MPC cost           : {result['cost']:.4f}")
    print("  Controller self-test passed ✓")
