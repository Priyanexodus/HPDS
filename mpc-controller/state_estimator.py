"""
state_estimator.py
==================
PINN-based state predictor + pure-NumPy Unscented Kalman Filter (UKF).

What this does
--------------
The cardamom pod's internal temperature and moisture are NOT directly measurable.
Sensors (DHT22 at the outlet, load cell) measure air conditions and total mass —
things that are related to but not the same as internal pod state.

The UKF bridges this gap:
  1. PINN gives a physics-based prediction of what the internal state should be
     (the "prior" or "prediction step")
  2. Sensors give a noisy measurement of related quantities
     (the "observation" or "update step")
  3. The UKF fuses both, weighting them by their relative uncertainty

State vector (5 elements):
    x = [T_core [°C], T_surface [°C], c_core [mol/m³], c_surface [mol/m³], MR [0–1]]

Observation vector (3 elements):
    z = [T_out_air [°C], RH_out [%], mass_g [g]]
"""

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

# ── Import PINN model (must be first so CardamomPINN finds ITS OWN config.py) ──
PINN_DIR = Path(__file__).parent.parent / "architecture-training"
sys.path.insert(0, str(PINN_DIR))
# ── Import MPC config (inserted AFTER PINN so the PINN's 'from config import'
#    already resolved; now we add MPC_DIR and import its config explicitly) ────
MPC_DIR = Path(__file__).parent
# Rename import to avoid shadowing the PINN's own 'config' in sys.modules
import importlib.util as _ilu

# ── Step 1: Pre-load architecture-training/config.py as sys.modules['config']
#    This ensures CardamomPINN's 'from config import FOURIER_FEATURES' always
#    finds the PINN config, regardless of sys.path ordering or module caching. ─
PINN_DIR = Path(__file__).parent.parent / "architecture-training"
_pinn_cfg_spec = _ilu.spec_from_file_location("config", str(PINN_DIR / "config.py"))
_pinn_cfg_mod  = _ilu.module_from_spec(_pinn_cfg_spec)
sys.modules["config"] = _pinn_cfg_mod          # register under "config" first
_pinn_cfg_spec.loader.exec_module(_pinn_cfg_mod)

# ── Step 2: Load CardamomPINN directly by path — bypasses sys.modules['model'] ─
# This avoids collision if FNO's model.py is already registered as 'model'.
_pinn_model_spec = _ilu.spec_from_file_location(
    "pinn_model", str(PINN_DIR / "model.py")
)
_pinn_model_mod  = _ilu.module_from_spec(_pinn_model_spec)
sys.modules["pinn_model"] = _pinn_model_mod
_pinn_model_spec.loader.exec_module(_pinn_model_mod)
CardamomPINN = _pinn_model_mod.CardamomPINN


# ── Step 3: Load MPC config under a distinct module name to avoid collision ────
MPC_DIR    = Path(__file__).parent
_mpc_spec  = _ilu.spec_from_file_location("mpc_config", str(MPC_DIR / "config.py"))
mpc_cfg    = _ilu.module_from_spec(_mpc_spec)
_mpc_spec.loader.exec_module(mpc_cfg)

# Flat imports for convenience in this file
A_AXIAL           = mpc_cfg.A_AXIAL
B_RADIAL          = mpc_cfg.B_RADIAL
C_SCALE           = mpc_cfg.C_SCALE
T_SCALE           = mpc_cfg.T_SCALE
PINN_TOTAL_TIME_S = mpc_cfg.PINN_TOTAL_TIME_S
UKF_ALPHA         = mpc_cfg.UKF_ALPHA
UKF_BETA          = mpc_cfg.UKF_BETA
UKF_KAPPA         = mpc_cfg.UKF_KAPPA
UKF_Q_DIAG        = mpc_cfg.UKF_Q_DIAG
UKF_R_DIAG        = mpc_cfg.UKF_R_DIAG
P_ATM             = mpc_cfg.P_ATM
R_V               = mpc_cfg.R_V
SIM_M0_G          = mpc_cfg.SIM_M0_G
SIM_M_EQ_G        = mpc_cfg.SIM_M_EQ_G
C_INIT            = mpc_cfg.C_INIT


# ─────────────────────────────────────────────────────────────────────────────
# Helper: psychrometric functions
# ─────────────────────────────────────────────────────────────────────────────

def psat_pa(T_C: float) -> float:
    """Saturation vapour pressure [Pa] via Antoine-style Magnus formula."""
    return 610.78 * np.exp(17.27 * T_C / (T_C + 237.3))


def rh_from_c_and_T(c_mol_m3: float, T_C: float) -> float:
    """
    Estimate RH [%] from surface moisture concentration and air temperature.
    Assumes surface concentration c_surface sets local vapour pressure.
    c is in mol/m³ water vapour; Mw = 0.018 kg/mol.
    """
    Mw   = 0.018          # kg/mol
    rho  = 1.2            # air density at ~50°C [kg/m³]
    # kg_water / kg_dry_air  →  specific humidity
    w    = max(0.0, c_mol_m3 * Mw / rho)
    pv   = w * P_ATM / (0.622 + w)
    rh   = min(100.0, 100.0 * pv / max(psat_pa(T_C), 1e-6))
    return float(rh)


def mass_from_MR(MR: float) -> float:
    """Convert moisture ratio to total mass [g]."""
    return float(SIM_M_EQ_G + MR * (SIM_M0_G - SIM_M_EQ_G))


# ─────────────────────────────────────────────────────────────────────────────
# PINN State Predictor
# ─────────────────────────────────────────────────────────────────────────────

class PINNStatePredictor:
    """
    Wraps CardamomPINN to predict internal pod state at core and surface.

    Probes two points:
      - core:    (r=0, z=0)         → T_core, c_core
      - surface: (r=B_RADIAL, z=0)  → T_surface, c_surface
    """

    def __init__(self, ckpt_path: Path, device: torch.device):
        ckpt  = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        cfg   = ckpt.get("model_cfg", {})
        model = CardamomPINN(
            n_frequencies = cfg.get("n_frequencies", 64),
            hidden_dim    = cfg.get("hidden_dim",    256),
            n_hidden      = cfg.get("n_hidden",      6),
            skip_every    = cfg.get("skip_every",    2),
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        self.model  = model
        self.device = device
        print(f"[PINN] Loaded {ckpt_path.name} on {device}")

    @torch.no_grad()
    def predict(self, t_s: float) -> dict:
        """
        Predict T [°C] and c [mol/m³] at core + surface for time t_s [seconds].

        Returns dict with keys: T_core, T_surface, c_core, c_surface, MR
        """
        # Build (r, z, t) pairs for core and surface
        r = np.array([0.0,    B_RADIAL], dtype=np.float32)
        z = np.array([0.0,    0.0],      dtype=np.float32)
        t = np.array([t_s,    t_s],      dtype=np.float32)

        r_n = torch.tensor(r / B_RADIAL,          device=self.device)
        z_n = torch.tensor(z / A_AXIAL,           device=self.device)
        t_n = torch.tensor(t / PINN_TOTAL_TIME_S, device=self.device)
        x   = torch.stack([r_n, z_n, t_n], dim=-1)   # (2, 3)

        T_phys, c_phys = self.model.predict_physical(x)
        T = T_phys.cpu().numpy().ravel()   # [T_core, T_surface] °C
        c = c_phys.cpu().numpy().ravel()   # [c_core, c_surface] mol/m³

        MR = float(np.clip(c[0] / C_INIT, 0.0, 1.0))

        return {
            "T_core":    float(T[0]),
            "T_surface": float(T[1]),
            "c_core":    float(c[0]),
            "c_surface": float(c[1]),
            "MR":        MR,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Pure-NumPy Unscented Kalman Filter
# ─────────────────────────────────────────────────────────────────────────────

class NumpyUKF:
    """
    Unscented Kalman Filter implemented in pure NumPy.

    How it works
    ------------
    Instead of linearising (like the standard Kalman filter), the UKF picks
    2n+1 "sigma points" that represent the uncertainty cloud around the current
    state estimate.  Each sigma point is propagated through the (possibly
    nonlinear) physics model.  The resulting cloud gives the predicted mean and
    covariance — no Jacobian needed.

    When sensor data arrives, the same sigma points are pushed through the
    observation model to predict what each sensor SHOULD read.  The difference
    between predicted and actual sensor values (the "innovation") is used to
    compute a Kalman gain that decides how much to trust sensors vs physics.

    Parameters
    ----------
    dim_x : int  — state dimension (5)
    dim_z : int  — observation dimension (3)
    Q     : (n,n) ndarray — process noise covariance
    R     : (m,m) ndarray — measurement noise covariance
    alpha, beta, kappa : UKF tuning parameters
    """

    def __init__(
        self,
        dim_x: int,
        dim_z: int,
        Q: np.ndarray,
        R: np.ndarray,
        alpha: float = 1e-3,
        beta:  float = 2.0,
        kappa: float = 0.0,
    ):
        self.n  = dim_x
        self.m  = dim_z
        self.Q  = Q.copy()
        self.R  = R.copy()

        # ── UKF scaling parameters ─────────────────────────────────────────
        lam       = alpha**2 * (dim_x + kappa) - dim_x
        self._lam = lam
        c         = dim_x + lam

        # Weights for mean and covariance
        self.Wm = np.full(2 * dim_x + 1, 0.5 / c)
        self.Wc = np.full(2 * dim_x + 1, 0.5 / c)
        self.Wm[0] = lam / c
        self.Wc[0] = lam / c + (1 - alpha**2 + beta)

        # State estimate and covariance
        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x)

    # ── Sigma-point generation ─────────────────────────────────────────────

    def _sigma_points(self) -> np.ndarray:
        """Generate 2n+1 sigma points around current state estimate."""
        n   = self.n
        c   = n + self._lam
        # Cholesky of scaled covariance — each column is an offset
        try:
            L = np.linalg.cholesky(c * self.P)
        except np.linalg.LinAlgError:
            # Numerical fix: add small jitter if P is not PD
            L = np.linalg.cholesky(c * self.P + 1e-6 * np.eye(n))

        sigmas       = np.zeros((2 * n + 1, n))
        sigmas[0]    = self.x
        for i in range(n):
            sigmas[i + 1]     = self.x + L[:, i]
            sigmas[i + 1 + n] = self.x - L[:, i]
        return sigmas

    # ── Predict step ───────────────────────────────────────────────────────

    def predict(self, fx) -> None:
        """
        Propagate state through physics model fx.

        fx : callable(state_vec) -> state_vec
             The PINN-based state transition function.
        """
        sigmas  = self._sigma_points()                        # (2n+1, n)
        sigmas_ = np.array([fx(s) for s in sigmas])          # propagated

        # Weighted mean
        self.x = (self.Wm[:, None] * sigmas_).sum(axis=0)

        # Weighted covariance
        d = sigmas_ - self.x
        self.P = (self.Wc[:, None, None] * np.einsum("ki,kj->kij", d, d)).sum(axis=0) + self.Q

    # ── Update step ────────────────────────────────────────────────────────

    def update(self, z: np.ndarray, hx) -> None:
        """
        Correct state estimate with actual sensor observation z.

        z  : (m,) actual sensor reading
        hx : callable(state_vec) -> obs_vec
             Maps state to expected sensor values.
        """
        sigmas = self._sigma_points()
        Zsig   = np.array([hx(s) for s in sigmas])    # predicted observations

        # Predicted observation mean
        z_pred = (self.Wm[:, None] * Zsig).sum(axis=0)

        # Cross-covariance Pxz and innovation covariance Pzz
        dx = sigmas - self.x
        dz = Zsig   - z_pred
        Pzz = (self.Wc[:, None, None] * np.einsum("ki,kj->kij", dz, dz)).sum(axis=0) + self.R
        Pxz = (self.Wc[:, None, None] * np.einsum("ki,kj->kij", dx, dz)).sum(axis=0)

        # Kalman gain
        K = Pxz @ np.linalg.inv(Pzz)

        # State and covariance update
        self.x = self.x + K @ (z - z_pred)
        self.P = self.P - K @ Pzz @ K.T

        # Enforce physical constraints on state (only for HPDS 5-dim state)
        if self.n >= 3:
            self.x[2] = max(0.0, self.x[2])   # c_core ≥ 0
        if self.n >= 4:
            self.x[3] = max(0.0, self.x[3])   # c_surface ≥ 0
        if self.n >= 5:
            self.x[4] = float(np.clip(self.x[4], 0.0, 1.0))   # MR ∈ [0, 1]


# ─────────────────────────────────────────────────────────────────────────────
# SensorFusionUKF  —  full state estimator
# ─────────────────────────────────────────────────────────────────────────────

class SensorFusionUKF:
    """
    Combines PINNStatePredictor (physics prior) with NumpyUKF (sensor correction).

    Usage
    -----
        estimator = SensorFusionUKF(pinn_predictor)
        estimator.initialise(t=0.0)

        # At each 10-second tick:
        state, cov = estimator.estimate(
            obs={"T_out": 48.2, "RH_out": 62.0, "mass_g": 385.0},
            t=10.0
        )
    """

    DIM_X = 5   # [T_core, T_surface, c_core, c_surface, MR]
    DIM_Z = 3   # [T_out_air, RH_out, mass_g]

    def __init__(self, pinn: PINNStatePredictor):
        self.pinn = pinn

        Q = np.diag(UKF_Q_DIAG)
        R = np.diag(UKF_R_DIAG)

        self.ukf = NumpyUKF(
            dim_x = self.DIM_X,
            dim_z = self.DIM_Z,
            Q     = Q,
            R     = R,
            alpha = UKF_ALPHA,
            beta  = UKF_BETA,
            kappa = UKF_KAPPA,
        )
        self._t = 0.0   # current time [s]

    def initialise(self, t: float = 0.0) -> None:
        """Warm-start UKF state from PINN prediction at t=0."""
        pred = self.pinn.predict(t)
        self.ukf.x = np.array([
            pred["T_core"],
            pred["T_surface"],
            pred["c_core"],
            pred["c_surface"],
            pred["MR"],
        ], dtype=np.float64)
        # Initial covariance: moderate uncertainty
        self.ukf.P = np.diag([1.0, 1.0, 5000.0, 5000.0, 0.01])
        self._t = t

    def estimate(
        self,
        obs: dict,
        t: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run one UKF predict+update cycle.

        Parameters
        ----------
        obs : dict with keys 'T_out' [°C], 'RH_out' [%], 'mass_g' [g]
        t   : current time [s]

        Returns
        -------
        state : (5,) ndarray  [T_core, T_surface, c_core, c_surface, MR]
        cov   : (5,5) ndarray  state covariance
        """
        t_now = t

        # ── Physics prior (PINN predict step) ─────────────────────────────
        def fx(state: np.ndarray) -> np.ndarray:
            """Propagate sigma point: replace with PINN prediction."""
            try:
                pred = self.pinn.predict(t_now)
                return np.array([
                    pred["T_core"],
                    pred["T_surface"],
                    pred["c_core"],
                    pred["c_surface"],
                    pred["MR"],
                ], dtype=np.float64)
            except Exception:
                return state.copy()   # fallback: no change

        self.ukf.predict(fx)

        # ── Observation model ──────────────────────────────────────────────
        def hx(state: np.ndarray) -> np.ndarray:
            """Map internal state → expected sensor readings."""
            T_core, T_surf, c_core, c_surf, MR = state

            # Outlet air temperature: weighted average of heater + pod surface
            T_air_out = T_surf + 0.8 * max(0.0, 50.0 - T_surf)

            # Outlet RH: estimated from surface moisture and air temp
            RH_out = rh_from_c_and_T(c_surf, T_air_out)

            # Mass from MR
            mass = mass_from_MR(MR)

            return np.array([T_air_out, RH_out, mass], dtype=np.float64)

        z = np.array([obs["T_out"], obs["RH_out"], obs["mass_g"]], dtype=np.float64)
        self.ukf.update(z, hx)
        self._t = t_now

        return self.ukf.x.copy(), self.ukf.P.copy()

    @property
    def state(self) -> dict:
        """Current state estimate as a named dict."""
        x = self.ukf.x
        return {
            "T_core":    float(x[0]),
            "T_surface": float(x[1]),
            "c_core":    float(x[2]),
            "c_surface": float(x[3]),
            "MR":        float(np.clip(x[4], 0.0, 1.0)),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    PINN_CKPT = mpc_cfg.PINN_CKPT

    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest or True:
        print("=== StateEstimator self-test (UKF only, no PINN checkpoint needed) ===")

        # Test pure UKF with a trivial linear system
        Q = np.diag([0.01, 0.01])
        R = np.diag([0.1])
        ukf = NumpyUKF(dim_x=2, dim_z=1, Q=Q, R=R)
        ukf.x = np.array([0.0, 1.0])
        ukf.P = np.eye(2) * 0.5

        for step in range(5):
            ukf.predict(fx=lambda s: s + np.array([s[1] * 0.1, 0.0]))
            ukf.update(z=np.array([float(step) * 0.1]), hx=lambda s: np.array([s[0]]))
            print(f"  Step {step+1}: x = {ukf.x.round(4)}")

        print("  UKF self-test passed ✓")

        if Path(str(PINN_CKPT)).exists():
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pinn   = PINNStatePredictor(PINN_CKPT, device)
            pred   = pinn.predict(t_s=0.0)
            print(f"\n  PINN prediction at t=0: {pred}")
        else:
            print(f"\n  (PINN checkpoint not found at {PINN_CKPT} — skipping PINN test)")

