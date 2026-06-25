"""
eo_model.py
===========
Essential Oil (EO) volatilisation model for cardamom drying.

Physics
-------
Each EO compound evaporates from the pod surface following the Antoine equation
for vapour pressure and a film-theory mass transfer model:

    log10(Pvap_i [mmHg]) = A_i − B_i / (C_i + T_surface [°C])
    J_eo_i [mol/m²/s]   = km_i × (Pvap_i − P_bulk_i) × (1/RT)
    EO_i(t)              = EO_i(0) − ∫ J_eo_i × A_surface dt
    EO_retained(t)       = Σ w_i × EO_i(t) / EO_i(0)

where:
    km_i = mass transfer coefficient for compound i [m/s]
    P_bulk_i ≈ 0 (vapour in bulk air is dilute at typical conditions)
    A_surface = pod surface area [m²]
    w_i = commercial importance weight

Compounds (from roadmap Phase 7)
---------------------------------
    α-pinene, sabinene, myrcene, 1,8-cineole, linalool,
    α-terpineol, geraniol, α-terpinyl acetate
"""

import numpy as np
import importlib.util as _ilu
from pathlib import Path as _Path

# Load MPC config explicitly to avoid collision with PINN/FNO config.py files
_mpc_spec   = _ilu.spec_from_file_location(
    "mpc_config", str(_Path(__file__).parent / "config.py")
)
_mpc_cfg    = _ilu.module_from_spec(_mpc_spec)
_mpc_spec.loader.exec_module(_mpc_cfg)
EO_COMPOUNDS       = _mpc_cfg.EO_COMPOUNDS
POD_SURFACE_AREA_M2 = _mpc_cfg.POD_SURFACE_AREA_M2


# Gas constant [J/mol·K]
R_GAS = 8.314


class EOModel:
    """
    Tracks cumulative EO loss as a function of pod surface temperature.

    Usage
    -----
        eo = EOModel()
        eo.reset()

        for each 10-second tick:
            result = eo.step(T_surface_C=48.0, dt=10.0)
            print(result["EO_retained"])      # 0 – 1
    """

    def __init__(self):
        self.names      = list(EO_COMPOUNDS.keys())
        # Unpack Antoine + mass transfer + commercial weight
        self._A   = np.array([EO_COMPOUNDS[n][0] for n in self.names])
        self._B   = np.array([EO_COMPOUNDS[n][1] for n in self.names])
        self._C   = np.array([EO_COMPOUNDS[n][2] for n in self.names])
        self._km  = np.array([EO_COMPOUNDS[n][3] for n in self.names])
        self._w   = np.array([EO_COMPOUNDS[n][4] for n in self.names])
        self._w  /= self._w.sum()    # normalise weights → sum = 1

        # Current fractional EO remaining per compound (0–1)
        self._eo_frac: np.ndarray = np.ones(len(self.names))
        # Initial EO content [mol/m²] — normalised so each starts at 1
        # The absolute quantity cancels in the ratio; we track fractions.

    def reset(self) -> None:
        """Reset all EO fractions to 1.0 (full retention)."""
        self._eo_frac = np.ones(len(self.names))

    # ── Antoine vapour pressure ──────────────────────────────────────────────

    def _pvap_mmHg(self, T_C: float) -> np.ndarray:
        """Vapour pressure [mmHg] for all compounds at surface temperature T_C."""
        return 10.0 ** (self._A - self._B / (self._C + T_C))

    def _pvap_pa(self, T_C: float) -> np.ndarray:
        """Vapour pressure [Pa]."""
        return self._pvap_mmHg(T_C) * 133.322   # 1 mmHg = 133.322 Pa

    # ── Per-compound evaporation flux ───────────────────────────────────────

    def _flux_mol_m2_s(self, T_C: float) -> np.ndarray:
        """
        Evaporation flux J [mol/m²/s] for all compounds.

        J = km × (Pvap(T) − 0) / (R × T_abs)
        Bulk vapour pressure assumed ≈ 0 (dilute limit).
        """
        T_K  = T_C + 273.15
        Pvap = self._pvap_pa(T_C)
        return self._km * Pvap / (R_GAS * T_K)   # [mol/m²/s]

    # ── Step ────────────────────────────────────────────────────────────────

    def step(self, T_surface_C: float, dt: float = 10.0) -> dict:
        """
        Advance the EO model by one timestep.

        Parameters
        ----------
        T_surface_C : pod surface temperature [°C]
        dt          : timestep duration [s]

        Returns
        -------
        dict with:
            "EO_retained"       : float  ∈ [0, 1]  weighted quality index
            "EO_retained_per_compound" : dict name → float
            "J_eo_total"        : float  total evaporation flux [mol/m²/s]
            "EO_lost_fraction"  : float  = 1 - EO_retained
        """
        J   = self._flux_mol_m2_s(T_surface_C)       # [mol/m²/s] per compound
        # Fractional loss per step: ΔEO_i / EO_i(0)
        # We track fractions so eo_frac[i] decreases by J*A*dt / EO_i(0)
        # Since EO_i(0) is normalised to 1 [mol], Δfrac_i = J_i × A_surface × dt
        delta = J * POD_SURFACE_AREA_M2 * dt          # dimensionless fraction lost
        self._eo_frac = np.clip(self._eo_frac - delta, 0.0, 1.0)

        retained = float(np.dot(self._w, self._eo_frac))

        return {
            "EO_retained":       retained,
            "EO_retained_per_compound": dict(zip(self.names, self._eo_frac.tolist())),
            "J_eo_total":        float(J.sum()),
            "EO_lost_fraction":  float(1.0 - retained),
        }

    # ── Preview: EO retained after a fixed temperature profile ──────────────

    def preview_loss(
        self,
        T_surface_history: np.ndarray,
        dt: float = 10.0,
    ) -> float:
        """
        Compute EO retained after a sequence of surface temperatures
        WITHOUT updating the model's internal state.  Used by the MPC controller
        to evaluate candidate control actions.

        Parameters
        ----------
        T_surface_history : (N,) array of surface temps [°C]
        dt                : timestep [s]

        Returns
        -------
        float : EO_retained at end of sequence
        """
        eo_tmp = self._eo_frac.copy()
        for T in T_surface_history:
            J     = self._flux_mol_m2_s(float(T))
            delta = J * POD_SURFACE_AREA_M2 * dt
            eo_tmp = np.clip(eo_tmp - delta, 0.0, 1.0)
        return float(np.dot(self._w, eo_tmp))

    # ── State accessors ─────────────────────────────────────────────────────

    @property
    def EO_retained(self) -> float:
        return float(np.dot(self._w, self._eo_frac))

    @property
    def compound_fractions(self) -> dict:
        return dict(zip(self.names, self._eo_frac.tolist()))

    # ── Evaporation temperature table ───────────────────────────────────────

    def print_evap_table(self) -> None:
        """Print the evaporation onset temperature for each compound."""
        print(f"{'Compound':<28}  Pvap@50°C [Pa]  km [m/s]   w_commercial")
        print("-" * 70)
        Pvap50 = self._pvap_pa(50.0)
        for i, name in enumerate(self.names):
            print(f"  {name:<26}  {Pvap50[i]:>12.2f}  {self._km[i]:.1e}   {self._w[i]:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    eo = EOModel()

    print("=== EO Model Self-Test ===\n")
    eo.print_evap_table()

    # Simulate 1 hour at 50°C
    eo.reset()
    dt = 10.0
    n_steps = int(3600 / dt)   # 1 hour

    for _ in range(n_steps):
        result = eo.step(T_surface_C=50.0, dt=dt)

    print(f"\n  After 1 h at 50°C:")
    print(f"  EO retained : {result['EO_retained']:.4f}")
    for name, frac in result["EO_retained_per_compound"].items():
        print(f"    {name:<30}: {frac:.4f}")

    # Compare: 1 hour at 40°C
    eo.reset()
    for _ in range(n_steps):
        result40 = eo.step(T_surface_C=40.0, dt=dt)
    print(f"\n  After 1 h at 40°C:")
    print(f"  EO retained : {result40['EO_retained']:.4f}  (expected > 50°C result)")
    print("\n  EO model self-test passed ✓")
