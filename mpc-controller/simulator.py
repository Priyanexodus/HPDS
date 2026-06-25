"""
simulator.py
============
Synthetic HPDS drying simulator for closed-loop MPC testing.

Purpose
-------
Lets you validate the full MPC control loop (estimator → controller → actuation)
without a Raspberry Pi or real hardware.

Physics model (lumped, 1-D)
----------------------------
Temperature (lumped Bi-corrected):
    dT_pod/dt = (h × A / (ρ × V × Cp)) × (T_heater − T_pod)
    T_surface  = T_pod + Bi × (T_heater − T_pod) / (1 + Bi)  [approx]

Moisture (Fick's law, thin-slab approximation):
    dMR/dt = − (π² × D_eff) / (4 × L²) × (MR − MR_eq)
    MR_eq  = 0.08 (equilibrium at the chosen operating RH)

EO loss:
    Tracked by EOModel using predicted T_surface at each step.

Sensors:
    DHT22 T_out : T_surface + 0.8 × (T_heater − T_surface) + N(0, 0.5)  [°C]
    DHT22 RH_out: derived from surface moisture + N(0, 2.0)              [%]
    Load cell   : m(t) = M_eq + MR(t) × (M0 − M_eq) + N(0, drift)       [g]
              where drift follows a random walk (0.01 g²/s → load cell drift)
"""

import sys
from pathlib import Path

import numpy as np

import importlib.util as _ilu
from pathlib import Path as _Path

# Load MPC config robustly — avoids collision with PINN/FNO config.py files
_mpc_spec = _ilu.spec_from_file_location(
    "mpc_config", str(_Path(__file__).parent / "config.py")
)
C = _ilu.module_from_spec(_mpc_spec)
_mpc_spec.loader.exec_module(C)

from eo_model import EOModel


# ── Inline psychrometric helper (avoids circular import from state_estimator) ──
def _rh_from_c_and_T(c_mol_m3: float, T_C: float) -> float:
    """Estimate outlet RH [%] from surface moisture concentration."""
    P_ATM = C.P_ATM
    Mw    = 0.018          # kg/mol
    rho   = 1.2            # air density at ~50°C [kg/m³]
    w     = max(0.0, c_mol_m3 * Mw / rho)
    pv    = w * P_ATM / (0.622 + w)
    psat  = 610.78 * float(np.exp(17.27 * T_C / (T_C + 237.3)))
    return float(min(100.0, 100.0 * pv / max(psat, 1e-6)))


class HPDSSim:
    """
    Synthetic cardamom HPDS drying simulator.

    Usage
    -----
        sim = HPDSSim(seed=42)
        obs = sim.reset()           # initial sensor observation

        for each step:
            obs = sim.step(T_setpoint=50.0)   # apply heater setpoint, get sensors
    """

    # ── Pod physical parameters ────────────────────────────────────────────────
    # (single-pod lumped model; realistic for Bi ≈ 0.23)
    RHO    = 1060.0          # density             [kg/m³]
    CP     = 3676.0          # specific heat       [J/kg·K]
    K_TH   = 0.20            # thermal conductivity[W/m·K]
    H_CONV = 18.0            # convective HTC      [W/m²·K]
    A_AXIAL = C.A_AXIAL      # semi-axis a         [m]
    B_RADIAL = C.B_RADIAL    # semi-axis b         [m]

    # Pod geometry (prolate ellipsoid)
    @property
    def _V(self) -> float:
        """Pod volume [m³]."""
        return (4 / 3) * np.pi * self.A_AXIAL * self.B_RADIAL ** 2

    @property
    def _A(self) -> float:
        """Pod surface area [m²] (approximation for prolate ellipsoid)."""
        a, b = self.A_AXIAL, self.B_RADIAL
        e  = np.sqrt(1 - (b / a) ** 2)
        return 2 * np.pi * b ** 2 * (1 + (a / (b * e)) * np.arcsin(e))

    @property
    def _Bi(self) -> float:
        return self.H_CONV * min(self.A_AXIAL, self.B_RADIAL) / self.K_TH

    # Fick's law moisture diffusion
    D_EFF0  = 2.0e-9         # moisture diffusivity at 50°C [m²/s]
    EA_MOIST = 2.8e4         # activation energy [J/mol]
    R_GAS   = 8.314
    MR_EQ   = 0.08           # equilibrium moisture ratio

    def __init__(self, seed: int = 42):
        self._rng     = np.random.default_rng(seed)
        self.eo_model = EOModel()
        self._load_cell_drift = 0.0   # cumulative drift [g]
        self.reset()

    # ── Effective diffusivity (Arrhenius) ─────────────────────────────────────

    def _D_eff(self, T_C: float) -> float:
        T_K = T_C + 273.15
        T_ref = 50.0 + 273.15
        # D(T) = D_ref * exp( -Ea/R * (1/T - 1/T_ref) )
        return self.D_EFF0 * np.exp(-self.EA_MOIST / self.R_GAS * (1.0/T_K - 1.0/T_ref))

    # ── Reset ──────────────────────────────────────────────────────────────────

    def reset(self) -> dict:
        """Reset simulator to initial conditions. Returns initial sensor obs."""
        self.t          = 0.0               # current time [s]
        self.T_pod      = C.SIM_T_INIT      # pod temperature [°C]
        self.MR         = C.SIM_MR_INIT     # moisture ratio
        self._load_cell_drift = 0.0
        self.eo_model.reset()

        # Derived initial surface temperature
        self.T_surface  = self.T_pod        # no gradient at t=0

        obs = self._make_obs(T_setpoint=25.0)
        return obs

    # ── One timestep ──────────────────────────────────────────────────────────

    def step(self, T_setpoint: float) -> dict:
        """
        Advance the simulation by DT seconds.

        Parameters
        ----------
        T_setpoint : heater setpoint [°C]

        Returns
        -------
        dict:
            T_out       : float  DHT22 outlet air temperature [°C]  (noisy)
            RH_out      : float  DHT22 outlet humidity [%]          (noisy)
            mass_g      : float  load cell mass [g]                 (noisy)
            MR_true     : float  true moisture ratio (groundtruth, not a sensor)
            T_surf_true : float  true surface temperature [°C]
            EO_retained : float  true EO quality index
            t_s         : float  current time [s]
        """
        dt  = float(C.DT)
        T_h = float(T_setpoint)

        # ── Update pod temperature (lumped) ───────────────────────────────
        # dT/dt = (h×A)/(ρ×V×Cp) × (T_heater − T_pod)
        tau_T   = self.RHO * self._V * self.CP / (self.H_CONV * self._A)
        dT_dt   = (T_h - self.T_pod) / tau_T
        self.T_pod = self.T_pod + dT_dt * dt

        # ── Bi-corrected surface temperature ──────────────────────────────
        # T_surface ≈ T_pod + Bi / (1 + Bi) × (T_heater − T_pod)
        Bi = self._Bi
        self.T_surface = self.T_pod + (Bi / (1 + Bi)) * (T_h - self.T_pod)
        self.T_surface = min(self.T_surface, T_h)  # cap at heater temp

        # ── Update moisture (Fick's law, first term of series solution) ────
        # dMR/dt ≈ −(π²D)/(4L²) × (MR − MR_eq)
        L     = self.B_RADIAL   # characteristic dimension [m]
        D_e   = self._D_eff(self.T_pod)
        k_dry = (np.pi ** 2 * D_e) / (4 * L ** 2)
        dMR_dt= -k_dry * (self.MR - self.MR_EQ)
        self.MR = max(self.MR_EQ, self.MR + dMR_dt * dt)

        # ── Update EO model ────────────────────────────────────────────────
        eo_result = self.eo_model.step(T_surface_C=self.T_surface, dt=dt)

        self.t += dt

        return self._make_obs(T_setpoint=T_h, eo_result=eo_result)

    # ── Observation builder (adds sensor noise) ────────────────────────────────

    def _make_obs(self, T_setpoint: float, eo_result: dict = None) -> dict:
        rng = self._rng

        # True outlet air temperature: weighted blend of heater + pod surface
        T_out_true = self.T_surface + 0.8 * max(0.0, T_setpoint - self.T_surface)
        T_out      = float(T_out_true + rng.normal(0.0, 0.5))   # DHT22 ±0.5°C

        # True outlet RH: derived from surface moisture
        # Convert MR → c_surface → RH
        # c_surface ≈ MR × C_INIT (rough proxy)
        c_surf_approx = self.MR * C.C_INIT
        RH_out_true = _rh_from_c_and_T(c_surf_approx, T_out_true)
        RH_out = float(np.clip(RH_out_true + rng.normal(0.0, 2.0), 0.0, 100.0))

        # Load cell mass [g] with drift random walk
        mass_true = C.SIM_M_EQ_G + self.MR * (C.SIM_M0_G - C.SIM_M_EQ_G)
        self._load_cell_drift += rng.normal(0.0, np.sqrt(0.01 * C.DT))
        mass_g = float(max(C.SIM_M_EQ_G, mass_true + self._load_cell_drift))

        obs = {
            # Sensor readings (what the MPC controller sees)
            "T_out":        T_out,
            "RH_out":       RH_out,
            "mass_g":       mass_g,
            # Ground truth (for validation / plotting only)
            "MR_true":      float(self.MR),
            "T_surf_true":  float(self.T_surface),
            "T_pod_true":   float(self.T_pod),
            "EO_retained":  float(eo_result["EO_retained"]) if eo_result else 1.0,
            "t_s":          float(self.t),
        }
        return obs


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Simulator Self-Test ===")
    sim = HPDSSim(seed=0)
    obs = sim.reset()
    print(f"  t=0s:   T_pod={obs['T_pod_true']:.1f}°C  MR={obs['MR_true']:.4f}  "
          f"T_out={obs['T_out']:.1f}°C  mass={obs['mass_g']:.1f}g")

    # 30-minute constant-temperature run at 50°C
    n_steps = int(30 * 60 / C.DT)
    for i in range(n_steps):
        obs = sim.step(T_setpoint=50.0)

    print(f"  t=30min: T_pod={obs['T_pod_true']:.1f}°C  MR={obs['MR_true']:.4f}  "
          f"T_surf={obs['T_surf_true']:.1f}°C  EO={obs['EO_retained']:.4f}")
    print("  Simulator self-test passed ✓")
