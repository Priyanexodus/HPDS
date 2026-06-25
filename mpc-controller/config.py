"""
config.py
=========
Centralised configuration for the Cardamom HPDS MPC Controller (Phase 5).

All tuneable knobs live here — edit this file without touching any other module.
Fan speed is a fixed constant; only the heater setpoint is optimised.
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent.parent          # HPDS/
PINN_CKPT       = BASE_DIR / "architecture-training" / "checkpoints" / "best.pt"
FNO_CKPT        = BASE_DIR / "architecture-training-FNO" / "checkpoints" / "best.pt"
FNO_GEOMETRY    = BASE_DIR / "synthetic_data" / "FNO" / "pod_geometry.h5"
RESULTS_DIR     = Path(__file__).parent / "results"

# ─────────────────────────────────────────────────────────────────────────────
# CONTROL LOOP TIMING
# ─────────────────────────────────────────────────────────────────────────────
DT          = 10          # [s]  control period (matches sensor sample rate)
HORIZON     = 108         # [steps]  MPC lookahead = HORIZON × DT = 1080 s = 18 min
TOTAL_TIME_S = 30_780     # [s]  total drying run = 8.55 h

# ─────────────────────────────────────────────────────────────────────────────
# CONTROL CANDIDATES  (fan speed is constant — NOT a control variable)
# ─────────────────────────────────────────────────────────────────────────────
FAN_SPEED   = 1.0                         # relative fan speed — fixed
T_CANDS     = [30.0, 35.0, 40.0, 45.0, 50.0, 55.0]   # [°C] heater setpoints to test

# ─────────────────────────────────────────────────────────────────────────────
# OBJECTIVE FUNCTION WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────
LAMBDA_MR   = 10.0   # penalise distance of final MR from target  ← dominant driver
LAMBDA_EO   = 0.5    # penalise EO loss (secondary quality guard)
LAMBDA_E    = 0.05   # penalise high heater temperature (energy cost proxy)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTRAINTS
# ─────────────────────────────────────────────────────────────────────────────
T_AIR_MIN       = 30.0   # [°C]  minimum heater setpoint
T_AIR_MAX       = 55.0   # [°C]  maximum heater setpoint
T_SURF_MAX      = 55.0   # [°C]  pod surface temperature hard limit (matches heater max)
RH_OUT_MAX      = 95.0   # [%]   outlet relative humidity hard limit

# ─────────────────────────────────────────────────────────────────────────────
# DRYING TARGETS
# ─────────────────────────────────────────────────────────────────────────────
MR_TARGET       = 0.11   # final moisture ratio target (wet basis)
MR_PHASE1_END   = 0.50   # Phase 1 → Phase 2 transition
MR_PHASE2_END   = 0.20   # Phase 2 → Phase 3 transition

# ─────────────────────────────────────────────────────────────────────────────
# PHYSICAL CONSTANTS  (cardamom pod — must match PINN training config)
# ─────────────────────────────────────────────────────────────────────────────
A_AXIAL     = 9.2e-3        # pod axial semi-axis     [m]
B_RADIAL    = 5.056e-3      # pod radial semi-axis    [m]
C_INIT      = 47082.0       # initial moisture conc.  [mol/m³]
C_SCALE     = 50000.0       # PINN normalisation scale for c
T_SCALE     = 60.0          # PINN normalisation scale for T   [°C]
PINN_TOTAL_TIME_S = 34200.0 # PINN time domain length  [s]

# FNO normalisation (must match architecture-training-FNO/config.py)
NORM_MEAN_X = [51.336,  0.5674, 58.184]
NORM_STD_X  = [25.534,  0.4210,  5.850]
NORM_MEAN_Y = [55.775,  0.5305]
NORM_STD_Y  = [30.933,  0.3971]
FNO_NX, FNO_NY = 8, 16

# ─────────────────────────────────────────────────────────────────────────────
# UKF SENSOR FUSION PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
# State:       [T_core(°C), T_surface(°C), c_core(mol/m³), c_surface(mol/m³), MR]
# Observation: [T_out_air(°C), RH_out(%), mass_g(g)]

UKF_ALPHA   = 1e-3    # sigma-point spread (small → conservative)
UKF_BETA    = 2.0     # prior knowledge of distribution (2 = Gaussian)
UKF_KAPPA   = 0.0     # secondary scaling (0 = standard)

# Process noise covariance Q (diagonal)
UKF_Q_DIAG  = [0.01,   # T_core   [°C²/step]
               0.01,   # T_surface
               100.0,  # c_core   [mol/m³]²/step
               100.0,  # c_surface
               1e-4]   # MR

# Measurement noise covariance R (diagonal)
UKF_R_DIAG  = [0.25,   # T_out_air  [°C²]      DHT22: ±0.5°C
               4.0,    # RH_out     [%²]        DHT22: ±2% RH
               0.01]   # mass_g     [g²]        load cell drift

# ─────────────────────────────────────────────────────────────────────────────
# PSYCHROMETRIC CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
P_ATM           = 101325.0   # atmospheric pressure  [Pa]
CP_DRY          = 1006.0     # dry air specific heat [J/kg·K]
CP_VAPOUR       = 1805.0     # vapour specific heat  [J/kg·K]
LAMBDA_LV       = 2.501e6    # latent heat of vaporisation [J/kg]
R_V             = 461.5      # water vapour gas constant [J/kg·K]

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATOR  (synthetic physics for demo without hardware)
# ─────────────────────────────────────────────────────────────────────────────
SIM_M0_G        = 450.0    # initial mass  [g]
SIM_M_EQ_G      = 49.5     # equilibrium mass at final RH  [g]
SIM_MR_INIT     = 0.80     # initial moisture ratio
SIM_T_INIT      = 25.0     # initial pod temperature  [°C]

# ─────────────────────────────────────────────────────────────────────────────
# ESSENTIAL OIL MODEL
# ─────────────────────────────────────────────────────────────────────────────
# Antoine coefficients: log10(Pvap [mmHg]) = A - B / (C + T[°C])
# Corrected to literature values that give physically sensible Pvap (1–500 Pa)
# at drying temperatures 30–55°C.
# References: NIST Webbook, Stull 1947, estimated from boiling points.
#
# Format: (A, B, C, km [m/s], w_commercial)
EO_COMPOUNDS = {
    # Monoterpenes — most volatile, evaporate at all drying temps
    "alpha_pinene":           (7.076, 1526.0,  214.0, 5e-4, 0.05),  # bp 155°C
    "sabinene":               (7.050, 1570.0,  212.0, 4e-4, 0.08),  # bp 163°C
    "myrcene":                (7.020, 1600.0,  210.0, 3e-4, 0.07),  # bp 167°C
    # Oxides — moderate volatility
    "cineole_18":             (7.090, 1650.0,  215.0, 2e-4, 0.20),  # bp 176°C
    # Alcohols — less volatile
    "linalool":               (7.300, 1840.0,  218.0, 1e-4, 0.15),  # bp 198°C
    "alpha_terpineol":        (7.350, 1970.0,  215.0, 8e-5, 0.10),  # bp 217°C
    "geraniol":               (7.500, 2100.0,  210.0, 5e-5, 0.15),  # bp 230°C
    # Esters — least volatile, most commercially important
    "alpha_terpinyl_acetate": (7.400, 2200.0,  210.0, 3e-5, 0.20),  # bp 220°C
}
POD_SURFACE_AREA_M2 = 2.1e-4   # ellipsoid surface area [m²]


if __name__ == "__main__":
    print("=== HPDS MPC Config ===")
    print(f"  Horizon        : {HORIZON} steps × {DT} s = {HORIZON*DT} s = {HORIZON*DT/60:.0f} min")
    print(f"  Heater cands   : {T_CANDS} °C")
    print(f"  Fan speed      : {FAN_SPEED} (constant)")
    print(f"  λ_MR / λ_EO / λ_E : {LAMBDA_MR} / {LAMBDA_EO} / {LAMBDA_E}")
    print(f"  PINN checkpoint: {PINN_CKPT}")
    print(f"  FNO  checkpoint: {FNO_CKPT}")
    print(f"  EO compounds   : {list(EO_COMPOUNDS.keys())}")
