"""
config.py
=========
Centralised hyperparameter configuration for the Cardamom HPDS PINN.
Edit this file to tune training without touching any other module.
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent           # HPDS/
DATA_PATH   = BASE_DIR / "synthetic_data" / "PINN" / "hpds_pinn_dataset.h5"
RESULTS_DIR = Path(__file__).parent / "results"
CKPT_DIR    = Path(__file__).parent / "checkpoints"

# ─────────────────────────────────────────────────────────────────────────────
# PHYSICAL CONSTANTS  (must match data generator)
# ─────────────────────────────────────────────────────────────────────────────
A_AXIAL    = 9.2e-3        # pod axial semi-axis    [m]
B_RADIAL   = 5.056e-3      # pod radial semi-axis   [m]
R_EFF      = (A_AXIAL * B_RADIAL ** 2) ** (1.0 / 3.0)   # equivalent sphere radius

ALPHA_T    = 1.2e-7        # thermal diffusivity    [m²/s]
K_COND     = 0.20          # thermal conductivity   [W/m·K]
H_CONV     = 18.0          # convective HTC         [W/m²·K]
H_M_CONV   = 2.0e-5        # convective mass transfer coeff [m/s]
BI_T       = H_CONV * R_EFF / K_COND

D_EFF_50   = 2.0e-9        # moisture diffusivity at 50°C  [m²/s]
T_REF_K    = 323.15        # reference temp for Arrhenius  [K]
EA_MOIST   = 2.8e4         # activation energy             [J/mol]
R_GAS      = 8.314         # gas constant                  [J/mol·K]

C_INIT     = 47082.0       # initial bulk moisture conc.   [mol/m³]
T_INIT     = 25.0          # initial pod temperature       [°C]  (approx)

TOTAL_TIME_S = 34200.0     # total drying duration         [s]  (9.5 h)

# ─────────────────────────────────────────────────────────────────────────────
# NORMALISATION SCALES  (used to non-dimensionalise network outputs)
# ─────────────────────────────────────────────────────────────────────────────
T_SCALE    = 60.0          # [°C]   — rough peak temperature
C_SCALE    = 50000.0       # [mol/m³] — approximate initial bulk value

# ─────────────────────────────────────────────────────────────────────────────
# NETWORK ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
FOURIER_FEATURES  = 64     # number of Fourier random frequencies (per axis)
HIDDEN_DIM        = 256    # neurons per hidden layer
N_HIDDEN          = 6      # number of hidden layers
ACTIVATION        = "tanh" # "tanh" | "silu"
SKIP_EVERY        = 2      # add residual skip connection every N layers

# ─────────────────────────────────────────────────────────────────────────────
# LOSS WEIGHTS  (λ values for each term)
# ─────────────────────────────────────────────────────────────────────────────
LAMBDA_PHYSICS_T  = 0.5    # PDE residual — heat equation  (reduced to not override IC)
LAMBDA_PHYSICS_C  = 0.5    # PDE residual — moisture diffusion (reduced: was dominating IC)
LAMBDA_IC_T       = 10.0   # initial condition — temperature (raised: must survive Phase 2)
LAMBDA_IC_C       = 200.0  # initial condition — concentration (strongly protected:
                            #   Phase 2 physics was pulling c to equilibrium instead of IC=0.942)
LAMBDA_NON_NEG    = 5.0    # non-negativity of c (physical constraint, c ≥ 0 always)
LAMBDA_BC_T       = 2.0    # Robin BC at pod surface (temperature)
LAMBDA_BC_C       = 2.0    # Mass transfer BC at pod surface (moisture)
LAMBDA_DATA_T     = 10.0   # COMSOL anchor T supervision (raised to compete with physics)
LAMBDA_DATA_C     = 10.0   # COMSOL anchor c supervision (raised to compete with physics)
LAMBDA_MR         = 50.0   # load-cell MR(t) supervision (key observable — raised: drying curve must be followed)

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────
SEED              = 42
BATCH_COLL        = 2048   # collocation mini-batch size
LR                = 1e-3   # initial Adam learning rate
LR_MIN            = 1e-5   # cosine annealing floor
WARMUP_STEPS      = 500    # linear LR warm-up steps
PHASE1_EPOCHS     = 2_000  # pretrain on data/IC/BC only  (no physics residual)
PHASE2_EPOCHS     = 13_000 # full PINN training
TOTAL_EPOCHS      = PHASE1_EPOCHS + PHASE2_EPOCHS  # 15 000

ADAPTIVE_LAMBDA_EVERY = 500   # gradient-norm rebalancing interval  [epochs]
GRAD_CLIP_NORM    = 1.0
LOG_EVERY         = 100       # print / TensorBoard log interval
SAVE_EVERY        = 500       # checkpoint save interval
VAL_EVERY         = 500       # full validation interval

# ─────────────────────────────────────────────────────────────────────────────
# L-BFGS PHASE 3  (quasi-Newton refinement after Adam)
# ─────────────────────────────────────────────────────────────────────────────
# Raissi et al. (2019): Adam first, then L-BFGS to reach the precision floor.
# L-BFGS is a FULL-BATCH optimizer — uses all fixed data + a fixed large
# collocation batch each step. Each "epoch" calls up to LBFGS_MAX_ITER
# line-search steps via the strong Wolfe condition.
LBFGS_EPOCHS        = 2_000   # number of L-BFGS outer "epochs"
LBFGS_MAX_ITER      = 20      # max inner iterations per step (line searches)
LBFGS_HISTORY_SIZE  = 100     # number of past gradients stored (curvature memory)
LBFGS_LR            = 1.0    # step size — always 1.0 with strong Wolfe search
LBFGS_COLL_PTS      = 4_096  # fixed collocation batch size for the closure
LBFGS_LOG_EVERY     = 10      # L-BFGS logging interval [epochs]
LBFGS_SAVE_EVERY    = 100     # L-BFGS checkpoint interval [epochs]
LBFGS_VAL_THRESHOLD = 1.2   # Adam plateaus at ~1.07 on this problem; 1.2 gives headroom
# ─────────────────────────────────────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = "cuda"   # falls back to "cpu" automatically in train.py
