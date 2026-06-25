"""
config.py
=========
Centralised hyperparameter configuration for the Cardamom HPDS FNO.
Edit this file to tune training without touching any other module.

FNO Task:
  Input  : (3, 8, 16)  — [T_current, M_current, T_heater]  (3 channels, 8×16 grid)
  Output : (2, 8, 16)  — [T_future, M_future]               (2 channels, 6 min ahead)
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent.parent           # HPDS/
DATA_DIR      = BASE_DIR / "synthetic_data" / "FNO"
TRAIN_PATH    = DATA_DIR / "fno_train.h5"
VAL_PATH      = DATA_DIR / "fno_val.h5"
TEST_PATH     = DATA_DIR / "fno_test.h5"
GEOMETRY_PATH = DATA_DIR / "pod_geometry.h5"
METADATA_PATH = DATA_DIR / "metadata.json"
RESULTS_DIR   = Path(__file__).parent / "results"
CKPT_DIR      = Path(__file__).parent / "checkpoints"

# ─────────────────────────────────────────────────────────────────────────────
# GRID DIMENSIONS  (must match data generator)
# ─────────────────────────────────────────────────────────────────────────────
NX = 8           # spatial grid width
NY = 16          # spatial grid height
IN_CHANNELS  = 3  # T_current, M_current, T_heater
OUT_CHANNELS = 2  # T_future, M_future

# ─────────────────────────────────────────────────────────────────────────────
# PHYSICAL CONSTANTS  (cardamom pod — matches FNO_data_gen.py)
# ─────────────────────────────────────────────────────────────────────────────
L_X     = 9.0e-3          # pod width  [m]
L_Y     = 18.0e-3         # pod length [m]
RHO     = 1100.0          # density             [kg/m³]
CP      = 3500.0          # specific heat       [J/(kg·K)]
K_TH    = 0.45            # thermal conductivity[W/(m·K)]
D_EFF   = 8.0e-10         # moisture diffusivity[m²/s]
M_EQ    = 0.08            # equilibrium moisture[kg_w/kg_dry]
H_CONV  = 30.0            # convective HTC      [W/(m²·K)]

# ─────────────────────────────────────────────────────────────────────────────
# NORMALISATION STATISTICS  (from metadata.json — training set stats)
# ─────────────────────────────────────────────────────────────────────────────
# Input channels: [T_current, M_current, T_heater]
NORM_MEAN_X = [51.336,  0.5674, 58.184]   # per-channel means
NORM_STD_X  = [25.534,  0.4210,  5.850]   # per-channel stds

# Output channels: [T_future, M_future]
NORM_MEAN_Y = [55.775,  0.5305]
NORM_STD_Y  = [30.933,  0.3971]

# ─────────────────────────────────────────────────────────────────────────────
# FNO ARCHITECTURE
# ─────────────────────────────────────────────────────────────────────────────
LATENT_CHANNELS  = 64       # channel width in spectral layers
NUM_FNO_LAYERS   = 4        # number of FNO spectral blocks
NUM_MODES_X      = 4        # Fourier modes to keep along x (≤ NX//2+1 = 5)
NUM_MODES_Y      = 8        # Fourier modes to keep along y (≤ NY//2+1 = 9)
DECODER_HIDDEN   = 128      # width of the MLP decoder (latent → output)
DECODER_LAYERS   = 2        # depth of the MLP decoder

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────
SEED           = 42
BATCH_SIZE     = 64          # training mini-batch
LR             = 1e-3        # initial Adam learning rate
LR_MIN         = 1e-5        # cosine annealing floor
WARMUP_STEPS   = 200         # linear LR warm-up epochs
TOTAL_EPOCHS   = 500         # full training budget
GRAD_CLIP_NORM = 1.0
NUM_WORKERS    = 0           # DataLoader workers (0 = main process)

# ─────────────────────────────────────────────────────────────────────────────
# LOSS WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────
LAMBDA_T = 1.0    # weight on temperature output loss
LAMBDA_M = 5.0    # weight on moisture output loss (harder to learn)

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING / CHECKPOINTING
# ─────────────────────────────────────────────────────────────────────────────
LOG_EVERY  = 10    # print / TensorBoard log interval [epochs]
SAVE_EVERY = 50    # periodic checkpoint interval [epochs]
VAL_EVERY  = 10    # validation interval [epochs]

# ─────────────────────────────────────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = "cuda"    # falls back to "cpu" automatically in train.py
