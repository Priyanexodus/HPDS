"""
=======================================================================
  Synthetic Data Generator for FNO — Single Cardamom Pod Drying
=======================================================================

  The cardamom pod is modelled as an 8×16 spatial grid.
  Physical size: ~18 mm long × ~9 mm wide × ~8 mm thick (cross-section)
  Each cell covers ≈ 1.125 mm × 1.125 mm of pod flesh.

  ┌─ FNO INPUT  (3, 8, 16) ──────────────────────────────────────────┐
  │  Channel 0 — Current temperature   T(x,y)   [°C]               │
  │  Channel 1 — Current moisture      M(x,y)   [kg_water/kg_dry]  │
  │  Channel 2 — Heater temperature    T_h(x,y) [°C]  (uniform)    │
  └───────────────────────────────────────────────────────────────────┘

  ┌─ FNO OUTPUT (2, 8, 16) ──────────────────────────────────────────┐
  │  Channel 0 — Temperature after 6 min  T'(x,y)  [°C]            │
  │  Channel 1 — Moisture after 6 min     M'(x,y)  [kg_water/kg_dry]│
  └───────────────────────────────────────────────────────────────────┘

  Physics:
    • 2-D Fourier heat conduction within pod flesh
    • Newton convection from heater air at pod surface
    • Fickian moisture diffusion (2-D) within pod flesh
    • Temperature-dependent surface evaporation (Arrhenius)
    • Latent heat cooling during evaporation

  Solver:  Forward Euler,  dt = 3 s,  120 steps per 6-min step

  Outputs:
    fno_train.h5       —  8 000 training samples
    fno_val.h5         —  1 000 validation samples
    fno_test.h5        —    500 test samples
    metadata.json      —  physical params + normalization stats
    pod_geometry.h5    —  masks, coordinate grids, r-field

  Array shapes
    X  (N, 3, 8, 16)  — input  [T, M, T_heater]
    y  (N, 2, 8, 16)  — output [T_future, M_future]
=======================================================================
"""

import numpy as np
from scipy.ndimage import gaussian_filter, binary_erosion
import json, time, os
from pathlib import Path
import h5py

# ─────────────────────────────────────────────────────────────────────
# 1.  PHYSICAL CONSTANTS — cardamom pod
# ─────────────────────────────────────────────────────────────────────

# Grid dimensions
Nx, Ny = 8, 16
# Physical pod size
L_x    = 9.0e-3                     # pod width  [m]  9 mm
L_y    = 18.0e-3                    # pod length [m]  18 mm
dx     = L_x / Nx                   # ≈ 1.125 mm per cell
dy     = L_y / Ny                   # ≈ 1.125 mm per cell

# Thermal properties (cardamom flesh, wet basis)
rho    = 1100.0                     # density             [kg/m³]
cp     = 3500.0                     # specific heat       [J/(kg·K)]
k_th   = 0.45                       # thermal conductivity[W/(m·K)]
alpha  = k_th / (rho * cp)          # thermal diffusivity [m²/s] ≈ 1.17e-7

# Moisture diffusion
D_eff  = 8.0e-10                    # effective diffusivity [m²/s]
M_eq   = 0.08                       # equilibrium moisture [kg_w/kg_dry]
# NOTE: M_eq corresponds to ~7.5% wet-basis, typical for dried spices

# Surface heat transfer from hot-air dryer
h_conv = 30.0                       # convective HTC [W/(m²·K)]

# Surface evaporation — Arrhenius kinetics
k_evap_ref  = 3.5e-4               # rate const at 60°C   [1/s]
T_evap_ref  = 333.15               # reference temperature [K]
Ea_evap     = 38_000.0             # activation energy     [J/mol]
R_gas       = 8.314                # gas constant          [J/(mol·K)]
L_vap       = 2.45e6               # latent heat of water  [J/kg_water]

# Stability check — Forward Euler requires dt < dx²/(2α)
dt_max = 0.5 * dx**2 / alpha       # ≈ 5.4 s
dt     = 3.0                       # chosen simulation dt  [s]  < dt_max ✓
t_pred = 360.0                     # 6-minute prediction   [s]
n_sim  = int(t_pred / dt)          # steps per prediction  = 120

assert dt < dt_max, f"Unstable! dt={dt} >= dt_max={dt_max:.2f}"
assert dt < 0.5 * dx**2 / D_eff * 1e6, "Check moisture diffusion stability"

# ─────────────────────────────────────────────────────────────────────
# 2.  POD GEOMETRY — elliptical mask on 8×16 grid
# ─────────────────────────────────────────────────────────────────────

def build_geometry(Nx=8, Ny=16, fill_fraction=0.88):
    """
    Returns:
      pod_mask   (Nx, Ny) bool  — True inside pod
      surf_mask  (Nx, Ny) bool  — True at pod boundary (surface cells)
      r_field    (Nx, Ny) f32   — normalised radial distance 0=centre 1=edge
      x_grid     (Nx, Ny) f32   — x coordinate [m]
      y_grid     (Nx, Ny) f32   — y coordinate [m]
    """
    ii, jj = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')

    cx = (Nx - 1) / 2.0
    cy = (Ny - 1) / 2.0
    rx = (Nx / 2.0) * fill_fraction
    ry = (Ny / 2.0) * fill_fraction

    dist2 = ((ii - cx) / rx)**2 + ((jj - cy) / ry)**2
    pod_mask  = dist2 <= 1.0
    surf_mask = pod_mask & ~binary_erosion(pod_mask)
    r_field   = np.sqrt(dist2).clip(0, 1).astype(np.float32)

    x_grid = ((ii + 0.5) * dx).astype(np.float32)
    y_grid = ((jj + 0.5) * dy).astype(np.float32)

    return pod_mask, surf_mask, r_field, x_grid, y_grid


POD_MASK, SURF_MASK, R_FIELD, X_GRID, Y_GRID = build_geometry()

# Precompute surface and interior masks for display
N_POD  = POD_MASK.sum()
N_SURF = SURF_MASK.sum()
print(f"Pod mask: {N_POD} cells  |  Surface: {N_SURF} cells  |  "
      f"Interior: {N_POD - N_SURF} cells")

# ─────────────────────────────────────────────────────────────────────
# 3.  FINITE-DIFFERENCE SIMULATOR (batched over N pods at once)
# ─────────────────────────────────────────────────────────────────────

def _laplacian(F):
    """
    2-D Laplacian with Neumann (zero-flux) boundary conditions.
    F : (B, Nx, Ny) float32
    Returns ∇²F same shape, in units of m⁻² (dx = dy assumed).
    """
    Fp = np.pad(F, [(0, 0), (1, 1), (1, 1)], mode='edge')
    return (Fp[:, :-2, 1:-1] + Fp[:, 2:, 1:-1]
          + Fp[:, 1:-1, :-2] + Fp[:, 1:-1, 2:]
          - 4 * Fp[:, 1:-1, 1:-1]) / dx**2


def simulate_batch(T0, M0, T_h, n_steps=120, dt=3.0):
    """
    Simulate one 6-minute drying step for a batch of B pods.

    Parameters
    ----------
    T0  : (B, Nx, Ny) float32  Initial temperature  [°C]
    M0  : (B, Nx, Ny) float32  Initial moisture     [kg_w/kg_dry]
    T_h : (B,)        float32  Heater temperature   [°C]

    Returns
    -------
    T_f : (B, Nx, Ny) float32  Temperature after 6 min
    M_f : (B, Nx, Ny) float32  Moisture   after 6 min
    """
    T  = T0.copy()
    M  = M0.copy()
    pm = POD_MASK[None]                         # (1, Nx, Ny)
    sm = SURF_MASK[None]                        # (1, Nx, Ny)
    Th = T_h[:, None, None].astype(np.float32) # (B, 1, 1)

    for _ in range(n_steps):
        T_K = T + 273.15

        # ── Heat ────────────────────────────────────────────────────
        lap_T     = _laplacian(T)
        dT        = alpha * lap_T               # conduction (everywhere in pod)

        # Surface convection  [°C/s]
        dT_conv   = (h_conv / (rho * cp * dx)) * (Th - T)

        # Arrhenius evaporation rate  [kg_w/(kg_dry · s)]
        k_evap    = k_evap_ref * np.exp(
                        (Ea_evap / R_gas) * (1.0 / T_evap_ref - 1.0 / T_K))
        evap_rate = k_evap * np.clip(M - M_eq, 0.0, None)

        # Latent-heat cooling at surface  [°C/s]
        dT_lat    = -(L_vap / cp) * evap_rate

        # Apply surface terms
        dT        = np.where(sm, dT + dT_conv + dT_lat, dT)

        T         = T + dt * dT
        T         = np.where(pm, T, Th)         # air outside pod = T_heater
        T         = T.clip(20.0, 90.0)

        # ── Moisture ─────────────────────────────────────────────────
        lap_M     = _laplacian(M)
        dM        = D_eff * lap_M               # Fickian diffusion

        # Surface evaporation loss
        dM        = np.where(sm, dM - evap_rate, dM)

        M         = M + dt * dM
        M         = np.where(pm, M.clip(0.0, 3.0), 0.0)

    return T.astype(np.float32), M.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────
# 4.  INITIAL CONDITION SAMPLER
# ─────────────────────────────────────────────────────────────────────

# Realistic drying-stage parameter envelopes
STAGE_PARAMS = {
    # stage  T_pod(°C)  M_centre(db)  T_heater(°C)
    'fresh': dict(T_pod=(28, 38), M_c=(1.5, 2.5), T_h=(50, 62)),
    'mid':   dict(T_pod=(40, 55), M_c=(0.40, 1.5), T_h=(55, 68)),
    'late':  dict(T_pod=(52, 65), M_c=(0.08, 0.40), T_h=(58, 75)),
}
STAGE_WEIGHTS = [0.40, 0.35, 0.25]


def sample_initial_states(B: int, rng: np.random.Generator, stage='mixed'):
    """
    Generate B physically plausible initial pod states.

    Returns
    -------
    T0  : (B, Nx, Ny) float32   initial temperature  [°C]
    M0  : (B, Nx, Ny) float32   initial moisture     [kg_w/kg_dry]
    T_h : (B,)        float32   heater temperature   [°C]
    """
    if stage == 'mixed':
        stages = rng.choice(list(STAGE_PARAMS), size=B, p=STAGE_WEIGHTS)
    else:
        stages = np.array([stage] * B)

    T0  = np.zeros((B, Nx, Ny), dtype=np.float32)
    M0  = np.zeros((B, Nx, Ny), dtype=np.float32)
    T_h = np.zeros(B,           dtype=np.float32)

    for i, stg in enumerate(stages):
        p = STAGE_PARAMS[stg]

        T_pod = rng.uniform(*p['T_pod'])       # mean pod temperature
        M_c   = rng.uniform(*p['M_c'])         # centre moisture
        T_hi  = rng.uniform(*p['T_h'])         # heater setpoint

        # ── Temperature field ────────────────────────────────────
        # Surface cells are warmer (convection already acting)
        T_surf_excess = rng.uniform(1.0, 5.0)
        T_field = T_pod + T_surf_excess * R_FIELD    # radial ramp

        # Spatially correlated random noise (± 2°C)
        noise  = rng.normal(0.0, 1.5, (Nx, Ny)).astype(np.float32)
        T_field += gaussian_filter(noise, sigma=1.2)
        T_field[~POD_MASK] = T_hi                    # air = heater temp

        # ── Moisture field ───────────────────────────────────────
        # Centre-heavy radial profile blended from two basis functions
        f1 = rng.uniform(0.5, 0.85)           # fraction in Gaussian core
        a  = rng.uniform(1.0, 2.5)            # Gaussian width parameter
        M_field = M_c * (
            f1       * np.exp(-a * R_FIELD**2)       # core Gaussian
            + (1-f1) * np.clip(1.0 - R_FIELD, 0, 1) # linear outer shell
        )

        # Small spatially-correlated noise (±0.03 db)
        noise_m = rng.normal(0.0, 0.02, (Nx, Ny)).astype(np.float32)
        M_field += gaussian_filter(noise_m, sigma=1.0)
        M_field  = M_field.clip(0.0, 3.0)
        M_field[~POD_MASK] = 0.0

        T0[i]  = T_field.astype(np.float32)
        M0[i]  = M_field.astype(np.float32)
        T_h[i] = T_hi

    return T0, M0, T_h


# ─────────────────────────────────────────────────────────────────────
# 5.  TRAJECTORY SAMPLER (MPC-style — T_h changes every 6 min)
# ─────────────────────────────────────────────────────────────────────

def generate_trajectory(rng, n_steps=15):
    """
    One drying trajectory: starts fresh, T_heater varies every 6 min.
    Returns list of (T_in, M_in, T_h_grid, T_out, M_out) tuples.
    """
    T0, M0, T_h0 = sample_initial_states(1, rng, stage='fresh')
    T = T0[0]; M = M0[0]; T_h_curr = float(T_h0[0])

    records = []
    for step in range(n_steps):
        # MPC-like random walk on heater temperature
        T_h_curr = float(np.clip(T_h_curr + rng.uniform(-4.0, 4.0), 50.0, 75.0))

        T_in = T.copy();  M_in = M.copy()
        T_h_grid = np.full((Nx, Ny), T_h_curr, dtype=np.float32)

        T_out, M_out = simulate_batch(T[None], M[None], np.array([T_h_curr]))
        T = T_out[0]; M = M_out[0]

        records.append((T_in, M_in, T_h_grid, T_out[0].copy(), M_out[0].copy()))

    return records


# ─────────────────────────────────────────────────────────────────────
# 6.  DATASET GENERATION
# ─────────────────────────────────────────────────────────────────────

def build_dataset(n_target: int, batch_size=100, rng=None,
                  traj_fraction=0.65, n_traj_steps=14):
    """
    Generate n_target (X, y) pairs.

    Strategy
    --------
    traj_fraction  of samples come from full drying trajectories
                   → exposes FNO to temporal correlations
    1-traj_fraction come from random initial conditions
                   → broad coverage of state space

    Returns X (N, 3, Nx, Ny), y (N, 2, Nx, Ny)
    """
    if rng is None:
        rng = np.random.default_rng(0)

    n_traj_samples   = int(n_target * traj_fraction)
    n_random_samples = n_target - n_traj_samples

    X_list, y_list = [], []

    # ── Trajectory-based samples ────────────────────────────────────
    n_trajectories = max(1, n_traj_samples // n_traj_steps)
    print(f"  Generating {n_trajectories} trajectories × {n_traj_steps} steps ...")

    for t in range(n_trajectories):
        records = generate_trajectory(rng, n_steps=n_traj_steps)
        for (T_in, M_in, T_h_grid, T_out, M_out) in records:
            X_list.append(np.stack([T_in, M_in, T_h_grid]))        # (3, Nx, Ny)
            y_list.append(np.stack([T_out, M_out]))                 # (2, Nx, Ny)
        if (t + 1) % 100 == 0:
            print(f"    trajectory {t+1}/{n_trajectories}")

    # ── Random-initial-condition samples ────────────────────────────
    print(f"  Generating {n_random_samples} random-IC samples ...")
    n_batches = int(np.ceil(n_random_samples / batch_size))

    for b in range(n_batches):
        bsz = min(batch_size, n_random_samples - b * batch_size)
        T0, M0, T_h = sample_initial_states(bsz, rng, stage='mixed')

        T_f, M_f = simulate_batch(T0, M0, T_h)

        T_h_grid = T_h[:, None, None] * np.ones((bsz, Nx, Ny), dtype=np.float32)
        X = np.stack([T0, M0, T_h_grid], axis=1)   # (B, 3, Nx, Ny)
        y = np.stack([T_f, M_f],         axis=1)   # (B, 2, Nx, Ny)

        X_list.extend(list(X))
        y_list.extend(list(y))

        if (b + 1) % 10 == 0:
            print(f"    random batch {b+1}/{n_batches}")

    X_all = np.stack(X_list, axis=0).astype(np.float32)[:n_target]
    y_all = np.stack(y_list, axis=0).astype(np.float32)[:n_target]

    return X_all, y_all


# ─────────────────────────────────────────────────────────────────────
# 7.  NORMALISATION STATISTICS
# ─────────────────────────────────────────────────────────────────────

def compute_stats(X_train, y_train):
    """
    Compute per-channel mean & std over training set (pod cells only).
    Returns dict for JSON serialisation.
    """
    pm = POD_MASK.flatten()

    stats = {}
    ch_names_X = ['T_current', 'M_current', 'T_heater']
    ch_names_y = ['T_future',  'M_future']

    for ci, name in enumerate(ch_names_X):
        vals = X_train[:, ci, :, :].reshape(-1, Nx * Ny)[:, pm].ravel()
        stats[f'X_ch{ci}_{name}'] = {
            'mean': float(vals.mean()), 'std': float(vals.std()),
            'min':  float(vals.min()),  'max': float(vals.max()),
        }

    for ci, name in enumerate(ch_names_y):
        vals = y_train[:, ci, :, :].reshape(-1, Nx * Ny)[:, pm].ravel()
        stats[f'y_ch{ci}_{name}'] = {
            'mean': float(vals.mean()), 'std': float(vals.std()),
            'min':  float(vals.min()),  'max': float(vals.max()),
        }

    return stats


# ─────────────────────────────────────────────────────────────────────
# 8.  MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    OUT = Path('/home/priyan/physicsnemo/HPDS/synthetic_data/FNO/')
    OUT.mkdir(exist_ok=True)

    rng = np.random.default_rng(2024)

    total_samples = 9_500
    n_train       = 8_000
    n_val         = 1_000
    n_test        = 500

    # ── Generate all data ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Generating {total_samples:,} samples for FNO cardamom model")
    print(f"  Pod grid: {Nx}×{Ny}  |  dt={dt}s  |  6-min horizon")
    print(f"{'='*60}\n")

    t0 = time.time()
    X_all, y_all = build_dataset(total_samples, batch_size=100, rng=rng)
    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s  →  {X_all.shape}  {y_all.shape}\n")

    # ── Sanity checks ────────────────────────────────────────────────
    pm_flat = POD_MASK.ravel()
    T_in_pod = X_all[:, 0].reshape(-1, Nx*Ny)[:, pm_flat]
    M_in_pod = X_all[:, 1].reshape(-1, Nx*Ny)[:, pm_flat]
    T_out_pod = y_all[:, 0].reshape(-1, Nx*Ny)[:, pm_flat]
    M_out_pod = y_all[:, 1].reshape(-1, Nx*Ny)[:, pm_flat]

    print("Sanity checks:")
    print(f"  T_in  range : {T_in_pod.min():.1f} – {T_in_pod.max():.1f} °C")
    print(f"  T_out range : {T_out_pod.min():.1f} – {T_out_pod.max():.1f} °C")
    print(f"  M_in  range : {M_in_pod.min():.3f} – {M_in_pod.max():.3f} kg/kg")
    print(f"  M_out range : {M_out_pod.min():.3f} – {M_out_pod.max():.3f} kg/kg")
    assert (M_out_pod >= 0).all(), "Negative moisture found!"
    assert (T_out_pod >= 20).all(), "Temperature below ambient!"
    print("  All checks passed ✓\n")

    # ── Shuffle and split ────────────────────────────────────────────
    idx = rng.permutation(len(X_all))
    X_all = X_all[idx]
    y_all = y_all[idx]

    splits = {
        'train': (X_all[:n_train],            y_all[:n_train]),
        'val':   (X_all[n_train:n_train+n_val], y_all[n_train:n_train+n_val]),
        'test':  (X_all[n_train+n_val:],       y_all[n_train+n_val:]),
    }

    # ── Save data splits ─────────────────────────────────────────────
    for split_name, (X, y) in splits.items():
        path = OUT / f'fno_{split_name}.h5'
        with h5py.File(str(path), 'w') as f:
            f.create_dataset('X', data=X, compression='gzip')
            f.create_dataset('y', data=y, compression='gzip')
        print(f"  Saved {path.name}  X{X.shape}  y{y.shape}  "
              f"({path.stat().st_size/1e6:.1f} MB)")

    # ── Save pod geometry ────────────────────────────────────────────
    geo_path = OUT / 'pod_geometry.h5'
    with h5py.File(str(geo_path), 'w') as f:
        f.create_dataset('pod_mask', data=POD_MASK.astype(np.int8))
        f.create_dataset('surf_mask', data=SURF_MASK.astype(np.int8))
        f.create_dataset('r_field', data=R_FIELD)
        f.create_dataset('x_grid', data=X_GRID)
        f.create_dataset('y_grid', data=Y_GRID)
    print(f"  Saved {geo_path.name}")

    # ── Save metadata ────────────────────────────────────────────────
    X_tr, y_tr = splits['train']
    stats = compute_stats(X_tr, y_tr)

    metadata = {
        "description": "FNO synthetic data — single cardamom pod drying",
        "grid": {"Nx": Nx, "Ny": Ny,
                 "dx_mm": round(dx * 1e3, 4),
                 "dy_mm": round(dy * 1e3, 4)},
        "pod_geometry": {
            "length_mm": L_y * 1e3,
            "width_mm":  L_x * 1e3,
            "n_pod_cells": int(N_POD),
            "n_surface_cells": int(N_SURF),
        },
        "physics": {
            "rho_kg_m3":          rho,
            "cp_J_kgK":           cp,
            "k_thermal_W_mK":     k_th,
            "alpha_m2_s":         alpha,
            "D_eff_m2_s":         D_eff,
            "M_eq_kg_kg":         M_eq,
            "h_conv_W_m2K":       h_conv,
            "k_evap_ref_per_s":   k_evap_ref,
            "T_evap_ref_K":       T_evap_ref,
            "Ea_evap_J_mol":      Ea_evap,
            "L_vap_J_kg":         L_vap,
        },
        "simulation": {
            "dt_s":               dt,
            "t_prediction_s":     t_pred,
            "n_steps":            n_sim,
            "solver":             "Forward Euler",
            "BC_temperature":     "Neumann (zero-flux walls), Dirichlet air=T_heater",
            "BC_moisture":        "Neumann (zero-flux walls), surface evaporation sink",
        },
        "input_channels": [
            {"ch": 0, "name": "T_current",   "units": "°C",
             "description": "Current temperature distribution inside pod"},
            {"ch": 1, "name": "M_current",   "units": "kg_water/kg_dry",
             "description": "Current moisture content distribution inside pod"},
            {"ch": 2, "name": "T_heater",    "units": "°C",
             "description": "Heater air temperature (uniform, MPC control variable)"},
        ],
        "output_channels": [
            {"ch": 0, "name": "T_future",    "units": "°C",
             "description": "Temperature distribution 6 min later"},
            {"ch": 1, "name": "M_future",    "units": "kg_water/kg_dry",
             "description": "Moisture content distribution 6 min later"},
        ],
        "dataset": {
            "n_train": n_train, "n_val": n_val, "n_test": n_test,
            "traj_fraction": 0.65,
            "n_traj_steps":  14,
            "heater_range_degC": [50, 75],
        },
        "normalisation_stats": stats,
        "units_note": (
            "Non-pod cells are masked: T=T_heater, M=0. "
            "When feeding to FNO, zero-out non-pod cells or apply pod_mask."
        ),
    }

    meta_path = OUT / 'metadata.json'
    with open(str(meta_path), 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved {meta_path.name}")

    print(f"\n{'='*60}")
    print(f"  Dataset ready in  {OUT}/")
    print(f"{'='*60}\n")

    return splits, metadata


if __name__ == '__main__':
    splits, metadata = main()

