"""
dataset.py
==========
HDF5 dataset loader for the Cardamom HPDS PINN.

Loads all four data categories from hpds_pinn_dataset.h5 and exposes
them as PyTorch tensors.  Provides:
  - CardamomPINNDataset  — torch.utils.data.Dataset for collocation batching
  - load_all_tensors     — returns fixed tensors (BC, anchors, MR) to device
  - bc_interpolators     — scipy interpolators for T_air(t), c_air(t), RH(t)
"""

import numpy as np
import torch
import h5py
from scipy.interpolate import interp1d
from torch.utils.data import Dataset

from config import (
    A_AXIAL, B_RADIAL, C_INIT, C_SCALE, T_SCALE, TOTAL_TIME_S, DATA_PATH
)


# ─────────────────────────────────────────────────────────────────────────────
# HDF5 reader helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_h5(path: str) -> dict:
    """Load all datasets from the HDF5 file into numpy arrays."""
    data = {}
    with h5py.File(path, "r") as f:
        # Category 1: collocation
        data["coords"] = f["category_1_collocation/coords"][:]
        # (N, 3) already normalised: [r_norm, z_norm, t_norm]

        # Category 2: boundary (sensor telemetry)
        data["t_sec_bc"] = f["category_2_boundary/t_sec"][:]
        data["T_air"]    = f["category_2_boundary/T_air"][:]
        data["RH_air"]   = f["category_2_boundary/RH_air"][:]
        data["c_air"]    = f["category_2_boundary/c_air"][:]   # [mol/m³]

        # Category 3a: load-cell MR
        data["t_sec_mr"] = f["category_3a_mr/t_sec"][:]
        data["MR"]       = f["category_3a_mr/MR"][:]

        # Category 3b: COMSOL anchors  (cols: r_m, z_m, t_sec, T_C, c_mol_m3)
        anch = f["category_3b_comsol/anchors"][:]
        data["anch_r"]   = anch[:, 0]
        data["anch_z"]   = anch[:, 1]
        data["anch_t"]   = anch[:, 2]
        data["anch_T"]   = anch[:, 3]
        data["anch_c"]   = anch[:, 4]

        # File-level attrs
        data["total_time_s"] = float(f.attrs.get("total_time_s", TOTAL_TIME_S))

    return data


# ─────────────────────────────────────────────────────────────────────────────
# Interpolators for boundary conditions
# ─────────────────────────────────────────────────────────────────────────────

def build_bc_interpolators(raw: dict):
    """
    Returns callables T_air_fn, c_air_fn that accept t_norm ∈ [0,1] and
    return T_air [°C] and c_air [mol/m³] as numpy arrays.
    """
    T_total = raw["total_time_s"]
    t_norm  = raw["t_sec_bc"] / T_total

    T_air_fn = interp1d(t_norm, raw["T_air"], kind="linear",
                        fill_value="extrapolate")
    c_air_fn = interp1d(t_norm, raw["c_air"], kind="linear",
                        fill_value="extrapolate")
    RH_air_fn = interp1d(t_norm, raw["RH_air"], kind="linear",
                         fill_value="extrapolate")

    def T_air_torch(t_norm_tensor: torch.Tensor) -> torch.Tensor:
        """Evaluate T_air interpolation on a torch tensor, returns tensor."""
        t_np  = t_norm_tensor.detach().cpu().numpy().ravel()
        T_np  = T_air_fn(t_np).astype(np.float32)
        return torch.tensor(T_np, dtype=torch.float32,
                            device=t_norm_tensor.device).unsqueeze(-1)

    return T_air_fn, c_air_fn, RH_air_fn, T_air_torch


# ─────────────────────────────────────────────────────────────────────────────
# Collocation Dataset (mini-batching over interior pts)
# ─────────────────────────────────────────────────────────────────────────────

class CardamomPINNDataset(Dataset):
    """
    Dataset wrapping the collocation coordinates (Category 1).
    Each item is one (r_norm, z_norm, t_norm) coordinate.

    Parameters
    ----------
    path : str | Path
        Path to hpds_pinn_dataset.h5.
    augment : bool
        If True, randomly resample interior points each epoch (online).
        If False, use the fixed 8192 stored collocation points.
    n_aug : int
        Number of online-resampled points (used when augment=True).
    """

    def __init__(self, path=None, augment: bool = False, n_aug: int = 8192):
        if path is None:
            path = DATA_PATH
        self._raw  = _read_h5(str(path))
        self._coords = torch.tensor(
            self._raw["coords"], dtype=torch.float32
        )  # (N, 3)
        self.augment = augment
        self.n_aug   = n_aug

    def __len__(self) -> int:
        return self.n_aug if self.augment else len(self._coords)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self.augment:
            return self._sample_one()
        return self._coords[idx]

    def _sample_one(self) -> torch.Tensor:
        """Sample one random interior point via rejection sampling."""
        while True:
            r_n = np.random.uniform(0.0, 1.0)
            z_n = np.random.uniform(-1.0, 1.0)
            if r_n ** 2 + z_n ** 2 < 1.0:
                t_n = np.random.uniform(0.0, 1.0)
                return torch.tensor([r_n, z_n, t_n], dtype=torch.float32)

    def get_raw(self) -> dict:
        return self._raw


# ─────────────────────────────────────────────────────────────────────────────
# Fixed tensors for BC, anchors, MR  (small enough to keep on GPU entirely)
# ─────────────────────────────────────────────────────────────────────────────

def load_fixed_tensors(raw: dict, device: torch.device) -> dict:
    """
    Convert all fixed (non-collocation) data to device tensors.

    Returns a dict with keys:
      x_anch   (K, 3) normalised anchor coords
      T_anch   (K, 1) anchor T [°C]
      c_anch   (K, 1) anchor c [mol/m³]
      t_mr     (T,)   normalised MR time points
      MR       (T,)   load-cell MR
      x_bc     (M, 3) surface boundary coords sampled at sensor time-points
      T_air_bc (M, 1) air temperature at those time-points [°C]
    """
    T_total = raw["total_time_s"]

    # ── COMSOL anchors ────────────────────────────────────────────────────
    r_n = torch.tensor(raw["anch_r"] / B_RADIAL,   dtype=torch.float32)
    z_n = torch.tensor(raw["anch_z"] / A_AXIAL,    dtype=torch.float32)
    t_n = torch.tensor(raw["anch_t"] / T_total,    dtype=torch.float32)
    x_anch = torch.stack([r_n, z_n, t_n], dim=1).to(device)   # (K, 3)
    T_anch = torch.tensor(raw["anch_T"], dtype=torch.float32).unsqueeze(1).to(device)
    c_anch = torch.tensor(raw["anch_c"], dtype=torch.float32).unsqueeze(1).to(device)

    # ── MR targets ────────────────────────────────────────────────────────
    t_mr = torch.tensor(
        raw["t_sec_mr"] / T_total, dtype=torch.float32
    ).to(device)
    MR = torch.tensor(raw["MR"], dtype=torch.float32).to(device)

    # ── Boundary (surface) points at sensor time-steps ────────────────────
    # Generate boundary points on the ellipsoidal surface ξ=1 at sensor times.
    # We scatter points along one side of the cross-section: θ ∈ [-π/2, π/2]
    n_t_bc  = len(raw["t_sec_bc"])
    n_theta = 8    # azimuthal sample points per time step
    theta   = np.linspace(-np.pi / 2, np.pi / 2, n_theta)
    r_surf  = np.cos(theta)            # r_norm at surface (positive)
    z_surf  = np.sin(theta)            # z_norm at surface

    r_bc_list, z_bc_list, t_bc_list, T_air_list, c_air_list = [], [], [], [], []
    for i, t_sec in enumerate(raw["t_sec_bc"]):
        t_n_i = float(t_sec / T_total)
        for j in range(n_theta):
            r_bc_list.append(r_surf[j])
            z_bc_list.append(z_surf[j])
            t_bc_list.append(t_n_i)
            T_air_list.append(float(raw["T_air"][i]))
            c_air_list.append(float(raw["c_air"][i]))

    x_bc = torch.tensor(
        list(zip(r_bc_list, z_bc_list, t_bc_list)), dtype=torch.float32
    ).to(device)
    T_air_bc = torch.tensor(
        T_air_list, dtype=torch.float32
    ).unsqueeze(1).to(device)
    c_air_bc = torch.tensor(
        c_air_list, dtype=torch.float32
    ).unsqueeze(1).to(device)

    return {
        "x_anch":    x_anch,
        "T_anch":    T_anch,
        "c_anch":    c_anch,
        "t_mr":      t_mr,
        "MR":        MR,
        "x_bc":      x_bc,
        "T_air_bc":  T_air_bc,
        "c_air_bc":  c_air_bc,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: load everything at once
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(path=None, device: torch.device = None):
    """
    Returns
    -------
    dataset   : CardamomPINNDataset   (for DataLoader)
    fixed     : dict                  (BC, anchors, MR tensors on device)
    T_air_fn  : callable(t_norm) → np.array  (T_air interpolator)
    T_air_torch : callable(tensor) → tensor
    """
    if device is None:
        device = torch.device("cpu")
    if path is None:
        path = DATA_PATH

    dataset = CardamomPINNDataset(path, augment=True, n_aug=8192)
    raw     = dataset.get_raw()
    fixed   = load_fixed_tensors(raw, device)
    T_air_fn, c_air_fn, RH_air_fn, T_air_torch = build_bc_interpolators(raw)

    return dataset, fixed, T_air_fn, T_air_torch


if __name__ == "__main__":
    ds, fixed, T_air_fn, _ = load_dataset()
    print(f"Collocation points : {len(ds)}")
    print(f"Anchor points      : {fixed['x_anch'].shape}")
    print(f"MR time points     : {fixed['t_mr'].shape}")
    print(f"BC points          : {fixed['x_bc'].shape}")
    print(f"T_air range        : {fixed['T_air_bc'].min():.1f} – {fixed['T_air_bc'].max():.1f} °C")
