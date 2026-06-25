"""
dataset.py
==========
HDF5 dataset loader for the Cardamom HPDS FNO.

Loads pre-generated (X, y) pairs from the FNO HDF5 files and exposes
them as normalised PyTorch tensors for training.

  X : (N, 3, 8, 16)  — [T_current, M_current, T_heater]   (raw °C / kg_w/kg)
  y : (N, 2, 8, 16)  — [T_future,  M_future]               (raw °C / kg_w/kg)

Normalisation is applied per-channel using training-set statistics
stored in config.py (sourced from metadata.json).
"""

import numpy as np
import torch
import h5py
from torch.utils.data import Dataset, DataLoader

from config import (
    TRAIN_PATH, VAL_PATH, TEST_PATH, GEOMETRY_PATH,
    IN_CHANNELS, OUT_CHANNELS, NX, NY,
    NORM_MEAN_X, NORM_STD_X, NORM_MEAN_Y, NORM_STD_Y,
    BATCH_SIZE, NUM_WORKERS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _norm_x(X: np.ndarray) -> np.ndarray:
    """
    Normalise input channels (N, 3, H, W) → zero-mean unit-variance per channel.
    """
    mean = np.array(NORM_MEAN_X, dtype=np.float32)[:, None, None]   # (3,1,1)
    std  = np.array(NORM_STD_X,  dtype=np.float32)[:, None, None]
    return (X - mean) / std


def _norm_y(y: np.ndarray) -> np.ndarray:
    """
    Normalise output channels (N, 2, H, W) → zero-mean unit-variance per channel.
    """
    mean = np.array(NORM_MEAN_Y, dtype=np.float32)[:, None, None]   # (2,1,1)
    std  = np.array(NORM_STD_Y,  dtype=np.float32)[:, None, None]
    return (y - mean) / std


def _denorm_y(y_norm: torch.Tensor) -> torch.Tensor:
    """
    Inverse normalisation for output tensor (…, 2, H, W).
    Returns physical fields: [T in °C, M in kg_w/kg_dry].
    """
    mean = torch.tensor(NORM_MEAN_Y, dtype=torch.float32,
                        device=y_norm.device).view(1, OUT_CHANNELS, 1, 1)
    std  = torch.tensor(NORM_STD_Y,  dtype=torch.float32,
                        device=y_norm.device).view(1, OUT_CHANNELS, 1, 1)
    return y_norm * std + mean


# ─────────────────────────────────────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────────────────────────────────────

class FNOCardamomDataset(Dataset):
    """
    PyTorch Dataset for FNO cardamom pod drying.

    Parameters
    ----------
    h5_path : path-like
        Path to one of fno_train.h5 / fno_val.h5 / fno_test.h5.
    normalise : bool
        If True (default) apply per-channel normalisation to X and y.
    augment : bool
        If True, apply light spatial augmentations (random flip along y-axis).
        Only used during training.
    """

    def __init__(self, h5_path, normalise: bool = True, augment: bool = False):
        self.h5_path   = str(h5_path)
        self.normalise = normalise
        self.augment   = augment
        self._load()

    def _load(self):
        with h5py.File(self.h5_path, "r") as f:
            X = f["X"][:]    # (N, 3, 8, 16) float32
            y = f["y"][:]    # (N, 2, 8, 16) float32

        if self.normalise:
            X = _norm_x(X)
            y = _norm_y(y)

        self.X = torch.from_numpy(X)    # (N, 3, 8, 16)
        self.y = torch.from_numpy(y)    # (N, 2, 8, 16)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x_i = self.X[idx].clone()
        y_i = self.y[idx].clone()

        if self.augment:
            # Random horizontal flip (y-axis) — preserves physics symmetry
            if torch.rand(1).item() > 0.5:
                x_i = x_i.flip(-1)
                y_i = y_i.flip(-1)
            # Random vertical flip (x-axis)
            if torch.rand(1).item() > 0.5:
                x_i = x_i.flip(-2)
                y_i = y_i.flip(-2)

        return x_i, y_i


# ─────────────────────────────────────────────────────────────────────────────
# Geometry loader (pod mask, surface mask, r-field)
# ─────────────────────────────────────────────────────────────────────────────

def load_geometry(device: torch.device = None) -> dict:
    """
    Load pod geometry arrays from pod_geometry.h5.

    Returns dict with keys:
      pod_mask   : (8, 16) bool tensor   — True inside pod
      surf_mask  : (8, 16) bool tensor   — True at pod surface
      r_field    : (8, 16) float tensor  — normalised radial distance [0, 1]
    """
    if device is None:
        device = torch.device("cpu")

    with h5py.File(str(GEOMETRY_PATH), "r") as f:
        pod_mask  = torch.tensor(f["pod_mask"][:].astype(bool),  device=device)
        surf_mask = torch.tensor(f["surf_mask"][:].astype(bool), device=device)
        r_field   = torch.tensor(f["r_field"][:].astype(np.float32), device=device)

    return {"pod_mask": pod_mask, "surf_mask": surf_mask, "r_field": r_field}


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def make_loaders(batch_size: int = BATCH_SIZE,
                 num_workers: int = NUM_WORKERS):
    """
    Create train, val, and test DataLoaders.

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    train_ds = FNOCardamomDataset(TRAIN_PATH, normalise=True, augment=True)
    val_ds   = FNOCardamomDataset(VAL_PATH,   normalise=True, augment=False)
    test_ds  = FNOCardamomDataset(TEST_PATH,  normalise=True, augment=False)

    pin = torch.cuda.is_available()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tr, va, te = make_loaders(batch_size=32)
    x, y = next(iter(tr))
    print(f"Training samples : {len(tr.dataset):,}")
    print(f"Val     samples  : {len(va.dataset):,}")
    print(f"Test    samples  : {len(te.dataset):,}")
    print(f"Batch X shape    : {x.shape}   dtype={x.dtype}")
    print(f"Batch y shape    : {y.shape}   dtype={y.dtype}")
    print(f"X mean / std     : {x.mean():.4f} / {x.std():.4f}  (expect ≈ 0 / 1)")
    print(f"y mean / std     : {y.mean():.4f} / {y.std():.4f}  (expect ≈ 0 / 1)")
    geo = load_geometry()
    print(f"Pod mask cells   : {geo['pod_mask'].sum().item()}")
    print(f"Surface mask     : {geo['surf_mask'].sum().item()}")
