"""
infer.py
========
Inference utility for the Cardamom HPDS FNO.

Given a pod state (T_current field, M_current field, T_heater scalar),
predicts (T_future, M_future) 6 minutes ahead.

Usage
-----
  cd HPDS/architecture-training-FNO

  # Single-step prediction from a random test sample
  uv run python infer.py --sample

  # Rollout — auto-regressive 6-step (36 min) prediction
  uv run python infer.py --rollout --steps 6

  # From a specific checkpoint
  uv run python infer.py --ckpt checkpoints/best.pt --sample
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CKPT_DIR, RESULTS_DIR,
    NX, NY, IN_CHANNELS, OUT_CHANNELS,
    NORM_MEAN_X, NORM_STD_X, NORM_MEAN_Y, NORM_STD_Y,
    LATENT_CHANNELS, NUM_FNO_LAYERS, NUM_MODES_X, NUM_MODES_Y,
    DECODER_HIDDEN, DECODER_LAYERS,
)
from model   import CardamomFNO
from dataset import FNOCardamomDataset, load_geometry, _denorm_y, TEST_PATH


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: Path, device: torch.device) -> CardamomFNO:
    ckpt  = torch.load(str(ckpt_path), map_location=device)
    cfg   = ckpt.get("model_cfg", {})
    model = CardamomFNO(
        latent_channels=cfg.get("latent_channels", LATENT_CHANNELS),
        num_fno_layers =cfg.get("num_fno_layers",  NUM_FNO_LAYERS),
        modes1         =cfg.get("modes1",           NUM_MODES_X),
        modes2         =cfg.get("modes2",           NUM_MODES_Y),
        decoder_hidden =DECODER_HIDDEN,
        decoder_layers =DECODER_LAYERS,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[infer] Loaded {ckpt_path.name}  —  device: {device}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _norm_input(T: np.ndarray, M: np.ndarray, Th: float) -> torch.Tensor:
    """
    Build a normalised input tensor from physical arrays.

    Parameters
    ----------
    T  : (8, 16) temperature field [°C]
    M  : (8, 16) moisture field    [kg_w/kg_dry]
    Th : float   heater temperature [°C]

    Returns
    -------
    x_norm : (1, 3, 8, 16) float32 tensor
    """
    Th_field = np.full((NX, NY), Th, dtype=np.float32)
    X = np.stack([T, M, Th_field], axis=0)              # (3, 8, 16)
    mean = np.array(NORM_MEAN_X, dtype=np.float32)[:, None, None]
    std  = np.array(NORM_STD_X,  dtype=np.float32)[:, None, None]
    X_norm = (X - mean) / std
    return torch.from_numpy(X_norm).unsqueeze(0)         # (1, 3, 8, 16)


def _denorm_output(y_norm: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """
    Denormalise output tensor.

    Returns T_future (8,16) [°C] and M_future (8,16) [kg_w/kg_dry].
    """
    y_phys = _denorm_y(y_norm)
    T_fut  = y_phys[0, 0].cpu().numpy()    # (8, 16)
    M_fut  = y_phys[0, 1].cpu().numpy()
    return T_fut, M_fut


# ─────────────────────────────────────────────────────────────────────────────
# Prediction function
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_step(
    model: CardamomFNO,
    T: np.ndarray,
    M: np.ndarray,
    T_heater: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One 6-minute forward prediction step.

    Parameters
    ----------
    T        : (8, 16) current temperature field [°C]
    M        : (8, 16) current moisture field    [kg_w/kg_dry]
    T_heater : float   heater setpoint           [°C]
    device   : torch.device

    Returns
    -------
    T_next : (8, 16) temperature 6 min ahead [°C]
    M_next : (8, 16) moisture    6 min ahead [kg_w/kg_dry]
    """
    x_norm = _norm_input(T, M, T_heater).to(device)
    y_norm = model(x_norm)
    return _denorm_output(y_norm)


# ─────────────────────────────────────────────────────────────────────────────
# CLI modes
# ─────────────────────────────────────────────────────────────────────────────

def run_sample(model, device, sample_idx: int = 0):
    """Load one test sample and show prediction vs truth."""
    ds = FNOCardamomDataset(TEST_PATH, normalise=True, augment=False)
    x_norm, y_norm = ds[sample_idx]
    x_norm = x_norm.unsqueeze(0).to(device)
    y_norm = y_norm.unsqueeze(0).to(device)

    y_pred = model(x_norm)

    pred_phys = _denorm_y(y_pred).cpu().numpy()[0]   # (2, 8, 16)
    true_phys = _denorm_y(y_norm).cpu().numpy()[0]

    geo      = load_geometry()
    pod_mask = geo["pod_mask"].cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    titles = [["True T_future [°C]", "Pred T_future [°C]", "|Error T| [°C]"],
              ["True M_future [kg/kg]", "Pred M_future [kg/kg]", "|Error M| [kg/kg]"]]
    cmaps  = [plt.cm.hot, plt.cm.Blues]

    for ci in range(2):
        true = np.where(pod_mask, true_phys[ci], np.nan)
        pred = np.where(pod_mask, pred_phys[ci], np.nan)
        err  = np.where(pod_mask, np.abs(true_phys[ci] - pred_phys[ci]), np.nan)

        vmin, vmax = np.nanmin(true), np.nanmax(true)
        axes[ci, 0].imshow(true, origin="upper", cmap=cmaps[ci], vmin=vmin, vmax=vmax)
        im = axes[ci, 1].imshow(pred, origin="upper", cmap=cmaps[ci], vmin=vmin, vmax=vmax)
        axes[ci, 2].imshow(err,  origin="upper", cmap="Reds")
        plt.colorbar(im, ax=axes[ci, 1], fraction=0.046, pad=0.04)

        for j, title in enumerate(titles[ci]):
            axes[ci, j].set_title(title, fontsize=9)
            axes[ci, j].axis("off")

    rmse_T = float(np.sqrt(np.nanmean((true_phys[0] - pred_phys[0]) ** 2)))
    rmse_M = float(np.sqrt(np.nanmean((true_phys[1] - pred_phys[1]) ** 2)))
    fig.suptitle(f"FNO single-step prediction — sample {sample_idx}\n"
                 f"RMSE_T={rmse_T:.3f}°C   RMSE_M={rmse_M:.5f} kg/kg", fontsize=11)
    fig.tight_layout()

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"infer_sample_{sample_idx}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[infer] RMSE T = {rmse_T:.4f} °C   RMSE M = {rmse_M:.5f} kg/kg")
    print(f"[infer] Figure → {out}")


def run_rollout(model, device, n_steps: int = 6, T_heater: float = 62.0):
    """
    Auto-regressive rollout starting from a random test sample.
    Plots mean T and M over time.
    """
    ds = FNOCardamomDataset(TEST_PATH, normalise=False, augment=False)  # raw
    x_raw, _ = ds[0]    # (3, 8, 16) raw physical

    T_curr = x_raw[0].numpy()   # (8, 16)
    M_curr = x_raw[1].numpy()

    geo      = load_geometry()
    pod_mask = geo["pod_mask"].cpu().numpy()

    T_history = [np.where(pod_mask, T_curr, np.nan).mean()]
    M_history = [np.where(pod_mask, M_curr, np.nan).mean()]
    times_min = [0.0]

    print(f"[infer] Rollout {n_steps} steps × 6 min  (T_heater={T_heater}°C)")
    print(f"  Step 0: T_mean={T_history[0]:.2f}°C   M_mean={M_history[0]:.4f}")

    for step in range(n_steps):
        T_next, M_next = predict_step(model, T_curr, M_curr, T_heater, device)
        # Apply pod mask: outside pod T → T_heater, M → 0
        T_next = np.where(pod_mask, T_next, T_heater)
        M_next = np.where(pod_mask, M_next.clip(0.0, 3.0), 0.0)
        T_curr, M_curr = T_next, M_next
        t_min = (step + 1) * 6.0
        T_mean = np.nanmean(np.where(pod_mask, T_curr, np.nan))
        M_mean = np.nanmean(np.where(pod_mask, M_curr, np.nan))
        T_history.append(T_mean)
        M_history.append(M_mean)
        times_min.append(t_min)
        print(f"  Step {step+1}: T_mean={T_mean:.2f}°C   M_mean={M_mean:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(times_min, T_history, "o-", color="#e03131", lw=2)
    ax1.set(xlabel="Time [min]", ylabel="Mean T [°C]",
            title="FNO Rollout — Mean Temperature")
    ax1.grid(True, alpha=0.3)

    ax2.plot(times_min, M_history, "s-", color="#1971c2", lw=2)
    ax2.set(xlabel="Time [min]", ylabel="Mean M [kg_w/kg_dry]",
            title="FNO Rollout — Mean Moisture")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"FNO Auto-Regressive Rollout  (T_heater={T_heater}°C)", fontsize=11)
    fig.tight_layout()
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "infer_rollout.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[infer] Rollout figure → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.ckpt) if args.ckpt else (CKPT_DIR / "best.pt")

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run train.py first or pass --ckpt <path>."
        )

    model = load_model(ckpt_path, device)

    if args.rollout:
        run_rollout(model, device, n_steps=args.steps,
                    T_heater=args.T_heater)
    else:
        run_sample(model, device, sample_idx=args.idx)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="HPDS FNO inference")
    ap.add_argument("--ckpt",     type=str,   default=None)
    ap.add_argument("--sample",   action="store_true",
                    help="Run single-step prediction on a test sample (default)")
    ap.add_argument("--rollout",  action="store_true",
                    help="Run auto-regressive rollout")
    ap.add_argument("--steps",    type=int,   default=6,
                    help="Number of 6-min rollout steps (default: 6 = 36 min)")
    ap.add_argument("--T_heater", type=float, default=62.0,
                    help="Heater setpoint for rollout [°C] (default: 62)")
    ap.add_argument("--idx",      type=int,   default=0,
                    help="Test sample index for --sample mode (default: 0)")
    main(ap.parse_args())
