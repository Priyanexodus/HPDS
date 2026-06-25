"""
validate.py
===========
Diagnostic validation & visualisation for the Cardamom HPDS FNO.

Loads the best checkpoint and produces:
  1. Predicted vs true T and M fields — spatial heatmaps (sample grid)
  2. Per-channel R² and RMSE on the full test set
  3. Scatter plots: predicted vs true (T and M) on test set
  4. Error distribution histograms per channel

All figures saved to RESULTS_DIR/figures/.

Usage
-----
  cd HPDS/architecture-training-FNO
  uv run python validate.py                          # uses best checkpoint
  uv run python validate.py --ckpt checkpoints/best.pt
  uv run python validate.py --split val              # run on val set instead
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CKPT_DIR, RESULTS_DIR, NX, NY,
    NORM_MEAN_Y, NORM_STD_Y,
    LATENT_CHANNELS, NUM_FNO_LAYERS, NUM_MODES_X, NUM_MODES_Y,
    DECODER_HIDDEN, DECODER_LAYERS,
)
from model   import CardamomFNO
from dataset import make_loaders, load_geometry, _denorm_y
from losses  import r2_score


# ─────────────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":       150,
    "font.family":      "DejaVu Sans",
    "axes.spines.top":  False,
    "axes.spines.right": False,
})
FIG_DIR = RESULTS_DIR / "figures"


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: Path, device: torch.device) -> CardamomFNO:
    ckpt = torch.load(str(ckpt_path), map_location=device)
    cfg  = ckpt.get("model_cfg", {})
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
    epoch = ckpt.get("epoch", "?")
    val   = ckpt.get("val_loss", float("nan"))
    print(f"[validate] Loaded {ckpt_path.name}  epoch={epoch}  val_loss={val:.4e}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 1. Spatial heatmaps — predicted vs true fields
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def plot_field_comparison(model: CardamomFNO, loader, device: torch.device,
                          n_samples: int = 4):
    """
    Plot T and M field predictions vs ground truth for n_samples from loader.
    """
    geo = load_geometry(device)
    pod_mask = geo["pod_mask"].cpu().numpy()    # (8, 16) bool

    x_batch, y_batch = next(iter(loader))
    x_batch = x_batch[:n_samples].to(device)
    y_batch = y_batch[:n_samples].to(device)

    y_pred = model(x_batch)

    # Denormalise to physical units
    y_pred_phys = _denorm_y(y_pred).cpu().numpy()   # (n, 2, 8, 16)
    y_true_phys = _denorm_y(y_batch).cpu().numpy()

    fig = plt.figure(figsize=(14, 3.5 * n_samples))
    gs  = gridspec.GridSpec(n_samples, 6, figure=fig,
                            hspace=0.4, wspace=0.3)

    ch_info = [
        ("T_future",  "°C",           plt.cm.hot),
        ("M_future",  "kg_w/kg_dry",  plt.cm.Blues),
    ]

    for s in range(n_samples):
        col_off = 0
        for ci, (ch_name, unit, cmap) in enumerate(ch_info):
            pred = y_pred_phys[s, ci]   # (8, 16)
            true = y_true_phys[s, ci]

            # Mask non-pod cells
            pred_m = np.where(pod_mask, pred, np.nan)
            true_m = np.where(pod_mask, true, np.nan)
            err_m  = np.where(pod_mask, np.abs(pred - true), np.nan)

            vmin = np.nanmin(true_m);  vmax = np.nanmax(true_m)

            ax_t = fig.add_subplot(gs[s, col_off])
            ax_p = fig.add_subplot(gs[s, col_off + 1])
            ax_e = fig.add_subplot(gs[s, col_off + 2])

            ax_t.imshow(true_m, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
            im_p = ax_p.imshow(pred_m, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
            ax_e.imshow(err_m,  origin="upper", cmap="Reds")

            for ax, title in [(ax_t, f"True {ch_name}"),
                              (ax_p, f"Pred {ch_name}"),
                              (ax_e, f"|Error| [{unit}]")]:
                ax.set_title(f"Sample {s+1} — {title}", fontsize=8)
                ax.axis("off")

            plt.colorbar(im_p, ax=ax_p, fraction=0.046, pad=0.04, label=unit)
            col_off += 3

    out = FIG_DIR / "field_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Full test-set metrics
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_test_metrics(model: CardamomFNO, loader, device: torch.device) -> dict:
    """
    Compute RMSE, MAPE, R² for T and M on the full dataset.
    All in physical (denormalised) space.
    """
    all_pred_T, all_true_T = [], []
    all_pred_M, all_true_M = [], []

    for x, y_true in loader:
        x      = x.to(device)
        y_true = y_true.to(device)
        y_pred = model(x)

        # Denorm
        pred_p = _denorm_y(y_pred)
        true_p = _denorm_y(y_true)

        all_pred_T.append(pred_p[:, 0].cpu())
        all_true_T.append(true_p[:, 0].cpu())
        all_pred_M.append(pred_p[:, 1].cpu())
        all_true_M.append(true_p[:, 1].cpu())

    pred_T = torch.cat(all_pred_T).numpy().ravel()
    true_T = torch.cat(all_true_T).numpy().ravel()
    pred_M = torch.cat(all_pred_M).numpy().ravel()
    true_M = torch.cat(all_true_M).numpy().ravel()

    def _rmse(p, t):
        return float(np.sqrt(np.mean((p - t) ** 2)))

    def _mape(p, t):
        return float(np.mean(np.abs((p - t) / (np.abs(t) + 1e-8))) * 100.0)

    def _r2(p, t):
        ss_res = np.sum((p - t) ** 2)
        ss_tot = np.sum((t - t.mean()) ** 2)
        return 1.0 - ss_res / (ss_tot + 1e-12)

    metrics = {
        "T_RMSE_K":  _rmse(pred_T, true_T),
        "T_MAPE_%":  _mape(pred_T, true_T),
        "T_R2":      _r2(pred_T, true_T),
        "M_RMSE":    _rmse(pred_M, true_M),
        "M_MAPE_%":  _mape(pred_M, true_M),
        "M_R2":      _r2(pred_M, true_M),
    }
    return metrics, pred_T, true_T, pred_M, true_M


# ─────────────────────────────────────────────────────────────────────────────
# 3. Scatter plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_scatter(pred_T, true_T, pred_M, true_M, metrics: dict):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, pred, true, label, unit, r2_key, rmse_key in [
        (axes[0], pred_T, true_T, "Temperature", "°C",          "T_R2", "T_RMSE_K"),
        (axes[1], pred_M, true_M, "Moisture",    "kg_w/kg_dry", "M_R2", "M_RMSE"),
    ]:
        # Subsample for speed
        idx = np.random.choice(len(pred), min(5000, len(pred)), replace=False)
        ax.scatter(true[idx], pred[idx], s=4, alpha=0.35, color="#3b5bdb")
        lo = min(true.min(), pred.min())
        hi = max(true.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.2, label="perfect")
        ax.set(
            xlabel=f"True {label} [{unit}]",
            ylabel=f"Predicted {label} [{unit}]",
            title=(f"{label}  R²={metrics[r2_key]:.4f}  "
                   f"RMSE={metrics[rmse_key]:.4f} {unit}"),
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = FIG_DIR / "scatter_pred_vs_true.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Error distribution histograms
# ─────────────────────────────────────────────────────────────────────────────

def plot_error_histograms(pred_T, true_T, pred_M, true_M):
    err_T = pred_T - true_T
    err_M = pred_M - true_M

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, err, label, unit in [
        (axes[0], err_T, "Temperature error", "°C"),
        (axes[1], err_M, "Moisture error",    "kg_w/kg_dry"),
    ]:
        ax.hist(err, bins=80, color="#4c6ef5", alpha=0.75, edgecolor="white", lw=0.3)
        ax.axvline(0, color="red", lw=1.5, linestyle="--")
        ax.set(xlabel=f"Error [{unit}]", ylabel="Count",
               title=f"{label}  μ={err.mean():.4f}  σ={err.std():.4f}")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = FIG_DIR / "error_histograms.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(args.ckpt) if args.ckpt else (CKPT_DIR / "best.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run train.py first or pass --ckpt <path>."
        )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    model = load_model(ckpt_path, device)

    print("[validate] Loading data loaders ...")
    _, val_loader, test_loader = make_loaders(batch_size=64)
    loader = val_loader if args.split == "val" else test_loader
    split_name = args.split

    print(f"[validate] Running on {split_name} set  ({len(loader.dataset):,} samples)")
    print("[validate] Generating field comparison plots ...")
    plot_field_comparison(model, loader, device, n_samples=4)

    print("[validate] Computing metrics ...")
    metrics, pred_T, true_T, pred_M, true_M = compute_test_metrics(model, loader, device)

    print("[validate] Generating scatter plots ...")
    plot_scatter(pred_T, true_T, pred_M, true_M, metrics)

    print("[validate] Generating error histograms ...")
    plot_error_histograms(pred_T, true_T, pred_M, true_M)

    print(f"\n[validate] ── Summary ({split_name}) ──────────────────────────")
    print(f"  Temperature  RMSE : {metrics['T_RMSE_K']:.4f} °C  "
          f"  MAPE : {metrics['T_MAPE_%']:.2f} %  "
          f"  R²   : {metrics['T_R2']:.4f}")
    print(f"  Moisture     RMSE : {metrics['M_RMSE']:.4f} kg/kg  "
          f"  MAPE : {metrics['M_MAPE_%']:.2f} %  "
          f"  R²   : {metrics['M_R2']:.4f}")
    print(f"\n  Figures → {FIG_DIR}\n")

    # Target checks from roadmap
    T_pass = metrics["T_RMSE_K"] < 0.5
    M_pass = metrics["M_R2"]     > 0.99
    print(f"  [{'✓' if T_pass else '✗'}] T RMSE < 0.5 K  →  {metrics['T_RMSE_K']:.4f}")
    print(f"  [{'✓' if M_pass else '✗'}] M R² > 0.99     →  {metrics['M_R2']:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Validate HPDS FNO")
    ap.add_argument("--ckpt",  type=str, default=None,
                    help="Path to checkpoint (default: checkpoints/best.pt)")
    ap.add_argument("--split", type=str, default="test",
                    choices=["val", "test"],
                    help="Dataset split to evaluate on (default: test)")
    main(ap.parse_args())
