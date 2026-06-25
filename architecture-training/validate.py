"""
validate.py
===========
Diagnostic validation & visualisation for the Cardamom HPDS PINN.

Loads the best checkpoint and produces:
  1. T(ξ, t) and c(ξ, t) spatial profiles at selected time slices
  2. MR(t): predicted vs load-cell measurement
  3. Physics residual map over the (r_norm, z_norm) plane at fixed t
  4. Summary metrics (MAPE, RMSE)

All figures are saved to RESULTS_DIR/figures/.

Usage
-----
  cd HPDS/architecture-training
  uv run python validate.py                          # use best checkpoint
  uv run python validate.py --ckpt checkpoints/best.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    CKPT_DIR, RESULTS_DIR, TOTAL_TIME_S,
    B_RADIAL, A_AXIAL, C_INIT, T_SCALE, C_SCALE,
)
from model   import CardamomPINN
from dataset import load_dataset


# ─────────────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":      150,
    "font.family":     "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})
FIG_DIR = RESULTS_DIR / "figures"


# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────

def load_model(ckpt_path: Path, device: torch.device) -> CardamomPINN:
    ckpt = torch.load(str(ckpt_path), map_location=device)
    cfg  = ckpt.get("model_cfg", {})
    model = CardamomPINN(
        n_frequencies=cfg.get("n_frequencies", 64),
        hidden_dim   =cfg.get("hidden_dim",    256),
        n_hidden     =cfg.get("n_hidden",      6),
        skip_every   =cfg.get("skip_every",    2),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    epoch    = ckpt.get("epoch", "?")
    val_loss = ckpt.get("val_loss")
    val_loss_str = f"{val_loss:.4e}" if isinstance(val_loss, (int, float)) else str(val_loss)
    print(f"[validate] Loaded checkpoint: {ckpt_path.name}  "
          f"(epoch={epoch}, val_loss={val_loss_str})")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 1. Spatial profiles: T and c along radial coordinate ξ
# ─────────────────────────────────────────────────────────────────────────────

def plot_spatial_profiles(model: CardamomPINN, device: torch.device):
    """Plot T(ξ) and c(ξ) at several time slices (axial mid-plane z=0)."""
    r_n   = torch.linspace(0.0, 1.0, 200)
    z_n   = torch.zeros(200)
    t_slices  = [0.0, 0.25, 0.5, 0.75, 1.0]
    colours   = plt.cm.plasma(np.linspace(0.15, 0.9, len(t_slices)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    with torch.no_grad():
        for t_n, col in zip(t_slices, colours):
            t_vec = torch.full_like(r_n, t_n)
            x     = torch.stack([r_n, z_n, t_vec], dim=1).to(device)
            T_norm, c_norm = model(x)
            T = (T_norm * T_SCALE).cpu().numpy().ravel()
            c = (c_norm * C_SCALE).cpu().numpy().ravel()
            xi = r_n.numpy()
            lbl = f"t={t_n * TOTAL_TIME_S / 60:.0f} min"
            axes[0].plot(xi, T, color=col, lw=1.8, label=lbl)
            axes[1].plot(xi, c, color=col, lw=1.8, label=lbl)

    axes[0].set(xlabel="ξ (r_norm)", ylabel="Temperature [°C]",
                title="T(ξ, t) — axial mid-plane")
    axes[1].set(xlabel="ξ (r_norm)", ylabel="Moisture conc. [mol/m³]",
                title="c(ξ, t) — axial mid-plane")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = FIG_DIR / "spatial_profiles.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. MR(t): predicted vs load-cell
# ─────────────────────────────────────────────────────────────────────────────

def plot_mr(model: CardamomPINN, fixed: dict, device: torch.device):
    """Plot Moisture Ratio MR(t): PINN volume-average vs load-cell."""
    t_mr  = fixed["t_mr"].cpu().numpy()
    MR_lc = fixed["MR"].cpu().numpy()

    n_quad = 128
    MR_pred = []

    with torch.no_grad():
        for t_n_i in t_mr:
            r_n  = torch.rand(n_quad, 1, device=device)
            z_n  = torch.FloatTensor(n_quad, 1).uniform_(-1.0, 1.0).to(device)
            inside = (r_n.pow(2) + z_n.pow(2)).sqrt() < 1.0
            r_n  = r_n[inside].view(-1, 1)
            z_n  = z_n[inside].view(-1, 1)
            t_q  = torch.full_like(r_n, float(t_n_i))
            x_q  = torch.cat([r_n, z_n, t_q], dim=1)
            _, c_norm = model(x_q)
            c_avg = ((c_norm * C_SCALE * r_n).sum() / r_n.sum()).item()
            MR_pred.append(c_avg / C_INIT)

    MR_pred = np.array(MR_pred)
    t_min   = t_mr * TOTAL_TIME_S / 60.0

    mape = float(np.mean(np.abs((MR_pred - MR_lc) / (MR_lc + 1e-8))) * 100.0)
    rmse = float(np.sqrt(np.mean((MR_pred - MR_lc) ** 2)))
    print(f"  [MR]  MAPE={mape:.2f}%   RMSE={rmse:.4f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_min, MR_lc,   "o-",  lw=2, label="Load cell (measured)", color="#2c7bb6")
    ax.plot(t_min, MR_pred, "s--", lw=2, label=f"PINN prediction  (MAPE={mape:.1f}%)",
            color="#d7191c")
    ax.set(xlabel="Time [min]", ylabel="Moisture Ratio (MR)",
           title="MR(t): PINN vs Load Cell")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "mr_prediction.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {out.name}")
    return mape, rmse


# ─────────────────────────────────────────────────────────────────────────────
# 3. Physics residual map
# ─────────────────────────────────────────────────────────────────────────────

def plot_residual_map(model: CardamomPINN, device: torch.device, t_norm: float = 0.5):
    """Heatmap of |physics residual| in the (r_norm, z_norm) plane at fixed t."""
    nr, nz = 60, 60
    r_g = torch.linspace(0.0, 1.0, nr)
    z_g = torch.linspace(-1.0, 1.0, nz)
    R, Z = torch.meshgrid(r_g, z_g, indexing="ij")

    inside = (R.pow(2) + Z.pow(2)) < 1.0

    r_flat = R[inside]
    z_flat = Z[inside]
    t_flat = torch.full_like(r_flat, t_norm)
    x_flat = torch.stack([r_flat, z_flat, t_flat], dim=1).to(device)
    x_flat.requires_grad_(True)

    T_norm, c_norm = model(x_flat)

    from losses import _grad, _laplacian_spherical, ALPHA_T, _D_eff
    from config  import TOTAL_TIME_S as TTS

    dT_dt = _grad(T_norm, x_flat)[:, 2:3] * (T_SCALE / TTS)
    dc_dt = _grad(c_norm, x_flat)[:, 2:3] * (C_SCALE / TTS)

    lap_T  = _laplacian_spherical(T_norm, x_flat) * T_SCALE
    T_phys = T_norm.detach() * T_SCALE
    D      = _D_eff(T_phys)
    lap_c  = _laplacian_spherical(c_norm, x_flat) * C_SCALE

    from config import EA_MOIST, R_GAS, B_RADIAL, A_AXIAL
    T_K = T_phys + 273.15
    dD_dT = D * (EA_MOIST / (R_GAS * T_K.pow(2)))
    g_T_full = _grad(T_norm, x_flat)
    g_c_full = _grad(c_norm, x_flat)
    
    dT_dr = g_T_full[:, 0:1] * (T_SCALE / B_RADIAL)
    dT_dz = g_T_full[:, 1:2] * (T_SCALE / A_AXIAL)
    dD_dr = dD_dT * dT_dr.detach()
    dD_dz = dD_dT * dT_dz.detach()
    
    dc_dr = g_c_full[:, 0:1] * (C_SCALE / B_RADIAL)
    dc_dz = g_c_full[:, 1:2] * (C_SCALE / A_AXIAL)
    gradD_dot_gradc = dD_dr * dc_dr + dD_dz * dc_dz

    res_T = (dT_dt - ALPHA_T * lap_T).abs().detach().cpu().numpy().ravel()
    res_c = (dc_dt - (D * lap_c + gradD_dot_gradc)).abs().detach().cpu().numpy().ravel()

    # Build 2-D maps
    R_np = R.numpy(); Z_np = Z.numpy()
    ins  = inside.numpy()

    map_T = np.full((nr, nz), np.nan)
    map_c = np.full((nr, nz), np.nan)
    map_T[ins] = res_T
    map_c[ins] = res_c

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, title, unit in zip(
        axes,
        [map_T, map_c],
        ["Heat eq. residual |∂T/∂t − α∇²T|",
         "Moisture eq. residual |∂c/∂t − D∇²c|"],
        ["°C/s", "mol/(m³·s)"],
    ):
        im = ax.pcolormesh(
            Z_np, R_np, data,
            norm=mcolors.LogNorm(vmin=1e-6, vmax=None),
            cmap="inferno", shading="auto",
        )
        plt.colorbar(im, ax=ax, label=unit)
        # Draw ellipse outline
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.sin(theta), np.abs(np.cos(theta)), "w--", lw=0.8, alpha=0.7)
        ax.set(xlabel="z_norm", ylabel="r_norm", title=title)

    t_min_label = t_norm * TOTAL_TIME_S / 60.0
    fig.suptitle(f"Physics residual map  (t = {t_min_label:.0f} min)", fontsize=13)
    fig.tight_layout()
    out = FIG_DIR / f"residual_map_t{int(t_min_label):03d}min.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. COMSOL anchor scatter
# ─────────────────────────────────────────────────────────────────────────────

def plot_anchor_comparison(model: CardamomPINN, fixed: dict, device: torch.device):
    """Scatter plot of predicted vs target T and c at COMSOL anchor pts."""
    with torch.no_grad():
        T_norm, c_norm = model(fixed["x_anch"])
        T_pred = (T_norm * T_SCALE).cpu().numpy().ravel()
        c_pred = (c_norm * C_SCALE).cpu().numpy().ravel()
    T_true = fixed["T_anch"].cpu().numpy().ravel()
    c_true = fixed["c_anch"].cpu().numpy().ravel()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, pred, true, lbl, unit in zip(
        axes,
        [T_pred, c_pred],
        [T_true, c_true],
        ["Temperature", "Concentration"],
        ["°C", "mol/m³"],
    ):
        lo = min(pred.min(), true.min())
        hi = max(pred.max(), true.max())
        ax.scatter(true, pred, s=10, alpha=0.6, color="#4575b4")
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.2, label="perfect")
        rmse = np.sqrt(np.mean((pred - true) ** 2))
        ax.set(xlabel=f"COMSOL {lbl} [{unit}]",
               ylabel=f"PINN {lbl} [{unit}]",
               title=f"{lbl}  RMSE={rmse:.3f} {unit}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = FIG_DIR / "anchor_scatter.png"
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

    print("[validate] Loading dataset ...")
    _, fixed, _, _ = load_dataset(device=device)

    print("[validate] Generating plots ...")
    plot_spatial_profiles(model, device)
    mape, rmse = plot_mr(model, fixed, device)
    plot_residual_map(model, device, t_norm=0.5)
    plot_anchor_comparison(model, fixed, device)

    print(f"\n[validate] Summary")
    print(f"  MR MAPE : {mape:.2f} %")
    print(f"  MR RMSE : {rmse:.4f}")
    print(f"  Figures  → {FIG_DIR}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Validate HPDS PINN")
    ap.add_argument("--ckpt", type=str, default=None,
                    help="Path to checkpoint (default: checkpoints/best.pt)")
    main(ap.parse_args())
