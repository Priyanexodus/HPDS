"""
infer.py
========
Inference utility for the Cardamom HPDS PINN.

Given physical coordinates (r [m], z [m], t [s]), predicts T [°C] and c [mol/m³].

Usage
-----
  cd HPDS/architecture-training

  # Single point
  uv run python infer.py --r 0.0 --z 0.0 --t 3600

  # Grid evaluation → saves CSV
  uv run python infer.py --grid --t 7200 --out results/grid_t7200.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from config import A_AXIAL, B_RADIAL, C_SCALE, T_SCALE, TOTAL_TIME_S, CKPT_DIR
from model  import CardamomPINN


# ─────────────────────────────────────────────────────────────────────────────
# Loader
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
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Inference function
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    model: CardamomPINN,
    r: np.ndarray,
    z: np.ndarray,
    t: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict T [°C] and c [mol/m³] at physical coordinates.

    Parameters
    ----------
    r : (N,) radial   [m],  0 ≤ r ≤ B_RADIAL
    z : (N,) axial    [m], -A_AXIAL ≤ z ≤ A_AXIAL
    t : (N,) time     [s],  0 ≤ t ≤ TOTAL_TIME_S
    """
    r_n = torch.tensor(r / B_RADIAL,     dtype=torch.float32, device=device)
    z_n = torch.tensor(z / A_AXIAL,      dtype=torch.float32, device=device)
    t_n = torch.tensor(t / TOTAL_TIME_S, dtype=torch.float32, device=device)
    x   = torch.stack([r_n, z_n, t_n], dim=-1)

    with torch.no_grad():
        T_phys, c_phys = model.predict_physical(x)

    return T_phys.cpu().numpy().ravel(), c_phys.cpu().numpy().ravel()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.ckpt) if args.ckpt else (CKPT_DIR / "best.pt")

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run train.py first, or pass --ckpt <path>."
        )

    model = load_model(ckpt_path, device)
    print(f"[infer] Loaded {ckpt_path.name}  —  device: {device}")

    if args.grid:
        # ── Grid evaluation ──────────────────────────────────────────────
        nr, nz = 40, 40
        r_vec  = np.linspace(0.0, B_RADIAL, nr)
        z_vec  = np.linspace(-A_AXIAL, A_AXIAL, nz)
        R, Z   = np.meshgrid(r_vec, z_vec, indexing="ij")
        t_arr  = np.full(R.ravel().shape, float(args.t))

        inside = (R / B_RADIAL) ** 2 + (Z / A_AXIAL) ** 2 < 1.0
        r_flat = R.ravel()[inside.ravel()]
        z_flat = Z.ravel()[inside.ravel()]

        T_pred, c_pred = predict(model, r_flat, z_flat,
                                 np.full(r_flat.shape, float(args.t)), device)

        import pandas as pd
        df = pd.DataFrame({
            "r_m": r_flat, "z_m": z_flat,
            "t_s": float(args.t),
            "T_C": T_pred, "c_mol_m3": c_pred,
        })
        out = Path(args.out) if args.out else Path(f"results/grid_t{int(args.t):06d}s.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"[infer] Grid saved → {out}  ({len(df)} interior pts)")

    else:
        # ── Single-point prediction ──────────────────────────────────────
        r = np.array([float(args.r)])
        z = np.array([float(args.z)])
        t = np.array([float(args.t)])
        T, c = predict(model, r, z, t, device)
        print(f"\n[infer] Point prediction")
        print(f"  r = {r[0]*1e3:.2f} mm   z = {z[0]*1e3:.2f} mm   t = {t[0]/60:.1f} min")
        print(f"  T = {T[0]:.2f} °C")
        print(f"  c = {c[0]:.1f} mol/m³")
        mr = c[0] / (47082.0)   # C_INIT
        print(f"  MR ≈ {mr:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="HPDS PINN inference")
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--r",    type=float, default=0.0,  help="Radial coord [m]")
    ap.add_argument("--z",    type=float, default=0.0,  help="Axial coord  [m]")
    ap.add_argument("--t",    type=float, default=0.0,  help="Time [s]")
    ap.add_argument("--grid", action="store_true",
                    help="Evaluate on a 2-D grid at the given time")
    ap.add_argument("--out",  type=str, default=None,
                    help="Output CSV path for grid evaluation")
    main(ap.parse_args())
