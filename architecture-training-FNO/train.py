"""
train.py
========
Training loop for the Cardamom HPDS FNO surrogate.

FNO learns: (T_current, M_current, T_heater) → (T_future, M_future)
            on an 8×16 spatial grid with a 6-minute prediction horizon.

Features
--------
  • PhysicsNeMo Module (AMP flags respected)
  • Adam + cosine-annealing LR with linear warm-up
  • Gradient clipping
  • Physics-guided penalties (monotone moisture, temperature range)
  • TensorBoard logging (every LOG_EVERY epochs)
  • Best-checkpoint saving by validation total loss
  • Resume from checkpoint

Usage
-----
  cd HPDS/architecture-training-FNO
  uv run python train.py                         # full training
  uv run python train.py --epochs 20             # smoke-test
  uv run python train.py --resume checkpoints/best.pt
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

# ─── project imports ─────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    SEED, LR, LR_MIN, WARMUP_STEPS,
    TOTAL_EPOCHS, GRAD_CLIP_NORM,
    LOG_EVERY, SAVE_EVERY, VAL_EVERY,
    DEVICE, RESULTS_DIR, CKPT_DIR,
    LATENT_CHANNELS, NUM_FNO_LAYERS, NUM_MODES_X, NUM_MODES_Y,
    DECODER_HIDDEN, DECODER_LAYERS, BATCH_SIZE,
)
from model   import CardamomFNO
from losses  import FNOLoss, relative_l2, r2_score
from dataset import make_loaders, _denorm_y


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _get_device() -> torch.device:
    if DEVICE == "cuda" and torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[device] GPU — {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("[device] CPU")
    return dev


def _make_lr_lambda(warmup_steps: int, total_steps: int, eta_min_ratio: float):
    """Linear warm-up followed by cosine decay."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return eta_min_ratio + (1.0 - eta_min_ratio) * cosine
    return lr_lambda


def _save_checkpoint(
    model: CardamomFNO,
    optimiser: Adam,
    epoch: int,
    val_loss: float,
    path: Path,
) -> None:
    torch.save({
        "epoch":     epoch,
        "val_loss":  val_loss,
        "model":     model.state_dict(),
        "optimiser": optimiser.state_dict(),
        "model_cfg": {
            "in_channels":     model.in_channels,
            "out_channels":    model.out_channels,
            "latent_channels": model.latent_channels,
            "num_fno_layers":  model.num_fno_layers,
            "modes1":          model.modes1,
            "modes2":          model.modes2,
        },
    }, path)


def _load_checkpoint(
    path: Path,
    model: CardamomFNO,
    optimiser: Adam,
    device: torch.device,
) -> tuple[int, float]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimiser.load_state_dict(ckpt["optimiser"])
    return ckpt["epoch"], ckpt["val_loss"]


# ─────────────────────────────────────────────────────────────────────────────
# Validation pass
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _validate(
    model: CardamomFNO,
    val_loader,
    loss_fn: FNOLoss,
    device: torch.device,
) -> dict:
    """Full validation pass. Returns dict of averaged metrics."""
    model.eval()
    total_loss = 0.0
    logs_sum   = {}
    r2_T_sum = r2_M_sum = 0.0
    n_batches = 0

    for x, y_true in val_loader:
        x      = x.to(device)
        y_true = y_true.to(device)

        y_pred = model(x)
        loss, logs = loss_fn(y_pred, y_true, x)

        total_loss += loss.item()
        for k, v in logs.items():
            logs_sum[k] = logs_sum.get(k, 0.0) + v
        rt, rm = r2_score(y_pred, y_true)
        r2_T_sum += rt
        r2_M_sum += rm
        n_batches += 1

    model.train()
    n = max(n_batches, 1)
    result = {k: v / n for k, v in logs_sum.items()}
    result["r2_T"] = r2_T_sum / n
    result["r2_M"] = r2_M_sum / n
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    torch.manual_seed(SEED)
    device = _get_device()

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    print("[data] Loading datasets ...")
    train_loader, val_loader, _ = make_loaders(batch_size=BATCH_SIZE)
    print(f"[data] Train: {len(train_loader.dataset):,}  "
          f"Val: {len(val_loader.dataset):,}  "
          f"Batch: {BATCH_SIZE}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CardamomFNO(
        latent_channels=LATENT_CHANNELS,
        num_fno_layers=NUM_FNO_LAYERS,
        modes1=NUM_MODES_X,
        modes2=NUM_MODES_Y,
        decoder_hidden=DECODER_HIDDEN,
        decoder_layers=DECODER_LAYERS,
    ).to(device)
    print(f"[model] CardamomFNO  —  {model.count_parameters():,} trainable params")
    print(f"        Latent={LATENT_CHANNELS}  Layers={NUM_FNO_LAYERS}  "
          f"Modes=({NUM_MODES_X},{NUM_MODES_Y})")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    # Must be created BEFORE _load_checkpoint (which restores its state_dict).
    optimiser    = Adam(model.parameters(), lr=LR)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_val    = float("inf")
    if args.resume and Path(args.resume).exists():
        start_epoch, best_val = _load_checkpoint(
            Path(args.resume), model, optimiser, device
        )
        print(f"[resume] Epoch {start_epoch}  val_loss={best_val:.4e}")

    # ── Scheduler ─────────────────────────────────────────────────────────────
    # Build AFTER checkpoint load so scheduler can fast-forward to right LR.
    # Without this, resuming always restarts LR at 1e-3 causing a damaging spike.
    total_epochs = args.epochs or TOTAL_EPOCHS
    lr_lambda    = _make_lr_lambda(
        warmup_steps=WARMUP_STEPS,
        total_steps=total_epochs,
        eta_min_ratio=LR_MIN / LR,
    )
    scheduler = LambdaLR(optimiser, lr_lambda=lr_lambda)
    if start_epoch > 0:
        for _ in range(start_epoch):
            scheduler.step()
        cur_lr = optimiser.param_groups[0]['lr']
        print(f"[resume] LR fast-forwarded to epoch {start_epoch} → lr={cur_lr:.2e}")

    # ── Loss function ─────────────────────────────────────────────────────────
    loss_fn = FNOLoss()

    # AMP: disabled for FNO because SpectralConv2d uses FFT → ComplexHalf
    # which is not supported on most GPUs. model.meta.amp is authoritative.
    use_amp = device.type == "cuda" and model.meta.amp
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── TensorBoard ───────────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=str(RESULTS_DIR / "tb_logs"))

    # ─────────────────────────────────────────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[train] Starting — total {total_epochs} epochs\n")
    t_start = time.time()

    for epoch in range(start_epoch, total_epochs):

        model.train()
        epoch_loss = 0.0
        epoch_logs: dict = {}

        for x, y_true in train_loader:
            x      = x.to(device)
            y_true = y_true.to(device)

            optimiser.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                y_pred = model(x)
                total, logs = loss_fn(y_pred, y_true, x)

            scaler.scale(total).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimiser)
            scaler.update()

            epoch_loss += total.item()
            for k, v in logs.items():
                epoch_logs[k] = epoch_logs.get(k, 0.0) + v

        scheduler.step()

        n_batches  = len(train_loader)
        epoch_loss /= n_batches
        for k in epoch_logs:
            epoch_logs[k] /= n_batches

        # ── Logging ───────────────────────────────────────────────────────────
        if (epoch + 1) % LOG_EVERY == 0 or epoch == 0:
            elapsed = time.time() - t_start
            lr_cur  = optimiser.param_groups[0]["lr"]
            print(
                f"[ep {epoch+1:4d}/{total_epochs}]  "
                f"loss={epoch_loss:.4e}  "
                f"T={epoch_logs.get('loss_T', 0):.4e}  "
                f"M={epoch_logs.get('loss_M', 0):.4e}  "
                f"lr={lr_cur:.2e}  "
                f"t={elapsed:.0f}s"
            )
            writer.add_scalar("Loss/train_total", epoch_loss, epoch)
            writer.add_scalar("LR", lr_cur, epoch)
            for k, v in epoch_logs.items():
                writer.add_scalar(f"Loss/train_{k}", v, epoch)

        # ── Validation & checkpointing ────────────────────────────────────────
        if (epoch + 1) % VAL_EVERY == 0 or (epoch + 1) == total_epochs:
            val_metrics = _validate(model, val_loader, loss_fn, device)
            val_loss    = val_metrics["total"]

            writer.add_scalar("Loss/val_total", val_loss, epoch)
            writer.add_scalar("Val/r2_T", val_metrics["r2_T"], epoch)
            writer.add_scalar("Val/r2_M", val_metrics["r2_M"], epoch)
            for k, v in val_metrics.items():
                writer.add_scalar(f"Val/{k}", v, epoch)

            print(
                f"  → val={val_loss:.4e}  "
                f"R²_T={val_metrics['r2_T']:.4f}  "
                f"R²_M={val_metrics['r2_M']:.4f}  "
                f"(best={best_val:.4e})"
            )

            if val_loss < best_val:
                best_val = val_loss
                _save_checkpoint(model, optimiser, epoch + 1, val_loss,
                                 CKPT_DIR / "best.pt")
                print(f"  ✓ saved best checkpoint  (ep {epoch+1})")

        if (epoch + 1) % SAVE_EVERY == 0:
            _save_checkpoint(model, optimiser, epoch + 1, epoch_loss,
                             CKPT_DIR / f"epoch_{epoch+1:05d}.pt")

    writer.close()
    print(f"\n[done] Training complete.  Best val loss: {best_val:.4e}")
    print(f"       Checkpoints in  {CKPT_DIR}")
    print(f"       Results in      {RESULTS_DIR}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Cardamom HPDS FNO trainer")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override total training epochs (default from config)")
    ap.add_argument("--resume", type=str, default=None,
                    help="Path to checkpoint to resume from")
    ap.add_argument("--batch",  type=int, default=None,
                    help="Override batch size")
    args = ap.parse_args()
    if args.batch:
        import config as _cfg
        _cfg.BATCH_SIZE = args.batch
    train(args)
