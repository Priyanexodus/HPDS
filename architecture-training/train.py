"""
train.py
========
Three-phase PINN training loop for Cardamom HPDS drying.

Phase 1  (epochs 0 → PHASE1_EPOCHS):
  Adam on IC + BC + data loss only — warms up the network.

Phase 2  (epochs PHASE1_EPOCHS → TOTAL_EPOCHS):
  Full PINN: all seven loss terms active.

Phase 3  (L-BFGS refinement after Adam):
  Full-batch quasi-Newton optimiser using the strong Wolfe line search.
  Follows Raissi et al. (2019) recommendation: Adam first, then L-BFGS.
  Each epoch calls torch.optim.LBFGS.step(closure) with up to
  LBFGS_MAX_ITER inner iterations. Uses a fixed collocation batch
  (no DataLoader) so the Hessian estimate stays consistent.

Features
--------
  • PhysicsNeMo Module (auto-diff/AMP flags respected)
  • Adam + cosine-annealing LR with linear warm-up
  • Gradient clipping (max_norm)
  • Adaptive loss weight rebalancing every ADAPTIVE_LAMBDA_EVERY epochs
  • L-BFGS Phase 3 with strong Wolfe line search
  • TensorBoard logging
  • Best-checkpoint saving by validation total loss

Usage
-----
  cd HPDS/architecture-training
  uv run python train.py                        # full Adam (P1+P2)
  uv run python train.py --lbfgs               # Adam P1+P2 then L-BFGS P3
  uv run python train.py --lbfgs-only           # skip Adam, run L-BFGS only
  uv run python train.py --resume checkpoints/best.pt --lbfgs
  uv run python train.py --epochs 500          # smoke-test
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch
from torch.optim import Adam, LBFGS
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ─── project imports ─────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    SEED, BATCH_COLL, LR, LR_MIN, WARMUP_STEPS,
    PHASE1_EPOCHS, PHASE2_EPOCHS, TOTAL_EPOCHS,
    ADAPTIVE_LAMBDA_EVERY, GRAD_CLIP_NORM,
    LOG_EVERY, SAVE_EVERY, VAL_EVERY,
    LBFGS_EPOCHS, LBFGS_MAX_ITER, LBFGS_HISTORY_SIZE,
    LBFGS_LR, LBFGS_COLL_PTS, LBFGS_LOG_EVERY, LBFGS_SAVE_EVERY, LBFGS_VAL_THRESHOLD,
    DEVICE, RESULTS_DIR, CKPT_DIR,
    HIDDEN_DIM, N_HIDDEN, FOURIER_FEATURES,
)
from model   import CardamomPINN
from losses  import PINNLoss
from dataset import load_dataset


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


def _adaptive_lambda_update(
    model: CardamomPINN,
    loss_fn: PINNLoss,
    batch: dict,
    fixed: dict,
    device: torch.device,
) -> None:
    """
    NTK-inspired loss rebalancing:
    normalise each λ so that λ·|∇_θ L_k| is equal across all active terms.
    Applied only in Phase 2.
    """
    if loss_fn.phase != 2:
        return

    grad_norms = {}
    for key in ("ic_T", "ic_c", "non_neg", "bc_T", "bc_c", "data_T", "data_c", "mr", "physics_T", "physics_c"):
        model.zero_grad()
        # Compute individual loss
        with torch.enable_grad():
            if key == "ic_T":
                from losses import loss_ic
                val, _ = loss_ic(model, device)
            elif key == "ic_c":
                from losses import loss_ic
                _, val = loss_ic(model, device)
            elif key == "non_neg":
                from losses import loss_non_neg
                val = loss_non_neg(model, device)
            elif key == "bc_T":
                from losses import loss_bc_temperature
                val = loss_bc_temperature(model, fixed["x_bc"], fixed["T_air_bc"])
            elif key == "bc_c":
                from losses import loss_bc_concentration
                val = loss_bc_concentration(model, fixed["x_bc"], fixed["c_air_bc"])
            elif key == "data_T":
                from losses import loss_data
                val, _ = loss_data(model, fixed["x_anch"], fixed["T_anch"], fixed["c_anch"])
            elif key == "data_c":
                from losses import loss_data
                _, val = loss_data(model, fixed["x_anch"], fixed["T_anch"], fixed["c_anch"])
            elif key == "mr":
                from losses import loss_mr
                val = loss_mr(model, fixed["t_mr"], fixed["MR"], device=device)
            elif key in ("physics_T", "physics_c"):
                from losses import loss_physics
                lt, lc = loss_physics(model, batch["x_coll"])
                val = lt if key == "physics_T" else lc
            else:
                continue

        val.backward()
        gnorm = sum(
            p.grad.norm().item() ** 2
            for p in model.parameters() if p.grad is not None
        ) ** 0.5
        grad_norms[key] = max(gnorm, 1e-8)

    model.zero_grad()

    if not grad_norms:
        return

    # Normalise: λ_k ← mean_norm / grad_norm_k
    # We use an Exponential Moving Average (EMA) to update the weights
    # safely without the exponential compounding bug (multiplying by current lam[k]).
    mean_norm = sum(grad_norms.values()) / len(grad_norms)
    
    alpha = 0.02   # very conservative: slow EMA so ic_c/mr weights don't erode during Phase 2
    new_lam = {}
    for k in grad_norms:
        hat_lam = mean_norm / grad_norms[k]
        # Hard clamp: prevents near-zero grad_norms from producing astronomically
        # large hat_lam values that EMA cannot damp quickly enough → explosion.
        # Upper bound raised to 200 so ic_c and mr can stay dominant.
        hat_lam = float(max(0.1, min(hat_lam, 200.0)))
        new_lam[k] = (1.0 - alpha) * loss_fn.lam[k] + alpha * hat_lam
        
    loss_fn.update_lambdas(new_lam)


def _save_checkpoint(
    model: CardamomPINN,
    optimiser: Adam,
    epoch: int,
    val_loss: float,
    path: Path,
) -> None:
    torch.save({
        "epoch":      epoch,
        "val_loss":   val_loss,
        "model":      model.state_dict(),
        "optimiser":  optimiser.state_dict(),
        "model_cfg": {
            "n_frequencies": model.n_frequencies,
            "hidden_dim":    model.hidden_dim,
            "n_hidden":      model.n_hidden,
            "skip_every":    model.skip_every,
        },
    }, path)


def _load_checkpoint(path: Path, model: CardamomPINN, optimiser: Adam, device: torch.device, load_opt: bool = True):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if load_opt:
        try:
            optimiser.load_state_dict(ckpt["optimiser"])
        except Exception as e:
            print(f"[resume] Warning: Could not load optimiser state ({type(e).__name__}: {e})")
    return ckpt["epoch"], ckpt["val_loss"]


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    torch.manual_seed(SEED)
    device = _get_device()

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────────────────────
    print("[data] Loading dataset ...")
    dataset, fixed, T_air_fn, T_air_torch = load_dataset(device=device)
    loader = DataLoader(dataset, batch_size=BATCH_COLL, shuffle=True,
                        num_workers=0, pin_memory=(device.type == "cuda"))
    print(f"[data] Collocation: {len(dataset):,}  |  Anchors: {fixed['x_anch'].shape[0]}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CardamomPINN(
        n_frequencies=FOURIER_FEATURES,
        hidden_dim=HIDDEN_DIM,
        n_hidden=N_HIDDEN,
    ).to(device)
    print(f"[model] {model.count_parameters():,} trainable parameters")

    # ── Optimiser ────────────────────────────────────────────────────────────
    # Must be created BEFORE _load_checkpoint (which restores its state_dict).
    optimiser = Adam(model.parameters(), lr=LR)

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 0
    best_val    = float("inf")
    if args.resume and Path(args.resume).exists():
        start_epoch, best_val = _load_checkpoint(
            Path(args.resume), model, optimiser, device, load_opt=not args.lbfgs_only
        )
        print(f"[resume] Epoch {start_epoch}  val_loss={best_val:.4e}")

    # ── Scheduler ─────────────────────────────────────────────────────────────
    # Build AFTER checkpoint load so we can fast-forward to the correct LR.
    # Without this, resuming always restarts LR at 1e-3 causing a damaging spike.
    total_epochs = args.epochs or TOTAL_EPOCHS
    lr_lambda  = _make_lr_lambda(
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
    phase    = 1 if start_epoch < PHASE1_EPOCHS else 2
    loss_fn  = PINNLoss(phase=phase)

    # ── TensorBoard ───────────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=str(RESULTS_DIR / "tb_logs"))

    # ─────────────────────────────────────────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────────────────────────────────────────
    if not args.lbfgs_only:
        print(f"\n[train] Starting — total {total_epochs} epochs\n")
        t_start = time.time()

        for epoch in range(start_epoch, total_epochs):

            # Switch to Phase 2
            if epoch == PHASE1_EPOCHS and loss_fn.phase == 1:
                loss_fn.set_phase(2)
                print(f"\n[phase] Switching to Phase 2 (full PINN) at epoch {epoch}\n")

            model.train()
            epoch_loss = 0.0
            epoch_logs = {}

            for batch_coords in loader:
                batch_coords = batch_coords.to(device)
                batch = {"x_coll": batch_coords, **fixed}

                optimiser.zero_grad()
                total, logs = loss_fn(model, batch, device, n_ic_pts=256)

                if not torch.isfinite(total):
                    print(f"[warn] non-finite loss at epoch {epoch} — skipping optimizer step")
                    optimiser.zero_grad()
                    continue

                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimiser.step()

                epoch_loss += total.item()
                for k, v in logs.items():
                    epoch_logs[k] = epoch_logs.get(k, 0.0) + v

            scheduler.step()

            n_batches = len(loader)
            epoch_loss /= n_batches
            for k in epoch_logs:
                epoch_logs[k] /= n_batches

            # ── Logging ───────────────────────────────────────────────────────────
            if (epoch + 1) % LOG_EVERY == 0 or epoch == 0:
                elapsed = time.time() - t_start
                lr_cur  = optimiser.param_groups[0]["lr"]
                phase_s = "P1" if loss_fn.phase == 1 else "P2"
                print(
                    f"[{phase_s}] ep {epoch+1:5d}/{total_epochs}  "
                    f"loss={epoch_loss:.4e}  lr={lr_cur:.2e}  "
                    f"t={elapsed:.0f}s"
                )
                writer.add_scalar("Loss/total",    epoch_loss, epoch)
                writer.add_scalar("LR",            lr_cur,     epoch)
                for k, v in epoch_logs.items():
                    writer.add_scalar(f"Loss/{k}", v, epoch)

            # ── Adaptive lambda rebalancing ───────────────────────────────────────
            if loss_fn.phase == 2 and (epoch + 1) % ADAPTIVE_LAMBDA_EVERY == 0:
                _adaptive_lambda_update(model, loss_fn, batch, fixed, device)
                writer.add_scalars("Lambdas", loss_fn.lam, epoch)
            # ── Validation & checkpointing ────────────────────────────────────────
            if (epoch + 1) % VAL_EVERY == 0 or (epoch + 1) == total_epochs:
                val_loss = _validate(model, fixed, loss_fn, device, T_air_torch)
                writer.add_scalar("Val/total", val_loss, epoch)
                print(f"  → val_loss = {val_loss:.4e}  (best={best_val:.4e})")

                if val_loss < best_val:
                    best_val = val_loss
                    _save_checkpoint(model, optimiser, epoch + 1, val_loss,
                                     CKPT_DIR / "best.pt")
                    print(f"  ✓ saved best checkpoint  (ep {epoch+1})")

            if (epoch + 1) % SAVE_EVERY == 0:
                _save_checkpoint(model, optimiser, epoch + 1, epoch_loss,
                                 CKPT_DIR / f"epoch_{epoch+1:05d}.pt")

        print(f"\n[Adam done] Best val loss: {best_val:.4e}")

    # ── Phase 3: L-BFGS refinement ───────────────────────────────────────────
    if args.lbfgs or args.lbfgs_only:
        if best_val >= LBFGS_VAL_THRESHOLD:
            print(f"[phase3] Skipped — val_loss {best_val:.3e} not below "
                  f"{LBFGS_VAL_THRESHOLD:.2f}; raise LBFGS_VAL_THRESHOLD in config.py if Adam has plateaued.")
        else:
            best_val = _lbfgs_phase(
                model, loss_fn, fixed, writer, device, best_val, start_epoch
            )

    writer.close()
    print(f"\n[done] Training complete.  Best val loss: {best_val:.4e}")
    print(f"       Checkpoints in  {CKPT_DIR}")
    print(f"       Results in      {RESULTS_DIR}")


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3 — L-BFGS quasi-Newton refinement
# ──────────────────────────────────────────────────────────────────────────────

def _lbfgs_phase(
    model: CardamomPINN,
    loss_fn: PINNLoss,
    fixed: dict,
    writer: SummaryWriter,
    device: torch.device,
    best_val: float,
    start_epoch: int = 0,
) -> float:
    """
    Phase 3: L-BFGS quasi-Newton refinement.

    L-BFGS is a FULL-BATCH optimizer — we fix one large collocation batch
    for the entire phase so the curvature estimate stays consistent.
    The closure recomputes loss + backward on every L-BFGS inner step.

    Returns the updated best_val.
    """
    lbfgs_start = max(0, start_epoch - TOTAL_EPOCHS)
    print(f"\n[phase3] Starting L-BFGS refinement  "
          f"(epochs={LBFGS_EPOCHS}, max_iter={LBFGS_MAX_ITER}, "
          f"history={LBFGS_HISTORY_SIZE}, starting from ep {lbfgs_start})")

    lbfgs = LBFGS(
        model.parameters(),
        lr=LBFGS_LR,
        max_iter=LBFGS_MAX_ITER,
        max_eval=int(LBFGS_MAX_ITER * 1.25),
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=LBFGS_HISTORY_SIZE,
        line_search_fn="strong_wolfe",
    )

    # NOTE: Do NOT use a LR scheduler with L-BFGS + strong_wolfe.
    # The Wolfe line search already computes the optimal step size internally.
    # An external scheduler corrupts the curvature memory (Hessian approximation)
    # built when lr=1.0, causing NaN when the effective lr shrinks too far.
    # lr=1.0 is the correct and standard setting (Raissi et al. 2019).

    # Fix one large collocation batch for the whole phase
    # requires_grad=True is ESSENTIAL: torch.autograd.grad() in the BC/physics
    # loss differentiates w.r.t. the input coordinates.
    rng    = torch.Generator(device="cpu")
    r_n    = torch.rand(LBFGS_COLL_PTS, 1, generator=rng)
    z_n    = torch.FloatTensor(LBFGS_COLL_PTS, 1).uniform_(-1.0, 1.0)
    t_n    = torch.rand(LBFGS_COLL_PTS, 1, generator=rng)
    inside = (r_n.pow(2) + z_n.pow(2)).sqrt() < 1.0
    r_n    = r_n[inside].view(-1, 1).to(device)
    z_n    = z_n[inside].view(-1, 1).to(device)
    t_n    = t_n[inside[:, 0]].view(-1, 1).to(device)
    x_coll_fixed = torch.cat([r_n, z_n, t_n], dim=1).requires_grad_(True)  # (M, 3)

    # Also enable grad on x_bc — loss_bc_temperature calls autograd.grad w.r.t. x_bc
    fixed = {k: (v.requires_grad_(True) if k == "x_bc" else v)
             for k, v in fixed.items()}

    # Full Phase 2 loss (physics + data + IC + BC + MR)
    loss_fn.set_phase(2)

    t_start   = time.time()
    epoch_off = TOTAL_EPOCHS   # offset for TensorBoard global step

    # Build batch once — x_coll_fixed and fixed are constant for all L-BFGS epochs.
    # Defining batch + closure outside the loop avoids Python's by-reference
    # closure capture issue (stale variable if batch were reassigned per epoch).
    batch = {"x_coll": x_coll_fixed, **fixed}

    _last_finite_loss = [None]   # mutable container so closure can update it

    def closure():
        lbfgs.zero_grad()
        total, _ = loss_fn(model, batch, device, n_ic_pts=512)
        if not torch.isfinite(total):
            # Bad step — return last known finite loss and zero grads so L-BFGS
            # can backtrack via the Wolfe conditions instead of poisoning the
            # Hessian memory with NaN gradients.
            lbfgs.zero_grad()
            if _last_finite_loss[0] is not None:
                return _last_finite_loss[0]
            return total   # first call — nothing to fall back to
        _last_finite_loss[0] = total.detach()
        total.backward()
        return total

    nan_streak = 0   # consecutive NaN epochs; abort if too many

    for epoch in range(lbfgs_start, LBFGS_EPOCHS):

        loss_val = lbfgs.step(closure)
        # no scheduler — lr stays at 1.0 for strong_wolfe


        # ── Logging ───────────────────────────────────────────────────────
        if (epoch + 1) % LBFGS_LOG_EVERY == 0 or epoch == 0:
            elapsed = time.time() - t_start
            lr_cur = lbfgs.param_groups[0]["lr"]
            # Re-evaluate for logging breakdown
            _, logs = loss_fn(model, batch, device, n_ic_pts=512)
            loss_finite = loss_val if (loss_val is not None and torch.isfinite(torch.tensor(float(loss_val)))) else float("nan")
            print(
                f"[L-BFGS ep {epoch+1:4d}/{LBFGS_EPOCHS}]  "
                f"loss={loss_finite:.4e}  lr={lr_cur:.2e}  "
                f"T={logs.get('physics_T', 0):.3e}  "
                f"c={logs.get('physics_c', 0):.3e}  "
                f"data={logs.get('data_T', 0):.3e}  "
                f"t={elapsed:.0f}s"
            )
            if math.isfinite(loss_finite):
                writer.add_scalar("Loss/lbfgs_total", loss_finite, epoch_off + epoch)
                writer.add_scalar("LR_lbfgs", lr_cur, epoch_off + epoch)
                for k, v in logs.items():
                    writer.add_scalar(f"Loss/lbfgs_{k}", float(v), epoch_off + epoch)

        # ── NaN streak abort ──────────────────────────────────────────────
        if loss_val is None or not math.isfinite(float(loss_val)):
            nan_streak += 1
            if nan_streak >= 5:
                print(f"[phase3] NaN loss for {nan_streak} consecutive epochs — "
                      f"aborting early. Best val so far: {best_val:.4e}")
                break
        else:
            nan_streak = 0

        # ── Validation & checkpointing ─────────────────────────────────────
        if (epoch + 1) % LBFGS_SAVE_EVERY == 0 or (epoch + 1) == LBFGS_EPOCHS:
            val_loss = _validate(model, fixed, loss_fn, device, None)
            writer.add_scalar("Val/lbfgs_total", val_loss, epoch_off + epoch)
            print(f"  → val_loss = {val_loss:.4e}  (best={best_val:.4e})")

            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    "epoch":    TOTAL_EPOCHS + epoch + 1,
                    "val_loss": val_loss,
                    "model":    model.state_dict(),
                    "optimiser": lbfgs.state_dict(),
                    "model_cfg": {
                        "n_frequencies": model.n_frequencies,
                        "hidden_dim":    model.hidden_dim,
                        "n_hidden":      model.n_hidden,
                        "skip_every":    model.skip_every,
                    },
                }, CKPT_DIR / "best_lbfgs.pt")
                print(f"  ✓ saved best L-BFGS checkpoint  (ep {epoch+1})")

            torch.save({
                "epoch":    TOTAL_EPOCHS + epoch + 1,
                "val_loss": val_loss,
                "model":    model.state_dict(),
                "optimiser": lbfgs.state_dict(),
                "model_cfg": {
                    "n_frequencies": model.n_frequencies,
                    "hidden_dim":    model.hidden_dim,
                    "n_hidden":      model.n_hidden,
                    "skip_every":    model.skip_every,
                },
            }, CKPT_DIR / f"lbfgs_epoch_{epoch+1:05d}.pt")

    print(f"[phase3] L-BFGS done.  Best val loss: {best_val:.4e}")
    return best_val


# ──────────────────────────────────────────────────────────────────────────────
# Validation pass (no physics grad — faster)
# ──────────────────────────────────────────────────────────────────────────────

def _validate(
    model: CardamomPINN,
    fixed: dict,
    loss_fn: PINNLoss,
    device: torch.device,
    T_air_torch,
) -> float:
    from losses import loss_data, loss_mr
    model.eval()
    with torch.no_grad():   # covers ALL terms — avoids building a compute graph during validation
        L_dT, L_dc = loss_data(model, fixed["x_anch"], fixed["T_anch"], fixed["c_anch"])
        L_mr_val = loss_mr(model, fixed["t_mr"], fixed["MR"], device=device)
    model.train()
    return (L_dT + L_dc + L_mr_val).item()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Cardamom HPDS PINN trainer")
    ap.add_argument("--epochs",     type=int,  default=None,
                    help="Override total Adam epochs (default from config)")
    ap.add_argument("--resume",     type=str,  default=None,
                    help="Path to checkpoint to resume from")
    ap.add_argument("--lbfgs",      action="store_true",
                    help="Run L-BFGS Phase 3 after Adam training")
    ap.add_argument("--lbfgs-only", action="store_true",
                    help="Skip Adam, load checkpoint and run L-BFGS only")
    args = ap.parse_args()

    if args.lbfgs_only and not args.resume:
        ap.error("--lbfgs-only requires --resume <checkpoint>")

    train(args)
