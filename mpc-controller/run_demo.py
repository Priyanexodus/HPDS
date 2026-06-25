"""
run_demo.py
===========
End-to-end closed-loop MPC demo for the Cardamom HPDS Digital Twin.

What this script does
----------------------
1. Loads the trained PINN and FNO checkpoints
2. Builds the state estimator (PINN + UKF) and MPC controller
3. Runs a full synthetic 8.55-hour drying simulation
4. Saves 5 diagnostic plots to results/

Usage
-----
    cd HPDS/mpc-controller

    # Full 8.55-hour simulation
    python run_demo.py

    # Custom checkpoint paths
    python run_demo.py \\
        --pinn_ckpt ../architecture-training/checkpoints/best.pt \\
        --fno_ckpt  ../architecture-training-FNO/checkpoints/best.pt

    # Short 60-second smoke test (--dry_run)
    python run_demo.py --dry_run

Output plots (saved to results/)
---------------------------------
    mpc_control_timeline.png   — heater setpoint over time
    mpc_drying_curve.png       — MR true vs predicted + target
    mpc_eo_retention.png       — EO quality index over time
    mpc_phase_indicator.png    — drying phase Gantt bar
    mpc_cost_history.png       — MPC objective cost per step
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Path setup ─────────────────────────────────────────────────────────────────
import importlib.util as _ilu

MPC_DIR  = Path(__file__).parent
PINN_DIR = MPC_DIR.parent / "architecture-training"
FNO_DIR  = MPC_DIR.parent / "architecture-training-FNO"

# Load MPC config robustly
_mpc_spec = _ilu.spec_from_file_location("mpc_config", str(MPC_DIR / "config.py"))
C         = _ilu.module_from_spec(_mpc_spec)
_mpc_spec.loader.exec_module(C)

# Remove inserting paths to sys.path since the sub-modules now handle their own imports robustly
sys.path.insert(0, str(MPC_DIR))

from eo_model        import EOModel
from state_estimator import PINNStatePredictor, SensorFusionUKF
from controller      import HPDSController, load_fno
from simulator       import HPDSSim


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "blue":   "#1c7ed6",
    "orange": "#e8590c",
    "green":  "#2f9e44",
    "red":    "#c92a2a",
    "purple": "#862e9c",
    "gray":   "#868e96",
}

plt.rcParams.update({
    "figure.dpi":       120,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.family":      "sans-serif",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
})


def _time_axis(times_s: np.ndarray) -> np.ndarray:
    return times_s / 3600.0   # hours


def plot_control_timeline(times, T_setpoints, out_dir):
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.step(
        _time_axis(times), T_setpoints,
        where="post", color=PALETTE["orange"], lw=2, label="T_setpoint (MPC)"
    )
    ax.axhline(C.T_SURF_MAX, ls="--", color=PALETTE["red"], lw=1.2, label=f"T_surf limit {C.T_SURF_MAX}°C")
    ax.set(
        xlabel="Time [h]", ylabel="Heater setpoint [°C]",
        title="MPC Control Timeline — Heater Setpoint",
        ylim=(25, 60),
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    p = out_dir / "mpc_control_timeline.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {p.name}")


def plot_drying_curve(times, MR_true, MR_predicted, out_dir):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(_time_axis(times), MR_true,      color=PALETTE["blue"],  lw=2,   label="MR true (simulator)")
    ax.plot(_time_axis(times), MR_predicted, color=PALETTE["orange"], lw=1.5, ls="--", label="MR predicted (FNO horizon)")
    ax.axhline(C.MR_TARGET,     ls=":",  color=PALETTE["red"],   lw=1.5, label=f"MR target = {C.MR_TARGET}")
    ax.axhline(C.MR_PHASE1_END, ls="--", color=PALETTE["gray"],  lw=1.0, alpha=0.7, label="Phase 1/2 boundary")
    ax.axhline(C.MR_PHASE2_END, ls="--", color=PALETTE["gray"],  lw=1.0, alpha=0.5, label="Phase 2/3 boundary")
    ax.set(
        xlabel="Time [h]", ylabel="Moisture Ratio MR",
        title="Drying Curve — True vs MPC Predicted",
        ylim=(-0.02, 0.90),
    )
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    p = out_dir / "mpc_drying_curve.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {p.name}")


def plot_eo_retention(times, EO_retained, out_dir):
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(_time_axis(times), EO_retained, color=PALETTE["green"], lw=2, label="EO retained (weighted)")
    ax.axhline(0.85, ls="--", color=PALETTE["blue"],  lw=1.2, label="Quality target ≥ 0.85")
    ax.axhline(0.70, ls=":",  color=PALETTE["red"],   lw=1.2, label="Grade downgrade < 0.70")
    ax.set(
        xlabel="Time [h]", ylabel="EO retained [0–1]",
        title="Essential Oil Quality Retention",
        ylim=(0.5, 1.05),
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    p = out_dir / "mpc_eo_retention.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {p.name}")


def plot_phase_indicator(times, phases, out_dir):
    phase_color = {
        "Phase1_ConstantRate": PALETTE["blue"],
        "Phase2_FallingRate":  PALETTE["orange"],
        "Phase3_Equilibrium":  PALETTE["green"],
    }
    fig, ax = plt.subplots(figsize=(11, 1.6))
    t_h = _time_axis(times)
    prev_phase = phases[0]
    seg_start  = t_h[0]
    for i in range(1, len(phases)):
        if phases[i] != prev_phase or i == len(phases) - 1:
            ax.barh(
                0, t_h[i] - seg_start, left=seg_start,
                color=phase_color.get(prev_phase, PALETTE["gray"]),
                height=0.5, alpha=0.8,
            )
            seg_start  = t_h[i]
            prev_phase = phases[i]

    patches = [mpatches.Patch(color=v, label=k.replace("_", " ")) for k, v in phase_color.items()]
    ax.legend(handles=patches, loc="lower right", frameon=False, fontsize=8)
    ax.set(xlabel="Time [h]", yticks=[], title="Drying Phase Indicator")
    ax.set_xlim(t_h[0], t_h[-1])
    fig.tight_layout()
    p = out_dir / "mpc_phase_indicator.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {p.name}")


def plot_cost_history(times, costs, out_dir):
    fig, ax = plt.subplots(figsize=(11, 3.5))
    finite_costs = np.where(np.isfinite(costs), costs, np.nan)
    ax.plot(_time_axis(times), finite_costs, color=PALETTE["purple"], lw=1.2, alpha=0.8)
    ax.set(
        xlabel="Time [h]", ylabel="MPC objective cost",
        title="MPC Cost History",
    )
    fig.tight_layout()
    p = out_dir / "mpc_cost_history.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {p.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main simulation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(
    pinn_ckpt: Path,
    fno_ckpt: Path,
    duration_s: float,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Run the full closed-loop MPC simulation.

    Returns a dict of time-series arrays for plotting.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Demo] Device: {device}")
    print(f"[Demo] Duration: {duration_s/3600:.2f} h  ({int(duration_s/C.DT)} steps)")

    # ── Load models ───────────────────────────────────────────────────────────
    use_pinn = pinn_ckpt.exists()
    use_fno  = fno_ckpt.exists()

    if use_pinn:
        pinn = PINNStatePredictor(pinn_ckpt, device)
    else:
        print(f"[Demo] WARNING: PINN checkpoint not found at {pinn_ckpt}")
        print(f"         Running with mock state estimator (PINN-bypass mode).")
        pinn = None

    if use_fno:
        fno = load_fno(fno_ckpt, device)
    else:
        print(f"[Demo] WARNING: FNO checkpoint not found at {fno_ckpt}")
        print(f"         Running with random FNO weights (demo mode).")
        fno = CardamomFNO().to(device)  # random weights — for smoke testing
        fno.eval()

    # ── Build components ──────────────────────────────────────────────────────
    eo      = EOModel()
    eo.reset()
    sim     = HPDSSim(seed=seed)

    # State estimator
    if pinn is not None:
        estimator = SensorFusionUKF(pinn)
        estimator.initialise(t=0.0)
    else:
        # Mock estimator: uses simulator ground truth directly
        class MockEstimator:
            def __init__(self, sim_ref):
                self._sim = sim_ref
            def estimate(self, obs, t):
                import numpy as np
                x = np.array([self._sim.T_pod, self._sim.T_surface,
                               self._sim.MR * C.C_INIT,
                               self._sim.MR * C.C_INIT * 0.8,
                               self._sim.MR])
                return x, np.eye(5)
            @property
            def state(self):
                return {
                    "T_core":    self._sim.T_pod,
                    "T_surface": self._sim.T_surface,
                    "c_core":    self._sim.MR * C.C_INIT,
                    "c_surface": self._sim.MR * C.C_INIT * 0.8,
                    "MR":        self._sim.MR,
                }
        estimator = MockEstimator(sim)

    ctrl = HPDSController(fno, estimator, eo, device)

    # ── Simulation loop ────────────────────────────────────────────────────────
    obs_init = sim.reset()

    times        = []
    T_setpoints  = []
    MR_true_list = []
    MR_pred_list = []
    EO_list      = []
    phase_list   = []
    cost_list    = []

    n_steps   = int(duration_s / C.DT)
    log_every = max(1, n_steps // 20)
    obs       = obs_init
    T_set     = 50.0   # initial setpoint

    print(f"[Demo] Starting simulation ...")

    for step in range(n_steps):
        t = step * C.DT

        # MPC step
        result = ctrl.step(sensor_obs=obs, t=float(t))
        T_set  = result["T_setpoint"]

        # Apply control to simulator → get next observation
        obs = sim.step(T_setpoint=T_set)

        # Record
        times.append(t)
        T_setpoints.append(T_set)
        MR_true_list.append(obs["MR_true"])
        MR_pred_list.append(result["predicted_MR"])
        EO_list.append(obs["EO_retained"])
        phase_list.append(result["drying_phase"])
        cost_list.append(result["cost"])

        if verbose and (step % log_every == 0 or step == n_steps - 1):
            print(
                f"  t={t/3600:.2f}h  T_set={T_set:.0f}°C  "
                f"MR={obs['MR_true']:.3f}  EO={obs['EO_retained']:.3f}  "
                f"phase={result['drying_phase']}  cost={result['cost']:.4f}"
            )

        # Stop early if target MR reached
        if obs["MR_true"] <= C.MR_TARGET:
            print(f"\n[Demo] MR target {C.MR_TARGET} reached at t={t/3600:.2f} h!")
            break

    data = {
        "times":       np.array(times),
        "T_setpoints": np.array(T_setpoints),
        "MR_true":     np.array(MR_true_list),
        "MR_pred":     np.array(MR_pred_list),
        "EO_retained": np.array(EO_list),
        "phases":      phase_list,
        "costs":       np.array(cost_list),
    }

    # Final summary
    print(f"\n{'='*55}")
    print(f"  Simulation complete")
    print(f"  Duration           : {times[-1]/3600:.2f} h")
    print(f"  Final MR (true)    : {MR_true_list[-1]:.4f}  (target: {C.MR_TARGET})")
    print(f"  Final EO retained  : {EO_list[-1]:.4f}  (target: ≥ 0.80)")
    print(f"  MR target reached  : {'YES ✓' if MR_true_list[-1] <= C.MR_TARGET + 0.02 else 'NO (extend run)'}")
    print(f"{'='*55}\n")

    return data


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="HPDS MPC closed-loop demo")
    ap.add_argument("--pinn_ckpt",  type=str, default=str(C.PINN_CKPT))
    ap.add_argument("--fno_ckpt",   type=str, default=str(C.FNO_CKPT))
    ap.add_argument("--duration_h", type=float, default=8.55,
                    help="Simulation duration in hours (default: 8.55)")
    ap.add_argument("--dry_run",    action="store_true",
                    help="Run only 60 seconds (smoke test)")
    ap.add_argument("--seed",       type=int, default=42)
    args = ap.parse_args()

    pinn_ckpt = Path(args.pinn_ckpt)
    fno_ckpt  = Path(args.fno_ckpt)
    duration  = 60.0 if args.dry_run else args.duration_h * 3600.0

    if args.dry_run:
        print("[Demo] DRY RUN mode — 60 seconds only")

    # Run simulation
    data = run_simulation(pinn_ckpt, fno_ckpt, duration, seed=args.seed)

    # Generate plots
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)

    print("[Demo] Generating plots ...")
    plot_control_timeline(data["times"], data["T_setpoints"], out_dir)
    plot_drying_curve(data["times"], data["MR_true"], data["MR_pred"], out_dir)
    plot_eo_retention(data["times"], data["EO_retained"], out_dir)
    plot_phase_indicator(data["times"], data["phases"], out_dir)
    plot_cost_history(data["times"], data["costs"], out_dir)

    print(f"\n[Demo] All plots saved to: {out_dir}/")
    print("[Demo] Done ✓")


if __name__ == "__main__":
    # Need CardamomFNO import even in fallback mode
    from model import CardamomFNO
    main()
