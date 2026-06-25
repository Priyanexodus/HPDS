"""
losses.py
=========
Multi-term PINN loss for Cardamom HPDS drying model.

PDE residuals are computed via torch.autograd.grad (requires_grad on inputs).

Loss inventory
--------------
  L_physics_T  — heat equation residual     (at collocation points)
  L_physics_c  — moisture diffusion residual (at collocation points)
  L_ic_T       — initial condition: temperature at t=0
  L_ic_c       — initial condition: concentration at t=0  (separate λ — key signal)
  L_non_neg    — non-negativity of c everywhere            (physical constraint)
  L_bc_T       — Robin BC (convective) at pod surface at sensor time-points
  L_bc_c       — mass-transfer Robin BC at pod surface
  L_data_T     — COMSOL anchor temperature supervision
  L_data_c     — COMSOL anchor concentration supervision
  L_mr         — load-cell MR(t) volume-average supervision
"""

import math
import torch
from config import (
    A_AXIAL, B_RADIAL, ALPHA_T, D_EFF_50, T_REF_K, EA_MOIST,
    R_GAS, H_CONV, H_M_CONV, K_COND, R_EFF, C_INIT, T_SCALE, C_SCALE,
    TOTAL_TIME_S,
    LAMBDA_PHYSICS_T, LAMBDA_PHYSICS_C,
    LAMBDA_IC_T, LAMBDA_IC_C,
    LAMBDA_NON_NEG,
    LAMBDA_BC_T, LAMBDA_BC_C,
    LAMBDA_DATA_T, LAMBDA_DATA_C,
    LAMBDA_MR,
)


# ─────────────────────────────────────────────────────────────────────────────
# Thermophysical helpers
# ─────────────────────────────────────────────────────────────────────────────

def _D_eff(T_C: torch.Tensor) -> torch.Tensor:
    """Arrhenius effective moisture diffusivity [m²/s] given T in [°C]."""
    T_K = (T_C + 273.15).clamp(min=1.0)
    return D_EFF_50 * torch.exp(
        (-EA_MOIST / R_GAS) * (1.0 / T_K - 1.0 / T_REF_K)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Auto-diff helpers (1st & 2nd order)
# ─────────────────────────────────────────────────────────────────────────────

def _grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """∂y/∂x, summed over output dim if needed.  Retains graph."""
    (g,) = torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True,
    )
    return g


def _laplacian_spherical(u: torch.Tensor, x_coll: torch.Tensor) -> torch.Tensor:
    """
    Approximate 2-D spherical Laplacian at collocation points via autograd.

    x_coll : (N, 3) — [r_norm, z_norm, t_norm], requires_grad=True.
    u      : (N, 1) — scalar field.

    Returns ∇²u ≈ (1/R²) [ ∂²u/∂r_n² + (2/r_n)·∂u/∂r_n ] (1-D approximation
    along normalised ellipsoidal coordinate ξ), which is sufficient for the
    radially-dominant diffusion in this near-spherical pod.
    """
    # Decompose gradients wrt each normalised coordinate
    g = _grad(u, x_coll)            # (N, 3): [∂u/∂r_n, ∂u/∂z_n, ∂u/∂t_n]
    du_dr  = g[:, 0:1]              # ∂u/∂r_norm
    du_dz  = g[:, 1:2]              # ∂u/∂z_norm

    # Second-order in r_n
    d2u_dr2 = _grad(du_dr, x_coll)[:, 0:1]
    d2u_dz2 = _grad(du_dz, x_coll)[:, 1:2]

    r_n = x_coll[:, 0:1].clamp(min=1e-6)   # avoid div-by-zero at centre

    # Physical Laplacian  (∂²/∂r² + (1/r)∂/∂r) / scale² + same in z / scale²
    lap_r = (d2u_dr2 + (1.0 / r_n) * du_dr) / (B_RADIAL ** 2)
    lap_z = d2u_dz2 / (A_AXIAL ** 2)

    return lap_r + lap_z


# ─────────────────────────────────────────────────────────────────────────────
# Individual loss terms
# ─────────────────────────────────────────────────────────────────────────────

def loss_physics(
    model,
    x_coll: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    PDE residual losses at interior collocation points.

    Returns
    -------
    L_T : scalar — heat equation residual MSE
    L_c : scalar — moisture diffusion residual MSE
    """
    x_coll = x_coll.detach().requires_grad_(True)

    T_norm, c_norm = model(x_coll)
    T = T_norm * T_SCALE        # physical temperature [°C]
    c = c_norm * C_SCALE        # physical concentration [mol/m³]

    # Full gradients via autograd
    g_T_full = _grad(T_norm, x_coll)
    g_c_full = _grad(c_norm, x_coll)

    # Time derivatives via autograd
    g_T = g_T_full[:, 2:3]   # ∂T_norm/∂t_norm
    g_c = g_c_full[:, 2:3]   # ∂c_norm/∂t_norm

    # Convert to physical time derivatives
    dT_dt = g_T * (T_SCALE / TOTAL_TIME_S)   # [°C/s]
    dc_dt = g_c * (C_SCALE / TOTAL_TIME_S)   # [mol/m³/s]

    # Spatial Laplacians
    lap_T = _laplacian_spherical(T_norm, x_coll) * T_SCALE   # [°C/m²]
    lap_c = _laplacian_spherical(c_norm, x_coll) * C_SCALE   # [mol/m³/m²]

    D = _D_eff(T.detach())          # Arrhenius D_eff [m²/s], detach so grad only through c
    T_K = (T.detach() + 273.15).clamp(min=1.0)
    dD_dT = D * (EA_MOIST / (R_GAS * T_K.pow(2)))

    dT_dr = g_T_full[:, 0:1] * (T_SCALE / B_RADIAL)
    dT_dz = g_T_full[:, 1:2] * (T_SCALE / A_AXIAL)
    dD_dr = dD_dT * dT_dr.detach()
    dD_dz = dD_dT * dT_dz.detach()

    dc_dr = g_c_full[:, 0:1] * (C_SCALE / B_RADIAL)
    dc_dz = g_c_full[:, 1:2] * (C_SCALE / A_AXIAL)
    gradD_dot_gradc = dD_dr * dc_dr + dD_dz * dc_dz

    # PDE residuals
    res_T = dT_dt - ALPHA_T * lap_T   # heat eq:      ∂T/∂t = α·∇²T
    res_c = dc_dt - (D * lap_c + gradD_dot_gradc) # moisture eq:  ∂c/∂t = ∇·(D_eff ∇c)

    return res_T.pow(2).mean(), res_c.pow(2).mean()


def loss_ic(
    model,
    device: torch.device,
    n_pts: int = 512,
    T_init: float = 25.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Initial condition loss: at t_norm=0, T→T_init, c→C_INIT.

    Returns L_ic_T and L_ic_c SEPARATELY so the caller can apply
    independent lambda weights to temperature and concentration ICs.
    Concentration IC is the dominant failure mode — it needs LAMBDA_IC_C
    much higher than LAMBDA_IC_T to prevent the physics terms from
    pulling c away from its correct initial value of C_INIT.

    Samples random (r,z) points inside the ellipse at t=0.
    """
    # Sample interior points (r_norm ∈ [0,1], z_norm ∈ [-1,1])
    r_n = torch.rand(n_pts, 1, device=device)
    z_n = torch.FloatTensor(n_pts, 1).uniform_(-1.0, 1.0).to(device)
    # Keep only those inside ellipse
    inside = (r_n.pow(2) + z_n.pow(2)).sqrt() < 1.0
    r_n = r_n[inside].view(-1, 1)
    z_n = z_n[inside].view(-1, 1)
    t_n = torch.zeros_like(r_n)

    x_ic = torch.cat([r_n, z_n, t_n], dim=1)
    T_norm, c_norm = model(x_ic)

    T_target = torch.full_like(T_norm, T_init / T_SCALE)
    c_target = torch.full_like(c_norm, C_INIT / C_SCALE)   # ≈ 0.9416

    L_T = (T_norm - T_target).pow(2).mean()
    L_c = (c_norm - c_target).pow(2).mean()
    return L_T, L_c


def loss_non_neg(
    model,
    device: torch.device,
    n_pts: int = 512,
) -> torch.Tensor:
    """
    Non-negativity constraint on concentration: c_norm ≥ 0 everywhere.

    Penalises relu(-c_norm)² at random interior and boundary points.
    Without this, the network can satisfy the PDE residual at wrong scale
    by producing negative concentrations at the surface (confirmed in results).

    This is a hard physical constraint — moisture cannot be negative.
    """
    r_n = torch.rand(n_pts, 1, device=device)
    z_n = torch.FloatTensor(n_pts, 1).uniform_(-1.0, 1.0).to(device)
    t_n = torch.rand(n_pts, 1, device=device)
    inside = (r_n.pow(2) + z_n.pow(2)).sqrt() < 1.0
    r_n = r_n[inside].view(-1, 1)
    z_n = z_n[inside].view(-1, 1)
    t_n = t_n[inside].view(-1, 1)

    if r_n.shape[0] == 0:
        return torch.tensor(0.0, device=device)

    x_pts = torch.cat([r_n, z_n, t_n], dim=1)
    _, c_norm = model(x_pts)
    violation = torch.relu(-c_norm)   # = 0 where c_norm >= 0, else |c_norm|
    return violation.pow(2).mean()


def loss_bc_temperature(
    model,
    x_bc: torch.Tensor,
    T_air_vals: torch.Tensor,
) -> torch.Tensor:
    """
    Robin BC at the pod surface (r_n²+z_n²=1), fully non-dimensionalised:

        ∂T_norm/∂n_norm + Bi_T · (T_norm − T_air_norm) = 0

    derived from the physical law −k·∂T/∂n = h·(T_surf − T_air).
    Bi_T = h·L_ref/k  is the (dimensionless) thermal Biot number.

    NOTE: this residual is algebraically a constant positive multiple
    (L_ref / T_SCALE) of the raw-physical-unit residual — i.e. it is the
    *same* Robin BC condition (same zero set / same converged solution),
    just rescaled so it sits at O(1), on the same footing as L_ic / L_data
    / L_mr, instead of O(10^3-10^4) [°C/m] silently dominating every
    gradient step.
    """
    x_bc = x_bc.detach().requires_grad_(True)
    T_norm, _ = model(x_bc)
    g = _grad(T_norm, x_bc)              # [∂T_norm/∂r_n, ∂T_norm/∂z_n, ...]

    r_n = x_bc[:, 0:1]
    z_n = x_bc[:, 1:2]

    # Outward unit normal of the ellipse r_n²+z_n²=1 in PHYSICAL direction,
    # expressed via normalised coords: n̂ ∝ (r_n/B_RADIAL, z_n/A_AXIAL).
    # (NOT divided by B_RADIAL², A_AXIAL² — that was the second bug here.)
    nR = r_n / B_RADIAL
    nZ = z_n / A_AXIAL
    n_len = torch.sqrt(nR**2 + nZ**2)
    nR, nZ = nR / n_len, nZ / n_len

    L_ref = B_RADIAL    # reference length for non-dimensionalisation
    dT_dn_norm = L_ref * (g[:, 0:1] / B_RADIAL * nR + g[:, 1:2] / A_AXIAL * nZ)

    T_air_norm = T_air_vals / T_SCALE
    Bi_T = H_CONV * L_ref / K_COND     # ≈ 0.46, dimensionless

    res = dT_dn_norm + Bi_T * (T_norm - T_air_norm)
    return res.pow(2).mean()


def loss_bc_concentration(
    model,
    x_bc: torch.Tensor,
    c_air_vals: torch.Tensor,
) -> torch.Tensor:
    """
    Mass-transfer Robin BC at the pod surface, same non-dimensional treatment:

        ∂c_norm/∂n_norm + Bi_m(T) · (c_norm − c_air_norm) = 0

    Bi_m = h_m·L_ref / D_eff(T)  is the (dimensionless) mass-transfer Biot
    number (≈ 50 at D_eff_50) — a real, finite, deliberately-derived number,
    not the previous "accidentally O(1)" raw-unit residual (which only
    looked sane because H_M_CONV happens to be tiny — same disease as the
    bc_T bug, milder symptom).
    """
    x_bc = x_bc.detach().requires_grad_(True)
    T_norm, c_norm = model(x_bc)
    T_surf = T_norm * T_SCALE         # [°C], needed only for D_eff(T)
    g_c = _grad(c_norm, x_bc)

    r_n = x_bc[:, 0:1]
    z_n = x_bc[:, 1:2]

    nR = r_n / B_RADIAL
    nZ = z_n / A_AXIAL
    n_len = torch.sqrt(nR**2 + nZ**2)
    nR, nZ = nR / n_len, nZ / n_len

    L_ref = B_RADIAL
    dc_dn_norm = L_ref * (g_c[:, 0:1] / B_RADIAL * nR + g_c[:, 1:2] / A_AXIAL * nZ)

    c_air_norm = c_air_vals / C_SCALE
    D = _D_eff(T_surf.detach())
    Bi_m = H_M_CONV * L_ref / D        # varies with T via D(T); ~O(10-100)

    res = dc_dn_norm + Bi_m * (c_norm - c_air_norm)
    return res.pow(2).mean()


def loss_data(
    model,
    x_anch: torch.Tensor,
    T_anch: torch.Tensor,
    c_anch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sparse field supervision from COMSOL anchor points.

    x_anch : (K, 3) normalised coordinates
    T_anch : (K, 1) target temperature [°C]
    c_anch : (K, 1) target concentration [mol/m³]
    """
    T_norm, c_norm = model(x_anch)
    L_T = (T_norm - (T_anch / T_SCALE)).pow(2).mean()
    L_c = (c_norm - (c_anch / C_SCALE)).pow(2).mean()
    return L_T, L_c


def loss_mr(
    model,
    t_mr: torch.Tensor,
    MR_target: torch.Tensor,
    n_quad: int = 64,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Moisture Ratio supervision from load cell.

    MR(t) = <c(r,z,t)> / C_INIT   (area average over 2D ellipse cross-section)

    The model is 2D planar (r_norm, z_norm) — NOT axisymmetric 3D.
    For a uniform 2D area average in normalised coordinates, all interior
    points get equal weight (the Jacobian B_RADIAL * A_AXIAL cancels in
    the ratio). So the correct estimator is simply the mean of c over
    uniformly sampled interior points — NOT a r_n-weighted sum.

    t_mr      : (T,)  normalised time points where MR is observed
    MR_target : (T,)  load-cell MR values
    """
    if device is None:
        device = t_mr.device

    n_t = t_mr.shape[0]
    MR_pred = []

    use_grad = model.training   # need grad during training, not during validation

    for i in range(n_t):
        # Sample n_quad random interior points
        r_n = torch.rand(n_quad, 1, device=device)
        z_n = torch.FloatTensor(n_quad, 1).uniform_(-1.0, 1.0).to(device)
        inside = (r_n.pow(2) + z_n.pow(2)).sqrt() < 1.0
        r_n = r_n[inside].view(-1, 1)
        z_n = z_n[inside].view(-1, 1)

        if r_n.shape[0] == 0:
            MR_pred.append(torch.tensor(1.0, device=device))
            continue

        t_n = t_mr[i].expand(r_n.shape[0], 1)
        x_q = torch.cat([r_n, z_n, t_n], dim=1)

        if use_grad:
            _, c_norm = model(x_q)
        else:
            with torch.no_grad():
                _, c_norm = model(x_q)

        # 2D planar ellipse: uniform area average (no r_n weighting).
        # MR = <c> / C_INIT = mean(c_norm * C_SCALE) / C_INIT
        #                    = mean(c_norm) * (C_SCALE / C_INIT)
        c_avg_norm = c_norm.mean()
        MR_pred.append(c_avg_norm * (C_SCALE / C_INIT))

    MR_pred_t = torch.stack(MR_pred)
    return (MR_pred_t - MR_target).pow(2).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Composite PINN loss
# ─────────────────────────────────────────────────────────────────────────────

class PINNLoss:
    """
    Weighted composite loss.

    Parameters
    ----------
    lambdas : dict  — override any default λ weight.
    phase   : int   — 1 = data/IC/BC only; 2 = full PINN (physics + data).
    """

    def __init__(self, lambdas: dict | None = None, phase: int = 2):
        self.lam = {
            "physics_T": LAMBDA_PHYSICS_T,
            "physics_c": LAMBDA_PHYSICS_C,
            # Temperature and concentration ICs have SEPARATE lambdas.
            # Concentration IC is the key failure mode — LAMBDA_IC_C >> LAMBDA_IC_T.
            "ic_T":       LAMBDA_IC_T,
            "ic_c":       LAMBDA_IC_C,
            "non_neg":    LAMBDA_NON_NEG,
            "bc_T":       LAMBDA_BC_T,
            "bc_c":       LAMBDA_BC_C,
            "data_T":     LAMBDA_DATA_T,
            "data_c":     LAMBDA_DATA_C,
            "mr":         LAMBDA_MR,
        }
        if lambdas:
            self.lam.update(lambdas)
        self.phase = phase

    def __call__(
        self,
        model,
        batch: dict,
        device: torch.device,
        n_ic_pts: int = 256,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute the full composite loss.

        batch contains:
          "x_coll"   : (B, 3) collocation coordinates
          "x_bc"     : (M, 3) surface boundary coordinates
          "T_air_bc" : (M, 1) T_air at those surface/time points [°C]
          "c_air_bc" : (M, 1) c_air at those surface/time points [mol/m³]
          "x_anch"   : (K, 3) COMSOL anchor coordinates
          "T_anch"   : (K, 1) anchor T [°C]
          "c_anch"   : (K, 1) anchor c [mol/m³]
          "t_mr"     : (T,)   normalised MR time points
          "MR"       : (T,)   load-cell MR values

        Returns total loss scalar + dict of component scalars.
        """
        logs = {}
        total = torch.tensor(0.0, device=device)

        # ── Phase 1 & 2: IC (separate T and c lambdas) ──────────────────────
        L_ic_T, L_ic_c = loss_ic(model, device, n_pts=n_ic_pts)
        logs["ic_T"] = L_ic_T.item()
        logs["ic_c"] = L_ic_c.item()
        total = total + self.lam["ic_T"] * L_ic_T + self.lam["ic_c"] * L_ic_c

        # ── Phase 1 & 2: Non-negativity of c ────────────────────────────────
        L_nn = loss_non_neg(model, device, n_pts=n_ic_pts)
        logs["non_neg"] = L_nn.item()
        total = total + self.lam["non_neg"] * L_nn

        # ── Phase 1 & 2: BC ─────────────────────────────────────────────────
        if "x_bc" in batch and "T_air_bc" in batch:
            L_bc = loss_bc_temperature(model, batch["x_bc"], batch["T_air_bc"])
            logs["bc_T"] = L_bc.item()
            total = total + self.lam["bc_T"] * L_bc

        if "x_bc" in batch and "c_air_bc" in batch:
            L_bc_c = loss_bc_concentration(model, batch["x_bc"], batch["c_air_bc"])
            logs["bc_c"] = L_bc_c.item()
            total = total + self.lam["bc_c"] * L_bc_c

        # ── Phase 1 & 2: Data supervision ────────────────────────────────────
        if "x_anch" in batch:
            L_dT, L_dc = loss_data(
                model, batch["x_anch"], batch["T_anch"], batch["c_anch"]
            )
            logs["data_T"] = L_dT.item()
            logs["data_c"] = L_dc.item()
            total = total + self.lam["data_T"] * L_dT + self.lam["data_c"] * L_dc

        # ── Phase 1 & 2: MR supervision ──────────────────────────────────────
        if "t_mr" in batch:
            L_mr = loss_mr(model, batch["t_mr"], batch["MR"], device=device)
            logs["mr"] = L_mr.item()
            total = total + self.lam["mr"] * L_mr

        # ── Phase 2 only: Physics residuals ──────────────────────────────────
        if self.phase == 2 and "x_coll" in batch:
            L_pT, L_pc = loss_physics(model, batch["x_coll"])
            logs["physics_T"] = L_pT.item()
            logs["physics_c"] = L_pc.item()
            total = total + self.lam["physics_T"] * L_pT + self.lam["physics_c"] * L_pc

        logs["total"] = total.item()
        return total, logs

    def update_lambdas(self, new_lam: dict):
        """Adaptive lambda update (called from trainer)."""
        self.lam.update(new_lam)

    def set_phase(self, phase: int):
        self.phase = phase