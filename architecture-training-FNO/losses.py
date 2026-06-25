"""
losses.py
=========
Loss functions for the Cardamom HPDS FNO.

The FNO is purely data-driven (no PDE residuals). Losses:

  L_T    — MSE on normalised temperature output channel
  L_M    — MSE on normalised moisture output channel
  L_phys — physics-guided penalty: moisture monotonically decreasing
            (M_future ≤ M_current + ε, in physical space)

Weighted combination:
  L_total = λ_T · L_T + λ_M · L_M + λ_phys · L_phys
"""

import torch
import torch.nn as nn

from config import (
    LAMBDA_T, LAMBDA_M,
    NORM_MEAN_X, NORM_STD_X,
    NORM_MEAN_Y, NORM_STD_Y,
    IN_CHANNELS, OUT_CHANNELS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Denormalisation helpers (for physics penalty)
# ─────────────────────────────────────────────────────────────────────────────

def _denorm_channel(tensor: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return tensor * std + mean


# ─────────────────────────────────────────────────────────────────────────────
# Individual loss terms
# ─────────────────────────────────────────────────────────────────────────────

def loss_mse(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-channel MSE between predicted and target normalised fields.

    Parameters
    ----------
    y_pred : (B, 2, H, W)  normalised FNO predictions
    y_true : (B, 2, H, W)  normalised targets

    Returns
    -------
    L_T : scalar  — temperature channel MSE
    L_M : scalar  — moisture channel MSE
    """
    L_T = nn.functional.mse_loss(y_pred[:, 0], y_true[:, 0])
    L_M = nn.functional.mse_loss(y_pred[:, 1], y_true[:, 1])
    return L_T, L_M


def loss_physics_monotone(
    x_norm: torch.Tensor,
    y_pred: torch.Tensor,
    slack: float = 0.01,
) -> torch.Tensor:
    """
    Physics-guided penalty: moisture must not *increase* during drying.
    Penalises M_future > M_current + slack (in physical space).

    Uses only the input moisture channel (ch=1) and predicted output moisture (ch=1).

    Parameters
    ----------
    x_norm : (B, 3, H, W)  normalised inputs
    y_pred : (B, 2, H, W)  normalised predictions
    slack  : float         allowable moisture increase [kg_w/kg_dry] before penalty

    Returns
    -------
    L_mono : scalar ≥ 0
    """
    # Denormalise moisture channels to physical space
    M_curr = _denorm_channel(x_norm[:, 1],
                             NORM_MEAN_X[1], NORM_STD_X[1])   # (B, H, W)
    M_fut  = _denorm_channel(y_pred[:, 1],
                             NORM_MEAN_Y[1], NORM_STD_Y[1])   # (B, H, W)

    # Violation = max(0, M_future - M_current - slack)
    violation = torch.relu(M_fut - M_curr - slack)
    return violation.pow(2).mean()


def loss_physics_temp_range(
    y_pred: torch.Tensor,
    T_min: float = 20.0,
    T_max: float = 90.0,
) -> torch.Tensor:
    """
    Physics-guided penalty: predicted temperature must stay within [T_min, T_max] °C.

    Parameters
    ----------
    y_pred : (B, 2, H, W)  normalised predictions
    T_min  : float         minimum physical temperature [°C]
    T_max  : float         maximum physical temperature [°C]
    """
    T_pred = _denorm_channel(y_pred[:, 0], NORM_MEAN_Y[0], NORM_STD_Y[0])
    below  = torch.relu(T_min - T_pred)
    above  = torch.relu(T_pred - T_max)
    return (below.pow(2) + above.pow(2)).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Relative L2 error  (for validation / reporting)
# ─────────────────────────────────────────────────────────────────────────────

def relative_l2(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Relative L2 error per channel.

    Returns
    -------
    err_T : scalar  — relative L2 for temperature
    err_M : scalar  — relative L2 for moisture
    """
    err_T = (y_pred[:, 0] - y_true[:, 0]).norm() / (y_true[:, 0].norm() + eps)
    err_M = (y_pred[:, 1] - y_true[:, 1]).norm() / (y_true[:, 1].norm() + eps)
    return err_T, err_M


def r2_score(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
) -> tuple[float, float]:
    """
    Coefficient of determination R² per output channel.
    Computed in normalised space.

    Returns
    -------
    r2_T, r2_M : float
    """
    def _r2(pred, true):
        ss_res = (pred - true).pow(2).sum()
        ss_tot = (true - true.mean()).pow(2).sum()
        return 1.0 - (ss_res / (ss_tot + 1e-8))

    return _r2(y_pred[:, 0], y_true[:, 0]).item(), \
           _r2(y_pred[:, 1], y_true[:, 1]).item()


# ─────────────────────────────────────────────────────────────────────────────
# Composite FNO loss
# ─────────────────────────────────────────────────────────────────────────────

class FNOLoss:
    """
    Weighted composite loss for FNO training.

    Parameters
    ----------
    lambda_T       : float  — weight on temperature MSE
    lambda_M       : float  — weight on moisture MSE
    lambda_mono    : float  — weight on monotone physics penalty
    lambda_T_range : float  — weight on temperature range penalty
    """

    def __init__(
        self,
        lambda_T:       float = LAMBDA_T,
        lambda_M:       float = LAMBDA_M,
        lambda_mono:    float = 0.5,
        lambda_T_range: float = 0.1,
    ):
        self.lambda_T       = lambda_T
        self.lambda_M       = lambda_M
        self.lambda_mono    = lambda_mono
        self.lambda_T_range = lambda_T_range

    def __call__(
        self,
        y_pred:  torch.Tensor,
        y_true:  torch.Tensor,
        x_norm:  torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute total loss.

        Parameters
        ----------
        y_pred  : (B, 2, H, W)  model predictions (normalised)
        y_true  : (B, 2, H, W)  targets (normalised)
        x_norm  : (B, 3, H, W)  normalised inputs (for physics penalties)

        Returns
        -------
        total  : scalar loss tensor
        logs   : dict of component loss values (for logging)
        """
        L_T, L_M = loss_mse(y_pred, y_true)
        L_mono   = loss_physics_monotone(x_norm, y_pred)
        L_Trange = loss_physics_temp_range(y_pred)

        total = (self.lambda_T       * L_T
               + self.lambda_M       * L_M
               + self.lambda_mono    * L_mono
               + self.lambda_T_range * L_Trange)

        logs = {
            "loss_T":     L_T.item(),
            "loss_M":     L_M.item(),
            "loss_mono":  L_mono.item(),
            "loss_Trange": L_Trange.item(),
            "total":      total.item(),
        }
        return total, logs
