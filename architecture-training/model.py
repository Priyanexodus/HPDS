"""
model.py
========
PhysicsNeMo-based PINN for Cardamom HPDS Drying.

Builds on PhysicsNeMo primitives:
  - physicsnemo.core.Module          → base class (AMP, JIT, ONNX metadata)
  - physicsnemo.core.ModelMetaData   → capability flags
  - physicsnemo.nn.FCLayer           → Xavier-init dense layer w/ optional weight-norm
  - physicsnemo.nn.get_activation    → unified activation factory
  - physicsnemo.nn.FourierLayer      → random-Fourier-feature input embedding

Architecture
------------
  Input  : (r_norm, z_norm, t_norm)  — 3 normalised coordinates
  ↓
  FourierLayer (Gaussian random freqs, frozen)   → 2*N_FREQ dim
  Concatenate raw coords + Fourier features      → 3 + 2*N_FREQ
  ↓
  FCLayer(input_proj) + Tanh
  ↓
  N_HIDDEN × FCLayer + Tanh  (skip: every SKIP_EVERY layers adds embed back)
  ↓
  Head_T : FCLayer → 1     (T_norm = T / T_SCALE)
  Head_c : FCLayer → 1     (c_norm = c / C_SCALE)

Supports auto-diff for PDE residuals (func_torch=True, auto_grad=True).
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from physicsnemo.core import ModelMetaData, Module
from physicsnemo.nn import FCLayer, FourierLayer, get_activation

from config import (
    A_AXIAL, B_RADIAL, C_SCALE, FOURIER_FEATURES,
    HIDDEN_DIM, N_HIDDEN, SKIP_EVERY, T_SCALE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Metadata flags (enables AMP, JIT, auto-diff in PhysicsNeMo)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PINNMetaData(ModelMetaData):
    name: str = "CardamomPINN"
    # Optimisation
    jit: bool = False          # disabled – auto-diff graphs prevent scripting
    cuda_graphs: bool = False
    amp: bool = True
    torch_fx: bool = False
    # Inference
    onnx: bool = False
    onnx_runtime: bool = False
    # Physics-informed
    func_torch: bool = True    # enables functorch / vmap compatibility
    auto_grad: bool = True     # marks that this model is used with autograd


# ─────────────────────────────────────────────────────────────────────────────
# Main PINN module
# ─────────────────────────────────────────────────────────────────────────────

class CardamomPINN(Module):
    """
    Physics-Informed Neural Network for cardamom pod drying.

    Parameters
    ----------
    n_frequencies : int
        Number of Gaussian random Fourier frequencies (per input dim).
        Total embedding size = 2 * n_frequencies.
    hidden_dim : int
        Width of each hidden FCLayer.
    n_hidden : int
        Number of hidden FCLayers.
    skip_every : int
        Add a residual connection from the Fourier-embed to the hidden
        representation every `skip_every` layers.
    activation : str
        Activation name recognised by physicsnemo.nn.get_activation
        (e.g. ``"tanh"``, ``"silu"``, ``"gelu"``).
    fourier_sigma : float
        Standard deviation of the Gaussian from which random frequencies
        are drawn. Larger values capture higher-frequency features.
    """

    def __init__(
        self,
        n_frequencies: int   = FOURIER_FEATURES,
        hidden_dim:    int   = HIDDEN_DIM,
        n_hidden:      int   = N_HIDDEN,
        skip_every:    int   = SKIP_EVERY,
        activation:    str   = "tanh",
        fourier_sigma: float = 1.0,
    ) -> None:
        super().__init__(meta=PINNMetaData())

        self.n_frequencies = n_frequencies
        self.hidden_dim    = hidden_dim
        self.n_hidden      = n_hidden
        self.skip_every    = skip_every

        # ── Fourier feature layer (physicsnemo.nn.FourierLayer) ───────────────
        # We use the "gaussian" frequency mode: draws N(0, sigma) frequencies.
        # FourierLayer freezes the frequency matrix (not trained).
        self.fourier = FourierLayer(
            in_features=3,
            frequencies=("gaussian", fourier_sigma, n_frequencies),
        )
        ff_out_dim = self.fourier.out_features()   # = 2 * n_frequencies

        # Total input to the network after concatenating raw + Fourier features
        embed_dim = 3 + ff_out_dim

        # ── Activation ────────────────────────────────────────────────────────
        act_fn = get_activation(activation)

        # ── Entry projection (embed → hidden) ─────────────────────────────────
        self.input_proj = FCLayer(embed_dim, hidden_dim, activation_fn=act_fn)

        # ── Hidden layers with optional skip connections ───────────────────────
        # PhysicsNeMo's FCLayer handles Xavier init, optional weight-norm, etc.
        self.hidden_layers = nn.ModuleList()
        self.skip_projs    = nn.ModuleList()

        for i in range(n_hidden):
            self.hidden_layers.append(
                FCLayer(hidden_dim, hidden_dim, activation_fn=act_fn)
            )
            # At every skip_every-th layer, project the raw embed into hidden_dim
            # and add it as a residual.  The skip projection has NO activation.
            if (i + 1) % skip_every == 0:
                self.skip_projs.append(
                    FCLayer(embed_dim, hidden_dim, activation_fn=None)
                )
            else:
                self.skip_projs.append(None)

        # ── Output heads (no activation, near-zero init for stable start) ─────
        self.head_T = nn.Linear(hidden_dim, 1)
        self.head_c = nn.Linear(hidden_dim, 1)
        nn.init.uniform_(self.head_T.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.head_T.bias)
        nn.init.uniform_(self.head_c.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.head_c.bias)

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor  shape (N, 3)
            Normalised coordinates [r_norm ∈ [0,1], z_norm ∈ [-1,1], t_norm ∈ [0,1]].

        Returns
        -------
        T_norm : (N, 1)   — T / T_SCALE
        c_norm : (N, 1)   — c / C_SCALE
        """
        ff    = self.fourier(x)                    # (N, 2*n_freq)
        z_in  = torch.cat([x, ff], dim=-1)         # (N, 3 + 2*n_freq)

        h = self.input_proj(z_in)                  # (N, hidden_dim)

        for i, (layer, skip) in enumerate(zip(self.hidden_layers, self.skip_projs)):
            h = layer(h)
            if skip is not None:
                h = h + skip(z_in)

        T_norm = self.head_T(h)                    # (N, 1)
        c_norm = self.head_c(h)                    # (N, 1)
        return T_norm, c_norm

    # ── Convenience ────────────────────────────────────────────────────────────

    def predict_physical(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return T [°C] and c [mol/m³] from normalised input coordinates."""
        T_norm, c_norm = self.forward(x)
        T = T_norm * T_SCALE
        c = torch.clamp(c_norm * C_SCALE, min=0.0)
        return T, c

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ── Coordinate helpers ─────────────────────────────────────────────────────

    @staticmethod
    def normalise_coords(r: torch.Tensor, z: torch.Tensor,
                         t: torch.Tensor) -> torch.Tensor:
        """
        Stack physical coordinates → normalised input tensor.

        r : radial [m]  →  r / B_RADIAL  ∈ [0, 1]
        z : axial  [m]  →  z / A_AXIAL   ∈ [-1, 1]
        t : time   [s]  →  t / TOTAL_TIME_S ∈ [0, 1]
        """
        from config import TOTAL_TIME_S
        r_n = r / B_RADIAL
        z_n = z / A_AXIAL
        t_n = t / TOTAL_TIME_S
        return torch.stack([r_n, z_n, t_n], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity-check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    net = CardamomPINN()
    x   = torch.rand(128, 3)
    T, c = net(x)
    print(f"CardamomPINN (PhysicsNeMo)")
    print(f"  Trainable params : {net.count_parameters():,}")
    print(f"  T_norm shape     : {T.shape}")
    print(f"  c_norm shape     : {c.shape}")
    print(f"  JIT / AMP / auto_grad : {net.meta.jit} / {net.meta.amp} / {net.meta.auto_grad}")
