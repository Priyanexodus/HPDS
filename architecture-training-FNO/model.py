"""
model.py
========
PhysicsNeMo-based FNO (Fourier Neural Operator) for Cardamom HPDS Drying.

Builds on PhysicsNeMo primitives:
  - physicsnemo.core.Module          → base class (AMP, JIT, ONNX metadata)
  - physicsnemo.core.ModelMetaData   → capability flags
  - physicsnemo.nn.SpectralConv2d    → core FNO spectral convolution layer
  - physicsnemo.nn.FCLayer           → Xavier-init dense layer for lifting/decoder

Architecture  (standard FNO for 2-D spatial fields)
-----------
  Input  : (B, 3, 8, 16)  — normalised [T_current, M_current, T_heater]
  ↓
  Lifting layer  : Conv2d  3 → LATENT_CHANNELS
  ↓
  N × FNO blocks: SpectralConv2d + pointwise Conv2d  + residual add
  ↓
  Decoder MLP    : Conv2d LATENT_CHANNELS → DECODER_HIDDEN → OUT_CHANNELS
  Output : (B, 2, 8, 16)  — normalised [T_future, M_future]

Reference: Li et al., "Fourier Neural Operator for Parametric PDEs", ICLR 2021.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from physicsnemo.core import ModelMetaData, Module
from physicsnemo.nn import SpectralConv2d, FCLayer, get_activation

from config import (
    IN_CHANNELS, OUT_CHANNELS,
    LATENT_CHANNELS, NUM_FNO_LAYERS, NUM_MODES_X, NUM_MODES_Y,
    DECODER_HIDDEN, DECODER_LAYERS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Metadata flags
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FNOMetaData(ModelMetaData):
    name: str = "CardamomFNO"
    # Optimisation
    jit: bool = False
    cuda_graphs: bool = False
    # AMP disabled: SpectralConv2d does FFT → complex tensors.
    # ComplexHalf (float16 complex) is not supported on most consumer GPUs.
    amp: bool = False
    torch_fx: bool = False
    # Inference
    onnx: bool = True
    onnx_runtime: bool = True
    # Physics-informed — FNO is purely data-driven (no autograd for PDEs)
    func_torch: bool = False
    auto_grad: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Single FNO Block
# ─────────────────────────────────────────────────────────────────────────────

class FNOBlock2d(nn.Module):
    """
    One Fourier Neural Operator block for 2-D spatial data.

    Computes:
        h = activate( SpectralConv2d(x) + W(x) )

    where W is a pointwise (1×1) convolution (the "bypass" branch).

    Parameters
    ----------
    channels : int
        Number of input = output channels.
    modes1   : int
        Fourier modes to keep along the first spatial axis (H).
    modes2   : int
        Fourier modes to keep along the second spatial axis (W).
    activation : str
        Activation name from physicsnemo.nn.get_activation.
    """

    def __init__(
        self,
        channels: int,
        modes1:   int,
        modes2:   int,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, modes1, modes2)
        self.bypass   = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.act      = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, H, W)

        Returns
        -------
        h : (B, C, H, W)
        """
        return self.act(self.spectral(x) + self.bypass(x))


# ─────────────────────────────────────────────────────────────────────────────
# Full FNO model
# ─────────────────────────────────────────────────────────────────────────────

class CardamomFNO(Module):
    """
    Fourier Neural Operator surrogate for cardamom pod drying.

    Maps (T, M, T_heater) → (T_future, M_future) 6 minutes ahead
    on an 8×16 spatial grid.

    Parameters
    ----------
    in_channels      : int   — number of input channels (default 3)
    out_channels     : int   — number of output channels (default 2)
    latent_channels  : int   — width of spectral layers
    num_fno_layers   : int   — number of FNO blocks
    modes1           : int   — Fourier modes along H
    modes2           : int   — Fourier modes along W
    decoder_hidden   : int   — hidden width of output MLP
    decoder_layers   : int   — depth of output MLP
    activation       : str   — activation (e.g. "gelu", "relu")
    """

    def __init__(
        self,
        in_channels:     int = IN_CHANNELS,
        out_channels:    int = OUT_CHANNELS,
        latent_channels: int = LATENT_CHANNELS,
        num_fno_layers:  int = NUM_FNO_LAYERS,
        modes1:          int = NUM_MODES_X,
        modes2:          int = NUM_MODES_Y,
        decoder_hidden:  int = DECODER_HIDDEN,
        decoder_layers:  int = DECODER_LAYERS,
        activation:      str = "gelu",
    ) -> None:
        super().__init__(meta=FNOMetaData())

        self.in_channels    = in_channels
        self.out_channels   = out_channels
        self.latent_channels = latent_channels
        self.num_fno_layers = num_fno_layers
        self.modes1 = modes1
        self.modes2 = modes2

        # ── Lifting layer: project input channels → latent space ──────────────
        self.lifting = nn.Conv2d(in_channels, latent_channels,
                                 kernel_size=1, bias=True)

        # ── Stack of FNO blocks ───────────────────────────────────────────────
        self.fno_blocks = nn.ModuleList([
            FNOBlock2d(latent_channels, modes1, modes2, activation=activation)
            for _ in range(num_fno_layers)
        ])

        # ── Decoder: latent → output via pointwise MLP ────────────────────────
        # Build as a sequence of Conv2d (kernel_size=1) layers acting as MLP
        # over the channel dimension at every spatial point.
        decoder_sizes = [latent_channels] + [decoder_hidden] * decoder_layers + [out_channels]
        act_fn = get_activation(activation)
        layers = []
        for i in range(len(decoder_sizes) - 1):
            layers.append(nn.Conv2d(decoder_sizes[i], decoder_sizes[i + 1],
                                    kernel_size=1, bias=True))
            if i < len(decoder_sizes) - 2:       # no activation on final layer
                layers.append(act_fn)
        self.decoder = nn.Sequential(*layers)

        # Weight initialisation — small near-zero for output stability
        nn.init.uniform_(self.decoder[-1].weight, -1e-2, 1e-2)
        nn.init.zeros_(self.decoder[-1].bias)

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 3, 8, 16)   normalised input [T_norm, M_norm, Th_norm]

        Returns
        -------
        y : (B, 2, 8, 16)   normalised prediction [T_future_norm, M_future_norm]
        """
        # Lift to latent space
        h = self.lifting(x)                   # (B, latent, 8, 16)

        # Spectral blocks
        for block in self.fno_blocks:
            h = block(h)                      # (B, latent, 8, 16)

        # Decode to output channels
        y = self.decoder(h)                   # (B, 2, 8, 16)
        return y

    # ── Convenience methods ────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def predict_physical(
        self,
        x_norm: torch.Tensor,
        denorm_fn,
    ) -> torch.Tensor:
        """
        Run forward pass and denormalise output to physical units.

        Parameters
        ----------
        x_norm   : (B, 3, 8, 16) normalised input
        denorm_fn: callable — dataset._denorm_y

        Returns
        -------
        y_phys : (B, 2, 8, 16)  [T_future °C, M_future kg_w/kg_dry]
        """
        with torch.no_grad():
            y_norm = self.forward(x_norm)
        return denorm_fn(y_norm)


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity-check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    net = CardamomFNO()
    x   = torch.rand(8, 3, 8, 16)
    y   = net(x)
    print(f"CardamomFNO (PhysicsNeMo)")
    print(f"  Trainable params : {net.count_parameters():,}")
    print(f"  Input  shape     : {x.shape}")
    print(f"  Output shape     : {y.shape}")
    print(f"  AMP / ONNX       : {net.meta.amp} / {net.meta.onnx}")
