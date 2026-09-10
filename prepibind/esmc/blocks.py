# Vendored from esm 3.4.0, esm/layers/blocks.py. MIT, Copyright 2026 Chan Zuckerberg Biohub,
# Inc. -- see LICENSE-esm.md.
# Stripped: geometric attention (ESMC builds every block with n_layers_geom=0) and gelu_ln_ffn
# (ffn_type is always "swiglu"), and with them the Affine3D import; and `output_attentions`,
# which only ever forwarded a flag to the attention branch this package does not carry. With it
# gone, forward returns the tensor rather than 3.4.0's (x, attn_weights) pair.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import FlashMultiHeadAttention, MultiHeadAttention


def swiglu_correction_fn(expansion_ratio: float, d_model: int) -> int:
    # set hidden dimesion to nearest multiple of 256 after expansion ratio
    return int(((expansion_ratio * d_model) + 255) // 256 * 256)


class SwiGLU(nn.Module):
    """
    SwiGLU activation function as an nn.Module, allowing it to be used within nn.Sequential.
    This module splits the input tensor along the last dimension and applies the SiLU (Swish)
    activation function to the first half, then multiplies it by the second half.
    """

    def __init__(self):
        super(SwiGLU, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2


def swiglu_ln_ffn(d_model: int, expansion_ratio: float, bias: bool):
    # SwiGLU has to be a real module: it holds index 2 of the Sequential, and the checkpoint's
    # keys are ffn.0 / ffn.1 / ffn.3.
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(
            d_model, swiglu_correction_fn(expansion_ratio, d_model) * 2, bias=bias
        ),
        SwiGLU(),
        nn.Linear(swiglu_correction_fn(expansion_ratio, d_model), d_model, bias=bias),
    )


class UnifiedTransformerBlock(nn.Module):
    """
    A transformer block: multi-head attention, then a SwiGLU feed-forward, each added back to the
    residual stream after division by `residue_scaling_factor`.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        use_flash_attn: bool = False,
        bias: bool = False,
        expansion_ratio: float = 4.0,
        residue_scaling_factor: float = 1,
        qk_layernorm: bool = True,
    ):
        super().__init__()
        if use_flash_attn:
            self.attn = FlashMultiHeadAttention(
                d_model, n_heads, bias, qk_layernorm=qk_layernorm
            )
        else:
            self.attn = MultiHeadAttention(
                d_model, n_heads, bias, qk_layernorm=qk_layernorm
            )
        self.ffn = swiglu_ln_ffn(d_model, expansion_ratio, bias)
        self.scaling_factor = residue_scaling_factor

    def forward(self, x: torch.Tensor, sequence_id: torch.Tensor) -> torch.Tensor:
        r1 = self.attn(x, sequence_id)
        x = x + r1 / self.scaling_factor

        r3 = self.ffn(x) / self.scaling_factor
        x = x + r3

        return x
