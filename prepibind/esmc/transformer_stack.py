# Vendored from esm 3.4.0, esm/layers/transformer_stack.py. MIT, Copyright 2026 Chan Zuckerberg
# Biohub, Inc. -- see LICENSE-esm.md.
# Stripped: v_heads / n_layers_geom / mask_and_zero_frameless / ffn_type, and the affine,
# affine_mask and chain_id arguments of forward -- all of them fed geometric attention only --
# and `output_attentions`, so forward returns three values where 3.4.0 returns four.

import math

import torch
import torch.nn as nn

from .blocks import UnifiedTransformerBlock


class TransformerStack(nn.Module):
    """
    A stack of transformer blocks.

    Args:
        d_model (int): The dimensionality of the input and output feature vectors.
        n_heads (int): The number of attention heads.
        n_layers (int): The number of transformer blocks in the stack.
        scale_residue (bool, optional): Whether to scale the residue connections in each block.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        scale_residue: bool = True,
        bias: bool = False,
        qk_layernorm: bool = True,
        expansion_ratio: float = 8 / 3,
        use_flash_attn: bool = False,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                UnifiedTransformerBlock(
                    d_model,
                    n_heads,
                    use_flash_attn=use_flash_attn,
                    residue_scaling_factor=(
                        math.sqrt(n_layers / 36) if scale_residue else 1.0
                    ),
                    expansion_ratio=expansion_ratio,
                    bias=bias,
                    qk_layernorm=qk_layernorm,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        sequence_id: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        """
        Returns:
            post_norm: The output tensor of shape (batch_size, sequence_length, d_model).
            pre_norm: The embedding of shape (batch_size, sequence_length, d_model).
            hidden_states: Hidden states from each layer.
        """
        all_hidden_states: list[torch.Tensor] = []
        for block in self.blocks:
            x = block(x, sequence_id)
            all_hidden_states.append(x)
        # 3.4.0 freezes the per-layer outputs into a tuple before returning them; kept.
        hidden_states = tuple(all_hidden_states)
        return self.norm(x), x, hidden_states
