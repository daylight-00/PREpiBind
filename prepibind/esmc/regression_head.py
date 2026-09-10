# Vendored from esm 3.4.0, esm/layers/regression_head.py. MIT, Copyright 2026 Chan Zuckerberg
# Biohub, Inc. -- see LICENSE-esm.md. Upstream's 3.4.0 and 3.1.6 copies of this file are
# byte-identical, and the body below is that file unchanged.

import torch.nn as nn


def RegressionHead(
    d_model: int, output_dim: int, hidden_dim: int | None = None
) -> nn.Module:
    """Single-hidden layer MLP for supervised output.

    Args:
        d_model: input dimension
        output_dim: dimensionality of the output.
        hidden_dim: optional dimension of hidden layer, defaults to d_model.
    Returns:
        output MLP module.
    """
    hidden_dim = hidden_dim if hidden_dim is not None else d_model
    return nn.Sequential(
        nn.Linear(d_model, hidden_dim),
        nn.GELU(),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, output_dim),
    )
