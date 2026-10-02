"""Semantic prefix head for duration-preserving audio representations."""

from __future__ import annotations

from torch import Tensor, nn

from ..modules import Float32RMSNorm, MaskedTemporalReducer, SelfAttentionBlock


class SemanticPrefixHead(nn.Module):
    """Reduce a sequence locally and adapt it to a language-model width."""

    def __init__(
        self,
        model_dim: int,
        output_dim: int,
        *,
        reduction: int,
        num_blocks: int,
        num_heads: int,
        ff_hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_blocks < 0:
            raise ValueError("num_blocks must be non-negative")
        self.reduction = MaskedTemporalReducer(model_dim, reduction=reduction)
        self.blocks = nn.ModuleList(
            SelfAttentionBlock(
                model_dim,
                num_heads=num_heads,
                ff_hidden_dim=ff_hidden_dim,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        )
        self.output = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, output_dim),
            Float32RMSNorm(output_dim),
        )

    def forward(
        self,
        values: Tensor,
        *,
        padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        values, reduced_padding_mask = self.reduction(
            values,
            padding_mask=padding_mask,
        )
        for block in self.blocks:
            values = block(values, padding_mask=reduced_padding_mask)
        return self.output(values), reduced_padding_mask
