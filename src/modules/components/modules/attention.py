"""Mask-aware reusable attention blocks."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .feed_forward import FeedForward


def _validate_padding_mask(values: Tensor, padding_mask: Tensor | None) -> None:
    if padding_mask is None:
        return
    if padding_mask.shape != values.shape[:2]:
        raise ValueError("padding_mask must have shape [batch, frames]")
    if padding_mask.dtype is not torch.bool:
        raise TypeError("padding_mask must be boolean")
    if padding_mask.all(dim=1).any():
        raise ValueError("every sample must contain at least one valid position")


class SelfAttentionBlock(nn.Module):
    """Pre-normalized self-attention and feed-forward residual block."""

    def __init__(
        self,
        model_dim: int,
        *,
        num_heads: int,
        ff_hidden_dim: int | None = None,
        ff_expansion: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(model_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feed_forward_norm = nn.LayerNorm(model_dim)
        self.feed_forward = FeedForward(
            model_dim,
            hidden_dim=ff_hidden_dim,
            expansion=ff_expansion,
            dropout=dropout,
        )

    def forward(
        self,
        values: Tensor,
        *,
        padding_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        _validate_padding_mask(values, padding_mask)
        normalized = self.attention_norm(values)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=padding_mask,
            attn_mask=attention_mask,
            need_weights=False,
        )
        values = values + attended
        return values + self.feed_forward(self.feed_forward_norm(values))


class CrossAttentionBlock(nn.Module):
    """Pre-normalized cross-attention and feed-forward residual block."""

    def __init__(
        self,
        model_dim: int,
        *,
        num_heads: int,
        ff_hidden_dim: int | None = None,
        ff_expansion: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(model_dim)
        self.source_norm = nn.LayerNorm(model_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feed_forward_norm = nn.LayerNorm(model_dim)
        self.feed_forward = FeedForward(
            model_dim,
            hidden_dim=ff_hidden_dim,
            expansion=ff_expansion,
            dropout=dropout,
        )

    def forward(
        self,
        query: Tensor,
        source: Tensor,
        *,
        source_padding_mask: Tensor | None = None,
    ) -> Tensor:
        _validate_padding_mask(source, source_padding_mask)
        normalized_source = self.source_norm(source)
        attended, _ = self.attention(
            self.query_norm(query),
            normalized_source,
            normalized_source,
            key_padding_mask=source_padding_mask,
            need_weights=False,
        )
        query = query + attended
        return query + self.feed_forward(self.feed_forward_norm(query))


class AttentionPool(nn.Module):
    """Pool a variable-length sequence with one learned attention query."""

    def __init__(self, model_dim: int, *, num_heads: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.empty(1, 1, model_dim))
        nn.init.normal_(self.query, std=0.02)
        self.source_norm = nn.LayerNorm(model_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        values: Tensor,
        *,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        _validate_padding_mask(values, padding_mask)
        query = self.query.expand(values.shape[0], -1, -1)
        normalized = self.source_norm(values)
        pooled, _ = self.attention(
            query=query,
            key=normalized,
            value=normalized,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        return self.output_norm(pooled[:, 0])


class LearnedQueryPool(nn.Module):
    """Produce a configured number of summaries from a source sequence."""

    def __init__(
        self,
        model_dim: int,
        *,
        output_tokens: int,
        num_heads: int,
        ff_expansion: float = 2.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if output_tokens <= 0:
            raise ValueError("output_tokens must be positive")
        self.queries = nn.Parameter(torch.empty(1, output_tokens, model_dim))
        nn.init.normal_(self.queries, std=0.02)
        self.cross_attention = CrossAttentionBlock(
            model_dim,
            num_heads=num_heads,
            ff_expansion=ff_expansion,
            dropout=dropout,
        )
        self.output_norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        values: Tensor,
        *,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        queries = self.queries.expand(values.shape[0], -1, -1)
        return self.output_norm(
            self.cross_attention(
                queries,
                values,
                source_padding_mask=padding_mask,
            )
        )
