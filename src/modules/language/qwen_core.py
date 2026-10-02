"""Frozen Qwen runtime used as the semantic language core."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, AutoTokenizer


_TORCH_DTYPES: Mapping[str, torch.dtype] = MappingProxyType(
    {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
)


@dataclass(frozen=True)
class QwenLanguageCoreConfig:
    """Reconstruction configuration for an external frozen Qwen model."""

    model_name: str = "Qwen/Qwen3-1.7B"
    revision: str | None = None
    dtype: str = "auto"
    device_map: str | Mapping[str, Any] | None = None
    trust_remote_code: bool = False
    use_fast_tokenizer: bool = True
    expected_hidden_size: int | None = None
    gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name must not be empty")
        if self.dtype != "auto" and self.dtype not in _TORCH_DTYPES:
            raise ValueError(
                f"unsupported dtype {self.dtype!r}; "
                f"expected one of {['auto', *_TORCH_DTYPES]}"
            )
        if self.expected_hidden_size is not None and self.expected_hidden_size <= 0:
            raise ValueError("expected_hidden_size must be positive")

    @classmethod
    def from_config(
        cls,
        config: QwenLanguageCoreConfig | Mapping[str, Any],
    ) -> QwenLanguageCoreConfig:
        if isinstance(config, cls):
            return config
        return cls(**dict(config))

    def to_config(self) -> dict[str, Any]:
        return asdict(self)

    def resolved_dtype(self) -> str | torch.dtype:
        if self.dtype == "auto":
            return "auto"
        return _TORCH_DTYPES[self.dtype]


class QwenLanguageCore(nn.Module):
    """Non-checkpointed frozen Qwen with a differentiable input path."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: QwenLanguageCoreConfig | Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.config = QwenLanguageCoreConfig.from_config(config)
        self.model = model
        self.tokenizer = tokenizer
        self._freeze()
        self._validate_hidden_size()

    @classmethod
    def from_config(
        cls,
        config: QwenLanguageCoreConfig | Mapping[str, Any],
        *,
        device: str | torch.device | None = None,
        model_factory: Any = AutoModelForCausalLM,
        tokenizer_factory: Any = AutoTokenizer,
    ) -> QwenLanguageCore:
        resolved = QwenLanguageCoreConfig.from_config(config)
        if device is not None and resolved.device_map is not None:
            raise ValueError("device and device_map cannot both be specified")

        common_kwargs: dict[str, Any] = {
            "trust_remote_code": resolved.trust_remote_code,
        }
        if resolved.revision is not None:
            common_kwargs["revision"] = resolved.revision

        tokenizer = tokenizer_factory.from_pretrained(
            resolved.model_name,
            use_fast=resolved.use_fast_tokenizer,
            **common_kwargs,
        )
        model_kwargs = dict(common_kwargs)
        model_kwargs["dtype"] = resolved.resolved_dtype()
        if resolved.device_map is not None:
            model_kwargs["device_map"] = resolved.device_map
        model = model_factory.from_pretrained(resolved.model_name, **model_kwargs)

        if resolved.gradient_checkpointing:
            enable = getattr(model, "gradient_checkpointing_enable", None)
            if enable is None:
                raise TypeError("configured Qwen model lacks gradient checkpointing")
            enable()

        core = cls(model=model, tokenizer=tokenizer, config=resolved)
        if device is not None:
            core.to(device)
        return core

    @property
    def hidden_size(self) -> int:
        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is None:
            text_config = getattr(self.model.config, "text_config", None)
            hidden_size = getattr(text_config, "hidden_size", None)
        if hidden_size is None:
            raise AttributeError("Qwen configuration does not expose hidden_size")
        return int(hidden_size)

    @property
    def embedding_layer(self) -> nn.Module:
        embedding_layer = self.model.get_input_embeddings()
        if embedding_layer is None:
            raise RuntimeError("Qwen model does not expose input embeddings")
        return embedding_layer

    @property
    def device(self) -> torch.device:
        weight = getattr(self.embedding_layer, "weight", None)
        if weight is not None:
            return weight.device
        parameter = next(self.model.parameters(), None)
        return parameter.device if parameter is not None else torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        weight = getattr(self.embedding_layer, "weight", None)
        if weight is not None:
            return weight.dtype
        parameter = next(self.model.parameters(), None)
        return parameter.dtype if parameter is not None else torch.float32

    def tokenize(self, text: str | list[str], **kwargs):
        """Tokenize text without silently moving it to a model device."""
        return self.tokenizer(text, **kwargs)

    def embed_tokens(self, input_ids: Tensor) -> Tensor:
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must contain integer token identifiers")
        return self.embedding_layer(input_ids)

    def forward(
        self,
        *,
        inputs_embeds: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs,
    ):
        if inputs_embeds.ndim != 3:
            raise ValueError("inputs_embeds must have shape [batch, tokens, hidden]")
        if inputs_embeds.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Qwen expects hidden size {self.hidden_size}, "
                f"received {inputs_embeds.shape[-1]}"
            )
        past_key_values = kwargs.get("past_key_values")
        if attention_mask is not None:
            if past_key_values is None and attention_mask.shape != inputs_embeds.shape[:2]:
                raise ValueError("attention_mask must have shape [batch, tokens]")
            if past_key_values is not None and (
                attention_mask.shape[0] != inputs_embeds.shape[0]
                or attention_mask.shape[1] < inputs_embeds.shape[1]
            ):
                raise ValueError(
                    "cached attention_mask must match the batch and include "
                    "the current input tokens"
                )
        if labels is not None and labels.shape != inputs_embeds.shape[:2]:
            raise ValueError("labels must have shape [batch, tokens]")

        return self.model(
            inputs_embeds=inputs_embeds.to(dtype=self.dtype),
            attention_mask=attention_mask,
            labels=labels,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

    def train(self, mode: bool = True) -> QwenLanguageCore:
        del mode
        super().train(False)
        self._freeze()
        return self

    def to(self, *args, **kwargs) -> QwenLanguageCore:
        super().to(*args, **kwargs)
        self._freeze()
        return self

    def offload(self) -> QwenLanguageCore:
        return self.to(torch.device("cpu"))

    def get_config(self) -> dict[str, Any]:
        return self.config.to_config()

    def _freeze(self) -> None:
        nn.Module.train(self, False)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def _validate_hidden_size(self) -> None:
        expected = self.config.expected_hidden_size
        if expected is not None and self.hidden_size != expected:
            raise ValueError(
                f"configured Qwen hidden size {expected} does not match "
                f"loaded model hidden size {self.hidden_size}"
            )
