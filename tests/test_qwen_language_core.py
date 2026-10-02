"""Tests for the external frozen Qwen language-core boundary."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from src.modules import QwenLanguageCore, QwenLanguageCoreConfig


class FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        del kwargs
        batch_size = len(text) if isinstance(text, list) else 1
        return {
            "input_ids": torch.ones(batch_size, 2, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, 2, dtype=torch.long),
        }


class FakeQwen(nn.Module):
    def __init__(self, hidden_size: int = 16) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(32, hidden_size)
        self.output = nn.Linear(hidden_size, 32, bias=False)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, *, inputs_embeds, **kwargs):
        del kwargs
        return SimpleNamespace(logits=self.output(inputs_embeds))


class RecordingFactory:
    def __init__(self, value) -> None:
        self.value = value
        self.calls = []

    def from_pretrained(self, model_name, **kwargs):
        self.calls.append((model_name, kwargs))
        return self.value


def make_core() -> QwenLanguageCore:
    return QwenLanguageCore(
        model=FakeQwen(),
        tokenizer=FakeTokenizer(),
        config=QwenLanguageCoreConfig(
            model_name="test/qwen",
            expected_hidden_size=16,
        ),
    )


class QwenLanguageCoreTests(unittest.TestCase):
    def test_initializes_frozen_and_stays_in_eval_mode(self) -> None:
        core = make_core()
        core.train(True)

        self.assertFalse(core.training)
        self.assertFalse(core.model.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in core.parameters()))

    def test_forward_preserves_gradient_to_input_embeddings(self) -> None:
        core = make_core()
        inputs = torch.randn(2, 5, 16, requires_grad=True)
        output = core(
            inputs_embeds=inputs,
            attention_mask=torch.ones(2, 5, dtype=torch.long),
        )

        output.logits.sum().backward()

        self.assertIsNotNone(inputs.grad)
        self.assertTrue(all(parameter.grad is None for parameter in core.parameters()))

    def test_embedding_and_tokenizer_boundaries(self) -> None:
        core = make_core()
        tokens = core.tokenize(["one", "two"])
        embeddings = core.embed_tokens(tokens["input_ids"])

        self.assertEqual(embeddings.shape, (2, 2, 16))
        self.assertEqual(core.hidden_size, 16)

    def test_rejects_wrong_input_width(self) -> None:
        core = make_core()
        with self.assertRaises(ValueError):
            core(inputs_embeds=torch.randn(1, 2, 8))

    def test_configuration_factory_loads_external_objects(self) -> None:
        model_factory = RecordingFactory(FakeQwen())
        tokenizer_factory = RecordingFactory(FakeTokenizer())
        config = {
            "model_name": "test/qwen",
            "revision": "test-revision",
            "dtype": "float32",
            "expected_hidden_size": 16,
        }

        core = QwenLanguageCore.from_config(
            config,
            model_factory=model_factory,
            tokenizer_factory=tokenizer_factory,
        )

        self.assertEqual(core.get_config(), QwenLanguageCoreConfig.from_config(config).to_config())
        self.assertEqual(model_factory.calls[0][0], "test/qwen")
        self.assertEqual(model_factory.calls[0][1]["dtype"], torch.float32)
        self.assertEqual(tokenizer_factory.calls[0][1]["revision"], "test-revision")

    def test_expected_hidden_size_is_enforced(self) -> None:
        with self.assertRaises(ValueError):
            QwenLanguageCore(
                model=FakeQwen(hidden_size=8),
                tokenizer=FakeTokenizer(),
                config={"model_name": "test/qwen", "expected_hidden_size": 16},
            )


if __name__ == "__main__":
    unittest.main()
