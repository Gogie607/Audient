"""Mixed-precision normalization behavior."""

from __future__ import annotations

import unittest
import warnings

import torch

from src.modules.components.modules import Float32RMSNorm


class Float32RMSNormTests(unittest.TestCase):
    def test_bfloat16_input_returns_bfloat16_without_dtype_warning(self):
        layer = Float32RMSNorm(8)
        values = torch.randn(2, 4, 8, dtype=torch.bfloat16, requires_grad=True)

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            output = layer(values)
            output.float().sum().backward()

        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(layer.weight.dtype, torch.float32)
        self.assertIsNotNone(layer.weight.grad)
        self.assertFalse(any(
            "Mismatch dtype between input and weight" in str(item.message)
            for item in captured
        ))

    def test_matches_explicit_float32_reference(self):
        layer = Float32RMSNorm(4, eps=1e-6)
        values = torch.randn(3, 4, dtype=torch.bfloat16)

        output = layer(values)
        reference = torch.nn.functional.rms_norm(
            values.float(),
            (4,),
            layer.weight,
            1e-6,
        ).to(torch.bfloat16)

        torch.testing.assert_close(output, reference)


if __name__ == "__main__":
    unittest.main()
