import tempfile
import unittest
from pathlib import Path

import torch

from runweaver_ml.phase_control import TrainingModelProtocol
from runweaver_ml.phase_control.execution import system_state, system_trainable

from src.modules import (
    ModelComponent,
    TrainingModelWrapper,
    create_component,
)


class LinearTestComponent(ModelComponent):
    component_type = "foundation_test_linear"

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.projection = torch.nn.Linear(input_dim, output_dim)
        self.initialize_frozen()

    def forward(self, value):
        return self.projection(value)

    def get_config(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
        }


class ParameterlessComponent(ModelComponent):
    component_type = "foundation_test_parameterless"

    def __init__(self):
        super().__init__()
        self.initialize_frozen()

    def get_config(self) -> dict:
        return {}


class ModelFoundationTest(unittest.TestCase):
    def test_registry_builds_component_from_stable_type(self):
        component = create_component(
            "foundation_test_linear",
            {"input_dim": 3, "output_dim": 2},
        )

        self.assertIsInstance(component, LinearTestComponent)
        self.assertEqual(component.get_config(), {"input_dim": 3, "output_dim": 2})
        self.assertFalse(component.training)
        self.assertFalse(component.get_trainable())

    def test_model_bundle_round_trip_preserves_config_and_weights(self):
        component = LinearTestComponent(3, 2)
        with torch.no_grad():
            component.projection.weight.fill_(2.5)
            component.projection.bias.fill_(-0.5)

        wrapper = TrainingModelWrapper({"adapter": component})

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            wrapper.save(path)
            loaded = TrainingModelWrapper.load(path)

        restored = loaded.get("adapter")
        self.assertEqual(restored.get_config(), component.get_config())
        self.assertFalse(restored.training)
        self.assertFalse(restored.get_trainable())
        self.assertTrue(
            torch.equal(restored.projection.weight, component.projection.weight)
        )
        self.assertTrue(
            torch.equal(restored.projection.bias, component.projection.bias)
        )

    def test_wrapper_satisfies_runweaver_protocol(self):
        wrapper = TrainingModelWrapper(
            {"adapter": LinearTestComponent(3, 2)}
        )
        self.assertIsInstance(wrapper, TrainingModelProtocol)

    def test_mode_and_trainability_contexts_restore_state(self):
        component = LinearTestComponent(3, 2)
        wrapper = TrainingModelWrapper({"adapter": component})

        wrapper.set_trainable({"adapter": True})
        wrapper.set_mode({"adapter": True})
        with system_state(wrapper, {"adapter": False}):
            self.assertFalse(component.training)
        self.assertTrue(component.training)

        with system_trainable(wrapper, ["adapter"]):
            self.assertTrue(component.get_trainable())
        self.assertTrue(component.get_trainable())

        wrapper.set_trainable({"adapter": False})
        with system_trainable(wrapper, ["adapter"]):
            self.assertTrue(component.get_trainable())
        self.assertFalse(component.get_trainable())

    def test_parameterless_component_has_cpu_device(self):
        self.assertEqual(ParameterlessComponent().device, torch.device("cpu"))

    def test_components_initialize_frozen_and_in_eval_mode(self):
        component = LinearTestComponent(3, 2)

        self.assertFalse(component.training)
        self.assertFalse(component.get_trainable())
        self.assertTrue(
            all(not parameter.requires_grad for parameter in component.parameters())
        )


if __name__ == "__main__":
    unittest.main()
