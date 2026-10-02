"""Concrete model container satisfying RunWeaver's model-control protocol."""

from collections.abc import Mapping

from torch import nn

from .base import Controllable, ModelComponent, Serializable
from .persistence import ModelPersistenceManager
from .registry import create_component


class TrainingModelWrapper(Controllable):
    def __init__(self, components: Mapping[str, ModelComponent] | None = None):
        self._components: dict[str, ModelComponent] = {}
        for name, component in (components or {}).items():
            self.add_component(name, component)

    @classmethod
    def from_config(cls, component_specs: list[dict], *, device=None):
        wrapper = cls()
        for spec in component_specs:
            name = spec["name"]
            component = create_component(
                spec["type"],
                spec.get("config", {}),
            )
            wrapper.add_component(name, component)

        if device is not None:
            wrapper.to(device)
        return wrapper

    @classmethod
    def load(cls, path, *, map_location="cpu", device=None):
        wrapper = cls(
            ModelPersistenceManager.load_model_bundle(
                path,
                map_location=map_location,
            )
        )
        if device is not None:
            wrapper.to(device)
        return wrapper

    def save(self, path):
        return ModelPersistenceManager.save_model_bundle(
            path,
            self._components,
        )

    def add_component(
        self,
        name: str,
        component: ModelComponent,
        *,
        overwrite: bool = False,
    ) -> ModelComponent:
        if not isinstance(name, str) or not name:
            raise TypeError("component name must be a non-empty string")
        if not isinstance(component, (nn.Module, Controllable, Serializable)):
            raise TypeError(
                "component must implement nn.Module, Controllable, and Serializable"
            )
        if not isinstance(component, ModelComponent):
            raise TypeError("component must inherit from ModelComponent")
        if name in self._components and not overwrite:
            raise KeyError(f"Component {name!r} already exists")

        self._components[name] = component
        return component

    def get(self, name: str) -> ModelComponent:
        try:
            return self._components[name]
        except KeyError as error:
            raise KeyError(f"Unknown model component {name!r}") from error

    def items(self):
        return self._components.items()

    def values(self):
        return self._components.values()

    def keys(self):
        return self._components.keys()

    def get_mode(self) -> dict[str, object]:
        return {
            name: component.get_mode()
            for name, component in self._components.items()
        }

    def set_mode(self, state: Mapping[str, object]):
        self._apply_component_state(state, "set_mode")
        return self

    def get_trainable(self) -> dict[str, object]:
        return {
            name: component.get_trainable()
            for name, component in self._components.items()
        }

    def set_trainable(self, state: Mapping[str, object]):
        self._apply_component_state(state, "set_trainable")
        return self

    def _apply_component_state(self, state, method_name: str) -> None:
        if not isinstance(state, Mapping):
            raise TypeError(f"{method_name} expects a component-state mapping")
        for name, value in state.items():
            component = self.get(name)
            getattr(component, method_name)(value)

    def trainable_parameters(self):
        seen = set()
        for component in self._components.values():
            for parameter in component.parameters():
                if parameter.requires_grad and id(parameter) not in seen:
                    seen.add(id(parameter))
                    yield parameter

    def to(self, device):
        for component in self._components.values():
            component.to(device)
        return self

