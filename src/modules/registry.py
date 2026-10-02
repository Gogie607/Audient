"""Model-component registration and configuration construction."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base.model_component import ModelComponent


MODEL_COMPONENT_REGISTRY: dict[str, type["ModelComponent"]] = {}


def register_component(
    component_type: str,
    component_class: type["ModelComponent"],
) -> None:
    existing = MODEL_COMPONENT_REGISTRY.get(component_type)
    if existing is not None and existing is not component_class:
        raise RuntimeError(
            f"Component type {component_type!r} is already registered by "
            f"{existing.__module__}.{existing.__qualname__}"
        )
    MODEL_COMPONENT_REGISTRY[component_type] = component_class


def get_component_class(component_type: str) -> type["ModelComponent"]:
    try:
        return MODEL_COMPONENT_REGISTRY[component_type]
    except KeyError as error:
        available = ", ".join(sorted(MODEL_COMPONENT_REGISTRY)) or "<none>"
        raise ValueError(
            f"Unknown model component type {component_type!r}. "
            f"Registered types: {available}"
        ) from error


def create_component(component_type: str, config: dict):
    return get_component_class(component_type).from_config(dict(config))


def registered_components() -> dict[str, type["ModelComponent"]]:
    return dict(MODEL_COMPONENT_REGISTRY)

