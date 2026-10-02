"""Model-component and model-bundle persistence."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import torch

from .registry import get_component_class


MODEL_BUNDLE_FORMAT_VERSION = 1


class ModelPersistenceManager:
    """Persist model components without training-session state."""

    @staticmethod
    def reconstruct_component(payload: dict):
        component_type = payload.get("component_type")
        if not component_type:
            raise ValueError("Serialized component is missing component_type")
        component_class = get_component_class(component_type)
        return component_class.from_serialized(payload)

    @classmethod
    def save_model_bundle(cls, path, components) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "format_version": MODEL_BUNDLE_FORMAT_VERSION,
            "components": {
                name: component.serialize()
                for name, component in components.items()
            },
        }

        with NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)

        try:
            torch.save(payload, temporary_path)
            temporary_path.replace(destination)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

        return destination

    @classmethod
    def load_model_bundle(cls, path, *, map_location="cpu") -> dict:
        payload = torch.load(
            Path(path),
            map_location=map_location,
            weights_only=False,
        )
        version = payload.get("format_version")
        if version != MODEL_BUNDLE_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported model bundle format {version!r}; "
                f"expected {MODEL_BUNDLE_FORMAT_VERSION}"
            )

        serialized = payload.get("components")
        if not isinstance(serialized, dict):
            raise ValueError("Model bundle components must be a mapping")

        return {
            name: cls.reconstruct_component(component_payload)
            for name, component_payload in serialized.items()
        }

