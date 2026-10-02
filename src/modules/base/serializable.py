"""Configuration-plus-weights serialization contract."""

from abc import ABC, abstractmethod
from typing import ClassVar


class Serializable(ABC):
    """A component reconstructable from configuration and a state dictionary."""

    SERIALIZATION_FORMAT_VERSION: ClassVar[int] = 1
    component_type: ClassVar[str | None] = None

    @abstractmethod
    def get_config(self) -> dict:
        """Return the complete constructor configuration for this component."""

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)

    def serialize(self) -> dict:
        if self.component_type is None:
            raise RuntimeError(
                f"{type(self).__name__} does not declare component_type"
            )

        return {
            "format_version": self.SERIALIZATION_FORMAT_VERSION,
            "component_type": self.component_type,
            "config": self.get_config(),
            "state_dict": self.state_dict(),
        }

    @classmethod
    def from_serialized(cls, payload: dict):
        version = payload.get("format_version")
        if version != cls.SERIALIZATION_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported serialization format {version!r}; "
                f"expected {cls.SERIALIZATION_FORMAT_VERSION}"
            )

        stored_type = payload.get("component_type")
        if stored_type != cls.component_type:
            raise ValueError(
                f"Serialized component type {stored_type!r} does not match "
                f"{cls.component_type!r}"
            )

        component = cls.from_config(dict(payload["config"]))
        component.load_state_dict(payload["state_dict"], strict=True)
        return component

