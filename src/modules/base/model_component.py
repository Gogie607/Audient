"""Base class for concrete model components."""

from abc import ABC

from .controllable import ControllableNode
from .serializable import Serializable


class ModelComponent(ControllableNode, Serializable, ABC):
    """Trainable component with stable type registration and persistence."""

    component_type = None

    def initialize_frozen(self):
        """Place a fully constructed component in its required initial state.

        Concrete constructors call this after attaching all child modules and
        parameters. Phase setup is responsible for enabling training later.
        """
        self.set_trainable(False)
        self.set_mode(False)
        return self

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

        component_type = getattr(cls, "component_type", None)
        if component_type is None:
            return
        if not isinstance(component_type, str) or not component_type.strip():
            raise TypeError("component_type must be a non-empty string")

        from ..registry import register_component

        register_component(component_type, cls)
