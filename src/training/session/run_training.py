"""YAML-facing training entry point."""

from __future__ import annotations

from pathlib import Path

from ...utils.config_loader import load_config
from .training_session import SessionFactories, TrainingSession


def run_training(
    config_path: str | Path,
    *,
    factories: SessionFactories | None = None,
    execution_builders=None,
) -> TrainingSession:
    """Consume one complete YAML file and execute its training session."""
    config = load_config(config_path)
    session = TrainingSession.from_config(
        config,
        factories=factories,
        execution_builders=execution_builders,
    )
    return session.run()

