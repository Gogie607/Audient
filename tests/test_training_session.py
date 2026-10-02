"""Tests for the YAML-to-RunWeaver training-session boundary."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.training import SessionFactories, TrainingSession, run_training


class FakeModel:
    def __init__(self):
        self.parameter = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.mode = {"audio_provider": {"head": False}}
        self.trainable = {"audio_provider": {"head": False}}
        self.saved_paths = []

    def get_mode(self):
        return {
            "audio_provider": dict(self.mode["audio_provider"]),
        }

    def set_mode(self, state):
        self.mode = state

    def get_trainable(self):
        return {
            "audio_provider": dict(self.trainable["audio_provider"]),
        }

    def set_trainable(self, state):
        self.trainable = state
        self.parameter.requires_grad_(state["audio_provider"]["head"])

    def trainable_parameters(self):
        if self.parameter.requires_grad:
            yield self.parameter

    def save(self, path):
        self.saved_paths.append(Path(path))
        return Path(path)


class FakeDataset:
    def __init__(self, split):
        self.split = split
        self.active = None

    def set_active_datasets(self, active):
        self.active = active


class FakeLogger:
    def __init__(self):
        self.records = []
        self.closed = False

    def log(self, tag, step, data):
        self.records.append((tag, step, data))

    def flush(self):
        pass

    def close(self):
        self.closed = True


class FakeOrchestrator:
    calls = []

    @classmethod
    def train_epoch(cls, **kwargs):
        cls.calls.append(kwargs)
        return kwargs["loop_config"].max_steps


def make_config(save_dir: str):
    return {
        "system": {"device": "cpu"},
        "models": {
            "components": [{"name": "audio_provider", "type": "fake", "config": {}}],
            "language_core": {"model_name": "fake", "device_map": None},
            "save_dir": save_dir,
        },
        "datasets": {
            "train": {
                "requires": ["audio_encoding", "txt", "speech_traits"],
                "batch_size": 2,
                "datasets": [],
            },
            "validation": {
                "requires": ["audio_encoding", "txt", "speech_traits"],
                "batch_size": 2,
                "datasets": [],
            },
        },
        "training": {
            "optimizer": {"type": "adamw", "lr": 0.001},
            "runtime": {"schedule": {}, "amp": {"enabled": False}},
            "phases": [
                {
                    "name": "first",
                    "enabled": True,
                    "run": {
                        "steps": 3,
                        "trainable": ["audio_provider.head"],
                        "execution": "fake_execution",
                    },
                    "targets": {
                        "semantic_weight": 1.0,
                        "trait_smooth_l1_beta": 1.0,
                    },
                    "schedules": {},
                }
            ],
        },
    }


def make_factories(model, logger):
    def dataset_builder(config, runtime_artifact_repository=None):
        del runtime_artifact_repository
        return FakeDataset(config.get("name", "split"))

    return SessionFactories(
        dataset_builder=dataset_builder,
        loader_builder=lambda dataset, **kwargs: (dataset, kwargs),
        model_builder=lambda specs, device=None: model,
        model_loader=lambda *args, **kwargs: model,
        language_core_builder=lambda config, device=None: object(),
        logger_builder=lambda config: logger,
        orchestrator=FakeOrchestrator,
    )


class TrainingSessionTests(unittest.TestCase):
    def setUp(self):
        FakeOrchestrator.calls = []

    def test_session_constructs_once_and_runs_phase_through_orchestrator(self):
        model = FakeModel()
        logger = FakeLogger()
        config = make_config("unused/models")
        session = TrainingSession.from_config(
            config,
            factories=make_factories(model, logger),
            execution_builders={
                "fake_execution": lambda session, context, params: object()
            },
        )

        result = session.run()

        self.assertIs(result, session)
        self.assertEqual(session.global_step, 3)
        self.assertEqual(len(FakeOrchestrator.calls), 1)
        self.assertEqual(model.saved_paths, [Path("unused/models/00_first.pt")])
        self.assertFalse(model.parameter.requires_grad)
        self.assertTrue(logger.closed)

    def test_run_training_consumes_complete_yaml(self):
        model = FakeModel()
        logger = FakeLogger()
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "training.yaml"
            config_path.write_text(
                """
system: {device: cpu}
models:
  components: [{name: audio_provider, type: fake, config: {}}]
  language_core: {model_name: fake, device_map: null}
  save_dir: unused/models
datasets:
  train: {requires: [audio_encoding, txt], batch_size: 1, datasets: []}
  validation: {requires: [audio_encoding, txt], batch_size: 1, datasets: []}
training:
  optimizer: {type: adam, lr: 0.001}
  runtime: {schedule: {}, amp: {enabled: false}}
  phases:
    - name: first
      enabled: true
      run: {steps: 2, trainable: [audio_provider.head], execution: fake_execution}
      targets: {semantic_weight: 1.0}
      schedules: {}
""",
                encoding="utf-8",
            )
            session = run_training(
                config_path,
                factories=make_factories(model, logger),
                execution_builders={
                    "fake_execution": lambda session, context, params: object()
                },
            )
        self.assertEqual(session.global_step, 2)

    def test_missing_required_top_level_section_fails_early(self):
        with self.assertRaisesRegex(KeyError, "datasets"):
            TrainingSession.from_config({
                "system": {"device": "cpu"},
                "models": {},
                "training": {},
            })


if __name__ == "__main__":
    unittest.main()
