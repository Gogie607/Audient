"""Construction and lifecycle ownership for one reproducible training run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from runweaver_ml.dataset_management import build_datasets, create_loader
from runweaver_ml.dataset_management.artifacts import ARTIFACT_REGISTRY
from runweaver_ml.dataset_management.runtime_artifact_repository import (
    RuntimeArtifactRepository,
)
from runweaver_ml.phase_control import (
    ParamProvider,
    ParameterWrapper,
    PhaseManager,
    PhaseOrchestrator,
    ScheduleEngine,
    TargetRegistry,
    TrainLoopConfig,
    build_schedule_block,
)
from runweaver_ml.phase_control.execution import (
    apply_trainable,
    clear_trainable,
    system_trainable,
)

from ...modules import QwenLanguageCore, TrainingModelWrapper
from ..audio_prefix import (
    AudioPrefixSnapshotCollector,
    JsonSnapshotWriter,
    create_audio_prefix_training_module,
)
from .logger import RunLogger


ExecutionBuilder = Callable[..., Any]


@dataclass(frozen=True)
class SessionFactories:
    """Replaceable construction seams used by tests and future recipes."""

    dataset_builder: Callable = build_datasets
    loader_builder: Callable = create_loader
    model_builder: Callable = TrainingModelWrapper.from_config
    model_loader: Callable = TrainingModelWrapper.load
    language_core_builder: Callable = QwenLanguageCore.from_config
    logger_builder: Callable = RunLogger.from_config
    orchestrator: Any = PhaseOrchestrator


@dataclass
class PreparedPhase:
    context: Any
    params: ParameterWrapper
    execution: Any
    optimizer: torch.optim.Optimizer
    scheduler: Any
    train_loader: Any
    validation_loader: Any
    loop_config: TrainLoopConfig


class TrainingSession:
    """Own all long-lived objects and prepare RunWeaver phases."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        device: torch.device,
        model: TrainingModelWrapper,
        language_core: QwenLanguageCore,
        train_dataset,
        validation_dataset,
        artifact_repository,
        logger,
        phase_manager: PhaseManager,
        factories: SessionFactories,
        execution_builders: Mapping[str, ExecutionBuilder],
    ) -> None:
        self.config = dict(config)
        self.device = device
        self.model = model
        self.language_core = language_core
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.artifact_repository = artifact_repository
        self.logger = logger
        self.phase_manager = phase_manager
        self.factories = factories
        self.execution_builders = dict(execution_builders)
        self.global_step = 0

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        factories: SessionFactories | None = None,
        execution_builders: Mapping[str, ExecutionBuilder] | None = None,
    ) -> TrainingSession:
        cls._validate_config(config)
        resolved_factories = factories or SessionFactories()
        device = torch.device(config["system"]["device"])
        model_cfg = config["models"]

        load_path = model_cfg.get("load_path")
        if load_path:
            model = resolved_factories.model_loader(
                load_path,
                map_location="cpu",
                device=device,
            )
        else:
            model = resolved_factories.model_builder(
                model_cfg["components"],
                device=device,
            )

        qwen_cfg = model_cfg["language_core"]
        qwen_device = None if qwen_cfg.get("device_map") is not None else device
        language_core = resolved_factories.language_core_builder(
            qwen_cfg,
            device=qwen_device,
        )

        repository = cls._build_artifact_repository(
            config["datasets"].get("shared_resources", [])
        )
        train_dataset = resolved_factories.dataset_builder(
            config["datasets"]["train"],
            runtime_artifact_repository=repository,
        )
        validation_dataset = resolved_factories.dataset_builder(
            config["datasets"]["validation"],
            runtime_artifact_repository=repository,
        )
        logger = resolved_factories.logger_builder(
            config["training"]["runtime"].get("logger")
        )
        phase_manager = PhaseManager(
            config["training"]["phases"],
            config.get("global", {}),
            config["training"]["runtime"],
            logger=logger,
        )
        builders = {"audio_prefix": cls._build_audio_prefix_execution}
        builders.update(execution_builders or {})
        return cls(
            config=config,
            device=device,
            model=model,
            language_core=language_core,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            artifact_repository=repository,
            logger=logger,
            phase_manager=phase_manager,
            factories=resolved_factories,
            execution_builders=builders,
        )

    def run(self) -> TrainingSession:
        try:
            for context in self.phase_manager:
                if not context.enabled:
                    self.logger.log(
                        "phase_skipped",
                        self.global_step,
                        {"phase": context.name, "index": context.idx},
                    )
                    continue
                with system_trainable(self.model, context.trainable):
                    prepared = self.prepare_phase(context)
                    self.execute_phase(prepared)
        finally:
            self.logger.close()
        return self

    def prepare_phase(self, context) -> PreparedPhase:
        self._set_active_datasets(context.run.get("active_datasets"))
        dataset_cfg = self.config["datasets"]
        train_loader = self.factories.loader_builder(
            self.train_dataset,
            required_modalities=dataset_cfg["train"]["requires"],
            batch_size=context.run.get(
                "batch_size", dataset_cfg["train"]["batch_size"]
            ),
        )
        validation_loader = self.factories.loader_builder(
            self.validation_dataset,
            required_modalities=dataset_cfg["validation"]["requires"],
            batch_size=context.run.get(
                "batch_size", dataset_cfg["validation"]["batch_size"]
            ),
        )

        registry = TargetRegistry(context.targets)
        engine = ScheduleEngine(registry=registry, last_step=0)
        for name, schedule_config in context.schedules.items():
            engine.register(name, build_schedule_block(schedule_config))
        params = ParameterWrapper(
            engine,
            ParamProvider(
                run={
                    **context.run,
                    "phase_name": context.name,
                    "phase_idx": context.idx,
                },
                global_cfg=context.global_cfg,
                runtime=context.runtime,
                registry=registry,
            ),
        )

        recipe = context.run.get("execution", "audio_prefix")
        try:
            builder = self.execution_builders[recipe]
        except KeyError as error:
            raise ValueError(f"unknown training execution recipe {recipe!r}") from error
        execution = builder(self, context, params)
        optimizer = self._build_optimizer(context)
        runtime_cfg = self._phase_runtime_config(context.run)
        loop_config = TrainLoopConfig.from_config(
            runtime_cfg,
            max_steps=context.steps,
        )
        return PreparedPhase(
            context=context,
            params=params,
            execution=execution,
            optimizer=optimizer,
            scheduler=None,
            train_loader=train_loader,
            validation_loader=validation_loader,
            loop_config=loop_config,
        )

    def execute_phase(self, prepared: PreparedPhase) -> None:
        context = prepared.context
        self.logger.log(
            "phase_begin",
            self.global_step,
            {
                "phase": context.name,
                "index": context.idx,
                "run": context.run,
                "targets": context.targets,
            },
        )
        completed_steps = self.factories.orchestrator.train_epoch(
            trainer=prepared.execution,
            loop_config=prepared.loop_config,
            optimizer=prepared.optimizer,
            train_loader=prepared.train_loader,
            val_loader=prepared.validation_loader,
            logger=self.logger,
            scheduler=prepared.scheduler,
        )
        self.global_step += completed_steps
        destination = self._phase_model_path(context)
        self.model.save(destination)
        self.logger.log(
            "phase_end",
            self.global_step,
            {
                "phase": context.name,
                "index": context.idx,
                "completed_steps": completed_steps,
                "model_path": str(destination),
            },
        )
        self.logger.flush()

    def _build_optimizer(self, context) -> torch.optim.Optimizer:
        config = {
            **self.config["training"].get("optimizer", {}),
            **context.run.get("optimizer", {}),
        }
        optimizer_name = str(config.pop("type", "adamw")).lower()
        if "lr" not in config:
            raise KeyError(f"[{context.name}] optimizer requires lr")
        optimizers = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
        }
        try:
            optimizer_class = optimizers[optimizer_name]
        except KeyError as error:
            raise ValueError(f"unsupported optimizer {optimizer_name!r}") from error
        parameters = list(self.model.trainable_parameters())
        if not parameters:
            raise RuntimeError(f"[{context.name}] phase has no trainable parameters")
        return optimizer_class(parameters, **config)

    def _phase_runtime_config(self, run_config: Mapping[str, Any]) -> dict:
        runtime = dict(self.config["training"]["runtime"])
        if "amp" in run_config:
            runtime["amp"] = {
                **runtime.get("amp", {}),
                **run_config["amp"],
            }
        if "schedule" in run_config:
            runtime["schedule"] = {
                **runtime.get("schedule", {}),
                **run_config["schedule"],
            }
        return runtime

    def _set_active_datasets(self, active) -> None:
        if active is None:
            return
        for dataset in (self.train_dataset, self.validation_dataset):
            setter = getattr(dataset, "set_active_datasets", None)
            if setter is None:
                raise TypeError("configured dataset does not support active_datasets")
            setter(active)

    def _phase_model_path(self, context) -> Path:
        output_dir = Path(self.config["models"]["save_dir"])
        return output_dir / f"{context.idx:02d}_{context.name}.pt"

    def _checkpoint_model_path(self, context, step: int) -> Path:
        output_dir = Path(self.config["models"]["save_dir"]) / "checkpoints"
        return output_dir / f"{context.idx:02d}_{context.name}_{step:08d}.pt"

    @staticmethod
    def _build_audio_prefix_execution(session, context, params):
        trainer = context.run.get("trainer", {})
        train_mode = apply_trainable(
            clear_trainable(session.model.get_mode()),
            context.trainable,
        )
        eval_mode = clear_trainable(session.model.get_mode())
        collector = AudioPrefixSnapshotCollector(
            max_samples=int(trainer.get("snapshot_samples", 0)),
            writer=(
                JsonSnapshotWriter(trainer["snapshot_dir"])
                if trainer.get("snapshot_dir")
                else None
            ),
        )

        def checkpoint_handler(module, optimizer, scheduler):
            del optimizer, scheduler
            session.model.save(
                session._checkpoint_model_path(context, module.current_step)
            )

        return create_audio_prefix_training_module(
            model=session.model,
            language_core=session.language_core,
            params=params,
            train_mode=train_mode,
            eval_mode=eval_mode,
            semantic_weight_parameter=trainer.get(
                "semantic_weight_parameter", "semantic_weight"
            ),
            semantic_objective=trainer.get("semantic_objective", "standard"),
            position_weight_parameters=trainer.get("position_weight_parameters"),
            enable_audio_contrast=bool(
                trainer.get("enable_audio_contrast", False)
            ),
            contrast_weight_parameter=trainer.get(
                "contrast_weight_parameter", "semantic_contrast_weight"
            ),
            contrast_margin_parameter=trainer.get(
                "contrast_margin_parameter", "semantic_contrast_margin"
            ),
            contrast_position_weight_parameters=trainer.get(
                "contrast_position_weight_parameters"
            ),
            trait_weight_parameters=trainer.get("trait_weight_parameters"),
            trait_beta_parameter=trainer.get(
                "trait_beta_parameter", "trait_smooth_l1_beta"
            ),
            include_traits=bool(trainer.get("include_traits", True)),
            prompt=str(trainer.get("prompt", "")),
            temporal_trait_groups=tuple(
                trainer.get(
                    "temporal_trait_groups", ["timing", "energy", "pitch"]
                )
            ),
            collector=collector,
            checkpoint_handler=checkpoint_handler,
        )

    @staticmethod
    def _build_artifact_repository(resources) -> RuntimeArtifactRepository:
        repository = RuntimeArtifactRepository(ARTIFACT_REGISTRY)
        for resource in resources:
            repository.register(
                name=resource["name"],
                handler=resource.get("handler"),
                config=resource.get("config"),
                instance=resource.get(
                    "instance", RuntimeArtifactRepository.MISSING
                ),
            )
        return repository

    @staticmethod
    def _validate_config(config: Mapping[str, Any]) -> None:
        for section in ("system", "models", "datasets", "training"):
            if section not in config:
                raise KeyError(f"training configuration missing {section!r}")
        if "device" not in config["system"]:
            raise KeyError("system.device is required")
        for name in ("components", "language_core", "save_dir"):
            if name not in config["models"]:
                raise KeyError(f"models.{name} is required")
        for split in ("train", "validation"):
            if split not in config["datasets"]:
                raise KeyError(f"datasets.{split} is required")
            for name in ("requires", "batch_size"):
                if name not in config["datasets"][split]:
                    raise KeyError(f"datasets.{split}.{name} is required")
        if "runtime" not in config["training"]:
            raise KeyError("training.runtime is required")
        if not config["training"].get("phases"):
            raise ValueError("training.phases must contain at least one phase")
