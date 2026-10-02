# LI Concept Model v2

Clean successor workspace for the Language Independent Concept Model research
project.

This repository is being assembled incrementally from reviewed, working
components. The retired `LI_concept_model_v1` repository is a read-only
reference; this project must never import code from it at runtime.

## Initial scope

The initial codebase retains three responsibility areas:

- `src.modules` — approved model components, the model wrapper, and model
  persistence;
- `src.training` — research-specific training assembly; and
- `src.utils` — small shared utilities, beginning with YAML configuration
  loading.

Dataset construction and streaming are provided by the independently maintained
[RunWeaver ML](https://github.com/Gogie607/RunWeaver-ML) package. RunWeaver is
an external dependency and must not be copied into this repository.

## Development environment

Development initially reuses the existing Conda environment:

```bash
/home/ronald/anaconda3/bin/conda run -n P12Cu13_env python
```

RunWeaver is currently installed editable from:

```text
/home/ronald/PycharmProjects/runweaver-ml
```

No new environment or package installation is required by the bootstrap.

## Repository layout

```text
configs/       new, verified configurations only
src/modules/   model components and persistence
src/training/  training assembly
src/utils/     configuration and shared utilities
tests/         focused tests for migrated components
```

Directories begin intentionally small. Components are added only after their
implementation and dependencies have been reviewed in the reference project.

## Model foundation

`src.modules` now provides the shared boundary required before the first model:

- `ModelComponent`, combining PyTorch module behavior, boolean leaf control,
  stable type registration, and configuration-plus-weights serialization;
- `TrainingModelWrapper`, which manages named components and satisfies
  RunWeaver's model-control protocol;
- `ModelPersistenceManager`, which saves and loads model bundles without
  optimizer, scheduler, or training-session state; and
- registry helpers for configuration-driven component construction.

The YAML loader in `src.utils` supports recursive relative includes, local
overrides, and include-cycle detection.

## Training sessions

`run_training` consumes one complete YAML file and creates a `TrainingSession`.
The session owns model/runtime construction, DSM datasets, phase parameters,
optimizers, logging, collectors, and phase-level persistence. RunWeaver remains
responsible for executing each prepared phase loop.

The first complete recipe is:

```text
configs/training/audio_prefix_libritts.yaml
```

Run it from the repository root with the project Conda environment:

```bash
/home/ronald/anaconda3/bin/conda run -n P12Cu13_env \
  python -m scripts.train configs/training/audio_prefix_libritts.yaml
```

The recipe uses cached Whisper encodings. It does not load or execute a Whisper
backbone during training. Qwen is reconstructed as a frozen external runtime
and is not included in provider model bundles.

The trained provider can be compared with Qwen's text prior using the paired
audio-conditioning evaluation:

```bash
/home/ronald/anaconda3/bin/conda run -n P12Cu13_env \
  python -m scripts.evaluate_audio_conditioning \
  configs/evaluation/audio_conditioning_libritts.yaml
```

This read-only evaluation compares correct, batch-shuffled, zero, BOS-only,
and frame-shuffled prefixes and writes both aggregate and per-sample results.

## Migration rules

1. Never import from `LI_concept_model_v1`.
2. Do not copy old experiment configurations into `configs/`.
3. Migrate one approved component and only its necessary dependencies at a
   time.
4. Preserve configuration-driven type registration until a specific change has
   been reviewed and approved.
5. Keep loading trained model weights separate from recovering an interrupted
   training session.
6. Do not restore optimizer, scheduler, or global-step state unless explicitly
   requested later.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the model, training-session, and
RunWeaver boundaries. [The experiment design note](docs/design.md) describes
how the current audio-prefix recipe composes those software components.
