# Changelog

## 2026-09-30

- Created the clean `LI_concept_model_v2` research repository.
- Added the initial `src.modules`, `src.training`, and `src.utils` package layout.
- Added project documentation, architecture notes, packaging metadata, and Git ignore rules.
- Added a YAML configuration loader with recursive relative includes, local overrides, and cycle detection.
- Added the `Controllable` and `ControllableNode` model-state interfaces.
- Added the `Serializable` configuration-and-weights persistence interface.
- Added `ModelComponent` as the common base for registered, controllable, serializable PyTorch components.
- Added stable component-type registration and configuration-driven component construction.
- Added `TrainingModelWrapper` for named components, mode control, trainability control, and RunWeaver integration.
- Added versioned model-bundle persistence without optimizer, scheduler, trainer, or global-step state.
- Added focused tests for YAML loading, component registration, model persistence, state restoration, device handling, and the RunWeaver model protocol.
- Confirmed that RunWeaver remains an external editable dependency; no RunWeaver source was copied into this repository.

