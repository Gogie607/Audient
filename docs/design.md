# Current Experiment Composition

This note records how the current audio-prefix experiment uses the reusable
software architecture. It documents the configured components and assembly
points, not an assessment of training results. The stable training architecture
is described in [ARCHITECTURE.md](../ARCHITECTURE.md).

## Model components

The model bundle contains the configured `audio_provider`, a registered
`WhisperProvider` component. Its shared input stem feeds a semantic prefix head
and speech-trait heads. `AudioRepresentation` carries their outputs and timing
metadata. Whisper itself is outside the model bundle: this recipe consumes
cached Whisper-medium `audio_encoding` tensors described by an
`AudioBackboneSpec`.

The language model is a separately constructed `QwenLanguageCore`, configured
as Qwen3-1.7B. It is frozen and remains external to the model bundle. The
audio-prefix forward strategy combines its token embeddings with the provider's
semantic stream and computes the recipe's causal target-token objective.
Speech-trait predictions are consumed by the recipe's auxiliary objectives.

## Configuration and assembly

The active recipe is
[`configs/training/audio_prefix_libritts.yaml`](../configs/training/audio_prefix_libritts.yaml).
The configuration selects model components and language-core settings, train
and validation dataset inputs, base optimizer/runtime settings, and a
`semantic_and_traits` phase. That phase selects trainable provider paths,
objective target values, and trainer-specific arguments.

The `audio_prefix` execution builder in `TrainingSession` assembles
`AudioPrefixForward`, `SemanticTokenObjective`, optional
`SpeechTraitObjective` instances, `AudioObjectiveComposer`, metrics, snapshot
collector, and a model-only checkpoint callback into RunWeaver's
`TrainingModule`. The composer reuses RunWeaver's objective-composition pattern
while returning the forward payload used by V2 handlers.

The built-in builder is recipe-specific. Other model/objective combinations
can reuse the session lifecycle and RunWeaver loop by supplying a compatible
execution builder; a different topology must provide the matching forward and
objective assembly rather than relying on YAML alone.
