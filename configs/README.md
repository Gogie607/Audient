# Configurations

This directory contains only configurations authored and verified for the new
codebase. Do not copy historical training configurations here wholesale.

`models/whisper_provider.yaml` defines the first V2 composite consuming
Whisper-medium encoder states. It contains a duration-preserving semantic head
and newly initialized speech-trait heads; it does not restore V1 checkpoints.

`models/qwen_language_core.yaml` identifies the frozen external Qwen runtime
used to consume semantic prefixes. Its foundation weights are loaded at runtime
and are not part of the model bundle.

`training/audio_prefix_libritts.yaml` is the first complete V2 training recipe.
It uses the existing LibriTTS tar modalities (`audio_encoding`, `txt`, and
`speech_traits`), constructs one training session, and delegates its phase loop
to RunWeaver. Objective weights live in the phase target registry so they can
remain constant or acquire schedules without changing trainer construction.
