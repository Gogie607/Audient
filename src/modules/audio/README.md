# Audio provider contract

An audio provider is a composite model component that owns the trainable modules
adapting backbone features for the language model. It does not own the frozen
audio backbone, training objectives, dataset routing, or experiment state.

The provider returns an `AudioRepresentation` instead of an anonymous tensor.
The representation always has a semantic stream and may carry a separate
acoustic stream and named speech-trait streams. Timing, frame rate, masks, and
representation kinds are explicit so continuous, discrete, RVQ, offline, and
streamed providers can be compared without changing downstream interfaces.

Waveform reconstruction is an optional, separate `AudioDecoder` capability.
Concrete Whisper, RVQ, external-tokenizer, and MOSS-oriented implementations
will be added only after their representation and rate decisions are made.

The backbone is a runtime-only `AudioBackbone` preprocessor reconstructed from
configuration. Its weights are not part of provider checkpoints. The provider
persists an `AudioBackboneSpec` identifying the exact feature boundary it
expects.

As a normal PyTorch module, `provider(backbone_features)` runs the complete
prefix-producing composite. Training modules decide whether those features came
from pregenerated dataset fields or a live backbone call. The provider only
validates compatibility and transforms them.
