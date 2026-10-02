"""Bounded, opt-in snapshots for later qualitative inspection."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from tempfile import NamedTemporaryFile

from runweaver_ml.phase_control import CollectorBase


class AudioPrefixSnapshotCollector(CollectorBase):
    """Collect detached examples without making artifact policy a trainer concern."""

    def __init__(self, *, max_samples: int = 16, writer: Callable | None = None):
        if max_samples < 0:
            raise ValueError("max_samples must be non-negative")
        self.max_samples = max_samples
        self.writer = writer
        self.snapshots: list[dict] = []
        self.step = None

    def begin(self, params=None, step=None):
        del params
        self.snapshots = []
        self.step = step

    def update(self, payload):
        remaining = self.max_samples - len(self.snapshots)
        if remaining <= 0:
            return
        for index in range(min(payload.batch_size, remaining)):
            record = {
                "sample_id": self._at(payload.sample_ids, index),
                "domain": self._at(payload.domains, index),
                "target_text": self._at(payload.target_text, index),
                "objective_losses": dict(payload.objective_losses),
            }
            if payload.predicted_token_ids is not None:
                record["predicted_token_ids"] = payload.predicted_token_ids[index].tolist()
            if payload.target_token_ids is not None:
                record["target_token_ids"] = payload.target_token_ids[index].tolist()
            if payload.target_token_mask is not None:
                record["target_token_mask"] = payload.target_token_mask[index].tolist()
            record.update(payload.snapshot_values)
            self.snapshots.append(record)

    def finalize(self):
        if self.writer is not None:
            self.writer(tuple(self.snapshots), step=self.step)
        return tuple(self.snapshots)

    def save(self):
        return tuple(self.snapshots)

    @staticmethod
    def _at(values, index):
        return values[index] if index < len(values) else None


class JsonSnapshotWriter:
    """Persist one immutable JSON snapshot artifact per validation step."""

    def __init__(self, output_dir: str | Path, *, prefix: str = "validation"):
        self.output_dir = Path(output_dir)
        self.prefix = prefix

    def __call__(self, snapshots, *, step=None):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        step_name = "unknown" if step is None else f"{int(step):08d}"
        destination = self.output_dir / f"{self.prefix}_{step_name}.json"
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self.output_dir,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(list(snapshots), handle, indent=2)
        temporary.replace(destination)
        return destination
