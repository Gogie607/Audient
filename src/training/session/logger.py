"""Small structured logger used by training sessions."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class RunLogger:
    """Buffer records by step and optionally append them as JSON lines."""

    def __init__(
        self,
        *,
        output_dir: str | Path | None = None,
        prefix: str = "run",
        print_enabled: bool = True,
    ) -> None:
        self.print_enabled = print_enabled
        self.current_step: int | None = None
        self.current_record: dict[str, Any] = {}
        self.path: Path | None = None
        if output_dir is not None:
            directory = Path(output_dir)
            directory.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.path = directory / f"{prefix}_{timestamp}.jsonl"

    @classmethod
    def from_config(cls, config: dict | None) -> RunLogger:
        values = config or {}
        return cls(
            output_dir=values.get("output_dir"),
            prefix=values.get("prefix", "run"),
            print_enabled=bool(values.get("print", True)),
        )

    def log(self, tag: str, step: int, data) -> None:
        if self.current_step is not None and step != self.current_step:
            self.flush()
        if self.current_step is None:
            self.current_step = step
            self.current_record = {"step": step}
        self.current_record[tag] = self._json_value(data)
        if self.print_enabled:
            print(f"[{step}] {tag}: {self.current_record[tag]}")

    def flush(self) -> None:
        if not self.current_record:
            return
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(self.current_record) + "\n")
        self.current_step = None
        self.current_record = {}

    def close(self) -> None:
        self.flush()

    @staticmethod
    def _json_value(value):
        if hasattr(value, "summarize"):
            value = value.summarize()
        if isinstance(value, dict):
            return {
                str(key): RunLogger._json_value(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [RunLogger._json_value(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

