"""YAML loading with relative recursive mapping includes."""

from pathlib import Path

import yaml


def load_config(path) -> dict:
    source = Path(path).resolve()
    return _load_mapping(source, stack=())


def _load_mapping(path: Path, *, stack: tuple[Path, ...]) -> dict:
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"Cyclic configuration include: {chain}")

    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)

    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise TypeError(f"Configuration root must be a mapping: {path}")

    return _resolve_includes(
        value,
        base_dir=path.parent,
        stack=(*stack, path),
    )


def _resolve_includes(value, *, base_dir: Path, stack: tuple[Path, ...]):
    if isinstance(value, dict):
        if "include" in value:
            include_value = value["include"]
            if not isinstance(include_value, str):
                raise TypeError("include must be a relative or absolute path string")

            include_path = Path(include_value)
            if not include_path.is_absolute():
                include_path = base_dir / include_path

            included = _load_mapping(include_path.resolve(), stack=stack)
            local = {
                key: item
                for key, item in value.items()
                if key != "include"
            }
            included.update(
                _resolve_includes(local, base_dir=base_dir, stack=stack)
            )
            return included

        return {
            key: _resolve_includes(item, base_dir=base_dir, stack=stack)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            _resolve_includes(item, base_dir=base_dir, stack=stack)
            for item in value
        ]

    return value

