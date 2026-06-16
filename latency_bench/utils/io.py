from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from enum import Enum
from json import JSONDecodeError
from pathlib import Path
from typing import Any

import numpy as np


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {
            field.name: to_jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(obj), f, sort_keys=True, indent=2)
        f.write("\n")


def write_jsonl(path: str | Path, records: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.writelines(
            json.dumps(to_jsonable(record), sort_keys=True, ensure_ascii=False) + "\n"
            for record in records
        )


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file does not exist: {jsonl_path}")

    records: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {jsonl_path}:{line_number}: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL record must be a JSON object at {jsonl_path}:{line_number}")
            records.append(record)
    return records
