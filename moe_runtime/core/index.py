from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from safetensors import safe_open


@dataclass(frozen=True, slots=True)
class TensorRecord:
    name: str
    filename: str
    shape: tuple[int, ...] | None = None
    dtype: str | None = None


class TensorIndex:
    def __init__(self, model_path: str | Path, records: dict[str, TensorRecord]) -> None:
        self.model_path = Path(model_path)
        self.records = records

    @classmethod
    def from_safetensors_index(cls, model_path: str | Path) -> "TensorIndex":
        root = Path(model_path)
        index_path = root / "model.safetensors.index.json"
        with index_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        records = {
            name: TensorRecord(name=name, filename=filename)
            for name, filename in data["weight_map"].items()
        }
        return cls(root, records)

    def names(self) -> list[str]:
        return list(self.records)

    def filenames(self) -> list[str]:
        return sorted({record.filename for record in self.records.values()})

    def require(self, name: str) -> TensorRecord:
        try:
            return self.records[name]
        except KeyError as exc:
            raise KeyError(f"Tensor not found in checkpoint index: {name}") from exc

    def materialize_metadata(self, names: Iterable[str] | None = None) -> None:
        needed = set(names) if names is not None else set(self.records)
        by_file: dict[str, list[str]] = {}
        for name in needed:
            by_file.setdefault(self.require(name).filename, []).append(name)

        updated = dict(self.records)
        for filename, tensor_names in by_file.items():
            with safe_open(self.model_path / filename, framework="pt", device="cpu") as shard:
                for name in tensor_names:
                    tensor_slice = shard.get_slice(name)
                    old = updated[name]
                    updated[name] = TensorRecord(
                        name=name,
                        filename=old.filename,
                        shape=tuple(tensor_slice.get_shape()),
                        dtype=str(tensor_slice.get_dtype()),
                    )
        self.records = updated

    def load_tensor(self, name: str, *, device: str = "cpu"):
        record = self.require(name)
        with safe_open(self.model_path / record.filename, framework="pt", device=device) as shard:
            return shard.get_tensor(name)
