from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch


def parse_bytes(value: int | float | str | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)

    text = value.strip().lower()
    units = {
        "b": 1,
        "kb": 1024,
        "mb": 1024**2,
        "gb": 1024**3,
        "kib": 1024,
        "mib": 1024**2,
        "gib": 1024**3,
    }
    for suffix, scale in sorted(units.items(), key=lambda item: len(item[0]), reverse=True):
        if text.endswith(suffix):
            return int(float(text[: -len(suffix)].strip()) * scale)
    return int(float(text))


def parse_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    normalized = dtype.lower().replace("torch.", "")
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return aliases[normalized]


@dataclass(slots=True)
class OffloadConfig:
    model_path: str | Path
    device: str | torch.device = "cuda"
    dtype: str | torch.dtype = torch.bfloat16
    cpu_cache_bytes: int | float | str | None = "8GiB"
    gpu_cache_bytes: int | float | str | None = "2GiB"
    pin_memory: bool = True
    prefetch: bool = True
    offload_experts: bool = True
    low_cpu_mem_usage: bool = True

    def __post_init__(self) -> None:
        self.model_path = Path(self.model_path)
        self.device = torch.device(self.device)
        self.dtype = parse_dtype(self.dtype)
        self.cpu_cache_bytes = parse_bytes(self.cpu_cache_bytes)
        self.gpu_cache_bytes = parse_bytes(self.gpu_cache_bytes)
