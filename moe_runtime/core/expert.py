from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, order=True, slots=True)
class ExpertId:
    layer: int
    expert: int


@dataclass(slots=True)
class ExpertWeights:
    gate_proj: torch.Tensor
    up_proj: torch.Tensor
    down_proj: torch.Tensor

    def to(
        self,
        device: torch.device | str,
        *,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> "ExpertWeights":
        return ExpertWeights(
            gate_proj=self.gate_proj.to(device=device, dtype=dtype, non_blocking=non_blocking),
            up_proj=self.up_proj.to(device=device, dtype=dtype, non_blocking=non_blocking),
            down_proj=self.down_proj.to(device=device, dtype=dtype, non_blocking=non_blocking),
        )

    def pin_memory(self) -> "ExpertWeights":
        if not torch.cuda.is_available():
            return self
        return ExpertWeights(
            gate_proj=self.gate_proj.pin_memory(),
            up_proj=self.up_proj.pin_memory(),
            down_proj=self.down_proj.pin_memory(),
        )

    @property
    def nbytes(self) -> int:
        return self.gate_proj.nbytes + self.up_proj.nbytes + self.down_proj.nbytes
