from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

from .expert import ExpertId, ExpertWeights
from .index import TensorIndex


class MoEAdapter(ABC):
    @abstractmethod
    def is_expert_tensor(self, tensor_name: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def expert_id_for_tensor(self, tensor_name: str) -> ExpertId:
        raise NotImplementedError

    @abstractmethod
    def expert_tensor_names(self, expert_id: ExpertId) -> dict[str, str]:
        raise NotImplementedError

    @abstractmethod
    def discover_experts(self, index: TensorIndex) -> set[ExpertId]:
        raise NotImplementedError

    @abstractmethod
    def load_expert_weights(self, index: TensorIndex, expert_id: ExpertId) -> ExpertWeights:
        raise NotImplementedError

    @abstractmethod
    def replace_experts(self, model, manager) -> None:
        raise NotImplementedError

    @abstractmethod
    def non_expert_tensor_names(self, index: TensorIndex) -> Iterable[str]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def supports(cls, model_path: str | Path) -> bool:
        raise NotImplementedError
