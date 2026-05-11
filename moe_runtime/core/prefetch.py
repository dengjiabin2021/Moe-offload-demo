from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from collections.abc import Iterable

from .expert import ExpertId


class PrefetchManager:
    def __init__(self, expert_manager, *, max_workers: int = 1) -> None:
        self.expert_manager = expert_manager
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="moe-prefetch")

    def submit(self, expert_ids: Iterable[ExpertId]) -> Future:
        ids = tuple(expert_ids)
        return self.executor.submit(self.expert_manager.prefetch, ids)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=True)
