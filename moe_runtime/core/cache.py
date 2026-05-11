from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Generic, TypeVar

from .expert import ExpertId

T = TypeVar("T")


@dataclass(slots=True)
class ExpertCacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    puts: int = 0

    @property
    def requests(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        if self.requests == 0:
            return 0.0
        return self.hits / self.requests


class ExpertCache(Generic[T]):
    def __init__(self, capacity_bytes: int | None, *, name: str) -> None:
        self.capacity_bytes = capacity_bytes
        self.name = name
        self.current_bytes = 0
        self.stats = ExpertCacheStats()
        self._items: OrderedDict[ExpertId, tuple[T, int]] = OrderedDict()

    def __contains__(self, key: ExpertId) -> bool:
        return key in self._items

    def __len__(self) -> int:
        return len(self._items)

    def get(self, key: ExpertId) -> T | None:
        item = self._items.get(key)
        if item is None:
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        value, size = item
        self._items.move_to_end(key)
        return value

    def peek(self, key: ExpertId) -> T | None:
        item = self._items.get(key)
        return None if item is None else item[0]

    def put(self, key: ExpertId, value: T, size_bytes: int) -> None:
        old = self._items.pop(key, None)
        if old is not None:
            self.current_bytes -= old[1]

        self._items[key] = (value, size_bytes)
        self.current_bytes += size_bytes
        self.stats.puts += 1
        self._evict_until_fit(protected=key)

    def clear(self) -> None:
        self._items.clear()
        self.current_bytes = 0

    def keys(self) -> list[ExpertId]:
        return list(self._items.keys())

    def _evict_until_fit(self, *, protected: ExpertId) -> None:
        if self.capacity_bytes is None:
            return
        while self.current_bytes > self.capacity_bytes and len(self._items) > 1:
            key, (_, size) = self._items.popitem(last=False)
            if key == protected:
                self._items[key] = self._items.pop(key)
                break
            self.current_bytes -= size
            self.stats.evictions += 1
