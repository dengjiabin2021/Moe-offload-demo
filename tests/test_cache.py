from moe_runtime.core.cache import ExpertCache
from moe_runtime.core.expert import ExpertId


def test_lru_cache_hits_and_evicts_oldest_item():
    cache = ExpertCache[int](capacity_bytes=10, name="test")
    first = ExpertId(0, 0)
    second = ExpertId(0, 1)
    third = ExpertId(0, 2)

    cache.put(first, 1, 5)
    cache.put(second, 2, 5)
    assert cache.get(first) == 1
    cache.put(third, 3, 5)

    assert cache.peek(first) == 1
    assert cache.peek(second) is None
    assert cache.peek(third) == 3
    assert cache.stats.evictions == 1
    assert cache.stats.hits == 1
