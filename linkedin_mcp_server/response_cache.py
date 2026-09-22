"""Small process-local cache for repeated read-only profile scrapes."""

from collections import OrderedDict
from copy import deepcopy
from time import monotonic
from typing import Any, Hashable


class TTLResultCache:
    """Bounded TTL cache that returns defensive copies of tool payloads."""

    def __init__(self, *, ttl_seconds: float = 1800, max_entries: int = 512) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._entries: OrderedDict[Hashable, tuple[float, dict[str, Any]]] = (
            OrderedDict()
        )

    def get(self, key: Hashable) -> dict[str, Any] | None:
        entry = self._entries.pop(key, None)
        if entry is None:
            return None
        expires_at, value = entry
        if monotonic() >= expires_at:
            return None
        self._entries[key] = entry
        return deepcopy(value)

    def set(self, key: Hashable, value: dict[str, Any]) -> None:
        self._entries.pop(key, None)
        self._entries[key] = (monotonic() + self.ttl_seconds, deepcopy(value))
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()


profile_result_cache = TTLResultCache()
