"""Tests for the bounded profile response cache."""

from linkedin_mcp_server.response_cache import TTLResultCache


def test_cache_returns_defensive_copies():
    cache = TTLResultCache(ttl_seconds=60, max_entries=2)
    original = {"sections": {"main_profile": "Phoenix"}}

    cache.set("alice", original)
    first = cache.get("alice")
    assert first is not None
    first["sections"]["main_profile"] = "Changed"

    assert cache.get("alice") == original


def test_cache_evicts_least_recently_used_entry():
    cache = TTLResultCache(ttl_seconds=60, max_entries=2)
    cache.set("alice", {"value": 1})
    cache.set("bob", {"value": 2})
    assert cache.get("alice") == {"value": 1}

    cache.set("carol", {"value": 3})

    assert cache.get("bob") is None
    assert cache.get("alice") == {"value": 1}
    assert cache.get("carol") == {"value": 3}
