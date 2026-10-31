import asyncio
import ast
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from pydantic import TypeAdapter
from typing_extensions import ReadOnly, TypedDict

import pytest

from litellm._internal_context import current_service_target
from litellm.caching.dual_cache import DualCache
from litellm.caching.redis_cache import RedisCache
from litellm.redis.stores._connection import redis_kwargs
from litellm.redis.stores._routing_prefetch import selector_read
from litellm.redis.stores.router_cooldowns import CooldownCacheValue, RouterCooldownStore

_WIRE: Final = Path(__file__).parents[4] / "litellm-rust/crates/redis-store/tests/fixtures/router_wire.json"


class _WireCooldown(TypedDict):
    cooldown: ReadOnly[dict[str, str | int | float]]
    str_dict: ReadOnly[str]
    json: ReadOnly[str]
    ttl_seconds: ReadOnly[float]
    redis_ttl_seconds: ReadOnly[int]


class _Wire(TypedDict):
    cooldowns: ReadOnly[list[_WireCooldown]]


def _entry(model_id: str = "dep") -> CooldownCacheValue:
    return {"exception_received": model_id, "status_code": "429", "timestamp": 1000.0, "cooldown_time": 120.0}


@dataclass
class FakeCooldowns:
    redis_kwargs: Mapping[str, object] | None
    namespace: str | None
    stored: dict[str, CooldownCacheValue] = field(default_factory=dict)
    fails: dict[str, int] = field(default_factory=dict)
    targets: list[str | None] = field(default_factory=list)
    smallest: float | None = None

    def put(self, model_id: str, cooldown: Mapping[str, object], ttl_seconds: float) -> None:
        self.targets.append(current_service_target())
        self.stored[model_id] = CooldownCacheValue(**cooldown)  # pyright: ignore[reportArgumentType]  # test double

    def increment_allowed_fails(self, model_id: str, suffix: str | None, ttl_seconds: float) -> int:
        self.targets.append(current_service_target())
        self.fails[model_id] = self.fails.get(model_id, 0) + 1
        return self.fails[model_id]

    def active(self, model_ids: Sequence[str]) -> list[tuple[str, CooldownCacheValue]]:
        self.targets.append(current_service_target())
        return [(model_id, self.stored[model_id]) for model_id in model_ids if model_id in self.stored]

    def aactive(self, model_ids: Sequence[str]) -> "asyncio.Future[list[tuple[str, CooldownCacheValue]]]":
        future: Final[asyncio.Future[list[tuple[str, CooldownCacheValue]]]] = asyncio.get_running_loop().create_future()
        future.set_result(self.active(model_ids))
        return future

    def minimum(self, model_ids: Sequence[str]) -> float | None:
        return self.smallest

    def evict_local(self, model_id: str) -> None:
        self.stored.pop(model_id, None)

    def flush_local_counters(self) -> None:
        self.fails.clear()

    def snapshot(self) -> dict[str, CooldownCacheValue]:
        return dict(self.stored)


@dataclass
class Handles:
    built: list[FakeCooldowns] = field(default_factory=list)

    def __call__(self, connection: RedisCache | None) -> FakeCooldowns:
        handle: Final = FakeCooldowns(
            None if connection is None else redis_kwargs(connection),
            None if connection is None else connection.namespace,
        )
        self.built.append(handle)
        return handle


def _redis(namespace: str | None = None) -> RedisCache:
    return RedisCache(host="redis.internal", port=6380, namespace=namespace)


def test_the_store_hands_the_native_handle_redis_kwargs_and_namespace_not_a_client() -> None:
    handles: Final = Handles()
    RouterCooldownStore(_redis("tenant"), handle=handles)
    assert handles.built[0].redis_kwargs == {"host": "redis.internal", "port": 6380, "socket_timeout": 5.0}
    assert handles.built[0].namespace == "tenant"


def test_late_attach_rebuilds_once_and_later_attaches_are_ignored() -> None:
    handles: Final = Handles()
    store: Final = RouterCooldownStore(None, handle=handles)
    store.put("dep", _entry(), 60.0)
    store.attach_redis(None)
    store.attach_redis(_redis())
    store.attach_redis(_redis("other"))
    assert [handle.redis_kwargs is not None for handle in handles.built] == [False, True]
    assert handles.built[1].namespace is None
    assert store.active_sync(["dep"], None) == ()


def test_writes_and_reads_carry_the_router_cooldowns_service_target() -> None:
    handles: Final = Handles()
    store: Final = RouterCooldownStore(_redis(), handle=handles)
    store.put("dep", _entry(), 60.0)
    assert store.increment_allowed_fails("dep", "timeout", 30.0) == 1
    assert store.active_sync(["dep", "other"], None) == (("dep", _entry()),)
    assert handles.built[0].targets == ["router_cooldowns"] * 3
    assert current_service_target() is None


@pytest.mark.parametrize(("smallest", "expected"), ((None, 60.0), (0, 60.0), (5, 5)))
def test_minimum_falls_back_to_the_default_when_nothing_or_zero_is_stored(
    smallest: float | None, expected: float
) -> None:
    handles: Final = Handles()
    store: Final = RouterCooldownStore(None, handle=handles)
    handles.built[0].smallest = smallest
    assert store.minimum(["dep"], 60.0, None) == expected


@pytest.mark.asyncio
async def test_the_routing_read_pairs_native_cooldowns_with_the_selector_counters() -> None:
    handles: Final = Handles()
    store: Final = RouterCooldownStore(None, handle=handles)
    store.put("dep", _entry(), 60.0)
    selector: Final = DualCache()
    await selector.async_set_cache(  # pyright: ignore[reportUnknownMemberType]  # legacy DualCache is untyped
        "dep:model:tpm:12-34", 7
    )

    read: Final = await store.routing_read(["dep"], selector_read(selector, ["dep:model:tpm:12-34", "missing"]), None)
    alone: Final = await store.routing_read(["dep"], None, None)

    assert read.cooldowns == (("dep", _entry()),)
    assert list(read.selector_values or ()) == [7, None]
    assert alone.selector_values is None
    assert handles.built[0].targets[-1] == "router_cooldowns"


def test_the_wire_fixture_is_what_mains_codec_writes_and_reads() -> None:
    wire: Final = TypeAdapter(_Wire).validate_json(_WIRE.read_text())
    for case in wire["cooldowns"]:
        assert str(case["cooldown"]) == case["str_dict"]
        assert ast.literal_eval(case["str_dict"]) == case["cooldown"]
        assert json.loads(case["json"]) == case["cooldown"]
        assert int(case["ttl_seconds"]) == case["redis_ttl_seconds"]


@pytest.mark.parametrize("setting", ({"ssl_ciphers": "HIGH"}, {"credential_provider": object()}))
def test_a_redis_setting_the_native_client_cannot_honor_fails_construction(setting: dict[str, object]) -> None:
    redis: Final = RedisCache(host="127.0.0.1", port=1, **setting)
    with pytest.raises(ValueError, match=next(iter(setting))):
        RouterCooldownStore(redis)
