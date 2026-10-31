"""
``RustCore``: the router core served by litellm-rust/crates/router-core through ``_native``.

The Rust core owns the registry, selection, retries, fallbacks and cooldowns. It reaches outside
itself only through its host, and the host never calls back into a Python router:

- every attempt runs through the host: ``router_attempt`` builds the request for the one
  deployment Rust picked and ``PythonHost.invoke(op)`` calls ``litellm.<op>`` with it
- the core records every outcome (cooldowns, allowed-fails counters, usage) itself, in the routing
  cache the host builds and owns (``PythonHost.routing_cache``)
- sleeps, fallback access and budget checks, and cache reads and writes are host steps

``RustCore`` does not build, wrap or read a ``PythonCore``. Settings are resolved from the
``RouterConfig`` and the host's process settings; the views (``registry()``, ``settings()``,
``health()``) are read from the Rust handle. A member the Rust core does not serve raises
``RustCoreUnsupported`` (an ``AttributeError``) instead of reaching a Python core, so deleting
``router_python_core.py`` cannot change what this core does.
"""

import dataclasses
import inspect
import json
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, TypeAlias, cast

from pydantic import TypeAdapter

from litellm.integrations.custom_guardrail import is_guardrail_intervention
from litellm.litellm_core_utils.add_retry_fallback_headers import (
    add_fallback_headers_to_response,
    add_retry_headers_to_response,
    apply_remaining_usage_headers,
    apply_response_model_id,
    ensure_response_additional_headers,
    prepare_response_for_header_attachment,
)
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.routing_common_utils import (
    format_content_policy_fallback_hint,
    format_fallback_outcome_message,
    format_no_fallback_group_message,
    truncate_fallback_error_detail,
)
from litellm.litellm_core_utils.secret_redaction import redact_string
from litellm.router_api import (
    ApplyResult,
    ChangeSet,
    RouterHealth,
    RouterRegistry,
    RouterSettings,
    RoutingContext,
    RoutingObservers,
)
from litellm.router_attempt import (
    attempt_kwargs,
    for_requested_model,
    metadata_field,
    prepare_call,
    request_from_call,
    stamp_attempt_error,
)
from litellm.router_config import RouterConfig, RouterDynamicConfig
from litellm.router_python_host import PythonHost
from litellm.types.router import Deployment, RouterErrors, RouterRateLimitError
from litellm.types.utils import LlmProviders

if TYPE_CHECKING:
    from litellm.caching.dual_cache import DualCache
    from litellm.router_utils.cooldown_cache import CooldownCache
    from litellm.rust_bridge._native import (  # pyright: ignore[reportPrivateUsage]  # the stub's step shapes
        RouterCall,
        RouterCoreHandle,
        _AttemptFailed,  # pyright: ignore[reportPrivateUsage]  # the stub's step shape
    )
    from litellm.types.router import CustomRoutingStrategyBase

# Async endpoints the Rust core routes: select a deployment, then run the attempt on the host.
RUST_SERVED_OPS: Final[frozenset[str]] = frozenset(
    {
        "acompletion",
        "atext_completion",
        "aembedding",
        "aresponses",
        "aanthropic_messages",
        "aimage_generation",
        "atranscription",
        "aspeech",
        "arerank",
        "amoderation",
    }
)
# Sync endpoints, routed the same way with the host's sync endpoint and sync cache.
SYNC_OPS: Final = MappingProxyType(
    {
        "completion": "acompletion",
        "text_completion": "atext_completion",
        "embedding": "aembedding",
        "image_generation": "aimage_generation",
    }
)

_FAILURE_KINDS: Final = (
    "RateLimitError",
    "ContextWindowExceededError",
    "ContentPolicyViolationError",
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
    "BadRequestError",
    "Timeout",
    "APIConnectionError",
    "InternalServerError",
    "ServiceUnavailableError",
    "BadGatewayError",
    "GuardrailIntervention",
    "CallerCredentialError",
)
_DECLINED: Final = frozenset({"MalformedRequestFieldError", "UnsupportedEndpointError"})
_STEP: Final = TypeAdapter(Mapping[str, object])
# The litellm function an op's attempt calls, where it differs from the op's name.
_HOST_ENDPOINTS: Final = MappingProxyType({"aanthropic_messages": "anthropic_messages"})
_PROVIDERS: Final = tuple(provider.value for provider in LlmProviders)
_DEFAULT_COOLDOWN_SECONDS: Final = 5.0

# Constructor arguments this core reads. Any other argument set to something other than its
# unset value keeps the router on the Python core while both exist (make_core decides that from
# the config alone, before any core is built).
_SERVED_ARGUMENTS: Final = frozenset(
    {
        "model_list",
        "routing_strategy",
        "model_group_alias",
        "fallbacks",
        "default_fallbacks",
        "context_window_fallbacks",
        "content_policy_fallbacks",
        "num_retries",
        "max_fallbacks",
        "allowed_fails",
        "cooldown_time",
        "disable_cooldowns",
        "allowed_fails_policy",
        "retry_after",
        "timeout",
        "default_litellm_params",
        "fallback_access_check",
        "fallback_budget_check",
        "redis_url",
        "redis_host",
        "redis_port",
        "redis_password",
        "redis_db",
        "cache_kwargs",
        "set_verbose",
        "debug_level",
    }
)
# Arguments that may be passed at their default without asking for anything.
_UNSET_DEFAULTS: Final[Mapping[str, object]] = MappingProxyType(
    {
        "client_ttl": 3600,
        "cache_responses": False,
        "routing_strategy_args": {},
        "enable_pre_call_checks": False,
        "enable_tag_filtering": False,
        "retry_policy": None,
        "model_group_retry_policy": {},
    }
)
# Per-request routing overrides this core does not read yet: such a request is refused, not
# rerouted to another core.
_UNSERVED_REQUEST_KEYS: Final = frozenset(
    {
        "fallbacks",
        "context_window_fallbacks",
        "content_policy_fallbacks",
        "default_fallbacks",
        "max_fallbacks",
        "model_group_retry_policy",
        "retry_policy",
    }
)
# Settings a changeset may change. Static ones rebuild the Rust core (settings are fixed for a
# core's life); dynamic ones are handed to it as a new dynamic document.
_STATIC_SETTINGS: Final = frozenset(
    {"routing_strategy", "cooldown_time", "disable_cooldowns", "allowed_fails_policy", "retry_after", "timeout"}
)
_DYNAMIC_SETTINGS: Final = frozenset(
    {
        "fallbacks",
        "context_window_fallbacks",
        "content_policy_fallbacks",
        "num_retries",
        "max_fallbacks",
        "allowed_fails",
        "model_group_alias",
        "default_litellm_params",
        "fallback_access_check",
        "fallback_budget_check",
    }
)


class RustCoreUnsupported(AttributeError):
    """A member or request the Rust core does not serve. It is never forwarded to a Python core."""


# --- failures -----------------------------------------------------------------------------------


def failure_kind(error: BaseException) -> str:
    names: Final = tuple(cls.__name__ for cls in type(error).__mro__)
    if not _DECLINED.isdisjoint(names):
        return "Declined"
    if is_guardrail_intervention(error):  # pyright: ignore[reportArgumentType]  # every attempt error is an Exception
        return "GuardrailIntervention"
    return next((name for name in names if name in _FAILURE_KINDS), "APIError")


def _retry_after_seconds(error: BaseException) -> float | None:
    response: Final = getattr(error, "response", None)
    headers: Final[object] = getattr(error, "litellm_response_headers", None) or getattr(response, "headers", None)
    get: Final[object] = getattr(headers, "get", None)
    if headers is None or not callable(get):
        return None
    milliseconds: Final[object] = get("retry-after-ms")
    seconds: Final[object] = get("retry-after")
    try:
        value: Final = float(str(milliseconds)) / 1000 if milliseconds is not None else float(str(seconds))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value >= 0 else None


def attempt_failure(error: BaseException) -> Mapping[str, object]:
    status: Final = getattr(error, "status_code", None)
    num_retries: Final = getattr(error, "num_retries", None)
    return {
        "ok": False,
        "kind": failure_kind(error),
        "status": status if isinstance(status, int) and 100 <= status <= 599 else None,
        "retry_after": _retry_after_seconds(error),
        "num_retries": num_retries if isinstance(num_retries, int) and num_retries >= 0 else None,
        "detail": str(error)[:500],
    }


def total_tokens(response: object) -> int:
    usage: Final = getattr(response, "usage", None)
    tokens: Final = getattr(usage, "total_tokens", None)
    return tokens if isinstance(tokens, int) else 0


# --- settings -----------------------------------------------------------------------------------


def _json_default(value: object) -> object:
    dump: Final = getattr(value, "model_dump", None)
    return dump(exclude_none=True) if callable(dump) else str(value)


def _plain(value: object) -> object:
    """``value`` as plain JSON values, the shape the Rust core reads."""
    loaded: Final[object] = json.loads(json.dumps(value, default=_json_default))  # pyright: ignore[reportAny]  # plain JSON
    return loaded


def _document(value: object) -> Mapping[str, object]:
    """A mapping as the plain JSON document the Rust core reads."""
    return as_str_mapping(_plain(value)) or {}


def _as_list(value: object) -> Sequence[object]:
    return value if isinstance(value, list) else ()  # pyright: ignore[reportUnknownVariableType]  # a list's items are read as objects


@dataclasses.dataclass(frozen=True, slots=True)
class Resolved:
    """The router's settings, resolved from its arguments and the process settings the way the
    Python router's constructor resolves them. Replaced, never changed, by ``apply``."""

    routing_strategy: str
    num_retries: int
    max_fallbacks: int
    allowed_fails: int
    explicit_allowed_fails: int | None
    cooldown_time: float
    disable_cooldowns: bool | None
    allowed_fails_policy: object
    retry_after: float
    timeout: float | None
    fallbacks: list[object] | None  # mutable-ok: the setting as Python spells it
    context_window_fallbacks: list[object] | None  # mutable-ok: same
    content_policy_fallbacks: list[object] | None  # mutable-ok: same
    model_group_alias: Mapping[str, object]
    default_litellm_params: Mapping[str, object]
    fallback_access_check: object
    fallback_budget_check: object
    assigned: frozenset[str] = frozenset()

    @classmethod
    def of(cls, config: RouterConfig, dynamic: RouterDynamicConfig) -> "Resolved":
        get: Final = config.get
        configured: Final[Sequence[object] | None] = list(_as_list(get("fallbacks"))) or (
            list(dynamic.fallbacks) if dynamic.fallbacks else None
        )
        default_fallbacks: Final = get("default_fallbacks") or dynamic.default_fallbacks
        if isinstance(default_fallbacks, str):
            raise TypeError(
                f"default_fallbacks must be a list of model names, got '{default_fallbacks}'. Use ['{default_fallbacks}']"
            )
        fallbacks: Final[Sequence[object] | None] = (
            configured if default_fallbacks is None else [*(configured or ()), {"*": list(_as_list(default_fallbacks))}]
        )
        num_retries: Final = get("num_retries")
        max_fallbacks: Final = get("max_fallbacks")
        allowed_fails: Final = get("allowed_fails")
        timeout: Final = get("timeout")
        return cls(
            routing_strategy=str(get("routing_strategy") or "simple-shuffle"),
            num_retries=int(str(num_retries))
            if num_retries is not None
            else dynamic.num_retries
            if dynamic.num_retries is not None
            else 2,
            max_fallbacks=int(str(max_fallbacks))
            if max_fallbacks is not None
            else dynamic.max_fallbacks
            if dynamic.max_fallbacks is not None
            else dynamic.router_max_fallbacks,
            allowed_fails=int(str(allowed_fails)) if allowed_fails is not None else dynamic.allowed_fails,
            explicit_allowed_fails=int(str(allowed_fails)) if allowed_fails is not None else None,
            cooldown_time=float(str(get("cooldown_time") or _DEFAULT_COOLDOWN_SECONDS)),
            disable_cooldowns=_optional_bool(get("disable_cooldowns")),
            allowed_fails_policy=get("allowed_fails_policy"),
            retry_after=float(str(get("retry_after") or 0)),
            timeout=float(str(timeout)) if timeout else dynamic.request_timeout,
            fallbacks=fallbacks,
            context_window_fallbacks=list(_as_list(get("context_window_fallbacks")))
            or (list(dynamic.context_window_fallbacks) if dynamic.context_window_fallbacks else None),
            content_policy_fallbacks=list(_as_list(get("content_policy_fallbacks")))
            or (list(dynamic.content_policy_fallbacks) if dynamic.content_policy_fallbacks else None),
            model_group_alias=MappingProxyType(dict(as_str_mapping(get("model_group_alias")) or {})),
            default_litellm_params=MappingProxyType(dict(as_str_mapping(get("default_litellm_params")) or {})),
            fallback_access_check=get("fallback_access_check"),
            fallback_budget_check=get("fallback_budget_check"),
        )

    def changed(self, values: Mapping[str, object]) -> "Resolved":
        from litellm.router_api import ASSIGNABLE_ALIASES, PLAIN_SETTING_ATTRIBUTES, SETTING_FIELDS

        unknown: Final = sorted(set(values) - {*SETTING_FIELDS, *PLAIN_SETTING_ATTRIBUTES, *ASSIGNABLE_ALIASES})
        if unknown:
            raise ValueError(f"not assignable router settings: {unknown}")
        unserved: Final = sorted(set(values) - _STATIC_SETTINGS - _DYNAMIC_SETTINGS)
        if unserved:
            raise RustCoreUnsupported(f"the Rust router core does not serve the setting(s) {', '.join(unserved)} yet")
        updates: Final[dict[str, object]] = {}  # mutable-ok: built here, then applied once
        for name, value in values.items():
            if name == "model_group_alias" or name == "default_litellm_params":
                updates[name] = MappingProxyType(dict(as_str_mapping(value) or {}))
            elif name in ("num_retries", "max_fallbacks", "allowed_fails"):
                updates[name] = int(str(value))
                if name == "allowed_fails":
                    updates["explicit_allowed_fails"] = int(str(value))
            elif name in ("cooldown_time", "retry_after"):
                updates[name] = float(str(value or 0)) or (
                    _DEFAULT_COOLDOWN_SECONDS if name == "cooldown_time" else 0.0
                )
            elif name == "timeout":
                updates[name] = float(str(value)) if value is not None else None
            elif name == "routing_strategy":
                updates[name] = str(value)
            elif name == "disable_cooldowns":
                updates[name] = _optional_bool(value)
            elif name in ("fallbacks", "context_window_fallbacks", "content_policy_fallbacks"):
                updates[name] = list(_as_list(value)) or None
            else:
                updates[name] = value
        return dataclasses.replace(self, **updates, assigned=self.assigned | frozenset(values))  # pyright: ignore[reportArgumentType]  # each value converted to its field's type above

    def static_document(self, model_list: Sequence[object]) -> Mapping[str, object]:
        values: Final = {
            "model_list": _plain(list(model_list)),
            "routing_strategy": self.routing_strategy,
            "cooldown_time": self.cooldown_time,
            "disable_cooldowns": self.disable_cooldowns,
            "allowed_fails_policy": _plain(self.allowed_fails_policy),
            "retry_after": self.retry_after,
            "timeout": self.timeout,
        }
        return {name: value for name, value in values.items() if value is not None}

    def dynamic_document(self, version: int, model_names: Sequence[str]) -> Mapping[str, object]:
        values: Final = {
            "version": version,
            "model_group_alias": _plain(dict(self.model_group_alias)),
            "fallbacks": _plain(self.fallbacks),
            "context_window_fallbacks": _plain(self.context_window_fallbacks),
            "content_policy_fallbacks": _plain(self.content_policy_fallbacks),
            "num_retries": self.num_retries,
            "max_fallbacks": self.max_fallbacks,
            "allowed_fails": self._router_allowed_fails(),
            "fallback_keys": self._fallback_keys(model_names),
        }
        return {name: value for name, value in values.items() if value is not None}

    def _router_allowed_fails(self) -> int | None:
        """``allowed_fails`` when it counts as set on the router (``_is_allowed_fails_set_on_router``):
        passed explicitly, or different from the process default."""
        import litellm

        if self.explicit_allowed_fails is not None:
            return self.explicit_allowed_fails
        if self.allowed_fails != litellm.allowed_fails:
            return self.allowed_fails
        return None

    def _fallback_keys(self, model_names: Sequence[str]) -> Mapping[str, object]:
        """The spellings a fallback key matches besides the group name (`get_fallback_model_group`)."""
        from litellm.router_utils.fallback_event_handlers import (
            _provider_prefixed_model_group,  # pyright: ignore[reportPrivateUsage]  # the fallback walk's own spelling rule
        )

        keys: Final = tuple(  # comprehension-ok: the first group of each fallback entry, across the three lists
            str(next(iter(mapping)))
            for entries in (self.fallbacks, self.context_window_fallbacks, self.content_policy_fallbacks)
            for entry in _as_list(entries)
            if (mapping := as_str_mapping(entry))
        )
        names: Final = {*model_names, *self.model_group_alias}
        prefixed: Final = {
            name: spelled for name in names if (spelled := _provider_prefixed_model_group(name, keys)) is not None
        }
        return {"providers": _PROVIDERS, "provider_prefixed": prefixed}


def _optional_bool(value: object) -> bool | None:
    return None if value is None else bool(value)


def unsupported_arguments(config: RouterConfig) -> frozenset[str]:
    """Arguments the Rust core does not serve."""
    unserved: Final = frozenset(
        name
        for name, value in config.arguments.items()
        if name not in _SERVED_ARGUMENTS and value not in (None, _UNSET_DEFAULTS.get(name))
    )
    strategy: Final = config.get("routing_strategy") or "simple-shuffle"
    # `simple_shuffle` raises on a non-numeric weight when it picks; only the Python core keeps that.
    odd_weight: Final = any(
        (params := as_str_mapping((as_str_mapping(deployment) or {}).get("litellm_params"))) is not None
        and params.get("weight") is not None
        and (isinstance(params.get("weight"), bool) or not isinstance(params.get("weight"), int | float))
        for deployment in _as_list(config.get("model_list"))
    )
    return (
        unserved
        | ({"routing_strategy"} if strategy != "simple-shuffle" else frozenset[str]())
        | ({"model_list"} if odd_weight else frozenset[str]())
    )


def _team_id(request: Mapping[str, object]) -> str | None:
    metadata: Final = as_str_mapping(request.get("metadata") or request.get("litellm_metadata"))
    team_id: Final = metadata.get("user_api_key_team_id") if metadata is not None else None
    return team_id if isinstance(team_id, str) else None


def _same_deployment(old: Deployment, new: Deployment) -> bool:
    """An upsert that changes nothing (``_upsert_deployment``'s equality check)."""
    return (
        old.model_name == new.model_name
        and old.litellm_params == new.litellm_params
        and old.model_info == new.model_info
    )


def _refuse_unserved_deployments(model_list: Sequence[object]) -> None:
    """Deployment fields the Rust core does not read yet: refused, never silently ignored."""
    for entry in model_list:
        model_info = as_str_mapping((as_str_mapping(_record(entry)) or {}).get("model_info")) or {}
        if model_info.get("allowed_fails_policy") is not None:
            raise RustCoreUnsupported(
                "a deployment's model_info.allowed_fails_policy is not served by the Rust router core yet"
            )


def _start_fields(request: Mapping[str, object]) -> Mapping[str, object]:
    """The request fields the Rust core routes on."""
    fields: Final = {
        key: request[key]
        for key in ("specific_deployment", "num_retries", "disable_fallbacks")
        if request.get(key) is not None
    }
    weights: Final = request.get("_router_weights")
    team_id: Final = _team_id(request)
    return {
        **fields,
        **({"router_weights": weights} if weights is not None else {}),
        **({"team_id": team_id} if team_id else {}),
    }


def _deployment(record: object) -> Deployment:
    return Deployment(**dict(as_str_mapping(record) or {}))  # pyright: ignore[reportArgumentType]  # a model_list entry


# --- views --------------------------------------------------------------------------------------


class RustRegistry(RouterRegistry):
    """``registry()`` read from the Rust handle. Queries it does not serve raise ``RustCoreUnsupported``."""

    __slots__ = ("_core",)

    def __init__(self, core: "RustCore") -> None:  # pyright: ignore[reportMissingSuperCall]  # reads the Rust core, not a Router
        self._core = core

    def __getattribute__(self, name: str) -> object:
        if name in _REGISTRY_SERVED or name.startswith("__") or name == "_core":
            found: Final[object] = object.__getattribute__(self, name)  # pyright: ignore[reportAny]  # the view's own member
            return found
        raise RustCoreUnsupported(f"registry().{name} is not served by the Rust router core", name=name, obj=self)

    @property
    def version(self) -> int:
        return self._core._version  # pyright: ignore[reportPrivateUsage]  # the registry reads its own core

    def _records(self) -> Sequence[Mapping[str, object]]:
        return [as_str_mapping(record) or {} for record in self._core._handle.deployments()]  # pyright: ignore[reportPrivateUsage]  # see above

    def deployment(self, id: str) -> Deployment | None:
        record: Final = self._core._handle.deployment(id)  # pyright: ignore[reportPrivateUsage]  # see above
        return None if record is None else _deployment(record)

    def has_deployment(self, id: str) -> bool:
        return self._core._handle.deployment(id) is not None  # pyright: ignore[reportPrivateUsage]  # see above

    def deployment_dict(self, id: str) -> dict[str, object] | None:  # pyright: ignore[reportIncompatibleMethodOverride]  # mutable-ok: a fresh dict
        record: Final = self._core._handle.deployment(id)  # pyright: ignore[reportPrivateUsage]  # see above
        return None if record is None else dict(as_str_mapping(record) or {})

    def deployments(  # pyright: ignore[reportIncompatibleMethodOverride]  # the "group" source only
        self,
        model_name: str | None = None,
        *,
        source: str = "group",
        team_id: str | None = None,
        **_: object,  # kwargs-ok: the RouterRegistry signature
    ) -> list[Mapping[str, object]] | None:  # mutable-ok: a fresh list, as get_model_list returns
        if source != "group":
            raise RustCoreUnsupported(
                f"registry().deployments(source={source!r}) is not served by the Rust router core"
            )
        records: Final = self._records()
        if model_name is None:
            return list(records)
        target: Final = self._target(model_name)
        found: Final = [
            record for record in records if record.get("model_name") == target and _usable_by(record, team_id)
        ]
        return found

    def all(self) -> list[Mapping[str, object]]:  # pyright: ignore[reportIncompatibleMethodOverride]  # mutable-ok: a fresh list
        return list(self._records())

    def names(  # pyright: ignore[reportIncompatibleMethodOverride]  # the served kinds only
        self,
        model_name: str | None = None,
        *,
        kind: str = "groups",
        team_id: str | None = None,
        exclude_team_models: bool = False,
    ) -> list[str] | set[str]:  # mutable-ok: fresh, as the Python core returns them
        records: Final = self._records()
        if kind == "ids":
            return [
                str((as_str_mapping(record.get("model_info")) or {}).get("id"))
                for record in records
                if (model_name is None or record.get("model_name") == model_name)
                and not (exclude_team_models and _team_of(record) is not None)
            ]
        if kind == "group_set":
            return {str(record.get("model_name")) for record in records}
        if kind == "groups":
            names: Final[list[str]] = []  # mutable-ok: filled here, then returned
            for record in records:
                team: str | None = _team_of(record)
                if team is None:
                    names.append(str(record.get("model_name")))
                elif team_id is not None and team == team_id:
                    public = (as_str_mapping(record.get("model_info")) or {}).get("team_public_model_name")
                    if isinstance(public, str):
                        names.append(public)
            aliases: Final = self._core._resolved.model_group_alias  # pyright: ignore[reportPrivateUsage]  # see above
            names.extend(alias for alias, target in aliases.items() if not _hidden(target))
            return names
        raise RustCoreUnsupported(f"registry().names(kind={kind!r}) is not served by the Rust router core")

    def resolve(  # pyright: ignore[reportIncompatibleMethodOverride]  # the served kinds only
        self,
        name: str | None,
        *,
        kind: str = "group",
        team_id: str | None = None,
        **_: object,  # kwargs-ok: the RouterRegistry signature
    ) -> str | bool | None:
        aliases: Final = self._core._resolved.model_group_alias  # pyright: ignore[reportPrivateUsage]  # see above
        if kind == "from_id":
            record: Final = self._core._handle.deployment(str(name))  # pyright: ignore[reportPrivateUsage]  # see above
            return None if record is None else str((as_str_mapping(record) or {}).get("model_name"))
        if name is None:
            raise TypeError(f"registry().resolve(kind={kind!r}) needs a name")
        if kind == "alias":
            return _alias_target(aliases.get(name)) if name in aliases else None
        if kind == "group":
            resolution: Final = self._core._handle.resolve(name)  # pyright: ignore[reportPrivateUsage]  # see above
            if resolution is not None:
                return str(resolution["model_group"])
            return _alias_target(aliases.get(name)) if name in aliases else name
        if kind == "recognized":
            return self._core._handle.resolve(name) is not None  # pyright: ignore[reportPrivateUsage]  # see above
        raise RustCoreUnsupported(f"registry().resolve(kind={kind!r}) is not served by the Rust router core")

    def aliases(self) -> dict[str, object]:  # pyright: ignore[reportIncompatibleMethodOverride]  # mutable-ok: a fresh dict
        return dict(self._core._resolved.model_group_alias)  # pyright: ignore[reportPrivateUsage]  # see above

    def _target(self, name: str) -> str:
        aliases: Final = self._core._resolved.model_group_alias  # pyright: ignore[reportPrivateUsage]  # see above
        return _alias_target(aliases.get(name)) or name if name in aliases else name


_REGISTRY_SERVED: Final = frozenset(
    {
        "version",
        "deployment",
        "has_deployment",
        "deployment_dict",
        "deployments",
        "all",
        "names",
        "resolve",
        "aliases",
        "_records",
        "_target",
    }
)


def _team_of(record: Mapping[str, object]) -> str | None:
    team: Final = (as_str_mapping(record.get("model_info")) or {}).get("team_id")
    return team if isinstance(team, str) else None


def _usable_by(record: Mapping[str, object], team_id: str | None) -> bool:
    team: Final = _team_of(record)
    return team is None or team == team_id


def _hidden(target: object) -> bool:
    mapping: Final = as_str_mapping(target)
    return mapping is not None and mapping.get("hidden") is True


def _alias_target(target: object) -> str | None:
    mapping: Final = as_str_mapping(target)
    if mapping is not None:
        model: Final = mapping.get("model")
        return model if isinstance(model, str) else None
    return target if isinstance(target, str) else None


class RustSettings(RouterSettings):
    """``settings()``: the resolved settings as a frozen snapshot."""

    def as_dict(self) -> Mapping[str, object]:
        router: Final = self._router
        if not isinstance(router, _SettingsSource):
            return super().as_dict()
        return router.settings_dict()


class _SettingsSource:
    def settings_dict(self) -> Mapping[str, object]:
        raise NotImplementedError


class RustHealth(RouterHealth):
    """``health()``: cooldowns, fail counts and usage, read through the host's routing cache and
    written through the Rust core."""

    __slots__ = ("_core",)

    def __init__(self, core: "RustCore") -> None:  # pyright: ignore[reportMissingSuperCall]  # reads the Rust core, not a Router
        self._core = core

    def __getattribute__(self, name: str) -> object:
        if name in _HEALTH_SERVED or name.startswith("__") or name == "_core":
            found: Final[object] = object.__getattribute__(self, name)  # pyright: ignore[reportAny]  # the view's own member
            return found
        raise RustCoreUnsupported(f"health().{name} is not served by the Rust router core", name=name, obj=self)

    @property
    def store(self) -> "DualCache":
        return self._core._cache  # pyright: ignore[reportPrivateUsage]  # the health view reads its own core

    @property
    def cooldowns(self) -> "CooldownCache":
        return self._core._cooldowns  # pyright: ignore[reportPrivateUsage]  # see above

    def failures_this_minute(self, deployment_id: str) -> int:
        return int(self._core._handle.outcomes_this_minute(deployment_id)[1])  # pyright: ignore[reportPrivateUsage]  # see above

    def record_failure(self, deployment_id: str, exception: Exception, exception_status: str | int) -> bool:
        failure: Final = dict(attempt_failure(exception))
        if failure["status"] is None and str(exception_status).isdigit():
            failure["status"] = int(str(exception_status))
        handle: Final = self._core._handle  # pyright: ignore[reportPrivateUsage]  # see above
        failed: Final = cast("_AttemptFailed", failure)  # cast-ok: attempt_failure builds that shape
        call: Final = handle.report_failure(deployment_id, failed)
        done: Final = self._core._drive_sync_steps(call)  # pyright: ignore[reportPrivateUsage]  # see above
        return bool(done.get("cooled"))

    async def usage(  # pyright: ignore[reportIncompatibleMethodOverride]  # the same two kinds
        self, model_group: str, *, kind: Literal["current", "remaining"] = "current"
    ) -> tuple[int | None, int | None] | Mapping[str, int]:
        keys: Final = self._core._handle.usage_keys(model_group)  # pyright: ignore[reportPrivateUsage]  # see above
        if keys is None:
            if kind == "current":
                return None, None
            return {}
        values: Final = await self.store.async_batch_get_cache(  # pyright: ignore[reportUnknownMemberType]  # DualCache's reads are untyped
            keys=[tpm for _, tpm, _ in keys] + [rpm for _, _, rpm in keys]
        )
        if kind == "current":
            tpm_values: Final = (values or [])[: len(keys)]
            rpm_values: Final = (values or [])[len(keys) :]
            return _summed(tpm_values), _summed(rpm_values)
        if kind != "remaining":
            raise ValueError(f"unknown usage kind {kind!r}")
        current: Final = await self.usage(model_group)
        current_tpm, current_rpm = current if isinstance(current, tuple) else (None, None)
        limits: Final = [
            as_str_mapping((as_str_mapping(self._core._handle.deployment(id)) or {}).get("litellm_params")) or {}  # pyright: ignore[reportPrivateUsage]  # see above
            for id, _, _ in keys
        ]
        tpm_limit: Final = _limit(limits, "tpm")
        rpm_limit: Final = _limit(limits, "rpm")
        remaining: Final[dict[str, int]] = {}  # mutable-ok: the header map, built here
        if tpm_limit is not None:
            remaining["x-ratelimit-remaining-tokens"] = tpm_limit - (current_tpm or 0)
            remaining["x-ratelimit-limit-tokens"] = tpm_limit
        if rpm_limit is not None:
            remaining["x-ratelimit-remaining-requests"] = rpm_limit - (current_rpm or 0)
            remaining["x-ratelimit-limit-requests"] = rpm_limit
        return remaining


_HEALTH_SERVED: Final = frozenset({"store", "cooldowns", "failures_this_minute", "record_failure", "usage"})


def _summed(values: Sequence[object]) -> int | None:
    numbers: Final = [int(value) for value in values if isinstance(value, int | float)]
    return sum(numbers) if numbers else None


def _limit(params: Sequence[Mapping[str, object]], key: str) -> int | None:
    values: Final = [int(value) for p in params if isinstance((value := p.get(key)), int | float)]
    return sum(values) if values else None


# --- the core -----------------------------------------------------------------------------------


_EndpointImpl: TypeAlias = Callable[..., Awaitable[object]]


class RustCore(_SettingsSource):
    """A ``RouterCore`` served by the Rust router core over a ``PythonHost``. It holds no Python core."""

    def __init__(self, config: RouterConfig, host: PythonHost, resolved: Resolved) -> None:
        from litellm.router_utils.cooldown_cache import CooldownCache

        self._config: Final = config
        self._host: Final = host
        self._resolved: Resolved = resolved
        self._cache: Final = host.routing_cache(config)
        self._cooldowns: CooldownCache = CooldownCache(cache=self._cache, default_cooldown_time=resolved.cooldown_time)
        self._version: int = 1
        self._retired: bool = False
        self._settings: RustSettings | None = None
        self._handle: RouterCoreHandle = self._new_handle(list(_as_list(config.get("model_list"))))

    @classmethod
    def build(cls, config: RouterConfig, host: PythonHost) -> "RustCore":
        from litellm.rust_bridge._native import RustBridgeDeclined

        extra: Final = unsupported_arguments(config)
        if extra:
            raise RustBridgeDeclined(f"router arguments not served by the Rust core: {sorted(extra)}")
        return cls(config, host, Resolved.of(config, host.dynamic()))

    def __getattr__(self, name: str) -> object:
        # Only reached for members this class does not define: never forwarded anywhere.
        raise RustCoreUnsupported(f"'{name}' is not served by the Rust router core", name=name, obj=self)

    # -- building --------------------------------------------------------------------------------

    def _new_handle(self, model_list: Sequence[object]) -> "RouterCoreHandle":
        from litellm.rust_bridge._native import RouterCoreHandle

        _refuse_unserved_deployments(model_list)
        names: Final = [
            str(name)
            for entry in model_list
            if isinstance(name := (as_str_mapping(entry) or {}).get("model_name"), str)
        ]
        return RouterCoreHandle(
            self._resolved.static_document(model_list), self._resolved.dynamic_document(self._version, names)
        )

    def _rebuild(self) -> None:
        """A new Rust core for changed static settings, over the same deployments."""
        old: Final = self._handle
        self._handle = self._new_handle(list(old.deployments()))
        old.retire()

    # -- entry points ----------------------------------------------------------------------------

    def registry(self) -> RustRegistry:
        return RustRegistry(self)

    def settings(self) -> RustSettings:
        """A cached snapshot, replaced when a changeset changes a setting."""
        if self._settings is None:
            self._settings = self._settings_snapshot()
        return self._settings

    def _settings_snapshot(self) -> RustSettings:
        r: Final = self._resolved
        return RustSettings(
            routing_strategy=r.routing_strategy,
            fallbacks=r.fallbacks,
            context_window_fallbacks=r.context_window_fallbacks,
            content_policy_fallbacks=r.content_policy_fallbacks,
            default_litellm_params=dict(r.default_litellm_params),
            cooldown_time=r.cooldown_time,
            allowed_fails=r.allowed_fails,
            allowed_fails_policy=r.allowed_fails_policy,  # pyright: ignore[reportArgumentType]  # the policy as passed
            disable_cooldowns=r.disable_cooldowns,
            enable_pre_call_checks=False,
            enable_health_check_routing=False,
            enable_tag_filtering=False,
            fallback_access_check=r.fallback_access_check,  # pyright: ignore[reportArgumentType]  # the check as passed
            fallback_budget_check=r.fallback_budget_check,  # pyright: ignore[reportArgumentType]  # the check as passed
            cache_responses=False,
            _router=self,  # pyright: ignore[reportArgumentType]  # the snapshot reads as_dict from this core
            _unset=frozenset(
                {
                    "default_deployment",
                    "default_max_parallel_requests",
                    "health_check_ignore_transient_errors",
                    "background_health_check_model_groups",
                    "tag_filtering_match_any",
                    "tag_routing_prefix",
                    "router_general_settings",
                    "provider_budget_config",
                    "provider_default_deployment_ids",
                    "kubernetes_pod_discovery",
                    "search_tools",
                    "guardrail_list",
                }
            ),
        )

    def settings_dict(self) -> Mapping[str, object]:
        r: Final = self._resolved
        return {
            "routing_strategy_args": {},
            "routing_strategy": r.routing_strategy,
            "allowed_fails": r.allowed_fails,
            "cooldown_time": r.cooldown_time,
            "num_retries": r.num_retries,
            "timeout": r.timeout,
            "retry_after": r.retry_after,
            "fallbacks": r.fallbacks,
            "context_window_fallbacks": r.context_window_fallbacks,
            "model_group_alias": dict(r.model_group_alias),
            "routing_groups": [],
        }

    def health(self) -> RustHealth:
        return RustHealth(self)

    def observe(self) -> RoutingObservers:
        raise RustCoreUnsupported("observe() is not served by the Rust router core yet", name="observe", obj=self)

    def register_plugin(self, plugin: "CustomRoutingStrategyBase") -> None:
        # Plugins will run through the host's select step; until then a plugin is refused, never
        # silently served by another core.
        raise RustCoreUnsupported(
            f"register_plugin({type(plugin).__name__}) is not served by the Rust router core yet",
            name="register_plugin",
            obj=self,
        )

    def retire(self) -> None:
        self._retired = True
        self._handle.retire()

    def apply(self, changes: ChangeSet) -> ApplyResult:
        if changes.redis_cache is not None or changes.refresh_model_info:
            raise RustCoreUnsupported("apply(redis_cache / refresh_model_info) is not served by the Rust router core")
        values: Final = {**changes.assign, **changes.settings}
        if values:
            resolved: Final = self._resolved.changed(values)
            static_changed: Final = bool(_STATIC_SETTINGS & set(values))
            self._resolved = resolved
            self._settings = None
            if "cooldown_time" in values:
                from litellm.router_utils.cooldown_cache import CooldownCache

                self._cooldowns = CooldownCache(cache=self._cache, default_cooldown_time=resolved.cooldown_time)
            if static_changed:
                self._rebuild()
            else:
                names: Final = [str(d.get("model_name")) for d in self._handle.deployments()]
                self._handle.apply({"settings": resolved.dynamic_document(self._version, names)})
        upserted: Final[list[Deployment | None]] = []  # mutable-ok: filled per upsert
        deleted: Final[list[Deployment | None]] = []  # mutable-ok: filled per delete
        # The registry as each write leaves it, in order: upserts first, then deletes
        # (`apply_changes`). Writes that change nothing are not sent and keep the version.
        view: Final[dict[str, Deployment | None]] = {}  # mutable-ok: filled per write
        records: Final[list[object]] = []  # mutable-ok: the upserts that change a deployment
        deletes: Final[list[str]] = []  # mutable-ok: the deletes of a deployment that exists

        def current(id: str) -> Deployment | None:
            if id in view:
                return view[id]
            record: Final = self._handle.deployment(id)
            return None if record is None else _deployment(record)

        for deployment in changes.upsert:
            record = _plain(_record(deployment))
            new = _deployment(record)
            id = str(new.model_info.id)
            old = current(id)
            if old is not None and _same_deployment(old, new):
                upserted.append(None)
                continue
            view[id] = new
            records.append(record)
            upserted.append(new)
        for id in changes.delete:
            old = current(id)
            deleted.append(old)
            if old is not None:
                view[id] = None
                deletes.append(id)
        if records or deletes:
            self._handle.apply({"upsert": records, "delete": deletes})
            self._version += 1
        return ApplyResult(version=self._version, upserted=upserted, deleted=deleted)

    def upsert_deployment(self, deployment: Deployment) -> Deployment | None:
        from litellm.router_deprecated import (
            _note,  # pyright: ignore[reportPrivateUsage]  # the router's deprecation notice
        )

        _note("upsert_deployment", "apply(ChangeSet(upsert=[deployment]))")
        return self.apply(ChangeSet(upsert=[deployment])).upserted[0]

    def delete_deployment(self, id: str) -> Deployment | None:
        from litellm.router_deprecated import _note  # pyright: ignore[reportPrivateUsage]  # see above

        _note("delete_deployment", "apply(ChangeSet(delete=[id]))")
        return self.apply(ChangeSet(delete=[id])).deleted[0]

    def __setattr__(self, name: str, value: object) -> None:
        # Settings are fixed after construction: only apply() changes them. A public attribute
        # write would be silently ignored, so it is refused.
        if not name.startswith("_"):
            raise RustCoreUnsupported(
                f"writing {name!r} is not served by the Rust router core; use apply(ChangeSet(assign=...))",
                name=name,
                obj=self,
            )
        object.__setattr__(self, name, value)

    async def select(  # pyright: ignore[reportIncompatibleMethodOverride]  # the RouterCore selection
        self,
        model: str,
        request_kwargs: dict[str, object] | None = None,  # mutable-ok: the RouterCore signature
        *,
        kind: str = "pick",
        specific_deployment: bool | None = None,
        **_: object,  # kwargs-ok: the RouterCore select options this core ignores
    ) -> Mapping[str, object] | Sequence[Mapping[str, object]]:
        """Pick a deployment for ``model`` without calling it (``pick``), or list the deployments
        a pick would choose among (``healthy``)."""
        if kind not in ("pick", "healthy"):
            raise RustCoreUnsupported(f"select(kind={kind!r}) is not served by the Rust router core")
        fields: Final = dict(_start_fields(request_kwargs or {}))
        if specific_deployment:
            fields["specific_deployment"] = True
        call: Final = (self._handle.select if kind == "pick" else self._handle.healthy)(model, _document(fields))
        step: Mapping[str, object] = {}  # rebind-ok: one step per round
        outcome: object = None  # rebind-ok: the answer to the last step
        while True:
            step = _STEP.validate_python(call.next(outcome))  # pyright: ignore[reportArgumentType]  # Rust validates each outcome
            match step["kind"]:
                case "cache":
                    outcome = await self._answer_cache(step)
                case "select":
                    outcome = step["candidates"]
                case "done" if kind == "healthy":
                    records = (self._handle.deployment(str(i)) for i in _as_list(step.get("deployment_ids")))
                    return [for_requested_model(as_str_mapping(record) or {}, model) for record in records]
                case "done":
                    record = self._handle.deployment(str(step["deployment_id"]))
                    return for_requested_model(as_str_mapping(record) or {}, model)
                case "failed":
                    raise self._failure_error(step, model, None)
                case other:
                    raise RuntimeError(f"router-core returned an unknown step {other!r}")

    # -- endpoints -------------------------------------------------------------------------------

    async def call(self, op: str, request: Mapping[str, object]) -> object:
        if op in SYNC_OPS:
            return self._call_sync(op, request)
        if op not in RUST_SERVED_OPS:
            raise RustCoreUnsupported(f"endpoint {op!r} is not served by the Rust router core yet", name=op, obj=self)
        return await self._drive(op, request)

    async def acompletion(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        return await self.call("acompletion", request_from_call("acompletion", args, kwargs))

    async def atext_completion(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        return await self.call("atext_completion", request_from_call("atext_completion", args, kwargs))

    async def aembedding(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        return await self.call("aembedding", request_from_call("aembedding", args, kwargs))

    async def aresponses(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        return await self.call("aresponses", request_from_call("aresponses", args, kwargs))

    async def aanthropic_messages(self, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        return await self.call("aanthropic_messages", kwargs)

    async def anthropic_messages(self, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        """The proxy's ``/v1/messages`` route name for ``aanthropic_messages``; async, as in PythonCore."""
        return await self.call("aanthropic_messages", kwargs)

    async def aimage_generation(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        return await self.call("aimage_generation", request_from_call("aimage_generation", args, kwargs))

    async def atranscription(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        return await self.call("atranscription", request_from_call("atranscription", args, kwargs))

    async def aspeech(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        return await self.call("aspeech", request_from_call("aspeech", args, kwargs))

    async def arerank(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        return await self.call("arerank", request_from_call("arerank", args, kwargs))

    async def amoderation(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        return await self.call("amoderation", request_from_call("amoderation", args, kwargs))

    def completion(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        return self._call_sync("completion", request_from_call("completion", args, kwargs))

    def text_completion(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        return self._call_sync("text_completion", request_from_call("text_completion", args, kwargs))

    def embedding(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        return self._call_sync("embedding", request_from_call("embedding", args, kwargs))

    def image_generation(self, *args: object, **kwargs: object) -> object:  # kwargs-ok: endpoint payload
        return self._call_sync("image_generation", request_from_call("image_generation", args, kwargs))

    # -- driving a call --------------------------------------------------------------------------

    def _start(self, op: str, request: Mapping[str, object]) -> "tuple[RouterCall, Mapping[str, object]]":
        if self._retired:
            raise RuntimeError("this router core was retired")
        unserved: Final = sorted(
            key for key in request if key in _UNSERVED_REQUEST_KEYS or key.startswith("mock_testing_")
        )
        if unserved:
            raise RustCoreUnsupported(f"request keys not served by the Rust router core yet: {unserved}")
        model: Final = request.get("model")
        if not isinstance(model, str):
            raise TypeError(f"{op}() needs a model name")
        field: Final = metadata_field(op)
        if request.get(field) is not None and not isinstance(request.get(field), Mapping):
            raise TypeError(f"{field} must be a mapping")
        aliases: Final = self._resolved.model_group_alias
        prepared: Final = prepare_call(op, request, model if model in aliases else None)
        return self._handle.start(op, _document({"model": model, **_start_fields(request)})), prepared

    async def _drive(self, op: str, request: Mapping[str, object]) -> object:
        call, prepared = self._start(op, request)
        driver: Final = _Driver(self, op, prepared)
        outcome: object = None  # rebind-ok: the answer to the last step
        while True:
            step = _STEP.validate_python(call.next(outcome))  # pyright: ignore[reportArgumentType]  # Rust validates each outcome against the step it answers
            driver.take_events(step.get("events"))
            match step["kind"]:
                case "attempt":
                    outcome = await driver.attempt(step)
                case "sleep":
                    await self._host.sleep(float(str(step["seconds"])))
                    outcome = None
                case "allow_fallback":
                    outcome = await driver.allow_fallback(str(step["target"]))
                case "cache":
                    outcome = await self._answer_cache(step)
                case "select":
                    outcome = step["candidates"]
                case "done":
                    return await driver.finish_async(step)
                case "failed":
                    raise driver.failed(step)
                case other:
                    raise RuntimeError(f"router-core returned an unknown step {other!r}")

    def _call_sync(self, op: str, request: Mapping[str, object]) -> object:
        async_op: Final = SYNC_OPS[op]
        call, prepared = self._start(async_op, request)
        driver: Final = _Driver(self, op, prepared)
        outcome: object = None  # rebind-ok: the answer to the last step
        while True:
            step = _STEP.validate_python(call.next(outcome))  # pyright: ignore[reportArgumentType]  # see _drive
            driver.take_events(step.get("events"))
            match step["kind"]:
                case "attempt":
                    outcome = driver.attempt_sync(step)
                case "sleep":
                    self._host.sleep_sync(float(str(step["seconds"])))
                    outcome = None
                case "allow_fallback":
                    # The fallback checks are awaited on the async path only, as in the Python router.
                    outcome = True
                case "cache":
                    outcome = self._answer_cache_sync(step)
                case "select":
                    outcome = step["candidates"]
                case "done":
                    return driver.finish(step)
                case "failed":
                    raise driver.failed(step)
                case other:
                    raise RuntimeError(f"router-core returned an unknown step {other!r}")

    def _drive_sync_steps(self, call: "RouterCall") -> Mapping[str, object]:
        """Drive a call whose only steps are cache ops (``report_failure``) to its end."""
        outcome: object = None  # rebind-ok: the answer to the last step
        while True:
            step = _STEP.validate_python(call.next(outcome))  # pyright: ignore[reportArgumentType]  # see _drive
            match step["kind"]:
                case "cache":
                    outcome = self._answer_cache_sync(step)
                case "done":
                    return step
                case "failed":
                    raise RuntimeError(f"router-core ({step.get('error_kind')}): {step.get('error')}")
                case other:
                    raise RuntimeError(f"router-core asked for {other!r} while recording an outcome")

    # -- cache steps -----------------------------------------------------------------------------

    def _store_for(self, key: str) -> "DualCache":
        """Cooldowns live in the cooldown store; every other routing key in the routing cache."""
        return self._cooldowns.cooldown_store if key.endswith(":cooldown") else self._cache

    async def _answer_cache(self, step: Mapping[str, object]) -> object:
        key: Final = str(step.get("key"))
        ttl: Final = _ttl(step)
        match step.get("op"):
            case "get":
                return _plain(await self._store_for(key).async_get_cache(key))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # DualCache's reads are untyped
            case "batch_get":
                return _plain(
                    [await self._store_for(str(k)).async_get_cache(str(k)) for k in _as_list(step.get("keys"))]  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # see above
                )
            case "set":
                store: Final = self._store_for(key)
                if ttl is None:
                    await store.async_set_cache(key, step.get("value"))  # pyright: ignore[reportUnknownMemberType]  # DualCache's writes are untyped
                else:
                    await store.async_set_cache(key, step.get("value"), ttl=ttl)  # pyright: ignore[reportUnknownMemberType]  # see above
                return None
            case "increment":
                by: Final = _amount(step.get("value"))
                counter: Final = self._store_for(key)
                total: Final = (
                    await counter.async_increment_cache(key, by)  # pyright: ignore[reportUnknownMemberType]  # see above
                    if ttl is None
                    else await counter.async_increment_cache(key, by, ttl=ttl)  # pyright: ignore[reportUnknownMemberType]  # see above
                )
                return by if total is None else total
            case "delete":
                await self._store_for(key).async_delete_cache(key)
                return None
            case other:
                raise RuntimeError(f"router-core asked for an unknown cache op {other!r}")

    def _answer_cache_sync(self, step: Mapping[str, object]) -> object:
        key: Final = str(step.get("key"))
        ttl: Final = _ttl(step)
        match step.get("op"):
            case "get":
                return _plain(self._store_for(key).get_cache(key))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # DualCache's reads are untyped
            case "batch_get":
                return _plain([self._store_for(str(k)).get_cache(str(k)) for k in _as_list(step.get("keys"))])  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # see above
            case "set":
                store: Final = self._store_for(key)
                if ttl is None:
                    store.set_cache(key, step.get("value"))  # pyright: ignore[reportUnknownMemberType]  # DualCache's writes are untyped
                else:
                    store.set_cache(key, step.get("value"), ttl=ttl)  # pyright: ignore[reportUnknownMemberType]  # see above
                return None
            case "increment":
                by: Final = _amount(step.get("value"))
                counter: Final = self._store_for(key)
                # DualCache.increment_cache is annotated int but adds floats as given.
                return (
                    counter.increment_cache(key, by)  # pyright: ignore[reportUnknownMemberType, reportArgumentType]  # see above
                    if ttl is None
                    else counter.increment_cache(key, by, ttl=ttl)  # pyright: ignore[reportUnknownMemberType, reportArgumentType]  # see above
                )
            case "delete":
                self._store_for(key).delete_cache(key)  # pyright: ignore[reportUnknownMemberType]  # DualCache's delete is untyped
                return None
            case other:
                raise RuntimeError(f"router-core asked for an unknown cache op {other!r}")

    # -- errors the core raises when no attempt failed ------------------------------------------

    def _failure_error(self, step: Mapping[str, object], model: str, last: BaseException | None) -> BaseException:
        import litellm.exceptions as litellm_exceptions

        kind: Final = step.get("error_kind")
        aliases: Final = self._resolved.model_group_alias
        if kind == "unknown_model":
            named: Final = _alias_target(aliases.get(model)) or model if model in aliases else model
            return litellm_exceptions.BadRequestError(
                message=f"You passed in model={named}. {RouterErrors.no_healthy_deployments.value}",
                model=named,
                llm_provider="",
            )
        if kind == "no_healthy_deployments":
            group: Final = self.registry().resolve(model)
            ids: Final = self.registry().names(str(group), kind="ids")
            active: Final = self._cooldowns.get_active_cooldowns(list(ids), None)  # pyright: ignore[reportArgumentType]  # a list of ids
            cooling: Final = [id for id, _ in active]
            min_cooldown: Final = min(
                (_seconds(entry.get("cooldown_time"), self._resolved.cooldown_time) for _, entry in active),
                default=self._resolved.cooldown_time,
            )
            return RouterRateLimitError(
                model=model,
                cooldown_time=min_cooldown,
                enable_pre_call_checks=False,
                cooldown_list=cooling,
                model_ids=list(ids),  # pyright: ignore[reportArgumentType]  # a list of ids
            )
        if last is not None:
            return last
        return RuntimeError(f"router-core ({kind}): {step.get('error')}")


def _seconds(value: object, default: float) -> float | int:
    """A cooldown time as stored: an int stays an int, so error texts read ``60`` like Python's."""
    if isinstance(value, int | float) and not isinstance(value, bool):
        return value
    return int(default) if float(default).is_integer() else default


def _ttl(step: Mapping[str, object]) -> float | None:
    ttl: Final = step.get("ttl")
    return ttl if isinstance(ttl, int | float) else None


def _amount(value: object) -> float | int:
    return value if isinstance(value, int | float) and not isinstance(value, bool) else 1


def _record(deployment: object) -> object:
    dump: Final = getattr(deployment, "model_dump", None)
    return dump(exclude_none=True) if callable(dump) else deployment


class _Driver:
    """Answers the attempt and fallback steps of one call, and shapes its result.

    Each attempt is one host call against the deployment Rust picked. The driver keeps the error
    each fallback level ended on: a level that fails raises its own error, whatever its fallbacks
    raised, as the Python router does.
    """

    def __init__(self, core: RustCore, op: str, request: Mapping[str, object]) -> None:
        self._core: Final = core
        self._op: Final = op
        self._request: Final = request
        self._last_response: object = None
        self._last_error: BaseException | None = None
        self._level_errors: Final[dict[int, BaseException]] = {}  # mutable-ok: filled as the call runs
        self._served: tuple[str | None, Mapping[str, object]] = (None, MappingProxyType({}))

    def take_events(self, events: object) -> None:
        for raw_event in _as_list(events):
            event = as_str_mapping(raw_event)
            if event is None:
                continue
            if event.get("kind") != "retries_exhausted":
                continue
            error: BaseException | None = self._last_error
            # `async_function_with_retries` stamps the error its exhausted retry loop raises.
            if error is not None and type(error) in self._core._host.dynamic().exception_types:  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
                error.max_retries = event.get("max")  # pyright: ignore[reportAttributeAccessIssue]  # Python's router stamps the same attributes
                error.num_retries = event.get("attempted")  # pyright: ignore[reportAttributeAccessIssue]  # same

    def _kwargs(self, step: Mapping[str, object]) -> tuple[Mapping[str, object], Mapping[str, object]]:
        name: Final = step.get("model")
        requested: Final = str(name) if name is not None else None
        deployment: Final = for_requested_model(as_str_mapping(step.get("deployment")) or {}, requested)
        resolved: Final = self._core._resolved  # pyright: ignore[reportPrivateUsage]  # see above
        group: Final = str(step["model_group"])
        kwargs: Final = attempt_kwargs(
            self._op,
            self._request,
            deployment,
            model_group=requested if requested is not None and "*" in group else group,
            default_litellm_params=resolved.default_litellm_params,
            timeout=resolved.timeout,
        )
        self._served = (requested, kwargs)
        return deployment, kwargs

    def _failed(
        self, step: Mapping[str, object], deployment: Mapping[str, object], error: BaseException
    ) -> Mapping[str, object]:
        stamp_attempt_error(error, deployment)
        self._last_error = error
        level: Final = step.get("fallback_level")
        self._level_errors[level if isinstance(level, int) else 0] = error
        return attempt_failure(error)

    async def attempt(self, step: Mapping[str, object]) -> Mapping[str, object]:
        deployment, kwargs = self._kwargs(step)
        try:
            host: Final = self._core._host  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
            endpoint: Final = host.invoke(_HOST_ENDPOINTS.get(self._op, self._op))
            # Every op the Rust core serves names an async endpoint, so the call returns an awaitable.
            response: Final[object] = await cast("Awaitable[object]", endpoint(**kwargs))  # cast-ok: see above
        except Exception as error:  # noqa: BLE001  # every attempt failure is classified and handed to the Rust call loop
            return self._failed(step, deployment, error)
        self._last_response = response
        return {"ok": True, "total_tokens": total_tokens(response)}

    def attempt_sync(self, step: Mapping[str, object]) -> Mapping[str, object]:
        deployment, kwargs = self._kwargs(step)
        try:
            response: Final = self._core._host.invoke(self._op)(**kwargs)  # pyright: ignore[reportPrivateUsage]  # a sync endpoint
        except Exception as error:  # noqa: BLE001  # see attempt
            return self._failed(step, deployment, error)
        if inspect.isawaitable(response):
            raise TypeError(f"the host's {self._op} endpoint is not synchronous")
        self._last_response = response
        return {"ok": True, "total_tokens": total_tokens(response)}

    async def allow_fallback(self, target: str) -> bool:
        """`_is_fallback_target_authorized`, then `_is_fallback_target_within_budget`."""
        if target == self._request.get("model"):
            return True
        ctx: Final = RoutingContext(self._core)  # pyright: ignore[reportArgumentType]  # the checks read registry() and settings()
        request: Final = dict(self._request)
        if not await ctx.fallback_access_allowed(model=target, request_kwargs=request):
            return False
        return await ctx.fallback_budget_allowed(model=target, request_kwargs=request)

    def finish(self, step: Mapping[str, object]) -> object:
        """The response with the router's headers: the deployment id, the model group it was
        served as, and the retry and fallback counts (``set_response_headers``)."""
        response: object = prepare_response_for_header_attachment(
            self._last_response
        )  # rebind-ok: gains its headers in turn
        if response is None:
            return self._last_response
        name, kwargs = self._served
        additional: Final = ensure_response_additional_headers(response)
        apply_response_model_id(response, as_str_mapping(kwargs.get(metadata_field(self._op))))
        additional["x-litellm-model-group"] = name
        retries: Final = step.get("attempted_retries")
        if isinstance(retries, int):
            max_retries: Final = step.get("max_retries")
            response = add_retry_headers_to_response(  # rebind-ok: see above
                response=response,
                attempted_retries=retries,
                max_retries=max_retries if isinstance(max_retries, int) and retries > 0 else None,
            )
        fallbacks: Final = step.get("attempted_fallbacks")
        return add_fallback_headers_to_response(
            response=response, attempted_fallbacks=fallbacks if isinstance(fallbacks, int) else 0
        )

    async def finish_async(self, step: Mapping[str, object]) -> object:
        response: Final = self.finish(step)
        name: Final = self._served[0]
        if name is not None and response is not None:
            remaining: Final = await self._core.health().usage(name, kind="remaining")
            if isinstance(remaining, Mapping):
                apply_remaining_usage_headers(ensure_response_additional_headers(response), dict(remaining))
        return response

    def failed(self, step: Mapping[str, object]) -> BaseException:
        model: Final = str(self._request.get("model"))
        primary: Final = self._level_errors.get(0)
        if primary is None:
            return self._core._failure_error(step, model, self._last_error)  # pyright: ignore[reportPrivateUsage]  # see above
        if self._core._host.dynamic().expose_router_debug_in_errors and hasattr(primary, "message"):  # pyright: ignore[reportPrivateUsage]  # see above
            primary.message += self._explanation(primary, model)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # checked above
        return primary

    def _explanation(self, primary: BaseException, model: str) -> str:
        """What ``_async_function_with_fallbacks_common_utils`` appends to the error it re-raises
        under ``expose_router_debug_in_errors``, from the resolved settings and what the call did."""
        import litellm.exceptions as litellm_exceptions
        from litellm.router_utils.fallback_event_handlers import get_fallback_model_group_for_lookup_groups

        resolved: Final = self._core._resolved  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
        lookup: Final = (model,)
        context_window: Final = isinstance(primary, litellm_exceptions.ContextWindowExceededError)
        content_policy: Final = isinstance(primary, litellm_exceptions.ContentPolicyViolationError)
        # The error kind's own list when it is set, else the generic one with a hint to set it.
        own: Final = (
            resolved.context_window_fallbacks
            if context_window
            else resolved.content_policy_fallbacks
            if content_policy
            else None
        )
        hint: Final = (
            f"\nmodel={model}. context_window_fallbacks=None. fallbacks={resolved.fallbacks}.\n\n"
            "Set 'context_window_fallback' - https://docs.litellm.ai/docs/routing#fallbacks"
            if context_window and own is None
            else "\n" + format_content_policy_fallback_hint(model, None, resolved.fallbacks)
            if content_policy and own is None
            else ""
        )
        source: Final = own if own is not None else resolved.fallbacks
        entries: Final = [dict(m) for entry in _as_list(source) if (m := as_str_mapping(entry))]
        found, generic = (
            get_fallback_model_group_for_lookup_groups(entries, lookup)  # pyright: ignore[reportArgumentType]  # the fallbacks config shape
            if entries
            else (None, None)
        )
        chain: Final[Sequence[object] | None] = (
            found if found is not None else _as_list(entries[generic].get("*")) if generic is not None else None
        )
        if chain is None and entries and own is None:
            return hint + format_no_fallback_group_message(lookup, entries)  # pyright: ignore[reportArgumentType]  # the fallbacks config shape
        if chain is None:
            return hint + format_fallback_outcome_message(model, None, "")
        return hint + format_fallback_outcome_message(model, chain, self._fallback_detail(chain))

    def _fallback_detail(self, chain: Sequence[object]) -> str:
        """The error the fallback walk ended on: the first fallback level's last error, or the
        unknown-model error of a target with no deployment, which fails before any attempt."""
        last: Final = str(chain[-1]) if chain else None
        attempted: Final = self._level_errors.get(1)
        unknown: Final = last is not None and self._core._handle.resolve(last) is None  # pyright: ignore[reportPrivateUsage]  # the driver serves its own core
        error: Final = (
            attempted
            if attempted is not None or not unknown or last is None
            else self._core._failure_error({"error_kind": "unknown_model"}, last, None)  # pyright: ignore[reportPrivateUsage]  # see above
        )
        return truncate_fallback_error_detail(redact_string(str(error))) if error is not None else ""


__all__ = [
    "RUST_SERVED_OPS",
    "Resolved",
    "RustCore",
    "RustCoreUnsupported",
    "RustHealth",
    "RustRegistry",
    "RustSettings",
    "attempt_failure",
    "failure_kind",
    "total_tokens",
    "unsupported_arguments",
]
