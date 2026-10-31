"""Writes redis_resolution.json, the settings litellm/_redis.py resolves for each case.

The Rust resolver in crates/config replays the file in tests/redis.rs, and
tests/unit/test_redis.py re-derives it so a change on either side fails a test.
Regenerate from the repo root after changing either resolver:

    PYTHONPATH=. python litellm-rust/crates/config/tests/generate_redis_resolution.py
"""

import json
import os
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

import redis
import redis.asyncio as async_redis

from litellm._redis import (
    _credential_provider_auth_kwargs,
    _get_redis_client_logic,
    _get_redis_cluster_kwargs,
    _get_redis_kwargs,
    _get_redis_url_kwargs,
)
from litellm._redis_credential_provider import (
    AzureADCredentialProvider,
    ElastiCacheIAMCredentialProvider,
    GCPIAMCredentialProvider,
)

FIXTURE: Final = Path(__file__).with_name("redis_resolution.json")
ENVIRONMENT_PREFIXES: Final = ("REDIS_", "AZURE_", "AWS_")

GCP_ACCOUNT: Final = "projects/-/serviceAccounts/sa@project.iam.gserviceaccount.com"
AZURE_SECRETS: Final = {"azure_client_id": "client", "azure_tenant_id": "tenant", "azure_client_secret": "secret"}
AWS_IAM: Final = {"aws_iam_auth": True, "aws_iam_user_name": "iam-user", "aws_iam_cache_name": "cache-1"}

CASES: Final[tuple[tuple[str, Mapping[str, object], Mapping[str, str]], ...]] = (
    ("host_defaults", {"host": "redis"}, {}),
    (
        "host_with_every_basic_setting",
        {
            "host": "redis",
            "port": 6380,
            "db": 2,
            "ssl": True,
            "username": "user",
            "password": "pass",
            "socket_timeout": 1.5,
        },
        {},
    ),
    (
        "environment_host_port_and_password",
        {},
        {
            "REDIS_HOST": "environment",
            "REDIS_PORT": "6390",
            "REDIS_PASSWORD": "environment-secret",
            "REDIS_SSL": "True",
        },
    ),
    ("string_numbers_are_coerced", {"host": "redis", "port": "6380", "db": "3", "socket_timeout": "2.5"}, {}),
    ("socket_timeout_from_the_environment", {"host": "redis"}, {"REDIS_SOCKET_TIMEOUT": "0.75"}),
    ("environment_url", {}, {"REDIS_URL": "redis://environment:6379/1"}),
    ("environment_url_beats_the_environment_host", {}, {"REDIS_URL": "redis://from-url:6379", "REDIS_HOST": "ignored"}),
    ("an_explicit_host_drops_the_environment_url", {"host": "configured"}, {"REDIS_URL": "redis://environment:6379"}),
    (
        "a_url_drops_host_db_and_password_and_keeps_the_username",
        {"url": "rediss://cache:6380", "host": "dropped", "db": 4, "username": "user", "password": "dropped"},
        {},
    ),
    (
        "a_url_ignores_tls_settings_outside_the_url",
        {"url": "rediss://cache:6380", "ssl_ca_certs": "/ca.pem", "ssl_cert_reqs": "none", "ssl_check_hostname": True},
        {},
    ),
    (
        "startup_nodes_beat_a_url",
        {
            "url": "redis://single:6379",
            "startup_nodes": [{"host": "node-a", "port": 7000}],
            "password": "cluster-secret",
        },
        {},
    ),
    (
        "environment_cluster_nodes",
        {},
        {"REDIS_CLUSTER_NODES": '[{"host": "node-a", "port": "7000"}, {"host": "node-b", "port": 7001}]'},
    ),
    (
        "cluster_tls_options",
        {
            "startup_nodes": [{"host": "node-a", "port": 7000}],
            "ssl": "true",
            "ssl_ca_certs": "/ca.pem",
            "ssl_check_hostname": True,
        },
        {},
    ),
    (
        "sentinel_from_settings",
        {
            "sentinel_nodes": [["sentinel-a", 26379], ["sentinel-b", 26380]],
            "service_name": "primary",
            "sentinel_password": "sentinel-secret",
            "password": "data-secret",
        },
        {},
    ),
    (
        "sentinel_from_the_environment",
        {},
        {
            "REDIS_SENTINEL_NODES": '[["sentinel-a", 26379]]',
            "REDIS_SERVICE_NAME": "primary",
            "REDIS_SENTINEL_PASSWORD": "sentinel-secret",
        },
    ),
    (
        "sentinels_beat_a_host",
        {"host": "ignored", "sentinel_nodes": [["sentinel-a", 26379]], "service_name": "primary"},
        {},
    ),
    (
        "a_url_beats_sentinels",
        {"url": "redis://single:6379", "sentinel_nodes": [["sentinel-a", 26379]], "service_name": "primary"},
        {},
    ),
    (
        "an_explicit_credential_provider_beats_cloud_auth",
        {
            "host": "redis",
            "password": "dropped",
            "credential_provider": True,
            "gcp_service_account": "account",
            "aws_iam_auth": True,
        },
        {"REDIS_AZURE_AD_TOKEN": "true"},
    ),
    (
        "gcp_beats_azure_and_aws",
        {"host": "redis", "password": "dropped", "azure_redis_ad_token": "True", "aws_iam_auth": True},
        {"REDIS_GCP_SERVICE_ACCOUNT": GCP_ACCOUNT},
    ),
    (
        "gcp_ca_certs_replace_the_configured_ones_under_ssl",
        {
            "host": "redis",
            "ssl": True,
            "ssl_ca_certs": "/configured-ca.pem",
            "gcp_service_account": "account",
            "gcp_ssl_ca_certs": "/gcp-ca.pem",
        },
        {},
    ),
    (
        "azure_beats_aws_and_takes_the_environment_username",
        {"host": "redis", "username": "dropped", "azure_redis_ad_token": True, **AZURE_SECRETS, "aws_iam_auth": True},
        {"REDIS_USERNAME": "environment-user"},
    ),
    (
        "azure_without_a_username_sends_only_the_token",
        {"host": "redis", "azure_redis_ad_token": True, **AZURE_SECRETS},
        {},
    ),
    ("azure_needs_the_literal_true", {"host": "redis", "azure_redis_ad_token": "1", **AZURE_SECRETS}, {}),
    (
        "aws_iam_over_tls_with_the_default_region",
        {"host": "cache-1", "ssl": True, "password": "dropped", **AWS_IAM, "aws_iam_auth": "yes"},
        {"AWS_DEFAULT_REGION": "us-east-1"},
    ),
    (
        "aws_region_beats_the_default_region",
        {"host": "cache-1", "ssl": True, **AWS_IAM},
        {"AWS_REGION": "eu-central-1", "AWS_DEFAULT_REGION": "us-east-1"},
    ),
    (
        "aws_iam_serverless_over_a_rediss_url_drops_its_userinfo",
        {
            "url": "rediss://user:pass@cache-1:6379",
            **AWS_IAM,
            "aws_iam_region": "eu-west-1",
            "aws_iam_serverless": True,
        },
        {},
    ),
    ("aws_iam_without_tls_fails", {"host": "cache-1", **AWS_IAM, "aws_iam_region": "us-east-1"}, {}),
    ("aws_iam_without_a_region_fails", {"host": "cache-1", "ssl": True, **AWS_IAM}, {}),
    (
        "tls_options_from_settings",
        {"host": "redis", "ssl": True, "ssl_ca_certs": "/ca.pem", "ssl_cert_reqs": "none", "ssl_check_hostname": True},
        {},
    ),
    (
        "tls_options_from_the_environment",
        {"host": "redis"},
        {"REDIS_SSL": "true", "REDIS_SSL_CA_CERTS": "/env-ca.pem", "REDIS_SSL_CERT_REQS": "optional"},
    ),
    (
        "connect_timeout_is_separate_from_the_socket_timeout",
        {"host": "redis", "socket_timeout": 2.5, "socket_connect_timeout": 0.25},
        {},
    ),
    (
        "environment_references_resolve",
        {"host": "os.environ/COORD_HOST", "password": "os.environ/COORD_PASSWORD"},
        {"COORD_HOST": "referenced", "COORD_PASSWORD": "referenced-secret"},
    ),
    ("no_target_fails", {"port": 6379}, {}),
    ("ssl_ciphers", {"host": "redis", "ssl": True, "ssl_ciphers": "HIGH"}, {}),
    ("retry_on_timeout", {"host": "redis", "retry_on_timeout": True}, {}),
    ("malformed_port", {"host": "redis", "port": "abc"}, {}),
    ("malformed_db", {"host": "redis", "db": "first"}, {}),
    ("socket_timeout_not_a_number", {"host": "redis", "socket_timeout": "slow"}, {}),
    ("malformed_port_from_the_environment", {}, {"REDIS_HOST": "redis", "REDIS_PORT": "abc"}),
    ("protocol_3", {"host": "redis", "protocol": 3}, {}),
    ("retry_on_error_from_the_environment", {"host": "redis"}, {"REDIS_RETRY_ON_ERROR": "true"}),
    ("unix_socket_path", {"host": "redis", "unix_socket_path": "/tmp/redis.sock"}, {}),
)


@contextmanager
def environment(values: Mapping[str, str]) -> Iterator[None]:
    saved: Final = dict(os.environ)
    try:
        for name in [name for name in os.environ if name.startswith(ENVIRONMENT_PREFIXES)]:
            del os.environ[name]
        os.environ.update(values)
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def _picked(kwargs: Mapping[str, object], allowed: Collection[str]) -> dict[str, object]:
    return {key: value for key, value in kwargs.items() if key in allowed}


def _tls(kwargs: Mapping[str, object]) -> dict[str, object]:
    return {
        "ca_certs": kwargs.get("ssl_ca_certs"),
        "verify_certificate": kwargs.get("ssl_cert_reqs", "required") != "none",
        "check_hostname": bool(kwargs.get("ssl_check_hostname", False)),
    }


def _node(node: Mapping[str, object] | Sequence[object]) -> str:
    match node:
        case {"host": host, "port": port}:
            return f"{host}:{port}"
        case [host, port]:
            return f"{host}:{port}"
        case _:
            raise ValueError(f"not a Redis node: {node!r}")


def _nodes(nodes: Iterable[Mapping[str, object] | Sequence[object]]) -> list[str]:
    return [_node(node) for node in nodes]


def _target(kwargs: Mapping[str, object]) -> tuple[dict[str, object], Mapping[str, object]]:
    if "startup_nodes" in kwargs:
        client_kwargs = _picked(kwargs, _get_redis_cluster_kwargs(async_redis.RedisCluster))
        tls = _tls(client_kwargs) if client_kwargs.get("ssl") else None
        return {"mode": "cluster", "nodes": _nodes(kwargs["startup_nodes"]), "tls": tls}, client_kwargs
    if kwargs.get("url") is not None:
        client_kwargs = _picked(kwargs, _get_redis_url_kwargs(client=async_redis.Redis))
        url = str(kwargs["url"])
        tls = _tls(client_kwargs) if urlsplit(url).scheme.lower() == "rediss" else None
        return {"mode": "url", "url": url, "tls": tls}, client_kwargs
    if "sentinel_nodes" in kwargs and "service_name" in kwargs:
        client_kwargs = _picked(kwargs, _get_redis_kwargs())
        return {
            "mode": "sentinel",
            "nodes": _nodes(kwargs["sentinel_nodes"]),
            "service_name": kwargs["service_name"],
            "sentinel_password": kwargs.get("sentinel_password"),
            "db": client_kwargs.get("db", 0),
            "tls": _tls(client_kwargs) if client_kwargs.get("ssl") else None,
        }, client_kwargs
    return {
        "mode": "host",
        "host": kwargs["host"],
        "port": kwargs.get("port", 6379),
        "db": kwargs.get("db", 0),
        "tls": _tls(kwargs) if kwargs.get("ssl") else None,
    }, kwargs


def _auth(client_kwargs: Mapping[str, object]) -> dict[str, object]:
    provider: Final = client_kwargs.get("credential_provider")
    match provider:
        case None:
            return {
                "kind": "static",
                "username": client_kwargs.get("username"),
                "password": client_kwargs.get("password"),
            }
        case GCPIAMCredentialProvider():
            return {"kind": "gcp_iam", "service_account": provider._gcp_service_account}
        case AzureADCredentialProvider():
            return {"kind": "azure_ad", "username": provider._username}
        case ElastiCacheIAMCredentialProvider():
            return {
                "kind": "aws_iam",
                "user_name": provider._user_name,
                "cache_name": provider._cache_name,
                "region": provider._region,
                "serverless": provider._is_serverless,
            }
        case _:
            return {"kind": "host"}


def resolve(kwargs: Mapping[str, object], values: Mapping[str, str]) -> dict[str, object]:
    with environment(values):
        try:
            resolved: Final = _credential_provider_auth_kwargs(_get_redis_client_logic(**kwargs))
        except ValueError:
            return {"rejected": True}
    target, client_kwargs = _target(resolved)
    return {
        **target,
        "auth": _auth(client_kwargs),
        "socket_timeout": client_kwargs.get("socket_timeout"),
        "socket_connect_timeout": client_kwargs.get("socket_connect_timeout"),
    }


def derive() -> dict[str, object]:
    return {
        "redis_py": redis.__version__,
        "cases": [
            {"name": name, "kwargs": kwargs, "environment": values, "python": resolve(kwargs, values)}
            for name, kwargs, values in CASES
        ],
    }


def render() -> str:
    return json.dumps(derive(), indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    FIXTURE.write_text(render())
