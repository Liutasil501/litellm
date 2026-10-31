use litellm_circuit_breaker::Breaker;
use litellm_config::{RedisAuth, RedisEndpoint, RedisTls, ResolvedRedis};
use litellm_redis_client::{
    ClusterEndpoint, ClusterNode, Credentials, DEFAULT_TIMEOUT, Endpoint, HostEndpoint, Namespace,
    RedisClient, RedisConfig, SentinelEndpoint, Timeouts, TlsOptions,
};

use crate::Error;

pub fn connect(resolved: ResolvedRedis, breaker: Option<Breaker>) -> Result<RedisClient, Error> {
    tracing::info!(endpoint = ?resolved.endpoint, namespace = ?resolved.namespace, "redis client configured");
    Ok(RedisClient::new(RedisConfig {
        endpoint: endpoint(resolved.endpoint),
        credentials: credentials(resolved.auth)?,
        namespace: resolved.namespace.map(Namespace::new).unwrap_or_default(),
        timeouts: Timeouts::new(
            resolved.timeout.unwrap_or(DEFAULT_TIMEOUT),
            resolved.connect_timeout,
        ),
        breaker,
        events: None,
    })?)
}

fn credentials(auth: RedisAuth) -> Result<Credentials, Error> {
    match auth {
        RedisAuth::Static { username, password } => Ok(Credentials { username, password }),
        RedisAuth::Host => Err(Error::NoHostCredentialProvider),
        RedisAuth::GcpIam { .. } => Err(Error::CloudAuth {
            setting: "gcp_service_account",
        }),
        RedisAuth::AzureAd(_) => Err(Error::CloudAuth {
            setting: "azure_redis_ad_token",
        }),
        RedisAuth::AwsIam(_) => Err(Error::CloudAuth {
            setting: "aws_iam_auth",
        }),
    }
}

fn endpoint(endpoint: RedisEndpoint) -> Endpoint {
    let nodes = |nodes: Vec<litellm_config::RedisNode>| {
        nodes
            .into_iter()
            .map(|node| ClusterNode {
                host: node.host,
                port: node.port,
            })
            .collect()
    };
    match endpoint {
        RedisEndpoint::Url { url, tls } => Endpoint::Url {
            url,
            tls: tls_options(tls),
        },
        RedisEndpoint::Host(host) => Endpoint::Host(HostEndpoint {
            host: host.host,
            port: host.port,
            tls: host.tls.map(tls_options),
            db: host.db,
        }),
        RedisEndpoint::Cluster(cluster) => Endpoint::Cluster(ClusterEndpoint {
            nodes: nodes(cluster.nodes),
            tls: cluster.tls.map(tls_options),
        }),
        RedisEndpoint::Sentinel(sentinel) => Endpoint::Sentinel(SentinelEndpoint {
            sentinels: nodes(sentinel.sentinels),
            service_name: sentinel.service_name,
            sentinel_password: sentinel.sentinel_password,
            tls: sentinel.tls.map(tls_options),
            db: sentinel.db,
        }),
    }
}

fn tls_options(tls: RedisTls) -> TlsOptions {
    TlsOptions {
        ca_certs: tls.ca_certs.map(Into::into),
        certfile: tls.certfile.map(Into::into),
        keyfile: tls.keyfile.map(Into::into),
        verify_certificate: tls.verify_certificate,
        check_hostname: tls.check_hostname,
    }
}
