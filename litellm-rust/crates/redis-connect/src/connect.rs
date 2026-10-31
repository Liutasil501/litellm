use std::sync::Arc;

use litellm_circuit_breaker::{Breaker, Config as BreakerConfig};
use litellm_config::{RedisAuth, RedisEndpoint, RedisTls, ResolvedRedis};
use litellm_core_utils::settings::Lookup;
use litellm_redis_client::{
    ClusterEndpoint, ClusterNode, Credentials, CredentialsProvider, DEFAULT_TIMEOUT, Endpoint,
    HostEndpoint, Namespace, RedisClient, RedisConfig, RedisEvents, SentinelEndpoint, Timeouts,
    TlsOptions,
};

use crate::{
    Error, RefreshingCredentials,
    sources::{AzureAd, ElastiCacheIam, GcpIam},
};

pub type SharedEnvironment = Arc<dyn Lookup + Send + Sync>;

pub struct HostServices {
    pub events: Option<Arc<dyn RedisEvents>>,
    pub breaker: Option<BreakerConfig>,
    pub http: litellm_http::Client,
    pub environment: SharedEnvironment,
}

pub fn connect(resolved: ResolvedRedis, host: HostServices) -> Result<RedisClient, Error> {
    tracing::info!(endpoint = ?resolved.endpoint, namespace = ?resolved.namespace, "redis client configured");
    let credentials = credentials(resolved.auth, &host)?;
    let events = host.events;
    let breaker = host.breaker.map(|config| {
        let observer = events.clone();
        Breaker::new(
            config,
            Box::new(move |transition| {
                if let Some(events) = &observer {
                    events.breaker(transition);
                }
            }),
        )
    });
    Ok(RedisClient::new(RedisConfig {
        endpoint: endpoint(resolved.endpoint),
        credentials,
        namespace: resolved.namespace.map(Namespace::new).unwrap_or_default(),
        timeouts: Timeouts::new(
            resolved.timeout.unwrap_or(DEFAULT_TIMEOUT),
            resolved.connect_timeout,
        ),
        breaker,
        events,
    })?)
}

fn credentials(auth: RedisAuth, host: &HostServices) -> Result<Credentials, Error> {
    let refreshing = |provider: RefreshingCredentials| {
        Credentials::Provider(CredentialsProvider(Arc::new(provider)))
    };
    Ok(match auth {
        RedisAuth::Static { username, password } => Credentials::Static { username, password },
        RedisAuth::Host => return Err(Error::NoHostCredentialProvider),
        RedisAuth::AzureAd(auth) => refreshing(RefreshingCredentials::new(AzureAd::new(auth))),
        RedisAuth::AwsIam(auth) => refreshing(RefreshingCredentials::new(ElastiCacheIam::new(
            auth,
            host.environment.clone(),
        ))),
        RedisAuth::GcpIam { service_account } => refreshing(RefreshingCredentials::new(
            GcpIam::new(host.http.clone(), service_account, host.environment.clone()),
        )),
    })
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
