#![allow(dead_code)] // each test binary compiles this module on its own and uses a different subset

use std::time::Duration;

use litellm_redis_client::{
    ClusterEndpoint, ClusterNode, Credentials, Endpoint, Namespace, RedisClient, RedisConfig,
    Timeouts, TlsOptions,
};
use litellm_redis_testing::RedisDeployment;

pub fn connect(deployment: &RedisDeployment) -> RedisClient {
    connect_with(deployment, None)
}

pub fn connect_guarded(
    deployment: &RedisDeployment,
    breaker: litellm_circuit_breaker::Breaker,
) -> RedisClient {
    connect_with(deployment, Some(breaker))
}

fn connect_with(
    deployment: &RedisDeployment,
    breaker: Option<litellm_circuit_breaker::Breaker>,
) -> RedisClient {
    let endpoint = match deployment {
        RedisDeployment::Standalone(server) => Endpoint::Url {
            url: litellm_secret_value::SecretValue::new(server.url().to_owned()),
            tls: TlsOptions::verified(),
        },
        RedisDeployment::Cluster(cluster) => Endpoint::Cluster(ClusterEndpoint {
            nodes: cluster
                .nodes()
                .map(|(host, port)| ClusterNode {
                    host: host.to_owned(),
                    port,
                })
                .collect(),
            tls: None,
        }),
    };
    RedisClient::new(RedisConfig {
        endpoint,
        credentials: Credentials::NONE,
        namespace: Namespace::default(),
        timeouts: Timeouts::new(Duration::from_millis(500), None),
        breaker,
        events: None,
    })
    .unwrap()
}

pub fn wire() -> serde_json::Value {
    serde_json::from_str(include_str!("../fixtures/router_wire.json")).unwrap()
}

pub fn cases(name: &str) -> Vec<serde_json::Value> {
    wire()[name].as_array().unwrap().clone()
}

pub fn text(case: &serde_json::Value, field: &str) -> String {
    case[field].as_str().unwrap().to_owned()
}

/// A TTL Redis reports for a key written `whole_seconds` ago at most a few seconds back.
pub fn within(ttl: Duration, whole_seconds: u64) -> bool {
    ttl <= Duration::from_secs(whole_seconds)
        && ttl > Duration::from_secs(whole_seconds.saturating_sub(5))
}
