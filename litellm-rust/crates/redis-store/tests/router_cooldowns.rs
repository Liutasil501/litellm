mod support;

use std::time::Duration;

use litellm_redis_client::{Expiry, Ttl};
use litellm_redis_store::{
    Number,
    router_cooldowns::{Cooldown, RouterCooldowns},
};
use litellm_redis_testing::{RedisDeployment, Topology};
use rstest::rstest;
use support::{cases, connect, text, within};

fn number(value: &serde_json::Value) -> Number {
    value
        .as_i64()
        .map(Number::Int)
        .unwrap_or_else(|| Number::Float(value.as_f64().unwrap()))
}

fn cooldown(case: &serde_json::Value) -> Cooldown {
    let fields = &case["cooldown"];
    Cooldown {
        exception_received: text(fields, "exception_received"),
        status_code: text(fields, "status_code"),
        timestamp: number(&fields["timestamp"]),
        cooldown_time: number(&fields["cooldown_time"]),
    }
}

fn ids(case: &serde_json::Value) -> Vec<String> {
    vec![text(case, "model_id")]
}

/// A clock fixed just after the fixture timestamps, so every fixture cooldown is still active.
fn fixture_clock() -> litellm_cache::Clock {
    std::sync::Arc::new(|| Duration::from_secs_f64(1_730_000_000.75))
}

#[rstest]
#[tokio::test]
async fn put_writes_mains_str_dict_bytes_with_whole_second_ttl(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let store = RouterCooldowns::new(Some(connect(&deployment)));

    for case in cases("cooldowns") {
        let ttl = Duration::from_secs_f64(case["ttl_seconds"].as_f64().unwrap());
        store
            .put(&text(&case, "model_id"), &cooldown(&case), ttl)
            .await
            .unwrap();
        let key = text(&case, "key");

        let written: Option<String> = raw.get(&key).await.unwrap();
        let Ttl::Expires(remaining) = raw.ttl(&key).await.unwrap() else {
            panic!("{key} has no TTL");
        };
        assert_eq!(written, Some(text(&case, "str_dict")));
        assert!(
            within(remaining, case["redis_ttl_seconds"].as_u64().unwrap()),
            "{remaining:?}"
        );
    }
}

#[rstest]
#[tokio::test]
async fn reads_cooldowns_python_wrote_as_str_dict_or_json(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
    #[values("str_dict", "json")] format: &str,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let store = RouterCooldowns::with_clock(Some(connect(&deployment)), fixture_clock());

    for case in cases("cooldowns") {
        raw.set(
            &text(&case, "key"),
            text(&case, format),
            Duration::from_secs(60),
        )
        .await
        .unwrap();

        assert_eq!(
            store.active(&ids(&case)).await.unwrap(),
            vec![(text(&case, "model_id"), cooldown(&case))]
        );
    }
}

#[rstest]
#[tokio::test]
async fn an_ended_cooldown_is_not_active_but_still_counts_toward_the_minimum(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let late = std::sync::Arc::new(|| Duration::from_secs(1_730_000_100));
    let store = RouterCooldowns::with_clock(Some(connect(&deployment)), late);
    let models = cases("cooldowns")
        .iter()
        .map(|case| text(case, "model_id"))
        .collect::<Vec<_>>();
    for case in cases("cooldowns") {
        raw.set(
            &text(&case, "key"),
            text(&case, "str_dict"),
            Duration::from_secs(60),
        )
        .await
        .unwrap();
    }

    assert_eq!(
        store.minimum(&models).await.unwrap(),
        Some(Number::Float(0.5))
    );
    assert_eq!(store.active(&models).await.unwrap(), vec![]);
}

#[rstest]
#[tokio::test]
async fn allowed_fails_count_across_routers_and_keep_the_first_ttl(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let first = RouterCooldowns::new(Some(connect(&deployment)));
    let second = RouterCooldowns::new(Some(connect(&deployment)));

    for case in cases("allowed_fails") {
        let key = text(&case, "key");
        raw.del(&[&key]).await.unwrap();
        let suffix = case["suffix"].as_str();

        let counts = [
            first
                .increment_allowed_fails("dep-a", suffix, Duration::from_secs_f64(30.9))
                .await
                .unwrap(),
            second
                .increment_allowed_fails("dep-a", suffix, Duration::from_secs(120))
                .await
                .unwrap(),
        ];
        let Ttl::Expires(remaining) = raw.ttl(&key).await.unwrap() else {
            panic!("{key} has no TTL");
        };

        assert_eq!(counts, [1, 2], "{key}");
        assert!(within(remaining, 30), "{remaining:?}");
    }
}

#[rstest]
#[tokio::test]
async fn python_written_fail_counts_continue_in_rust(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let store = RouterCooldowns::new(Some(connect(&deployment)));
    raw.incr_by(
        "deployment:dep-a:allowed_fails",
        4,
        Expiry::KeepExisting(Duration::from_secs(60)),
    )
    .await
    .unwrap();

    assert_eq!(
        store
            .increment_allowed_fails("dep-a", None, Duration::from_secs(60))
            .await
            .unwrap(),
        5
    );
}

#[rstest]
#[tokio::test]
async fn a_router_without_redis_keeps_its_cooldowns_and_counts_in_memory() {
    let store = RouterCooldowns::with_clock(None, fixture_clock());
    let case = &cases("cooldowns")[0];

    store
        .put(
            &text(case, "model_id"),
            &cooldown(case),
            Duration::from_secs(90),
        )
        .await
        .unwrap();
    let fails = [
        store
            .increment_allowed_fails("dep-a", None, Duration::from_secs(60))
            .await
            .unwrap(),
        store
            .increment_allowed_fails("dep-a", None, Duration::from_secs(60))
            .await
            .unwrap(),
    ];
    store.flush_local_counters().unwrap();

    assert_eq!(
        store.active(&ids(case)).await.unwrap(),
        vec![(text(case, "model_id"), cooldown(case))]
    );
    assert_eq!(fails, [1, 2]);
    assert_eq!(
        store
            .increment_allowed_fails("dep-a", None, Duration::from_secs(60))
            .await
            .unwrap(),
        1
    );
}

#[rstest]
#[tokio::test]
async fn a_redis_outage_serves_memory_and_counts_fails_locally() {
    let deployment = RedisDeployment::start(Topology::Standalone).await.unwrap();
    let store = RouterCooldowns::with_clock(Some(connect(&deployment)), fixture_clock());
    let case = &cases("cooldowns")[0];
    store
        .put(
            &text(case, "model_id"),
            &cooldown(case),
            Duration::from_secs(90),
        )
        .await
        .unwrap();
    deployment.stop().await.unwrap();

    let active = store
        .active(&[text(case, "model_id"), "dep-missing".to_owned()])
        .await
        .unwrap();
    let fails = store
        .increment_allowed_fails("dep-a", None, Duration::from_secs(60))
        .await
        .unwrap();

    assert_eq!(active, vec![(text(case, "model_id"), cooldown(case))]);
    assert_eq!(fails, 1);
}

struct StepClock(std::sync::Mutex<Duration>);

impl StepClock {
    fn at(seconds: f64) -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self(std::sync::Mutex::new(Duration::from_secs_f64(
            seconds,
        ))))
    }

    fn advance(&self, by: Duration) {
        *self.0.lock().unwrap() += by;
    }

    fn clock(self: &std::sync::Arc<Self>) -> litellm_cache::Clock {
        let this = self.clone();
        std::sync::Arc::new(move || *this.0.lock().unwrap())
    }
}

#[rstest]
#[tokio::test]
async fn a_miss_is_read_from_redis_again_only_after_the_one_second_refresh(
    #[values(Topology::Standalone, Topology::Cluster)] topology: Topology,
) {
    let deployment = RedisDeployment::start(topology).await.unwrap();
    let raw = connect(&deployment);
    let time = StepClock::at(1_730_000_000.75);
    let store = RouterCooldowns::with_clock(Some(connect(&deployment)), time.clock());
    let case = &cases("cooldowns")[0];

    let before = store.active(&ids(case)).await.unwrap();
    raw.set(
        &text(case, "key"),
        text(case, "str_dict"),
        Duration::from_secs(60),
    )
    .await
    .unwrap();
    let throttled = store.active(&ids(case)).await.unwrap();
    time.advance(Duration::from_secs(1));
    let refreshed = store.active(&ids(case)).await.unwrap();

    assert_eq!(before, vec![]);
    assert_eq!(throttled, vec![]);
    assert_eq!(refreshed, vec![(text(case, "model_id"), cooldown(case))]);
}

#[rstest]
#[tokio::test]
async fn a_breaker_refusal_does_not_hold_back_the_next_read() {
    let deployment = RedisDeployment::start(Topology::Standalone).await.unwrap();
    let raw = connect(&deployment);
    let breaker = litellm_circuit_breaker::Breaker::new(
        litellm_circuit_breaker::Config {
            failure_threshold: 1,
            recovery_timeout: Duration::from_millis(200),
            ..litellm_circuit_breaker::Config::default()
        },
        Box::new(|_| {}),
    );
    let time = StepClock::at(1_730_000_000.75);
    let guarded = support::connect_guarded(&deployment, breaker);
    let store = RouterCooldowns::with_clock(Some(guarded.clone()), time.clock());
    let case = &cases("cooldowns")[0];
    deployment.stop().await.unwrap();
    store.active(&ids(case)).await.unwrap();
    time.advance(Duration::from_secs(1));
    let refused = store.active(&ids(case)).await.unwrap();
    deployment.resume().await.unwrap();
    litellm_redis_testing::eventually(async || guarded.ping().await.ok()).await;
    raw.set(
        &text(case, "key"),
        text(case, "str_dict"),
        Duration::from_secs(60),
    )
    .await
    .unwrap();

    let recovered = store.active(&ids(case)).await.unwrap();

    assert_eq!(refused, vec![]);
    assert_eq!(recovered, vec![(text(case, "model_id"), cooldown(case))]);
}
