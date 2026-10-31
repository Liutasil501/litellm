mod support;

use std::time::Duration;

use litellm_redis_client::{Endpoint, Error, HostEndpoint};
use litellm_redis_testing::RedisServer;
use litellm_secret_value::SecretValue;
use rstest::rstest;

use support::connect_to;

#[rstest]
#[tokio::test]
async fn a_rejected_password_is_a_transport_failure() {
    let server = RedisServer::start().await.unwrap();
    server
        .cli(&["ACL", "SETUSER", "default", "on", ">right"])
        .await
        .unwrap();
    let port = server.url().rsplit(':').next().unwrap().parse().unwrap();
    let client = connect_to(
        Endpoint::Host(HostEndpoint {
            host: "127.0.0.1".to_owned(),
            port,
            username: None,
            password: Some(SecretValue::new("wrong")),
            tls: false,
            db: 0,
        }),
        "",
        Duration::from_secs(2),
        None,
    );

    let rejected = client.ping().await;

    assert!(matches!(rejected, Err(Error::Transport(_))), "{rejected:?}");
}
