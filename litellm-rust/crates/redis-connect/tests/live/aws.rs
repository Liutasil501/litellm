use rstest::{fixture, rstest};

use super::{LiveConfig, config, roundtrip};

#[fixture]
fn aws_config() -> Option<LiveConfig> {
    config("LITELLM_REDIS_AWS_LIVE_CONFIG")
}

#[rstest]
#[tokio::test]
async fn credentials_authenticate_existing_and_fresh_connections(aws_config: Option<LiveConfig>) {
    let Some(config) = aws_config else { return };
    roundtrip(config).await;
}
