use std::{path::PathBuf, sync::Arc, time::Duration};

use coordinate_systems::Ground;
use linear_algebra::nalgebra;
use ros_z::{
    prelude::*,
    time::{Clock, Time},
};
use types::{
    ball_detection::BallPercept, ball_position::BallPosition,
    multivariate_normal_distribution::MultivariateNormalDistribution,
};

// Exercise the actual filter publisher and unchanged visual selector together.
// Seed a known sighting on their public topic, then stop all sensor streams.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn visual_target_expires_during_sensor_silence_but_survives_brief_gaps() {
    let clock = Clock::logical(Time::zero());
    let context = Arc::new(
        ContextBuilder::default()
            .with_namespace("/ball_filter_percept_expiry")
            .with_clock(clock.clone())
            .with_mode("peer")
            .disable_multicast_scouting()
            .with_connect_endpoints(std::iter::empty::<&str>())
            .with_listen_endpoints(std::iter::empty::<&str>())
            .with_parameter_layers([
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../etc/parameters/base")
            ])
            .build()
            .await
            .unwrap(),
    );
    let node = context.create_node("test_driver").build().await.unwrap();
    let output = node
        .subscriber::<Option<BallPosition<Ground>>>("visual_kick/ball_position")
        .build()
        .await
        .unwrap();
    let sighting = node
        .publisher::<Vec<BallPercept>>("ball_filter/ball_percepts")
        .build()
        .await
        .unwrap();
    let filter = tokio::spawn(ball_filter::run_boxed(context.clone()));
    let selector = tokio::spawn(visual_kick_ball_selector::run_boxed(context));
    tokio::time::timeout(Duration::from_secs(5), async {
        while !sighting.has_subscribers() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    sighting
        .publish_with_source_time(
            &vec![BallPercept {
                percept_in_ground: MultivariateNormalDistribution {
                    mean: nalgebra::vector![0.5, 0.0],
                    covariance: nalgebra::Matrix2::identity(),
                },
                image_location: Default::default(),
            }],
            Time::zero(),
        )
        .await
        .unwrap();
    let initial = tokio::time::timeout(Duration::from_secs(5), output.recv())
        .await
        .unwrap()
        .unwrap();
    assert!(initial.is_some());
    // Let the independently starting filter install its timer at clock zero.
    tokio::time::sleep(Duration::from_millis(100)).await;
    for millis in [40, 120, 240, 280, 400] {
        clock
            .set_time(Time::from_nanos(millis * 1_000_000))
            .unwrap();
        let value = tokio::time::timeout(Duration::from_secs(2), output.recv())
            .await
            .expect("visual output stopped during sensor silence")
            .unwrap();
        assert_eq!(value.is_some(), millis <= 250, "at {millis}ms");
    }
    filter.abort();
    selector.abort();
    let _ = filter.await;
    let _ = selector.await;
}
