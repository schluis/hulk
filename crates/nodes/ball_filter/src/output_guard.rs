//! Quarantine isolated, physically surprising reacquisitions in publication only.
//! The original hypothesis still controls association, confidence and existence.
use crate::{
    hypothesis::{BallHypothesis, BallMode},
    reacquisition,
};
use coordinate_systems::Ground;
use linear_algebra::{IntoFramed, Isometry2};
use nalgebra::{Matrix2, Matrix4, Vector2};
use ros_z::{Message, time::Time};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use types::{
    ball_position::BallPosition,
    multivariate_normal_distribution::MultivariateNormalDistribution as Gaussian,
    obstacles::Obstacle, parameters::BallFilterParameters,
};

#[derive(Clone, Debug, Serialize, Deserialize, Message)]
pub struct OutputGuard {
    mode: BallMode,
    motion_evidence: Option<crate::hypothesis::MotionEvidence>,
    last_seen: Time,
    rejected: Option<RejectedEvidence>,
}

#[derive(Clone, Debug, Serialize, Deserialize, Message)]
struct RejectedEvidence {
    time: Time,
    position: Vector2<f32>,
    count: u8,
}

impl OutputGuard {
    fn proxy(&self) -> BallHypothesis {
        let mut proxy = BallHypothesis::new(
            Gaussian {
                mean: nalgebra::Vector4::zeros(),
                covariance: Matrix4::identity(),
            },
            self.last_seen,
        );
        proxy.mode = self.mode.clone();
        proxy.motion_evidence = self.motion_evidence.clone();
        proxy
    }

    pub fn position(&self) -> BallPosition<Ground> {
        self.proxy().position()
    }

    pub fn predict(
        &mut self,
        dt: Duration,
        odometry: Isometry2<Ground, Ground>,
        decay: f32,
        moving_noise: Matrix4<f32>,
        resting_noise: Matrix2<f32>,
        zero_velocity_threshold: f32,
    ) {
        let mut proxy = self.proxy();
        proxy.predict(
            dt,
            odometry,
            decay,
            moving_noise,
            resting_noise,
            zero_velocity_threshold,
        );
        self.mode = proxy.mode;
        self.motion_evidence = proxy.motion_evidence;
        if let Some(evidence) = &mut self.rejected {
            evidence.position = (odometry * evidence.position.framed().as_point())
                .inner
                .coords;
        }
    }
}

pub fn observe(
    hypothesis: &mut BallHypothesis,
    time: Time,
    measurement: Gaussian<2>,
    obstacles: Option<&[Obstacle]>,
    measurement_size_supported: bool,
    parameters: &BallFilterParameters,
) {
    let distance = parameters.output_reacquisition_distance;
    if !distance.is_finite() || distance <= 0.0 {
        hypothesis.output_guard = None;
        return;
    }
    if time <= hypothesis.last_seen {
        return;
    }
    let was_guarded = hypothesis.output_guard.is_some();
    // Only close observations whose image size and filtered location agreed
    // supply a trusted reference. Stored confidence alone can belong to clutter.
    if !was_guarded
        && (hypothesis.output_guard_observation_supported != Some(true)
            || hypothesis.position().position.coords().norm_squared() > 1.5_f32.powi(2))
    {
        return;
    }
    let mut guard = hypothesis
        .output_guard
        .take()
        .unwrap_or_else(|| OutputGuard {
            mode: hypothesis.mode.clone(),
            motion_evidence: hypothesis.motion_evidence.clone(),
            last_seen: hypothesis.last_seen,
            rejected: None,
        });
    let mut proxy = guard.proxy();
    let displacement = measurement.mean - proxy.position().position.inner.coords;
    if displacement.norm_squared() <= distance * distance {
        if !was_guarded {
            return;
        }
        // A consistent observation restores trust. Keep the independent state
        // only while the baseline has not caught up with the trusted position.
        proxy.update(time, measurement, 0.0);
        guard.mode = proxy.mode;
        guard.motion_evidence = proxy.motion_evidence;
        guard.last_seen = time;
        guard.rejected = None;
        if (guard.position().position - hypothesis.position().position).norm() > 0.05 {
            hypothesis.output_guard = Some(guard);
        }
        return;
    }
    if !reacquisition::protect_prior(&proxy, time, obstacles, parameters) {
        return;
    }
    if !measurement_size_supported {
        guard.rejected = None;
        hypothesis.output_guard = Some(guard);
        return;
    }
    // Three geometrically supported observations may describe a real unseen kick. Do not
    // freeze the previous location indefinitely on a physical plausibility prior.
    let count = match guard.rejected {
        Some(RejectedEvidence {
            time: previous_time,
            position: previous_position,
            count,
        }) if time > previous_time
            && time.duration_since(previous_time) <= Duration::from_millis(200)
            && (measurement.mean - previous_position).norm() <= 0.2 =>
        {
            count + 1
        }
        _ => 1,
    };
    if count < 3 {
        guard.rejected = Some(RejectedEvidence {
            time,
            position: measurement.mean,
            count,
        });
        hypothesis.output_guard = Some(guard);
    }
}

pub fn supports_observation(
    hypothesis: &BallHypothesis,
    percept: &types::ball_detection::BallPercept,
    camera: Option<&projection::camera_matrix::CameraMatrix>,
    ball_radius: f32,
) -> bool {
    let position = percept.percept_in_ground.mean;
    if position.norm_squared() > 1.5_f32.powi(2)
        || (position - hypothesis.position().position.inner.coords).norm_squared() > 0.2_f32.powi(2)
    {
        return false;
    }
    supports_size(percept, camera, ball_radius)
}

pub fn supports_size(
    percept: &types::ball_detection::BallPercept,
    camera: Option<&projection::camera_matrix::CameraMatrix>,
    ball_radius: f32,
) -> bool {
    let position = percept.percept_in_ground.mean;
    let Some(camera) = camera else {
        return false;
    };
    let camera_position =
        camera.ground_to_camera * linear_algebra::point![position.x, position.y, ball_radius];
    let depth = camera_position.z();
    if !depth.is_finite() || depth <= 0.0 {
        return false;
    }
    let expected = ball_radius * camera.intrinsics.focals.x.min(camera.intrinsics.focals.y) / depth;
    let observed = percept.image_location.radius;
    expected.is_finite()
        && expected > 0.0
        && observed.is_finite()
        && observed > 0.0
        && observed <= 1.5 * expected
        && expected <= 1.5 * observed
}

#[cfg(test)]
mod tests {
    use super::*;
    fn track(x: f32) -> BallHypothesis {
        let mut hypothesis = BallHypothesis::new(
            Gaussian {
                mean: nalgebra::vector![x, 0.0, 0.0, 0.0],
                covariance: Matrix4::identity() * 0.01,
            },
            Time::zero(),
        );
        hypothesis.validity = 5.0;
        hypothesis.output_guard_observation_supported = Some(true);
        hypothesis
    }
    fn parameters() -> BallFilterParameters {
        BallFilterParameters {
            output_reacquisition_distance: 0.3,
            output_reacquisition_blend: 1.0,
            velocity_decay_factor: 0.998,
            ..Default::default()
        }
    }
    fn observation(x: f32) -> Gaussian<2> {
        Gaussian {
            mean: nalgebra::vector![x, 0.0],
            covariance: Matrix2::identity() * 0.1,
        }
    }
    #[test]
    fn isolated_jump_is_guarded_without_changing_baseline_state_or_confidence() {
        let mut original = track(1.0);
        let mut guarded = original.clone();
        for step in 0..3 {
            let time = Time::from_nanos(500_000_000 + step * 40_000_000);
            let measurement = observation(3.0);
            observe(
                &mut guarded,
                time,
                measurement,
                Some(&[]),
                true,
                &parameters(),
            );
            original.update(time, measurement, 1.0);
            guarded.update(time, measurement, 1.0);
            assert_eq!(original.position().position, guarded.position().position);
            assert_eq!(
                original.position_covariance(),
                guarded.position_covariance()
            );
            assert_eq!(original.validity, guarded.validity);
            assert_eq!(original.last_seen, guarded.last_seen);
            if step < 2 {
                assert_eq!(guarded.output_position(1.0).position.x(), 1.0);
            }
        }
        assert!(
            guarded.output_guard.is_none(),
            "coherent remote motion must release the guard"
        );
    }
    #[test]
    fn repeated_size_inconsistent_boxes_do_not_confirm_remote_motion() {
        let mut hypothesis = track(1.0);
        for step in 0..8 {
            let time = Time::from_nanos(500_000_000 + step * 40_000_000);
            let measurement = observation(3.0);
            observe(
                &mut hypothesis,
                time,
                measurement,
                Some(&[]),
                false,
                &parameters(),
            );
            hypothesis.update(time, measurement, 1.0);
            assert_eq!(hypothesis.output_position(1.0).position.x(), 1.0);
        }
    }

    #[test]
    fn unknown_geometry_kicking_reach_and_ordinary_updates_are_not_guarded() {
        for (x, measurement, obstacles) in [
            (1.0, 3.0, None),
            (0.3, 3.0, Some(&[][..])),
            (1.0, 1.1, Some(&[][..])),
        ] {
            let mut hypothesis = track(x);
            observe(
                &mut hypothesis,
                Time::from_nanos(500_000_000),
                observation(measurement),
                obstacles,
                true,
                &parameters(),
            );
            assert!(hypothesis.output_guard.is_none());
        }
    }
    #[test]
    fn unconfirmed_or_distant_history_cannot_override_a_new_detection() {
        for (x, validity) in [(1.0, 1.0), (3.0, 10.0)] {
            let mut hypothesis = track(x);
            hypothesis.validity = validity;
            if validity < 3.0 {
                hypothesis.output_guard_observation_supported = None;
            }
            observe(
                &mut hypothesis,
                Time::from_nanos(500_000_000),
                observation(0.4),
                Some(&[]),
                true,
                &parameters(),
            );
            assert!(hypothesis.output_guard.is_none());
        }
    }

    #[test]
    fn guard_and_remote_evidence_follow_odometry() {
        let mut hypothesis = track(1.0);
        observe(
            &mut hypothesis,
            Time::from_nanos(500_000_000),
            observation(3.0),
            Some(&[]),
            true,
            &parameters(),
        );
        let guard = hypothesis.output_guard.as_mut().unwrap();
        guard.predict(
            Duration::from_millis(40),
            Isometry2::from_parts(linear_algebra::vector![-0.5, 0.0], 0.0),
            0.998,
            Matrix4::identity() * 0.001,
            Matrix2::identity() * 0.001,
            0.5,
        );
        assert!((guard.position().position.x() - 0.5).abs() < 1e-5);
        assert!((guard.rejected.as_ref().unwrap().position.x - 2.5).abs() < 1e-5);
    }
}
