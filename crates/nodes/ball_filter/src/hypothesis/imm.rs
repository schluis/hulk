//! Two-model interacting Kalman filter: strongly damped/resting and rolling.
//! Standard IMM mixing includes between-model mean spread in covariance.
use super::moving::{MovingPredict, MovingUpdate};
use coordinate_systems::Ground;
use linear_algebra::Isometry2;
use nalgebra::{Matrix2, Matrix4};
use ros_z::Message;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use types::multivariate_normal_distribution::MultivariateNormalDistribution as Gaussian;

#[derive(Clone, Debug, Serialize, Deserialize, Message)]
pub struct Imm {
    pub states: [Gaussian<4>; 2],
    pub moving_probability: f32,
    pub transition_rate: f32,
    #[serde(default = "unit_scale")]
    pub measurement_scale: f32,
    #[serde(default = "unit_scale")]
    pub process_scale: f32,
    #[serde(default = "unit_scale")]
    pub output_blend: f32,
}

fn mixture(a: Gaussian<4>, b: Gaussian<4>, weight_b: f32) -> Gaussian<4> {
    let mean = a.mean * (1.0 - weight_b) + b.mean * weight_b;
    let da = a.mean - mean;
    let db = b.mean - mean;
    Gaussian {
        mean,
        covariance: (a.covariance + da * da.transpose()) * (1.0 - weight_b)
            + (b.covariance + db * db.transpose()) * weight_b,
    }
}

impl Imm {
    pub fn new(state: Gaussian<4>, transition_rate: f32) -> Self {
        Self {
            states: [state; 2],
            moving_probability: 0.5,
            transition_rate,
            measurement_scale: 1.0,
            process_scale: 1.0,
            output_blend: 1.0,
        }
    }
    pub fn combined(&self) -> Gaussian<4> {
        mixture(self.states[0], self.states[1], self.moving_probability)
    }
    pub fn predict(
        &mut self,
        dt: Duration,
        odometry: Isometry2<Ground, Ground>,
        decay: f32,
        moving_noise: Matrix4<f32>,
        resting_noise: Matrix2<f32>,
    ) {
        let moving_noise = moving_noise * self.process_scale;
        let resting_noise = resting_noise * self.process_scale;
        let switch = 0.5 * -(-2.0 * self.transition_rate * dt.as_secs_f32()).exp_m1();
        let p = self.moving_probability;
        let prior = (p * (1.0 - switch) + (1.0 - p) * switch).clamp(1e-6, 1.0 - 1e-6);
        let resting = mixture(self.states[0], self.states[1], p * switch / (1.0 - prior));
        let moving = mixture(self.states[0], self.states[1], p * (1.0 - switch) / prior);
        self.states = [resting, moving];
        self.moving_probability = prior;
        let mut resting_q = Matrix4::zeros();
        resting_q
            .fixed_view_mut::<2, 2>(0, 0)
            .copy_from(&resting_noise);
        resting_q[(2, 2)] = 1e-6;
        resting_q[(3, 3)] = 1e-6;
        MovingPredict::predict(&mut self.states[0], dt, odometry, 0.90, resting_q);
        MovingPredict::predict(&mut self.states[1], dt, odometry, decay, moving_noise);
    }
    pub fn update(&mut self, mut measurement: Gaussian<2>) {
        measurement.covariance *= self.measurement_scale;
        let likelihood = |state: Gaussian<4>| {
            let residual = measurement.mean - state.mean.xy();
            let covariance =
                state.covariance.fixed_view::<2, 2>(0, 0).into_owned() + measurement.covariance;
            covariance
                .cholesky()
                .map(|factor| {
                    -0.5 * residual.dot(&factor.solve(&residual))
                        - factor.l().diagonal().map(|x| x.ln()).sum()
                })
                .unwrap_or(f32::NEG_INFINITY)
        };
        let a = likelihood(self.states[0]) + (1.0 - self.moving_probability).ln();
        let b = likelihood(self.states[1]) + self.moving_probability.ln();
        if a.is_finite() && b.is_finite() {
            self.moving_probability = (1.0 / (1.0 + (a - b).exp())).clamp(1e-6, 1.0 - 1e-6);
        }
        for state in &mut self.states {
            MovingUpdate::update(state, measurement);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector4;
    #[test]
    fn mixture_retains_between_model_uncertainty() {
        let a = Gaussian {
            mean: Vector4::zeros(),
            covariance: Matrix4::identity(),
        };
        let b = Gaussian {
            mean: Vector4::new(2.0, 0.0, 0.0, 0.0),
            ..a
        };
        let mixed = mixture(a, b, 0.5);
        assert_eq!(mixed.mean.x, 1.0);
        assert_eq!(mixed.covariance[(0, 0)], 2.0);
    }
    #[test]
    fn coherent_kick_shifts_probability_and_stationary_measurements_restore_rest() {
        let mut imm = Imm::new(
            Gaussian {
                mean: Vector4::zeros(),
                covariance: Matrix4::identity() * 0.01,
            },
            1.0,
        );
        let q = Matrix4::from_diagonal(&Vector4::new(1e-6, 1e-6, 1e-3, 1e-3));
        for i in 1..=20 {
            imm.predict(
                Duration::from_millis(40),
                Isometry2::identity(),
                1.0,
                q,
                Matrix2::identity() * 1e-7,
            );
            imm.update(Gaussian {
                mean: nalgebra::Vector2::new(i as f32 * 0.04, 0.0),
                covariance: Matrix2::identity() * 1e-4,
            });
        }
        assert!(imm.moving_probability > 0.9);
        assert!((imm.combined().mean.z - 1.0).abs() < 0.2);
        for _ in 0..50 {
            imm.predict(
                Duration::from_millis(40),
                Isometry2::identity(),
                1.0,
                q,
                Matrix2::identity() * 1e-7,
            );
            imm.update(Gaussian {
                mean: nalgebra::Vector2::new(0.8, 0.0),
                covariance: Matrix2::identity() * 1e-4,
            });
            assert!(imm.combined().covariance.cholesky().is_some());
        }
        assert!(imm.moving_probability < 0.5);
        assert!(imm.combined().mean.z.abs() < 0.05);
    }
}

fn unit_scale() -> f32 {
    1.0
}
