//! Conditional PDA for moving tracks with ambiguous, unclaimed measurements.
//! Posterior moment matching retains association uncertainty. Existing hard
//! assignment controls existence/confirmation; this is not the full PKF paper.
use crate::hypothesis::{BallHypothesis, BallMode};
use nalgebra::{Matrix4, Vector4};
use ros_z::time::Time;
use types::multivariate_normal_distribution::MultivariateNormalDistribution as Gaussian;

pub fn update(
    hypothesis: &mut BallHypothesis,
    time: Time,
    selected: Gaussian<2>,
    bonus: f32,
    alternatives: &[(Gaussian<2>, f32)],
) {
    if alternatives.len() < 2
        || !matches!(hypothesis.mode, BallMode::Moving(_))
        || time <= hypothesis.last_seen
    {
        hypothesis.update(time, selected, bonus);
        return;
    }
    let mut posteriors = Vec::with_capacity(alternatives.len());
    let total: f32 = alternatives.iter().map(|(_, w)| w).sum();
    for &(measurement, weight) in alternatives {
        let mut posterior = hypothesis.clone();
        posterior.update(time, measurement, bonus);
        let BallMode::Moving(state) = posterior.mode else {
            unreachable!("moving update retains mode")
        };
        posteriors.push((state, weight / total));
    }
    let mean = posteriors
        .iter()
        .fold(Vector4::zeros(), |sum, (state, w)| sum + state.mean * *w);
    let covariance = posteriors.iter().fold(Matrix4::zeros(), |sum, (state, w)| {
        let residual = state.mean - mean;
        sum + (state.covariance + residual * residual.transpose()) * *w
    });
    hypothesis.update(time, selected, bonus);
    hypothesis.mode = BallMode::Moving(Gaussian { mean, covariance });
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix2, Vector2};
    #[test]
    fn symmetric_ambiguity_preserves_mean_and_accounts_for_spread() {
        let prior = BallHypothesis::new(
            Gaussian {
                mean: Vector4::zeros(),
                covariance: Matrix4::identity(),
            },
            Time::zero(),
        );
        let measurement = |x| Gaussian {
            mean: Vector2::new(x, 0.0),
            covariance: Matrix2::identity() * 0.01,
        };
        let mut hard = prior.clone();
        let mut soft = prior.clone();
        let time = Time::from_nanos(40_000_000);
        hard.update(time, measurement(0.5), 1.0);
        update(
            &mut soft,
            time,
            measurement(0.5),
            1.0,
            &[(measurement(-0.5), 0.5), (measurement(0.5), 0.5)],
        );
        assert!(soft.position().position.x().abs() < 1e-6);
        assert!(soft.position_covariance()[(0, 0)] > hard.position_covariance()[(0, 0)] + 0.2);
        assert_eq!(soft.validity, hard.validity);
        assert_eq!(soft.last_seen, hard.last_seen);
        assert!(soft.position_covariance().cholesky().is_some());
    }
}
