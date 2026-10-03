//! Student-t-inspired innovation reweighting (Gaussian assumed-density update).
//! A bounded approximation to the latent noise precision update, not a complete
//! Student-t state posterior or variational-Bayes reproduction of a paper.
use nalgebra::{Matrix2, Vector2};
use types::multivariate_normal_distribution::MultivariateNormalDistribution;

pub fn reweight(
    prediction: Vector2<f32>,
    covariance: Matrix2<f32>,
    mut measurement: MultivariateNormalDistribution<2>,
    robustness: f32,
) -> MultivariateNormalDistribution<2> {
    if !robustness.is_finite() || robustness <= 0.0 {
        return measurement;
    }
    let residual = measurement.mean - prediction;
    let Some(factor) = (covariance + measurement.covariance).cholesky() else {
        return measurement;
    };
    let distance = residual.dot(&factor.solve(&residual)).max(0.0);
    // nu ranges from approximately Gaussian to heavy-tailed. Do not sharpen
    // nominal measurements, and cap inflation to retain numerical conditioning.
    let nu = 2.0 + 30.0 * (1.0 - robustness.clamp(0.0, 1.0));
    let inflation = ((nu + distance) / (nu + 2.0)).clamp(1.0, 100.0);
    measurement.covariance *= inflation;
    measurement
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reweighting_preserves_inliers_and_reduces_outlier_influence() {
        let sample = |x| MultivariateNormalDistribution {
            mean: Vector2::new(x, 0.0),
            covariance: Matrix2::identity() * 0.01,
        };
        let p = Matrix2::identity() * 0.01;
        assert_eq!(
            reweight(Vector2::zeros(), p, sample(0.1), 1.0).covariance,
            sample(0.1).covariance
        );
        let outlier = reweight(Vector2::zeros(), p, sample(1.0), 1.0);
        assert!(outlier.covariance[(0, 0)] > 0.1);
        assert_eq!(outlier.mean, sample(1.0).mean);
        assert_eq!(
            reweight(Vector2::zeros(), p, sample(1.0), 0.0).covariance,
            sample(1.0).covariance
        );
        assert!(outlier.covariance.cholesky().is_some());
    }
}
