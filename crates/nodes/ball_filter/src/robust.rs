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
    // The Student-t latent precision increases for ordinary small innovations,
    // and decreases for outliers. Inlier sharpening is essential here: the
    // association gate excludes innovations beyond the inflation threshold.
    // A bounded degrees-of-freedom range retains numerical conditioning.
    let nu = 2.0 + 30.0 * (1.0 - robustness.clamp(0.0, 1.0));
    let precision_scale = ((nu + distance) / (nu + 2.0)).clamp(nu / (nu + 2.0), 100.0);
    let inflation = 1.0 + robustness.clamp(0.0, 1.0) * (precision_scale - 1.0);
    measurement.covariance *= inflation;
    measurement
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reweighting_sharpens_inliers_and_reduces_outlier_influence() {
        let sample = |x| MultivariateNormalDistribution {
            mean: Vector2::new(x, 0.0),
            covariance: Matrix2::identity() * 0.01,
        };
        let p = Matrix2::identity() * 0.01;
        assert!(
            reweight(Vector2::zeros(), p, sample(0.1), 1.0).covariance[(0, 0)]
                < sample(0.1).covariance[(0, 0)]
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
    #[test]
    fn accepted_innovation_changes_update_and_disabled_mode_is_exact() {
        // Association uses prediction covariance and accepts d <= 0.25.
        let covariance = Matrix2::identity();
        let measurement = MultivariateNormalDistribution {
            mean: Vector2::new(0.4, 0.0),
            covariance: Matrix2::identity(),
        };
        let association_distance = measurement.mean.dot(&measurement.mean);
        assert!(association_distance < 0.25);
        let weighted = reweight(Vector2::zeros(), covariance, measurement, 1.0);
        let gain = (covariance + weighted.covariance)
            .cholesky()
            .unwrap()
            .solve(&measurement.mean);
        let baseline_gain = (covariance + measurement.covariance)
            .cholesky()
            .unwrap()
            .solve(&measurement.mean);
        assert!(gain.x > baseline_gain.x);
        assert!(weighted.covariance.cholesky().is_some());
        assert_eq!(
            reweight(Vector2::zeros(), covariance, measurement, 0.0),
            measurement
        );
    }
}
