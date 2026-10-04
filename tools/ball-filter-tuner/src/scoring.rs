use crate::recording::{Cycle, Recording, Reference};
use ball_filter::tracker::Tracker;
use color_eyre::{Result, eyre::ensure};
use coordinate_systems::{Field, Ground};
use linear_algebra::{Isometry2, Point2, Point3};
use ros_z::time::Time;
use serde::Serialize;
use types::{ball_position::BallPosition, parameters::BallFilterParameters};

pub const MISSING_PENALTY_MULTIPLIER: f64 = 1.25;
pub const OUT_OF_FIELD_DECAY_METRES: f64 = 0.3;
pub const CLOSE_RANGE_LOSS_WEIGHT: f64 = 4.0;
/// Association radius used by the user-approved correct-ball availability guards.
pub const CORRECT_TRACK_RADIUS_METRES: f64 = 0.5;

/// Stored with every report so scores from different objectives are not confused.
#[derive(Serialize)]
pub struct Objective {
    pub version: &'static str,
    pub position_loss: &'static str,
    pub missing_loss: &'static str,
    pub false_track_loss: &'static str,
    pub normalization: &'static str,
    pub reference_weight: &'static str,
    pub out_of_field_decay_metres: f64,
    pub close_range_loss: &'static str,
    pub close_range_weight: f64,
    pub motion_reference: &'static str,
}

pub const OBJECTIVE: Objective = Objective {
    version: "single_ball_close_accuracy_v8",
    position_loss: "p^2 * d^2 / (p^2 + d^2); d = distance to the single labelled ball",
    missing_loss: "1.25 * p^2",
    false_track_loss: "p^2",
    normalization: "all-scene weighted mean plus close_range_weight times the separately normalized close-range mean; p = penalty_metres",
    reference_weight: "exp(-distance_outside_field_metres / out_of_field_decay_metres); distance outside Field rectangle expanded by ball radius; absent or unknown Field pose weight=1",
    out_of_field_decay_metres: OUT_OF_FIELD_DECAY_METRES,
    close_range_loss: "Add a separately time-normalized spatial/missing loss for truth within 1 m in Ground, without field-boundary downweighting. Missing costs more than any finite position error. Per-recording close-range RMSE may increase by at most 0.01 m and RMS/mean absolute spatial lag by at most 0.04 s. Aggregate accuracy and false-track time may not worsen. Correct-ball unavailable time (missing or error greater than 0.5 m), its close-range subset and its longest uninterrupted gap must not worsen per recording or in aggregate; raw missing time remains diagnostic. False-track guards remain strict per recording. Signed mean lag is diagnostic only because opposing errors can cancel.",
    close_range_weight: CLOSE_RANGE_LOSS_WEIGHT,
    motion_reference: "Use timestamp-matched simulation/ball_ground_truth_field for motion derivatives when standard simulator labels are selected; otherwise derive Field positions from the selected reference and recorded pose. Ground position/close-range scoring remains unchanged.",
};

#[derive(Default, Debug, Serialize)]
pub struct Score {
    pub loss: f64,
    pub position_rmse_metres: Option<f64>,
    pub labelled_seconds: f64,
    pub weighted_seconds: f64,
    pub out_of_field_seconds: f64,
    pub unlabelled_seconds: f64,
    pub present_seconds: f64,
    pub missing_seconds: f64,
    /// Present truth but selected output is farther than 0.5 m away.
    pub wrong_track_seconds: f64,
    /// Missing output or output farther than 0.5 m from present truth.
    pub correct_track_missing_seconds: f64,
    pub longest_correct_track_gap_seconds: f64,
    /// Includes initial acquisition and unavailable field transforms, not just lost tracks.
    pub missing_runs: u64,
    pub longest_missing_seconds: f64,
    pub close_range_position_rmse_metres: Option<f64>,
    pub close_range_present_seconds: f64,
    pub close_range_missing_seconds: f64,
    pub close_range_correct_track_missing_seconds: f64,
    pub close_range_loss: Option<f64>,
    /// Correct selected-output time; missing output contributes zero success.
    pub close_range_within_10cm_seconds: f64,
    pub close_range_within_20cm_seconds: f64,
    /// Signed along the ball's motion: negative means the estimate is behind.
    pub along_motion_error_metres: Option<f64>,
    /// Equivalent spatial lag, positive behind. Not measured processing latency.
    pub motion_lag_seconds: Option<f64>,
    /// RMS equivalent spatial lag: opposing lead/lag errors cannot cancel.
    pub motion_lag_rms_seconds: Option<f64>,
    /// Mean absolute spatial lag; unlike abs(mean lag), lead/lag cannot cancel.
    pub motion_lag_absolute_seconds: Option<f64>,
    pub moving_reference_seconds: f64,
    pub absent_seconds: f64,
    pub false_track_seconds: f64,
    pub missing_transform_seconds: f64,
    #[serde(skip)]
    squared_error: f64,
    #[serde(skip)]
    current_missing_seconds: f64,
    #[serde(skip)]
    current_correct_track_gap_seconds: f64,
    #[serde(skip)]
    close_range_squared_error: f64,
    #[serde(skip)]
    along_motion_error_integral: f64,
    #[serde(skip)]
    motion_lag_integral: f64,
    #[serde(skip)]
    motion_lag_squared_integral: f64,
    #[serde(skip)]
    motion_lag_absolute_integral: f64,
    #[serde(skip)]
    close_loss_integral: f64,
    #[serde(skip)]
    close_loss_seconds: f64,
}

/// Eligibility constraint, separate from the spatial objective. Compare against
/// the immutable baseline on the same training recordings, never held-out data.
/// Correct-ball unavailable durations include out-of-field frames and initial
/// acquisition. Removing an already wrong output does not reduce availability.
pub fn preserves_baseline_continuity(candidate: &Score, baseline: &Score) -> bool {
    if !baseline.labelled_seconds.is_finite() || baseline.labelled_seconds < 0.0 {
        return false;
    }
    // Time sums can differ by roundoff when different frame subsets have the
    // same duration. This is about 5.5e-11 seconds for a 240-second dataset, not
    // an allowed extra missing frame or a tunable behavioral margin.
    let roundoff = 1024.0 * f64::EPSILON * baseline.labelled_seconds.max(1.0);
    [
        (
            candidate.correct_track_missing_seconds,
            baseline.correct_track_missing_seconds,
        ),
        (
            candidate.close_range_correct_track_missing_seconds,
            baseline.close_range_correct_track_missing_seconds,
        ),
        (
            candidate.longest_correct_track_gap_seconds,
            baseline.longest_correct_track_gap_seconds,
        ),
    ]
    .into_iter()
    .all(|(candidate, baseline)| {
        candidate.is_finite()
            && baseline.is_finite()
            && candidate >= 0.0
            && baseline >= 0.0
            && (candidate <= baseline || candidate - baseline <= roundoff)
    })
}

/// A cheaper global fit cannot buy worse near-ball accuracy or motion tracking.
/// Missing diagnostics are only acceptable when the baseline also lacks them.
pub fn preserves_baseline_quality(candidate: &Score, baseline: &Score) -> bool {
    preserves_quality_with_margins(candidate, baseline, 0.0, 0.0)
}

/// User-approved per-clip tolerance; aggregate accuracy remains non-increasing.
pub fn preserves_recording_quality(candidate: &Score, baseline: &Score) -> bool {
    preserves_quality_with_margins(candidate, baseline, 0.01, 0.04)
}

fn preserves_quality_with_margins(
    candidate: &Score,
    baseline: &Score,
    close: f64,
    lag: f64,
) -> bool {
    fn non_increasing(candidate: Option<f64>, baseline: Option<f64>, margin: f64) -> bool {
        match (candidate, baseline) {
            (Some(candidate), Some(baseline)) => {
                candidate.is_finite()
                    && baseline.is_finite()
                    && candidate >= 0.0
                    && baseline >= 0.0
                    && candidate <= baseline + margin + 1024.0 * f64::EPSILON * baseline.max(1.0)
            }
            (None, None) => true,
            (Some(candidate), None) => candidate.is_finite() && candidate >= 0.0,
            (None, Some(_)) => false,
        }
    }
    preserves_baseline_continuity(candidate, baseline)
        && non_increasing(
            Some(candidate.false_track_seconds),
            Some(baseline.false_track_seconds),
            0.0,
        )
        && non_increasing(
            candidate.close_range_position_rmse_metres,
            baseline.close_range_position_rmse_metres,
            close,
        )
        && non_increasing(
            candidate.motion_lag_rms_seconds,
            baseline.motion_lag_rms_seconds,
            lag,
        )
        && non_increasing(
            candidate.motion_lag_absolute_seconds,
            baseline.motion_lag_absolute_seconds,
            lag,
        )
}

/// Continuous distance to feasibility for population exploration, never used as
/// an alternative acceptance criterion. Exact guards above still select winners.
pub fn quality_violation(candidate: &Score, baseline: &Score, per_recording: bool) -> f64 {
    let pairs = [
        (
            Some(candidate.correct_track_missing_seconds),
            Some(baseline.correct_track_missing_seconds),
        ),
        (
            Some(candidate.close_range_correct_track_missing_seconds),
            Some(baseline.close_range_correct_track_missing_seconds),
        ),
        (
            Some(candidate.false_track_seconds),
            Some(baseline.false_track_seconds),
        ),
        (
            Some(candidate.longest_correct_track_gap_seconds),
            Some(baseline.longest_correct_track_gap_seconds),
        ),
        (
            candidate.close_range_position_rmse_metres,
            baseline.close_range_position_rmse_metres,
        ),
        (
            candidate.motion_lag_rms_seconds,
            baseline.motion_lag_rms_seconds,
        ),
        (
            candidate.motion_lag_absolute_seconds,
            baseline.motion_lag_absolute_seconds,
        ),
    ];
    pairs
        .into_iter()
        .enumerate()
        .map(
            |(index, (candidate, baseline))| match (candidate, baseline) {
                (Some(c), Some(b)) if c.is_finite() && b.is_finite() => {
                    let margin = if per_recording {
                        match index {
                            4 => 0.01,
                            5 | 6 => 0.04,
                            _ => 0.0,
                        }
                    } else {
                        0.0
                    };
                    ((c - b - margin) / b.max(0.02)).max(0.0)
                }
                (None, None) | (Some(_), None) => 0.0,
                _ => f64::INFINITY,
            },
        )
        .sum()
}

fn close_reference(cycle: &Cycle) -> Option<Point2<Ground>> {
    let truth = match cycle.reference.as_ref()? {
        Reference::Ground(points) => match points.as_slice() {
            [point] => point.xy(),
            _ => return None,
        },
        Reference::Field(points) => match points.as_slice() {
            [point] => cycle.ground_to_field?.inverse() * point.xy(),
            _ => return None,
        },
    };
    (truth.coords().norm_squared() <= 1.0).then_some(truth)
}

impl Score {
    fn observe_cycle(
        &mut self,
        cycle: &Cycle,
        estimate: Option<BallPosition<Ground>>,
        penalty: f64,
    ) {
        let loss_before = self.loss;
        match &cycle.reference {
            Some(Reference::Ground(reference)) => {
                self.observe(Some(reference), estimate, cycle.seconds, penalty)
            }
            Some(Reference::Field(reference)) => self.observe_field(
                reference,
                estimate,
                cycle.ground_to_field,
                cycle.seconds,
                penalty,
            ),
            None => self.observe::<Ground>(None, estimate, cycle.seconds, penalty),
        }
        if cycle.reference.is_some() {
            let distance = reference_distance_outside_field(cycle);
            let weight = (-distance / OUT_OF_FIELD_DECAY_METRES).exp();
            self.weighted_seconds += cycle.seconds * weight;
            if distance > 0.0 {
                self.out_of_field_seconds += cycle.seconds;
            }
            // Only the optimization objective is weighted. Raw spatial errors,
            // missing durations and uninterrupted gap lengths remain unchanged.
            self.loss = loss_before + (self.loss - loss_before) * weight;
        }
        // Normalize this independently of far-ball/empty-scene duration. Sparse
        // kick opportunities must not disappear in a long recording's average.
        if let Some(truth) = close_reference(cycle) {
            let cap = penalty.powi(2);
            let loss = estimate.map_or(MISSING_PENALTY_MULTIPLIER * cap, |ball| {
                let squared = f64::from((ball.position - truth).norm_squared());
                cap * squared / (cap + squared)
            });
            self.close_loss_integral += cycle.seconds * loss;
            self.close_loss_seconds += cycle.seconds;
        }
    }

    fn observe_diagnostics(
        &mut self,
        cycle: &Cycle,
        estimate: Option<BallPosition<Ground>>,
        previous: &mut Option<(Time, Point2<Field>)>,
    ) {
        // Diagnose a ball the robot could kick now, within 1 m of its Ground origin.
        let mut close_squared = None::<f64>;
        let mut close_present = false;
        let mut observe_ground = |truth: Point2<Ground>| {
            if truth.coords().norm_squared() <= 1.0 {
                close_present = true;
                if let Some(estimate) = estimate {
                    let squared = f64::from((truth - estimate.position).norm_squared());
                    close_squared = Some(close_squared.map_or(squared, |old| old.min(squared)));
                }
            }
        };
        let single_field = match &cycle.reference {
            Some(Reference::Ground(points)) => {
                for point in points {
                    observe_ground(point.xy());
                }
                match points.as_slice() {
                    [point] => cycle.ground_to_field.map(|pose| pose * point.xy()),
                    _ => None,
                }
            }
            Some(Reference::Field(points)) => {
                if let Some(pose) = cycle.ground_to_field {
                    let inverse = pose.inverse();
                    for point in points {
                        observe_ground(inverse * point.xy());
                    }
                }
                match points.as_slice() {
                    [point] => Some(point.xy()),
                    _ => None,
                }
            }
            None => None,
        };
        if close_present {
            if close_squared.is_none_or(|squared| {
                !squared.is_finite() || squared > CORRECT_TRACK_RADIUS_METRES.powi(2)
            }) {
                self.close_range_correct_track_missing_seconds += cycle.seconds;
            }
            self.close_range_present_seconds += cycle.seconds;
            if let Some(squared) = close_squared {
                self.close_range_squared_error += squared * cycle.seconds;
                if squared <= 0.01 {
                    self.close_range_within_10cm_seconds += cycle.seconds;
                }
                if squared <= 0.04 {
                    self.close_range_within_20cm_seconds += cycle.seconds;
                }
            } else {
                self.close_range_missing_seconds += cycle.seconds;
            }
        }
        // Never infer identities between multiple balls. Field coordinates are
        // essential: differences in Ground include robot translation/rotation.
        let single_field = match &cycle.motion_reference {
            Some(points) => match points.as_slice() {
                [point] => Some(point.xy()),
                _ => None,
            },
            None => single_field,
        };
        let Some(truth) = single_field else {
            *previous = None;
            return;
        };
        let old = previous.replace((cycle.time, truth));
        let Some((time, position)) = old else { return };
        let dt = (cycle.time.as_nanos() - time.as_nanos()) as f64 * 1e-9;
        if !(0.0 < dt && dt <= 0.1) {
            return;
        }
        let velocity = (truth - position) / dt as f32;
        let speed = velocity.norm();
        // Exclude stationary numerical jitter and implausible discontinuities.
        if !(0.5..=15.0).contains(&speed) {
            return;
        }
        if let Some(estimate) = estimate.zip(cycle.ground_to_field).map(|(b, p)| p * b) {
            let along = f64::from((estimate.position - truth).dot(&velocity) / speed);
            self.along_motion_error_integral += along * cycle.seconds;
            self.motion_lag_integral += -along / f64::from(speed) * cycle.seconds;
            self.motion_lag_squared_integral += (along / f64::from(speed)).powi(2) * cycle.seconds;
            self.motion_lag_absolute_integral += (along / f64::from(speed)).abs() * cycle.seconds;
            self.moving_reference_seconds += cycle.seconds;
        }
    }

    fn observe_field(
        &mut self,
        reference: &[Point3<Field>],
        estimate: Option<BallPosition<Ground>>,
        pose: Option<Isometry2<Ground, Field>>,
        seconds: f64,
        penalty: f64,
    ) {
        if pose.is_none() {
            self.missing_transform_seconds += seconds;
        }
        if reference.is_empty() {
            // A false track is still a false track if geometry is missing.
            self.observe::<Ground>(Some(&[]), estimate, seconds, penalty);
        } else {
            self.observe(
                Some(reference),
                estimate.and_then(|ball| pose.map(|pose| pose * ball)),
                seconds,
                penalty,
            );
        }
    }

    fn observe<Frame>(
        &mut self,
        reference: Option<&[Point3<Frame>]>,
        estimate: Option<BallPosition<Frame>>,
        seconds: f64,
        penalty: f64,
    ) {
        let Some(reference) = reference else {
            self.unlabelled_seconds += seconds;
            self.current_missing_seconds = 0.0;
            self.current_correct_track_gap_seconds = 0.0;
            return;
        };
        self.labelled_seconds += seconds;
        if reference.is_empty() || estimate.is_some() {
            self.current_missing_seconds = 0.0;
        }
        let correct_track_missing = reference.first().is_some_and(|truth| {
            estimate.as_ref().is_none_or(|ball| {
                let squared = f64::from((truth.xy() - ball.position).norm_squared());
                !squared.is_finite() || squared > CORRECT_TRACK_RADIUS_METRES.powi(2)
            })
        });
        if correct_track_missing {
            self.correct_track_missing_seconds += seconds;
            self.current_correct_track_gap_seconds += seconds;
            self.longest_correct_track_gap_seconds = self
                .longest_correct_track_gap_seconds
                .max(self.current_correct_track_gap_seconds);
            if estimate.is_some() {
                self.wrong_track_seconds += seconds;
            }
        } else {
            self.current_correct_track_gap_seconds = 0.0;
        }
        match (reference.first(), estimate) {
            (Some(truth), Some(estimate)) => {
                self.present_seconds += seconds;
                // Optimization recordings have exactly one target when present.
                let squared = f64::from((truth.xy() - estimate.position).norm_squared());
                self.squared_error += seconds * squared;
                // Keep the raw RMSE above for honesty about spatial accuracy.
                // Bound only the optimization loss, retaining a gradient even
                // for distant predictions. Dropping a difficult track must not
                // beat any finite position error at that instant.
                let cap = penalty.powi(2);
                self.loss += seconds * cap * (squared / (cap + squared));
            }
            (Some(_), None) => {
                self.present_seconds += seconds;
                self.missing_seconds += seconds;
                if self.current_missing_seconds == 0.0 && seconds > 0.0 {
                    self.missing_runs += 1;
                }
                self.current_missing_seconds += seconds;
                self.longest_missing_seconds = self
                    .longest_missing_seconds
                    .max(self.current_missing_seconds);
                self.loss += seconds * MISSING_PENALTY_MULTIPLIER * penalty.powi(2);
            }
            (None, Some(_)) => {
                self.absent_seconds += seconds;
                self.false_track_seconds += seconds;
                self.loss += seconds * penalty.powi(2);
            }
            (None, None) => {
                self.absent_seconds += seconds;
            }
        }
    }
}

/// Scenario likelihood comes from reference truth, never a candidate estimate.
/// Expanding the rectangle by ball radius keeps a straddling ball at full weight.
/// Ground-only labels without a Field transform cannot establish field bounds.
fn reference_distance_outside_field(cycle: &Cycle) -> f64 {
    let point = match &cycle.reference {
        Some(Reference::Field(points)) => points.first().map(|point| point.xy()),
        Some(Reference::Ground(points)) => points
            .first()
            .zip(cycle.ground_to_field)
            .map(|(point, pose)| pose * point.xy()),
        None => None,
    };
    let Some(point) = point else { return 0.0 };
    let dimensions = &cycle.dimensions;
    let outside_x = (point.x().abs() - dimensions.length / 2.0).max(0.0);
    let outside_y = (point.y().abs() - dimensions.width / 2.0).max(0.0);
    f64::from((outside_x.hypot(outside_y) - dimensions.ball_radius).max(0.0))
}

/// Verify replay against live outputs before trying any candidates. Missing input,
/// parameter mismatch and scheduling drift must not silently change the experiment.
pub fn verify(recording: &Recording, parameters: &BallFilterParameters) -> Result<()> {
    let mut tracker = Tracker::default();
    for cycle in &recording.cycles {
        for input in &cycle.inputs {
            tracker.advance_with_obstacles(
                input.time,
                input.odometry,
                input.detections.as_deref(),
                input.camera.as_ref(),
                input.obstacles.as_ref(),
                parameters,
                &cycle.dimensions,
            )?;
        }
        let replay = tracker.finish_with_field_pose(
            cycle.time,
            parameters,
            &cycle.dimensions,
            cycle.field_prior_pose,
        );
        let matches = match (replay, cycle.recorded_estimate) {
            (None, None) => true,
            (Some(a), Some(b)) => {
                (a.position - b.position).norm() < 1e-4
                    && (a.velocity - b.velocity).norm() < 1e-4
                    && a.last_seen == b.last_seen
            }
            _ => false,
        };
        ensure!(
            matches,
            "replay differs from live filter at {:?} in {}; verify baseline parameters and capture completeness",
            cycle.time,
            recording.path
        );
    }
    Ok(())
}

pub fn evaluate(
    recordings: &[Recording],
    parameters: &BallFilterParameters,
    penalty: f64,
) -> Result<Score> {
    let mut score = Score::default();
    for recording in recordings {
        // Separate recordings do not establish a continuous observation gap.
        score.current_missing_seconds = 0.0;
        score.current_correct_track_gap_seconds = 0.0;
        let mut previous_reference = None;
        let mut tracker = Tracker::default();
        for cycle in &recording.cycles {
            if let Some(reference) = &cycle.reference {
                reference.validate_for_optimization()?;
            }
            for input in &cycle.inputs {
                tracker.advance_with_obstacles(
                    input.time,
                    input.odometry,
                    input.detections.as_deref(),
                    input.camera.as_ref(),
                    input.obstacles.as_ref(),
                    parameters,
                    &cycle.dimensions,
                )?;
            }
            let estimate = tracker.finish_with_field_pose(
                cycle.time,
                parameters,
                &cycle.dimensions,
                cycle.field_prior_pose,
            );
            score.observe_diagnostics(cycle, estimate, &mut previous_reference);
            score.observe_cycle(cycle, estimate, penalty);
        }
    }
    score.finish();
    Ok(score)
}

pub fn export_frames(
    recording: &Recording,
    parameters: &BallFilterParameters,
    path: &std::path::Path,
) -> Result<()> {
    use std::io::Write;
    let mut writer = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut tracker = Tracker::default();
    for cycle in &recording.cycles {
        let mut percepts = Vec::new();
        for input in &cycle.inputs {
            percepts.extend(tracker.advance_with_obstacles(
                input.time,
                input.odometry,
                input.detections.as_deref(),
                input.camera.as_ref(),
                input.obstacles.as_ref(),
                parameters,
                &cycle.dimensions,
            )?);
        }
        let estimate = tracker.finish_with_field_pose(
            cycle.time,
            parameters,
            &cycle.dimensions,
            cycle.field_prior_pose,
        );
        let truth = match &cycle.reference {
            Some(Reference::Ground(points)) => points.first().map(|p| p.xy()),
            Some(Reference::Field(points)) => points
                .first()
                .zip(cycle.ground_to_field)
                .map(|(p, pose)| pose.inverse() * p.xy()),
            None => None,
        };
        let row = serde_json::json!({
            "time": cycle.time.as_nanos(), "seconds": cycle.seconds,
            "truth": truth, "estimate": estimate,
            "motion_truth": cycle.motion_reference,
            "ground_to_field": cycle.ground_to_field,
            "odometry": cycle.inputs.last().map(|input| input.odometry),
            "percepts": percepts, "hypotheses": tracker.filter.hypotheses,
        });
        serde_json::to_writer(&mut writer, &row)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

impl Score {
    fn finish(&mut self) {
        let score = self;
        score.loss /= score.weighted_seconds;
        score.close_range_loss = (score.close_loss_seconds > 0.0)
            .then(|| score.close_loss_integral / score.close_loss_seconds);
        score.loss += CLOSE_RANGE_LOSS_WEIGHT * score.close_range_loss.unwrap_or(0.0);
        let matched = score.present_seconds - score.missing_seconds;
        score.position_rmse_metres =
            (matched > 0.0).then(|| (score.squared_error / matched).sqrt());
        let close_matched = score.close_range_present_seconds - score.close_range_missing_seconds;
        score.close_range_position_rmse_metres =
            (close_matched > 0.0).then(|| (score.close_range_squared_error / close_matched).sqrt());
        score.along_motion_error_metres = (score.moving_reference_seconds > 0.0)
            .then(|| score.along_motion_error_integral / score.moving_reference_seconds);
        score.motion_lag_seconds = (score.moving_reference_seconds > 0.0)
            .then(|| score.motion_lag_integral / score.moving_reference_seconds);
        score.motion_lag_rms_seconds = (score.moving_reference_seconds > 0.0)
            .then(|| (score.motion_lag_squared_integral / score.moving_reference_seconds).sqrt());
        score.motion_lag_absolute_seconds = (score.moving_reference_seconds > 0.0)
            .then(|| score.motion_lag_absolute_integral / score.moving_reference_seconds);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use linear_algebra::{Vector2, point};
    use ros_z::time::Time;

    fn continuity_baseline() -> Score {
        Score {
            loss: 1.0,
            labelled_seconds: 240.0,
            correct_track_missing_seconds: 2.0,
            close_range_correct_track_missing_seconds: 0.4,
            longest_correct_track_gap_seconds: 1.2,
            false_track_seconds: 3.0,
            ..Default::default()
        }
    }

    #[test]
    fn close_availability_counts_wrong_and_missing_outputs_equally() {
        let cycle = diagnostic_cycle(0, 0.5, 0.0);
        let mut score = Score::default();
        let mut previous = None;
        for position in [Some(2.0), None, Some(0.6)] {
            let estimate = position.map(|x| BallPosition {
                position: point![x, 0.0],
                velocity: Vector2::zeros(),
                last_seen: Time::zero(),
            });
            score.observe_diagnostics(&cycle, estimate, &mut previous);
        }
        assert_eq!(
            score.close_range_correct_track_missing_seconds,
            2.0 * cycle.seconds
        );
        assert_eq!(score.close_range_missing_seconds, cycle.seconds);
    }

    #[test]
    fn removing_wrong_outputs_is_allowed_but_losing_correct_outputs_is_not() {
        let baseline = continuity_baseline();
        let mut candidate = continuity_baseline();
        candidate.missing_seconds = 100.0;
        candidate.close_range_missing_seconds = 50.0;
        candidate.longest_missing_seconds = 40.0;
        assert!(preserves_baseline_continuity(&candidate, &baseline));
        assert_eq!(quality_violation(&candidate, &baseline, true), 0.0);
        candidate.correct_track_missing_seconds += 0.002;
        assert!(!preserves_baseline_continuity(&candidate, &baseline));
        assert!(quality_violation(&candidate, &baseline, true) > 0.0);
    }

    fn accuracy_baseline() -> Score {
        Score {
            close_range_position_rmse_metres: Some(0.42),
            motion_lag_seconds: Some(0.43),
            motion_lag_absolute_seconds: Some(0.43),
            motion_lag_rms_seconds: Some(0.5),
            ..continuity_baseline()
        }
    }

    #[test]
    fn cancelling_signed_errors_do_not_block_better_absolute_and_rms_lag() {
        let mut baseline = accuracy_baseline();
        baseline.motion_lag_seconds = Some(0.0);
        let mut candidate = accuracy_baseline();
        candidate.motion_lag_seconds = Some(0.1);
        candidate.motion_lag_absolute_seconds = Some(0.1);
        candidate.motion_lag_rms_seconds = Some(0.2);
        assert!(preserves_baseline_quality(&candidate, &baseline));
        candidate.false_track_seconds = baseline.false_track_seconds + 0.04;
        assert!(!preserves_baseline_quality(&candidate, &baseline));
    }

    #[test]
    fn wrong_output_is_not_correct_ball_retention() {
        let truth = [linear_algebra::point![<Ground>, 0.0, 0.0, 0.1]];
        let estimate = |x| BallPosition {
            position: linear_algebra::point![x, 0.0],
            velocity: linear_algebra::Vector2::zeros(),
            last_seen: Time::zero(),
        };
        let mut score = Score::default();
        score.observe(Some(&truth), Some(estimate(2.0)), 1.0, 2.0);
        score.observe(Some(&truth), None, 2.0, 2.0);
        score.observe(Some(&truth), Some(estimate(0.1)), 1.0, 2.0);
        score.observe(Some(&truth), None, 1.0, 2.0);
        assert_eq!(score.wrong_track_seconds, 1.0);
        assert_eq!(score.missing_seconds, 3.0);
        assert_eq!(score.correct_track_missing_seconds, 4.0);
        assert_eq!(score.longest_correct_track_gap_seconds, 3.0);
        assert_eq!(score.longest_missing_seconds, 2.0);
    }

    #[test]
    fn recording_tolerances_are_bounded_and_do_not_relax_aggregate_or_retention() {
        let baseline = accuracy_baseline();
        let mut candidate = accuracy_baseline();
        candidate.close_range_position_rmse_metres =
            baseline.close_range_position_rmse_metres.map(|x| x + 0.01);
        candidate.motion_lag_rms_seconds = baseline.motion_lag_rms_seconds.map(|x| x + 0.04);
        candidate.motion_lag_absolute_seconds =
            baseline.motion_lag_absolute_seconds.map(|x| x + 0.04);
        assert!(preserves_recording_quality(&candidate, &baseline));
        assert!(!preserves_baseline_quality(&candidate, &baseline));
        for metric in 0..5 {
            let mut worse = accuracy_baseline();
            match metric {
                0 => {
                    worse.close_range_position_rmse_metres = baseline
                        .close_range_position_rmse_metres
                        .map(|x| x + 0.010001)
                }
                1 => {
                    worse.motion_lag_rms_seconds =
                        baseline.motion_lag_rms_seconds.map(|x| x + 0.040001)
                }
                2 => {
                    worse.motion_lag_absolute_seconds =
                        baseline.motion_lag_absolute_seconds.map(|x| x + 0.040001)
                }
                3 => worse.correct_track_missing_seconds += 0.002,
                _ => worse.false_track_seconds += 0.002,
            }
            assert!(!preserves_recording_quality(&worse, &baseline));
            assert!(quality_violation(&worse, &baseline, true) > 0.0);
        }
    }

    #[test]
    fn lower_global_loss_cannot_buy_worse_close_accuracy_or_lag() {
        let baseline = accuracy_baseline();
        assert!(preserves_baseline_quality(&baseline, &baseline));
        for metric in 0..3 {
            let mut candidate = accuracy_baseline();
            candidate.loss = 0.0;
            candidate.correct_track_missing_seconds = 0.0;
            candidate.false_track_seconds = 0.0;
            match metric {
                0 => candidate.close_range_position_rmse_metres = Some(1.10),
                1 => candidate.motion_lag_absolute_seconds = Some(0.70),
                2 => candidate.motion_lag_rms_seconds = Some(0.70),
                _ => unreachable!(),
            }
            assert!(!preserves_baseline_quality(&candidate, &baseline));
        }
        let mut candidate = accuracy_baseline();
        candidate.close_range_position_rmse_metres = Some(0.10);
        candidate.motion_lag_seconds = Some(0.1);
        candidate.motion_lag_rms_seconds = Some(0.2);
        assert!(preserves_baseline_quality(&candidate, &baseline));
    }

    #[test]
    fn missing_or_invalid_accuracy_metrics_cannot_bypass_quality_guards() {
        let baseline = accuracy_baseline();
        for invalid in [None, Some(f64::NAN), Some(f64::INFINITY), Some(-0.01)] {
            let mut candidate = accuracy_baseline();
            candidate.close_range_position_rmse_metres = invalid;
            assert!(!preserves_baseline_quality(&candidate, &baseline));
            candidate = accuracy_baseline();
            candidate.motion_lag_rms_seconds = invalid;
            assert!(!preserves_baseline_quality(&candidate, &baseline));
        }
        // Recordings with no close/moving observations have no comparison to
        // enforce. Acquiring a previously completely missing ball is allowed.
        let absent = continuity_baseline();
        assert!(preserves_baseline_quality(&absent, &absent));
        assert!(preserves_baseline_quality(&baseline, &absent));
    }

    #[test]
    fn long_far_ball_intervals_cannot_dilute_the_close_loss_component() {
        let close = diagnostic_cycle(0, 0.5, 0.0);
        let close_estimate = BallPosition {
            position: point![0.7, 0.0],
            velocity: Vector2::zeros(),
            last_seen: Time::zero(),
        };
        let mut short = Score::default();
        short.observe_cycle(&close, Some(close_estimate), 2.0);
        short.finish();
        let mut long = Score::default();
        long.observe_cycle(&close, Some(close_estimate), 2.0);
        let mut far = diagnostic_cycle(50, 3.0, 0.0);
        far.seconds = 300.0;
        long.observe_cycle(
            &far,
            Some(BallPosition {
                position: point![3.0, 0.0],
                velocity: Vector2::zeros(),
                last_seen: Time::zero(),
            }),
            2.0,
        );
        long.finish();
        assert_eq!(short.close_range_loss, long.close_range_loss);
        assert!(long.loss >= CLOSE_RANGE_LOSS_WEIGHT * short.close_range_loss.unwrap());
        let mut missing = Score::default();
        missing.observe_cycle(&close, None, 2.0);
        missing.finish();
        assert!(missing.close_range_loss > short.close_range_loss);
    }

    #[test]
    fn close_term_uses_truth_distance_and_survives_field_downweighting() {
        let mut score = Score::default();
        let far_truth = diagnostic_cycle(0, 3.0, 0.0);
        score.observe_cycle(
            &far_truth,
            Some(BallPosition {
                position: point![0.1, 0.0],
                velocity: Vector2::zeros(),
                last_seen: Time::zero(),
            }),
            2.0,
        );
        assert_eq!(score.close_loss_seconds, 0.0);
        let outside = diagnostic_cycle(50, 10.5, 10.0);
        score.observe_cycle(&outside, None, 2.0);
        assert_eq!(score.close_loss_seconds, outside.seconds);
        assert_eq!(score.close_loss_integral, outside.seconds * 5.0);
    }

    #[test]
    fn stationary_field_truth_does_not_inherit_motion_from_pose_timing() {
        let mut score = Score::default();
        let mut previous = None;
        for (time, robot) in [(0, 0.0), (50, 0.2), (100, -0.1)] {
            let mut cycle = diagnostic_cycle(time, 1.0, robot);
            // A stale pose changes the reconstructed world coordinate despite
            // the physical ball remaining stationary in its timestamped label.
            cycle.reference = Some(Reference::Ground(vec![linear_algebra::point![
                1.0, 0.0, 0.1
            ]]));
            cycle.motion_reference = Some(vec![linear_algebra::point![1.0, 0.0, 0.1]]);
            score.observe_diagnostics(
                &cycle,
                Some(BallPosition {
                    position: point![1.0, 0.0],
                    velocity: Vector2::zeros(),
                    last_seen: cycle.time,
                }),
                &mut previous,
            );
        }
        assert_eq!(score.moving_reference_seconds, 0.0);
        assert!((score.close_range_present_seconds - 0.15).abs() < 1e-12);
    }

    #[test]
    fn rms_lag_prevents_opposing_errors_from_cancelling() {
        let mut score = Score::default();
        let mut previous = Some((Time::zero(), point![0.5, 0.0]));
        for (time, truth, estimate) in [(50, 0.6, 0.4), (100, 0.7, 0.9)] {
            score.observe_diagnostics(
                &diagnostic_cycle(time, truth, 0.0),
                Some(BallPosition {
                    position: point![estimate, 0.0],
                    velocity: Vector2::zeros(),
                    last_seen: Time::zero(),
                }),
                &mut previous,
            );
        }
        assert!((score.motion_lag_integral / score.moving_reference_seconds).abs() < 1e-6);
        assert!(
            (score.motion_lag_squared_integral / score.moving_reference_seconds).sqrt() > 0.099
        );
    }

    #[test]
    fn continuity_accepts_baseline_and_improvements_with_only_roundoff_tolerance() {
        let baseline = continuity_baseline();
        assert!(preserves_baseline_continuity(&baseline, &baseline));
        let mut candidate = continuity_baseline();
        candidate.correct_track_missing_seconds = 1.0;
        candidate.longest_correct_track_gap_seconds = 0.5;
        candidate.close_range_correct_track_missing_seconds = 0.3;
        assert!(preserves_baseline_continuity(&candidate, &baseline));

        let mut baseline = continuity_baseline();
        baseline.close_range_correct_track_missing_seconds = 0.3;
        candidate.close_range_correct_track_missing_seconds = 0.1 + 0.2;
        assert!(preserves_baseline_continuity(&candidate, &baseline));
        candidate.close_range_correct_track_missing_seconds = 0.3 + 1e-9;
        assert!(!preserves_baseline_continuity(&candidate, &baseline));
    }

    #[test]
    fn each_continuity_limit_independently_rejects_a_regression() {
        let baseline = continuity_baseline();
        for metric in 0..3 {
            let mut candidate = continuity_baseline();
            match metric {
                0 => candidate.correct_track_missing_seconds += 0.002,
                1 => candidate.close_range_correct_track_missing_seconds += 0.002,
                2 => candidate.longest_correct_track_gap_seconds += 0.002,
                _ => unreachable!(),
            }
            assert!(!preserves_baseline_continuity(&candidate, &baseline));
        }
    }

    #[test]
    fn false_track_reduction_and_lower_loss_cannot_pay_for_worse_continuity() {
        let baseline = continuity_baseline();
        let mut candidate = continuity_baseline();
        candidate.false_track_seconds = 0.0;
        candidate.loss = 0.0;
        candidate.correct_track_missing_seconds += 0.002;
        assert!(!preserves_baseline_continuity(&candidate, &baseline));
    }

    #[test]
    fn nonfinite_or_negative_continuity_metrics_are_ineligible() {
        let baseline = continuity_baseline();
        for invalid in [f64::NAN, f64::INFINITY, -0.1] {
            let mut candidate = continuity_baseline();
            candidate.longest_correct_track_gap_seconds = invalid;
            assert!(!preserves_baseline_continuity(&candidate, &baseline));
        }
    }

    fn diagnostic_cycle(time_ms: i64, x: f32, robot_x: f32) -> Cycle {
        Cycle {
            inputs: Vec::new(),
            time: Time::from_nanos(time_ms * 1_000_000),
            dimensions: Default::default(),
            reference: Some(Reference::Field(vec![point![x, 0.0, 0.1]])),
            motion_reference: None,
            ground_to_field: Some(Isometry2::from(linear_algebra::vector![robot_x, 0.0])),
            recorded_estimate: None,
            field_prior_pose: None,
            seconds: 0.05,
        }
    }

    #[test]
    fn field_weight_decays_with_whole_ball_distance_and_uses_field_coordinates() {
        let mut cycle = diagnostic_cycle(0, 0.0, 0.0);
        cycle.dimensions = types::field_dimensions::FieldDimensions::SPL_2025;
        let edge = cycle.dimensions.length / 2.0 + cycle.dimensions.ball_radius;
        let mut previous_weight = 1.0;
        for distance in [0.0_f32, 0.3, 1.0, 2.0] {
            cycle.reference = Some(Reference::Field(vec![point![edge + distance, 0.0, 0.1]]));
            let measured = reference_distance_outside_field(&cycle);
            assert!((measured - f64::from(distance)).abs() < 1e-6);
            let weight = (-measured / OUT_OF_FIELD_DECAY_METRES).exp();
            assert!(weight <= previous_weight);
            previous_weight = weight;
        }
        cycle.reference = Some(Reference::Field(vec![point![edge - 0.01, 0.0, 0.1]]));
        assert_eq!(reference_distance_outside_field(&cycle), 0.0);
        cycle.reference = Some(Reference::Field(vec![point![
            cycle.dimensions.length / 2.0 + 0.3,
            cycle.dimensions.width / 2.0 + 0.4,
            0.1
        ]]));
        assert!(
            (reference_distance_outside_field(&cycle)
                - f64::from(0.5 - cycle.dimensions.ball_radius))
            .abs()
                < 1e-6
        );

        // A distant ball in Ground can still be inside the field when the robot
        // is translated and rotated. Geometry must not follow robot coordinates.
        let pose = Isometry2::<Ground, Field>::from_parts(linear_algebra::vector![8.0, -5.0], 1.2);
        for truth in [point![0.0, 0.0], point![edge + 0.3, 0.0]] {
            let ground = pose.inverse() * truth;
            cycle.reference = Some(Reference::Ground(vec![point![ground.x(), ground.y(), 0.1]]));
            cycle.ground_to_field = Some(pose);
            let expected = (truth.x().abs() - edge).max(0.0);
            let actual = reference_distance_outside_field(&cycle);
            // Inverting and reapplying a float32 pose several metres away
            // introduces a few micrometres of roundoff.
            assert!(
                (actual - f64::from(expected)).abs() < 5e-6,
                "distance {actual}, expected {expected}"
            );
        }
        cycle.ground_to_field = None;
        assert_eq!(reference_distance_outside_field(&cycle), 0.0);
    }

    #[test]
    fn outside_weight_changes_only_loss_and_its_denominator_not_raw_metrics_or_gaps() {
        let mut inside = diagnostic_cycle(0, 1.0, 0.0);
        inside.dimensions = types::field_dimensions::FieldDimensions::SPL_2025;
        inside.seconds = 1.0;
        let edge = inside.dimensions.length / 2.0 + inside.dimensions.ball_radius;
        let mut outside = diagnostic_cycle(0, edge + 0.3, 0.0);
        outside.dimensions = inside.dimensions;
        outside.seconds = 1.0;
        let weight =
            (-reference_distance_outside_field(&outside) / OUT_OF_FIELD_DECAY_METRES).exp();
        for error in [None, Some(0.0), Some(0.5)] {
            let estimate = |truth: f32| {
                error.map(|offset| BallPosition::<Ground> {
                    position: point![truth + offset, 0.0],
                    velocity: Vector2::zeros(),
                    last_seen: Time::zero(),
                })
            };
            let mut a = Score::default();
            let mut b = Score::default();
            a.observe_cycle(&inside, estimate(1.0), 2.0);
            b.observe_cycle(&outside, estimate(edge + 0.3), 2.0);
            assert!((b.loss - a.loss * weight).abs() < 1e-6);
            assert!((b.squared_error - a.squared_error).abs() < 1e-6);
            assert_eq!(a.missing_seconds, b.missing_seconds);
            assert_eq!(a.labelled_seconds, b.labelled_seconds);
            assert_eq!(b.out_of_field_seconds, 1.0);
            assert_eq!(b.weighted_seconds, weight);
        }
        let mut gaps = Score::default();
        gaps.observe_cycle(&inside, None, 2.0);
        gaps.observe_cycle(&outside, None, 2.0);
        assert_eq!(gaps.missing_runs, 1);
        assert_eq!(gaps.longest_missing_seconds, 2.0);
        assert_eq!(gaps.missing_seconds, 2.0);

        outside.reference = Some(Reference::Field(Vec::new()));
        let mut absent = Score::default();
        absent.observe_cycle(&outside, None, 2.0);
        assert_eq!(absent.weighted_seconds, 1.0);
        assert_eq!(absent.out_of_field_seconds, 0.0);
        outside.reference = None;
        absent.observe_cycle(&outside, None, 2.0);
        assert_eq!(absent.weighted_seconds, 1.0);
        assert_eq!(absent.unlabelled_seconds, 1.0);
    }

    #[test]
    fn spatial_lag_has_signed_direction_and_ignores_robot_translation() {
        let estimate = BallPosition::<Ground> {
            position: point![0.2, 0.0],
            velocity: Vector2::zeros(),
            last_seen: Time::zero(),
        };
        let mut score = Score::default();
        let mut previous = None;
        score.observe_diagnostics(&diagnostic_cycle(0, 0.5, 0.0), None, &mut previous);
        // Ball moves +0.1 m in 50 ms = 2 m/s; robot itself moved +0.2 m.
        // Estimate world x=.4 trails truth x=.6 by .2 m = .1 s spatial lag.
        score.observe_diagnostics(
            &diagnostic_cycle(50, 0.6, 0.2),
            Some(estimate),
            &mut previous,
        );
        assert!((score.along_motion_error_integral / 0.05 + 0.2).abs() < 1e-6);
        assert!((score.motion_lag_integral / 0.05 - 0.1).abs() < 1e-6);
        assert_eq!(score.moving_reference_seconds, 0.05);
        assert!((score.close_range_squared_error / 0.05 - 0.04).abs() < 1e-6);
        assert_eq!(score.close_range_present_seconds, 0.1);
        assert_eq!(score.close_range_missing_seconds, 0.05);

        let mut ahead = Score::default();
        let mut previous = Some((Time::zero(), point![0.5, 0.0]));
        ahead.observe_diagnostics(
            &diagnostic_cycle(50, 0.6, 0.6),
            Some(estimate),
            &mut previous,
        );
        assert!(ahead.along_motion_error_integral > 0.0);
        assert!(ahead.motion_lag_integral < 0.0);
    }

    #[test]
    fn lag_skips_multiple_balls_stationary_balls_and_reference_gaps() {
        let estimate = BallPosition::<Ground> {
            position: point![0.5, 0.0],
            velocity: Vector2::zeros(),
            last_seen: Time::zero(),
        };
        let mut score = Score::default();
        let mut previous = None;
        let mut cycle = diagnostic_cycle(0, 0.5, 0.0);
        score.observe_diagnostics(&cycle, Some(estimate), &mut previous);
        cycle = diagnostic_cycle(50, 0.5, 0.2);
        score.observe_diagnostics(&cycle, Some(estimate), &mut previous);
        cycle.reference = Some(Reference::Field(vec![
            point![0.6, 0.0, 0.1],
            point![2.0, 0.0, 0.1],
        ]));
        score.observe_diagnostics(&cycle, Some(estimate), &mut previous);
        assert!(previous.is_none());
        score.observe_diagnostics(
            &diagnostic_cycle(100, 0.7, 0.2),
            Some(estimate),
            &mut previous,
        );
        score.observe_diagnostics(
            &diagnostic_cycle(500, 1.0, 0.2),
            Some(estimate),
            &mut previous,
        );
        assert_eq!(score.moving_reference_seconds, 0.0);
    }

    #[test]
    fn dropping_a_difficult_prediction_never_improves_instantaneous_loss() {
        let reference = [point![0.0, 0.0, 0.1]];
        let mut missing = Score::default();
        missing.observe::<Ground>(Some(&reference), None, 1.0, 2.0);
        assert_eq!(missing.loss, 5.0);
        let mut previous_loss = -1.0;
        for error in [0.0, 0.5, 2.0, 5.0, 20.0, 1000.0] {
            let estimate = BallPosition::<Ground> {
                position: point![error, 0.0],
                velocity: Vector2::zeros(),
                last_seen: Time::zero(),
            };
            let mut predicted = Score::default();
            predicted.observe(Some(&reference), Some(estimate), 1.0, 2.0);
            assert!(predicted.loss < missing.loss);
            assert!(predicted.loss < 4.0);
            assert!(predicted.loss > previous_loss);
            assert_eq!(predicted.squared_error, f64::from(error * error));
            previous_loss = predicted.loss;
        }
    }

    #[test]
    fn retaining_a_track_after_the_ball_is_absent_is_penalized() {
        let mut missing = Score::default();
        let mut false_track = Score::default();
        let estimate = BallPosition::<Ground> {
            position: point![0.0, 0.0],
            velocity: Vector2::zeros(),
            last_seen: Time::zero(),
        };
        missing.observe::<Ground>(Some(&[]), None, 1.0, 2.0);
        false_track.observe(Some(&[]), Some(estimate), 1.0, 2.0);
        assert_eq!(missing.loss, 0.0);
        assert_eq!(false_track.loss, 4.0);
        assert_eq!(false_track.missing_runs, 0);
    }

    #[test]
    fn gaps_end_at_estimates_absence_or_unknown_labels() {
        let reference = [point![0.0, 0.0, 0.1]];
        let estimate = BallPosition::<Ground> {
            position: point![0.0, 0.0],
            velocity: Vector2::zeros(),
            last_seen: Time::zero(),
        };
        let mut score = Score::default();
        score.observe::<Ground>(Some(&reference), None, 0.0, 2.0);
        assert_eq!(score.missing_runs, 0);
        score.observe::<Ground>(Some(&reference), None, 0.25, 2.0);
        score.observe::<Ground>(Some(&reference), None, 0.75, 2.0);
        assert_eq!(score.missing_runs, 1);
        assert_eq!(score.longest_missing_seconds, 1.0);
        score.observe(Some(&reference), Some(estimate), 0.5, 2.0);
        score.observe::<Ground>(Some(&reference), None, 0.5, 2.0);
        score.observe::<Ground>(Some(&[]), None, 5.0, 2.0);
        score.observe::<Ground>(Some(&reference), None, 0.75, 2.0);
        score.observe::<Ground>(None, None, 5.0, 2.0);
        score.observe::<Ground>(Some(&reference), None, 1.5, 2.0);
        assert_eq!(score.missing_runs, 4);
        assert_eq!(score.longest_missing_seconds, 1.5);
        assert_eq!(score.missing_seconds, 3.75);
    }

    #[test]
    fn spatial_score_uses_the_single_labelled_target() {
        let balls = [point![2.0, 0.0, 0.1]];
        let mut score = Score::default();
        let mut estimate = BallPosition::<Ground> {
            position: point![2.0, 0.0],
            velocity: Vector2::zeros(),
            last_seen: Time::zero(),
        };
        score.observe(Some(&balls), Some(estimate), 1.0, 2.0);
        assert_eq!(score.loss, 0.0);
        estimate.position = point![0.0, 0.0];
        score.observe(Some(&balls), Some(estimate), 1.0, 2.0);
        assert_eq!(score.loss, 2.0);
    }
    #[test]
    fn field_score_includes_ground_pose_and_never_hides_false_tracks_when_pose_is_missing() {
        let ball = BallPosition::<Ground> {
            position: point![1.0, 0.0],
            velocity: Vector2::zeros(),
            last_seen: Time::zero(),
        };
        let pose = Isometry2::<Ground, Field>::from(linear_algebra::vector![2.0, 1.0]);
        let mut score = Score::default();
        score.observe_field(&[point![3.0, 1.0, 0.105]], Some(ball), Some(pose), 1.0, 2.0);
        assert!(score.loss < 1e-9);
        score.observe_field(&[point![3.0, 1.0, 0.105]], Some(ball), None, 2.0, 2.0);
        score.observe_field(&[], Some(ball), None, 3.0, 2.0);
        assert_eq!(score.loss, 22.0);
        assert_eq!(score.missing_seconds, 2.0);
        assert_eq!(score.false_track_seconds, 3.0);
        assert_eq!(score.missing_transform_seconds, 5.0);
    }

    #[test]
    fn missing_and_false_tracks_are_penalized_unknown_is_not_absent() {
        let mut score = Score::default();
        let ball: BallPosition<Ground> = BallPosition {
            position: point![1.0, 0.0],
            velocity: Vector2::zeros(),
            last_seen: Time::zero(),
        };
        score.observe(None, Some(ball), 2.0, 2.0);
        score.observe::<Ground>(Some(&[point![1.0, 0.0, 0.105]]), None, 3.0, 2.0);
        score.observe(Some(&[]), Some(ball), 4.0, 2.0);
        assert_eq!(score.loss, 31.0);
        assert_eq!(score.labelled_seconds, 7.0);
        assert_eq!(score.unlabelled_seconds, 2.0);
        assert_eq!(score.missing_seconds, 3.0);
        assert_eq!(score.false_track_seconds, 4.0);
    }
}

impl From<&Score> for types::ball_filter_tuning::Metrics {
    fn from(score: &Score) -> Self {
        Self {
            loss: score.loss,
            position_rmse_metres: score.position_rmse_metres,
            missing_seconds: score.missing_seconds,
            missing_runs: score.missing_runs,
            longest_missing_seconds: score.longest_missing_seconds,
            close_range_position_rmse_metres: score.close_range_position_rmse_metres,
            close_range_present_seconds: score.close_range_present_seconds,
            close_range_missing_seconds: score.close_range_missing_seconds,
            along_motion_error_metres: score.along_motion_error_metres,
            motion_lag_seconds: score.motion_lag_seconds,
            moving_reference_seconds: score.moving_reference_seconds,
            false_track_seconds: score.false_track_seconds,
            missing_transform_seconds: score.missing_transform_seconds,
        }
    }
}
