//! Synthetic detector output at the production vision/filter boundary.
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
    time::Duration,
};

use color_eyre::Result;
use coordinate_systems::{Field, Ground, Pixel};
use geometry::rectangle::Rectangle;
use linear_algebra::{Isometry2, Point2, Point3, point};
use projection::{Projection, camera_matrix::CameraMatrix};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};
use ros_z::{
    prelude::*,
    time::{Clock, Time},
};
use ros_z_streams::{AnnouncingPublisher, CreateAnnouncingPublisher};
use serde::{Deserialize, Serialize};
use tokio::{
    runtime::Handle,
    task::{JoinHandle, JoinSet},
};
use types::{
    ball_position::BallPosition,
    bounding_box::BoundingBox,
    object_detection::{Object, RobocupObjectLabel},
    time_wrapper::TimeWrapper,
};

#[derive(Debug, Clone, Serialize, Deserialize, Message)]
#[serde(default, deny_unknown_fields)]
pub struct Parameters {
    /// Logical seconds between camera frames, independent of rendering speed.
    pub frame_period: f64,
    /// Exposure-to-detector-delivery delay; announcements still happen at exposure.
    pub delivery_delay_seconds: f64,
    /// Uniform delay variation on either side of delivery_delay_seconds.
    pub delivery_jitter_seconds: f64,
    /// Deterministic missed-frame bursts after eight visible close-ball frames.
    /// Empty disables this challenge. Physical balls and ground truth are retained.
    pub close_dropout_pattern: Vec<u32>,
    /// Independent standard deviation of bounding-box center x/y, in pixels.
    pub center_noise_pixels: f32,
    /// Fixed detector-center bias per recording, in pixels.
    pub center_bias_pixels: [f32; 2],
    /// Independent probability of suppressing true detections on a frame.
    pub dropout_probability: f32,
    /// Probability of starting a correlated run of missed true detections.
    pub dropout_burst_probability: f32,
    pub dropout_burst_frames: u32,
    /// Frames for which a false detection persists at one fixed field location.
    /// A value of one preserves independent uniformly sampled image outliers.
    pub false_positive_burst_frames: u32,
    /// Probability of starting an additional false-detection burst per frame.
    pub false_positive_probability: f32,
    pub detection_confidence: f32,
    /// Pixel radius of the additional false detection.
    pub false_positive_radius: f32,
    pub seed: u64,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            frame_period: 0.04,
            delivery_delay_seconds: 0.0,
            delivery_jitter_seconds: 0.0,
            close_dropout_pattern: Vec::new(),
            center_noise_pixels: 1.0,
            center_bias_pixels: [0.0; 2],
            dropout_probability: 0.0,
            dropout_burst_probability: 0.0,
            dropout_burst_frames: 1,
            false_positive_burst_frames: 1,
            false_positive_probability: 0.04,
            detection_confidence: 0.9,
            false_positive_radius: 8.0,
            seed: 42,
        }
    }
}

impl Parameters {
    pub fn validate(&self) -> Result<(), String> {
        if !self.delivery_delay_seconds.is_finite()
            || !self.delivery_jitter_seconds.is_finite()
            || self.delivery_jitter_seconds < 0.0
            || self.delivery_delay_seconds < self.delivery_jitter_seconds
            || self.delivery_delay_seconds + self.delivery_jitter_seconds > 0.5
        {
            return Err(
                "detector delay must satisfy 0 <= jitter <= delay and delay+jitter <= 0.5 seconds"
                    .into(),
            );
        }
        if self
            .close_dropout_pattern
            .iter()
            .any(|frames| !(1..=8).contains(frames))
        {
            return Err("close dropout lengths must be between one and eight frames".into());
        }
        if !self.frame_period.is_finite() || self.frame_period < 0.002 {
            return Err(
                "ball_perception.frame_period must be finite and at least 0.002 seconds".into(),
            );
        }
        if !self.center_noise_pixels.is_finite() || self.center_noise_pixels < 0.0 {
            return Err(
                "ball_perception.center_noise_pixels must be finite and nonnegative".into(),
            );
        }
        if !self.center_bias_pixels.iter().all(|v| v.is_finite())
            || self.dropout_burst_frames == 0
            || self.false_positive_burst_frames == 0
        {
            return Err("detector biases must be finite and burst lengths must be positive".into());
        }
        for (name, value) in [
            ("dropout_probability", self.dropout_probability),
            ("dropout_burst_probability", self.dropout_burst_probability),
            (
                "false_positive_probability",
                self.false_positive_probability,
            ),
            ("detection_confidence", self.detection_confidence),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(format!(
                    "ball_perception.{name} must be between zero and one"
                ));
            }
        }
        if !self.false_positive_radius.is_finite() || self.false_positive_radius <= 0.0 {
            return Err("ball_perception.false_positive_radius must be finite and positive".into());
        }
        Ok(())
    }
}

pub(crate) const FALSE_DETECTIONS_TOPIC: &str = "simulation/false_ball_detections";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Message)]
pub enum FalseDetectionProjection {
    GroundPlane,
    CameraRay,
}

/// Simulator-only provenance; never used as a production percept or reference ball.
/// Correlated artifacts mark their fixed physical anchor, not their noisy image estimate.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, Message)]
pub struct FalseDetectionMarker {
    pub position: Point3<Field>,
    pub projection: FalseDetectionProjection,
}

struct DetectionFrame {
    detections: Vec<Object<RobocupObjectLabel>>,
    false_markers: Vec<FalseDetectionMarker>,
}

fn false_detection_marker(
    camera: &CameraMatrix,
    ground_to_field: Isometry2<Ground, Field>,
    pixel: Point2<Pixel>,
    radius: f32,
) -> FalseDetectionMarker {
    let (ground, projection) = match camera.pixel_to_ground_with_z(pixel, radius) {
        Ok(position)
            if position
                .inner
                .coords
                .iter()
                .all(|coordinate| coordinate.is_finite()) =>
        {
            (
                nalgebra::point![position.x(), position.y(), radius],
                FalseDetectionProjection::GroundPlane,
            )
        }
        _ => {
            // Above-horizon false pixels have no ground intersection. Keep them
            // visible as explicitly diagnostic markers three metres down the ray.
            let bearing = camera.bearing(pixel).inner.normalize();
            (
                camera.ground_to_camera.inverse().inner * nalgebra::Point3::from(bearing * 3.0),
                FalseDetectionProjection::CameraRay,
            )
        }
    };
    let field = ground_to_field * point![ground.x, ground.y];
    FalseDetectionMarker {
        position: point![field.x(), field.y(), ground.z],
        projection,
    }
}

pub struct Detector {
    rng: ChaCha8Rng,
    seed: u64,
    dropout_remaining: u32,
    false_remaining: u32,
    false_center: Point2<Pixel>,
    false_anchor: Option<Point2<Field>>,
    close_dropouts: crate::ball_approach::BriefDropouts,
}

/// Upright physical occluder in the same ground frame as the camera and balls.
pub struct Occluder {
    pub center: Point3<Ground>,
    pub radius: f32,
    pub height: f32,
}

pub(crate) fn occludes(camera: &CameraMatrix, ball: Point3<Ground>, obstacle: &Occluder) -> bool {
    let origin = camera.ground_to_camera.inverse().inner.translation.vector;
    let delta = ball.inner.coords - origin;
    let relative = origin - obstacle.center.inner.coords;
    let a = delta.xy().norm_squared();
    let c = relative.xy().norm_squared() - obstacle.radius.powi(2);
    let (mut enter, mut exit) = (0.0_f32, 1.0_f32);
    if a < 1e-12 {
        if c > 0.0 {
            return false;
        }
    } else {
        let b = relative.xy().dot(&delta.xy());
        let discriminant = b * b - a * c;
        if discriminant < 0.0 {
            return false;
        }
        let root = discriminant.sqrt();
        enter = enter.max((-b - root) / a);
        exit = exit.min((-b + root) / a);
    }
    if delta.z.abs() < 1e-6 {
        if relative.z.abs() > obstacle.height / 2.0 {
            return false;
        }
    } else {
        let low = (-obstacle.height / 2.0 - relative.z) / delta.z;
        let high = (obstacle.height / 2.0 - relative.z) / delta.z;
        enter = enter.max(low.min(high));
        exit = exit.min(low.max(high));
    }
    enter < exit && exit > 0.0 && enter < 1.0
}

impl Detector {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
            seed,
            dropout_remaining: 0,
            false_remaining: 0,
            false_center: Point2::origin(),
            false_anchor: None,
            close_dropouts: Default::default(),
        }
    }

    #[cfg(test)]
    pub fn detect(
        &mut self,
        camera: &CameraMatrix,
        balls: &[Point3<Ground>],
        radius: f32,
        parameters: &Parameters,
    ) -> Vec<Object<RobocupObjectLabel>> {
        self.detect_with_provenance(
            camera,
            Isometry2::identity(),
            balls,
            &[],
            radius,
            parameters,
        )
        .detections
    }

    #[allow(clippy::too_many_arguments)]
    fn detect_with_provenance(
        &mut self,
        camera: &CameraMatrix,
        ground_to_field: Isometry2<Ground, Field>,
        balls: &[Point3<Ground>],
        obstacles: &[Occluder],
        radius: f32,
        parameters: &Parameters,
    ) -> DetectionFrame {
        if self.seed != parameters.seed {
            *self = Self::new(parameters.seed);
        }
        let mut detections = Vec::new();
        let mut false_markers = Vec::new();
        if self.dropout_remaining == 0
            && parameters.dropout_burst_probability > 0.0
            && self.rng.random::<f32>() < parameters.dropout_burst_probability
        {
            self.dropout_remaining = parameters.dropout_burst_frames;
        }
        let close_visible = balls.iter().any(|ball| {
            ball.xy().coords().norm() <= 1.0
                && camera
                    .ground_with_z_to_pixel(ball.xy(), ball.z())
                    .is_ok_and(|pixel| in_image(camera, pixel))
                && !obstacles
                    .iter()
                    .any(|obstacle| occludes(camera, *ball, obstacle))
        });
        let scheduled_drop = self
            .close_dropouts
            .step(close_visible, &parameters.close_dropout_pattern);
        let drop_frame = scheduled_drop
            || self.dropout_remaining > 0
            || (parameters.dropout_probability > 0.0
                && self.rng.random::<f32>() < parameters.dropout_probability);
        self.dropout_remaining = self.dropout_remaining.saturating_sub(1);
        for ball in balls.iter().filter(|ball| {
            !drop_frame
                && !obstacles
                    .iter()
                    .any(|obstacle| occludes(camera, **ball, obstacle))
        }) {
            let Ok(center) = camera.ground_with_z_to_pixel(ball.xy(), ball.z()) else {
                continue;
            };
            if !in_image(camera, center) {
                continue;
            }
            let in_camera = camera.ground_to_camera * *ball;
            // Pinhole approximation for a sphere, using its actual depth (including flight).
            let pixel_radius =
                radius * camera.intrinsics.focals.x.min(camera.intrinsics.focals.y) / in_camera.z();
            if !pixel_radius.is_finite() || pixel_radius <= 0.0 {
                continue;
            }
            let dx: f32 = StandardNormal.sample(&mut self.rng);
            let dy: f32 = StandardNormal.sample(&mut self.rng);
            let noisy = point![
                center.x() + dx * parameters.center_noise_pixels + parameters.center_bias_pixels[0],
                center.y() + dy * parameters.center_noise_pixels + parameters.center_bias_pixels[1]
            ];
            if in_image(camera, noisy) {
                detections.push(detection(
                    noisy,
                    pixel_radius,
                    parameters.detection_confidence,
                ));
            }
        }
        if self.false_remaining == 0
            && self.rng.random::<f32>() < parameters.false_positive_probability
        {
            self.false_anchor = None;
            if parameters.false_positive_burst_frames == 1 {
                // Preserve independent image outliers, including above-horizon pixels.
                self.false_center = point![
                    self.rng.random::<f32>() * camera.image_size.x(),
                    self.rng.random::<f32>() * camera.image_size.y()
                ];
                self.false_remaining = 1;
            } else {
                // A shadow or field marking stays put as the robot walks. Sample
                // boundedly so an upward-looking camera cannot stall simulation.
                for _ in 0..32 {
                    let pixel = point![
                        self.rng.random::<f32>() * camera.image_size.x(),
                        self.rng.random::<f32>() * camera.image_size.y()
                    ];
                    let Ok(ground) = camera.pixel_to_ground_with_z(pixel, radius) else {
                        continue;
                    };
                    if !ground.inner.coords.iter().all(|value| value.is_finite())
                        || ground.inner.coords.norm() > 15.0
                        || balls
                            .iter()
                            .any(|ball| (ball.xy() - ground).norm() < 2.0 * radius + 0.1)
                    {
                        continue;
                    }
                    let point = point![ground.x(), ground.y(), radius];
                    if obstacles
                        .iter()
                        .any(|obstacle| occludes(camera, point, obstacle))
                    {
                        continue;
                    }
                    self.false_anchor = Some(ground_to_field * ground);
                    self.false_remaining = parameters.false_positive_burst_frames;
                    break;
                }
            }
        }
        if self.false_remaining > 0 {
            self.false_remaining -= 1;
            let emission = if let Some(anchor) = self.false_anchor {
                let ground = ground_to_field.inverse() * anchor;
                let point = point![ground.x(), ground.y(), radius];
                let obscured = obstacles
                    .iter()
                    .any(|obstacle| occludes(camera, point, obstacle));
                // If a real ball rolls through the artifact, do not emit a duplicate
                // synthetic ball there. The anchor itself never follows that ball.
                let overlaps_ball = balls
                    .iter()
                    .any(|ball| (ball.xy() - ground).norm() < 2.0 * radius + 0.1);
                camera
                    .ground_with_z_to_pixel(ground, radius)
                    .ok()
                    .filter(|center| in_image(camera, *center) && !obscured && !overlaps_ball)
                    .and_then(|center| {
                        let dx: f32 = StandardNormal.sample(&mut self.rng);
                        let dy: f32 = StandardNormal.sample(&mut self.rng);
                        let noisy = point![
                            center.x()
                                + dx * parameters.center_noise_pixels
                                + parameters.center_bias_pixels[0],
                            center.y()
                                + dy * parameters.center_noise_pixels
                                + parameters.center_bias_pixels[1]
                        ];
                        in_image(camera, noisy).then_some((
                            noisy,
                            FalseDetectionMarker {
                                position: point![anchor.x(), anchor.y(), radius],
                                projection: FalseDetectionProjection::GroundPlane,
                            },
                        ))
                    })
            } else {
                Some((
                    self.false_center,
                    false_detection_marker(camera, ground_to_field, self.false_center, radius),
                ))
            };
            if let Some((pixel, marker)) = emission {
                false_markers.push(marker);
                detections.push(detection(
                    pixel,
                    parameters.false_positive_radius,
                    parameters.detection_confidence,
                ));
            }
        }
        DetectionFrame {
            detections,
            false_markers,
        }
    }
}

pub(crate) fn in_image(camera: &CameraMatrix, point: Point2<Pixel>) -> bool {
    (0.0..camera.image_size.x()).contains(&point.x())
        && (0.0..camera.image_size.y()).contains(&point.y())
}

fn detection(center: Point2<Pixel>, radius: f32, confidence: f32) -> Object<RobocupObjectLabel> {
    Object {
        label: RobocupObjectLabel::Ball,
        bounding_box: BoundingBox {
            area: Rectangle {
                min: point![center.x() - radius, center.y() - radius],
                max: point![center.x() + radius, center.y() + radius],
            },
            confidence,
        },
    }
}

/// Metrics are conditional on an estimate and real ball existing; misses and false tracks
/// are counted separately because conditional position error alone rewards suppression.
#[derive(Debug, Default, Clone, Serialize, Deserialize, Message)]
pub struct Metrics {
    pub time: Time,
    pub matched_samples: u64,
    pub missing_estimates: u64,
    pub estimates_without_ball: u64,
    pub empty_samples: u64,
    pub unmatched_timestamps: u64,
    pub position_error_metres: Option<f64>,
    pub position_rmse_metres: Option<f64>,
    pub field_matched_samples: u64,
    pub missing_field_transforms: u64,
    pub field_transform_age_seconds: Option<f64>,
    pub field_position_error_metres: Option<f64>,
    pub field_position_rmse_metres: Option<f64>,
}

impl Metrics {
    fn observe(
        &mut self,
        time: Time,
        truth: &[Point3<Ground>],
        estimate: Option<BallPosition<Ground>>,
    ) {
        self.time = time;
        self.position_error_metres = None;
        match (truth.is_empty(), estimate) {
            (true, Some(_)) => self.estimates_without_ball += 1,
            (true, None) => self.empty_samples += 1,
            (false, None) => self.missing_estimates += 1,
            (false, Some(estimate)) => {
                let error = truth
                    .iter()
                    .map(|ball| (ball.xy() - estimate.position).norm() as f64)
                    .min_by(f64::total_cmp)
                    .expect("truth is nonempty");
                self.matched_samples += 1;
                let mean_square = self.position_rmse_metres.unwrap_or(0.0).powi(2);
                self.position_error_metres = Some(error);
                self.position_rmse_metres = Some(
                    (mean_square + (error * error - mean_square) / self.matched_samples as f64)
                        .sqrt(),
                );
            }
        }
    }

    fn observe_field(
        &mut self,
        truth: &TruthFrame,
        estimate: Option<BallPosition<Ground>>,
        pose: Option<(Time, Isometry2<Ground, Field>)>,
    ) {
        self.field_position_error_metres = None;
        self.field_transform_age_seconds = None;
        let Some(estimate) = estimate.filter(|_| !truth.balls.is_empty()) else {
            return;
        };
        let Some((stamp, pose)) = pose
            .filter(|(stamp, _)| self.time.duration_since(*stamp) <= Duration::from_millis(100))
        else {
            self.missing_field_transforms += 1;
            return;
        };
        self.field_transform_age_seconds = Some(self.time.duration_since(stamp).as_secs_f64());
        let position = pose * estimate.position;
        let error = truth
            .balls
            .iter()
            .map(|ball| (truth.ground_to_field * ball.xy() - position).norm() as f64)
            .min_by(f64::total_cmp)
            .expect("truth is nonempty");
        self.field_matched_samples += 1;
        let mean_square = self.field_position_rmse_metres.unwrap_or(0.0).powi(2);
        self.field_position_error_metres = Some(error);
        self.field_position_rmse_metres = Some(
            (mean_square + (error * error - mean_square) / self.field_matched_samples as f64)
                .sqrt(),
        );
    }
}

#[derive(Clone)]
struct TruthFrame {
    balls: Vec<Point3<Ground>>,
    ground_to_field: Isometry2<Ground, Field>,
}

type TruthHistory = Arc<Mutex<BTreeMap<Time, TruthFrame>>>;

pub struct PerceptionIo {
    runtime: Handle,
    detector: Detector,
    detections: Arc<AnnouncingPublisher<TimeWrapper<Vec<Object<RobocupObjectLabel>>>>>,
    truth: Publisher<TimeWrapper<Vec<Point3<Ground>>>>,
    field_truth: Publisher<TimeWrapper<Vec<Point3<Field>>>>,
    false_detections: Arc<Publisher<TimeWrapper<Vec<FalseDetectionMarker>>>>,
    history: TruthHistory,
    metrics_task: JoinHandle<()>,
    last_frame: Option<Time>,
    clock: Clock,
    deliveries: JoinSet<Result<()>>,
    delivery_rng: ChaCha8Rng,
    delivery_seed: u64,
}

impl PerceptionIo {
    pub async fn new(node: &Node, runtime: &Handle) -> Result<Self> {
        let detections = node.announcing_publisher("detected_objects").await?;
        let truth = node
            .publisher("simulation/ball_ground_truth")
            .build()
            .await?;
        let field_truth = node
            .publisher("simulation/ball_ground_truth_field")
            .build()
            .await?;
        let false_detections = node.publisher(FALSE_DETECTIONS_TOPIC).build().await?;
        let estimates = node
            .subscriber::<Option<BallPosition<Ground>>>("ball_filter/ball_position")
            .build()
            .await?;
        let metrics_pub = node
            .publisher::<Metrics>("simulation/ball_filter_metrics")
            .build()
            .await?;
        let poses = node
            .subscriber::<Isometry2<Ground, Field>>("ground_to_field")
            .build()
            .await?;
        let history = TruthHistory::default();
        let metric_history = history.clone();
        let metrics_task = runtime.spawn(async move {
            let mut metrics = Metrics::default();
            let mut last_time = None;
            let mut field_poses = BTreeMap::new();
            loop {
                let sample = tokio::select! {
                    pose = poses.recv_with_metadata() => {
                        let Ok(pose) = pose else { break; };
                        field_poses.insert(pose.source_time, pose.message);
                        while field_poses.len() > 10_000 { field_poses.pop_first(); }
                        continue;
                    }
                    sample = estimates.recv_with_metadata() => {
                        let Ok(sample) = sample else { break; };
                        sample
                    }
                };
                if last_time.is_some_and(|last| sample.source_time <= last) {
                    continue;
                }
                last_time = Some(sample.source_time);
                let truth = metric_history
                    .lock()
                    .expect("truth history lock poisoned")
                    .get(&sample.source_time)
                    .cloned();
                if let Some(truth) = truth {
                    metrics.observe(sample.source_time, &truth.balls, sample.message);
                    metrics.observe_field(
                        &truth,
                        sample.message,
                        field_poses
                            .range(..=sample.source_time)
                            .next_back()
                            .map(|(time, pose)| (*time, *pose)),
                    );
                } else {
                    metrics.time = sample.source_time;
                    metrics.position_error_metres = None;
                    metrics.field_position_error_metres = None;
                    metrics.field_transform_age_seconds = None;
                    metrics.unmatched_timestamps += 1;
                }
                if let Err(error) = metrics_pub
                    .publish_with_source_time(&metrics, sample.source_time)
                    .await
                {
                    bevy::log::warn!("cannot publish ball filter metrics: {error}");
                    break;
                }
            }
        });
        Ok(Self {
            runtime: runtime.clone(),
            detector: Detector::new(42),
            detections: Arc::new(detections),
            truth,
            field_truth,
            false_detections: Arc::new(false_detections),
            history,
            metrics_task,
            last_frame: None,
            clock: node.clock().clone(),
            deliveries: JoinSet::new(),
            delivery_rng: ChaCha8Rng::seed_from_u64(42 ^ 0xd311_0e12_u64),
            delivery_seed: 42,
        })
    }

    pub fn publish(
        &mut self,
        time: Time,
        ground_to_field: Isometry2<Ground, Field>,
        camera: &CameraMatrix,
        balls: Vec<Point3<Ground>>,
        radius: f32,
        parameters: &Parameters,
    ) -> Result<()> {
        self.publish_with_occluders(
            time,
            ground_to_field,
            camera,
            balls,
            radius,
            parameters,
            &[],
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn publish_with_occluders(
        &mut self,
        time: Time,
        ground_to_field: Isometry2<Ground, Field>,
        camera: &CameraMatrix,
        balls: Vec<Point3<Ground>>,
        radius: f32,
        parameters: &Parameters,
        obstacles: &[Occluder],
    ) -> Result<()> {
        while let Some(result) = self.deliveries.try_join_next() {
            result??;
        }
        // Keep exact ground truth for delayed filter outputs, before announcing either input.
        {
            let mut history = self.history.lock().expect("truth history lock poisoned");
            history.insert(
                time,
                TruthFrame {
                    balls: balls.clone(),
                    ground_to_field,
                },
            );
            while history.len() > 10_000 {
                history.pop_first();
            }
        }
        let frame_due = self.last_frame.is_none_or(|last| {
            time.duration_since(last).as_secs_f64() + 1e-9 >= parameters.frame_period
        });
        self.runtime.block_on(async {
            // Record reference positions at physics cadence, matching odometry timestamps.
            // An absent message is unknown; an empty vector explicitly means no ball.
            self.truth
                .publish_with_source_time(
                    &TimeWrapper {
                        time,
                        inner: balls.clone(),
                    },
                    time,
                )
                .await?;
            self.field_truth
                .publish_with_source_time(
                    &TimeWrapper {
                        time,
                        inner: balls
                            .iter()
                            .map(|ball| {
                                let field = ground_to_field * ball.xy();
                                point![field.x(), field.y(), ball.z()]
                            })
                            .collect(),
                    },
                    time,
                )
                .await?;
            if frame_due {
                let frame = self.detector.detect_with_provenance(
                    camera,
                    ground_to_field,
                    &balls,
                    obstacles,
                    radius,
                    parameters,
                );
                let false_markers = frame.false_markers;
                if self.delivery_seed != parameters.seed {
                    self.delivery_seed = parameters.seed;
                    self.delivery_rng =
                        ChaCha8Rng::seed_from_u64(parameters.seed ^ 0xd311_0e12_u64);
                }
                let jitter = if parameters.delivery_jitter_seconds > 0.0 {
                    self.delivery_rng.random_range(
                        -parameters.delivery_jitter_seconds..=parameters.delivery_jitter_seconds,
                    )
                } else {
                    0.0
                };
                let due =
                    time + Duration::from_secs_f64(parameters.delivery_delay_seconds + jitter);
                let payload = TimeWrapper {
                    time,
                    inner: frame.detections,
                };
                if due == time {
                    self.detections
                        .announce(time)
                        .await?
                        .publish(&payload)
                        .await?;
                    if !false_markers.is_empty() {
                        self.false_detections
                            .publish_with_source_time(
                                &TimeWrapper {
                                    time,
                                    inner: false_markers,
                                },
                                time,
                            )
                            .await?;
                    }
                } else {
                    let detections = self.detections.clone();
                    let markers = self.false_detections.clone();
                    let clock = self.clock.clone();
                    let (announced, ready) = tokio::sync::oneshot::channel();
                    self.deliveries.spawn_on(
                        async move {
                            let pending = detections.announce(time).await?;
                            let _ = announced.send(());
                            clock.sleep_until(due).await;
                            pending.publish(&payload).await?;
                            if !false_markers.is_empty() {
                                markers
                                    .publish_with_source_time(
                                        &TimeWrapper {
                                            time,
                                            inner: false_markers,
                                        },
                                        time,
                                    )
                                    .await?;
                            }
                            Ok(())
                        },
                        &self.runtime,
                    );
                    // Register the exposure before later odometry can commit its timestamp.
                    ready.await?;
                }
                self.last_frame = Some(time);
            }
            Ok(())
        })
    }

    pub fn brief_dropout_counts(&self) -> &[u64; 8] {
        &self.detector.close_dropouts.bursts
    }
}

impl Drop for PerceptionIo {
    fn drop(&mut self) {
        self.metrics_task.abort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use linear_algebra::{Isometry3, vector};
    use ros_z::{qos::QosDurability, time::Clock};

    fn camera() -> CameraMatrix {
        use std::f32::consts::FRAC_PI_2;
        CameraMatrix::from_normalized_focal_and_center(
            nalgebra::vector![0.5, 0.5],
            nalgebra::point![0.5, 0.5],
            vector![640.0, 544.0],
            Isometry3::identity(),
            Isometry3::identity(),
            Isometry3::wrap(
                nalgebra::Isometry3::rotation(nalgebra::Vector3::y() * -FRAC_PI_2)
                    * nalgebra::Isometry3::rotation(nalgebra::Vector3::x() * FRAC_PI_2)
                    * nalgebra::Isometry3::translation(0.0, 0.0, -0.8),
            ),
        )
    }

    fn clean() -> Parameters {
        Parameters {
            center_noise_pixels: 0.0,
            false_positive_probability: 0.0,
            ..Default::default()
        }
    }

    #[test]
    fn false_diagnostic_preserves_detection_sequence_and_reports_every_emission() {
        let camera = camera();
        let parameters = Parameters {
            false_positive_probability: 1.0,
            false_positive_burst_frames: 3,
            dropout_probability: 0.1,
            center_noise_pixels: 2.0,
            ..clean()
        };
        let mut ordinary = Detector::new(parameters.seed);
        let mut diagnostic = Detector::new(parameters.seed);
        let mut count = 0;
        for _ in 0..200 {
            let expected = ordinary.detect(&camera, &[point![2.0, 0.0, 0.105]], 0.105, &parameters);
            let actual = diagnostic.detect_with_provenance(
                &camera,
                Isometry2::identity(),
                &[point![2.0, 0.0, 0.105]],
                &[],
                0.105,
                &parameters,
            );
            assert_eq!(
                serde_json::to_value(&expected).unwrap(),
                serde_json::to_value(&actual.detections).unwrap()
            );
            for marker in actual.false_markers {
                count += 1;
                assert!(in_image(
                    &camera,
                    actual.detections.last().unwrap().bounding_box.area.center()
                ));
                assert!(
                    marker
                        .position
                        .inner
                        .coords
                        .iter()
                        .all(|value| value.is_finite())
                );
            }
        }
        assert!(
            count > 100,
            "static artifacts should remain visible in this fixed camera"
        );
        // Both streams consumed exactly the same RNG draws, including future frames.
        assert_eq!(ordinary.rng.random::<u64>(), diagnostic.rng.random::<u64>());
    }

    fn static_detector(anchor: Point2<Field>, frames: u32) -> Detector {
        let mut detector = Detector::new(42);
        detector.false_anchor = Some(anchor);
        detector.false_remaining = frames;
        detector
    }

    #[test]
    fn correlated_false_detection_stays_in_field_when_robot_and_camera_move() {
        let original = camera();
        let mut rotated = original.clone();
        rotated.robot_to_head =
            Isometry3::wrap(nalgebra::Isometry3::rotation(nalgebra::Vector3::z() * 0.08));
        rotated.compute_memoized();
        let anchor = point![2.0, 0.0];
        let mut detector = static_detector(anchor, 3);
        let mut previous_pixel: Option<Point2<Pixel>> = None;
        for (camera, pose) in [
            (&original, Isometry2::identity()),
            (&rotated, Isometry2::identity()),
            (&rotated, Isometry2::from_parts(vector![0.2, 0.1], 0.1)),
        ] {
            let frame = detector.detect_with_provenance(camera, pose, &[], &[], 0.105, &clean());
            assert_eq!(frame.detections.len(), 1);
            assert_eq!(frame.false_markers.len(), 1);
            let pixel = frame.detections[0].bounding_box.area.center();
            let ground = camera.pixel_to_ground_with_z(pixel, 0.105).unwrap();
            assert!((pose * ground - anchor).norm() < 1e-4);
            assert_eq!(frame.false_markers[0].position.xy(), anchor);
            if let Some(previous) = previous_pixel {
                assert!(
                    (pixel - previous).norm() > 1.0,
                    "physical artifact must not stick to the image"
                );
            }
            previous_pixel = Some(pixel);
        }
    }

    #[test]
    fn static_artifact_survives_hidden_frames_without_moving_or_emitting_markers() {
        let camera = camera();
        let anchor = point![2.0, 0.0];
        let mut detector = static_detector(anchor, 4);
        let parameters = Parameters {
            dropout_probability: 1.0,
            ..clean()
        };
        let blocker = Occluder {
            center: point![1.0, 0.0, 0.5],
            radius: 0.3,
            height: 1.0,
        };
        let frame = detector.detect_with_provenance(
            &camera,
            Isometry2::identity(),
            &[],
            &[blocker],
            0.105,
            &parameters,
        );
        assert!(frame.detections.is_empty());
        assert!(frame.false_markers.is_empty());
        let away = Isometry2::from_parts(vector![0.0, 0.0], std::f32::consts::PI);
        let frame = detector.detect_with_provenance(&camera, away, &[], &[], 0.105, &parameters);
        assert!(frame.detections.is_empty());
        assert!(frame.false_markers.is_empty());
        // A real ball passing through the artifact does not make a second physical ball.
        let frame = detector.detect_with_provenance(
            &camera,
            Isometry2::identity(),
            &[point![2.0, 0.0, 0.105]],
            &[],
            0.105,
            &parameters,
        );
        assert!(frame.detections.is_empty());
        assert!(frame.false_markers.is_empty());
        let frame = detector.detect_with_provenance(
            &camera,
            Isometry2::identity(),
            &[],
            &[],
            0.105,
            &parameters,
        );
        assert_eq!(frame.detections.len(), 1);
        assert_eq!(frame.false_markers[0].position.xy(), anchor);
        assert_eq!(detector.false_remaining, 0);
        assert!(detector.detect(&camera, &[], 0.105, &parameters).is_empty());
    }

    #[test]
    fn static_artifact_has_noisy_measurements_but_fixed_provenance() {
        let camera = camera();
        let anchor = point![2.0, 0.0];
        let mut detector = static_detector(anchor, 200);
        let parameters = Parameters {
            center_noise_pixels: 2.0,
            ..clean()
        };
        let expected_pixel = camera.ground_with_z_to_pixel(point![anchor.x(), anchor.y()], 0.105);
        let expected_pixel = expected_pixel.unwrap();
        let mut squared_error = 0.0;
        for _ in 0..200 {
            let frame = detector.detect_with_provenance(
                &camera,
                Isometry2::identity(),
                &[],
                &[],
                0.105,
                &parameters,
            );
            assert_eq!(frame.false_markers.len(), frame.detections.len());
            assert_eq!(frame.detections.len(), 1);
            assert_eq!(frame.false_markers[0].position.xy(), anchor);
            squared_error +=
                (frame.detections[0].bounding_box.area.center() - expected_pixel).norm_squared();
        }
        assert!((4.0..12.0).contains(&(squared_error / 200.0)));
    }

    #[test]
    fn static_artifact_sampling_avoids_true_balls_even_when_they_are_missed() {
        let camera = camera();
        let parameters = Parameters {
            false_positive_probability: 1.0,
            false_positive_burst_frames: 3,
            dropout_probability: 1.0,
            ..clean()
        };
        let first = Detector::new(42).detect_with_provenance(
            &camera,
            Isometry2::identity(),
            &[],
            &[],
            0.105,
            &parameters,
        );
        let occupied = first.false_markers[0].position;
        let second = Detector::new(42).detect_with_provenance(
            &camera,
            Isometry2::identity(),
            &[point![occupied.x(), occupied.y(), occupied.z()]],
            &[],
            0.105,
            &parameters,
        );
        assert_eq!(second.false_markers.len(), 1);
        assert!((second.false_markers[0].position.xy() - occupied.xy()).norm() >= 0.31);
    }

    #[test]
    fn independent_image_outliers_keep_the_original_seeded_sampling() {
        let camera = camera();
        let parameters = Parameters {
            false_positive_probability: 1.0,
            ..clean()
        };
        let mut expected_rng = ChaCha8Rng::seed_from_u64(42);
        let _: f32 = expected_rng.random(); // Burst-start probability.
        let expected = point![
            expected_rng.random::<f32>() * 640.0,
            expected_rng.random::<f32>() * 544.0
        ];
        let frame = Detector::new(42).detect_with_provenance(
            &camera,
            Isometry2::identity(),
            &[],
            &[],
            0.105,
            &parameters,
        );
        assert_eq!(frame.detections[0].bounding_box.area.center(), expected);
        assert_eq!(frame.false_markers.len(), 1);
    }

    #[test]
    fn false_marker_uses_ground_plane_or_explicit_above_horizon_ray() {
        let camera = camera();
        let pose = Isometry2::from_parts(vector![2.0, 3.0], std::f32::consts::FRAC_PI_2);
        let ground = point![2.0, 0.2];
        let pixel = camera.ground_with_z_to_pixel(ground, 0.105).unwrap();
        let marker = false_detection_marker(&camera, pose, pixel, 0.105);
        assert_eq!(marker.projection, FalseDetectionProjection::GroundPlane);
        assert!((marker.position.xy() - pose * ground).norm() < 1e-4);
        assert_eq!(marker.position.z(), 0.105);
        let high_pixel = point![320.0, 0.0];
        assert!(camera.pixel_to_ground_with_z(high_pixel, 0.105).is_err());
        let ray = false_detection_marker(&camera, Isometry2::identity(), high_pixel, 0.105);
        assert_eq!(ray.projection, FalseDetectionProjection::CameraRay);
        let camera_origin = camera.ground_to_camera.inverse().inner.translation.vector;
        assert!(((ray.position.inner.coords - camera_origin).norm() - 3.0).abs() < 1e-5);
    }

    #[test]
    fn occlusion_requires_an_obstacle_between_camera_and_ball_at_ray_height() {
        let camera = camera();
        let ball = point![3.0, 0.0, 0.105];
        let mut obstacle = Occluder {
            center: point![1.5, 0.0, 0.425],
            radius: 0.22,
            height: 0.85,
        };
        assert!(occludes(&camera, ball, &obstacle));
        obstacle.center = point![4.0, 0.0, 0.425];
        assert!(!occludes(&camera, ball, &obstacle));
        obstacle.center = point![1.5, 1.0, 0.425];
        assert!(!occludes(&camera, ball, &obstacle));
        obstacle.center = point![1.5, 0.0, 2.0];
        assert!(!occludes(&camera, ball, &obstacle));
    }

    #[test]
    fn clean_detections_round_trip_and_respect_camera_view() {
        let camera = camera();
        let balls = [
            point![2.0, 0.2, 0.105],
            point![-2.0, 0.0, 0.105],
            point![2.0, 20.0, 0.105],
        ];
        let detections = Detector::new(42).detect(&camera, &balls, 0.105, &clean());
        assert_eq!(detections.len(), 1);
        let position = camera
            .pixel_to_ground_with_z(detections[0].bounding_box.area.center(), 0.105)
            .unwrap();
        assert!((position - balls[0].xy()).norm() < 1e-5);
        let airborne = point![2.0, 0.2, 0.5];
        let detection = Detector::new(42).detect(&camera, &[airborne], 0.105, &clean());
        assert!(
            (detection[0].bounding_box.area.center()
                - camera
                    .ground_with_z_to_pixel(airborne.xy(), airborne.z())
                    .unwrap())
            .norm()
                < 1e-4
        );
    }

    #[test]
    fn seeded_noise_is_repeatable_and_has_requested_variance() {
        let camera = camera();
        let ball = point![2.0, 0.0, 0.105];
        let center = camera.ground_with_z_to_pixel(ball.xy(), ball.z()).unwrap();
        let parameters = Parameters {
            center_noise_pixels: 3.0,
            ..clean()
        };
        let mut first = Detector::new(42);
        let mut second = Detector::new(42);
        let mut sum = nalgebra::Vector2::<f64>::zeros();
        let mut squares = nalgebra::Vector2::<f64>::zeros();
        for _ in 0..10_000 {
            let a = first.detect(&camera, &[ball], 0.105, &parameters);
            let b = second.detect(&camera, &[ball], 0.105, &parameters);
            assert_eq!(
                a[0].bounding_box.area.center(),
                b[0].bounding_box.area.center()
            );
            let residual = (a[0].bounding_box.area.center() - center)
                .inner
                .cast::<f64>();
            sum += residual;
            squares += residual.component_mul(&residual);
        }
        assert!((sum / 10_000.0).amax() < 0.1);
        assert!((squares / 10_000.0 - nalgebra::Vector2::repeat(9.0)).amax() < 0.4);
    }

    #[test]
    fn dropout_and_false_detection_bursts_persist_for_the_configured_frames() {
        let camera = camera();
        let ball = point![2.0, 0.0, 0.105];
        let mut detector = Detector::new(42);
        let mut parameters = Parameters {
            dropout_burst_probability: 1.0,
            dropout_burst_frames: 3,
            false_positive_probability: 1.0,
            false_positive_burst_frames: 3,
            ..clean()
        };
        let first = detector.detect(&camera, &[ball], 0.105, &parameters);
        assert_eq!(first.len(), 1); // True ball dropped, false ball remains.
        parameters.dropout_burst_probability = 0.0;
        parameters.false_positive_probability = 0.0;
        for _ in 0..2 {
            let frame = detector.detect(&camera, &[ball], 0.105, &parameters);
            assert_eq!(frame.len(), 1);
            assert_eq!(
                frame[0].bounding_box.area.center(),
                first[0].bounding_box.area.center()
            );
        }
        let recovered = detector.detect(&camera, &[ball], 0.105, &parameters);
        assert_eq!(recovered.len(), 1);
        let projected = camera
            .pixel_to_ground_with_z(recovered[0].bounding_box.area.center(), 0.105)
            .unwrap();
        assert!((projected - ball.xy()).norm() < 1e-5);
    }

    #[test]
    fn constant_center_bias_is_applied_and_invalid_bursts_are_rejected() {
        let camera = camera();
        let ball = point![2.0, 0.0, 0.105];
        let mut parameters = Parameters {
            center_bias_pixels: [3.0, -2.0],
            ..clean()
        };
        let detection = Detector::new(42).detect(&camera, &[ball], 0.105, &parameters);
        let center = camera.ground_with_z_to_pixel(ball.xy(), ball.z()).unwrap();
        assert!(
            (detection[0].bounding_box.area.center() - center - vector![3.0, -2.0]).norm() < 1e-4
        );
        parameters.false_positive_burst_frames = 0;
        assert!(parameters.validate().is_err());
        parameters.false_positive_burst_frames = 1;
        parameters.dropout_probability = f32::NAN;
        assert!(parameters.validate().is_err());
    }

    #[test]
    fn false_detections_also_occur_without_real_balls() {
        let camera = camera();
        let mut detector = Detector::new(42);
        assert!(detector.detect(&camera, &[], 0.105, &clean()).is_empty());
        let parameters = Parameters {
            false_positive_probability: 1.0,
            ..clean()
        };
        let detections = detector.detect(&camera, &[], 0.105, &parameters);
        assert_eq!(detections.len(), 1);
        assert!(in_image(&camera, detections[0].bounding_box.area.center()));
        assert_eq!(detections[0].label, RobocupObjectLabel::Ball);
        assert!(
            detector
                .detect(&camera, &[point![-2.0, 0.0, 0.105]], 0.105, &clean())
                .is_empty()
        );
    }

    #[test]
    fn metrics_include_misses_and_false_tracks() {
        let mut metrics = Metrics::default();
        let truth = [point![2.0, 0.0, 0.105]];
        let estimate = BallPosition {
            position: point![2.3, 0.4],
            velocity: vector![0.0, 0.0],
            last_seen: Time::zero(),
        };
        metrics.observe(Time::zero(), &truth, Some(estimate));
        assert!((metrics.position_rmse_metres.unwrap() - 0.5).abs() < 1e-6);
        metrics.observe(Time::zero(), &truth, None);
        metrics.observe(Time::zero(), &[], Some(estimate));
        metrics.observe(Time::zero(), &[], None);
        assert_eq!(
            (
                metrics.matched_samples,
                metrics.missing_estimates,
                metrics.estimates_without_ball,
                metrics.empty_samples
            ),
            (1, 1, 1, 1)
        );
        assert_eq!(metrics.position_error_metres, None);
    }

    #[test]
    fn field_transform_error_is_separate_from_primary_ball_error() {
        let time = Time::from_nanos(200_000_000);
        let truth = TruthFrame {
            balls: vec![point![2.0, 0.0, 0.105]],
            ground_to_field: Isometry2::identity(),
        };
        let estimate = Some(BallPosition {
            position: point![2.0, 0.0],
            velocity: vector![0.0, 0.0],
            last_seen: time,
        });
        let mut metrics = Metrics::default();
        metrics.observe(time, &truth.balls, estimate);
        let incorrect_pose =
            Isometry2::wrap(nalgebra::Isometry2::new(nalgebra::vector![0.3, 0.4], 0.0));
        metrics.observe_field(&truth, estimate, Some((time, incorrect_pose)));
        assert_eq!(metrics.position_rmse_metres, Some(0.0));
        assert!((metrics.field_position_rmse_metres.unwrap() - 0.5).abs() < 1e-6);
        metrics.observe_field(&truth, estimate, Some((Time::zero(), incorrect_pose)));
        assert_eq!(metrics.missing_field_transforms, 1);
        assert_eq!(metrics.field_position_error_metres, None);
        assert_eq!(metrics.position_rmse_metres, Some(0.0));
    }

    #[test]
    fn rejects_invalid_noise_configuration() {
        assert!(Parameters::default().validate().is_ok());
        for (delay, jitter) in [
            (f64::NAN, 0.0),
            (-0.1, 0.0),
            (0.02, 0.03),
            (0.5, 0.01),
            (0.05, f64::INFINITY),
        ] {
            assert!(
                Parameters {
                    delivery_delay_seconds: delay,
                    delivery_jitter_seconds: jitter,
                    ..clean()
                }
                .validate()
                .is_err()
            );
        }
        assert!(
            Parameters {
                close_dropout_pattern: vec![0],
                ..clean()
            }
            .validate()
            .is_err()
        );
        for value in [-1.0, f32::NAN, f32::INFINITY] {
            assert!(
                Parameters {
                    center_noise_pixels: value,
                    ..clean()
                }
                .validate()
                .is_err()
            );
        }
        assert!(
            Parameters {
                false_positive_probability: 1.1,
                ..clean()
            }
            .validate()
            .is_err()
        );
        assert!(
            Parameters {
                frame_period: 0.0,
                ..clean()
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn production_filter_consumes_announcements_and_metrics_match_capture_time() {
        check_production_filter_delivery(0.0);
    }

    #[test]
    fn delayed_detector_payloads_keep_exposure_geometry_in_production_filter() {
        check_production_filter_delivery(0.05);
    }

    fn check_production_filter_delivery(delay: f64) {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let clock = Clock::logical(Time::zero());
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("tcp/127.0.0.1:{}", listener.local_addr().unwrap().port());
        drop(listener);
        let (
            context,
            node,
            mut perception,
            camera_pub,
            metrics,
            estimates,
            task,
            odometry,
            pose_pub,
        ) = runtime.block_on(async {
            let context = Arc::new(
                ContextBuilder::default()
                    .with_namespace("/ball_perception_test")
                    .with_mode("router")
                    .disable_multicast_scouting()
                    .with_connect_endpoints(std::iter::empty::<&str>())
                    .with_listen_endpoints([endpoint.as_str()])
                    .with_parameter_layers([std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                        .join("../../etc/parameters/base")])
                    .with_clock(clock.clone())
                    .build()
                    .await
                    .unwrap(),
            );
            let node = context.create_node("test_io").build().await.unwrap();
            let perception = PerceptionIo::new(&node, runtime.handle()).await.unwrap();
            let field = node
                .publisher::<types::field_dimensions::FieldDimensions>("field_dimensions")
                .qos(QosProfile {
                    durability: QosDurability::TransientLocal,
                    ..Default::default()
                })
                .build()
                .await
                .unwrap();
            field
                .publish(&types::field_dimensions::FieldDimensions {
                    ball_radius: 0.105,
                    ..types::field_dimensions::FieldDimensions::SPL_2025
                })
                .await
                .unwrap();
            let camera_pub = node
                .publisher::<TimeWrapper<CameraMatrix>>("camera_matrix")
                .build()
                .await
                .unwrap();
            let metrics = node
                .subscriber::<Metrics>("simulation/ball_filter_metrics")
                .cache(1)
                .build()
                .await
                .unwrap();
            let estimates = node
                .subscriber::<Option<BallPosition<Ground>>>("ball_filter/ball_position")
                .cache(1)
                .build()
                .await
                .unwrap();
            let odometry = node
                .announcing_publisher::<linear_algebra::Pose2<coordinate_systems::Odometry>>(
                    "inputs/odometry",
                )
                .await
                .unwrap();
            let pose_pub = node
                .publisher::<Isometry2<Ground, Field>>("ground_to_field")
                .build()
                .await
                .unwrap();
            let task = tokio::spawn(ball_filter::run_boxed(context.clone()));
            // Keep the retained publisher alive throughout the test.
            (
                context,
                (node, field),
                perception,
                camera_pub,
                metrics,
                estimates,
                task,
                odometry,
                pose_pub,
            )
        });
        let camera = camera();
        for tick in 1..=120 {
            let time = Time::from_nanos(tick * 20_000_000);
            clock.set_time(time).unwrap();
            runtime
                .block_on(camera_pub.publish(&TimeWrapper {
                    time,
                    inner: camera.clone(),
                }))
                .unwrap();
            // The robot translates and turns; a stationary world ball changes Ground coordinates.
            let pose = nalgebra::Isometry3::new(
                nalgebra::vector![tick as f32 * 0.001, 0.0, 0.0],
                nalgebra::vector![0.0, 0.0, tick as f32 * 0.0005],
            );
            let ball = Point3::wrap(pose.inverse() * nalgebra::point![2.0, 0.0, 0.105]);
            let ground_to_field = crate::behavior_inputs::ground_to_field(
                pose,
                types::field_dimensions::GlobalFieldSide::Home,
            );
            // Register truth before the filter can receive either input at this time.
            perception
                .publish(
                    time,
                    ground_to_field,
                    &camera,
                    vec![ball],
                    0.105,
                    &Parameters {
                        delivery_delay_seconds: delay,
                        ..clean()
                    },
                )
                .unwrap();
            runtime.block_on(async {
                pose_pub
                    .publish_with_source_time(&ground_to_field, time)
                    .await
                    .unwrap();
                odometry
                    .announce(time)
                    .await
                    .unwrap()
                    .publish(&linear_algebra::Pose2::new(
                        point![pose.translation.x, pose.translation.y],
                        pose.rotation.euler_angles().2,
                    ))
                    .await
                    .unwrap();
                tokio::time::sleep(Duration::from_millis(5)).await;
            });
            assert!(!task.is_finished(), "production filter exited");
        }
        let result = metrics
            .get_latest()
            .expect("production filter did not produce metrics");
        assert!(result.matched_samples > 10, "{result:?}");
        assert_eq!(result.unmatched_timestamps, 0, "{result:?}");
        assert!(result.position_rmse_metres.unwrap() < 0.15, "{result:?}");
        assert!(estimates.get_latest().unwrap().is_some());
        assert!(result.field_matched_samples > 10, "{result:?}");
        assert!(
            result.field_position_rmse_metres.unwrap() < 0.15,
            "{result:?}"
        );
        task.abort();
        runtime.block_on(async {
            let _ = task.await;
        });
        drop(perception);
        drop(node);
        context.shutdown().unwrap();
    }
}
