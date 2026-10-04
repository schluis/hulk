use crate::hypothesis::BallHypothesis;
use projection::camera_matrix::CameraMatrix;
use types::ball_detection::BallPercept;

pub fn observe(
    hypothesis: &mut BallHypothesis,
    percept: &BallPercept,
    camera: Option<&CameraMatrix>,
    ball_radius: f32,
) {
    let error = camera
        .and_then(|camera| {
            let position = percept.percept_in_ground.mean;
            let camera_position = camera.ground_to_camera
                * linear_algebra::point![position.x, position.y, ball_radius];
            let depth = camera_position.z();
            (depth.is_finite() && depth > 0.0).then(|| {
                ball_radius * camera.intrinsics.focals.x.min(camera.intrinsics.focals.y) / depth
            })
        })
        .filter(|expected| expected.is_finite() && *expected > 0.0)
        .and_then(|expected| {
            let observed = percept.image_location.radius;
            (observed.is_finite() && observed > 0.0).then(|| (expected / observed).ln().max(0.0))
        })
        .filter(|error| error.is_finite());
    // Oversized images can represent an airborne ball, so only undersized
    // observations provide negative size evidence.
    // Unknown geometry provides no negative evidence. A new valid measurement
    // gradually replaces history so one noisy radius cannot dominate selection.
    hypothesis.size_consistency_error = error.map(|new| {
        hypothesis
            .size_consistency_error
            .map_or(new, |old| 0.75 * old + 0.25 * new)
    });
}
