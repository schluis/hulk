//! Visibility challenges and command outcomes, separate from ball-position loss.
use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use behavior_node::node::Blackboard;
use color_eyre::Result;
use ros_z::{prelude::*, time::Time};
use serde::Serialize;
use tokio::task::JoinHandle;
use types::{motion_command::MotionCommand, primary_state::PrimaryState};

/// Wait for a visible close ball before each burst; never remove its physical body.
#[derive(Default)]
pub struct BriefDropouts {
    visible_frames: u32,
    remaining: u32,
    next: usize,
    pub bursts: [u64; 8],
}

impl BriefDropouts {
    pub fn step(&mut self, close_visible: bool, pattern: &[u32]) -> bool {
        if pattern.is_empty() {
            self.visible_frames = 0;
            self.remaining = 0;
            return false;
        }
        if self.remaining > 0 {
            self.remaining -= 1;
            return true;
        }
        if !close_visible {
            self.visible_frames = 0;
            return false;
        }
        if self.visible_frames < 8 {
            self.visible_frames += 1;
            return false;
        }
        let frames = pattern[self.next % pattern.len()];
        self.next = self.next.wrapping_add(1);
        self.visible_frames = 0;
        self.remaining = frames - 1;
        self.bursts[frames as usize - 1] += 1;
        true
    }
}

#[derive(Clone, Serialize)]
pub struct StopEvent {
    pub time: Time,
    pub model_distance_metres: f32,
    pub visual_present: bool,
    pub visual_age_seconds: Option<f64>,
}

#[derive(Clone, Default, Serialize)]
pub struct Summary {
    pub samples: u64,
    pub close_upright_playing_seconds: f64,
    pub kick_seconds: f64,
    pub stand_seconds: f64,
    pub stand_after_kick_with_visual_under_100ms_seconds: f64,
    pub kick_to_stand: u64,
    pub timestamp_gaps: u64,
    pub events: Vec<StopEvent>,
    #[serde(skip)]
    previous: Option<Sample>,
    #[serde(skip)]
    interrupted_kick: bool,
}

#[derive(Clone)]
struct Sample {
    time: Time,
    eligible: bool,
    kick: bool,
    stand: bool,
    distance: f32,
    visual_present: bool,
    visual_age: Option<f64>,
}

impl Sample {
    fn from_blackboard(board: &Blackboard) -> Self {
        let world = &board.world_state;
        let distance = world
            .ball
            .map_or(f32::INFINITY, |ball| ball.ball_in_ground.coords().norm());
        Self {
            time: world.now,
            eligible: world.robot.primary_state == PrimaryState::Playing
                && world
                    .fall_detection
                    .is_some_and(|fall| fall.is_upright(world.now))
                && distance <= 1.0
                && !board.remote_control_enabled
                && !board.is_injected_motion_command,
            kick: matches!(board.last_motion_command, MotionCommand::Kick { .. }),
            stand: matches!(board.last_motion_command, MotionCommand::Stand { .. }),
            distance,
            visual_present: board.visual_kick_ball_position.is_some(),
            visual_age: board
                .visual_kick_ball_position
                .and_then(|ball| ball.age_at(world.now))
                .map(|age| age.as_secs_f64()),
        }
    }
}

impl Summary {
    fn observe(&mut self, sample: Sample) {
        self.samples += 1;
        if let Some(previous) = &self.previous {
            let dt = sample.time.duration_since(previous.time);
            if sample.time <= previous.time || dt > Duration::from_millis(100) {
                self.timestamp_gaps += 1;
                self.interrupted_kick = false;
            } else {
                if previous.eligible {
                    self.close_upright_playing_seconds += dt.as_secs_f64();
                    if previous.kick {
                        self.kick_seconds += dt.as_secs_f64();
                    }
                    if previous.stand {
                        self.stand_seconds += dt.as_secs_f64();
                        if self.interrupted_kick
                            && previous.visual_age.is_some_and(|age| age <= 0.1)
                        {
                            self.stand_after_kick_with_visual_under_100ms_seconds +=
                                dt.as_secs_f64();
                        }
                    }
                }
                if previous.eligible && previous.kick && sample.eligible && sample.stand {
                    self.kick_to_stand += 1;
                    self.interrupted_kick = true;
                    if self.events.len() < 100 {
                        self.events.push(StopEvent {
                            time: sample.time,
                            model_distance_metres: sample.distance,
                            visual_present: sample.visual_present,
                            visual_age_seconds: sample.visual_age,
                        });
                    }
                }
            }
        }
        if !sample.stand || !sample.eligible {
            self.interrupted_kick = false;
        }
        self.previous = Some(sample);
    }

    pub fn status(&self) -> &'static str {
        if self.kick_to_stand > 0 {
            "review_required"
        } else if self.kick_seconds < 0.2 {
            "insufficient_coverage"
        } else {
            "no_interruption_observed"
        }
    }
}

pub struct Observer {
    summary: Arc<Mutex<Summary>>,
    task: JoinHandle<()>,
}

impl Observer {
    pub async fn new(node: &Node) -> Result<Self> {
        let subscriber = node
            .subscriber::<Blackboard>("behavior/blackboard")
            .build()
            .await?;
        let summary = Arc::new(Mutex::new(Summary::default()));
        let output = summary.clone();
        let task = tokio::spawn(async move {
            while let Ok(board) = subscriber.recv().await {
                output
                    .lock()
                    .expect("approach summary lock poisoned")
                    .observe(Sample::from_blackboard(&board));
            }
        });
        Ok(Self { summary, task })
    }

    pub fn snapshot(&self) -> Summary {
        self.summary
            .lock()
            .expect("approach summary lock poisoned")
            .clone()
    }
}

impl Drop for Observer {
    fn drop(&mut self) {
        self.task.abort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brief_gaps_cover_one_two_three_frames_only_after_close_visibility() {
        let mut schedule = BriefDropouts::default();
        for _ in 0..30 {
            assert!(!schedule.step(false, &[1, 2, 3]));
        }
        for burst in [1, 2, 3, 1] {
            for _ in 0..8 {
                assert!(!schedule.step(true, &[1, 2, 3]));
            }
            for _ in 0..burst {
                assert!(schedule.step(true, &[1, 2, 3]));
            }
        }
        assert_eq!(schedule.bursts[..3], [2, 1, 1]);
        assert!(!schedule.step(true, &[]));
    }

    #[test]
    fn report_detects_stopping_with_retained_model_and_fresh_vision_return() {
        let mut report = Summary::default();
        let sample = |ms: i64, kick, stand, visual_age: Option<f64>| Sample {
            time: Time::from_nanos(ms * 1_000_000),
            eligible: true,
            kick,
            stand,
            distance: 0.65,
            visual_present: visual_age.is_some(),
            visual_age,
        };
        report.observe(sample(0, true, false, Some(0.08)));
        report.observe(sample(20, false, true, Some(0.101)));
        report.observe(sample(40, false, true, Some(0.06)));
        report.observe(sample(60, true, false, Some(0.08)));
        assert_eq!(report.kick_to_stand, 1);
        assert!((report.stand_seconds - 0.04).abs() < 1e-9);
        assert!((report.stand_after_kick_with_visual_under_100ms_seconds - 0.02).abs() < 1e-9);
        assert_eq!(report.status(), "review_required");
        report.observe(sample(1000, false, true, None));
        assert_eq!(
            report.kick_to_stand, 1,
            "a recording gap is not a continuous transition"
        );
        assert_eq!(Summary::default().status(), "insufficient_coverage");
    }
}
