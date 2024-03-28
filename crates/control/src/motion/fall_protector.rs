use std::{
    ops::Range,
    time::{Duration, SystemTime},
};

use color_eyre::Result;
use context_attribute::context;
use framework::MainOutput;
use serde::{Deserialize, Serialize};
use types::{
    cycle_time::CycleTime,
    fall_state::{Direction, Side},
    joints::{
        arm::ArmJoints, body::BodyJoints, head::HeadJoints, leg::LegJoints, mirror::Mirror, Joints,
    },
    motion_command::MotionCommand,
    motion_selection::{MotionSafeExits, MotionVariant},
    motor_commands::MotorCommands,
    sensor_data::SensorData,
};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Phase {
    Early,
    Late,
}

#[derive(Default, Serialize, Deserialize)]
pub struct FallProtector {}

#[context]
pub struct CreationContext {}

#[context]
pub struct CycleContext {
    cycle_time: Input<CycleTime, "cycle_time">,
    motion_command: Input<MotionCommand, "motion_command">,
    sensor_data: Input<SensorData, "sensor_data">,

    front_early: Parameter<Joints<f32>, "fall_protection.front_early">,
    front_late: Parameter<Joints<f32>, "fall_protection.front_late">,
    back_early: Parameter<Joints<f32>, "fall_protection.back_early">,
    back_late: Parameter<Joints<f32>, "fall_protection.back_late">,

    early_protection_timeout: Parameter<Duration, "fall_protection.early_protection_timeout">,
    reached_threshold: Parameter<f32, "fall_protection.reached_threshold">,
    head_stiffness: Parameter<Range<f32>, "fall_protection.head_stiffness">,
    arm_stiffness: Parameter<Range<f32>, "fall_protection.arm_stiffness">,
    leg_stiffness: Parameter<Range<f32>, "fall_protection.leg_stiffness">,
}

#[context]
#[derive(Default)]
pub struct MainOutputs {
    pub fall_protection_command: MainOutput<MotorCommands<Joints<f32>>>,
}

impl FallProtector {
    pub fn new(_context: CreationContext) -> Result<Self> {
        Ok(Self::default())
    }

    pub fn cycle(&mut self, context: CycleContext) -> Result<MainOutputs> {
        let MotionCommand::FallProtection {
            start_time,
            direction,
        } = context.motion_command
        else {
            return Ok(MainOutputs::default());
        };

        let duration_in_fall = context
            .cycle_time
            .start_time
            .duration_since(*start_time)
            .unwrap();
        let phase = if duration_in_fall < *context.early_protection_timeout {
            Phase::Early
        } else {
            Phase::Late
        };

        let measured_positions = context.sensor_data.positions;
        let protection_angles = match (direction, phase) {
            (Direction::Forward { side: Side::Left }, Phase::Early) => {
                prevent_stuck_arms(context.front_early.mirrored(), measured_positions)
            }
            (Direction::Forward { side: Side::Left }, Phase::Late) => {
                prevent_stuck_arms(context.front_late.mirrored(), measured_positions)
            }
            (Direction::Forward { side: Side::Right }, Phase::Early) => {
                prevent_stuck_arms(*context.front_early, measured_positions)
            }
            (Direction::Forward { side: Side::Right }, Phase::Late) => {
                prevent_stuck_arms(*context.front_late, measured_positions)
            }
            (Direction::Backward { side: Side::Left }, Phase::Early) => {
                context.back_early.mirrored()
            }
            (Direction::Backward { side: Side::Left }, Phase::Late) => context.back_late.mirrored(),
            (Direction::Backward { side: Side::Right }, Phase::Early) => *context.back_early,
            (Direction::Backward { side: Side::Right }, Phase::Late) => *context.back_late,
        };

        let is_head_protected = (measured_positions.head.pitch - protection_angles.head.pitch)
            .abs()
            < *context.reached_threshold
            && (measured_positions.head.yaw - protection_angles.head.yaw).abs()
                < *context.reached_threshold;

        let head_stiffnesses = if is_head_protected {
            HeadJoints::fill(context.head_stiffness.end)
        } else {
            HeadJoints::fill(context.head_stiffness.start)
        };

        let body_stiffnesses = match phase {
            Phase::Early => BodyJoints {
                left_arm: ArmJoints::fill(context.arm_stiffness.start),
                right_arm: ArmJoints::fill(context.arm_stiffness.start),
                left_leg: LegJoints::fill(context.leg_stiffness.start),
                right_leg: LegJoints::fill(context.leg_stiffness.start),
            },
            Phase::Late => BodyJoints {
                left_arm: ArmJoints::fill(context.arm_stiffness.end),
                right_arm: ArmJoints::fill(context.arm_stiffness.end),
                left_leg: LegJoints::fill(context.leg_stiffness.end),
                right_leg: LegJoints::fill(context.leg_stiffness.end),
            },
        };

        let motor_commands = MotorCommands {
            positions: protection_angles,
            stiffnesses: Joints::from_head_and_body(head_stiffnesses, body_stiffnesses),
        };

        Ok(MainOutputs {
            fall_protection_command: motor_commands.into(),
        })
    }
}

fn prevent_stuck_arms(request: Joints<f32>, measured_positions: Joints<f32>) -> Joints<f32> {
    let left_arm = if measured_positions.left_arm.shoulder_roll < 0.0
        && measured_positions.left_arm.shoulder_pitch > 1.6
    {
        ArmJoints {
            shoulder_pitch: 0.0,
            shoulder_roll: 0.35,
            elbow_yaw: 0.0,
            elbow_roll: 0.0,
            wrist_yaw: 0.0,
            hand: 0.0,
        }
    } else {
        request.left_arm
    };
    let right_arm = if measured_positions.right_arm.shoulder_roll > 0.0
        && measured_positions.right_arm.shoulder_pitch > 1.6
    {
        ArmJoints {
            shoulder_pitch: 0.0,
            shoulder_roll: -0.35,
            elbow_yaw: 0.0,
            elbow_roll: 0.0,
            wrist_yaw: 0.0,
            hand: 0.0,
        }
    } else {
        request.right_arm
    };
    Joints {
        left_arm,
        right_arm,
        ..request
    }
}
