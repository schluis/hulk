use path_serde::{PathDeserialize, PathIntrospect, PathSerialize};
use serde::{Deserialize, Serialize};
use types::{
    joints::body::BodyJoints, motion_command::KickVariant, motor_commands::MotorCommands,
    step::Step, support_foot::Side,
};

use crate::{
    step_plan::StepPlan, step_state::StepState, stiffness::Stiffness as _, Context, WalkTransition,
};

use super::{starting::Starting, Mode};

#[derive(
    Default,
    Clone,
    Copy,
    Debug,
    Serialize,
    Deserialize,
    PathSerialize,
    PathDeserialize,
    PathIntrospect,
)]
pub struct Standing {}

impl WalkTransition for Standing {
    fn stand(self, _context: &Context) -> Mode {
        Mode::Standing(Standing {})
    }

    fn walk(self, context: &Context, step: Step) -> Mode {
        let is_requested_step_towards_left = step.left.is_sign_positive();
        let support_side = if is_requested_step_towards_left {
            Side::Left
        } else {
            Side::Right
        };
        Mode::Starting(Starting::new(context, support_side))
    }

    fn kick(
        self,
        context: &Context,
        _variant: KickVariant,
        kicking_side: Side,
        _strength: f32,
    ) -> Mode {
        let support_side = if kicking_side == Side::Left {
            Side::Left
        } else {
            Side::Right
        };
        Mode::Starting(Starting::new(context, support_side))
    }
}

impl Standing {
    pub fn compute_commands(&self, context: &Context) -> MotorCommands<BodyJoints> {
        let plan = StepPlan::stand(context);
        let zero_step = StepState::new(plan);

        zero_step.compute_joints(context).apply_stiffness(
            context.parameters.stiffnesses.leg_stiffness_stand,
            context.parameters.stiffnesses.arm_stiffness,
        )
    }

    pub fn tick(&mut self, _context: &Context) {}
}
