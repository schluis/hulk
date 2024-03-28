use types::{fall_state::FallState, motion_command::MotionCommand, world_state::WorldState};

pub fn execute(world_state: &WorldState, has_ground_contact: bool) -> Option<MotionCommand> {
    match (world_state.robot.fall_state, has_ground_contact) {
        (
            FallState::Falling {
                start_time,
                direction,
            },
            true,
        ) => Some(MotionCommand::FallProtection {
            start_time,
            direction,
        }),
        _ => None,
    }
}
