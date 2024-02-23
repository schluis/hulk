use source_analyzer::{
    cyclers::{CyclerKind, Cyclers},
    error::Error,
    manifest::{CyclerManifest, FrameworkManifest},
};

pub fn collect_hulk_cyclers() -> Result<Cyclers, Error> {
    let manifest = FrameworkManifest {
        cyclers: vec![
            CyclerManifest {
                name: "Vision",
                kind: CyclerKind::Perception,
                instances: vec!["Top", "Bottom"],
                setup_nodes: vec!["vision::image_receiver"],
                nodes: vec![
                    "vision::ball_detection",
                    "vision::camera_matrix_extractor",
                    "vision::feet_detection",
                    "vision::field_border_detection",
                    "vision::field_color_detection",
                    "vision::image_segmenter",
                    "vision::limb_projector",
                    "vision::line_detection",
                    "vision::perspective_grid_candidates_provider",
                    "vision::segment_filter",
                ],
            },
            CyclerManifest {
                name: "Control",
                kind: CyclerKind::RealTime,
                instances: vec![""],
                setup_nodes: vec!["control::sensor_data_receiver"],
                nodes: vec![
                    "control::active_vision",
                    "control::ball_filter",
                    "control::ball_state_composer",
                    "control::behavior::node",
                    "control::button_filter",
                    "control::camera_matrix_calculator",
                    "control::center_of_mass_provider",
                    "control::fall_state_estimation",
                    "control::foot_bumper_filter",
                    "control::game_controller_filter",
                    "control::game_controller_state_filter",
                    "control::ground_contact_detector",
                    "control::ground_provider",
                    "control::kick_selector",
                    "control::kinematics_provider",
                    "control::led_status",
                    "control::localization",
                    "control::motion::arms_up_squat",
                    "control::motion::command_sender",
                    "control::motion::condition_input_provider",
                    "control::motion::dispatching_interpolator",
                    "control::motion::fall_protector",
                    "control::motion::head_motion",
                    "control::motion::jump_left",
                    "control::motion::jump_right",
                    "control::motion::look_around",
                    "control::motion::look_at",
                    "control::motion::motion_selector",
                    "control::motion::motor_commands_collector",
                    "control::motion::motor_commands_optimizer",
                    "control::motion::sit_down",
                    "control::motion::stand_up_back",
                    "control::motion::stand_up_front",
                    "control::motion::stand_up_sitting",
                    "control::motion::step_planner",
                    "control::motion::walk_manager",
                    "control::motion::walking_engine",
                    "control::obstacle_filter",
                    "control::odometry",
                    "control::orientation_filter",
                    "control::penalty_shot_direction_estimation",
                    "control::primary_state_filter",
                    "control::role_assignment",
                    "control::rule_obstacle_composer",
                    "control::sole_pressure_filter",
                    "control::sonar_filter",
                    "control::support_foot_estimation",
                    "control::time_to_reach_kick_position",
                    "control::visual_referee_filter",
                    "control::whistle_filter",
                    "control::world_state_composer",
                ],
            },
            CyclerManifest {
                name: "SplNetwork",
                kind: CyclerKind::Perception,
                instances: vec![""],
                setup_nodes: vec!["spl_network::message_receiver"],
                nodes: vec!["spl_network::message_filter"],
            },
            CyclerManifest {
                name: "Audio",
                kind: CyclerKind::Perception,
                instances: vec![""],
                setup_nodes: vec!["audio::microphone_recorder"],
                nodes: vec!["audio::whistle_detection"],
            },
        ],
    };

    let root = "..";
    Cyclers::try_from_manifest(manifest, root)
}
