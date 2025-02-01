use std::f32::consts::FRAC_PI_2;

use color_eyre::{
    eyre::{bail, eyre, Context as EyreContext, ContextCompat},
    Result,
};
use context_attribute::context;
use coordinate_systems::{Field, Ground, Robot, UpcomingSupport, Walk};
use filtering::low_pass_filter::LowPassFilter;
use framework::{deserialize_not_implemented, AdditionalOutput, MainOutput};
use hardware::PathsInterface;
use kinematics::forward;
use linear_algebra::{vector, Isometry2, Isometry3, Orientation3, Point2, Point3, Vector3};
use openvino::{CompiledModel, Core, DeviceType, InferenceError::GeneralError};
use serde::{Deserialize, Serialize};
use types::{
    cycle_time::CycleTime,
    joints::body::BodyJoints,
    motion_selection::{MotionSafeExits, MotionType},
    motor_commands::MotorCommands,
    obstacle_avoiding_arms::{ArmCommand, ArmCommands},
    sensor_data::SensorData,
    support_foot::Side,
    walk_command::WalkCommand,
};
use walking_engine::{kick_steps::KickSteps, mode::Mode, parameters::Parameters, Context, Engine};

#[derive(Serialize, Deserialize)]
pub struct WalkingEngine {
    engine: Engine,
    last_actuated_joints: BodyJoints,
    filtered_gyro: LowPassFilter<nalgebra::Vector3<f32>>,

    #[serde(skip, default = "deserialize_not_implemented")]
    network: CompiledModel,
}

#[context]
pub struct CreationContext {
    parameters: Parameter<Parameters, "walking_engine">,
    hardware_interface: HardwareInterface,
}

#[context]
#[derive(Debug)]
pub struct CycleContext {
    parameters: Parameter<Parameters, "walking_engine">,
    kick_steps: Parameter<KickSteps, "kick_steps">,

    motion_safe_exits: CyclerState<MotionSafeExits, "motion_safe_exits">,
    ground_to_upcoming_support:
        CyclerState<Isometry2<Ground, UpcomingSupport>, "ground_to_upcoming_support">,

    cycle_time: Input<CycleTime, "cycle_time">,
    center_of_mass: Input<Point3<Robot>, "center_of_mass">,
    sensor_data: Input<SensorData, "sensor_data">,
    walk_command: Input<WalkCommand, "walk_command">,
    robot_to_ground: Input<Option<Isometry3<Robot, Ground>>, "robot_to_ground?">,
    robot_orientation: RequiredInput<Option<Orientation3<Field>>, "robot_orientation?">,
    obstacle_avoiding_arms: Input<ArmCommands, "obstacle_avoiding_arms">,
    zero_moment_point: Input<Point2<Ground>, "zero_moment_point">,
    number_of_consecutive_cycles_zero_moment_point_outside_support_polygon:
        Input<i32, "number_of_consecutive_cycles_zero_moment_point_outside_support_polygon">,

    debug_output: AdditionalOutput<Engine, "walking.engine">,
    last_actuated_joints: AdditionalOutput<BodyJoints, "walking.last_actuated_joints">,
    robot_to_walk: AdditionalOutput<Isometry3<Robot, Walk>, "walking.robot_to_walk">,
    walking_engine_mode: CyclerState<Mode, "walking_engine_mode">,
}

#[context]
#[derive(Default)]
pub struct MainOutputs {
    pub walk_motor_commands: MainOutput<MotorCommands<BodyJoints<f32>>>,
}

const EXPECTED_OUTPUT_NAME: &str = "output";

impl WalkingEngine {
    pub fn new(context: CreationContext<impl PathsInterface>) -> Result<Self> {
        let paths = context.hardware_interface.get_paths();
        let neural_network_folder = paths.neural_networks;

        let model_xml_name = "robust-dust-41-policy-ov.xml";

        let model_path = neural_network_folder.join(model_xml_name);
        let weights_path = neural_network_folder
            .join(model_xml_name)
            .with_extension("bin");

        let mut core = Core::new()?;
        let network = core
            .read_model_from_file(
                model_path
                    .to_str()
                    .wrap_err("failed to get detection model path")?,
                weights_path
                    .to_str()
                    .wrap_err("failed to get detection weights path")?,
            )
            .map_err(|error| match error {
                GeneralError => eyre!("{error}: possible incomplete OpenVino installation"),
                _ => eyre!("{error}: failed to create detection network"),
            })?;

        let number_of_inputs = network
            .get_inputs_len()
            .wrap_err("failed to get number of inputs")?;
        let output_name = network.get_output_by_index(0)?.get_name()?;
        dbg!(&output_name);
        dbg!(&number_of_inputs);
        if number_of_inputs != 1 || output_name != EXPECTED_OUTPUT_NAME {
            bail!(
                "expected exactly one input and output name to be '{}'",
                EXPECTED_OUTPUT_NAME
            );
        }

        Ok(Self {
            engine: Engine::default(),
            last_actuated_joints: Default::default(),
            filtered_gyro: LowPassFilter::with_smoothing_factor(
                nalgebra::Vector3::zeros(),
                context.parameters.gyro_balancing.low_pass_factor,
            ),
            network: core.compile_model(&network, DeviceType::CPU)?,
        })
    }

    pub fn cycle(&mut self, mut cycle_context: CycleContext) -> Result<MainOutputs> {
        self.filtered_gyro.update(
            cycle_context
                .sensor_data
                .inertial_measurement_unit
                .angular_velocity
                .inner,
        );

        let torso_tilt_compensation_factor = cycle_context
            .parameters
            .swinging_arms
            .torso_tilt_compensation_factor;

        let arm_compensation = compensate_arm_motion_with_torso_tilt(
            &cycle_context.obstacle_avoiding_arms.left_arm,
            torso_tilt_compensation_factor,
        ) + compensate_arm_motion_with_torso_tilt(
            &cycle_context.obstacle_avoiding_arms.right_arm,
            torso_tilt_compensation_factor,
        );

        let robot_to_walk = Isometry3::from_parts(
            vector![
                cycle_context.parameters.base.torso_offset,
                0.0,
                cycle_context.parameters.base.walk_height,
            ],
            Orientation3::new(
                Vector3::y_axis() * (cycle_context.parameters.base.torso_tilt + arm_compensation),
            ),
        );

        let mut context = Context {
            parameters: cycle_context.parameters,
            kick_steps: cycle_context.kick_steps,
            cycle_time: cycle_context.cycle_time,
            center_of_mass: cycle_context.center_of_mass,
            force_sensitive_resistors: &cycle_context.sensor_data.force_sensitive_resistors,
            robot_orientation: cycle_context.robot_orientation,
            robot_to_ground: cycle_context.robot_to_ground,
            gyro: self.filtered_gyro.state(),
            current_joints: self.last_actuated_joints,
            robot_to_walk,
            obstacle_avoiding_arms: cycle_context.obstacle_avoiding_arms,
            zero_moment_point: cycle_context.zero_moment_point,
            number_of_consecutive_cycles_zero_moment_point_outside_support_polygon: cycle_context
                .number_of_consecutive_cycles_zero_moment_point_outside_support_polygon,
            network: &mut self.network,
            sensor_data: cycle_context.sensor_data,
        };

        match *cycle_context.walk_command {
            WalkCommand::Stand => self.engine.stand(&context),
            WalkCommand::Walk { step } => self.engine.walk(&context, step),
            WalkCommand::Kick {
                variant,
                side,
                strength,
            } => self.engine.kick(&context, variant, side, strength),
        };

        self.engine.tick(&context);

        let motor_commands = self.engine.compute_commands(&mut context);

        self.last_actuated_joints = motor_commands.positions;

        *cycle_context.ground_to_upcoming_support = self
            .calculate_return_offset(cycle_context.parameters, robot_to_walk)
            .unwrap_or_default();
        cycle_context.motion_safe_exits[MotionType::Walk] = self.engine.is_standing();

        cycle_context
            .debug_output
            .fill_if_subscribed(|| self.engine.clone());
        cycle_context
            .last_actuated_joints
            .fill_if_subscribed(|| self.last_actuated_joints);
        cycle_context
            .robot_to_walk
            .fill_if_subscribed(|| robot_to_walk);
        *cycle_context.walking_engine_mode = self.engine.mode;

        Ok(MainOutputs {
            walk_motor_commands: motor_commands.into(),
        })
    }

    fn calculate_return_offset(
        &self,
        parameters: &Parameters,
        robot_to_walk: Isometry3<Robot, Walk>,
    ) -> Option<Isometry2<Ground, UpcomingSupport>> {
        let left_sole = robot_to_walk
            * forward::left_sole_to_robot(&self.last_actuated_joints.left_leg).as_pose();
        let right_sole = robot_to_walk
            * forward::right_sole_to_robot(&self.last_actuated_joints.right_leg).as_pose();
        let support_side = self.engine.support_side()?;
        let swing_sole = match support_side {
            Side::Left => right_sole,
            Side::Right => left_sole,
        };
        let swing_sole_base_offset = match support_side {
            Side::Left => parameters.base.foot_offset_right,
            Side::Right => parameters.base.foot_offset_left,
        };

        let forward = swing_sole.position().x();
        let left = swing_sole.position().y() - swing_sole_base_offset.y();
        let turn = swing_sole.orientation().inner.euler_angles().2;
        Some(Isometry2::from_parts(vector![forward, left], turn).inverse())
    }
}

fn compensate_arm_motion_with_torso_tilt(
    arm_command: &ArmCommand,
    torso_tilt_compensation_factor: f32,
) -> f32 {
    (arm_command.shoulder_pitch() - FRAC_PI_2) * torso_tilt_compensation_factor
}
