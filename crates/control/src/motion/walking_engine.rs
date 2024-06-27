use std::f32::consts::FRAC_PI_2;
use std::f32::consts::FRAC_PI_8;
use std::time::Duration;

use color_eyre::Result;
use context_attribute::context;
use coordinate_systems::{Ground, Robot, Walk};
use filtering::kalman_filter::KalmanFilter;
use filtering::low_pass_filter::LowPassFilter;
use framework::{AdditionalOutput, MainOutput};
use kinematics::forward;
use linear_algebra::{vector, Isometry3, Orientation3, Point2, Point3, Vector3};
use nalgebra::matrix;
use serde::{Deserialize, Serialize};
use types::multivariate_normal_distribution::MultivariateNormalDistribution;
use types::{
    cycle_time::CycleTime,
    joints::body::BodyJoints,
    motion_selection::{MotionSafeExits, MotionType},
    motor_commands::MotorCommands,
    obstacle_avoiding_arms::{ArmCommand, ArmCommands},
    sensor_data::SensorData,
    step_plan::Step,
    support_foot::Side,
    walk_command::WalkCommand,
};
use walking_engine::{kick_steps::KickSteps, parameters::Parameters, Context, Engine};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkingEngine {
    engine: Engine,
    last_actuated_joints: BodyJoints,
    filtered_gyro: LowPassFilter<nalgebra::Vector3<f32>>,
    torso_tilt_factor_controller: PIDController,
    filtered_zero_moment_point: MultivariateNormalDistribution<3>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIDController {
    pub last_error: f32,
    pub i_error: f32,
    pub k_p: f32,
    pub k_i: f32,
    pub k_d: f32,
    pub k_v: f32,
    pub k_a: f32,
    pub setpoint: f32,
    pub anti_wind_up_clamp: f32,
}

impl Default for PIDController {
    fn default() -> Self {
        Self {
            anti_wind_up_clamp: f32::MAX,
            last_error: 0.0,
            i_error: 0.0,
            k_p: 0.0,
            k_i: 0.0,
            k_d: 0.0,
            setpoint: 0.0,
            k_v: 0.0,
            k_a: 0.0,
        }
    }
}

impl PIDController {
    pub fn control(&mut self, process_variable: f32, time_passed: Duration) -> f32 {
        let error = self.setpoint - process_variable;
        let d_error = (error - self.last_error) / time_passed.as_secs_f32();

        if (error * self.i_error).is_sign_negative() && error > 0.06 {
            self.i_error /= 1.1;
        }

        self.i_error += error * time_passed.as_secs_f32();
        self.last_error = error;
        self.k_p * error
            + self.k_i
                * self
                    .i_error
                    .clamp(-self.anti_wind_up_clamp, self.anti_wind_up_clamp)
            + self.k_d * d_error
    }

    pub fn feed_forward(&self, position: f32, velocity: f32, acceleration: f32) -> f32 {
        position + self.k_v * velocity + self.k_a * acceleration
    }
}

#[context]
pub struct CreationContext {
    parameters: Parameter<Parameters, "walking_engine">,
}

#[context]
#[derive(Debug)]
pub struct CycleContext {
    parameters: Parameter<Parameters, "walking_engine">,
    kick_steps: Parameter<KickSteps, "kick_steps">,

    motion_safe_exits: CyclerState<MotionSafeExits, "motion_safe_exits">,
    walk_return_offset: CyclerState<Step, "walk_return_offset">,

    cycle_time: Input<CycleTime, "cycle_time">,
    center_of_mass: Input<Point3<Robot>, "center_of_mass">,
    sensor_data: Input<SensorData, "sensor_data">,
    walk_command: Input<WalkCommand, "walk_command">,
    robot_to_ground: Input<Option<Isometry3<Robot, Ground>>, "robot_to_ground?">,
    obstacle_avoiding_arms: Input<ArmCommands, "obstacle_avoiding_arms">,
    zero_moment_point: Input<Point2<Ground>, "zero_moment_point">,
    number_of_consecutive_cycles_zero_moment_point_outside_support_polygon:
        Input<i32, "number_of_consecutive_cycles_zero_moment_point_outside_support_polygon">,

    debug_output: AdditionalOutput<Engine, "walking.engine">,
    last_actuated_joints: AdditionalOutput<BodyJoints, "walking.last_actuated_joints">,
    robot_to_walk: AdditionalOutput<Isometry3<Robot, Walk>, "walking.robot_to_walk">,
    filtered_zero_moment_point_mean:
        AdditionalOutput<nalgebra::Vector3<f32>, "walking.filtered_zero_moment_point_mean">,
    torso_tilt_compensation_factor: AdditionalOutput<f32, "walking.torso_tilt_compensation_factor">,
}
#[context]
#[derive(Default)]
pub struct MainOutputs {
    pub walk_motor_commands: MainOutput<MotorCommands<BodyJoints<f32>>>,
}

impl WalkingEngine {
    pub fn new(context: CreationContext) -> Result<Self> {
        Ok(Self {
            engine: Engine::default(),
            last_actuated_joints: Default::default(),
            filtered_gyro: LowPassFilter::with_smoothing_factor(
                nalgebra::Vector3::zeros(),
                context.parameters.gyro_balancing.low_pass_factor,
            ),
            torso_tilt_factor_controller: PIDController::default(),
            filtered_zero_moment_point: MultivariateNormalDistribution {
                mean: nalgebra::Vector3::zeros(),
                covariance: nalgebra::Matrix3::identity(),
            },
        })
    }

    pub fn cycle(&mut self, mut cycle_context: CycleContext) -> Result<MainOutputs> {
        self.torso_tilt_factor_controller.k_p = cycle_context.parameters.base.torso_tilt_factor_k_p;
        self.torso_tilt_factor_controller.k_i = cycle_context.parameters.base.torso_tilt_factor_k_i;
        self.torso_tilt_factor_controller.k_d = cycle_context.parameters.base.torso_tilt_factor_k_d;
        self.torso_tilt_factor_controller.anti_wind_up_clamp = cycle_context
            .parameters
            .base
            .torso_tilt_factor_anti_wind_up_clamp;
        self.torso_tilt_factor_controller.k_v = cycle_context.parameters.base.torso_tilt_factor_k_v;
        self.torso_tilt_factor_controller.k_a = cycle_context.parameters.base.torso_tilt_factor_k_a;

        self.filtered_gyro.update(
            cycle_context
                .sensor_data
                .inertial_measurement_unit
                .angular_velocity
                .inner,
        );

        // state vector:
        // [zero_moment_point_x;
        //  zero_moment_point_ẋ;
        //  zero_moment_point_ẍ]

        let passed_time = cycle_context.cycle_time.last_cycle_duration.as_secs_f32();
        let state_transition_model = matrix![1.0, passed_time, passed_time.powi(2) / 2.0;
                                             0.0, 1.0,         passed_time;
                                             0.0, 0.0,         1.0];
        let control_input_model = nalgebra::Matrix3::identity();
        let control_vector = nalgebra::vector![0.0, 0.0, 0.0]; // todo: add control input, i.e. zmp
        let process_noise = nalgebra::Matrix3::from_diagonal(
            &nalgebra::Vector3::new(
                passed_time.powi(3) / 6.0,
                passed_time.powi(2) / 2.0,
                passed_time,
            ) * cycle_context
                .parameters
                .base
                .zero_moment_point_process_noise,
        );

        self.filtered_zero_moment_point.predict(
            state_transition_model,
            control_input_model,
            control_vector,
            process_noise,
        );

        let measurement_model = matrix![1.0, 0.0, 0.0];
        let measurement = nalgebra::vector![cycle_context.zero_moment_point.x()];
        let measurement_noise = matrix![
            cycle_context
                .parameters
                .base
                .zero_moment_point_measurement_noise
        ];

        self.filtered_zero_moment_point
            .update(measurement_model, measurement, measurement_noise);

        let zero_moment_point_angle = f32::atan(
            self.filtered_zero_moment_point.mean[0] / cycle_context.parameters.base.walk_height,
        );

        let mut torso_tilt_compensation_factor = self.torso_tilt_factor_controller.control(
            zero_moment_point_angle,
            cycle_context.cycle_time.last_cycle_duration,
        );
        torso_tilt_compensation_factor = self.torso_tilt_factor_controller.feed_forward(
            torso_tilt_compensation_factor,
            self.filtered_zero_moment_point.mean[1],
            self.filtered_zero_moment_point.mean[2],
        );
        // .clamp(-FRAC_PI_8, FRAC_PI_8);
        // .clamp(0.0, 0.0);

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
                Vector3::y_axis() * (torso_tilt_compensation_factor + arm_compensation),
            ),
        );

        let context = Context {
            parameters: cycle_context.parameters,
            kick_steps: cycle_context.kick_steps,
            cycle_time: cycle_context.cycle_time,
            center_of_mass: cycle_context.center_of_mass,
            sensor_data: cycle_context.sensor_data,
            robot_to_ground: cycle_context.robot_to_ground,
            gyro: self.filtered_gyro.state(),
            current_joints: self.last_actuated_joints,
            robot_to_walk,
            obstacle_avoiding_arms: cycle_context.obstacle_avoiding_arms,
            zero_moment_point: cycle_context.zero_moment_point,
            number_of_consecutive_cycles_zero_moment_point_outside_support_polygon: cycle_context
                .number_of_consecutive_cycles_zero_moment_point_outside_support_polygon,
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

        let motor_commands = self.engine.compute_commands(&context);

        self.last_actuated_joints = motor_commands.positions;

        *cycle_context.walk_return_offset = self
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
        cycle_context
            .filtered_zero_moment_point_mean
            .fill_if_subscribed(|| self.filtered_zero_moment_point.mean);
        cycle_context
            .torso_tilt_compensation_factor
            .fill_if_subscribed(|| torso_tilt_compensation_factor);

        Ok(MainOutputs {
            walk_motor_commands: motor_commands.into(),
        })
    }

    fn calculate_return_offset(
        &self,
        parameters: &Parameters,
        robot_to_walk: Isometry3<Robot, Walk>,
    ) -> Option<Step> {
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

        Some(Step {
            forward: swing_sole.position().x(),
            left: swing_sole.position().y() - swing_sole_base_offset.y(),
            turn: swing_sole.orientation().inner.euler_angles().2,
        })
    }
}

fn compensate_arm_motion_with_torso_tilt(
    arm_command: &ArmCommand,
    torso_tilt_compensation_factor: f32,
) -> f32 {
    (arm_command.shoulder_pitch() - FRAC_PI_2) * torso_tilt_compensation_factor
}
