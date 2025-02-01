use openvino::{ElementType, Tensor};
use path_serde::{PathDeserialize, PathIntrospect, PathSerialize};
use serde::{Deserialize, Serialize};
use types::{
    joints::{arm::ArmJoints, body::BodyJoints, leg::LegJoints, Joints},
    motion_command::KickVariant,
    motor_commands::MotorCommands,
    sensor_data::SensorData,
    step::Step,
    support_foot::Side,
};

use crate::{
    parameters::Stiffnesses, step_plan::StepPlan, step_state::StepState, stiffness::Stiffness as _,
    Context, WalkTransition,
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
    pub fn compute_commands(&self, context: &mut Context) -> MotorCommands<BodyJoints> {
        let mut tensor = Tensor::new(
            ElementType::F32,
            &context.network.get_input().unwrap().get_shape().unwrap(),
        )
        .unwrap();

        load_into_scratchpad(tensor.get_data_mut().unwrap(), context.sensor_data);

        let mut infer_request = context.network.create_infer_request().unwrap();
        infer_request.set_input_tensor(&tensor).unwrap();
        infer_request.infer().unwrap();

        let prediction = infer_request.get_output_tensor_by_index(0).unwrap();
        let prediction = prediction.get_data::<f32>().unwrap();

        motor_commands_from(prediction, 0.15)

        // let plan = StepPlan::stand(context);
        // let zero_step = StepState::new(plan);
        //
        // zero_step.compute_joints(context).apply_stiffness(
        //     context.parameters.stiffnesses.leg_stiffness_stand,
        //     context.parameters.stiffnesses.arm_stiffness,
        // )
    }

    pub fn tick(&mut self, _context: &Context) {}
}

fn motor_commands_from(prediction: &[f32], stiffness: f32) -> MotorCommands<BodyJoints> {
    assert!(stiffness <= 0.5, "please don't fry the motors");

    MotorCommands {
        positions: BodyJoints {
            left_arm: ArmJoints {
                shoulder_pitch: prediction[11],
                shoulder_roll: prediction[12],
                elbow_yaw: prediction[13],
                elbow_roll: prediction[14],
                wrist_yaw: prediction[15],
                hand: 0.0f32,
            },
            right_arm: ArmJoints {
                shoulder_pitch: prediction[16],
                shoulder_roll: prediction[17],
                elbow_yaw: prediction[18],
                elbow_roll: prediction[19],
                wrist_yaw: prediction[20],
                hand: 0.0f32,
            },
            left_leg: LegJoints {
                ankle_pitch: prediction[4],
                ankle_roll: prediction[5],
                hip_pitch: prediction[2],
                hip_roll: prediction[1],
                hip_yaw_pitch: prediction[0],
                knee_pitch: prediction[3],
            },
            right_leg: LegJoints {
                ankle_pitch: prediction[9],
                ankle_roll: prediction[10],
                hip_pitch: prediction[7],
                hip_roll: prediction[6],
                hip_yaw_pitch: prediction[0],
                knee_pitch: prediction[8],
            },
        },
        stiffnesses: BodyJoints::fill(stiffness),
    }
}

fn load_into_scratchpad(scratchpad: &mut [f32], sensor_data: &SensorData) {
    scratchpad.copy_from_slice(&[
        sensor_data.positions.head.yaw,
        sensor_data.positions.head.pitch,
        sensor_data.positions.left_leg.hip_yaw_pitch,
        sensor_data.positions.left_leg.hip_roll,
        sensor_data.positions.left_leg.hip_pitch,
        sensor_data.positions.left_leg.knee_pitch,
        sensor_data.positions.left_leg.ankle_pitch,
        sensor_data.positions.left_leg.ankle_roll,
        sensor_data.positions.right_leg.hip_roll,
        sensor_data.positions.right_leg.hip_pitch,
        sensor_data.positions.right_leg.knee_pitch,
        sensor_data.positions.right_leg.ankle_pitch,
        sensor_data.positions.right_leg.ankle_roll,
        sensor_data.positions.left_arm.shoulder_pitch,
        sensor_data.positions.left_arm.shoulder_roll,
        sensor_data.positions.left_arm.elbow_yaw,
        sensor_data.positions.left_arm.elbow_roll,
        sensor_data.positions.left_arm.wrist_yaw,
        sensor_data.positions.right_arm.shoulder_pitch,
        sensor_data.positions.right_arm.shoulder_roll,
        sensor_data.positions.right_arm.elbow_yaw,
        sensor_data.positions.right_arm.elbow_roll,
        sensor_data.positions.right_arm.wrist_yaw,
        sensor_data
            .inertial_measurement_unit
            .angular_velocity
            .inner
            .x,
        sensor_data
            .inertial_measurement_unit
            .angular_velocity
            .inner
            .y,
        sensor_data
            .inertial_measurement_unit
            .angular_velocity
            .inner
            .z,
        sensor_data
            .inertial_measurement_unit
            .linear_acceleration
            .inner
            .x,
        sensor_data
            .inertial_measurement_unit
            .linear_acceleration
            .inner
            .y,
        sensor_data
            .inertial_measurement_unit
            .linear_acceleration
            .inner
            .z,
        sensor_data.force_sensitive_resistors.left.mean(),
        sensor_data.force_sensitive_resistors.right.mean(),
    ]);

    // scratchpad.copy_from_slice(&[
    //     2.17833254e-06,
    //     1.46821881e-04,
    //     2.00733336e-03,
    //     4.31524352e-03,
    //     -3.08132262e-01,
    //     8.53828172e-01,
    //     -5.39959373e-01,
    //     -4.25303940e-03,
    //     -4.24714955e-03,
    //     -3.08359738e-01,
    //     8.53640055e-01,
    //     -5.39560976e-01,
    //     4.20299377e-03,
    //     1.57084739e+00,
    //     9.45262874e-02,
    //     -1.56998563e+00,
    //     -3.49070734e-02,
    //     -2.48942127e-06,
    //     1.57089481e+00,
    //     -9.45336957e-02,
    //     1.56998560e+00,
    //     3.49013771e-02,
    //     2.47832502e-06,
    //     -7.84287842e-03,
    //     -2.97802196e-03,
    //     1.15524255e-03,
    //     -1.02747153e+00,
    //     1.24009788e-01,
    //     1.04023793e+01,
    //     5.77711950e-01,
    //     4.75285076e-01,
    // ]);

    assert_eq!(
        scratchpad.len(),
        31,
        "last time we checked there should be 31 sensors in the observation space"
    );
}
