from pathlib import Path

import mujoco as mj
import pandas as pd
import plotly.express as px
from learned_identification.recording import (
    load_recorded_actuator_positions,
    load_recorded_sensor_data,
)
from learned_identification.simulation import simulate_recording

SENSORS = [
    "head.yaw",
    "head.pitch",
    "left_leg.hip_yaw_pitch",
    "left_leg.hip_roll",
    "left_leg.hip_pitch",
    "left_leg.knee_pitch",
    "left_leg.ankle_pitch",
    "left_leg.ankle_roll",
    "right_leg.hip_roll",
    "right_leg.hip_pitch",
    "right_leg.knee_pitch",
    "right_leg.ankle_pitch",
    "right_leg.ankle_roll",
    "left_arm.shoulder_pitch",
    "left_arm.shoulder_roll",
    "left_arm.elbow_yaw",
    "left_arm.elbow_roll",
    "left_arm.wrist_yaw",
    "right_arm.shoulder_pitch",
    "right_arm.shoulder_roll",
    "right_arm.elbow_yaw",
    "right_arm.elbow_roll",
    "right_arm.wrist_yaw",
    "inertial_measurement_unit.angular_velocity.x",
    "inertial_measurement_unit.angular_velocity.y",
    "inertial_measurement_unit.angular_velocity.z",
    "inertial_measurement_unit.linear_acceleration.x",
    "inertial_measurement_unit.linear_acceleration.y",
    "inertial_measurement_unit.linear_acceleration.z",
    "force_sensitive_resistors.left",
    "force_sensitive_resistors.right",
]

recording_path = Path("recording.mcap").resolve().as_posix()
spec_path = Path("../model/scene.xml").resolve().as_posix()

recorded_sensor_data = load_recorded_sensor_data(
    recording_path,
    SENSORS,
)

spec = mj.MjSpec.from_file(spec_path)
recorded_actuator_positions = load_recorded_actuator_positions(
    spec,
    recording_path,
)

simulated_sensor_data = simulate_recording(
    spec,
    recorded_actuator_positions,
    positions=SENSORS[:-8],
    sensors=SENSORS,
    video_path=Path().resolve().as_posix() + "/video.mp4",
)


df = pd.DataFrame(
    {
        "simulated": simulated_sensor_data[-1],
        "recorded": recorded_actuator_positions[-1],
    }
)

print(df.describe())
px.bar(df, barmode="group", x=SENSORS, y=["simulated", "recorded"]).show()
