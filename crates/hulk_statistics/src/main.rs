#![recursion_limit = "256"]
use crate::execution::Replayer;
use chrono::{DateTime, Utc};
use color_eyre::{
    eyre::{Result, WrapErr},
    install,
};
use hardware::{
    ActuatorInterface, CameraInterface, IdInterface, MicrophoneInterface, NetworkInterface,
    PathsInterface, RecordingInterface, SensorInterface, SpeakerInterface,
};
use polars::prelude::*;
use std::fs::create_dir_all;
use std::{env::args, path::PathBuf, sync::Arc};
use types::{
    audio::SpeakerRequest,
    camera_position::CameraPosition,
    fall_state::{Direction, FallState, Kind, Side},
    hardware::{Ids, Paths},
    joints::Joints,
    led::Leds,
    messages::{IncomingMessage, OutgoingMessage},
    samples::Samples,
    sensor_data::SensorData,
    ycbcr422_image::YCbCr422Image,
};

pub trait HardwareInterface:
    ActuatorInterface
    + CameraInterface
    + IdInterface
    + MicrophoneInterface
    + NetworkInterface
    + PathsInterface
    + RecordingInterface
    + SensorInterface
    + SpeakerInterface
{
}

include!(concat!(env!("OUT_DIR"), "/generated_code.rs"));

struct ReplayerHardwareInterface {
    ids: Ids,
}

impl ActuatorInterface for ReplayerHardwareInterface {
    fn write_to_actuators(
        &self,
        _positions: Joints<f32>,
        _stiffnesses: Joints<f32>,
        _leds: Leds,
    ) -> Result<()> {
        Ok(())
    }
}

impl CameraInterface for ReplayerHardwareInterface {
    fn read_from_camera(&self, _camera_position: CameraPosition) -> Result<YCbCr422Image> {
        panic!("Replayer cannot produce data from hardware")
    }
}

impl IdInterface for ReplayerHardwareInterface {
    fn get_ids(&self) -> Ids {
        self.ids.clone()
    }
}

impl MicrophoneInterface for ReplayerHardwareInterface {
    fn read_from_microphones(&self) -> Result<Samples> {
        panic!("Replayer cannot produce data from hardware")
    }
}

impl NetworkInterface for ReplayerHardwareInterface {
    fn read_from_network(&self) -> Result<IncomingMessage> {
        panic!("Replayer cannot produce data from hardware")
    }

    fn write_to_network(&self, _message: OutgoingMessage) -> Result<()> {
        Ok(())
    }
}

impl PathsInterface for ReplayerHardwareInterface {
    fn get_paths(&self) -> Paths {
        Paths {
            motions: "etc/motions".into(),
            neural_networks: "etc/neural_networks".into(),
            sounds: "etc/sounds".into(),
        }
    }
}

impl RecordingInterface for ReplayerHardwareInterface {
    fn should_record(&self) -> bool {
        false
    }

    fn set_whether_to_record(&self, _enable: bool) {}
}

impl SensorInterface for ReplayerHardwareInterface {
    fn read_from_sensors(&self) -> Result<SensorData> {
        panic!("Replayer cannot produce data from hardware")
    }
}

impl SpeakerInterface for ReplayerHardwareInterface {
    fn write_to_speakers(&self, _request: SpeakerRequest) {}
}

impl HardwareInterface for ReplayerHardwareInterface {}

fn main() -> Result<()> {
    install()?;

    let replay_path = args()
        .nth(1)
        .expect("expected replay path as first parameter");

    let output_folder = PathBuf::from(
        args()
            .nth(2)
            .expect("expected output path as second parameter"),
    );

    let path_buf = PathBuf::from(replay_path.clone());
    let nao_number = path_buf.parent().unwrap().file_name().unwrap();

    let parameters_directory = args().nth(3).unwrap_or(replay_path.clone());
    let ids = Ids {
        body_id: "replayer".into(),
        head_id: "replayer".into(),
    };
    let hardware_interface = ReplayerHardwareInterface { ids: ids.clone() };

    let mut replayer = Replayer::new(
        Arc::new(hardware_interface),
        parameters_directory,
        ids,
        replay_path,
    )
    .wrap_err("failed to create image extractor")?;

    let mut control_receiver = replayer.control_receiver();
    let instance_name = "Control";

    create_dir_all(output_folder.clone()).expect("failed to create output folder");

    let unknown_indices_error_message =
        format!("could not find recording indices for `{instance_name}`");
    let timings: Vec<_> = replayer
        .get_recording_indices()
        .get(instance_name)
        .expect(&unknown_indices_error_message)
        .iter()
        .collect();

    let timings_series = Series::new(
        "Timestamps",
        timings
            .iter()
            .map(|timing| {
                let datetime: DateTime<Utc> = timing.timestamp.into();
                datetime.format("%d/%m/%Y %T").to_string()
            })
            .collect::<Series>(),
    );

    let mut fall_state = vec![];

    for timing in timings {
        let frame = replayer
            .get_recording_indices_mut()
            .get_mut(instance_name)
            .map(|index| {
                index
                    .find_latest_frame_up_to(timing.timestamp)
                    .expect("failed to find latest frame")
            })
            .expect(&unknown_indices_error_message);

        if let Some(frame) = frame {
            replayer
                .replay(instance_name, frame.timing.timestamp, &frame.data)
                .expect("failed to replay frame");

            let (_, database) = &*control_receiver.borrow_and_mark_as_seen();

            match database.main_outputs.fall_state {
                FallState::Upright => fall_state.push("upright"),
                FallState::Falling {
                    start_time: _,
                    direction,
                } => match direction {
                    Direction::Forward { side } => match side {
                        Side::Left => fall_state.push("falling forward left"),
                        Side::Right => fall_state.push("falling forward right"),
                    },
                    Direction::Backward { side } => match side {
                        Side::Left => fall_state.push("falling backward left"),
                        Side::Right => fall_state.push("falling backward right"),
                    },
                },
                FallState::Fallen { kind } => match kind {
                    Kind::FacingDown => fall_state.push("fallen down"),
                    Kind::FacingUp => fall_state.push("fallen up"),
                    Kind::Sitting => fall_state.push("sitting"),
                },
                FallState::StandingUp {
                    start_time: _,
                    kind,
                } => match kind {
                    Kind::FacingDown => fall_state.push("standing up facing down"),
                    Kind::FacingUp => fall_state.push("standing up facing up"),
                    Kind::Sitting => fall_state.push("standing up sitting"),
                },
            }
        }
    }

    let fall_state_series = Series::new("Fall State", fall_state);

    let mut control_dataframe = df![
        "timestamps" => timings_series,
        "fall state " => fall_state_series]
    .unwrap();

    let mut file =
        std::fs::File::create(output_folder.join(format!("{}.csv", instance_name))).unwrap();
    CsvWriter::new(&mut file)
        .finish(&mut control_dataframe)
        .unwrap();

    Ok(())
}
