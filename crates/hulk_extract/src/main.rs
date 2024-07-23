#![recursion_limit = "256"]
use crate::execution::Replayer;
use chrono::{DateTime, Utc};
use color_eyre::{
    eyre::{Result, WrapErr},
    install,
};
use hardware::{
    ActuatorInterface, CameraInterface, IdInterface, MicrophoneInterface, NetworkInterface,
    PathsInterface, RecordingInterface, SensorInterface, SpeakerInterface, TimeInterface,
};
use polars::prelude::*;
use std::fs::create_dir_all;
use std::time::{SystemTime, UNIX_EPOCH};
use std::{env::args, path::PathBuf, sync::Arc};
use types::{
    audio::SpeakerRequest,
    camera_position::CameraPosition,
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

    let mut export_dataframe = DataFrame::default();

    let control_receiver = replayer.control_receiver();
    // let vision_top_receiver = replayer.vision_top_receiver();
    // let vision_bottom_receiver = replayer.vision_bottom_receiver();

    for (instance_name, mut receiver) in [
        ("Control", control_receiver),
        // ("VisionTop", vision_top_receiver),
        // ("VisionBottom", vision_bottom_receiver),
    ] {
        let mut current_dataframe = DataFrame::default();

        let output_folder = &output_folder.join(instance_name);
        create_dir_all(output_folder).expect("failed to create output folder");

        let unknown_indices_error_message =
            format!("could not find recording indices for `{instance_name}`");
        let timings: Vec<_> = replayer
            .get_recording_indices()
            .get(instance_name)
            .expect(&unknown_indices_error_message)
            .iter()
            .collect();

        export_dataframe.insert_column(
            0,
            timings
                .iter()
                .map(|timing| {
                    let datetime: DateTime<Utc> = timing.timestamp.into();
                    datetime.format("%d/%m/%Y %T").to_string()
                })
                .collect::<Series>(),
        );

        // for timing in timings {
        //     let frame = replayer
        //         .get_recording_indices_mut()
        //         .get_mut(instance_name)
        //         .map(|index| {
        //             index
        //                 .find_latest_frame_up_to(timing.timestamp)
        //                 .expect("failed to find latest frame")
        //         })
        //         .expect(&unknown_indices_error_message);
        //
        //     if let Some(frame) = frame {
        //         replayer
        //             .replay(instance_name, frame.timing.timestamp, &frame.data)
        //             .expect("failed to replay frame");
        //
        //         let (_, database) = &*receiver.borrow_and_mark_as_seen();
        //
        //         let output_file = output_folder.join(format!(
        //             "{}.png",
        //             frame
        //                 .timing
        //                 .timestamp
        //                 .duration_since(UNIX_EPOCH)
        //                 .unwrap()
        //                 .as_secs()
        //         ));
        //
        //         database
        //             .main_outputs
        //             .image
        //             .save_to_ycbcr_444_file(output_file)
        //             .expect("failed to write file");
        //     }
        // }
    }

    let mut file = std::fs::File::create("test.csv").unwrap();
    CsvWriter::new(&mut file)
        .finish(&mut export_dataframe)
        .unwrap();

    Ok(())
}
