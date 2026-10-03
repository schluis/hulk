//! A fixed-pose sensor regression, through production geometry/filter/behavior nodes.
//! It checks command continuity, not physical kick execution or locomotion dynamics.
use std::{path::Path, sync::Arc, time::Duration};

use color_eyre::{Result, eyre::ensure};
use hsl_network_messages::PlayerNumber;
use linear_algebra::Point3;
use mujoco_rs::prelude::{MjData, MjSpec, MjtObj};
use ros_z::{
    prelude::*,
    qos::QosDurability,
    time::{Clock, Time},
};
use serde_json::json;
use types::{
    field_dimensions::{FieldDimensions, GlobalFieldSide},
    filtered_game_controller_state::FilteredGameControllerState,
    filtered_game_state::FilteredGameState,
};

use crate::{
    ball_approach::Observer,
    ball_perception::{Parameters, PerceptionIo},
    behavior_inputs::BehaviorInputs,
    geometry_inputs::GeometryInputs,
    robot_io::RobotBinding,
};

pub fn run(output: &Path, parameter_root: &Path, location: &str) -> Result<()> {
    ensure!(
        !output.exists(),
        "regression output already exists: {}",
        output.display()
    );
    std::fs::create_dir_all(output)?;
    // Exercise normal ball behavior even when the development defaults inject
    // a zero-velocity motion command for motion-stack bring-up.
    let behavior_parameters = output.join("behavior-parameters");
    std::fs::create_dir(&behavior_parameters)?;
    std::fs::write(
        behavior_parameters.join("behavior_node.json5"),
        "{ control: { injected_motion_command: null } }\n",
    )?;
    let runtime = tokio::runtime::Runtime::new()?;
    let mut reports = Vec::new();
    for (name, delay, gaps) in [
        ("control", 0.0, false),
        ("brief-gaps", 0.0, true),
        ("delivery-delay", 0.05, false),
        ("gaps-and-delay", 0.05, true),
    ] {
        eprintln!("Ball approach regression: {name}");
        let noise = Parameters {
            center_noise_pixels: 0.0,
            false_positive_probability: 0.0,
            delivery_delay_seconds: delay,
            delivery_jitter_seconds: if delay > 0.0 { 0.015 } else { 0.0 },
            close_dropout_pattern: if gaps { vec![1, 2, 3] } else { Vec::new() },
            ..Default::default()
        };
        let report = run_case(
            runtime.handle(),
            output,
            parameter_root,
            location,
            name,
            noise,
        )?;
        std::fs::write(
            output.join(format!("{name}.json")),
            serde_json::to_vec_pretty(&report)?,
        )?;
        eprintln!("{name}: {}", report["status"]);
        reports.push(report);
    }
    let passed = reports.iter().all(|report| report["status"] == "pass");
    std::fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&json!({
            "passed": passed,
            "scope": "Fixed MuJoCo robot pose and stationary synthetic ball ground truth at 0.7m. Production measured-sensor kinematics, Ground, torso-reference localization, odometry, ball filter, visual selector and complete behavior tree. Commands recorded but not applied; no SDK/physical-kick success claim.",
            "requirement": "Reach close-ball kick mode, exercise all requested gaps, and do not interrupt approach into Stand while the real ball remains present. Missing coverage fails independently of interruption.",
            "cases": reports,
        }))?,
    )?;
    ensure!(
        passed,
        "ball approach regression failed; inspect {}/report.json and MCAP traces",
        output.display()
    );
    Ok(())
}

fn run_case(
    runtime: &tokio::runtime::Handle,
    output: &Path,
    parameter_root: &Path,
    location: &str,
    name: &str,
    noise: Parameters,
) -> Result<serde_json::Value> {
    noise
        .validate()
        .map_err(|error| color_eyre::eyre::eyre!(error))?;
    let clock = Clock::logical(Time::from_nanos(1_000_000_000));
    let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
    let endpoint = format!("tcp/127.0.0.1:{}", listener.local_addr()?.port());
    drop(listener);
    let (
        context,
        _node,
        mut perception,
        geometry,
        behavior,
        low,
        field,
        player,
        game,
        observer,
        recording,
        mut tasks,
    ) = runtime.block_on(async {
        let context = Arc::new(
            ContextBuilder::default()
                .with_namespace("/approach_regression")
                .with_mode("router")
                .disable_multicast_scouting()
                .with_connect_endpoints(std::iter::empty::<&str>())
                .with_listen_endpoints([endpoint.as_str()])
                .with_parameter_layers([
                    parameter_root.join("base"),
                    parameter_root.join("location").join(location),
                    output.join("behavior-parameters"),
                ])
                .with_clock(clock.clone())
                .build()
                .await?,
        );
        let node = context.create_node("sensors").build().await?;
        let perception = PerceptionIo::new(&node, runtime).await?;
        let geometry = GeometryInputs::new(&node).await?;
        let behavior = BehaviorInputs::new(&node, true).await?;
        let low = node
            .publisher::<booster::LowState>("inputs/low_state")
            .build()
            .await?;
        let retained = QosProfile {
            durability: QosDurability::TransientLocal,
            ..Default::default()
        };
        let field = node
            .publisher::<FieldDimensions>("field_dimensions")
            .qos(retained)
            .build()
            .await?;
        let player = node
            .publisher::<PlayerNumber>("player_number")
            .qos(retained)
            .build()
            .await?;
        let game = node
            .publisher::<FilteredGameControllerState>("filtered_game_controller_state")
            .build()
            .await?;
        let observer = Observer::new(&node).await?;
        let mut tasks = tokio::task::JoinSet::new();
        tasks.spawn(fall_detection::run_boxed(context.clone()));
        tasks.spawn(kinematics_provider::run_boxed(context.clone()));
        tasks.spawn(support_foot_estimator::run_boxed(context.clone()));
        tasks.spawn(ground_provider::run_boxed(context.clone()));
        tasks.spawn(camera_matrix_calculator::run_boxed(context.clone()));
        tasks.spawn(odometry::run_boxed(context.clone()));
        tasks.spawn(localization_2d::run_boxed(context.clone()));
        tasks.spawn(ball_filter::run_boxed(context.clone()));
        tasks.spawn(visual_kick_ball_selector::run_boxed(context.clone()));
        tasks.spawn(ball_state_composer::run_boxed(context.clone()));
        tasks.spawn(behavior_node::run_boxed(context.clone()));
        tokio::time::sleep(Duration::from_millis(150)).await;
        if let Some(result) = tasks.try_join_next() {
            return Err(color_eyre::eyre::eyre!(
                "production node failed to start: {result:?}"
            ));
        }
        // The fixed-pose harness has no scene/viewer publishers. All remaining
        // topics must be discoverable after the production nodes have started.
        let topics: Vec<_> = crate::ball_tuning::TOPICS
            .iter()
            .copied()
            .filter(|topic| {
                !matches!(
                    *topic,
                    "simulation/ball_poses_world"
                        | "simulation/ball_velocities_world"
                        | "simulation/obstacle_positions_world"
                        | "simulation/scenario"
                        | "simulation/parameters"
                )
            })
            .collect();
        let recording =
            mcap_recorder::Recording::start(&node, output.join(format!("{name}.mcap")), &topics)
                .await?;
        Ok::<_, color_eyre::Report>((
            context, node, perception, geometry, behavior, low, field, player, game, observer,
            recording, tasks,
        ))
    })?;
    let dimensions = FieldDimensions {
        ball_radius: 0.105,
        ..FieldDimensions::SPL_2025
    };
    let game_state = FilteredGameControllerState {
        game_state: FilteredGameState::Playing {
            ball_is_free: true,
            kick_off: false,
        },
        global_field_side: GlobalFieldSide::Home,
        ..Default::default()
    };
    runtime.block_on(async {
        field.publish(&dimensions).await?;
        player.publish(&PlayerNumber::Three).await?;
        Ok::<_, color_eyre::Report>(())
    })?;
    let mut spec = MjSpec::from_xml(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/k1_robot.xml"))?;
    let mut data = MjData::new(Box::new(spec.compile()?));
    let pitch = data
        .model()
        .name_to_id(MjtObj::mjOBJ_JOINT, "Head_pitch")
        .ok_or_else(|| color_eyre::eyre::eyre!("robot model has no head pitch joint"))?;
    let address = data.model().jnt_qposadr()[pitch] as usize;
    data.qpos_mut()[2] = 0.6;
    data.qpos_mut()[address] = 0.55;
    data.forward();
    let binding = RobotBinding::new(&data, "")?;
    for _ in 0..3000 {
        let time = clock.advance(Duration::from_millis(2))?;
        let observation = binding.observe(&data);
        let ground = binding.ground_to_world(&data);
        let ball = Point3::wrap(binding.point_in_ground(&data, [0.7, 0.0, 0.105]));
        perception.publish(
            time,
            crate::behavior_inputs::ground_to_field(ground, GlobalFieldSide::Home),
            &observation.camera_matrix,
            vec![ball],
            0.105,
            &noise,
        )?;
        runtime.block_on(async {
            low.publish_with_source_time(&observation.low_state, time)
                .await?;
            geometry
                .publish(&observation, time, GlobalFieldSide::Home)
                .await?;
            behavior
                .publish(
                    ground,
                    None,
                    Vec::new(),
                    [0.2; 2],
                    GlobalFieldSide::Home,
                    time,
                )
                .await?;
            game.publish(&game_state).await?;
            behavior.publish_game(&game_state).await?;
            Ok::<_, color_eyre::Report>(())
        })?;
        std::thread::sleep(Duration::from_millis(2));
        ensure!(
            tasks.try_join_next().is_none(),
            "a production node exited during {name}"
        );
    }
    // End behavioral scoring before the intentional sensor shutdown/drain.
    std::thread::sleep(Duration::from_millis(30));
    let summary = observer.snapshot();
    clock.advance(Duration::from_millis(100))?;
    std::thread::sleep(Duration::from_millis(200));
    let written = runtime.block_on(recording.finish())?;
    let bursts = *perception.brief_dropout_counts();
    let coverage = summary.kick_seconds >= 0.2
        && noise
            .close_dropout_pattern
            .iter()
            .all(|n| bursts[*n as usize - 1] > 0);
    let status = if !coverage {
        "insufficient_coverage"
    } else if summary.kick_to_stand > 0 {
        "fail"
    } else {
        "pass"
    };
    tasks.abort_all();
    runtime.block_on(async { while tasks.join_next().await.is_some() {} });
    drop(context);
    Ok(
        json!({ "case": name, "status": status, "noise": noise, "messages": written, "brief_dropout_bursts_by_frames": bursts, "approach": summary }),
    )
}
