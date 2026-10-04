//! Publish existing offline search reports to Twix without restarting workers.
use clap::Parser;
use color_eyre::{
    Result,
    eyre::{Context as _, ensure},
};
use ros_z::prelude::*;
use ros_z::qos::{QosDurability, QosHistory};
use serde_json::Value;
use std::{
    fs,
    path::{Path, PathBuf},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use types::ball_filter_tuning::{
    Metrics, NAMESPACE, PROGRESS_TOPIC, Progress, ROUTER, RemoteProgress, RemoteWorker,
    SearchProgress,
};

#[derive(Parser)]
struct Args {
    /// Directory containing worker output directories and their report.json files.
    directory: PathBuf,
    /// A completed report defining the training data, objective and baseline to compare.
    #[arg(long)]
    reference_report: PathBuf,
    /// Only include worker directories whose names start with these prefixes.
    #[arg(long, required = true)]
    prefix: Vec<String>,
    #[arg(long, default_value = ROUTER)]
    listen: String,
    /// Print one snapshot without opening a ROS-Z endpoint.
    #[arg(long)]
    once: bool,
}

fn comparable(report: &Value, reference: &Value) -> bool {
    [
        "objective",
        "training_recordings",
        "validation_recordings",
        "baseline_parameters",
        "reference_frame",
        "reference_topic",
        "namespace",
        "penalty_metres",
    ]
    .iter()
    .all(|key| report.get(key).is_some() && report.get(key) == reference.get(key))
}

fn search(report: &Value) -> Result<SearchProgress> {
    let baseline: Metrics = serde_json::from_value(report["training"]["baseline"].clone())?;
    let best: Metrics = serde_json::from_value(report["training"]["optimized"].clone())?;
    ensure!(
        baseline.loss.is_finite() && best.loss.is_finite(),
        "non-finite training loss"
    );
    let trials = report["trials"]
        .as_u64()
        .ok_or_else(|| color_eyre::eyre::eyre!("missing trials"))?;
    Ok(SearchProgress {
        reference_frame: report["reference_frame"]
            .as_str()
            .unwrap_or_default()
            .into(),
        trial: trials,
        trials,
        // Reports do not record the trial where the best candidate was found.
        best_trial: 0,
        baseline,
        best,
        best_parameters: serde_json::from_value(report["optimized_parameters"].clone())?,
        validation_baseline: Some(serde_json::from_value(
            report["validation"]["baseline"].clone(),
        )?),
        validation_best: Some(serde_json::from_value(
            report["validation"]["optimized"].clone(),
        )?),
    })
}

fn worker_outputs(root: &Path, prefixes: &[String]) -> Result<Vec<(u64, PathBuf)>> {
    let mut active = Vec::new();
    for entry in fs::read_dir("/proc")? {
        let entry = entry?;
        let Ok(pid) = entry.file_name().to_string_lossy().parse::<u64>() else {
            continue;
        };
        let Ok(command) = fs::read(entry.path().join("cmdline")) else {
            continue;
        };
        let args: Vec<_> = command
            .split(|byte| *byte == 0)
            .map(String::from_utf8_lossy)
            .collect();
        let Some(output) = args
            .windows(2)
            .find(|pair| pair[0] == "--output")
            .map(|pair| PathBuf::from(pair[1].as_ref()))
        else {
            continue;
        };
        let output = if output.is_absolute() {
            output
        } else {
            let Ok(cwd) = fs::read_link(entry.path().join("cwd")) else {
                continue;
            };
            cwd.join(output)
        };
        if output.parent() == Some(root)
            && output.file_name().is_some_and(|name| {
                prefixes
                    .iter()
                    .any(|prefix| name.to_string_lossy().starts_with(prefix))
            })
        {
            active.push((pid, output));
        }
    }
    active.sort_by_key(|(pid, _)| *pid);
    Ok(active)
}

fn snapshot(root: &Path, prefixes: &[String], reference: &Value) -> Result<Progress> {
    let active = worker_outputs(root, prefixes)?;
    let mut remote = RemoteProgress {
        host: fs::read_to_string("/proc/sys/kernel/hostname")?
            .trim()
            .into(),
        workers: active
            .iter()
            .map(|(pid, path)| RemoteWorker {
                run: path.file_name().unwrap().to_string_lossy().into(),
                worker: *pid,
                round: 0,
                status: "Running; trial count available when report completes".into(),
            })
            .collect(),
        ..Default::default()
    };
    let mut best: Option<SearchProgress> = None;
    let mut completed = 0;
    let mut skipped = 0;
    let mut invalid = 0;
    let mut entries = fs::read_dir(root)?.collect::<std::io::Result<Vec<_>>>()?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        if !entry.file_type()?.is_dir()
            || !prefixes
                .iter()
                .any(|prefix| entry.file_name().to_string_lossy().starts_with(prefix))
        {
            continue;
        }
        let path = entry.path().join("report.json");
        if !path.exists() {
            continue;
        }
        let parsed = (|| -> Result<(Value, SearchProgress)> {
            let report: Value = serde_json::from_slice(&fs::read(&path)?)?;
            let state = search(&report)?;
            Ok((report, state))
        })();
        let Ok((report, state)) = parsed else {
            invalid += 1;
            continue;
        };
        if !comparable(&report, reference) {
            skipped += 1;
            continue;
        }
        completed += 1;
        remote.completed_trials += state.trials;
        if best
            .as_ref()
            .is_none_or(|old| state.best.loss < old.best.loss)
        {
            remote.best_candidate = entry.path().display().to_string();
            best = Some(state);
        }
    }
    Ok(Progress {
        status: format!(
            "Offline search: {} active workers, {completed} completed rounds, {} completed trials",
            active.len(),
            remote.completed_trials
        ),
        phase: "Monitoring completed worker reports".into(),
        output_directory: root.display().to_string(),
        search: best,
        remote: Some(remote),
        remote_updated_unix_seconds: Some(
            SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64(),
        ),
        live_status: Some(format!(
            "Read-only monitor; updates after each completed round. Best-trial index is unavailable (shown as 0). {skipped} incompatible reports excluded; {invalid} unreadable/incomplete reports will be retried."
        )),
        viewer_status: Some("No live simulator or 3D preview is attached to this monitor.".into()),
        ..Default::default()
    })
}

fn main() -> Result<()> {
    color_eyre::install()?;
    let args = Args::parse();
    let root = args.directory.canonicalize()?;
    let reference: Value = serde_json::from_slice(&fs::read(&args.reference_report)?)?;
    search(&reference).wrap_err("invalid reference report")?;
    ensure!(
        comparable(&reference, &reference),
        "reference report lacks comparison metadata"
    );
    if args.once {
        println!(
            "{}",
            serde_json::to_string_pretty(&snapshot(&root, &args.prefix, &reference)?)?
        );
        return Ok(());
    }
    tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build()?.block_on(async {
        let context = ContextBuilder::default().with_mode("router").disable_multicast_scouting()
            .with_connect_endpoints(std::iter::empty::<&str>()).with_listen_endpoints([args.listen.as_str()]).build().await?;
        let node = context.create_node("offline_ball_filter_monitor").with_namespace(NAMESPACE).build().await?;
        let publisher = node.publisher::<Progress>(PROGRESS_TOPIC).qos(QosProfile {
            durability: QosDurability::TransientLocal, history: QosHistory::from_depth(1), ..Default::default()
        }).build().await?;
        eprintln!("Publishing {} on {NAMESPACE}/{PROGRESS_TOPIC} via {}", root.display(), args.listen);
        let mut tick = tokio::time::interval(Duration::from_secs(2));
        let mut previous = String::new();
        loop {
            tokio::select! {
                _ = tokio::signal::ctrl_c() => return Ok(()),
                _ = tick.tick() => {
                    // Do not refresh timestamps if reading worker state fails.
                    let progress = snapshot(&root, &args.prefix, &reference)?;
                    if previous != progress.status { eprintln!("{}", progress.status); previous.clone_from(&progress.status); }
                    publisher.publish(&progress).await?;
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn different_datasets_or_objectives_are_never_ranked_together() {
        let mut reference = serde_json::json!({});
        for key in [
            "objective",
            "training_recordings",
            "validation_recordings",
            "baseline_parameters",
            "reference_frame",
            "reference_topic",
            "namespace",
            "penalty_metres",
        ] {
            reference[key] = serde_json::json!(key);
        }
        assert!(comparable(&reference, &reference));
        for key in [
            "objective",
            "training_recordings",
            "baseline_parameters",
            "reference_frame",
        ] {
            let mut other = reference.clone();
            other[key] = Value::Null;
            assert!(!comparable(&other, &reference));
        }
        assert!(!comparable(&serde_json::json!({}), &serde_json::json!({})));
    }
    #[test]
    fn snapshot_counts_rounds_and_selects_training_best_not_holdout_best() {
        let directory = tempfile::tempdir().unwrap();
        let metrics = serde_json::to_value(Metrics::default()).unwrap();
        let mut reference = serde_json::json!({
            "objective": "test", "training_recordings": ["train"],
            "validation_recordings": ["holdout"], "baseline_parameters": {},
            "reference_frame": "ground", "reference_topic": "truth", "namespace": "",
            "penalty_metres": 2.0, "trials": 100,
            "optimized_parameters": json5::from_str::<types::parameters::BallFilterParameters>(include_str!("../../../../etc/parameters/base/ball_filter.json5")).unwrap(),
            "training": {"baseline": metrics, "optimized": metrics},
            "validation": {"baseline": metrics, "optimized": metrics}
        });
        reference["training"]["optimized"]["loss"] = serde_json::json!(2.0);
        let write = |name: &str, report: &Value| {
            let path = directory.path().join(name);
            fs::create_dir(&path).unwrap();
            fs::write(
                path.join("report.json"),
                serde_json::to_vec(report).unwrap(),
            )
            .unwrap();
        };
        write("worker-01", &reference);
        let mut better = reference.clone();
        better["training"]["optimized"]["loss"] = serde_json::json!(1.0);
        better["validation"]["optimized"]["loss"] = serde_json::json!(5.0);
        write("worker-02", &better);
        let mut incompatible = better.clone();
        incompatible["training_recordings"] = serde_json::json!(["different"]);
        write("worker-03", &incompatible);
        fs::create_dir(directory.path().join("worker-04")).unwrap();
        fs::write(directory.path().join("worker-04/report.json"), "{").unwrap();
        let state = snapshot(directory.path(), &["worker-".into()], &reference).unwrap();
        let remote = state.remote.unwrap();
        assert_eq!(remote.completed_trials, 200);
        assert!(remote.best_candidate.ends_with("worker-02"));
        assert_eq!(state.search.unwrap().best.loss, 1.0);
        assert!(
            state
                .live_status
                .unwrap()
                .contains("1 incompatible reports excluded; 1 unreadable")
        );
    }
    /// Run explicitly against a monitor to verify the same typed topic Twix observes.
    #[test]
    #[ignore = "requires a running monitor at MONITOR_TEST_ENDPOINT"]
    fn receives_live_progress() {
        let endpoint = std::env::var("MONITOR_TEST_ENDPOINT").unwrap();
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap()
            .block_on(async {
                let context = ContextBuilder::default()
                    .with_mode("client")
                    .with_router_endpoint(endpoint)
                    .unwrap()
                    .build()
                    .await
                    .unwrap();
                let node = context
                    .create_node("monitor_verification")
                    .with_namespace(NAMESPACE)
                    .build()
                    .await
                    .unwrap();
                let subscriber = node
                    .subscriber::<Progress>(PROGRESS_TOPIC)
                    .build()
                    .await
                    .unwrap();
                let sample = tokio::time::timeout(Duration::from_secs(15), subscriber.recv())
                    .await
                    .unwrap()
                    .unwrap();
                assert!(sample.remote.is_some());
                assert!(sample.search.is_some());
                println!("Received: {}", sample.status);
            });
    }
}
