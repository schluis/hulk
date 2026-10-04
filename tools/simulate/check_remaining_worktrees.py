#!/usr/bin/env python3
"""Check preserved ball-filter implementations with the current fixed v8 scorer.

Uses a disposable source tree; never modifies experimental worktrees. All inputs
are already inspected development data. This is a fixed-candidate audit, not an
equal-budget search or an independent promotion evaluation.
"""
import argparse
import concurrent.futures
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import tarfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def violations(candidate, baseline, per_clip):
    fields = [
        "correct_track_missing_seconds", "close_range_correct_track_missing_seconds",
        "longest_correct_track_gap_seconds", "false_track_seconds",
        "close_range_position_rmse_metres", "motion_lag_rms_seconds",
        "motion_lag_absolute_seconds",
    ]
    bad = {}
    for key in fields:
        b, c = baseline[key], candidate[key]
        if b is None:
            continue
        margin = (0.01 if key == fields[4] else 0.04 if key.startswith("motion_lag_") else 0) if per_clip else 0
        epsilon = 1024 * math.ulp(1.0) * max(1, baseline["labelled_seconds"] if key.endswith("seconds") and not key.startswith("motion_lag_") else b)
        if c is None or not math.isfinite(c) or c > b + margin + epsilon:
            bad[key] = {"baseline": b, "candidate": c, "allowance": margin}
    return bad


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--resume", action="store_true", help="Reuse validated binaries; rerun fixed evaluations")
    args = parser.parse_args()
    root, out, scratch = args.root.resolve(), args.output.resolve(), args.scratch.resolve()
    out.mkdir(parents=True, exist_ok=args.resume)
    if not args.resume:
        scratch.mkdir(parents=True, exist_ok=False)
        with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(["git", "-C", str(root), "archive", "HEAD"]))) as archive:
            archive.extractall(scratch, filter="data")
    history = root / "logs/ball-improvement-20261003"
    clips = json.loads((history / "development-144-inputs.json").read_text())["train"] + json.loads((history / "sixth-audit-inputs.json").read_text())["recordings"]
    assert len(clips) == len(set(clips)) == 168
    validation = str(history / "fresh-long-occlusion/validation-4243.mcap")
    capture = history / "fresh-approach/baseline.json5"
    manifest = {"protocol": "Fixed candidates; all 168 clips are inspected development data; no promotion claim", "clips": {p: digest(Path(p)) for p in clips}, "capture": {"path": str(capture), "sha256": digest(capture)}, "root_head": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(), "models": {}}
    if args.resume:
        previous = json.loads((out / "manifest.json").read_text())
        for key in ["clips", "capture", "root_head"]:
            assert manifest[key] == previous[key], (key, "changed since build")
        manifest = previous
    jobs = []
    selected = json.loads((history / "method-comparison-training-selection.json").read_text())["methods"]
    models = ["student-t", "imm", "pda", "output-imm", "selection-only", "soft-geometry", "output-reacquisition", "publication-filter"]
    for model in models:
        work = root.parent / ("hulk-ball-" + model)
        if args.resume and model in manifest["models"]:
            entry = manifest["models"][model]
            assert entry["head"] == subprocess.check_output(["git", "-C", str(work), "rev-parse", "HEAD"], text=True).strip()
            assert all(digest(work / path) == value for path, value in entry["source_hashes"].items())
            binary = out / (model + "-tuner")
            assert digest(binary) == entry["binary_sha256"]
            for name, variant in entry["variants"].items():
                path = out / f"{model}-{name}.json"
                assert digest(path) == variant["parameters_sha256"]
                for amplitude in [0.0, 0.1, 0.25, 0.5]:
                    jobs.append((model, name, amplitude, binary, path))
            continue
        entry = {"worktree": str(work), "head": subprocess.check_output(["git", "-C", str(work), "rev-parse", "HEAD"], text=True).strip(), "status": subprocess.check_output(["git", "-C", str(work), "status", "--short"], text=True), "source_hashes": {}}
        for relative in ["crates/nodes/ball_filter/src", "crates/types/src", "tools/ball-filter-tuner/src"]:
            shutil.rmtree(scratch / relative)
            shutil.copytree(work / relative, scratch / relative)
            # copytree preserves historical mtimes. Refresh them so Cargo cannot
            # reuse another branch's dependency artifact at this shared path.
            for path in (scratch / relative).rglob("*"):
                if path.is_file():
                    path.touch()
            for path in sorted((work / relative).rglob("*.rs")):
                entry["source_hashes"][str(path.relative_to(work))] = digest(path)
        # Keep algorithm-specific parameter encode/decode intact. Only fixed
        # evaluations are used; scoring/recording are shared byte-for-byte.
        for name in ["scoring.rs", "recording.rs"]:
            shutil.copyfile(root / "tools/ball-filter-tuner/src" / name, scratch / "tools/ball-filter-tuner/src" / name)
        lib = scratch / "tools/ball-filter-tuner/src/lib.rs"
        text = lib.read_text().replace("args.trials > 0 && args.penalty_metres.is_finite()", "args.penalty_metres.is_finite()")
        if "pub field_prior_wobble_metres" not in text:
            text = text.replace("    pub trials: usize,", "    pub trials: usize,\n    #[arg(long, default_value_t = 0.0)]\n    pub field_prior_wobble_metres: f32,")
            text = text.replace("    replay_matches_live: bool,", "    replay_matches_live: bool,\n    field_prior_wobble_metres: f32,\n    field_prior_stressed_cycles: usize,")
            text = text.replace("let train = read(&args.train)?;", "let mut train = read(&args.train)?;").replace("let validation = read(&args.validation)?;", "let mut validation = read(&args.validation)?;")
            text = text.replace("    let baseline: BallFilterParameters = match", "    ensure!(args.field_prior_wobble_metres.is_finite() && args.field_prior_wobble_metres >= 0.0, \"invalid prior wobble\");\n    let field_prior_stressed_cycles = train.iter_mut().chain(&mut validation).map(|r| r.apply_prior_wobble(args.field_prior_wobble_metres)).sum();\n    let baseline: BallFilterParameters = match")
            text = text.replace("        replay_matches_live: true,", "        replay_matches_live: true,\n        field_prior_wobble_metres: args.field_prior_wobble_metres,\n        field_prior_stressed_cycles,")
        # The scorer contains the actual v8 guards; remove stale v7 prose.
        lines = text.splitlines()
        current_policy = next(line for line in (root / "tools/ball-filter-tuner/src/lib.rs").read_text().splitlines() if line.strip().startswith("continuity_policy: \""))
        text = "\n".join(current_policy if line.strip().startswith("continuity_policy: \"") else line for line in lines) + "\n"
        lib.write_text(text)
        entry["effective_tuner_hashes"] = {p.name: digest(p) for p in lib.parent.glob("*.rs")}
        with tarfile.open(out / (model + "-source.tar.gz"), "w:gz") as archive:
            for relative in ["crates/nodes/ball_filter", "crates/types", "tools/ball-filter-tuner"]:
                archive.add(scratch / relative, arcname=relative)
        cargo = ["cargo", "+1.98.1"]
        common = ["--offline", "--release", "--manifest-path", str(scratch / "Cargo.toml"), "--target-dir", str(root / "target")]
        for action, options in [("build", ["-p", "ball-filter-tuner", "--bin", "ball-filter-tuner"]), ("test", ["-p", "ball_filter", "-p", "ball-filter-tuner", "--lib"]), ("clippy", ["-p", "ball_filter", "-p", "ball-filter-tuner", "--lib", "--", "-D", "warnings"])]:
            with (out / f"{model}-{action}.log").open("w") as log:
                result = subprocess.run(cargo + [action] + common + options, cwd=scratch, stdout=log, stderr=subprocess.STDOUT)
            entry[action + "_returncode"] = result.returncode
            print(model, action, result.returncode, flush=True)
            if action == "build" and result.returncode:
                raise RuntimeError(f"{model} build failed; see log")
        binary = out / (model + "-tuner")
        shutil.copyfile(root / "target/release/ball-filter-tuner", binary)
        binary.chmod(0o755)
        entry["binary_sha256"] = digest(binary)
        variants = {"disabled": json.loads(capture.read_text())}
        if model in selected:
            variants["historical-selected"] = selected[model]["parameters"]
            if model == "imm":
                variants["enabled-probe"] = dict(variants["disabled"], imm_transition_rate=1.0)
        elif model == "publication-filter":
            variants["production"] = json.loads((history / "candidate144-before-sixth-audit.json").read_text())
        else:
            grid = {"output-imm": "output-imm-grid-72", "selection-only": "selection-only-grid-72", "soft-geometry": "soft-geometry-v4-grid-72", "output-reacquisition": "output-guard-v5-grid-72"}[model]
            summary = json.loads((history / grid / "summary.json").read_text())
            if summary["selected"]:
                variants["historical-selected"] = json.loads(Path(summary["selected"]["parameters"]).read_text())
            variants["historical-best-loss-probe"] = json.loads(Path(summary["results"][0]["parameters"]).read_text())
        entry["variants"] = {}
        for name, config in variants.items():
            path = out / f"{model}-{name}.json"
            path.write_text(json.dumps(config, indent=2) + "\n")
            entry["variants"][name] = {"parameters_sha256": digest(path)}
            for amplitude in [0.0, 0.1, 0.25, 0.5]:
                jobs.append((model, name, amplitude, binary, path))
        manifest["models"][model] = entry
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def evaluate(job):
        model, name, amplitude, binary, parameters = job
        destination = out / f"{model}-{name}-stress-{amplitude}"
        cmd = [str(binary), "--train", *clips, "--validation", validation, "--parameters", str(capture), "--evaluation-parameters", str(parameters), "--trials", "0", "--field-prior-wobble-metres", str(amplitude), "--output", str(destination)]
        with Path(str(destination) + ".log").open("w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads((destination / "report.json").read_text())
        assert report["replay_matches_live"] and report["objective"]["version"] == "single_ball_close_accuracy_v8"
        assert report["training_recordings"] == clips and report["trials"] == 0
        assert (report["field_prior_stressed_cycles"] > 0) == (amplitude > 0)
        assert report["training"]["baseline"] == report["training"]["optimized"]
        print("evaluated", model, name, amplitude, flush=True)
        return (model, name, amplitude), report

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        reports = dict(pool.map(evaluate, jobs))
    verdict = {"protocol": manifest["protocol"], "recordings": len(clips), "results": []}
    for (model, name, amplitude), report in reports.items():
        references = {"original": reports[(model, "disabled", amplitude)], "production": reports[("publication-filter", "production", amplitude)]}
        item = {"model": model, "variant": name, "amplitude": amplitude, "score": report["training"]["baseline"], "comparisons": {}}
        for label, reference in references.items():
            bad = {}
            for path, c, b in zip(clips, report["training_per_recording"], reference["training_per_recording"], strict=True):
                v = violations(c["baseline"], b["baseline"], True)
                if v:
                    bad[path] = v
            v = violations(item["score"], reference["training"]["baseline"], False)
            if v:
                bad["aggregate"] = v
            item["comparisons"][label] = {"passes_all_guards": not bad, "failures": bad}
        verdict["results"].append(item)
    # Disabled controls must reproduce a common estimator, not just pass guards.
    for amplitude in [0.0, 0.1, 0.25, 0.5]:
        reference = reports[("publication-filter", "disabled", amplitude)]
        for model in models:
            assert reports[(model, "disabled", amplitude)]["training_per_recording"] == reference["training_per_recording"], (model, amplitude, "disabled drift")
    frozen = json.loads((history / "final168-verification/report.json").read_text())
    reproduced = reports[("publication-filter", "production", 0.0)]
    assert reproduced["training_per_recording"] == frozen["training_per_recording"] + frozen["validation_per_recording"]
    verdict["disabled_controls_identical"] = True
    verdict["production_scores_reproduce_all_168"] = True
    verdict["all_builds_tests_clippy_pass"] = all(
        entry[action + "_returncode"] == 0
        for entry in manifest["models"].values()
        for action in ["build", "test", "clippy"]
    )
    verdict["fixed_evaluations"] = len(reports)
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print("Completed", out, flush=True)


if __name__ == "__main__":
    main()
