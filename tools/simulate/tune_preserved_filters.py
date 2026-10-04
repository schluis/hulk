#!/usr/bin/env python3
"""Search enabled methods against unchanged v8 guards; retain rejected evidence."""
import argparse
import concurrent.futures
import copy
import itertools
import json
from pathlib import Path
import subprocess

from check_remaining_worktrees import digest, violations


def configurations(model, baseline):
    spaces = {
        "student-t": ("student_t_robustness", [.01, .05, .1, .25, .5, 1.]),
        "imm": ("imm_transition_rate", [.001, .01, .05, .1, .25, 1.]),
        "pda": ("association_temperature", [.01, .05, .1, .25, .5, 1., 2., 4.]),
        "output-imm": ("output_imm_blend", [.005, .01, .025, .05, .1, .25]),
        "selection-only": ("hypothesis_uncertainty_weight", [.001, .01, .05, .1, .25, .5, 1., 2.]),
        "soft-geometry": ("selection_size_consistency_weight", [.001, .01, .05, .1, .25, .5, 1., 2.]),
        "output-reacquisition": ("output_reacquisition_blend", [.005, .01, .025, .05, .1, .25, .5, 1.]),
    }
    key, values = spaces[model]
    for value, noise in itertools.product(values, [3., 4., 5., 6., 8.]):
        config = copy.deepcopy(baseline)
        config[key] = value
        config["noise"]["detection_noise"] = [noise, noise]
        if model == "output-imm":
            config.update(imm_transition_rate=1., output_imm_measurement_scale=.001, output_imm_process_scale=1.)
        elif model == "output-reacquisition":
            config["output_reacquisition_distance"] = .1
        yield config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["imm", "pda", "output-imm", "selection-only", "soft-geometry", "output-reacquisition"])
    parser.add_argument("--student-binary", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    root, audit, out = args.root.resolve(), args.audit.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    history = root / "logs/ball-improvement-20261003"
    old = json.loads((audit / "manifest.json").read_text())
    clips = list(old["clips"])
    assert len(clips) == 168
    screen = json.loads((history / "full-suite-inputs.json").read_text())
    screen = screen["train"] + screen["validation"] + json.loads((history / "sixth-audit-inputs.json").read_text())["recordings"]
    assert len(screen) == len(set(screen)) == 48 and set(screen) <= set(clips)
    capture = history / "fresh-approach/baseline.json5"
    baseline = json.loads(capture.read_text())
    validation = str(history / "fresh-long-occlusion/validation-4243.mcap")
    references = {amplitude: json.loads((audit / f"publication-filter-disabled-stress-{amplitude}/report.json").read_text()) for amplitude in [0., .1, .25, .5]}
    by_clip = {amplitude: dict(zip(clips, [x["baseline"] for x in report["training_per_recording"]], strict=True)) for amplitude, report in references.items()}
    manifest = {"protocol": "Enabled-method development tuning, strict unchanged v8 guards; all 168 recordings inspected", "screen": screen, "clips_sha256": old["clips"], "binaries": {}, "cases": []}
    for model in args.models:
        binary = args.student_binary.resolve() if model == "student-t" and args.student_binary else audit / (model + "-tuner")
        assert binary.exists()
        manifest["binaries"][model] = {"path": str(binary), "sha256": digest(binary)}
        for index, config in enumerate(configurations(model, baseline)):
            name = f"{model}-{index:03}"
            parameters = out / (name + ".json")
            parameters.write_text(json.dumps(config, indent=2) + "\n")
            manifest["cases"].append({"model": model, "name": name, "parameters": str(parameters), "sha256": digest(parameters)})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def evaluate(case, selected_clips, amplitude, stage):
        destination = out / (case["name"] + f"-{stage}-{amplitude}")
        binary = manifest["binaries"][case["model"]]["path"]
        cmd = [binary, "--train", *selected_clips, "--validation", validation, "--parameters", str(capture), "--evaluation-parameters", case["parameters"], "--trials", "0", "--field-prior-wobble-metres", str(amplitude), "--output", str(destination)]
        with Path(str(destination) + ".log").open("w") as log:
            result = subprocess.run(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            return {"returncode": result.returncode, "report": str(destination / "report.json"), "failures": {"execution": result.returncode}}
        r = json.loads((destination / "report.json").read_text())
        assert r["replay_matches_live"] and r["trials"] == 0 and r["objective"]["version"] == "single_ball_close_accuracy_v8"
        bad = {}
        for path, c in zip(selected_clips, r["training_per_recording"], strict=True):
            v = violations(c["baseline"], by_clip[amplitude][path], True)
            if v:
                bad[path] = v
        if selected_clips == clips:
            v = violations(r["training"]["baseline"], references[amplitude]["training"]["baseline"], False)
            if v:
                bad["aggregate"] = v
        return {"returncode": 0, "report": str(destination / "report.json"), "score": r["training"]["baseline"], "failures": bad}

    def run(case):
        result = dict(case)
        result["screen"] = evaluate(case, screen, 0., "screen")
        if not result["screen"]["failures"]:
            result["full"] = {}
            for amplitude in [0., .1, .25, .5]:
                r = evaluate(case, clips, amplitude, "full")
                result["full"][str(amplitude)] = r
                if r["failures"]:
                    break
            result["passes_all_guards"] = len(result["full"]) == 4 and all(not r["failures"] for r in result["full"].values())
        else:
            result["passes_all_guards"] = False
        (out / (case["name"] + "-verdict.json")).write_text(json.dumps(result, indent=2))
        print(case["name"], "screen failures", len(result["screen"]["failures"]), "all guards", result["passes_all_guards"], flush=True)
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run, manifest["cases"]))
    summary = {"results": results, "guard_feasible": [r for r in results if r["passes_all_guards"]], "note": "Feasibility alone is insufficient: verify active effect by disabled-feature ablation and substantial aggregate gains before selection."}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("finished", len(results), "cases; feasible", len(summary["guard_feasible"]), flush=True)


if __name__ == "__main__":
    main()
