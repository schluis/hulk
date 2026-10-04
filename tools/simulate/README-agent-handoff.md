# Ball-filter optimization: agent handoff

Exported 2026-10-04 from the optimization session. Repository: `/home/schluis/hulk`. This file records the goal, decisions, completed work, experimental worktrees, evidence, and practical continuation context. Detailed chronological evidence remains in the linked notes and local artifacts.

## Start here

The user requested substantial improvement to ball detection/filtering by running the simulator on this server, adding tunable parameters or changing the filter as necessary. They explicitly suggested researching established/state-of-the-art filtering methods, implementing alternatives in separate worktrees, and comparing them. They wanted sustained autonomous work and better use of the available CPU, not just a plan or a small tuning gain.

**That optimization goal was completed and marked complete.** The final geometry-supported auxiliary filter is integrated and enabled. On 24 fresh independent recordings it reduced close-range RMSE by **20.58%** and RMS spatial lag by **17.13%**, passing all agreed per-recording and aggregate safeguards, including localization stress. The session took about 5 hours 21 minutes. Do not automatically reopen the completed goal or launch another search merely because this handoff is read.

The user's latest request is to export session knowledge for a new agent, explicitly including the other worktrees and the original improvement goal. Immediately before that, they asked what was happening with the other implementations. Those experiments are retained, not currently running. Only the auxiliary publication-filter implementation was promoted. No worktree cleanup was requested.

Read these in order:

1. This handoff.
2. [Current results and method comparison](ball-filter-method-comparison.md), especially the top status and “Sixth independent audit — final configuration.”
3. [Chronological tuning notes](ball-filter-tuning-notes.txt). Earlier rejected/pending/reverted statuses are historical; the final status supersedes them.
4. [Simulator documentation](README.md) for operation and capture details. This file already has unrelated uncommitted changes; preserve them.
5. Local `logs/ball-improvement-20261003/final-completion-audit.json` and the manifests listed below.

## User decisions and scope

- Substantial overall improvement is required. A tiny feasible scalar-loss gain is insufficient.
- The user approved **up to 0.01 m additional close-range RMSE per recording** and **up to 0.04 s additional RMS and mean-absolute spatial lag per recording**. Aggregate accuracy remains non-increasing.
- The user explicitly chose: **“Protect correct-ball availability; allow removing clearly wrong outputs.”** Raw missing time previously credited estimates metres away from the true ball as retained. Do not reinstate that obsolete interpretation.
- Objective v8 treats a present ball as unavailable if output is missing **or more than 0.5 m from truth**. It protects correct-ball unavailable time, its close-range subset, longest correct-track gap, and false-track time, per clip and in aggregate. Close range means truth distance at most 1 m. Use the actual evaluator/verdict implementation for numerical details and tolerances.
- Retention and false-track safeguards remain strict. Large aggregate gains never excuse a failed per-clip safeguard.
- Keep production kinematics and the simulator's absolute torso treatment. Synthetic filter experiments do not run CNN inference.
- Parameter changes, filter changes, server-side simulation/search and experiment checkpoints were within the authorized task. The user prefers continued useful work over repeated permission questions.
- Preserve unrelated monitor work. Do not use a coordination board for this task.

## Integrated implementation

Root branch: `codex/ball-filter-realism-20261002`.

- Source/configuration integration commit: `28ff65b02`.
- Final results/documentation commit and root HEAD at export: `0b11dbe5265f020fdea5494d751e337e7222f2ca`.
- Frozen winning experimental source: publication-filter worktree commit `9c15aecea5e7e3fa60648aedd4f5a85efbb719a9`.
- Integration copied the validated changes into the root branch; do not describe the experimental branch as Git-merged. Relevant core source was checked byte-for-byte against the frozen source.

The existing main filter still decides association, hypothesis existence and whether an output is present. A separate auxiliary Kalman history uses physically plausible image-size observations and survives clearing of a main hypothesis. When the main selected observation is physically too small, the published estimate can be partially corrected using this retained history. Plausible ordinary observations retain their original estimate. An old auxiliary observation cannot create output on its own, and evidence older than the configured maximum age falls back to the original estimate. Correction preserves the older observation timestamp.

The auxiliary history avoids accumulating localization-dependent stored-validity decay. It retains a soft field-ranking prior with a wider uncertainty buffer. The main field margin remains 0.5 m. This distinction mattered under localization wobble.

Enabled parameters in `etc/parameters/base/ball_filter.json5`:

| Parameter | Final value |
| --- | ---: |
| `publication_filter_blend` | 0.7 |
| `publication_detection_noise` | 0.2 |
| `publication_maximum_age` | `{ secs: 5, nanos: 0 }` |
| `publication_maximum_distance` | 0.0 (no distance cutoff) |
| `publication_field_boundary_margin` | 1.5 |

Omitted/default experimental configuration remains disabled; the repository base configuration explicitly enables this validated setting. The experimental worktree's configuration need not equal the final root configuration, even though the integrated core source matches.

Key code: `crates/nodes/ball_filter/src/{tracker,lib,hypothesis,filter,field_prior}.rs`, `crates/types/src/parameters.rs`, and `tools/ball-filter-tuner`. Consult the integration commit for the complete change rather than assuming this is an exhaustive file list.

## Final measured outcome

Independent ordinary inputs, 24 recordings:

| Metric | Baseline | Final |
| --- | ---: | ---: |
| Close-range RMSE | 0.461851 m | 0.366802 m |
| RMS spatial lag | 0.819072 s | 0.678751 s |
| Mean absolute spatial lag | 0.232810 s | 0.200614 s |
| Correct-ball unavailable | 229.670 s | 219.390 s |
| Close correct-ball unavailable | 17.870 s | 16.802 s |
| Initial acquisition unavailable | 49.660 s | 49.660 s |
| Established-track unavailable | 180.010 s | 169.730 s |
| Wrong selected output | 98.090 s | 87.810 s |
| False-track time | 109.576 s | 109.576 s |

All false-track time here comes from the empty-scene family. Completed reacquisition mean changed from 0.158419 to 0.167395 s; maximum stayed 2.392 s. Event counts changed from 129 to 119, with no supported censored events; established loss runs decreased from 138 to 132. These are not matched-event comparisons. **Do not claim faster reacquisition.** Initial acquisition was unchanged and established unavailable duration improved.

The 144 development recordings improved close RMSE from approximately 0.420717 to 0.348359 m and RMS spatial lag from 0.783866 to 0.654177 s. Correct-ball unavailable time fell from 1286.682 to 1235.402 s; false-track time stayed 616.936 s.

Validation completed:

- All independent ordinary and localization-wobble per-clip/aggregate guards pass.
- Integrated production source/configuration reproduces all 168 ordinary recording scores and parameters exactly.
- 108 ball-filter tests plus 38 tuner tests pass (146 total).
- Strict library Clippy and simulator build pass.
- Four live approach continuity checks pass: fixed-pose control, brief gaps, delivery delay, combined.
- Final audit checked frozen hashes, coverage, actual 1–5-frame dropout coverage, guards, source equality, integrated scores, tests/build and approach reports.

These are **synthetic detector-noise results**, not CNN or real-robot validation. False detections use a simplistic fixed 8-pixel box model. Spatial lag is derived from positional error and motion, not processing latency. The approach checks establish fixed-pose command continuity, not physical kicking success.

## Dataset separation and audit history

The original matched method comparison used 24 recordings: 16 training and 8 held out, across eight families. Inspected held-out recordings subsequently became development data. Five inspected 24-recording audits expanded the development set to 144. Rejected candidates remain documented; do not reuse an inspected audit as independent evidence for a newly selected model.

The final sixth audit contains 24 fresh recordings, three in each family: `stationary-close`, `approach`, `fast-crossing`, `brief-gaps`, `contested`, `long-occlusion`, `empty-false`, `sideline`. All passed capture coverage on the first attempt. Seeds start at offset 119,000,000 with 10,000 family spacing. Filenames such as `train-42.mcap` inside this reserved audit do **not** mean they were training data; manifests define the split.

Parameters/evaluator were frozen before sixth-audit inspection. Selection used all 144 ordinary development recordings and localization stress on 48 inspected recordings. Final selection compared blends 0.4–1.0 and auxiliary field margins 1.5/2 m, selecting the lowest ordinary development loss satisfying every guard; ties favored the smaller margin.

Predeclared stress amplitudes were 0, 0.1, 0.25 and 0.5 m per axis. Wobble affects only the filter's field prior, after exact capture-baseline replay; scoring truth, camera transforms and odometry stay unchanged. Periods are 4 s for x and 7 s for y. Every final guard passed at every amplitude. See the comparison report for full per-family and stress metrics.

For a future optimization, the sixth audit is now inspected. If it informs changes, treat it as development and reserve new recordings for an independent final claim.

## Other worktrees: preserved experiments

All paths below were checked at export. “Clean” means no uncommitted changes; it does not imply merged or promoted. Branches are retained for reproducibility. No experiment worker matching `^python3 .*ball-improvement-20261003` was running at export; this is not a claim that the whole server has no other jobs.

| Worktree under `/home/schluis/` | Branch | HEAD | State / result |
| --- | --- | --- | --- |
| `hulk-ball-student-t` | `codex/ball-student-t-20261003` | `647e500921e4145271dd87bf522c8627daa9c272` | Clean; robust innovation adaptation had no effective gain in the matched comparison. |
| `hulk-ball-imm` | `codex/ball-imm-20261003` | `76953b381ab50d7dee458190b9276533e0969c8a` | Clean; best search disabled IMM. |
| `hulk-ball-pda` | `codex/ball-pda-20261003` | `afa0ef7f2133ff73f7461dcc2838fd64b3cc13ca` | Clean; no qualifying gain, failed held-out checks. |
| `hulk-ball-output-imm` | `codex/ball-output-imm-20261004` | `bd7c00c52144a354322616cb831d39fd38cb0c1b` | Clean; small feasible accuracy gain, insufficient overall improvement. |
| `hulk-ball-selection-only` | `codex/ball-selection-only-20261004` | `a608e2777de35c8b5760d7a6be2bfda4753304ad` | Clean; early gain disappeared under expanded guards/data. |
| `hulk-ball-soft-geometry` | `codex/ball-soft-geometry-20261004` | `776dfe1e8df133ec0e215016d30eaa6beaa72559` | **Dirty follow-up experiment**, no feasible improving v4 result. |
| `hulk-ball-output-reacquisition` | `codex/ball-output-reacquisition-20261004` | `a4b8f74ca3fad2c5e4c9c265cc260a5976beecf5` | Clean; failed grids, but diagnosis informed the winner. |
| `hulk-ball-publication-filter` | `codex/ball-publication-filter-20261004` | `9c15aecea5e7e3fa60648aedd4f5a85efbb719a9` | Clean; **winning source integrated into root**. |

### Student-t, IMM and PDA matched comparison

Common source checkpoint `9fc3bb824`; objective v7, original 16/8 split. Each method, including the existing filter, received eight searches of 1,024 trials: four coordinate and four differential-evolution runs, same seeds/common bounds/training-only warm start. All capture replays verified.

| Method | Best training loss | Interpretation |
| --- | ---: | --- |
| Existing filter | 2.283688 | Won the training comparison, but selected candidate failed held-out guards. |
| Student-t approximation | 2.289060 | `student_t_robustness = 0.227606`; disabling it yielded identical scores. |
| IMM | 2.289060 | `imm_transition_rate = 0`, i.e. disabled. |
| Conditional PDA | 2.289061 | `association_temperature = 4`; ablation improved loss by less than 0.000001. |

All selected winners failed original held-out quality/false-track checks and were not promoted. This was an earlier dataset and objective, **not a fresh equal-budget comparison against the final winner on all 168 clips under v8**.

Student-t inflates measurement covariance from normalized innovation while keeping a Gaussian state; it is not a full Student-t latent recursion/smoother. IMM mixes resting/rolling models, states, covariance and mode probabilities. PDA moment-matches gated moving-track posteriors; hard assignment still controls existence and excludes detections assigned elsewhere. It is not full JPDA or a reproduction of PKF's EM/permanent machinery. These are research-inspired adaptations; their results do not prove the published methods ineffective or establish a new SOTA result.

Primary research links, also discussed in the comparison report:

- [Roth et al., Student-t filtering and smoothing](https://arxiv.org/abs/1703.02428).
- [RoboCup map-based multiple-model tracking](https://rse-lab.cs.washington.edu/papers/tracking-robocup-04.pdf).
- [Li and Sun, robust IMM, 2019](https://www.mdpi.com/1424-8220/19/22/4830).
- [Probabilistic Kalman Filter, arXiv 2411.06378](https://arxiv.org/abs/2411.06378), revised September 2025; inspiration, not reproduction.
- [Berlin United modeling context](https://docs.berlinunited.org/teamreport/modeling/).

### Follow-up implementations

- **Output-only IMM:** separates auxiliary correction from baseline association/existence. Parameters include `output_imm_measurement_scale`, `output_imm_process_scale`, `output_imm_blend`; original IMM rate 1 enables the experiment. A 48-case grid on 72 development clips found four feasible cases, all blend 0.1. Best close RMSE was 0.369419 → 0.364672 m (~1.3%), lag 0.799899 → 0.798907 s. Too small for promotion. 103 filter tests and strict Clippy passed.
- **Selection-only:** `tools/simulate/selection_grid.py` adjusts uncertainty weight and confidence cap without Rust changes in that worktree. On 48 development clips, 168 cases found a feasible ~5% close-error and ~18% lag reduction with identical raw availability. On 72 clips, 168 cases yielded no improving feasible candidate; four feasible cases were baseline-identical.
- **Soft geometry:** `selection_size_consistency_weight` ranks hypotheses using an EWMA of absolute log expected/observed radius. Follow-ups used pinhole expected radius, a log(1.5) deadband, and recovery weighting. The v4 80-case grid on 72 clips had no feasible improving result. 109 filter tests passed; do not assume strict Clippy was rerun for final v4 edits. Uncommitted files are `crates/nodes/ball_filter/src/filter.rs`, `lib.rs`, and `size_consistency.rs`. Preserve them.
- **Output reacquisition:** parameters `output_reacquisition_distance` and `output_reacquisition_blend` guarded suspicious distant reacquisition with per-hypothesis auxiliary state. Quiet-case conditions included gap >120 ms, pre-gap speed ≤0.2 m/s, own ball >0.6 m, and no nearby opponent. v1–v5 32-case grids on 72 clips found no feasible result. Diagnosis: history stored on the main hypothesis vanished when that hypothesis was deleted. This motivated the independent retained publication history. 111 tests and strict Clippy passed.
- **Publication filter:** the successful follow-up described above. Retained as a frozen source snapshot; it is not an active optimization worker.
- Quiet-track reacquisition gating and speed-based resting transitions also had failed fixed probes and stayed disabled. See the chronological notes for those earlier root-branch experiments.

## Geometry tuner and CPU context

The “geometry tuner” was an archived experimental replay-tuner binary/search setup, not a separate visual detector or a claim of a novel filtering method. Local `logs/ball-improvement-20261003/search_geometry.py` runs the `geometry-tuner` binary across 16 independent seeded 4,096-trial searches using approach and brief-gap recordings. These are early experiments, not the final validation protocol.

The user noticed low CPU usage and wanted the server used for optimization. Searches were parallelized across independent runs/cases. Archived scripts encode their worker counts. If resuming, inspect available CPU/memory and active workloads before choosing concurrency; do not infer poor search progress solely from low utilization during capture/build/serial validation phases. No further search is currently required by the completed goal.

The final publication controls are configurable through simulator parameter files and fixed `--evaluation-parameters` / `--trials 0` evaluations. The existing continuous search has 21 dimensions and holds these extra controls fixed; explicit archived grids evaluated them. Do not assume adding a configuration field automatically makes it a search dimension.

## Evidence and reproduction map

All artifact paths below are relative to `/home/schluis/hulk/logs/ball-improvement-20261003/`. This directory contains ignored/local recordings, reports, scripts and immutable binaries. They are available on this server, **not guaranteed by a clean Git checkout**. Preserve them and use new output directories for new evaluations.

| Artifact | Purpose |
| --- | --- |
| `fresh-approach/baseline.json5` | Original capture baseline parameters. |
| `full-suite-inputs.json` | Original matched-method recording suite. |
| `method-comparison-manifest.json` | Frozen original method comparison provenance. |
| `method-comparison-training-selection.json` | Original training-only selections. |
| `ablation-summary.log` | Student-t/IMM/PDA ablation results. |
| `development-144-inputs.json` | Final development recording list. |
| `candidate144-before-sixth-audit.json` | Frozen final candidate parameters. |
| `candidate144-before-sixth-audit-manifest.json` | Selection rule, source and parameter/binary hashes. |
| `buffered-prior-tuner` | Frozen final evaluator binary. |
| `sixth-audit-inputs.json` | Independent final audit recording list. |
| `sixth-audit-complete-manifest.json` | Capture commands, hashes and coverage provenance. |
| `sixth-audit-verdict.json` | Every ordinary/stress guard and outcome. |
| `sixth-audit-{amplitude}-{baseline,candidate}/report.json` | Fixed evaluations; amplitudes 0.0, 0.1, 0.25, 0.5. |
| `sixth-audit-diagnostics-{baseline,candidate}/availability.json` | Acquisition/reacquisition diagnostics. |
| `final168-verification/report.json` | Integrated replay evaluation. |
| `final168-integration-verdict.json` | Exact score/parameter equality against the frozen candidate. |
| `final168-tests.log` | 108 + 38 passing tests. |
| `final168-clippy.log` | Strict library lint validation. |
| `final168-simulator-build.log` | Simulator build validation. |
| `final168-approach/report.json` | Four live continuity cases. |
| `final_completion_audit.py` | Assertions joining final provenance and validation evidence. |
| `final-completion-audit.json` | Completed final audit result. |

Frozen candidate SHA-256: `57c13f43e5ac265b075e80802b87296b3b257d65cbfdfe9808a44383076fd9e9`.
Frozen evaluator SHA-256: `7be77a89be38b4139daec20e21ba39438279ca87fd519ff9828c497f2704e999`.

Read capture manifests/scripts and the evaluator's `--help` for exact commands. Do not blindly rerun historical scripts: some use absolute paths, obsolete objectives, or existing output directories. In particular, `final_completion_audit.py` writes `final-completion-audit.json` and stamps the current HEAD; reading it is safe, but rerunning it would replace historical evidence. Copy/adapt it to a new destination if another verification is needed.

Archived original-method binaries have a stale free-text v6 `continuity_policy` description; their objective-v7 text and actual guard code apply the approved 1 cm/40 ms margins. Source descriptions were corrected. Final acceptance uses v8, so distinguish objective versions when comparing old reports.

## Working-tree changes to preserve

At export, root already had these unrelated changes, before this handoff was created:

```text
 M Cargo.lock
 M tools/ball-filter-tuner/Cargo.toml
 M tools/simulate/README.md
?? tools/ball-filter-tuner/src/bin/
```

They belong to separate monitor work. Do not stage, reset, overwrite or attribute them to the completed ball-filter integration. This handoff is a new standalone file so it does not disturb the existing README edits.

There is also an unrelated detached worktree at `/tmp/hulk-twix-monitor-compat`, HEAD `deeec050205c28410640822676ac7048bbed581e`. It has changes in `Cargo.lock`, `crates/types/src/ball_filter_tuning.rs`, `tools/ball-filter-tuner/Cargo.toml`, and untracked `tools/ball-filter-tuner/src/bin/`. Preserve it. A monitor has used `127.0.0.1:7449`; check current processes before interacting with it, and do not stop it as experiment cleanup.

The soft-geometry edits listed above are unfinished experimental changes, separate from both root integration and monitor work. All seven other ball experiment worktrees were clean at export. Recheck `git status` before any future operation because another session may modify them.

No subagents were used for this optimization. Separate worktrees contained alternative implementations; they were not independent ongoing agent sessions.

## If the user requests further improvement

Keep the validated configuration as a baseline. First identify a concrete remaining failure from per-recording diagnostics; avoid repeating the exhausted grids without a new hypothesis. Promising validation gaps include realistic detector outputs/false-box geometry, real-robot recordings, and physical closed-loop behavior, but these are recommendations for future work, not completed results or newly authorized deployments.

If revisiting Student-t/IMM/PDA, make the comparison fair under the current v8 guards and expanded development data. Their earlier failures do not settle their performance on the final objective. Freeze new candidates before new independent captures, retain failed evidence, report per-clip guards and aggregate gains, and distinguish improved spatial estimates from availability, reacquisition latency and compute cost.

Do not delete worktrees or logs merely because a branch was not promoted. The user explicitly cares about the alternative implementations and their status. Explain their measured results and limitations when asked.
