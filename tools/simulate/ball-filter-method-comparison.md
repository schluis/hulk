# Ball filter method comparison, 2026-10-03

The experiment compares the current multi-hypothesis filter with three research-inspired alternatives. These are small, independently written adaptations to HULKs 2D ground tracking problem, not claims to reproduce published benchmark results or to establish state of the art. All alternatives default to disabled and use the same original capture parameters for exact replay verification.

## Research and implementations

- **Student-t innovation reweighting:** heavy-tailed measurement models reduce the influence of outliers. The experiment inflates measurement covariance according to normalized innovation, using a bounded Student-t precision approximation. It retains a Gaussian state; it does not implement the full latent-variable recursion or smoothing. Reference: Roth et al., [Robust Bayesian Filtering and Smoothing Using Student's t Distribution](https://arxiv.org/abs/1703.02428). Worktree: `/home/schluis/hulk-ball-student-t`, parameter `student_t_robustness`.
- **Interacting multiple models (IMM):** keep both a strongly damped and a rolling motion model, mix states including between-model covariance, and update mode probabilities from measurement likelihoods. This targets abrupt kicks and stops that the current hard resting/moving transition may handle late. IMM is an established method, not a newly invented SOTA algorithm. Relevant RoboCup precedent: [Map-based Multiple Model Tracking of a Moving Object](https://rse-lab.cs.washington.edu/papers/tracking-robocup-04.pdf). Robust IMM literature: [Li and Sun, 2019](https://www.mdpi.com/1424-8220/19/22/4830). Our variant uses Gaussian models and no map constraints. Worktree: `/home/schluis/hulk-ball-imm`, parameter `imm_transition_rate`.
- **Conditional probabilistic data association:** moment-match moving-track posteriors across ambiguous, gated detections, preserving covariance from association uncertainty. Hard assignment still controls track existence and confirmation; detections assigned to other tracks are excluded. This is a deliberately limited PDA variant. The recent [PKF paper, revised September 2025](https://arxiv.org/abs/2411.06378) motivated testing ambiguity handling, but our implementation does not reproduce its EM update, joint association probabilities, or matrix permanents. Worktree: `/home/schluis/hulk-ball-pda`, parameter `association_temperature`.

RoboCup context: [Berlin United's modeling report](https://docs.berlinunited.org/teamreport/modeling/) uses a multi-hypothesis EKF ball model to distinguish consistent tracks from sporadic false detections. A more elaborate estimator is therefore not automatically better than improving the existing hypothesis management. The tuned current filter remains a full competitor.

## Frozen comparison protocol

Common source checkpoint: `9fc3bb824`. Frozen suite: `logs/ball-improvement-20261003/full-suite-inputs.json`, 16 training and 8 held-out 40-second recordings across eight families. Original capture baseline: `fresh-approach/baseline.json5` under that log directory. Per-recording outputs are retained. No parameter or algorithm choice uses held-out scores.

Objective v7: time-normalized spatial/missing loss plus separately normalized close-range loss. User-approved per-clip limits are baseline close RMSE +0.01 m and baseline RMS/mean-absolute spatial lag +0.04 s. Aggregate accuracy remains non-increasing. Missing time, close missing time, longest missing run and false-track time remain strict both per clip and in aggregate. Spatial lag is a motion-normalized position error, not compute latency.

Compare equal numbers of seeded search trials, identical common parameter bounds and a common training-only warm start. Report algorithm-specific parameter values, including when an experiment prefers the disabled baseline. Record wall-clock search time separately because IMM costs more per evaluation. Select each contender by training loss subject to guards; inspect held-out only after selection. Substantial overall improvement, not merely feasibility or a small scalar gain, is required for promotion.

Limitations: injected detection noise instead of running the image detector, simplistic false-box size model, relatively few seeds, considerable out-of-field time in some moving scenes, no localization wobble in the sideline recipe yet. Replay measures estimator quality on fixed inputs; it cannot prove closed-loop kicking success or real-robot performance.

## Equal-budget training results

Eight runs per method, 1,024 trials per run; four coordinate searches and four differential-evolution searches. Each run passed capture replay verification. Training loss (lower is better):

| Method | Best training loss | Experimental setting |
| --- | ---: | --- |
| Current filter | 2.283688 | no experimental model |
| Student-t approximation | 2.289060 | robustness 0.227606 |
| IMM | 2.289060 | transition rate 0: disabled |
| Conditional PDA | 2.289061 | temperature 4 |

The current filter won this comparison. Its training missing time fell from 73.700 s to 51.260 s, RMS spatial lag from 0.9583 s to 0.5637 s, and close RMSE from 0.4246 m to 0.4184 m. These are training results, not a promotion decision.

Ablating the best Student-t configuration produced exactly the same training scores: its robust downweighting had no effect on these accepted associations. The best IMM configuration already disabled the alternative. Removing PDA very slightly improved its training loss (less than 0.000001). These experiments do not establish that the general published methods are ineffective; they show that these particular adaptations did not improve this dataset under the agreed guards and search budget.

Frozen manifests, binary hashes, training-only selection and ablation outputs live under `logs/ball-improvement-20261003/`: `method-comparison-manifest.json`, `method-comparison-training-selection.json`, and `ablation-summary.log`. Branch checkpoints: Student-t `efa7860fd`, IMM `a05f398f7`, PDA `71a741d15`. Archived binaries have a stale free-text `continuity_policy` description from v6; their objective v7 text and actual guard implementation correctly apply the approved 1 cm / 40 ms per-clip margins. Source descriptions have been corrected.

## Follow-up, separate from the frozen comparison

The current-filter branch now also experiments with a distance limit for size-consistency checks, retaining the existing global check when the limit is zero. This targets close false associations without rejecting uncertain distant geometry. The feature remains disabled in production defaults. Follow-up search uses training only. Fresh audit recordings are being captured with offsets starting at 9,000,000; every file in that audit, including files named `train-*`, was reserved until the candidate was frozen.

The selected current-filter candidate failed the original held-out check: loss 2.4010 -> 1.8637 and missing time 33.768 -> 20.194 s, but close RMSE 0.2850 -> 0.3232 m, RMS spatial lag 1.2191 -> 1.6085 s, and absent-scene false-track time 37.320 -> 37.480 s. The other method winners also failed quality/false-track checks. No candidate was promoted.

Those eight inspected clips have now joined the 16 development clips. A separate fresh audit contains 24 new coverage-checked clips across all eight families, reserved entirely for final evaluation. Development search remains under v7 guards. An additional disabled-by-default selection-confidence cap is under test; it changes ranking only, with existing confirmation, eligibility and storage preserved. Additional diagnostics count wrong output farther than 0.5 m from present truth, total correct-track unavailability, and longest correct-track gap. They do not currently change acceptance. A user clarification is pending because raw missing time credits even a clearly wrong output as retained.


## First fresh audit and expanded development set

The candidate frozen in `candidate24-before-fresh-audit.json` improved fresh-audit aggregate close RMSE from 0.4701 to 0.1951 m and RMS spatial lag from 0.7254 to 0.3697 s. It nevertheless failed four per-clip checks: stationary close error increased by 3.2 cm, approach RMS lag increased by 91 ms, empty-scene false output increased by 132 ms, and one sideline clip lost retention. It was rejected. Exact results are in `fresh-audit-first-verdict.json`; strong aggregate gains do not override the agreed safeguards.

Those 24 inspected audit clips now join the original 24 as a **48-clip development set** (`development-48-inputs.json`). They are no longer independent validation. An excluded older long-occlusion recording supplies the CLI's required distinct validation input during development searches; its scores are diagnostic only and cannot support promotion.

A selection-only grid tested 168 combinations of uncertainty weight and confidence cap, keeping all association, prediction, confirmation and existence parameters at the original baseline. Sixteen combinations satisfied every development guard. The best used `hypothesis_uncertainty_weight = 0.05` and `selection_confidence_cap = 30`: close RMSE 0.4301 -> 0.4076 m, RMS spatial lag 0.9052 -> 0.7396 s, mean absolute spatial lag 0.2836 -> 0.2402 s. Missing time (224.072 s) and false-track time (210.164 s) were exactly unchanged, both overall and per clip. Correct-track unavailability improved from 426.144 to 411.800 s, but remains a diagnostic rather than a relaxed retention rule. Grid script: `/home/schluis/hulk-ball-selection-only/tools/simulate/selection_grid.py`; complete scores: `selection-only-grid-v2/summary.json`.

This selection-only candidate seeds further 48-clip searches. A second independent audit uses new seed offsets starting at 19,000,000, with all model metrics withheld until candidate freeze. Failed upright/coverage captures are excluded and replaced using predetermined seed increments; the first fast-crossing capture fell at 37.7 s. Capture provenance and replacements are retained. Production parameters are unchanged.

Quiet-track reacquisition gating and speed-based resting transitions are also implemented as disabled experiments. Fixed probes failed the guards; neither is recommended. Current checks pass 35 tuner library tests, 105 ball-filter tests, and strict release Clippy for the tuner/filter libraries.


The 48-clip candidate was frozen before inspecting the second audit: uncertainty weight **0.05**, selection confidence cap **22**, visibility uncertainty scale **0.0025**, with all other parameters restored to the original baseline. Development close RMSE is 0.3611 m (16.0% lower), RMS spatial lag 0.7061 s (22.0% lower), missing time 222.992 s (1.080 s lower), and false-track time unchanged. It passes all per-clip guards. Ablation showed the visibility margin contributes most of the additional close-range gain; restoring the original confidence decay rates sacrifices very little loss and avoids two unnecessary changes. A larger visibility scale of 0.005 failed retention and false-track checks and was rejected using development data alone.

Freeze parameters and hash: `candidate48-before-final-audit.json` and `candidate48-before-final-audit-selection.json`. The rebuilt current simulator passed all four independent approach-continuity cases for this candidate (`approach48-candidate/report.json`). These observe behavior commands on fixed physical inputs and do not establish physical kick success. The second audit remains the promotion gate.


## Second audit: rejected; 72 development clips

The frozen three-setting candidate also failed independent validation. Close RMSE improved 0.2137 -> 0.1896 m and RMS spatial lag 0.5433 -> 0.4647 s, but false-track duration rose 100.388 -> 100.588 s. Brief-gaps/train-42 exceeded its close-error allowance (0.3277 -> 0.3429 m); contested/train-43 added 40 ms missing time and exceeded both spatial-lag allowances. Two empty-false clips increased false output. Exact per-clip verdict: `final-audit-verdict.json`. Production remains unchanged.

These 24 inspected clips now join development (`development-72-inputs.json`). Repeating the 168-case selection-only grid on all 72 found no improving feasible setting: the four feasible combinations reproduce baseline outputs. This rules out promoting the previously promising uncertainty/cap pair. A separate worktree, `/home/schluis/hulk-ball-soft-geometry`, now tests soft size-consistency evidence for selection, retaining baseline association/existence. A third audit is reserved at offsets starting 39,000,000; its scores remain uninspected. The synthetic false boxes have a simplified fixed radius, so any gains from size evidence require particular caution about real-detector transfer.


Further 72-clip experiments: soft size-consistency ranking (worktree commit `776dfe1e8`) found no feasible improving combination in its 80-case revised grid. The initial version incorrectly let tiny size differences veto confirmed stale-track recovery; the revision preserves that rule and has 107 passing filter tests and strict library Clippy. Both sets of results are retained, with `soft-geometry-v2-grid-72/summary.json` describing the corrected experiment. No production change.

An independent output IMM (`/home/schluis/hulk-ball-output-imm`, commit `bd7c00c52`) leaves the original hypothesis state, association and existence untouched, and blends a separate resting/rolling estimate only at publication. Its 48-case grid found four feasible settings, but the best only improved close RMSE 0.3694 -> 0.3647 m (~1.3%) and RMS spatial lag 0.7999 -> 0.7989 s. This is insufficient for the requested substantial improvement and has not consumed the third audit. It passes 103 filter tests and strict library Clippy. All tested per-clip missing/false durations were unchanged. See `output-imm-grid-72/summary.json`.

The third audit completed all eight families (24 clips) without requiring replacement; model metrics remain uninspected. Work continues on publication-only protection against isolated, distant reacquisition of a previously quiet ball, preserving the baseline association/existence state and releasing protection after coherent remote observations or a possible kick. This is experimental, not promoted.


Publication-only reacquisition experiments (`/home/schluis/hulk-ball-output-reacquisition`) also failed the 72-clip guards. The final tested version is `a4b8f74ca` (`output-guard-v5-grid-72/summary.json`), with 111 filter tests and strict library Clippy. Full frame diagnostics show why per-hypothesis protection cannot fix the main stationary failure: it preserves the correct position at 29.948 s, but the baseline hypothesis itself is deleted at 30.118 s, destroying that history. Assertions confirmed missing and false durations unchanged; position guards still failed. Frame exports use the optimized parameters, so `output-guard-fast-diagnostic` is NOT a fixed-candidate trace (its one trial changed noise). The full 72-clip `output-guard-development-diagnostic` and `output-guard-v4-diagnostic` reports explicitly match evaluation/optimized parameters and are the valid traces.

The size experiments also exposed an approximation mismatch: `get_pixel_radius` uses angular field of view, while injected sphere sizes use focal length and camera depth. Corrected experimental size checks use the pinhole model. The corrected size-ranking grid, with a 1.5x tolerance before penalizing stale-recovery support (`soft-geometry-v4-grid-72/summary.json`), still found no feasible improvement. This is not a claimed model improvement.

Current experiment: `/home/schluis/hulk-ball-publication-filter` maintains a separate geometry-filtered tracker, but gates every publication on baseline availability. The auxiliary history survives baseline deletion. An unrestricted 32-case blend grid failed accuracy checks, despite preserving all missing/false durations. The next version uses the auxiliary estimate only when the baseline's last detection is undersized for a sphere at its ground-plane projection. Larger images may be airborne balls and are deliberately left unchanged. This is a prototype, not promoted. The third audit remains uninspected.

### Correct-ball availability approval (objective v8)

The user explicitly approved protecting correct-ball availability while allowing removal of clearly wrong outputs. The tuner now constrains missing-or-more-than-0.5-m-wrong duration, its close-range (truth within 1 m) subset, and the longest such gap per recording and in aggregate. Raw missing duration remains reported but is no longer an eligibility guard. False-track guards remain strict; per-clip accuracy allowances remain +0.01 m RMSE and +0.04 s RMS/absolute spatial lag, with no aggregate accuracy regression. The spatial search loss is unchanged for comparability; v8 changes eligibility and its exploration distance, not historical measurements. The third 24-clip audit remains reserved.

### Third independent audit: auxiliary publication filter rejected

The v8 hard radius-gate grid (32 combinations, 72 development clips) had no improving feasible candidate: the best loss still regressed correct-ball availability on 17 clips. The separate auxiliary history was more promising. It uses physically plausible detections and only corrects selected outputs whose latest detector image is too small for the projected physical ball. A 32-case age/distance grid selected full correction with detection noise 0.2 and alternate distance at most 1 m, improving development close RMSE 0.369419 -> 0.202155 m and RMS spatial lag 0.799899 -> 0.766103 s with all v8 guards passing.

The candidate was frozen before inspecting the third audit (`candidate72-before-third-audit-manifest.json`, source `e3c880e1a`). On its 24 fresh clips, close RMSE improved 0.648953 -> 0.527440 m, close correct-ball unavailability 44.536 -> 30.380 s, RMS lag 0.536680 -> 0.519183 s, and false-track time stayed 103.502 s. However, sideline/train-42 lost 0.100 s of correct-ball availability (10.920 -> 11.020 s), so it was rejected. Artifacts: `third-audit-verdict.json`. These 24 clips are now development data, forming `development-96-inputs.json`; a fourth audit uses fresh capture seeds starting 59,000,000. No production parameters changed. The prototype has 108 passing filter tests, 37 tuner tests, a passing targeted stale-alternate regression test, and strict library Clippy.

### Fixed-parameter availability diagnostics

Use `ball-filter-tuner --trials 0 --export-baseline-training-frames` to export the exact evaluation baseline rather than the search winner. Run `python3 tools/simulate/analyze_ball_availability.py <run>/report.json --output <run>/availability.json` afterward. This checks exported correct-ball unavailability against the report, groups metrics by scenario family, separates initial acquisition from subsequent losses, and reports completed and censored reacquisition events. Reacquisition starts at a delivered projected percept within 0.5 m of current truth: this is a supporting-detection proxy, not measured physical visibility or detector identity. Keep its censor counts alongside mean/max delays. Inspecting an audit this way is diagnostic only after its frozen-candidate verdict; it must not become parameter-selection data while still described as held out.

The revised 96-clip candidate (auxiliary age <=5 s, no distance cutoff) passes all development guards with 20% lower close RMSE and 10% lower RMS lag. It also passes all four live approach continuity checks; `approach96-verdict.json` includes parameter and simulator hashes. On the previously failing third-audit sideline clip, fixed exports show initial acquisition unchanged at 0.174 s and established-track unavailability improving 10.746 -> 9.806 s. The fourth audit remains pending; no production promotion.
