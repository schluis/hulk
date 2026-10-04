# Ball filter method comparison, 2026-10-03

## Current result — validated and enabled, 2026-10-04

The final geometry-supported auxiliary filter is enabled in `etc/parameters/base/ball_filter.json5` (source commit `28ff65b02`). On **24 fresh independent recordings**, close-range RMSE improves **20.6%** (0.461851 -> 0.366802 m) and RMS spatial lag **17.1%** (0.819072 -> 0.678751 s). Correct-ball unavailability improves; false-track time is unchanged. Every per-clip and aggregate safeguard passes on ordinary inputs and at each predeclared localization-wobble amplitude (0.1, 0.25, 0.5 m per axis).

Selection used 144 development recordings, with wobble variants of 48 inspected recordings. The final 24 recordings were reserved until parameters and evaluator were frozen. Integrated source and production parameters reproduce all **168** ordinary recording scores exactly. **146 filter/tuner tests**, strict library Clippy, simulator build and all four live approach continuity checks pass. See the final-result section below for per-family data and provenance. Earlier rejection/pending checkpoints are historical and superseded by this result.

This validates the filter with synthetic detector noise, not CNN inference or real-robot accuracy. Spatial lag is a position-derived metric, not processing latency. Live approach checks establish fixed-pose command continuity, not physical kick success.


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

### Fourth ordinary audit passed; later localization stress failed

The frozen age-limited candidate passed **all 24 fourth-audit recordings and all v8 per-recording/aggregate guards**. Its source is integrated in the main worktree. The parameter enablement was subsequently reverted after the additional localization stress below exposed a regression. The 96 development recordings were used for selection; the final 24 recordings were new seeds across all eight scenario families. Fast-crossing was replaced once after a coverage/upright rejection, before model metrics were inspected. The frozen candidate and binary hashes are in `candidate96-before-fourth-audit-manifest.json`; coverage provenance is in `fourth-audit-complete-manifest.json`; the verdict is `fourth-audit-verdict.json`.

Independent aggregate results:

| Metric | Baseline | Candidate |
|---|---:|---:|
| Close-range RMSE | 0.311333 m | 0.247224 m |
| RMS spatial lag | 0.845534 s | 0.720110 s |
| Correct-ball unavailable | 165.902 s | 152.818 s |
| Close correct-ball unavailable | 21.140 s | 16.620 s |
| Established-track unavailable | 116.290 s | 103.206 s |
| Initial acquisition unavailable | 49.612 s | 49.612 s |
| Wrong selected output | 76.410 s | 63.326 s |
| False-track time | 104.820 s | 104.820 s |

Per-family independent results (three recordings each):

| Family | Close RMSE, m (base → candidate) | RMS spatial lag, s (base → candidate) | Correct-ball unavailable, s (base → candidate) |
|---|---:|---:|---:|
| approach | 0.102 → 0.102 | 0.447 → 0.314 | 3.648 → 2.168 |
| brief-gaps | 0.355 → 0.310 | 0.214 → 0.213 | 12.772 → 12.372 |
| contested | 0.093 → 0.093 | 0.296 → 0.282 | 56.270 → 54.070 |
| empty-false | — → — | — → — | 0.000 → 0.000 |
| fast-crossing | 0.174 → 0.174 | 1.659 → 1.420 | 18.588 → 17.948 |
| long-occlusion | — → — | — → — | 12.108 → 8.588 |
| sideline | 0.267 → 0.103 | 0.324 → 0.301 | 49.888 → 49.124 |
| stationary-close | 0.390 → 0.305 | — → — | 12.628 → 8.548 |

Reacquisition diagnostics use delivered supporting percepts, not physical visibility: completed-event mean 0.162752 -> 0.155210 s; maximum 3.064 -> 2.854 s; zero supported censored events in either run. Event counts differ (133 -> 119) because the candidate prevents some losses, so the means are not matched-event latency measurements. Full per-recording values are in `fourth-audit-diagnostics-{baseline,candidate}/availability.json`.

Implementation: an independent Kalman history admits observations consistent with physical ball size and survives spurious clearing of the ordinary track. It corrects only selected tracks whose latest detector image is implausibly small; ordinary supported observations retain their original output. Main output presence remains authoritative, so this change does not create extra false tracks. The chosen history uses detection-noise scale 0.2, full correction blend 1, maximum observation age 5 s, and no distance cutoff. Corrections retain the older observation timestamp. All four controls are serialized simulator/robot parameters; `publication_filter_blend=0` disables the auxiliary history for old configurations.

Scope: these results validate the filter against synthetic detector noise and physically simulated trajectories, not the CNN detector or real-robot accuracy. The four live approach checks cover fixed-pose kick-command continuity during gaps/delivery delay, not physical kick success. Student-t, IMM and PDA alternatives remain separate experiments; none earned promotion in their matched comparison.

### Additional localization stress: enablement reverted pending hardening

The notes also requested localization wobble. Added `--field-prior-wobble-metres`: first verify exact captured replay, then perturb only the optional filter prior with independent 4 s / 7 s sinusoidal Field-position errors. Ground truth, scoring transforms, cameras and odometry stay unchanged; reports record the amplitude and number of actually changed cycles. Zero disables the stress. A regression test protects this separation.

The frozen candidate passed all sideline per-clip guards at 0.1, 0.25 and 0.5 m per axis; the 0.25 m sideline-only aggregate mean-absolute lag rose 1.4 ms. Broader all-family stress exposed a real per-clip regression: fourth-audit brief-gaps/train-42 RMS lag 0.502 -> 0.703 s at 0.25 m, and 0.976 -> 1.510 s at 0.5 m. The 0.5 m full-suite RMS aggregate also regressed. Accordingly the production parameter enablement is reverted, despite the earlier independent ordinary-input pass. Source remains optional/default-off. Next experiment isolates the auxiliary geometry history from field-prior localization error while leaving main output eligibility unchanged. A fifth independent suite uses fresh seeds starting 89,000,000. Goal remains active.

### Buffered auxiliary prior selected for fifth audit

Removing the auxiliary field prior entirely fixes the inspected wobble cases but regresses an older sideline clip's RMS lag (0.485 -> 0.588 s). Disabling only stored-validity decay preserves the 120 ordinary clips but leaves the wobble regression. A five-value auxiliary margin grid (0.5, 0.75, 1.0, 1.5, 2.0 m), with auxiliary stored-validity field decay disabled, found 1.5 and 2.0 m feasible on all 120 ordinary clips and on all 24 inspected clips at each of 0.1, 0.25 and 0.5 m prior wobble. The main filter's 0.5 m margin and all main eligibility settings remain unchanged.

The 1.5 m auxiliary margin won by ordinary development loss. Close RMSE 0.428504 -> 0.343991 m; RMS lag 0.769037 -> 0.663603 s; correct-ball unavailable 1049.874 -> 980.750 s; false time unchanged 518.874 s. Source `9c15aecea`, immutable `buffered-prior-tuner`, frozen `candidate120-before-fifth-audit.json` and manifest. Fifth audit uses 24 fresh recordings and predeclared wobble amplitudes 0, 0.1, 0.25, 0.5 m. It remains unseen. Prototype tests: 108 filter + 38 tuner, strict library Clippy pass. New parameter `publication_field_boundary_margin` widens only the auxiliary ranking prior and never narrows the main margin. Production enablement remains reverted pending these results.

### Fifth audit: ordinary pass, two small wobble lag failures

The buffered 1.5 m / full-correction candidate passed all 24 fresh ordinary recordings: close RMSE 0.376630 -> 0.313779 m (16.7%); RMS lag 0.856418 -> 0.484896 s (43.4%); correct-ball unavailable 236.808 -> 217.664 s; false time unchanged 98.062 s. All families passed coverage on the first attempt. However sideline/train-42 under 0.1 m prior wobble had RMS lag 1.220624 -> 1.268117 s (+47.5 ms), and under 0.25 m 1.213805 -> 1.267212 s (+53.4 ms). Both exceed the allowed +40 ms; enablement remains reverted. Artifacts: fifth-audit-verdict.json and fifth-audit-diagnostics-{baseline,candidate}/availability.json.

These recordings are now development, totaling 144 ordinary clips plus stress variants of 48 inspected clips. A fixed blend grid (0.4 through 1.0) at auxiliary margin1.5 m finds blend0.7 feasible across every ordinary and stress guard. Blends0.4–0.6 regress one original sideline clip's correct availability; blends0.8–1.0 exceed the fifth-audit wobble lag guard. A second margin2 m blend grid is being completed before the next freeze. Sixth audit captures fresh seeds starting119,000,000; its metrics remain unseen.

### Sixth-audit freeze: partial correction

Both 1.5 m and 2 m auxiliary margin grids select blend0.7 and have identical ordinary scores. The declared tie-break chooses the smaller margin1.5 m. Development144 close RMSE0.420717->0.348359m (17.2%), RMSlag0.783866->0.654177s (16.5%), correct-unavailable1286.682->1235.402s, false616.936s unchanged. All144 ordinary clips and all48 inspected clips at each .1/.25/.5m wobble pass every per-clip and aggregate guard. Frozen `candidate144-before-sixth-audit.json` and manifest; unchanged source9c15aecea / buffered-prior-tuner. The candidate also passes all four live approach cases (`approach144-verdict.json`). Sixth fresh capture uses seeds119,000,000+; ordinary and all three stress amplitudes are predeclared, and no sixth scores have been inspected. Main parameter enablement remains reverted pending this independent verdict.

## Sixth independent audit — final configuration

All eight families passed capture coverage on the first attempt. The frozen configuration uses:

| Parameter | Value | Purpose |
|---|---:|---|
| `publication_filter_blend` | 0.7 | Correct implausible selected observations with a partial auxiliary estimate; zero disables the auxiliary filter. |
| `publication_detection_noise` | 0.2 | Auxiliary measurement-noise scale. |
| `publication_maximum_age` | 5 s | Fall back to the original estimate when auxiliary visual evidence is older. |
| `publication_maximum_distance` | 0 | No distance cutoff. |
| `publication_field_boundary_margin` | 1.5 m | Auxiliary ranking uncertainty buffer; main field margin remains 0.5 m. |

The auxiliary history does not accumulate localization-dependent stored-validity decay, but retains a soft field-ranking prior. Main association and output presence remain authoritative. Ordinary physically plausible observations retain their original estimate; correction uses the older observation timestamp. These controls can be varied through simulator parameter files or fixed `--evaluation-parameters` / `--trials 0` comparisons. They are held fixed by the existing 21-dimension continuous search; the archived grids explicitly evaluate the additional controls.

Independent ordinary-input results (24 clips):

| Metric | Baseline | Final |
|---|---:|---:|
| Close-range RMSE | 0.461851 m | 0.366802 m |
| RMS spatial lag | 0.819072 s | 0.678751 s |
| Mean absolute spatial lag | 0.232810 s | 0.200614 s |
| Correct-ball unavailable | 229.670 s | 219.390 s |
| Close correct-ball unavailable | 17.870 s | 16.802 s |
| Initial acquisition unavailable | 49.660 s | 49.660 s |
| Established-track unavailable | 180.010 s | 169.730 s |
| Wrong selected output | 98.090 s | 87.810 s |
| False-track time | 109.576 s | 109.576 s |

Per-family independent results (three recordings each):

| Family | Close RMSE, m (base → final) | RMS spatial lag, s (base → final) | Correct-ball unavailable, s (base → final) |
|---|---:|---:|---:|
| approach | 0.113 → 0.104 | 1.083 → 1.072 | 6.234 → 6.074 |
| brief-gaps | 0.400 → 0.319 | 0.245 → 0.216 | 15.190 → 14.350 |
| contested | — → — | 0.730 → 0.491 | 97.634 → 96.474 |
| empty-false | — → — | — → — | 0.000 → 0.000 |
| fast-crossing | 0.395 → 0.383 | 0.358 → 0.162 | 8.550 → 8.470 |
| long-occlusion | — → — | — → — | 22.972 → 15.092 |
| sideline | 1.748 → 1.336 | 1.426 → 0.888 | 77.478 → 77.318 |
| stationary-close | 0.011 → 0.011 | — → — | 1.612 → 1.612 |

All false-track time belongs to the empty-scene family and is unchanged. Completed reacquisition-event mean is 0.158419 -> 0.167395 s, maximum 2.392 s unchanged, with zero supported censored events. Event counts differ (129 -> 119) and established loss runs decrease (138 -> 132), so these means are not matched-event latency measurements and are not evidence of faster reacquisition. Initial acquisition is unchanged; established-track unavailable duration decreases.

Predeclared independent localization stress (same 24 clips, scoring truth/cameras/odometry unchanged):

| Prior wobble per axis | Close RMSE, m (base → final) | RMS spatial lag, s (base → final) | All guards |
|---|---:|---:|---|
| 0.1 m | 0.468 → 0.372 | 0.842 → 0.697 | pass |
| 0.25 m | 0.499 → 0.375 | 0.953 → 0.784 | pass |
| 0.5 m | 0.507 → 0.255 | 1.153 → 0.851 | pass |

Reproduction/provenance artifacts live under `logs/ball-improvement-20261003` on this server: `candidate144-before-sixth-audit-manifest.json` (frozen source/parameters/binary hashes), `sixth-audit-complete-manifest.json` (capture hashes and commands), `sixth-audit-verdict.json` (every ordinary/wobble guard), `sixth-audit-diagnostics-{baseline,candidate}/availability.json` (per-recording acquisition/reacquisition), `final168-verification/report.json` and `final168-integration-verdict.json` (exact integrated replay), `final168-tests.log`, `final168-clippy.log`, `final168-simulator-build.log`, `final168-approach/report.json`, and `final-completion-audit.json`. Captures and immutable binaries are local artifacts, not a clean-checkout download guarantee. Use a new output directory when reproducing evaluator commands; existing reports are protected from overwrite.

The Student-t, IMM and PDA adaptations remain separate worktree experiments. Their matched comparison is documented above; the promoted change is the geometry-supported auxiliary history, not a claim of reproducing a published SOTA method. No parameters were selected from the sixth audit.

## Preserved-worktree continuation audit — 2026-10-04

At the user's request, all eight ball-filter worktrees were checked again, including the uncommitted soft-geometry v4 implementation. Each implementation was copied into a disposable source tree and evaluated with the current, shared v8 scoring and recording code. Algorithm-specific tracking source, parameter types and search encoding were retained. The evaluator adapter permits zero trials and adds prior wobble after exact capture replay verification. Original worktrees, dirty edits, production parameters and monitor work were preserved.

This audit evaluated **19 fixed configurations × 4 localization amplitudes = 76 cases**, each across all **168 inspected recordings**. The historical selected configurations were reused; where a grid had no qualifying improvement, its lowest-loss rejected candidate was included as an explicitly rejected diagnostic probe. IMM also received an enabled probe because its historical winner disabled the model. This is a fixed-candidate comparison under common v8 guards, **not a new equal-budget optimization or independent validation**. The sixth audit is now part of this inspected comparison dataset.

Ordinary-input results below compare against the original capture filter. The failing-clip column excludes aggregate failures; all safeguards must pass for qualification.

| Fixed configuration | Close RMSE, m | RMS spatial lag, s | Failing clips / 168 |
|---|---:|---:|---:|
| Original filter / every disabled control | 0.427070 | 0.788793 | 0 |
| Historical selected Student-t | 0.345643 | 0.586990 | 70 |
| Historical selected IMM (model disabled) | 0.345643 | 0.586990 | 70 |
| Enabled IMM probe, otherwise capture parameters | 0.966421 | 1.127217 | 119 |
| Historical selected PDA | 0.345652 | 0.586958 | 69 |
| Historical selected output-only IMM | 0.424413 | 0.788131 | 37 |
| Lowest-loss rejected output-only IMM probe | 0.408232 | 0.787864 | 57 |
| Historical selected selection-only (baseline-identical) | 0.427070 | 0.788793 | 0 |
| Lowest-loss rejected selection-only probe | 0.404496 | 0.522033 | 17 |
| Lowest-loss rejected soft-geometry v4 probe | 0.406584 | 0.622236 | 17 |
| Lowest-loss rejected output-reacquisition v5 probe | 0.424300 | 0.789880 | 1 |
| Integrated publication-filter configuration | **0.351157** | **0.657605** | **0** |

Student-t, selected IMM and PDA also fail the aggregate false-track guard: **726.512 → 741.144 s**. Output-reacquisition also fails aggregate RMS/mean-absolute spatial lag. None of the tested improving alternative configurations passes every original-filter safeguard, and none passes every safeguard against the integrated publication filter. Their lower aggregate metrics in some columns do not justify promotion. This does not rule out retuning these methods or combining them with the publication history; neither was tested here.

All disabled controls reproduce identical per-recording scores at every amplitude, and the publication configuration reproduces all 168 previously frozen ordinary per-recording scores exactly. The publication filter also passes every per-recording and aggregate guard on **all 168 recordings** at each of 0, 0.1, 0.25 and 0.5 m per-axis prior wobble, expanding stress coverage beyond the earlier 48 inspected development plus 24 independent clips. At zero wobble, correct-ball unavailable time improves **1516.352 → 1454.792 s**, with false-track time unchanged at **726.512 s**.

Each copied filter passes its library tests: Student-t 101, IMM 102, PDA 101, output-only IMM 103, selection-only 105, soft geometry 109, output reacquisition 111, publication filter 108. The shared evaluator passes 38 tests with each implementation. Every release build and strict library Clippy check passes, including the previously unverified final soft-geometry edits.

Evidence is under `logs/ball-improvement-20261003/worktree-audit-20261004-v2/`: `manifest.json` records worktree HEADs/status, source hashes, binary hashes, parameter hashes and all 168 input hashes; `*-source.tar.gz` preserves effective source; `*-{build,test,clippy}.log` records checks; each case directory contains its full `report.json`; `verdict.json` contains failures against both original and production references; `completion-audit.json` verifies coverage, hashes and unchanged worktrees. Earlier `worktree-audit-20261004/` artifacts record an abandoned adapter build and are not the completed audit.

To reproduce, choose unused output and scratch directories:

```bash
python3 tools/simulate/check_remaining_worktrees.py \
  --output logs/ball-improvement-20261003/worktree-audit-new \
  --scratch /tmp/hulk-worktree-audit-new --workers 16
```

The runner uses the locally installed Rust 1.98.1 toolchain and offline dependencies. `--resume` verifies frozen input/source/binary/parameter hashes before reusing built evaluators and rerunning the fixed cases. Synthetic detector and physical closed-loop limitations remain unchanged; this audit introduces no new production setting.
