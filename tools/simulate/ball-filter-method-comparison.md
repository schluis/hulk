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

The current-filter branch now also experiments with a distance limit for size-consistency checks, retaining the existing global check when the limit is zero. This targets close false associations without rejecting uncertain distant geometry. The feature remains disabled in production defaults. Follow-up search uses training only. Fresh audit recordings are being captured with offsets starting at 9,000,000; every file in that audit, including files named `train-*`, is reserved for evaluation.

The selected current-filter candidate failed the original held-out check: loss 2.4010 -> 1.8637 and missing time 33.768 -> 20.194 s, but close RMSE 0.2850 -> 0.3232 m, RMS spatial lag 1.2191 -> 1.6085 s, and absent-scene false-track time 37.320 -> 37.480 s. The other method winners also failed quality/false-track checks. No candidate was promoted.

Those eight inspected clips have now joined the 16 development clips. A separate fresh audit contains 24 new coverage-checked clips across all eight families, reserved entirely for final evaluation. Development search remains under v7 guards. An additional disabled-by-default selection-confidence cap is under test; it changes ranking only, with existing confirmation, eligibility and storage preserved. Additional diagnostics count wrong output farther than 0.5 m from present truth, total correct-track unavailability, and longest correct-track gap. They do not currently change acceptance. A user clarification is pending because raw missing time credits even a clearly wrong output as retained.
