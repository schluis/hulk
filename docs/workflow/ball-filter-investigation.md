# Ball-filter investigation — living plan and experiment memory

Last updated: 2026-10-05. Owner: the agent continuing this investigation.

This is the canonical index of current work, decisions, and explored ideas.
Update it **before starting an experiment, after obtaining a result, and before
ending a work session**. Keep rejected ideas and their evidence; do not replace
history with only the latest plan. Detailed reports stay in linked artifacts.
This is maintained during work sessions, not by an unattended background job.

## 1. Goal and constraints

### 1.1 Desired behavior

Track the real ball with a persistent model whose **position and velocity** are
accurate. Reduce duplicate hypotheses, wrong associations, and wrong selected
outputs. One persistent track is the target for an unambiguous single ball;
temporary alternatives are legitimate under ambiguity. A small hypothesis count
alone is not success: deleting the ball or merging clutter into it is worse.

Priority: position and velocity within **1 m**, particularly fast shots near the
robot, kicks, and recovery after a robot obstacle occludes the ball.

### 1.2 Working constraints

- Prefer the existing extended filter and small demonstrated improvements.
- Keep experimental algorithm changes inside the ball filter; instrumentation,
  scoring, and simulator work may support evaluation.
- Commit locally; **do not push** without a new user instruction.
- Use `/home/schluis/hulk-ball-filter-preferred-simulator`, branch
  `dev/ball-filter-preferred-simulator-20261004`. Leave unrelated main-worktree
  changes untouched. Confirm checkout and status before edits.
- CPU use may reach 100% at low priority. Aggregate compute memory limit: 45 GiB
  (40 GiB high watermark, no swap); use the existing resource runner.
- Previous agents are stopped. Do not restart parallel tuning automatically.
- Parameter limits have literal numeric semantics; zero is not infinity.

## 2. Current checkpoint

### 2.1 Retained implementation

At `0ed7a0910`, the latest investigation retained a **selection tie fix**:
equal ranks prefer recent observations, then effective confidence, without
relaxing confirmation eligibility. The accepted benchmark remains unchanged; see the experimental checkout below.
Resting/moving transition experiments were rejected. Tooling now includes v11
fast-close velocity loss, initial-position-covariance tuning, seeded no-search
demos, and the image observer.

E06 association tracing is complete. E07 tested a full-span motion-significance
check with fixed parameters and was rejected on all three replay partitions.
The retained baseline runtime and parameters are those of `0ed7a0910`. The accepted baseline runtime and parameters remain the comparison reference.
Goal mode is active at the user’s request. E08 failed fresh validation; E10
retention calibration found a publication dependency (E15). A joint primary/auxiliary
parameter search completed (E16) on all 54 inspected development clips. The
parameter-only ablation-0260 candidate failed the second audit’s per-suite
close-position guard. E18 completed the stationary-noise calibration; E19
rejected local-0280 on a fresh coasting-position regression. E20 produced
local-0322; E21 rejected it on fresh suite and fast-velocity regressions.
Stop only after a candidate beats the frozen current best with improved velocity
and passes fresh validation; do not declare diagnostic progress a completed goal.

Next: calibrate measurement noise around E70's verified covariance geometry, using
original-runtime controls; E69/E70 fixed-parameter variants fail close accuracy,
and extend simulator/evaluation coverage for the user's fast and airborne kicks.
E67's smooth position-weighted velocity credit is implemented as a diagnostic;
the publication candidate slightly worsens it. E66's moving reconfirmation fails
full replay and is archived, with working runtime restored.
E56 isolates excessive velocity noise as a cause of direction jitter. E62's paired
216-recording audit confirms E61's one-second clear-miss gate removes 57.132 seconds
of wrong output, loses no correct output, and changes no remaining estimate.
Correctly associated close-velocity error and coverage are unchanged: this is a
false-output improvement, not a velocity winner. The production objective remains
v11; supplementary scoring is diagnostic only. No accepted winner. Experimental
runtimes are archived and prior runtime restored; no evaluation jobs remain running.
Reserved67/68 remain unscored. Working runtime has
experimental optional cap. At the user’s explicit request to push for local testing,
the checkout now uses E43 parameters; this is not acceptance or a merge recommendation.


### 2.2 Baseline and terminology

- PR #2944 is the game-version reference supplied by the user; do not substitute
  a later preferred snapshot when reporting “game” results.
- `e5672d82e` means **before the latest motion investigation**, not main or game.
- The preferred branch is the extended normal filter, not a PDA implementation.
- Historical method comparisons used different data/objectives. Their absolute
  scores cannot be pasted into the current comparison as if equivalent.
- v10 and v11 scalar losses are not comparable. Compare physical metrics on
  identical inputs, or re-score both implementations with one objective.

## 3. Next work — ordered plan

### 3.1 A: Establish an association trace [INITIAL TWO-CLIP AUDIT COMPLETE]

**Question:** For each returning/fast-ball detection, why did the existing real-ball
hypothesis fail to receive it?

1. Freeze the current code, parameters, recordings, and evaluator identity.
2. Start with fast-near-shot and contested/occluded-kick recordings from the
   existing image-interface capture. Inspect both successful and failed matches.
3. Trace timestamps, detection projection, each predicted position/velocity and
   covariance, gate outcomes, association cost, assignment, spawn/merge/delete,
   resting/moving transitions, confirmation, and selected/published output.
4. Use diagnostic stable identities where necessary. A vector index is not a
   persistent hypothesis identity. Keep instrumentation out of normal hot paths.
5. Classify failures: projection/timing, gating, assignment competition, poor
   velocity initialization, premature resting, retention/deletion, or selection.

**Deliverable:** a small set of reproducible annotated failure windows with the
first incorrect decision identified. Separate measured causes from hypotheses.
**Exit:** select the smallest correction supported by those traces.

### 3.2 B: Improve velocity acquisition [ACTIVE; see E56–E63]

#### B1: Two-observation velocity initialization [TESTED; REJECTED E25]

E25 tested this idea and failed the isolated-false-confirmation check. Do not
repeat without a new mechanism addressing that failure. Original question:
whether compatible timestamped observations can initialize velocity with
appropriate uncertainty, leaving isolated detections tentative. Account for
robot motion and observation interval. Check noise amplification at short
intervals and false pairing across obstacles/clutter.

This is **not** the same as the rejected experiment that merely delayed resting
until multiple observations (E02 below). Do not conflate their conclusions.

#### B2: Existing-model velocity correction [DIAGNOSED; CALIBRATION ONGOING]

E56 confirms excessive velocity process noise causes rapid reversals; lowering
it helps stability but has recovery tradeoffs. E63 checks publication calibration
using correctly tracked velocity before further search. Original question:
whether uncertainty allocation makes updates correct position while leaving
velocity too small. Initial position covariance is now tunable, but its search
with the previous resting experiment did not yield an accepted candidate.
Inspect the actual update before introducing another model or parameter.

### 3.3 C: Improve recovery and matching [INVESTIGATED; NOT RESOLVED]

#### C1: Propagation and gates after occlusion

Verify elapsed-time/odometry propagation, uncertainty growth, Euclidean limits,
and statistical gates against returning-ball residuals. Identify the rejecting
gate before loosening it. Test unrelated nearby detections as negative examples.

#### C2: Assignment, duplicate creation, and retention

Check whether multiple valid assignments compete incorrectly and whether new
hypotheses repeat the same ball history. Fix the cause before adding aggressive
merging. Analyze stale-track deletion separately from selection and publication.

### 3.4 D: Calibrate trajectory prediction [DEFERRED until association improves]

Keep the current Kalman framework initially. Calibrate damping using full-filter
position/velocity and forecast errors, not only true-state trajectory fits.
Only test extra dynamics if residuals show a systematic unexplained effect.
Behavior/interception code extrapolates separately; changing filter dynamics does
not automatically change interception. Behavior changes are outside this step.

### 3.5 E: Validate and decide [REQUIRED for each candidate]

- Change one mechanism at a time; preserve a baseline and perform an ablation.
- Report within-1m position and velocity RMSE, fast-close velocity RMSE, first
  200 ms after kicks/returns, and forecast errors at 100/300/500 ms.
- Add truth-attributed duplicate births, wrong associations, wrong selections,
  track survival/switches, and reacquisition delay. These identity-based metrics
  are **planned**, not yet established by total hypothesis counts.
- Report missing/correct-ball-unavailable time and clutter failures alongside
  candidate reduction, so suppressed output cannot look like an improvement.
- Preserve per-scenario results and reference durations. Current hard optimizer
  guards remain aggregate close position/velocity RMSE; proposed new metrics
  need explicit definitions and a frozen acceptance rule before selection.
- Freeze the candidate before fresh evaluation. Inspected holdouts become
  development data. Keep coverage failures visible; do not cherry-pick seeds.
- Verify capture-version live/replay parity. Cross-algorithm evaluation is not
  evidence of matching the original live output.

## 4. Experiment tree — preserve outcomes

### Mechanism index — read this before reopening an idea

The stable experiment IDs below preserve chronology and evidence. This index groups
related trials by mechanism; later trials supersede earlier plans, not their results.

- **Tracking and motion**
  - Association, initialization and reacquisition: E06/E07, E09, E13, E25, E29, E45/E46.
  - Resting transitions and trajectory models: E02, E04/E05, E24.
  - Velocity stability, process noise and damping: E18–E22, E27, E34, E53, E56–E58.
- **Candidate lifecycle and selection**
  - Ranking, confirmation and confidence caps: E01, E23, E39–E44, E52, E54.
  - Retention, visibility and duplicate merging: E10–E12, E14, E36–E38, E47, E55.
  - Publication versus internal retention: E15/E16, E26, E31/E32, E50/E51, E59–E61.
- **Evaluation and decision making**
  - Parameter calibration and held-out failures: E03, E08, E17–E22, E27/E28,
    E30, E33–E44, E48.
  - User checkout provenance: E49.
  - Correctness-aware loss and paired coverage audit: E62.
- **Archived alternatives**: section 4.5; no active alternative-filter search.

### 4.1 Selection

#### E01 — Equal-rank tie handling [RETAINED]

Problem: cap zero/saturated scores allowed iteration order to select a stale
confirmed track despite a recently observed moving one. Original fast-near-shot
at 10.190 s: moving hypothesis estimated 5.77 m/s, selected resting track was
7.188 s old.

Change: recency, then effective confidence, breaks ranking ties. On nine seed-4243
recordings, close position RMSE improved 0.22013 → 0.15759 m; close velocity
0.58925 → 0.58672 m/s. This fixes selection, not hypothesis creation.

Fresh seed 8675309 completed only stationary-close and approach: position
0.17811 → 0.14417 m; velocity unchanged at 0.39275 m/s. Fast velocity and
hypothesis counts were unchanged. Third scenario failed required reacquisition
coverage. **Not a completed independent nine-scenario audit.**

Evidence: `M/tie-only/`, `M/final-before/`, `M/final-retained/`,
`M/final-holdout-capture.log`. Retained code: `a364d14bb`.

### 4.2 Resting transitions and initialization

#### E02 — Delay resting using uncertainty or observation count [REJECTED]

- Covariance-aware resting plus tuning: promising training results, rejected
  development regressions. Preserved at
  `experiment/ball-filter-covariance-resting-20261005` (`77230d7a0`).
- Require multiple distinct observations before speed-based resting: 38 tested
  configurations, no accepted validation improvement. Preserved at
  `experiment/ball-filter-observation-rest-20261005` (`6f8f013f7`).

These outcomes reject the tested variants, not all velocity initialization work.
Reopen only with a trace-supported mechanism or materially different evidence.
Evidence: `M/v11/decision.json`, `M/observation-rest/decision.json`.

### 4.3 Parameter search and scoring

#### E03 — Lower-noise close-range tuning [CANDIDATES REJECTED]

v10: 32,768 trials + 96 probes. Aggregate gains hid poor first-200-ms fast-shot
behavior; narrow matching distances were a concern.

v11: independently normalized fast-close velocity loss added so stationary time
cannot dilute it; shared initial x/y covariance added to search. 16,384 trials +
56 probes, stopped at four-batch budget, **not a proven plateau**. Training winner
worsened development close position by 10.3% and velocity by 14.5%, despite better
fast-close velocity. Candidate defaults were not adopted. Scoring/search tooling
was retained. Evidence: `M/v11/comparison.json`, `M/v11/protocol.json`.

### 4.4 Motion prediction

#### E04 — Exponential damping versus constant deceleration [DIAGNOSTIC ONLY]

Current transition already integrates `dx/dt=v`, `dv/dt=-lambda*v`: fixed heading
with decreasing speed. An offline fit on truth favored calibrated exponential
damping over constant deceleration on held-out smooth fast-close simulator rolls.

| Oracle-initialized prediction | 300 ms RMSE | 500 ms RMSE |
| --- | ---: | ---: |
| Current damping | 0.11963 m | 0.24401 m |
| Fitted exponential damping | 0.06834 m | 0.10102 m |
| Fitted constant deceleration | 0.07358 m | 0.11832 m |

Starts from true position/velocity. Smoothness uses future truth; windows overlap.
This is not end-to-end tracking, a real-world validation, or evidence of fewer
hypotheses. No model or damping change was adopted.
Evidence: `M/trajectory-audit.json`, `M/trajectory_audit.py`.

#### E05 — Curved/spin-aware or extra motion models [DEFERRED]

No current evidence justifies this complexity. First distinguish smooth-model
error from missed observations, incorrect velocity, contacts, and association.
Do not treat this as a tested rejection.

### 4.5 Earlier alternative filters [ARCHIVED; not an active search]

The earlier exploration includes conditional PDA, Student-t reweighting, internal
IMM, output IMM, selection-only, soft geometry, reacquisition protection, and an
auxiliary publication filter. Implementations and outcomes are recorded in
[the historical method comparison](../../tools/simulate/ball-filter-method-comparison.md)
and [the lab history](ball-filter-lab.md). The auxiliary publication correction
was integrated into the extended baseline; it is not a new candidate here.

The selected parameter candidate was evaluated on the normal filter **without
PDA**. Equal normal/PDA benchmark results do not establish a PDA benefit. Earlier
rejections depended on their particular variants, objectives, and safeguards;
consult the original report before making a stronger claim or repeating work.
Reopen an archived approach only with a new trace-supported reason and a minimal
comparison against the current normal filter.

### 4.6 Association and evidence diagnostics

#### E06 — Detection-to-track assignment trace [COMPLETE, 2026-10-05]

Baseline `bf0e9df22`, unchanged preferred parameters. Replayed original
fast-near-shot and contested recordings with temporary instrumentation of
predicted states, measurement covariance, gate outcomes, assignment, and updates.
The two compact frame exports are **byte-identical** to the frozen uninstrumented
baseline. Instrumentation was removed after freezing its executable and patch.
No runtime logging or new dependency remains in the preferred branch.

Findings (times relative to the recording's first output cycle):

- **Fast-near-shot, second shot:** ball-supporting detections at 10.002 and
  10.042 s match the recent resting hypothesis; the next input at 10.082 s
  already sees it moving. No repeated ball birth in this window. A new detection
  at 10.202 s is about 4.97 m from the nearby truth sample, while the supporting
  detection still matches the moving hypothesis. It is unsafe to call every
  additional hypothesis a duplicate of the ball.
- **Contested:** returning detection at 12.522 s spawns one new hypothesis.
  The two closest retained predictions are 1.41/1.58 m away and were last seen
  11.64/7.96 s earlier; both fail reacquisition distance. These are stale
  predictions, not evidence that widening the gate would restore the true track.
  Subsequent detections match the fresh track. It becomes resting before its
  second observation, and internal moving mode resumes at 12.962 s, 440 ms after
  birth. Published/selected moving mode appears at 12.998 s.
- Three-observation motion confirmation first rejects velocity inconsistency,
  then repeatedly rejects one insufficiently significant short displacement.
  For example, the input at 12.682 s has displacement significances 11.83 and
  3.08 against a required 9, while velocity inconsistency is only 2.74 against
  its maximum 9. This is delayed motion recovery despite successful association.
- The statistical association distance uses prediction covariance `P`, excluding
  measurement covariance `R`. Recomputing with `P+R` changes **zero** pair
  admissions in these two traces after the existing distance gates. This remains
  a modeling question, but is not the demonstrated cause of these failures.

Birth diagnostic (not truth identity classification): fast-near-shot has 27
births, only one within 0.3 m of the nearby cycle's truth; contested has 17 births,
three within 0.3 m. This includes initial acquisition. Truth is sampled at the
next output cycle, not interpolated to detector time, and percept projection
has error. Consequently “outside 0.3 m” is a supporting diagnostic, **not proof
of a false detection**. Broad claims about real-world duplicate frequency are
not warranted from these two synthetic clips.

Row indices are local to an exposure and change when hypotheses are sorted or
removed. The trace associates an assigned row with its before/after state and
last-observation time; it does not introduce permanent track identity metrics.
Spawns are measured from unassigned percepts, not changes in vector length.

Evidence root `A` =
`/home/schluis/hulk/logs/ball-filter-association-20261005/`:
`baseline.json`, `instrumentation.patch`, `trace-evaluate`, `trace.log`,
`trace-parity.json`, `analyze.py`, `pairs.json`, `motion.py`,
`motion-reasons.json`, and `replay/`. Original capture-version parity was verified
previously; this new parity check establishes instrumentation equivalence only.

#### E07 — Full-span motion significance [REJECTED, 2026-10-05]

**Hypothesis:** test significance over the full three-observation displacement
instead of requiring each short displacement to exceed three sigma. Retain three
observations, direction agreement, velocity consistency, time-gap handling, and
correlated initial covariance. No parameter search or new parameters.

The occluded-kick example improves selected-mode activation from 12.998 to
12.696 s (302 ms earlier), but aggregate accuracy worsens:

| Replay partition | Close position, before → candidate (m) | Close velocity, before → candidate (m/s) |
| --- | ---: | ---: |
| Original seed 42, nine clips | 0.18484 → 0.20317 | 0.65101 → 0.75155 |
| Training seeds 42/43, 18 clips | 0.37891 → 0.43835 | 0.75328 → 0.89849 |
| Development seed 4243, nine clips | 0.15759 → 0.23286 | 0.58672 → 0.71360 |

Fast-close velocity also worsens in all partitions. These are identical-input,
fixed-parameter comparisons against the retained tie fix. All data were already
inspected; no fresh held-out audit was consumed. Passing 118 unit tests establishes
code invariants, not quality. Rejected before promotion, despite the better
individual example. Do not retry this relaxation unchanged.

Preserved local branch: `experiment/ball-filter-span-motion-20261005`, commit
`609a6acef4dc4247c9cec600ac18c5ca816ae7e3`. Evidence: `A/decision.json`,
`A/span-experiment.patch`, `A/span-evaluate`, `A/span-{original,training,validation}/`,
`A/build-candidate.log`, and `A/evaluate_candidate.py`.

**Next question:** can a calibrated initial velocity and its uncertainty improve
accuracy, rather than merely activating moving mode earlier? Separately determine
whether long-lived off-truth births remain visible because of uncertainty,
occlusion, retention rules, or real subsequent support. Do not conflate that
lifetime problem with failed matching of consecutive real-ball observations.

### 4.7 Active goal: beat the current tuned best

#### E08 — Calibrate existing motion/update model [REJECTED AFTER FRESH AUDIT]

Frozen baseline: `ed1d39b67` runtime, original preferred parameters, tie-only
frozen evaluator. Goal: better close velocity with no close-position regression,
then independent fresh validation. Inspect fast velocity, missing-ball time,
wrong/false output and hypothesis behavior before promotion.

First isolate parameter effects on the **retained** algorithm, unlike previous
searches combined with rejected resting changes. Probe damping, moving position
versus velocity process noise, initial position covariance, and resting threshold.
No new algorithm complexity. All 36 existing recordings (original nine, training
18, prior development nine) are now explicitly development data. Candidates must
pass close position/velocity checks in each partition; final fresh data remain
unseen until candidate freeze. A parameter probe is not an adopted candidate.

Evidence: `/home/schluis/hulk/logs/ball-filter-goal-20261005/`, including
`protocol.json`, `probe.py`, `configs/`, and immutable evaluator identity.
16 concurrent low-priority evaluation processes through the 45 GiB resource
runner; no subagents. Further experiments follow evidence, not a fixed trial cap.

38 initial probes completed. A 192-combination search then found combo-032,
which improves close position, close velocity, and fast-close velocity in all
three partitions. Pooled close position 0.29143 → 0.22018 m; close velocity
0.68842 → 0.65429 m/s; fast-close velocity 2.28775 → 2.23075 m/s. False-track and
velocity-unavailable durations are unchanged; wrong-output time varies and must
be reported. These are development results only, not goal completion.
All 192 local perturbations completed. Frozen winner: **refine-106**, selected
by summed close/fast-close velocity ratios among candidates that improve all
three accuracy metrics in every development partition. Across 36 recordings:
close position 0.29143 → 0.21530 m (−26.1%); close velocity 0.68842 → 0.63838 m/s
(−7.27%); fast-close velocity 2.28775 → 2.20538 m/s (−3.60%).

Diagnostic limitations: global velocity RMSE worsens 2.48%; first-200-ms velocity
RMSE improves only 3.11421 → 3.09140 m/s, with position almost unchanged
(0.29572 → 0.29595 m). Mean total hypotheses rises 6.547 → 6.670 (maximum 15),
so this is not evidence of fewer candidates. This count includes clutter.
Fresh validation is running with the frozen candidate and protocol; no success
claim until that completes. Seed 50000042 failed fast-crossing coverage (no
reacquisition opportunity) before accuracy evaluation. Seed 50000043 completed
all nine scenarios; seed 50000044 is running. No fresh accuracy results have
been inspected. Preferred default parameters are still unchanged.
`fresh-protocol.json` predeclares two complete new nine-scenario suites with
coverage-only replacements and no accuracy-based seed selection.

Fresh audit completed on seeds 50000043/50000044 (18 recordings), with exact
baseline live/replay parity. Close position improved 0.23391 → 0.22171 m (5.21%),
but close velocity improved only 0.61622 → 0.61405 m/s (0.35%) and fast velocity
2.39468 → 2.38157 m/s (0.55%). Wrong-track time rose 73.688 → 88.034 s;
false output rose 68.696 → 76.492 s. The second suite regressed both close metrics.
**Rejected; goal not reached.** Evidence: `fresh-comparison.json`,
`fresh-breakdown.json`, `audit-decision.json`. No preferred parameters changed.
The inspected 18 recordings now join development (54 total).

#### E10 — Retention and observability calibration [COMPLETED; NO PROMOTION]

E08's fresh failure shows that noise/damping gains alone can preserve or select
more clutter. Test existing hypothesis timeout, hidden confidence decay, and
visibility uncertainty margin with both original and E08 motion parameters.
No new algorithm or confirmation relaxation. Retained covariance grows while
hidden; a 3-sigma rectangle often cannot certify a clear camera miss, leaving
low-confidence clutter governed by weak hidden decay and a 20-second timeout.
This is a code-supported mechanism, not yet a proven correction.

Acceptance includes close position/velocity and fast-close velocity on each of
four development partitions, plus false/wrong output and availability diagnostics.
Do not accept a smaller hypothesis count achieved by losing the real ball.
Fresh data for the next frozen candidate must use new seeds, separate from E08.
Evidence: `ball-filter-goal-20261005/retention/`.

64 ablations completed. Blanket shorter timeouts damage close-position accuracy
by deleting established hidden tracks. Lower visibility margins can reduce false
output but lose close-ball availability. The modest feasible motion/retention
combinations still retain E08's wrong-output regression. Do not promote them.

#### E11 — Expire weak, unsupported hypotheses sooner [COMPLETED; NOT ADOPTED]

E10 shows established and tentative tracks cannot share a shorter timeout safely.
Test a one-second unseen timeout only while validity is below the existing
confirmation threshold (max of 3 and output threshold); established tracks retain
the original timeout. Use a diagnostic constant first; add a clearly named literal
parameter only if evidence warrants production adoption. This is current support,
not a claim to store historical confirmation. Check hidden tracks whose confidence
has decayed as well as single false births. Compare both baseline and E08 motion
parameters on 54 development recordings; record actual hypothesis counts and
availability before any new fresh audit.

#### E09 — Preserve the birth observation for motion evidence [NOT ADOPTED]

The existing three-observation motion test starts only when updating a resting
track. A newly spawned track discards its first measured position before entering
rest, effectively requiring four detections. Test retaining that actual birth
measurement and its covariance, transforming it with odometry, and preserving
it through the initial moving-to-resting transition. Keep all three-observation
significance/consistency checks. Clear evidence after a moving update or activation.
This differs from E02/E07: it preserves valid data rather than relaxing evidence.
Compare fixed parameters first, then promising E08 parameters if warranted.

Completed on all 36 development recordings with baseline and E08 combo-032
parameters. Close position, close velocity, and fast-close velocity metrics were
exactly unchanged; overall wrong-track time increased by 0.388/0.160 s,
respectively. 118 tests passed, including birth-to-third-exposure motion and
stationary behavior. No demonstrated benefit to justify adopting the change.
Preserved branch: `experiment/ball-filter-birth-evidence-20261005`;
`ball-filter-goal-20261005/birth-evidence/decision.json` contains its commit and
metrics. Preferred runtime source restored.


#### E12 — Recheck physical size gating against new clutter failures [REJECTED]

Revisit the existing maximum_detection_radius_ratio parameter (currently 1e6),
with limits 2, 3, 5, and 10, using baseline and E08 motion settings. Earlier size
gates/soft geometry failed different datasets and safeguards; E08's new false
output regression motivates this bounded recheck, not a claim that it is novel.
Preserve old rejections. Synthetic false boxes can make size filtering look too
favorable; require true-ball availability and fast-shot coverage, and disclose
real-detector transfer uncertainty. No new geometry model. Eight probes completed: ratio 2 cuts false time by about
44 s but worsens close position on the new audit data by about 77%; ratio 10 has
no meaningful effect. Intermediate limits do not pass all accuracy partitions.
This confirms, rather than overturns, the earlier rejection.

#### E13 — Weighted three-observation velocity initialization [NOT ADOPTED]

Fit position and velocity jointly to the existing three timestamped observations,
weighted by their full measurement covariance. Use fit residuals to reject
inconsistent motion and velocity covariance to require significant motion.
Retain interval and direction checks. Unlike E07, this estimates the initial
state and its full correlated uncertainty jointly; it does not just loosen two
displacement tests while copying the last position. First probe: squared velocity
significance 18, residual consistency 9. No extra history or predictor model.
Compare baseline parameters, E08 parameters, and E08 with weak timeout 2 s.
Record stationary/outlier behavior and reject aggregate regression.

The initial joint-fit activation rule failed: close velocity regressed 9.5% on
seed 4243 with motion/weak settings (and 12–18% across partitions with baseline
parameters). Next ablation restores **all original motion-confirmation checks**
and changes only the weighted initial state/covariance estimate. This separates
a better estimator from a relaxed activation decision. First binary and patch
are preserved in `weighted-init/`; the ablation is in `original-gates/`.
The original-guard ablation produced only tiny mixed changes (roughly 0.1% on
baseline velocity), insufficient to justify the added estimator. Restored the
original motion-evidence algorithm; keep the saved source and binaries as evidence.

#### E14 — Joint parameter search with weak-track retention [COMPLETED; FOLLOW-UP E16/E17]

E11's weak-track timeout reduces mean hypotheses from roughly six to about two,
without broadly expiring supported hidden tracks, but its interaction with motion
noise needs calibration. Search existing motion/noise/association settings plus
the optional weak timeout on the 54 inspected recordings. Keep all original
motion-confirmation checks. Reuse recordings in memory across low-priority worker
threads rather than reload them for each candidate. Baseline parity is checked
before search. No fresh validation data are included. Also probe the existing auxiliary detection
noise and blending fraction, which were not dimensions of the preceding native
22-parameter search. The auxiliary tracker can dominate the published position
and velocity when the primary observation is geometrically implausible; tuning
only primary detection noise does not calibrate those updates. Keep the original
availability guard. Include no-weak-timeout controls to avoid adopting new code
if parameter-only changes suffice.

#### E15 — Remove dependency on unrelated primary clutter [REJECTED AS PROPOSED]

Tracing E11's lost close output found an architectural coupling in the existing
auxiliary publication correction. Fresh-50000043/approach at 11.304 s: selected
primary is an 8.036-second-old resting hypothesis at Ground (1.44, 2.55), validity
0.795, with physically implausible last image size. Yet the published estimate
is near (-0.162, 0.092), from the auxiliary tracker; truth is (-0.606, 0.056).
Deleting the weak primary removes publication eligibility, hiding that separate
estimate. This is not loss of the primary's accurate ball trajectory.
Evidence: `tentative/approach-11.304.json` and its earlier snapshots.

Test independently publishing the **existing** geometry-supported auxiliary
estimate when no primary is eligible. Require confirmation-level effective
validity under the primary field prior, plus the existing publication age and
distance limits. Relative covariance comparison continues to apply when blending
with a primary. No additional tracker, no change to ordinary primary updates.
Evaluate alone and with weak expiry; scrutinize absent-ball false output before
promotion. This deliberately revisits the old baseline-availability constraint
because it obstructs removing unrelated clutter; it is a behavior change, not a
claim that the old preservation safeguard was implemented incorrectly.

The existing clear-view-miss test exposes an unacceptable consequence: unrestricted
fallback republishes old auxiliary history after the primary correctly disappears.
Do not weaken that test to promote the candidate. Restored the original availability
guard. A viable future design needs independent negative evidence/freshness, not
just auxiliary confidence. The lost-output trace also shows that this is not
simply historical confirmation decaying: the weak primary was a lone unrelated
birth (validity 1), while the accurate published estimate came from elsewhere.
Thus adding a historical-confirmation bit would not fix this particular loss.

### E16 — Joint primary/auxiliary calibration on 54 development clips

**Status:** preparing/running, 2026-10-05. The preceding primary-only searches
left `publication_detection_noise` and blending fixed, despite the auxiliary
tracker dominating some published estimates. Test these existing controls jointly
with motion/process noise and optional weak expiry, without a new tracker.

`G/joint/protocol.json` freezes 1,130 configurations (seed 2026100524), including
baseline/motion controls, an auxiliary-noise grid, selection-cap controls, and
1,024 random combinations. Four disjoint groups are all development data now.
Require no group close-position/velocity/fast-velocity regression; inspect bounded
losses and missing output to reject gains caused by hiding difficult estimates.
False/wrong output must improve before consuming another independent audit.
A shared-recording threaded evaluator reduces memory and redundant replay work.

Preflight detected a stale build: source restoration preserved old modification
times, so Cargo reused the rejected weighted-initialization object code. This was
caught by baseline parity **before launching the search** (119 tests rather than
118, and small metric discrepancies). Touch restored runtime sources and rebuild;
exact baseline parity was subsequently verified across all four groups / 54 clips
(`G/joint/parity-receipt.json`), with 118 tests passing. Preserve the failed
preflight output; do not interpret it as a parameter result. Future restorations
must update modification times and check baseline parity.

The first 1,130 configurations completed. Thirty passed close physical RMSE
checks, but none improved while also meeting all bounded-loss and false/wrong
output checks: weak expiry hides additional output in the diagnosed approach
case. A focused 2,048-configuration local search now holds weak expiry disabled,
varies primary/auxiliary noise, damping, matching and hidden decay around four
centres, and keeps the baseline selection cap at literal zero (recency breaks
ranking ties; this is not an uncapped setting). Seed 2026100525, exact protocol and configs
under `G/joint/refine-*`. Do not promote conditional-error gains caused by loss
of availability. All results use the parity-verified rebuilt binary.


### E17 — Simplified parameter candidate, second independent audit

**Status:** frozen before capture; running. E16 plus focused retention/selection
and ablation searches used only the same 54 development clips. Exact configs,
protocols, reports, and summaries are under `G/joint/`. `ablation-0260` retains
almost all of local-0855's roughly 30% close-position and 6.5% close-velocity
improvement with only damping, matching distance, and primary noise changes.
No auxiliary-parameter change or new runtime mechanism is necessary.

The earlier E16 screening rule also required global false and wrong output to
improve. No substantial close-range winner satisfied that extra requirement.
Before capturing this audit, explicitly return to the user's stated priority:
close-range position and velocity are the hard requirements; global behavior is
reported and inspected, rather than an automatic veto. This candidate has no
extra missing output and less close-range wrong tracking on development data,
but about 5–6% more global wrong-track time and 3.4% more false output. Do not
hide that tradeoff or call it fewer wrong candidates overall. E08 remains rejected
because its second fresh suite regressed close position and velocity as well.

`G/audit2/freeze.json` hashes candidate and frozen baseline runtime binaries.
`G/audit2/protocol.json` freezes acceptance and seed order 50000045–50000052.
Capture the first two complete nine-scenario suites, replacing only capture
coverage failures before scoring; no replacement on accuracy. Both suites must
have non-regressing close position/velocity; pooled position/velocity and bounded
close losses must improve, fast-close velocity must not regress, and missing close
velocity time must not grow. Inspect scenario-specific failures and initial shot
recovery. No candidate changes after this freeze. Captures use baseline runtime
and parameters; evaluate baseline/candidate on identical inputs, strictly verify
baseline live/replay equality. Parameter SHA256:
`c6f596878caaaf1d4d3b4f7d2260c37e2a94bae4a49f08e8bd4edcc4b49d041c`.

Optional weak-expiry runtime code was restored out of the working tree; its
experiment branch and artifacts preserve the rejected investigation.


#### E17.1 Development evidence and operational checkpoint

Against the retained tuned baseline on 54 inspected clips:

| Metric | Baseline | Frozen candidate | Change |
| --- | ---: | ---: | ---: |
| Close position RMSE (m) | 0.272425 | 0.190507 | -30.07% |
| Close velocity RMSE (m/s) | 0.663743 | 0.620764 | -6.48% |
| Fast-close velocity RMSE (m/s) | 2.319617 | 2.257681 | -2.67% |
| Close velocity missing (s) | 6.506 | 6.506 | unchanged |
| Close correct-track missing (s) | 33.588 | 32.802 | -2.34% |
| Global wrong-track time (s) | 261.318 | 276.400 | +5.77% |
| False-output time (s) | 227.616 | 235.432 | +3.43% |
| Mean primary hypotheses | 6.3555 | 7.1587 | +12.64% |

Maximum primary hypothesis count remains 15. Counts include clutter, not tracked
truth identities. This is an accuracy/velocity candidate, **not** a demonstrated
reduction in all wrong candidates or a single-model solution. V11 aggregate loss
falls 0.543304 → 0.512535 (-5.66%). First-200ms kick/impact velocity RMSE improves
3.314619 → 3.264105 m/s (-1.52%) over 4.188 s of event coverage, with no missing
output. Original occluded-kick published motion starts 162 ms earlier. Recovery
plots and timing: `G/audit2/development-recovery.{png,svg}` and timing JSON.

Prepared configuration passes 117 filter tests and 43 tuner/observer tests
(one existing observer test ignored). No added runtime code. All calibration
uses the parity-verified binary; final candidate replays through the original
frozen baseline binary, independently of the abandoned weak-expiry feature.

Parallel audit capture hit the fixed Zenoh port 7448. Preserved seed-50000046's
startup failure; retry **the same seed** sequentially, not a coverage/accuracy
replacement. Seed 50000045 completed all nine scenarios. Seed 50000046 failed the existing
robot-fall validity check during fast-near-shot (21.4 s fallen), after three
completed scenarios; preserved and excluded before candidate scoring. The next
predeclared seed, 50000047, is now recording.
`G/audit2/resume_capture.py` resumes capture; `evaluate_fresh.py` waits for two
complete suites and then verifies/scores both frozen configurations. Artifacts
and logs persist under `G/audit2/`. Aggregate RAM remains under 10 GiB here;
45 GiB cgroup cap continues to apply. No subagents, no pushes.


#### E17.2 Independent audit result — rejected, improvement is not uniform

Seeds 50000045 and 50000048 completed all nine scenarios; seeds 46 and 47 were
excluded by existing fall-validation checks before candidate scoring. All 18
accepted clips passed strict baseline live/replay verification. Frozen candidate:

| Fresh metric | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| Close position RMSE (m) | 0.227709 | 0.177237 | -22.17% |
| Close velocity RMSE (m/s) | 0.621104 | 0.591335 | -4.79% |
| Fast-close velocity RMSE (m/s) | 1.967700 | 1.931275 | -1.85% |
| Close velocity missing (s) | 1.968 | 1.968 | unchanged |
| Close correct-track missing (s) | 6.802 | 7.278 | +0.476 s |
| Wrong-track time (s) | 67.132 | 71.188 | +6.04% |
| False-output time (s) | 78.880 | 78.880 | unchanged |

Both suites improve velocity. Suite 1 improves position 0.286639 → 0.191221 m,
but suite 2 regresses 0.145025 → 0.161785 m (+11.56%, +1.676 cm), violating the
predeclared guard. The stationary-close family regresses 0.137273 → 0.167123 m.
Do not change the criterion after seeing these results. Preserve candidate on
`experiment/ball-filter-velocity-audit2-20261005`; exact commit and decision at
`G/audit2/decision.json`. Restored original preferred parameter file. These 18
recordings are now development data, never again an independent heldout.

### E18 — Separate stationary-noise regression from motion improvement

**Status:** preparing, 2026-10-05. E17's optional parameter simplification grouped
all noise fields together, so it unnecessarily retained a higher resting process
noise. E16 one-at-time ablation already showed that restoring resting noise kept
almost all development gains. Recheck resting and detection noise with damping,
moving noise and matching distance, using all 72 now-inspected clips, with each
fresh suite separate. Runtime stays the original retained filter; no weak expiry,
new tracker, or new publication mechanism. Freeze any winner before another audit.


#### E18.1 Result and selected candidate

Completed 1,787 grid/random configurations and 1,024 focused refinements on the
72 development recordings. Baseline parity was checked across all seven groups;
the original three aggregate reports match exactly and four individual audit
suites match the frozen evaluator's per-recording moments within aggregation
roundoff. Protocols, all configs/reports, binary hash and selection script:
`G/stationary-calibration/`.

Select `local-0280`: minimum pooled v11 bounded loss among candidates satisfying
close-position/velocity guards in every group, unchanged close velocity
availability, and at least a 0.5% margin from every group's regression boundary.
Its worst group ratio is 0.992515. Pooled close position RMSE -22.90%, close
velocity RMSE -5.15%, fast-close velocity RMSE -3.02%; bounded v11 loss -5.71%.
This is still parameter-only. Global wrong/false tracking and hypothesis counts
remain diagnostics/tradeoffs rather than claimed improvements.

### E19 — Third frozen independent audit

**Status:** frozen and capturing, 2026-10-05. Candidate `local-0280`, SHA256
`3267d22db59e920d7467cbcbe9c55eec83690d6f4cfd515d429f4ae1f8ce50eb`.
`G/audit3/freeze.json` records candidate, evaluator and simulator identities.
Same acceptance rules as E17 (`G/audit3/protocol.json`): improve pooled close
position/velocity and bounded losses, no per-suite close p/v regression, no
pooled fast-close velocity regression or extra missing velocity output. Inspect
scenario failures, kick recovery and clutter diagnostics. No post-freeze tuning.

Fresh seeds are 50000049–50000060 in order. Run four captures concurrently using
**unprivileged user/network namespaces**, loopback enabled, so each uses the
simulator's unchanged fixed port 7448 without collisions. The frozen capture
binary and baseline parameters are unchanged. Memory remains in the existing
45 GiB aggregate cgroup at nice 15. Select the first two successful nine-scenario
suites in listed seed order; preserve failures and unused successful captures.
Only runtime/coverage validation may replace a suite, never accuracy. Strictly
verify baseline live/replay equality before candidate comparison. Root is the
only agent. Initial four simulations are running under about 8 GiB aggregate RAM.


#### E19.1 Independent result — rejected on coasting position

Seeds 50000049 and 50000050 are the first two complete suites by frozen seed
order. Seed 51 also completed but remains **unused/unscored**; seed 52 failed
fast-crossing reacquisition coverage (0 < 1). Both scored suites passed strict
baseline live/replay verification. Pooled close position RMSE 0.134035 → 0.139413 m
(+4.01%), velocity 0.709789 → 0.688203 m/s (-3.04%), fast velocity 2.481866 →
2.394730 m/s (-3.51%). Missing velocity time unchanged at 2.506 s. Bounded position
loss rises 0.378%; total v11 loss improves 4.86%. Suite 1 position regresses
0.171929 → 0.182354 m, violating the frozen rule. Reject despite the consistent
velocity gain; defaults remain unchanged. Preserved branch:
`experiment/ball-filter-velocity-audit3-20261005`, exact commit in decision JSON.

Diagnosis: seed49 fast-near-shot around 21.2 s, both filters retain the same
moving, size-plausible observed primary, last seen about 0.84 s ago. Published
output equals that primary (not auxiliary correction). Candidate has moved farther
past a ball that slowed during the gap: errors about 1.53 vs 1.21 m at the worst
close-range sample. There is no current detector observation to associate, and
both have eight hypotheses. Thus auxiliary publication changes or indiscriminate
merging do not target this failure. Evidence: `G/audit3/diagnosis/`; observer
8767 serves this candidate/clip with baseline capture parity checked.

### E20 — Motion-noise/damping tradeoff during unobserved slowing

**Status:** running, 2026-10-05. Keep the existing filter; calibrate a broader
range of damping jointly with moving noise, measurement noise and matching range
on 90 inspected recordings, with each old audit suite separate. Preserve the
velocity gains while reducing excess integrated position during unseen slowing.
Do not claim this predicts unobserved collisions perfectly; do not add obstacle
physics or another tracker without evidence. Seed51 remains untouched as unused
validation material. Freeze before another independent audit.


E20 protocol: `G/coasting-calibration/` contains 2,107 frozen grid/random
configurations (seed 2026100530), nine disjoint development groups, baseline
parity receipt, source and frozen evaluator. The grid preserves baseline resting
noise while varying existing moving/detection noise, damping and matching range.
Require close p/v guards on every group, unchanged availability, improved bounded
losses before freezing. No fresh seed51 data used. Memory cap/nice scheduling
unchanged. Moving prediction was inspected: it already integrates damped motion
and continuous process covariance, including cross terms. The large covariance
seen here comes from configured noise, not a missing elapsed-time integration.


#### E20.1 Completed calibration

2,107 initial configurations plus 768 local refinements completed on the same
90 development clips. Select `local-0322` by minimum pooled v11 bounded loss among
feasible candidates with at least 0.5% margin in every group's close p/v RMSE.
Worst ratio 0.992401; pooled position -21.71%, velocity -4.46%, fast velocity -1.57%,
v11 bounded loss -3.41%. Availability unchanged. Original resting-process noise
is retained; only existing damping, detection/moving noise and matching distance
change. No runtime change. Exact selection: `G/audit4/development-selection.json`.

### E21 — Fourth frozen independent audit

**Status:** capturing, 2026-10-05. Freeze `local-0322`, SHA256
`6b57303480d6a41a2ac34d86da7375e12c96b3ade002a6b9bed4dbb64639ab8d`.
Artifacts/protocol under `G/audit4/`. Same acceptance criteria as E17/E19, fixed
before capture. Seeds 50000053–50000064 in order, four concurrent isolated
network namespaces; first two complete valid suites by listed order, no
accuracy-based replacement. Verify baseline live/replay before scoring.
Unused seed51 remains unscored. No candidate modifications after freeze.


#### E21.1 Rejected result and duplicate assessment

All four suites captured successfully. First two seeds53/54 were selected as
prespecified; seeds55/56 remain unused, alongside audit3 seed51. Strict baseline
replay passed. Close position 0.461140 → 0.432526 m (-6.21%), velocity
0.765589 → 0.692426 m/s (-9.56%), but fast velocity 2.287083 → 2.324364
m/s (+1.63%). Suite1 regresses both position (0.169313 → 0.204365 m) and
velocity (0.598235 → 0.609586 m/s); bounded position loss +4.96%. Reject.
Wrong output 123.946 → 131.374 s; false output unchanged79.8 s, missing
close velocity unchanged2.376 s. Do not promote pooled improvements.

Instrumented replay has exact score/parameter parity. Within1m, time with
multiple primary hypotheses within0.30m of truth declines37.090 → 35.176 s,
but zero-near time increases17.074 → 17.824 s. Mean total primary count
increases6.889 → 7.175. Geometric proximity is not persistent identity and
excludes the auxiliary tracker. Evidence: `G/audit4/hypothesis-proximity.json`.
Candidate is preserved unchanged for the user's local inspection; defaults
unchanged. `G/audit4/decision.json` records rejection.

### E22 — Conservative parameter calibration after four failed audits

**Status:** running. New development set adds only inspected seeds53/54, for
108 clips in11 separate groups. Independently interpolate moving noise,
detection noise, damping and matching range between original baseline and E21.
This tests whether smaller changes preserve gains without the regressions from
large process-noise increases. 272 configurations including controls; strengthen
development screen to require fast-velocity non-regression in each group as well
as close position/velocity and availability. Fresh audit remains required.
Artifacts: `G/conservative-calibration/`. No runtime changes, agents or pushes.

Initial272 complete: only identical baseline configurations pass every guard.
Closest grid0173/0218 improve pooled velocity about6%, but the older development
suite regresses0.19–0.22%. Run768 seeded local refinements around those two,
retaining the stricter fast-per-group screen. Baseline matches previous nine
groups exactly and two new verified groups to1e-10 RMSE. Build passes117 tests.

E21 fast-velocity diagnosis: largest incremental squared error in seed54
fast-crossing at2.6s: truth field velocity about(1.98,0.38)m/s, baseline reports
rest, candidate predicts motion roughly opposite, last observation38ms ago.
Approach5.4–5.8s similarly retains obsolete motion during observation gaps.
Artifacts: `G/audit4/diagnosis/fast-regressions.json`; these are paired sample
errors, not proof of a new association failure. No runtime change inferred yet.

### E23 — Cap accumulated confidence (user suggestion)

**Status:** experimental. Stored validity currently accumulates per matched
observation without a ceiling. The existing selection cap affects output ranking,
not stored support, pruning or slot retention. Add an optional literal maximum
stored confidence, applied before pruning/merging. Omission preserves historical
behavior; zero means zero support. Test caps3,5,10,20,50,100 on baseline and
E22 local candidates. Assess missing output and long occlusions, not just counts.
No default cap chosen or promoted. Artifacts: `G/confidence-cap/`.

21 replays complete (three parameter sets × six caps plus uncapped controls).
118 tests pass; uncapped scores exactly match all108 original recordings.
On retained baseline, cap10 improves pooled close position1.505%, velocity0.122%,
fast velocity unchanged, false-output time -0.986s, wrong-output time -0.500s;
worst per-group close p/v ratio1.001239. Cap5 cuts false-output3.740s but a group
regresses17.4%. Cap3 increases pooled close-position RMSE33.3% and close correct
track missing time14.122s. Thus confidence saturation can bound stale history,
but low caps damage retention/selection and do not solve fast velocity.
Preserve experiment branch `experiment/ball-filter-confidence-cap-20261005`;
restore original runtime/defaults. Revisit only with a candidate that passes
close-position/velocity guards; do not add unused production complexity.

E22 completed1,040 configurations; none beats baseline under all eleven p/v/fast
group guards. Several improve all close p/v groups but regress fast velocity in
one or two. Full results retained for targeted follow-up, not fresh evidence.

### E24 — Existing speed threshold and obsolete velocity

**Status:** running. E21 regressions include opposite-direction residual motion
where baseline reports rest. Vary the existing resting-speed threshold0.03–0.5m/s
and resting noise ×0.7/1/1.3 on original and three E22 centres (88 configs).
No covariance-transition algorithm or additional tracker. Same eleven-group
close p/v/fast and availability guards. Defaults remain unchanged; original
frozen E22 evaluator used, not confidence-cap experimental binary.
Artifacts: `G/resting-calibration/`.

E24 complete:88 configurations, no passing improvement. Preserved results and
unchanged defaults. No additional covariance-transition mechanism introduced.

### E25 — Two-observation initialization for clearly fast motion

**Status:** experimental. Keep three-observation confirmation except for an
associated pair20–100ms apart, displacement significance>36 and conservative
three-sigma speed lower bound>2m/s. Preserve position/velocity covariance cross
terms. Hypothesis: reduce fast-shot onset delay without promoting noisy or slow
motion. Unlike E07, do not loosen all three-point consistency gates. Risks:
isolated outliers or genuine sudden reversals; require existing tests and all108
recordings before fresh validation. Artifact: `G/fast-start/`. Original source
saved; no defaults or new user parameters added.

E25 rejected at unit-test gate:118 pass, existing
`isolated_false_observation_does_not_confirm_motion` fails. An isolated outlier
can satisfy both the six-sigma displacement and conservative speed criteria;
two observations cannot distinguish it from a shot. Preserve experiment source,
restore original confirmation. Do not weaken the test or claim replay benefit.

### E26 — Revisit existing output correction with expanded evidence

**Status:** running. E16 explored this on54 clips and found its extra changes
unnecessary; now108 inspected clips expose new fast-velocity regressions in the
E22 candidates. Test only existing auxiliary detection noise, output blend and
covariance ratio around original and three E22 centres. No additional tracker,
new blending rule or runtime change. Same eleven-group guards and missing-time
checks. This is an explicit revisit of E16, not a new mechanism. Artifacts:
`G/publication-recalibration/`. Fresh validation still required.

E26 complete:244 configurations, no passing improvement. The output correction
changes do not fix the remaining per-group fast-velocity regressions. Preserve
results; do not add changes to publication defaults.

### E27 — Independent-axis measurement and process noise

**Status:** running. E20/E22 tied forward/lateral measurement and motion noise.
Allow independent axes around three E22 candidates; camera projection and robot
viewing direction can produce unequal errors. 1,024 seeded samples plus baseline,
resting noise and publication settings unchanged. All existing parameters; no
new mechanism. Same eleven-group p/v/fast and availability guards, frozen original
evaluator,108 development clips. Artifacts: `G/axis-calibration/`.

E27 initial1,025 configurations complete, no full pass. Best worst-group
p/v/fast regression about0.15%. Run1,024 local refinements around random0227,
0345,0455; add fine independent resting-noise and threshold variation, rather
than E24's coarse steps. Seed2026100534, artifacts include exact centres/ranges.

Next independent audit is prespecified as the previously captured, **unscored**
reserved suites audit3 seed51 and audit4 seed55. They were not used for selection,
diagnosis or tuning. Freeze candidate before evaluating them, verify baseline
live/replay, apply unchanged fresh acceptance rules. Seed56 remains unused.
Protocol prepared at `G/audit5/protocol.json`; candidate not yet selected.

E27 local refinements complete:2,049 total configs, still no nonbaseline full
pass. Continue with an explicit minimax search: up to8 generations ×64 samples
around the best four worst-group p/v/fast ratios, shrinking noise perturbations,
seed2026100535. Preserve missing/bounded-loss constraints; allow small existing
publication blend variation. Every generation records its centres and settings.
Stop early only after all group ratios<0.999, then freeze for reserved validation.
No new data enters optimization.

E27 minimax completes8×64 samples; best evo-7-015 still regresses worst group
0.0158%, no full pass. Runtime/defaults unchanged, reserved validation untouched.

### E28 — Association refinement around the minimax candidate

**Status:** running. Vary existing association maximum cost, uncertainty penalty,
and quiet far-ball reacquisition range around E27 evo-7-015. These dimensions
were fixed during E27; aim to remove the remaining small mismatched-output
regression without reducing availability. 182 configs including controls,
same108 clips and eleven-group p/v/fast checks. No runtime change. Artifacts:
`G/association-refinement/`. No held-out candidate evaluation yet.

### E29 — Initialize current velocity rather than interval-average velocity

**Status:** experimental, parallel to frozen-binary E28 parameter evaluation.
Three-point confirmation and its significance/consistency checks remain intact.
Current endpoint slope is an interval-average velocity, while prediction expects
velocity at the newest observation. For existing exponential damping, multiply
velocity by x/expm1(x), x=lambda×span, and transform covariance cross/velocity
blocks consistently. No new parameters. Unit-test exact damped trajectory and
run existing false-observation safeguards, then compare on108 clips against the
original runtime with identical parameter files. Artifacts:
`G/damped-initialization/`. This is not E13's constant-velocity weighted fit.

E28 complete:182 configurations, no full pass; association variations do not
remove the tiny development velocity regression. No changes retained.

E29 complete:118 tests pass. Original parameters improve pooled position0.27%,
velocity0.13%, fast velocity0.008%, but worst group regresses0.029%. With E27
centre, worst group regresses0.318% versus0.0158% without the change. Preserve
source and reject as unnecessary mixed-benefit complexity; restore original
runtime. Evidence includes paired fixed-parameter controls.

### E30 — Reserved independent audit of minimax parameter candidate

**Status:** frozen before evaluation. Select E27 evo-7-015 by smallest maximum
of all11 development-group p/v/fast ratios after the prespecified8 generations.
Its worst ratio is1.00015809, so it **does not pass** E22's extra strict all-group
screen. All group positions improve; pooled p/v/fast improve. The0.0158%
velocity tradeoff is recorded, not rounded to a pass. Use the original fresh
acceptance rules (E17/E19/E21), unchanged, to assess generalization on reserved
seed51/55. This explicitly changes development candidate-selection policy from
requiring an exact all-group pass to minimax selection; it does not change any
fresh success criterion or claim success on development. No held-out scores have
been inspected before this decision. No runtime algorithm changes. Exact config,
hashes,selection,protocol: `G/audit5/`. Defaults remain original pending outcome.

E30 rejected. Both baseline replays verified. Fresh18: close position
0.158841→0.158387m (-0.286%), velocity0.749706→0.709934m/s (-5.305%),
fast velocity2.673446→2.578348m/s (-3.557%). Wrong-output53.332→51.748s,
false79.6s unchanged, close velocity missing1.424s unchanged. Suite2 position
0.148938→0.154374m (+3.65%,5.44mm) fails the unchanged frozen criterion.
Stationary family0.094206→0.105496m; restoring baseline output blend and resting
measurement settings is the next ablation. Seed51/55 are now inspected development;
seed56 remains untouched. Decision and exact frozen candidate at `G/audit5/`.

### E31 — Simplify the candidate to preserve stationary behavior

**Status:** running. Add the18 rejected validation clips to development (126 total,
13 groups). Factorially restore baseline detection noise, resting noise,
publication blend, resting-speed threshold in the E30 candidate (16 ablations
plus baseline/candidate controls). The original moving/damping/matching changes
remain to test whether extra changes caused stationary regressions. No source
algorithm change. Original runtime rebuilt with enlarged batch evaluator;
require original score parity before interpreting. Artifacts:
`G/simplification-calibration/`. No fresh scores or default edits yet.

E31 initial18 and motion-component32 ablations complete; no nonbaseline full
pass. In seed55 stationary at30.304s, candidate selects recent resting hypothesis
(0.587,0.725), error0.360m, while baseline retains resting estimate error0.028m.
Observer selected hypothesis has size_plausible=false, age0.078s, covariance
trace1.1646, confidence6.177; published output equals this uncorrected primary.
This is a wrong fresh selection/correction failure, not velocity drift. Observer
8768 and `G/audit5/diagnosis/candidate-frame30.json` preserve evidence.

### E32 — Loosen the existing correction covariance gate

**Status:** running. E26 only tried ratios through12. Test12,24,48,96,1000,1e6
with blends baseline/0.98/1 on E30 and simplified parameters (36 configs) to
identify whether useful auxiliary history is rejected relative to a newly
observed false primary. It is a diagnostic hypothesis, not yet proof: inspect
paired output at the failing frame, and reject broader close-range regressions.
No new runtime mechanism. Artifacts: `G/simplification-calibration/correction-gate/`.

E31 baseline parity: first11 groups exact, new2 grouped RMSE within1e-10 of
verified replay. Original runtime117 tests pass. E30 proximity diagnostics also
complete: close multiple-near time22.000→20.806s, zero-near15.422→15.048s,
mean total primary count6.199→6.459; geometric counts exclude auxiliary tracker.

E32 confirms the correction-gate hypothesis: at the failing stationary frame,
ratio1000/blend0.98 changes error0.35970→0.03408m; published last-seen returns
to retained history at27.322s. All13 development groups close p/v improve.
Pooled position -19.32%, velocity -5.89%, fast velocity -2.20%; wrong-output
-9.022s, false-output +8.184s, close correct-track missing -6.466s, close velocity
missing unchanged. One group fast velocity +0.064% remains; do not call the
stricter all-fast development screen a pass. Ratio1e6 yields virtually identical
loss and slightly more wrong time; freeze1000 at the observed plateau.

### E33 — Sixth frozen independent audit

**Status:** frozen/capturing. Exact candidate `G/audit6/candidate.json5` and hash
in freeze.json; original retained runtime, parameters only. First suite is reserved
unscored seed56, second first valid new seed57–64 in fixed order. Capture two
concurrently in isolated namespaces, preserve unused successful extras. No scoring
before freeze and no accuracy-based replacements. Fresh rules unchanged: pooled
close p/v improve, each suite close p/v non-regressing, pooled fast velocity and
availability non-regressing, bounded close losses improve; inspect scenarios,
events and wrong/duplicate hypotheses. Defaults remain unchanged until success.

E33 numerical checks all pass on reserved56/new57 (baseline replay verified).
Seed58 captured successfully but remains unscored/uninspected. Fresh18: position
0.138001→0.109498m (-20.65%), velocity0.659313→0.655969m/s (-0.507%),
fast velocity2.442831→2.418012m/s (-1.016%). Both suites improve close p/v.
Wrong-output91.284→86.434s, false78.32→79.48s, close correct-track missing
6.504→5.514s; missing velocity2.338s unchanged. All numeric checks pass.

Manual review does **not** establish the full tracking goal yet. Seed57 contested
close velocity0.933644→1.056514m/s (+13.16%), fast2.735359→3.049205m/s
(+11.47%). At28.4s candidate continues old opposite motion while baseline has
rested; at28.6s baseline has initialized the new velocity while candidate lags.
Close multiple-near primary time35.574→43.846s, zero-near8.052→7.148s;
mean total primary7.054→7.501. Counts are geometric, not identities. First200ms
impact velocity3.37370→3.37517m/s (essentially unchanged). Exploratory200–1000ms
recovery improves2.56377→2.32736m/s (-9.22%), with no missing output. Preserve
as a numerical benchmark, not a completed goal or changed default. Detailed
paired diagnostics/proximity/decision are in `G/audit6/`.

### E34 — Velocity response after reversal with the corrected output gate

**Status:** running. E33 moving-position noise is about2.6× original; it may let
new positions update without correcting velocity promptly. The old stationary
regression is now addressed by the correction gate, so retest lower moving
position noise against velocity noise/damping while preserving that gate.
182 configurations on144 inspected clips/15 groups, plus explicit review of
seed57 contested reversal and duplicates before another freeze. Seed58 remains
unused. All changes are existing parameters; runtime unchanged. Artifacts:
`G/velocity-response/`. Original baseline remains the reference, not E33.

E34 baseline parity verified: first13 groups exact, new2 verified pooled moments
within1e-10; separate seed57 contested diagnostic exactly matches its baseline
per-recording score and is not double-counted. Coarse182 complete. Grid0149
improves pooled position33.4%, velocity7.14%, fast3.97%, and the targeted reversal
close velocity14.4%, fast24.7%, but worst group regresses0.368%. Refine768
configurations around0149/0073, seed2026100536, with small independent noise,
damping, threshold, matching-distance and blend perturbations. Require reversal
close/fast non-regression explicitly, not just pooled groups. No fresh data used.
Paired reversal plot inspected: `G/audit6/diagnosis/reversal.png`; it shows old
candidate's wrong-direction velocity at28.45s and subsequent lag despite lower
late position error. No assertion that first200ms recovery has been solved.

E34 complete:950 configurations (182 controls/grid plus768 refinements). Only
local0381 beats original baseline while passing every15-group close p/v guard,
pooled fast/bounded-loss/availability checks **and** the separately scored
reversal close+fast velocity guard. Pooled p -28.86%, v -7.18%, fast -4.10%;
worst close p/v ratio0.99989. Target reversal close velocity0.933644→0.858168,
fast2.735359→2.314772, position0.037595→0.031689. Wrong-output -7.166s,
false-output +9.122s, close correct-track missing -6.450s; velocity missing
unchanged. This is development evidence only; no runtime changes or default edits.

### E35 — Seventh frozen independent audit

**Status:** frozen/capturing. `G/audit7/candidate.json5`, exact hash in freeze.json.
Reserved unscored seed58 plus first valid new seed59–66, fixed order, two parallel
isolated captures; extras preserved unused. Original fresh numeric rules and
manual review unchanged. Explicitly inspect reversal/recovery and duplicate
hypotheses, since the preceding numerical pass did not establish the full goal.
No scores from these recordings informed selection. All runtime code remains
original retained extended filter; only existing parameters differ.

E35 rejected. Seed59 failed fast-crossing reacquisition coverage0<1 after two
complete scenarios; next ordered seed60 valid. Strict baseline verification passed
on reserved58/new60. Fresh18 pooled p0.194773→0.105326m (-45.92%),
v0.629042→0.618628m/s (-1.655%), fast2.213166→2.144758m/s (-3.091%).
Suite1 velocity0.660066→0.663052m/s (+0.452%) fails unchanged criterion;
both suites position improve. Wrong-output102.628→96.104s, false79.24s unchanged,
close correct missing9.076→5.504s, velocity missing2.36s unchanged.

Recovery gain is now concrete: first200ms velocity3.385829→3.201773m/s (-5.44%),
position0.239321→0.236870m, no missing. Contested family close velocity
1.125471→0.797944m/s (-29.10%), fast2.416058→1.643323m/s (-31.98%).
But close multiple-near hypotheses23.296→41.224s, zero-near10.990→9.278s,
mean total primary6.304→6.567. Preserve results, do not hide duplicate regression
behind position gains. Defaults unchanged. All successful unused captures have
now been inspected; next independent validation requires newly captured seeds.

### E36 — Existing merge distance with improved velocity response

**Status:** running. 191 configurations: merge distance0.05/0.1/0.15/0.2/0.25/
0.3/0.4m crossed with small moving-position/velocity noise and damping factors,
plus baseline/E35 controls. Keep all existing same-image, velocity-consistency,
confirmation and observation-history merge safeguards. No runtime change or new
mechanism. Dataset162 inspected clips,17groups; separate seed57 reversal
score is not counted twice. Assess near duplicates and availability explicitly
before another freeze. Artifacts: `G/merge-response/`.

E36 coarse191 complete: no nonbaseline full pass; best worst-group close p/v
regression0.35%. Expand with144 controlled probes of initial position covariance
and hidden-confidence decay, alongside merge/damping. This explicitly revisits
E08/E10 under the now-improved velocity settings/correction gate and162 clips;
not a new mechanism. Existing hidden decay can shorten weak clutter retention
while stronger observed support survives. Availability must not regress. Exact
ranges and centre in `G/merge-response/retention-protocol.json`.

E36 retention probes144 complete: no full pass; stronger hidden decay sometimes
reduces wrong-output but adds missing velocity (up to2.548s in leading rows), so
those candidates remain excluded. Baseline162-clip parity verified: first15
groups exact, new2 moments within1e-10.

A controlled merge-distance comparison does demonstrate a substantial duplicate
benefit. On the18 recent recordings, grid0041 (merge0.1) vs grid0149 (merge0.3),
otherwise identical parameters: close multiple-near time41.230→4.802s,
zero-near9.910s unchanged, one-near163.748→200.176s; total hypothesis integral
1411.902→1262.962 over214.888s. These are primary hypotheses within0.30m of
truth, not persistent identity; exact parameter/score parity checked. Evidence:
`G/merge-response/merge-counts.json`. No missing-velocity increase in these configs.

Continue with up to8×64 minimax refinements, seed2026100537, fixed merge0.3.
Select centres only with no added missing velocity, improved bounded losses,
non-regressing pooled fast velocity and targeted reversal close/fast velocity.
Minimize worst17-group close p/v ratio; generation centres/ranges saved. No
fresh data or new code mechanism. This targets the residual0.35% velocity
regression while retaining the demonstrated duplicate reduction.

E36 minimax stopped early after2×64 samples as prespecified: evo-1-056 passes
all17 group close p/v checks with worst ratio0.998818, plus pooled fast/bounded
loss/availability and targeted reversal checks. Total463 evaluated configurations.
Pooled position -28.48%, velocity -6.33%, fast -3.43%; wrong-output -14.756s,
false-output +9.326s, close correct missing -11.096s, velocity missing unchanged.
It keeps baseline hidden decay, uses initial xy covariance1, and merge0.3.

Exact-score diagnostic replay on the recent18 clips confirms duplicate benefit:
original baseline close multiple-near23.296s → candidate5.202s (-77.67%),
zero-near10.990→9.796s, one-near180.602→199.890s. Total primary hypothesis
integral1354.646→1272.238 over214.888s. No missing-velocity increase. This is
still development evidence, with geometric proximity rather than track identity.
Artifacts: `G/merge-response/final-development-counts.json`.

### E37 — Eighth frozen independent audit

**Status:** frozen/capturing. `G/audit8/candidate.json5`, exact hash in freeze.json.
All-new seeds61–72 in fixed order, four concurrent isolated captures, first two
complete valid suites selected before candidate scoring; extras unused. Frozen
capture/evaluator hashes verified. Fresh numeric criteria and manual recovery/
duplicate assessment unchanged. Runtime/defaults remain original pending outcome;
no confidence cap or other experimental algorithm change is retained.

E37 rejected. All four captures valid; first61/62 scored,63/64 remain unscored.
Baseline replay verified. Pooled position0.237569→0.288403m (+21.40%) fails;
velocity0.671039→0.648825m/s (-3.31%), fast2.464599→2.361903m/s (-4.17%)
improve in both suites. Sideline family position0.412873→0.815225m dominates
regression; other position families largely stable or better. Missing velocity
1.398s unchanged, false77.64s unchanged, wrong104.426→106.560s. First200ms
velocity3.714401→3.645527m/s (-1.85%).

Duplicate reduction generalized: close multiple-near35.398→1.814s (-94.88%),
zero-near10.930→9.730s, one-near154.664→189.448s, mean total primary
5.657→4.990. This is geometric proximity, not permanent identity. Preserve the
benefit but do not adopt while close-position accuracy regresses.

Sideline diagnosis: seed62 at36.228s, truth Field(5.211,-3.176) is within0.88m
of robot, slightly outside the9×6m field. Candidate reports a resting track last
seen9.838s ago, position error3.981m; baseline's2.238s-old track errs1.510m.
Possible field-prior suppression of the genuine nearby ball must be isolated;
this is not a new velocity error (both outputs at rest). Evidence:
`G/audit8/diagnosis/sideline-regressions.json`.

### E38 — Field-boundary confidence near a real off-field ball

**Status:** running. 202 configurations (200 grid +2 controls), existing field
margin, confidence distance, stored decay rate and publication margin only.
Keep localization assertion true and all velocity/merge settings frozen at E37.
180 inspected clips/19groups, original reference unchanged; no scoring, pose,
physics or runtime changes. Require close p/v, pooled fast/bounded/availability
and targeted reversal guards; inspect sideline and duplicates. Artifacts:
`G/boundary-calibration/`. Reserved63/64 remain untouched until next freeze.

E38 complete:202 configurations, no full pass. Doubling boundary confidence
distance0.3→0.6 reduces the new suite's position regression from41.33% to3.07%,
with other close p/v groups passing. Do not promote; field tuning alone did not
close the failure.

Observer shows why recency ranking alone cannot recover: selected9.854s-old
primary has confidence73.045; newer1.454s/2.254s tracks have2.626/1.876, below
the existing confirmation threshold3. The established-track rule excludes them
before the zero-capped ranking ties are broken by recency. Selected size flag is
false already, so adding a new radius-gate bypass would not target this case.
Evidence: `G/audit8/diagnosis/candidate-frame36.json`, observer8769.

### E39 — Stored-confidence cap near the confirmation threshold

**Status:** experimental revisit of E23, motivated by the user's cap suggestion
and this observed failure. E23 tested3/5/10/etc on earlier parameters; cap3 loses
confirmed status after any decay, explaining why that setting was especially
harmful. Test3.25/3.5/3.75/4/4.5/5/6/8/10 plus None, crossed with boundary
confidence distances0.3/0.45/0.6/0.75 (41 configs including original baseline).
At baseline hidden decay, cap4 falls below confirmation3 after8.3s, while a
recently observed track remains confirmed. Keep the existing confirmation rule.

Reuse exactly E23's optional stored-confidence ceiling, no additional mechanism.
Require uncapped replay parity and existing tests before comparison. Current
working runtime is experimental with defaultNone; production parameters remain
unchanged. Reserve63/64 stays unscored. Artifacts: `G/confidence-revisit/`.

E39 initial41 complete:118 tests pass; uncapped baseline has exact score parity.
Cap4 alone lowers audit8b position ratio1.41332→1.11841; combining boundary
distance0.75 lowers it to1.002814, with all other close p/v groups passing and
no extra missing velocity. Target reversal unchanged. Still no accepted winner.
Refine48 combinations around cap3.85–4.25 and distance0.65–1.5 on development
only; reserved63/64 remain untouched.

E39 refine complete89 total: refine-019 (cap4, boundary distance0.8) passes all19
development close p/v groups; pooled position−26.33%, velocity−6.04%, fast−3.51%;
wrong−14.488s, false+12.184s, correct-missing−11.614s, velocity-missing unchanged.
Reversal improvement preserved. Freeze exact candidate/runtime in `G/audit9/`
before scoring previously reserved63/64. Fresh criteria unchanged; review false
time and duplicates as well as numerical guards. Default parameters unchanged.

### E40 — Audit9 confidence-cap candidate

Fresh reserved63/64 baseline replay verified; all numerical guards pass.
Close position0.232324→0.198630m (−14.50%); velocity0.650333→0.639321m/s
(−1.69%); fast velocity2.265821→2.175009m/s (−4.01%). Both suites close
position/velocity nonregress; missing velocity2.21s unchanged. Correct-track
missing9.924→5.274s, but wrong68.704→70.052s and false78.422→79.800s.
Manual review outstanding: approach position0.277227→0.477615m despite
less wrong time; contested wrong18.942→26.930s, initial recovery and primary
duplicate counts. Do not adopt or mark goal complete yet. These18 recordings
are now inspected; they cannot be fresh validation for a retuned candidate.
Artifacts `G/audit9/comparison.json`, report/frame files, freeze and hashes.

E40 manual review: primary close duplicates19.190→7.606s (−60.4%); zero-near
7.756→8.776s. Initial200ms velocity3.068974→3.024124m/s (−1.46%) with
no missing output. However seed64 approach has0.52s additional wrong output,
including3.61m error at true range0.196m. Reject adoption despite numerical pass.
At51.280s: nearby primary support2.811, newer distant clutter1.0; no confirmed
track>=3, so ranking cap0 selects newest clutter. State in
`G/audit9/diagnosis/approach-state`; exact diagnostic replay preserves runtime.

### E41 — Positive selection cap with bounded stored support

Hypothesis: cap0 discards useful relative support when all hypotheses fall below
confirmation. Test existing ranking caps0/0.5/1/2/3/4/8/1000, stored ceilings
None/4/5/8, uncertainty0/0.01/0.1 (97 configs including original baseline).
All other E39 parameters fixed,198 inspected clips and21 groups. No new runtime
mechanism beyond E39 optional cap. Preserve close/fast/availability guards and
inspect approach/contested before freezing new candidate. Artifacts
`G/confidence-selection/`. Need new seeds65+ for fresh validation.

E41 complete97 configs: positive ranking cap3, uncertainty0.1 reduces approach
position0.622→0.160m, but regresses audit4b position36.2%; smaller passing
ranking settings retain the outlier. No adoption. These caps alter confirmed
competition as well as the failing unconfirmed case.

### E42 — Support-first ties only without a confirmed incumbent

Test a small selector change: on equal ranking score, if no confirmed incumbent,
compare effective support before recency. Confirmed-track ties retain recency
first, preserving E01. No new parameter. Regression test mirrors observed
2.8-support nearby track versus newer1.0 clutter in both vector orders.
Three configs: original parameters, E39 cap4 candidate, uncapped candidate.
Compare against original E41 baseline results, not the modified-runtime baseline.
198 inspected recordings; review approach and all existing guards.
Artifacts `G/unconfirmed-selection/`. Default config still unchanged.

E42 rejected:119 tests passed and approach RMSE0.622→0.162m, but cap4
full replay regresses original position4.816x and several other groups.
Uncapped variant also misses audit8b. Preserve source/patch under
`G/unconfirmed-selection/`; restore original selector byte-for-byte.

### E43 — Boundary decay with bounded confidence

Cap4 approach confidence drops below3 after about1s, much sooner than8.3s
from hidden decay alone. Boundary stored decay penalizes actual off-field
ball tracks too. Test stored caps3.5/4/4.5/5/6 crossed with boundary decay
0/0.05/0.1/0.2/0.25/0.35/0.5 and confidence distance0.6/0.8/1.2.
106 configurations including original baseline,198 inspected recordings.
Original selector restored; only optional stored cap is experimental runtime.
Require uncapped original score parity, existing guards and targeted approach
review; fresh validation needed. Artifacts `G/confidence-decay/`.

E43 initial106 complete: original baseline score parity exact,118 tests pass.
No full pass; best cap4/rate0/distance0.6 misses only audit8b position by0.565%.
The prior E39 stored field-decay rate was2.0 (checked exact candidate file).
Refine80 configs around cap3.75–4.15, rate0/0.25/0.5/1, distance0.55–0.85.
No fresh data scored.

E43 refinement186 total: refine-040 cap3.95, boundary decay0.5 and confidence
distance0.55 passes all21 close p/v groups, pooled fast/bounded/availability and
reversal guards. Pooled position−28.25%, velocity−5.66%, fast−3.55%; wrong
−13.576s, false+13.602s, close correct-missing−16.704s, velocity missing unchanged.
Prior audit9 approach position0.358202→0.163544m; catastrophic clutter selection
removed. Remaining additional close wrong time0.04s at error0.5007–0.5074m
after1.5s gap. Contested extra wrong4.986/3.002s occurs at true ranges>=3.288/
1.796m respectively, no extra close wrong time; document long-range extrapolation
tradeoff. These18 clips are development, not new validation.

### E44 — Audit10 frozen cap/decay candidate

Freeze E43 refine-040 and evaluator hashes under `G/audit10/` before new captures.
Seeds65–76, first two complete nine-scenario suites by ordered coverage success;
four isolated captures at a time, extras preserved unscored. Same acceptance
criteria as preceding audits, including manual review. Runtime only adds optional
stored-confidence cap; production parameter defaults unchanged until acceptance.

E44 audit10 rejected: all4 captures succeeded, first65/66 scored;67/68 reserved
unscored. Baseline replay verified. Close position0.127733→0.135965m (+6.45%),
velocity0.677975→0.668177m/s (−1.45%), fast2.414109→2.345040 (−2.86%).
Suite65 close p/v regress. Wrong68.150→70.720s, false79.12 unchanged, close
correct-missing5.294→5.334s, velocity-missing2.526 unchanged.
Close primary duplicates45.994→5.284s (−88.5%), zero-near6.218→6.114s.
Initial200ms velocity3.980079→3.563681m/s (−10.46%), position0.650101→0.767090;
200–1000ms velocity2.842105→2.359496, no missing output in either window.
Tests118 filter+43 tuner passed;1 existing tuner test ignored. Receipts under
`G/audit10/checks/`. Goal incomplete and defaults unchanged.

Approach65 diagnosis: at43.424s candidate primary recent resting support3.95
outputs velocity0 while baseline retains older size-inconsistent primary and
uses auxiliary velocity3.64m/s, close to truth3.76m/s. At48.320s candidate
selects2.158s-old resting track,1.10m error; true recent track has support1.0,
while baseline matches the true track with45.8 support. Difference is in
association/merging and publication eligibility, not only Kalman velocity gains.
Source states and paired metrics: `G/audit10/diagnosis/`.

### E45 — Merge and reacquisition calibration on the new approach failure

Test existing merge distances0.1/0.15/0.2/0.25/0.275/0.3, matching1/1.2/1.4/1.6/2,
reacquisition0.1365(original)/0.25/0.5:90 plus baseline and prior candidate.
216 inspected clips,23 groups, plus separate seed65 approach diagnostic.
Retain original runtime with optional cap; no new algorithm. Require existing
close/fast/availability guards and inspect approach/reversal/duplicates before
new frozen validation. Reserved67/68 untouched. `G/merge-reacquisition/`.

E45 initial92 configs: no full pass, best worst-group close position remains
+9.29% on audit10a. Expand90 configurations to reacquisition1/1.5/3m because
observed residual1.10m is outside the original and first grid limits.

Additional diagnosis: candidate last observed true ball at46.162s while moving
(~0.76m/s) and within0.28m. At48.32s it has transitioned to Resting by prediction
and lies0.612m away; quiet-ball reacquisition protection may now apply. Its
current velocity is zero, so `protect_prior` cannot reconstruct pre-gap motion
by undoing damping. A future targeted fix could preserve last-observed speed
across prediction-only resting transitions; first test wider existing gates.
Do not confuse deterministic slowing with observed evidence that the ball rested.

E45 complete182 configs: no full pass. Wider gates1–3m worsen the targeted
approach position, so do not loosen them globally.

### E46 — Retain speed at the last observation across predicted rest

Hypothesis: `protect_prior` reconstructs observed speed by undoing damping, but
cannot do so once prediction converts the model to Resting (velocity becomes0).
Preserve optional speed-at-last-observation metadata on hypothesis updates and
coherent merges; use it for quiet-ball protection, with old inverse-damping
fallback for missing metadata. New observations overwrite the metadata; pure
prediction leaves it unchanged. No new tuning parameter. Test transition to
predicted rest followed by an actual stationary observation, plus existing tests.
Compare original parameters and frozen E43 parameters against the unchanged
original E45 baseline216-clip results; do not redefine baseline with new runtime.
Sources before/after and artifacts under `G/observed-motion/`. Default parameters
unchanged, reserved67/68 remain unscored.

E46 rejected for current candidate:119 tests pass. Original parameters close
metrics essentially unchanged; cap candidate worst-group position1.14519 vs
1.09291 without metadata; approach position1.16996x baseline. Quiet-guard
eligibility alone does not fix overall matching. Preserve before/after sources
and decision under `G/observed-motion/`; restore all5 modified files exactly.

### E47 — Spawn support and predicted-rest transition with a stored cap

Existing nearby-spawn factor transfers up to1.5 confidence from parent to newborn.
For cap3.95 this can unconfirm the parent, unlike the unbounded baseline.
Test spawn factors0/0.25/0.5/0.75, resting-speed thresholds0/0.02/0.05/0.075/0.1,
and merge0.1/0.2/0.25/0.3 (80 plus controls;82 total). Later predicted rest also
retains moving Kalman state through observation gaps, unlike E46 metadata alone.
216 inspected clips; original selector/runtime plus optional stored cap only.
All23 group close p/v and existing fast/availability guards remain; inspect
approach and reversal. Reserved67/68 remain unscored.
Artifacts `G/motion-confirmation/`.

E47 complete82 configurations: no full pass; best worst-group position+9.09%,
approach+10.97%. Changing spawn transfer alone had little effect here; do not
treat that proposed cause as proven. Lower resting thresholds improve pooled
velocity but introduce other position regressions.

### E48 — Joint minimax parameter calibration with all inspected failures

The single-family scans did not generalize. Jointly calibrate existing noise,
damping, matching/merge, rest/spawn, boundary and publication parameters; include
both capped and uncapped original anchors rather than assuming cap is required.
Fixed seed2026100548, up to12 generations of128;216 inspected recordings.
Rank by worst close p/v ratio across23 groups and target approach, with pooled
fast/bounded/availability and reversal guards. Early stop only when fully
feasible and pooled position/velocity each improve>=1%; then manual review and
fresh validation on untouched67/68. No algorithm additions beyond optional cap.
Artifacts `G/joint-calibration/`. Default parameters remain unchanged.
Result: all12 generations finished (1540 non-baseline reports including seed
configurations); zero full passes. Best minimax evo-6-102 still regresses one
suite velocity0.2713%, despite pooled position -0.7494%, velocity -0.1906%.
Rejected; do not restart this completed job.

### E49 — User-requested simulator checkout and push

The user requested the best current candidate and then explicitly requested a
push so they can pull it. Apply exactly E43/refine-040 (`G/audit10/candidate.json5`,
SHA256 e4f2c2a63ced11611e37dca14fa85f449cf5f497944cc8b1543a997f0832e4a6)
to the simulator branch defaults, with the tested optional stored-confidence cap.
This is an experimental local-testing checkout, not adoption: audit10 close
position regressed6.45%, while close velocity improved1.45%; details in E44.
The accepted comparison baseline remains `ed1d39b67`, and goal mode continues.
E48 uses frozen binaries/configurations and is unaffected by this checkout change.
Push only this branch to `schluis`; no other branches or PR are requested.
Verification: the checked-out defaults reproduce the frozen candidate's parsed
parameters and seed65 approach score exactly. All 118 ball-filter tests and 41
tuner library tests pass, along with tuner binary targets. Unrelated confidence
accounting fixtures explicitly disable the cap; the dedicated test covers the cap
and literal zero. The legacy warm-start test's expected configured boundary decay
is updated to 0.5. Local diagnostic examples are excluded from this commit.

### E50 — Exhaustive baseline/candidate parameter-block crossover

Prepared while E48 finishes: 256 combinations of eight parameter blocks from
the original baseline and E43. Blocks are confidence cap, field decay, merging,
rest threshold, matching distance, noise/covariance, damping, and publication.
This separates the useful changes from interactions that caused the E44 failures;
random mutation in E48 currently gravitates toward near-baseline candidates with
small remaining regressions. Use the same 216 development clips, unchanged E48
guards including the approach and reversal diagnostics, and frozen cap-only
runtime. Reserved seeds67/68 remain untouched. Evidence and exact block mapping:
`G/block-ablation/protocol.json`. No additional runtime complexity is introduced.

Result: all256 finished; none passes the full guards. Publication-only block128
is closest: pooled close position -8.89%, velocity -0.079%, wrong output -67.636s,
correct-ball missing time -14.092s, no additional missing velocity. All23 suite
position scores improve/nonregress, but some suite velocity scores regress up to
0.200%, and pooled fast velocity regresses0.0125%. Adding merge changes gives no
velocity benefit; cap/field changes introduce other regressions. Rejected as a
winner, but motivates calibrating publication parameters in isolation.

### E51 — Isolated publication-parameter calibration

Run98 combinations of existing publication blend, detection noise, and covariance
ratio, holding every primary parameter at the original baseline and leaving the
confidence cap disabled. This specifically tests E50's near-feasible simple
alternative without changing primary association or adding new filter code.
Same216 clips and full E48 guards; fresh67/68 remain reserved. Exact grid and
results: `G/publication-isolation/`. No acceptance based on pooled averages alone.
Result: all98 finished, zero full passes. Closest grid009 improves pooled
position9.13% and velocity0.188%, but worst guard remains +0.1245%. Rejected.

### E52 — Candidate persistence: visibility uncertainty versus confidence cap

User questions whether candidates decay and whether cap3.95 is too low. Code
audit: primary clear misses decay/expire; hidden decay half-life is about20s;
unknown visibility pauses decay. The auxiliary tracker disables unmatched decay;
both trackers retain the20s age timeout. Confidence is a support score, not a
probability, and3.95 is only slightly above confirmed-support threshold3.

Concrete E44 evidence: seed65 approach at48.320s selects a track last observed at
46.162s, support3.853, with no current validity-decay evidence and no accumulated
clear misses. Its3-sigma covariance bounds extend about20.18m by19.42m on either
side. Requiring all corners to be visible can protect increasingly uncertain
stale candidates by returning Unknown, which pauses decay. This is distinct
from cap-induced loss of confirmation.

Test56 configurations: baseline/E43 centres, visibility uncertainty scales
0/0.1/0.25/0.5/1/2/3 and confidence caps None/3.95/8/16. Other parameters remain
fixed within each centre. Use the216 development clips and unchanged full guards;
do not assume less conservative visibility is safe around real occlusions.
Evidence: `G/visibility-cap/protocol.json`; original state in audit10 candidate
frames-1.jsonl. No new runtime code or changes outside the filter.
Result: all56 finished. Grid020 (baseline, scale2, no cap) technically passes
numeric guards but improves position only0.00133% and velocity0.00000355%; this
does not resolve the observed problems and is not adopted. Smaller uncertainty
scales/higher caps alone do not produce an acceptable winner.

### E53 — Position/velocity continuity and rapid direction changes

User observes jumping candidates and very rapid velocity direction changes.
Code audit confirms MovingPredict integrates damped velocity into position and
applies odometry; measurement correction can change both immediately. E43's
velocity process-noise variances are6.37x/12.28x baseline, permitting much larger
velocity changes. This is a plausible mechanism, not yet identification of the
user's particular jump. Published output can also switch primary/auxiliary
histories; visual hypothesis indices are not persistent identities.

Diagnostic on audit10's18 recordings: transform outputs into Field, compare
displacement with trapezoidal integrated output velocity for adjacent<=120ms
samples, both true ranges<=1m. This includes corrections/switches, not just the
prediction model. Candidate discrepancy RMS0.02781m versus baseline0.03738m;
over10cm exposure2.666s versus2.690s; over50cm0.036s versus0.256s. No>90deg
estimated reversals while both estimated/true speeds>0.5m/s and truth direction
is steady in either run. Thus this close-range published-output diagnostic does
not yet reproduce the user's direction-jitter observation; do not claim it is
fixed or disprove a candidate-level problem. Exact protocol/events and script:
`G/continuity/`. Expand to individual-hypothesis corrections as needed.

### E54 — One-percept candidates controlling robot motion

User observed the robot walking toward an empty, wrongly selected location for
three seconds. An isolated birth has support1, yet output threshold is0.5.
The hard3-support confirmation rule only protects an established competitor;
it does not prevent a lone unconfirmed hypothesis from becoming published output.
Combined with Unknown visibility pausing decay, this can permit the reported
behavior. No exact user replay timestamp is available; distinguish the proven
code path from a confirmed attribution of that particular event.

At25Hz uninterrupted clear misses, isolated support1 crosses discard0.2 after
approximately3 near misses or17 farther misses (about80ms/640ms after first miss).
Inherited support, fresh matches, visibility changes and other removal paths
alter this. Unknown visibility can defer removal to20s without a new match.

Test existing output thresholds0.5/1.1/2/3, caps None/3.95/8, baseline/E43 centres:
24 configurations on216 clips. Quantify false/wrong output and acquisition/missing
time explicitly; a confirmation fix cannot be declared a tuning winner merely
because it suppresses outputs. Keep established occlusion handling separate from
tentative clutter. Evidence: `G/publication-confirmation/`.

Result: all24 finished; no full winner. On original baseline, output threshold3
reduces wrong-track time178.734s but adds17.904s close velocity missing time and
13.082s correct-close-ball missing time across216 clips; velocity RMSE also
regresses0.247%. On E43 with cap3.95, threshold3 adds104.988s missing velocity
versus original baseline, while reducing wrong time196.610s. This is not an
acceptable standalone fix; narrow cap headroom compounds confirmation loss.

Additional code-audit interaction: competing-track suppression requires leader
support10 (`competition::MINIMUM_LEADER_VALIDITY`). A cap3.95 keeps support below
that level under normal one/two-exposure update cadence, disabling that mechanism
in practice. Raising a cap alone still cannot prevent weak output eligibility or
Unknown visibility from pausing decay. Treat these as separate policies.

### E55 — Tentative candidates must earn uncertainty-based miss protection

Next experiment, not yet implemented: only established support earns the
covariance-corner protection in hypothesis visibility. For tentative support<3,
use the existing center visibility classification, which still respects actual
robot occlusion and camera geometry. This avoids treating large initial
uncertainty as grounds to ignore an empty, otherwise visible location. Keep
existing clear-miss rates/timeouts first to isolate this one change. Test a
single false birth disappearing versus an actual occluded track being retained;
then full216-clip baseline/candidate comparisons, including missing/fast recovery.
No automatic adoption and no further pushes. User's three-second false pursuit
is a concrete behavioral failure requiring targeted validation.

Implemented and tested:119 unit tests pass, including tentative broad-covariance
track removal on clear misses and retention behind a robot. Full216 replay:
with original parameters, wrong time drops220.664s and correct-close missing
time drops7.244s, but close velocity missing increases2.714s and audit3a position
regresses53.28%. With E43, wrong time drops222.226s versus original baseline,
but audit8b position regresses63.98% and missing velocity rises0.680s. Reject as
a standalone change. Preserve patch, before/after sources, tests, frozen binaries
and results in `G/tentative-visibility/`; restore prior runtime. No test weakened:
the old uncertainty-protection test now explicitly uses established support3;
new test exercises tentative support1 in clear and physically occluded cases.

### E56 — Reproduce and isolate excessive velocity direction changes

E53 extended beyond1m **does reproduce** the user's concern: across audit10's18
recordings, candidate has352>90degree output-velocity reversals during steady
true motion, versus baseline21. Eligible steady-motion exposure155.278s versus
174.824s; rates are therefore also substantially worse. All reversal samples
have recently observed primary tracks (age<120ms). Over50cm displacement/velocity
discrepancy exposure rises4.432s versus3.878s. Close-only metrics hid this failure.
These are discrete output transitions, not persistent-track identities; do not
attribute every reversal to the same hypothesis or to process noise without
ablation. Evidence `G/continuity/audit10-all-ranges.json` and its exact protocol.

Run E43 with only velocity process-noise variances restored to0.1/0.1 from
1.2277/0.6370. Frozen pre-E55 cap-only evaluator; reuse already-inspected audit10
recordings and baseline frames. Compare direction reversals, position continuity,
close/full-range velocity errors and missing outputs. Evidence/protocol:
`G/velocity-continuity/`. No fresh seeds or production defaults changed.

Result: restoring only velocity noise to0.1/0.1 cuts all-range reversal samples
from352 to14 (baseline21); eligible steady-motion exposure140.126s. Over50cm
discrepancy exposure drops to3.002s. Audit10 close position0.127120m versus
baseline0.127733m and E43 0.135965m; close velocity0.676847m/s versus baseline
0.677975 and E43 0.668177. Full-range velocity0.918606 remains worse than baseline
0.887787, but improves E43 0.989390. Plot inspected:
`G/velocity-continuity/velocity-noise-ablation.png`; it shows large oscillating
red velocities and much smoother low-noise estimates during a steady trajectory.

On all216 clips, five symmetric variance values0.05/0.1/0.2/0.3/0.5 fail the full
guards. At0.1, all suite close-position guards pass and pooled p/v improve31.25%/
3.24%, but some suite velocities regress and targeted reversal velocity worsens
26.54%. Lower noise alone is not a winner. Initial decimal filenames collided in
the batch helper's output naming; discard that ambiguous report, preserve it in
`invalid-name-collision`, and rerun all five under unique `qv-0p1`-style names.
The independent18-clip causal ablation was unaffected.

### E57 — Calibrate damping jointly with bounded velocity noise

E43's decay0.997069 per2ms gives velocity half-life about0.472s, compared with
baseline0.999444 (about2.494s). High velocity process noise can compensate for
over-strong damping while detections arrive, yet produces jitter and poor gap
prediction. Test that hypothesis rather than treating global high noise as a
solution to abrupt motion recovery. On E43's remaining parameters, cross
velocity noise0.1/0.2/0.3/0.5, seven damping values including baseline, and caps
None/3.95/8:84 configurations. Same216 clips, full existing guards, plus inspect
all-range velocity/stability before any fresh validation. Exact protocol and
results in `G/damping-noise/`. No runtime changes or production defaults changed.
Result: all84 finished, no full passes. Closest grid007 (noise0.1, original
damping, cap3.95) improves pooled close p13.39%, close v1.19%, full-range v1.95%,
but worst suite position still regresses5.69% and multiple velocity suites worsen.
The large E43 changes cannot simply be repaired by one noise/damping substitution.

### E58 — Conservative motion calibration with/without tentative-miss change

Use E51 grid009 as the centre: original stable primary parameters, no confidence
cap, and only publication blend/noise/covariance settings changed. Cross symmetric
velocity variances0.1/0.11/0.12/0.15/0.2, position-process-noise scales0.5/0.75/1/1.25,
and damping0.9993/original/0.9996:60 configurations. Evaluate the identical grid
under the retained runtime and frozen E55 runtime (120 evaluations total). This
tests whether the real clutter-removal benefit can be retained after modest
calibration, rather than adding further algorithms or adopting a jittery model.
Both compare to original baseline results and unchanged full guards; additionally
inspect full-range velocity and E56 stability before fresh67/68. Evidence under
`G/conservative-motion/` and `G/conservative-motion-tentative/`. No new sources,
production parameter changes, agents or pushes.
Result: both60-configuration grids completed, no full passes. Retained runtime's
closest grid007 is the E51 centre unchanged in primary parameters (worst guard
+0.1245%). Every E55-runtime configuration increases missing close velocity;
the best minimax entries still add at least2.634s. Keep the demonstrated unit
fix as a rejected/incomplete experiment, not a production change.

### E59 — Freshness of tentative publication, distinct from internal retention

Next, not implemented. Read-only consumer audit explains why retaining weak
published balls matters: `ball_state_composer::compose_ball_state` forwards the
filter's Some output, and behavior's `LastBall.age` is refreshed to now whenever
that ball exists. Its250ms timeout starts only once ball output becomes None;
that path does not itself expire a weak estimate based on last physical sighting.
Do not modify behavior or other components: the publication decision belongs in
the ball filter for this investigation.

Test suppressing a tentative primary's direct output after the existing120ms
recovery grace without deleting its internal hypothesis. Repeated real matches
keep it fresh while confirmation accumulates. Confirmed occluded tracks retain
their current policy. Keep existing auxiliary correction eligible under its
existing gates so this does not repeat E10/E15's deletion/auxiliary-availability
coupling; no new auxiliary-only fallback. Apply the freshness check to direct
primary output and primary fallback when auxiliary correction is unavailable.
Test single false detection then no observations, confirmed occlusion, fresh
reacquisition, and existing auxiliary availability tests. Quantify raw output
missing separately from correct-ball missing; removing a wrong output must not
be mistaken for losing a previously correct ball. Keep original guard results
visible and do not declare success from censored RMSE or a narrow unit test.

Result:119 tests pass after explicitly distinguishing localization correction
from a real new sighting for weak evidence. All216-clip comparisons fail full
guards. Original parameters reduce wrong time139.302s but add2.886s missing close
velocity and0.640s missing correctly tracked close ball; bounded close velocity
loss worsens6.87%. Reject age-only publication expiry and restore runtime/tests.
Artifacts/patch/binaries/snapshots: `G/tentative-publication/`.

### E60 — Clear-miss publication gate without deleting uncertain tentative history

Unlike E59's blind age gate, require accumulated clear detector exposure at the
tentative candidate's center. If its uncertainty corners prevent a full clear-miss
classification but its center is visible and not robot-occluded, accumulate the
existing negative-evidence clear-miss clock without deleting the uncertain
history. After120ms, suppress direct weak primary output until an actual matched
observation resets evidence. Confirmed tracks and physical occlusion keep their
existing policy. Do not add a new tracker or metadata field. Existing clear-miss
deletion/decay still applies where uncertainty already permits it; no independent
auxiliary fallback is introduced. Tests cover clear empty location, robot
occlusion, confirmation, internal state retention, and fresh reacquisition.
Then original/E43/publication parameter centres on216 clips with original guards.
Evidence: `G/tentative-clear-publication/`.
Result:119 tests pass. Original parameters reduce wrong time109.506s but still
add0.320s missing correctly tracked close ball and1.160s raw missing close
velocity; bounded close velocity loss worsens2.76%. Better than age-only gating,
but not accepted. All three parameter centres fail the unchanged full guards.
Patch, frozen binaries and before/after sources preserved.

### E61 — Calibrate clear exposure using the existing literal timeout

Replace E60's hard-coded120ms publication threshold with the existing
`visible_missed_detection_timeout`. It already expresses accumulated clear miss
exposure; use the same meaning for uncertain tentative output rather than adding
a second timing knob. Preserve internal uncertain history for positive timeouts.
Zero means removal on the first known clear point miss, while unknown/occluded
frames remain non-evidence. Test250/500/1000/2000ms on original/publication/E43
centres (12 configurations,216 clips); unchanged full guards and explicit wrong,
raw-missing and correctly-tracked-missing comparisons. Evidence:
`G/clear-publication-exposure/`. This is calibration, not acceptance of a weakened
availability criterion.
Result:120 tests pass, all12 configurations evaluated. Original parameters with
1000ms/2000ms reduce wrong output57.132s, leave aggregate correct-close missing
unchanged, and add0.080s raw missing velocity. At500ms wrong time drops81.698s,
correct-close missing is unchanged, and raw missing rises0.200s. The1s original
case nevertheless worsens old bounded position loss0.125% and velocity loss0.190%.
That demonstrates an objective conflict, not proof of a correctly tracked ball
being lost. Need per-frame comparison to exclude offsetting losses/gains hidden
by equal aggregate correct-close missing. No adoption yet; runtime restored,
120-test patch, sources and frozen executables preserved in the experiment root.

### E62 — Correctness-aware velocity and missing-output scoring audit

Code audit found a mismatch with the user's explicit false-pursuit concern.
The existing v11 position loss deliberately makes **every finite wrong estimate
cheaper than missing output** (`scoring.rs` comment and rational bounded loss;
missing penalty is1.25 times the asymptotic cap). Velocity is scored even when
the position is unrelated to the real ball. Consequently a stationary wrong
candidate can earn perfect velocity credit while directing the robot elsewhere.
Do not interpret this score preference as proof that ghost output is useful.

Prospective work, before further selection:

1. Preserve v11 raw RMSE, bounded losses and availability reports unchanged.
   Baseline remains the original ed1d39b67 implementation/parameters; no new model
   is silently substituted as the comparison reference.
2. Add supplementary joint position/velocity diagnostics. Use the already existing
   correct-track radius0.5m (not the0.3m duplicate-count radius). Velocity receives
   correct-ball credit only for a spatially associated estimate; wrong and absent
   estimates both lack a correctly tracked velocity. Keep truth-based denominators
   and report correctly associated RMSE plus missing coverage to prevent censoring.
3. Audit the optimizer's position penalty: clearly wrong output must not be
   preferred to absence merely because its finite error is bounded below the
   missing penalty. Prespecify and test any v12 objective before optimizing under
   it; retain side-by-side v11 results. Correct-position noise should retain a
   smooth loss, and empty-scene false output must still be penalized.
4. Re-score existing development data first and inspect **paired** E61 removed
   outputs: was previously correct position/velocity actually lost, or only a
   wrong candidate withheld? A0 aggregate coverage delta alone cannot prove this.
5. No completion claim from a changed score. Raw close-position accuracy must
   still improve/nonregress; true-ball velocity, fast/occlusion recovery, wrong
   and duplicate tracks, real-ball coverage, all-range stability, and untouched
   fresh67/68 validation remain required. This addresses a metric mismatch raised
   by the user's observed failure; it is not permission to hide regressions.

Implementation plan: start with a supplementary streaming diagnostic over compact
frame exports and counterexample tests (wrong position with numerically perfect
velocity must not receive correct-ball credit). Then decide/version the search
objective from that evidence. Current frozen E61 evaluators are available; frozen
original `M/tie-only/evaluate` can generate baseline compact frames. About52GiB
disk free at this checkpoint; avoid unnecessary full hypothesis-state exports.

#### E62.1 Completed paired audit — 216 development recordings

Evidence: `G/correctness-audit/{protocol.json,analyze.py,joint-comparison.json}`;
compact baseline/candidate frame exports and unchanged v11 reports accompany it.
Four counterexample tests passed, and streamed raw metrics reconcile with the
Rust reports within 1e-4. The candidate is E61 with original parameters and a
one-second timeout; its comparator is the original accepted baseline, not E43.

- Removed output: **57.132 seconds, all spatially incorrect** at the existing
  0.5 m correct-track radius; zero correctly tracked output lost or gained.
- All remaining published estimates unchanged; hypothesis counts unchanged.
- Correctly associated close-velocity RMSE **0.626021676 m/s**, identical for both;
  associated close-velocity missing time **103.618 seconds**, also unchanged.
- Therefore the improvement is suppression of wrong publication, not improved
  velocity or aggregate cancellation of correct-output losses and gains.

Decision: retain the mechanism as a promising experiment, not an accepted runtime
change. Production scoring and branch parameters remain unchanged. Next compare
publication-only E51 and its combination with E61 using paired correctness and
velocity diagnostics; freeze any revised objective before further optimization.
Fresh seeds 67/68 remain untouched. No goal-completion claim.

### E63 — Publication tuning with correctness-aware paired velocity [COMPLETE; NOT A VELOCITY WINNER]

Compare E51 grid-009 and the same publication parameters with E61's one-second
clear-miss gate against the original accepted baseline on all 216 development
recordings. Reuse E62 baseline exports. Freeze evaluator/parameter hashes in each
protocol before evaluation; fresh seeds 67/68 remain untouched. Preserve v11
reports and the supplementary E62 diagnostics.

Also measure close-range velocity on identical frames where both outputs are
within the existing 0.5 m correct-track radius. Report lost/gained correct coverage
separately, and per-suite results, so changes in the evaluated population cannot
masquerade as velocity improvement. This diagnostic definition was written before
reading the candidate results. No revised optimization objective is adopted yet.

Evidence: `G/correctness-publication/`, including both protocols, compact frame
exports, and `common_velocity.py`. Both evaluator and analysis jobs completed successfully at nice 15 under the
45 GiB aggregate cap; observed aggregate memory about 8 GiB during export.

#### E63.1 Results and decision

Combined E51/E61 versus original baseline, all 216 development recordings:

| Diagnostic | Baseline | Combined candidate |
| --- | ---: | ---: |
| Raw close position RMSE (m) | 0.243550 | 0.221130 |
| Raw close velocity RMSE (m/s) | 0.678003 | 0.676738 |
| Correctly associated close velocity RMSE (m/s) | 0.626022 | 0.625058 |
| Correctly associated close velocity missing time (s) | 103.618 | 89.348 |
| Velocity RMSE on identical correct close frames (m/s) | 0.625915 | 0.625780 |
| Correctly associated fast-close velocity RMSE (m/s) | 2.304482 | 2.305480 |

Common-frame velocity improves only 0.02158%; several suites regress (worst about
0.146%). Close correct-position coverage gains 17.116 s but loses 2.870 s elsewhere;
this net improvement must not conceal those losses. For velocity-eligible frames,
the corresponding gains/losses are 17.108/2.838 s. The two eligibility definitions
explain the small difference; do not mix them.

Compared with original baseline, the combined candidate withholds 53.422 s of
previously wrong publication and zero correct publication. Publication tuning
also changes remaining estimates; hypothesis counts stay identical. E51 alone
and E51+E61 have identical correctly associated velocity metrics and common-frame
velocity: the clear-miss gate adds no measured velocity benefit. Legacy reports
remain preserved; diagnostic counterexample tests and raw-metric reconciliation
passed. No production source/parameter changes, no fresh-seed evaluation.

Decision: do not promote this as the velocity improvement. Keep the clear-miss
mechanism as a separate promising correction. Next isolate the largest original
baseline velocity-error windows by motion, observation age and scenario, then
choose a targeted model/update change rather than another undirected search.

### E64 — Largest baseline velocity failures [DIAGNOSED; EXPERIMENT PENDING]

`G/velocity-error-audit/` partitions all 216 development recordings by truth speed,
selected primary age, mode, and position correctness. Raw close velocity RMSE
reconciles to 0.678003185 m/s. Fast frames (>=2 m/s) account for 785.042 of
1137.193 time-integrated squared velocity error, despite only 142.098 of 2473.834
seconds of available close velocity. Mode/age are primary diagnostics and do not
prove the published estimate came from that primary rather than auxiliary output.

Largest correct-position 200 ms window: recording 111, fast-near-shot, 143.2–143.4 s;
velocity RMSE 5.946 m/s while position RMSE is 0.161 m. Around 143.10 s truth and
estimate velocities agree near (-1.4, 5.5) m/s. By 143.19 s truth has reversed to
(2.45, -1.53), while the estimate remains near (-1.43, 5.61). Fresh observations
then correct velocity slowly. This is evidence for abrupt-motion recovery, not
proof of a particular physical contact. Proposed experiment: reuse three-point
consistent motion evidence in moving tracks to detect significant velocity
changes; preserve outlier rejection. No implementation or acceptance yet.

User clarified velocity credit should decrease with position error. Plan a
smooth product of position and velocity credits with truth-based denominators;
retain raw metrics, coverage and false-output penalties. Not implemented yet;
E62's hard-radius prototype remains supplementary, not the production objective.

### E65 — B-Human reference and expanded fast/airborne requirements

Inspected official 2025 public release at commit
`d89d603f0388208cecd1c3144b6e8eb92aef4a93`; local source snapshots in
`G/bhuman-reference/`. This is architectural evidence, not a shared benchmark or
knowledge of their current private competition code.

- [Estimator](https://github.com/bhuman/BHumanCodeRelease/blob/d89d603f0388208cecd1c3144b6e8eb92aef4a93/Src/Modules/Modeling/BallStateEstimator/BallStateEstimator.cpp):
  separate stationary/rolling Kalman banks, likelihood ranking and bounded banks;
  initialize new rolling candidates from recent observation pairs with friction
  compensation. New candidates normally need a later update before selection;
  published velocity additionally requires minimum measurement support.
- [Defaults](https://github.com/bhuman/BHumanCodeRelease/blob/d89d603f0388208cecd1c3144b6e8eb92aef4a93/Src/Modules/Modeling/BallStateEstimator/BallStateEstimator.h):
  ten hypotheses per mode, four measurements for nonzero rolling output;
  disappearance evidence after seven expected-visible misses within one metre.
  This updates disappearance metadata, not automatic deletion of the estimate.
- [Percept filtering](https://github.com/bhuman/BHumanCodeRelease/blob/d89d603f0388208cecd1c3144b6e8eb92aef4a93/Src/Modules/Modeling/BallStateEstimator/BallPerceptFilter.cpp):
  verification buffers and trajectory checks precede estimation, including weaker
  percepts supported by coherent rolling motion. This helps explain why copying
  their two-point births alone would not reproduce their robustness (cf. E25).
- [Contact handling](https://github.com/bhuman/BHumanCodeRelease/blob/d89d603f0388208cecd1c3144b6e8eb92aef4a93/Src/Modules/Modeling/BallStateEstimator/BallContactCheckerProvider.cpp):
  foot geometry/motion supplies explicit collision correction. Their inspected
  estimator is planar stationary/rolling, not evidence of full airborne tracking.

New user requirement: prepare for extremely fast, long kicks and airborne balls
from the next opponent. Do not treat present rolling tests as coverage of this.
Our detector projection intersects the ray with fixed ball-centre height and
tracks 2D position/velocity. Larger-image tolerance does not estimate height.
The scorer currently excludes truth speeds above 15 m/s; thus it cannot establish
performance above that speed. Exact opponent speed/flight envelope is not known.

Required next validation work: explicit high-speed and airborne scenarios,
measurement/projection audit (including rays above the ground intersection), and
separate onset, flight, bounce/landing and occlusion recovery metrics. Preserve
original baseline metrics and fresh 67/68 validation; new challenge scenarios
must be versioned separately. Assess whether available image centre/radius and
camera pose support height/range inference before choosing added state/dynamics.
Do not claim airborne support from changes to 2D process noise alone.

### E66 — Reconfirm changed velocity in an existing moving hypothesis [REJECTED]

Motivated by E64's measured reversal lag. Reuse existing three-observation motion
consistency/significance checks while moving; transform their history through
odometry. If that trajectory's velocity differs from the current estimate by
squared Mahalanobis distance >9 using the sum of velocity covariances, replace the
state with its correlated position/velocity initialization. This is a heuristic
change detector, not a claim that the overlapping estimates are independent.
Replace rather than fuse to avoid double-counting the same observations.

No new parameters. Moving-track merges conservatively clear the short history.
120 tests pass, including new moving-reversal and isolated-outlier tests. Frozen
binaries, hashes, before/after source and patch: `G/moving-reconfirmation/`.
Production working source restored after building; evaluation uses frozen binaries.
Two configurations (original and publication-only) completed on all 216 development clips
at nice 15 under the existing memory cap. No airborne/faster-than-15m/s claim.
Evaluate against the original accepted baseline, not the pushed E43 parameters.

#### E66.1 Full development result — do not adopt

Fixed original parameters versus original accepted baseline:

| Metric | Baseline | Reconfirmation | Change |
| --- | ---: | ---: | ---: |
| Close position RMSE (m) | 0.243549712 | 0.243576927 | +0.0112% |
| Close velocity RMSE (m/s) | 0.678003173 | 0.678105884 | +0.0151% |
| Fast-close velocity RMSE (m/s) | 2.350457712 | 2.349902364 | -0.0236% |
| All-range velocity RMSE (m/s) | 0.781474412 | 0.798654136 | +2.1984% |

Wrong-output time increases 0.924 s; close correct-track missing increases 0.094 s;
raw close-velocity missing is unchanged. Worst per-suite close p/v regressions are
0.983%/1.012% (audit2b), and all-range velocity regresses 17.077% in audit4a.
Ten suites change. Publication configuration shows the same type of deltas versus
its matching E51 control. Unit tests establish the mechanism and outlier invariant,
not performance on real replay inputs. Both evaluator jobs terminal; source already
restored. Evidence: `G/moving-reconfirmation/{results,differences.json}`.

Reopening condition: inspect the actual reset windows to distinguish bad observation
triples from a mismatched statistical gate before changing thresholds. Do not repeat
this broad three-observation replacement unchanged. High-speed/airborne scenario
coverage and smooth diagnostic integration remain outstanding; no fresh seeds used.

### E67 — Smooth joint position/velocity credit [DIAGNOSTIC COMPLETE]

User requested velocity credit proportional to positional correctness. Prespecified
supplementary diagnostic in `G/joint-credit/score.py`:

`credit = max(0, 1 - position_error_squared / 0.5^2)^2
          / (1 + 0.3^2 * velocity_error_squared / 0.5^2)`

The 0.5 m support radius is the existing correct-track radius; 0.3 s is the
existing velocity-displacement horizon. Position weight falls smoothly to zero
with zero derivative at the boundary. Missing/nonfinite output earns zero credit;
truth-based denominators prevent dropping hard frames from improving the mean.
This is a bounded credit, not an error multiplied by a weight (which could reward
wrong positions). False-output penalties and raw position/velocity/coverage stay
separate. Four monotonicity/missing/counterexample tests pass.

On 216 clips, baseline mean close credit is 0.892923874; E51+E61 is 0.892763726.
Fast-close credit is 0.546846047 versus 0.546666624. Thus this publication candidate
is slightly worse on the new diagnostic, despite improved positional RMSE and
correct-track availability. Record the unfavorable result; do not tune the metric
to reverse it. Formula remains supplementary, not a production objective change.
Existing <=15m/s evaluation domain retained solely for old-data comparability;
new high-speed/airborne challenge validation must explicitly cover its envelope.

### E68 — Trace harmful moving reconfirmation [DIAGNOSIS COMPLETE]

Paired nine-clip audit4a exports and analysis: `G/moving-reconfirmation/failure-audit/`.
Instrumented frozen reset evaluator/source/log: `G/moving-reconfirmation/trace/`.
120 tests pass for instrumentation builds; production source restored afterward.
All export, analysis and trace jobs completed. No new runtime retained.

Dominant regression is fast-near-shot at 134.2–134.4 s, true range about 7.16 m:
mean truth speed 4.14 m/s, baseline estimated speed 3.07 m/s, experimental estimate
9.29 m/s. Extra integrated squared velocity error in that window alone is 26.207.
Ground-truth height remains about 0.105 m: this failure is not airborne motion.
Paired audit loses 0.040 s correct all-range position and gains none; close metrics
in this suite are effectively unchanged. Do not hide the far-range failure.

Exact reset at detector time 134.162 s has support 43.951; a confirmation-support
guard would not fix it. Old state velocity (2.076, 1.856) becomes (-11.904, 2.968).
Three observation positions transformed into the same current robot frame:
(7.035, -0.047), (6.347, 0.092), (6.083, 0.190), at 40 ms intervals. Their x
variances are 0.01501, 0.01003, 0.00798 m^2. The existing significance and velocity
consistency gates accept this apparently coherent backward trajectory. Trace
appears twice because diagnostic evaluator scores the clip and aggregate separately;
these are repeated replays, not two runtime resets in one run.

Measurement-model audit: filter pixel variance is `(detected_radius * relative_noise)^2`;
simulator centre noise is additive Gaussian with fixed pixel standard deviation
(independent of apparent radius). Thus distant small images can receive too-small
pixel uncertainty under this model. This is a code-supported mismatch, not yet
proof that a noise floor fixes the regression. Also audit the projection Jacobian:
mean uses ball-centre height while `project_noise_to_ground` uses the ground plane.

Next controlled experiment: fixed additive pixel-noise floor, first on baseline
without reconfirmation, then combined only if justified. Evaluate near-range
accuracy and fast onset as well as far-range stability; no threshold tuning against
fresh seeds. A documented pixel uncertainty floor is more physically grounded than
silently restricting recovery by distance or confidence. Do not infer measured
real-world pixel noise from the simulator's default.

### E69 — One-pixel measurement uncertainty floor [REJECTED AS STANDALONE]

Test `max((detected_radius * relative_noise)^2, 1 pixel^2)` independently of E66.
Diagnostic constant only; no new configuration parameter. 118 existing tests pass.
Original and publication configurations evaluated on all 216 development recordings.
Frozen binaries, hashes, source/patch and reports: `G/pixel-noise-floor/`.
Working runtime restored after build; both replay configurations completed.

With original parameters: close position 0.243549712 -> 0.243545289 m (-0.0018%);
close velocity 0.678003173 -> 0.679038474 m/s (+0.1527%); fast-close velocity
2.350457712 -> 2.354979581 m/s (+0.1924%). All-range velocity improves
0.781474412 -> 0.768029458 m/s (-1.7205%), wrong-output time decreases 1.342 s,
close correct-track and raw velocity missing time unchanged. Worst suite close
velocity +2.842%, fast-close +3.605% (audit8a). Publication configuration shows
similar tradeoffs. No promotion: far-range benefit does not satisfy close velocity.
A calibrated floor may still be useful with physically consistent covariance, but
this result does not justify claiming success or retuning on fresh seeds.

### E70 — Consistent ball-centre plane for covariance [GEOMETRY VERIFIED; FIXED-PARAMETER CANDIDATE REJECTED]

Mean projection uses ball-centre height, whereas existing covariance projection
uses the z=0 homography. Test the analytic covariance Jacobian on the same plane
as the mean, independently of E69 and E66. Implemented locally within ball filter
for the experiment; no shared projection API changes. Numerical finite differences
at 0.5, 2 and 7 m verify the covariance at heights 0 and 0.105 m to 0.5% relative
tolerance, with positive-definite covariance and z=0 parity with the existing API.
119 tests pass. This proves derivative consistency, not end-to-end improvement.

Frozen artifacts: `G/height-covariance/`. Original/publication configurations on
216 development recordings; no held-out seeds consumed. Working source restored
after freezing binaries; replay runs at nice 15 under aggregate45GiB memory cap.

#### E70.1 Replay result and next experiment

Both configurations completed. With original parameters, close position RMSE
0.243549712 -> 0.245966206 m (+0.992%); close velocity 0.678003173 -> 0.681125861
m/s (+0.461%); fast-close velocity 2.350457712 -> 2.362341944 m/s (+0.506%).
All-range velocity improves 0.443%, but wrong-output time grows 0.628 s and close
correct-track missing grows 0.418 s. Worst-suite close position +49.55% (audit7a).
Publication control also regresses. Do not adopt under current tuned parameters.

The finite-difference test still establishes a real geometry mismatch; the tuned
noise parameters may compensate for it. Next, bounded calibration of measurement
noise and motion-confirmation behavior around physically consistent uncertainty,
with original-runtime controls, is justified; unconditionally increasing uncertainty
is not. Keep baseline frozen and compare per-suite failures, not only pooled scores.
E69/E70 source restored, frozen artifacts preserved, no evaluation jobs live, fresh
67/68 untouched. No production config or scoring changes.

## 5. Evidence map and operational handoff

- `G` = `/home/schluis/hulk/logs/ball-filter-goal-20261005/`.
- `M` = `/home/schluis/hulk/logs/ball-filter-motion-fixes-20261005/`.
- Original diagnostic capture:
  `/home/schluis/hulk/logs/ball-filter-image-interface-20261005/`.
- Resource runner:
  `/home/schluis/hulk/logs/filter-method-comparison-20261004/resource_run.py`.
- [Lab history](ball-filter-lab.md): parameter semantics, branches, objective
  history, latest validation, and experiment provenance.
- [Simulator guide](../../tools/simulate/README.md): demo and image API usage.
- Port 8765 last served the original nine recordings using their frozen binary;
  it is not a live view of the current branch. Check process state before reuse.
- Artifacts under `logs/` are local evidence, not guaranteed present in a fresh
  clone. Preserve manifests/parameters/binaries before deleting old runs.

### 5.1 Record template for every new idea

Add an entry under the relevant mechanism, using a stable ID:

- **Status/date:** untested, running, retained, rejected, inconclusive, deferred.
- **Question and predicted effect:** including why existing evidence is insufficient.
- **Change:** code commit, exact parameter file, baseline, and ablation.
- **Protocol:** recordings/seeds, objective, acceptance rule, budget, resource cap.
- **Result:** absolute metrics, differences, coverage, and failure examples.
- **Decision:** adopt/reject/defer; limitation; concrete condition for reopening.
- **Evidence:** reports, scripts, source/binary identity, reproducible command.

### 5.2 Session update checklist

Update the current checkpoint and relevant experiment entries; record running
processes or confirm completion; preserve artifacts; state the next concrete
action and unresolved questions. Commit this note with the corresponding work.
Do not silently remove an unsuccessful approach or describe inspected data as
still held out.
