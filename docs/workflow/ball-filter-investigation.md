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

Next: E81 identifies delayed reversal as E80's remaining velocity regression.
E82 restricts changed association to resting-only competition, improving pooled
close/fast metrics and wrong time, but audit8b position still regresses0.501mm RMSE.
E83 traces that regression to inherited support making a newly matched false-size
resting track eligible to win recency. E84's re-confirmation restriction fixes it and
improves pooled position/wrong time, but original-suite velocity regresses0.109%.
E85 proves original raw-velocity regression rewards a misplaced stationary output;
common-correct-frame velocity and joint credit improve across216, but near-truth
duplicate exposure increases28.398 hypothesis-seconds, chiefly recordings144/177.
E86 finds stale nearby resting histories with zero clear-miss exposure; E87 isolates
selection-only improvement (position/wrong output, no fast-velocity gain). Investigate
stale-history retirement without broad merge changes before fresh validation.
E88 confirms one duplicate's center is visible but covariance makes exposure unknown;
the other is physically hidden. Do not apply a common clear-miss deletion rule.
E89 conditional merging removes dominant two-clip duplicate increases and preserves
pooled velocity gains, but development position regresses0.124%; inspect before
fresh validation. Broad0.21m merge control regresses one suite8.64% and is rejected.
E90 localizes remaining discrepancy to approach49.532s; E91's stale precise-state
fusion endpoint does not change that frame and is rejected. Instrument the actual
merge/selection before further policy changes. E92 identifies the stale-age versus
observation-gap mismatch; E93 aligns them and fixes the exact frame, while a smaller
+0.059%audit6a position regression remains. E94 traces it to the hard auxiliary
covariance gate switching output to a94.45% stale-history blend; test continuous
publication weighting before further merge tuning. E95 taper fixes the local jump
but fixed settings fail position; E96 calibration grid012 improves all-suite position
and pooled velocity, with small per-suite velocity regressions still requiring the
position-conditioned and duplicate audit before fresh validation. E97 completes that
audit: duplicates improve31%, but joint credit worsens and audit5b has a real
velocity regression at brief-gaps190.4–190.6s. Trace primary/auxiliary output there
before changing calibration or consuming fresh validation; no candidate accepted.
E98 confirms older resting auxiliary suppresses newer moving primary velocity.
E99 preserves primary velocity for that combination: fixes the target frame but
worsens pooled velocity and audit6a, rejected. Inspect that counterexample before
adding any motion-dependent publication rule.
High-speed velocity acquisition and airborne projection remain open. Explicit >15m/s/airborne captures expose large baseline errors;
they supplement, not replace, the original close-accuracy/fresh-validation goal.
E73's longer-window fallback is rejected; do not relax activation blindly.
E67's smooth position-weighted velocity credit is implemented as a diagnostic;
the publication candidate slightly worsens it. E66's moving reconfirmation fails
full replay and is archived, with working runtime restored.
E56 isolates excessive velocity noise as a cause of direction jitter. E62's paired
216-recording audit confirms E61's one-second clear-miss gate removes 57.132 seconds
of wrong output, loses no correct output, and changes no remaining estimate.
Correctly associated close-velocity error and coverage are unchanged: this is a
false-output improvement, not a velocity winner. The production objective remains
v11; supplementary scoring is diagnostic only. No accepted winner. Experimental
runtimes are archived and prior runtime restored; E74 captures and E75 analysis completed.
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
  - Association, initialization and reacquisition: E06/E07, E09, E13, E25, E29, E45/E46, E64/E66/E68, E72/E73.
  - Resting transitions and trajectory models: E02, E04/E05, E24.
  - Velocity stability, process noise and damping: E18–E22, E27, E34, E53, E56–E58, E69–E71.
- **Candidate lifecycle and selection**
  - Ranking, confirmation and confidence caps: E01, E23, E39–E44, E52, E54.
  - Retention, visibility and duplicate merging: E10–E12, E14, E36–E38, E47, E55.
  - Publication versus internal retention: E15/E16, E26, E31/E32, E50/E51, E59–E61.
- **Evaluation and decision making**
  - Parameter calibration and held-out failures: E03, E08, E17–E22, E27/E28,
    E30, E33–E44, E48.
  - User checkout provenance: E49.
  - Correctness-aware loss and paired coverage audit: E62/E63, E67/E72.
  - External reference and airborne requirements: E65.
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

### E71 — Controlled noise calibration around corrected covariance [COMPLETE; NO SCREEN SURVIVOR]

45 prespecified configurations per runtime (90 total): relative detection-noise
scale [0.7,0.8,0.9,1,1.1], moving position-noise scale [0.75,1,1.25], moving
velocity-noise scale [0.75,1,1.5]. All other original parameters unchanged.
Evaluate identical grid under original and E70 corrected-height runtimes, each
on all 216 development clips. The original-runtime arm distinguishes parameter
benefit from geometry benefit; accepted baseline remains fixed. No E66 reset or
E69 floor. Frozen runtime hashes and protocol: `G/height-calibration/`.

16 low-priority workers per process, aggregate memory observed about8GiB under45GiB
cap during loading. Screening requires pooled close p/v improvement, per-suite
p/v nonregression, fast-close nonregression and no worse wrong/correct-close/raw
missing coverage; all-range metrics also reported. Any screen survivor still needs
paired smooth-credit/velocity analysis, failure-window review and fresh frozen
validation. No new objective adopted; no faster-than15m/s or airborne support claim.

#### E71.1 Result

Both jobs completed all45 configurations,90 results total. No screen survivor;
original-runtime grid031 reproduces baseline exactly (control parity).
Corrected-height grid040 is a useful diagnostic compromise: close position
0.243451123 m (-0.0405%), close velocity 0.675066018 m/s (-0.4332%), fast-close
2.331299339 m/s (-0.8151%), all-range velocity0.772565852 m/s (-1.1400%).
Wrong time -0.308 s, close correct missing -0.092 s, raw velocity missing unchanged.
However worst per-suite close position+0.489% (audit7a), close velocity+1.316%
(original), all-range velocity+4.455% (audit3b). No promotion or fresh evaluation.

The matching original-runtime grid040 also improves pooled p/v but regresses
per-suite close velocity up to2.148%; thus geometry modifies the tradeoff, not
uniformly resolves it. Corrected-height grid036 improves close velocity0.527%
but worsens pooled position0.137%. Do not select purely by pooled error.

Evidence: `G/height-calibration/{protocol.json,rank.py,ranked.json,original,height}`.
Next inspect grid040's original-suite velocity regression on identical correctly
tracked frames and measure smooth joint credit before deciding whether refinement
is justified. This is diagnosis, not relaxation of the goal's accuracy requirements.
Runtime/defaults remain unchanged, no jobs live, no fresh seeds consumed.

### E72 — Paired diagnosis of corrected covariance calibration [COMPLETE]

Original nine-clip suite, E71 corrected-height grid040 versus original baseline.
Artifacts: `G/height-calibration/paired-original/` (frozen protocol, exports,
paired diagnostics, all-range and close-only error windows, smooth joint credit).
All jobs terminal; runtime untouched. Previously inspected data, no fresh seeds.

Close correctly associated velocity RMSE 0.640577568 -> 0.649581754 m/s, while
associated missing time remains exactly4.834 s. No correct close-position frames
lost or gained. Thus the velocity regression is real, not recovered difficult
frames changing the evaluated population. Smooth close joint credit decreases
0.883812121 -> 0.880734074; fast credit0.533123264 -> 0.532906549.

Top close-error windows: sideline365.0–365.2 s, truth speed1.827m/s at0.453m,
baseline estimated speed0.748m/s versus candidate0.00146m/s. Brief-gaps183.0–183.8s:
truth slows from0.72 to0.60m/s while candidate velocity is zero; baseline follows
about0.69 to0.59m/s. Earlier all-range windows also show delayed motion acquisition.
These observations support a motion-confirmation bottleneck as uncertainty grows;
they do not justify simply lowering every significance gate.

Next hypothesis: preserve existing three-point fast confirmation, but allow a
longer coherent observation window to establish slower motion when short intervals
are individually insignificant. Distinct from rejected E07 (relaxing the same
three-point span), E25 (two-point confirmation), and E13 (three-point weighted fit).
Must retain isolated-outlier, odometry and long-gap invariants, compare fixed
original parameters first, and assess fast onset/false motion before calibration.
No new algorithm implemented at this checkpoint; high-speed/airborne validation
extension remains required and is not covered by this slow-motion diagnosis.

### E73 — Five-observation fallback for slower coherent motion [REJECTED]

Keep original three-point confirmation on the latest three observations unchanged.
Retain up to five observations; when the short-window test fails, require every
consecutive displacement to point along the net displacement and intermediate
points2/4 to agree with interpolation on segments1–3/3–5 within squared Mahalanobis9.
Then run the original significance/velocity-consistency tests on points1/3/5.
This accumulates temporal displacement without lowering significance thresholds.
No E66 reset, no covariance correction, no new parameter. Existing time-gap and
odometry transforms apply to the whole history.

120 tests pass, including a0.7m/s case that confirms on observation5 while remaining
insignificant on the short window, and false-percept rejection at every position
in a five-observation stationary sequence. Existing fast/outlier/odometry/gap tests
remain intact. Tests prove intended mechanics, not end-to-end performance.

Frozen source/patch/binaries/hashes and protocol: `G/longer-motion-evidence/`.
Two fixed configurations (original and publication) on216 development recordings,
nice15, aggregate45GiB memory cap. Source restored after freezing. No fresh seeds,
no high-speed/airborne support claim. Evaluate against matching original-runtime
controls; inspect wrong/missing/duplicates before any promotion.

#### E73.1 Full replay rejection

Original parameters: close position0.243549712 -> 0.278115923m (+14.193%),
close velocity0.678003173 -> 0.719463502m/s (+6.115%), fast-close velocity
2.350457712 -> 2.525985683m/s (+7.468%). Wrong-output time+4.440s and close
correct-track missing+9.052s; raw missing unchanged. Worst suite close position
+175.897% (audit4a). Publication control also regresses strongly. All-range velocity
is nearly unchanged, illustrating why its aggregate cannot hide close failures.

Reject despite120 passing tests. Frozen patch/source and results retained; working
source already restored; both jobs terminal. Do not repeat longer-span promotion
unchanged. Earlier E07/E13/E25 and now E73 show that easier motion activation alone
is not a robust solution; the downstream hypothesis/selection effects matter.

Next prioritize explicit fast/airborne scenario coverage requested in E65. Current
scenario impulse is planar and old scoring excludes speeds above15m/s. Add an
explicit vertical kick input and coverage checks without altering old recipes or
baseline scores, then measure projection/flight/landing failures before choosing
more filter changes. This does not replace the original close-accuracy goal or its
fresh validation requirement. No adopted runtime changes and no fresh seeds used.

### E74 — Explicit airborne and high-speed challenge captures [CAPTURES COMPLETE]

Simulator recipe adds optional `ball_vertical_impulse` (upward world-frame N s,
default0), applied through the existing one-step external-force integration. Existing
planar recipes remain unchanged. Capture coverage adds peak ball-centre height,
time >0.1m above resting height, and time with horizontal speed>15m/s. Separate
`scenarios/ball-filter-challenges/{airborne,high-speed}.json` recipes check actual
challenge coverage; the old nine-scenario demo and legacy scores are unchanged.

Following the user's wall-height question, recipe `wall_height_metres` is configurable
(default1m, airborne3m). Physical capture enclosure uses it; ordinary scene/visual
walls retain1m. Explicitly separate artificial wall rebounds from natural flight
and landing when assessing performance. This is capture infrastructure, not a
claim of airborne filter support or a change to the competition filter.

81 simulator tests pass,1 ignored. New tests verify vertical momentum/no persistent
force, rise/fall under gravity, elevated rebound at2m against a3m wall, recipe
validation and old planar defaults. Fixed stale capture-layer test to compare
field-boundary decay to the actual base layer instead of old hardcoded2.0; E43
checkout default is0.5. Documentation includes the capture command and limitations.

Artifacts: `G/airborne-challenges/`. Build complete; frozen simulator available.
First capture failed due missing ONNX runtime environment before useful physics;
failed output/log preserved as `airborne-missing-ort*`, generated core dump removed.
Retried with established ORT_DYLIB_PATH. Airborne capture currently live (exec77041),
using original accepted parameter file, new seed offset80000000, no opponents.
These are challenge-development captures, not untouched final validation. High-speed
capture must run after airborne because both use router port7448. Observed aggregate
memory about5GiB, nice15,45GiB cap. Inspect live handle before restarting any job.

Next: verify actual challenge coverage, export original-baseline estimates, and
measure high-speed/flight/landing accuracy without the legacy15m/s exclusion. Keep
those supplemental metrics distinct from old scores. Fresh67/68 remain unscored.

### E75 — Baseline performance on airborne and >15m/s challenges [COMPLETE]

Both three-recording captures completed with required coverage. Airborne peaks
0.677–0.724m, airborne time0.772–0.872s per recording. High-speed peaks20.762–22.640m/s.
Original accepted parameters used; source/scenario/evaluator hashes in
`G/airborne-challenges/challenge-protocol.json`. New development seed offset80000000;
filenames include `validation` from capture CLI but these inspected recordings are
not held-out final validation. Fresh67/68 untouched. Airborne replay matches captured
live output (`airborne-verified/report.json`, live_replay_verified=true).

Supplemental analyzer `analyze_challenge.py` scores all finite Field velocities
with valid time/pose, without the legacy15m/s exclusion. Position/velocity are
horizontal only. Flight: centre>radius+0.1m; landing window: first500ms after descent
to radius+0.02m following flight; wall approach: within0.25m of enclosure boundary.
These are geometric diagnostic categories, not a contact classifier. Missing and
incorrect coverage reported alongside RMSE; old objective/reports unchanged.

| Diagnostic interval, pooled3 recordings | Time (s) | Position RMSE (m) | Velocity RMSE (m/s) |
| --- | ---: | ---: | ---: |
| Airborne recipe: flight | 2.486 | 1.4914 | 3.0380 |
| Airborne recipe: close flight | 0.964 | 0.4666 | 3.1572 |
| Airborne recipe: flight before first wall approach | 2.366 | 1.5287 | 3.1126 |
| Airborne recipe: first500ms after landing | 3.240 | 2.8522 | 2.9196 |
| High-speed recipe: >15m/s | 1.796 | 5.3585 | 18.5111 |
| High-speed recipe: close and >15m/s | 0.294 | 1.5121 | 19.9643 |

No missing estimates in these intervals: most failure is wrong published output.
Close >15m/s is position-incorrect (>0.5m) for0.268 of0.294s. This is only three
stress scenarios and a short close exposure; not a population-level opponent estimate.
Airborne figure `airborne-baseline.png` was rendered and inspected: estimated
horizontal motion lags true motion during flight and after landing, before the
wall rebound. This supports investigating projection and acquisition, not blaming
wall physics alone.

Reconciliation: supplemental airborne close velocity matches legacy within1e-4.
Close position differs0.000107m because supplemental position uses velocity-eligible
Field-range frames while legacy position uses Ground-reference range independently;
recorded in `airborne-analysis/reconciliation.json`, not hidden by loosening tolerance.

Measurement caveat: simulator detection centres have Gaussian pixel noise, while
ball radius is generated directly from true depth. A size-based height/range method
must be stress-tested with imperfect radius estimates before claiming real-world
benefit; do not exploit synthetic perfect radius. Next trace high-speed first
observations and ground-projection rejection during flight before choosing an
implementation. All capture/export/analysis jobs terminal; no filter change adopted.

### E76 — First fast-kick acquisition trace [DIAGNOSED, NOT FIXED]

Frozen accepted-baseline evaluator and original parameters replayed E75 high-speed
train-42 with full hypothesis states; job completed successfully. Evidence:
`G/airborne-challenges/first-kick-trace/{frames-0.jsonl,report.json,summarize.py,first-observations.json}`.
This inspected challenge remains development data. No production changes.

- At output3.026s, detection timestamp3.002s updates the selected resting track;
  support62.528, estimated velocity zero. Actual ball has started moving.
- At output3.068s, timestamp3.042s supplies one ball detection. Another resting
  hypothesis now has support1.902 and one motion observation at x≈0.730m;
  selected track remains at x≈1.493m, last_seen3.002s. Hypothesis count remains4.
  Do not call this a new birth: an existing weak track may have absorbed it.
- At output3.100s, timestamp3.082s supplies another detection. The other track
  has support2.666, two motion observations and resting state x≈0.434m. Its latest
  size-plausibility flag is false. Selected stale track remains x≈1.469m while
  truth is already x≈−0.558m in current Ground coordinates.
- Subsequent timestamps3.122/3.162 contain no ball detections. The alternative
  never obtains a third observation in this acquisition window. The selected
  track retains support62.528 and zero velocity.

This distinguishes observation availability, association/selection, and motion
confirmation: the filter receives two post-kick detections but does not publish
motion before losing observations. It does not yet prove which association gate
redirected the first percept, or whether the second percept is accurate (its size
flag is false). The baseline maximum distance is1.601m, reacquisition gate0.1365m;
inspect actual assignment costs/protection before blaming either threshold.
The robot had walked to Field x≈2m by kick time, shortening observation opportunity.
A two-observation velocity alone cannot be trusted here: the paired projected
positions imply substantially slower motion than truth. Next audit raw detection
geometry/timestamps and assignment decisions, then test a narrowly justified
change against E25/E73 false-motion counterexamples and the complete development
suite. Fresh67/68 untouched; memory≈4.83GB; no agents or pushes.

### E77 — Assignment costs explain first fast-kick capture [DIAGNOSED]

Instrumented baseline association immediately before Hungarian assignment, using
frozen original parameters on E75 train-42. `G/first-kick-association/` preserves
before/after source, frozen evaluator, complete trace, and exact aggregate parity
receipt against E76. Diagnostic source restored after build. Replay terminal.

At3.042s the true-looking percept projects to(0.75495,−0.02943)m, radius32.0529px.
The established track is eligible: gated Mahalanobis score−1.94945, after uncertainty
penalty−2.32348. An existing diffuse weak track scores−0.08200, then−1.03709 and
wins. Thus neither maximum-distance nor reacquisition gate blocked the primary;
normalized residual favors the diffuse history despite the bounded penalty.
At3.082s, the next percept projects to(0.44404,0.05420)m with radius **exactly8px**,
matching the simulator's fixed false-positive radius. Combined with its false
size-plausibility flag and inconsistent motion, this strongly identifies an
outlier rather than a second useful true-ball measurement. Raw detections do not
carry a truth/provenance label, so this is a source-supported inference.
The weak updated track again wins (−0.64150 versus−2.35844). Do not interpret E76's
two ball-labelled detections as two verified observations of the real ball.
Auxiliary publication-filter traces are also present; distinguish them by their
single-hypothesis state/configuration. Per-recording and aggregate replay each
execute tracking, explaining repeated trace blocks.

### E78 — Relative logarithmic association uncertainty [REJECTED]

Question: can an unsaturated covariance penalty prevent diffuse weak candidates
from absorbing a fast ball's first displaced observation? Change only association
uncertainty penalty, original parameters otherwise. `G/association-log-penalty/`
preserves sources, config, binaries, build logs and SHA256 protocol.

First attempt changed trace/(1+trace) to log(1+trace). It failed two existing tests:
the sole eligible diffuse track was rejected against the unmatched dummy. Archived
as `lib-after.rs`, not adopted. Revised version subtracts the minimum log penalty
among eligible tracks for each percept, so uncertainty expresses relative track
preference; sole eligible track has zero relative penalty. All118 tests pass.
Source archived as `lib-after-relative.rs`; working source restored after build.

Challenge replay terminal: at3.068s the established track now accepts the3.042s
observation and output x moves from1.493 to0.731m. It still has zero velocity and
subsequent outlier/publication behavior remains problematic (output x1.412 at3.1s).
This verifies the association mechanism, not a complete tracking solution.
Full216-development replay completed under the45GiB aggregate cap (observed~9GB),
nice15, original configuration only. `comparison.json` records:

| Metric | Baseline | Candidate | Change |
|---|---:|---:|---:|
| Close position RMSE (m) |0.243549712|0.243278659|−0.1113%|
| Close velocity RMSE (m/s) |0.678003173|0.679248701|+0.1837%|
| Fast close velocity RMSE (m/s) |2.350457712|2.349591298|−0.0369%|
| All velocity RMSE (m/s) |0.781474412|0.784191563|+0.3477%|

Wrong output increases10.050s, correct-close missing decreases2.128s, raw close
velocity missing unchanged. Worst suite audit3a close position+41.582%, velocity
+4.462%. Reject despite diagnostic association improvement and pooled position
gain. No fresh67/68 evaluation. All jobs terminal, filter source restored. Next
inspect the audit3a regressions alongside the first-kick counterexample before
choosing a more selective association or motion-evidence rule; unsaturated
uncertainty preference alone is insufficient. Preserve single-outlier protection.

### E79 — Logarithmic association regression is outlier capture [DIAGNOSED]

Paired audit3a exports locate E78's worst close-position regression in fast-near-shot
seed50000049,143.6–144.6s. Evidence `G/association-log-penalty/failure-audit/`
includes compact paired frames, position-window rankings and detailed failure window;
`trace/` freezes instrumented candidate source/binary/assignment log. Source restored.

At143.642s, after280ms without a matched observation, a percept projects to
(0.57835,−0.61464)m with radius exactly8px (consistent with synthetic false positives).
The established moving track predicts(−0.08721,0.34990)m, covariance trace≈4.267.
E78 gives that track score−0.64371 versus−2.03999 for the diffuse competitor, thus
redirecting this likely outlier into the established history. At output143.664s,
truth is(0.20708,0.31488)m, baseline output(−0.08859,0.40169), E78 output
(0.50977,−0.67252). Subsequent missing detections let corrupted prediction persist.
In close window144.2–144.4s, baseline position SSE0.376772m²s versus1.276504m²s,
equivalent RMSE1.37254m versus2.52636m over0.2s. Both are wrong; candidate is worse.
This explains why E77's beneficial preference cannot be applied indiscriminately.

### E80 — Size-qualified logarithmic association [NOT ACCEPTED]

Use E78's relative log uncertainty preference only when existing radius_consistency
returns Some(true); retain baseline bounded penalty for false/unknown size geometry.
This does not hard-code synthetic8px, change perception, or reject airborne enlarged
images: the existing check rejects only substantially undersized apparent balls.
Still requires eventual radius-noise stress; synthetic true-ball radius is ideal.
Original parameters unchanged, all118 filter tests pass. `G/association-size-log/`
contains source before/after, binary/config SHA256 protocol, build log and evaluator.
Working source restored immediately after build. Full216 development evaluation
completed under45GiB/nice15; held-out67/68 untouched. Results (`comparison.json`):

| Metric | Baseline | Candidate | Change |
|---|---:|---:|---:|
| Close position RMSE (m) |0.243549712|0.243414374|−0.05557%|
| Close velocity RMSE (m/s) |0.678003173|0.677706196|−0.04380%|
| Fast close velocity RMSE (m/s) |2.350457712|2.349078535|−0.05868%|
| All velocity RMSE (m/s) |0.781474412|0.782775693|+0.16652%|

Wrong output−3.060s, correct-close missing−0.198s, raw close velocity missing0change.
Worst close position+0.2665%audit9a; close velocity+1.2244%original, fast+1.7893%
original; all velocity+6.1522%audit6a. Does not satisfy existing suite guards.
Challenge first match improves but zero velocity remains, and at3.100s output
reverts toward stale publication history despite the size-qualified association.
`challenge/first-observations.json` records this; do not count first-match recovery
as end-to-end high-speed tracking. All jobs terminal, no adopted runtime change.
Next inspect original-suite velocity regression and candidate primary versus
auxiliary output after the suspicious second percept; do not weaken acceptance
criteria or fresh-test this configuration merely because pooled gains are positive.

### E81 — Size-qualified preference delays a real reversal [DIAGNOSED]

`G/association-size-log/original-audit/` contains paired compact original-suite
exports, close-velocity200ms window ranking and detailed failure-window.json.
The remaining original-suite close-velocity regression concentrates in fast-near-shot
133.4–133.8s. Outputs are identical through133.314s; true motion reverses around133.3s.
At133.336s baseline moving track is unmatched (age54ms), while E80 keeps a moving
track matched (age14ms), with velocity still pointing the old way. At133.422s the
baseline output velocity is(−4.8804,1.7115)m/s, while E80 is(2.3581,−0.0325)m/s.
By133.516s truth is at0.938m range; baseline position error0.042m versus E80's0.158m,
velocity(−4.4812,2.1455) versus(1.0111,0.1755). E80 subsequently adapts slowly.

During the0.118s close portion of133.4–133.6s, true mean speed3.1503m/s, baseline
mean speed4.8371, candidate0.7338; integrated velocity squared error0.340375 versus
1.489320. This is not merely velocity credited at an incorrect position: baseline
position is accurate after recovery. Matched-history persistence prevents the
rapid alternative-track recovery seen in baseline. Avoid changing moving-track
competition just to repair resting-track acquisition.

### E82 — Restrict logarithmic preference to resting-track competition [DEVELOPMENT ONLY]

E80's size-qualified relative log uncertainty is applied only when **every eligible
hypothesis is resting**. If any eligible moving hypothesis exists, preserve baseline
bounded uncertainty preference for that percept. Eligibility uses the same finite
and maximum-cost gates as assignment. This targets E77's initially stationary
competition while retaining baseline reversal association; no scenario/time checks.
Original parameters unchanged. `G/association-resting-log/` preserves sources,
config, build log, evaluators and SHA256 protocol. All118 existing filter tests pass.
Full216 development replay completed; working source restored. Results:

| Metric | Baseline | Candidate | Change |
|---|---:|---:|---:|
| Close position RMSE (m) |0.243549712|0.243503639|−0.01892%|
| Close velocity RMSE (m/s) |0.678003173|0.677856122|−0.02169%|
| Fast close velocity RMSE (m/s) |2.350457712|2.349776931|−0.02896%|
| All velocity RMSE (m/s) |0.781474412|0.780696452|−0.09955%|

Wrong output1045.142→1042.400s (−2.742s); correct-close missing106.924→106.886s;
raw missing67.652s and close missing28.900s unchanged, missing runs192 unchanged.
Fast-close velocity no worse in every suite. Worst close-velocity change+0.000465%
audit9a; worst close position+0.19670%audit8b,0.254627421→0.255128268m (+0.501mm).
This small regression remains a failed guard, not permission to declare success.
Worst all-velocity+0.00826%audit10a. Duplicate hypotheses still need paired review
before any promotion; score report lacks a duplicate count. No accepted runtime or
fresh67/68 evaluation. Next inspect audit8b's position regression and test narrowly
justified corrections, preserving reversal and isolated-outlier counterexamples.
All jobs terminal; peak observed RAM<10GiB, CPU nice15, no pushes.

### E83 — Resting-association regression exposes inherited-support selection [DIAGNOSED]

Paired audit8b development exports and a full-state baseline/candidate replay of
brief-gaps seed50000062 are preserved in `G/association-resting-log/position-audit/`.
Scripts `position_windows.py` and `duplicates.py`, failure-window/selection-states
and duplicate/count reports make this reproducible. All replay jobs terminal.

The+0.501mm suite RMSE hides a large short failure at202.538s: truth
(0.19759,0.14792)m, baseline(0.12629,0.18308)m (≈0.080m error), candidate
(0.72754,−0.56090)m (≈0.885m error), zero published velocity. In202.4–202.6s,
position SSE0.00137850→0.03225701m²s (0.08302→0.40160m RMSE over0.2s).
The existing moving hypothesis is identical in position/covariance/velocity in both
replays. Its support is47.231 baseline,47.275 candidate. The false-size resting
hypothesis at the wrong location has support2.657 baseline versus14.967 candidate.
Both last matched202.522s; moving track last matched202.282s. This is a selection
switch caused by different inherited support, not corruption of the moving state.

`filter.rs::select_hypothesis` requires support≥3 to compete against established
history; with original selection_confidence_cap=0, eligible ranking ties resolve
by last_seen. Baseline wrong track fails that confirmation threshold; candidate
passes and wins recency. A previously supported track can therefore regain output
on one size-inconsistent percept. This qualifies the comment that a lone false
observation cannot trigger recovery: it prevents a newborn from doing so, not a
retained formerly supported hypothesis. No fix claimed; relaxing association gates
or ignoring the tiny aggregate regression would conceal this failure.

Hypothesis counts across all9 audit8b clips:2165.446→2183.726 hypothesis-seconds,
counts differ for18.280s over359.904s. These counts alone are not duplicate counts.
On the full-state brief-gaps clip, near-truth duplicate proxy (extra hypotheses
within0.3m of sole truth) is unchanged1.552 hypothesis-seconds, multiple-near-truth
duration1.552s. Total hypothesis-seconds223.226→224.906; near-truth hypothesis-seconds
37.968→38.008. First state/count divergence190.104s. This is only one clip's spatial
duplicate assessment, not evidence of full216 duplicate nonregression.

Next inspect/restrict re-confirmation of retained, newly matched size-inconsistent
histories before they displace a supported track, preserving ordinary true-ball
recovery and existing false-percept counterexamples. Consider existing leadership
observation evidence before adding state. E82 remains unaccepted; source unchanged,
fresh67/68 untouched, RAM≈4.85GB after completed jobs, no agents or pushes.

### E84 — Reconfirm size-inconsistent resting challengers [DEVELOPMENT ONLY]

Combine E82 association with a selection restriction: when a supported (effective
validity≥3) moving track has a size-plausible last observation, a size-inconsistent
resting challenger must have three observations in its existing MotionEvidence
window before it is eligible. Do not exclude a lone track, size-plausible resting
track, or a moving challenger. Existing motion history ignores repeated/too-close
timestamps and clears after gaps>200ms. LeadershipEvidence cannot be reused here:
competition.rs populates it only for the current leader, so requiring it of a
challenger would create a circular eligibility dependency.

`G/resting-reconfirmation/` preserves all three source before/after snapshots,
config, binaries and hashes. New unit test covers both vector orders, lone-track
fallback and size-plausible recovery; all119 filter tests pass. Working sources
restored after build; no schema/state added. Full216 development evaluation and
motivating brief-gaps full-state replay completed under45GiB/nice15 (observed<10GiB).
At202.538s it retains the baseline moving estimate, error0.079501m instead of E82's
0.885m; `diagnostic/motivating-frame.json` records exact values.

| Metric | Baseline | Candidate | Change |
|---|---:|---:|---:|
| Close position RMSE (m) |0.243549712|0.242773335|−0.31878%|
| Close velocity RMSE (m/s) |0.678003173|0.677871787|−0.01938%|
| Fast close velocity RMSE (m/s) |2.350457712|2.349776931|−0.02896%|
| All velocity RMSE (m/s) |0.781474412|0.780476858|−0.12765%|

Wrong output−7.154s, correct-close missing−1.508s, raw close velocity missing unchanged.
Worst close position+0.000301%audit8a; close velocity+0.108775%original; fast close
no suite regression. Worst all velocity+0.020731%audit7a. This improves the mechanism
and pooled scores, but original-suite velocity still fails the existing guard.
Next audit position correctness on that velocity regression and compare against
E82 to isolate the new selection restriction. Do not claim the goal or spend fresh
seeds based only on the pooled gain. Three-observation recovery needs an explicit
positive test and independent selection-only ablation before adoption if this line
survives replay. Thresholds and fresh-validation requirements unchanged; fresh67/68
untouched. All jobs terminal; no retained runtime change or push.

### E85 — Correctness and duplicate audit of re-confirmation [COMPLETE]

Paired original-suite exports in `G/resting-reconfirmation/original-audit/` localize
raw velocity regression to contested238.6–240.2s. At238.734s baseline publishes
stationary false position0.611m from truth; candidate publishes moving estimate
0.065m from truth. At239.064s errors are0.761m baseline versus0.146m candidate.
Thus raw velocity favors zero velocity on a wrong ball location in this window.
This does not prove every E84 difference is benign: retain raw reports.

Fixed E63 common-correct-frame audit on all9 original clips finds identical velocity
RMSE0.640577568m/s over108.514s, zero correct-close coverage lost,1.398s gained.
Fixed E67 smooth joint credit improves0.883812121→0.890130948; fast credit unchanged
0.533123264. No formula or threshold changed to obtain this result. The original
raw velocity guard still fails; this evidence explains its meaning rather than
silently marking it passed. The user explicitly requested position-dependent
velocity credit, so evaluate the existing diagnostic alongside raw guards.

Full216 paired export in `G/resting-reconfirmation/full-audit/` completed in9-clip
chunks, retaining hypothesis positions but removing full states after each chunk.
Replays used the resource runner, nice15, aggregate45GiB; observed~5GiB. All export
and analysis jobs terminal. Common-velocity and credit formulas were unchanged:

- Common correct close frames2395.788s: velocity RMSE0.625915140→0.625754701m/s
  (−0.02563%); correct-close coverage gained1.550s, lost0.040s (recording85).
- Two tiny common-frame suite regressions remain: audit3b+4.29e−8m/s,
  audit9a+3.09e−6m/s. Preserve these rather than silently round them into passes.
- Smooth close joint credit0.892923874→0.893315691; fast credit
  0.546846047→0.547359098. Raw legacy metrics/failing guards remain in E84.
- Across8634.348s, hypothesis-seconds54014.472→54099.194. Extra hypotheses
  within0.3m of sole truth693.982→722.380 (+28.398,+4.09%); exposure with
  multiple near truth624.000→652.398s. All nearby-pair seconds2703.062→2738.952.

`duplicates.py` defines these as spatial proxies, not proof of track identity.
The duplicate increase is concentrated in recording144 (+17.280 hypothesis-seconds),
177 (+9.080),22 (+1.200),183 (+0.842),73 (+0.086); ranking saved in
`duplicate-regressions.json`. Thus improved output does not imply fewer candidates.
Next inspect144/177 duplicate survival and isolate selection-only behavior before
promoting added association complexity. Candidate remains development-only; fresh
67/68 untouched, no adopted code or push. Full-state positive recovery test and
selection-only ablation remain outstanding, as do fast/airborne challenge failures.

### E86 — Dominant duplicate windows [DIAGNOSED]

E85's duplicate-window audit is saved as `full-audit/duplicate-windows.json`.
Recording144 (stationary-close seed50000058) has two candidate hypotheses near
truth from12.230 to29.480s, versus one baseline: positions(0.70262,−0.13389) and
(0.83517,−0.10234)m at onset, separated≈0.136m; truth(0.69703,−0.13486).
Candidate published position is accurate (≈0.0057m error); baseline publication
is≈0.074m away. Thus the better output coexists with another nearby history.
Recording177 (long-occlusion seed50000062),259.512–268.560s: candidate positions
(1.44489,−0.09330) and(1.59724,−0.10622), separated≈0.153m; truth
(1.60122,−0.10885), output≈0.0048m error. Baseline has one near-truth hypothesis.
These separations exceed baseline merge distance0.1m. Do not infer that widening
merging is safe: E36 already shows wider-merge position regressions. Full-state
candidate replay completed; `duplicate-states/nearby-histories.json` preserves:
stationary-close extra resting track last_seen9.482s, support3.802, versus observed
track last_seen12.202s/support279.514. Long-occlusion extra resting track last_seen
257.202s/support159.120, versus current259.482s/support4.880. Both stale histories
have clear-miss duration0 and no last_clear_frame despite their proximity to the
observed ball. This is consistent with covariance/visibility ambiguity preserving
history (E52), not a newly growing trail of moving hypotheses. Exact exposure
classification still needs trace before asserting a new decay rule is safe.

### E87 — Re-confirmation without association changes [COMPLETE ABLATION]

Independent E84 ablation: original association, original parameters, only resting
size-inconsistent challenger selection restriction and existing motion-history
query. `G/resting-reconfirmation-only/` freezes source before/after, binaries/config
and SHA256 protocol. Added history test verifies three observations, reset after
>200ms gap, repeated timestamp not counting, and re-earning three observations.
All120 filter tests pass. Working source restored after build. Full216 replay
completed under45GiB/nice15; fresh67/68 untouched. `comparison.json`:

| Metric | Baseline | Selection only | Change |
|---|---:|---:|---:|
| Close position RMSE (m) |0.243549712|0.242858959|−0.28362%|
| Close velocity RMSE (m/s) |0.678003173|0.678034044|+0.00455%|
| Fast close velocity RMSE (m/s) |2.350457712|2.350457712|unchanged|
| All velocity RMSE (m/s) |0.781474412|0.781223533|−0.03210%|

No suite position regression; worst raw close velocity+0.108775%original (E85
explains its correctness confound). Wrong output−4.340s, correct-close missing
−1.398s, raw close velocity missing unchanged. This isolates a useful position/
wrong-selection benefit but is **not a velocity winner**: E84's small fast-velocity
gain comes from changed association. Full duplicate comparison of this ablation
not yet run; do not infer exact hypothesis parity because selection influences
competition decay. No promotion. All jobs terminal, source restored, no pushes.

Next investigate retirement of nearby stale histories left by the improved
association without blindly widening merge distance, preserving separate observed
balls and fast-moving transitions. Any such experiment must revisit all216 metrics
and duplicate exposure, then freeze before fresh validation. Independent high-speed
and airborne limitations remain.

### E88 — Duplicate visibility differs between clear and occluded scenes [DIAGNOSED]

Instrumented E84 association/decay at exact detector timestamps12.202s and259.482s.
`G/duplicate-visibility/` preserves frozen instrumented source/binary, build log,
trace and exact aggregate parity receipt against E86's two-clip replay. All source
files restored after build; replay terminal. Repeated log entries are per-recording
and aggregate evaluation, not repeated detector updates.

- Stationary-close stale track last_seen9.482s, position(0.83483,−0.10687), support
  3.802, covariance diagonal≈20.47594m², visibility scale3: center **Visible**,
  full uncertainty footprint **Unknown**. The3σ half-width is≈13.575m. This is
  uncertainty conservatism preventing clear-miss exposure, not physical occlusion.
- Long-occlusion stale track last_seen257.202s, position(1.44489,−0.09329), support
  159.340, covariance diagonal≈17.00335m²: center **Hidden**, full **Hidden**.
  Do not count this as a clear miss merely because a nearby ball is observed.

`negative_evidence::classify_with_detections` uses Robot detections as occluders;
Ball-labelled detections do not directly shield all nearby candidates from misses.
Thus two hypotheses sharing one percept is not the visibility mechanism here.
Resting covariance growth is confirmed by source: Qrest≈0.014915m² per reference
2ms, i.e.≈7.458m²/s per axis. This explains≈20m² after2.7s. It is a tuned adaptation
noise scale, not a physically bounded stationary-ball displacement model. E02's
covariance-aware resting transition is a different rejected experiment; do not
conflate it with prediction-noise/visibility policy.

Do not delete both stale tracks under one center-visible rule. Next evaluate a
narrow duplicate-merge condition using existing covariance agreement and disjoint
observation histories, distinguishing recently observed resting evidence from
stale alternatives and preserving two simultaneous observations. A broader0.3m
merge remains rejected (E36). Alternatively investigate resting uncertainty growth
with explicit reacquisition/occlusion guards; neither is currently adopted.
All jobs terminal, fresh67/68 untouched, no pushes. Existing candidate still has
unresolved duplicate increase and fast/airborne limitations; goal not complete.

### E89 — Conditional resting duplicate merging [DEVELOPMENT ONLY]

E84 combined with configured merge maximum0.21m and a physical resting-track
limit: ordinary resting pairs limited to one ball radius; recent confirmed
size-plausible resting track (age≤120ms, support≥3) with stale resting history
(age≥1s) may use up to a diameter. Actual limit is min(configured, physical), so
zero still disables merging and the parameter remains an upper bound. Existing
same-exposure/disjoint-history and covariance agreement checks remain. Moving
pairs retain configured-distance behavior (thus parameter increase affects them).
This is explicitly not a proof that overlapping predicted means imply one ball.

`G/stale-resting-merge/` preserves both source attempts, build logs, SHA256 protocol,
binaries/config. First attempt accidentally limited moving merges and failed an
existing test; new test also assumed simulator0.105m radius while fixture used
SPL2025's0.05m. Corrected moving behavior and explicit fixture radius; all120 tests
pass, including fresh/stale positive, same exposure, weak, moving and zero-limit
negative cases. Working sources restored after build.

Two-clip diagnostic near-truth duplicate hypothesis-seconds:

| Recording | Baseline | E84 | E89 |
|---|---:|---:|---:|
|144 stationary-close|6.880|24.160|2.280|
|177 long-occlusion|0|9.080|0|

Full216 replay completed, plus E84 runtime with same0.21m parameter as broad control,
under45GiB/nice15 (observed~9.22GiB). Results vs accepted baseline:

| Metric | Conditional E89 | Broad0.21m control |
|---|---:|---:|
| Close position RMSE (m) |0.242464272 (−0.44568%)|0.239268363 (−1.75790%)|
| Close velocity RMSE (m/s) |0.677871787 (−0.01938%)|0.677897957 (−0.01552%)|
| Fast close velocity RMSE (m/s) |2.349776931 (−0.02896%)|same|
| Wrong output delta (s) |−7.962|−8.222|
| Correct-close missing delta (s) |−2.060|−3.430|
| Worst suite close position |+0.12372%development|+8.64018%audit6a|

Raw close velocity missing unchanged. E85's known original raw-velocity correctness
confound remains+0.108775%. Conditional E89's full duplicate metrics are not yet
exported; two-clip cleanup is not full-suite evidence. This experiment supports the
need for conservative conditions but still fails a position guard. Broad control
is rejected despite larger pooled gain. Next trace development position regression
(index27–35) before acceptance; avoid adding more policy merely to fit one case.
All jobs terminal; source restored; no adopted runtime or push. Fresh67/68 unused.

### E90 — Conditional-merge development regression [DIAGNOSED]

`G/stale-resting-merge/development-audit/` contains paired9-clip compact exports,
position-window ranking and first100 differing estimates. Regression is in approach
(index28 globally), first differing output49.532s: same last_seen49.482s, zero
velocity both, position(0.200788,0.022069) baseline versus(0.195616,0.024506) E89,
a≈5.7mm displacement. Truth(0.220862,0.001281). The difference persists into coasting
and already-wrong output:51.0–51.2s position SSE0.119454→0.121068m²s. This is
consistent with covariance-intersection mean perturbation; no instrumented merge
call yet proves it exclusively. E91 tests that mechanism rather than assuming it.

### E91 — Preserve precise recent state in stale resting merges [REJECTED]

E89 plus a valid covariance-intersection endpoint: choose the recent resting
Gaussian unchanged when observation times differ≥1s, recent support≥3, recent size
plausible, and old covariance minus recent covariance is positive definite. Otherwise
retain existing equal-weight covariance intersection. Existing merge eligibility,
observation intervals, max-support policy and evidence-reset behavior are unchanged.
No moving-state fusion change. This avoids shifting/inflating a precise recent state
merely to consolidate a much less certain resting history; it does not prove identity.

`G/stale-resting-endpoint/` freezes all source before/after, config, tested binaries
and SHA256 protocol. Unit test checks both merge orders, exact mean/covariance
preservation, and recent/weak/implausible negative cases; all121 tests pass. Full216
replay completed under45GiB/nice15. Working source restored after build. Original
parameters except E89's0.21m merge maximum; fresh67/68 untouched, no adopted change.

Results essentially unchanged from E89: close position0.242464325m (−0.44565%),
close velocity0.677871787m/s (−0.01938%), fast2.349776931m/s (−0.02896%). Worst
position remains development+0.123765%; wrong−7.962s, correct-close missing−2.060s.
At diagnostic49.532s, output remains exactly(0.195616439,0.024505677), identical to
E89. Thus this endpoint rule does not address E90's observed discrepancy. Reject
added complexity; do not claim the older-less-precise fusion hypothesis proven.
Full-state diagnostic saved at `diagnostic/frames-0.jsonl`. Next instrument the
actual49.532s merge/selection (including auxiliary output) and test whether the
ordinary resting limit0.105m versus baseline0.1m is involved, before another change.
All jobs terminal, memory returned≈4.86GB, no pushes. Goal remains incomplete.

### E92 — Merge trace proves an age-reference mismatch [DIAGNOSED]

`G/merge-shift-trace/` freezes instrumented E89 runtime, exact one-recording score
parity receipt and trace. At49.532s primary filter (blend0.94453126, not auxiliary)
merges resting states using0.21m limit. Fresh state last_seen49.482s, support136.819,
mean(0.2007875,0.0220691), covariance diagonal≈0.373. Old state last_seen48.522s,
support1.9153, mean(0.0911950,0.0736878), covariance diagonal≈7.5323, false size flag.
Their separation≈0.121m exceeds baseline0.1m but meets the conditional diameter gate.
Equal-weight intersection produces E90's shifted output.

E89 tests ages against current time: old1.010s, fresh0.050s, so merging is allowed.
E91 compares observation timestamps: separation0.960s, so its≥1s endpoint condition
is false. This explains exactly why E91 could not fix this frame. Ordinary0.105m
limit and publication switching are ruled out for this event. Repeated trace block
is per-recording/aggregate evaluation, not two physical merges. Sources restored.

### E93 — Align fusion age reference with merge eligibility [DEVELOPMENT ONLY]

E91 endpoint threshold is now880ms = stale minimum1s minus fresh maximum120ms.
This follows the existing eligible interval rather than tuning to49.532s. Other
precision, support, size, covariance agreement/history checks unchanged. Added960ms
positive case in both-order unit test; all121 tests pass. `G/stale-resting-endpoint-aligned/`
preserves source/config/binary SHA256 protocol and build log. Full216 replay and
one-clip diagnostic completed,45GiB cap/nice15 (observed~9.12GB). Working sources
restored after build. E91's original failed variant stays archived.

At49.532s the estimate now equals baseline exactly:(0.200787514,0.022069115),
zero velocity, last_seen49.482s. This controlled change supports E92's diagnosis.
`diagnostic/motivating-frame.json` saves the frame. Full results:
close position0.242459171m (−0.44777%), close velocity0.677871787m/s (−0.01938%),
fast close2.349776931m/s (−0.02896%), all velocity0.780471671m/s (−0.12831%).
Wrong output−7.970s, correct-close missing−2.068s, raw close velocity missing unchanged.
Worst position regression now+0.05898%audit6a; original raw velocity+0.108775%
remains the E85 correctness confound. No fast-close suite regression. This resolves
the traced development failure but not all guards. Full paired duplicate/coverage
review of E93 remains outstanding; do not reuse E84's audit as proof for changed
merge behavior. Next inspect audit6a and quantify the remaining error, then freeze
only a defensible candidate before fresh validation. No fresh67/68 evaluation,
no adopted code/push. All jobs terminal; goal remains incomplete.

### E94 — Remaining position regression is publication gate discontinuity [DIAGNOSED]

`G/stale-resting-endpoint-aligned/audit6a/` contains paired9-clip exports, position
window rankings and failure-window.json. Close-position regression concentrates
in brief-gaps seed50000056,184.8–185.0s:0.162s exposure, SSE0.0343424 baseline
versus0.0376839 candidate (RMSE≈0.4604→0.4823m in that short interval). At184.818s,
selected primary age/mode remain0.216s/resting in both, but published last_seen
changes to183.482s for candidate, while baseline retains184.602s until later.
This points to auxiliary publication rather than primary selection.

Instrumented E93 `G/publication-switch-trace/` proves the branch transition, with
exact per-recording score parity receipt and source/binary preserved. At184.800s,
primary covariance trace3.1888766, auxiliary19.659359, ratio≈6.165>configured6.048:
auxiliary rejected. At184.818s, traces3.4573545 and19.927834, ratio≈5.764<6.048:
auxiliary admitted. The blend immediately becomes0.94453126 instead of0. Thus a
roughly0.36m primary/auxiliary separation produces≈0.34m output change with both
velocities zero. Auxiliary last_seen183.482s, older than primary184.602s. Candidate
primary position(0.299012,−0.342024) differs only submillimetres from baseline;
published candidate(0.157928,−0.032094) is the blended older history.

Source trace matches tracker.rs's binary covariance_supported gate followed by a
fixed publication_filter_blend. Increasing primary process uncertainty alone can
therefore abruptly enable a stale auxiliary estimate. This explains the residual
position regression and is a concrete output-jump mechanism; it does not prove
all observed jumps share this cause. Do not further tune merge distances to conceal
this independent publication discontinuity.

Next test a continuous covariance-dependent blend within the existing admissible
region, preserving the maximum covariance ratio as rejection boundary and literal
zero behavior. Must compare stationary corrections, close velocity and false-output
retention against all216 plus E93; E51's prior publication grids remain relevant.
All diagnostic jobs terminal, all source restored, memory≈4.87GB, no pushes, fresh
67/68 untouched. Goal not complete; high-speed/airborne and full E93 duplicate audit
remain outstanding.

### E95 — Continuous publication covariance blend [FIXED CONFIGURATION REJECTED]

E93 plus a covariance weight inside the existing auxiliary admission boundary.
Let P=primary covariance trace, A=auxiliary trace, C=maximum_ratio*P and
F=min(P,C/2). Weight1 for A≤F; weight(C−A)/(C−F) for F<A≤C; outside C the original
gate rejects the auxiliary. Existing publication_filter_blend multiplies the weight.
Zero resulting blend returns primary with its own timestamp. Both-zero covariance,
zero maximum ratio, nonfinite inputs and infinite cutoff handled explicitly. No
new parameter. This targets E94's covariance discontinuity only; age/distance and
size-based admission still have their original switches.

`G/continuous-publication/` preserves sources, failed/passing build logs, frozen
binaries/config and SHA256 protocol. Added continuity/monotonicity/zero/NaN tests.
Initial run failed an old assertion equating precise-auxiliary output with a much
less precise auxiliary under a finite1e6 cutoff. Under a taper those outputs differ
slightly by design. Replaced that assertion with the explicit configured-blend
expectation for the precise auxiliary; original reject/retain/zero-limit assertions
remain. All122 tests pass. Source restored after build. Full216 replay plus E94
one-clip diagnostic completed under45GiB/nice15, no new parameters or fresh seeds.
Ground-coordinate output displacement184.800→184.818s falls0.339906→0.018651m;
this includes ordinary odometry over18ms, not a pure world-frame discontinuity metric.
`diagnostic/switch.json` preserves both traces. Mechanism works locally but fixed
configuration fails full replay: close position0.252103653m (+3.51220%), worst
suite development+18.67648%; close velocity0.677837464m/s (−0.02444%), fast
2.349709965m/s (−0.03181%), wrong+32.978s, correct-close missing+7.364s. Raw close
velocity missing unchanged. Reject configuration; do not promote smoother output
at the expense of accuracy. Test existing-parameter calibration before discarding
continuous weighting itself.

### E96 — Calibrate existing parameters for continuous weighting [COMPLETE, UNACCEPTED]

Frozen E95 runtime,15 configurations combining maximum covariance ratios8/12/20/50/
1000 and publication blend0.85/original0.94453126/1.0. All other E95 parameters
unchanged. Protocol and scripts in `G/continuous-publication/calibration/`. Purpose:
continuous taper changes useful correction strength, so test the existing controls
rather than adding a transition-width parameter. Full216 inspected development
recordings only; scoring unchanged, no fresh67/68 data. One shared replay process,
15 workers, resource runner/nice15/45GiB. All15 runs terminal, ranked.json preserved;
zero configurations satisfy every original raw position/velocity/fast suite guard.

Two informative candidates (ratio1000):

| Config | Blend | Close p RMSE | Close v RMSE | Fast v RMSE | Wrong delta |
|---|---:|---:|---:|---:|---:|
|grid-012|0.85|0.223471874 (−8.24384%)|0.677020136 (−0.14499%)|2.349989098 (−0.01994%)|−59.980s|
|grid-013|0.94453126|0.222029026 (−8.83626%)|0.677278211 (−0.10693%)|2.350053019 (−0.01722%)|−65.520s|

Both improve position in every suite. Grid012 worst close velocity+0.12318%audit5b,
fast+0.17916%audit5b; correct-close missing−12.130s. Grid013 worst close velocity
+0.16467%audit6b, fast+0.20622%audit5b; correct-close missing−13.594s. E51/E63's
publication-only gains and correctness confounds are relevant controls; do not
attribute all gains to the new taper or association/merge changes. Ratio1000 also
makes taper very weak over most ordinary ratios, so reduced diagnostic jump under
the original ratio does not establish jump reduction for these calibrated configs.

Next prioritize grid012 for fixed common-correct-frame velocity, existing E67 joint
credit and spatial duplicate audit; compare with E93 and the earlier publication
control before freezing a fresh-validation candidate. No adopted runtime change,
source restored, fresh67/68 untouched, no pushes. Goal remains incomplete.

### E97 — Grid012 paired audit and matched publication-only control [COMPLETE]

`G/continuous-publication/calibration/audit-grid012/` freezes216 files and protocol;
candidate chunked export retains hypothesis positions for E63 common-correct-frame,
E67 fixed joint credit and spatial duplicate metrics. Baseline frames reuse E85's
exact accepted-runtime full export. No fresh67/68 data. Resource runner/nice15/45GiB.

Found existing E51 grid007 with the same publication settings: blend0.85,
maximum covariance ratio1000, detection noise0.2 (same f32 as0.20000000298).
Its other parameters/runtime remain baseline. Existing216 results provide control:

| Metric | Publication parameters only | Full E96 grid012 |
|---|---:|---:|
| Close position RMSE (m) |0.223245931|0.223471874 (+0.10121%)|
| Close velocity RMSE (m/s) |0.677140456|0.677020136 (−0.01777%)|
| Fast close velocity RMSE (m/s) |2.350628428|2.349989098 (−0.02720%)|
| Wrong output (s) |986.664|985.162|
| Correct-close missing (s) |94.490|94.794|

Thus most position gain comes from two existing publication parameters; do not
attribute it to the additional runtime changes. The extra velocity gain is small,
and correct-close missing is slightly worse than this simpler control. A second
chunked export uses the baseline evaluator with E51 grid007, to compare exact
correct frames and duplicates. `publication-control.json` preserves summary.
Both216 exports completed. Paired analyses completed successfully under resource
runner (exec61773); memory4.6GiB, nice15, aggregate45GiB cap. Source unchanged.

| Fixed diagnostic | Accepted baseline | Full grid012 |
|---|---:|---:|
| Common-correct close velocity RMSE (m/s), same2392.720s |0.625819627|0.625630901 (−0.03016%)|
| Extra near-truth hypothesis-seconds (0.3m spatial proxy) |693.982|479.302 (−30.9345%)|
| Multiple-near-truth seconds |624.000|447.288|
| All hypothesis-seconds |54014.472|53517.560|
| E67 joint position/velocity mean credit |0.892923874|0.892590140 (worse)|
| E67 fast mean credit |0.546846047|0.547176677 (better)|

Candidate gains15.264s correct-close coverage but loses3.108s elsewhere. Control
has unchanged hypotheses, common-frame velocity0.625911241→0.625874650, gains
15.364s and loses2.906s. Comparing full candidate directly to control on identical
2407.726s: velocity0.625175647→0.625009869 (−0.02652%), correct coverage gains
0.258s/loses0.560s. Thus algorithm changes add a meaningful duplicate reduction,
but tiny velocity improvement and slightly worse correct coverage. Spatial proximity
is not proof of duplicate identity. Do not call candidate accepted: joint credit
worsens and genuine common-correct velocity regressions remain.

Largest common-correct suite regression is audit5b (indices117–125):
0.719462398→0.720498556m/s. `velocity_windows.py` localizes its entire positive SSE
difference to brief-gaps recording121 at190.4–190.6s: SSE1.387486→1.547929.
At190.520s true Field velocity(0.9986,3.5173); baseline published Ground velocity
(0.5208,1.2450), last_seen190.482; candidate(0.0813,0.1944), last_seen190.362.
Both positions remain within0.5m, so this is a real velocity regression, not false
position credit. The publication-only control has nearly identical suite regression,
implicating publication calibration rather than resting association changes; exact
primary/auxiliary trace is the next step before modifying policy.

Evidence: root `common-velocity.json`, `duplicates.json`, `candidate-credit.json`,
`publication_only-credit.json`, `control-audit/`, `incremental-audit/`, and
`velocity-regression-windows.json`. Fresh67/68 remain unscored. No pushes.
Acceptance requires full goal, not merely positive pooled scores.

### E98 — Brief-gap auxiliary velocity trace [COMPLETE]

E97 localizes a genuine velocity regression to recording121 at190.4–190.6s.
Instrument frozen E96 runtime without behavior changes, then verify exact replay
parity with its exported frames before interpreting primary/auxiliary state.
Artifacts `G/brief-gap-publication-trace/`, original sources backed up there; restore
after terminal build. Same grid012 parameters; no new data or parameter search.
Resource runner, nice15,45GiB aggregate,3 build jobs. No agents or pushes.

E98 exact parity passes all7552 frames and every preserved export field. At190.520s
primary is Moving, velocity(0.520755,1.245024), covariance trace0.286004, observed
190.482. Auxiliary is stationary, velocity zero, trace2.357432, observed190.362.
Ratio8.243 exceeds original6.048 guard but is admitted at1000. Its high blend
suppresses velocity while correcting position. All instrumentation restored.

### E99 — Preserve newer moving velocity with older resting position correction [REJECTED]

Isolate one mechanism atop E96 grid012: when primary mode is Moving and auxiliary
mode Resting, with strictly newer primary observation, retain primary velocity;
position blend and conservative timestamp stay unchanged. No new parameter. Equal
or newer auxiliary timestamps retain old behavior. This may expose false primary
velocities, so evaluate all216 and reject if actual velocity/credit regressions
outweigh recovery. Fresh67/68 remain untouched. Fixture covers older/equal/newer
auxiliary, asserting position correction remains. Artifacts
`G/publication-moving-velocity/`, source backups,3-job build/resource runner.

E99 complete:123 tests pass, both binaries built, sources restored byte-for-byte.
All216 replay results and targeted recording121 complete. The190.520s output now
retains primary velocity exactly, with identical blended position/timestamp. Overall
position unchanged from E96 grid0120.223471874m (−8.24384% versus accepted),
wrong time and correct-close missing unchanged. But close velocity worsens to
0.678231953m/s (+0.03374% versus accepted; E96 was0.677020136), worst suite
audit6a+1.08891%. Fast-close velocity improves2.349748850m/s (−0.03016%), but
worst fast suite still+0.04607%audit9a. Do not adopt: simply preserving newer
primary velocity exchanges this gap failure for larger errors elsewhere.
Results `ranked.json`, full `results/candidate.json`, targeted `diagnostic/`.
No live jobs from E98/E99; no fresh validation consumed, no pushes. Next inspect
E99 audit6a primary velocities on position-correct frames to determine whether
motion evidence can distinguish true reacquisition from false moving-primary
corrections. Preserve the simpler-control comparison; no additional complexity
accepted based on a single successfully fixed frame.

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
