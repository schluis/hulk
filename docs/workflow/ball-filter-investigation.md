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
relaxing confirmation eligibility. Preferred parameter values are unchanged.
Resting/moving transition experiments were rejected. Tooling now includes v11
fast-close velocity loss, initial-position-covariance tuning, seeded no-search
demos, and the image observer.

E06 association tracing is complete. E07 tested a full-span motion-significance
check with fixed parameters and was rejected on all three replay partitions.
Runtime code and preferred parameters remain unchanged from `0ed7a0910`.
Goal mode is active at the user’s request. E08 failed fresh validation; E10
retention calibration found a publication dependency (E15). A joint primary/auxiliary
parameter search is being prepared (E16) on all 54 inspected development clips.
Stop only after a candidate beats the frozen current best with improved velocity
and passes fresh validation; do not declare diagnostic progress a completed goal.

Next: tune primary and auxiliary measurement noise together, damping, process
noise, existing selection controls, and optional weak-track retention. Weighted
initialization was tested and rejected (E13); unrestricted auxiliary publication
failed the clear-view-miss test (E15). Require bounded close losses and availability
checks as well as conditional RMSE before freezing another fresh-audit candidate.

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

### 3.2 B: Improve velocity acquisition [NEXT; informed by E06/E07]

#### B1: Two-observation velocity initialization [UNTESTED]

Test whether compatible timestamped observations can initialize velocity with
appropriate uncertainty, leaving isolated detections tentative. Account for
robot motion and observation interval. Check noise amplification at short
intervals and false pairing across obstacles/clutter.

This is **not** the same as the rejected experiment that merely delayed resting
until multiple observations (E02 below). Do not conflate their conclusions.

#### B2: Existing-model velocity correction [UNTESTED diagnosis]

Check whether uncertainty allocation makes updates correct position while leaving
velocity too small. Initial position covariance is now tunable, but its search
with the previous resting experiment did not yield an accepted candidate.
Inspect the actual update before introducing another model or parameter.

### 3.3 C: Improve recovery and matching [PLANNED; depends on A]

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

#### E11 — Expire weak, unsupported hypotheses sooner [RUNNING]

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

#### E14 — Joint parameter search with weak-track retention [RUNNING]

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
centres, and keeps selection uncapped. Seed 2026100525, exact protocol and configs
under `G/joint/refine-*`. Do not promote conditional-error gains caused by loss
of availability. All results use the parity-verified rebuilt binary.

## 5. Evidence map and operational handoff

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
