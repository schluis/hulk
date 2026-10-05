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
The retained baseline runtime and parameters are those of `0ed7a0910`. Runtime code and preferred parameter defaults remain unchanged.
Goal mode is active at the user’s request. E08 failed fresh validation; E10
retention calibration found a publication dependency (E15). A joint primary/auxiliary
parameter search completed (E16) on all 54 inspected development clips. The
parameter-only ablation-0260 candidate failed the second audit’s per-suite
close-position guard. E18 completed the stationary-noise calibration; E19
rejected local-0280 on a fresh coasting-position regression. E20 produced
local-0322; E21 rejected it on fresh suite and fast-velocity regressions.
Stop only after a candidate beats the frozen current best with improved velocity
and passes fresh validation; do not declare diagnostic progress a completed goal.

Next: E34 improve velocity response after E33 numeric pass but unresolved reversal lag. Runtime/default
parameters remain at the original retained baseline until fresh validation succeeds.


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
