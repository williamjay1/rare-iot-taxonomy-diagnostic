# Shared predictor refit protocol

## External process-loss continuation (2026-10-19 04:10 local)

A fresh Win32_Process inventory found no Python workers or previous paired /
grouped coordinators; old exec session IDs were unavailable. Available model
logs had no exception. The cause is unknown, not attributed to deadlock or an
intentional model stop. At detection, paired seeds42/43 had completed 26 cells
plus the seed44 flat result (27 total), and grouped seed42 had completed 9 cells
plus five seed43 cells (14 total). Fifty completed checkpoint digests were
verified before continuation, recorded in
`recovery/external_interruption_20261019T0410.json`.

The same resume wrapper and resource queue were continued from immutable D
caches/checkpoints. No completed model or saved probability was refitted.
Unsaved in-flight probability computation must be repeated. Seeds, estimators,
rows, float64 output precision, two-thread limits and the 8 GiB launch gate are
unchanged. The queue now tracks the new paired launcher, rather than stale PIDs.
Prior grouped attempt metadata is preserved separately. Actual wall intervals
include interruptions and resource waits; exact active times of interrupted
attempts are unavailable and are not invented or silently added to completed
invocation durations. Final reporting rechecks all 50 pre-interruption digests.

## Resource-gated continuation amendment (2026-10-19)

The original paired process (launcher 21448, model child 36896) remains running,
without code or parameter changes. Its runner SHA256 remains
086cce012ba141739a0307dcd5fda7330b07ec59ac47c9f9533b941cf7c25698.
Only the idle sequential continuation waiter was replaced. The replacement
`continue_resource_queue.ps1` may run at most two independent grouped37 seed
workers. The initial gate was 10 GiB free RAM and at most four actual Python
runtimes. At 00:37 the idle coordinator was updated to read `resource_policy.json`:
at least 8 GiB free RAM and at most five actual Python runtimes before launch,
at most two grouped workers. The paired estimator process was not restarted.
Each estimator still uses n_jobs=2 and
BLAS/OpenMP=2. At launch this amendment started no additional training worker:
seven actual Python runtimes and only about 5 GiB free RAM were observed.

`run_grouped37_seed_worker.py` calls the unchanged grouped37 setup, role and fit
functions. It acquires a per-seed Windows exclusive byte-range lock, uses its
own seed log and runtime, and never writes shared summaries. Shared aggregation
runs once, after all workers exit. The finalizer distinguishes the actual
earliest-worker-start to last-worker-finish wall interval from summed worker
CPU/wall durations and aggregation-only overhead. The original preflight
contracts, estimators, rows, precision and calibration permissions are unchanged.

The completion-only `MANUSCRIPT_HANDOFF.json` is created by the report builder
only after all 78 paired46 and 54 grouped37 cells, completion markers, independent
verification and ECE10 derivation have passed. No partial handoff is emitted.

The paired46 protocol has 13 arms per seed; grouped37 has the frozen 9 core arms
per seed, including taxonomy benefit gates. Grouped37 does not include the flat
temperature-calibrated blend or Newton log-probability stacker. Its confirmation
therefore cannot exclude those calibrated flat meta-control explanations to the
same extent as paired46. No arms are added to either protocol after inspection.

The 2048-row layout-only prediction probe returned bitwise identical float64
probabilities for F- and C-contiguous float32 inputs, but did not establish a
speedup (F: 3.20/4.31 s; C: 3.87/3.89 s under concurrent load). No production
layout change was made. Repeated feature cleaning took about 0.012 s per 200k
rows; real-hierarchy checkpoint non-builder overhead was bounded at about
8.60 s versus 917.92 s within the prediction builder. Neither supports treating
cleaning or checkpoint serialization as the main prediction bottleneck.

An additional 8192-row read-only batch probe returned bitwise identical outputs
for one full batch, 2048-row chunks and 512-row chunks: 20.5545, 20.2531 and
18.8867 s respectively, with two threads and identical estimator checkpoint.
Concurrent machine load changed during this small probe, so this is not proof
of a reproducible throughput improvement and no production change was made.
The real/cyclic/support maps have no identical local label sets in seed42;
cross-map family-model prediction reuse cannot be assumed mathematically valid.
Per-map predictions are already reused across all applicable output arms.

At 00:54:08 local time, during flat LightGBM prediction, a read-only Windows
performance-counter sample measured 193% process CPU (about two cores), 0.794
GiB resident memory, 3 system hard page reads/second and 0 process I/O bytes/s.
This short observation supports a CPU/tree-traversal bottleneck, not a claim
that paging was absent throughout the entire prediction interval. The saved
flat model has n_jobs=2, Booster num_threads=2 and 12,920 trees (380 rounds
times 34 classes). Its complete three-role prediction took 1353.17 seconds.

The additional 8193-row, label-free `prediction_execution_audit.json` checked
the flat model and real hierarchy, including both fine and family outputs.
Explicit num_threads=2 and 8192-row chunks both produced bitwise identical
float64 arrays (maximum difference zero), with unchanged checkpoint hashes
and estimator parameters. Loaded threadpool libraries were also limited to 2.
Flat timings were 21.448/20.799/21.672/21.673 seconds for reference, explicit
threads, batching and repeated reference; hierarchy timings were
18.452/17.871/18.045/18.204 seconds. These do not establish a reliable large
speedup. `prediction_execution.py` remains a tested prototype, NOT installed
in either production run. The original vectorized predictions already use two
threads; no timing improvement is credited in the ETA.

The first grouped worker started at 01:11:44 local time only after the queue
observed 8.04 GiB free RAM. It uses the original worker implementation. For
future single-seed workers, an execution-only memory amendment deletes the
unused full input table and group representative table after role preparation
and before fitting, then collects garbage and rechecks the capped fit row-order
hash. The role frames, features, encoder, models and probabilities are retained
unchanged. Each worker state records its script hash and whether these setup
tables were released. No running paired or grouped seed42 process was modified.

`representation_verification.json` records a full 1,429,753-row streamed audit
of both D parquet caches. All 37 Arrow fields have type float (float32), and
all corresponding pandas dtypes are float32. There are zero nonfinite values
and zero values changed by the frozen cleaning/cast. Hashes were recomputed
from DataFrame(clean_features(...)) with index=False. Raw37 and actual cleaned
float32 partitions are identical, with 1,388,399 unique groups. Therefore no
additional representation protocol or new cutpoints are required. The existing
independent grouped-role audit also uses the actual cleaned float32 values.

### Metadata-only recovery of paired seed42

The Newton fit completed and its model was atomically serialized at
2026-10-18T17:15:23.094608Z. The original process then raised in the generic
model-parameter metadata audit, which incorrectly required every n_jobs field
to equal 2. Frozen sklearn LogisticRegression has n_jobs=None; installed 1.9
documents this field as having no effect. Its numerical libraries remain
limited to two threads. No convergence warning occurred; the recovered model
has 11 iterations, 68 inputs and 34 classes. Its SHA256 is
b51dbc941b9af56ccb0f8d28dbc2687485a6dc93846c41f01b21df18d19bc17e.

`resume_paired_execution.py` preserves both frozen runner files and all model
parameters, accepts only the exact reference Newton constructor parameters,
and keeps the two-thread assertion for tree estimators. The serialized stacker
was validated, copied (not moved) to recovery/ and reused without refitting.
Its exact perf_counter duration was lost at the assertion, so checkpoint
elapsed_seconds is null and its method training_seconds is missing. The
592.068014-second preceding-log-to-artifact interval is only a wall upper
bound, not a substituted exact fit duration. Other timings remain measured.
The final runtime records distinguish resumed invocation time from the wall
interval spanning the first original role event to completion, including the
recovery gap and excluding startup before that event.

The original training-source parquet currently hashes to e0061e1f80952abe7d3628986e33723afca50a8ff800babfccfebfd549c82afe,
whereas the frozen D training cache still exactly hashes to
bcc477f125aa7cecbeba50cd3b9641e20dcf521e882c437a704f73c30c96b353.
The test source/cache retain their original frozen hash. No source was changed
by this task. Resume verifies contract metadata, source-code hashes and both
unchanged D cache hashes, and reconstructs precisely the same model roles from
those D caches; it never recopies the changed original training source.
The source drift is recorded in recovery/cache_only_resume_preflight.json.
Seed42 completed at 17:27:10Z and seed43 began at 17:27:30Z.

The original resource queue retains ownership of grouped dispatch. A separate
`finish_after_recovery.ps1` waits for the resumed paired launcher AND that
queue to exit before any fallback final aggregation/audit. It does not stop or
restart training workers. This handles the original queue's obsolete paired
PID without racing its shared-summary writer. Complete markers for all twelve
seed runs remain mandatory before a handoff can be created.

Route: EI engineering empirical validation, as clarified by the user. The present task is a reviewer
requested representation sensitivity experiment, not manuscript revision.
Primary standards: identical data roles, fair model refits, group leakage audit,
unchanged model parameters, held out evaluation, and reproducible outputs.
Do not apply social science causal identification requirements.
Indexing is not a method standard. Workflow: applicable engineering validation
modules, without social science causal requirements or generic top-journal gates.

## Question and boundary

Does the discovery corpus taxonomy advantage survive removing the nine
mirror-only predictors? Input: existing public subsample parquet files, verified
against the v14 supplementary SHA256 values. All 37 shared names come from
results/ijismd_revision_feature_overlap_v1/feature_overlap_report.json.
No file in the abandoned harmonised_features_v1 directory is used or read.
No archive data are retrained. This ablation cannot establish why discovery and
archive results differ, because source, sampling and measurement differ too.
No directional outcome is assumed before fitting.

## Frozen comparison

The reference is results/ijismd_v10_ciciot_grouped_core_6split, the PRIMARY
six-holdout pipeline. Seeds: 42, 43, 44, 45, 46, 47. Original 46-predictor exact
groups determine the outer 20 percent test role and the remaining development
roles (70 percent fit pool, 15 percent utility, 15 percent selection). RNG seeds
are seed, seed+10000, seed+20000 respectively. Cap sampling and shuffle use seed.
Per-label caps are 25000, except BenignTraffic at 80000. Rare classes are computed
from the uncapped fit pool at the original 1 percent threshold. Support matched
maps use only capped fit counts; each is checked against the PRIMARY map.
All original role counts, test row order, test labels and rare IDs are checked
against the PRIMARY ledgers. Ordered row IDs are saved for all four roles and
the capped fit set. Full-46 group overlaps must be zero.

An unchanged-role feature ablation need not remain group-disjoint after a
projection. Raw shared-37 and cleaned float32 shared-37 group overlaps are
therefore separately reported, never silently removed or regrouped. Such a
regrouping would change the question and cease to be the requested paired ablation.

XGBoost: 520 rounds, depth 9, learning rate .055, min child weight 2, row fraction
.9, column fraction .95, lambda 1.5, alpha .02, histogram trees, multiclass
soft probabilities, mlogloss. Flat weights have frequency exponent .35.
LightGBM: 380 rounds, learning rate .055, 127 leaves, unlimited depth, minimum
child samples 25, row fraction .9, column fraction .95, lambda 1.5, alpha .02.
Family weights use exponent .25; within-family weights .50; flat LightGBM .35.
Model RNG is the holdout seed. Constructor parameters are inherited unchanged
from the reference module except n_jobs=2. BLAS and OpenMP threads are also 2.
Missing and infinite features are converted to zero, float32, exactly as PRIMARY.
No preprocessing statistics are learned from utility, selection or test rows.

## Arms and permissions

| Arms | Fit | Utility | Selection | Test |
| --- | --- | --- | --- | --- |
| Weighted flat XGBoost and LightGBM | capped fit | none | none | evaluation |
| Real, cyclic, support matched hierarchies | capped fit | none | none | evaluation |
| Equal flat/hierarchy blends | inherited | none | none; fixed .5 | evaluation |
| Soft and hard real taxonomy | inherited | none | none | evaluation |
| Benefit gates, all three maps | inherited | utility model | fixed grid | evaluation |
| Equal flat XGBoost/LightGBM blend | inherited | none | none; fixed .5 | evaluation |
| Temperature calibrated flat blend | inherited | two temperatures | none | evaluation |
| Newton flat log probability stacker | inherited | meta model | none | evaluation |

The reference utility regressor, objective, power/threshold/beta grid and
tie-breaking are reused without rewriting. Objective is Macro F1 + .04 rare
recall; powers .75/1; thresholds -.5/-.25/0/.1/.2/.3/.4/.5; betas .25/.5/.75/1.
Utility regressor: 150 iterations, rate .05, 15 leaves, minimum leaf 50, L2 .5,
first four diagnostics categorical, no early stopping, RNG=seed. Utility weights
are clipped square roots of maximum fit-pool support divided by class support.
Temperature scaling bounds .25 to 5, bounded log-temperature optimization,
xatol=.0001. Stacker: C=1, Newton Cholesky, maximum 100 iterations; convergence
warnings are fatal. These match the existing flat-control scripts.
The 2660-round tree-budget matched flat LightGBM is outside this scoped run.

Reference permissions were checked against executed v10 scripts and metadata,
not inferred from prose. In run_v10_grouped_flat_comparators.py, the second
return of split_development is named calibration; temperatures and the lbfgs
stacker fit on that role (lines 305-371). The third role is named selection and
is used only for diagnostics. In run_v10_grouped_newton_flat_stacker.py the same
second return supplies the converged Newton stacker (lines 124-166). This is
exactly the PRIMARY utility role, not the PRIMARY selection role. The present
code retains that permission and records it in the run contract. The PRIMARY
benefit gate uses utility for the regressor and selection for its fixed grid.
The abandoned reduced-feature script is not the reference and is not evidence.

## Additional projection-grouped confirmation

The user requires a second core confirmation if projection collisions exist.
Keep the unchanged PRIMARY-role ablation; separately construct all four roles
using exact hashes of the actual cleaned float32 37-variable model input.
Use the same six seeds, fractions, caps, tree parameters and .5 blends. Retain
every observed fine label. If a projected group has conflicting labels, choose
its modal fine label only as a group-stratification key, resolving ties by
lexicographic label order; do not relabel, discard, or split its member rows.
This rule is fixed before model outcomes are available. Save both conflicting
group statistics and all group-disjointness checks. The initial minimum request
listed four core arms: weighted flat, real equal blend, cyclic equal blend and
support matched equal blend. Before grouped model fitting, the executable
contract froze nine arms, retaining those four and adding the real soft/hard
factorisations and the three corresponding taxonomy benefit gates:

1. `flat_weighted_xgboost`
2. `equal_flat_taxonomy_blend`
3. `equal_flat_cyclic_blend`
4. `equal_flat_support_matched_blend`
5. `soft_taxonomy_factorisation`
6. `hard_top_down_taxonomy`
7. `calibrated_taxonomy_benefit_gate`
8. `calibrated_cyclic_taxonomy_control`
9. `calibrated_support_matched_control`

This is 9 arms x 6 seeds = 54 cells, not a later expansion based on results.
The separate paired46 protocol has 13 arms x 6 seeds = 78 cells. Grouped37 does
not include the paired protocol's flat LightGBM, equal flat ensemble, flat
temperature ensemble or Newton stacker. In particular, no claim that grouped37
equally excludes calibrated flat meta-control explanations is justified.
Recompute rare IDs from the new uncapped fit pool and support maps from the new
capped fit set. This is a separate protocol, not a like-for-like paired
comparison against original-46 PRIMARY scores.

## Physical storage correction and authorization

On inspection, <LEGACY_PROJECT> is a junction to
F:/AcademicData/D_Project_Migration_20261001/codex/sci/iot_ids_sci.
Both authorized directories are physically on F, despite their D path spelling.
The initial six-seed role-only preflight completed before this was identified;
its free-space log incorrectly called the F volume D. No model was fitted.
The runner refuses all model and preflight output writes unless the output
directory and its temporary directory resolve to physical D. Merely removing
Path.resolve would conceal this problem, not solve it. Creating a new physical
D work directory was subsequently explicitly authorized by the user:
<WORKSPACE>/harmonized. <WORKSPACE_ROOT> and its project directory
were verified to be ordinary directories, not reparse points; Path.resolve
returns D. Available physical D storage was 213416935424 bytes at validation.

The authoritative scripts now live in that physical D work directory. PRIMARY
code is imported read-only with -B and sys.dont_write_bytecode. Original parquet
files are hashed read-only, copied once into datasets/source_inputs, hash-checked
again, and marked read-only. All later parquet reads use the D cache. All models,
predictions, logs, temporary files and result tables go under results/paired46
or results/grouped37 in the authorized physical D directory. The initial role
preflight artifacts were copied, not moved or deleted, into
preflight_original_roles_COPY. No training checkpoint existed on the old volume.

## Execution and analysis

Existing ML environment: <LEGACY_PROJECT>/.venv/Scripts/python.exe.
Its Python and six recorded package versions must match the PRIMARY metadata.
All authoritative writing is restricted to this physical D harmonized directory;
the original dedicated result directory is F-backed and is no longer written.
Inputs are read-only; no downloads,
publishing, other workers, agents, or manuscript changes.
Disk preflight requires at least 20 GiB free on D. CPU only, 2 workers per
estimator. Six sequential seeds, each with one global model, three hierarchies,
three gates, one flat LightGBM, two temperatures, and one converged stacker.
Runtime will be measured, not claimed from historical workstation benchmarks.

Models and role probabilities are independently resumable, with atomic saves,
SHA256 validation, and a source/input/config contract. Method-level outputs use
unquantized float64 probabilities, allowing exact metric recomputation. A run
with mismatching sources, features, versions, inputs or configuration is rejected.

Analysis: per-seed and mean/sample-SD Macro F1, rare recall, accuracy, PR AUC,
Brier score, and 15-bin top-label ECE; per-class scores; real-blend paired deltas
against flat and wrong maps and calibrated flat controls; within-seed 37 versus
PRIMARY-46 comparisons. No choosing the best arm or seed on outer outcomes.
Repeated holdouts overlap and are not independent corpus replications. Means,
SDs, ranges and paired directions are descriptive; no independence-based p
values or confidence intervals are introduced by this refit.

### ECE reporting correction requested during execution

The frozen training runner and all original 15-bin metric files remain unchanged.
The final main table uses additional `ece_10` and `rare_ece_10` fields derived
from exactly the same saved float64 predictions, without refitting or choosing
any model. The implementation exactly matches Euclid's frozen extension:
`np.clip(np.digitize(confidence,np.linspace(0,1,11),right=True)-1,0,9)`.
Each nonempty bin is weighted by `mask.mean()`. This is right-closed binning,
with exact zero clamped into the first bin and exact one included in the last;
calling the original comparator with bins=10 would not match its boundaries.
Probability normalization matches Euclid aligned_probability (the same v10
comparator normalise). Original 15-bin columns remain available and gain explicit
`ece_15`/`rare_ece_15` aliases only in the new derived tables. The standalone
derive_ece10.py records prediction digests, unchanged original metric digests,
and bin counts, and independently checks the interval formulation at 1e-15.
The derived key set is checked against the complete frozen seed x method
Cartesian product: 78 paired46 and 54 grouped37 rows, with no duplicate keys,
missing cells or nonfinite explicit ECE10 fields at finalization.
The user confirmed that no real edge device is available; no hardware-deployment
or real-device latency claim will be made from these workstation experiments.
