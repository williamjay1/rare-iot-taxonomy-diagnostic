# Harmonized 37-predictor refit: completed experiments

Route: EI engineering empirical validation. No manuscript files were edited.

## Scope and storage

Authoritative physical work directory: `<WORKSPACE>\harmonized`.
The old project path is a junction to F. It supplies read-only source and original input files,
not training I/O. The parquet files were copied once, SHA256 verified and marked read-only
under `datasets/source_inputs` on physical D. Both training runs read those D caches.
Python and installed packages are read-only in the original environment. `-B` and
`sys.dont_write_bytecode` disable old-directory bytecode writes. Temp files are on D.

The original F-backed run was role-only preflight, not model training. Its artifacts were
copied into `preflight_original_roles_COPY`; the originals were not moved or deleted.

## Frozen design

Seeds: 42, 43, 44, 45, 46, 47. Outer 20%; development 70% fit pool, 15% utility,
15% selection. Split seeds are s, s+10000, s+20000; cap and shuffle seed is s.
Cap per fine label: 25000; BenignTraffic: 80000. Rare threshold is 1% of uncapped fit pool.
XGBoost 520 and hierarchy/flat LightGBM 380 trees; all model parameters, class weights,
gate grid and tie-breaking are inherited from PRIMARY. The sole constructor override
is n_jobs=2; BLAS/OpenMP limits are also 2. Complete values are in `EXPERIMENT_PROTOCOL.md`,
each `run_contract.json`, and each model checkpoint's parameter sidecar.

Temperature calibration and Newton stacking use the second split_development return:
`calibration` in v10 flat-control code, identical to PRIMARY `utility`. They do not fit
on PRIMARY `selection`. Gate regressors fit utility; gate grids use selection. Test is
evaluation only. The flat 2660-tree budget-matched control was not included.

## Projection collision audit

See `projection_collision_audit.json` and `projection_collision_by_seed.csv` for exact
counts and denominators, including capped-fit/test and all-development/test overlaps.
Fixed-role ablation deliberately retains these collisions to answer the paired question.
The additional grouped37 experiment assigns every actual cleaned float32 37-input hash
to one role. It changes role membership and is not paired to the original46 PRIMARY run.

Total rows: 1429753; original46 groups: 1390064;
shared37 groups: 1388399. Duplicate excess rows:
41354; rows in any duplicate shared37 group:
69096. Newly merged projection groups:
851, containing 2735 rows
(0.191292% of the full sample). Conflicting-label groups:
0. Rare rows in new collisions:
0. Raw37 and float32 partitions agree:
True.

| Seed | Test rows | Test sharing capped fit | Test % | Test sharing any development | Test % | Rare test affected |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 285884 | 374 | 0.130822 | 477 | 0.166851 | 0 |
| 43 | 286156 | 412 | 0.143977 | 496 | 0.173332 | 0 |
| 44 | 286178 | 397 | 0.138725 | 513 | 0.179259 | 0 |
| 45 | 285899 | 424 | 0.148304 | 510 | 0.178385 | 0 |
| 46 | 285971 | 385 | 0.134629 | 471 | 0.164702 | 0 |
| 47 | 285977 | 393 | 0.137424 | 501 | 0.175189 | 0 |

## paired46 results

Mean +/- sample SD across the six overlapping holdouts; probabilities were saved
without quantization and all saved metrics were independently recomputed.

| Method | Macro F1 | Rare recall | ECE (10 bins) |
| --- | ---: | ---: | ---: |
| Flat XGBoost | 0.853258 +/- 0.001174 | 0.487510 +/- 0.005215 | 0.004516 +/- 0.000347 |
| Flat LightGBM | 0.843496 +/- 0.001881 | 0.444943 +/- 0.007350 | 0.002090 +/- 0.000164 |
| Flat equal blend | 0.855787 +/- 0.000936 | 0.484294 +/- 0.003032 | 0.005093 +/- 0.000217 |
| Flat temperature blend | 0.855936 +/- 0.000980 | 0.485087 +/- 0.003574 | 0.004935 +/- 0.000287 |
| Flat Newton stacker | 0.858874 +/- 0.002318 | 0.469437 +/- 0.009100 | 0.001775 +/- 0.000296 |
| Real equal blend | 0.881692 +/- 0.001331 | 0.591590 +/- 0.005161 | 0.007201 +/- 0.000132 |
| Cyclic equal blend | 0.867492 +/- 0.002190 | 0.526531 +/- 0.007956 | 0.006257 +/- 0.000492 |
| Support-matched equal blend | 0.861775 +/- 0.001343 | 0.496212 +/- 0.004984 | 0.006626 +/- 0.000387 |
| Soft real taxonomy | 0.872810 +/- 0.001653 | 0.577837 +/- 0.006126 | 0.001446 +/- 0.000392 |
| Hard real taxonomy | 0.867357 +/- 0.001539 | 0.579724 +/- 0.006217 | 0.039827 +/- 0.000287 |
| Real benefit gate | 0.881645 +/- 0.001277 | 0.594141 +/- 0.007361 | 0.009010 +/- 0.001108 |
| Cyclic benefit gate | 0.868787 +/- 0.001547 | 0.535309 +/- 0.003769 | 0.007738 +/- 0.000949 |
| Support-matched benefit gate | 0.862753 +/- 0.001683 | 0.505333 +/- 0.005370 | 0.007889 +/- 0.000263 |

Real equal blend minus each control (percentage points):

| Control | Mean Macro F1 delta | Range | Positive / 6 | Mean rare-recall delta |
| --- | ---: | ---: | ---: | ---: |
| Flat XGBoost | +2.8434 | [+2.6562, +3.0548] | 6 | +10.4080 |
| Flat LightGBM | +3.8196 | [+3.4348, +4.0672] | 6 | +14.6647 |
| Flat equal blend | +2.5904 | [+2.4973, +2.6475] | 6 | +10.7296 |
| Flat temperature blend | +2.5756 | [+2.4874, +2.6804] | 6 | +10.6502 |
| Flat Newton stacker | +2.2818 | [+2.0086, +2.6580] | 6 | +12.2152 |
| Cyclic equal blend | +1.4200 | [+1.2592, +1.5462] | 6 | +6.5059 |
| Support-matched equal blend | +1.9917 | [+1.8037, +2.2083] | 6 | +9.5377 |
| Soft real taxonomy | +0.8882 | [+0.7062, +1.0964] | 6 | +1.3753 |
| Hard real taxonomy | +1.4334 | [+1.2777, +1.5996] | 6 | +1.1865 |
| Real benefit gate | +0.0047 | [-0.0290, +0.0829] | 3 | -0.2551 |
| Cyclic benefit gate | +1.2905 | [+1.2433, +1.3248] | 6 | +5.6281 |
| Support-matched benefit gate | +1.8939 | [+1.7354, +1.9970] | 6 | +8.6257 |

Actual invocation wall time: 38096.745 seconds.

## grouped37 results

Mean +/- sample SD across the six overlapping holdouts; probabilities were saved
without quantization and all saved metrics were independently recomputed.

| Method | Macro F1 | Rare recall | ECE (10 bins) |
| --- | ---: | ---: | ---: |
| Flat XGBoost | 0.853386 +/- 0.001571 | 0.487484 +/- 0.008455 | 0.004468 +/- 0.000427 |
| Real equal blend | 0.882402 +/- 0.002619 | 0.595036 +/- 0.010433 | 0.007044 +/- 0.000487 |
| Cyclic equal blend | 0.867699 +/- 0.001117 | 0.527242 +/- 0.005848 | 0.005943 +/- 0.000357 |
| Support-matched equal blend | 0.862272 +/- 0.001684 | 0.497243 +/- 0.006609 | 0.006524 +/- 0.000301 |
| Soft real taxonomy | 0.873401 +/- 0.003144 | 0.580272 +/- 0.012338 | 0.001599 +/- 0.000353 |
| Hard real taxonomy | 0.867805 +/- 0.002928 | 0.581038 +/- 0.012258 | 0.039761 +/- 0.000437 |
| Real benefit gate | 0.882301 +/- 0.002327 | 0.598763 +/- 0.011230 | 0.008428 +/- 0.000988 |
| Cyclic benefit gate | 0.869242 +/- 0.001214 | 0.540260 +/- 0.003793 | 0.007178 +/- 0.001106 |
| Support-matched benefit gate | 0.863307 +/- 0.001793 | 0.506054 +/- 0.008763 | 0.007839 +/- 0.000348 |

Real equal blend minus each control (percentage points):

| Control | Mean Macro F1 delta | Range | Positive / 6 | Mean rare-recall delta |
| --- | ---: | ---: | ---: | ---: |
| Flat XGBoost | +2.9016 | [+2.5019, +3.2439] | 6 | +10.7553 |
| Cyclic equal blend | +1.4703 | [+1.1998, +1.7057] | 6 | +6.7794 |
| Support-matched equal blend | +2.0130 | [+1.5180, +2.4203] | 6 | +9.7793 |
| Soft real taxonomy | +0.9001 | [+0.5690, +1.1639] | 6 | +1.4764 |
| Hard real taxonomy | +1.4596 | [+1.1770, +1.6973] | 6 | +1.3999 |
| Real benefit gate | +0.0101 | [-0.0534, +0.1139] | 3 | -0.3726 |
| Cyclic benefit gate | +1.3160 | [+0.9945, +1.6606] | 6 | +5.4777 |
| Support-matched benefit gate | +1.9095 | [+1.4178, +2.2467] | 6 | +8.8983 |

Actual invocation wall time: 22703.734 seconds.

## Verification and interpretation limits

Primary reporting ECE uses 10 equal-width bins with the exact Euclid endpoint:
`clip(digitize(confidence, linspace(0,1,11), right=True)-1,0,9)` and each bin's
`mask.mean()` as weight, after the same comparator normalization. Both `ece_10` and
`rare_ece_10` are explicit fields. Original 15-bin fields/files are retained unchanged,
with `ece_15` and `rare_ece_15` aliases in derived tables. No model was refitted or
reselected for this metric convention correction. Bin counts, interval cross-checks
and prediction digests are in `ece10_*` files and `ece10_derivation_report.json`.

`verification_report.json` independently checks complete partitions, role order hashes,
caps, rare IDs, PRIMARY test row/label equality, saved probabilities, estimator worker
limits, checkpoint SHA256s and metric recomputation (absolute tolerance 1e-12).
All old imported reference-source hashes were rechecked during report generation.

The grouping representation is exactly the classifiers' cleaned float32 input:
infinities become missing, missing values become zero, then values are cast to float32.
A full streamed check of both D caches found all 37 source fields already float32,
zero nonfinite values and zero changed values on conversion. Raw and cleaned
partitions are identical, with 1,388,399 groups; see `representation_verification.json`.

`all_seed_runtimes.csv` separates seed wall time from the sum of unique checkpoint
builder times; shared models must not be counted repeatedly across methods. Inference
seconds in method CSVs cover utility+selection+test prediction stages, not test-only
benchmarks. Concurrent work and the two-worker cap preclude historical timing comparisons.

A label-free 8193-row inference execution audit verified bitwise equality for explicit
two-thread calls and 8192-row batching, without changing estimator parameters or
checkpoint bytes. No reliable large speedup was established, so neither production
run installs the batching prototype. Future independent grouped workers release
unused full-data setup tables before fitting; per-worker source hashes and flags are
preserved in `all_grouped_worker_execution.json`.

Seed42 Newton converged and was atomically serialized before a generic n_jobs audit
raised: the frozen LogisticRegression uses n_jobs=None, which has no effect in
installed sklearn 1.9; BLAS/OpenMP remain limited to 2. Recovery validated the
unchanged model (11 iterations), copied it for audit and added its missing metadata,
without refitting or changing coefficients. Exact builder time was lost and remains
null, not reconstructed. Its log-to-artifact wall upper bound is 592.068014 seconds.
The Newton method's seed42 training_seconds is therefore missing; prediction metrics
are complete. Invocation wall intervals include the recovery gap and explicitly
exclude startup preceding the first original role event. Recovery preflight also
detected original-source hash drift, while both D caches still match frozen hashes;
only those unchanged D caches were used, never re-copied from the changed source.

At 04:10 local time, the previous paired/grouped processes and their coordinators
were no longer present; available logs contained no model exception. The external
cause is unknown. Fifty completed checkpoints were hash-verified before continuation
and again here. Saved models/probabilities were reused; interrupted unsaved inference
was recomputed without changing rows, precision or estimators. Wall intervals include
this interruption and queue waits; exact active durations of interrupted attempts are
unknown. Summed completed-worker invocation times exclude those interrupted attempts.

These are repeated holdouts from one public subsample, not six independent corpus
replications. Deltas, sample SDs and directions are descriptive; no independence-based
p-values or confidence intervals are claimed. Grouped37 changes the grouping unit and
must be kept distinct from the PRIMARY-paired feature ablation. Neither experiment
identifies why discovery and archive corpora differ. All arms and seeds are retained.

Grouped37 is the frozen nine-arm core confirmation, not the thirteen-arm paired46
experiment. It includes taxonomy benefit gates but not the flat temperature-calibrated
blend or Newton log-probability stacker. Consequently, grouped37 cannot exclude
those calibrated flat meta-control explanations to the same extent as paired46.

No abandoned harmonised_features_v1 results were used as evidence.
