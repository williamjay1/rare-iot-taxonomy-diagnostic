# Standards-Grounded Mechanism Control, v15

Scope: a retrospective control on the existing CICIoT2023 public development subsample. This is not an expert-approved taxonomy, independent confirmation dataset, real timestamp experiment, or physical-edge experiment. No manuscript edits are made here.

## Frozen Mechanism Mapping

`mechanism_spec.json` declares the grouping before new outer predictions. `frozen/standard_mapping.csv` expands it to all 34 canonical labels, with official CAPEC identifiers, definition names, URLs, version, match strength, ambiguity, rationale, and limitations. All 33 attack labels have references, but this is reference coverage, not 33 exact annotations. Seven labels have broad/partial/conditional evidence. Three web-compromise labels share a clearly marked unresolved fallback bucket. Related conditional CAPEC subtypes are not asserted as ground truth.

The 12 groups separate TCP, UDP, ICMP, fragmentation, HTTP resource exhaustion, other flooding, active probing, input injection, unresolved web compromise, credential guessing, traffic redirection, and benign traffic. This differs from v10's five broad tactic/proxy groups. The partition is researcher-constructed using official definitions; CAPEC does not publish this CICIoT mapping.

Official CAPEC 3.9 XML and supporting official CIC/ATT&CK HTML snapshots are preserved under `F:\AcademicData\iot_ids_sci\raw\standards_sources_20261018_v1`. `source_manifest.json` records URLs, retrieval timestamps, sizes, and SHA-256. The design lock records their hashes and the code/spec hashes. Source snapshots are marked read-only.

## Roles And Learners

- Seeds: 42, 43, 44, 45, 46, 47. Same original 46 predictors and order.
- Exact-predictor-hash grouped stratified outer split: 20% held out. Development: 70% base, 15% utility, 15% selection using original seed offsets and functions. No hash group crosses roles. These are not chronological splits.
- Fit cap: 25,000 per attack class, 80,000 benign, original order and sampling implementation. Rare labels come from the uncapped base role, as in the core.
- New hierarchy: original family/local LightGBM factories, 380 estimators, learning rate 0.055, 127 leaves, min child 25, subsample 0.9, column subsample 0.95, lambda 1.5, alpha 0.02. Family/local inverse-frequency exponents remain 0.25/0.50. Only thread count changes to `n_jobs=2` for both new arms.
- Standard and wrong-standard blends: fixed `0.5 * core_flat_probability + 0.5 * new_hierarchy_probability`. No gate, calibration, parameter search, or test-based selection. Utility/selection remain unused by the new fixed blends.
- Canonical input parquet and original core predictions are copied once to physical D cache and SHA-256 checked. Training reads only D caches. Imported project helpers are read-only; bytecode writes are disabled.
- Outer test is reordered to canonical NPZ `row_id`; row set, prediction order, target labels, and rare IDs must match exactly. Core-flat hard predictions must reproduce canonical classification metrics to absolute tolerance 1e-12.

## Wrong Control: Exact Size And Support

For each seed, construct the wrong map using capped-fit label counts only, before new outer prediction access. Integer assignment constraints preserve exactly (1) the number of fine labels per named group and (2) total capped-fit rows per group. Benign remains in its singleton group, and no attack can enter it. Exact matching is asserted with integer counts after solving.

The objective is lexicographic: first minimize original within-group attack-label pairs that remain together, then minimize retained named attack memberships. This prevents a mere permutation of group names from being accepted as a wrong taxonomy. Each original pair has objective weight 35, exceeding all 33 possible attack-membership penalties. Solver: HiGHS through SciPy MILP, two threads, 120-second budget, relative gap target zero; the returned exact-support assignment is frozen, with solver status/gap recorded. If optimality is not proved, the valid incumbent is reported as such, not called maximally wrong. A solution with an unchanged partition is rejected.

Support matching means **capped-fit row support**, not test support, uncapped base support, or identical local per-class count vectors. Full zero-retention is not imposed: unique small singleton supports can force retention. The report must show retained labels, retained co-pairs, and partition ARI. Both arms have identical group cardinalities, local-model counts, tree hyperparameters, class-weight rules, and fit data. Actual model sizes and runtimes need not be equal.

## Metrics, Precision, And Interpretation

Primary: macro-F1 of standardblend versus wrongstandardblend. Secondary: accuracy, macro-recall, rare recall, worst recall, macro one-vs-rest average precision, and all per-class support/precision/recall/F1. Also compare to reused coreflat. Negative or null contrasts are retained.

The canonical flat cache stores probabilities as float16 and separately stores original hard predictions. `coreflat` uses original hard predictions for classification metrics and cached probabilities for PR-AUC; `coreflat_cached_probability` exposes the quantized argmax sensitivity. Both blends use the identical unrenormalized promoted cached flat probabilities, matching v10's equal-blend construction. No flat refit or invented full-precision recovery is claimed. New blend metrics are computed in memory; saved float32 probabilities and original computed hard predictions are retained.

Six overlapping grouped splits are summarized descriptively with mean, sample SD, minimum, maximum, and paired seed differences. They are not six independent populations; no t-test, independence-based confidence interval, or preregistration claim is made.

Cost is new hierarchy training/inference/serialization, serialized bytes, process peak RSS, and total runner wall time. Flat cost is excluded because predictions are reused. These CPU measurements are not physical-edge latency, end-to-end feature extraction throughput, or energy measurements.

## Execution

Run `run_standards_control.py self-test`, then `prepare`, then `run` with the project's existing Python environment and `-B`. Preparation freezes all six wrong maps before evaluating new outer predictions. `verify` checks code, specification, frozen role/map artifacts, canonical caches, and fit caches. All new scripts, cache, frozen artifacts, logs, and results stay under this physical D `standards` directory.

Results directory: `results/ijismd_v15_standards_20261018`. A complete status requires all six paired prediction files, all twelve newly fitted hierarchy arms, coverage, cost, role/order/support audits, and successful post-run verification. A script or prepared map alone does not count as an executed experiment.
