"""Summarize completed physical-D experiments without fitting or model choice."""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import run_harmonized_refit as run
import pandas as pd

LABELS = {
    "flat_weighted_xgboost": "Flat XGBoost",
    "flat_weighted_lightgbm": "Flat LightGBM",
    "flat_equal_probability_blend": "Flat equal blend",
    "flat_temperature_calibrated_blend": "Flat temperature blend",
    "flat_newton_log_probability_stacker": "Flat Newton stacker",
    "equal_flat_taxonomy_blend": "Real equal blend",
    "equal_flat_cyclic_blend": "Cyclic equal blend",
    "equal_flat_support_matched_blend": "Support-matched equal blend",
    "soft_taxonomy_factorisation": "Soft real taxonomy",
    "hard_top_down_taxonomy": "Hard real taxonomy",
    run.benefit.REAL_ID: "Real benefit gate",
    run.benefit.CYCLIC_ID: "Cyclic benefit gate",
    run.benefit.SUPPORT_ID: "Support-matched benefit gate",
}


def describe_deltas(frame):
    rows = []
    treatment = "equal_flat_taxonomy_blend"
    for control in LABELS:
        if control == treatment or control not in set(frame.run_id):
            continue
        indexed = frame.set_index(["seed", "run_id"])
        for metric in ("f1_macro", "rare_recall", "pr_auc_macro", "accuracy", "ece_10", "rare_ece_10"):
            delta = indexed.xs(treatment, level="run_id")[metric] - indexed.xs(control, level="run_id")[metric]
            rows.append({"treatment": treatment, "control": control, "metric": metric,
                         "seeds": len(delta), "mean_delta": delta.mean(),
                         "sample_sd_delta": delta.std(ddof=1), "minimum_delta": delta.min(),
                         "maximum_delta": delta.max(), "positive_seeds": int((delta > 0).sum()),
                         "zero_seeds": int((delta == 0).sum()), "negative_seeds": int((delta < 0).sum())})
    return pd.DataFrame(rows)


def manuscript_protocol_metadata(protocol, metrics, summary, deltas):
    output = run.WORK / "results" / protocol
    contract = json.loads((output / "run_contract.json").read_text())
    metric_names = [name for name in ("accuracy", "f1_macro", "rare_recall", "pr_auc_macro",
                    "brier_multiclass", "ece_10", "rare_ece_10", "ece_15", "rare_ece_15")
                    if name in metrics.columns]
    ranges = []
    for method, subset in metrics.groupby("run_id", sort=False):
        for metric in metric_names:
            values = subset[metric]
            ranges.append({"run_id": method, "metric": metric, "seeds": len(values),
                           "mean": values.mean(), "sample_sd": values.std(ddof=1),
                           "minimum": values.min(), "maximum": values.max()})
    ranges = pd.DataFrame(ranges)
    run.csv_write(output / "metric_ranges.csv", ranges)
    summary = summary.rename(columns={"ece_mean": "ece_15_mean"})
    run.csv_write(output / "manuscript_summary.csv", summary)
    definitions = {
        "seed": "Holdout seed; 42,43,44,45,46,47; not independent corpus replications",
        "run_id": "Frozen method identifier; no test-based model selection",
        "n_features": "37 shared predictors",
        "n": "Number of held-out outer-test rows",
        "accuracy": "Fraction of outer-test rows classified correctly by argmax",
        "f1_macro": "Macro F1 over the frozen fine-class universe",
        "recall_macro": "Macro recall over the frozen fine-class universe",
        "rare_recall": "Macro recall over fit-pool-defined rare fine classes",
        "worst_recall": "Minimum per-class recall over the frozen fine-class universe",
        "pr_auc_macro": "Macro one-vs-rest average precision, frozen reference implementation",
        "brier_multiclass": "Mean row sum of squared class-probability errors, not divided by class count",
        "ece_10": "Primary top-label ECE; exact Euclid right-closed 10-bin endpoint",
        "rare_ece_10": "Same ECE10 restricted to test rows with a fit-defined rare true label",
        "ece_15": "Audit alias of unchanged original comparator 15-bin top-label ECE",
        "rare_ece_15": "Audit alias of unchanged original 15-bin rare-row ECE",
        "ece_top_label": "Original 15-bin comparator ECE, retained unchanged for audit",
        "rare_ece_top_label": "Original 15-bin comparator rare-row ECE, retained unchanged",
        "training_seconds": "Composed method builder times; do not sum shared models. Missing for seed42 Newton because exact builder time was lost during metadata-only recovery",
        "inference_seconds": "Composed utility+selection+test times, not test-only deployment latency",
    }
    return {"completion": True, "seeds": sorted(int(seed) for seed in metrics.seed.unique()),
            "completed_cells": len(metrics),
            "calibrated_flat_meta_controls_included": protocol == "paired46",
            "calibrated_flat_meta_control_ids": [
                "flat_temperature_calibrated_blend",
                "flat_newton_log_probability_stacker"] if protocol == "paired46" else [],
            "calibrated_flat_meta_control_scope": (
                "Frozen utility-fit flat temperature and Newton controls are present"
                if protocol == "paired46" else
                "Not included in this nine-arm grouping confirmation; cannot exclude these competing explanations to the same extent as paired46"),
            "result_csv": str(output / "harmonized_results_ece10.csv"),
            "original_15bin_result_csv": str(output / "harmonized_results.csv"),
            "metric_schema": [{"field": name, "pandas_dtype": str(metrics[name].dtype),
                               "definition": definitions.get(name, "Frozen reference metric; see protocol and metric source")}
                              for name in metrics.columns],
            "summary_csv": str(output / "manuscript_summary.csv"),
            "delta_csv": str(output / "real_blend_delta_summary.csv"),
            "ranges_csv": str(output / "metric_ranges.csv"),
            "summary": summary.to_dict(orient="records"),
            "deltas_real_equal_blend_minus_control": deltas.to_dict(orient="records"),
            "ranges": ranges.to_dict(orient="records"),
            "frozen_settings": contract["config"],
            "run_contract": str(output / "run_contract.json"),
            "run_contract_sha256": run.sha256(output / "run_contract.json")}


def collect_protocol(protocol):
    output = run.WORK / "results" / protocol
    contract = json.loads((output / "run_contract.json").read_text())
    metrics = pd.read_csv(output / "harmonized_results_ece10.csv", float_precision="round_trip")
    assert sorted(metrics.seed.unique()) == run.SEEDS
    assert len(metrics) == 6 * len(contract["config"]["methods"])
    assert not metrics.duplicated(["seed", "run_id"]).any()
    assert set(metrics[["seed", "run_id"]].itertuples(index=False, name=None)) == {
        (seed, method) for seed in run.SEEDS for method in contract["config"]["methods"]}
    assert metrics[["ece_10", "rare_ece_10"]].notna().all().all()
    summary = pd.read_csv(output / "harmonized_summary.csv").merge(
        pd.read_csv(output / "harmonized_summary_ece10.csv")[["run_id", "ece_10_mean", "ece_10_sd", "rare_ece_10_mean", "rare_ece_10_sd"]],
        on="run_id", validate="one_to_one")
    deltas = describe_deltas(metrics)
    run.csv_write(output / "real_blend_delta_summary.csv", deltas)
    runtimes = []
    roles = []
    checkpoints = []
    for seed in run.SEEDS:
        directory = output / f"seed_{seed}"
        completion = json.loads((directory / "COMPLETED.json").read_text())
        assert completion["run_contract_sha256"] == run.sha256(output / "run_contract.json")
        assert completion["methods"] == contract["config"]["methods"]
        runtime = json.loads((directory / "runtime.json").read_text())
        proof = json.loads((directory / "data_role_proof.json").read_text())
        roles.append({"protocol": protocol, "seed": seed,
                      **{key: proof[key] for key in ("base_pool_rows_before_cap", "base_fit_rows_after_cap",
                                                      "gate_train_rows", "gate_select_rows", "outer_test_rows")},
                      "fit_order_sha256": proof["fit_order_sha256"]})
        elapsed = 0.0
        unknown_builder_times = 0
        for metadata in sorted(directory.glob("*.joblib.json")):
            value = json.loads(metadata.read_text())
            model = metadata.with_name(metadata.name.removesuffix(".json"))
            assert run.sha256(model) == value["sha256"]
            if value["elapsed_seconds"] is None:
                unknown_builder_times += 1
            else:
                elapsed += value["elapsed_seconds"]
            checkpoints.append({"protocol": protocol, "seed": seed, "checkpoint": str(model),
                                "sha256": value["sha256"], "builder_seconds": value["elapsed_seconds"],
                                "builder_timing_status": value.get("duration_status", "Exact measured perf_counter duration"),
                                "builder_seconds_upper_bound": value.get("duration_upper_bound_seconds"),
                                "saved_utc": value["saved_utc"]})
        runtimes.append({"protocol": protocol, "seed": seed,
                         "seed_wall_seconds_this_invocation": runtime["wall_seconds_this_invocation"],
                         "unique_checkpoint_builder_seconds": elapsed if not unknown_builder_times else None,
                         "unique_checkpoint_known_builder_seconds": elapsed,
                         "unknown_checkpoint_builder_durations": unknown_builder_times,
                         "wall_seconds_first_roles_to_completion_including_recovery_gap": runtime.get("wall_seconds_first_roles_to_completion_including_recovery_gap"),
                         "completed_utc": completion["completed_utc"]})
    invocation = json.loads((output / "invocation_runtime.json").read_text())
    return metrics, summary, deltas, roles, runtimes, checkpoints, invocation


def main():
    run.require_physical_d_output(run.WORK)
    verified = json.loads((run.WORK / "verification_report.json").read_text())
    assert {item["protocol"] for item in verified["results"]} == {"paired46", "grouped37"}
    assert all(item["status"] == "COMPLETE_PASS" for item in verified["results"])
    ece_report = json.loads((run.WORK / "ece10_derivation_report.json").read_text())
    assert all(item["status"] == "COMPLETE_PASS" for item in ece_report["results"])
    interruption = json.loads((run.WORK / "recovery" / "external_interruption_20261019T0410.json").read_text())
    for checkpoint in interruption["checkpoint_sha256_verified_before_resume"]:
        assert run.sha256(checkpoint["path"]) == checkpoint["sha256"], checkpoint["path"]
    provenance = json.loads((run.WORK / "runtime_provenance.json").read_text())
    for path, digest in provenance["read_only_reference_sources"].items():
        assert run.sha256(path) == digest, f"Read-only imported source changed: {path}"
    prediction_audit = json.loads((run.WORK / "prediction_execution_audit.json").read_text())
    assert prediction_audit["status"] == "EXACT_PASS"
    assert run.sha256(run.WORK / "prediction_execution.py") == prediction_audit["adapter_sha256"]
    assert all(all(case["exact_array_equality"]) for checkpoint in prediction_audit["checkpoints"]
               for case in checkpoint["cases"])
    representation = json.loads((run.WORK / "representation_verification.json").read_text())
    assert representation["status"] == "PASS"
    assert representation["actual_model_dtype"] == "float32"
    assert representation["raw37_and_actual_clean_float32_partitions_identical"]
    assert representation["actual37_unique_groups"] == 1388399
    worker_records = [json.loads((run.WORK / "dispatch" / f"grouped37_seed_{seed}.json").read_text())
                      for seed in run.SEEDS]
    assert all(record["state"] == "complete" for record in worker_records)
    run.json_write(run.WORK / "all_grouped_worker_execution.json", worker_records)
    collision = json.loads((run.WORK / "projection_collision_audit.json").read_text())
    lines = ["# Harmonized 37-predictor refit: completed experiments", "",
             "Route: EI engineering empirical validation. No manuscript files were edited.", "",
             "## Scope and storage", "",
             f"Authoritative physical work directory: `{run.WORK.resolve()}`.",
             "The old project path is a junction to F. It supplies read-only source and original input files,",
             "not training I/O. The parquet files were copied once, SHA256 verified and marked read-only",
             "under `datasets/source_inputs` on physical D. Both training runs read those D caches.",
             "Python and installed packages are read-only in the original environment. `-B` and",
             "`sys.dont_write_bytecode` disable old-directory bytecode writes. Temp files are on D.", "",
             "The original F-backed run was role-only preflight, not model training. Its artifacts were",
             "copied into `preflight_original_roles_COPY`; the originals were not moved or deleted.", "",
             "## Frozen design", "",
             "Seeds: 42, 43, 44, 45, 46, 47. Outer 20%; development 70% fit pool, 15% utility,",
             "15% selection. Split seeds are s, s+10000, s+20000; cap and shuffle seed is s.",
             "Cap per fine label: 25000; BenignTraffic: 80000. Rare threshold is 1% of uncapped fit pool.",
             "XGBoost 520 and hierarchy/flat LightGBM 380 trees; all model parameters, class weights,",
             "gate grid and tie-breaking are inherited from PRIMARY. The sole constructor override",
             "is n_jobs=2; BLAS/OpenMP limits are also 2. Complete values are in `EXPERIMENT_PROTOCOL.md`,",
             "each `run_contract.json`, and each model checkpoint's parameter sidecar.", "",
             "Temperature calibration and Newton stacking use the second split_development return:",
             "`calibration` in v10 flat-control code, identical to PRIMARY `utility`. They do not fit",
             "on PRIMARY `selection`. Gate regressors fit utility; gate grids use selection. Test is",
             "evaluation only. The flat 2660-tree budget-matched control was not included.", "",
             "## Projection collision audit", "",
             "See `projection_collision_audit.json` and `projection_collision_by_seed.csv` for exact",
             "counts and denominators, including capped-fit/test and all-development/test overlaps.",
             "Fixed-role ablation deliberately retains these collisions to answer the paired question.",
             "The additional grouped37 experiment assigns every actual cleaned float32 37-input hash",
             "to one role. It changes role membership and is not paired to the original46 PRIMARY run.", ""]
    global_collision = collision["global"]
    lines += [f"Total rows: {global_collision['rows']}; original46 groups: {global_collision['original46_unique_groups']};",
              f"shared37 groups: {global_collision['shared37_unique_groups']}. Duplicate excess rows:",
              f"{global_collision['shared37_duplicate_excess_rows']}; rows in any duplicate shared37 group:",
              f"{global_collision['shared37_rows_in_duplicate_groups']}. Newly merged projection groups:",
              f"{global_collision['projection_new_collision_groups']}, containing {global_collision['projection_new_collision_rows']} rows",
              f"({global_collision['projection_new_collision_row_pct']:.6f}% of the full sample). Conflicting-label groups:",
              f"{global_collision['shared37_conflicting_label_groups']}. Rare rows in new collisions:",
              f"{global_collision['rare_rows_in_projection_new_collision_groups']}. Raw37 and float32 partitions agree:",
              f"{global_collision['raw37_and_float32_group_partitions_equal']}.", "",
              "| Seed | Test rows | Test sharing capped fit | Test % | Test sharing any development | Test % | Rare test affected |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for item in collision["per_seed"]:
        lines.append(f"| {item['seed']} | {item['test_rows']} | {item['test_rows_sharing_fit_vector']} | {item['test_rows_sharing_fit_vector_pct']:.6f} | {item['test_rows_sharing_any_development_vector']} | {item['test_rows_sharing_any_development_vector_pct']:.6f} | {item['rare_test_rows_sharing_any_development_vector']} |")
    lines.append("")
    all_metrics, role_rows, runtime_rows, checkpoint_rows, invocation_rows = [], [], [], [], []
    manuscript_protocols = {}
    for protocol in ("paired46", "grouped37"):
        metrics, summary, deltas, roles, runtimes, checkpoints, invocation = collect_protocol(protocol)
        manuscript_protocols[protocol] = manuscript_protocol_metadata(protocol, metrics, summary, deltas)
        metrics.insert(0, "protocol", protocol)
        all_metrics.append(metrics)
        role_rows.extend(roles)
        runtime_rows.extend(runtimes)
        checkpoint_rows.extend(checkpoints)
        invocation_rows.append({"protocol": protocol, **invocation})
        lines += [f"## {protocol} results", "",
                  "Mean +/- sample SD across the six overlapping holdouts; probabilities were saved",
                  "without quantization and all saved metrics were independently recomputed.", "",
                  "| Method | Macro F1 | Rare recall | ECE (10 bins) |", "| --- | ---: | ---: | ---: |"]
        scores = summary.set_index("run_id")
        for method, label in LABELS.items():
            if method not in scores.index:
                continue
            row = scores.loc[method]
            lines.append(f"| {label} | {row.f1_macro_mean:.6f} +/- {row.f1_macro_sd:.6f} | {row.rare_recall_mean:.6f} +/- {row.rare_recall_sd:.6f} | {row.ece_10_mean:.6f} +/- {row.ece_10_sd:.6f} |")
        lines += ["", "Real equal blend minus each control (percentage points):", "",
                  "| Control | Mean Macro F1 delta | Range | Positive / 6 | Mean rare-recall delta |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for control in deltas.control.drop_duplicates():
            f1 = deltas.loc[(deltas.control == control) & (deltas.metric == "f1_macro")].iloc[0]
            rare = deltas.loc[(deltas.control == control) & (deltas.metric == "rare_recall")].iloc[0]
            lines.append(f"| {LABELS[control]} | {100*f1.mean_delta:+.4f} | [{100*f1.minimum_delta:+.4f}, {100*f1.maximum_delta:+.4f}] | {int(f1.positive_seeds)} | {100*rare.mean_delta:+.4f} |")
        lines += ["", f"Actual invocation wall time: {invocation['elapsed_seconds']:.3f} seconds.", ""]
    run.csv_write(run.WORK / "all_protocol_results.csv", pd.concat(all_metrics, ignore_index=True))
    run.csv_write(run.WORK / "all_data_roles.csv", pd.DataFrame(role_rows))
    run.csv_write(run.WORK / "all_seed_runtimes.csv", pd.DataFrame(runtime_rows))
    run.csv_write(run.WORK / "all_checkpoint_manifest.csv", pd.DataFrame(checkpoint_rows))
    run.json_write(run.WORK / "all_invocation_runtimes.json", invocation_rows)
    lines += ["## Verification and interpretation limits", "",
              "Primary reporting ECE uses 10 equal-width bins with the exact Euclid endpoint:",
              "`clip(digitize(confidence, linspace(0,1,11), right=True)-1,0,9)` and each bin's",
              "`mask.mean()` as weight, after the same comparator normalization. Both `ece_10` and",
              "`rare_ece_10` are explicit fields. Original 15-bin fields/files are retained unchanged,",
              "with `ece_15` and `rare_ece_15` aliases in derived tables. No model was refitted or",
              "reselected for this metric convention correction. Bin counts, interval cross-checks",
              "and prediction digests are in `ece10_*` files and `ece10_derivation_report.json`.", "",
              "`verification_report.json` independently checks complete partitions, role order hashes,",
              "caps, rare IDs, PRIMARY test row/label equality, saved probabilities, estimator worker",
              "limits, checkpoint SHA256s and metric recomputation (absolute tolerance 1e-12).",
              "All old imported reference-source hashes were rechecked during report generation.", "",
              "The grouping representation is exactly the classifiers' cleaned float32 input:",
              "infinities become missing, missing values become zero, then values are cast to float32.",
              "A full streamed check of both D caches found all 37 source fields already float32,",
              "zero nonfinite values and zero changed values on conversion. Raw and cleaned",
              "partitions are identical, with 1,388,399 groups; see `representation_verification.json`.", "",
              "`all_seed_runtimes.csv` separates seed wall time from the sum of unique checkpoint",
              "builder times; shared models must not be counted repeatedly across methods. Inference",
              "seconds in method CSVs cover utility+selection+test prediction stages, not test-only",
              "benchmarks. Concurrent work and the two-worker cap preclude historical timing comparisons.", "",
              "A label-free 8193-row inference execution audit verified bitwise equality for explicit",
              "two-thread calls and 8192-row batching, without changing estimator parameters or",
              "checkpoint bytes. No reliable large speedup was established, so neither production",
              "run installs the batching prototype. Future independent grouped workers release",
              "unused full-data setup tables before fitting; per-worker source hashes and flags are",
              "preserved in `all_grouped_worker_execution.json`.", "",
              "Seed42 Newton converged and was atomically serialized before a generic n_jobs audit",
              "raised: the frozen LogisticRegression uses n_jobs=None, which has no effect in",
              "installed sklearn 1.9; BLAS/OpenMP remain limited to 2. Recovery validated the",
              "unchanged model (11 iterations), copied it for audit and added its missing metadata,",
              "without refitting or changing coefficients. Exact builder time was lost and remains",
              "null, not reconstructed. Its log-to-artifact wall upper bound is 592.068014 seconds.",
              "The Newton method's seed42 training_seconds is therefore missing; prediction metrics",
              "are complete. Invocation wall intervals include the recovery gap and explicitly",
              "exclude startup preceding the first original role event. Recovery preflight also",
              "detected original-source hash drift, while both D caches still match frozen hashes;",
              "only those unchanged D caches were used, never re-copied from the changed source.", "",
              "At 04:10 local time, the previous paired/grouped processes and their coordinators",
              "were no longer present; available logs contained no model exception. The external",
              "cause is unknown. Fifty completed checkpoints were hash-verified before continuation",
              "and again here. Saved models/probabilities were reused; interrupted unsaved inference",
              "was recomputed without changing rows, precision or estimators. Wall intervals include",
              "this interruption and queue waits; exact active durations of interrupted attempts are",
              "unknown. Summed completed-worker invocation times exclude those interrupted attempts.", "",
              "These are repeated holdouts from one public subsample, not six independent corpus",
              "replications. Deltas, sample SDs and directions are descriptive; no independence-based",
              "p-values or confidence intervals are claimed. Grouped37 changes the grouping unit and",
              "must be kept distinct from the PRIMARY-paired feature ablation. Neither experiment",
              "identifies why discovery and archive corpora differ. All arms and seeds are retained.", "",
              "Grouped37 is the frozen nine-arm core confirmation, not the thirteen-arm paired46",
              "experiment. It includes taxonomy benefit gates but not the flat temperature-calibrated",
              "blend or Newton log-probability stacker. Consequently, grouped37 cannot exclude",
              "those calibrated flat meta-control explanations to the same extent as paired46.", "",
              "No abandoned harmonised_features_v1 results were used as evidence.", ""]
    (run.WORK / "EXPERIMENT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    run.json_write(run.WORK / "DELIVERY_MANIFEST.json", {
        "status": "COMPLETE_VERIFIED", "seeds_per_protocol": run.SEEDS,
        "metric_rows": sum(len(item) for item in all_metrics),
        "authoritative_physical_directory": str(run.WORK.resolve()),
        "verified_reference_source_count": len(provenance["read_only_reference_sources"]),
        "report_sha256": run.sha256(run.WORK / "EXPERIMENT_REPORT.md"),
        "all_results_sha256": run.sha256(run.WORK / "all_protocol_results.csv"),
        "verification_report_sha256": run.sha256(run.WORK / "verification_report.json")})
    # This completion-only handoff must never be emitted for partial experiments.
    assert sum(item["completed_cells"] for item in manuscript_protocols.values()) == 132
    assert all(item["seeds"] == run.SEEDS for item in manuscript_protocols.values())
    run.json_write(run.WORK / "MANUSCRIPT_HANDOFF.json", {
        "completion": True, "status": "COMPLETE_VERIFIED",
        "physical_work_directory": str(run.WORK.resolve()),
        "protocols": manuscript_protocols,
        "verification": {"report": str(run.WORK / "verification_report.json"),
                         "sha256": run.sha256(run.WORK / "verification_report.json"),
                         "results": verified["results"],
                         "ece10_report": str(run.WORK / "ece10_derivation_report.json"),
                         "ece10_results": ece_report["results"],
                         "grouped_actual_input_hash_role_overlap_csv": str(run.WORK / "results" / "grouped37" / "independent_actual37_role_overlap.csv"),
                         "model_input_representation_report": str(run.WORK / "representation_verification.json"),
                         "model_input_dtype": representation["actual_model_dtype"],
                         "raw_and_clean_group_partitions_identical": representation["raw37_and_actual_clean_float32_partitions_identical"],
                         "read_only_reference_sources_rechecked": len(provenance["read_only_reference_sources"])},
        "frozen_settings": run.CONFIG,
        "full_frozen_settings_sources": [str(run.WORK / "EXPERIMENT_PROTOCOL.md"),
                                         str(run.WORK / "all_checkpoint_manifest.csv")],
        "operational_resource_policy": str(run.WORK / "resource_policy.json"),
        "inference_execution_audit": {"report": str(run.WORK / "prediction_execution_audit.json"),
                                      "status": prediction_audit["status"],
                                      "production_batching_installed": False,
                                      "production_live_processes_modified": False,
                                      "worker_execution_records": str(run.WORK / "all_grouped_worker_execution.json")},
        "ece10_endpoint": {"bins": 10, "equal_width": True, "right_closed": True,
                           "formula": "clip(digitize(confidence, linspace(0,1,11), right=True)-1,0,9)",
                           "bin_weight": "mask.mean()", "normalization": "frozen v10 comparator.normalise",
                           "original_15bin_outputs_preserved": True, "model_reselection": False},
        "data_role_proof_csv": str(run.WORK / "all_data_roles.csv"),
        "collision_report": str(run.WORK / "projection_collision_audit.json"),
        "runtime_csv": str(run.WORK / "all_seed_runtimes.csv"),
        "execution_recovery": {"audit": str(run.WORK / "recovery" / "seed42_stacker_recovery.json"),
                               "external_interruption_audit": str(run.WORK / "recovery" / "external_interruption_20261019T0410.json"),
                               "pre_interruption_checkpoints_reverified_unchanged": len(interruption["checkpoint_sha256_verified_before_resume"]),
                               "cache_preflight": str(run.WORK / "recovery" / "cache_only_resume_preflight.json"),
                               "no_stacker_refit": True, "scientific_parameters_changed": False,
                               "seed42_exact_stacker_builder_seconds": None,
                               "seed42_stacker_wall_upper_bound_seconds": 592.068014},
        "known_measurement_limits": [
            "No actual edge device measurement or deployment claim",
            "Method inference times cover utility+selection+test, not test-only latency",
            "Concurrent jobs and n_jobs=2 preclude direct historical timing comparisons",
            "Seed42 Newton exact builder duration is unavailable after metadata-only recovery; its training_seconds is missing, while all prediction metrics remain complete",
            "External worker loss observed at 04:10 has unknown cause/duration; unchanged checkpoints were reused, unsaved inference recomputed, and wall intervals retain interruption gaps",
            "Original training-source hash drifted after the one-time copy; all fits use unchanged, frozen-SHA256 verified D caches",
            "Six overlapping holdouts from one subsample are descriptive, not independent replications",
            "Fixed-role paired37 retains measured projection collisions; grouped37 is a separate grouping confirmation",
            "Grouped37 changes row roles and is not paired to the original 46-feature PRIMARY results",
            "Grouped37 has nine core arms including taxonomy gates, but no flat temperature or Newton meta controls; it cannot equally exclude those competing explanations",
            "No causal explanation of cross-corpus performance differences is identified",
            "No independent-holdout p-values or confidence intervals are inferred from these six means",
            "The 2660-tree budget-matched flat control is not part of this refit"],
        "manuscript_edited": False, "abandoned_harmonised_results_used": False})
    print(json.dumps({"status": "COMPLETE_VERIFIED", "result_rows": sum(len(item) for item in all_metrics),
                      "report": str(run.WORK / "EXPERIMENT_REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()
