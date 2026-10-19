"""Bounded, no-refit capture audit and fail-closed publication correction."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
import run_capture_controls as capture
import run_matched_extensions as run

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from threadpoolctl import threadpool_limits

import run_sequential_window_drift_study as historical_source

FRESH = ("flat_xgb", "flat_lgbm", "flat_equal_blend",
         "temperature_flat_blend", "newton_flat_stacker")
ARCHIVED = {
    "legacy_flat_xgb": "archived_flat_xgb",
    "taxonomy_blend": "archived_taxonomy_control",
    "cyclic_blend": "archived_cyclic_control",
    "support_matched_blend": "archived_support_control",
}
REASON = (
    "Original window archives contain hard predictions only. No original flat or "
    "hierarchy probability arrays or serialized estimators were found in the "
    "audited archive locations. Fresh flat predictions disagree with the old "
    "flat component. A common-flat taxonomy/control blend cannot be computed "
    "without unavailable components or a prohibited hierarchy refit."
)
VERSION = "capture_component_scope_v1"


def contract(path):
    frame = pd.read_csv(path)
    return {"path": str(path), "rows": len(frame), "sha256": run.file_hash(path),
            "schema": [{"column": name, "dtype": str(frame[name].dtype)} for name in frame]}


def split_results(frame):
    if set(frame.method) != set(FRESH) | set(ARCHIVED):
        raise ValueError("Unexpected capture methods: refusing an implicit rename")
    if frame.duplicated(["window", "method"]).any():
        raise ValueError("Duplicate capture method/window")
    counts = frame.groupby("window").size()
    if set(counts.index) != set(capture.WINDOWS) or not counts.eq(9).all():
        raise ValueError("All four complete windows are required")
    fresh = frame.loc[frame.method.isin(FRESH)].copy()
    archived = frame.loc[frame.method.isin(ARCHIVED)].copy()
    archived["original_extension_method"] = archived.method
    archived["original_archive_key"] = archived.method.map(capture.FROZEN_METHODS)
    archived["method"] = archived.method.map(ARCHIVED)
    archived["publication_scope"] = "audit_only_not_common_fitted_flat"
    return fresh, archived


def retained_records(values):
    return [row for row in values if row.get("endpoint") != "capture_windows"]


def audit_inputs(folder, protocol, historical):
    encoder = LabelEncoder().fit(protocol["fine_labels"])
    role_rows = []
    for role in ("base", "gate_train", "gate_select"):
        path = folder / "datasets" / f"{role}.parquet"
        source = Path(historical["samples_dir"]) / path.name
        frame = pd.read_parquet(path)
        y = encoder.transform(frame.Label.astype(str))
        X = run.trf.clean_features(frame, protocol["feature_columns"])
        source_hash, cache_hash = run.file_hash(source), run.file_hash(path)
        if source_hash != cache_hash:
            raise ValueError(f"Source/cache bytes differ: {role}")
        sequences = sorted(int(v) for v in frame._source_sequence.unique())
        expected = {"base": list(range(1, 46)), "gate_train": list(range(46, 51)),
                    "gate_select": list(range(51, 55))}[role]
        if sequences != expected or len(frame) != protocol["role_rows"][role]:
            raise ValueError(f"Role row permission mismatch: {role}")
        row = {"role": role, "rows": len(frame), "sequences": sequences,
               "current_source": str(source), "physical_d_cache": str(path),
               "current_source_sha256": source_hash, "cache_sha256": cache_hash,
               "ordered_clean_float32_X_sha256": run.array_hash(X),
               "ordered_encoded_y_sha256": run.array_hash(y),
               "ordered_source_rows_sha256": run.array_hash(
                   frame[["_source_sequence", "_source_row"]].to_numpy(dtype=np.int64)),
               "ordered_feature_hash_sha256": run.array_hash(frame._feature_hash.to_numpy()),
               "class_counts": np.bincount(y, minlength=len(encoder.classes_)).tolist(),
               "historical_as_fitted_row_fingerprint_available": False}
        if role == "base":
            weights = run.trf.inverse_frequency_weights(y, 0.35)
            row.update({"ordered_float32_weights_sha256": run.array_hash(weights),
                        "weight_exponent": 0.35, "weight_normalization": "divide by row mean"})
        role_rows.append(row)
    return role_rows


def archive_inventory(root):
    locations = [run.ROOT / "results" / name for name in (
        "jit_sequential_window_drift_study_v2", "jit_sequential_window_drift_study",
        "jit_ciciot2023_official_archive_confirmation")]
    rows, probability_candidates, models = [], [], []
    model_extensions = {".joblib", ".pkl", ".pickle", ".ubj", ".model", ".bin"}
    for location in locations:
        for path in sorted(location.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix in model_extensions:
                models.append(str(path))
            item = {"path": str(path), "bytes": path.stat().st_size,
                    "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()}
            if path.suffix == ".npz":
                item["sha256"] = run.file_hash(path)
                with np.load(path, allow_pickle=False) as data:
                    item["arrays"] = []
                    for key in data.files:
                        a = data[key]
                        item["arrays"].append({"key": key, "shape": list(a.shape), "dtype": str(a.dtype)})
                        if a.ndim == 2 and a.shape[1] == 34 and a.shape[0] > 1000:
                            probability_candidates.append({"path": str(path), "key": key})
            rows.append(item)
    # Search model suffixes in the old project too, excluding interpreter fixtures.
    command = ["rg", "--files", "--hidden", "-g", "*.joblib", "-g", "*.pkl",
               "-g", "*.pickle", "-g", "*.ubj", "-g", "*.model", "-g", "*.bin",
               "-g", "!**/.venv/**", "-g", "!**/site-packages/**", str(run.ROOT)]
    search = subprocess.run(command, capture_output=True, text=True, timeout=180, check=False)
    if search.returncode not in (0, 1):
        raise RuntimeError(f"Bounded model search failed: {search.stderr}")
    candidates = search.stdout.splitlines()
    excluded = [path for path in candidates if Path(path).suffix == ".bin"
                and Path(path).name.startswith("_nop_") and Path(path).parent.name == "cache"]
    project_models = [path for path in candidates if path not in excluded]
    run.write_json(root / "archive_component_inventory.json", {
        "locations": [str(p) for p in locations], "files": rows,
        "serialized_models_in_archive_locations": models,
        "possible_34class_probability_matrices": probability_candidates,
        "additional_project_model_search_command": command,
        "additional_project_model_candidates": project_models,
        "excluded_unrelated_candidates": [{"path": path, "sha256": run.file_hash(Path(path)),
                                            "reason": "LibreOffice formula OpenCL kernel profile cache, not a fitted estimator"}
                                           for path in excluded],
        "scope": "Local project and named original archive locations only; not a disk-wide absence proof"})
    if models or probability_candidates or project_models:
        raise RuntimeError("An original component candidate exists; inspect it before publishing a blocked result")
    return locations


def model_audit(folder, protocol):
    current = joblib.load(folder / "xgb.joblib")
    run.trf.RANDOM_STATE = capture.SEED
    nominal = run._global_factory(len(protocol["fine_labels"]), 520)
    actual = {key: repr(value) for key, value in current.get_params().items()}
    expected = {key: repr(value) for key, value in nominal.get_params().items()}
    differences = {key: {"fresh_fitted": actual.get(key), "current_historical_factory": expected.get(key)}
                   for key in actual.keys() | expected.keys() if actual.get(key) != expected.get(key)}
    if set(differences) != {"n_jobs"} or current.n_features_in_ != 39 or current.n_jobs != 2:
        raise ValueError(f"Unexplained current factory discrepancy: {differences}")
    first = next(iter(capture.WINDOWS))
    part = sorted((folder / first).glob("part_*.parquet"))[0]
    frame = pd.read_parquet(part).iloc[:1024]
    probability = current.predict_proba(run.trf.clean_features(frame, protocol["feature_columns"]))
    with np.load(part.with_suffix(".npz")) as saved:
        np.testing.assert_array_equal(probability.argmax(axis=1), saved["flat_xgb"][:len(frame)])
    np.testing.assert_allclose(probability.sum(axis=1), 1, rtol=0, atol=2e-6)
    return {"current_model": str(folder / "xgb.joblib"),
            "current_model_sha256": run.file_hash(folder / "xgb.joblib"),
            "fresh_fitted_parameters": actual, "current_historical_factory_parameters": expected,
            "only_observed_constructor_difference": differences,
            "same_fit_function_object": historical_source.nested.fit_weighted_xgb is run.nested.fit_weighted_xgb,
            "current_model_classes": current.classes_.tolist(),
            "booster_config": json.loads(current.get_booster().save_config()),
            "fresh_cache_reproduced_on_first_1024_rows": True,
            "historical_serialized_model_available": False,
            "causal_diagnosis": "Unresolved. n_jobs is a documented difference, not an established cause.",
            "no_extra_fit": True}


def publish(root, correction, diagnostic):
    old_path = correction / "original_MANUSCRIPT_HANDOFF.json"
    if not old_path.exists():
        shutil.copyfile(root / "MANUSCRIPT_HANDOFF.json", old_path)
    original = json.loads(old_path.read_text())
    raw = pd.read_csv(root / "capture_windows_results.csv")
    fresh, archived = split_results(raw)
    run.write_csv(correction / "capture_fresh_flat_results.csv", fresh)
    run.write_csv(correction / "capture_original_hybrid_audit.csv", archived)
    registry = []
    for method in FRESH:
        registry.append({"original_extension_method": method, "corrected_method": method,
                         "archive_key": "not_used", "actual_source_operation": "fresh_flat_control",
                         "official_fresh_flat_table_eligible": True, "common_flat_taxonomy_claim_eligible": False})
    for method, name in ARCHIVED.items():
        operation = "archived_flat_argmax" if method == "legacy_flat_xgb" else "source_fixed_half_flat_plus_half_hierarchy_not_gate"
        registry.append({"original_extension_method": method, "corrected_method": name,
                         "archive_key": capture.FROZEN_METHODS[method], "actual_source_operation": operation,
                         "official_fresh_flat_table_eligible": False, "common_flat_taxonomy_claim_eligible": False})
    run.write_csv(correction / "capture_method_registry.csv", pd.DataFrame(registry))
    run.write_csv(correction / "common_flat_repair_coverage.csv", pd.DataFrame([
        {"window": window, "computed_original_case_complete": True, "common_flat_repair_complete": False,
         "fresh_methods_complete": 5, "required_reblends_unavailable": 3, "incomplete_reason": REASON}
        for window in capture.WINDOWS]))
    coverage = pd.read_csv(root / "endpoint_coverage.csv").fillna("")
    coverage["computed_original_units"] = coverage.completed_units
    mask = coverage.endpoint.eq("capture_windows")
    coverage.loc[mask, "completed_units"] = 0
    coverage.loc[mask, "strong_control_evaluations_completed"] = 0
    coverage.loc[mask, "status"] = "blocked_common_component"
    coverage.loc[mask, "incomplete_reason"] = REASON
    coverage.loc[mask, "result_csv"] = str(correction / "capture_fresh_flat_results.csv")
    run.write_csv(correction / "endpoint_coverage.csv", coverage)
    safe_names = ("all_endpoint_summary.csv", "all_endpoint_per_seed_paired_results.csv",
                  "all_endpoint_control_comparisons.csv", "best_competitor_by_endpoint_metric.csv")
    for name in safe_names:
        frame = pd.read_csv(root / name)
        if name == "all_endpoint_summary.csv":
            keep = ~frame.endpoint.eq("capture_windows") | frame.method.isin(FRESH)
            frame = frame.loc[keep].copy()
            mask = frame.endpoint.eq("capture_windows")
            frame.loc[mask, "facet"] = "fresh_flat_only:" + frame.loc[mask, "facet"].astype(str)
        else:
            frame = frame.loc[~frame.endpoint.eq("capture_windows")].copy()
        run.write_csv(correction / name, frame)
    comparisons = []
    for window, group in fresh.groupby("window"):
        index = group.set_index("method")
        for method in FRESH:
            if method == "flat_xgb":
                continue
            for metric in ("macro_f1", "rare_recall"):
                comparisons.append({"endpoint": "capture_windows", "window": window, "seed": capture.SEED,
                                    "method": method, "reference": "flat_xgb", "metric": metric,
                                    "method_value": float(index.loc[method, metric]),
                                    "reference_value": float(index.loc["flat_xgb", metric]),
                                    "method_minus_reference": float(index.loc[method, metric] - index.loc["flat_xgb", metric]),
                                    "scope": "fresh_flat_only_no_taxonomy_inference"})
    run.write_csv(correction / "capture_fresh_flat_paired_results.csv", pd.DataFrame(comparisons))
    summaries = []
    for method, group in fresh.groupby("method"):
        for metric in ("macro_f1", "rare_recall"):
            summaries.append({"method": method, "metric": metric, "mean": float(group[metric].mean()),
                              "sample_sd_across_windows": float(group[metric].std(ddof=1)),
                              "n_windows": 4, "n_fitted_seeds": 1,
                              "inference": "descriptive variation across four dependent windows; no confidence interval or independent-replicate test"})
    run.write_csv(correction / "capture_fresh_flat_window_summary.csv", pd.DataFrame(summaries))
    handoff = {"completion": False, "bounded_diagnostic_complete": True, "version": VERSION,
               "original_computed_units": 55, "expected_matched_units": 55, "accepted_full_matched_units": 51,
               "common_flat_capture_windows_complete": 0, "common_flat_capture_windows_expected": 4,
               "fresh_flat_only_windows_complete": 4, "original_hybrid_windows_preserved": 4,
               "incomplete_reason": REASON, "diagnostic": str(correction / "diagnostic.json"),
               "original_handoff_preserved": str(old_path), "original_handoff_sha256": run.file_hash(old_path),
               "csv_contracts": {p.name: contract(p) for p in sorted(correction.glob("*.csv"))},
               "formal_capture_table": str(correction / "capture_fresh_flat_results.csv"),
               "formal_capture_methods": list(FRESH),
               "original_hybrid_audit_only": str(correction / "capture_original_hybrid_audit.csv"),
               "no_common_component_taxonomy_or_best_competitor_capture_claim": True,
               "naming_resolution": diagnostic["naming_resolution"],
               "review_facing_prose": [
                   "Four capture-order windows were evaluated, but the archived hierarchy composites and fresh flat controls do not share an identical fitted flat component. The common-component comparison is therefore withheld, not declared matched or complete.",
                   "Only the five internally consistent fresh flat arms are eligible for the revised capture table. The three archived hierarchy composites and archived flat remain separately available for audit, without cross-generation rankings.",
                   "Capture-window source code uses fixed equal mixtures for cyclic and support controls despite gate-like legacy identifiers. No capture gate result is established by those identifiers.",
                   "The cause of the approximately 2.3 percent old-versus-new flat hard-label disagreement is unresolved. Current nominal inputs, labels, weights and model settings match apart from thread count, but no historical as-fitted row/model fingerprint survives to establish component identity.",
                   "The four windows are dependent file-order proxies from one fitted seed, not four independent training replicates or timestamp-based temporal validation. No real edge hardware was used."],
               "repair_requirements": ["Locate original serialized estimators or independent original hierarchy probabilities with valid row/class/fit provenance", "Otherwise, a separately authorized full matched hierarchy refit would be required; none was performed"],
               "not_performed": ["hierarchy refitting", "flat refitting", "2*oldblend-oldflat reconstruction", "test-based tuning", "window selection"]}
    run.write_json(root / "CAPTURE_CORRECTION_HANDOFF.json", handoff)
    updated = copy.deepcopy(original)
    updated.update({"completion": False, "computed_completion": True,
                    "completed_endpoint_categories": 4, "completed_units": 51,
                    "computed_units": 55, "strong_control_unit_evaluations": 102,
                    "status": "capture_common_component_repair_blocked",
                    "correction_handoff": str(root / "CAPTURE_CORRECTION_HANDOFF.json"),
                    "original_handoff_preserved": str(old_path),
                    "capture_comparisons_by_window": [],
                    "capture_flat_reproduction_audit": diagnostic})
    updated["endpoint_method_rows"] = 1071
    updated["additional_fresh_flat_only_capture_rows"] = 20
    updated["computed_original_endpoint_method_rows"] = original["endpoint_method_rows"]
    for row in updated["coverage"]:
        if row["endpoint"] == "capture_windows":
            row.update({"completed_units": 0, "computed_original_units": 4,
                        "strong_control_evaluations_completed": 0, "status": "blocked_common_component",
                        "incomplete_reason": REASON, "result_csv": handoff["formal_capture_table"],
                        "result_scope": "five fresh flat methods only; requested taxonomy comparisons unavailable"})
    for key in ("best_competitor_by_endpoint_metric", "reversals_control_better"):
        updated[key] = retained_records(updated[key])
    updated["all_endpoint_summary"] = json.loads(pd.read_csv(
        correction / "all_endpoint_summary.csv").to_json(orient="records", double_precision=15))
    updated["csv_contracts"] = {name: value for name, value in updated["csv_contracts"].items()
                                if "capture" not in name and name not in (*safe_names, "endpoint_coverage.csv")}
    updated["csv_contracts"].update(handoff["csv_contracts"])
    updated["review_facing_prose"] = [p for p in original["review_facing_prose"] if "Capture" not in p]
    updated["review_facing_prose"].extend(handoff["review_facing_prose"])
    updated["frozen_protocol_and_class_universes"]["capture_method_ids"] = list(FRESH)
    updated["frozen_protocol_and_class_universes"]["capture_legacy_method_aliases"] = ARCHIVED
    updated["protocol_boundaries"]["capture_common_component_comparison_withheld"] = True
    updated["verification_scope"] = "Original numeric and source verification is preserved; it does not certify the now-withheld common-flat capture comparisons"
    run.write_json(root / "MANUSCRIPT_HANDOFF.json", updated)
    status_path = root / "CURRENT_RUN_STATUS.json"
    if status_path.exists() and not (correction / "original_CURRENT_RUN_STATUS.json").exists():
        shutil.copyfile(status_path, correction / "original_CURRENT_RUN_STATUS.json")
    run.write_json(status_path, {"state": "all_original_computations_complete_capture_matching_withheld",
                                "computed_units": 55, "accepted_full_matched_units": 51,
                                "capture_fresh_flat_only_windows": 4, "common_flat_capture_windows": 0,
                                "correction_handoff": str(root / "CAPTURE_CORRECTION_HANDOFF.json"),
                                "updated_utc": datetime.now(timezone.utc).isoformat(), "active_extension_jobs": 0})
    return handoff


def main():
    root = run.require_physical_d(run.OUTPUT)
    correction = root / VERSION
    correction.mkdir(exist_ok=True)
    folder = root / "capture"
    protocol = json.loads((folder / "protocol.json").read_text())
    historical = json.loads((folder / "frozen" / "run_metadata.json").read_text())
    immutable = [root / name for name in ("lofo_results.csv", "scarcity_results.csv", "calibration_results.csv",
                 "alert_budget_results.csv", "capture_windows_results.csv", "table4_nine_arm_ece10.csv")]
    immutable += list((folder / "frozen").glob("*.npz")) + list(folder.glob("*.joblib"))
    hashes = {str(path): run.file_hash(path) for path in immutable}
    print("AUDIT original archives and component availability", flush=True)
    locations = archive_inventory(correction)
    print("AUDIT nominal fit rows, labels, weights and constructors; no refit", flush=True)
    roles = audit_inputs(folder, protocol, historical)
    model = model_audit(folder, protocol)
    naming = {"source_script": str(Path(historical_source.__file__)),
              "source_sha256": run.file_hash(Path(historical_source.__file__)),
              "documentation_and_code": "Module docstring: Methods without gates; lines 742-744: fixed .5 flat + .5 hierarchy",
              "source_actual_operation": "fixed_half_flat_plus_half_hierarchy_for_real_cyclic_and_support",
              "not_established": "A capture benefit gate cannot be inferred from the calibrated_* legacy ID",
              "ordinary_archive_endpoint_uses_gate_but_is_not_this_capture_window_generator": True,
              "corrected_audit_ids": list(ARCHIVED.values()),
              "formal_cyclic_gate_or_support_gate_created": False,
              "historical_source_snapshot_at_fit_available": False}
    for key in ("fine_labels",):
        if protocol[key] != historical[key]:
            raise ValueError(f"Class encoding mismatch: {key}")
    if any(protocol["package_versions"][key] != value for key, value in historical["package_versions"].items()):
        raise ValueError("Package versions differ")
    if any(protocol["group_overlap_counts"].values()):
        raise ValueError("Development roles overlap")
    diagnostic = {"status": "bounded_diagnostic_complete_repair_unavailable", "version": VERSION,
                  "cause_established": False, "thread_difference_is_not_causal_proof": True,
                  "current_source_and_cache_sha_match": True,
                  "same_historical_as_fitted_rows_or_model_proven": False,
                  "hierarchy_fit_role_source_scope": "Current window source fits all hierarchies on the same base frame, files 1-45; no historical hierarchy model/fit ledger exists to independently attest actual fitted bytes or ordered rows",
                  "fit_role_rows_and_fingerprints": roles, "model": model,
                  "naming_resolution": naming, "archive_locations": [str(p) for p in locations],
                  "source_fit_function_sha256": run.hashlib.sha256(inspect.getsource(run.nested.fit_weighted_xgb).encode()).hexdigest(),
                  "original_probability_components_available": False,
                  "serialized_original_models_available": False,
                  "window_disagreement": pd.read_csv(root / "capture_flat_reproduction_audit.csv").to_dict("records"),
                  "incomplete_reason": REASON, "no_refit_no_reconstruction": True,
                  "no_new_scientific_parameters_or_row_selection": True,
                  "finished_utc": datetime.now(timezone.utc).isoformat()}
    run.write_json(correction / "diagnostic.json", diagnostic)
    handoff = publish(root, correction, diagnostic)
    for path, checksum in hashes.items():
        if run.file_hash(Path(path)) != checksum:
            raise RuntimeError(f"Immutable result/source modified: {path}")
    run.write_json(correction / "preservation_audit.json", {
        "status": "passed", "original_artifacts_unchanged": hashes,
        "original_handoff_preserved_sha256": handoff["original_handoff_sha256"],
        "bounded_audit_completed": True, "common_flat_repair_completed": False})
    print(json.dumps({"computed": 55, "matched_accepted": 51,
                      "common_flat_capture_complete": 0, "fresh_flat_capture_complete": 4,
                      "correction_handoff": str(root / "CAPTURE_CORRECTION_HANDOFF.json")}), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
