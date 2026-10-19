"""Audit every new capture probability, then publish the corrected contract."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from threadpoolctl import threadpool_limits

import run_capture_common_flat as common
import build_manuscript_handoff as builder


def preserve(path, destination):
    if path.exists() and not destination.exists():
        shutil.copyfile(path, destination)


def original_artifact_path(path):
    path = Path(path)
    archived = common.OUTPUT / f"pre_refit_{path.name}"
    if path == common.run.OUTPUT / "capture_windows_results.csv" and archived.exists():
        return archived
    return path


def audit_flat_constructors(protocol, encoder, check):
    common.run.trf.RANDOM_STATE = common.capture.SEED
    actual = {}
    for name, factory, trees in (("xgb.joblib", common.run.trf.make_global_model, 520),
                                  ("lgbm.joblib", common.run.trf.make_lgbm, 380)):
        model = common.joblib.load(common.capture.OUTPUT / name)
        observed = {key: repr(value) for key, value in model.get_params().items()}
        expected = {key: repr(value) for key, value in factory(len(encoder.classes_), trees).get_params().items()}
        check(f"actual_flat_constructor_{name}", observed == expected)
        check(f"actual_flat_shape_{name}", model.n_features_in_ == 39
              and np.array_equal(model.classes_, np.arange(34)))
        actual[name] = {"actual_parameters_repr": observed, "expected_parameters_repr": expected,
                        "constructor_exact_match": True, "features": 39, "classes": 34,
                        "fit_role": "base files 1-45", "fit_rows": 720747,
                        "extra_row_cap": None, "weight_exponent": .35,
                        "base_fingerprint": protocol["base_fingerprint"],
                        "checkpoint_sha256": common.run.file_hash(common.capture.OUTPUT / name),
                        "fit_scope": "existing fresh checkpoint and frozen fit-source protocol; no new flat refit"}
        del model
    calibration = pd.read_parquet(common.capture.OUTPUT / "datasets" / "gate_train.parquet", columns=["Label"])
    with np.load(common.capture.OUTPUT / "calibration_inputs.npz") as saved:
        np.testing.assert_array_equal(saved["y_true"], encoder.transform(calibration.Label.astype(str)))
        check("meta_input_role_and_shape", saved["xgb"].shape == saved["lgbm"].shape == (329477, 34))
    meta = common.joblib.load(common.capture.OUTPUT / "calibrators.joblib")
    parameters = {key: repr(value) for key, value in meta["stacker"].get_params().items()}
    check("actual_meta_frozen_constructor", parameters["solver"] == "'newton-cholesky'"
          and parameters["C"] == "1.0" and parameters["max_iter"] == "100"
          and int(meta["stacker_iterations"]) < 100)
    check("actual_temperatures_finite_positive", all(np.isfinite(meta[key]) and meta[key] > 0
          for key in ("xgb_temperature", "lgbm_temperature")))
    actual["calibrators.joblib"] = {"stacker_actual_parameters_repr": parameters,
        "stacker_iterations": int(meta["stacker_iterations"]), "fit_role": "utility/calibration files 46-50",
        "rows": 329477, "selection_role_used_for_fit": False,
        "calibration_inputs_sha256": common.run.file_hash(common.capture.OUTPUT / "calibration_inputs.npz"),
        "xgb_temperature": meta["xgb_temperature"], "lgbm_temperature": meta["lgbm_temperature"],
        "checkpoint_sha256": common.run.file_hash(common.capture.OUTPUT / "calibrators.joblib")}
    return actual


def audit_sampling_roles(protocol, check):
    folder = common.capture.OUTPUT
    fresh = json.loads((folder / "protocol.json").read_text())
    path = folder / "datasets" / "sample_metadata.json"
    sample = json.loads(path.read_text())
    check("frozen_sampling_metadata_sha", common.run.file_hash(path) == fresh["sample_sha256"][path.name])
    expected_caps = {"base_attack": 25000, "base_benign": 80000,
                     "gate_train_per_label": 15000, "gate_select_per_label": 15000}
    check("frozen_original_sampling_caps", sample["caps"] == expected_caps)
    check("same_ordered_features_and_label_universe", sample["feature_columns"] == protocol["features"]
          and sample["canonical_labels"] == protocol["fine_labels"])
    roles, groups = {}, {}
    for role, expected in (("base", list(range(1, 46))), ("gate_train", list(range(46, 51))),
                            ("gate_select", list(range(51, 55)))):
        path = folder / "datasets" / f"{role}.parquet"
        frame = pd.read_parquet(path, columns=["Label", "_source_sequence", "_feature_hash"])
        check(f"actual_role_source_sha_{role}", common.run.file_hash(path) == fresh["sample_sha256"][path.name])
        check(f"actual_role_rows_sequences_{role}", len(frame) == protocol["role_rows"][role]
              and sorted(frame._source_sequence.unique().tolist()) == expected)
        counts = frame.Label.value_counts().sort_index()
        limits = {label: expected_caps["base_benign"] if label == "BenignTraffic" else expected_caps["base_attack"]
                  for label in counts.index} if role == "base" else dict.fromkeys(counts.index, expected_caps[f"{role}_per_label"])
        check(f"actual_frozen_label_caps_{role}", all(int(counts[label]) <= limits[label] for label in counts.index))
        groups[role] = np.unique(frame._feature_hash.to_numpy())
        roles[role] = {"rows": len(frame), "sequences": expected, "class_counts": counts.to_dict(),
                       "original_label_caps": limits, "additional_cap_in_new_refit": None,
                       "source_sha256": common.run.file_hash(path)}
    for left, right in (("base", "gate_train"), ("base", "gate_select"), ("gate_train", "gate_select")):
        check(f"actual_disjoint_role_groups_{left}_{right}", not len(np.intersect1d(groups[left], groups[right])))
    return {"original_sample_caps": expected_caps, "roles": roles,
            "selection_is_audit_only": True, "test_windows_no_additional_sampling": True}


def audit_window_sources(protocol, check):
    folder = common.capture.OUTPUT
    path = folder / "datasets" / "official_member_manifest.csv"
    copied = json.loads(path.with_suffix(".csv.copy.json").read_text())
    check("original_member_manifest_sha", common.run.file_hash(path) == copied["sha256"]
          == common.run.file_hash(Path(copied["source"])))
    manifest = pd.read_csv(path)
    sequences = sorted(sequence for window in protocol["windows"].values() for sequence in window)
    members = manifest.loc[manifest.sequence.isin(sequences)].sort_values("sequence")
    check("all_nine_window_source_members", len(members) == 9 and members.sequence.tolist() == sequences)
    rows = []
    for member in members.itertuples():
        source = folder / "bytes" / Path(member.member).name
        copied = json.loads(source.with_suffix(".csv.copy.json").read_text())
        digest = common.run.file_hash(source)
        check(f"original_window_source_sha_{member.sequence}", digest == member.sha256 == copied["sha256"]
              and source.stat().st_size == member.uncompressed_bytes == copied["source_size"])
        rows.append({"sequence": member.sequence, "member": member.member, "raw_rows": member.rows,
                     "D_cached_source": str(source), "original_read_only_source": copied["source"],
                     "bytes": source.stat().st_size, "sha256": digest, "original_manifest_sha_match": True})
    frame = pd.DataFrame(rows)
    common.run.write_csv(common.OUTPUT / "window_source_sha_audit.csv", frame)
    return frame.set_index("sequence").raw_rows.to_dict()


def new_quality_state():
    return {"n": 0, "count": np.zeros(10, dtype=np.int64), "confidence": np.zeros(10),
            "correct": np.zeros(10), "brier_sum": 0.0}


def add_quality(state, y, p):
    prediction = p.argmax(axis=1)
    confidence = p.max(axis=1).astype(np.float64)
    correct = y == prediction
    bins = np.clip(np.digitize(confidence, np.linspace(0, 1, 11), right=True) - 1, 0, 9)
    state["n"] += len(y)
    state["count"] += np.bincount(bins, minlength=10)
    state["confidence"] += np.bincount(bins, weights=confidence, minlength=10)
    state["correct"] += np.bincount(bins, weights=correct, minlength=10)
    probability = p.astype(np.float64, copy=False)
    losses = np.square(probability).sum(axis=1) - 2 * probability[np.arange(len(y)), y] + 1
    state["brier_sum"] += float(losses.sum())


def finish_quality(state):
    return {"rows": state["n"],
            "ece": float(np.abs(state["correct"] - state["confidence"]).sum() / state["n"]),
            "brier": state["brier_sum"] / state["n"]}


def audit(protocol):
    root, out = common.run.OUTPUT, common.OUTPUT
    checks = []
    def check(name, value):
        checks.append({"check": name, "passed": bool(value)})
        if not value:
            raise AssertionError(name)
    check("frozen_protocol_signature", common.signature({key: value for key, value in protocol.items()
          if key != "protocol_sha256"}) == protocol["protocol_sha256"])
    check("all_four_windows_complete", len(list(out.glob("w*/completed.json"))) == 4)
    check("fixed_weights_and_tree_counts", protocol["fixed_blend_weights"] == [.5, .5]
          and protocol["flat_xgb_trees"] == 520 and protocol["flat_lgbm_trees"] == 380
          and protocol["hierarchy_family_and_local_lgbm_trees"] == 380)
    check("all_threads_two", protocol["n_jobs"] == 2)
    fresh_protocol = json.loads((common.capture.OUTPUT / "protocol.json").read_text())
    check("same_runtime_versions_as_fresh_flat_fit", common.capture.archive.package_versions()
          == fresh_protocol["package_versions"])
    check("all_role_group_intersections_zero", not any(protocol["role_group_overlap"].values()))
    check("same_fresh_flat_meta_checkpoints", all(common.run.file_hash(Path(v["path"])) == v["sha256"]
          for v in protocol["model_checkpoints"].values()))
    check("frozen_maps_unmodified", json.loads((out / "frozen_maps.json").read_text()) == protocol["maps"])
    check("original_support_map_sha_and_labels", common.run.file_hash(Path(protocol["support_map_source"]))
          == protocol["support_map_source_sha256"] and json.loads(Path(protocol["support_map_source"]).read_text())
          == protocol["maps"]["support"])
    check("source_scripts_unmodified_since_freeze", all(common.run.file_hash(Path(path)) == digest
          for path, digest in protocol["implementation_sources_sha256"].items()))
    prior = json.loads((root / "capture_component_scope_v1" / "preservation_audit.json").read_text())
    check("all_original_results_and_archives_preserved", all(common.run.file_hash(original_artifact_path(path)) == digest
          for path, digest in prior["original_artifacts_unchanged"].items()))
    original = json.loads((root / "capture_component_scope_v1" / "original_MANUSCRIPT_HANDOFF.json").read_text())
    required = ("lofo_results.csv", "scarcity_results.csv", "calibration_results.csv",
                "alert_budget_retrieval_results.csv", "table4_nine_arm_ece10.csv")
    check("non_capture_numeric_contracts_unchanged", all(
        common.run.file_hash(Path(original["csv_contracts"][name]["path"]))
        == original["csv_contracts"][name]["sha256"] for name in required))
    check("refit_configuration_unchanged", common.run.file_hash(common.run.CONFIG_PATH)
          == original["verification"]["config_sha256"])
    check("canonical_e3_e4_source_audit_retained", json.loads((root / "frozen_source_audit.json").read_text())
          == original["canonical_frozen_probability_source_audit"])
    common.run.write_json(out / "sampling_role_audit.json", audit_sampling_roles(protocol, check))
    source_row_counts = audit_window_sources(protocol, check)
    base = pd.read_parquet(common.capture.OUTPUT / "datasets" / "base.parquet")
    X = common.run.trf.clean_features(base, protocol["features"])
    encoder = LabelEncoder().fit(protocol["fine_labels"])
    actual_models = audit_flat_constructors(protocol, encoder, check)
    expected_components = 0
    fit_rows = []
    for name, mapping in protocol["maps"].items():
        values = base.Label.astype(str).map(mapping)
        family_encoder = LabelEncoder().fit(values)
        specs = [("family", X, family_encoder.transform(values), .25)]
        for index, family in enumerate(family_encoder.classes_):
            mask = values.to_numpy() == family
            local_encoder = LabelEncoder().fit(base.loc[mask, "Label"].astype(str))
            if len(local_encoder.classes_) > 1:
                specs.append((f"local_{index:02d}", X[mask], local_encoder.transform(base.loc[mask, "Label"].astype(str)), .5))
        expected_components += len(specs)
        for component, xx, yy, exponent in specs:
            folder = out / "models" / name
            spec = json.loads((folder / f"{component}.completed.json").read_text())
            check(f"fit_lineage_{name}_{component}", spec["ordered_X_sha256"] == common.run.array_hash(xx)
                  and spec["ordered_y_sha256"] == common.run.array_hash(yy)
                  and spec["ordered_weight_sha256"] == common.run.array_hash(common.run.trf.inverse_frequency_weights(yy, exponent))
                  and spec["model_sha256"] == common.run.file_hash(folder / f"{component}.joblib")
                  and spec["parameters_repr"]["n_jobs"] == "2" and spec["parameters_repr"]["n_estimators"] == "380")
            model = common.joblib.load(folder / f"{component}.joblib")["model"]
            observed = {key: repr(value) for key, value in model.get_params().items()}
            expected = {key: repr(value) for key, value in common.run.trf.make_lgbm(len(np.unique(yy)), 380).get_params().items()}
            check(f"actual_hierarchy_constructor_{name}_{component}", observed == expected == spec["parameters_repr"]
                  and model.n_features_in_ == 39 and np.array_equal(model.classes_, np.arange(len(np.unique(yy)))))
            actual_models[f"{name}/{component}"] = {"actual_parameters_repr": observed,
                "expected_parameters_repr": expected, "constructor_exact_match": True,
                "fit_role": "base files 1-45, family-conditioned subset for locals", "fit_rows": len(yy),
                "extra_row_cap": None, "weight_exponent": exponent,
                "fit_input_sha256": spec["input_sha256"], "checkpoint_sha256": spec["model_sha256"]}
            del model
            check(f"map_frozen_before_fit_{name}_{component}", (out / "frozen_maps.json").stat().st_mtime_ns
                  <= (folder / f"{component}.input.json").stat().st_mtime_ns)
            fit_rows.append({"map": name, "component": component, "rows": len(yy),
                             "classes": len(np.unique(yy)), "n_jobs": 2, "trees": 380,
                             "fit_seconds": spec["fit_seconds"], "model_sha256": spec["model_sha256"],
                             "fit_input_sha256": spec["input_sha256"]})
    check("all_expected_components_fitted", expected_components == len(list((out / "models").glob("*/*.completed.json"))))
    common.run.write_csv(out / "hierarchy_fit_audit.csv", pd.DataFrame(fit_rows))
    common.run.write_json(out / "actual_model_constructor_audit.json", actual_models)
    del base, X, specs
    results, components, quality, part_audits, part_times, row_audits = [], [], [], [], [], []
    for window, sources in protocol["prediction_parts"].items():
        matrices = {method: np.zeros((34, 34), dtype=np.int64) for method in common.ALL_METHODS}
        states = {method: {"all": new_quality_state(), "rare": new_quality_state()} for method in common.ALL_METHODS}
        y_parts, hash_parts = [], []
        for source in sources:
            path = out / window / Path(source["parquet"]).with_suffix(".npz").name
            marker = json.loads(path.with_suffix(".completed.json").read_text())
            part_times.append({"window": window, "part": path.stem, "rows": marker["rows"],
                               "elapsed_seconds": marker["elapsed_seconds"], "finished_utc": marker["finished_utc"]})
            check(f"part_sha_{window}_{path.stem}", marker["prediction_file_sha256"] == common.run.file_hash(path)
                  and marker["protocol_sha256"] == protocol["protocol_sha256"]
                  and common.run.file_hash(Path(source["parquet"])) == source["sha256"]
                  and common.run.file_hash(Path(source["fresh_predictions"])) == source["fresh_prediction_sha256"])
            with np.load(path) as saved, np.load(source["fresh_predictions"]) as old:
                y = saved["y_true"]
                np.testing.assert_array_equal(saved["rare_ids"], protocol["rare_ids"])
                y_parts.append(y.copy())
                frame = pd.read_parquet(source["parquet"], columns=[*protocol["features"], "Label",
                    "feature_hash", "source_sequence", "source_row"])
                recomputed = pd.util.hash_pandas_object(frame[protocol["features"]], index=False,
                    categorize=True).to_numpy(dtype=np.uint64)
                np.testing.assert_array_equal(recomputed, frame.feature_hash.to_numpy())
                check(f"independently_recomputed_feature_hash_{window}_{path.stem}", True)
                check(f"original_source_row_bounds_{window}_{path.stem}",
                    frame.source_sequence.isin(protocol["windows"][window]).all()
                    and (frame.source_row >= 0).all()
                    and (frame.source_row < frame.source_sequence.map(source_row_counts)).all())
                hash_parts.append(recomputed)
                row_audits.append({"window": window, "part": path.stem, "rows": len(frame),
                    "predictors": len(protocol["features"]), "independently_recomputed_feature_hash": True,
                    "ordered_feature_hash_sha256": common.run.array_hash(recomputed),
                    "ordered_source_keys_sha256": common.run.array_hash(frame[["source_sequence", "source_row"]].to_numpy()),
                    "source_parquet_sha256": source["sha256"]})
                np.testing.assert_array_equal(y, encoder.transform(frame.Label.astype(str)))
                for key in ("y_true", "feature_hash", "source_sequence", "source_row"):
                    np.testing.assert_array_equal(saved[key], old[key])
                    if key != "y_true":
                        np.testing.assert_array_equal(saved[key], frame[key].to_numpy())
                check(f"row_alignment_{window}_{path.stem}", True)
                for method in common.FRESH:
                    expected = old[f"{method}_probability"].argmax(axis=1) if method in common.run.STRONG else old[method]
                    np.testing.assert_array_equal(saved[method], expected)
                    if method in common.run.STRONG:
                        np.testing.assert_array_equal(saved[f"{method}_probability"], old[f"{method}_probability"])
                check(f"five_fresh_arms_preserved_{window}_{path.stem}", True)
                flat = saved["flat_xgb_probability"]
                for name in common.HIERARCHIES:
                    np.testing.assert_array_equal(saved[f"{common.BLENDS[name]}_probability"],
                        common.equal_blend(flat, saved[f"{common.HIERARCHIES[name]}_probability"]))
                check(f"shared_flat_fixed_half_identity_{window}_{path.stem}", True)
                rare = np.isin(y, protocol["rare_ids"])
                for method in common.ALL_METHODS:
                    p = saved[f"{method}_probability"]
                    audit_row = common.probability_audit(p)
                    np.testing.assert_array_equal(saved[method], p.argmax(axis=1))
                    check(f"probability_{window}_{path.stem}_{method}", audit_row == marker["probabilities"][method])
                    matrices[method] += confusion_matrix(y, saved[method], labels=np.arange(34))
                    add_quality(states[method]["all"], y, p)
                    add_quality(states[method]["rare"], y[rare], p[rare])
                    part_audits.append({"window": window, "part": path.stem, "method": method,
                                       "rows": len(y), "probability_sha256": audit_row["sha256"],
                                       "hard_sha256": audit_row["hard_sha256"],
                                       "row_sum_max_error": audit_row["max_row_sum_error"]})
        y_all = np.concatenate(y_parts)
        all_hashes = np.concatenate(hash_parts)
        check(f"window_feature_groups_unique_{window}", len(np.unique(all_hashes)) == len(all_hashes))
        with np.load(common.capture.OUTPUT / "frozen" / f"window_predictions_{window}.npz") as old:
            np.testing.assert_array_equal(y_all, old["y_true"])
            np.testing.assert_array_equal(protocol["rare_ids"], old["rare_ids"])
        check(f"entire_original_window_order_{window}", True)
        rows = []
        for method in common.ALL_METHODS:
            metrics, _ = common.capture.archive.confusion_metrics(matrices[method], np.asarray(protocol["rare_ids"]))
            metrics["macro_f1"] = metrics.pop("f1_macro")
            row = {"endpoint": "capture_windows", "window": window, "seed": common.capture.SEED,
                   "method": method, "test_rows": len(y_all), **metrics}
            rows.append(row)
            qa, qr = (finish_quality(states[method][key]) for key in ("all", "rare"))
            quality.append({"window": window, "seed": common.capture.SEED, "method": method,
                            "rows": qa["rows"], "rare_rows": qr["rows"], "ece_all": qa["ece"],
                            "ece_rare_rows": qr["ece"], "brier_all": qa["brier"], "brier_rare_rows": qr["brier"],
                            "nbins": 10, "right_closed": True})
        actual = pd.DataFrame(rows)
        stored = pd.read_csv(out / window / "component_results.csv")
        pd.testing.assert_frame_equal(actual, stored, check_exact=False, rtol=1e-13, atol=1e-15)
        check(f"independent_confusion_metrics_{window}", True)
        components.append(actual)
        results.append(actual.loc[actual.method.isin(common.METHODS)])
    formal = pd.concat(results, ignore_index=True)
    check("all_four_by_nine_formal_results", len(formal) == 36 and not formal.isna().any().any())
    old_fresh = pd.read_csv(root / "capture_windows_results.csv")
    old_fresh = old_fresh.loc[old_fresh.method.isin(common.FRESH)].sort_values(["window", "method"]).reset_index(drop=True)
    new_fresh = formal.loc[formal.method.isin(common.FRESH)].sort_values(["window", "method"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(old_fresh, new_fresh, check_exact=False, rtol=1e-13, atol=1e-15)
    check("all_original_fresh_flat_numeric_results_preserved", True)
    common.run.write_csv(out / "capture_windows_results.csv", formal)
    common.run.write_csv(out / "capture_hierarchy_component_results.csv", pd.concat(components, ignore_index=True))
    common.run.write_csv(out / "capture_probability_quality_ece10.csv", pd.DataFrame(quality))
    common.run.write_csv(out / "prediction_array_audit.csv", pd.DataFrame(part_audits))
    common.run.write_csv(out / "inference_part_timings.csv", pd.DataFrame(part_times))
    common.run.write_csv(out / "capture_row_hash_audit.csv", pd.DataFrame(row_audits))
    common.run.write_csv(out / "verification_checks.csv", pd.DataFrame(checks))
    report = {"status": "passed", "checks_passed": len(checks), "completed_windows": 4,
              "formal_method_rows": 36, "prediction_parts": sum(map(len, protocol["prediction_parts"].values())),
              "hierarchy_components": expected_components, "probability_matrices": len(part_audits),
              "common_fitted_flat_verified": True, "fixed_half_blend_verified": True,
              "independently_recomputed_feature_hash_rows": int(pd.DataFrame(row_audits).rows.sum()),
              "original_window_source_files_sha_verified": len(source_row_counts),
              "maps_frozen_before_fitting": True, "all_fresh_checkpoints_unchanged": True,
              "hierarchy_component_fit_seconds_sum": float(pd.DataFrame(fit_rows).fit_seconds.sum()),
              "hierarchy_fit_seconds_by_map": pd.DataFrame(fit_rows).groupby("map").fit_seconds.sum().to_dict(),
              "inference_part_seconds_sum": float(pd.DataFrame(part_times).elapsed_seconds.sum()),
              "ece_nbins": 10, "protocol_sha256": protocol["protocol_sha256"],
              "package_versions": common.capture.archive.package_versions(),
              "verifier_sha256": common.run.file_hash(Path(__file__)), "finished_utc": common.now()}
    common.run.write_json(out / "verification_report.json", report)
    return formal, report


def publish(formal, report, protocol):
    root, out = common.run.OUTPUT, common.OUTPUT
    original = json.loads((root / "capture_component_scope_v1" / "original_MANUSCRIPT_HANDOFF.json").read_text())
    frames = {name: pd.read_csv(root / f"{name}_results.csv") for name in ("lofo", "scarcity", "calibration")}
    frames["alert_budget"] = pd.read_csv(root / "alert_budget_retrieval_results.csv")
    frames["capture_windows"] = formal
    comparisons = builder.paired_comparisons(frames)
    summary, best, paired = builder.all_endpoint_tables(frames, comparisons)
    outputs = {"all_endpoint_control_comparisons.csv": comparisons, "all_endpoint_summary.csv": summary,
               "best_competitor_by_endpoint_metric.csv": best, "all_endpoint_per_seed_paired_results.csv": paired}
    wide = []
    for window, values in formal.groupby("window"):
        index = values.set_index("method")
        for control in common.run.STRONG:
            wide.append({"window": window, "control": control,
                         **{f"taxonomy_blend_minus_control_{metric}": float(index.loc["taxonomy_blend", metric] - index.loc[control, metric])
                            for metric in ("macro_f1", "rare_recall")}})
    outputs["capture_paired_deltas.csv"] = pd.DataFrame(wide)
    map_pairs = []
    for window, values in formal.groupby("window"):
        index = values.set_index("method")
        for control in ("cyclic_blend", "support_matched_blend"):
            for metric in ("macro_f1", "rare_recall"):
                taxonomy, baseline = (float(index.loc[method, metric]) for method in ("taxonomy_blend", control))
                map_pairs.append({"window": window, "seed": common.capture.SEED, "control": control,
                    "metric": metric, "taxonomy_value": taxonomy, "control_value": baseline,
                    "taxonomy_minus_control": taxonomy - baseline,
                    "direction": "taxonomy_better" if taxonomy > baseline else "control_better" if taxonomy < baseline else "tie",
                    "scope": "same fitted flat; frozen maps; dependent windows; descriptive, no significance inference"})
    outputs["capture_shared_flat_map_comparisons.csv"] = pd.DataFrame(map_pairs)
    coverage = pd.read_csv(root / "endpoint_coverage.csv").fillna("")
    coverage.loc[coverage.endpoint.eq("alert_budget"), "result_csv"] = str(root / "alert_budget_retrieval_results.csv")
    coverage.loc[coverage.endpoint.eq("capture_windows"), "result_csv"] = str(root / "capture_windows_results.csv")
    coverage.loc[coverage.endpoint.eq("capture_windows"), "status"] = "completed"
    outputs["endpoint_coverage.csv"] = coverage
    registry = []
    for method in common.ALL_METHODS:
        is_hierarchy = method in common.HIERARCHIES.values()
        is_blend = method in common.BLENDS.values()
        registry.append({"method": method, "formal_table_arm": method in common.METHODS,
            "new_fit": "family/local LGBM only" if is_hierarchy else "no; derived half mixture" if is_blend else "no; reused fresh checkpoint/probability",
            "fit_role": "base files 1-45" if is_hierarchy else "utility files 46-50, existing checkpoint" if method in common.run.STRONG else "base files 1-45, existing checkpoint" if method in common.FRESH else "not applicable: fixed arithmetic",
            "blend_weights": "0.5/0.5" if is_blend or method in ("flat_equal_blend", "temperature_flat_blend") else "not applicable",
            "gate": False, "flat_component_sha256": protocol["model_checkpoints"]["xgb.joblib"]["sha256"] if is_blend or method == "flat_xgb" else "not applicable"})
    outputs["capture_method_registry.csv"] = pd.DataFrame(registry)
    for name, frame in outputs.items():
        common.run.write_csv(out / name, frame)
    published_frames = {**outputs, "capture_windows_results.csv": formal}
    published = {}
    for name, frame in published_frames.items():
        previous = out / f"pre_refit_{name}"
        preserve(root / name, previous)
        common.run.write_csv(root / name, frame)
        if common.run.file_hash(root / name) != common.run.file_hash(out / name):
            raise AssertionError(f"Root/versioned publication bytes differ: {name}")
        published[name] = {"authoritative": builder.csv_contract(root / name),
                           "versioned": builder.csv_contract(out / name),
                           "previous_archived_path": str(previous) if previous.exists() else None,
                           "previous_sha256": common.run.file_hash(previous) if previous.exists() else None}
    publication_manifest = {"version": "capture_common_flat_v2", "state": "csvs_published_after_full_audit",
        "protocol_sha256": protocol["protocol_sha256"], "published_utc": common.now(),
        "root_completion_guard": str(root / "MANUSCRIPT_HANDOFF.json"), "artifacts": published,
        "verification_report": str(out / "verification_report.json"),
        "actual_model_constructor_audit": str(out / "actual_model_constructor_audit.json"),
        "original_hybrid_scope": "Archived unchanged, audit only; not used for corrected model ranking"}
    common.run.write_json(out / "PUBLICATION_MANIFEST.json", publication_manifest)
    for name in ("MANUSCRIPT_HANDOFF.json", "CAPTURE_CORRECTION_HANDOFF.json", "PROTOCOL_CONTRACT.json"):
        preserve(root / name, out / f"pre_refit_{name}")
    handoff = copy.deepcopy(original)
    handoff.update({"completion": True, "computed_completion": True, "completed_endpoint_categories": 5,
        "completed_units": 55, "expected_units": 55, "endpoint_method_rows": 1107,
        "status": "completed_with_verified_common_flat_capture_refits", "coverage": builder.records(coverage),
        "capture_common_flat_verification": report,
        "capture_original_generation_drift_audit": original["capture_flat_reproduction_audit"],
        "capture_comparisons_by_window": builder.records(comparisons.loc[comparisons.endpoint.eq("capture_windows")]),
        "all_endpoint_summary": builder.records(summary), "best_competitor_by_endpoint_metric": builder.records(best),
        "reversals_control_better": builder.records(comparisons.loc[comparisons.direction.eq("control_better")]),
        "formal_capture_result_csv": str(root / "capture_windows_results.csv"),
        "versioned_formal_capture_result_csv": str(out / "capture_windows_results.csv"),
        "capture_publication_manifest": str(out / "PUBLICATION_MANIFEST.json"),
        "capture_publication_manifest_sha256": common.run.file_hash(out / "PUBLICATION_MANIFEST.json"),
        "capture_actual_model_constructor_audit": str(out / "actual_model_constructor_audit.json"),
        "capture_sampling_role_audit": str(out / "sampling_role_audit.json"),
        "capture_independent_row_hash_audit": str(out / "capture_row_hash_audit.csv"),
        "capture_original_window_source_sha_audit": str(out / "window_source_sha_audit.csv"),
        "capture_shared_flat_map_comparisons": builder.records(outputs["capture_shared_flat_map_comparisons.csv"]),
        "capture_exact_command": "& 'legacy_project\\.venv\\Scripts\\python.exe' -B -u 'workspace\\extensions\\scripts\\run_capture_common_flat.py'",
        "capture_run_log": str(out / "run.log"),
        "correction_handoff": str(root / "CAPTURE_CORRECTION_HANDOFF.json")})
    handoff["original_computational_verification"] = copy.deepcopy(original["verification"])
    handoff["verification"].update({
        "scope": "Composite verification: unchanged audited LOFO/scarcity/E3/E4 contracts plus the new common-flat capture audit; original capture rankings are retired",
        "capture_common_flat_verification_report": str(out / "verification_report.json"),
        "capture_common_flat_checks_passed": report["checks_passed"],
        "capture_common_flat_protocol_sha256": protocol["protocol_sha256"]})
    handoff["capture_flat_reproduction_audit"]["scope"] = "Historical old-versus-new generation drift only; not used in the corrected formal comparisons"
    handoff["capture_flat_reproduction_audit"]["current_common_flat_verified"] = True
    replaced = set(outputs) | {"capture_windows_results.csv"}
    handoff["csv_contracts"] = {name: value for name, value in handoff["csv_contracts"].items() if name not in replaced}
    handoff["csv_contracts"].update({path.name: builder.csv_contract(path) for path in sorted(out.glob("*.csv"))})
    handoff["csv_contracts"].update({name: value["authoritative"] for name, value in published.items()})
    contract = handoff["frozen_protocol_and_class_universes"]
    contract["capture_protocol"] = protocol
    contract["capture_method_ids"] = list(common.METHODS)
    contract["fit_and_reuse_scope"].update({
        "capture_fresh_model_fitting": "Three new family/local LGBM hierarchy bundles, 380 trees each component; flat XGB 520, flat LGBM 380 and temperature/stacker checkpoints reused unchanged",
        "capture_fresh_result_arms": list(common.METHODS), "capture_reused_hard_prediction_arms": [],
        "capture_reuse_checks": "All fresh five-arm cached results exactly preserved; source-row/class alignment audited for all 28 parts; blend probabilities exactly equal .5 times the shared current-flat probability plus .5 hierarchy probability",
        "capture_comparison_boundary": "Same current fitted flat component, same base rows and 39 predictors; all three maps frozen before new fits, all four windows included without tuning",
        "capture_all_arms_retrained": False})
    contract["capture_legacy_method_aliases"] = {"status": "retired audit-only IDs; source reused gate-like names for fixed blends",
        "original_hybrid_audit_csv": str(root / "capture_component_scope_v1" / "capture_original_hybrid_audit.csv")}
    prose = [p for p in original["review_facing_prose"] if "capture" not in p.lower()]
    prose.append("The four capture-order windows now compare three newly fitted hierarchy maps against the same existing fresh flat component. Family/local LGBM components use 380 trees and the frozen base rows; flat XGB 520, flat LGBM 380 and utility-calibrated meta-models were reused, not retrained. Each hierarchy blend uses fixed equal weights. No gate was fitted for these capture controls.")
    cap_summary = summary.loc[summary.endpoint.eq("capture_windows") & summary.facet.eq("all_four_windows")]
    index = cap_summary.set_index(["method", "metric"])
    for control in common.run.STRONG:
        for metric in ("macro_f1", "rare_recall"):
            tv, cv = (float(index.loc[(method, metric), "mean"]) for method in ("taxonomy_blend", control))
            prose.append(f"Across all four dependent capture windows, taxonomy blend {metric} averaged {tv:.6f} versus {cv:.6f} for {control}; taxonomy minus control was {tv-cv:.6f}. These are descriptive means from one fitted seed, not independent-replicate inference.")
    for control in ("cyclic_blend", "support_matched_blend"):
        for metric in ("macro_f1", "rare_recall"):
            values = outputs["capture_shared_flat_map_comparisons.csv"]
            values = values.loc[values.control.eq(control) & values.metric.eq(metric)]
            prose.append(f"For the same shared-flat comparison against {control}, mean taxonomy-minus-control {metric} was {values.taxonomy_minus_control.mean():.6f}; taxonomy was higher in {int(values.taxonomy_minus_control.gt(0).sum())}/4 dependent windows and lower in {int(values.taxonomy_minus_control.lt(0).sum())}/4. These descriptive directions do not establish population significance or universal taxonomy superiority.")
    prose.append("All nine Table 4 arms, including the taxonomy, cyclic and support gates, retain the original canonical-v10 six-split ECE10 results. Capture cyclic/support arms are separate fixed blends, not those gates. Capture file order is a proxy, not timestamp validation; no physical edge hardware was available.")
    handoff["review_facing_prose"] = prose
    handoff["protocol_boundaries"]["capture_common_component_comparison_withheld"] = False
    handoff["protocol_boundaries"]["capture_common_fitted_flat_verified"] = True
    common.run.write_json(root / "PROTOCOL_CONTRACT.json", contract)
    correction = {"completion": True, "version": "capture_common_flat_v2", "completed_matched_units": 55,
        "capture_common_flat_windows_completed": 4, "formal_capture_methods": list(common.METHODS),
        "formal_capture_result_csv": str(root / "capture_windows_results.csv"),
        "versioned_formal_capture_result_csv": str(out / "capture_windows_results.csv"),
        "publication_manifest": str(out / "PUBLICATION_MANIFEST.json"),
        "verification": report, "protocol_json": str(out / "protocol.json"),
        "csv_contracts": {path.name: builder.csv_contract(path) for path in sorted(out.glob("*.csv"))},
        "original_hybrid_audit_only": str(root / "capture_component_scope_v1" / "capture_original_hybrid_audit.csv"),
        "supersedes_blocked_handoff": str(out / "pre_refit_CAPTURE_CORRECTION_HANDOFF.json"),
        "all_endpoint_summary_csv": str(root / "all_endpoint_summary.csv"),
        "all_endpoint_per_seed_paired_results_csv": str(root / "all_endpoint_per_seed_paired_results.csv"),
        "best_competitor_by_endpoint_metric_csv": str(root / "best_competitor_by_endpoint_metric.csv"),
        "review_facing_prose": [p for p in prose if "capture" in p.lower() or "shared-flat" in p.lower()],
        "all_nine_arms_retrained": False,
        "method_scope_change": "The ninth formal capture arm is newly available real_taxonomy, replacing the audit-only legacy_flat_xgb; the other five fresh flat and three blend IDs remain. Cyclic/support standalone hierarchies are additional audit arms.",
        "no_test_tuning_no_outcome_map_selection": True, "no_probability_reconstruction": True,
        "old_unresolved_flat_drift_cause_required_for_current_comparison": False}
    # Only publish completion after the full probability/source audit has passed.
    common.run.write_json(out / "MANUSCRIPT_HANDOFF.json", handoff)
    common.run.write_json(root / "CAPTURE_CORRECTION_HANDOFF.json", correction)
    common.run.write_json(root / "MANUSCRIPT_HANDOFF.json", handoff)
    print("PUBLISHED common-flat capture: 4/4 windows, 9 formal arms; full extension contract 55/55", flush=True)


def verify_and_publish(protocol):
    formal, report = audit(protocol)
    publish(formal, report, protocol)


if __name__ == "__main__":
    protocol = json.loads((common.OUTPUT / "protocol.json").read_text())
    with threadpool_limits(limits=2):
        verify_and_publish(protocol)
