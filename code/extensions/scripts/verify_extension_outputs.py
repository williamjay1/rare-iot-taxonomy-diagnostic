"""Validate actual extension results; missing cells remain explicitly missing."""

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

import run_matched_extensions as run


def finite(frame, columns):
    if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
        raise ValueError(f"Nonfinite result values: {columns}")


def verify_nested_aggregation(root, endpoint, expected_cases):
    frame = pd.read_csv(root / f"{endpoint}_results.csv")
    metrics = ["macro_f1", "rare_recall"]
    if endpoint == "lofo":
        metrics += ["msp_unknown_auroc", "msp_selection_fpr_0.05_unknown_recall",
                    "msp_selection_fpr_0.05_outer_known_fpr"]
    counts = frame.groupby(["seed", "method"]).size()
    if (set(frame.seed) != {42, 43, 44} or not counts.eq(expected_cases).all()
            or frame.duplicated(["seed", "case", "method"]).any()):
        raise ValueError(f"Incomplete nested seed summaries: {endpoint}")
    expected = frame.groupby(["seed", "method"])[metrics].mean().sort_index()
    actual = pd.read_csv(root / f"{endpoint}_per_seed_summary.csv").set_index(["seed", "method"]).sort_index()
    if (not actual.index.equals(expected.index)
            or not actual.nested_cases_completed.eq(expected_cases).all()
            or not actual.nested_cases_expected.eq(expected_cases).all()):
        raise ValueError(f"Nested summary counts disagree: {endpoint}")
    np.testing.assert_allclose(actual[metrics], expected[metrics], rtol=1e-12, atol=1e-12)
    rows = []
    for method, values in expected.reset_index().groupby("method"):
        for metric in metrics:
            rows.append({"endpoint": endpoint, "method": method, "metric": metric,
                         "mean_of_seed_means": float(values[metric].mean()),
                         "sample_sd_of_three_seed_means": float(values[metric].std(ddof=1)),
                         "seeds": len(values), "cases_within_each_seed": expected_cases,
                         "inference": "descriptive; three seed summaries, not independent nested cases"})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument("--check-probabilities", action="store_true")
    args = parser.parse_args()
    root = run.require_physical_d(run.OUTPUT)
    metric_protocol = json.loads((root / "calibration_metric_protocol.json").read_text())
    if metric_protocol["nbins"] != 10 or metric_protocol["bin_edges"] != [i / 10 for i in range(11)]:
        raise ValueError("ECE must use the declared ten equal-width right-closed bins")
    coverage = run.summarize(json.loads(run.CONFIG_PATH.read_text()))
    checks = []
    for endpoint, expected, keys, metrics in (
        ("calibration", 60, ["seed", "method"], ["ece_all", "brier_all", "ece_rare_rows", "brier_rare_rows"]),
        ("alert_budget", 660, ["seed", "method", "budget_id"], ["budget", "rare_hits", "rare_rows", "rare_recall", "rare_precision"]),
    ):
        frame = pd.read_csv(root / f"{endpoint}_results.csv")
        if len(frame) != expected or frame.duplicated(keys).any() or frame.isna().any().any():
            raise ValueError(f"Incomplete {endpoint} table")
        finite(frame, metrics)
        checks.append({"check": endpoint, "rows": len(frame), "expected_rows": expected, "status": "passed"})
        if endpoint == "calibration":
            table = frame.groupby("method").ece_all.agg(["mean", "std", "count"])
            table = table.loc[metric_protocol["table4_methods_in_order"]].reset_index()
            table.columns = ["method", "ece_all_mean", "ece_all_sample_sd", "seeds"]
            table["nbins"] = 10
            table["bin_closure"] = "right; zero included in first bin"
            table["display_mean_sample_sd"] = table.apply(
                lambda row: f"{row.ece_all_mean:.4f} +/- {row.ece_all_sample_sd:.4f}", axis=1)
            if not (table.seeds == 6).all():
                raise ValueError("Table 4 must have all six seeds for all nine arms")
            run.write_csv(root / "table4_nine_arm_ece10.csv", table)
    for marker in sorted((root / "cells").glob("*/completed.json")):
        folder = marker.parent
        protocol = json.loads(marker.read_text())
        frame = pd.read_csv(folder / "results.csv")
        if len(frame) != len(run.METHODS) or set(frame.method) != set(run.METHODS) or frame.method.duplicated().any():
            raise ValueError(f"Missing paired methods: {folder.name}")
        finite(frame, frame.select_dtypes("number").columns.tolist())
        if protocol["n_jobs"] != 2 or any(value for key, value in protocol.items() if key.endswith("group_overlap")):
            raise ValueError(f"Invalid grouped roles/threads: {folder.name}")
        if protocol["calibrator_fit_role"] != "calibration" or protocol["selection_used_for_model_or_hyperparameter_choice"]:
            raise ValueError(f"Calibration/selection role violation: {folder.name}")
        if not 0 < protocol["stacker_iterations"] < 100:
            raise ValueError(f"Unconverged stacker: {folder.name}")
        for name in ("xgb_temperature", "lgbm_temperature"):
            if not .25 <= protocol[name] <= 5:
                raise ValueError(f"Invalid temperature: {folder.name}")
        if args.check_probabilities:
            with np.load(folder / "paired_predictions.npz") as data:
                if run.array_hash(data["row_id"]) != protocol["known_outer_row_id_sha256"]:
                    raise ValueError(f"Outer row hash mismatch: {folder.name}")
                for name in data.files:
                    if name.endswith("_probability"):
                        p = data[name]
                        if not np.isfinite(p).all() or (p < 0).any():
                            raise ValueError(f"Invalid probabilities: {folder.name}/{name}")
                        np.testing.assert_allclose(p.sum(axis=1), 1, atol=2e-6)
        checks.append({"check": folder.name, "rows": len(frame), "expected_rows": len(run.METHODS), "status": "passed"})
    for marker in sorted((root / "capture").glob("w*/completed.json")):
        frame = pd.read_csv(marker.with_name("results.csv"))
        if len(frame) != 9 or frame.method.nunique() != 9 or not set(run.STRONG).issubset(frame.method):
            raise ValueError(f"Incomplete capture controls: {marker.parent.name}")
        finite(frame, frame.select_dtypes("number").columns.tolist())
        complete = json.loads(marker.read_text())
        protocol = json.loads((root / "capture" / "protocol.json").read_text())
        cache = json.loads(marker.with_name("cache_complete.json").read_text())
        if (protocol["n_jobs"] != 2 or any(protocol["group_overlap_counts"].values())
                or protocol["test_used_for_fitting_selection_or_tuning"]
                or complete["protocol_sha256"] != protocol["protocol_sha256"]
                or cache["protocol_sha256"] != protocol["protocol_sha256"]
                or not 0 < complete["stacker_iterations"] < 100):
            raise ValueError(f"Invalid capture roles, protocol or convergence: {marker.parent.name}")
        if args.check_probabilities:
            labels = []
            parts = sorted(marker.parent.glob("part_*.parquet"))
            if [path.name for path in parts] != sorted(cache["parts"]):
                raise ValueError("Capture cached part list changed")
            for part in parts:
                source = pd.read_parquet(part, columns=["feature_hash", "source_sequence", "source_row"])
                with np.load(part.with_suffix(".npz")) as data:
                    labels.append(data["y_true"])
                    for key in source.columns:
                        np.testing.assert_array_equal(data[key], source[key].to_numpy())
                    for method in run.STRONG:
                        p = data[f"{method}_probability"]
                        if (p.shape != (len(source), len(protocol["fine_labels"]))
                                or not np.isfinite(p).all() or (p < 0).any()):
                            raise ValueError(f"Invalid capture probability: {part.name}/{method}")
                        np.testing.assert_allclose(p.sum(axis=1), 1, atol=2e-6)
            ordered_y = np.concatenate(labels)
            if (len(ordered_y) != complete["rows"]
                    or run.array_hash(ordered_y) != complete["ordered_y_sha256"]
                    or complete["ordered_y_sha256"] != cache["ordered_y_sha256"]):
                raise ValueError("Capture ordered labels changed after cache verification")
        checks.append({"check": marker.parent.name, "rows": len(frame), "expected_rows": 9, "status": "passed"})
    completed = sum(row["completed_units"] for row in coverage)
    nested_audits = []
    for endpoint, expected_cases in (("lofo", 7), ("scarcity", 6)):
        covered = next(row for row in coverage if row["endpoint"] == endpoint)
        if covered["status"] == "completed":
            nested_audits.append(verify_nested_aggregation(root, endpoint, expected_cases))
            checks.append({"check": f"{endpoint}_nested_seed_aggregation", "rows": 3,
                           "expected_rows": 3, "status": "passed"})
    if nested_audits:
        run.write_csv(root / "nested_aggregation_audit.csv", pd.concat(nested_audits, ignore_index=True))
    policy = {
        "legacy_lofo_and_scarcity_headlines": "RETIRED; do not combine with new matched controls",
        "reason": ["Legacy endpoints used original random mirrors instead of the new grouped roles.",
                   "Legacy LOFO novelty thresholds used outer-known scores; new thresholds use selection only.",
                   "Legacy scarcity rare IDs changed after capping; new rare IDs are fixed per uncapped fit pool."],
        "replacement": "Use only complete newly paired grouped tables; scarcity is base-fit label scarcity with additional calibration labels.",
        "calibration_role": "utility/calibration fits temperatures and stacker; selection is not the model-fitting role",
        "ece_protocol": "10 equal-width, right-closed bins; six-seed mean and sample SD; all nine Table 4 arms, including three gate variants",
        "coverage_units_completed": completed, "coverage_units_expected": 55,
        "endpoint_categories_completed": sum(row["status"] == "completed" for row in coverage),
        "capture_boundary": "File capture order proxy only, not actual timestamp validation.",
    }
    run.write_json(root / "headline_replacement_policy.json", policy)
    run.write_csv(root / "verification_checks.csv", pd.DataFrame(checks))
    run.write_json(root / "verification_report.json", {
        "status": "passed" if completed == 55 else "available_results_passed_but_coverage_incomplete",
        "completed_units": completed, "expected_units": 55, "checks": len(checks),
        "probability_matrices_checked": args.check_probabilities,
        "nested_seed_aggregation_checked": len(nested_audits) == 2,
        "script_sha256": {path.name: run.file_hash(path) for path in Path(__file__).parent.glob("*.py")},
        "config_sha256": run.file_hash(run.CONFIG_PATH),
    })
    print(pd.DataFrame(coverage).to_string(index=False))
    print(f"Validated {len(checks)} available result sets; completed units={completed}/55", flush=True)
    if args.require_all and completed != 55:
        raise SystemExit("Required full 55-unit coverage is not complete")


if __name__ == "__main__":
    main()
