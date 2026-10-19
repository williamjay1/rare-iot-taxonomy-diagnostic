"""Publish the numeric contract only after all five endpoint categories pass."""

from __future__ import annotations

import json
import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd

import run_matched_extensions as run

EXPECTED_ROWS = {"lofo": 189, "scarcity": 162, "calibration": 60,
                 "alert_budget": 660, "capture_windows": 36}
DISPLAY = {"temperature_flat_blend": "temperature-calibrated flat blend",
           "newton_flat_stacker": "converged Newton-Cholesky flat stacker"}


def require_ready(root):
    correction = root / "CAPTURE_CORRECTION_HANDOFF.json"
    if correction.exists() and not json.loads(correction.read_text()).get("completion"):
        raise RuntimeError("Capture common-component correction is incomplete; do not republish the obsolete 55-unit matched claim")
    if correction.exists():
        raise RuntimeError("Use publish_capture_common_flat.py: the original frozen capture loader is retired")
    files = [root / name for name in ("endpoint_coverage.csv", "verification_report.json",
                                      "dispatch/final_publication.json")]
    if not all(path.exists() for path in files):
        raise RuntimeError("Final coverage, publication and probability verification must exist first")
    coverage = pd.read_csv(files[0]).fillna("")
    if (set(coverage.endpoint) != set(EXPECTED_ROWS) or len(coverage) != 5
            or not coverage.status.eq("completed").all()
            or not coverage.completed_units.eq(coverage.expected_units).all()
            or int(coverage.completed_units.sum()) != 55
            or not coverage.incomplete_reason.eq("").all()):
        raise RuntimeError("Refusing manuscript handoff: all five categories and 55 units are required")
    verification = json.loads(files[1].read_text())
    if (verification.get("status") != "passed" or verification.get("completed_units") != 55
            or not verification.get("probability_matrices_checked")
            or not verification.get("nested_seed_aggregation_checked")
            or verification.get("config_sha256") != run.file_hash(run.CONFIG_PATH)):
        raise RuntimeError("The complete frozen-configuration probability audit has not passed")
    publication = json.loads(files[2].read_text())
    if publication.get("frozen_config_sha256") != verification["config_sha256"]:
        raise RuntimeError("Publication configuration does not match verification")
    source_audit = root / "frozen_source_audit.json"
    if not source_audit.exists():
        raise RuntimeError("Canonical frozen probability source audit must exist first")
    source = json.loads(source_audit.read_text())
    if (source.get("status") != "passed" or source.get("source_files_checked") != 18
            or source.get("calibration_rows_reproduced") != 60 or source.get("alert_budget_rows_reproduced") != 660
            or source.get("config_sha256") != verification["config_sha256"]):
        raise RuntimeError("Canonical frozen probability source audit is incomplete")
    capture_audit = json.loads((root / "capture_flat_reproduction_audit.json").read_text())
    if (capture_audit.get("windows_checked") != 4
            or not capture_audit.get("no_refit_or_metric_replacement")
            or capture_audit.get("protocol_sha256") != json.loads((root / "capture" / "protocol.json").read_text())["protocol_sha256"]):
        raise RuntimeError("Capture fresh-versus-archived flat reproduction audit is incomplete")
    return coverage, verification


def records(frame):
    return json.loads(frame.to_json(orient="records", double_precision=15))


def endpoint_spec(endpoint, frame):
    if endpoint == "calibration":
        return ["ece_all", "brier_all", "ece_rare_rows", "brier_rare_rows"], ["seed"]
    if endpoint == "alert_budget":
        return ["rare_row_retrieval_recall", "rare_row_retrieval_precision", "rare_hits"], ["seed", "budget_id"]
    metrics = ["macro_f1", "rare_recall"]
    if endpoint == "lofo":
        metrics += [name for name in frame if name.endswith(("unknown_auroc", "unknown_recall", "outer_known_fpr"))]
    return metrics, ["seed", "window"] if endpoint == "capture_windows" else ["seed", "case"]


def paired_comparisons(frames):
    rows = []
    for endpoint, frame in frames.items():
        metrics, keys = endpoint_spec(endpoint, frame)
        for key, group in frame.groupby(keys, sort=True):
            index = group.set_index("method")
            identifiers = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
            for control in run.STRONG:
                for metric in metrics:
                    taxonomy, baseline = float(index.loc["taxonomy_blend", metric]), float(index.loc[control, metric])
                    lower = metric.startswith(("ece_", "brier_")) or metric.endswith("outer_known_fpr")
                    delta = taxonomy - baseline
                    benefit = -delta if lower else delta
                    rows.append({"endpoint": endpoint, **identifiers, "control": control, "metric": metric,
                                 "taxonomy_value": taxonomy, "control_value": baseline,
                                 "taxonomy_minus_control": delta, "higher_is_better": not lower,
                                 "taxonomy_benefit": benefit,
                                 "direction": "taxonomy_better" if benefit > 0 else "control_better" if benefit < 0 else "tie"})
    return pd.DataFrame(rows)


def all_endpoint_tables(frames, comparisons):
    rows = []

    def append(endpoint, facet, data, metrics, aggregation, unit):
        for method, values in data.groupby("method", sort=True):
            for metric in metrics:
                rows.append({"endpoint": endpoint, "facet": str(facet), "metric": metric, "method": method,
                             "mean": float(values[metric].mean()),
                             "sample_sd": float(values[metric].std()) if len(values) > 1 else None,
                             "n_summary_units": len(values), "n_seeds": int(values.seed.nunique()),
                             "summary_unit": unit, "aggregation": aggregation,
                             "higher_is_better": not (metric.startswith(("ece_", "brier_")) or metric.endswith("outer_known_fpr")),
                             "inference": "descriptive; no population significance or independent nested-case inference"})

    for endpoint, frame in frames.items():
        metrics, _ = endpoint_spec(endpoint, frame)
        if endpoint in ("lofo", "scarcity"):
            seed_means = frame.groupby(["seed", "method"])[metrics].mean().reset_index()
            append(endpoint, "all_nested_cases", seed_means, metrics,
                   "equal family/cap mean within seed, then equal mean across three seed summaries", "seed summary")
        elif endpoint == "alert_budget":
            for budget, values in frame.groupby("budget_id"):
                append(endpoint, budget, values, metrics, "equal mean across six frozen outer-test splits at this budget", "seed split")
        elif endpoint == "capture_windows":
            append(endpoint, "all_four_windows", frame, metrics, "equal window mean; one fitted pair and dependent file-order windows", "dependent capture window")
            for window, values in frame.groupby("window"):
                append(endpoint, window, values, metrics, "single capture window, one fitted pair", "capture window")
        else:
            append(endpoint, "all_six_splits", frame, metrics, "equal mean across six frozen outer-test splits", "seed split")
    summary = pd.DataFrame(rows)
    best = []
    for key, group in summary.groupby(["endpoint", "facet", "metric"]):
        index = group.set_index("method")
        candidates = group.loc[~group.method.isin(["taxonomy_blend", "legacy_flat_xgb"])].copy()
        higher = bool(group.higher_is_better.iloc[0])
        candidates = candidates.sort_values(["mean", "method"], ascending=[not higher, True], kind="stable")
        flat = candidates.loc[candidates.method.isin(run.STRONG)]
        winner, flat_winner = candidates.iloc[0], flat.iloc[0]
        taxonomy = float(index.loc["taxonomy_blend", "mean"])
        sign = 1 if higher else -1
        best.append(dict(zip(("endpoint", "facet", "metric"), key)) | {
            "taxonomy_blend_mean": taxonomy,
            "temperature_flat_blend_mean": float(index.loc[run.STRONG[0], "mean"]),
            "newton_flat_stacker_mean": float(index.loc[run.STRONG[1], "mean"]),
            "best_observed_competitor": winner.method, "best_observed_competitor_mean": float(winner["mean"]),
            "best_observed_competitor_ties": ";".join(candidates.loc[candidates["mean"].eq(winner["mean"]), "method"]),
            "taxonomy_benefit_vs_best_competitor": sign * (taxonomy - float(winner["mean"])),
            "best_strong_flat_control": flat_winner.method, "best_strong_flat_control_mean": float(flat_winner["mean"]),
            "taxonomy_benefit_vs_best_strong_flat": sign * (taxonomy - float(flat_winner["mean"])),
            "higher_is_better": higher,
            "scope": "metric-specific post-hoc descriptive comparison; no test-set tuning; legacy_flat_xgb is an audit-only arm"})
    per_seed = comparisons.copy()
    per_seed["facet"] = "all_six_splits"
    for endpoint in ("lofo", "scarcity"):
        per_seed.loc[per_seed.endpoint == endpoint, "facet"] = "all_nested_cases"
    for endpoint, column in (("alert_budget", "budget_id"), ("capture_windows", "window")):
        mask = per_seed.endpoint == endpoint
        per_seed.loc[mask, "facet"] = per_seed.loc[mask, column]
    keys = ["endpoint", "facet", "seed", "control", "metric"]
    metrics = ["taxonomy_value", "control_value", "taxonomy_minus_control", "taxonomy_benefit"]
    paired = per_seed.groupby(keys)[metrics].mean().reset_index()
    paired["within_seed_cases"] = per_seed.groupby(keys).size().to_numpy()
    paired["scope"] = "equal-case mean within seed for LOFO/scarcity; no independent-family/cap inference"
    return summary, pd.DataFrame(best), paired


def describe_nested(comparisons):
    subset = comparisons.loc[comparisons.endpoint.isin(["lofo", "scarcity"])]
    metrics = ["taxonomy_value", "control_value", "taxonomy_minus_control", "taxonomy_benefit"]
    seed = subset.groupby(["endpoint", "seed", "control", "metric"])[metrics].mean().reset_index()
    seed["nested_cases"] = subset.groupby(["endpoint", "seed", "control", "metric"]).size().to_numpy()
    expected = seed.endpoint.map({"lofo": 7, "scarcity": 6})
    if not seed.nested_cases.eq(expected).all():
        raise ValueError("A per-seed nested summary is incomplete")
    groups = subset.groupby(["endpoint", "case", "control", "metric"])[metrics].agg(["mean", "std", "count"])
    groups.columns = ["_".join(key) for key in groups.columns]
    directions = []
    for key, group in seed.groupby(["endpoint", "control", "metric"]):
        if len(group) != 3:
            raise ValueError("Exactly three complete seed summaries are required")
        values = group.taxonomy_benefit
        directions.append(dict(zip(("endpoint", "control", "metric"), key)) | {
            "taxonomy_better_seed_means": int((values > 0).sum()), "control_better_seed_means": int((values < 0).sum()),
            "tied_seed_means": int((values == 0).sum()), "seeds": 3,
            "mean_taxonomy_minus_control": float(group.taxonomy_minus_control.mean()),
            "sample_sd_of_seed_differences": float(group.taxonomy_minus_control.std()),
            "scope": "descriptive direction across three splits; families/caps are nested, not independent replications"})
    return seed, groups.reset_index(), pd.DataFrame(directions)


def csv_contract(path):
    frame = pd.read_csv(path)
    return {"path": str(path.resolve()), "rows": len(frame), "sha256": run.file_hash(path),
            "schema": [{"column": name, "dtype": str(dtype)} for name, dtype in frame.dtypes.items()]}


def protocol_contract(root):
    primary_dir = run.ROOT / "results" / "ijismd_v10_ciciot_grouped_core_6split"
    primary = json.loads((primary_dir / "run_metadata.json").read_text())
    primary_audit = pd.read_csv(primary_dir / "nested_split_audit.csv")
    primary_audit["rare_labels"] = primary_audit.rare_labels.map(json.loads)
    primary_audit["rare_count_threshold"] = np.maximum(1, np.ceil(primary_audit.base_pool_rows_before_cap * .01)).astype(int)
    capture = json.loads((root / "capture" / "protocol.json").read_text())
    counts = pd.read_csv(root / "capture" / "datasets" / "selected_label_counts.csv")
    source_n = int(counts.loc[counts.role == "base", "source_rows"].sum())
    cells = []
    for marker in sorted((root / "cells").glob("*/completed.json")):
        cell = json.loads(marker.read_text())
        cells.append({key: cell[key] for key in ("cell_id", "endpoint", "case", "seed", "fine_labels", "rare_ids",
                                                  "fit_pool_rows", "fit_rows", "calibration_rows", "selection_rows",
                                                  "known_outer_rows", "unknown_outer_rows")} | {
            "rare_labels": [cell["fine_labels"][index] for index in cell["rare_ids"]],
            "rare_count_threshold": max(1, int(np.ceil(cell["fit_pool_rows"] * .01))),
            "unknown_labels": sorted(set(primary["fine_labels"]) - set(cell["fine_labels"]))})
    return {
        "metadata_scope": "frozen protocol contract, not evidence of unrun-cell completion",
        "new_refit_config": json.loads(run.CONFIG_PATH.read_text()),
        "new_refit_config_sha256": run.file_hash(run.CONFIG_PATH),
        "new_refit_seeds": [42, 43, 44], "frozen_primary_seeds": list(range(42, 48)),
        "expected_endpoint_units": {"lofo": 21, "scarcity": 18, "calibration": 6, "alert_budget": 6, "capture_windows": 4},
        "expected_endpoint_method_rows": EXPECTED_ROWS, "expected_total_units": 55,
        "primary_grouped_arguments": primary["arguments"],
        "primary_class_labels": primary["fine_labels"], "primary_label_to_family": primary["label_to_family"],
        "primary_feature_columns": primary["features"], "primary_feature_count": len(primary["features"]),
        "primary_grouped_seed_roles_and_rarity": records(primary_audit),
        "new_refit_label_universe": "Scarcity: all 34 labels. LOFO: 34 minus the held-out family's labels; known-label macro-F1 does not score unknown-family labels as known classes.",
        "completed_refit_cell_label_and_rare_universes": cells,
        "rarity_formula": "positive class support <= max(1, ceil(0.01 * uncapped reference rows)); if no class qualifies, select up to eight least-supported present classes",
        "old_rare_definition_text_is_approximate": "Some cell metadata says 'below 1%'; the authoritative code has always used inclusive <= ceil threshold, not strict <.",
        "new_refit_rarity_reference": "Uncapped fit pool after removal of held-out LOFO family; never recomputed after scarcity cap",
        "capture_protocol": capture,
        "capture_rarity_source_rows": source_n,
        "capture_rarity_count_threshold": max(1, int(np.ceil(.01 * source_n))),
        "capture_rarity_reference": "source_rows per class in original base-file counts, not the selected 720747-row base sample",
        "capture_rare_labels": [capture["fine_labels"][index] for index in capture["rare_ids"]],
        "capture_method_ids": ["flat_xgb", "flat_lgbm", "flat_equal_blend", *run.STRONG,
                               "legacy_flat_xgb", "taxonomy_blend", "cyclic_blend", "support_matched_blend"],
        "fit_and_reuse_scope": {
            "lofo_scarcity": "Per-cell fresh flat XGB/LGBM and real/cyclic hierarchy bundles; derived blends and calibration-only temperature/stacker controls use exactly matched grouped rows",
            "calibration_alert_budget": "All ten arms recomputed from strictly aligned frozen probabilities; no new base or meta-model fitting",
            "capture_fresh_model_fitting": "Only flat XGB 520 trees, flat LGBM 380 trees, two temperatures and the converged Newton-Cholesky stacker; 39 features",
            "capture_fresh_result_arms": ["flat_xgb", "flat_lgbm", "flat_equal_blend", *run.STRONG],
            "capture_reused_hard_prediction_arms": ["legacy_flat_xgb", "taxonomy_blend", "cyclic_blend", "support_matched_blend"],
            "capture_reuse_checks": "Exact source SHA, all five dedup audit counts, full ordered labels and rare IDs match the original frozen windows; no refit of taxonomy/cyclic/support arms",
            "capture_comparison_boundary": "Matched inputs and scientific settings, not an exact common-fitted-component ablation: fresh two-thread flat predictions differ from archived legacy flat predictions. See capture_flat_reproduction_audit.csv.",
            "capture_all_arms_retrained": False},
        "capture_legacy_method_aliases": json.loads((root / "capture" / "legacy_method_audit.json").read_text()),
        "meta_model": {"fit_role": "utility/calibration only", "temperature_bounds": [.25, 5.0],
                       "temperature_log_bound_optimizer": "bounded minimize_scalar, xatol=1e-4, multiclass log loss",
                       "temperature_blend_weights": [.5, .5], "probability_clip_epsilon": run.flat.EPS,
                       "stacker_features": "concatenated log-normalized base probabilities", "stacker_C": 1.0,
                       "stacker_solver": "newton-cholesky", "stacker_max_iter": 100,
                       "convergence_warning_is_failure": True, "outer_test_tuning": False},
        "scarcity_label_budget": "base-fitting cap only; utility/calibration labels additional; not a total annotation-budget study",
        "uncertainty_and_ci_assumptions": {
            "confidence_intervals_computed": False, "reported_uncertainty": "sample SD, ddof=1, descriptive only",
            "lofo_scarcity_unit": "three seed summaries, each an equal-weight mean across its seven families or six caps",
            "core_e3_e4_unit": "six grouped outer-test split summaries; shared source observations are not independent population samples",
            "capture_unit": "four dependent file-order windows, one seed/fitted pair; any SD is window variability, not a four-replicate CI",
            "no_independent_21_18_fold_test": True, "no_sign_p_value": True,
            "no_bootstrap_or_normal_theory_ci_claim": True},
        "runtime_only_changes": "independent output roots, two-thread limits, resource admission, checkpoint ownership and atomic final merge; frozen model/sample parameters unchanged",
        "real_edge_device_available": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-only", action="store_true")
    args = parser.parse_args()
    root = run.require_physical_d(run.OUTPUT)
    if args.protocol_only:
        run.write_json(root / "PROTOCOL_CONTRACT.json", protocol_contract(root))
        print("PROTOCOL_CONTRACT.json written; no endpoint completion is asserted", flush=True)
        return
    coverage, verification = require_ready(root)
    frames = {endpoint: pd.read_csv(root / f"{endpoint}_results.csv") for endpoint in EXPECTED_ROWS}
    for endpoint, frame in frames.items():
        if len(frame) != EXPECTED_ROWS[endpoint] or frame.isna().any().any():
            raise ValueError(f"Missing or unexpected final numeric rows: {endpoint}")
        numeric = frame.select_dtypes("number")
        if not np.isfinite(numeric.to_numpy()).all():
            raise ValueError(f"Nonfinite numeric contract: {endpoint}")
    table4 = pd.read_csv(root / "table4_nine_arm_ece10.csv")
    if len(table4) != 9 or not table4.nbins.eq(10).all() or not table4.seeds.eq(6).all():
        raise ValueError("All nine Table 4 arms must have six-seed ten-bin ECE")
    retrieval = frames["alert_budget"].rename(columns={
        "rare_recall": "rare_row_retrieval_recall", "rare_precision": "rare_row_retrieval_precision"})
    run.write_csv(root / "alert_budget_retrieval_results.csv", retrieval)
    frames["alert_budget"] = retrieval
    comparisons = paired_comparisons(frames)
    seed, groups, directions = describe_nested(comparisons)
    all_summary, best_competitors, all_seed_pairs = all_endpoint_tables(frames, comparisons)
    outputs = {"all_endpoint_control_comparisons.csv": comparisons,
               "all_endpoint_summary.csv": all_summary,
               "best_competitor_by_endpoint_metric.csv": best_competitors,
               "all_endpoint_per_seed_paired_results.csv": all_seed_pairs,
               "nested_control_comparisons_per_seed.csv": seed,
               "nested_control_comparisons_per_family_or_cap.csv": groups,
               "nested_control_seed_directions.csv": directions}
    for name, frame in outputs.items():
        run.write_csv(root / name, frame)
    prose = []
    for row in directions.loc[directions.metric.isin(["macro_f1", "rare_recall", "msp_unknown_auroc"])].itertuples():
        metric = "class-balanced rare-class recall" if row.metric == "rare_recall" else row.metric.replace("_", " ")
        prose.append(f"For {row.endpoint.upper()} {metric}, the mean taxonomy-blend minus {DISPLAY[row.control]} "
                     f"difference across the three equal-case seed summaries was {row.mean_taxonomy_minus_control:.6f} "
                     f"(sample SD {row.sample_sd_of_seed_differences:.6f}); taxonomy-favouring/control-favouring "
                     f"seed directions were {row.taxonomy_better_seed_means}/{row.control_better_seed_means}. "
                     "Families or caps are nested within splits, so these directions are descriptive, not an independent-fold significance test.")
    core = comparisons.loc[comparisons.endpoint == "calibration"]
    core_summary = core.groupby(["control", "metric"])[["taxonomy_value", "control_value", "taxonomy_minus_control"]].agg(["mean", "std"])
    core_summary.columns = ["_".join(key) for key in core_summary.columns]
    budget = comparisons.loc[comparisons.endpoint == "alert_budget"]
    budget_summary = budget.groupby(["budget_id", "control", "metric"])[["taxonomy_value", "control_value", "taxonomy_minus_control"]].agg(["mean", "std"])
    budget_summary.columns = ["_".join(key) for key in budget_summary.columns]
    run.write_csv(root / "calibration_control_comparisons_summary.csv", core_summary.reset_index())
    run.write_csv(root / "alert_budget_control_comparisons_summary.csv", budget_summary.reset_index())
    for row in core_summary.reset_index().itertuples():
        prose.append(f"For {row.metric}, taxonomy blend averaged {row.taxonomy_value_mean:.6f} versus "
                     f"{row.control_value_mean:.6f} for the {DISPLAY[row.control]} across six frozen outer-test splits; lower is better.")
    prose.extend([
        "Calibration and alert-budget results were recomputed from the original v10 grouped probability archives, never compact hard-label ledgers. Frozen float16 probabilities are normalized consistently; ECE correctness uses their own argmax. Sensitivity to the separately saved canonical hard predictions is reported without switching the declared metric.",
        "Alert-budget recall is the fraction of true rare rows retrieved by the ranked alert list; it is not the unweighted mean of rare-class recalls. Budgets, scoring and tie order were fixed without outer-label tuning.",
        "Scarcity limits base-fitting labels per class only. Utility/calibration labels remain additional and fit temperatures and the converged stacker. It is not a total-annotation-budget experiment.",
        "Capture windows are deduplicated file-order windows with matched strong flat controls, not timestamp-based temporal validation or real edge-device measurements.",
        "Capture comparisons reuse archived taxonomy/cyclic/support composite hard predictions against fresh two-thread flat controls. Verified inputs, features, seed and tree budgets match, but fresh flat hard predictions are not identical to the archived flat component. This is not an exact common-fitted-component ablation; per-window drift is disclosed without refitting or replacing results.",
        "Legacy random-mirror LOFO/scarcity headline numbers are retired. Only newly paired grouped results may support revised extension claims."])
    names = [f"{name}_results.csv" for name in EXPECTED_ROWS] + [
        "lofo_per_seed_summary.csv", "scarcity_per_seed_summary.csv", "lofo_summary.csv", "scarcity_summary.csv",
        "paired_deltas_per_seed.csv", "paired_deltas_per_family_or_cap.csv", "paired_seed_direction_consistency.csv",
        "nested_aggregation_audit.csv",
        "frozen_source_hash_audit.csv", "frozen_ece_probability_vs_hard_prediction_sensitivity.csv",
        "compact_support_gate_seed42_differences.csv",
        "capture_flat_reproduction_audit.csv",
        "calibration_summary.csv", "alert_budget_summary.csv", "capture_paired_deltas.csv", "table4_nine_arm_ece10.csv",
        "alert_budget_retrieval_results.csv", "calibration_control_comparisons_summary.csv", "alert_budget_control_comparisons_summary.csv", *outputs]
    handoff = {
        "completion": True, "completed_endpoint_categories": 5, "completed_units": 55, "expected_units": 55,
        "endpoint_method_rows": 1107, "strong_control_unit_evaluations": 110,
        "coverage": records(coverage), "verification": verification,
        "canonical_frozen_probability_source_audit": json.loads((root / "frozen_source_audit.json").read_text()),
        "capture_flat_reproduction_audit": json.loads((root / "capture_flat_reproduction_audit.json").read_text()),
        "calibration_metric_protocol": json.loads((root / "calibration_metric_protocol.json").read_text()),
        "frozen_protocol_and_class_universes": protocol_contract(root),
        "csv_contracts": {name: csv_contract(root / name) for name in names},
        "per_seed_nested_results": records(seed), "per_family_or_cap_results": records(groups),
        "all_endpoint_summary": records(all_summary), "best_competitor_by_endpoint_metric": records(best_competitors),
        "seed_direction_consistency": records(directions),
        "table4_all_nine_arms_ece10": records(table4),
        "calibration_comparisons": records(core_summary.reset_index()),
        "alert_budget_comparisons_by_budget": records(budget_summary.reset_index()),
        "capture_comparisons_by_window": records(comparisons.loc[comparisons.endpoint == "capture_windows"]),
        "reversals_control_better": records(comparisons.loc[comparisons.direction == "control_better"]),
        "metric_semantics": {
            "classification_rare_recall": "unweighted mean recall over rare class IDs fixed from the uncapped fit pool",
            "alert_budget_raw_rare_recall": "rare_hits / true_rare_rows; row-retrieval metric, NOT class-balanced recall",
            "alert_budget_raw_rare_precision": "rare_hits / alert_budget; row-retrieval precision",
            "ece": "10 equal-width right-closed bins; zero included in first bin; row-share weighting",
            "no_independent_21_or_18_replication_inference": True, "no_sign_p_values": True},
        "review_facing_prose": prose,
        "protocol_boundaries": {
            "frozen_parameters_unchanged": True, "calibrator_fit_role": "utility/calibration only",
            "lofo_threshold_fit_role": "selection only; strict exceedance of higher quantile",
            "outer_test_tuning": False, "scarcity_calibration_labels_additional": True,
            "capture_is_file_order_proxy": True, "real_edge_device_available": False,
            "legacy_unmatched_headlines_retired": True},
    }
    run.write_json(root / "MANUSCRIPT_HANDOFF.json", handoff)
    print("MANUSCRIPT_HANDOFF.json published: all five categories, 55 units, 1107 endpoint-method rows", flush=True)


if __name__ == "__main__":
    main()
