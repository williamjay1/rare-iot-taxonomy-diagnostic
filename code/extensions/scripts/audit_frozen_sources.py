"""Audit canonical frozen probability sources without changing any endpoint result."""

from __future__ import annotations

import json
from pathlib import Path

import run_matched_extensions as run

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


SOURCES = {
    "core": ("ijismd_v10_ciciot_grouped_core_6split", {
        "flat_xgb": "flat_weighted_xgboost",
        "real_taxonomy": "soft_taxonomy_factorisation",
        "taxonomy_blend": "equal_flat_taxonomy_blend",
        "taxonomy_benefit_gate": "calibrated_taxonomy_benefit_gate",
        "cyclic_gate": "calibrated_cyclic_taxonomy_control",
        "support_matched_gate": "calibrated_support_matched_control",
    }),
    "flat_comparators": ("ijismd_v10_grouped_flat_comparators_6split", {
        "flat_lgbm": "flat_weighted_lightgbm",
        "flat_equal_blend": "flat_equal_probability_blend",
        "temperature_flat_blend": "flat_temperature_calibrated_blend",
    }),
    "stacker": ("ijismd_v10_grouped_newton_flat_stacker_6split", {
        "newton_flat_stacker": "flat_newton_log_probability_stacker",
    }),
}


def ece_with_hard_labels(y, probability, hard):
    confidence = probability.max(axis=1)
    correct = hard == y
    bins = np.clip(np.digitize(confidence, np.linspace(0, 1, 11), right=True) - 1, 0, 9)
    return float(sum(np.mean(bins == b) * abs(correct[bins == b].mean() - confidence[bins == b].mean())
                     for b in range(10) if np.any(bins == b)))


def main():
    root = run.require_physical_d(run.OUTPUT)
    calibration = pd.read_csv(root / "calibration_results.csv").set_index(["seed", "method"])
    budget = pd.read_csv(root / "alert_budget_results.csv").set_index(["seed", "method"]).sort_index()
    parent_path = root.parent / "primary_audit" / "canonical_v10" / "input_manifest.csv"
    parent = pd.read_csv(parent_path)
    parent_hashes = {str(Path(row.source).resolve()): row.sha256 for row in parent.itertuples()}
    source_rows, sensitivity = [], []
    budget_checks = 0
    for seed in range(42, 48):
        saved = json.loads((root / "frozen" / f"seed{seed}" / "completed.json").read_text())
        saved_hashes = {str(Path(path).resolve()): sha for path, sha in saved["source_sha256"].items()}
        anchor_path = run.ROOT / "results" / SOURCES["core"][0] / f"paired_predictions_seed_{seed}.npz"
        with np.load(anchor_path) as anchor:
            y, rare = anchor["y_true"].astype(int), anchor["rare_ids"].astype(int)
            rare_mask = np.isin(y, rare)
            for kind, (directory, methods) in SOURCES.items():
                path = (run.ROOT / "results" / directory / f"paired_predictions_seed_{seed}.npz").resolve()
                sha = run.file_hash(path)
                if sha != saved_hashes[str(path)]:
                    raise ValueError(f"Frozen source changed: {path}")
                if kind in ("core", "stacker") and sha != parent_hashes[str(path)]:
                    raise ValueError(f"Source differs from parent canonical manifest: {path}")
                source_rows.append({"seed": seed, "source_kind": kind, "raw_source": str(path),
                                    "sha256_now": sha, "sha256_recorded_at_computation": saved_hashes[str(path)],
                                    "matches_computation_manifest": True,
                                    "parent_canonical_sha256": parent_hashes.get(str(path), "not listed; flat comparator source"),
                                    "compact_used": False})
                with np.load(path) as data:
                    for method, key in methods.items():
                        p = run.aligned_probability(data, anchor, key + "_probability")
                        calculated = run.calibration_metrics(y, p, rare)
                        expected = calibration.loc[(seed, method)]
                        for metric, value in calculated.items():
                            np.testing.assert_allclose(value, expected[metric], rtol=0, atol=1e-12)
                        hard = data[key] if key in data.files else None
                        mismatch = p.argmax(axis=1) != hard if hard is not None else None
                        hard_ece = ece_with_hard_labels(y, p, hard) if hard is not None else None
                        hard_rare_ece = ece_with_hard_labels(y[rare_mask], p[rare_mask], hard[rare_mask]) if hard is not None else None
                        sensitivity.append({
                            "seed": seed, "method": method, "source_kind": kind, "rows": len(y),
                            "probability_dtype_on_disk": str(data[key + "_probability"].dtype),
                            "probability_sha256": run.array_hash(data[key + "_probability"]),
                            "canonical_hard_prediction_available": hard is not None,
                            "canonical_hard_prediction_sha256": run.array_hash(hard) if hard is not None else None,
                            "probability_argmax_vs_canonical_hard_differences": int(mismatch.sum()) if hard is not None else None,
                            "rare_row_hard_differences": int((mismatch & rare_mask).sum()) if hard is not None else None,
                            "ece10_declared_probability_argmax": calculated["ece_all"],
                            "ece10_sensitivity_canonical_hard": hard_ece,
                            "ece10_sensitivity_minus_declared": hard_ece - calculated["ece_all"] if hard is not None else None,
                            "rare_ece10_declared_probability_argmax": calculated["ece_rare_rows"],
                            "rare_ece10_sensitivity_canonical_hard": hard_rare_ece,
                            "rare_ece10_sensitivity_minus_declared": hard_rare_ece - calculated["ece_rare_rows"] if hard is not None else None,
                            "all_four_calibration_metrics_reproduced": True,
                        })
                        order = np.argsort(-p[:, rare].max(axis=1), kind="stable")
                        for row in budget.loc[(seed, method)].itertuples():
                            hits = int(rare_mask[order[:int(row.budget)]].sum())
                            if hits != row.rare_hits:
                                raise ValueError(f"Frozen alert budget differs: seed{seed}/{method}/{row.budget_id}")
                            np.testing.assert_allclose([hits / rare_mask.sum(), hits / row.budget],
                                                       [row.rare_recall, row.rare_precision], rtol=0, atol=1e-12)
                            budget_checks += 1
        print(f"Audited canonical frozen seed{seed}: 10 calibration arms and 110 budget rows", flush=True)
    compact_path = (run.ROOT / "results" / "jit_compact_prediction_ledgers_v1" /
                    "ciciot2023_public_subsample" / "paired_predictions_seed_42.npz")
    canonical_path = run.ROOT / "results" / SOURCES["core"][0] / "paired_predictions_seed_42.npz"
    key = "calibrated_support_matched_control"
    with np.load(canonical_path) as canonical, np.load(compact_path) as compact:
        for name in ("row_id", "y_true", "rare_ids"):
            np.testing.assert_array_equal(canonical[name], compact[name])
        changed = canonical[key] != compact[key]
        p = run.flat.normalise(canonical[key + "_probability"].astype(float))
        run.write_csv(root / "compact_support_gate_seed42_differences.csv", pd.DataFrame({
            "row_id": canonical["row_id"][changed], "y_true": canonical["y_true"][changed],
            "canonical_hard": canonical[key][changed], "compact_hard": compact[key][changed],
            "canonical_probability_argmax": p.argmax(axis=1)[changed],
            "canonical_probability_confidence": p.max(axis=1)[changed],
        }))
        compact_audit = {"path": str(compact_path.resolve()), "sha256": run.file_hash(compact_path),
                         "used_for_extension_calibration_or_budget": False,
                         "has_support_gate_probability": key + "_probability" in compact.files,
                         "support_gate_hard_differences_vs_canonical": int(changed.sum())}
    sources, sensitive = pd.DataFrame(source_rows), pd.DataFrame(sensitivity)
    run.write_csv(root / "frozen_source_hash_audit.csv", sources)
    run.write_csv(root / "frozen_ece_probability_vs_hard_prediction_sensitivity.csv", sensitive)
    support = sensitive.loc[sensitive.method == "support_matched_gate"]
    report = {
        "status": "passed", "source_files_checked": len(sources), "calibration_rows_reproduced": len(sensitive),
        "alert_budget_rows_reproduced": budget_checks, "config_sha256": run.file_hash(run.CONFIG_PATH),
        "parent_canonical_manifest": str(parent_path), "parent_manifest_sha256": run.file_hash(parent_path),
        "all_core_and_stacker_source_hashes_match_parent": True, "compact_seed42_audit": compact_audit,
        "ece_definition_unchanged": "ten right-closed equal-width bins; correctness is argmax of the same normalized frozen probability row",
        "canonical_probability_precision": "original v10 float16 archives, not recovered full-precision predictions",
        "sensitivity_only_not_a_metric_switch": True,
        "support_gate_ece10_mean_declared": float(support.ece10_declared_probability_argmax.mean()),
        "support_gate_ece10_mean_canonical_hard_sensitivity": float(support.ece10_sensitivity_canonical_hard.mean()),
        "support_gate_rare_ece10_mean_declared": float(support.rare_ece10_declared_probability_argmax.mean()),
        "support_gate_rare_ece10_mean_canonical_hard_sensitivity": float(support.rare_ece10_sensitivity_canonical_hard.mean()),
        "impact": "The discrepant compact hard labels cannot affect these endpoints: compact was never read for calibration/budget. Float16 argmax versus stored canonical hard prediction is separately tabulated. Brier scores and alert ordering use probabilities and do not depend on either saved hard-label vector.",
    }
    run.write_json(root / "frozen_source_audit.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
