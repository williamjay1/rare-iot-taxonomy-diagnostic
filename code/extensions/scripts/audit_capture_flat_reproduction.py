"""Describe fresh-versus-archived flat drift without refitting or changing metrics."""

import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import joblib
import numpy as np
import pandas as pd

import run_capture_controls as capture
import run_matched_extensions as run


def main():
    root = run.require_physical_d(run.OUTPUT)
    folder = root / "capture"
    protocol = json.loads((folder / "protocol.json").read_text())
    historical = json.loads((folder / "frozen" / "run_metadata.json").read_text())
    arguments = historical["arguments"]
    if (int(arguments["seed"]) != capture.SEED or int(arguments["global_trees"]) != 520
            or int(arguments["local_trees"]) != 380):
        raise ValueError("Historical capture seed/tree settings differ")
    run.trf.RANDOM_STATE = capture.SEED
    current = joblib.load(folder / "xgb.joblib")
    factory = run._global_factory(len(protocol["fine_labels"]), 520)
    actual = {key: repr(value) for key, value in current.get_params().items()}
    original_factory = {key: repr(value) for key, value in factory.get_params().items()}
    differences = {key: {"fresh": actual.get(key), "historical_source_factory": original_factory.get(key)}
                   for key in actual.keys() | original_factory.keys()
                   if actual.get(key) != original_factory.get(key)}
    if set(differences) != {"n_jobs"} or current.n_features_in_ != 39:
        raise ValueError(f"Unexpected capture factory/feature discrepancy: {differences}")
    if not all(protocol["package_versions"][name] == value
               for name, value in historical["package_versions"].items()):
        raise ValueError("Historical capture package versions differ")
    rows = []
    for window in capture.WINDOWS:
        complete = json.loads((folder / window / "completed.json").read_text())
        labels, predictions = [], []
        for part in sorted((folder / window).glob("part_*.npz")):
            with np.load(part) as data:
                labels.append(data["y_true"])
                predictions.append(data["flat_xgb"])
        y, fresh = np.concatenate(labels), np.concatenate(predictions)
        with np.load(folder / "frozen" / f"window_predictions_{window}.npz") as source:
            np.testing.assert_array_equal(y, source["y_true"])
            legacy = source["flat_weighted_xgboost"]
            changed = int(np.count_nonzero(fresh != legacy))
        if len(y) != complete["rows"] or run.array_hash(y) != complete["ordered_y_sha256"]:
            raise ValueError("Capture reproduction audit row identity mismatch")
        agreement = float(np.mean(fresh == legacy))
        np.testing.assert_allclose(agreement, complete["flat_xgb_agreement_with_original"], rtol=0, atol=1e-15)
        results = pd.read_csv(folder / window / "results.csv").set_index("method")
        row = {"window": window, "rows": len(y), "differing_hard_predictions": changed,
               "agreement": agreement, "same_fitted_component": changed == 0}
        for metric in ("macro_f1", "rare_recall"):
            row[f"fresh_flat_xgb_{metric}"] = float(results.loc["flat_xgb", metric])
            row[f"legacy_flat_xgb_{metric}"] = float(results.loc["legacy_flat_xgb", metric])
            row[f"fresh_minus_legacy_{metric}"] = row[f"fresh_flat_xgb_{metric}"] - row[f"legacy_flat_xgb_{metric}"]
        rows.append(row)
    run.write_csv(root / "capture_flat_reproduction_audit.csv", pd.DataFrame(rows))
    run.write_json(root / "capture_flat_reproduction_audit.json", {
        "status": "passed_with_nonidentical_predictions", "windows_checked": len(rows),
        "protocol_sha256": protocol["protocol_sha256"], "window_results": rows,
        "fresh_model_parameters_repr": actual, "source_factory_parameters_repr": original_factory,
        "source_factory_parameter_differences": differences,
        "historical_serialized_estimator_available": False,
        "parameter_comparison_scope": "Current fresh fitted estimator versus the read-only historical source factory at the recorded seed/tree settings; not an unavailable serialized historical estimator",
        "package_versions_match_historical_metadata": True,
        "strict_identical_fitted_flat_component_claim_supported": False,
        "cause": "Not established. Two-thread fresh fitting versus historical source n_jobs=-1 is a documented runtime difference, not proof of the cause.",
        "interpretation": "Fresh two-thread flat controls share verified inputs, features, labels, seed and tree budgets with the archived capture protocol. Taxonomy/cyclic/support composite hard predictions are archived, not refitted, and contain a nonidentical legacy flat component. These are matched-input exploratory comparisons, not an exact common-fitted-component ablation.",
        "no_refit_or_metric_replacement": True,
    })
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
