"""Independently verify saved roles, worker limits, checkpoints and metrics."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
import run_harmonized_refit as run
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def check_workers(value):
    if isinstance(value, dict):
        if "params" in value and "n_jobs" in value["params"]:
            if value.get("type") == "LogisticRegression":
                parameters = value["params"]
                expected = run.newton.LogisticRegression(C=1.0, max_iter=100,
                    solver="newton-cholesky", random_state=parameters["random_state"]).get_params(deep=False)
                assert parameters == expected and parameters["random_state"] in run.SEEDS, value
            else:
                assert value["params"]["n_jobs"] == 2, value
        for item in value.values():
            check_workers(item)
    elif isinstance(value, list):
        for item in value:
            check_workers(item)


def verify(protocol, allow_incomplete=False):
    output = run.WORK / "results" / protocol
    run.require_physical_d_output(output)
    contract = json.loads((output / "run_contract.json").read_text())
    config = contract["config"]
    assert config["seeds"] == run.SEEDS and config["n_jobs"] == 2
    assert len(contract["features"]) == 37
    assert contract["package_versions"] == run.benefit.package_versions()
    cache_paths = [Path(item["training_cache_path"]) for item in contract["inputs"]]
    actual_groups = None
    if protocol == "grouped37":
        inputs = pd.concat([pd.read_parquet(path, columns=contract["features"]) for path in cache_paths], ignore_index=True)
        cleaned = pd.DataFrame(run.trf.clean_features(inputs, contract["features"]), columns=contract["features"])
        actual_groups = pd.util.hash_pandas_object(cleaned, index=False).to_numpy(dtype=np.uint64)
        del inputs, cleaned
        collisions = json.loads((run.WORK / "projection_collision_audit.json").read_text())["global"]
        assert len(np.unique(actual_groups)) == collisions["shared37_unique_groups"]
    full = pd.concat([pd.read_parquet(path, columns=["Label"]) for path in cache_paths], ignore_index=True)
    encoder = LabelEncoder().fit(full["Label"].astype(str))
    labels = encoder.transform(full["Label"].astype(str))
    method_count = 0
    completed = []
    missing = []
    group_pairs = []
    stacker_audits = []
    for seed in run.SEEDS:
        directory = output / f"seed_{seed}"
        if not (directory / "role_row_ids.npz").exists():
            missing.append(f"seed_{seed}/roles")
            continue
        proof = json.loads((directory / "data_role_proof.json").read_text())
        with np.load(directory / "role_row_ids.npz") as values:
            roles = {name: values[f"{name}_row_id"] for name in ("fit_pool", "utility", "selection", "test")}
            fit = values["fit_ordered_row_id"]
            rare = values["rare_ids"]
        partition = np.concatenate(list(roles.values()))
        assert len(partition) == len(full) and len(np.unique(partition)) == len(full), seed
        assert partition.min() == 0 and partition.max() == len(full) - 1
        assert np.isin(fit, roles["fit_pool"]).all() and len(np.unique(fit)) == len(fit)
        assert run.array_hash(fit) == proof["fit_order_sha256"]
        for name, rows in roles.items():
            assert run.array_hash(rows) == proof["role_order_sha256"][name]
        if actual_groups is not None:
            role_groups = {name: np.unique(actual_groups[rows]) for name, rows in roles.items()}
            for left, right in itertools.combinations(roles, 2):
                count = len(np.intersect1d(role_groups[left], role_groups[right], assume_unique=True))
                assert count == 0, (seed, left, right, count)
                group_pairs.append({"seed": seed, "left": left, "right": right,
                                    "independently_recomputed_overlap_groups": count})
        fit_counts = full.iloc[fit]["Label"].astype(str).value_counts()
        for label, n in fit_counts.items():
            assert n <= (80000 if label == "BenignTraffic" else 25000)
        expected_rare = run.trf.rare_class_ids(labels[roles["fit_pool"]])
        assert np.array_equal(rare, expected_rare)
        overlap = pd.read_csv(directory / "predictor_group_overlap.csv")
        basis = "original46" if protocol == "paired46" else "shared37_float32_model_input"
        assert (overlap.loc[overlap.basis == basis, "overlap_groups"] == 0).all()
        if protocol == "paired46":
            with np.load(run.CORE / f"paired_predictions_seed_{seed}.npz") as original:
                assert np.array_equal(roles["test"], original["row_id"])
                assert np.array_equal(labels[roles["test"]], original["y_true"])
                assert np.array_equal(rare, original["rare_ids"])
        for model in directory.glob("*.joblib"):
            information = json.loads(model.with_name(model.name + ".json").read_text())
            assert run.sha256(model) == information["sha256"], str(model)
            check_workers(information["models"])
            if information["models"].get("type") == "LogisticRegression":
                stacker = run.joblib.load(model)
                assert stacker.get_params(deep=False) == information["models"]["params"]
                assert (stacker.n_iter_ > 0).all() and (stacker.n_iter_ < 100).all()
                assert stacker.n_features_in_ == 68
                assert np.array_equal(stacker.classes_, np.arange(len(encoder.classes_)))
                assert np.isfinite(stacker.coef_).all() and np.isfinite(stacker.intercept_).all()
                stacker_audits.append({"seed": seed, "n_iter": stacker.n_iter_.tolist(),
                                      "n_features_in": 68, "solver": stacker.solver,
                                      "n_jobs": stacker.n_jobs, "runtime_threadpool_limit": 2,
                                      "fit_role": "utility", "checkpoint_sha256": information["sha256"]})
        for method in config["methods"]:
            path = directory / f"metrics_{method}.csv"
            if not path.exists():
                missing.append(f"seed_{seed}/{method}")
                continue
            row = pd.read_csv(path).iloc[0]
            with np.load(directory / f"predictions_{method}.npz") as saved:
                assert np.array_equal(saved["row_id"], roles["test"])
                assert np.array_equal(saved["y_true"], labels[roles["test"]])
                assert np.array_equal(saved["rare_ids"], rare)
                probability = saved["probability"]
                y = saved["y_true"]
            assert probability.dtype == np.float64 and probability.shape == (len(y), len(encoder.classes_))
            assert np.isfinite(probability).all() and probability.min() >= 0
            assert np.max(np.abs(probability.sum(axis=1) - 1)) < 1e-5
            recomputed = run.trf.metric_row(method, y, probability, rare)
            recomputed["brier_multiclass"] = run.comparator.multiclass_brier(y, probability)
            recomputed["ece_top_label"] = run.comparator.expected_calibration_error(y, probability)
            recomputed["rare_ece_top_label"] = run.comparator.expected_calibration_error(y[np.isin(y, rare)], probability[np.isin(y, rare)])
            for key in ("accuracy", "f1_macro", "rare_recall", "recall_macro", "worst_recall", "pr_auc_macro", "brier_multiclass", "ece_top_label", "rare_ece_top_label"):
                assert np.isclose(float(row[key]), float(recomputed[key]), atol=1e-12, rtol=0), (seed, method, key)
            method_count += 1
        if (directory / "COMPLETED.json").exists():
            complete = json.loads((directory / "COMPLETED.json").read_text())
            assert complete["methods"] == config["methods"]
            assert complete["run_contract_sha256"] == run.sha256(output / "run_contract.json")
            completed.append(seed)
    if not allow_incomplete:
        assert not missing, missing
        assert completed == run.SEEDS
        assert method_count == len(run.SEEDS) * len(config["methods"])
        table = pd.read_csv(output / "harmonized_results.csv")
        assert len(table) == method_count and not table.duplicated(["seed", "run_id"]).any()
    if group_pairs:
        run.csv_write(output / "independent_actual37_role_overlap.csv", pd.DataFrame(group_pairs))
    return {"protocol": protocol, "status": "COMPLETE_PASS" if not missing and completed == run.SEEDS else "INCOMPLETE",
            "completed_seeds": completed, "verified_method_rows": method_count,
            "actual_float32_groups_independently_recomputed": actual_groups is not None,
            "independently_verified_group_role_pairs": len(group_pairs),
            "stacker_convergence_audits": stacker_audits,
            "expected_method_rows": len(run.SEEDS) * len(config["methods"]), "missing": missing}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--protocol", choices=["paired46", "grouped37", "both"], default="both")
    args = parser.parse_args()
    started = time.perf_counter()
    results = [verify(name, args.allow_incomplete) for name in (["paired46", "grouped37"] if args.protocol == "both" else [args.protocol])]
    output = {"results": results, "elapsed_seconds": time.perf_counter() - started}
    run.json_write(run.WORK / "verification_report.json", output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
