"""Derive Euclid-compatible 10-bin ECE without modifying frozen result ledgers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
import run_harmonized_refit as run
import numpy as np
import pandas as pd

E3_SOURCE = Path(r"workspace\extensions\scripts\run_matched_extensions.py")


def ece10(y, probability):
    """Input is already normalized, as in Euclid's frozen endpoint."""
    confidence = probability.max(axis=1)
    correct = probability.argmax(axis=1) == y
    bins = np.clip(np.digitize(confidence, np.linspace(0, 1, 11), right=True) - 1, 0, 9)
    return float(sum(np.mean(bins == b) * abs(correct[bins == b].mean() - confidence[bins == b].mean())
                     for b in range(10) if np.any(bins == b)))


def bin_audit(y, probability, seed, method, population):
    confidence = probability.max(axis=1)
    correct = probability.argmax(axis=1) == y
    rows = []
    reference = 0.0
    edges = np.linspace(0, 1, 11)
    for b, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        # Independent interval expression, including exact zero in the first bin.
        mask = (confidence >= lower if b == 0 else confidence > lower) & (confidence <= upper)
        n = int(mask.sum())
        share = float(mask.mean())
        mean_confidence = float(confidence[mask].mean()) if n else None
        accuracy = float(correct[mask].mean()) if n else None
        contribution = share * abs(accuracy - mean_confidence) if n else 0.0
        reference += contribution
        rows.append({"seed": seed, "run_id": method, "population": population,
                     "bin": b, "lower": lower, "upper": upper, "rows": n,
                     "population_rows": len(y), "row_share": share,
                     "mean_confidence": mean_confidence, "accuracy": accuracy,
                     "weighted_absolute_gap": contribution})
    assert sum(item["rows"] for item in rows) == len(y)
    return reference, rows


def validate_metric_keys(frame, seeds, methods, allow_incomplete):
    assert len(set(seeds)) == len(seeds) and len(set(methods)) == len(methods)
    assert not frame.duplicated(["seed", "run_id"]).any(), "Duplicate ECE seed x method keys"
    observed = set(frame[["seed", "run_id"]].itertuples(index=False, name=None))
    expected = {(seed, method) for seed in seeds for method in methods}
    assert observed <= expected, "Unexpected ECE seed x method key"
    complete = observed == expected
    assert complete or allow_incomplete, "Missing ECE seed x method keys"
    return complete


def derive_protocol(protocol, allow_incomplete):
    output = run.WORK / "results" / protocol
    contract = json.loads((output / "run_contract.json").read_text())
    rows, bins, provenance = [], [], []
    original_hashes = {}
    for seed in run.SEEDS:
        directory = output / f"seed_{seed}"
        for method in contract["config"]["methods"]:
            metric = directory / f"metrics_{method}.csv"
            prediction = directory / f"predictions_{method}.npz"
            if not metric.exists():
                assert allow_incomplete, str(metric)
                continue
            original_hashes[str(metric)] = run.sha256(metric)
            original = pd.read_csv(metric, float_precision="round_trip")
            assert len(original) == 1, f"Expected one frozen metric row: {metric}"
            row = original.iloc[0].to_dict()
            assert row["seed"] == seed and row["run_id"] == method, str(metric)
            with np.load(prediction) as saved:
                y = saved["y_true"]
                probability = run.comparator.normalise(saved["probability"].astype(float))
                rare_mask = np.isin(y, saved["rare_ids"])
            row["ece_15"] = row["ece_top_label"]
            row["rare_ece_15"] = row["rare_ece_top_label"]
            row["ece_10"] = ece10(y, probability)
            row["rare_ece_10"] = ece10(y[rare_mask], probability[rare_mask])
            for population, selected, value in (("all", np.ones(len(y), dtype=bool), row["ece_10"]),
                                                 ("rare_rows", rare_mask, row["rare_ece_10"])):
                reference, diagnostic = bin_audit(y[selected], probability[selected], seed, method, population)
                assert np.isclose(reference, value, atol=1e-15, rtol=0)
                bins.extend(diagnostic)
            rows.append(row)
            provenance.append({"seed": seed, "run_id": method, "prediction_path": str(prediction),
                               "prediction_sha256": run.sha256(prediction),
                               "original_metric_sha256": original_hashes[str(metric)]})
    if not rows:
        return {"protocol": protocol, "status": "INCOMPLETE", "rows": 0}
    frame = pd.DataFrame(rows).sort_values(["seed", "run_id"])
    expected = len(run.SEEDS) * len(contract["config"]["methods"])
    complete = validate_metric_keys(frame, run.SEEDS, contract["config"]["methods"], allow_incomplete)
    assert np.isfinite(frame[["ece_10", "rare_ece_10"]].to_numpy(dtype=float)).all()
    suffix = "" if complete else "_partial"
    run.csv_write(output / f"harmonized_results_ece10{suffix}.csv", frame)
    summary = frame.groupby("run_id").agg(
        seeds=("seed", "nunique"), f1_macro_mean=("f1_macro", "mean"), f1_macro_sd=("f1_macro", "std"),
        rare_recall_mean=("rare_recall", "mean"), rare_recall_sd=("rare_recall", "std"),
        ece_10_mean=("ece_10", "mean"), ece_10_sd=("ece_10", "std"),
        rare_ece_10_mean=("rare_ece_10", "mean"), rare_ece_10_sd=("rare_ece_10", "std"),
        ece_15_mean=("ece_15", "mean"), ece_15_sd=("ece_15", "std"),
        rare_ece_15_mean=("rare_ece_15", "mean"), rare_ece_15_sd=("rare_ece_15", "std")).reset_index()
    run.csv_write(output / f"harmonized_summary_ece10{suffix}.csv", summary)
    run.csv_write(output / f"ece10_reliability_bins{suffix}.csv", pd.DataFrame(bins))
    run.csv_write(output / f"ece10_prediction_provenance{suffix}.csv", pd.DataFrame(provenance))
    for path, digest in original_hashes.items():
        assert run.sha256(path) == digest, f"Frozen metric file modified: {path}"
    return {"protocol": protocol, "status": "COMPLETE_PASS" if complete else "INCOMPLETE",
            "rows": len(frame), "expected_rows": expected,
            "duplicate_seed_method_keys": 0,
            "exact_frozen_seed_method_key_set": complete,
            "explicit_ece10_fields_finite": True,
            "frozen_per_method_metric_files_unchanged": True,
            "independent_interval_check_tolerance": 1e-15}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    run.require_physical_d_output(run.WORK)
    results = [derive_protocol(protocol, args.allow_incomplete) for protocol in ("paired46", "grouped37")]
    report = {"results": results, "elapsed_seconds": time.perf_counter() - started,
              "ece_10_definition": "np.clip(np.digitize(confidence,np.linspace(0,1,11),right=True)-1,0,9); bin weight=mask.mean()",
              "probability_preparation": "v10 comparator normalise(saved float64 probabilities), identical to Euclid aligned_probability",
              "reference_ece_15_definition": "unchanged comparator default 15 bins, left-closed except inclusive final upper edge",
              "reference_frozen_endpoint_source": str(E3_SOURCE),
              "reference_frozen_endpoint_source_sha256": run.sha256(E3_SOURCE),
              "script_sha256": run.sha256(__file__),
              "model_refitting_or_selection": False}
    run.json_write(run.WORK / "ece10_derivation_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
