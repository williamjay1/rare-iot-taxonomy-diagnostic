"""Bounded layout-only prediction audit using an existing frozen checkpoint."""
from __future__ import annotations

import hashlib
import argparse
import json
import sys
import time

sys.dont_write_bytecode = True

import run_harmonized_refit as run
import joblib
import numpy as np
import pyarrow.parquet as pq
from sklearn.preprocessing import LabelEncoder
from threadpoolctl import threadpool_limits


def digest(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-check", action="store_true")
    args = parser.parse_args()
    run.require_physical_d_output(run.WORK)
    output = run.WORK / "results" / "paired46"
    contract = json.loads((output / "run_contract.json").read_text())
    directory = output / "seed_42"
    path = directory / "taxonomy_real.joblib"
    original_hash = run.sha256(path)
    assert original_hash == json.loads(path.with_name(path.name + ".json").read_text())["sha256"]
    batch = next(pq.ParquetFile(contract["inputs"][0]["training_cache_path"])
                 .iter_batches(batch_size=8192 if args.batch_check else 2048,
                               columns=contract["features"], use_threads=False))
    # No labels are read from the input. The frozen mapping defines class order.
    frame = batch.to_pandas(use_threads=False)
    proof = json.loads((directory / "data_role_proof.json").read_text())
    mapping = proof["mapping"]["real"]
    encoder = LabelEncoder().fit(list(mapping))
    values = run.trf.clean_features(frame, contract["features"])
    layouts = {"F": np.asfortranarray(values), "C": np.ascontiguousarray(values)}
    assert np.array_equal(layouts["F"], layouts["C"])
    assert values.dtype == np.float32
    model = joblib.load(path)
    params_before = run.model_parameters(model)
    reference = None
    rows = []
    with threadpool_limits(limits=2):
        cases = [("F", len(values)), ("F", 2048), ("F", 512)] if args.batch_check else [(order, len(values)) for order in ("F", "C", "C", "F")]
        for order, chunk_rows in cases:
            started = time.perf_counter()
            chunks = []
            for start in range(0, len(values), chunk_rows):
                chunks.append(run.trf.hierarchy_probabilities(
                    np.asarray(layouts[order][start:start + chunk_rows], order=order),
                    model["family_model"], model["family_encoder"],
                    encoder, mapping, model["local_models"]))
            probability = tuple(np.concatenate([part[index] for part in chunks], axis=0)
                                for index in range(2))
            elapsed = time.perf_counter() - started
            if reference is None:
                reference = probability
            differences = [float(np.max(np.abs(a - b))) for a, b in zip(reference, probability)]
            equal = [np.array_equal(a, b) for a, b in zip(reference, probability)]
            rows.append({"layout": order, "batch_rows": chunk_rows, "seconds": elapsed,
                         "exact_array_equality": equal, "max_absolute_difference": differences,
                         "probability_sha256": [digest(a) for a in probability]})
            print(json.dumps(rows[-1]), flush=True)
            assert all(equal), differences
    assert params_before == run.model_parameters(model)
    assert run.sha256(path) == original_hash
    report = {"probe_only": True, "rows": len(frame), "n_features": values.shape[1],
              "n_jobs": 2, "fit_performed": False, "labels_read": False,
              "production_changes": "None; current running process is unchanged",
              "original_C_contiguous": bool(values.flags.c_contiguous),
              "original_F_contiguous": bool(values.flags.f_contiguous),
              "input_float32_sha256": digest(values),
              "checkpoint": str(path), "checkpoint_sha256_before_after": original_hash,
              "estimators_parameters_unchanged": True, "timings": rows,
              "caveat": "Bounded read-only probe; not a full-role throughput benchmark"}
    run.json_write(run.WORK / ("prediction_batch_audit.json" if args.batch_check else "prediction_layout_audit.json"), report)


if __name__ == "__main__":
    main()
