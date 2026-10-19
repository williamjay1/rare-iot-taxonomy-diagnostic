"""Verify an execution-only adapter against unchanged completed checkpoints."""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

from datetime import datetime, timezone
import json
import time
import run_harmonized_refit as run
import prediction_execution as execution
import joblib
import numpy as np
import pyarrow.parquet as pq
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
from threadpoolctl import threadpool_info, threadpool_limits


def main():
    run.require_physical_d_output(run.WORK)
    directory = run.OUT / "seed_42"
    contract = json.loads((run.OUT / "run_contract.json").read_text())
    batch = next(pq.ParquetFile(contract["inputs"][0]["training_cache_path"])
                 .iter_batches(batch_size=execution.BATCH_ROWS + 1,
                               columns=contract["features"], use_threads=False))
    frame = batch.to_pandas(use_threads=False)
    values = run.trf.clean_features(frame, contract["features"])
    mapping = json.loads((directory / "data_role_proof.json").read_text())["mapping"]["real"]
    encoder = LabelEncoder().fit(list(mapping))
    original = LGBMClassifier.predict_proba
    reports = []
    with threadpool_limits(limits=2):
        pools = threadpool_info()
        assert all(pool["num_threads"] == 2 for pool in pools), pools
        for filename in ("flat_lgbm.joblib", "taxonomy_real.joblib"):
            path = directory / filename
            before_hash = run.sha256(path)
            model = joblib.load(path)
            parameters = run.model_parameters(model)
            models = [model] if filename.startswith("flat") else [model["family_model"], *[
                part[0] for part in model["local_models"].values() if part[0] is not None]]
            model_details = [{"n_jobs": part.n_jobs,
                              "booster_num_threads": part.booster_.params.get("num_threads"),
                              "num_trees": part.booster_.num_trees(),
                              "num_classes": part.booster_.num_model_per_iteration()}
                             for part in models]
            assert all(part["n_jobs"] == part["booster_num_threads"] == 2 for part in model_details)

            def invoke():
                if filename.startswith("flat"):
                    return (model.predict_proba(values),)
                return run.trf.hierarchy_probabilities(values, model["family_model"],
                    model["family_encoder"], encoder, mapping, model["local_models"])

            rows = []
            reference = None
            try:
                for mode in ("reference", "explicit_threads", "bounded_8192", "reference_repeat"):
                    if mode == "explicit_threads":
                        LGBMClassifier.predict_proba = lambda self, x, *a, **k: original(self, x, *a, num_threads=2, **k)
                    elif mode == "bounded_8192":
                        LGBMClassifier.predict_proba = lambda self, x, *a, **k: execution.bounded_predict(original, self, x, *a, **k)
                    else:
                        LGBMClassifier.predict_proba = original
                    started = time.perf_counter()
                    probabilities = invoke()
                    elapsed = time.perf_counter() - started
                    if reference is None:
                        reference = probabilities
                    equality = [bool(np.array_equal(a, b)) for a, b in zip(reference, probabilities)]
                    differences = [float(np.max(np.abs(a - b))) for a, b in zip(reference, probabilities)]
                    row = {"mode": mode, "seconds": elapsed, "exact_array_equality": equality,
                           "max_absolute_difference": differences,
                           "dtypes": [str(part.dtype) for part in probabilities]}
                    rows.append(row)
                    print(json.dumps({"checkpoint": filename, **row}), flush=True)
                    assert all(equality), row
            finally:
                LGBMClassifier.predict_proba = original
            assert run.model_parameters(model) == parameters
            assert run.sha256(path) == before_hash
            reports.append({"checkpoint": str(path), "checkpoint_sha256_before_after": before_hash,
                            "estimators_unchanged": True, "model_details": model_details, "cases": rows})
            del model, models, reference, probabilities
    run.json_write(run.WORK / "prediction_execution_audit.json", {
        "status": "EXACT_PASS", "created_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(values), "batch_rows": execution.BATCH_ROWS, "features": values.shape[1],
        "input_dtype": str(values.dtype), "labels_read": False, "fit_performed": False,
        "explicit_prediction_num_threads": 2, "threadpools": pools,
        "adapter_sha256": run.sha256(execution.__file__), "checkpoints": reports,
        "caveat": "Bounded identity audit under shared CPU load; not a full-role speed benchmark",
        "application": "Eligible for future grouped workers only; paired live process unchanged"})


if __name__ == "__main__":
    main()
