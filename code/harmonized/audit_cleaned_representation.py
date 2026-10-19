"""Stream the D cache to verify the exact model-input grouping representation."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True

from datetime import datetime, timezone
import json
import time
import run_harmonized_refit as run
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def main():
    run.require_physical_d_output(run.WORK)
    started = time.perf_counter()
    contract_path = run.OUT / "run_contract.json"
    contract = json.loads(contract_path.read_text())
    features = contract["features"]
    raw_hashes, clean_hashes, sources = [], [], []
    nonfinite = np.zeros(len(features), dtype=np.int64)
    rounded = np.zeros(len(features), dtype=np.int64)
    rows = 0
    for item in contract["inputs"]:
        path = item["training_cache_path"]
        run.require_physical_d_output(path)
        parquet = pq.ParquetFile(path)
        source = {"path": path, "source_sha256": item["sha256"],
                  "arrow_types": {name: str(parquet.schema_arrow.field(name).type) for name in features}}
        for batch in parquet.iter_batches(batch_size=65536, columns=features, use_threads=False):
            frame = batch.to_pandas(use_threads=False)
            dtypes = {name: str(frame[name].dtype) for name in features}
            if "pandas_dtypes" in source:
                assert source["pandas_dtypes"] == dtypes
            source["pandas_dtypes"] = dtypes
            raw = frame.to_numpy(dtype=np.float64)
            actual = run.trf.clean_features(frame, features)
            assert actual.dtype == np.float32 and np.isfinite(actual).all()
            nonfinite += (~np.isfinite(raw)).sum(axis=0)
            sanitized = np.where(np.isfinite(raw), raw, 0.0)
            rounded += (actual.astype(np.float64) != sanitized).sum(axis=0)
            raw_hashes.append(pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64))
            clean_hashes.append(pd.util.hash_pandas_object(pd.DataFrame(actual, columns=features),
                                                        index=False).to_numpy(dtype=np.uint64))
            rows += len(frame)
        sources.append(source)
    raw_hash = np.concatenate(raw_hashes)
    actual_hash = np.concatenate(clean_hashes)
    raw_codes, raw_unique = pd.factorize(raw_hash, sort=False)
    actual_codes, actual_unique = pd.factorize(actual_hash, sort=False)
    partitions_identical = bool(np.array_equal(raw_codes, actual_codes))
    assert partitions_identical and len(actual_unique) == 1388399
    report = {"status": "PASS", "created_utc": datetime.now(timezone.utc).isoformat(),
              "rows": rows, "features": features, "sources": sources,
              "actual_model_dtype": "float32", "actual_model_all_finite": True,
              "precast_nonfinite_policy": "+inf/-inf -> NaN; NaN -> 0; then float32 cast",
              "hash_values": "pandas hash_pandas_object(DataFrame(clean_features), index=False)",
              "clean_features_source": str(run.trf.__file__),
              "clean_features_source_sha256": run.sha256(run.trf.__file__),
              "contract_sha256": run.sha256(contract_path),
              "nonfinite_values_before_cleaning_per_feature": dict(zip(features, nonfinite.tolist())),
              "cast_changed_finite_values_per_feature": dict(zip(features, rounded.tolist())),
              "raw37_unique_groups": len(raw_unique), "actual37_unique_groups": len(actual_unique),
              "raw37_and_actual_clean_float32_partitions_identical": partitions_identical,
              "additional_protocol_needed": False, "fit_performed": False,
              "elapsed_seconds": time.perf_counter() - started}
    run.json_write(run.WORK / "representation_verification.json", report)
    print(json.dumps({key: report[key] for key in ("status", "rows", "actual_model_dtype",
          "raw37_and_actual_clean_float32_partitions_identical", "elapsed_seconds")}), flush=True)


if __name__ == "__main__":
    main()
