"""Audit measured checkpoint costs and time feature cleaning, without fitting."""
from __future__ import annotations
import json
from datetime import datetime
import statistics
import sys
import time
sys.dont_write_bytecode = True
import run_harmonized_refit as run
import numpy as np
import pyarrow.parquet as pq


def main():
    run.require_physical_d_output(run.WORK)
    output = run.WORK / "results" / "paired46"
    contract = json.loads((output / "run_contract.json").read_text())
    source = contract["inputs"][0]["training_cache_path"]
    # Small read-only cost probe; no labels or model outcomes are consulted.
    batch = next(pq.ParquetFile(source).iter_batches(batch_size=200000,
                 columns=contract["features"], use_threads=False))
    frame = batch.to_pandas(use_threads=False)
    timings = []
    reference = None
    for _ in range(5):
        started = time.perf_counter()
        values = run.trf.clean_features(frame, contract["features"])
        timings.append(time.perf_counter() - started)
        if reference is None:
            reference = values
        else:
            assert np.array_equal(reference, values)
    assert reference.dtype == np.float32 and reference.shape == (len(frame), 37)
    events = []
    for line in (output / "execution.log").read_text().splitlines():
        timestamp, message = line.split(" ", 1)
        events.append((datetime.fromisoformat(timestamp), message))
    model_saved = next(stamp for stamp, message in events if message.startswith("SAVED seed_42\\taxonomy_real.joblib "))
    probabilities_saved = next(stamp for stamp, message in events if message.startswith("SAVED seed_42\\taxonomy_real_probabilities.joblib "))
    builder = json.loads((output / "seed_42" / "taxonomy_real_probabilities.joblib.json").read_text())["elapsed_seconds"]
    wall_between = (probabilities_saved - model_saved).total_seconds()
    proof = json.loads((output / "seed_42" / "data_role_proof.json").read_text())
    predicted_rows = sum(proof[name] for name in ("gate_train_rows", "gate_select_rows", "outer_test_rows"))
    value = {"fit_or_model_choice_performed": False,
             "probe_cache_source": source, "probe_rows": len(frame),
             "probe_clean_seconds": timings, "probe_clean_median_seconds": statistics.median(timings),
             "probe_float32_exact_repeat_equality": True,
             "seed42_real_three_role_rows": predicted_rows,
             "linear_clean_cost_proxy_seconds": statistics.median(timings) * predicted_rows / len(frame),
             "linear_proxy_caveat": "A read-only 200k-row cost probe, not a direct profile of every production conversion",
             "actual_real_probability_builder_seconds": builder,
             "wall_real_model_save_to_probabilities_save_seconds": wall_between,
             "non_builder_overhead_upper_proxy_seconds": wall_between - builder,
             "overhead_caveat": "Includes setup, joblib compression, digest, metadata and log writes; not an isolated serialization benchmark",
             "static_call_audit": {
                 "flat_xgb": "One prediction per role per seed; cached and reused across every taxonomy arm",
                 "each_hierarchy": "One prediction per role per seed; cached and reused for equal blend, soft/hard and gate",
                 "cleaning": "One clean_features call per role per hierarchy, outside the loop over local family models",
                 "family_predictions": "One family and one conditional per non-singleton family, all required for soft factorization",
                 "flat_lgbm_selection": "Selection prediction is cached although not consumed by current flat outputs; potentially avoidable in a future version",
                 "production_changes": "None: current runner and contracts remain unchanged; no skipped rows, precision changes or estimator changes"},
             "worker_plan": "One active estimator with n_jobs=2; grouped37 waits for paired completion; validation stages run sequentially",
             "source_sha256": {str(run.__file__): run.sha256(run.__file__),
                               str(run.trf.__file__): run.sha256(run.trf.__file__),
                               str(run.benefit.__file__): run.sha256(run.benefit.__file__)}}
    run.json_write(run.WORK / "runtime_cost_audit.json", value)
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
