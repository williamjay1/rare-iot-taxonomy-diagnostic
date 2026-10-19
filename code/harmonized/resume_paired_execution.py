"""Resume frozen paired fits after correcting a serialization-only worker audit."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True

from datetime import datetime, timezone
import json
import msvcrt
import shutil
import time
import run_harmonized_refit as run
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression

_model_parameters = run.model_parameters
_checkpoint = run.checkpoint


def initialize_frozen_cache():
    run.require_physical_d_output(run.WORK)
    contract = json.loads((run.OUT / "run_contract.json").read_text())
    assert contract["config"] == run.CONFIG
    assert contract["package_versions"] == run.benefit.package_versions()
    assert run.sha256(run.CORE / "run_metadata.json") == contract["primary_metadata_sha256"]
    assert run.sha256(run.CORE / "nested_split_audit.csv") == contract["primary_role_audit_sha256"]
    assert run.sha256(run.OVERLAP) == contract["overlap_report_sha256"]
    for path, digest in contract["sources"].items():
        assert run.sha256(path) == digest, path
    free = shutil.disk_usage(run.WORK).free
    assert free >= 20 * 1024**3
    sources = []
    cached_paths = []
    for item in contract["inputs"]:
        path = Path(item["training_cache_path"])
        run.require_physical_d_output(path)
        assert run.sha256(path) == item["sha256"]
        assert path.stat().st_size == item["bytes"]
        current_source_hash = run.sha256(item["source_path"])
        sources.append({"frozen_source_path": item["source_path"], "frozen_sha256": item["sha256"],
                        "current_original_source_sha256": current_source_hash,
                        "original_source_matches_frozen": current_source_hash == item["sha256"],
                        "training_cache_path": str(path), "cache_matches_frozen": True,
                        "source_recopied": False})
        cached_paths.append(path)
    run.json_write(run.WORK / "recovery" / "cache_only_resume_preflight.json", {
        "checked_utc": datetime.now(timezone.utc).isoformat(), "inputs": sources,
        "scientific_training_inputs_changed": False, "original_contract_preserved": True,
        "D_free_bytes": free, "resume_script_sha256": run.sha256(__file__)})
    run.nested.TRAIN_PATH, run.nested.TEST_PATH = cached_paths
    primary = json.loads((run.CORE / "run_metadata.json").read_text())
    features = contract["features"]
    assert features == json.loads(run.OVERLAP.read_text())["shared_features"]
    full = pd.concat([pd.read_parquet(path) for path in cached_paths], ignore_index=True)
    original_features = run.nested.exp.feature_cols(full)
    assert original_features == primary["features"] and len(original_features) == 46
    run.benefit.assign_exact_predictor_groups(full, original_features)
    full["__shared37_raw_hash"] = pd.util.hash_pandas_object(full[features], index=False).to_numpy(dtype=np.uint64)
    clean = pd.DataFrame(run.trf.clean_features(full, features), columns=features)
    full["__shared37_model_hash"] = pd.util.hash_pandas_object(clean, index=False).to_numpy(dtype=np.uint64)
    full["__source_row_id"] = np.arange(len(full), dtype=np.int64)
    run.event("RESUME PREFLIGHT frozen D caches verified; no old-source recopy; original contract unchanged")
    return full, features, primary


def audited_model_parameters(value):
    if isinstance(value, LogisticRegression):
        expected = LogisticRegression(C=1.0, max_iter=100, solver="newton-cholesky",
                                      random_state=value.random_state).get_params(deep=False)
        assert value.get_params(deep=False) == expected
        assert value.random_state in run.SEEDS
        return {"type": type(value).__name__, "params": value.get_params(deep=False)}
    return _model_parameters(value)


def checkpoint_with_unknown_timing(path, builder):
    value, seconds = _checkpoint(path, builder)
    # Unknown cost is not an invented duration. Scientific predictions are unchanged.
    return value, float("nan") if seconds is None else seconds


def recover_serialized_stacker():
    directory = run.OUT / "seed_42"
    path = directory / "flat_newton_stacker.joblib"
    metadata = path.with_name(path.name + ".json")
    recovery = run.WORK / "recovery"
    recovery.mkdir(exist_ok=True)
    if metadata.exists():
        assert run.sha256(path) == json.loads(metadata.read_text())["sha256"]
        return
    assert path.exists(), "No serialized stacker to recover; refusing an unrecorded refit"
    digest = run.sha256(path)
    model = joblib.load(path)
    parameters = audited_model_parameters(model)
    assert model.random_state == 42
    assert np.isfinite(model.coef_).all() and np.isfinite(model.intercept_).all()
    assert np.array_equal(model.classes_, np.arange(34))
    assert model.n_features_in_ == 68
    assert (model.n_iter_ > 0).all() and (model.n_iter_ < 100).all()
    previous = [line for line in (run.OUT / "execution.log").read_text().splitlines()
                if "METRIC seed=42 flat_temperature_calibrated_blend" in line][-1]
    preceding_metric = datetime.fromisoformat(previous.split()[0])
    artifact_written = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    backup = recovery / "orphan_flat_newton_stacker_seed_42_COPY.joblib"
    if not backup.exists():
        shutil.copy2(path, backup)
    assert run.sha256(backup) == digest
    details = {
        "sha256": digest, "elapsed_seconds": None, "models": parameters,
        "saved_utc": artifact_written.isoformat(), "recovered_serialized_model": True,
        "fit_performed_during_recovery": False, "n_iter": model.n_iter_.tolist(),
        "duration_status": "Exact perf_counter duration lost when metadata assertion raised",
        "duration_upper_bound_seconds": (artifact_written - preceding_metric).total_seconds(),
        "duration_upper_bound_scope": "preceding metric log to completed serialized artifact, includes non-fit overhead",
        "original_file_sha256_unchanged": True,
        "convergence_evidence": "Frozen fit_converged_stacker returned before atomic joblib save; it raises on ConvergenceWarning"}
    run.json_write(metadata, details)
    run.json_write(recovery / "seed42_stacker_recovery.json", {
        "recovery_utc": datetime.now(timezone.utc).isoformat(), "checkpoint": str(path),
        "copy": str(backup), "details": details,
        "reason": "Generic serialization audit incorrectly required LogisticRegression.n_jobs=2; frozen reference is None",
        "scientific_estimator_parameters_changed": False, "original_runner_changed": False,
        "runtime_threadpool_limit": 2, "recovery_script_sha256": run.sha256(__file__)})
    run.event("RECOVERED serialized seed42 stacker; frozen LogisticRegression n_jobs=None; exact builder duration unknown")


def finalize_runtime():
    recovery = run.WORK / "recovery"
    path = run.OUT / "invocation_runtime.json"
    original = json.loads(path.read_text())
    run.json_write(recovery / "resumed_invocation_runtime.json", original)
    first_line = next(line for line in (run.OUT / "execution.log").read_text().splitlines()
                      if "ROLES VERIFIED seed=42" in line)
    first_role = datetime.fromisoformat(first_line.split()[0])
    finished = datetime.fromisoformat(original["finished_utc"])
    original.update({"elapsed_seconds_resume_invocation": original["elapsed_seconds"],
                     "elapsed_seconds": (finished - first_role).total_seconds(),
                     "timing_scope": "First original seed42 roles-verified event through final resumed finish; includes metadata recovery and external interruption gaps",
                     "startup_before_first_role_event_included": False,
                     "first_original_role_event_utc": first_role.isoformat(),
                     "recovery_report": str(recovery / "seed42_stacker_recovery.json")})
    run.json_write(path, original)
    seed_path = run.OUT / "seed_42" / "runtime.json"
    seed_runtime = json.loads(seed_path.read_text())
    completion = json.loads((run.OUT / "seed_42" / "COMPLETED.json").read_text())
    seed_runtime["wall_seconds_first_roles_to_completion_including_recovery_gap"] = (
        datetime.fromisoformat(completion["completed_utc"]) - first_role).total_seconds()
    seed_runtime["exact_original_stacker_builder_seconds_available"] = False
    run.json_write(seed_path, seed_runtime)
    log_lines = (run.OUT / "execution.log").read_text().splitlines()
    for seed in run.SEEDS:
        first = next(line for line in log_lines if f"ROLES VERIFIED seed={seed} " in line)
        first_time = datetime.fromisoformat(first.split()[0])
        directory = run.OUT / f"seed_{seed}"
        value = json.loads((directory / "runtime.json").read_text())
        completed = json.loads((directory / "COMPLETED.json").read_text())
        value["wall_seconds_first_roles_to_completion_including_recovery_gap"] = (
            datetime.fromisoformat(completed["completed_utc"]) - first_time).total_seconds()
        value["first_roles_verified_utc"] = first_time.isoformat()
        value["wall_interval_includes_any_external_interruption_gap"] = True
        run.json_write(directory / "runtime.json", value)


def main():
    run.require_physical_d_output(run.WORK)
    lease_path = run.WORK / "dispatch" / "paired_resume.lock"
    with lease_path.open("a+b") as lease:
        if lease.tell() == 0:
            lease.write(b"0")
            lease.flush()
        lease.seek(0)
        msvcrt.locking(lease.fileno(), msvcrt.LK_NBLCK, 1)
        run.model_parameters = audited_model_parameters
        run.checkpoint = checkpoint_with_unknown_timing
        run.initialize = initialize_frozen_cache
        recover_serialized_stacker()
        run.main()
        finalize_runtime()


if __name__ == "__main__":
    main()
