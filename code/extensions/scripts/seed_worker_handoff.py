"""Controlled checkpoint handoff and final-only publication of isolated workers."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
import pandas as pd

import run_matched_extensions as run

ROOT = run.OUTPUT
SEEDS = (42, 43, 44)


def expected_cells(config, seed):
    return {f"{endpoint}_seed{seed}_{str(case).replace('-', '_')}": endpoint
            for endpoint, cases in (("lofo", config["lofo_families"]),
                                    ("scarcity", config["scarcity_caps"])) for case in cases}


def same_checkpoint(left, right):
    for name in ("completed.json", "results.csv", "paired_predictions.npz"):
        if not (right / name).exists() or run.file_hash(left / name) != run.file_hash(right / name):
            raise ValueError(f"Conflicting checkpoint: {left.name}/{name}")


def copy_checkpoint(source, destination):
    if destination.exists():
        same_checkpoint(source, destination)
        return
    temporary = destination.with_name(destination.name + ".handoff_tmp")
    shutil.copytree(source, temporary, dirs_exist_ok=True)
    same_checkpoint(source, temporary)
    temporary.replace(destination)


def prepare_seed42():
    status = json.loads((ROOT / "run_status.json").read_text())
    boundary = ROOT / "cells" / "lofo_seed42_DoS"
    boundary_evidence = status.get("error", "") + status.get("traceback", "")
    if (status["status"] != "failed" or "FileExistsError" not in status["error"]
            or "lofo_seed42_DoS" not in boundary_evidence):
        raise ValueError("Original queue has not reached the documented pre-fit boundary")
    if not (ROOT / "cells" / "lofo_seed42_DDoS" / "completed.json").exists():
        raise ValueError("Current DDoS cell must finish before handoff")
    if not boundary.is_file() or not boundary.read_text().startswith("CONTROLLED CELL-BOUNDARY FENCE"):
        raise ValueError("Unexpected boundary path; refusing any removal")
    run.write_json(ROOT / "dispatch" / "original_queue_exit.json", {
        "classification": "controlled cell-boundary exit, not a failed model fit",
        "mechanism": "next cell mkdir saw a sentinel file after the full prior cell completed",
        "no_fit_was_killed_or_interrupted": True,
        "original_status_preserved": status,
    })
    worker = ROOT / "workers" / "seed42" / "cells"
    worker.mkdir(parents=True, exist_ok=True)
    copied = []
    for marker in sorted((ROOT / "cells").glob("*_seed42_*/completed.json")):
        copy_checkpoint(marker.parent, worker / marker.parent.name)
        copied.append(marker.parent.name)
    boundary.unlink()
    run.write_json(ROOT / "dispatch" / "seed42_handoff.json", {
        "status": "checkpoint_copy_verified", "cells_reused_without_refit": copied,
        "boundary_sentinel_removed_after_original_exit": True,
        "worker_root": str(worker.parent), "n_jobs_per_worker": 2,
    })
    print(f"Handoff verified: reused {len(copied)} complete seed42 cells", flush=True)


def inspect_workers(config):
    available, by_seed = {}, {}
    config_hash = run.file_hash(run.CONFIG_PATH)
    for seed in SEEDS:
        expected = expected_cells(config, seed)
        worker = ROOT / "workers" / f"seed{seed}"
        count = 0
        for marker in sorted((worker / "cells").glob("*/completed.json")):
            folder = marker.parent
            if folder.name not in expected:
                raise ValueError(f"Unexpected cell in seed root: {folder}")
            protocol = json.loads(marker.read_text())
            if protocol["seed"] != seed or protocol["config_sha256"] != config_hash:
                raise ValueError(f"Mixed seed or frozen configuration: {folder}")
            frame = pd.read_csv(folder / "results.csv")
            if len(frame) != 9 or frame.method.nunique() != 9 or set(frame.method) != set(run.METHODS):
                raise ValueError(f"Incomplete paired methods: {folder}")
            available[folder.name] = folder
            count += 1
        by_seed[seed] = {"completed_cells": count, "expected_cells": len(expected), "worker_root": str(worker)}
        pending = [path for path in (worker / "cells").glob("*/protocol.json")
                   if not path.with_name("completed.json").exists()]
        if pending:
            active = max(pending, key=lambda path: path.stat().st_mtime_ns).parent
            artifacts = sorted(path.name for path in active.iterdir() if path.is_file())
            by_seed[seed]["current_cell"] = active.name
            by_seed[seed]["checkpoint_files"] = artifacts
            by_seed[seed]["stage"] = (
                "outer inference/evaluation/checkpoint serialization" if "calibrators.joblib" in artifacts else
                "temperature/stacker calibration" if "calibration_inputs.npz" in artifacts else
                "utility pair predictions" if "cyclic_hierarchy.joblib" in artifacts else
                "cyclic hierarchy fit" if "real_hierarchy.joblib" in artifacts else
                "real hierarchy fit" if "lgbm.joblib" in artifacts else
                "LightGBM fit" if "xgb.joblib" in artifacts else "XGBoost fit")
        else:
            by_seed[seed]["stage"] = "not started/resource wait" if count == 0 else "between cells or queue finished; inspect run_status"
        status_path = worker / "run_status.json"
        if status_path.exists():
            by_seed[seed]["worker_status"] = json.loads(status_path.read_text())["status"]
    # Before seed42 is handed off, its original completed cells remain real evidence.
    for marker in sorted((ROOT / "cells").glob("*/completed.json")):
        if marker.parent.name in available:
            same_checkpoint(marker.parent, available[marker.parent.name])
        else:
            available[marker.parent.name] = marker.parent
    frozen = sum((ROOT / "frozen" / f"seed{seed}" / "completed.json").exists() for seed in range(42, 48))
    captures = sum((ROOT / "capture" / window / "completed.json").exists()
                   for window in ("w1_55_56", "w2_57_58", "w3_59_60", "w4_61_63"))
    all_units_complete = len(available) == 39 and frozen == 6 and captures == 4
    handoff_path = ROOT / "MANUSCRIPT_HANDOFF.json"
    handoff_verified = False
    if handoff_path.exists():
        handoff = json.loads(handoff_path.read_text())
        verification = handoff.get("verification", {})
        handoff_verified = (
            all_units_complete and handoff.get("completion") is True
            and handoff.get("completed_units") == 55
            and verification.get("status") == "passed"
            and verification.get("probability_matrices_checked") is True
            and verification.get("nested_seed_aggregation_checked") is True
            and verification.get("config_sha256") == config_hash)
    report_status = ("completed_and_verified" if handoff_verified else
                     "all_units_completed_pending_final_audit" if all_units_complete else
                     "running_or_incomplete")
    report = {"status": report_status, "snapshot_utc": datetime.now(timezone.utc).isoformat(),
              "refit_cells_completed": len(available),
              "refit_cells_expected": 39, "capture_windows_completed": captures,
              "calibration_units_completed": frozen, "alert_budget_units_completed": frozen,
              "total_units_completed": len(available) + 2 * frozen + captures,
              "total_units_expected": 55, "seeds": by_seed,
              "frozen_config_sha256": config_hash, "n_jobs_per_worker": 2,
              "no_worker_results_are_published_to_shared_root_until_final_merge": True}
    run.write_json(ROOT / "dispatch" / "worker_progress.json", report)
    run.write_json(ROOT / "CURRENT_RUN_STATUS.json", {
        "status": "completed_and_verified" if handoff_verified else "isolated_workers_running_or_pending_final_publication",
        "progress": report,
        "current_logs": {str(seed): str(ROOT / f"seed{seed}_run.log") for seed in SEEDS},
        "capture_log": str(ROOT / "capture_fit_run.log"),
        "old_matched_refit_run_log_is_not_the_active_queue": True,
        "old_queue_error_classification": "controlled next-cell boundary after complete DDoS; no fit interrupted",
        "old_error_audit": str(ROOT / "dispatch" / "original_queue_exit.json"),
        "handoff_audit": str(ROOT / "dispatch" / "seed42_handoff.json"),
        "shared_publication_rule": "only after all 55 units; exclusive publication lease and atomic per-cell copies",
        "manuscript_handoff_available": handoff_verified,
        "manuscript_handoff_path": str(handoff_path) if handoff_verified else None,
    })
    return available, report


def publish(config):
    available, report = inspect_workers(config)
    expected = {name for seed in SEEDS for name in expected_cells(config, seed)}
    if set(available) != expected or report["total_units_completed"] != 55:
        raise ValueError("Final publication requires all 39 refit cells, 12 E3/E4 units and four capture windows")
    if list((ROOT / "dispatch" / "leases").glob("*.lock")):
        raise ValueError("A seed worker still owns a lease")
    if (ROOT / "dispatch" / "capture_fit.lock").exists():
        raise ValueError("The capture fit still owns its output lease")
    for seed in SEEDS:
        status = json.loads((ROOT / "workers" / f"seed{seed}" / "run_status.json").read_text())
        if status["status"] != "requested_queue_finished" or status["missing_cells"]:
            raise ValueError(f"Worker seed{seed} did not finish cleanly")
    for name in expected:
        target = ROOT / "cells" / name
        run.require_physical_d(target)
        if target.exists() and (not target.is_dir() or target.is_symlink() or target.is_junction()):
            raise ValueError(f"Publication destination is not a plain directory: {target}")
    for name, source in available.items():
        target = ROOT / "cells" / name
        if source != target:
            copy_checkpoint(source, target)
    import run_capture_controls as capture
    capture.summarize()
    coverage = run.summarize(config)
    if sum(row["completed_units"] for row in coverage) != 55:
        raise ValueError("Published coverage does not equal the verified 55 units")
    run.write_json(ROOT / "dispatch" / "final_publication.json", {
        "status": "published_all_55_units_pending_final_probability_verifier",
        "workers": report["seeds"], "frozen_config_sha256": run.file_hash(run.CONFIG_PATH),
        "copied_cells": sorted(expected), "sample_or_tree_counts_reduced": False,
    })
    print("Published all 55 units; run the full probability verifier next", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare-seed42", "status", "publish"))
    args = parser.parse_args()
    run.require_physical_d(ROOT)
    config = json.loads(run.CONFIG_PATH.read_text())
    if args.action == "prepare-seed42":
        prepare_seed42()
    elif args.action == "publish":
        lock = ROOT / "dispatch" / "publication.lock"
        with lock.open("x", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "scope": "single writer for final shared publication"}, stream)
        try:
            publish(config)
        finally:
            lock.unlink()
    else:
        _, report = inspect_workers(config)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
