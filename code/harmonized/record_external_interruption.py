"""Preserve checkpoint evidence after observing all previous workers absent."""
import sys
sys.dont_write_bytecode = True
from datetime import datetime, timezone
import json
from pathlib import Path
import run_harmonized_refit as run


def main():
    destination = run.WORK / "recovery" / "external_interruption_20261019T0410.json"
    assert not destination.exists()
    checkpoints = []
    for protocol in ("paired46", "grouped37"):
        for sidecar in sorted((run.WORK / "results" / protocol).glob("seed_*/*.joblib.json")):
            value = json.loads(sidecar.read_text())
            checkpoint = Path(str(sidecar)[:-5])
            digest = run.sha256(checkpoint)
            assert digest == value["sha256"], str(checkpoint)
            checkpoints.append({"path": str(checkpoint), "sha256": digest})
    run.json_write(destination, {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "observed_previous_processes_absent_local_time": "2026-10-19T04:10:39+08:00",
        "absent_previous_pids": [30508, 52244, 58796, 7524],
        "observation": "Win32_Process inventory contained no python.exe; previous exec sessions no longer existed",
        "cause": "Unknown external process loss; no model exception found in available logs",
        "completed_seed_counts": {"paired46": 2, "grouped37": 1},
        "metric_cell_counts": {"paired46": 27, "grouped37": 14},
        "checkpoint_sha256_verified_before_resume": checkpoints,
        "completed_checkpoints_refitted": False,
        "incomplete_unsaved_probability_stages_recomputed": True,
        "scientific_settings_or_rows_changed": False})
    print(f"Verified {len(checkpoints)} immutable checkpoints before resume", flush=True)


if __name__ == "__main__":
    main()
