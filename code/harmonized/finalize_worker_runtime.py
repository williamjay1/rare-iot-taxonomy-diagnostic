"""Record actual grouped worker wall interval, separate from finalizer overhead."""
from datetime import datetime
import json
import sys

sys.dont_write_bytecode = True
import run_harmonized_refit as run


def main():
    output = run.WORK / "results" / "grouped37"
    original = json.loads((output / "invocation_runtime.json").read_text())
    values = []
    for seed in run.SEEDS:
        assert (output / f"seed_{seed}" / "COMPLETED.json").exists()
        value = json.loads((run.WORK / "dispatch" / f"grouped37_seed_{seed}.json").read_text())
        assert value["state"] == "complete"
        values.append(value)
    started = min(datetime.fromisoformat(item.get("first_started_utc", item["started_utc"])) for item in values)
    finished = max(datetime.fromisoformat(item["finished_utc"]) for item in values)
    for item in values:
        directory = output / f"seed_{item['seed']}"
        runtime_path = directory / "runtime.json"
        runtime = json.loads(runtime_path.read_text())
        first_role = next(line for line in (directory / "worker_execution.log").read_text().splitlines()
                          if "GROUPED37 ROLES VERIFIED" in line)
        role_time = datetime.fromisoformat(first_role.split()[0])
        runtime["wall_seconds_first_roles_to_completion_including_recovery_gap"] = (
            datetime.fromisoformat(item["finished_utc"]) - role_time).total_seconds()
        runtime["worker_wall_seconds_first_start_to_finish_including_gaps"] = (
            datetime.fromisoformat(item["finished_utc"]) -
            datetime.fromisoformat(item.get("first_started_utc", item["started_utc"]))
        ).total_seconds()
        run.json_write(runtime_path, runtime)
    run.json_write(output / "aggregation_runtime.json", original)
    run.json_write(output / "invocation_runtime.json", {
        "elapsed_seconds": (finished - started).total_seconds(),
        "started_utc": started.isoformat(), "finished_utc": finished.isoformat(),
        "execution_mode": "resource_gated_independent_seed_workers",
        "wall_interval_includes_between_worker_waits": True,
        "sum_worker_seconds_not_wall_time": sum(item["elapsed_seconds"] for item in values),
        "sum_worker_seconds_scope": "Completed worker invocations only; excludes interrupted attempts whose exact active duration is unknown",
        "interrupted_attempt_records": [path for item in values for path in item.get("prior_attempt_records", [])],
        "aggregation_only_seconds": original["elapsed_seconds"], "workers": values})


if __name__ == "__main__":
    main()
