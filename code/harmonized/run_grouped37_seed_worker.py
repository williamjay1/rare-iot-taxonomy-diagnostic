"""Execute one exclusively leased grouped37 seed; never write shared summaries."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import msvcrt
import os
import sys
import time
import warnings

sys.dont_write_bytecode = True
import run_grouped37_confirmation as grouped
from sklearn.preprocessing import LabelEncoder
from threadpoolctl import threadpool_limits

base = grouped.base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=base.SEEDS)
    args = parser.parse_args()
    base.require_physical_d_output(grouped.OUT)
    dispatch = base.WORK / "dispatch"
    dispatch.mkdir(exist_ok=True)
    directory = grouped.OUT / f"seed_{args.seed}"
    directory.mkdir(exist_ok=True)
    # Windows byte-range locks survive stale lock files but not a live owner.
    with (dispatch / f"grouped37_seed_{args.seed}.lock").open("a+b") as lease:
        if lease.tell() == 0:
            lease.write(b"0")
            lease.flush()
        lease.seek(0)
        msvcrt.locking(lease.fileno(), msvcrt.LK_NBLCK, 1)
        if (directory / "COMPLETED.json").exists():
            print(f"Seed {args.seed} already complete; no fit", flush=True)
            return
        started_utc = datetime.now(timezone.utc).isoformat()
        started = time.perf_counter()
        state_path = dispatch / f"grouped37_seed_{args.seed}.json"
        previous = json.loads(state_path.read_text()) if state_path.exists() else None
        prior_attempts = list(previous.get("prior_attempt_records", [])) if previous else []
        if previous:
            history = dispatch / f"grouped37_seed_{args.seed}_prior_attempt_{len(prior_attempts) + 1}.json"
            assert not history.exists(), str(history)
            base.json_write(history, previous)
            prior_attempts.append(str(history))
        state = {"seed": args.seed, "pid": os.getpid(), "state": "running",
                 "started_utc": started_utc, "n_jobs": 2,
                 "first_started_utc": previous.get("first_started_utc", previous["started_utc"]) if previous else started_utc,
                 "prior_attempt_records": prior_attempts,
                 "resumed_from_immutable_checkpoints": previous is not None,
                 "shared_summary_writes": False, "worker_script_sha256": base.sha256(__file__)}
        base.json_write(state_path, state)

        def seed_event(message):
            line = f"{datetime.now(timezone.utc).isoformat()} {message}"
            print(line, flush=True)
            with (directory / "worker_execution.log").open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")

        base.event = seed_event
        warnings.filterwarnings("ignore", category=UserWarning)
        full, features, representative, primary = grouped.setup()
        base.OUT = grouped.OUT
        encoder = LabelEncoder().fit(full["Label"].astype(str))
        values = grouped.prepare_seed(full, features, representative, primary, encoder, args.seed)
        seed_event(f"GROUPED37 ROLES VERIFIED seed={args.seed} counts={len(values[0])}/{len(values[1])}/{len(values[2])}/{len(values[3])}")
        # This worker has no next seed; only the already materialized roles are needed.
        del full, representative, primary
        gc.collect()
        assert base.array_hash(values[0]["__source_row_id"]) == values[-1]["fit_order_sha256"]
        state["unused_setup_tables_released_before_fit"] = True
        base.json_write(state_path, state)
        fit_started = time.perf_counter()
        with threadpool_limits(limits=2):
            grouped.fit_seed(args.seed, *values, encoder, features)
        base.json_write(directory / "runtime.json", {
            "wall_seconds_this_invocation": time.perf_counter() - fit_started,
            "worker_total_seconds_including_setup": time.perf_counter() - started,
            "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(),
            "first_started_utc": state["first_started_utc"],
            "prior_attempt_records": prior_attempts,
            "execution_mode": "exclusively_leased_single_seed_worker", "pid": os.getpid()})
        state.update({"state": "complete", "finished_utc": datetime.now(timezone.utc).isoformat(),
                      "elapsed_seconds": time.perf_counter() - started})
        base.json_write(state_path, state)
        seed_event(f"COMPLETED GROUPED37 SEED {args.seed}")


if __name__ == "__main__":
    main()
