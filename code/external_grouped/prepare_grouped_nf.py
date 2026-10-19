"""Freeze three independent-source grouped allocations before model outcomes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

BASE = Path(r"workspace\external_grouped")
SOURCE = BASE.parent / "temporal"
sys.path.insert(0, str(SOURCE / "runtime"))
import duckdb


def save(path, content):
    path.write_text(json.dumps(content, indent=2), encoding="utf-8")


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    assert BASE.resolve().drive.upper() == "D:"
    contract = {
        "source": "NF-ToN-IoT-v3",
        "reason": "Chronological future contains no fitted attack label; add a separately labelled grouped replication, without changing the chronological cutpoints.",
        "frozen_before_new_model_outcomes": True,
        "seeds": [42, 43, 44],
        "group_role": "hash(predictor_hash, seed) modulo 1000000; fit <560000, calibration <680000, selection <800000, otherwise test",
        "hash_domain": "All 48 cleaned float32 predictors, excluding absolute times, IP addresses, transaction identifier and labels",
        "roles": ["fit", "calibration", "selection", "test"],
        "caps": {"fit_per_attack": 25000, "fit_benign": 80000, "calibration_total_uniform": 60000, "selection_total_uniform": 60000, "test_total_uniform": 300000},
        "sampling": "Hash original row ID with 20261018, label blind except prespecified fitting caps",
        "conflicts": "Do not remove or relabel conflicting groups; assign a predictor group to one role without consulting labels",
        "model_contract": "Same eleven arms and original 520/380 estimators as chronological experiment; ten-bin right-closed ECE; all ten labels primary Macro F1",
        "interpretation": "Three grouped random allocations in one additional testbed corpus, not chronological validation, production traffic or three independent populations",
    }
    path = BASE / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == contract
    else:
        save(path, contract)
    metadata = json.loads((SOURCE / "data_audit.json").read_text())
    template = json.loads((SOURCE / "protocol.json").read_text())
    connection = duckdb.connect(str(SOURCE / "cache.duckdb"), read_only=True)
    connection.execute("SET threads=2")
    connection.execute("SET memory_limit='2GB'")
    for seed in contract["seeds"]:
        directory = BASE / f"allocation{seed}"
        (directory / "datasets").mkdir(parents=True, exist_ok=True)
        if (directory / "PREPARED.json").exists():
            continue
        protocol = dict(template)
        protocol.update({"random_fit_seeds": [seed], "chronology": "Not chronological: predictor-grouped random allocation",
                         "interpretation": contract["interpretation"], "duplicates": contract["conflicts"]})
        save(directory / "protocol.json", protocol)
        save(directory / "data_audit.json", {
            "source_rows": metadata["source_rows"],
            "features": metadata["features"],
            "allocation": "Predictor-grouped random allocation, not chronological",
            "role_audit": "role_audit.json",
            "source_audit": "../../temporal/data_audit.json",
        })
        records, overlap = [], []
        conditions = {"fit": "bucket<560000", "calibration": "bucket>=560000 AND bucket<680000",
                      "selection": "bucket>=680000 AND bucket<800000", "test": "bucket>=800000"}
        connection.execute(f"CREATE OR REPLACE TEMP VIEW assigned AS SELECT *,hash(predictor_hash,{seed})%1000000 AS bucket FROM flows")
        for role, condition in conditions.items():
            eligible = f"SELECT * EXCLUDE(bucket) FROM assigned WHERE {condition}"
            view = f"nf_grouped_{role}"
            connection.execute(f"CREATE OR REPLACE TEMP VIEW {view} AS {eligible}")
            support = connection.execute(f"SELECT attack,count(*) AS count FROM {view} GROUP BY attack ORDER BY attack").df()
            support.insert(0, "role", role)
            records.extend(support.to_dict("records"))
            if role == "fit":
                sample = f"SELECT * EXCLUDE(rk) FROM (SELECT *, row_number() OVER (PARTITION BY attack ORDER BY hash(row_id,20261018),row_id) AS rk FROM {view}) WHERE rk<=CASE WHEN attack='Benign' THEN 80000 ELSE 25000 END"
            else:
                cap = contract["caps"][f"{role}_total_uniform"]
                sample = f"SELECT * FROM {view} ORDER BY hash(row_id,20261018),row_id LIMIT {cap}"
            output = str(directory / "datasets" / f"{role}.parquet").replace("\\", "/")
            connection.execute(f"COPY ({sample}) TO '{output}' (FORMAT PARQUET, COMPRESSION ZSTD)")
            print(f"GROUPED allocation{seed} {role}: {int(support['count'].sum())} eligible rows", flush=True)
        pd.DataFrame(records).to_csv(directory / "role_class_support_full.csv", index=False)
        roles = list(conditions)
        for i, left in enumerate(roles):
            for right in roles[i + 1:]:
                # The assignment is a function of the predictor hash alone.
                count = connection.execute(f"SELECT count(*) FROM (SELECT DISTINCT predictor_hash FROM nf_grouped_{left}) l INNER JOIN (SELECT DISTINCT predictor_hash FROM nf_grouped_{right}) r USING (predictor_hash)").fetchone()[0]
                assert count == 0
                overlap.append({"left": left, "right": right, "overlap_hashes": count})
        save(directory / "role_audit.json", {"allocation_seed": seed, "overlap": overlap, "time_fields_used_by_model": False,
                                             "source_class_conflicts_label_cleaned": False})
        save(directory / "PREPARED.json", {"seed": seed, "protocol_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    connection.close()
    print("ALL GROUPED ALLOCATIONS PREPARED", flush=True)


if __name__ == "__main__":
    main()
