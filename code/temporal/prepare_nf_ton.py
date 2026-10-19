"""Acquire immutable official raw ZIP; build physical-D chronological work data."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import requests

BASE = Path(r"workspace\temporal")
RAW = Path(r"F:\AcademicData\iot_ids_sci\raw\nf_ton_iot_v3_20261018")
API = "https://api.rdm.uq.edu.au/datasets/343e2e8c-6e6e-4a0c-813d-a46acea1b7f4/download"
EXPECTED_SHA1 = "24854ca3072ab7c2ade9ebfb202345a666a94cfe"
EXCLUDE = ["FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS", "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "DNS_QUERY_ID", "Label", "Attack"]


def dump(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def sqlpath(path):
    return str(path).replace("\\", "/").replace("'", "''")


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    (BASE / "temp").mkdir(exist_ok=True)
    os.environ.update(TMP=str(BASE / "temp"), TEMP=str(BASE / "temp"))
    assert BASE.resolve().drive.upper() == "D:"
    protocol = {
        "source": "NF-ToN-IoT-v3", "doi": "10.48610/44d7c5e", "api": API,
        "license": "Permitted Re-Use with Commercial Use Restriction",
        "frozen_before_model_outcomes": True,
        "chronology": "Global start-time 60/15/10/15 percentiles; ties stay together; end time cannot cross the next role boundary",
        "roles": ["fit", "calibration", "selection", "test"],
        "duplicates": "For every later role remove predictor hashes observed in any earlier retained role; labels do not determine this rule",
        "caps": {"fit_per_attack": 25000, "fit_benign": 80000, "calibration_total_uniform": 60000, "selection_total_uniform": 60000, "test_total_uniform": 300000},
        "sampling": "Within each role deterministic hash of original row id, seed 20261018; no test label based sampling",
        "random_fit_seeds": [42, 43, 44], "excluded_predictors": EXCLUDE,
        "rare_rule": "attack labels at most 1 percent of pre-cap fit role; absent fit classes reported separately, not learned",
        "interpretation": "Independent testbed source refitting plus one global chronological future test; seeds repeat fitting, not independent time replications",
        "models": "weighted XGBoost 520 trees, LightGBM 380 trees, equal/calibrated flat blend, converged stacker, true/cyclic/support hierarchy and equal blends; same adapter and roles",
        "parameters": {"xgb_depth": 9, "learning_rate": .055, "row_subsample": .9, "colsample": .95, "reg_lambda": 1.5, "reg_alpha": .02, "xgb_weight_exponent": .35, "family_weight_exponent": .25, "local_weight_exponent": .50, "lgb_leaves": 127, "lgb_min_child_samples": 25, "threads": 2},
        "family_mapping": {"Benign": "Benign", "DoS": "Availability", "DDoS": "Availability", "Injection": "InputManipulation", "XSS": "InputManipulation", "MITM": "InterceptionCredential", "Password": "InterceptionCredential", "Backdoor": "PersistenceImpact", "Ransomware": "PersistenceImpact", "Scanning": "Reconnaissance"},
        "mapping_status": "Researcher grouping from official attack descriptions, not an independently validated native taxonomy or expert annotation",
    }
    contract = BASE / "protocol.json"
    if contract.exists():
        assert json.loads(contract.read_text()) == protocol
    else:
        dump(contract, protocol)
    target = RAW / "NF-ToN-IoT-v3_official_20261018.zip"
    if not target.exists():
        response = requests.get(API, timeout=90)
        response.raise_for_status()
        url = response.json()["url"]
        part = target.with_suffix(".zip.part")
        # Partial transfers are kept, but a fresh signed request is restarted unless range is explicitly verified.
        if part.exists():
            part = target.with_suffix(f".zip.{int(time.time())}.part")
        digest = hashlib.sha256()
        total = 0
        last = time.time()
        with requests.get(url, stream=True, timeout=(60, 180)) as remote, part.open("xb") as out:
            remote.raise_for_status()
            for block in remote.iter_content(4 * 1024 * 1024):
                if not block:
                    continue
                out.write(block)
                digest.update(block)
                total += len(block)
                if time.time() - last > 15:
                    print(f"Downloaded {total / 1024**2:.1f} MiB", flush=True)
                    last = time.time()
        assert total == 417476635, total
        part.rename(target)
        dump(BASE / "download_manifest.json", {"source_api": API, "raw_path": str(target), "bytes": total, "sha256": digest.hexdigest(), "downloaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        target.chmod(0o444)
    csv = BASE / "datasets" / "NF-ToN-IoT-v3_normalized_working.csv"
    csv.parent.mkdir(exist_ok=True)
    if not csv.exists():
        sha1 = hashlib.sha1()
        normsha = hashlib.sha256()
        with zipfile.ZipFile(target) as archive:
            member = next(name for name in archive.namelist() if name.endswith("/NF-ToN-IoT-v3.csv"))
            part = csv.with_suffix(".csv.tmp")
            tail = b""
            with archive.open(member) as stream, part.open("wb") as out:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    sha1.update(block)
                    block = tail + block
                    tail, block = block[-3:], block[:-3]
                    # CRCRLF from the official file is normalized only in the work copy.
                    normalized = block.replace(b"\r", b"")
                    out.write(normalized)
                    normsha.update(normalized)
                normalized = tail.replace(b"\r", b"")
                out.write(normalized)
                normsha.update(normalized)
            assert sha1.hexdigest() == EXPECTED_SHA1, sha1.hexdigest()
            part.replace(csv)
            for name in archive.namelist():
                if name.endswith("NetFlow_v3_Features.csv"):
                    dictionary = csv.parent / "NetFlow_v3_Features.csv"
                    if not dictionary.exists():
                        dictionary.write_bytes(archive.read(name))
        dump(BASE / "working_csv_manifest.json", {"official_csv_sha1": sha1.hexdigest(), "official_sha1_verified": True, "normalized_sha256": normsha.hexdigest(), "normalization": "remove CR bytes only from work copy; raw ZIP unchanged"})
        print("Official CSV SHA1 verified", flush=True)
    runtime = BASE / "runtime"
    sys.path.insert(0, str(runtime))
    try:
        import duckdb
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-cache-dir", "--target", str(runtime), "duckdb==1.4.3"], check=True)
        importlib.invalidate_caches()
        import duckdb
    con = duckdb.connect(str(BASE / "cache.duckdb"))
    con.execute("SET threads=2")
    con.execute("SET memory_limit='3GB'")
    con.execute(f"SET temp_directory='{sqlpath(BASE / 'temp')}'")
    header = csv.open(encoding="utf-8").readline().strip().split(",")
    features = [x for x in header if x not in EXCLUDE]
    assert len(features) == 48, features
    if not con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name='flows'").fetchone()[0]:
        parts = []
        for column in features:
            # Hash the exact float32 values the models receive, including the
            # declared zero replacement for nonfinite or missing predictors.
            parts.append(f'CASE WHEN isfinite(try_cast("{column}" AS FLOAT)) THEN try_cast("{column}" AS FLOAT) ELSE 0::FLOAT END AS "{column}"')
        projection = ",".join(parts)
        print("Building full D-disk feature cache", flush=True)
        con.execute(f'''CREATE TABLE flows AS SELECT row_number() OVER ()::BIGINT AS row_id,
            try_cast(FLOW_START_MILLISECONDS AS BIGINT) AS start_ms,
            try_cast(FLOW_END_MILLISECONDS AS BIGINT) AS end_ms,
            CASE lower(Attack) WHEN 'benign' THEN 'Benign' WHEN 'dos' THEN 'DoS' WHEN 'ddos' THEN 'DDoS' WHEN 'mitm' THEN 'MITM' WHEN 'xss' THEN 'XSS' ELSE upper(substr(Attack,1,1)) || lower(substr(Attack,2)) END AS attack,
            Attack AS attack_raw, {projection}
            FROM read_csv('{sqlpath(csv)}', header=true, all_varchar=true, parallel=false)''')
        con.execute(f'''ALTER TABLE flows ADD COLUMN predictor_hash UBIGINT''')
        con.execute(f'''UPDATE flows SET predictor_hash=hash({','.join('"'+x+'"' for x in features)})''')
    observed = con.execute("SELECT attack,attack_raw,count(*) AS n,min(start_ms),max(start_ms),min(end_ms),max(end_ms) FROM flows GROUP BY 1,2 ORDER BY 1,2").df()
    observed.to_csv(BASE / "source_class_audit.csv", index=False)
    assert set(observed.attack) == set(protocol["family_mapping"]), set(observed.attack)
    cuts = con.execute("SELECT quantile_disc(start_ms, [0.60,0.75,0.85]) FROM flows WHERE start_ms IS NOT NULL AND end_ms>=start_ms").fetchone()[0]
    bad = con.execute("SELECT count(*) FROM flows WHERE start_ms IS NULL OR end_ms IS NULL OR end_ms<start_ms").fetchone()[0]
    source_rows = con.execute("SELECT count(*) FROM flows").fetchone()[0]
    records = []
    earlier = []
    for i, role in enumerate(protocol["roles"]):
        lo = cuts[i - 1] if i else None
        hi = cuts[i] if i < 3 else None
        condition = "start_ms IS NOT NULL AND end_ms>=start_ms"
        if lo is not None:
            condition += f" AND start_ms>{lo}"
        if hi is not None:
            condition += f" AND start_ms<={hi} AND end_ms<={hi}"
        query = f"SELECT * FROM flows WHERE {condition}"
        if earlier:
            query += " AND predictor_hash NOT IN (" + " UNION ".join(f"SELECT predictor_hash FROM kept_{x}" for x in earlier) + ")"
        con.execute(f"CREATE OR REPLACE TABLE kept_{role} AS {query}")
        summary = con.execute(f"SELECT attack,count(*) AS count,min(start_ms) AS first_start,max(start_ms) AS last_start,max(end_ms) AS last_end FROM kept_{role} GROUP BY 1 ORDER BY 1").df()
        summary.insert(0, "role", role)
        records.extend(summary.to_dict("records"))
        output = BASE / "datasets" / f"{role}.parquet"
        if role == "fit":
            sample = f"SELECT * EXCLUDE(rank_in_class) FROM (SELECT *, row_number() OVER (PARTITION BY attack ORDER BY hash(row_id,20261018),row_id) AS rank_in_class FROM kept_{role}) WHERE rank_in_class<=CASE WHEN attack='Benign' THEN 80000 ELSE 25000 END"
        else:
            cap = protocol["caps"][f"{role}_total_uniform"]
            sample = f"SELECT * FROM kept_{role} ORDER BY hash(row_id,20261018),row_id LIMIT {cap}"
        con.execute(f"COPY ({sample}) TO '{sqlpath(output)}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        print(f"Cached {role}: {sum(summary['count'])} eligible source rows", flush=True)
        earlier.append(role)
    import pandas as pd
    pd.DataFrame(records).to_csv(BASE / "role_class_support_full.csv", index=False)
    overlap = []
    for i, a in enumerate(protocol["roles"]):
        for b in protocol["roles"][i+1:]:
            n = con.execute(f"SELECT count(*) FROM (SELECT DISTINCT predictor_hash FROM kept_{a}) a INNER JOIN (SELECT DISTINCT predictor_hash FROM kept_{b}) b USING (predictor_hash)").fetchone()[0]
            overlap.append({"left": a, "right": b, "overlap_hashes": n})
            assert n == 0
    dump(BASE / "data_audit.json", {"source_rows": source_rows, "invalid_time_rows": bad, "cutpoints_ms": cuts, "features": features, "predictor_overlap": overlap, "duckdb_version": duckdb.__version__, "note": "All source rows audited. Training/test models use deterministic bounded work subsets, not the full 27.5M population. Global within-source duplicate conflicts are not label-cleaned."})
    con.close()
    print("PREPARATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
