"""Expand one predeclared split's genuine random-map reference to 19 refits."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "2"

import pandas as pd

ROOT = Path(r"legacy_project").resolve()
OUT = Path(r"workspace\pseudo_null\expanded_seed42")
SCRIPTS = ROOT / "working/ijismd_v11_mechanical_completion_20260923/scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "scripts"))
import run_v10_grouped_pseudo_taxonomy as runner


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert OUT.resolve().drive.upper() == "D:"
    contract = {
        "split_seed": 42, "choice": "First numerical discovery split, not a selected best split",
        "total_random_maps": 19, "existing_maps_carried_forward": 4,
        "new_refits": 15, "trees": 380, "group_size_profile_unchanged": True,
        "map_seed_base": 20260923, "n_jobs": 2,
        "comparison": "Real equal taxonomy blend versus equal blends with random size-matched class partitions",
        "tail": "(1 + number random metric >= real metric)/(19 + 1); one split conditional random-map reference, not an independent-data significance test",
        "inference_limits": "No Fisher pooling across reused corpus splits; heterogeneous KMeans/GMM controls not counted as exchangeable random draws. Two metric tails reported separately without an omnibus significance claim.",
    }
    protocol = OUT / "protocol.json"
    if protocol.exists():
        assert json.loads(protocol.read_text()) == contract
    else:
        protocol.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    source = ROOT / "results/ijismd_v10_grouped_pseudo_taxonomy_capacity"
    partial = OUT / "pseudo_taxonomy_metrics_partial.csv"
    if not partial.exists():
        frame = pd.read_csv(source / "pseudo_taxonomy_metrics.csv")
        prior = frame[frame.seed.eq(42) & frame.map_type.eq("random_partition")]
        assert len(prior) == 4
        prior.to_csv(partial, index=False)
        (OUT / "maps").mkdir(exist_ok=True)
        for name in (source / "maps").glob("seed_42_random_partition_*.json"):
            shutil.copyfile(name, OUT / "maps" / name.name)
        audit = pd.read_csv(source / "grouped_role_audit.csv")
        audit[audit.seed.eq(42)].to_csv(OUT / "grouped_role_audit_partial.csv", index=False)
    for attr, filename in (("TRAIN_PATH", "train-00000-of-00001.parquet"), ("TEST_PATH", "test-00000-of-00001.parquet")):
        path = Path(r"workspace\harmonized\datasets\source_inputs") / filename
        assert path.exists() and path.resolve().drive.upper() == "D:"
        setattr(runner.nested, attr, path)
    sys.argv = [str(Path(__file__)), "--seeds", "42", "--random-replicates", "19", "--kmeans-replicates", "0", "--gmm-replicates", "0", "--n-jobs", "2", "--out-dir", str(OUT), "--resume"]
    runner.main()
    result = pd.read_csv(OUT / "pseudo_taxonomy_metrics.csv")
    assert len(result) == 19
    real = pd.read_csv(ROOT / "results/ijismd_v10_ciciot_grouped_core_6split/nested_split_results.csv")
    real = real[real.seed.eq(42) & real.run_id.eq("equal_flat_taxonomy_blend")].iloc[0]
    rows = []
    for metric in ("f1_macro", "rare_recall"):
        count = int((result[metric] >= real[metric]).sum())
        rows.append({"seed": 42, "metric": metric, "real": real[metric], "random_draws": 19,
                     "exceedances": count, "plus_one_tail": (count + 1) / 20,
                     "random_mean": result[metric].mean(), "random_min": result[metric].min(), "random_max": result[metric].max()})
    pd.DataFrame(rows).to_csv(OUT / "expanded_null_summary.csv", index=False)
    (OUT / "COMPLETED.json").write_text(json.dumps({"random_refits": 19, "new_refits": 15,
        "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest()}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
