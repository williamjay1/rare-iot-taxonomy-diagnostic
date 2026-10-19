"""Read-only shared-37 collision audit. Prints measured JSON, never fits models."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path(r"legacy_project")
WORK = Path(r"workspace\harmonized")
OUT = WORK / "preflight_original_roles_COPY"
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import pandas as pd
import run_ciciot_agreement_gated_trf_confirmation as nested
import run_ciciot_taxonomy_residual_fusion as trf


def main():
    started = time.perf_counter()
    shared = json.loads((ROOT / "results" / "ijismd_revision_feature_overlap_v1" / "feature_overlap_report.json").read_text())["shared_features"]
    cache = WORK / "datasets" / "source_inputs"
    full = pd.concat([pd.read_parquet(cache / nested.TRAIN_PATH.name), pd.read_parquet(cache / nested.TEST_PATH.name)], ignore_index=True)
    original = nested.exp.feature_cols(full)
    label = full["Label"].astype(str)
    h46 = pd.util.hash_pandas_object(full[original], index=False).to_numpy(dtype=np.uint64)
    h37 = pd.util.hash_pandas_object(full[shared], index=False).to_numpy(dtype=np.uint64)
    hmodel = pd.util.hash_pandas_object(pd.DataFrame(trf.clean_features(full, shared), columns=shared), index=False).to_numpy(dtype=np.uint64)
    assert np.array_equal(h37 == h37[0], hmodel == hmodel[0])
    groups = pd.DataFrame({"h37": h37, "h46": h46, "label": label})
    counts = groups.groupby("h37", sort=False).agg(rows=("label", "size"), labels=("label", "nunique"), original46_groups=("h46", "nunique"))
    newly = counts.index[counts.original46_groups > 1]
    conflicting = counts.index[counts.labels > 1]
    base = {"rows": len(full), "original46_unique_groups": int(pd.Series(h46).nunique()),
            "shared37_unique_groups": len(counts),
            "shared37_duplicate_excess_rows": int(len(full) - len(counts)),
            "shared37_rows_in_duplicate_groups": int(counts.loc[counts.rows > 1, "rows"].sum()),
            "projection_new_collision_groups": len(newly),
            "projection_new_collision_rows": int(counts.loc[newly, "rows"].sum()),
            "projection_new_collision_row_pct": float(100 * counts.loc[newly, "rows"].sum() / len(full)),
            "shared37_conflicting_label_groups": len(conflicting),
            "shared37_conflicting_label_rows": int(counts.loc[conflicting, "rows"].sum()),
            "shared37_conflicting_labels": label[np.isin(h37, conflicting)].value_counts().to_dict(),
            "raw37_and_float32_group_partitions_equal": bool(np.array_equal(pd.factorize(h37)[0], pd.factorize(hmodel)[0]))}
    by_seed = []
    rare_collision = []
    for seed in range(42, 48):
        directory = OUT / f"seed_{seed}"
        proof = json.loads((directory / "data_role_proof.json").read_text())
        rare_labels = proof["rare_labels"]
        with np.load(directory / "role_row_ids.npz") as roles:
            fit = roles["fit_ordered_row_id"]
            pool = roles["fit_pool_row_id"]
            test = roles["test_row_id"]
            utility = roles["utility_row_id"]
            selection = roles["selection_row_id"]
        common = np.intersect1d(h37[fit], h37[test])
        test_hit = np.isin(h37[test], common)
        fit_hit = np.isin(h37[fit], common)
        rare_test = label.iloc[test].isin(rare_labels).to_numpy()
        rare_fit = label.iloc[fit].isin(rare_labels).to_numpy()
        dev = np.concatenate([pool, utility, selection])
        any_dev = np.isin(h37[test], h37[dev])
        record = {"seed": seed, "fit_rows_after_cap": len(fit), "test_rows": len(test),
                  "shared37_fit_test_overlap_groups": len(common),
                  "fit_rows_sharing_test_vector": int(fit_hit.sum()),
                  "test_rows_sharing_fit_vector": int(test_hit.sum()),
                  "test_rows_sharing_fit_vector_pct": float(100 * test_hit.mean()),
                  "rare_test_rows": int(rare_test.sum()),
                  "rare_test_rows_sharing_fit_vector": int((test_hit & rare_test).sum()),
                  "rare_test_rows_sharing_fit_vector_pct": float(100 * (test_hit & rare_test).sum() / rare_test.sum()),
                  "rare_fit_rows_sharing_test_vector": int((fit_hit & rare_fit).sum()),
                  "test_rows_sharing_any_development_vector": int(any_dev.sum()),
                  "test_rows_sharing_any_development_vector_pct": float(100 * any_dev.mean()),
                  "rare_test_rows_sharing_any_development_vector": int((any_dev & rare_test).sum())}
        for lab in rare_labels:
            mask = label.iloc[test].eq(lab).to_numpy()
            rare_collision.append({"seed": seed, "label": lab, "test_rows": int(mask.sum()),
                                   "test_rows_sharing_fit_vector": int((mask & test_hit).sum()),
                                   "test_rows_sharing_any_development_vector": int((mask & any_dev).sum())})
        by_seed.append(record)
    rare_names = json.loads((OUT / "seed_42" / "data_role_proof.json").read_text())["rare_labels"]
    base["rare_rows_in_projection_new_collision_groups"] = int((label.isin(rare_names).to_numpy() & np.isin(h37, newly)).sum())
    base["rare_rows_in_shared37_conflicting_groups"] = int((label.isin(rare_names).to_numpy() & np.isin(h37, conflicting)).sum())
    output = {"global": base, "per_seed": by_seed,
              "rare_per_class_nonzero": [r for r in rare_collision if r["test_rows_sharing_any_development_vector"]],
              "elapsed_seconds": time.perf_counter() - started,
              "note": "Read-only diagnostic. No training or manuscript edits. Percent denominators are explicit."}
    (WORK / "projection_collision_audit.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    pd.DataFrame(by_seed).to_csv(WORK / "projection_collision_by_seed.csv", index=False)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
