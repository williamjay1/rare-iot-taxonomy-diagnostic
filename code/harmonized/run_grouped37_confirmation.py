"""Independent 37-input-hash grouped confirmation of the core taxonomy arms."""
from __future__ import annotations

import argparse
import gc
import itertools
import json
from pathlib import Path
import sys
import time
import warnings
from datetime import datetime, timezone

sys.dont_write_bytecode = True
import run_harmonized_refit as base
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from threadpoolctl import threadpool_limits

OUT = base.WORK / "results" / "grouped37"
METHODS = ["flat_weighted_xgboost", "equal_flat_taxonomy_blend",
           "equal_flat_cyclic_blend", "equal_flat_support_matched_blend",
           base.benefit.REAL_ID, base.benefit.CYCLIC_ID, base.benefit.SUPPORT_ID,
           "soft_taxonomy_factorisation", "hard_top_down_taxonomy"]


def setup():
    base.require_physical_d_output(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "temp").mkdir(exist_ok=True)
    import os
    os.environ["TEMP"] = os.environ["TMP"] = os.environ["JOBLIB_TEMP_FOLDER"] = str(OUT / "temp")
    primary = json.loads((base.CORE / "run_metadata.json").read_text())
    assert base.benefit.package_versions() == primary["package_versions"]
    parent_contract = json.loads((base.WORK / "results" / "paired46" / "run_contract.json").read_text())
    frames = []
    for item in parent_contract["inputs"]:
        path = Path(item["training_cache_path"])
        base.require_physical_d_output(path)
        assert base.sha256(path) == item["sha256"]
        frames.append(pd.read_parquet(path))
    full = pd.concat(frames, ignore_index=True)
    features = parent_contract["features"]
    clean = pd.DataFrame(base.trf.clean_features(full, features), columns=features)
    full[base.benefit.FEATURE_GROUP_COLUMN] = pd.util.hash_pandas_object(clean, index=False).to_numpy(dtype=np.uint64)
    full["__source_row_id"] = np.arange(len(full), dtype=np.int64)
    del clean
    # The observed projection has no conflicting groups. Modal labels remain a
    # stratification-only convention if a future immutable input contains any.
    counts = full.groupby([base.benefit.FEATURE_GROUP_COLUMN, "Label"], sort=False).size().rename("count").reset_index()
    counts["Label"] = counts["Label"].astype(str)
    representative = counts.sort_values([base.benefit.FEATURE_GROUP_COLUMN, "count", "Label"], ascending=[True, False, True]).drop_duplicates(base.benefit.FEATURE_GROUP_COLUMN).set_index(base.benefit.FEATURE_GROUP_COLUMN)["Label"]
    conflicts = int(counts.duplicated(base.benefit.FEATURE_GROUP_COLUMN, keep=False).groupby(counts[base.benefit.FEATURE_GROUP_COLUMN]).any().sum())
    config = {**base.CONFIG, "methods": METHODS,
              "grouping_basis": "37 cleaned float32 predictor values (the actual model input)",
              "stratification": "group modal fine label, ties lexicographic; observed labels never altered",
              "observed_conflicting_groups": conflicts,
              "roles_are_paired_with_original_PRIMARY": False,
              "calibrated_flat_controls": "not part of this additional core confirmation"}
    contract = {"config": config, "inputs": parent_contract["inputs"], "features": features,
                "parent_reference_contract_sha256": base.sha256(base.WORK / "results" / "paired46" / "run_contract.json"),
                "script_sha256": base.sha256(__file__), "base_script_sha256": base.sha256(base.__file__),
                "package_versions": base.benefit.package_versions()}
    path = OUT / "run_contract.json"
    if path.exists():
        assert json.loads(path.read_text()) == contract, "Grouped-37 contract changed"
    else:
        base.json_write(path, contract)
    return full, features, representative, primary


def split_groups(frame, representative, seed, train_size=None, test_size=None):
    groups = frame[base.benefit.FEATURE_GROUP_COLUMN].drop_duplicates()
    left, _ = train_test_split(groups.to_numpy(dtype=np.uint64), train_size=train_size,
                               test_size=test_size, random_state=seed,
                               stratify=representative.loc[groups].to_numpy())
    mask = frame[base.benefit.FEATURE_GROUP_COLUMN].isin(left)
    return frame.loc[mask], frame.loc[~mask]


def prepare_seed(full, features, representative, primary, encoder, seed):
    development, outer = split_groups(full, representative, seed, test_size=.20)
    pool, gate_pool = split_groups(development, representative, seed + 10000, train_size=.70)
    utility, selection = split_groups(gate_pool, representative, seed + 20000, train_size=.50)
    fit = base.nested.cap_fit_rows(pool, seed, 25000, 80000)
    y_pool = encoder.transform(pool["Label"].astype(str))
    rare = base.trf.rare_class_ids(y_pool)
    mapping, support_audit = base.support.support_matched_wrong_map(fit, primary["label_to_family"])
    directory = OUT / f"seed_{seed}"
    directory.mkdir(exist_ok=True)
    roles = {"fit_pool": pool, "utility": utility, "selection": selection, "test": outer}
    groups = {k: set(v[base.benefit.FEATURE_GROUP_COLUMN]) for k, v in roles.items()}
    overlaps = []
    for left, right in itertools.combinations(roles, 2):
        n = len(groups[left].intersection(groups[right]))
        assert n == 0, (seed, left, right, n)
        overlaps.append({"seed": seed, "basis": "shared37_float32_model_input", "left": left, "right": right, "overlap_groups": n})
    base.csv_write(directory / "predictor_group_overlap.csv", pd.DataFrame(overlaps))
    base.csv_write(directory / "support_match_audit.csv", support_audit)
    base.npz_write(directory / "role_row_ids.npz", **{f"{k}_row_id": v.index.to_numpy(dtype=np.int64) for k, v in roles.items()},
                   fit_ordered_row_id=fit["__source_row_id"].to_numpy(dtype=np.int64), rare_ids=rare)
    proof = {"seed": seed, "grouping_basis": "37 cleaned float32 input values",
             "roles_are_paired_with_PRIMARY": False,
             "base_pool_rows_before_cap": len(pool), "base_fit_rows_after_cap": len(fit),
             "gate_train_rows": len(utility), "gate_select_rows": len(selection), "outer_test_rows": len(outer),
             "role_order_sha256": {k: base.array_hash(v.index) for k, v in roles.items()},
             "fit_order_sha256": base.array_hash(fit["__source_row_id"]),
             "split_rng": {"outer": seed, "fit_vs_gate": seed + 10000, "utility_vs_selection": seed + 20000, "cap_and_shuffle": seed},
             "mapping": {"real": primary["label_to_family"], "cyclic": primary["cyclic_control_map"], "support": mapping},
             "rare_labels": encoder.inverse_transform(rare).tolist()}
    base.json_write(directory / "data_role_proof.json", proof)
    class_counts = []
    for name, frame in {**roles, "fit_after_cap": fit}.items():
        for label, count in frame["Label"].astype(str).value_counts().items():
            class_counts.append({"seed": seed, "role": name, "label": label, "rows": count})
    base.csv_write(directory / "class_support_by_role.csv", pd.DataFrame(class_counts))
    return fit, utility, selection, outer, np.bincount(y_pool, minlength=len(encoder.classes_)), rare, proof


def fit_seed(seed, fit, utility, selection, outer, supports, rare, proof, encoder, features):
    directory = OUT / f"seed_{seed}"
    base.trf.RANDOM_STATE = seed
    frames = {"utility": utility, "selection": selection, "test": outer}
    y_utility = encoder.transform(utility["Label"].astype(str))
    y_selection = encoder.transform(selection["Label"].astype(str))
    flat, flat_seconds = base.checkpoint(directory / "flat_xgb.joblib", lambda: base.nested.fit_weighted_xgb(fit, features, encoder, 520)[0])
    flat_p, flat_pred_seconds = base.checkpoint(directory / "flat_probabilities.joblib",
            lambda: {role: np.asarray(flat.predict_proba(base.trf.clean_features(frame, features)), dtype=np.float64) for role, frame in frames.items()})
    base.save_method(directory, seed, "flat_weighted_xgboost", flat_p["test"], outer, encoder, rare, flat_seconds, flat_pred_seconds)
    del flat
    for kind, equal_name, gate_name in (("real", "equal_flat_taxonomy_blend", base.benefit.REAL_ID),
                                      ("cyclic", "equal_flat_cyclic_blend", base.benefit.CYCLIC_ID),
                                      ("support", "equal_flat_support_matched_blend", base.benefit.SUPPORT_ID)):
        base.event(f"GROUPED37 SEED {seed} hierarchy={kind}")
        mapping = proof["mapping"][kind]
        family_encoder = LabelEncoder().fit(list(mapping.values()))

        def build_taxonomy():
            if kind == "real":
                model, locals_, _ = base.nested.fit_real_taxonomy(fit, features, family_encoder, 380)
                return {"family_model": model, "family_encoder": family_encoder, "local_models": locals_}
            model, enc, locals_, _ = base.trf.fit_control_hierarchy(fit, features, mapping, 380)
            return {"family_model": model, "family_encoder": enc, "local_models": locals_}

        taxonomy, tax_seconds = base.checkpoint(directory / f"taxonomy_{kind}.joblib", build_taxonomy)
        tax_p, tax_pred_seconds = base.checkpoint(directory / f"taxonomy_{kind}_probabilities.joblib",
            lambda: {role: base.benefit.predict_taxonomy(frame, features, taxonomy["family_model"], taxonomy["family_encoder"], encoder, mapping, taxonomy["local_models"]) for role, frame in frames.items()})
        probability = .5 * flat_p["test"] + .5 * tax_p["test"][0]
        base.save_method(directory, seed, equal_name, probability, outer, encoder, rare, flat_seconds + tax_seconds, flat_pred_seconds + tax_pred_seconds)
        del probability
        if kind == "real":
            base.save_method(directory, seed, "soft_taxonomy_factorisation", tax_p["test"][0], outer, encoder, rare, tax_seconds, tax_pred_seconds)
            hard = base.nested.hard_hierarchy_probability(tax_p["test"][0], tax_p["test"][1], encoder, taxonomy["family_encoder"], mapping)
            base.save_method(directory, seed, "hard_top_down_taxonomy", hard, outer, encoder, rare, tax_seconds, tax_pred_seconds)
            del hard

        def build_gate():
            model, counts = base.benefit.fit_utility_model(y_utility, flat_p["utility"], *tax_p["utility"], encoder, taxonomy["family_encoder"], mapping, supports, rare, seed)
            chosen, grid = base.benefit.select_benefit_gate(y_selection, flat_p["selection"], *tax_p["selection"], model, encoder, taxonomy["family_encoder"], mapping, supports, rare, .04)
            base.csv_write(directory / f"selection_grid_{kind}.csv", grid)
            return {"model": model, "selection": chosen, "utility_counts": counts}

        gate, gate_seconds = base.checkpoint(directory / f"gate_{kind}.joblib", build_gate)
        started = time.perf_counter()
        probability, active, counts, score = base.benefit.evaluate_gate(encoder.transform(outer["Label"].astype(str)), flat_p["test"], *tax_p["test"], gate["model"], gate["selection"], encoder, taxonomy["family_encoder"], mapping, supports, rare)
        infer = time.perf_counter() - started
        base.json_write(directory / f"gate_{kind}_audit.json", {"selection": gate["selection"], "utility_counts": gate["utility_counts"], "outer_counts": counts,
                        "utility_fit_role": "utility", "hyperparameter_selection_role": "selection", "outer_used_for_fitting_or_selection": False})
        base.save_method(directory, seed, gate_name, probability, outer, encoder, rare, flat_seconds + tax_seconds + gate_seconds, flat_pred_seconds + tax_pred_seconds + infer)
        del taxonomy, tax_p, gate, probability, active, counts, score
        gc.collect()
    base.json_write(directory / "COMPLETED.json", {"seed": seed, "methods": METHODS,
                    "completed_utc": datetime.now(timezone.utc).isoformat(), "run_contract_sha256": base.sha256(OUT / "run_contract.json")})


def aggregate():
    available = [pd.read_csv(path) for seed in base.SEEDS for method in METHODS
                 if (path := OUT / f"seed_{seed}" / f"metrics_{method}.csv").exists()]
    if not available:
        return
    frame = pd.concat(available, ignore_index=True)
    base.csv_write(OUT / "harmonized_results_partial.csv", frame)
    if len(frame) != len(base.SEEDS) * len(METHODS):
        return
    assert all((OUT / f"seed_{s}" / "COMPLETED.json").exists() for s in base.SEEDS)
    base.csv_write(OUT / "harmonized_results.csv", frame)
    summary = frame.groupby("run_id").agg(seeds=("seed", "nunique"), f1_macro_mean=("f1_macro", "mean"), f1_macro_sd=("f1_macro", "std"), rare_recall_mean=("rare_recall", "mean"), rare_recall_sd=("rare_recall", "std"), accuracy_mean=("accuracy", "mean"), pr_auc_macro_mean=("pr_auc_macro", "mean"), brier_mean=("brier_multiclass", "mean"), ece_mean=("ece_top_label", "mean")).reset_index()
    base.csv_write(OUT / "harmonized_summary.csv", summary)
    deltas = []
    for seed in base.SEEDS:
        scores = frame.loc[frame.seed == seed].set_index("run_id")
        for control in ("flat_weighted_xgboost", "equal_flat_cyclic_blend", "equal_flat_support_matched_blend", base.benefit.CYCLIC_ID, base.benefit.SUPPORT_ID):
            deltas.append({"seed": seed, "treatment": "equal_flat_taxonomy_blend", "control": control,
                           **{f"delta_{metric}": scores.loc["equal_flat_taxonomy_blend", metric] - scores.loc[control, metric]
                              for metric in ("f1_macro", "rare_recall", "accuracy", "pr_auc_macro", "brier_multiclass", "ece_top_label")}})
    base.csv_write(OUT / "harmonized_paired_deltas.csv", pd.DataFrame(deltas))
    base.csv_write(OUT / "predictor_group_overlap_all_seeds.csv", pd.concat([pd.read_csv(OUT / f"seed_{s}" / "predictor_group_overlap.csv") for s in base.SEEDS], ignore_index=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)
    started = time.perf_counter()
    full, features, representative, primary = setup()
    base.OUT = OUT
    encoder = LabelEncoder().fit(full["Label"].astype(str))
    for seed in base.SEEDS:
        if not args.preflight_only and (OUT / f"seed_{seed}" / "COMPLETED.json").exists():
            base.event(f"RESUME COMPLETED GROUPED37 SEED {seed}")
            continue
        values = prepare_seed(full, features, representative, primary, encoder, seed)
        base.event(f"GROUPED37 ROLES VERIFIED seed={seed} counts={len(values[0])}/{len(values[1])}/{len(values[2])}/{len(values[3])}")
        if not args.preflight_only:
            seed_started = time.perf_counter()
            with threadpool_limits(limits=2):
                fit_seed(seed, *values, encoder, features)
            base.json_write(OUT / f"seed_{seed}" / "runtime.json", {"wall_seconds_this_invocation": time.perf_counter() - seed_started})
            aggregate()
        del values
        gc.collect()
    base.json_write(OUT / ("preflight_runtime.json" if args.preflight_only else "invocation_runtime.json"),
                    {"elapsed_seconds": time.perf_counter() - started, "preflight_only": args.preflight_only,
                     "python": sys.executable, "command": sys.argv, "finished_utc": datetime.now(timezone.utc).isoformat()})
    if not args.preflight_only:
        aggregate()


if __name__ == "__main__":
    main()
