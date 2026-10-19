"""Refit the frozen PRIMARY six holdouts using the 37 shared predictors.

Existing project modules are imported read-only. The only constructor override
is n_jobs=2. Role assignment remains based on the original 46 predictors so
this is a paired feature ablation, not a replacement grouped protocol.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import shutil
import sys
import time
import warnings
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = Path(r"legacy_project")
WORK = Path(r"workspace\harmonized")
OUT = WORK / "results" / "paired46"
# All temporary files, bytecode and model outputs stay inside the owned paths.
sys.dont_write_bytecode = True
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "2"
os.environ["NUMEXPR_MAX_THREADS"] = "2"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["PYTHONPYCACHEPREFIX"] = str(OUT / "temp" / "bytecode")
os.environ["TEMP"] = os.environ["TMP"] = str(OUT / "temp")
os.environ["JOBLIB_TEMP_FOLDER"] = str(OUT / "temp")
sys.path[:0] = [str(ROOT / "scripts"),
                str(ROOT / "working" / "ijismd_v10_risk_closure_20260923" / "scripts")]

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from threadpoolctl import threadpool_limits

import run_ciciot_agreement_gated_trf_confirmation as nested
import run_ciciot_calibrated_benefit_gate_confirmation as benefit
import run_ciciot_support_matched_control_v2 as support
import run_ciciot_taxonomy_residual_fusion as trf
import run_v10_grouped_flat_comparators as comparator
import run_v10_grouped_newton_flat_stacker as newton

SOURCE_INPUTS = (nested.TRAIN_PATH, nested.TEST_PATH)

SEEDS = [42, 43, 44, 45, 46, 47]
CORE = ROOT / "results" / "ijismd_v10_ciciot_grouped_core_6split"
OVERLAP = ROOT / "results" / "ijismd_revision_feature_overlap_v1" / "feature_overlap_report.json"
METHODS = [
    "flat_weighted_xgboost", "flat_weighted_lightgbm",
    "flat_equal_probability_blend", "flat_temperature_calibrated_blend",
    "flat_newton_log_probability_stacker", "equal_flat_taxonomy_blend",
    "soft_taxonomy_factorisation", "hard_top_down_taxonomy",
    benefit.REAL_ID, "equal_flat_cyclic_blend", benefit.CYCLIC_ID,
    "equal_flat_support_matched_blend", benefit.SUPPORT_ID,
]
CONFIG = {
    "seeds": SEEDS, "outer_test_frac": 0.20, "base_fraction": 0.70,
    "gate_train_fraction": 0.15, "gate_select_fraction": 0.15,
    "majority_cap": 25000, "benign_cap": 80000,
    "global_trees": 520, "local_trees": 380, "rare_lambda": 0.04,
    "n_jobs": 2, "group_disjoint": True,
    "grouping_basis": "original 46 predictors; identical PRIMARY roles",
    "temperature_bounds": [0.25, 5.0], "temperature_xatol": 1e-4,
    "stacker": {"C": 1.0, "max_iter": 100, "solver": "newton-cholesky"},
    "methods": METHODS,
    "metric_probability_precision": "float64 (checkpointed without quantization)",
    "venue_route": "EI engineering empirical validation; indexing is not a method standard",
    "flat_calibration_fit_role": "utility (same ordered rows as v10 reference calibration)",
    "flat_stacker_fit_role": "utility (same ordered rows as v10 reference calibration)",
    "flat_selection_role_permission": "diagnostic evaluation only, no fitting or model choice",
}

_global_factory = trf.make_global_model
_lgbm_factory = trf.make_lgbm


def limited_global(n_classes, n_estimators):
    return _global_factory(n_classes, n_estimators).set_params(n_jobs=2)


def limited_lgbm(n_classes, n_estimators):
    return _lgbm_factory(n_classes, n_estimators).set_params(n_jobs=2)


trf.make_global_model = limited_global
trf.make_lgbm = limited_lgbm


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_hash(values):
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def json_write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def csv_write(path, frame):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def npz_write(path, **values):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)


def event(message):
    line = f"{datetime.now(timezone.utc).isoformat()} {message}"
    print(line, flush=True)
    with (OUT / "execution.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def model_parameters(value):
    if hasattr(value, "get_params"):
        params = value.get_params(deep=False)
        if "n_jobs" in params:
            assert params["n_jobs"] == 2, params
        return {"type": type(value).__name__, "params": params}
    if isinstance(value, dict):
        return {str(k): model_parameters(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [model_parameters(v) for v in value]
    return str(value)


def checkpoint(path, builder):
    """Commit model/prediction plus digest; an interrupted write is never reused."""
    path = Path(path)
    meta = path.with_name(path.name + ".json")
    if path.exists() and meta.exists():
        info = json.loads(meta.read_text())
        if sha256(path) != info["sha256"]:
            raise RuntimeError(f"Checkpoint digest mismatch: {path}")
        event(f"RESUME {path.relative_to(OUT)}")
        return joblib.load(path), info["elapsed_seconds"]
    started = time.perf_counter()
    value = builder()
    elapsed = time.perf_counter() - started
    temporary = path.with_name(path.name + ".tmp")
    joblib.dump(value, temporary, compress=3)
    temporary.replace(path)
    json_write(meta, {"sha256": sha256(path), "elapsed_seconds": elapsed,
                      "models": model_parameters(value), "saved_utc": datetime.now(timezone.utc).isoformat()})
    event(f"SAVED {path.relative_to(OUT)} seconds={elapsed:.2f}")
    return value, elapsed


def require_physical_d_output(path):
    resolved = Path(path).resolve()
    if resolved.drive.upper() != "D:":
        raise RuntimeError(
            f"Storage blocker: output {path} resolves to {resolved}, not physical D:. "
            "The project root is a junction to F:. No training may start. "
            "Authorize a physical D: work directory and an output junction, or "
            "provide an owned output directory already resident on D:."
        )
    return resolved


def initialize():
    require_physical_d_output(WORK)
    require_physical_d_output(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "temp").mkdir(exist_ok=True)
    require_physical_d_output(OUT / "temp")
    free = shutil.disk_usage(OUT.resolve()).free
    if free < 20 * 1024**3:
        raise RuntimeError(f"D: needs at least 20 GiB free; available {free}")
    primary = json.loads((CORE / "run_metadata.json").read_text())
    for key in ("seeds", "outer_test_frac", "base_fraction", "gate_train_fraction",
                "gate_select_fraction", "majority_cap", "benign_cap",
                "global_trees", "local_trees", "rare_lambda", "group_disjoint"):
        assert CONFIG[key] == primary["arguments"][key], key
    versions = benefit.package_versions()
    assert versions == primary["package_versions"], (versions, primary["package_versions"])
    features = json.loads(OVERLAP.read_text())["shared_features"]
    assert len(features) == len(set(features)) == 37
    expected_hashes = ["bcc477f125aa7cecbeba50cd3b9641e20dcf521e882c437a704f73c30c96b353",
                       "16eb6b12e6443a270a09a90c08adf90279d0bcb374f2346f143ac069df99ac6b"]
    inputs = []
    cache = WORK / "datasets" / "source_inputs"
    require_physical_d_output(cache)
    cache.mkdir(parents=True, exist_ok=True)
    cached_paths = []
    for path, expected in zip(SOURCE_INPUTS, expected_hashes):
        digest = sha256(path)
        assert digest == expected, str(path)
        destination = cache / path.name
        if not destination.exists():
            temporary = destination.with_name(destination.name + ".tmp")
            shutil.copyfile(path, temporary)
            assert sha256(temporary) == expected
            temporary.replace(destination)
            import stat
            destination.chmod(stat.S_IREAD)
            event(f"COPIED READ-ONLY SOURCE ONCE {path} -> {destination}")
        assert sha256(destination) == expected, f"D cache digest mismatch: {destination}"
        cached_paths.append(destination)
        inputs.append({"source_path": str(path), "source_resolved": str(path.resolve()),
                       "training_cache_path": str(destination), "training_cache_resolved": str(destination.resolve()),
                       "sha256": digest, "bytes": path.stat().st_size})
    nested.TRAIN_PATH, nested.TEST_PATH = cached_paths
    sources = {str(Path(module.__file__).resolve()): sha256(module.__file__)
               for module in list(sys.modules.values())
               if getattr(module, "__file__", None)
               and str(Path(module.__file__).resolve()).startswith(str(ROOT / "scripts"))}
    for module in (comparator, newton):
        sources[str(Path(module.__file__))] = sha256(module.__file__)
    sources[str(Path(__file__).resolve())] = sha256(__file__)
    contract = {"config": CONFIG, "features": features, "inputs": inputs,
                "sources": sources, "package_versions": versions,
                "primary_metadata_sha256": sha256(CORE / "run_metadata.json"),
                "primary_role_audit_sha256": sha256(CORE / "nested_split_audit.csv"),
                "overlap_report_sha256": sha256(OVERLAP)}
    contract_path = OUT / "run_contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            assert not list(OUT.rglob("*.joblib")), "Contract changed; refusing stale model checkpoints"
            assert not list(OUT.rglob("metrics_*.csv")), "Contract changed; refusing stale metrics"
            old_hash = sha256(contract_path)
            archive = OUT / f"run_contract_preflight_only_{old_hash[:12]}.json"
            if not archive.exists():
                shutil.copyfile(contract_path, archive)
            json_write(contract_path, contract)
    else:
        json_write(contract_path, contract)
    event(f"PREFLIGHT python={sys.executable} D_free_GiB={free / 1024**3:.2f} versions={versions}")
    full = pd.concat([pd.read_parquet(cached_paths[0]), pd.read_parquet(cached_paths[1])], ignore_index=True)
    original_features = nested.exp.feature_cols(full)
    assert original_features == primary["features"] and len(original_features) == 46
    benefit.assign_exact_predictor_groups(full, original_features)
    # Keep raw and actual float32 model-input collisions separate in the audit.
    full["__shared37_raw_hash"] = pd.util.hash_pandas_object(full[features], index=False).to_numpy(dtype=np.uint64)
    clean = pd.DataFrame(trf.clean_features(full, features), columns=features)
    full["__shared37_model_hash"] = pd.util.hash_pandas_object(clean, index=False).to_numpy(dtype=np.uint64)
    del clean
    full["__source_row_id"] = np.arange(len(full), dtype=np.int64)
    return full, features, primary


def roles_for_seed(full, seed, encoder, primary):
    development, outer = benefit.stratified_group_split(full, test_size=0.20, seed=seed)
    pool, utility, selection = benefit.split_development(development, seed, .70, .15, .15, True)
    fit = nested.cap_fit_rows(pool, seed, 25000, 80000)
    roles = {"fit_pool": pool, "utility": utility, "selection": selection, "test": outer}
    counts = {"development_rows": len(development), "base_pool_rows_before_cap": len(pool),
              "base_fit_rows_after_cap": len(fit), "gate_train_rows": len(utility),
              "gate_select_rows": len(selection), "outer_test_rows": len(outer)}
    audit = pd.read_csv(CORE / "nested_split_audit.csv").set_index("seed").loc[seed]
    for key, value in counts.items():
        assert value == int(audit[key]), (seed, key, value, audit[key])
    y_pool = encoder.transform(pool["Label"].astype(str))
    rare = trf.rare_class_ids(y_pool)
    with np.load(CORE / f"paired_predictions_seed_{seed}.npz") as stored:
        assert np.array_equal(outer.index, stored["row_id"]), seed
        assert np.array_equal(encoder.transform(outer["Label"].astype(str)), stored["y_true"]), seed
        assert np.array_equal(rare, stored["rare_ids"]), seed
    support_map, support_audit = support.support_matched_wrong_map(fit, primary["label_to_family"])
    assert support_map == primary["support_maps_by_seed"][str(seed)]
    directory = OUT / f"seed_{seed}"
    directory.mkdir(exist_ok=True)
    csv_write(directory / "support_match_audit.csv", support_audit)
    npz_write(directory / "role_row_ids.npz", **{f"{k}_row_id": v.index.to_numpy(dtype=np.int64) for k, v in roles.items()},
              fit_ordered_row_id=fit["__source_row_id"].to_numpy(dtype=np.int64), rare_ids=rare)
    pairs = []
    for basis, column in (("original46", benefit.FEATURE_GROUP_COLUMN),
                          ("shared37_raw", "__shared37_raw_hash"),
                          ("shared37_float32_model_input", "__shared37_model_hash")):
        groups = {k: set(frame[column]) for k, frame in roles.items()}
        for left, right in itertools.combinations(roles, 2):
            common = groups[left].intersection(groups[right])
            if basis == "original46":
                assert not common, (seed, left, right)
            pairs.append({"seed": seed, "basis": basis, "left": left, "right": right,
                          "overlap_groups": len(common),
                          "left_rows_in_overlap": int(roles[left][column].isin(common).sum()),
                          "right_rows_in_overlap": int(roles[right][column].isin(common).sum())})
    csv_write(directory / "predictor_group_overlap.csv", pd.DataFrame(pairs))
    proof = {"seed": seed, **counts, "primary_outer_row_order_equal": True,
             "primary_labels_equal": True, "primary_rare_ids_equal": True,
             "primary_support_map_equal": True, "primary_grouping_features_equal": True,
             "role_order_sha256": {k: array_hash(v.index) for k, v in roles.items()},
             "fit_order_sha256": array_hash(fit["__source_row_id"]),
             "split_rng": {"outer": seed, "fit_vs_gate": seed + 10000,
                           "utility_vs_selection": seed + 20000, "cap_and_shuffle": seed},
             "mapping": {"real": primary["label_to_family"], "cyclic": primary["cyclic_control_map"],
                         "support": support_map},
             "rare_labels": encoder.inverse_transform(rare).tolist()}
    json_write(directory / "data_role_proof.json", proof)
    class_counts = []
    for name, frame in {**roles, "fit_after_cap": fit}.items():
        for label, count in frame["Label"].astype(str).value_counts().items():
            class_counts.append({"seed": seed, "role": name, "label": label, "rows": count})
    csv_write(directory / "class_support_by_role.csv", pd.DataFrame(class_counts))
    return fit, utility, selection, outer, np.bincount(y_pool, minlength=len(encoder.classes_)), rare, proof


def save_method(directory, seed, name, probability, outer, encoder, rare, train_seconds, infer_seconds):
    path = directory / f"metrics_{name}.csv"
    if path.exists():
        return
    probability = np.asarray(probability, dtype=np.float64)
    assert np.isfinite(probability).all() and probability.min() >= 0
    assert np.max(np.abs(probability.sum(axis=1) - 1)) < 1e-5
    y = encoder.transform(outer["Label"].astype(str))
    row = trf.metric_row(name, y, probability, rare)
    row.update({"seed": seed, "n_features": 37, "training_seconds": train_seconds,
                "inference_seconds": infer_seconds,
                "brier_multiclass": comparator.multiclass_brier(y, probability),
                "ece_top_label": comparator.expected_calibration_error(y, probability),
                "rare_ece_top_label": comparator.expected_calibration_error(y[np.isin(y, rare)], probability[np.isin(y, rare)])})
    npz_write(directory / f"predictions_{name}.npz", row_id=outer.index.to_numpy(dtype=np.int64),
              y_true=y.astype(np.int16), rare_ids=rare.astype(np.int16), probability=probability)
    csv_write(directory / f"classwise_{name}.csv", nested.classwise_frame(seed, name, y, probability, encoder.classes_.tolist(), rare))
    csv_write(path, pd.DataFrame([row]))
    event(f"METRIC seed={seed} {name} f1={row['f1_macro']:.7f} rare={row['rare_recall']:.7f}")


def fit_seed(seed, fit, utility, selection, outer, class_support, rare, encoder, features, proof):
    directory = OUT / f"seed_{seed}"
    trf.RANDOM_STATE = seed
    frames = {"utility": utility, "selection": selection, "test": outer}
    y_utility = encoder.transform(utility["Label"].astype(str))
    y_selection = encoder.transform(selection["Label"].astype(str))

    flat, flat_seconds = checkpoint(directory / "flat_xgb.joblib", lambda: nested.fit_weighted_xgb(fit, features, encoder, 520)[0])

    def predict_flat():
        return {role: np.asarray(flat.predict_proba(trf.clean_features(frame, features)), dtype=np.float64)
                for role, frame in frames.items()}

    flat_p, flat_pred_seconds = checkpoint(directory / "flat_probabilities.joblib", predict_flat)
    save_method(directory, seed, "flat_weighted_xgboost", flat_p["test"], outer, encoder, rare, flat_seconds, flat_pred_seconds)
    del flat
    gc.collect()

    for kind, equal_name, gate_name in (("real", "equal_flat_taxonomy_blend", benefit.REAL_ID),
                                       ("cyclic", "equal_flat_cyclic_blend", benefit.CYCLIC_ID),
                                       ("support", "equal_flat_support_matched_blend", benefit.SUPPORT_ID)):
        event(f"SEED {seed} hierarchy={kind}")
        mapping = proof["mapping"][kind]
        family_encoder = LabelEncoder().fit(list(mapping.values()))

        def build_taxonomy():
            if kind == "real":
                model, locals_, _ = nested.fit_real_taxonomy(fit, features, family_encoder, 380)
                return {"family_model": model, "family_encoder": family_encoder, "local_models": locals_}
            model, enc, locals_, _ = trf.fit_control_hierarchy(fit, features, mapping, 380)
            return {"family_model": model, "family_encoder": enc, "local_models": locals_}

        taxonomy, taxonomy_seconds = checkpoint(directory / f"taxonomy_{kind}.joblib", build_taxonomy)

        def predict_taxonomy():
            return {role: benefit.predict_taxonomy(frame, features, taxonomy["family_model"],
                    taxonomy["family_encoder"], encoder, mapping, taxonomy["local_models"])
                    for role, frame in frames.items()}

        tax_p, tax_pred_seconds = checkpoint(directory / f"taxonomy_{kind}_probabilities.joblib", predict_taxonomy)
        equal = 0.5 * flat_p["test"] + 0.5 * tax_p["test"][0]
        save_method(directory, seed, equal_name, equal, outer, encoder, rare, flat_seconds + taxonomy_seconds, flat_pred_seconds + tax_pred_seconds)
        del equal
        if kind == "real":
            save_method(directory, seed, "soft_taxonomy_factorisation", tax_p["test"][0], outer, encoder, rare, taxonomy_seconds, tax_pred_seconds)
            hard = nested.hard_hierarchy_probability(tax_p["test"][0], tax_p["test"][1], encoder, taxonomy["family_encoder"], mapping)
            save_method(directory, seed, "hard_top_down_taxonomy", hard, outer, encoder, rare, taxonomy_seconds, tax_pred_seconds)
            del hard

        def build_gate():
            model, counts = benefit.fit_utility_model(y_utility, flat_p["utility"], *tax_p["utility"],
                           encoder, taxonomy["family_encoder"], mapping, class_support, rare, seed)
            chosen, grid = benefit.select_benefit_gate(y_selection, flat_p["selection"], *tax_p["selection"],
                           model, encoder, taxonomy["family_encoder"], mapping, class_support, rare, .04)
            csv_write(directory / f"selection_grid_{kind}.csv", grid)
            return {"model": model, "selection": chosen, "utility_counts": counts}

        gate, gate_seconds = checkpoint(directory / f"gate_{kind}.joblib", build_gate)
        started = time.perf_counter()
        probability, active, counts, score = benefit.evaluate_gate(encoder.transform(outer["Label"].astype(str)),
                         flat_p["test"], *tax_p["test"], gate["model"], gate["selection"], encoder,
                         taxonomy["family_encoder"], mapping, class_support, rare)
        gate_pred_seconds = time.perf_counter() - started
        json_write(directory / f"gate_{kind}_audit.json", {"selection": gate["selection"],
                   "utility_counts": gate["utility_counts"], "outer_counts": counts,
                   "utility_fit_role": "utility", "hyperparameter_selection_role": "selection",
                   "outer_used_for_fitting_or_selection": False})
        save_method(directory, seed, gate_name, probability, outer, encoder, rare,
                    flat_seconds + taxonomy_seconds + gate_seconds, flat_pred_seconds + tax_pred_seconds + gate_pred_seconds)
        del probability, active, counts, score, gate, tax_p, taxonomy
        gc.collect()

    lgbm, lgbm_seconds = checkpoint(directory / "flat_lgbm.joblib", lambda: nested.fit_flat_lgbm(fit, features, encoder, 380)[0])
    lgbm_p, lgbm_pred_seconds = checkpoint(directory / "flat_lgbm_probabilities.joblib",
             lambda: {role: comparator.normalise(lgbm.predict_proba(trf.clean_features(frame, features))) for role, frame in frames.items()})
    save_method(directory, seed, "flat_weighted_lightgbm", lgbm_p["test"], outer, encoder, rare, lgbm_seconds, lgbm_pred_seconds)
    equal = .5 * flat_p["test"] + .5 * lgbm_p["test"]
    save_method(directory, seed, "flat_equal_probability_blend", equal, outer, encoder, rare, flat_seconds + lgbm_seconds, flat_pred_seconds + lgbm_pred_seconds)
    del equal, lgbm

    def build_flat_calibration():
        xgb_t, xgb_loss = comparator.fit_temperature(y_utility, flat_p["utility"])
        lgbm_t, lgbm_loss = comparator.fit_temperature(y_utility, lgbm_p["utility"])
        return {"xgb_temperature": xgb_t, "lgbm_temperature": lgbm_t,
                "xgb_utility_loss": xgb_loss, "lgbm_utility_loss": lgbm_loss}

    temperatures, calibration_seconds = checkpoint(directory / "flat_temperatures.joblib", build_flat_calibration)
    probability = .5 * comparator.temperature_probability(flat_p["test"], temperatures["xgb_temperature"]) + .5 * comparator.temperature_probability(lgbm_p["test"], temperatures["lgbm_temperature"])
    save_method(directory, seed, "flat_temperature_calibrated_blend", probability, outer, encoder, rare,
                flat_seconds + lgbm_seconds + calibration_seconds, flat_pred_seconds + lgbm_pred_seconds)
    del probability
    stacker, stacker_seconds = checkpoint(directory / "flat_newton_stacker.joblib",
            lambda: newton.fit_converged_stacker(comparator.probability_features(flat_p["utility"], lgbm_p["utility"]), y_utility, seed, 1., 100))
    started = time.perf_counter()
    probability = comparator.normalise(stacker.predict_proba(comparator.probability_features(flat_p["test"], lgbm_p["test"])))
    stacker_pred_seconds = time.perf_counter() - started
    save_method(directory, seed, "flat_newton_log_probability_stacker", probability, outer, encoder, rare,
                flat_seconds + lgbm_seconds + stacker_seconds, flat_pred_seconds + lgbm_pred_seconds + stacker_pred_seconds)
    json_write(directory / "flat_calibration_audit.json", {**temperatures, "role": "utility",
               "selection_used_for_choice": False, "outer_used_for_fitting_or_choice": False,
               "stacker_iterations": stacker.n_iter_.tolist(), "stacker_converged": True})
    for method in METHODS:
        assert (directory / f"metrics_{method}.csv").exists(), method
        assert (directory / f"predictions_{method}.npz").exists(), method
    json_write(directory / "COMPLETED.json", {"seed": seed, "methods": METHODS,
               "completed_utc": datetime.now(timezone.utc).isoformat(),
               "run_contract_sha256": sha256(OUT / "run_contract.json")})
    event(f"COMPLETED SEED {seed}")


def aggregate():
    files = [OUT / f"seed_{seed}" / f"metrics_{method}.csv" for seed in SEEDS for method in METHODS]
    available = [pd.read_csv(path) for path in files if path.exists()]
    if not available:
        return
    metrics = pd.concat(available, ignore_index=True).sort_values(["seed", "run_id"])
    csv_write(OUT / "harmonized_results_partial.csv", metrics)
    if len(metrics) != len(SEEDS) * len(METHODS):
        return
    for seed in SEEDS:
        assert (OUT / f"seed_{seed}" / "COMPLETED.json").exists()
    csv_write(OUT / "harmonized_results.csv", metrics)
    summary = metrics.groupby("run_id").agg(
        seeds=("seed", "nunique"), f1_macro_mean=("f1_macro", "mean"), f1_macro_sd=("f1_macro", "std"),
        rare_recall_mean=("rare_recall", "mean"), rare_recall_sd=("rare_recall", "std"),
        accuracy_mean=("accuracy", "mean"), pr_auc_macro_mean=("pr_auc_macro", "mean"),
        brier_mean=("brier_multiclass", "mean"), ece_mean=("ece_top_label", "mean")).reset_index()
    csv_write(OUT / "harmonized_summary.csv", summary)
    deltas = []
    for seed in SEEDS:
        values = metrics.loc[metrics.seed == seed].set_index("run_id")
        for treatment, control in [("equal_flat_taxonomy_blend", m) for m in (
            "flat_weighted_xgboost", "equal_flat_cyclic_blend", "equal_flat_support_matched_blend",
            "flat_equal_probability_blend", "flat_temperature_calibrated_blend", "flat_newton_log_probability_stacker")]:
            deltas.append({"seed": seed, "treatment": treatment, "control": control,
                           **{f"delta_{metric}": values.loc[treatment, metric] - values.loc[control, metric]
                              for metric in ("f1_macro", "rare_recall", "pr_auc_macro", "accuracy", "brier_multiclass", "ece_top_label")}})
    csv_write(OUT / "harmonized_paired_deltas.csv", pd.DataFrame(deltas))
    original = pd.read_csv(CORE / "nested_split_results.csv")
    comparison = metrics.merge(original, on=["seed", "run_id"], suffixes=("_37", "_46"))
    for metric in ("f1_macro", "rare_recall", "pr_auc_macro", "accuracy"):
        comparison[f"delta_{metric}_37_minus_46"] = comparison[f"{metric}_37"] - comparison[f"{metric}_46"]
    csv_write(OUT / "harmonized_vs_primary46.csv", comparison)
    overlaps = pd.concat([pd.read_csv(OUT / f"seed_{s}" / "predictor_group_overlap.csv") for s in SEEDS], ignore_index=True)
    csv_write(OUT / "predictor_group_overlap_all_seeds.csv", overlaps)
    event("ALL SIX SEEDS COMPLETE; paired deltas are descriptive, not independent replications")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--check-storage", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS, choices=SEEDS)
    args = parser.parse_args()
    if args.check_storage:
        print(require_physical_d_output(OUT))
        return
    warnings.filterwarnings("ignore", category=UserWarning)
    start = time.perf_counter()
    full, features, primary = initialize()
    encoder = LabelEncoder().fit(full["Label"].astype(str))
    cyc, shift = trf.cyclic_control_map(primary["label_to_family"])
    assert cyc == primary["cyclic_control_map"] and shift == primary["cyclic_control_shift"]
    for seed in args.seeds:
        if not args.preflight_only and (OUT / f"seed_{seed}" / "COMPLETED.json").exists():
            event(f"RESUME COMPLETED SEED {seed}")
            continue
        values = roles_for_seed(full, seed, encoder, primary)
        event(f"ROLES VERIFIED seed={seed} counts={values[-1]['base_fit_rows_after_cap']}/{len(values[1])}/{len(values[2])}/{len(values[3])}")
        if not args.preflight_only:
            seed_start = time.perf_counter()
            with threadpool_limits(limits=2):
                fit_seed(seed, *values[:6], encoder, features, values[-1])
            json_write(OUT / f"seed_{seed}" / "runtime.json", {"wall_seconds_this_invocation": time.perf_counter() - seed_start})
            aggregate()
        del values
        gc.collect()
    if not args.preflight_only:
        aggregate()
    json_write(OUT / ("preflight_runtime.json" if args.preflight_only else "invocation_runtime.json"),
               {"elapsed_seconds": time.perf_counter() - start, "seeds_requested": args.seeds,
                "preflight_only": args.preflight_only, "python": sys.executable,
                "command": sys.argv, "finished_utc": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    main()
