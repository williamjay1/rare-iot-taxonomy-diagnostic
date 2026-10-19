"""Checkpointed, group-disjoint exploratory controls; no outer-role tuning."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

sys.dont_write_bytecode = True
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from threadpoolctl import threadpool_limits

ROOT = Path(r"legacy_project").resolve()
OUTPUT = Path(r"workspace\extensions")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "working" / "ijismd_v11_mechanical_completion_20260923" / "scripts"))
import run_ciciot_agreement_gated_trf_confirmation as nested
import run_ciciot_calibrated_benefit_gate_confirmation as benefit
import run_ciciot_taxonomy_residual_fusion as trf
import run_v10_grouped_flat_comparators as flat
import run_v10_grouped_newton_flat_stacker as newton

CONFIG_PATH = Path(__file__).with_name("extension_config.json")
METHODS = (
    "flat_xgb", "flat_lgbm", "flat_equal_blend", "temperature_flat_blend",
    "newton_flat_stacker", "real_taxonomy", "taxonomy_blend",
    "cyclic_taxonomy", "cyclic_blend",
)
STRONG = ("temperature_flat_blend", "newton_flat_stacker")
GROUP = benefit.FEATURE_GROUP_COLUMN
_global_factory = trf.make_global_model
_local_factory = trf.make_lgbm


def limited_global(n_classes, trees):
    return _global_factory(n_classes, trees).set_params(n_jobs=2)


def limited_local(n_classes, trees):
    return _local_factory(n_classes, trees).set_params(n_jobs=2)


trf.make_global_model = limited_global
trf.make_lgbm = limited_local


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path, frame):
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def save_npz(path, **arrays):
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def checkpoint(path, builder):
    if path.exists():
        return joblib.load(path)
    print(f"  fitting {path.stem}", flush=True)
    value = builder()
    temporary = path.with_name(path.name + ".tmp")
    joblib.dump(value, temporary, compress=3)
    temporary.replace(path)
    return value


def array_hash(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_physical_d(path):
    resolved = path.resolve()
    if resolved.drive.upper() != "D:":
        raise ValueError(f"Compute storage must resolve to physical D:, not {resolved}. Supply an authorized --out-dir on D:.")
    return resolved


def working_mirrors(dataset_dir=None):
    dataset_dir = require_physical_d(dataset_dir) if dataset_dir is not None else OUTPUT / "datasets"
    dataset_dir.mkdir(exist_ok=True)
    paths = []
    for role, source in (("train", nested.TRAIN_PATH), ("test", nested.TEST_PATH)):
        destination = dataset_dir / f"{role}_working_copy.parquet"
        manifest = dataset_dir / f"{role}_copy_manifest.json"
        if manifest.exists():
            metadata = json.loads(manifest.read_text())
            if (not destination.exists() or file_hash(destination) != metadata["sha256"]
                    or source.stat().st_size != metadata["source_size"]
                    or source.stat().st_mtime_ns != metadata["source_mtime_ns"]):
                raise ValueError("Working copy or source changed; refusing stale mixed inputs")
        else:
            print(f"Caching {role} mirror on physical D: {destination}", flush=True)
            temporary = destination.with_suffix(".tmp.parquet")
            shutil.copyfile(source, temporary)
            if file_hash(source) != file_hash(temporary):
                raise ValueError("Working-copy checksum mismatch")
            temporary.replace(destination)
            write_json(manifest, {"source": str(source), "source_size": source.stat().st_size,
                                  "source_mtime_ns": source.stat().st_mtime_ns,
                                  "sha256": file_hash(destination)})
        paths.append(destination)
    return paths


def role_audit(roles):
    audit = {}
    names = list(roles)
    groups = {name: set(frame[GROUP]) for name, frame in roles.items()}
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = len(groups[left] & groups[right])
            audit[f"{left}_{right}_group_overlap"] = overlap
            if overlap:
                raise ValueError(f"Predictor groups cross {left}/{right}")
    return audit


def cap_rows(frame, cap, seed, benign_cap=None):
    pieces = []
    for label, group in frame.groupby("Label", sort=False):
        limit = benign_cap if benign_cap is not None and str(label) == "BenignTraffic" else cap
        pieces.append(group.sample(n=min(limit, len(group)), random_state=seed))
    return pd.concat(pieces).sample(frac=1, random_state=seed)


def classification_metrics(y, probability, rare):
    prediction = probability.argmax(axis=1)
    labels = np.arange(probability.shape[1])
    recalls = recall_score(y, prediction, labels=labels, average=None, zero_division=0)
    return {
        "macro_f1": float(f1_score(y, prediction, labels=labels, average="macro", zero_division=0)),
        "rare_recall": float(recalls[rare].mean()) if len(rare) else None,
        "test_rows": len(y),
        "test_rare_rows": int(np.isin(y, rare).sum()),
        "rare_labels_without_test_support": int(sum(not np.any(y == value) for value in rare)),
    }


def novelty_scores(probability):
    return {
        "msp": 1 - probability.max(axis=1),
        "entropy": -(probability * np.log(np.clip(probability, 1e-12, 1))).sum(axis=1),
    }


def novelty_metrics_from_scores(selection, known, unknown):
    result = {}
    y = np.r_[np.zeros(len(known["msp"])), np.ones(len(unknown["msp"]))]
    for name in ("msp", "entropy"):
        selection_score, known_score, unknown_score = (scores[name] for scores in (selection, known, unknown))
        result[f"{name}_unknown_auroc"] = float(roc_auc_score(y, np.r_[known_score, unknown_score]))
        for fpr in (0.01, 0.05, 0.10):
            # Strict exceedance makes ties conservative. No threshold sees outer labels.
            threshold = float(np.quantile(selection_score, 1 - fpr, method="higher"))
            suffix = f"{name}_selection_fpr_{fpr:g}"
            result[f"{suffix}_threshold"] = threshold
            result[f"{suffix}_selection_realized_fpr"] = float(np.mean(selection_score > threshold))
            result[f"{suffix}_outer_known_fpr"] = float(np.mean(known_score > threshold))
            result[f"{suffix}_unknown_recall"] = float(np.mean(unknown_score > threshold))
    return result


def novelty_metrics(selection, known, unknown):
    return novelty_metrics_from_scores(*(novelty_scores(p) for p in (selection, known, unknown)))


def fit_meta(xgb_calibration, lgbm_calibration, y_calibration, seed, config):
    start = time.perf_counter()
    xgb_temperature, xgb_loss = flat.fit_temperature(y_calibration, xgb_calibration)
    lgbm_temperature, lgbm_loss = flat.fit_temperature(y_calibration, lgbm_calibration)
    stacker = newton.fit_converged_stacker(
        flat.probability_features(xgb_calibration, lgbm_calibration), y_calibration,
        seed, config["stacker_C"], config["stacker_max_iter"],
    )
    return {
        "xgb_temperature": xgb_temperature, "lgbm_temperature": lgbm_temperature,
        "xgb_calibration_loss": xgb_loss, "lgbm_calibration_loss": lgbm_loss,
        "stacker": stacker, "seconds": time.perf_counter() - start,
        "stacker_iterations": int(stacker.n_iter_.max()),
    }


def probabilities(frame, features, encoder, family_encoder, mapping, models, meta):
    X = trf.clean_features(frame, features)
    xgb, lgbm, real, cyclic = models
    p_xgb = flat.normalise(xgb.predict_proba(X))
    p_lgbm = flat.normalise(lgbm.predict_proba(X))
    p_real, _ = trf.hierarchy_probabilities(X, real[0], family_encoder, encoder, mapping, real[1])
    p_cyclic, _ = trf.hierarchy_probabilities(X, cyclic[0], cyclic[1], encoder, cyclic[4], cyclic[2])
    p_real, p_cyclic = flat.normalise(p_real), flat.normalise(p_cyclic)
    return dict(zip(METHODS, (
        p_xgb, p_lgbm, flat.normalise(p_xgb + p_lgbm),
        flat.normalise(flat.temperature_probability(p_xgb, meta["xgb_temperature"]) +
                       flat.temperature_probability(p_lgbm, meta["lgbm_temperature"])),
        flat.normalise(meta["stacker"].predict_proba(flat.probability_features(p_xgb, p_lgbm))),
        p_real, flat.normalise(p_xgb + p_real), p_cyclic, flat.normalise(p_xgb + p_cyclic),
    )))


def run_cell(endpoint, case, seed, roles, features, config, config_hash):
    cell_id = f"{endpoint}_seed{seed}_{str(case).replace('-', '_')}"
    folder = OUTPUT / "cells" / cell_id
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "completed.json").exists():
        complete = json.loads((folder / "completed.json").read_text())
        if complete["config_sha256"] != config_hash:
            raise ValueError(f"Configuration changed for {cell_id}")
        return False
    start = time.perf_counter()
    fit_pool, calibration, selection, outer = (roles[name] for name in ("fit_pool", "calibration", "selection", "outer"))
    if endpoint == "lofo":
        fit_pool, calibration, selection = (
            frame.loc[frame["attack_class"].astype(str) != case]
            for frame in (fit_pool, calibration, selection)
        )
        fit = cap_rows(fit_pool, config["lofo_majority_cap"], seed, config["lofo_benign_cap"])
        if len(fit) > config["lofo_max_train_rows"]:
            fit = fit.sample(n=config["lofo_max_train_rows"], random_state=seed)
        known = outer.loc[outer["attack_class"].astype(str) != case]
        unknown = outer.loc[outer["attack_class"].astype(str) == case]
        known = known.sample(n=min(config["lofo_known_rows"], len(known)), random_state=seed)
        unknown = unknown.sample(n=min(config["lofo_unknown_rows"], len(unknown)), random_state=seed)
    else:
        fit = cap_rows(fit_pool, int(case), seed)
        known = outer.sample(n=min(config["scarcity_outer_rows"], len(outer)), random_state=seed)
        unknown = None
    encoder = LabelEncoder().fit(fit_pool["Label"].astype(str))
    family_encoder = LabelEncoder().fit(fit_pool["attack_class"].astype(str))
    if set(fit["Label"]) != set(encoder.classes_):
        raise ValueError("Fit cap omitted one or more known classes")
    mapping = fit_pool[["Label", "attack_class"]].drop_duplicates().set_index("Label")["attack_class"].astype(str).to_dict()
    y_pool = encoder.transform(fit_pool["Label"].astype(str))
    rare = trf.rare_class_ids(y_pool)
    y_calibration = encoder.transform(calibration["Label"].astype(str))
    if set(np.unique(y_calibration)) != set(range(len(encoder.classes_))):
        raise ValueError("Calibration lacks known classes")
    audit = role_audit({"fit_pool": fit_pool, "calibration": calibration, "selection": selection, "outer": outer})
    audit.update({
        "cell_id": cell_id, "endpoint": endpoint, "case": case, "seed": seed,
        "config_sha256": config_hash, "fit_rows": len(fit), "fit_pool_rows": len(fit_pool),
        "calibration_rows": len(calibration), "selection_rows": len(selection),
        "known_outer_rows": len(known), "unknown_outer_rows": 0 if unknown is None else len(unknown),
        "n_jobs": 2, "fine_labels": encoder.classes_.tolist(), "rare_ids": rare.tolist(),
        "rare_definition": "uncapped fit-pool class count below 1% of uncapped fit-pool rows",
        "calibrator_fit_role": "calibration", "novelty_threshold_fit_role": "selection",
        "selection_used_for_model_or_hyperparameter_choice": False,
        "held_out_family_in_fit_or_calibration_or_selection": False if endpoint == "lofo" else None,
        "calibration_label_budget_is_additional_to_fit_cap": True,
    })
    for name, frame in {"fit": fit, "calibration": calibration, "selection": selection, "known_outer": known}.items():
        audit[f"{name}_row_id_sha256"] = array_hash(frame.index.to_numpy(dtype=np.int64))
    existing_audit = folder / "protocol.json"
    if existing_audit.exists() and json.loads(existing_audit.read_text())["config_sha256"] != config_hash:
        raise ValueError(f"Refusing mixed checkpoint configurations in {cell_id}")
    write_json(existing_audit, audit)
    print(f"START {cell_id}: fit={len(fit)} calibration={len(calibration)} known={len(known)}", flush=True)
    trf.RANDOM_STATE = seed
    xgb = checkpoint(folder / "xgb.joblib", lambda: nested.fit_weighted_xgb(fit, features, encoder, config["global_trees"])[0])
    lgbm = checkpoint(folder / "lgbm.joblib", lambda: nested.fit_flat_lgbm(fit, features, encoder, config["local_trees"])[0])
    real = checkpoint(folder / "real_hierarchy.joblib", lambda: nested.fit_real_taxonomy(fit, features, family_encoder, config["local_trees"])[:2])
    def build_cyclic():
        wrong_map, _ = trf.cyclic_control_map(mapping)
        fitted = trf.fit_control_hierarchy(fit, features, wrong_map, config["local_trees"])
        return (*fitted, wrong_map)
    cyclic = checkpoint(folder / "cyclic_hierarchy.joblib", build_cyclic)
    calibration_path = folder / "calibration_inputs.npz"
    if calibration_path.exists():
        with np.load(calibration_path) as source:
            if not np.array_equal(source["row_id"], calibration.index) or not np.array_equal(source["y_true"], y_calibration):
                raise ValueError("Calibration checkpoint row mismatch")
            p_xgb, p_lgbm = source["xgb"].astype(float), source["lgbm"].astype(float)
    else:
        X = trf.clean_features(calibration, features)
        p_xgb, p_lgbm = flat.normalise(xgb.predict_proba(X)), flat.normalise(lgbm.predict_proba(X))
        save_npz(calibration_path, row_id=calibration.index.to_numpy(), y_true=y_calibration,
                 xgb=p_xgb, lgbm=p_lgbm)
        del X
    meta = checkpoint(folder / "calibrators.joblib", lambda: fit_meta(p_xgb, p_lgbm, y_calibration, seed, config))
    del p_xgb, p_lgbm
    models = (xgb, lgbm, real, cyclic)
    print(f"  calibrated: Newton iterations={meta['stacker_iterations']}; inferring frozen outer", flush=True)
    known_probability = probabilities(known, features, encoder, family_encoder, mapping, models, meta)
    selection_scores = None
    if unknown is not None:
        selection_probability = probabilities(selection, features, encoder, family_encoder, mapping, models, meta)
        # Only these exact row scores enter thresholds; release the larger matrices before outer inference.
        selection_scores = {method: novelty_scores(selection_probability.pop(method)) for method in METHODS}
        del selection_probability
    unknown_probability = probabilities(unknown, features, encoder, family_encoder, mapping, models, meta) if unknown is not None else None
    y_known = encoder.transform(known["Label"].astype(str))
    arrays = {"row_id": known.index.to_numpy(), "group_id": known[GROUP].to_numpy(), "y_true": y_known,
              "rare_ids": rare, "fine_labels": np.asarray(encoder.classes_, dtype=str)}
    rows = []
    for method in METHODS:
        p = known_probability[method]
        row = {"cell_id": cell_id, "endpoint": endpoint, "case": case, "seed": seed, "method": method,
               **classification_metrics(y_known, p, rare)}
        arrays[f"{method}_probability"] = p.astype(np.float32)
        if unknown is not None:
            row.update(novelty_metrics_from_scores(selection_scores[method], novelty_scores(p),
                                                   novelty_scores(unknown_probability[method])))
            arrays[f"{method}_unknown_probability"] = unknown_probability[method].astype(np.float32)
        rows.append(row)
    if unknown is not None:
        arrays["unknown_row_id"] = unknown.index.to_numpy()
        arrays["unknown_group_id"] = unknown[GROUP].to_numpy()
    save_npz(folder / "paired_predictions.npz", **arrays)
    write_csv(folder / "results.csv", pd.DataFrame(rows))
    audit.update({"stacker_iterations": meta["stacker_iterations"], "xgb_temperature": meta["xgb_temperature"],
                  "lgbm_temperature": meta["lgbm_temperature"], "elapsed_seconds_this_invocation": time.perf_counter() - start})
    write_json(folder / "completed.json", audit)
    print(f"DONE {cell_id}: {audit['elapsed_seconds_this_invocation']:.1f}s", flush=True)
    return True


def ece(y, probability, *, nbins=10):
    if nbins != 10:
        raise ValueError("The frozen extension metric protocol requires 10 equal-width ECE bins")
    confidence = probability.max(axis=1)
    correct = probability.argmax(axis=1) == y
    # E3 uses 10 bins and row-share weights, not the old len(boolean_mask) bug.
    bins = np.clip(np.digitize(confidence, np.linspace(0, 1, nbins + 1), right=True) - 1, 0, nbins - 1)
    return float(sum(np.mean(bins == b) * abs(correct[bins == b].mean() - confidence[bins == b].mean())
                     for b in range(nbins) if np.any(bins == b)))


def calibration_metrics(y, probability, rare):
    mask = np.isin(y, rare)
    losses = (probability ** 2).sum(axis=1) - 2 * probability[np.arange(len(y)), y] + 1
    return {"ece_all": ece(y, probability, nbins=10), "brier_all": float(losses.mean()),
            "ece_rare_rows": ece(y[mask], probability[mask], nbins=10) if mask.any() else None,
            "brier_rare_rows": float(losses[mask].mean()) if mask.any() else None}


def aligned_probability(source, anchor, key):
    if not np.array_equal(source["row_id"], anchor["row_id"]):
        raise ValueError("Frozen row order differs")
    if not np.array_equal(source["y_true"], anchor["y_true"]) or not np.array_equal(source["rare_ids"], anchor["rare_ids"]):
        raise ValueError("Frozen labels / rare definitions differ")
    return flat.normalise(source[key].astype(float))


def frozen_endpoints():
    for seed in range(42, 48):
        folder = OUTPUT / "frozen" / f"seed{seed}"
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "completed.json").exists():
            continue
        base = ROOT / "results"
        paths = [base / directory / f"paired_predictions_seed_{seed}.npz" for directory in (
            "ijismd_v10_ciciot_grouped_core_6split", "ijismd_v10_grouped_flat_comparators_6split",
            "ijismd_v10_grouped_newton_flat_stacker_6split")]
        with np.load(paths[0]) as core, np.load(paths[1]) as blend, np.load(paths[2]) as stacker:
            sources = {
                "flat_xgb": (core, "flat_weighted_xgboost_probability"),
                "real_taxonomy": (core, "soft_taxonomy_factorisation_probability"),
                "taxonomy_blend": (core, "equal_flat_taxonomy_blend_probability"),
                "taxonomy_benefit_gate": (core, "calibrated_taxonomy_benefit_gate_probability"),
                "cyclic_gate": (core, "calibrated_cyclic_taxonomy_control_probability"),
                "support_matched_gate": (core, "calibrated_support_matched_control_probability"),
                "flat_lgbm": (blend, "flat_weighted_lightgbm_probability"),
                "flat_equal_blend": (blend, "flat_equal_probability_blend_probability"),
                "temperature_flat_blend": (blend, "flat_temperature_calibrated_blend_probability"),
                "newton_flat_stacker": (stacker, "flat_newton_log_probability_stacker_probability"),
            }
            y, rare = core["y_true"].astype(int), core["rare_ids"].astype(int)
            rare_mask = np.isin(y, rare)
            cal_rows, alert_rows = [], []
            for method, (source, key) in sources.items():
                p = aligned_probability(source, core, key)
                cal_rows.append({"seed": seed, "method": method, **calibration_metrics(y, p, rare)})
                order = np.argsort(-p[:, rare].max(axis=1), kind="stable")
                specs = [(f"absolute_{k}", min(k, len(y))) for k in (1000, 2500, 5000, 10000, 20000, 30000, 50000, 80000, 100000)]
                specs += [(f"fraction_{fraction:g}", max(1, int(np.ceil(len(y) * fraction)))) for fraction in (0.005, 0.01)]
                for budget_id, budget in specs:
                    hits = int(rare_mask[order[:budget]].sum())
                    alert_rows.append({"seed": seed, "method": method, "budget_id": budget_id,
                                       "budget": budget, "rare_hits": hits, "rare_rows": int(rare_mask.sum()),
                                       "rare_recall": hits / int(rare_mask.sum()), "rare_precision": hits / budget,
                                       "test_rows": len(y), "score": "maximum rare-class probability",
                                       "ties": "stable original frozen row order"})
        write_csv(folder / "calibration.csv", pd.DataFrame(cal_rows))
        write_csv(folder / "alert_budget.csv", pd.DataFrame(alert_rows))
        write_json(folder / "completed.json", {"seed": seed, "status": "completed", "fitting": "none; frozen probabilities only",
                                               "historical_temperature_and_stacker_fit_role": "utility/calibration; selection was reported only",
                                               "probability_treatment": "all frozen float16 matrices renormalized consistently to unit row sums",
                                               "source_sha256": {str(path): file_hash(path) for path in paths}})
        print(f"DONE frozen E3/E4 seed{seed}", flush=True)


def summarize(config):
    frames = [pd.read_csv(path) for path in sorted((OUTPUT / "cells").glob("*/results.csv"))
              if path.with_name("completed.json").exists()]
    all_results = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not all_results.empty:
        write_csv(OUTPUT / "matched_extension_results.csv", all_results)
        for endpoint in ("lofo", "scarcity"):
            selected = all_results.loc[all_results.endpoint == endpoint].dropna(axis=1, how="all")
            if not selected.empty:
                write_csv(OUTPUT / f"{endpoint}_results.csv", selected)
                metrics = [column for column in selected.select_dtypes("number").columns
                           if column not in ("seed", "case")]
                summary = selected.groupby(["case", "method"])[metrics].agg(["mean", "std", "count"])
                summary.columns = ["_".join(column) for column in summary.columns]
                write_csv(OUTPUT / f"{endpoint}_summary.csv", summary.reset_index())
                per_seed = selected.groupby(["seed", "method"])[metrics].mean()
                per_seed["nested_cases_completed"] = selected.groupby(["seed", "method"]).size()
                per_seed["nested_cases_expected"] = 7 if endpoint == "lofo" else 6
                per_seed["aggregation"] = "equal-weight mean across nested families/caps within seed; not independent replications"
                write_csv(OUTPUT / f"{endpoint}_per_seed_summary.csv", per_seed.reset_index())
        paired = all_results.pivot(index=["cell_id", "endpoint", "case", "seed"], columns="method", values=["macro_f1", "rare_recall"])
        delta_rows = []
        for index, row in paired.iterrows():
            for control in STRONG:
                delta_rows.append(dict(zip(("cell_id", "endpoint", "case", "seed"), index)) | {
                    "control": control,
                    **{f"taxonomy_blend_minus_control_{metric}": row[(metric, "taxonomy_blend")] - row[(metric, control)]
                       for metric in ("macro_f1", "rare_recall")}})
        write_csv(OUTPUT / "paired_deltas.csv", pd.DataFrame(delta_rows))
        deltas = pd.DataFrame(delta_rows)
        delta_metrics = [f"taxonomy_blend_minus_control_{metric}" for metric in ("macro_f1", "rare_recall")]
        per_seed = deltas.groupby(["endpoint", "seed", "control"])[delta_metrics].mean().reset_index()
        per_seed["nested_cases_completed"] = deltas.groupby(["endpoint", "seed", "control"]).size().to_numpy()
        per_seed["nested_cases_expected"] = per_seed.endpoint.map({"lofo": 7, "scarcity": 6})
        per_seed["seed_summary_complete"] = per_seed.nested_cases_completed == per_seed.nested_cases_expected
        per_seed["inference_scope"] = "within-seed descriptive mean; three seed-level summaries, no independent-fold sign p value"
        write_csv(OUTPUT / "paired_deltas_per_seed.csv", per_seed)
        per_case = deltas.groupby(["endpoint", "case", "control"])[delta_metrics].agg(["mean", "std", "count"])
        per_case.columns = ["_".join(column) for column in per_case.columns]
        write_csv(OUTPUT / "paired_deltas_per_family_or_cap.csv", per_case.reset_index())
        direction_rows = []
        for (endpoint, control), values in per_seed.groupby(["endpoint", "control"]):
            values = values.loc[values.seed_summary_complete]
            for metric in delta_metrics:
                direction_rows.append({"endpoint": endpoint, "control": control, "metric": metric,
                                       "seed_means_positive": int((values[metric] > 0).sum()),
                                       "seed_means_zero": int((values[metric] == 0).sum()),
                                       "seed_means_negative": int((values[metric] < 0).sum()),
                                       "seeds_available": len(values), "seeds_expected": 3,
                                       "mean_of_available_seed_means": float(values[metric].mean()) if len(values) else None,
                                       "seed_level_sample_sd": float(values[metric].std()) if len(values) > 1 else None,
                                       "scope": "descriptive direction only; nested cases are dependent and no population-significance claim is made"})
        write_csv(OUTPUT / "paired_seed_direction_consistency.csv", pd.DataFrame(direction_rows))
    counts = {}
    counts["capture_windows"] = sum(
        path.with_name("results.csv").exists()
        for path in (OUTPUT / "capture").glob("w*/completed.json")
    )
    for endpoint in ("lofo", "scarcity"):
        counts[endpoint] = 0 if all_results.empty else int(all_results.loc[all_results.endpoint == endpoint, "cell_id"].nunique())
    for name in ("calibration", "alert_budget"):
        frozen_frames = [pd.read_csv(path) for path in sorted((OUTPUT / "frozen").glob(f"seed*/{name}.csv"))
                         if path.with_name("completed.json").exists()]
        counts[name] = len(frozen_frames)
        if frozen_frames:
            combined = pd.concat(frozen_frames, ignore_index=True)
            write_csv(OUTPUT / f"{name}_results.csv", combined)
            metrics = ["ece_all", "brier_all", "ece_rare_rows", "brier_rare_rows"] if name == "calibration" else ["rare_recall", "rare_precision", "rare_hits"]
            groups = ["method"] if name == "calibration" else ["method", "budget_id"]
            summary = combined.groupby(groups)[metrics].agg(["mean", "std", "count"])
            summary.columns = ["_".join(column) for column in summary.columns]
            write_csv(OUTPUT / f"{name}_summary.csv", summary.reset_index())
    expected = {"lofo": 21, "scarcity": 18, "calibration": 6, "alert_budget": 6, "capture_windows": 4}
    rows = []
    for endpoint, total in expected.items():
        completed = counts.get(endpoint, 0)
        reason = "" if completed == total else (
            "Capture windows require the separate 39-feature archive runner, run_capture_controls.py. See capture/run_status.json and per-window checkpoints; no missing window is imputed."
            if endpoint == "capture_windows" else "Sequential checkpointed queue is incomplete; see run_status.json and per-cell checkpoints. No unrun result is imputed."
        )
        rows.append({"endpoint": endpoint, "expected_units": total, "completed_units": completed,
                     "strong_control_evaluations_completed": 2 * completed,
                     "status": "completed" if completed == total else "incomplete", "incomplete_reason": reason,
                     "frozen_recompute": endpoint in ("calibration", "alert_budget"),
                     "result_csv": str(OUTPUT / f"{endpoint}_results.csv") if completed else ""})
    write_csv(OUTPUT / "endpoint_coverage.csv", pd.DataFrame(rows))
    return rows


def main():
    global OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--phases", nargs="+", choices=("lofo", "scarcity", "frozen"), default=["lofo", "scarcity", "frozen"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--max-new-cells", type=int, default=0)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--worker-only", action="store_true",
                        help="One leased seed, per-cell results only; final aggregate publication belongs to the merger")
    parser.add_argument("--lease-dir", type=Path)
    parser.add_argument("--dataset-cache-dir", type=Path,
                        help="Read the already verified physical-D working mirrors without making new training copies")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT,
                        help="Authorized physical D: directory for results, datasets, models and temporary files")
    args = parser.parse_args()
    if args.worker_only and (len(args.seeds) != 1 or args.lease_dir is None or "frozen" in args.phases):
        parser.error("--worker-only requires exactly one seed, --lease-dir, and refit phases only")
    config = json.loads(CONFIG_PATH.read_text())
    config_hash = file_hash(CONFIG_PATH)
    if not args.worker_only and list((OUTPUT / "dispatch" / "leases").glob("*.lock")):
        parser.error("Isolated seed workers own the active queue; shared training/summary commands are locked until final merge")
    OUTPUT = require_physical_d(args.out_dir)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "temp").mkdir(exist_ok=True)
    for name in ("TEMP", "TMP", "TMPDIR", "JOBLIB_TEMP_FOLDER"):
        os.environ[name] = str(OUTPUT / "temp")
    if args.summary_only:
        print(pd.DataFrame(summarize(config)).to_string(index=False))
        return
    lease = None
    if args.worker_only:
        lease_dir = require_physical_d(args.lease_dir)
        lease_dir.mkdir(parents=True, exist_ok=True)
        lease = lease_dir / f"seed{args.seeds[0]}.lock"
        with lease.open("x", encoding="utf-8") as stream:
            json.dump({"seed": args.seeds[0], "pid": os.getpid(), "output": str(OUTPUT),
                       "config_sha256": config_hash, "n_jobs": 2}, stream, indent=2)
    write_json(OUTPUT / "run_status.json", {"status": "running", "command": sys.executable + " -B " + " ".join(sys.argv),
                                           "config_sha256": config_hash, "n_jobs": 2, "pid": os.getpid(),
                                           "worker_only": args.worker_only, "seeds": args.seeds})
    if not args.worker_only:
        summarize(config)
    try:
        done = 0
        if any(phase in args.phases for phase in ("lofo", "scarcity")):
            train_path, test_path = working_mirrors(args.dataset_cache_dir)
            full = pd.concat([pd.read_parquet(train_path), pd.read_parquet(test_path)], ignore_index=True)
            features = nested.exp.feature_cols(full)
            benefit.assign_exact_predictor_groups(full, features)
            write_json(OUTPUT / "run_metadata.json", {
                "route": "SCI prediction and matched validation; no causal claim", "config": config,
                "source_files": [str(nested.TRAIN_PATH), str(nested.TEST_PATH)], "features": features,
                "source_sha256": {str(path): file_hash(path) for path in (nested.TRAIN_PATH, nested.TEST_PATH)},
                "package_versions": benefit.package_versions(), "thread_limit": 2,
                "legacy_endpoint_values_are_not_reused": True,
                "scarcity_scope": "Base-fitting cap only; full calibration role provides additional labels, reported explicitly. Not total-label scarcity.",
                "role_correction": "Primary grouped 20% outer; development 70/15/15. Calibration fits temperature and stacker; selection supplies novelty thresholds. Outer never tunes anything.",
                "legacy_discrepancies": ["Legacy LOFO/scarcity scripts used original random mirrors, not grouped roles.",
                                         "Legacy LOFO known-FPR thresholds used outer known scores; new thresholds use selection.",
                                         "Legacy scarcity code recomputed rare IDs after capping; new rare IDs use uncapped fit pool.",
                                         "v14 prose says selection-only stacker; actual primary comparator code fits calibration only."],
            })
            for endpoint in (phase for phase in args.phases if phase != "frozen"):
                for seed in args.seeds:
                    development, outer = benefit.stratified_group_split(full, seed=seed, test_size=0.20)
                    fit, calibration, selection = benefit.split_development(development, seed, .70, .15, .15, group_disjoint=True)
                    roles = dict(fit_pool=fit, calibration=calibration, selection=selection, outer=outer)
                    role_audit(roles)
                    cases = config["lofo_families"] if endpoint == "lofo" else config["scarcity_caps"]
                    for case in cases:
                        changed = run_cell(endpoint, case, seed, roles, features, config, config_hash)
                        done += int(changed)
                        if not args.worker_only:
                            summarize(config)
                        gc.collect()
                        if args.max_new_cells and done >= args.max_new_cells:
                            write_json(OUTPUT / "run_status.json", {"status": "checkpoint_limit_reached", "new_cells": done,
                                                                  "next_action": "Rerun identical command without --max-new-cells; completed cells and model checkpoints are reused."})
                            return
        if "frozen" in args.phases:
            frozen_endpoints()
        coverage = [] if args.worker_only else summarize(config)
        expected_cells = [f"{endpoint}_seed{seed}_{str(case).replace('-', '_')}"
                          for endpoint in args.phases if endpoint != "frozen" for seed in args.seeds
                          for case in (config["lofo_families"] if endpoint == "lofo" else config["scarcity_caps"])]
        missing = [name for name in expected_cells
                   if not (OUTPUT / "cells" / name / "completed.json").exists()
                   or not (OUTPUT / "cells" / name / "results.csv").exists()]
        write_json(OUTPUT / "run_status.json", {"status": "requested_queue_finished", "new_cells": done,
                                               "requested_cells": len(expected_cells), "missing_cells": missing,
                                               "worker_only": args.worker_only, "seeds": args.seeds,
                                               "completed_endpoint_count": sum(row["status"] == "completed" for row in coverage),
                                               "capture_windows_remain_incomplete": any(row["endpoint"] == "capture_windows" and row["status"] != "completed" for row in coverage)})
    except BaseException as error:
        write_json(OUTPUT / "run_status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                               "error": repr(error), "traceback": traceback.format_exc(), "new_cells": done})
        if not args.worker_only:
            summarize(config)
        raise
    finally:
        if lease is not None:
            lease.unlink()


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
