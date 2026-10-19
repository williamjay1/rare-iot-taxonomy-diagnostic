from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import os
import shutil
import stat
import sys
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = Path(r"legacy_project")
CACHE = ROOT / "cache"
OUT = ROOT / "results" / "ijismd_v15_standards_20261018"
FROZEN = ROOT / "frozen"
CORE = PROJECT / "results" / "ijismd_v10_ciciot_grouped_core_6split"
SEEDS = [42, 43, 44, 45, 46, 47]
PARAMETERS = {
    "seeds": SEEDS, "outer_test_frac": 0.2, "base_fraction": 0.70,
    "gate_train_fraction": 0.15, "gate_select_fraction": 0.15,
    "majority_cap": 25000, "benign_cap": 80000, "local_trees": 380,
    "blend_weight": 0.5, "n_jobs": 2, "wrong_solver_seconds": 120,
}
for directory in (CACHE, OUT, FROZEN, ROOT / "temp"):
    directory.mkdir(parents=True, exist_ok=True)
for variable in ("TEMP", "TMP", "TMPDIR", "JOBLIB_TEMP_FOLDER"):
    os.environ[variable] = str(ROOT / "temp")
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
sys.dont_write_bytecode = True
sys.path.insert(0, str(PROJECT / "scripts"))

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix
from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score, recall_score
from sklearn.preprocessing import LabelEncoder
from threadpoolctl import threadpool_limits

import run_ciciot_agreement_gated_trf_confirmation as nested
import run_ciciot_calibrated_benefit_gate_confirmation as benefit
import run_ciciot_taxonomy_residual_fusion as trf


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dump(path: Path, value: object, *, exclusive: bool = False) -> None:
    with path.open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def physical_d(path: Path) -> None:
    if path.drive.upper() != "D:":
        raise RuntimeError(f"Output is not on D: {path}")
    for parent in (path, *path.parents):
        if parent.exists() and getattr(parent.lstat(), "st_file_attributes", 0) & 0x400:
            raise RuntimeError(f"Output traverses a reparse point: {parent}")


def cached_copy(source: Path, destination: Path) -> dict:
    physical_d(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_hash = digest(source)
    if not destination.exists():
        shutil.copyfile(source, destination)
    if digest(destination) != source_hash:
        raise RuntimeError(f"Cache differs from source: {destination}")
    return {"source": str(source), "cache": str(destination), "bytes": destination.stat().st_size, "sha256": source_hash}


def standard_spec() -> tuple[pd.DataFrame, dict[str, str]]:
    spec = json.loads((ROOT / "mechanism_spec.json").read_text(encoding="utf-8"))
    definitions = pd.read_csv(ROOT / "capec_definitions.csv", dtype={"capec_id": str}).set_index("capec_id")
    rows = []
    ambiguous = {"partial_proxy", "broad_mechanism", "broad_proxy", "conditional_proxy"}
    for rule in spec["rules"]:
        ids = [str(x) for x in rule["capec_ids"]]
        for label in rule["labels"]:
            rows.append({
                "fine_label": label, "mechanism_group": rule["group"],
                "capec_ids": ";".join("CAPEC-" + x for x in ids),
                "capec_names": ";".join(str(definitions.loc[x, "name"]) for x in ids),
                "capec_urls": ";".join(str(definitions.loc[x, "url"]) for x in ids),
                "attack_ids": ";".join(rule.get("attack_ids", [])),
                "source_version": "CAPEC 3.9", "match": rule["match"],
                "ambiguity": rule["match"] in ambiguous,
                "basis": rule["basis"], "limitation": rule["limitation"],
                "expert_validated": False,
            })
    frame = pd.DataFrame(rows).sort_values("fine_label").reset_index(drop=True)
    if frame.fine_label.duplicated().any():
        raise ValueError("Repeated label in mechanism specification")
    return frame, dict(zip(frame.fine_label, frame.mechanism_group))


def wrong_map_exact(mapping: dict[str, str], counts: dict[str, int], seed: int) -> tuple[dict[str, str], dict]:
    labels = sorted(mapping)
    groups = sorted(set(mapping.values()))
    n, k = len(labels), len(groups)
    sizes = Counter(mapping.values())
    target = {g: sum(counts[l] for l in labels if mapping[l] == g) for g in groups}
    pairs = [(a, b) for a, b in itertools.combinations(range(n), 2)
             if mapping[labels[a]] == mapping[labels[b]] and mapping[labels[a]] != "Benign"]
    nx, nz = n * k, len(pairs) * k
    constraints, lower, upper = [], [], []

    def add(terms: dict[int, float], lo: float, hi: float) -> None:
        constraints.append(terms); lower.append(lo); upper.append(hi)

    for i in range(n):
        add({i * k + j: 1 for j in range(k)}, 1, 1)
    for j, group in enumerate(groups):
        add({i * k + j: 1 for i in range(n)}, sizes[group], sizes[group])
        add({i * k + j: counts[label] for i, label in enumerate(labels)}, target[group], target[group])
    for p, (a, b) in enumerate(pairs):
        for j in range(k):
            add({a * k + j: 1, b * k + j: 1, nx + p * k + j: -1}, -np.inf, 1)
    matrix = lil_matrix((len(constraints), nx + nz), dtype=float)
    for i, terms in enumerate(constraints):
        for j, v in terms.items():
            matrix[i, j] = v
    lo = np.zeros(nx + nz)
    hi = np.ones(nx + nz)
    # Benign stays fixed. Attack labels cannot occupy the benign singleton.
    benign_j = groups.index("Benign")
    for i, label in enumerate(labels):
        if label == "BenignTraffic":
            hi[i * k:(i + 1) * k] = 0
            lo[i * k + benign_j] = hi[i * k + benign_j] = 1
        else:
            hi[i * k + benign_j] = 0
    objective = np.zeros(nx + nz)
    objective[nx:] = n + 1
    for i, label in enumerate(labels):
        if label != "BenignTraffic":
            objective[i * k + groups.index(mapping[label])] += 1
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized options detected.*")
        result = milp(objective, integrality=np.ones(nx + nz), bounds=Bounds(lo, hi),
                      constraints=LinearConstraint(matrix.tocsc(), lower, upper),
                      options={"time_limit": PARAMETERS["wrong_solver_seconds"], "mip_rel_gap": 0.0, "threads": 2})
    if result.x is None:
        raise RuntimeError(f"Exact-support control solver failed: {result.message}")
    assignment = np.rint(result.x[:nx].reshape(n, k)).astype(int)
    if not np.all(assignment.sum(axis=1) == 1):
        raise RuntimeError("Nonintegral control solution")
    wrong = {label: groups[int(assignment[i].argmax())] for i, label in enumerate(labels)}
    actual = {g: sum(counts[l] for l in labels if wrong[l] == g) for g in groups}
    if Counter(wrong.values()) != sizes or actual != target:
        raise RuntimeError("Wrong control not exactly cardinality AND fit-support matched")
    original_pairs = len(pairs)
    retained_pairs = sum(wrong[labels[a]] == wrong[labels[b]] for a, b in pairs)
    if retained_pairs == original_pairs:
        raise RuntimeError("Control only renames groups; original partition unchanged")
    return wrong, {
        "seed": seed, "solver_status": int(result.status), "solver_message": str(result.message),
        "optimality_proven": result.status == 0, "mip_gap": float(result.mip_gap),
        "solver_seconds": time.perf_counter() - start,
        "label_cardinality_exact": True, "fit_row_support_exact": True,
        "support_scope": "capped fit rows only; not test or uncapped base-pool support",
        "benign_fixed": True,
        "attack_labels_retained": sum(wrong[l] == mapping[l] for l in labels if l != "BenignTraffic"),
        "attack_labels_changed": sum(wrong[l] != mapping[l] for l in labels if l != "BenignTraffic"),
        "original_attack_copairs": original_pairs, "retained_attack_copairs": retained_pairs,
        "partition_ari": float(adjusted_rand_score([mapping[l] for l in labels], [wrong[l] for l in labels])),
        "strict_zero_retention_claimed": False,
    }


def load_full() -> tuple[pd.DataFrame, list[str], LabelEncoder]:
    full = pd.concat([pd.read_parquet(CACHE / "train.parquet"), pd.read_parquet(CACHE / "test.parquet")], ignore_index=True)
    features = nested.exp.feature_cols(full)
    benefit.assign_exact_predictor_groups(full, features)
    encoder = LabelEncoder().fit(full.Label.astype(str))
    return full, features, encoder


def roles(full: pd.DataFrame, seed: int):
    development, outer = benefit.stratified_group_split(full, test_size=0.2, seed=seed)
    base, utility, selection = benefit.split_development(development, seed, 0.70, 0.15, 0.15, group_disjoint=True)
    fit = nested.cap_fit_rows(base, seed, 25000, 80000)
    return base, utility, selection, outer, fit


def prepare() -> None:
    physical_d(ROOT)
    if (FROZEN / "prepared.json").exists():
        raise FileExistsError("Preparation already frozen; use --run or --verify")
    sources = json.loads((ROOT / "source_manifest.json").read_text())
    for entry in sources:
        if digest(Path(entry["path"])) != entry["sha256"]:
            raise RuntimeError("Official snapshot hash changed")
        Path(entry["path"]).chmod(stat.S_IREAD)
    table, mapping = standard_spec()
    table.to_csv(FROZEN / "standard_mapping.csv", index=False)
    dump(FROZEN / "standard_map.json", mapping, exclusive=True)
    dump(FROZEN / "design_lock.json", {
        "frozen_at_utc": now(), "parameters": PARAMETERS,
        "mechanism_spec_sha256": digest(ROOT / "mechanism_spec.json"),
        "protocol_sha256": digest(ROOT / "PROTOCOL.md"),
        "runner_sha256": digest(Path(__file__)), "standard_map_sha256": digest(FROZEN / "standard_map.json"),
        "helper_hashes": {str(Path(m.__file__)): digest(Path(m.__file__)) for m in (nested, benefit, trf, nested.exp)},
        "official_sources": sources,
        "wrong_algorithm": "MILP: exact group label cardinality and capped-fit row support; integer lexicographic objective minimizes retained original co-pairs then retained named attack memberships; benign fixed; frozen solver assignment resolves ties",
        "evaluation": "0.5 cached core-flat + 0.5 newly fitted hierarchy; no calibration or outcome selection",
        "status": "retrospective v15 control frozen before its new outer predictions; not preregistered independent confirmation",
    }, exclusive=True)
    copies = [cached_copy(nested.TRAIN_PATH, CACHE / "train.parquet"), cached_copy(nested.TEST_PATH, CACHE / "test.parquet")]
    for seed in SEEDS:
        copies.append(cached_copy(CORE / f"paired_predictions_seed_{seed}.npz", CACHE / f"core_seed_{seed}.npz"))
    copies.append(cached_copy(CORE / "nested_split_results.csv", CACHE / "core_results.csv"))
    copies.append(cached_copy(CORE / "run_metadata.json", CACHE / "core_metadata.json"))
    full, features, encoder = load_full()
    if set(encoder.classes_) != set(mapping):
        raise ValueError("Standard map differs from full canonical 34-label set")
    metadata = json.loads((CACHE / "core_metadata.json").read_text())
    if features != metadata["features"]:
        raise ValueError("Feature order differs from core")
    for key in ("outer_test_frac", "base_fraction", "gate_train_fraction", "gate_select_fraction", "majority_cap", "benign_cap", "local_trees"):
        if metadata["arguments"][key] != PARAMETERS[key]:
            raise ValueError(f"Parameters differ from canonical core: {key}")
    audits, support_rows, wrong_meta = [], [], []
    for seed in SEEDS:
        print(f"Preparing fit-only wrong map seed {seed}", flush=True)
        base, utility, selection, outer, fit = roles(full, seed)
        counts = {str(k): int(v) for k, v in fit.Label.value_counts().items()}
        wrong, control_audit = wrong_map_exact(mapping, counts, seed)
        dump(FROZEN / f"wrong_map_seed_{seed}.json", wrong, exclusive=True)
        wrong_meta.append(control_audit)
        fit.to_parquet(CACHE / f"fit_seed_{seed}.parquet", index=False)
        np.savez_compressed(FROZEN / f"roles_seed_{seed}.npz", base_id=base.index.to_numpy(), utility_id=utility.index.to_numpy(), selection_id=selection.index.to_numpy(), outer_id=outer.index.to_numpy(), rare_ids=trf.rare_class_ids(encoder.transform(base.Label.astype(str))))
        frames = {"base": base, "utility": utility, "selection": selection, "outer": outer}
        overlaps = {f"{a}_{b}_overlap": benefit.group_overlap(frames[a], frames[b]) for a, b in itertools.combinations(frames, 2)}
        if any(overlaps.values()):
            raise RuntimeError("Predictor group overlaps roles")
        audits.append({"seed": seed, "fit_rows": len(fit), "outer_rows": len(outer), "fit_sha256": digest(CACHE / f"fit_seed_{seed}.parquet"), **overlaps})
        for group in sorted(set(mapping.values())):
            real_labels = [l for l in sorted(mapping) if mapping[l] == group]
            wrong_labels = [l for l in sorted(wrong) if wrong[l] == group]
            support_rows.append({"seed": seed, "group": group, "standard_label_count": len(real_labels), "wrong_label_count": len(wrong_labels), "standard_fit_rows": sum(counts[l] for l in real_labels), "wrong_fit_rows": sum(counts[l] for l in wrong_labels), "standard_labels": ";".join(real_labels), "wrong_labels": ";".join(wrong_labels)})
    pd.DataFrame(audits).to_csv(FROZEN / "role_audit.csv", index=False)
    pd.DataFrame(support_rows).to_csv(FROZEN / "support_match_audit.csv", index=False)
    dump(FROZEN / "wrong_control_audit.json", wrong_meta, exclusive=True)
    prepared = {"prepared_at_utc": now(), "rows": len(full), "features": features, "fine_labels": list(encoder.classes_), "cache_copies": copies, "files": {str(p.relative_to(ROOT)): digest(p) for p in sorted(FROZEN.iterdir()) if p.is_file()}}
    dump(FROZEN / "prepared.json", prepared, exclusive=True)
    print("PREPARED: all six fit-only wrong maps frozen; no core outer probabilities accessed.", flush=True)


def verify() -> dict:
    physical_d(ROOT)
    prepared = json.loads((FROZEN / "prepared.json").read_text())
    lock = json.loads((FROZEN / "design_lock.json").read_text())
    if digest(Path(__file__)) != lock["runner_sha256"]:
        raise RuntimeError("Runner changed after design freeze")
    if digest(ROOT / "mechanism_spec.json") != lock["mechanism_spec_sha256"]:
        raise RuntimeError("Mechanism specification changed")
    if digest(ROOT / "PROTOCOL.md") != lock["protocol_sha256"]:
        raise RuntimeError("Protocol changed after freeze")
    for path, expected in lock["helper_hashes"].items():
        if digest(Path(path)) != expected:
            raise RuntimeError(f"Imported helper changed: {path}")
    for relative, expected in prepared["files"].items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError(f"Frozen file changed: {relative}")
    for entry in prepared["cache_copies"]:
        if digest(Path(entry["cache"])) != entry["sha256"]:
            raise RuntimeError("Cached canonical input changed")
    for row in pd.read_csv(FROZEN / "role_audit.csv").to_dict("records"):
        if digest(CACHE / f"fit_seed_{row['seed']}.parquet") != row["fit_sha256"]:
            raise RuntimeError("Frozen fit cache changed")
    return prepared


def score(run_id: str, y: np.ndarray, probability: np.ndarray, rare: np.ndarray, prediction=None) -> dict:
    row = trf.metric_row(run_id, y, probability, rare)
    if prediction is not None:
        recalls = recall_score(y, prediction, labels=np.arange(probability.shape[1]), average=None, zero_division=0)
        row.update(accuracy=float(accuracy_score(y, prediction)), f1_macro=float(f1_score(y, prediction, average="macro", zero_division=0)), recall_macro=float(np.mean(recalls)), rare_recall=float(np.mean(recalls[rare])), worst_recall=float(np.min(recalls)))
    return row


def run() -> None:
    prepared = verify()
    mapping = json.loads((FROZEN / "standard_map.json").read_text())
    features = prepared["features"]
    encoder = LabelEncoder().fit(prepared["fine_labels"])
    full, actual_features, _ = load_full()
    if actual_features != features:
        raise RuntimeError("Feature order changed")
    core_metrics = pd.read_csv(CACHE / "core_results.csv")
    original_factory = trf.make_lgbm

    def factory(n_classes, n_estimators):
        return original_factory(n_classes, n_estimators).set_params(n_jobs=2)

    trf.make_lgbm = factory
    started = time.perf_counter()
    all_rows, costs, order_audits, class_rows = [], [], [], []
    for seed in SEEDS:
        progress = {"status": "RUNNING", "seed": seed, "at_utc": now(), "completed_seeds": sorted({r["seed"] for r in all_rows})}
        dump(OUT / "run_status.json", progress)
        print(f"RUN seed {seed}", flush=True)
        trf.RANDOM_STATE = seed
        role = np.load(FROZEN / f"roles_seed_{seed}.npz")
        core = np.load(CACHE / f"core_seed_{seed}.npz")
        ids = core["row_id"].astype(np.int64)
        if set(ids) != set(role["outer_id"]) or len(ids) != len(role["outer_id"]):
            raise RuntimeError("Core outer row set mismatch")
        outer = full.loc[ids]
        y = encoder.transform(outer.Label.astype(str))
        rare = role["rare_ids"].astype(int)
        if not np.array_equal(y, core["y_true"]) or not np.array_equal(rare, core["rare_ids"]):
            raise RuntimeError("Core label/rare class mismatch")
        flat = np.asarray(core["flat_weighted_xgboost_probability"], dtype=np.float64)
        hard = core["flat_weighted_xgboost"].astype(int)
        result = score("coreflat", y, flat, rare, hard)
        reference = core_metrics[(core_metrics.seed == seed) & (core_metrics.run_id == "flat_weighted_xgboost")].iloc[0]
        for metric in ("accuracy", "f1_macro", "recall_macro", "rare_recall", "worst_recall"):
            if not np.isclose(result[metric], reference[metric], atol=1e-12, rtol=0):
                raise RuntimeError(f"Saved core-flat {metric} does not match canonical result")
        result.update(seed=seed, provenance="reused canonical hard predictions; cached float16 probabilities for PR-AUC")
        all_rows.append(result)
        quantized = score("coreflat_cached_probability", y, flat, rare)
        quantized.update(seed=seed, provenance="argmax of cached float16 probabilities")
        all_rows.append(quantized)
        order_audits.append({"seed": seed, "row_set_exact": True, "prediction_order_exact": True, "y_true_exact": True, "rare_ids_exact": True, "row_order_sha256": array_digest(ids), "core_hard_metrics_match": True, "float16_argmax_disagreements": int(np.sum(hard != np.argmax(flat, axis=1))), "float16_probability_row_sum_max_error": float(np.max(np.abs(flat.sum(axis=1) - 1)))})
        fit = pd.read_parquet(CACHE / f"fit_seed_{seed}.parquet")
        probabilities = {}
        for run_id, group_map in (("standardblend", mapping), ("wrongstandardblend", json.loads((FROZEN / f"wrong_map_seed_{seed}.json").read_text()))):
            print(f"seed {seed}: training {run_id}, n_jobs=2", flush=True)
            with threadpool_limits(limits=2):
                family_model, family_encoder, local_models, training_seconds = benefit.taxonomy_bundle(fit, features, group_map, 380, None)
                start = time.perf_counter()
                taxonomy, _ = benefit.predict_taxonomy(outer, features, family_model, family_encoder, encoder, group_map, local_models)
                inference_seconds = time.perf_counter() - start
            probability = 0.5 * flat + 0.5 * taxonomy
            if not np.isfinite(probability).all() or np.min(probability) < 0:
                raise RuntimeError("Invalid blend probability")
            model_path = CACHE / f"models_{run_id}_seed_{seed}.joblib"
            save_start = time.perf_counter()
            joblib.dump({"family_model": family_model, "family_encoder": family_encoder, "local_models": local_models, "mapping": group_map, "features": features}, model_path, compress=3)
            row = score(run_id, y, probability, rare)
            row.update(seed=seed, provenance="new hierarchy fit; equal blend with cached float16 core-flat")
            all_rows.append(row)
            costs.append({"seed": seed, "run_id": run_id, "training_seconds": training_seconds, "hierarchy_inference_seconds": inference_seconds, "serialization_seconds": time.perf_counter() - save_start, "model_bytes": model_path.stat().st_size, "n_jobs": 2, "fit_rows": len(fit), "outer_rows": len(outer), "family_classes": len(family_encoder.classes_), "trained_local_models": sum(m[0] is not None for m in local_models.values()), "rss_peak_mib": benefit.process_peak_rss_mb()})
            probabilities[run_id] = probability.astype(np.float32)
            probabilities[f"{run_id}_prediction"] = np.argmax(probability, axis=1).astype(np.int16)
            classes = trf.classwise_frame(run_id, y, probability, list(encoder.classes_))
            classes["seed"] = seed
            classes["run_id"] = run_id
            class_rows.extend(classes.to_dict("records"))
            print(f"seed {seed}: {run_id} macro-F1={row['f1_macro']:.8f}; fit={training_seconds:.1f}s inference={inference_seconds:.1f}s", flush=True)
            del family_model, family_encoder, local_models, taxonomy, probability
        np.savez_compressed(OUT / f"paired_predictions_seed_{seed}.npz", row_id=ids, y_true=y.astype(np.int16), rare_ids=rare.astype(np.int16), coreflat_prediction=hard.astype(np.int16), **probabilities)
        pd.DataFrame(all_rows).to_csv(OUT / "model_results.csv", index=False)
        pd.DataFrame(costs).to_csv(OUT / "costs.csv", index=False)
        pd.DataFrame(order_audits).to_csv(OUT / "prediction_order_audit.csv", index=False)
        pd.DataFrame(class_rows).to_csv(OUT / "classwise_metrics.csv", index=False)
    results = pd.DataFrame(all_rows)
    metrics = ["accuracy", "f1_macro", "recall_macro", "rare_recall", "worst_recall", "pr_auc_macro"]
    results.groupby("run_id")[metrics].agg(["mean", "std", "min", "max"]).to_csv(OUT / "summary.csv")
    pivot = results.pivot(index="seed", columns="run_id", values=metrics)
    differences = []
    for seed in SEEDS:
        for metric in metrics:
            differences.append({"seed": seed, "metric": metric, "standard_minus_wrong": float(pivot.loc[seed, (metric, "standardblend")] - pivot.loc[seed, (metric, "wrongstandardblend")]), "standard_minus_coreflat": float(pivot.loc[seed, (metric, "standardblend")] - pivot.loc[seed, (metric, "coreflat")])})
    pd.DataFrame(differences).to_csv(OUT / "paired_differences.csv", index=False)
    dump(OUT / "run_metadata.json", {"started_elapsed_seconds": time.perf_counter() - started, "completed_at_utc": now(), "parameters": PARAMETERS, "frozen_prepared_sha256": digest(FROZEN / "prepared.json"), "package_versions": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "xgboost")}, "uncertainty": "Six overlapping grouped random splits: report descriptive mean/SD; not six independent datasets or time-held-out cohorts.", "cost_scope": "new hierarchy only; flat training/inference cost excluded because predictions reused; no physical-edge or energy experiment"})
    dump(OUT / "run_status.json", {"status": "COMPLETE", "completed_seeds": SEEDS, "completed_at_utc": now()})
    print("ALL SIX SEEDS COMPLETE", flush=True)


def self_test() -> None:
    mapping = {"BenignTraffic": "Benign", "a": "A", "b": "A", "c": "B", "d": "B", "e": "C"}
    wrong, audit = wrong_map_exact(mapping, {l: 100 if l != "e" else 17 for l in mapping}, 42)
    assert Counter(wrong.values()) == Counter(mapping.values())
    assert audit["fit_row_support_exact"] and audit["retained_attack_copairs"] < audit["original_attack_copairs"]
    assert wrong["e"] == "C"
    print(json.dumps(audit, indent=2))
    print("SELF TEST PASS: exact support, fixed benign, forced singleton retention, nontrivial partition")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "verify", "run", "self-test"])
    mode = parser.parse_args().mode
    if mode == "prepare": prepare()
    elif mode == "verify": verify(); print("FROZEN INPUTS VERIFIED")
    elif mode == "run": run()
    else: self_test()
