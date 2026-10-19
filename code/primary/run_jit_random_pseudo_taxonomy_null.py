"""Run a leakage controlled random pseudo taxonomy null experiment.

The experiment isolates the contribution of the family level prior.  For each
outer split, the frozen weighted flat probabilities supply the within-group
conditional probabilities.  Only the group posterior is learned again.

Two null families are supported:

* random_partition: random label partitions with the observed group size
  profile, independent of feature values;
* feature_subspace_cluster: balanced partitions obtained from class centroids
  in a randomly selected feature subspace.  Centroids and assignments use the
  fit role only.

Neither map uses validation or outer test records.  The script deliberately
does not tune a map against the outer test metrics.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, label_binarize

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = (
    ROOT
    / "data"
    / "raw"
    / "ciciot2023_hf_neto_subsample"
    / "random"
    / "train-00000-of-00001.parquet"
)
TEST_PATH = (
    ROOT
    / "data"
    / "raw"
    / "ciciot2023_hf_neto_subsample"
    / "random"
    / "test-00000-of-00001.parquet"
)

CANONICAL_DIR = (
    ROOT / "results" / "ijics_v2_ciciot_agreement_gated_trf_6split"
)
EPS = 1e-12
DEFAULT_SPLIT_SEEDS = [42, 43, 44, 45, 46, 47]
DEFAULT_MAP_SEED = 20260810
LABEL_COLUMNS = {"Label", "Label_orig", "attack_class", "label"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run random pseudo taxonomy null controls on CICIoT2023."
    )
    parser.add_argument("--n-replicates", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SPLIT_SEEDS)
    parser.add_argument("--map-seed", type=int, default=DEFAULT_MAP_SEED)
    parser.add_argument("--outer-test-frac", type=float, default=0.2)
    parser.add_argument("--inner-validation-frac", type=float, default=0.2)
    parser.add_argument("--majority-cap", type=int, default=25_000)
    parser.add_argument("--benign-cap", type=int, default=80_000)
    parser.add_argument("--family-trees", type=int, default=380)
    parser.add_argument(
        "--group-model",
        choices=["gaussian_nb", "lgbm"],
        default="gaussian_nb",
        help="Fast Gaussian family classifier for the null distribution, or LightGBM.",
    )
    parser.add_argument("--subspace-size", type=int, default=16)
    parser.add_argument("--kmeans-n-init", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["random_partition", "feature_subspace_cluster"],
        default=["random_partition", "feature_subspace_cluster"],
    )
    parser.add_argument(
        "--out-dir",
        default="jit_random_pseudo_taxonomy_null_v1",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume completed outer seeds from the partial metrics file.",
    )
    return parser.parse_args()


def cap_fit_rows(
    frame: pd.DataFrame,
    seed: int,
    majority_cap: int,
    benign_cap: int,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for label, group in frame.groupby("Label", sort=False):
        cap = benign_cap if str(label) == "BenignTraffic" else majority_cap
        if len(group) > cap:
            group = group.sample(n=cap, random_state=seed)
        parts.append(group)
    return (
        pd.concat(parts, ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


def feature_cols(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in LABEL_COLUMNS
        and pd.api.types.is_numeric_dtype(frame[column])
    ]


def feature_matrix(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    return (
        frame[features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .to_numpy(dtype=np.float32)
    )


def make_group_model(n_groups: int, trees: int, seed: int, n_jobs: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="multiclass",
        num_class=n_groups,
        n_estimators=trees,
        learning_rate=0.055,
        num_leaves=127,
        max_depth=-1,
        min_child_samples=25,
        subsample=0.9,
        colsample_bytree=0.95,
        reg_lambda=1.5,
        reg_alpha=0.02,
        random_state=seed,
        n_jobs=n_jobs,
        verbosity=-1,
    )


def inverse_frequency_weights(y: np.ndarray, exponent: float) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    raw = {
        int(c): float(len(y) / (len(classes) * n)) ** exponent
        for c, n in zip(classes, counts)
    }
    values = np.asarray([raw[int(v)] for v in y], dtype=np.float32)
    return values / np.mean(values)


def metric_row(
    method: str,
    seed: int,
    replicate: int | None,
    y_true: np.ndarray,
    probability: np.ndarray,
    rare_ids: np.ndarray,
    map_type: str,
    map_seed: int | None,
    feature_names: list[str],
    fit_rows: int,
    validation_rows: int,
    outer_rows: int,
    training_seconds: float,
) -> dict[str, object]:
    prediction = np.argmax(probability, axis=1)
    recalls = recall_score(
        y_true,
        prediction,
        labels=np.arange(probability.shape[1]),
        average=None,
        zero_division=0,
    )
    y_binary = label_binarize(y_true, classes=np.arange(probability.shape[1]))
    row: dict[str, object] = {
        "method": method,
        "seed": int(seed),
        "replicate": np.nan if replicate is None else int(replicate),
        "map_type": map_type,
        "map_seed": np.nan if map_seed is None else int(map_seed),
        "feature_count": int(len(feature_names)),
        "feature_names": ",".join(feature_names),
        "n_test": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "f1_macro": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, prediction, average="macro", zero_division=0)),
        "rare_recall": float(np.mean(recalls[rare_ids])),
        "worst_recall": float(np.min(recalls)),
        "pr_auc_macro": float(average_precision_score(y_binary, probability, average="macro")),
        "fit_rows": int(fit_rows),
        "validation_rows": int(validation_rows),
        "outer_rows": int(outer_rows),
        "training_seconds": float(training_seconds),
    }
    return row


def group_size_profile(label_to_family: dict[str, str]) -> list[int]:
    return sorted(Counter(label_to_family.values()).values(), reverse=True)


def random_partition_map(
    labels: list[str],
    sizes: list[int],
    seed: int,
) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(labels, dtype=object).copy()
    rng.shuffle(shuffled)
    result: dict[str, str] = {}
    start = 0
    for group_id, size in enumerate(sizes):
        group_name = f"Random_{group_id:02d}"
        for label in shuffled[start : start + size]:
            result[str(label)] = group_name
        start += size
    if len(result) != len(labels):
        raise RuntimeError("The random partition did not cover every class.")
    return result


def standardised_class_centroids(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    mean = np.nanmean(X_fit, axis=0)
    scale = np.nanstd(X_fit, axis=0)
    scale = np.where(scale > 1e-7, scale, 1.0)
    result = np.zeros((n_classes, X_fit.shape[1]), dtype=np.float64)
    for class_id in range(n_classes):
        mask = y_fit == class_id
        if not np.any(mask):
            raise RuntimeError(f"Class {class_id} is absent from the fit role.")
        result[class_id] = np.nanmean(X_fit[mask], axis=0)
    return (result - mean) / scale


def balanced_feature_cluster_map(
    labels: list[str],
    centroids: np.ndarray,
    sizes: list[int],
    feature_indices: np.ndarray,
    seed: int,
    kmeans_n_init: int,
) -> dict[str, str]:
    selected = np.nan_to_num(centroids[:, feature_indices], nan=0.0, posinf=0.0, neginf=0.0)
    kmeans = KMeans(
        n_clusters=len(sizes),
        n_init=kmeans_n_init,
        random_state=seed,
    )
    kmeans.fit(selected)
    centers = np.asarray(kmeans.cluster_centers_, dtype=np.float64)
    distance = np.sum((selected[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    slots = np.concatenate(
        [np.full(size, group_id, dtype=np.int64) for group_id, size in enumerate(sizes)]
    )
    cost = distance[:, slots]
    row_ind, column_ind = linear_sum_assignment(cost)
    if not np.array_equal(row_ind, np.arange(len(labels))):
        raise RuntimeError("The balanced assignment did not cover every class.")
    result = {
        str(label): f"Feature_{int(slots[column]):02d}"
        for label, column in zip(labels, column_ind)
    }
    if len(result) != len(labels):
        raise RuntimeError("The feature cluster map did not cover every class.")
    return result


def map_pairwise_agreement(
    labels: list[str],
    true_map: dict[str, str],
    pseudo_map: dict[str, str],
) -> tuple[float, float]:
    true_values = np.asarray([true_map[label] for label in labels], dtype=object)
    pseudo_values = np.asarray([pseudo_map[label] for label in labels], dtype=object)
    true_same = true_values[:, None] == true_values[None, :]
    pseudo_same = pseudo_values[:, None] == pseudo_values[None, :]
    upper = np.triu(np.ones_like(true_same, dtype=bool), k=1)
    pairwise = float(np.mean(true_same[upper] == pseudo_same[upper]))
    # The adjusted Rand index is not needed for inference, but it is useful
    # for describing how close each null map was to the real taxonomy.
    from sklearn.metrics import adjusted_rand_score

    ari = float(adjusted_rand_score(true_values, pseudo_values))
    return pairwise, ari


def fit_group_probability(
    fit_X: np.ndarray,
    outer_X: np.ndarray,
    fit_labels: pd.Series,
    fine_encoder: LabelEncoder,
    label_map: dict[str, str],
    trees: int,
    seed: int,
    n_jobs: int,
    feature_indices: np.ndarray,
    group_model: str,
) -> tuple[np.ndarray, LabelEncoder, float]:
    fit_class_labels = fit_labels.astype(str).to_numpy()
    group_values = np.asarray([label_map[str(label)] for label in fit_class_labels], dtype=object)
    group_encoder = LabelEncoder().fit(group_values)
    y_group = group_encoder.transform(group_values)
    start = time.perf_counter()
    if group_model == "gaussian_nb":
        model = GaussianNB(var_smoothing=1e-9)
        model.fit(
            fit_X[:, feature_indices],
            y_group,
            sample_weight=inverse_frequency_weights(y_group, 0.25),
        )
    elif group_model == "lgbm":
        model = make_group_model(len(group_encoder.classes_), trees, seed, n_jobs)
        model.fit(
            fit_X[:, feature_indices],
            y_group,
            sample_weight=inverse_frequency_weights(y_group, 0.25),
        )
    else:
        raise ValueError(f"Unsupported group model: {group_model}")
    probability = np.asarray(model.predict_proba(outer_X[:, feature_indices]), dtype=np.float64)
    return probability, group_encoder, time.perf_counter() - start


def group_only_taxonomy_probability(
    flat_probability: np.ndarray,
    group_probability: np.ndarray,
    labels: list[str],
    label_map: dict[str, str],
    group_encoder: LabelEncoder,
) -> np.ndarray:
    class_to_group = group_encoder.transform(
        np.asarray([label_map[str(label)] for label in labels], dtype=object)
    )
    flat_group = np.zeros((len(flat_probability), len(group_encoder.classes_)), dtype=np.float64)
    for class_id, group_id in enumerate(class_to_group):
        flat_group[:, group_id] += flat_probability[:, class_id]
    conditional = flat_probability / np.clip(flat_group[:, class_to_group], EPS, None)
    result = group_probability[:, class_to_group] * conditional
    return result / np.clip(result.sum(axis=1, keepdims=True), EPS, None)


def load_canonical_predictions(seed: int) -> dict[str, np.ndarray]:
    path = CANONICAL_DIR / f"paired_predictions_seed_{seed}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing canonical prediction file: {path}")
    with np.load(path) as archive:
        required = {
            "row_id",
            "y_true",
            "rare_ids",
            "flat_weighted_xgboost_probability",
            "equal_flat_taxonomy_blend_probability",
        }
        missing = required.difference(archive.files)
        if missing:
            raise RuntimeError(f"Canonical prediction file lacks keys: {sorted(missing)}")
        return {key: np.asarray(archive[key]) for key in required}


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    args = parse_args()
    out_dir = ROOT / "results" / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.n_replicates < 1:
        raise ValueError("--n-replicates must be positive.")
    if any(seed < 0 for seed in args.seeds):
        raise ValueError("Split seeds must be nonnegative.")

    print("Loading the public subsample and rebuilding only the declared split roles...")
    full = pd.concat(
        [pd.read_parquet(TRAIN_PATH), pd.read_parquet(TEST_PATH)],
        ignore_index=True,
    )
    features = feature_cols(full)
    fine_encoder = LabelEncoder().fit(full["Label"].astype(str))
    labels = list(fine_encoder.classes_)
    true_map = (
        full[["Label", "attack_class"]]
        .drop_duplicates()
        .set_index("Label")["attack_class"]
        .astype(str)
        .to_dict()
    )
    sizes = group_size_profile(true_map)
    if sum(sizes) != len(labels):
        raise RuntimeError("The observed family size profile does not cover all labels.")

    mode_maps: dict[str, dict[int, dict[str, str]]] = {}
    map_records: list[dict[str, object]] = []
    for mode in args.modes:
        if mode != "random_partition":
            continue
        mode_maps[mode] = {}
        for replicate in range(args.n_replicates):
            map_seed = args.map_seed + replicate
            mapping = random_partition_map(labels, sizes, map_seed)
            mode_maps[mode][replicate] = mapping
            pairwise, ari = map_pairwise_agreement(labels, true_map, mapping)
            map_records.append(
                {
                    "mode": mode,
                    "seed": np.nan,
                    "replicate": replicate,
                    "map_seed": map_seed,
                    "feature_count": len(features),
                    "feature_names": ",".join(features),
                    "pairwise_taxonomy_agreement": pairwise,
                    "adjusted_rand_index": ari,
                    "label_to_group": json.dumps(mapping, sort_keys=True),
                }
            )

    partial_metrics_path = out_dir / "pseudo_taxonomy_split_replicate_metrics_partial.csv"
    partial_audit_path = out_dir / "split_audit_partial.csv"
    all_rows: list[dict[str, object]] = []
    split_audit: list[dict[str, object]] = []
    completed_seeds: set[int] = set()
    if args.resume and partial_metrics_path.exists():
        existing_metrics = pd.read_csv(partial_metrics_path)
        all_rows = existing_metrics.to_dict(orient="records")
        if "method" in existing_metrics.columns:
            completed_seeds = set(
                existing_metrics.loc[
                    existing_metrics["method"] == "flat_weighted_xgboost", "seed"
                ]
                .dropna()
                .astype(int)
                .tolist()
            )
        if partial_audit_path.exists():
            split_audit = pd.read_csv(partial_audit_path).to_dict(orient="records")
        print(f"Resuming with completed seeds: {sorted(completed_seeds)}")

    for seed in args.seeds:
        if seed in completed_seeds:
            print(f"Skipping completed outer split seed {seed}")
            continue
        print(f"\nOuter split seed {seed}")
        development_df, outer_test_df = train_test_split(
            full,
            test_size=args.outer_test_frac,
            random_state=seed,
            stratify=full["Label"],
        )
        fit_pool_df, validation_df = train_test_split(
            development_df,
            test_size=args.inner_validation_frac,
            random_state=seed + 10_000,
            stratify=development_df["Label"],
        )
        fit_df = cap_fit_rows(
            fit_pool_df,
            seed,
            args.majority_cap,
            args.benign_cap,
        )
        canonical = load_canonical_predictions(seed)
        expected_row_id = outer_test_df.index.to_numpy(dtype=np.int64)
        if not np.array_equal(expected_row_id, canonical["row_id"]):
            raise RuntimeError(
                f"Canonical row order differs for seed {seed}; refusing to evaluate."
            )
        y_outer = canonical["y_true"].astype(np.int64)
        if not np.array_equal(
            y_outer,
            fine_encoder.transform(outer_test_df["Label"].astype(str)),
        ):
            raise RuntimeError(f"Canonical labels differ for seed {seed}.")
        rare_ids = canonical["rare_ids"].astype(np.int64)
        flat_probability = canonical["flat_weighted_xgboost_probability"].astype(np.float64)
        semantic_probability = canonical["equal_flat_taxonomy_blend_probability"].astype(np.float64)
        flat_probability /= np.clip(flat_probability.sum(axis=1, keepdims=True), EPS, None)
        semantic_probability /= np.clip(semantic_probability.sum(axis=1, keepdims=True), EPS, None)

        base_common = {
            "fit_rows": len(fit_df),
            "validation_rows": len(validation_df),
            "outer_rows": len(outer_test_df),
        }
        flat_row = metric_row(
            "flat_weighted_xgboost",
            seed,
            None,
            y_outer,
            flat_probability,
            rare_ids,
            "none",
            None,
            features,
            **base_common,
            training_seconds=np.nan,
        )
        semantic_row = metric_row(
            "real_semantic_blend",
            seed,
            None,
            y_outer,
            semantic_probability,
            rare_ids,
            "real_taxonomy",
            None,
            features,
            **base_common,
            training_seconds=np.nan,
        )
        all_rows.extend([flat_row, semantic_row])

        fit_X = feature_matrix(fit_df, features)
        outer_X = feature_matrix(outer_test_df, features)
        fit_y = fine_encoder.transform(fit_df["Label"].astype(str))
        centroids = standardised_class_centroids(fit_X, fit_y, len(labels))

        # This is the capacity matched reference for the random nulls.  It
        # uses the real family labels but the same group only construction as
        # every pseudo taxonomy.  The full semantic blend is retained as a
        # separate, stronger reference from the frozen prediction ledger.
        real_group_probability, real_group_encoder, real_group_seconds = fit_group_probability(
            fit_X,
            outer_X,
            fit_df["Label"],
            fine_encoder,
            true_map,
            args.family_trees,
            seed,
            args.n_jobs,
            np.arange(len(features), dtype=np.int64),
            args.group_model,
        )
        real_group_taxonomy = group_only_taxonomy_probability(
            flat_probability,
            real_group_probability,
            labels,
            true_map,
            real_group_encoder,
        )
        real_group_blend = 0.5 * (flat_probability + real_group_taxonomy)
        real_group_blend /= np.clip(real_group_blend.sum(axis=1, keepdims=True), EPS, None)
        real_group_row = metric_row(
            "real_group_only_blend",
            seed,
            None,
            y_outer,
            real_group_blend,
            rare_ids,
            "real_taxonomy_group_only",
            None,
            features,
            **base_common,
            training_seconds=real_group_seconds,
        )
        all_rows.append(real_group_row)

        modes_for_seed: list[tuple[str, int, dict[str, str], np.ndarray, int, list[str]]] = []
        for mode in args.modes:
            for replicate in range(args.n_replicates):
                if mode == "random_partition":
                    mapping = mode_maps[mode][replicate]
                    feature_indices = np.arange(len(features), dtype=np.int64)
                    map_seed = args.map_seed + replicate
                    feature_names = features
                else:
                    map_seed = args.map_seed + replicate + seed * 100_000
                    rng = np.random.default_rng(map_seed)
                    subspace_size = min(args.subspace_size, len(features))
                    feature_indices = np.sort(
                        rng.choice(len(features), size=subspace_size, replace=False)
                    ).astype(np.int64)
                    mapping = balanced_feature_cluster_map(
                        labels,
                        centroids,
                        sizes,
                        feature_indices,
                        map_seed,
                        args.kmeans_n_init,
                    )
                    feature_names = [features[index] for index in feature_indices]
                    pairwise, ari = map_pairwise_agreement(labels, true_map, mapping)
                    map_records.append(
                        {
                            "mode": mode,
                            "seed": seed,
                            "replicate": replicate,
                            "map_seed": map_seed,
                            "feature_count": len(feature_names),
                            "feature_names": ",".join(feature_names),
                            "pairwise_taxonomy_agreement": pairwise,
                            "adjusted_rand_index": ari,
                            "label_to_group": json.dumps(mapping, sort_keys=True),
                        }
                    )
                modes_for_seed.append(
                    (mode, replicate, mapping, feature_indices, map_seed, feature_names)
                )

        for mode, replicate, mapping, feature_indices, map_seed, feature_names in modes_for_seed:
            print(f"  {mode} replicate {replicate + 1}/{args.n_replicates}", flush=True)
            group_probability, group_encoder, training_seconds = fit_group_probability(
                fit_X,
                outer_X,
                fit_df["Label"],
                fine_encoder,
                mapping,
                args.family_trees,
                seed + replicate + 1,
                args.n_jobs,
                feature_indices,
                args.group_model,
            )
            pseudo_taxonomy = group_only_taxonomy_probability(
                flat_probability,
                group_probability,
                labels,
                mapping,
                group_encoder,
            )
            pseudo_blend = 0.5 * (flat_probability + pseudo_taxonomy)
            pseudo_blend /= np.clip(pseudo_blend.sum(axis=1, keepdims=True), EPS, None)
            method = (
                "random_partition_group_only"
                if mode == "random_partition"
                else "feature_subspace_cluster_group_only"
            )
            row = metric_row(
                method,
                seed,
                replicate,
                y_outer,
                pseudo_blend,
                rare_ids,
                mode,
                map_seed,
                feature_names,
                **base_common,
                training_seconds=training_seconds,
            )
            row["group_count"] = len(group_encoder.classes_)
            row["pairwise_taxonomy_agreement"] = (
                map_pairwise_agreement(labels, true_map, mapping)[0]
            )
            row["adjusted_rand_index"] = map_pairwise_agreement(labels, true_map, mapping)[1]
            row["delta_macro_f1_vs_flat"] = row["f1_macro"] - flat_row["f1_macro"]
            row["delta_rare_recall_vs_flat"] = row["rare_recall"] - flat_row["rare_recall"]
            all_rows.append(row)
            del group_probability, pseudo_taxonomy, pseudo_blend
            gc.collect()

        split_audit.append(
            {
                "seed": seed,
                "fit_pool_rows_before_cap": len(fit_pool_df),
                "fit_rows_after_cap": len(fit_df),
                "validation_rows": len(validation_df),
                "outer_test_rows": len(outer_test_df),
                "outer_row_order_matches_canonical": True,
                "validation_used_for_map_or_weight_selection": False,
                "outer_test_used_for_map_or_weight_selection": False,
            }
        )
        pd.DataFrame(all_rows).to_csv(partial_metrics_path, index=False)
        pd.DataFrame(split_audit).to_csv(partial_audit_path, index=False)
        del (
            fit_X,
            outer_X,
            centroids,
            flat_probability,
            semantic_probability,
            real_group_probability,
            real_group_taxonomy,
            real_group_blend,
        )
        gc.collect()

    metrics = pd.DataFrame(all_rows)
    for column in ("delta_macro_f1_vs_flat", "delta_rare_recall_vs_flat"):
        metrics[column] = np.nan
    for seed in args.seeds:
        flat = metrics[(metrics["seed"] == seed) & (metrics["method"] == "flat_weighted_xgboost")]
        if len(flat) != 1:
            raise RuntimeError(f"Expected exactly one flat row for seed {seed}.")
        flat_macro = float(flat.iloc[0]["f1_macro"])
        flat_rare = float(flat.iloc[0]["rare_recall"])
        mask = metrics["seed"] == seed
        metrics.loc[mask, "delta_macro_f1_vs_flat"] = (
            metrics.loc[mask, "f1_macro"].astype(float) - flat_macro
        )
        metrics.loc[mask, "delta_rare_recall_vs_flat"] = (
            metrics.loc[mask, "rare_recall"].astype(float) - flat_rare
        )

    metrics.to_csv(out_dir / "pseudo_taxonomy_split_replicate_metrics.csv", index=False)
    pd.DataFrame(map_records).to_json(
        out_dir / "pseudo_taxonomy_maps.jsonl",
        orient="records",
        lines=True,
        force_ascii=True,
    )
    pd.DataFrame(split_audit).to_csv(out_dir / "split_audit.csv", index=False)

    random_methods = [
        "random_partition_group_only",
        "feature_subspace_cluster_group_only",
    ]
    summary_rows: list[dict[str, object]] = []
    for method in [
        "flat_weighted_xgboost",
        "real_group_only_blend",
        "real_semantic_blend",
    ] + random_methods:
        frame = metrics[metrics["method"] == method]
        if frame.empty:
            continue
        for seed, split_frame in frame.groupby("seed", sort=True):
            summary_rows.append(
                {
                    "method": method,
                    "seed": int(seed),
                    "n_maps": int(split_frame["replicate"].notna().sum()),
                    "macro_delta_mean": float(split_frame["delta_macro_f1_vs_flat"].mean()),
                    "macro_delta_median": float(split_frame["delta_macro_f1_vs_flat"].median()),
                    "macro_delta_q025": float(split_frame["delta_macro_f1_vs_flat"].quantile(0.025)),
                    "macro_delta_q975": float(split_frame["delta_macro_f1_vs_flat"].quantile(0.975)),
                    "rare_delta_mean": float(split_frame["delta_rare_recall_vs_flat"].mean()),
                    "rare_delta_median": float(split_frame["delta_rare_recall_vs_flat"].median()),
                    "rare_delta_q025": float(split_frame["delta_rare_recall_vs_flat"].quantile(0.025)),
                    "rare_delta_q975": float(split_frame["delta_rare_recall_vs_flat"].quantile(0.975)),
                    "macro_positive_fraction": float(
                        np.mean(split_frame["delta_macro_f1_vs_flat"].astype(float) > 0)
                    ),
                    "rare_positive_fraction": float(
                        np.mean(split_frame["delta_rare_recall_vs_flat"].astype(float) > 0)
                    ),
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "pseudo_taxonomy_null_summary.csv", index=False)

    rank_rows: list[dict[str, object]] = []
    for seed in args.seeds:
        random_frame = metrics[
            (metrics["seed"] == seed)
            & (metrics["method"].isin(random_methods))
        ]
        real_frame = metrics[
            (metrics["seed"] == seed)
            & (metrics["method"] == "real_semantic_blend")
        ]
        real_group_frame = metrics[
            (metrics["seed"] == seed)
            & (metrics["method"] == "real_group_only_blend")
        ]
        if len(real_frame) != 1 or len(real_group_frame) != 1:
            raise RuntimeError(f"Expected one real semantic row for seed {seed}.")
        real_macro = float(real_frame.iloc[0]["delta_macro_f1_vs_flat"])
        real_rare = float(real_frame.iloc[0]["delta_rare_recall_vs_flat"])
        real_group_macro = float(real_group_frame.iloc[0]["delta_macro_f1_vs_flat"])
        real_group_rare = float(real_group_frame.iloc[0]["delta_rare_recall_vs_flat"])
        for method in random_methods:
            frame = random_frame[random_frame["method"] == method]
            if frame.empty:
                continue
            macro_values = frame["delta_macro_f1_vs_flat"].astype(float).to_numpy()
            rare_values = frame["delta_rare_recall_vs_flat"].astype(float).to_numpy()
            rank_rows.append(
                {
                    "seed": seed,
                    "null_method": method,
                    "n_maps": len(frame),
                    "real_group_only_macro_delta": real_group_macro,
                    "real_group_only_rare_delta": real_group_rare,
                    "real_semantic_macro_delta": real_macro,
                    "real_semantic_rare_delta": real_rare,
                    "null_macro_delta_ge_real_group_only_fraction": float(
                        np.mean(macro_values >= real_group_macro)
                    ),
                    "null_rare_delta_ge_real_group_only_fraction": float(
                        np.mean(rare_values >= real_group_rare)
                    ),
                    "null_macro_delta_ge_real_fraction": float(np.mean(macro_values >= real_macro)),
                    "null_rare_delta_ge_real_fraction": float(np.mean(rare_values >= real_rare)),
                    "null_macro_delta_mean": float(np.mean(macro_values)),
                    "null_rare_delta_mean": float(np.mean(rare_values)),
                    "null_macro_delta_q025": float(np.quantile(macro_values, 0.025)),
                    "null_macro_delta_q975": float(np.quantile(macro_values, 0.975)),
                    "null_rare_delta_q025": float(np.quantile(rare_values, 0.025)),
                    "null_rare_delta_q975": float(np.quantile(rare_values, 0.975)),
                }
            )
    pd.DataFrame(rank_rows).to_csv(
        out_dir / "real_semantic_vs_pseudo_null_rank.csv", index=False
    )

    metadata = {
        "purpose": "pre-registered random pseudo taxonomy null experiment",
        "dataset": "CICIoT2023 public subsample",
        "source_role": "third party development mirror",
        "canonical_predictions": str(CANONICAL_DIR),
        "n_replicates": args.n_replicates,
        "split_seeds": args.seeds,
        "map_seed": args.map_seed,
        "modes": args.modes,
        "features": features,
        "fine_labels": labels,
        "true_label_to_family": true_map,
        "observed_family_size_profile": sizes,
        "family_trees": args.family_trees,
        "group_model": args.group_model,
        "subspace_size": args.subspace_size,
        "kmeans_n_init": args.kmeans_n_init,
        "n_jobs": args.n_jobs,
        "map_generation": {
            "random_partition": "randomly assign fine labels to groups with the observed group size profile",
            "feature_subspace_cluster": "fit role class centroids, select a random feature subspace, cluster centroids, then solve a balanced assignment to the observed group size profile",
        },
        "probability_construction": "learned pseudo group posterior times the frozen weighted flat conditional probability within each pseudo group",
        "blend_weight": 0.5,
        "validation_used_for_selection": False,
        "outer_test_used_for_selection": False,
        "python": sys.version,
        "platform": platform.platform(),
        "command": " ".join(sys.argv),
    }
    (out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print("\nWrote random pseudo taxonomy null outputs to", out_dir)
    print(summary.to_string(index=False))
    print(pd.DataFrame(rank_rows).to_string(index=False))


if __name__ == "__main__":
    main()
