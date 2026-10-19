from __future__ import annotations

import argparse
import ctypes
import io
import importlib.metadata
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import run_ciciot_agreement_gated_trf_confirmation as nested
import run_ciciot_support_matched_control_v2 as support_control
import run_ciciot_taxonomy_residual_fusion as trf


ROOT = Path(__file__).resolve().parents[1]
REAL_ID = "calibrated_taxonomy_benefit_gate"
CYCLIC_ID = "calibrated_cyclic_taxonomy_control"
SUPPORT_ID = "calibrated_support_matched_control"
FEATURE_GROUP_COLUMN = "__exact_predictor_hash"


def process_peak_rss_mb() -> float:
    if platform.system() == "Windows":
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_current_process.restype = ctypes.c_void_p
        get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        get_process_memory_info.restype = ctypes.c_int
        handle = get_current_process()
        success = get_process_memory_info(
            handle,
            ctypes.byref(counters),
            counters.cb,
        )
        if success:
            return float(counters.PeakWorkingSetSize / (1024**2))
        return float("nan")

    try:
        import resource

        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if platform.system() == "Darwin":
            return value / (1024**2)
        return value / 1024
    except (ImportError, AttributeError):
        return float("nan")


def compressed_model_size_mb(model: object) -> float:
    buffer = io.BytesIO()
    joblib.dump(model, buffer, compress=3)
    return float(len(buffer.getvalue()) / (1024**2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen calibrated benefit gate contingency on six "
            "nested CICIoT2023 public subsample splits."
        )
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46, 47],
    )
    parser.add_argument("--outer-test-frac", type=float, default=0.2)
    parser.add_argument("--base-fraction", type=float, default=0.70)
    parser.add_argument("--gate-train-fraction", type=float, default=0.15)
    parser.add_argument("--gate-select-fraction", type=float, default=0.15)
    parser.add_argument("--majority-cap", type=int, default=25_000)
    parser.add_argument("--benign-cap", type=int, default=80_000)
    parser.add_argument("--global-trees", type=int, default=520)
    parser.add_argument("--local-trees", type=int, default=380)
    parser.add_argument("--rare-lambda", type=float, default=0.04)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Development preflight only. Formal runs leave this unset.",
    )
    parser.add_argument(
        "--group-disjoint",
        action="store_true",
        help=(
            "Assign every exact predictor hash to only one data role. "
            "This is the duplicate safe confirmation mode."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="ijics_v2_ciciot_calibrated_benefit_gate_6split",
    )
    return parser.parse_args()


def normalised_entropy(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, trf.EPS, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1)
    return entropy / np.log(probability.shape[1])


def probability_diagnostics(probability: np.ndarray) -> tuple[np.ndarray, ...]:
    top_two = np.partition(probability, -2, axis=1)[:, -2:]
    top = top_two[:, 1]
    margin = top_two[:, 1] - top_two[:, 0]
    return top, margin, normalised_entropy(probability)


def powered_probability(
    probability: np.ndarray,
    power: float,
) -> np.ndarray:
    if power == 1.0:
        return probability
    powered = np.power(np.clip(probability, trf.EPS, 1.0), power)
    return powered / np.clip(powered.sum(axis=1, keepdims=True), trf.EPS, None)


def diagnostic_matrix(
    flat_probability: np.ndarray,
    taxonomy_probability: np.ndarray,
    family_probability: np.ndarray,
    fine_encoder: LabelEncoder,
    family_encoder: LabelEncoder,
    label_to_family: dict[str, str],
    class_support: np.ndarray,
    rare_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    flat_leaf = np.argmax(flat_probability, axis=1)
    taxonomy_leaf = np.argmax(taxonomy_probability, axis=1)
    family_leaf = np.argmax(family_probability, axis=1)
    fine_family_ids = np.asarray(
        [
            int(family_encoder.transform([label_to_family[str(label)]])[0])
            for label in fine_encoder.classes_
        ],
        dtype=np.int16,
    )
    flat_family = fine_family_ids[flat_leaf]

    flat_top, flat_margin, flat_entropy = probability_diagnostics(
        flat_probability
    )
    taxonomy_top, taxonomy_margin, taxonomy_entropy = probability_diagnostics(
        taxonomy_probability
    )
    family_top, family_margin, family_entropy = probability_diagnostics(
        family_probability
    )
    row_ids = np.arange(len(flat_leaf))
    rare_mask = np.zeros(len(fine_encoder.classes_), dtype=np.float64)
    rare_mask[rare_ids] = 1.0

    matrix = np.column_stack(
        [
            flat_leaf,
            taxonomy_leaf,
            flat_family,
            family_leaf,
            flat_top,
            flat_margin,
            flat_entropy,
            taxonomy_top,
            taxonomy_margin,
            taxonomy_entropy,
            family_top,
            family_margin,
            family_entropy,
            np.abs(flat_probability - taxonomy_probability).sum(axis=1),
            flat_probability[row_ids, taxonomy_leaf],
            taxonomy_probability[row_ids, flat_leaf],
            np.log1p(class_support[flat_leaf]),
            np.log1p(class_support[taxonomy_leaf]),
            (flat_family == family_leaf).astype(np.float64),
            rare_mask[flat_leaf],
            rare_mask[taxonomy_leaf],
        ]
    ).astype(np.float64)
    disagreement = flat_leaf != taxonomy_leaf
    return matrix, disagreement


def utility_targets(
    y_true: np.ndarray,
    flat_probability: np.ndarray,
    taxonomy_probability: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    flat_leaf = np.argmax(flat_probability, axis=1)
    taxonomy_leaf = np.argmax(taxonomy_probability, axis=1)
    disagreement = flat_leaf != taxonomy_leaf
    utility = np.zeros(len(y_true), dtype=np.float64)
    utility[(taxonomy_leaf == y_true) & (flat_leaf != y_true)] = 1.0
    utility[(flat_leaf == y_true) & (taxonomy_leaf != y_true)] = -1.0
    return utility, disagreement


def utility_weights(
    y_true: np.ndarray,
    class_support: np.ndarray,
) -> np.ndarray:
    positive = class_support[class_support > 0]
    maximum = float(np.max(positive))
    weights_by_class = np.sqrt(
        maximum / np.clip(class_support.astype(float), 1.0, None)
    )
    weights_by_class = np.clip(weights_by_class, 1.0, 10.0)
    return weights_by_class[y_true]


def fit_utility_model(
    y_true: np.ndarray,
    flat_probability: np.ndarray,
    taxonomy_probability: np.ndarray,
    family_probability: np.ndarray,
    fine_encoder: LabelEncoder,
    family_encoder: LabelEncoder,
    label_to_family: dict[str, str],
    class_support: np.ndarray,
    rare_ids: np.ndarray,
    seed: int,
) -> tuple[HistGradientBoostingRegressor, dict[str, int]]:
    diagnostics, diagnostic_disagreement = diagnostic_matrix(
        flat_probability,
        taxonomy_probability,
        family_probability,
        fine_encoder,
        family_encoder,
        label_to_family,
        class_support,
        rare_ids,
    )
    utility, target_disagreement = utility_targets(
        y_true,
        flat_probability,
        taxonomy_probability,
    )
    if not np.array_equal(diagnostic_disagreement, target_disagreement):
        raise RuntimeError("Utility disagreement masks do not match.")
    if int(target_disagreement.sum()) < 100:
        raise RuntimeError("Too few disagreement rows to fit the utility model.")

    model = HistGradientBoostingRegressor(
        max_iter=150,
        learning_rate=0.05,
        max_leaf_nodes=15,
        min_samples_leaf=50,
        l2_regularization=0.5,
        categorical_features=[0, 1, 2, 3],
        early_stopping=False,
        random_state=seed,
    )
    model.fit(
        diagnostics[target_disagreement],
        utility[target_disagreement],
        sample_weight=utility_weights(
            y_true[target_disagreement],
            class_support,
        ),
    )
    counts = {
        "disagreement_rows": int(target_disagreement.sum()),
        "taxonomy_benefit_rows": int(
            np.sum(target_disagreement & (utility == 1.0))
        ),
        "taxonomy_harm_rows": int(
            np.sum(target_disagreement & (utility == -1.0))
        ),
        "both_wrong_rows": int(
            np.sum(target_disagreement & (utility == 0.0))
        ),
    }
    return model, counts


def apply_benefit_gate(
    flat_probability: np.ndarray,
    taxonomy_probability: np.ndarray,
    utility_score: np.ndarray,
    disagreement: np.ndarray,
    threshold: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    active = disagreement & (utility_score >= threshold)
    output = flat_probability.copy()
    output[active] += beta * (
        taxonomy_probability[active] - flat_probability[active]
    )
    output /= np.clip(output.sum(axis=1, keepdims=True), trf.EPS, None)
    return output, active


def correction_counts(
    y_true: np.ndarray,
    flat_probability: np.ndarray,
    final_probability: np.ndarray,
    active: np.ndarray,
) -> dict[str, int | float]:
    flat_prediction = np.argmax(flat_probability, axis=1)
    final_prediction = np.argmax(final_probability, axis=1)
    flat_correct = flat_prediction == y_true
    final_correct = final_prediction == y_true
    changed = flat_prediction != final_prediction
    return {
        "gate_active_rows": int(active.sum()),
        "gate_active_fraction": float(np.mean(active)),
        "changed_predictions": int(changed.sum()),
        "corrected_errors": int(np.sum((~flat_correct) & final_correct)),
        "introduced_errors": int(np.sum(flat_correct & (~final_correct))),
        "net_corrected_errors": int(
            np.sum((~flat_correct) & final_correct)
            - np.sum(flat_correct & (~final_correct))
        ),
    }


def select_benefit_gate(
    y_select: np.ndarray,
    flat_probability: np.ndarray,
    taxonomy_probability: np.ndarray,
    family_probability: np.ndarray,
    utility_model: HistGradientBoostingRegressor,
    fine_encoder: LabelEncoder,
    family_encoder: LabelEncoder,
    label_to_family: dict[str, str],
    class_support: np.ndarray,
    rare_ids: np.ndarray,
    rare_lambda: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    diagnostics, disagreement = diagnostic_matrix(
        flat_probability,
        taxonomy_probability,
        family_probability,
        fine_encoder,
        family_encoder,
        label_to_family,
        class_support,
        rare_ids,
    )
    utility_score = utility_model.predict(diagnostics)
    rows: list[dict[str, float]] = []
    for hierarchy_power in (0.75, 1.0):
        powered_taxonomy = powered_probability(
            taxonomy_probability,
            hierarchy_power,
        )
        for threshold in (-0.50, -0.25, 0.0, 0.10, 0.20, 0.30, 0.40, 0.50):
            for beta in (0.25, 0.50, 0.75, 1.0):
                probability, active = apply_benefit_gate(
                    flat_probability,
                    powered_taxonomy,
                    utility_score,
                    disagreement,
                    threshold,
                    beta,
                )
                (
                    objective,
                    macro_f1,
                    rare_recall,
                    worst_recall,
                    accuracy,
                ) = nested.validation_objective(
                    y_select,
                    probability,
                    rare_ids,
                    rare_lambda,
                )
                counts = correction_counts(
                    y_select,
                    flat_probability,
                    probability,
                    active,
                )
                rows.append(
                    {
                        "threshold": threshold,
                        "beta": beta,
                        "hierarchy_power": hierarchy_power,
                        "objective": objective,
                        "f1_macro": macro_f1,
                        "rare_recall": rare_recall,
                        "worst_recall": worst_recall,
                        "accuracy": accuracy,
                        **counts,
                    }
                )
    grid = pd.DataFrame(rows).sort_values(
        [
            "objective",
            "f1_macro",
            "rare_recall",
            "introduced_errors",
            "gate_active_fraction",
        ],
        ascending=[False, False, False, True, True],
    )
    return grid.iloc[0].to_dict(), grid


def evaluate_gate(
    y_true: np.ndarray,
    flat_probability: np.ndarray,
    taxonomy_probability: np.ndarray,
    family_probability: np.ndarray,
    utility_model: HistGradientBoostingRegressor,
    selection: dict[str, float],
    fine_encoder: LabelEncoder,
    family_encoder: LabelEncoder,
    label_to_family: dict[str, str],
    class_support: np.ndarray,
    rare_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float], np.ndarray]:
    diagnostics, disagreement = diagnostic_matrix(
        flat_probability,
        taxonomy_probability,
        family_probability,
        fine_encoder,
        family_encoder,
        label_to_family,
        class_support,
        rare_ids,
    )
    utility_score = utility_model.predict(diagnostics)
    powered_taxonomy = powered_probability(
        taxonomy_probability,
        float(selection["hierarchy_power"]),
    )
    probability, active = apply_benefit_gate(
        flat_probability,
        powered_taxonomy,
        utility_score,
        disagreement,
        float(selection["threshold"]),
        float(selection["beta"]),
    )
    counts = correction_counts(
        y_true,
        flat_probability,
        probability,
        active,
    )
    return probability, active, counts, utility_score


def taxonomy_bundle(
    fit_df: pd.DataFrame,
    features: list[str],
    mapping: dict[str, str],
    local_trees: int,
    real_family_encoder: LabelEncoder | None = None,
) -> tuple[
    object,
    LabelEncoder,
    dict[str, tuple[object | None, LabelEncoder | None, list[str]]],
    float,
]:
    if real_family_encoder is not None:
        family_model, local_models, seconds = nested.fit_real_taxonomy(
            fit_df,
            features,
            real_family_encoder,
            local_trees,
        )
        return family_model, real_family_encoder, local_models, seconds
    return trf.fit_control_hierarchy(
        fit_df,
        features,
        mapping,
        local_trees,
    )


def predict_taxonomy(
    frame: pd.DataFrame,
    features: list[str],
    family_model: object,
    family_encoder: LabelEncoder,
    fine_encoder: LabelEncoder,
    mapping: dict[str, str],
    local_models: dict[
        str,
        tuple[object | None, LabelEncoder | None, list[str]],
    ],
) -> tuple[np.ndarray, np.ndarray]:
    X = trf.clean_features(frame, features)
    return trf.hierarchy_probabilities(
        X,
        family_model,
        family_encoder,
        fine_encoder,
        mapping,
        local_models,
    )


def split_development(
    development_df: pd.DataFrame,
    seed: int,
    base_fraction: float,
    gate_train_fraction: float,
    gate_select_fraction: float,
    group_disjoint: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total = base_fraction + gate_train_fraction + gate_select_fraction
    if not np.isclose(total, 1.0):
        raise ValueError("Development fractions must sum to one.")
    if group_disjoint:
        base_df, gate_df = stratified_group_split(
            development_df,
            train_size=base_fraction,
            seed=seed + 10_000,
        )
    else:
        base_df, gate_df = train_test_split(
            development_df,
            train_size=base_fraction,
            random_state=seed + 10_000,
            stratify=development_df["Label"],
        )
    gate_train_share = gate_train_fraction / (
        gate_train_fraction + gate_select_fraction
    )
    if group_disjoint:
        gate_train_df, gate_select_df = stratified_group_split(
            gate_df,
            train_size=gate_train_share,
            seed=seed + 20_000,
        )
    else:
        gate_train_df, gate_select_df = train_test_split(
            gate_df,
            train_size=gate_train_share,
            random_state=seed + 20_000,
            stratify=gate_df["Label"],
        )
    return base_df, gate_train_df, gate_select_df


def stratified_group_split(
    frame: pd.DataFrame,
    *,
    seed: int,
    train_size: float | None = None,
    test_size: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if FEATURE_GROUP_COLUMN not in frame.columns:
        raise ValueError("Exact predictor groups have not been assigned.")
    group_labels = frame[[FEATURE_GROUP_COLUMN, "Label"]].drop_duplicates()
    conflicts = group_labels.duplicated(
        FEATURE_GROUP_COLUMN,
        keep=False,
    )
    if conflicts.any():
        raise ValueError(
            "An exact predictor group contains more than one fine label."
        )
    left_groups, right_groups = train_test_split(
        group_labels[FEATURE_GROUP_COLUMN].to_numpy(dtype=np.uint64),
        train_size=train_size,
        test_size=test_size,
        random_state=seed,
        stratify=group_labels["Label"].astype(str),
    )
    left_mask = frame[FEATURE_GROUP_COLUMN].isin(left_groups)
    left = frame.loc[left_mask]
    right = frame.loc[~left_mask]
    if len(left) + len(right) != len(frame):
        raise RuntimeError("Group split did not preserve every row.")
    if set(left[FEATURE_GROUP_COLUMN]).intersection(
        set(right[FEATURE_GROUP_COLUMN])
    ):
        raise RuntimeError("An exact predictor group crossed data roles.")
    return left, right


def assign_exact_predictor_groups(
    frame: pd.DataFrame,
    features: list[str],
) -> None:
    hashes = pd.util.hash_pandas_object(
        frame[features],
        index=False,
        categorize=True,
    ).to_numpy(dtype=np.uint64)
    frame[FEATURE_GROUP_COLUMN] = hashes
    group_labels = frame[[FEATURE_GROUP_COLUMN, "Label"]].drop_duplicates()
    if group_labels.duplicated(FEATURE_GROUP_COLUMN, keep=False).any():
        raise ValueError(
            "Exact predictor hashes contain conflicting fine labels."
        )


def group_overlap(left: pd.DataFrame, right: pd.DataFrame) -> int:
    if FEATURE_GROUP_COLUMN not in left.columns:
        return 0
    return int(
        len(
            set(left[FEATURE_GROUP_COLUMN]).intersection(
                set(right[FEATURE_GROUP_COLUMN])
            )
        )
    )


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in (
        "numpy",
        "pandas",
        "scikit-learn",
        "lightgbm",
        "xgboost",
        "scipy",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    args = parse_args()
    out_dir = ROOT / "results" / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and merging the historical public subsample...")
    full = pd.concat(
        [
            pd.read_parquet(nested.TRAIN_PATH),
            pd.read_parquet(nested.TEST_PATH),
        ],
        ignore_index=True,
    )
    if args.max_rows is not None and args.max_rows < len(full):
        full, _ = train_test_split(
            full,
            train_size=args.max_rows,
            random_state=20260729,
            stratify=full["Label"],
        )
        full = full.sort_index().copy()
    features = nested.exp.feature_cols(full)
    if args.group_disjoint:
        print("Assigning exact predictor groups...")
        assign_exact_predictor_groups(full, features)
    fine_encoder = LabelEncoder().fit(full["Label"].astype(str))
    family_encoder = LabelEncoder().fit(full["attack_class"].astype(str))
    labels = list(fine_encoder.classes_)
    label_to_family = (
        full[["Label", "attack_class"]]
        .drop_duplicates()
        .set_index("Label")["attack_class"]
        .astype(str)
        .to_dict()
    )
    cyclic_map, cyclic_shift = trf.cyclic_control_map(label_to_family)

    all_results: list[dict[str, object]] = []
    all_classwise: list[pd.DataFrame] = []
    all_selections: list[dict[str, object]] = []
    all_utility_audit: list[dict[str, object]] = []
    split_audit: list[dict[str, object]] = []
    support_maps: dict[str, dict[str, str]] = {}

    for seed in args.seeds:
        print(f"\nStarting calibrated benefit gate seed {seed}...")
        trf.RANDOM_STATE = seed
        if args.group_disjoint:
            development_df, outer_df = stratified_group_split(
                full,
                test_size=args.outer_test_frac,
                seed=seed,
            )
        else:
            development_df, outer_df = train_test_split(
                full,
                test_size=args.outer_test_frac,
                random_state=seed,
                stratify=full["Label"],
            )
        base_pool_df, gate_train_df, gate_select_df = split_development(
            development_df,
            seed,
            args.base_fraction,
            args.gate_train_fraction,
            args.gate_select_fraction,
            args.group_disjoint,
        )
        fit_df = nested.cap_fit_rows(
            base_pool_df,
            seed,
            args.majority_cap,
            args.benign_cap,
        )
        y_base_pool = fine_encoder.transform(
            base_pool_df["Label"].astype(str)
        )
        class_support = np.bincount(
            y_base_pool,
            minlength=len(fine_encoder.classes_),
        ).astype(np.int64)
        rare_ids = trf.rare_class_ids(y_base_pool)
        support_map, support_audit = (
            support_control.support_matched_wrong_map(
                fit_df,
                label_to_family,
            )
        )
        support_maps[str(seed)] = support_map
        support_audit.insert(0, "seed", seed)
        support_audit.to_csv(
            out_dir / f"support_match_audit_seed_{seed}.csv",
            index=False,
        )
        (out_dir / f"support_matched_map_seed_{seed}.json").write_text(
            json.dumps(support_map, indent=2),
            encoding="utf-8",
        )
        split_audit.append(
            {
                "seed": seed,
                "development_rows": int(len(development_df)),
                "base_pool_rows_before_cap": int(len(base_pool_df)),
                "base_fit_rows_after_cap": int(len(fit_df)),
                "gate_train_rows": int(len(gate_train_df)),
                "gate_select_rows": int(len(gate_select_df)),
                "outer_test_rows": int(len(outer_df)),
                "rare_labels": json.dumps([labels[i] for i in rare_ids]),
                "group_disjoint": bool(args.group_disjoint),
                "base_gate_overlap": group_overlap(
                    base_pool_df,
                    pd.concat([gate_train_df, gate_select_df]),
                ),
                "base_outer_overlap": group_overlap(
                    base_pool_df,
                    outer_df,
                ),
                "gate_outer_overlap": group_overlap(
                    pd.concat([gate_train_df, gate_select_df]),
                    outer_df,
                ),
            }
        )

        print(f"Seed {seed}: fitting the weighted flat predictor...")
        flat_model, flat_seconds = nested.fit_weighted_xgb(
            fit_df,
            features,
            fine_encoder,
            args.global_trees,
        )
        taxonomy_specs: dict[str, dict[str, object]] = {
            REAL_ID: {
                "mapping": label_to_family,
                "real_family_encoder": family_encoder,
            },
            CYCLIC_ID: {
                "mapping": cyclic_map,
                "real_family_encoder": None,
            },
            SUPPORT_ID: {
                "mapping": support_map,
                "real_family_encoder": None,
            },
        }
        fitted_taxonomies: dict[str, dict[str, object]] = {}
        for run_id, spec in taxonomy_specs.items():
            print(f"Seed {seed}: fitting {run_id}...")
            (
                taxonomy_family_model,
                taxonomy_family_encoder,
                taxonomy_local_models,
                taxonomy_seconds,
            ) = taxonomy_bundle(
                fit_df,
                features,
                spec["mapping"],
                args.local_trees,
                spec["real_family_encoder"],
            )
            fitted_taxonomies[run_id] = {
                "family_model": taxonomy_family_model,
                "family_encoder": taxonomy_family_encoder,
                "local_models": taxonomy_local_models,
                "training_seconds": taxonomy_seconds,
                "mapping": spec["mapping"],
            }

        X_gate_train = trf.clean_features(gate_train_df, features)
        X_gate_select = trf.clean_features(gate_select_df, features)
        X_outer = trf.clean_features(outer_df, features)
        y_gate_train = fine_encoder.transform(
            gate_train_df["Label"].astype(str)
        )
        y_gate_select = fine_encoder.transform(
            gate_select_df["Label"].astype(str)
        )
        y_outer = fine_encoder.transform(outer_df["Label"].astype(str))
        flat_gate_train = np.asarray(
            flat_model.predict_proba(X_gate_train),
            dtype=np.float64,
        )
        flat_gate_select = np.asarray(
            flat_model.predict_proba(X_gate_select),
            dtype=np.float64,
        )
        inference_start = time.perf_counter()
        flat_outer = np.asarray(
            flat_model.predict_proba(X_outer),
            dtype=np.float64,
        )
        flat_inference_seconds = time.perf_counter() - inference_start

        probability_map: dict[str, np.ndarray] = {
            "flat_weighted_xgboost": flat_outer,
        }
        gate_state: dict[str, dict[str, object]] = {}
        for run_id, fitted in fitted_taxonomies.items():
            mapping = fitted["mapping"]
            taxonomy_family_encoder = fitted["family_encoder"]
            taxonomy_gate_train, family_gate_train = predict_taxonomy(
                gate_train_df,
                features,
                fitted["family_model"],
                taxonomy_family_encoder,
                fine_encoder,
                mapping,
                fitted["local_models"],
            )
            utility_start = time.perf_counter()
            utility_model, utility_counts = fit_utility_model(
                y_gate_train,
                flat_gate_train,
                taxonomy_gate_train,
                family_gate_train,
                fine_encoder,
                taxonomy_family_encoder,
                mapping,
                class_support,
                rare_ids,
                seed,
            )
            utility_seconds = time.perf_counter() - utility_start
            taxonomy_gate_select, family_gate_select = predict_taxonomy(
                gate_select_df,
                features,
                fitted["family_model"],
                taxonomy_family_encoder,
                fine_encoder,
                mapping,
                fitted["local_models"],
            )
            selection, grid = select_benefit_gate(
                y_gate_select,
                flat_gate_select,
                taxonomy_gate_select,
                family_gate_select,
                utility_model,
                fine_encoder,
                taxonomy_family_encoder,
                mapping,
                class_support,
                rare_ids,
                args.rare_lambda,
            )
            grid.assign(seed=seed, run_id=run_id).to_csv(
                out_dir / f"gate_selection_grid_{run_id}_seed_{seed}.csv",
                index=False,
            )
            all_selections.append(
                {"seed": seed, "run_id": run_id, **selection}
            )
            all_utility_audit.append(
                {
                    "seed": seed,
                    "run_id": run_id,
                    "utility_training_seconds": utility_seconds,
                    **utility_counts,
                }
            )
            inference_start = time.perf_counter()
            taxonomy_outer, family_outer = predict_taxonomy(
                outer_df,
                features,
                fitted["family_model"],
                taxonomy_family_encoder,
                fine_encoder,
                mapping,
                fitted["local_models"],
            )
            taxonomy_inference_seconds = (
                time.perf_counter() - inference_start
            )
            inference_start = time.perf_counter()
            (
                final_outer,
                active_outer,
                correction_audit,
                utility_outer,
            ) = evaluate_gate(
                y_outer,
                flat_outer,
                taxonomy_outer,
                family_outer,
                utility_model,
                selection,
                fine_encoder,
                taxonomy_family_encoder,
                mapping,
                class_support,
                rare_ids,
            )
            gate_inference_seconds = time.perf_counter() - inference_start
            fitted["utility_model"] = utility_model
            fitted["selection"] = selection
            fitted["active_outer"] = active_outer
            fitted["utility_outer"] = utility_outer
            fitted["correction_audit"] = correction_audit
            fitted["utility_training_seconds"] = utility_seconds
            fitted["gate_inference_seconds"] = gate_inference_seconds
            fitted["taxonomy_inference_seconds"] = (
                taxonomy_inference_seconds
            )
            probability_map[run_id] = final_outer
            if run_id == REAL_ID:
                probability_map["soft_taxonomy_factorisation"] = (
                    taxonomy_outer
                )
                probability_map["hard_top_down_taxonomy"] = (
                    nested.hard_hierarchy_probability(
                        taxonomy_outer,
                        family_outer,
                        fine_encoder,
                        taxonomy_family_encoder,
                        mapping,
                    )
                )
                probability_map["equal_flat_taxonomy_blend"] = (
                    0.5 * flat_outer + 0.5 * taxonomy_outer
                )

        flat_artifact = {
            "flat_model": flat_model,
            "fine_encoder": fine_encoder,
            "features": features,
        }
        flat_model_size_mb = compressed_model_size_mb(flat_artifact)
        for run_id, fitted in fitted_taxonomies.items():
            taxonomy_artifact = {
                "family_model": fitted["family_model"],
                "family_encoder": fitted["family_encoder"],
                "local_models": fitted["local_models"],
                "fine_encoder": fine_encoder,
                "mapping": fitted["mapping"],
                "features": features,
            }
            fitted["taxonomy_model_size_mb"] = compressed_model_size_mb(
                taxonomy_artifact
            )
            fitted["blend_model_size_mb"] = compressed_model_size_mb(
                {
                    **flat_artifact,
                    **taxonomy_artifact,
                }
            )
            fitted["gate_model_size_mb"] = compressed_model_size_mb(
                {
                    **flat_artifact,
                    **taxonomy_artifact,
                    "utility_model": fitted["utility_model"],
                    "selection": fitted["selection"],
                    "class_support": class_support,
                    "rare_ids": rare_ids,
                }
            )
        peak_rss_mb = process_peak_rss_mb()

        prediction_store: dict[str, np.ndarray] = {
            "row_id": outer_df.index.to_numpy(dtype=np.int64),
            "y_true": y_outer.astype(np.int16),
            "rare_ids": rare_ids.astype(np.int16),
            "flat_weighted_xgboost_probability": flat_outer.astype(np.float16),
        }
        for run_id, probability in probability_map.items():
            row = trf.metric_row(run_id, y_outer, probability, rare_ids)
            training_seconds = flat_seconds
            inference_seconds = flat_inference_seconds
            model_size_mb = flat_model_size_mb
            selected_values: dict[str, object] = {}
            if run_id in fitted_taxonomies:
                fitted = fitted_taxonomies[run_id]
                training_seconds += float(fitted["training_seconds"])
                training_seconds += float(fitted["utility_training_seconds"])
                inference_seconds += float(
                    fitted["taxonomy_inference_seconds"]
                )
                inference_seconds += float(fitted["gate_inference_seconds"])
                model_size_mb = float(fitted["gate_model_size_mb"])
                selected_values = {
                    "threshold": fitted["selection"]["threshold"],
                    "beta": fitted["selection"]["beta"],
                    "hierarchy_power": fitted["selection"][
                        "hierarchy_power"
                    ],
                    **fitted["correction_audit"],
                }
                prediction_store[f"{run_id}_active"] = fitted[
                    "active_outer"
                ].astype(np.uint8)
                prediction_store[f"{run_id}_utility"] = fitted[
                    "utility_outer"
                ].astype(np.float32)
            elif run_id == "equal_flat_taxonomy_blend":
                fitted = fitted_taxonomies[REAL_ID]
                training_seconds += float(fitted["training_seconds"])
                inference_seconds += float(
                    fitted["taxonomy_inference_seconds"]
                )
                model_size_mb = float(fitted["blend_model_size_mb"])
            elif run_id in {
                "soft_taxonomy_factorisation",
                "hard_top_down_taxonomy",
            }:
                fitted = fitted_taxonomies[REAL_ID]
                training_seconds = float(fitted["training_seconds"])
                inference_seconds = float(
                    fitted["taxonomy_inference_seconds"]
                )
                model_size_mb = float(fitted["taxonomy_model_size_mb"])
            row.update(
                {
                    "dataset": "CICIoT2023 public subsample",
                    "source_role": "third party development mirror",
                    "seed": seed,
                    "split_id": f"outer_seed_{seed}",
                    "n_test": int(len(outer_df)),
                    "base_fit_rows": int(len(fit_df)),
                    "gate_train_rows": int(len(gate_train_df)),
                    "gate_select_rows": int(len(gate_select_df)),
                    "outer_test_rows": int(len(outer_df)),
                    "n_features": len(features),
                    "training_seconds": training_seconds,
                    "inference_seconds": inference_seconds,
                    "inference_ms_per_10000": (
                        inference_seconds * 1000.0 * 10_000.0 / len(outer_df)
                    ),
                    "model_size_mb": model_size_mb,
                    "peak_rss_mb": peak_rss_mb,
                    "threshold": np.nan,
                    "beta": np.nan,
                    "hierarchy_power": np.nan,
                    "gate_active_rows": np.nan,
                    "gate_active_fraction": np.nan,
                    "changed_predictions": np.nan,
                    "corrected_errors": np.nan,
                    "introduced_errors": np.nan,
                    "net_corrected_errors": np.nan,
                }
            )
            row.update(selected_values)
            all_results.append(row)
            all_classwise.append(
                nested.classwise_frame(
                    seed,
                    run_id,
                    y_outer,
                    probability,
                    labels,
                    rare_ids,
                )
            )
            prediction_store[f"{run_id}_probability"] = probability.astype(
                np.float16
            )
            prediction_store[run_id] = np.argmax(
                probability,
                axis=1,
            ).astype(np.int16)

        np.savez_compressed(
            out_dir / f"paired_predictions_seed_{seed}.npz",
            **prediction_store,
        )
        pd.DataFrame(all_results).to_csv(
            out_dir / "nested_split_results_partial.csv",
            index=False,
        )
        pd.concat(all_classwise, ignore_index=True).to_csv(
            out_dir / "nested_split_classwise_metrics_partial.csv",
            index=False,
        )
        pd.DataFrame(all_selections).to_csv(
            out_dir / "gate_selections_partial.csv",
            index=False,
        )
        pd.DataFrame(all_utility_audit).to_csv(
            out_dir / "utility_training_audit_partial.csv",
            index=False,
        )
        print(
            pd.DataFrame(
                [row for row in all_results if int(row["seed"]) == seed]
            )[
                [
                    "run_id",
                    "accuracy",
                    "f1_macro",
                    "rare_recall",
                    "worst_recall",
                ]
            ]
            .sort_values("f1_macro", ascending=False)
            .to_string(index=False)
        )

    results = pd.DataFrame(all_results)
    results.to_csv(out_dir / "nested_split_results.csv", index=False)
    pd.concat(all_classwise, ignore_index=True).to_csv(
        out_dir / "nested_split_classwise_metrics.csv",
        index=False,
    )
    pd.DataFrame(all_selections).to_csv(
        out_dir / "gate_selections.csv",
        index=False,
    )
    pd.DataFrame(all_utility_audit).to_csv(
        out_dir / "utility_training_audit.csv",
        index=False,
    )
    pd.DataFrame(split_audit).to_csv(
        out_dir / "nested_split_audit.csv",
        index=False,
    )
    aggregate = (
        results.groupby("run_id")
        .agg(
            splits=("seed", "nunique"),
            macro_f1_mean=("f1_macro", "mean"),
            macro_f1_sd=("f1_macro", "std"),
            rare_recall_mean=("rare_recall", "mean"),
            rare_recall_sd=("rare_recall", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_sd=("accuracy", "std"),
            pr_auc_macro_mean=("pr_auc_macro", "mean"),
            worst_recall_mean=("worst_recall", "mean"),
            gate_active_fraction_mean=("gate_active_fraction", "mean"),
            corrected_errors_mean=("corrected_errors", "mean"),
            introduced_errors_mean=("introduced_errors", "mean"),
        )
        .reset_index()
        .sort_values("macro_f1_mean", ascending=False)
    )
    aggregate.to_csv(out_dir / "nested_split_aggregate.csv", index=False)
    metadata = {
        "purpose": "frozen calibrated benefit gate contingency",
        "design_lock": (
            "docs/jit_20260729_v1/DUPLICATE_CONTROL_DECISION.md"
        ),
        "dataset": "CICIoT2023 public subsample",
        "source_role": "third party development mirror",
        "historical_public_subsample_role": "development evidence only",
        "resource_preflight": args.max_rows is not None,
        "merged_rows": int(len(full)),
        "features": features,
        "fine_labels": labels,
        "label_to_family": label_to_family,
        "cyclic_control_map": cyclic_map,
        "cyclic_control_shift": cyclic_shift,
        "support_maps_by_seed": support_maps,
        "arguments": vars(args),
        "command": " ".join(sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "package_versions": package_versions(),
        "utility_model": {
            "type": "HistGradientBoostingRegressor",
            "max_iter": 150,
            "learning_rate": 0.05,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 50,
            "l2_regularization": 0.5,
            "categorical_features": [0, 1, 2, 3],
            "traffic_features_used": False,
        },
        "threshold_grid": [-0.50, -0.25, 0.0, 0.10, 0.20, 0.30, 0.40, 0.50],
        "beta_grid": [0.25, 0.50, 0.75, 1.0],
        "taxonomy_probability_power_grid": [0.75, 1.0],
        "selection_rule": "gate selection partition only",
        "outer_test_role": "not used for fitting or parameter selection",
    }
    (out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print("\nAggregate results")
    print(aggregate.to_string(index=False))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
