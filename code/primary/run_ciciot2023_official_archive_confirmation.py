from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import prepare_ciciot2023_official_confirmation_samples as preparation
import run_ciciot_agreement_gated_trf_confirmation as nested
import run_ciciot_calibrated_benefit_gate_confirmation as benefit
import run_ciciot_support_matched_control_v2 as support_control
import run_ciciot_taxonomy_residual_fusion as taxonomy


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "data"
    / "processed"
    / "ciciot2023_official_validated"
    / "MERGED_CSV"
)
DEFAULT_MANIFEST = (
    ROOT
    / "results"
    / "jit_ciciot_official_archive_audit"
    / "official_member_manifest.csv"
)
DEFAULT_SAMPLES = (
    ROOT
    / "results"
    / "jit_ciciot2023_official_confirmation_samples"
)
DEFAULT_DUPLICATE_AUDIT = (
    ROOT
    / "results"
    / "jit_ciciot_official_split_duplicate_audit"
)
REAL_ID = benefit.REAL_ID
CYCLIC_ID = benefit.CYCLIC_ID
SUPPORT_ID = benefit.SUPPORT_ID
FLAT_ID = "flat_weighted_xgboost"
SOFT_ID = "soft_taxonomy_factorisation"
HARD_ID = "hard_top_down_taxonomy"
EQUAL_ID = "equal_flat_taxonomy_blend"
METHODS = (
    FLAT_ID,
    REAL_ID,
    CYCLIC_ID,
    SUPPORT_ID,
    SOFT_ID,
    HARD_ID,
    EQUAL_ID,
)
PAIRED_CONTRASTS = (
    (EQUAL_ID, FLAT_ID),
    (REAL_ID, FLAT_ID),
    (REAL_ID, EQUAL_ID),
    (EQUAL_ID, CYCLIC_ID),
    (EQUAL_ID, SUPPORT_ID),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen CICIoT2023 confirmation on the official "
            "source file partition."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--samples-dir", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument(
        "--duplicate-audit-dir",
        type=Path,
        default=DEFAULT_DUPLICATE_AUDIT,
        help=(
            "Completed official split duplicate audit used to construct "
            "the duplicate free final test sensitivity."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--global-trees", type=int, default=520)
    parser.add_argument("--local-trees", type=int, default=380)
    parser.add_argument("--rare-lambda", type=float, default=0.04)
    parser.add_argument(
        "--max-test-rows",
        type=int,
        default=None,
        help="Development preflight only. Formal runs leave this unset.",
    )
    parser.add_argument(
        "--out-dir",
        default="jit_ciciot2023_official_archive_confirmation",
    )
    return parser.parse_args()


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


def public_label_to_family() -> dict[str, str]:
    frame = pd.concat(
        [
            pd.read_parquet(
                nested.TRAIN_PATH,
                columns=["Label", "attack_class"],
            ),
            pd.read_parquet(
                nested.TEST_PATH,
                columns=["Label", "attack_class"],
            ),
        ],
        ignore_index=True,
    )
    pairs = frame.drop_duplicates()
    if pairs["Label"].duplicated().any():
        raise ValueError("A public fine label maps to multiple families.")
    return (
        pairs.set_index("Label")["attack_class"].astype(str).to_dict()
    )


def add_family(
    frame: pd.DataFrame,
    label_to_family: dict[str, str],
) -> pd.DataFrame:
    output = frame.copy()
    output["attack_class"] = output["Label"].map(label_to_family)
    if output["attack_class"].isna().any():
        missing = sorted(
            output.loc[output["attack_class"].isna(), "Label"]
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError(f"Missing attack family labels: {missing}")
    return output


def rare_ids_from_counts(
    counts: pd.DataFrame,
    fine_encoder: LabelEncoder,
    fraction: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, int]:
    base = counts.loc[counts["role"] == "base"].copy()
    available = base.set_index("label")["source_rows"].astype(int)
    expected = set(map(str, fine_encoder.classes_))
    if set(available.index.astype(str)) != expected:
        raise ValueError("The base source counts do not cover all labels.")
    support = np.asarray(
        [available.loc[str(label)] for label in fine_encoder.classes_],
        dtype=np.int64,
    )
    threshold = max(1, int(np.ceil(support.sum() * fraction)))
    selected = np.flatnonzero((support > 0) & (support <= threshold))
    if not len(selected):
        present = np.flatnonzero(support > 0)
        selected = present[np.argsort(support[present])][
            : min(8, len(present))
        ]
    return selected.astype(np.int16), support, threshold


def confusion_metrics(
    matrix: np.ndarray,
    rare_ids: np.ndarray,
) -> tuple[dict[str, float], pd.DataFrame]:
    matrix = np.asarray(matrix, dtype=np.int64)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = np.diag(matrix)
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=predicted > 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=support > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    total = int(matrix.sum())
    metrics = {
        "n_test": total,
        "accuracy": float(true_positive.sum() / total),
        "f1_macro": float(f1.mean()),
        "recall_macro": float(recall.mean()),
        "rare_recall": float(recall[rare_ids].mean()),
        "worst_recall": float(recall.min()),
    }
    classwise = pd.DataFrame(
        {
            "class_id": np.arange(len(support)),
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )
    return metrics, classwise


def update_confusion(
    matrix: np.ndarray,
    y_true: np.ndarray,
    prediction: np.ndarray,
) -> None:
    np.add.at(matrix, (y_true, prediction), 1)


def update_joint_counts(
    tensor: np.ndarray,
    y_true: np.ndarray,
    primary_prediction: np.ndarray,
    comparator_prediction: np.ndarray,
) -> None:
    np.add.at(
        tensor,
        (y_true, primary_prediction, comparator_prediction),
        1,
    )


def update_transitions(
    record: dict[str, np.ndarray | int],
    y_true: np.ndarray,
    flat_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
) -> None:
    corrected = (flat_prediction != y_true) & (
        candidate_prediction == y_true
    )
    introduced = (flat_prediction == y_true) & (
        candidate_prediction != y_true
    )
    both_correct = (flat_prediction == y_true) & (
        candidate_prediction == y_true
    )
    both_wrong = (flat_prediction != y_true) & (
        candidate_prediction != y_true
    )
    np.add.at(record["corrected_by_class"], y_true[corrected], 1)
    np.add.at(record["introduced_by_class"], y_true[introduced], 1)
    record["corrected"] += int(corrected.sum())
    record["introduced"] += int(introduced.sum())
    record["both_correct"] += int(both_correct.sum())
    record["both_wrong"] += int(both_wrong.sum())


def probability_prediction(probability: np.ndarray) -> np.ndarray:
    return np.argmax(probability, axis=1).astype(np.int16)


def membership_mask(
    values: np.ndarray,
    sorted_reference: np.ndarray,
) -> np.ndarray:
    positions = np.searchsorted(sorted_reference, values)
    valid = positions < len(sorted_reference)
    matched = np.zeros(len(values), dtype=bool)
    matched[valid] = (
        sorted_reference[positions[valid]] == values[valid]
    )
    return matched


def canonicalise_labels(
    frame: pd.DataFrame,
    canonical: dict[str, str],
) -> pd.DataFrame:
    output = frame.copy()
    mapped = output["Label"].astype(str).map(
        lambda value: canonical.get(preparation.canonical_key(value))
    )
    if mapped.isna().any():
        missing = sorted(
            output.loc[mapped.isna(), "Label"].astype(str).unique()
        )
        raise ValueError(f"Unmapped official labels: {missing}")
    output["Label"] = mapped.to_numpy()
    return output


def partial_results(
    confusions: dict[str, np.ndarray],
    rare_ids: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_id, matrix in confusions.items():
        if matrix.sum() == 0:
            continue
        metrics, _ = confusion_metrics(matrix, rare_ids)
        rows.append({"seed": seed, "run_id": run_id, **metrics})
    return pd.DataFrame(rows)


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    args = parse_args()
    source = args.source.resolve()
    manifest_path = args.manifest.resolve()
    samples_dir = args.samples_dir.resolve()
    duplicate_audit_dir = args.duplicate_audit_dir.resolve()
    out_dir = ROOT / "results" / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for required in (
        source,
        manifest_path,
        samples_dir / "base.parquet",
        samples_dir / "gate_train.parquet",
        samples_dir / "gate_select.parquet",
        samples_dir / "selected_label_counts.csv",
        samples_dir / "sample_metadata.json",
        duplicate_audit_dir / "official_split_duplicate_audit.json",
        duplicate_audit_dir / "base_unique_feature_hashes.npy",
        duplicate_audit_dir / "gate_train_unique_feature_hashes.npy",
        duplicate_audit_dir / "gate_select_unique_feature_hashes.npy",
        duplicate_audit_dir / "global_conflicting_feature_hashes.npy",
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    manifest = pd.read_csv(manifest_path)
    manifest["sequence"] = manifest["member"].map(
        preparation.sequence_from_member
    )
    if len(manifest) != 63 or set(manifest["status"]) != {"complete"}:
        raise ValueError("A complete official archive manifest is required.")
    if sorted(manifest["sequence"]) != list(range(1, 64)):
        raise ValueError("The official archive sequence is incomplete.")

    sample_metadata = json.loads(
        (samples_dir / "sample_metadata.json").read_text(encoding="utf-8")
    )
    development_feature_hashes = [
        np.load(
            duplicate_audit_dir / f"{role}_unique_feature_hashes.npy",
            mmap_mode="r",
        )
        for role in ("base", "gate_train", "gate_select")
    ]
    conflicting_feature_hashes = np.load(
        duplicate_audit_dir / "global_conflicting_feature_hashes.npy",
        mmap_mode="r",
    )
    features = [str(value) for value in sample_metadata["feature_columns"]]
    base_df = pd.read_parquet(samples_dir / "base.parquet")
    gate_train_df = pd.read_parquet(samples_dir / "gate_train.parquet")
    gate_select_df = pd.read_parquet(samples_dir / "gate_select.parquet")
    label_to_family = public_label_to_family()
    base_df = add_family(base_df, label_to_family)
    gate_train_df = add_family(gate_train_df, label_to_family)
    gate_select_df = add_family(gate_select_df, label_to_family)

    fine_encoder = LabelEncoder().fit(sorted(label_to_family))
    family_encoder = LabelEncoder().fit(
        sorted(set(label_to_family.values()))
    )
    labels = list(map(str, fine_encoder.classes_))
    selected_counts = pd.read_csv(
        samples_dir / "selected_label_counts.csv"
    )
    rare_ids, class_support, rare_threshold = rare_ids_from_counts(
        selected_counts,
        fine_encoder,
    )
    cyclic_map, cyclic_shift = taxonomy.cyclic_control_map(label_to_family)
    support_map, support_audit = (
        support_control.support_matched_wrong_map(
            base_df,
            label_to_family,
        )
    )
    support_audit.insert(0, "seed", args.seed)
    support_audit.to_csv(
        out_dir / "support_match_audit.csv",
        index=False,
    )
    (out_dir / "support_matched_map.json").write_text(
        json.dumps(support_map, indent=2),
        encoding="utf-8",
    )

    taxonomy.RANDOM_STATE = args.seed
    print("Fitting the official archive flat predictor...")
    flat_model, flat_training_seconds = nested.fit_weighted_xgb(
        base_df,
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
    fitted: dict[str, dict[str, object]] = {}
    for run_id, spec in taxonomy_specs.items():
        print(f"Fitting official archive method {run_id}...")
        (
            family_model,
            fitted_family_encoder,
            local_models,
            taxonomy_seconds,
        ) = benefit.taxonomy_bundle(
            base_df,
            features,
            spec["mapping"],
            args.local_trees,
            spec["real_family_encoder"],
        )
        fitted[run_id] = {
            "family_model": family_model,
            "family_encoder": fitted_family_encoder,
            "local_models": local_models,
            "mapping": spec["mapping"],
            "taxonomy_training_seconds": taxonomy_seconds,
        }

    X_gate_train = taxonomy.clean_features(gate_train_df, features)
    X_gate_select = taxonomy.clean_features(gate_select_df, features)
    y_gate_train = fine_encoder.transform(
        gate_train_df["Label"].astype(str)
    )
    y_gate_select = fine_encoder.transform(
        gate_select_df["Label"].astype(str)
    )
    flat_gate_train = np.asarray(
        flat_model.predict_proba(X_gate_train),
        dtype=np.float64,
    )
    flat_gate_select = np.asarray(
        flat_model.predict_proba(X_gate_select),
        dtype=np.float64,
    )
    selection_rows: list[dict[str, object]] = []
    utility_rows: list[dict[str, object]] = []
    for run_id, state in fitted.items():
        mapping = state["mapping"]
        taxonomy_gate_train, family_gate_train = (
            taxonomy.hierarchy_probabilities(
                X_gate_train,
                state["family_model"],
                state["family_encoder"],
                fine_encoder,
                mapping,
                state["local_models"],
            )
        )
        utility_start = time.perf_counter()
        utility_model, utility_counts = benefit.fit_utility_model(
            y_gate_train,
            flat_gate_train,
            taxonomy_gate_train,
            family_gate_train,
            fine_encoder,
            state["family_encoder"],
            mapping,
            class_support,
            rare_ids,
            args.seed,
        )
        utility_seconds = time.perf_counter() - utility_start
        taxonomy_gate_select, family_gate_select = (
            taxonomy.hierarchy_probabilities(
                X_gate_select,
                state["family_model"],
                state["family_encoder"],
                fine_encoder,
                mapping,
                state["local_models"],
            )
        )
        selection, grid = benefit.select_benefit_gate(
            y_gate_select,
            flat_gate_select,
            taxonomy_gate_select,
            family_gate_select,
            utility_model,
            fine_encoder,
            state["family_encoder"],
            mapping,
            class_support,
            rare_ids,
            args.rare_lambda,
        )
        grid.assign(seed=args.seed, run_id=run_id).to_csv(
            out_dir / f"gate_selection_grid_{run_id}.csv",
            index=False,
        )
        state["utility_model"] = utility_model
        state["utility_training_seconds"] = utility_seconds
        state["selection"] = selection
        selection_rows.append(
            {"seed": args.seed, "run_id": run_id, **selection}
        )
        utility_rows.append(
            {
                "seed": args.seed,
                "run_id": run_id,
                "utility_training_seconds": utility_seconds,
                **utility_counts,
            }
        )
    pd.DataFrame(selection_rows).to_csv(
        out_dir / "gate_selections.csv",
        index=False,
    )
    pd.DataFrame(utility_rows).to_csv(
        out_dir / "utility_training_audit.csv",
        index=False,
    )

    classes = len(labels)
    confusions = {
        run_id: np.zeros((classes, classes), dtype=np.int64)
        for run_id in METHODS
    }
    duplicate_free_confusions = {
        run_id: np.zeros((classes, classes), dtype=np.int64)
        for run_id in METHODS
    }
    paired_joint_counts = {
        (primary, comparator): np.zeros(
            (classes, classes, classes),
            dtype=np.int64,
        )
        for primary, comparator in PAIRED_CONTRASTS
    }
    duplicate_free_paired_joint_counts = {
        (primary, comparator): np.zeros(
            (classes, classes, classes),
            dtype=np.int64,
        )
        for primary, comparator in PAIRED_CONTRASTS
    }
    transitions = {
        run_id: {
            "corrected": 0,
            "introduced": 0,
            "both_correct": 0,
            "both_wrong": 0,
            "corrected_by_class": np.zeros(classes, dtype=np.int64),
            "introduced_by_class": np.zeros(classes, dtype=np.int64),
        }
        for run_id in METHODS
        if run_id != FLAT_ID
    }
    duplicate_free_transitions = {
        run_id: {
            "corrected": 0,
            "introduced": 0,
            "both_correct": 0,
            "both_wrong": 0,
            "corrected_by_class": np.zeros(classes, dtype=np.int64),
            "introduced_by_class": np.zeros(classes, dtype=np.int64),
        }
        for run_id in METHODS
        if run_id != FLAT_ID
    }
    inference_seconds: defaultdict[str, float] = defaultdict(float)
    active_rows: defaultdict[str, int] = defaultdict(int)
    canonical = preparation.canonical_label_map()
    expected_columns = features + ["Label"]
    processed_rows = 0
    duplicate_crossed_rows = 0
    conflicting_label_rows = 0
    repeated_final_rows = 0
    seen_final_hashes: set[int] = set()
    final_members = manifest.loc[
        manifest["sequence"].between(55, 63)
    ].sort_values("sequence")

    for _, member_row in final_members.iterrows():
        member = str(member_row["member"])
        path = source / Path(member).name
        member_rows = 0
        print(f"Predicting complete official test member {member}...")
        for chunk in pd.read_csv(
            path,
            chunksize=args.chunk_size,
            low_memory=False,
        ):
            if [str(column) for column in chunk.columns] != expected_columns:
                raise ValueError(f"Unexpected schema in {member}.")
            if args.max_test_rows is not None:
                remaining = args.max_test_rows - processed_rows
                if remaining <= 0:
                    break
                if len(chunk) > remaining:
                    chunk = chunk.iloc[:remaining].copy()
            chunk = canonicalise_labels(chunk, canonical)
            chunk = add_family(chunk, label_to_family)
            y_true = fine_encoder.transform(chunk["Label"].astype(str))
            X = taxonomy.clean_features(chunk, features)
            feature_hash = pd.util.hash_pandas_object(
                chunk[features],
                index=False,
                categorize=True,
            ).to_numpy(dtype=np.uint64)
            crossed = np.zeros(len(chunk), dtype=bool)
            for reference in development_feature_hashes:
                crossed |= membership_mask(feature_hash, reference)
            conflict = membership_mask(
                feature_hash,
                conflicting_feature_hashes,
            )
            first_final_occurrence = np.zeros(len(chunk), dtype=bool)
            for index, value in enumerate(feature_hash):
                key = int(value)
                if key not in seen_final_hashes:
                    seen_final_hashes.add(key)
                    first_final_occurrence[index] = True
            duplicate_free = (
                ~crossed
                & ~conflict
                & first_final_occurrence
            )
            duplicate_crossed_rows += int(crossed.sum())
            conflicting_label_rows += int(conflict.sum())
            repeated_final_rows += int((~first_final_occurrence).sum())

            start = time.perf_counter()
            flat_probability = np.asarray(
                flat_model.predict_proba(X),
                dtype=np.float64,
            )
            inference_seconds[FLAT_ID] += time.perf_counter() - start
            flat_prediction = probability_prediction(flat_probability)
            update_confusion(
                confusions[FLAT_ID],
                y_true,
                flat_prediction,
            )
            update_confusion(
                duplicate_free_confusions[FLAT_ID],
                y_true[duplicate_free],
                flat_prediction[duplicate_free],
            )
            chunk_predictions: dict[str, np.ndarray] = {
                FLAT_ID: flat_prediction,
            }

            for run_id, state in fitted.items():
                start = time.perf_counter()
                taxonomy_probability, family_probability = (
                    taxonomy.hierarchy_probabilities(
                        X,
                        state["family_model"],
                        state["family_encoder"],
                        fine_encoder,
                        state["mapping"],
                        state["local_models"],
                    )
                )
                taxonomy_seconds = time.perf_counter() - start
                start = time.perf_counter()
                (
                    gate_probability,
                    active,
                    _,
                    _,
                ) = benefit.evaluate_gate(
                    y_true,
                    flat_probability,
                    taxonomy_probability,
                    family_probability,
                    state["utility_model"],
                    state["selection"],
                    fine_encoder,
                    state["family_encoder"],
                    state["mapping"],
                    class_support,
                    rare_ids,
                )
                gate_seconds = time.perf_counter() - start
                inference_seconds[run_id] += (
                    taxonomy_seconds + gate_seconds
                )
                active_rows[run_id] += int(active.sum())
                gate_prediction = probability_prediction(gate_probability)
                chunk_predictions[run_id] = gate_prediction
                update_confusion(
                    confusions[run_id],
                    y_true,
                    gate_prediction,
                )
                update_confusion(
                    duplicate_free_confusions[run_id],
                    y_true[duplicate_free],
                    gate_prediction[duplicate_free],
                )
                update_transitions(
                    transitions[run_id],
                    y_true,
                    flat_prediction,
                    gate_prediction,
                )
                update_transitions(
                    duplicate_free_transitions[run_id],
                    y_true[duplicate_free],
                    flat_prediction[duplicate_free],
                    gate_prediction[duplicate_free],
                )

                if run_id == REAL_ID:
                    soft_prediction = probability_prediction(
                        taxonomy_probability
                    )
                    hard_probability = nested.hard_hierarchy_probability(
                        taxonomy_probability,
                        family_probability,
                        fine_encoder,
                        state["family_encoder"],
                        state["mapping"],
                    )
                    hard_prediction = probability_prediction(
                        hard_probability
                    )
                    equal_prediction = probability_prediction(
                        0.5 * flat_probability
                        + 0.5 * taxonomy_probability
                    )
                    for method, prediction in (
                        (SOFT_ID, soft_prediction),
                        (HARD_ID, hard_prediction),
                        (EQUAL_ID, equal_prediction),
                    ):
                        chunk_predictions[method] = prediction
                        update_confusion(
                            confusions[method],
                            y_true,
                            prediction,
                        )
                        update_confusion(
                            duplicate_free_confusions[method],
                            y_true[duplicate_free],
                            prediction[duplicate_free],
                        )
                        update_transitions(
                            transitions[method],
                            y_true,
                            flat_prediction,
                            prediction,
                        )
                        update_transitions(
                            duplicate_free_transitions[method],
                            y_true[duplicate_free],
                            flat_prediction[duplicate_free],
                            prediction[duplicate_free],
                        )
                    inference_seconds[SOFT_ID] += taxonomy_seconds
                    inference_seconds[HARD_ID] += taxonomy_seconds
                    inference_seconds[EQUAL_ID] += taxonomy_seconds

            for primary, comparator in PAIRED_CONTRASTS:
                update_joint_counts(
                    paired_joint_counts[(primary, comparator)],
                    y_true,
                    chunk_predictions[primary],
                    chunk_predictions[comparator],
                )
                update_joint_counts(
                    duplicate_free_paired_joint_counts[
                        (primary, comparator)
                    ],
                    y_true[duplicate_free],
                    chunk_predictions[primary][duplicate_free],
                    chunk_predictions[comparator][duplicate_free],
                )

            processed_rows += len(chunk)
            member_rows += len(chunk)
            if (
                args.max_test_rows is not None
                and processed_rows >= args.max_test_rows
            ):
                break
        if args.max_test_rows is None and member_rows != int(
            member_row["rows"]
        ):
            raise ValueError(
                f"Row count changed for {member}: "
                f"{member_rows} != {int(member_row['rows'])}"
            )
        partial_results(
            confusions,
            rare_ids,
            args.seed,
        ).to_csv(
            out_dir / "official_confirmation_results_partial.csv",
            index=False,
        )
        if (
            args.max_test_rows is not None
            and processed_rows >= args.max_test_rows
        ):
            break

    flat_artifact = {
        "flat_model": flat_model,
        "fine_encoder": fine_encoder,
        "features": features,
    }
    flat_size = benefit.compressed_model_size_mb(flat_artifact)
    real_state = fitted[REAL_ID]
    model_sizes: dict[str, float] = {FLAT_ID: flat_size}
    training_costs: dict[str, float] = {
        FLAT_ID: flat_training_seconds,
    }
    for run_id, state in fitted.items():
        taxonomy_artifact = {
            "family_model": state["family_model"],
            "family_encoder": state["family_encoder"],
            "local_models": state["local_models"],
            "fine_encoder": fine_encoder,
            "mapping": state["mapping"],
            "features": features,
        }
        taxonomy_size = benefit.compressed_model_size_mb(
            taxonomy_artifact
        )
        gate_size = benefit.compressed_model_size_mb(
            {
                **flat_artifact,
                **taxonomy_artifact,
                "utility_model": state["utility_model"],
                "selection": state["selection"],
                "class_support": class_support,
                "rare_ids": rare_ids,
            }
        )
        model_sizes[run_id] = gate_size
        training_costs[run_id] = float(
            flat_training_seconds
            + state["taxonomy_training_seconds"]
            + state["utility_training_seconds"]
        )
        state["taxonomy_size"] = taxonomy_size

    real_taxonomy_cost = float(
        real_state["taxonomy_training_seconds"]
    )
    real_taxonomy_size = float(real_state["taxonomy_size"])
    model_sizes[SOFT_ID] = real_taxonomy_size
    model_sizes[HARD_ID] = real_taxonomy_size
    model_sizes[EQUAL_ID] = benefit.compressed_model_size_mb(
        {
            **flat_artifact,
            "family_model": real_state["family_model"],
            "family_encoder": real_state["family_encoder"],
            "local_models": real_state["local_models"],
            "mapping": real_state["mapping"],
        }
    )
    training_costs[SOFT_ID] = real_taxonomy_cost
    training_costs[HARD_ID] = real_taxonomy_cost
    training_costs[EQUAL_ID] = (
        flat_training_seconds + real_taxonomy_cost
    )

    peak_rss = benefit.process_peak_rss_mb()
    result_rows: list[dict[str, object]] = []
    classwise_rows: list[pd.DataFrame] = []
    reported_inference_seconds: dict[str, float] = {}
    for run_id in METHODS:
        metrics, classwise = confusion_metrics(
            confusions[run_id],
            rare_ids,
        )
        method_inference = inference_seconds[run_id]
        if run_id in {REAL_ID, CYCLIC_ID, SUPPORT_ID, EQUAL_ID}:
            method_inference += inference_seconds[FLAT_ID]
        reported_inference_seconds[run_id] = method_inference
        result_rows.append(
            {
                "dataset": "CICIoT2023 official archive",
                "source_role": "official_archive_confirmation",
                "seed": args.seed,
                "split_id": "official_file_group_01_63",
                "run_id": run_id,
                **metrics,
                "pr_auc_macro": np.nan,
                "training_seconds": training_costs[run_id],
                "inference_seconds": method_inference,
                "inference_ms_per_10000": (
                    method_inference * 1000.0 * 10_000.0
                    / metrics["n_test"]
                ),
                "model_size_mb": model_sizes[run_id],
                "peak_rss_mb": peak_rss,
                "active_rows": active_rows.get(run_id, 0),
            }
        )
        classwise.insert(0, "run_id", run_id)
        classwise.insert(0, "split_id", "official_file_group_01_63")
        classwise.insert(0, "seed", args.seed)
        classwise.insert(0, "dataset", "CICIoT2023 official archive")
        classwise["class_label"] = labels
        classwise["is_rare"] = classwise["class_id"].isin(rare_ids)
        classwise_rows.append(classwise)

    pd.DataFrame(result_rows).to_csv(
        out_dir / "official_confirmation_results.csv",
        index=False,
    )
    pd.concat(classwise_rows, ignore_index=True).to_csv(
        out_dir / "official_confirmation_classwise.csv",
        index=False,
    )

    def write_transitions(
        records: dict[str, dict[str, np.ndarray | int]],
        path: Path,
    ) -> None:
        rows: list[dict[str, object]] = []
        rare_set = set(map(int, rare_ids))
        for run_id, record in records.items():
            for class_id, class_label in enumerate(labels):
                corrected = int(record["corrected_by_class"][class_id])
                introduced = int(record["introduced_by_class"][class_id])
                rows.append(
                    {
                        "run_id": run_id,
                        "class_id": class_id,
                        "class_label": class_label,
                        "is_rare": class_id in rare_set,
                        "corrected": corrected,
                        "introduced": introduced,
                        "net_corrected": corrected - introduced,
                    }
                )
            rows.append(
                {
                    "run_id": run_id,
                    "class_id": -1,
                    "class_label": "ALL",
                    "is_rare": False,
                    "corrected": int(record["corrected"]),
                    "introduced": int(record["introduced"]),
                    "net_corrected": int(
                        record["corrected"] - record["introduced"]
                    ),
                    "both_correct": int(record["both_correct"]),
                    "both_wrong": int(record["both_wrong"]),
                }
            )
        pd.DataFrame(rows).to_csv(path, index=False)

    write_transitions(
        transitions,
        out_dir / "official_error_transitions.csv",
    )
    write_transitions(
        duplicate_free_transitions,
        out_dir / "official_duplicate_free_error_transitions.csv",
    )

    duplicate_free_result_rows: list[dict[str, object]] = []
    duplicate_free_classwise_rows: list[pd.DataFrame] = []
    for run_id in METHODS:
        metrics, classwise = confusion_metrics(
            duplicate_free_confusions[run_id],
            rare_ids,
        )
        duplicate_free_result_rows.append(
            {
                "dataset": "CICIoT2023 official archive",
                "source_role": (
                    "official archive unique conflict free confirmation"
                ),
                "seed": args.seed,
                "split_id": "official_file_group_01_63",
                "run_id": run_id,
                **metrics,
                "pr_auc_macro": np.nan,
                "training_seconds": training_costs[run_id],
                "inference_seconds": reported_inference_seconds[run_id],
                "inference_ms_per_10000": (
                    reported_inference_seconds[run_id]
                    * 1000.0
                    * 10_000.0
                    / processed_rows
                ),
                "model_size_mb": model_sizes[run_id],
                "peak_rss_mb": peak_rss,
            }
        )
        classwise.insert(0, "run_id", run_id)
        classwise.insert(0, "split_id", "official_file_group_01_63")
        classwise.insert(0, "seed", args.seed)
        classwise.insert(0, "dataset", "CICIoT2023 official archive")
        classwise["class_label"] = labels
        classwise["is_rare"] = classwise["class_id"].isin(rare_ids)
        duplicate_free_classwise_rows.append(classwise)
    pd.DataFrame(duplicate_free_result_rows).to_csv(
        out_dir / "official_duplicate_free_results.csv",
        index=False,
    )
    pd.concat(duplicate_free_classwise_rows, ignore_index=True).to_csv(
        out_dir / "official_duplicate_free_classwise.csv",
        index=False,
    )
    for name, records in (
        ("official_paired_joint_counts.npz", paired_joint_counts),
        (
            "official_duplicate_free_paired_joint_counts.npz",
            duplicate_free_paired_joint_counts,
        ),
    ):
        np.savez_compressed(
            out_dir / name,
            rare_ids=rare_ids.astype(np.int16),
            **{
                f"{primary}__vs__{comparator}": tensor
                for (primary, comparator), tensor in records.items()
            },
        )

    audit = {
        "seed": args.seed,
        "base_rows": int(len(base_df)),
        "gate_train_rows": int(len(gate_train_df)),
        "gate_select_rows": int(len(gate_select_df)),
        "final_test_rows": int(processed_rows),
        "final_test_duplicate_crossed_rows": int(
            duplicate_crossed_rows
        ),
        "final_test_duplicate_crossing_fraction": float(
            duplicate_crossed_rows / processed_rows
        ),
        "final_test_conflicting_label_rows": int(conflicting_label_rows),
        "final_test_conflicting_label_fraction": float(
            conflicting_label_rows / processed_rows
        ),
        "final_test_repeated_feature_rows": int(repeated_final_rows),
        "final_test_repeated_feature_fraction": float(
            repeated_final_rows / processed_rows
        ),
        "final_test_unique_conflict_free_rows": int(
            duplicate_free_confusions[FLAT_ID].sum()
        ),
        "base_source_sequences": "01-45",
        "gate_train_source_sequences": "46-50",
        "gate_select_source_sequences": "51-54",
        "final_test_source_sequences": "55-63",
        "role_overlap_by_construction": 0,
        "rare_fraction": 0.01,
        "rare_support_threshold": rare_threshold,
        "rare_labels": [labels[int(value)] for value in rare_ids],
        "cyclic_shift": cyclic_shift,
        "resource_preflight": args.max_test_rows is not None,
    }
    pd.DataFrame([audit]).to_csv(
        out_dir / "official_split_audit.csv",
        index=False,
    )
    metadata = {
        "purpose": "frozen official CICIoT2023 source confirmation",
        "source": str(source),
        "manifest": str(manifest_path),
        "samples_dir": str(samples_dir),
        "duplicate_audit_dir": str(duplicate_audit_dir),
        "design_lock": (
            "docs/jit_20260729_v1/"
            "OFFICIAL_CICIOT2023_CONFIRMATION_LOCK.md"
        ),
        "command_arguments": vars(args),
        "methods": list(METHODS),
        "features": features,
        "label_to_family": label_to_family,
        "support_matched_map": support_map,
        "rare_definition": (
            "Fine labels with no more than one percent of complete base "
            "source rows before deterministic training caps."
        ),
        "official_test_probability_retained": False,
        "duplicate_free_test_sensitivity_generated": True,
        "duplicate_free_definition": (
            "first final occurrence of an exact predictor hash that is "
            "absent from every development role and absent from the "
            "global conflicting label hash list"
        ),
        "selection_uses_final_test": False,
        "paired_joint_counts_retained": True,
        "paired_contrasts": [
            {"primary": primary, "comparator": comparator}
            for primary, comparator in PAIRED_CONTRASTS
        ],
        "platform": platform.platform(),
        "python": platform.python_version(),
        "package_versions": package_versions(),
    }
    metadata["command_arguments"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in metadata["command_arguments"].items()
    }
    (out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(pd.DataFrame(result_rows).to_string(index=False))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
