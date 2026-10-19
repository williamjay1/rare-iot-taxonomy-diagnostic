"""Experiment A: drift aware sequential window evaluation.

The official CICIoT2023 merged files have no timestamp column in any locally
available distribution. The file number in Merged01 to Merged63 follows the
publication capture order and is used here as a disclosed capture order proxy.

Protocol (locked in docs/jit_20260810_drift_llm/EXPERIMENT_DESIGN_GATE.md):
- Train once on the frozen official archive base sample (files 1 to 45,
  attack cap 25k, benign cap 80k), identical to the archive confirmation run.
- Four sequential test windows: W1 files 55-56, W2 files 57-58, W3 files
  59-60, W4 files 61-63 (Amendment 1: files 46 to 54 carry the development
  roles, so the first locked window definitions degenerated to zero rows).
- Duplicate free per window: exclude exact predictor hashes observed in any
  development role (base, gate_train, gate_select) or in the global
  conflicting label hash list; keep the first occurrence per hash in the
  window.
- Methods without gates: weighted flat XGBoost, equal semantic blend (real
  taxonomy), equal cyclic control blend, equal support matched control blend.
- Drift diagnostics per window versus base: class mix Jensen-Shannon
  divergence, family mix JS, per feature PSI, benign fraction, class
  coverage.
- Paired stratified bootstrap 5,000 reps per window for blend versus flat,
  versus cyclic, versus support.

Test windows are never used for fitting or selection.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import prepare_ciciot2023_official_confirmation_samples as preparation
import run_ciciot2023_official_archive_confirmation as archive
import run_ciciot_agreement_gated_trf_confirmation as nested
import run_ciciot_calibrated_benefit_gate_confirmation as benefit
import run_ciciot_support_matched_control_v2 as support_control
import run_ciciot_taxonomy_residual_fusion as taxonomy
import analyze_ciciot_agreement_gated_trf_confirmation as nested_analysis


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
    / "jit_ciciot_official_archive_audit_final"
    / "official_member_manifest.csv"
)
DEFAULT_SAMPLES = (
    ROOT
    / "results"
    / "jit_ciciot2023_official_confirmation_samples_final"
)
DEFAULT_DUPLICATE_AUDIT = (
    ROOT
    / "results"
    / "jit_ciciot_official_split_duplicate_audit_final"
)
DEFAULT_FROZEN_ARCHIVE = (
    ROOT
    / "results"
    / "jit_ciciot2023_official_archive_confirmation"
)

FLAT_ID = "flat_weighted_xgboost"
BLEND_ID = "equal_flat_taxonomy_blend"
CYCLIC_ID = "calibrated_cyclic_taxonomy_control"
SUPPORT_ID = "calibrated_support_matched_control"
METHODS = (FLAT_ID, BLEND_ID, CYCLIC_ID, SUPPORT_ID)
CONTRASTS = (
    (BLEND_ID, FLAT_ID),
    (BLEND_ID, CYCLIC_ID),
    (BLEND_ID, SUPPORT_ID),
)
WINDOWS = {
    "w1_55_56": [55, 56],
    "w2_57_58": [57, 58],
    "w3_59_60": [59, 60],
    "w4_61_63": [61, 62, 63],
}
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--samples-dir", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument(
        "--duplicate-audit-dir",
        type=Path,
        default=DEFAULT_DUPLICATE_AUDIT,
    )
    parser.add_argument(
        "--frozen-archive-dir",
        type=Path,
        default=DEFAULT_FROZEN_ARCHIVE,
    )
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--global-trees", type=int, default=520)
    parser.add_argument("--local-trees", type=int, default=380)
    parser.add_argument("--bootstrap-reps", type=int, default=5_000)
    parser.add_argument(
        "--max-rows-per-member",
        type=int,
        default=None,
        help="Development preflight only. Formal runs leave this unset.",
    )
    parser.add_argument(
        "--windows",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional subset of window ids for resumable execution; "
            "formal runs leave this unset."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="jit_sequential_window_drift_study",
    )
    return parser.parse_args()


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in (
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "lightgbm",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def js_divergence(
    base_proportions: np.ndarray,
    window_proportions: np.ndarray,
) -> float:
    base_proportions = (
        np.asarray(base_proportions, dtype=np.float64) + EPS
    )
    window_proportions = (
        np.asarray(window_proportions, dtype=np.float64) + EPS
    )
    base_proportions = base_proportions / base_proportions.sum()
    window_proportions = (
        window_proportions / window_proportions.sum()
    )
    mixture = 0.5 * (base_proportions + window_proportions)
    left = base_proportions * np.log(base_proportions / mixture)
    right = window_proportions * np.log(
        window_proportions / mixture
    )
    return float(0.5 * (left.sum() + right.sum()))


def psi_from_counts(
    window_counts: np.ndarray,
    base_expected: np.ndarray,
) -> float:
    actual = window_counts.astype(np.float64) + EPS
    actual = actual / actual.sum()
    expected = base_expected + EPS
    expected = expected / expected.sum()
    return float(
        ((actual - expected) * np.log(actual / expected)).sum()
    )


def window_mix_profile(
    y_true: np.ndarray,
    fine_encoder: LabelEncoder,
    label_to_family: dict[str, str],
) -> tuple[np.ndarray, np.ndarray]:
    class_counts = np.bincount(y_true, minlength=len(fine_encoder.classes_))
    class_proportions = (
        class_counts.astype(np.float64) / max(1, class_counts.sum())
    )
    family_counts: dict[str, int] = {}
    for label, family in label_to_family.items():
        fine_id = int(fine_encoder.transform([label])[0])
        family_counts[family] = family_counts.get(family, 0) + int(
            class_counts[fine_id]
        )
    families = sorted(label_to_family.values())
    family_proportions = np.asarray(
        [family_counts.get(family, 0) for family in families],
        dtype=np.float64,
    )
    family_proportions = (
        family_proportions / max(1.0, family_proportions.sum())
    )
    return class_proportions, family_proportions


def first_occurrence_within_window(
    feature_hash: np.ndarray,
    seen_hashes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the first occurrence per hash inside one window.

    Chunks are visited in file order. The returned mask marks rows whose hash
    was not observed earlier in the window, and the second return value is the
    updated sorted array of hashes observed so far. This is the vectorized
    equivalent of a per row set membership loop.
    """
    unique_hashes, first_indices = np.unique(
        feature_hash,
        return_index=True,
    )
    first_mask = np.zeros(len(feature_hash), dtype=bool)
    first_mask[first_indices] = True
    new_mask = (
        ~archive.membership_mask(feature_hash, seen_hashes)
        & first_mask
    )
    seen_hashes = np.union1d(seen_hashes, unique_hashes)
    return new_mask, seen_hashes


def limit_model_threads(
    flat_model: object,
    fitted: dict[str, dict[str, object]],
) -> None:
    """Constrain inference threads to reduce OpenMP contention."""
    flat_model.set_params(n_jobs=8)
    for state in fitted.values():
        state["family_model"].n_jobs = 8
        state["family_model"]._n_jobs = 8
        for model, _, _ in state["local_models"].values():
            if model is not None:
                model.n_jobs = 8
                model._n_jobs = 8


def dedup_and_psi_from_files(
    members: pd.DataFrame,
    source: Path,
    features: list[str],
    base_edges: list[np.ndarray],
    psi_bins: int,
    chunk_size: int,
    canonical: dict[str, str],
    label_to_family: dict[str, str],
    development_feature_hashes: list[np.ndarray],
    conflicting_feature_hashes: np.ndarray,
    expected_columns: list[str],
) -> tuple[dict[str, int], list[np.ndarray]]:
    """Re-read member files to reproduce the window dedup audit and PSI bins.

    Used when a window was completed before a crash but its derived tables
    were not written. No model inference is performed, so the pass is fast.
    """
    audit = {
        "raw_rows": 0,
        "crossed_rows": 0,
        "conflict_rows": 0,
        "repeated_rows": 0,
        "kept_rows": 0,
    }
    psi_accumulator = [
        np.zeros(psi_bins, dtype=np.int64)
        for _ in features
    ]
    seen_hashes = np.empty(0, dtype=np.uint64)
    for _, member_row in members.iterrows():
        member = str(member_row["member"])
        path = source / Path(member).name
        member_rows = 0
        for chunk in pd.read_csv(
            path,
            chunksize=chunk_size,
            low_memory=False,
        ):
            if [str(column) for column in chunk.columns] != (
                expected_columns
            ):
                raise ValueError(f"Unexpected schema in {member}.")
            chunk = archive.canonicalise_labels(chunk, canonical)
            chunk = archive.add_family(chunk, label_to_family)
            X = taxonomy.clean_features(chunk, features)
            feature_hash = pd.util.hash_pandas_object(
                chunk[features],
                index=False,
                categorize=True,
            ).to_numpy(dtype=np.uint64)
            crossed = np.zeros(len(chunk), dtype=bool)
            for reference in development_feature_hashes:
                crossed |= archive.membership_mask(
                    feature_hash,
                    reference,
                )
            conflict = archive.membership_mask(
                feature_hash,
                conflicting_feature_hashes,
            )
            first_occurrence, seen_hashes = (
                first_occurrence_within_window(
                    feature_hash,
                    seen_hashes,
                )
            )
            duplicate_free = (
                ~crossed & ~conflict & first_occurrence
            )
            audit["raw_rows"] += len(chunk)
            audit["crossed_rows"] += int(crossed.sum())
            audit["conflict_rows"] += int(conflict.sum())
            audit["repeated_rows"] += int(
                (~first_occurrence).sum()
            )
            audit["kept_rows"] += int(duplicate_free.sum())
            for feature_id in range(len(features)):
                binned = np.digitize(
                    X[duplicate_free, feature_id],
                    base_edges[feature_id][1:-1],
                )
                psi_accumulator[feature_id] += np.bincount(
                    binned,
                    minlength=psi_bins,
                )
            member_rows += len(chunk)
        if member_rows != int(member_row["rows"]):
            raise ValueError(
                f"Row count changed for {member}: "
                f"{member_rows} != {int(member_row['rows'])}"
            )
    return audit, psi_accumulator


def append_frame(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    frame.to_csv(
        path,
        mode="a",
        header=not path.is_file(),
        index=False,
    )


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    args = parse_args()
    source = args.source.resolve()
    manifest_path = args.manifest.resolve()
    samples_dir = args.samples_dir.resolve()
    duplicate_audit_dir = args.duplicate_audit_dir.resolve()
    frozen_archive_dir = args.frozen_archive_dir.resolve()
    out_dir = ROOT / "results" / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    training_seconds: dict[str, float] = {}

    sample_metadata = json.loads(
        (samples_dir / "sample_metadata.json").read_text(encoding="utf-8")
    )
    features = [str(value) for value in sample_metadata["feature_columns"]]
    base_df = pd.read_parquet(samples_dir / "base.parquet")
    label_to_family = archive.public_label_to_family()
    base_df = archive.add_family(base_df, label_to_family)
    fine_encoder = LabelEncoder().fit(sorted(label_to_family))
    family_encoder = LabelEncoder().fit(
        sorted(set(label_to_family.values()))
    )
    labels = list(map(str, fine_encoder.classes_))
    families = list(map(str, family_encoder.classes_))

    selected_counts = pd.read_csv(
        samples_dir / "selected_label_counts.csv"
    )
    rare_ids, class_support, rare_threshold = (
        archive.rare_ids_from_counts(
            selected_counts,
            fine_encoder,
        )
    )
    cyclic_map, cyclic_shift = taxonomy.cyclic_control_map(
        label_to_family
    )
    support_map, support_audit = (
        support_control.support_matched_wrong_map(
            base_df,
            label_to_family,
        )
    )
    frozen_support_map = json.loads(
        (frozen_archive_dir / "support_matched_map.json").read_text(
            encoding="utf-8"
        )
    )
    if support_map != frozen_support_map:
        raise RuntimeError(
            "Recomputed support matched map differs from the frozen map."
        )

    taxonomy.RANDOM_STATE = args.seed
    print("Fitting flat predictor on the frozen base sample...")
    flat_start = time.perf_counter()
    flat_model, flat_seconds = (
        nested.fit_weighted_xgb(
            base_df,
            features,
            fine_encoder,
            args.global_trees,
        )
    )
    training_seconds[FLAT_ID] = flat_seconds
    taxonomy_specs: dict[str, dict[str, object]] = {
        "real": {
            "mapping": label_to_family,
            "family_encoder": family_encoder,
        },
        "cyclic": {
            "mapping": cyclic_map,
            "family_encoder": None,
        },
        "support": {
            "mapping": support_map,
            "family_encoder": None,
        },
    }
    fitted: dict[str, dict[str, object]] = {}
    for key, spec in taxonomy_specs.items():
        print(f"Fitting {key} taxonomy hierarchy...")
        (
            family_model,
            fitted_family_encoder,
            local_models,
            taxonomy_seconds,
        ) = benefit.taxonomy_bundle(
            base_df,
            features,
            spec["mapping"],  # type: ignore[arg-type]
            args.local_trees,
            spec["family_encoder"],  # type: ignore[arg-type]
        )
        fitted[key] = {
            "family_model": family_model,
            "family_encoder": fitted_family_encoder,
            "local_models": local_models,
            "mapping": spec["mapping"],
        }
        training_seconds[f"{key}_taxonomy"] = taxonomy_seconds
    training_seconds["total_training"] = (
        time.perf_counter() - flat_start
    )
    limit_model_threads(flat_model, fitted)

    # Base profiles for drift diagnostics.
    base_class_proportions, base_family_proportions = (
        window_mix_profile(
            fine_encoder.transform(base_df["Label"].astype(str)),
            fine_encoder,
            label_to_family,
        )
    )
    base_X = taxonomy.clean_features(base_df, features)
    psi_bins = 10
    base_edges: list[np.ndarray] = []
    base_expected: list[np.ndarray] = []
    for feature_id in range(base_X.shape[1]):
        column = base_X[:, feature_id]
        quantiles = np.quantile(
            column,
            np.linspace(0.0, 1.0, psi_bins + 1),
        )
        quantiles[0] = -np.inf
        quantiles[-1] = np.inf
        base_edges.append(quantiles)
        binned = np.digitize(column, quantiles[1:-1])
        expected = np.bincount(binned, minlength=psi_bins).astype(
            np.float64
        )
        base_expected.append(expected / max(1.0, expected.sum()))
    del base_X

    development_feature_hashes = [
        np.load(
            duplicate_audit_dir / f"{role}_unique_feature_hashes.npy"
        )
        for role in ("base", "gate_train", "gate_select")
    ]
    conflicting_feature_hashes = np.load(
        duplicate_audit_dir / "global_conflicting_feature_hashes.npy"
    )
    canonical = preparation.canonical_label_map()
    expected_columns = features + ["Label"]
    manifest = pd.read_csv(manifest_path)

    result_rows: list[dict[str, object]] = []
    classwise_rows: list[dict[str, object]] = []
    drift_rows: list[dict[str, object]] = []
    feature_psi_rows: list[dict[str, object]] = []
    mix_rows: list[dict[str, object]] = []
    bootstrap_summary_rows: list[dict[str, object]] = []
    bootstrap_sample_frames: list[pd.DataFrame] = []
    inference_seconds: dict[str, float] = {}
    window_audit: dict[str, dict[str, int]] = {}
    psi_accumulators: dict[str, list[np.ndarray]] = {
        window_id: [
            np.zeros(psi_bins, dtype=np.int64)
            for _ in features
        ]
        for window_id in WINDOWS
    }
    completed: dict[str, str] = {}
    progress_path = out_dir / "completed_windows.json"
    if progress_path.is_file():
        completed = json.loads(
            progress_path.read_text(encoding="utf-8")
        )

    window_ids = (
        list(WINDOWS) if args.windows is None else list(args.windows)
    )
    for window_id in window_ids:
        if window_id not in WINDOWS:
            raise ValueError(f"Unknown window id {window_id}.")
        sequences = WINDOWS[window_id]
        members = manifest.loc[
            manifest["sequence"].isin(sequences)
        ].sort_values("sequence")
        npz_path = out_dir / f"window_predictions_{window_id}.npz"
        if npz_path.is_file() and window_id in completed:
            print(
                f"Window {window_id} already completed; skipping.",
                flush=True,
            )
            continue
        if npz_path.is_file():
            print(
                f"Resuming analysis for {window_id} from saved "
                "predictions.",
                flush=True,
            )
            with np.load(npz_path) as data:
                y_true = data["y_true"]
                predictions = {
                    run_id: data[run_id]
                    for run_id in METHODS
                }
            confusions = {
                run_id: np.zeros(
                    (len(labels), len(labels)),
                    dtype=np.int64,
                )
                for run_id in METHODS
            }
            for run_id in METHODS:
                archive.update_confusion(
                    confusions[run_id],
                    y_true,
                    predictions[run_id],
                )
            window_audit[window_id], psi_accumulators[window_id] = (
                dedup_and_psi_from_files(
                    members,
                    source,
                    features,
                    base_edges,
                    psi_bins,
                    args.chunk_size,
                    canonical,
                    label_to_family,
                    development_feature_hashes,
                    conflicting_feature_hashes,
                    expected_columns,
                )
            )
            print(
                f"  resumed kept rows {len(y_true)}",
                flush=True,
            )
        else:
            print(f"Processing window {window_id}...", flush=True)
            seen_window_hashes = np.empty(0, dtype=np.uint64)
            confusions = {
                run_id: np.zeros(
                    (len(labels), len(labels)),
                    dtype=np.int64,
                )
                for run_id in METHODS
            }
            y_parts: list[np.ndarray] = []
            prediction_parts: dict[str, list[np.ndarray]] = {
                run_id: [] for run_id in METHODS
            }
            window_audit[window_id] = {
                "raw_rows": 0,
                "crossed_rows": 0,
                "conflict_rows": 0,
                "repeated_rows": 0,
                "kept_rows": 0,
            }

            for _, member_row in members.iterrows():
                member = str(member_row["member"])
                path = source / Path(member).name
                member_rows = 0
                print(f"  {member}...", flush=True)
                chunk_index = 0
                for chunk in pd.read_csv(
                    path,
                    chunksize=args.chunk_size,
                    low_memory=False,
                ):
                    chunk_index += 1
                    if args.max_rows_per_member is not None:
                        remaining = (
                            args.max_rows_per_member - member_rows
                        )
                        if remaining <= 0:
                            break
                        if len(chunk) > remaining:
                            chunk = chunk.iloc[:remaining].copy()
                    if [str(column) for column in chunk.columns] != (
                        expected_columns
                    ):
                        raise ValueError(
                            f"Unexpected schema in {member}."
                        )
                    chunk = archive.canonicalise_labels(
                        chunk,
                        canonical,
                    )
                    chunk = archive.add_family(
                        chunk,
                        label_to_family,
                    )
                    y_true_chunk = fine_encoder.transform(
                        chunk["Label"].astype(str)
                    )
                    X = taxonomy.clean_features(chunk, features)
                    feature_hash = pd.util.hash_pandas_object(
                        chunk[features],
                        index=False,
                        categorize=True,
                    ).to_numpy(dtype=np.uint64)
                    crossed = np.zeros(len(chunk), dtype=bool)
                    for reference in development_feature_hashes:
                        crossed |= archive.membership_mask(
                            feature_hash,
                            reference,
                        )
                    conflict = archive.membership_mask(
                        feature_hash,
                        conflicting_feature_hashes,
                    )
                    first_occurrence, seen_window_hashes = (
                        first_occurrence_within_window(
                            feature_hash,
                            seen_window_hashes,
                        )
                    )
                    duplicate_free = (
                        ~crossed & ~conflict & first_occurrence
                    )
                    print(
                        f"    chunk {chunk_index}: rows "
                        f"{len(chunk)}, kept "
                        f"{int(duplicate_free.sum())}",
                        flush=True,
                    )
                    window_audit[window_id]["raw_rows"] += len(chunk)
                    window_audit[window_id]["crossed_rows"] += int(
                        crossed.sum()
                    )
                    window_audit[window_id]["conflict_rows"] += int(
                        conflict.sum()
                    )
                    window_audit[window_id]["repeated_rows"] += int(
                        (~first_occurrence).sum()
                    )
                    window_audit[window_id]["kept_rows"] += int(
                        duplicate_free.sum()
                    )

                    start = time.perf_counter()
                    flat_probability = np.asarray(
                        flat_model.predict_proba(X),
                        dtype=np.float64,
                    )
                    flat_seconds = time.perf_counter() - start
                    inference_seconds[FLAT_ID] = (
                        inference_seconds.get(FLAT_ID, 0.0)
                        + flat_seconds
                    )
                    predictions: dict[str, np.ndarray] = {
                        FLAT_ID: archive.probability_prediction(
                            flat_probability
                        )
                    }
                    for key, state in fitted.items():
                        start = time.perf_counter()
                        taxonomy_probability, _ = (
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
                        run_id = {
                            "real": BLEND_ID,
                            "cyclic": CYCLIC_ID,
                            "support": SUPPORT_ID,
                        }[key]
                        inference_seconds[run_id] = (
                            inference_seconds.get(run_id, 0.0)
                            + taxonomy_seconds
                        )
                        blend_probability = (
                            0.5 * flat_probability
                            + 0.5 * taxonomy_probability
                        )
                        predictions[run_id] = (
                            archive.probability_prediction(
                                blend_probability
                            )
                        )

                    for run_id in METHODS:
                        prediction = predictions[run_id][
                            duplicate_free
                        ]
                        archive.update_confusion(
                            confusions[run_id],
                            y_true_chunk[duplicate_free],
                            prediction,
                        )
                        prediction_parts[run_id].append(prediction)
                    for feature_id in range(len(features)):
                        binned = np.digitize(
                            X[duplicate_free, feature_id],
                            base_edges[feature_id][1:-1],
                        )
                        psi_accumulators[window_id][feature_id] += (
                            np.bincount(
                                binned,
                                minlength=psi_bins,
                            )
                        )
                    y_parts.append(y_true_chunk[duplicate_free])
                    member_rows += len(chunk)
                if (
                    args.max_rows_per_member is None
                    and member_rows != int(member_row["rows"])
                ):
                    raise ValueError(
                        f"Row count changed for {member}: "
                        f"{member_rows} != {int(member_row['rows'])}"
                    )

            if len(y_parts) == 0 or all(
                len(part) == 0 for part in y_parts
            ):
                raise RuntimeError(
                    f"Window {window_id} has no duplicate free "
                    "kept rows."
                )
            y_true = np.concatenate(y_parts)
            predictions = {
                run_id: np.concatenate(parts)
                for run_id, parts in prediction_parts.items()
            }
            np.savez_compressed(
                npz_path,
                y_true=y_true,
                rare_ids=rare_ids,
                **{
                    run_id: predictions[run_id]
                    for run_id in METHODS
                },
            )
            del y_parts, prediction_parts
        if len(y_true) == 0:
            raise RuntimeError(
                f"Window {window_id} has no duplicate free kept rows."
            )
        analysis_marker = (
            len(result_rows),
            len(classwise_rows),
            len(drift_rows),
            len(feature_psi_rows),
            len(mix_rows),
            len(bootstrap_summary_rows),
            len(bootstrap_sample_frames),
        )

        class_proportions, family_proportions = window_mix_profile(
            y_true,
            fine_encoder,
            label_to_family,
        )
        benign_id = int(fine_encoder.transform(["BenignTraffic"])[0])
        benign_fraction = float(
            class_proportions[benign_id]
        )
        class_coverage = float(
            np.mean(class_proportions > 0)
        )
        feature_psi: dict[str, float] = {}
        for feature_id in range(len(features)):
            counts = psi_accumulators[window_id][feature_id]
            feature_psi[features[feature_id]] = psi_from_counts(
                counts,
                base_expected[feature_id],
            )
        mean_psi = float(np.mean(list(feature_psi.values())))
        worst_feature = max(
            feature_psi,
            key=feature_psi.get,  # type: ignore[arg-type]
        )
        drift_rows.append(
            {
                "window": window_id,
                "n_rows": int(len(y_true)),
                "class_js": js_divergence(
                    base_class_proportions,
                    class_proportions,
                ),
                "family_js": js_divergence(
                    base_family_proportions,
                    family_proportions,
                ),
                "benign_fraction": benign_fraction,
                "base_benign_fraction": float(
                    base_class_proportions[benign_id]
                ),
                "class_coverage": class_coverage,
                "mean_psi": mean_psi,
                "worst_psi_feature": worst_feature,
                "worst_psi": feature_psi[worst_feature],
            }
        )
        for feature_id in range(len(features)):
            feature_psi_rows.append(
                {
                    "window": window_id,
                    "feature": features[feature_id],
                    "psi": feature_psi[features[feature_id]],
                }
            )
        for class_id, label in enumerate(labels):
            mix_rows.append(
                {
                    "window": "base",
                    "item_type": "class",
                    "item": label,
                    "proportion": float(base_class_proportions[class_id]),
                }
            )
            mix_rows.append(
                {
                    "window": window_id,
                    "item_type": "class",
                    "item": label,
                    "proportion": float(class_proportions[class_id]),
                }
            )
        for family_id, family_name in enumerate(families):
            mix_rows.append(
                {
                    "window": "base",
                    "item_type": "family",
                    "item": family_name,
                    "proportion": float(
                        base_family_proportions[family_id]
                    ),
                }
            )
            mix_rows.append(
                {
                    "window": window_id,
                    "item_type": "family",
                    "item": family_name,
                    "proportion": float(
                        family_proportions[family_id]
                    ),
                }
            )

        for run_id in METHODS:
            metrics, classwise = archive.confusion_metrics(
                confusions[run_id],
                rare_ids,
            )
            result_rows.append(
                {
                    "window": window_id,
                    "run_id": run_id,
                    **metrics,
                }
            )
            for class_id, label in enumerate(labels):
                row = classwise.iloc[class_id]
                classwise_rows.append(
                    {
                        "window": window_id,
                        "run_id": run_id,
                        "label": label,
                        "support": int(row["support"]),
                        "precision": float(row["precision"]),
                        "recall": float(row["recall"]),
                        "f1": float(row["f1"]),
                    }
                )

        rng = np.random.default_rng(args.seed + sum(sequences))
        for primary_id, comparator_id in CONTRASTS:
            sample_frame, summary = (
                nested_analysis.paired_stratified_bootstrap(
                    y_true,
                    predictions[primary_id],
                    predictions[comparator_id],
                    rare_ids,
                    len(labels),
                    args.bootstrap_reps,
                    rng,
                )
            )
            sample_frame = sample_frame.assign(
                window=window_id,
                primary=primary_id,
                comparator=comparator_id,
            )
            bootstrap_sample_frames.append(sample_frame)
            bootstrap_summary_rows.append(
                {
                    "window": window_id,
                    "primary": primary_id,
                    "comparator": comparator_id,
                    **summary,
                }
            )
        del y_true, predictions, confusions

        append_frame(
            out_dir / "sequential_window_results.csv",
            pd.DataFrame(result_rows[analysis_marker[0]:]),
        )
        append_frame(
            out_dir / "sequential_window_classwise.csv",
            pd.DataFrame(classwise_rows[analysis_marker[1]:]),
        )
        append_frame(
            out_dir / "drift_diagnostics.csv",
            pd.DataFrame(drift_rows[analysis_marker[2]:]),
        )
        append_frame(
            out_dir / "feature_psi.csv",
            pd.DataFrame(feature_psi_rows[analysis_marker[3]:]),
        )
        append_frame(
            out_dir / "mix_profiles.csv",
            pd.DataFrame(mix_rows[analysis_marker[4]:]),
        )
        append_frame(
            out_dir / "paired_bootstrap_summary.csv",
            pd.DataFrame(
                bootstrap_summary_rows[analysis_marker[5]:]
            ),
        )
        if bootstrap_sample_frames[analysis_marker[6]:]:
            append_frame(
                out_dir / "paired_bootstrap_samples.csv",
                pd.concat(
                    bootstrap_sample_frames[analysis_marker[6]:],
                    ignore_index=True,
                ),
            )
        (out_dir / f"window_audit_{window_id}.json").write_text(
            json.dumps(window_audit[window_id], indent=2),
            encoding="utf-8",
        )
        completed[window_id] = time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(),
        )
        progress_path.write_text(
            json.dumps(completed, indent=2),
            encoding="utf-8",
        )
        print(
            f"Window {window_id} checkpoint written.",
            flush=True,
        )

    for window_id in WINDOWS:
        if window_id not in window_audit:
            audit_path = out_dir / f"window_audit_{window_id}.json"
            if audit_path.is_file():
                window_audit[window_id] = json.loads(
                    audit_path.read_text(encoding="utf-8")
                )

    final_frames = {
        "sequential_window_results.csv": pd.DataFrame(
            result_rows
        ),
        "sequential_window_classwise.csv": pd.DataFrame(
            classwise_rows
        ),
        "drift_diagnostics.csv": pd.DataFrame(drift_rows),
        "feature_psi.csv": pd.DataFrame(feature_psi_rows),
        "mix_profiles.csv": pd.DataFrame(mix_rows),
        "paired_bootstrap_summary.csv": pd.DataFrame(
            bootstrap_summary_rows
        ),
    }
    for csv_name, frame in final_frames.items():
        path = out_dir / csv_name
        if not path.is_file():
            frame.to_csv(path, index=False)
    samples_path = out_dir / "paired_bootstrap_samples.csv"
    if not samples_path.is_file() and bootstrap_sample_frames:
        pd.concat(bootstrap_sample_frames, ignore_index=True).to_csv(
            samples_path,
            index=False,
        )
    dedup_audit_path = out_dir / "window_dedup_audit.csv"
    if not dedup_audit_path.is_file():
        pd.DataFrame(
            [
                {
                    "window": window_id,
                    **audit,
                }
                for window_id, audit in window_audit.items()
            ]
        ).to_csv(dedup_audit_path, index=False)

    metadata = {
        "purpose": (
            "drift aware sequential window evaluation on the official "
            "CICIoT2023 archive"
        ),
        "design_lock": (
            "docs/jit_20260810_drift_llm/EXPERIMENT_DESIGN_GATE.md"
        ),
        "source": str(source),
        "manifest": str(manifest_path),
        "samples_dir": str(samples_dir),
        "duplicate_audit_dir": str(duplicate_audit_dir),
        "frozen_archive_dir": str(frozen_archive_dir),
        "windows": WINDOWS,
        "dedup_rule": (
            "exclude hashes observed in base, gate_train or gate_select "
            "or in the global conflicting label hash list; first "
            "occurrence per hash inside each window"
        ),
        "methods": list(METHODS),
        "contrasts": [list(pair) for pair in CONTRASTS],
        "fine_labels": labels,
        "families": families,
        "label_to_family": label_to_family,
        "cyclic_control_shift": cyclic_shift,
        "rare_ids": [int(value) for value in rare_ids],
        "rare_threshold": rare_threshold,
        "rare_definition": (
            "fine class support at most one percent of base source rows"
        ),
        "arguments": {
            key: str(value)
            for key, value in vars(args).items()
        },
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "window_audit": window_audit,
        "capture_order_proxy_disclosure": (
            "no timestamp column exists in any locally available "
            "CICIoT2023 distribution; Merged file numbers follow the "
            "publication capture order and are used as a disclosed "
            "capture order proxy, not a wall clock split"
        ),
        "started_at": started,
        "finished_at": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ",
            time.gmtime(),
        ),
        "resumed_from_checkpoint": bool(completed),
        "completed_windows": completed,
        "execution_note": (
            "per window predictions are checkpointed as "
            "window_predictions_<id>.npz together with appended derived "
            "tables, so an interrupted run resumes from the last "
            "completed window"
        ),
        "platform": platform.platform(),
        "package_versions": package_versions(),
    }
    (out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
