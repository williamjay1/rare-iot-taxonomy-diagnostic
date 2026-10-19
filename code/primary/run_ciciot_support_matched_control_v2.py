from __future__ import annotations

import argparse
import json
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import run_ciciot_agreement_gated_trf_confirmation as nested
import run_ciciot_taxonomy_residual_fusion as trf


ROOT = Path(__file__).resolve().parents[1]
CONTROL_ID = "support_matched_wrong_taxonomy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a support matched wrong taxonomy capacity control."
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46, 47])
    parser.add_argument("--outer-test-frac", type=float, default=0.2)
    parser.add_argument("--inner-validation-frac", type=float, default=0.2)
    parser.add_argument("--majority-cap", type=int, default=25_000)
    parser.add_argument("--benign-cap", type=int, default=80_000)
    parser.add_argument("--global-trees", type=int, default=520)
    parser.add_argument("--local-trees", type=int, default=380)
    parser.add_argument("--rare-lambda", type=float, default=0.04)
    parser.add_argument(
        "--out-dir",
        default="ijics_v2_ciciot_support_matched_wrong_taxonomy_6split",
    )
    return parser.parse_args()


def support_matched_wrong_map(
    fit_df: pd.DataFrame,
    label_to_family: dict[str, str],
) -> tuple[dict[str, str], pd.DataFrame]:
    labels = sorted(label_to_family)
    counts = fit_df["Label"].astype(str).value_counts().reindex(labels).astype(int)
    slot_families = np.asarray([label_to_family[label] for label in labels], dtype=object)
    log_support = np.log1p(counts.to_numpy(dtype=float))
    cost = np.abs(log_support[:, None] - log_support[None, :])
    same_family = np.asarray(
        [
            [label_to_family[label] == slot_family for slot_family in slot_families]
            for label in labels
        ],
        dtype=bool,
    )
    cost[same_family] = 1_000_000.0
    row_ids, slot_ids = linear_sum_assignment(cost)
    if not np.array_equal(row_ids, np.arange(len(labels))):
        raise RuntimeError("Unexpected assignment row order.")
    mapping = {
        labels[row_id]: str(slot_families[slot_id])
        for row_id, slot_id in zip(row_ids, slot_ids)
    }
    if any(mapping[label] == label_to_family[label] for label in labels):
        raise RuntimeError("The support matched control retained a real family.")

    rows: list[dict[str, object]] = []
    for family in sorted(set(label_to_family.values())):
        real_labels = [label for label in labels if label_to_family[label] == family]
        control_labels = [label for label in labels if mapping[label] == family]
        real_support = int(counts.loc[real_labels].sum())
        control_support = int(counts.loc[control_labels].sum())
        rows.append(
            {
                "family": family,
                "real_label_count": len(real_labels),
                "control_label_count": len(control_labels),
                "real_fit_support": real_support,
                "control_fit_support": control_support,
                "absolute_support_difference": abs(control_support - real_support),
                "relative_support_difference": (
                    abs(control_support - real_support) / real_support
                    if real_support
                    else 0.0
                ),
                "real_labels": json.dumps(real_labels),
                "control_labels": json.dumps(control_labels),
            }
        )
    audit = pd.DataFrame(rows)
    if not (audit["real_label_count"] == audit["control_label_count"]).all():
        raise RuntimeError("The control did not preserve labels per family.")
    return mapping, audit


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    args = parse_args()
    out_dir = ROOT / "results" / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and merging the historical public subsample...")
    full = pd.concat(
        [pd.read_parquet(nested.TRAIN_PATH), pd.read_parquet(nested.TEST_PATH)],
        ignore_index=True,
    )
    features = nested.exp.feature_cols(full)
    label_to_family = (
        full[["Label", "attack_class"]]
        .drop_duplicates()
        .set_index("Label")["attack_class"]
        .astype(str)
        .to_dict()
    )
    labels = sorted(full["Label"].astype(str).unique())
    fine_encoder = LabelEncoder().fit(labels)

    result_rows: list[dict[str, object]] = []
    classwise_rows: list[pd.DataFrame] = []
    selection_rows: list[dict[str, object]] = []
    map_records: dict[str, dict[str, str]] = {}
    for seed in args.seeds:
        print(f"\nStarting support matched control seed {seed}...")
        trf.RANDOM_STATE = seed
        fit_validation_df, outer_df = train_test_split(
            full,
            test_size=args.outer_test_frac,
            random_state=seed,
            stratify=full["Label"],
        )
        fit_pool_df, validation_df = train_test_split(
            fit_validation_df,
            test_size=args.inner_validation_frac,
            random_state=seed + 10_000,
            stratify=fit_validation_df["Label"],
        )
        fit_df = nested.cap_fit_rows(
            fit_pool_df,
            seed,
            args.majority_cap,
            args.benign_cap,
        )
        y_fit_pool = fine_encoder.transform(fit_pool_df["Label"].astype(str))
        rare_ids = trf.rare_class_ids(y_fit_pool)
        control_map, support_audit = support_matched_wrong_map(
            fit_df,
            label_to_family,
        )
        support_audit.insert(0, "seed", seed)
        support_audit.to_csv(
            out_dir / f"support_match_audit_seed_{seed}.csv",
            index=False,
        )
        (out_dir / f"support_matched_map_seed_{seed}.json").write_text(
            json.dumps(control_map, indent=2),
            encoding="utf-8",
        )
        map_records[str(seed)] = control_map

        print(f"Seed {seed}: fitting weighted flat model...")
        weighted_model, weighted_seconds = nested.fit_weighted_xgb(
            fit_df,
            features,
            fine_encoder,
            args.global_trees,
        )
        print(f"Seed {seed}: fitting support matched wrong taxonomy...")
        (
            family_model,
            family_encoder,
            local_models,
            control_seconds,
        ) = trf.fit_control_hierarchy(
            fit_df,
            features,
            control_map,
            args.local_trees,
        )

        X_validation = trf.clean_features(validation_df, features)
        y_validation = fine_encoder.transform(validation_df["Label"].astype(str))
        weighted_validation = np.asarray(
            weighted_model.predict_proba(X_validation),
            dtype=np.float64,
        )
        hierarchy_validation, family_validation = trf.hierarchy_probabilities(
            X_validation,
            family_model,
            family_encoder,
            fine_encoder,
            control_map,
            local_models,
        )
        selection, grid = nested.select_agreement_gate(
            y_validation,
            weighted_validation,
            hierarchy_validation,
            family_validation,
            fine_encoder,
            family_encoder,
            control_map,
            rare_ids,
            args.rare_lambda,
        )
        grid.assign(seed=seed).to_csv(
            out_dir / f"validation_support_control_grid_seed_{seed}.csv",
            index=False,
        )
        selection_rows.append({"seed": seed, **selection})

        print(f"Seed {seed}: evaluating outer rows...")
        X_outer = trf.clean_features(outer_df, features)
        y_outer = fine_encoder.transform(outer_df["Label"].astype(str))
        inference_start = nested.time.perf_counter()
        weighted_outer = np.asarray(
            weighted_model.predict_proba(X_outer),
            dtype=np.float64,
        )
        hierarchy_outer, family_outer = trf.hierarchy_probabilities(
            X_outer,
            family_model,
            family_encoder,
            fine_encoder,
            control_map,
            local_models,
        )
        control_outer, family_agreement = nested.gate.consensus_blend(
            weighted_outer,
            hierarchy_outer,
            family_outer,
            fine_encoder,
            family_encoder,
            control_map,
            selection["beta_agree"],
            selection["beta_disagree"],
            0.0,
            selection["hierarchy_power"],
        )
        inference_seconds = nested.time.perf_counter() - inference_start
        row = trf.metric_row(CONTROL_ID, y_outer, control_outer, rare_ids)
        row.update(
            {
                "dataset": "CICIoT2023 public subsample",
                "source_role": "third party development mirror",
                "seed": seed,
                "split_id": f"outer_seed_{seed}",
                "n_test": int(len(outer_df)),
                "fit_rows": int(len(fit_df)),
                "validation_rows": int(len(validation_df)),
                "outer_test_rows": int(len(outer_df)),
                "n_features": len(features),
                "training_seconds": weighted_seconds + control_seconds,
                "inference_seconds": inference_seconds,
                "inference_ms_per_10000": (
                    inference_seconds * 1000.0 * 10_000.0 / len(outer_df)
                ),
                "model_size_mb": np.nan,
                "peak_rss_mb": np.nan,
                "training_seconds_weighted_flat": weighted_seconds,
                "training_seconds_control": control_seconds,
                "inference_seconds_weighted_and_control": inference_seconds,
                "beta_agree": selection["beta_agree"],
                "beta_disagree": selection["beta_disagree"],
                "entropy_power": 0.0,
                "hierarchy_power": selection["hierarchy_power"],
                "family_agreement_fraction": float(np.mean(family_agreement)),
            }
        )
        result_rows.append(row)
        classwise_rows.append(
            nested.classwise_frame(
                seed,
                CONTROL_ID,
                y_outer,
                control_outer,
                labels,
                rare_ids,
            )
        )
        np.savez_compressed(
            out_dir / f"support_control_predictions_seed_{seed}.npz",
            y_true=y_outer.astype(np.int16),
            rare_ids=rare_ids.astype(np.int16),
            row_id=outer_df.index.to_numpy(dtype=np.int64),
            family_agreement=family_agreement.astype(np.uint8),
            support_matched_wrong_taxonomy_probability=control_outer.astype(
                np.float16
            ),
            support_matched_wrong_taxonomy=np.argmax(
                control_outer,
                axis=1,
            ).astype(np.int16),
        )
        pd.DataFrame(result_rows).to_csv(
            out_dir / "support_matched_control_results_partial.csv",
            index=False,
        )
        print(
            pd.DataFrame([row])[
                ["run_id", "accuracy", "f1_macro", "rare_recall", "worst_recall"]
            ].to_string(index=False)
        )

    results = pd.DataFrame(result_rows)
    results.to_csv(out_dir / "support_matched_control_results.csv", index=False)
    pd.concat(classwise_rows, ignore_index=True).to_csv(
        out_dir / "support_matched_control_classwise.csv",
        index=False,
    )
    pd.DataFrame(selection_rows).to_csv(
        out_dir / "support_matched_control_selections.csv",
        index=False,
    )
    metadata = {
        "purpose": "support matched wrong taxonomy capacity control",
        "dataset": "CICIoT2023 public subsample",
        "source_role": "third party development mirror",
        "features": features,
        "fine_labels": labels,
        "label_to_family": label_to_family,
        "maps_by_seed": map_records,
        "arguments": vars(args),
        "command": " ".join(sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "package_versions": nested.package_versions(),
        "selection_rule": "inner validation only",
        "outer_test_role": "not used for fitting or parameter selection",
    }
    (out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(results.to_string(index=False))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
