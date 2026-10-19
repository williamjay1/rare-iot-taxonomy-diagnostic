from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import run_ciciot_quick_experiments as exp
import run_ciciot_taxonomy_residual_fusion as trf
import screen_ciciot_residual_gate_variants as gate


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Six split nested confirmation for agreement gated taxonomy "
            "residual fusion."
        )
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45, 46, 47],
    )
    parser.add_argument("--outer-test-frac", type=float, default=0.2)
    parser.add_argument("--inner-validation-frac", type=float, default=0.2)
    parser.add_argument("--majority-cap", type=int, default=25_000)
    parser.add_argument("--benign-cap", type=int, default=80_000)
    parser.add_argument("--global-trees", type=int, default=520)
    parser.add_argument("--local-trees", type=int, default=380)
    parser.add_argument("--rare-lambda", type=float, default=0.04)
    parser.add_argument(
        "--out-dir",
        default="ijics_v2_ciciot_agreement_gated_trf_6split",
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


def hard_hierarchy_probability(
    hierarchy_probability: np.ndarray,
    family_probability: np.ndarray,
    fine_encoder: LabelEncoder,
    family_encoder: LabelEncoder,
    label_to_family: dict[str, str],
) -> np.ndarray:
    top_family = np.argmax(family_probability, axis=1)
    output = np.zeros_like(hierarchy_probability)
    for fine_id, label in enumerate(fine_encoder.classes_):
        family_id = int(family_encoder.transform([label_to_family[str(label)]])[0])
        mask = top_family == family_id
        output[mask, fine_id] = hierarchy_probability[mask, fine_id]
    row_sum = output.sum(axis=1, keepdims=True)
    empty = row_sum[:, 0] <= trf.EPS
    output[empty] = hierarchy_probability[empty]
    return output / np.clip(output.sum(axis=1, keepdims=True), trf.EPS, None)


def classwise_frame(
    seed: int,
    run_id: str,
    y_true: np.ndarray,
    probability: np.ndarray,
    labels: list[str],
    rare_ids: np.ndarray,
) -> pd.DataFrame:
    prediction = np.argmax(probability, axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        prediction,
        labels=np.arange(len(labels)),
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "seed": seed,
            "run_id": run_id,
            "class_id": np.arange(len(labels)),
            "class_label": labels,
            "is_rare": np.isin(np.arange(len(labels)), rare_ids),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )


def fit_flat_lgbm(
    fit_df: pd.DataFrame,
    features: list[str],
    fine_encoder: LabelEncoder,
    n_estimators: int,
) -> tuple[object, float]:
    X_fit = trf.clean_features(fit_df, features)
    y_fit = fine_encoder.transform(fit_df["Label"].astype(str))
    model = trf.make_lgbm(len(fine_encoder.classes_), n_estimators)
    start = time.perf_counter()
    model.fit(
        X_fit,
        y_fit,
        sample_weight=trf.inverse_frequency_weights(y_fit, 0.35),
    )
    return model, time.perf_counter() - start


def fit_unweighted_xgb(
    fit_df: pd.DataFrame,
    features: list[str],
    fine_encoder: LabelEncoder,
    n_estimators: int,
) -> tuple[object, float]:
    X_fit = trf.clean_features(fit_df, features)
    y_fit = fine_encoder.transform(fit_df["Label"].astype(str))
    model = trf.make_global_model(len(fine_encoder.classes_), n_estimators)
    start = time.perf_counter()
    model.fit(X_fit, y_fit)
    return model, time.perf_counter() - start


def fit_weighted_xgb(
    fit_df: pd.DataFrame,
    features: list[str],
    fine_encoder: LabelEncoder,
    n_estimators: int,
) -> tuple[object, float]:
    X_fit = trf.clean_features(fit_df, features)
    y_fit = fine_encoder.transform(fit_df["Label"].astype(str))
    model = trf.make_global_model(len(fine_encoder.classes_), n_estimators)
    start = time.perf_counter()
    model.fit(
        X_fit,
        y_fit,
        sample_weight=trf.inverse_frequency_weights(y_fit, 0.35),
    )
    return model, time.perf_counter() - start


def fit_real_taxonomy(
    fit_df: pd.DataFrame,
    features: list[str],
    family_encoder: LabelEncoder,
    local_trees: int,
) -> tuple[object, dict[str, tuple[object | None, LabelEncoder | None, list[str]]], float]:
    X_fit = trf.clean_features(fit_df, features)
    y_family = family_encoder.transform(fit_df["attack_class"].astype(str))
    start = time.perf_counter()
    family_model = trf.make_lgbm(len(family_encoder.classes_), local_trees)
    family_model.fit(
        X_fit,
        y_family,
        sample_weight=trf.inverse_frequency_weights(y_family, 0.25),
    )
    local_models: dict[
        str, tuple[object | None, LabelEncoder | None, list[str]]
    ] = {}
    family_values = fit_df["attack_class"].astype(str).to_numpy()
    for family_name in family_encoder.classes_:
        mask = family_values == str(family_name)
        labels = fit_df.loc[mask, "Label"].astype(str)
        unique_labels = sorted(labels.unique().tolist())
        if len(unique_labels) == 1:
            local_models[str(family_name)] = (None, None, unique_labels)
            continue
        local_encoder = LabelEncoder()
        y_local = local_encoder.fit_transform(labels)
        local_model = trf.make_lgbm(len(local_encoder.classes_), local_trees)
        local_model.fit(
            X_fit[mask],
            y_local,
            sample_weight=trf.inverse_frequency_weights(y_local, 0.50),
        )
        local_models[str(family_name)] = (
            local_model,
            local_encoder,
            list(local_encoder.classes_),
        )
    return family_model, local_models, time.perf_counter() - start


def validation_objective(
    y_true: np.ndarray,
    probability: np.ndarray,
    rare_ids: np.ndarray,
    rare_lambda: float,
) -> tuple[float, float, float, float, float]:
    prediction = np.argmax(probability, axis=1)
    recalls = recall_score(
        y_true,
        prediction,
        labels=np.arange(probability.shape[1]),
        average=None,
        zero_division=0,
    )
    macro_f1 = float(
        f1_score(y_true, prediction, average="macro", zero_division=0)
    )
    rare_recall = float(np.mean(recalls[rare_ids]))
    return (
        macro_f1 + rare_lambda * rare_recall,
        macro_f1,
        rare_recall,
        float(np.min(recalls)),
        float(accuracy_score(y_true, prediction)),
    )


def select_agreement_gate(
    y_validation: np.ndarray,
    flat_validation: np.ndarray,
    taxonomy_validation: np.ndarray,
    family_validation: np.ndarray,
    fine_encoder: LabelEncoder,
    family_encoder: LabelEncoder,
    label_to_family: dict[str, str],
    rare_ids: np.ndarray,
    rare_lambda: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    rows: list[dict[str, float]] = []
    betas = (0.0, 0.25, 0.5, 0.75, 1.0)
    for hierarchy_power in (0.75, 1.0):
        for beta_agree in betas:
            for beta_disagree in betas:
                if beta_disagree > beta_agree:
                    continue
                probability, agreement = gate.consensus_blend(
                    flat_validation,
                    taxonomy_validation,
                    family_validation,
                    fine_encoder,
                    family_encoder,
                    label_to_family,
                    beta_agree,
                    beta_disagree,
                    0.0,
                    hierarchy_power,
                )
                (
                    objective,
                    macro_f1,
                    rare_recall,
                    worst_recall,
                    accuracy,
                ) = validation_objective(
                    y_validation,
                    probability,
                    rare_ids,
                    rare_lambda,
                )
                rows.append(
                    {
                        "beta_agree": beta_agree,
                        "beta_disagree": beta_disagree,
                        "entropy_power": 0.0,
                        "hierarchy_power": hierarchy_power,
                        "objective": objective,
                        "f1_macro": macro_f1,
                        "rare_recall": rare_recall,
                        "worst_recall": worst_recall,
                        "accuracy": accuracy,
                        "agreement_fraction": float(np.mean(agreement)),
                        "gate_spread": beta_agree - beta_disagree,
                    }
                )
    grid = pd.DataFrame(rows).sort_values(
        [
            "objective",
            "f1_macro",
            "rare_recall",
            "gate_spread",
            "beta_agree",
        ],
        ascending=[False, False, False, True, True],
    )
    return grid.iloc[0].to_dict(), grid


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("numpy", "pandas", "scikit-learn", "lightgbm", "xgboost"):
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
        [pd.read_parquet(TRAIN_PATH), pd.read_parquet(TEST_PATH)],
        ignore_index=True,
    )
    features = exp.feature_cols(full)
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
    control_map, control_shift = trf.cyclic_control_map(label_to_family)

    all_results: list[dict[str, object]] = []
    all_classwise: list[pd.DataFrame] = []
    all_selection: list[dict[str, object]] = []
    split_audit: list[dict[str, object]] = []

    for seed in args.seeds:
        print(f"\nStarting outer split seed {seed}...")
        trf.RANDOM_STATE = seed
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
        y_fit_pool = fine_encoder.transform(fit_pool_df["Label"].astype(str))
        y_validation = fine_encoder.transform(validation_df["Label"].astype(str))
        y_outer = fine_encoder.transform(outer_test_df["Label"].astype(str))
        rare_ids = trf.rare_class_ids(y_fit_pool)

        split_audit.append(
            {
                "seed": seed,
                "fit_pool_rows_before_cap": int(len(fit_pool_df)),
                "fit_rows_after_cap": int(len(fit_df)),
                "validation_rows": int(len(validation_df)),
                "outer_test_rows": int(len(outer_test_df)),
                "rare_labels": json.dumps([labels[i] for i in rare_ids]),
                "fit_outer_overlap": 0,
                "validation_outer_overlap": 0,
            }
        )

        print(f"Seed {seed}: fitting weighted flat XGBoost...")
        weighted_xgb, weighted_xgb_seconds = fit_weighted_xgb(
            fit_df,
            features,
            fine_encoder,
            args.global_trees,
        )
        print(f"Seed {seed}: fitting real taxonomy models...")
        family_model, local_models, taxonomy_train_seconds = fit_real_taxonomy(
            fit_df,
            features,
            family_encoder,
            args.local_trees,
        )
        print(f"Seed {seed}: fitting cyclic taxonomy control...")
        (
            control_family_model,
            control_family_encoder,
            control_local_models,
            control_train_seconds,
        ) = trf.fit_control_hierarchy(
            fit_df,
            features,
            control_map,
            args.local_trees,
        )
        print(f"Seed {seed}: fitting flat LightGBM and unweighted XGBoost baselines...")
        flat_lgbm, flat_lgbm_seconds = fit_flat_lgbm(
            fit_df,
            features,
            fine_encoder,
            args.local_trees,
        )
        unweighted_xgb, unweighted_xgb_seconds = fit_unweighted_xgb(
            fit_df,
            features,
            fine_encoder,
            args.global_trees,
        )
        training_seconds_total = (
            weighted_xgb_seconds
            + taxonomy_train_seconds
            + control_train_seconds
            + flat_lgbm_seconds
            + unweighted_xgb_seconds
        )
        training_seconds_by_run = {
            "flat_unweighted_xgboost": unweighted_xgb_seconds,
            "flat_weighted_xgboost": weighted_xgb_seconds,
            "flat_weighted_lightgbm": flat_lgbm_seconds,
            "hard_top_down_taxonomy": taxonomy_train_seconds,
            "soft_taxonomy_factorisation": taxonomy_train_seconds,
            "equal_flat_taxonomy_blend": (
                weighted_xgb_seconds + taxonomy_train_seconds
            ),
            "global_taxonomy_residual_fusion": (
                weighted_xgb_seconds + taxonomy_train_seconds
            ),
            "agreement_gated_taxonomy_residual_fusion": (
                weighted_xgb_seconds + taxonomy_train_seconds
            ),
            "cyclic_taxonomy_control": (
                weighted_xgb_seconds + control_train_seconds
            ),
        }

        X_validation = trf.clean_features(validation_df, features)
        weighted_val = np.asarray(
            weighted_xgb.predict_proba(X_validation),
            dtype=np.float64,
        )
        hierarchy_val, family_val = trf.hierarchy_probabilities(
            X_validation,
            family_model,
            family_encoder,
            fine_encoder,
            label_to_family,
            local_models,
        )
        global_selection, global_grid = trf.select_global_gate(
            y_validation,
            weighted_val,
            hierarchy_val,
            family_val,
            rare_ids,
            args.rare_lambda,
        )
        agreement_selection, agreement_grid = select_agreement_gate(
            y_validation,
            weighted_val,
            hierarchy_val,
            family_val,
            fine_encoder,
            family_encoder,
            label_to_family,
            rare_ids,
            args.rare_lambda,
        )
        control_hierarchy_val, control_family_val = trf.hierarchy_probabilities(
            X_validation,
            control_family_model,
            control_family_encoder,
            fine_encoder,
            control_map,
            control_local_models,
        )
        control_selection, control_grid = select_agreement_gate(
            y_validation,
            weighted_val,
            control_hierarchy_val,
            control_family_val,
            fine_encoder,
            control_family_encoder,
            control_map,
            rare_ids,
            args.rare_lambda,
        )
        global_grid.assign(seed=seed).to_csv(
            out_dir / f"validation_global_grid_seed_{seed}.csv",
            index=False,
        )
        agreement_grid.assign(seed=seed).to_csv(
            out_dir / f"validation_agreement_grid_seed_{seed}.csv",
            index=False,
        )
        control_grid.assign(seed=seed).to_csv(
            out_dir / f"validation_cyclic_agreement_grid_seed_{seed}.csv",
            index=False,
        )
        all_selection.extend(
            [
                {
                    "seed": seed,
                    "kind": "global_real_taxonomy",
                    **global_selection.__dict__,
                },
                {
                    "seed": seed,
                    "kind": "agreement_real_taxonomy",
                    **agreement_selection,
                },
                {
                    "seed": seed,
                    "kind": "agreement_cyclic_control",
                    **control_selection,
                },
            ]
        )

        print(f"Seed {seed}: evaluating the untouched outer split...")
        X_outer = trf.clean_features(outer_test_df, features)
        inference_start = time.perf_counter()
        weighted_outer = np.asarray(weighted_xgb.predict_proba(X_outer), dtype=np.float64)
        unweighted_outer = np.asarray(unweighted_xgb.predict_proba(X_outer), dtype=np.float64)
        flat_lgbm_outer = np.asarray(flat_lgbm.predict_proba(X_outer), dtype=np.float64)
        hierarchy_outer, family_outer = trf.hierarchy_probabilities(
            X_outer,
            family_model,
            family_encoder,
            fine_encoder,
            label_to_family,
            local_models,
        )
        control_hierarchy_outer, control_family_outer = trf.hierarchy_probabilities(
            X_outer,
            control_family_model,
            control_family_encoder,
            fine_encoder,
            control_map,
            control_local_models,
        )
        global_outer = trf.blend(
            weighted_outer,
            hierarchy_outer,
            family_outer,
            global_selection.beta,
            global_selection.entropy_power,
            global_selection.hierarchy_power,
        )
        equal_outer = trf.blend(
            weighted_outer,
            hierarchy_outer,
            family_outer,
            0.5,
            0.0,
            1.0,
        )
        hard_outer = hard_hierarchy_probability(
            hierarchy_outer,
            family_outer,
            fine_encoder,
            family_encoder,
            label_to_family,
        )
        agreement_outer, family_agreement = gate.consensus_blend(
            weighted_outer,
            hierarchy_outer,
            family_outer,
            fine_encoder,
            family_encoder,
            label_to_family,
            agreement_selection["beta_agree"],
            agreement_selection["beta_disagree"],
            0.0,
            agreement_selection["hierarchy_power"],
        )
        cyclic_outer, cyclic_family_agreement = gate.consensus_blend(
            weighted_outer,
            control_hierarchy_outer,
            control_family_outer,
            fine_encoder,
            control_family_encoder,
            control_map,
            control_selection["beta_agree"],
            control_selection["beta_disagree"],
            0.0,
            control_selection["hierarchy_power"],
        )
        inference_seconds_total = time.perf_counter() - inference_start

        probability_map = {
            "flat_unweighted_xgboost": unweighted_outer,
            "flat_weighted_xgboost": weighted_outer,
            "flat_weighted_lightgbm": flat_lgbm_outer,
            "hard_top_down_taxonomy": hard_outer,
            "soft_taxonomy_factorisation": hierarchy_outer,
            "equal_flat_taxonomy_blend": equal_outer,
            "global_taxonomy_residual_fusion": global_outer,
            "agreement_gated_taxonomy_residual_fusion": agreement_outer,
            "cyclic_taxonomy_control": cyclic_outer,
        }
        prediction_store: dict[str, np.ndarray] = {
            "row_id": outer_test_df.index.to_numpy(dtype=np.int64),
            "y_true": y_outer.astype(np.int16),
            "rare_ids": rare_ids.astype(np.int16),
            "family_agreement": family_agreement.astype(np.uint8),
            "cyclic_family_agreement": cyclic_family_agreement.astype(np.uint8),
            "flat_weighted_xgboost_probability": weighted_outer.astype(np.float16),
            "equal_flat_taxonomy_blend_probability": equal_outer.astype(np.float16),
            "agreement_gated_taxonomy_residual_fusion_probability": (
                agreement_outer.astype(np.float16)
            ),
        }
        for run_id, probability in probability_map.items():
            row = trf.metric_row(run_id, y_outer, probability, rare_ids)
            selected_values: dict[str, float] = {}
            if run_id == "global_taxonomy_residual_fusion":
                selected_values = {
                    "beta": global_selection.beta,
                    "entropy_power": global_selection.entropy_power,
                    "hierarchy_power": global_selection.hierarchy_power,
                }
            elif run_id == "agreement_gated_taxonomy_residual_fusion":
                selected_values = {
                    "beta_agree": agreement_selection["beta_agree"],
                    "beta_disagree": agreement_selection["beta_disagree"],
                    "entropy_power": 0.0,
                    "hierarchy_power": agreement_selection["hierarchy_power"],
                }
            elif run_id == "cyclic_taxonomy_control":
                selected_values = {
                    "beta_agree": control_selection["beta_agree"],
                    "beta_disagree": control_selection["beta_disagree"],
                    "entropy_power": 0.0,
                    "hierarchy_power": control_selection["hierarchy_power"],
                }
            row.update(
                {
                    "dataset": "CICIoT2023 public subsample",
                    "source_role": "third party development mirror",
                    "seed": seed,
                    "split_id": f"outer_seed_{seed}",
                    "n_test": int(len(outer_test_df)),
                    "fit_rows": int(len(fit_df)),
                    "validation_rows": int(len(validation_df)),
                    "outer_test_rows": int(len(outer_test_df)),
                    "n_features": len(features),
                    "training_seconds": training_seconds_by_run[run_id],
                    "inference_seconds": np.nan,
                    "inference_ms_per_10000": np.nan,
                    "model_size_mb": np.nan,
                    "peak_rss_mb": np.nan,
                    "training_seconds_all_models": training_seconds_total,
                    "inference_seconds_all_models": inference_seconds_total,
                    "beta": np.nan,
                    "beta_agree": np.nan,
                    "beta_disagree": np.nan,
                    "entropy_power": np.nan,
                    "hierarchy_power": np.nan,
                    "family_agreement_fraction": (
                        float(np.mean(family_agreement))
                        if run_id
                        == "agreement_gated_taxonomy_residual_fusion"
                        else np.nan
                    ),
                }
            )
            row.update(selected_values)
            all_results.append(row)
            all_classwise.append(
                classwise_frame(
                    seed,
                    run_id,
                    y_outer,
                    probability,
                    labels,
                    rare_ids,
                )
            )
            prediction_store[run_id] = np.argmax(probability, axis=1).astype(np.int16)
        np.savez_compressed(
            out_dir / f"paired_predictions_seed_{seed}.npz",
            **prediction_store,
        )
        pd.DataFrame(all_results).to_csv(
            out_dir / "nested_split_results_partial.csv",
            index=False,
        )
        print(
            pd.DataFrame(
                [row for row in all_results if int(row["seed"]) == seed]
            )[["run_id", "accuracy", "f1_macro", "rare_recall", "worst_recall"]]
            .sort_values("f1_macro", ascending=False)
            .to_string(index=False)
        )

    results = pd.DataFrame(all_results)
    results.to_csv(out_dir / "nested_split_results.csv", index=False)
    pd.concat(all_classwise, ignore_index=True).to_csv(
        out_dir / "nested_split_classwise_metrics.csv",
        index=False,
    )
    pd.DataFrame(all_selection).to_csv(
        out_dir / "nested_split_selections.csv",
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
            worst_recall_sd=("worst_recall", "std"),
        )
        .reset_index()
        .sort_values("macro_f1_mean", ascending=False)
    )
    aggregate.to_csv(out_dir / "nested_split_aggregate.csv", index=False)
    metadata = {
        "purpose": "agreement gated taxonomy residual fusion confirmation",
        "dataset": "CICIoT2023 public subsample",
        "source_role": "third party development mirror",
        "historical_fixed_test_reused_as_final": False,
        "merged_rows": int(len(full)),
        "features": features,
        "fine_labels": labels,
        "label_to_family": label_to_family,
        "cyclic_control_map": control_map,
        "cyclic_control_shift": control_shift,
        "arguments": vars(args),
        "command": " ".join(sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "package_versions": package_versions(),
        "agreement_gate_grid": {
            "beta_agree": [0.0, 0.25, 0.5, 0.75, 1.0],
            "beta_disagree": [0.0, 0.25, 0.5, 0.75, 1.0],
            "constraint": "beta_disagree <= beta_agree",
            "entropy_power": [0.0],
            "hierarchy_power": [0.75, 1.0],
        },
        "selection_rule": "inner validation only",
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
