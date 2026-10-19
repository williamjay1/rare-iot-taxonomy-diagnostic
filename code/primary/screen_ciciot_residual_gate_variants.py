from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import run_ciciot_quick_experiments as exp
import run_ciciot_taxonomy_residual_fusion as trf


ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen residual gate variants before nested evaluation."
    )
    parser.add_argument("--train-n", type=int, default=300_000)
    parser.add_argument("--test-n", type=int, default=100_000)
    parser.add_argument("--validation-frac", type=float, default=0.2)
    parser.add_argument("--global-trees", type=int, default=520)
    parser.add_argument("--local-trees", type=int, default=380)
    parser.add_argument("--rare-lambda", type=float, default=0.04)
    parser.add_argument(
        "--out-dir",
        default="ijics_ciciot_residual_gate_variant_screen_300k",
    )
    return parser.parse_args()


def objective(
    y_true: np.ndarray,
    probability: np.ndarray,
    rare_ids: np.ndarray,
    rare_lambda: float,
) -> tuple[float, float, float]:
    prediction = np.argmax(probability, axis=1)
    macro_f1 = float(f1_score(y_true, prediction, average="macro", zero_division=0))
    recalls = recall_score(
        y_true,
        prediction,
        labels=np.arange(probability.shape[1]),
        average=None,
        zero_division=0,
    )
    rare_recall = float(np.mean(recalls[rare_ids]))
    return macro_f1 + rare_lambda * rare_recall, macro_f1, rare_recall


def fine_to_family_probability(
    fine_probability: np.ndarray,
    fine_encoder: LabelEncoder,
    family_encoder: LabelEncoder,
    label_to_family: dict[str, str],
) -> np.ndarray:
    output = np.zeros(
        (len(fine_probability), len(family_encoder.classes_)),
        dtype=np.float64,
    )
    for fine_id, label in enumerate(fine_encoder.classes_):
        family_id = int(family_encoder.transform([label_to_family[str(label)]])[0])
        output[:, family_id] += fine_probability[:, fine_id]
    output = np.clip(output, EPS, None)
    return output / output.sum(axis=1, keepdims=True)


def consensus_blend(
    flat_probability: np.ndarray,
    taxonomy_probability: np.ndarray,
    family_probability: np.ndarray,
    fine_encoder: LabelEncoder,
    family_encoder: LabelEncoder,
    label_to_family: dict[str, str],
    beta_agree: float,
    beta_disagree: float,
    entropy_power: float,
    hierarchy_power: float,
) -> tuple[np.ndarray, np.ndarray]:
    taxonomy = trf.power_normalise(taxonomy_probability, hierarchy_power)
    flat_family = fine_to_family_probability(
        flat_probability,
        fine_encoder,
        family_encoder,
        label_to_family,
    )
    agreement = np.argmax(flat_family, axis=1) == np.argmax(family_probability, axis=1)
    beta = np.where(agreement, beta_agree, beta_disagree)
    confidence = trf.family_confidence(family_probability)
    if entropy_power:
        beta = beta * np.power(confidence, entropy_power)
    mixed = flat_probability + beta[:, None] * (taxonomy - flat_probability)
    mixed = np.clip(mixed, EPS, None)
    return mixed / mixed.sum(axis=1, keepdims=True), agreement


def support_aware_blend(
    flat_probability: np.ndarray,
    taxonomy_probability: np.ndarray,
    family_probability: np.ndarray,
    rare_ids: np.ndarray,
    beta_common: float,
    beta_rare: float,
    entropy_power: float,
    hierarchy_power: float,
) -> np.ndarray:
    taxonomy = trf.power_normalise(taxonomy_probability, hierarchy_power)
    class_beta = np.full(flat_probability.shape[1], beta_common, dtype=np.float64)
    class_beta[rare_ids] = beta_rare
    confidence = trf.family_confidence(family_probability)
    gate = (
        np.ones_like(confidence)
        if entropy_power == 0
        else np.power(confidence, entropy_power)
    )
    mixed = flat_probability + gate[:, None] * class_beta[None, :] * (
        taxonomy - flat_probability
    )
    mixed = np.clip(mixed, EPS, None)
    return mixed / mixed.sum(axis=1, keepdims=True)


def select_consensus(
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
        for entropy_power in (0.0, 0.5):
            for beta_agree in betas:
                for beta_disagree in betas:
                    if beta_disagree > beta_agree:
                        continue
                    probability, agreement = consensus_blend(
                        flat_validation,
                        taxonomy_validation,
                        family_validation,
                        fine_encoder,
                        family_encoder,
                        label_to_family,
                        beta_agree,
                        beta_disagree,
                        entropy_power,
                        hierarchy_power,
                    )
                    score, macro_f1, rare_recall = objective(
                        y_validation,
                        probability,
                        rare_ids,
                        rare_lambda,
                    )
                    rows.append(
                        {
                            "beta_agree": beta_agree,
                            "beta_disagree": beta_disagree,
                            "entropy_power": entropy_power,
                            "hierarchy_power": hierarchy_power,
                            "objective": score,
                            "f1_macro": macro_f1,
                            "rare_recall": rare_recall,
                            "agreement_fraction": float(np.mean(agreement)),
                        }
                    )
    grid = pd.DataFrame(rows).sort_values(
        ["objective", "f1_macro", "rare_recall"],
        ascending=False,
    )
    best = grid.iloc[0].to_dict()
    return best, grid


def select_support_aware(
    y_validation: np.ndarray,
    flat_validation: np.ndarray,
    taxonomy_validation: np.ndarray,
    family_validation: np.ndarray,
    rare_ids: np.ndarray,
    rare_lambda: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    rows: list[dict[str, float]] = []
    betas = (0.0, 0.25, 0.5, 0.75, 1.0)
    for hierarchy_power in (0.75, 1.0):
        for entropy_power in (0.0, 0.5):
            for beta_common in betas:
                for beta_rare in betas:
                    probability = support_aware_blend(
                        flat_validation,
                        taxonomy_validation,
                        family_validation,
                        rare_ids,
                        beta_common,
                        beta_rare,
                        entropy_power,
                        hierarchy_power,
                    )
                    score, macro_f1, rare_recall = objective(
                        y_validation,
                        probability,
                        rare_ids,
                        rare_lambda,
                    )
                    rows.append(
                        {
                            "beta_common": beta_common,
                            "beta_rare": beta_rare,
                            "entropy_power": entropy_power,
                            "hierarchy_power": hierarchy_power,
                            "objective": score,
                            "f1_macro": macro_f1,
                            "rare_recall": rare_recall,
                        }
                    )
    grid = pd.DataFrame(rows).sort_values(
        ["objective", "f1_macro", "rare_recall"],
        ascending=False,
    )
    best = grid.iloc[0].to_dict()
    return best, grid


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    args = parse_args()
    out_dir = ROOT / "results" / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = exp.load_data(args.train_n, args.test_n)
    fit_df, validation_df = train_test_split(
        train_df,
        test_size=args.validation_frac,
        random_state=exp.RANDOM_STATE,
        stratify=train_df["Label"],
    )
    features = exp.feature_cols(train_df)
    fine_encoder = LabelEncoder().fit(train_df["Label"].astype(str))
    family_encoder = LabelEncoder().fit(train_df["attack_class"].astype(str))
    label_to_family = (
        train_df[["Label", "attack_class"]]
        .drop_duplicates()
        .set_index("Label")["attack_class"]
        .astype(str)
        .to_dict()
    )
    y_fit = fine_encoder.transform(fit_df["Label"].astype(str))
    y_validation = fine_encoder.transform(validation_df["Label"].astype(str))
    y_test = fine_encoder.transform(test_df["Label"].astype(str))
    rare_ids = trf.rare_class_ids(y_fit)

    print("Fitting flat and real taxonomy models...")
    flat_model, family_model, local_models, training_seconds = trf.fit_models(
        fit_df,
        features,
        fine_encoder,
        family_encoder,
        args.global_trees,
        args.local_trees,
    )
    X_validation = trf.clean_features(validation_df, features)
    flat_validation = np.asarray(
        flat_model.predict_proba(X_validation),
        dtype=np.float64,
    )
    taxonomy_validation, family_validation = trf.hierarchy_probabilities(
        X_validation,
        family_model,
        family_encoder,
        fine_encoder,
        label_to_family,
        local_models,
    )
    global_selection, global_grid = trf.select_global_gate(
        y_validation,
        flat_validation,
        taxonomy_validation,
        family_validation,
        rare_ids,
        args.rare_lambda,
    )
    consensus_selection, consensus_grid = select_consensus(
        y_validation,
        flat_validation,
        taxonomy_validation,
        family_validation,
        fine_encoder,
        family_encoder,
        label_to_family,
        rare_ids,
        args.rare_lambda,
    )
    support_selection, support_grid = select_support_aware(
        y_validation,
        flat_validation,
        taxonomy_validation,
        family_validation,
        rare_ids,
        args.rare_lambda,
    )
    global_grid.to_csv(out_dir / "validation_global_grid.csv", index=False)
    consensus_grid.to_csv(out_dir / "validation_consensus_grid.csv", index=False)
    support_grid.to_csv(out_dir / "validation_support_grid.csv", index=False)

    X_test = trf.clean_features(test_df, features)
    flat_test = np.asarray(flat_model.predict_proba(X_test), dtype=np.float64)
    taxonomy_test, family_test = trf.hierarchy_probabilities(
        X_test,
        family_model,
        family_encoder,
        fine_encoder,
        label_to_family,
        local_models,
    )
    global_test = trf.blend(
        flat_test,
        taxonomy_test,
        family_test,
        global_selection.beta,
        global_selection.entropy_power,
        global_selection.hierarchy_power,
    )
    consensus_test, test_agreement = consensus_blend(
        flat_test,
        taxonomy_test,
        family_test,
        fine_encoder,
        family_encoder,
        label_to_family,
        consensus_selection["beta_agree"],
        consensus_selection["beta_disagree"],
        consensus_selection["entropy_power"],
        consensus_selection["hierarchy_power"],
    )
    support_test = support_aware_blend(
        flat_test,
        taxonomy_test,
        family_test,
        rare_ids,
        support_selection["beta_common"],
        support_selection["beta_rare"],
        support_selection["entropy_power"],
        support_selection["hierarchy_power"],
    )
    probability_map = {
        "flat_weighted_xgboost": flat_test,
        "soft_taxonomy_factorisation": taxonomy_test,
        "global_taxonomy_residual_fusion": global_test,
        "consensus_taxonomy_residual_fusion": consensus_test,
        "support_aware_taxonomy_residual_fusion": support_test,
    }
    results = pd.DataFrame(
        [
            trf.metric_row(run_id, y_test, probability, rare_ids)
            for run_id, probability in probability_map.items()
        ]
    ).sort_values("f1_macro", ascending=False)
    results.to_csv(out_dir / "screen_results.csv", index=False)
    np.savez_compressed(
        out_dir / "screen_predictions.npz",
        y_true=y_test.astype(np.int16),
        rare_ids=rare_ids.astype(np.int16),
        **{
            run_id: np.argmax(probability, axis=1).astype(np.int16)
            for run_id, probability in probability_map.items()
        },
    )
    metadata = {
        "screening_only": True,
        "train_rows_loaded": int(len(train_df)),
        "fit_rows": int(len(fit_df)),
        "validation_rows": int(len(validation_df)),
        "development_test_rows": int(len(test_df)),
        "features": features,
        "rare_labels": [str(fine_encoder.classes_[idx]) for idx in rare_ids],
        "training_seconds": training_seconds,
        "global_selection": asdict(global_selection),
        "consensus_selection": consensus_selection,
        "support_selection": support_selection,
        "test_family_agreement_fraction": float(np.mean(test_agreement)),
        "arguments": vars(args),
    }
    (out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(results.to_string(index=False))
    print(json.dumps(metadata, indent=2))
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
