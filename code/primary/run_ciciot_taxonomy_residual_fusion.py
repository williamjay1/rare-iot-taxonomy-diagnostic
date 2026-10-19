from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from xgboost import XGBClassifier

import run_ciciot_quick_experiments as exp


ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-12
RANDOM_STATE = 42


@dataclass(frozen=True)
class Selection:
    beta: float
    entropy_power: float
    hierarchy_power: float
    objective: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen confidence gated taxonomy residual fusion on CICIoT2023."
    )
    parser.add_argument("--train-n", type=int, default=300_000)
    parser.add_argument("--test-n", type=int, default=100_000)
    parser.add_argument("--validation-frac", type=float, default=0.2)
    parser.add_argument("--out-dir", default="ijics_taxonomy_residual_fusion_screen_300k")
    parser.add_argument("--global-trees", type=int, default=420)
    parser.add_argument("--local-trees", type=int, default=320)
    parser.add_argument("--rare-lambda", type=float, default=0.04)
    return parser.parse_args()


def clean_features(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    return (
        df[features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .to_numpy(dtype=np.float32)
    )


def inverse_frequency_weights(y: np.ndarray, exponent: float) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    raw = {int(c): float(len(y) / (len(classes) * n)) ** exponent for c, n in zip(classes, counts)}
    weights = np.asarray([raw[int(v)] for v in y], dtype=np.float32)
    return weights / np.mean(weights)


def make_global_model(n_classes: int, n_estimators: int) -> XGBClassifier:
    return XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        n_estimators=n_estimators,
        max_depth=9,
        learning_rate=0.055,
        min_child_weight=2.0,
        subsample=0.9,
        colsample_bytree=0.95,
        reg_lambda=1.5,
        reg_alpha=0.02,
        tree_method="hist",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def make_lgbm(n_classes: int, n_estimators: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary" if n_classes == 2 else "multiclass",
        n_estimators=n_estimators,
        learning_rate=0.055,
        num_leaves=127,
        max_depth=-1,
        min_child_samples=25,
        subsample=0.9,
        colsample_bytree=0.95,
        reg_lambda=1.5,
        reg_alpha=0.02,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )


def power_normalise(probabilities: np.ndarray, power: float) -> np.ndarray:
    adjusted = np.power(np.clip(probabilities, EPS, 1.0), power)
    return adjusted / np.clip(adjusted.sum(axis=1, keepdims=True), EPS, None)


def family_confidence(family_probabilities: np.ndarray) -> np.ndarray:
    entropy = -np.sum(
        family_probabilities * np.log(np.clip(family_probabilities, EPS, 1.0)),
        axis=1,
    )
    return np.clip(1.0 - entropy / np.log(family_probabilities.shape[1]), 0.0, 1.0)


def rare_class_ids(y_fit: np.ndarray, fraction: float = 0.01) -> np.ndarray:
    counts = np.bincount(y_fit)
    threshold = max(1, int(np.ceil(len(y_fit) * fraction)))
    selected = np.flatnonzero((counts > 0) & (counts <= threshold))
    if len(selected):
        return selected
    present = np.flatnonzero(counts > 0)
    order = present[np.argsort(counts[present])]
    return order[: min(8, len(order))]


def metric_row(
    run_id: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    rare_ids: np.ndarray,
) -> dict[str, float | int | str]:
    prediction = np.argmax(probabilities, axis=1)
    recalls = recall_score(
        y_true,
        prediction,
        labels=np.arange(probabilities.shape[1]),
        average=None,
        zero_division=0,
    )
    y_binary = label_binarize(y_true, classes=np.arange(probabilities.shape[1]))
    return {
        "run_id": run_id,
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "f1_macro": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, prediction, average="macro", zero_division=0)),
        "rare_recall": float(np.mean(recalls[rare_ids])),
        "worst_recall": float(np.min(recalls)),
        "pr_auc_macro": float(
            average_precision_score(y_binary, probabilities, average="macro")
        ),
    }


def hierarchy_probabilities(
    X: np.ndarray,
    family_model: LGBMClassifier,
    family_encoder: LabelEncoder,
    fine_encoder: LabelEncoder,
    label_to_family: dict[str, str],
    local_models: dict[str, tuple[LGBMClassifier | None, LabelEncoder | None, list[str]]],
) -> tuple[np.ndarray, np.ndarray]:
    family_probability = np.asarray(family_model.predict_proba(X), dtype=np.float64)
    fine_probability = np.zeros((len(X), len(fine_encoder.classes_)), dtype=np.float64)
    for family_id, family_name in enumerate(family_encoder.classes_):
        model, local_encoder, labels = local_models[str(family_name)]
        if model is None or local_encoder is None:
            conditional = np.ones((len(X), 1), dtype=np.float64)
            local_labels = labels
        else:
            conditional = np.asarray(model.predict_proba(X), dtype=np.float64)
            local_labels = list(local_encoder.classes_)
        for local_id, label in enumerate(local_labels):
            fine_id = int(fine_encoder.transform([label])[0])
            fine_probability[:, fine_id] = family_probability[:, family_id] * conditional[:, local_id]
    row_sum = fine_probability.sum(axis=1, keepdims=True)
    fine_probability = fine_probability / np.clip(row_sum, EPS, None)
    expected = {str(v) for v in fine_encoder.classes_}
    if set(label_to_family) != expected:
        raise RuntimeError("The label to family map does not cover every fine label.")
    return fine_probability, family_probability


def fit_models(
    fit_df: pd.DataFrame,
    features: list[str],
    fine_encoder: LabelEncoder,
    family_encoder: LabelEncoder,
    global_trees: int,
    local_trees: int,
) -> tuple[
    XGBClassifier,
    LGBMClassifier,
    dict[str, tuple[LGBMClassifier | None, LabelEncoder | None, list[str]]],
    float,
]:
    X_fit = clean_features(fit_df, features)
    y_fine = fine_encoder.transform(fit_df["Label"].astype(str))
    y_family = family_encoder.transform(fit_df["attack_class"].astype(str))

    start = time.perf_counter()
    global_model = make_global_model(len(fine_encoder.classes_), global_trees)
    global_model.fit(X_fit, y_fine, sample_weight=inverse_frequency_weights(y_fine, 0.35))

    family_model = make_lgbm(len(family_encoder.classes_), local_trees)
    family_model.fit(X_fit, y_family, sample_weight=inverse_frequency_weights(y_family, 0.25))

    local_models: dict[
        str, tuple[LGBMClassifier | None, LabelEncoder | None, list[str]]
    ] = {}
    for family_name in family_encoder.classes_:
        mask = fit_df["attack_class"].astype(str).to_numpy() == str(family_name)
        labels = fit_df.loc[mask, "Label"].astype(str)
        unique_labels = sorted(labels.unique().tolist())
        if len(unique_labels) == 1:
            local_models[str(family_name)] = (None, None, unique_labels)
            continue
        local_encoder = LabelEncoder()
        y_local = local_encoder.fit_transform(labels)
        local_model = make_lgbm(len(local_encoder.classes_), local_trees)
        local_model.fit(
            X_fit[mask],
            y_local,
            sample_weight=inverse_frequency_weights(y_local, 0.50),
        )
        local_models[str(family_name)] = (
            local_model,
            local_encoder,
            list(local_encoder.classes_),
        )
    return global_model, family_model, local_models, time.perf_counter() - start


def cyclic_control_map(label_to_family: dict[str, str]) -> tuple[dict[str, str], int]:
    labels = sorted(label_to_family)
    family_slots = [label_to_family[label] for label in labels]
    best_mapping: dict[str, str] | None = None
    best_matches = len(labels) + 1
    best_shift = 0
    for shift in range(1, len(labels)):
        rotated = np.roll(np.asarray(family_slots, dtype=object), shift)
        candidate = {label: str(rotated[idx]) for idx, label in enumerate(labels)}
        matches = sum(candidate[label] == label_to_family[label] for label in labels)
        if matches < best_matches:
            best_mapping = candidate
            best_matches = matches
            best_shift = shift
    if best_mapping is None:
        raise RuntimeError("Could not construct the cyclic taxonomy control.")
    return best_mapping, best_shift


def fit_control_hierarchy(
    fit_df: pd.DataFrame,
    features: list[str],
    control_map: dict[str, str],
    local_trees: int,
) -> tuple[
    LGBMClassifier,
    LabelEncoder,
    dict[str, tuple[LGBMClassifier | None, LabelEncoder | None, list[str]]],
    float,
]:
    X_fit = clean_features(fit_df, features)
    control_family = fit_df["Label"].astype(str).map(control_map)
    if control_family.isna().any():
        raise RuntimeError("The cyclic control map does not cover every fitted label.")
    control_encoder = LabelEncoder().fit(control_family)
    y_family = control_encoder.transform(control_family)
    start = time.perf_counter()
    family_model = make_lgbm(len(control_encoder.classes_), local_trees)
    family_model.fit(
        X_fit,
        y_family,
        sample_weight=inverse_frequency_weights(y_family, 0.25),
    )
    local_models: dict[
        str, tuple[LGBMClassifier | None, LabelEncoder | None, list[str]]
    ] = {}
    family_values = control_family.to_numpy()
    for family_name in control_encoder.classes_:
        mask = family_values == str(family_name)
        labels = fit_df.loc[mask, "Label"].astype(str)
        unique_labels = sorted(labels.unique().tolist())
        if len(unique_labels) == 1:
            local_models[str(family_name)] = (None, None, unique_labels)
            continue
        local_encoder = LabelEncoder()
        y_local = local_encoder.fit_transform(labels)
        local_model = make_lgbm(len(local_encoder.classes_), local_trees)
        local_model.fit(
            X_fit[mask],
            y_local,
            sample_weight=inverse_frequency_weights(y_local, 0.50),
        )
        local_models[str(family_name)] = (
            local_model,
            local_encoder,
            list(local_encoder.classes_),
        )
    return (
        family_model,
        control_encoder,
        local_models,
        time.perf_counter() - start,
    )


def blend(
    global_probability: np.ndarray,
    hierarchy_probability: np.ndarray,
    family_probability: np.ndarray,
    beta: float,
    entropy_power: float,
    hierarchy_power: float,
    class_beta: np.ndarray | None = None,
) -> np.ndarray:
    hierarchy = power_normalise(hierarchy_probability, hierarchy_power)
    confidence = family_confidence(family_probability)
    gate = np.ones_like(confidence) if entropy_power == 0 else np.power(confidence, entropy_power)
    if class_beta is None:
        weight = beta * gate[:, None]
    else:
        weight = gate[:, None] * class_beta[None, :]
    mixed = global_probability + weight * (hierarchy - global_probability)
    mixed = np.clip(mixed, EPS, None)
    return mixed / mixed.sum(axis=1, keepdims=True)


def select_global_gate(
    y_val: np.ndarray,
    global_probability: np.ndarray,
    hierarchy_probability: np.ndarray,
    family_probability: np.ndarray,
    rare_ids: np.ndarray,
    rare_lambda: float,
) -> tuple[Selection, pd.DataFrame]:
    rows: list[dict[str, float]] = []
    for hierarchy_power in (0.75, 1.0, 1.25):
        for entropy_power in (0.0, 0.5, 1.0, 2.0):
            for beta in np.linspace(0.0, 1.0, 21):
                probability = blend(
                    global_probability,
                    hierarchy_probability,
                    family_probability,
                    float(beta),
                    float(entropy_power),
                    float(hierarchy_power),
                )
                prediction = np.argmax(probability, axis=1)
                recalls = recall_score(
                    y_val,
                    prediction,
                    labels=np.arange(probability.shape[1]),
                    average=None,
                    zero_division=0,
                )
                macro_f1 = float(
                    f1_score(y_val, prediction, average="macro", zero_division=0)
                )
                rare_recall = float(np.mean(recalls[rare_ids]))
                objective = macro_f1 + rare_lambda * rare_recall
                rows.append(
                    {
                        "beta": float(beta),
                        "entropy_power": float(entropy_power),
                        "hierarchy_power": float(hierarchy_power),
                        "objective": objective,
                        "accuracy": float(accuracy_score(y_val, prediction)),
                        "f1_macro": macro_f1,
                        "recall_macro": float(np.mean(recalls)),
                        "rare_recall": rare_recall,
                        "worst_recall": float(np.min(recalls)),
                    }
                )
    grid = pd.DataFrame(rows).sort_values(
        ["objective", "f1_macro", "rare_recall"],
        ascending=False,
    )
    best = grid.iloc[0]
    return (
        Selection(
            beta=float(best["beta"]),
            entropy_power=float(best["entropy_power"]),
            hierarchy_power=float(best["hierarchy_power"]),
            objective=float(best["objective"]),
        ),
        grid,
    )


def select_support_regularised_class_gate(
    y_val: np.ndarray,
    global_probability: np.ndarray,
    hierarchy_probability: np.ndarray,
    family_probability: np.ndarray,
    selection: Selection,
    shrinkage: float = 200.0,
) -> tuple[np.ndarray, pd.DataFrame]:
    n_classes = global_probability.shape[1]
    supports = np.bincount(y_val, minlength=n_classes)
    rows: list[dict[str, float | int]] = []
    selected = np.full(n_classes, selection.beta, dtype=np.float64)
    hierarchy = power_normalise(hierarchy_probability, selection.hierarchy_power)
    confidence = family_confidence(family_probability)
    gate = (
        np.ones_like(confidence)
        if selection.entropy_power == 0
        else np.power(confidence, selection.entropy_power)
    )
    for class_id in range(n_classes):
        y_binary = y_val == class_id
        best_beta = selection.beta
        best_f1 = -1.0
        for beta in np.linspace(0.0, 1.0, 21):
            score = global_probability[:, class_id] + beta * gate * (
                hierarchy[:, class_id] - global_probability[:, class_id]
            )
            threshold_prediction = score >= np.max(
                np.delete(global_probability, class_id, axis=1),
                axis=1,
            )
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_binary,
                threshold_prediction,
                average="binary",
                zero_division=0,
            )
            if f1 > best_f1:
                best_f1 = float(f1)
                best_beta = float(beta)
                best_precision = float(precision)
                best_recall = float(recall)
        reliability = supports[class_id] / (supports[class_id] + shrinkage)
        selected[class_id] = selection.beta + reliability * (best_beta - selection.beta)
        rows.append(
            {
                "class_id": class_id,
                "support": int(supports[class_id]),
                "raw_beta": best_beta,
                "regularised_beta": float(selected[class_id]),
                "validation_binary_precision": best_precision,
                "validation_binary_recall": best_recall,
                "validation_binary_f1": best_f1,
            }
        )
    return selected, pd.DataFrame(rows)


def classwise_frame(
    run_id: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
) -> pd.DataFrame:
    prediction = np.argmax(probabilities, axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        prediction,
        labels=np.arange(len(labels)),
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "run_id": run_id,
            "class_id": np.arange(len(labels)),
            "class_label": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    args = parse_args()
    out_dir = ROOT / "results" / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = exp.load_data(args.train_n, args.test_n)
    fit_df, val_df = train_test_split(
        train_df,
        test_size=args.validation_frac,
        random_state=RANDOM_STATE,
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
    y_val = fine_encoder.transform(val_df["Label"].astype(str))
    y_test = fine_encoder.transform(test_df["Label"].astype(str))
    rare_ids = rare_class_ids(y_fit)

    print("Fitting global, family, and family specific models...")
    global_model, family_model, local_models, train_seconds = fit_models(
        fit_df,
        features,
        fine_encoder,
        family_encoder,
        args.global_trees,
        args.local_trees,
    )
    control_map, control_shift = cyclic_control_map(label_to_family)
    print(
        "Fitting the cyclic taxonomy control "
        f"(shift={control_shift}, changed={sum(control_map[k] != label_to_family[k] for k in control_map)})..."
    )
    (
        control_family_model,
        control_family_encoder,
        control_local_models,
        control_train_seconds,
    ) = fit_control_hierarchy(
        fit_df,
        features,
        control_map,
        args.local_trees,
    )
    train_seconds += control_train_seconds

    X_val = clean_features(val_df, features)
    global_val = np.asarray(global_model.predict_proba(X_val), dtype=np.float64)
    hierarchy_val, family_val = hierarchy_probabilities(
        X_val,
        family_model,
        family_encoder,
        fine_encoder,
        label_to_family,
        local_models,
    )
    control_hierarchy_val, control_family_val = hierarchy_probabilities(
        X_val,
        control_family_model,
        control_family_encoder,
        fine_encoder,
        control_map,
        control_local_models,
    )
    selection, grid = select_global_gate(
        y_val,
        global_val,
        hierarchy_val,
        family_val,
        rare_ids,
        args.rare_lambda,
    )
    control_selection, control_grid = select_global_gate(
        y_val,
        global_val,
        control_hierarchy_val,
        control_family_val,
        rare_ids,
        args.rare_lambda,
    )
    class_beta, class_grid = select_support_regularised_class_gate(
        y_val,
        global_val,
        hierarchy_val,
        family_val,
        selection,
    )
    grid.to_csv(out_dir / "validation_global_gate_grid.csv", index=False)
    control_grid.to_csv(out_dir / "validation_cyclic_control_gate_grid.csv", index=False)
    class_grid.assign(
        class_label=[fine_encoder.classes_[i] for i in class_grid["class_id"]]
    ).to_csv(out_dir / "validation_class_gate.csv", index=False)

    X_test = clean_features(test_df, features)
    global_test = np.asarray(global_model.predict_proba(X_test), dtype=np.float64)
    hierarchy_test, family_test = hierarchy_probabilities(
        X_test,
        family_model,
        family_encoder,
        fine_encoder,
        label_to_family,
        local_models,
    )
    control_hierarchy_test, control_family_test = hierarchy_probabilities(
        X_test,
        control_family_model,
        control_family_encoder,
        fine_encoder,
        control_map,
        control_local_models,
    )
    global_gate = blend(
        global_test,
        hierarchy_test,
        family_test,
        selection.beta,
        selection.entropy_power,
        selection.hierarchy_power,
    )
    class_gate = blend(
        global_test,
        hierarchy_test,
        family_test,
        selection.beta,
        selection.entropy_power,
        selection.hierarchy_power,
        class_beta=class_beta,
    )
    control_gate = blend(
        global_test,
        control_hierarchy_test,
        control_family_test,
        control_selection.beta,
        control_selection.entropy_power,
        control_selection.hierarchy_power,
    )

    probability_map = {
        "global_weighted_xgboost": global_test,
        "taxonomy_factorisation": hierarchy_test,
        "global_taxonomy_blend": global_gate,
        "support_regularised_taxonomy_residual_fusion": class_gate,
        "cyclic_taxonomy_control": control_gate,
    }
    result_rows = [
        metric_row(run_id, y_test, probability, rare_ids)
        for run_id, probability in probability_map.items()
    ]
    results = pd.DataFrame(result_rows)
    results["train_seconds_total"] = train_seconds
    results.to_csv(out_dir / "screen_results.csv", index=False)
    pd.concat(
        [
            classwise_frame(run_id, y_test, probability, list(fine_encoder.classes_))
            for run_id, probability in probability_map.items()
        ],
        ignore_index=True,
    ).to_csv(out_dir / "screen_classwise_metrics.csv", index=False)

    metadata = {
        "screening_only": True,
        "train_rows_loaded": int(len(train_df)),
        "fit_rows": int(len(fit_df)),
        "validation_rows": int(len(val_df)),
        "development_test_rows": int(len(test_df)),
        "features": features,
        "fine_labels": list(fine_encoder.classes_),
        "family_labels": list(family_encoder.classes_),
        "label_to_family": label_to_family,
        "rare_labels": [str(fine_encoder.classes_[i]) for i in rare_ids],
        "selection": asdict(selection),
        "cyclic_control_selection": asdict(control_selection),
        "cyclic_control_shift": control_shift,
        "cyclic_control_map": control_map,
        "cyclic_control_changed_labels": int(
            sum(control_map[k] != label_to_family[k] for k in control_map)
        ),
        "class_beta": class_beta.tolist(),
        "arguments": vars(args),
        "warning": (
            "The historical public subsample test file has been used in earlier development. "
            "These results are screening evidence and are not an untouched confirmation."
        ),
    }
    (out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        out_dir / "screen_predictions.npz",
        y_true=y_test,
        rare_ids=rare_ids,
        **{name: np.argmax(prob, axis=1) for name, prob in probability_map.items()},
    )
    print(results.to_string(index=False))
    print(f"Selected gate: {selection}")
    print(f"Selected cyclic control gate: {control_selection}")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
