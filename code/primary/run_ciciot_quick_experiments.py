from __future__ import annotations

import argparse
import json
import pickle
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import BorderlineSMOTE, RandomOverSampler, SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from xgboost import XGBClassifier

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover - optional dependency
    CatBoostClassifier = None


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "raw" / "ciciot2023_hf_neto_subsample" / "random" / "train-00000-of-00001.parquet"
TEST_PATH = ROOT / "data" / "raw" / "ciciot2023_hf_neto_subsample" / "random" / "test-00000-of-00001.parquet"
OUT_DIR = ROOT / "results" / "ciciot_quick"
FIG_DIR = ROOT / "figures" / "ciciot_quick"
DOCS_DIR = ROOT / "docs"
LABEL_COLS = ["Label", "Label_orig", "attack_class", "label"]
RANDOM_STATE = 42


@dataclass
class RunConfig:
    run_id: str
    task: str
    target_col: str
    model_name: str
    imbalance: str = "none"
    feature_strategy: str = "all"
    top_k: int | None = None
    hierarchical: bool = False
    notes: str = ""


def ensure_dirs() -> None:
    for path in [OUT_DIR, FIG_DIR, DOCS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in LABEL_COLS and pd.api.types.is_numeric_dtype(df[c])]


def stratified_sample(df: pd.DataFrame, label_col: str, n: int | None) -> pd.DataFrame:
    if n is None or n >= len(df):
        return df.copy()
    parts = []
    counts = df[label_col].value_counts()
    for label, group in df.groupby(label_col, sort=False):
        frac = counts[label] / len(df)
        take = max(1, int(round(n * frac)))
        take = min(take, len(group))
        parts.append(group.sample(n=take, random_state=RANDOM_STATE))
    sampled = pd.concat(parts).sample(frac=1, random_state=RANDOM_STATE)
    if len(sampled) > n:
        sampled = sampled.sample(n=n, random_state=RANDOM_STATE)
    return sampled.reset_index(drop=True)


def load_data(train_n: int | None, test_n: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(TRAIN_PATH)
    test = pd.read_parquet(TEST_PATH)
    train = stratified_sample(train, "Label", train_n)
    test = stratified_sample(test, "Label", test_n)
    return train, test


def encode_target(y_train: pd.Series, y_test: pd.Series) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    enc = LabelEncoder()
    y_tr = enc.fit_transform(y_train.astype(str))
    y_te = enc.transform(y_test.astype(str))
    return y_tr, y_te, enc


def class_sample_weight(y: np.ndarray) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    weights = {cls: len(y) / (len(classes) * count) for cls, count in zip(classes, counts)}
    return np.array([weights[v] for v in y], dtype=np.float32)


def target_sampling_strategy(y: np.ndarray, task: str) -> dict[int, int]:
    classes, counts = np.unique(y, return_counts=True)
    if task == "fine":
        floor = 2500
        cap = 12000
    elif task == "category":
        floor = 7000
        cap = 25000
    else:
        floor = int(np.median(counts))
        cap = int(max(counts))
    strategy = {}
    for cls, count in zip(classes, counts):
        if count < floor:
            strategy[int(cls)] = min(cap, floor)
    return strategy


def apply_imbalance(
    X: np.ndarray,
    y: np.ndarray,
    task: str,
    imbalance: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, str]:
    if imbalance == "none":
        return X, y, None, "none"
    if imbalance == "class_weight":
        return X, y, class_sample_weight(y), "class_weight"
    strategy = target_sampling_strategy(y, task)
    if not strategy:
        return X, y, None, f"{imbalance}:no_small_classes"
    min_count = min(np.bincount(y))
    k_neighbors = max(1, min(5, min_count - 1))
    if imbalance == "random_over":
        sampler = RandomOverSampler(sampling_strategy=strategy, random_state=RANDOM_STATE)
    elif imbalance == "smote":
        sampler = SMOTE(sampling_strategy=strategy, random_state=RANDOM_STATE, k_neighbors=k_neighbors)
    elif imbalance == "borderline_smote":
        sampler = BorderlineSMOTE(sampling_strategy=strategy, random_state=RANDOM_STATE, k_neighbors=k_neighbors)
    elif imbalance == "smoteenn":
        sampler = SMOTEENN(
            sampling_strategy=strategy,
            random_state=RANDOM_STATE,
            smote=SMOTE(sampling_strategy=strategy, random_state=RANDOM_STATE, k_neighbors=k_neighbors),
        )
    else:
        raise ValueError(f"Unknown imbalance strategy: {imbalance}")
    X_res, y_res = sampler.fit_resample(X, y)
    return X_res, y_res, None, f"{imbalance}:{json.dumps(strategy, sort_keys=True)}"


def make_model(name: str, n_classes: int) -> object:
    if name == "lgbm":
        objective = "binary" if n_classes == 2 else "multiclass"
        return LGBMClassifier(
            objective=objective,
            n_estimators=220,
            learning_rate=0.06,
            num_leaves=63,
            max_depth=-1,
            min_child_samples=40,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=160,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            class_weight=None,
        )
    if name == "extratrees":
        return ExtraTreesClassifier(
            n_estimators=180,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            class_weight=None,
        )
    if name == "xgb":
        objective = "binary:logistic" if n_classes == 2 else "multi:softprob"
        params = {
            "objective": objective,
            "n_estimators": 220,
            "max_depth": 7,
            "learning_rate": 0.08,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "tree_method": "hist",
            "eval_metric": "logloss" if n_classes == 2 else "mlogloss",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }
        if n_classes > 2:
            params["num_class"] = n_classes
        return XGBClassifier(**params)
    if name == "catboost":
        if CatBoostClassifier is None:
            raise RuntimeError("catboost is not installed")
        return CatBoostClassifier(
            iterations=220,
            depth=7,
            learning_rate=0.08,
            loss_function="Logloss" if n_classes == 2 else "MultiClass",
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )
    raise ValueError(f"Unknown model: {name}")


def model_size_kb(model: object) -> float:
    try:
        return len(pickle.dumps(model)) / 1024
    except Exception:
        return float("nan")


def predict_proba_safe(model: object, X: np.ndarray, n_classes: int) -> np.ndarray | None:
    if not hasattr(model, "predict_proba"):
        return None
    proba = model.predict_proba(X)
    if isinstance(proba, list):
        return None
    proba = np.asarray(proba)
    if n_classes == 2 and proba.ndim == 1:
        proba = np.column_stack([1 - proba, proba])
    return proba


def metric_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray | None,
    labels: list[str],
    train_seconds: float,
    inference_seconds: float,
    n_features: int,
    model_kb: float,
) -> dict[str, float | int | str]:
    n_classes = len(labels)
    out: dict[str, float | int | str] = {
        "n_test": int(len(y_true)),
        "n_classes": int(n_classes),
        "n_features": int(n_features),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "train_seconds": train_seconds,
        "inference_seconds": inference_seconds,
        "inference_ms_per_1000": inference_seconds / max(1, len(y_true)) * 1_000_000,
        "model_size_kb": model_kb,
    }
    counts = np.bincount(y_true, minlength=n_classes)
    minority_classes = np.where(counts == counts[counts > 0].min())[0]
    recalls = recall_score(y_true, y_pred, labels=list(range(n_classes)), average=None, zero_division=0)
    out["minority_recall_mean"] = float(np.mean(recalls[minority_classes]))
    out["worst_class_recall"] = float(np.min(recalls[counts > 0]))
    out["worst_class_label"] = labels[int(np.argmin(np.where(counts > 0, recalls, np.inf)))]
    if n_classes == 2:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            out["false_positive_rate"] = fp / max(1, fp + tn)
            out["false_negative_rate"] = fn / max(1, fn + tp)
        if proba is not None:
            out["roc_auc"] = roc_auc_score(y_true, proba[:, 1])
            out["pr_auc"] = average_precision_score(y_true, proba[:, 1])
    else:
        if proba is not None:
            try:
                lb = LabelBinarizer()
                y_bin = lb.fit_transform(y_true)
                if y_bin.shape[1] == proba.shape[1]:
                    out["roc_auc_ovr_macro"] = roc_auc_score(y_bin, proba, average="macro", multi_class="ovr")
                    out["pr_auc_macro"] = average_precision_score(y_bin, proba, average="macro")
                    out["pr_auc_weighted"] = average_precision_score(y_bin, proba, average="weighted")
            except Exception as exc:
                out["auc_error"] = str(exc)
    return out


def classwise_frame(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], run_id: str) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "run_id": run_id,
            "class_id": list(range(len(labels))),
            "class_label": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )


def rank_features(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    strategy: str,
    max_rows: int = 60000,
) -> list[str]:
    if strategy == "all":
        return list(X_train.columns)
    sample_idx = np.arange(len(X_train))
    if len(sample_idx) > max_rows:
        rng = np.random.default_rng(RANDOM_STATE)
        sample_idx = rng.choice(sample_idx, size=max_rows, replace=False)
    X_sample = X_train.iloc[sample_idx]
    y_sample = y_train[sample_idx]
    if strategy == "mi":
        scores = mutual_info_classif(X_sample, y_sample, random_state=RANDOM_STATE, discrete_features=False)
        order = np.argsort(scores)[::-1]
        return [X_train.columns[i] for i in order]
    if strategy == "importance":
        model = LGBMClassifier(
            objective="multiclass" if len(np.unique(y_train)) > 2 else "binary",
            n_estimators=120,
            learning_rate=0.08,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(X_sample, y_sample)
        order = np.argsort(model.feature_importances_)[::-1]
        return [X_train.columns[i] for i in order]
    raise ValueError(f"Unknown feature ranking strategy: {strategy}")


def select_features(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    config: RunConfig,
    cache: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if config.feature_strategy == "all":
        cols = list(X_train.columns)
    else:
        if config.feature_strategy not in cache:
            cache[config.feature_strategy] = rank_features(X_train, y_train, config.feature_strategy)
        top_k = config.top_k or len(X_train.columns)
        cols = cache[config.feature_strategy][:top_k]
    return X_train[cols], X_test[cols], cols


def fit_predict_run(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: RunConfig,
    rank_cache: dict[tuple[str, str], list[str]],
) -> tuple[dict[str, object], pd.DataFrame]:
    base_features = feature_cols(train)
    X_train_all = train[base_features].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test_all = test[base_features].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train, y_test, enc = encode_target(train[config.target_col], test[config.target_col])
    labels = list(enc.classes_)
    cache_key = (config.target_col, config.feature_strategy)
    local_cache = {}
    if cache_key in rank_cache:
        local_cache[config.feature_strategy] = rank_cache[cache_key]
    X_train_df, X_test_df, cols = select_features(X_train_all, y_train, X_test_all, config, local_cache)
    if config.feature_strategy != "all":
        rank_cache[cache_key] = local_cache[config.feature_strategy]
    X_train_np = X_train_df.to_numpy(dtype=np.float32)
    X_test_np = X_test_df.to_numpy(dtype=np.float32)
    X_fit, y_fit, sample_weight, sampling_note = apply_imbalance(
        X_train_np,
        y_train,
        config.task,
        config.imbalance,
    )
    model = make_model(config.model_name, len(labels))
    start = time.perf_counter()
    if sample_weight is not None:
        model.fit(X_fit, y_fit, sample_weight=sample_weight)
    else:
        model.fit(X_fit, y_fit)
    train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    y_pred = model.predict(X_test_np)
    y_pred = np.asarray(y_pred).reshape(-1).astype(int)
    proba = predict_proba_safe(model, X_test_np, len(labels))
    inference_seconds = time.perf_counter() - start
    metrics = metric_dict(
        y_test,
        y_pred,
        proba,
        labels,
        train_seconds=train_seconds,
        inference_seconds=inference_seconds,
        n_features=len(cols),
        model_kb=model_size_kb(model),
    )
    metrics.update(asdict(config))
    metrics["sampling_note"] = sampling_note
    metrics["train_rows"] = int(len(y_train))
    metrics["fit_rows"] = int(len(y_fit))
    metrics["test_rows"] = int(len(y_test))
    metrics["feature_names"] = json.dumps(cols)
    cw = classwise_frame(y_test, y_pred, labels, config.run_id)
    return metrics, cw


def fit_binary_model(train: pd.DataFrame, features: list[str]) -> tuple[object, list[str]]:
    y = train["label"].to_numpy(dtype=int)
    X = train[features].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=np.float32)
    model = make_model("lgbm", 2)
    model.fit(X, y, sample_weight=class_sample_weight(y))
    return model, ["BenignTraffic", "Attack"]


def fit_category_model(train_attack: pd.DataFrame, features: list[str]) -> tuple[object, LabelEncoder]:
    y, _, enc = encode_target(train_attack["attack_class"], train_attack["attack_class"])
    X = train_attack[features].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=np.float32)
    model = make_model("lgbm", len(enc.classes_))
    model.fit(X, y, sample_weight=class_sample_weight(y))
    return model, enc


def fit_fine_models_by_category(train: pd.DataFrame, features: list[str]) -> dict[str, tuple[object, LabelEncoder, str]]:
    models: dict[str, tuple[object, LabelEncoder, str]] = {}
    for category, group in train[train["attack_class"] != "Benign"].groupby("attack_class"):
        labels = group["Label"].astype(str)
        majority = labels.value_counts().idxmax()
        if labels.nunique() < 2 or len(group) < 100:
            models[str(category)] = (None, None, majority)  # type: ignore[assignment]
            continue
        y, _, enc = encode_target(labels, labels)
        X = group[features].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=np.float32)
        model = make_model("lgbm", len(enc.classes_))
        model.fit(X, y, sample_weight=class_sample_weight(y))
        models[str(category)] = (model, enc, majority)
    return models


def fit_predict_hierarchical(train: pd.DataFrame, test: pd.DataFrame, top_k: int | None = 20) -> tuple[dict[str, object], pd.DataFrame]:
    start_total = time.perf_counter()
    base_features = feature_cols(train)
    y_fine_train, y_fine_test, fine_enc = encode_target(train["Label"], test["Label"])
    rank = rank_features(
        train[base_features].replace([np.inf, -np.inf], np.nan).fillna(0),
        y_fine_train,
        "importance",
        max_rows=70000,
    )
    selected = rank[:top_k] if top_k else rank
    start = time.perf_counter()
    binary_model, _ = fit_binary_model(train, selected)
    train_attack = train[train["label"] == 1].copy()
    category_model, category_enc = fit_category_model(train_attack, selected)
    fine_models = fit_fine_models_by_category(train, selected)
    train_seconds = time.perf_counter() - start
    X_test = test[selected].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=np.float32)
    start = time.perf_counter()
    binary_pred = np.asarray(binary_model.predict(X_test)).reshape(-1).astype(int)
    y_pred_label: list[str] = []
    attack_idx = np.where(binary_pred == 1)[0]
    category_pred_names: dict[int, str] = {}
    if len(attack_idx):
        cat_pred = category_model.predict(X_test[attack_idx])
        cat_pred = np.asarray(cat_pred).reshape(-1).astype(int)
        for idx, cat_id in zip(attack_idx, cat_pred):
            category_pred_names[int(idx)] = str(category_enc.inverse_transform([cat_id])[0])
    for idx, is_attack in enumerate(binary_pred):
        if is_attack == 0:
            y_pred_label.append("BenignTraffic")
            continue
        category = category_pred_names.get(idx)
        model, enc, majority = fine_models.get(str(category), (None, None, "DDoS-ICMP_Flood"))
        if model is None or enc is None:
            y_pred_label.append(majority)
        else:
            pred_id = int(np.asarray(model.predict(X_test[idx : idx + 1])).reshape(-1)[0])
            y_pred_label.append(str(enc.inverse_transform([pred_id])[0]))
    inference_seconds = time.perf_counter() - start
    y_pred = fine_enc.transform(pd.Series(y_pred_label).astype(str))
    labels = list(fine_enc.classes_)
    metrics = metric_dict(
        y_fine_test,
        y_pred,
        None,
        labels,
        train_seconds=train_seconds,
        inference_seconds=inference_seconds,
        n_features=len(selected),
        model_kb=float("nan"),
    )
    config = RunConfig(
        run_id="fine_hier_lgbm_top20_weighted",
        task="fine",
        target_col="Label",
        model_name="lgbm",
        imbalance="class_weight",
        feature_strategy="importance",
        top_k=top_k,
        hierarchical=True,
        notes="Binary attack gate, attack-class classifier, and per-category fine classifiers.",
    )
    metrics.update(asdict(config))
    metrics["sampling_note"] = "class_weight_per_stage"
    metrics["train_rows"] = int(len(train))
    metrics["fit_rows"] = int(len(train))
    metrics["test_rows"] = int(len(test))
    metrics["feature_names"] = json.dumps(selected)
    metrics["total_seconds"] = time.perf_counter() - start_total
    return metrics, classwise_frame(y_fine_test, y_pred, labels, config.run_id)


def experiment_configs(include_model_zoo: bool) -> list[RunConfig]:
    configs = [
        RunConfig("binary_lgbm_full_weighted", "binary", "label", "lgbm", imbalance="class_weight"),
        RunConfig("category_lgbm_full", "category", "attack_class", "lgbm"),
        RunConfig("category_lgbm_full_weighted", "category", "attack_class", "lgbm", imbalance="class_weight"),
        RunConfig("fine_lgbm_full", "fine", "Label", "lgbm"),
        RunConfig("fine_lgbm_full_weighted", "fine", "Label", "lgbm", imbalance="class_weight"),
        RunConfig("fine_lgbm_random_over", "fine", "Label", "lgbm", imbalance="random_over"),
        RunConfig("fine_lgbm_smote", "fine", "Label", "lgbm", imbalance="smote"),
        RunConfig("fine_lgbm_borderline_smote", "fine", "Label", "lgbm", imbalance="borderline_smote"),
        RunConfig("fine_lgbm_smoteenn", "fine", "Label", "lgbm", imbalance="smoteenn"),
        RunConfig("fine_lgbm_mi_top20", "fine", "Label", "lgbm", feature_strategy="mi", top_k=20),
        RunConfig("fine_lgbm_mi_top20_weighted", "fine", "Label", "lgbm", imbalance="class_weight", feature_strategy="mi", top_k=20),
        RunConfig("fine_lgbm_mi_top10_weighted", "fine", "Label", "lgbm", imbalance="class_weight", feature_strategy="mi", top_k=10),
        RunConfig("fine_lgbm_importance_top20_weighted", "fine", "Label", "lgbm", imbalance="class_weight", feature_strategy="importance", top_k=20),
        RunConfig("fine_lgbm_importance_top10_weighted", "fine", "Label", "lgbm", imbalance="class_weight", feature_strategy="importance", top_k=10),
    ]
    if include_model_zoo:
        configs.extend(
            [
                RunConfig("fine_rf_full_weighted", "fine", "Label", "rf", imbalance="class_weight"),
                RunConfig("fine_extratrees_full_weighted", "fine", "Label", "extratrees", imbalance="class_weight"),
                RunConfig("fine_xgb_full_weighted", "fine", "Label", "xgb", imbalance="class_weight"),
                RunConfig("fine_catboost_full_weighted", "fine", "Label", "catboost", imbalance="class_weight"),
            ]
        )
    return configs


def write_design_docs(train: pd.DataFrame, test: pd.DataFrame, train_n: int | None, test_n: int | None) -> None:
    route_card = """# Venue Route Card

| Field | Decision |
|---|---|
| Route | SCI |
| Reason | The core contribution is applied-scientific prediction and validation for IoT intrusion detection, not social-science causal explanation. |
| Primary standard | Data provenance, leakage-safe preprocessing, strong baselines, multi-metric evaluation, external/entity-held-out validation, explainability, and reproducibility. |
| Do not use | Do not judge the work by SSCI causal identification standards or present engineering deployment as the main contribution. |
| Next workflow | sci-full-workflow |
| Immediate Red risks | Public mirror provenance, weak external validation, overclaiming from a subsample, and reporting only accuracy. |
"""
    (DOCS_DIR / "VENUE_ROUTE_CARD.md").write_text(route_card, encoding="utf-8")
    class_counts = train["Label"].value_counts().to_frame("train_count").join(
        test["Label"].value_counts().to_frame("test_count"), how="outer"
    )
    class_counts.to_csv(OUT_DIR / "ciciot_label_distribution.csv")
    experiment_gate = f"""# Experiment Design Gate

## Scope

This first executable stage runs CICIoT2023 quick experiments before manuscript drafting.

## Data

- Train rows used: {len(train):,}
- Test rows used: {len(test):,}
- Train sampling argument: {train_n}
- Test sampling argument: {test_n}
- Feature columns: {len(feature_cols(train))}
- Binary target: `label`
- Attack-class target: `attack_class`
- Fine-grained target: `Label`

## Experiment Matrix

1. Binary attack detection with weighted LightGBM.
2. 8-class attack category classification with and without class weights.
3. Flat 34-class fine-grained classification baseline.
4. Imbalance handling: class weight, random oversampling, SMOTE, Borderline-SMOTE.
5. Feature-efficient variants: MI top-20/top-10 and LightGBM-importance top-20/top-10.
6. Model family check: RF, ExtraTrees, XGBoost, CatBoost when enabled.
7. Hierarchical classifier: binary gate -> attack class -> per-category fine label.

## Primary Metrics

- Fine-grained task: Macro-F1, minority recall mean, worst-class recall, macro PR-AUC.
- Binary task: PR-AUC, ROC-AUC, false positive rate, MCC.
- Lightweight task: Macro-F1 and PR-AUC retention, feature count, inference time.

## Red-Risk Controls

- The test split from the Hugging Face mirror is kept frozen.
- Feature selection is fitted only on the training split.
- Metrics include Macro-F1 and classwise recall, not only accuracy.
- Claims from this quick stage must be treated as preliminary until full-data or external validation is complete.
"""
    (DOCS_DIR / "EXPERIMENT_DESIGN_GATE.md").write_text(experiment_gate, encoding="utf-8")


def summarize_results(results: pd.DataFrame) -> str:
    cols = [
        "run_id",
        "task",
        "model_name",
        "imbalance",
        "feature_strategy",
        "top_k",
        "n_features",
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "f1_weighted",
        "minority_recall_mean",
        "worst_class_recall",
        "worst_class_label",
        "pr_auc_macro",
        "train_seconds",
        "inference_ms_per_1000",
    ]
    use_cols = [c for c in cols if c in results.columns]
    fine = results[results["task"] == "fine"].sort_values(["f1_macro", "minority_recall_mean"], ascending=False)
    lines = ["# CICIoT2023 Quick Result Ledger", ""]
    lines.append("## Top Fine-Grained Runs")
    lines.append("")
    lines.append(fine[use_cols].head(12).to_markdown(index=False))
    lines.append("")
    if "fine_lgbm_full" in set(results["run_id"]):
        base = results.loc[results["run_id"] == "fine_lgbm_full"].iloc[0]
        lines.append("## Main Comparisons Against Flat LightGBM Baseline")
        lines.append("")
        for _, row in fine.iterrows():
            if row["run_id"] == "fine_lgbm_full":
                continue
            delta = row["f1_macro"] - base["f1_macro"]
            rec_delta = row["minority_recall_mean"] - base["minority_recall_mean"]
            feat = int(row["n_features"])
            lines.append(
                f"- `{row['run_id']}`: Macro-F1 delta {delta:+.4f}, "
                f"minority-recall delta {rec_delta:+.4f}, features {feat}."
            )
    lines.append("")
    lines.append("## Claim Control")
    lines.append("")
    lines.append("- Strong claims are not allowed from this quick stage alone.")
    lines.append("- A publishable claim needs either full CICIoT2023 confirmation or independent external IoT validation.")
    lines.append("- If a lightweight/top-k run preserves Macro-F1 with lower latency, it can support a feature-efficient auxiliary claim.")
    lines.append("- If hierarchical classification beats the flat 34-class baseline on Macro-F1 or rare-class recall, it can become an independent paper route.")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run quick CICIoT2023 experiments.")
    parser.add_argument("--train-n", type=int, default=220_000)
    parser.add_argument("--test-n", type=int, default=80_000)
    parser.add_argument("--include-model-zoo", action="store_true")
    parser.add_argument("--skip-hierarchical", action="store_true")
    return parser.parse_args()


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    ensure_dirs()
    args = parse_args()
    train, test = load_data(args.train_n, args.test_n)
    write_design_docs(train, test, args.train_n, args.test_n)
    configs = experiment_configs(args.include_model_zoo)
    results: list[dict[str, object]] = []
    classwise: list[pd.DataFrame] = []
    rank_cache: dict[tuple[str, str], list[str]] = {}
    for config in configs:
        print(f"Running {config.run_id}...")
        try:
            metrics, cw = fit_predict_run(train, test, config, rank_cache)
            results.append(metrics)
            classwise.append(cw)
            pd.DataFrame(results).to_csv(OUT_DIR / "experiment_results_partial.csv", index=False)
        except Exception as exc:
            error_row = asdict(config)
            error_row["error"] = repr(exc)
            results.append(error_row)
            print(f"ERROR {config.run_id}: {exc!r}")
    if not args.skip_hierarchical:
        print("Running fine_hier_lgbm_top20_weighted...")
        try:
            metrics, cw = fit_predict_hierarchical(train, test, top_k=20)
            results.append(metrics)
            classwise.append(cw)
        except Exception as exc:
            error_row = asdict(
                RunConfig(
                    "fine_hier_lgbm_top20_weighted",
                    "fine",
                    "Label",
                    "lgbm",
                    imbalance="class_weight",
                    feature_strategy="importance",
                    top_k=20,
                    hierarchical=True,
                )
            )
            error_row["error"] = repr(exc)
            results.append(error_row)
            print(f"ERROR hierarchical: {exc!r}")
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "experiment_results.csv", index=False)
    if classwise:
        pd.concat(classwise, ignore_index=True).to_csv(OUT_DIR / "classwise_metrics.csv", index=False)
    (OUT_DIR / "RESULT_LEDGER.md").write_text(summarize_results(results_df), encoding="utf-8")
    print(f"Wrote {OUT_DIR / 'experiment_results.csv'}")
    print(f"Wrote {OUT_DIR / 'RESULT_LEDGER.md'}")


if __name__ == "__main__":
    main()
