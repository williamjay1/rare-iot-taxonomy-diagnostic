from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, t as student_t


ROOT = Path(__file__).resolve().parents[1]
PRIMARY = "agreement_gated_taxonomy_residual_fusion"
COMPARATORS = [
    "flat_weighted_xgboost",
    "soft_taxonomy_factorisation",
    "equal_flat_taxonomy_blend",
    "global_taxonomy_residual_fusion",
    "cyclic_taxonomy_control",
]
SUPPORT_COMPARATOR = "support_matched_wrong_taxonomy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse agreement gated taxonomy residual fusion results."
    )
    parser.add_argument(
        "--result-dir",
        default="ijics_v2_ciciot_agreement_gated_trf_6split",
    )
    parser.add_argument(
        "--support-result-dir",
        default="ijics_v2_ciciot_support_matched_control_6split",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260726)
    return parser.parse_args()


def metrics_from_confusion(
    confusion: np.ndarray,
    rare_ids: np.ndarray,
) -> tuple[float, float, float]:
    true_support = confusion.sum(axis=1)
    predicted_support = confusion.sum(axis=0)
    diagonal = np.diag(confusion).astype(np.float64)
    precision = np.divide(
        diagonal,
        predicted_support,
        out=np.zeros_like(diagonal),
        where=predicted_support > 0,
    )
    recall = np.divide(
        diagonal,
        true_support,
        out=np.zeros_like(diagonal),
        where=true_support > 0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(diagonal),
        where=(precision + recall) > 0,
    )
    present = true_support > 0
    macro_f1 = float(np.mean(f1[present]))
    rare_recall = float(np.mean(recall[rare_ids]))
    accuracy = float(diagonal.sum() / true_support.sum())
    return macro_f1, rare_recall, accuracy


def confusion_from_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    encoded = y_true.astype(np.int64) * n_classes + y_pred.astype(np.int64)
    return np.bincount(encoded, minlength=n_classes * n_classes).reshape(
        n_classes,
        n_classes,
    )


def paired_stratified_bootstrap(
    y_true: np.ndarray,
    primary_pred: np.ndarray,
    comparator_pred: np.ndarray,
    rare_ids: np.ndarray,
    n_classes: int,
    reps: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, float]]:
    strata: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, int]] = []
    for true_class in range(n_classes):
        mask = y_true == true_class
        paired_id = (
            primary_pred[mask].astype(np.int64) * n_classes
            + comparator_pred[mask].astype(np.int64)
        )
        pair_values, counts = np.unique(paired_id, return_counts=True)
        strata.append(
            (
                true_class,
                pair_values // n_classes,
                pair_values % n_classes,
                counts / counts.sum(),
                int(counts.sum()),
            )
        )

    samples = np.empty((reps, 3), dtype=np.float64)
    for rep in range(reps):
        primary_confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
        comparator_confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
        for true_class, primary_ids, comparator_ids, probabilities, support in strata:
            sampled = rng.multinomial(support, probabilities)
            np.add.at(primary_confusion[true_class], primary_ids, sampled)
            np.add.at(comparator_confusion[true_class], comparator_ids, sampled)
        primary_metrics = metrics_from_confusion(primary_confusion, rare_ids)
        comparator_metrics = metrics_from_confusion(comparator_confusion, rare_ids)
        samples[rep] = np.asarray(primary_metrics) - np.asarray(comparator_metrics)

    primary_point = metrics_from_confusion(
        confusion_from_predictions(y_true, primary_pred, n_classes),
        rare_ids,
    )
    comparator_point = metrics_from_confusion(
        confusion_from_predictions(y_true, comparator_pred, n_classes),
        rare_ids,
    )
    point = np.asarray(primary_point) - np.asarray(comparator_point)
    columns = ["macro_f1_delta", "rare_recall_delta", "accuracy_delta"]
    sample_frame = pd.DataFrame(samples, columns=columns)
    summary: dict[str, float] = {}
    for idx, metric in enumerate(columns):
        lower, upper = np.quantile(samples[:, idx], [0.025, 0.975])
        summary[f"{metric}_point"] = float(point[idx])
        summary[f"{metric}_lower"] = float(lower)
        summary[f"{metric}_upper"] = float(upper)
        summary[f"{metric}_bootstrap_positive_probability"] = float(
            np.mean(samples[:, idx] > 0)
        )
    return sample_frame, summary


def exact_sign_flip_p(values: np.ndarray) -> float:
    observed = abs(float(np.mean(values)))
    signs = np.asarray(list(itertools.product([-1.0, 1.0], repeat=len(values))))
    permuted = np.abs(np.mean(signs * values[None, :], axis=1))
    return float(np.mean(permuted >= observed - 1e-15))


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * float(p_values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def corrected_repeated_holdout(
    deltas: np.ndarray,
    test_fraction: float,
) -> dict[str, float]:
    n_splits = len(deltas)
    mean_delta = float(np.mean(deltas))
    if n_splits < 2:
        return {
            "corrected_t": np.nan,
            "corrected_df": np.nan,
            "corrected_two_sided_p": np.nan,
            "corrected_ci_lower": np.nan,
            "corrected_ci_upper": np.nan,
        }
    variance = float(np.var(deltas, ddof=1))
    train_fraction = 1.0 - test_fraction
    correction = 1.0 / n_splits + test_fraction / train_fraction
    standard_error = float(np.sqrt(correction * variance))
    if standard_error == 0:
        statistic = np.inf if mean_delta != 0 else 0.0
        p_value = 0.0 if mean_delta != 0 else 1.0
        lower = upper = mean_delta
    else:
        statistic = mean_delta / standard_error
        p_value = float(
            2.0 * student_t.sf(abs(statistic), df=n_splits - 1)
        )
        critical = float(student_t.ppf(0.975, df=n_splits - 1))
        lower = mean_delta - critical * standard_error
        upper = mean_delta + critical * standard_error
    return {
        "corrected_t": float(statistic),
        "corrected_df": float(n_splits - 1),
        "corrected_two_sided_p": p_value,
        "corrected_ci_lower": float(lower),
        "corrected_ci_upper": float(upper),
    }


def main() -> None:
    args = parse_args()
    result_dir = ROOT / "results" / args.result_dir
    support_result_dir = ROOT / "results" / args.support_result_dir
    metadata = json.loads((result_dir / "run_metadata.json").read_text(encoding="utf-8"))
    labels = list(metadata["fine_labels"])
    n_classes = len(labels)
    audit = pd.read_csv(result_dir / "nested_split_audit.csv")
    results = pd.read_csv(result_dir / "nested_split_results.csv")
    support_results_path = support_result_dir / "support_matched_control_results.csv"
    support_available = support_results_path.exists()
    if support_available:
        support_results = pd.read_csv(support_results_path)
        results_for_deltas = pd.concat(
            [results, support_results],
            ignore_index=True,
        )
    else:
        support_results = pd.DataFrame()
        results_for_deltas = results.copy()

    rng = np.random.default_rng(args.seed)
    summary_rows: list[dict[str, object]] = []
    all_samples: list[pd.DataFrame] = []
    mcnemar_rows: list[dict[str, object]] = []
    for seed in sorted(results["seed"].unique()):
        archive = np.load(result_dir / f"paired_predictions_seed_{seed}.npz")
        y_true = archive["y_true"].astype(np.int64)
        primary_pred = archive[PRIMARY].astype(np.int64)
        rare_labels = json.loads(
            audit.loc[audit["seed"] == seed, "rare_labels"].iloc[0]
        )
        rare_ids = np.asarray([labels.index(label) for label in rare_labels], dtype=int)
        comparator_predictions = {
            comparator: archive[comparator].astype(np.int64)
            for comparator in COMPARATORS
        }
        support_archive_path = (
            support_result_dir
            / f"support_control_predictions_seed_{seed}.npz"
        )
        if support_archive_path.exists():
            support_archive = np.load(support_archive_path)
            if not np.array_equal(
                archive["row_id"].astype(np.int64),
                support_archive["row_id"].astype(np.int64),
            ):
                raise ValueError(
                    f"Outer row identifiers differ for support control seed {seed}."
                )
            if not np.array_equal(
                y_true,
                support_archive["y_true"].astype(np.int64),
            ):
                raise ValueError(
                    f"Outer labels differ for support control seed {seed}."
                )
            comparator_predictions[SUPPORT_COMPARATOR] = support_archive[
                SUPPORT_COMPARATOR
            ].astype(np.int64)

        for comparator, comparator_pred in comparator_predictions.items():
            samples, summary = paired_stratified_bootstrap(
                y_true,
                primary_pred,
                comparator_pred,
                rare_ids,
                n_classes,
                args.bootstrap_reps,
                rng,
            )
            samples.insert(0, "comparator", comparator)
            samples.insert(0, "seed", int(seed))
            all_samples.append(samples)
            summary_rows.append(
                {
                    "seed": int(seed),
                    "primary": PRIMARY,
                    "comparator": comparator,
                    "bootstrap_reps": args.bootstrap_reps,
                    **summary,
                }
            )

            primary_correct = primary_pred == y_true
            comparator_correct = comparator_pred == y_true
            primary_only = int(np.sum(primary_correct & ~comparator_correct))
            comparator_only = int(np.sum(~primary_correct & comparator_correct))
            discordant = primary_only + comparator_only
            p_value = (
                float(binomtest(primary_only, discordant, 0.5).pvalue)
                if discordant
                else 1.0
            )
            mcnemar_rows.append(
                {
                    "seed": int(seed),
                    "comparator": comparator,
                    "primary_only_correct": primary_only,
                    "comparator_only_correct": comparator_only,
                    "discordant": discordant,
                    "mcnemar_exact_two_sided_p": p_value,
                }
            )

    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(result_dir / "paired_bootstrap_summary.csv", index=False)
    pd.concat(all_samples, ignore_index=True).to_csv(
        result_dir / "paired_bootstrap_samples.csv",
        index=False,
    )
    mcnemar_frame = pd.DataFrame(mcnemar_rows)
    mcnemar_frame["mcnemar_holm_p"] = mcnemar_frame.groupby(
        "seed",
        group_keys=False,
    )["mcnemar_exact_two_sided_p"].transform(
        lambda values: holm_adjust(values.to_numpy(dtype=float))
    )
    mcnemar_frame.to_csv(
        result_dir / "paired_mcnemar.csv",
        index=False,
    )

    split_delta_rows: list[dict[str, object]] = []
    outer_test_fraction = float(metadata["arguments"]["outer_test_frac"])
    delta_comparators = COMPARATORS + (
        [SUPPORT_COMPARATOR] if support_available else []
    )
    for comparator in delta_comparators:
        primary_rows = results_for_deltas[
            results_for_deltas["run_id"] == PRIMARY
        ].set_index("seed")
        comparator_rows = results_for_deltas[
            results_for_deltas["run_id"] == comparator
        ].set_index("seed")
        for metric in ("f1_macro", "rare_recall", "accuracy"):
            deltas = (
                primary_rows.loc[comparator_rows.index, metric]
                - comparator_rows[metric]
            ).to_numpy(dtype=float)
            split_delta_rows.append(
                {
                    "comparator": comparator,
                    "metric": metric,
                    "splits": len(deltas),
                    "mean_delta": float(np.mean(deltas)),
                    "sd_delta": (
                        float(np.std(deltas, ddof=1))
                        if len(deltas) > 1
                        else np.nan
                    ),
                    "minimum_delta": float(np.min(deltas)),
                    "maximum_delta": float(np.max(deltas)),
                    "positive_splits": int(np.sum(deltas > 0)),
                    "nonnegative_splits": int(np.sum(deltas >= 0)),
                    "exact_sign_flip_p": exact_sign_flip_p(deltas),
                    **corrected_repeated_holdout(
                        deltas,
                        outer_test_fraction,
                    ),
                }
            )
    split_delta_frame = pd.DataFrame(split_delta_rows)
    split_delta_frame.to_csv(
        result_dir / "split_level_delta_summary.csv",
        index=False,
    )

    delta_index = split_delta_frame.set_index(["comparator", "metric"])
    equal_macro = delta_index.loc[("equal_flat_taxonomy_blend", "f1_macro")]
    equal_rare = delta_index.loc[("equal_flat_taxonomy_blend", "rare_recall")]
    equal_accuracy = delta_index.loc[
        ("equal_flat_taxonomy_blend", "accuracy")
    ]
    flat_macro = delta_index.loc[("flat_weighted_xgboost", "f1_macro")]
    flat_rare = delta_index.loc[("flat_weighted_xgboost", "rare_recall")]
    cyclic_macro = delta_index.loc[("cyclic_taxonomy_control", "f1_macro")]
    cyclic_rare = delta_index.loc[("cyclic_taxonomy_control", "rare_recall")]

    equal_blend_pass = bool(
        float(equal_macro["mean_delta"]) >= 0.001
        and float(equal_rare["mean_delta"]) >= 0.002
        and int(equal_macro["nonnegative_splits"]) >= 5
        and int(equal_rare["nonnegative_splits"]) >= 5
        and float(equal_accuracy["mean_delta"]) >= -0.001
    )
    flat_baseline_pass = bool(
        float(flat_macro["mean_delta"]) >= 0.015
        and float(flat_rare["mean_delta"]) >= 0.05
    )
    cyclic_control_pass = bool(
        float(cyclic_macro["mean_delta"]) >= 0.005
        and float(cyclic_rare["mean_delta"]) >= 0.02
    )
    if support_available:
        support_macro = delta_index.loc[(SUPPORT_COMPARATOR, "f1_macro")]
        support_rare = delta_index.loc[(SUPPORT_COMPARATOR, "rare_recall")]
        support_matched_pass = bool(
            float(support_macro["mean_delta"]) >= 0.005
            and float(support_rare["mean_delta"]) >= 0.02
        )
        support_matched_status = "PASS" if support_matched_pass else "FAIL"
    else:
        support_matched_pass = False
        support_matched_status = "PENDING"
    completed_required_failure = not (
        equal_blend_pass
        and flat_baseline_pass
        and cyclic_control_pass
    )
    if completed_required_failure:
        complete_gate_status = "FAIL"
    elif support_matched_status == "PENDING":
        complete_gate_status = "PENDING"
    elif support_matched_pass:
        complete_gate_status = "PASS"
    else:
        complete_gate_status = "FAIL"
    gate_pass = complete_gate_status == "PASS"

    note_lines = [
        "# Agreement gated taxonomy confirmation result",
        "",
        f"* Outer splits: {results['seed'].nunique()}",
        f"* Paired class stratified bootstrap repetitions per comparison: {args.bootstrap_reps}",
        f"* Increment over the equal blend: {'PASS' if equal_blend_pass else 'FAIL'}",
        f"* Increment over weighted flat XGBoost: {'PASS' if flat_baseline_pass else 'FAIL'}",
        f"* Increment over cyclic wrong taxonomy: {'PASS' if cyclic_control_pass else 'FAIL'}",
        f"* Support matched wrong taxonomy: {support_matched_status}",
        f"* Complete engineering novelty gate: {complete_gate_status}",
        "",
        "## Aggregate results",
        "",
        results.groupby("run_id")[
            ["f1_macro", "rare_recall", "accuracy", "pr_auc_macro"]
        ]
        .agg(["mean", "std"])
        .to_markdown(),
        "",
        "## Split level deltas",
        "",
        split_delta_frame.to_markdown(index=False),
        "",
        "The historical public subsample files were merged before nested repeated "
        "holdouts. Each outer test partition was excluded from fitting and inner "
        "selection within its split. The results do not constitute a temporal or "
        "device independent validation.",
    ]
    gate_payload = {
        "equal_blend_pass": equal_blend_pass,
        "flat_baseline_pass": flat_baseline_pass,
        "cyclic_control_pass": cyclic_control_pass,
        "support_matched_pass": support_matched_pass,
        "support_matched_status": support_matched_status,
        "complete_gate_status": complete_gate_status,
        "complete_gate_pass": gate_pass,
        "thresholds": {
            "equal_macro_f1_delta": 0.001,
            "equal_rare_recall_delta": 0.002,
            "equal_nonnegative_splits": 5,
            "maximum_accuracy_loss": 0.001,
            "weighted_flat_macro_f1_delta": 0.015,
            "weighted_flat_rare_recall_delta": 0.05,
            "cyclic_macro_f1_delta": 0.005,
            "cyclic_rare_recall_delta": 0.02,
            "support_matched_macro_f1_delta": 0.005,
            "support_matched_rare_recall_delta": 0.02,
        },
    }
    (result_dir / "innovation_gate.json").write_text(
        json.dumps(gate_payload, indent=2),
        encoding="utf-8",
    )
    (result_dir / "AGREEMENT_GATE_RESULT_NOTE.md").write_text(
        "\n".join(note_lines) + "\n",
        encoding="utf-8",
    )
    print("\n".join(note_lines[:9]))


if __name__ == "__main__":
    main()
