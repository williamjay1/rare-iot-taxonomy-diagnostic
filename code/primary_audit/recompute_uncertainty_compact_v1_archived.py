"""Conditional paired intervals and overlap-aware sensitivity from frozen rows."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"

import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.metrics import f1_score, recall_score
from threadpoolctl import threadpool_limits

BASE = Path(r"workspace\primary_audit")
ROOT = Path(r"legacy_project\results")
N_BOOT = 5000
BLOCK_ROWS = 2000
RANDOM_STATE = 20261018
CONTRASTS = {
    "weighted_flat": "flat_weighted_xgboost",
    "converged_stacker": "flat_stacker",
    "cyclic_gate": "calibrated_cyclic_taxonomy_control",
    "support_matched_gate": "calibrated_support_matched_control",
}
TARGET = "equal_flat_taxonomy_blend"


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scores(cm, rare):
    cm = np.asarray(cm, dtype=float)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    support = cm.sum(axis=-1)
    predicted = cm.sum(axis=-2)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros_like(tp), where=(support + predicted) > 0).mean(axis=-1)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    return np.stack((f1, recall[..., rare].mean(axis=-1)), axis=-1)


def confusion(y, pred, classes):
    return np.bincount(y * classes + pred, minlength=classes**2).reshape(classes, classes)


def row_bootstrap(y, a, b, classes, rare, generator):
    ca = np.zeros((N_BOOT, classes, classes), dtype=np.int64)
    cb = np.zeros_like(ca)
    for label in range(classes):
        mask = y == label
        count = np.bincount(a[mask] * classes + b[mask], minlength=classes**2)
        occupied = np.flatnonzero(count)
        draw = generator.multinomial(int(mask.sum()), count[occupied] / count.sum(), size=N_BOOT)
        for index, pair in enumerate(occupied):
            left, right = divmod(int(pair), classes)
            ca[:, label, left] += draw[:, index]
            cb[:, label, right] += draw[:, index]
    return scores(ca, rare) - scores(cb, rare)


def block_bootstrap(y, a, b, row_ids, classes, rare, generator):
    order = np.argsort(row_ids, kind="stable")
    y, a, b = y[order], a[order], b[order]
    block = np.arange(len(y)) // BLOCK_ROWS
    count = int(block.max()) + 1
    shape = (count, classes**2)
    ca = sparse.coo_matrix((np.ones(len(y)), (block, y * classes + a)), shape=shape).tocsr()
    cb = sparse.coo_matrix((np.ones(len(y)), (block, y * classes + b)), shape=shape).tocsr()
    weights = generator.multinomial(count, np.repeat(1 / count, count), size=N_BOOT)
    result = np.empty((N_BOOT, 2))
    for start in range(0, N_BOOT, 250):
        w = weights[start:start + 250]
        left = ca.T.dot(w.T).T.reshape(-1, classes, classes)
        right = cb.T.dot(w.T).T.reshape(-1, classes, classes)
        result[start:start + len(w)] = scores(left, rare) - scores(right, rare)
    return result


def holm(values):
    values = np.asarray(values, float)
    order = np.argsort(values)
    corrected = np.maximum.accumulate(values[order] * (len(values) - np.arange(len(values))))
    result = np.empty(len(values))
    result[order] = np.minimum(1., corrected)
    return result


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    (BASE / "inputs").mkdir(exist_ok=True)
    assert BASE.resolve().drive.upper() == "D:"
    protocol = {
        "resamples": N_BOOT, "seed": RANDOM_STATE, "source_order_block_rows": BLOCK_ROWS,
        "point": "Mean of six per-holdout metric differences, not metric of pooled predictions",
        "row_interval": "Within each fixed holdout, class-stratified paired multinomial resampling of joint true/prediction cells; then average six resampled metric differences",
        "block_interval": "Within each fixed holdout sort by original source row ID, draw contiguous 2000-row blocks with replacement; then average six metric differences",
        "conditional_scope": "Intervals condition on fitted models and six selected holdouts; repeated source records across holdouts are not independent populations",
        "nadeau_sensitivity": "Approximate corrected repeated-holdout t interval; variance multiplier 1/6+0.20/0.80; df=5; no claim it recovers unavailable capture dependence",
        "sign_test": "Exact binomial direction summary under a hypothetical independent symmetric split reference; descriptive because holdouts overlap; Holm over eight metric/contrast summaries",
        "mcnemar": "Exact paired correctness test within each split; Holm over four declared contrasts separately in each split; no minimum-p pooling across splits",
        "contrasts": list(CONTRASTS),
        "precision": "Core methods use original saved hard predictions. Stacker uses argmax of saved float16 probabilities; compare its reconstructed metrics with original full-precision CSV and record sensitivity.",
    }
    path = BASE / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == protocol
    else:
        path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    arrays, provenance, precision = [], [], []
    original = pd.read_csv(ROOT / "ijismd_v10_grouped_newton_flat_stacker_6split" / "newton_flat_stacker_results.csv")
    for seed in range(42, 48):
        sources = {
            "core": ROOT / "jit_compact_prediction_ledgers_v1" / "ciciot2023_public_subsample" / f"paired_predictions_seed_{seed}.npz",
            "stacker": ROOT / "ijismd_v10_grouped_newton_flat_stacker_6split" / f"paired_predictions_seed_{seed}.npz",
        }
        cached = {}
        for name, source in sources.items():
            target = BASE / "inputs" / f"{name}_{seed}.npz"
            if not target.exists():
                shutil.copyfile(source, target)
            digest = file_hash(target)
            assert digest == file_hash(source)
            cached[name] = np.load(target)
            provenance.append({"seed": seed, "kind": name, "source": str(source), "sha256": digest})
        c, s = cached["core"], cached["stacker"]
        for name in ("row_id", "y_true", "rare_ids"):
            assert np.array_equal(c[name], s[name]), (seed, name)
        current = {key: c[key].astype(np.int64) for key in ("row_id", "y_true", "rare_ids", TARGET, *[x for x in CONTRASTS.values() if x != "flat_stacker"])}
        current["flat_stacker"] = s["flat_newton_log_probability_stacker_probability"].argmax(axis=1).astype(np.int64)
        current["seed"] = seed
        arrays.append(current)
        o = original[(original.seed == seed) & (original.method == "flat_newton_log_probability_stacker")].iloc[0]
        reconstructed = scores(confusion(current["y_true"], current["flat_stacker"], 34), current["rare_ids"])
        precision.append({"seed": seed, "cached_macro_f1": reconstructed[0], "original_macro_f1": o.f1_macro,
                          "cached_rare_recall": reconstructed[1], "original_rare_recall": o.rare_recall,
                          "macro_f1_difference": reconstructed[0] - o.f1_macro, "rare_recall_difference": reconstructed[1] - o.rare_recall})
        c.close()
        s.close()
    pd.DataFrame(provenance).to_csv(BASE / "input_manifest.csv", index=False)
    pd.DataFrame(precision).to_csv(BASE / "stacker_precision_sensitivity.csv", index=False)
    rng = np.random.default_rng(RANDOM_STATE)
    rows, split_rows, mcnemar = [], [], []
    for name, reference in CONTRASTS.items():
        row_draw, block_draw = np.zeros((N_BOOT, 2)), np.zeros((N_BOOT, 2))
        differences = []
        for current in arrays:
            y, rare, a, b = (current[x] for x in ("y_true", "rare_ids", TARGET, reference))
            delta = scores(confusion(y, a, 34), rare) - scores(confusion(y, b, 34), rare)
            assert abs(scores(confusion(y, a, 34), rare)[0] - f1_score(y, a, average="macro")) < 1e-12
            assert abs(scores(confusion(y, a, 34), rare)[1] - recall_score(y, a, labels=np.arange(34), average=None, zero_division=0)[rare].mean()) < 1e-12
            differences.append(delta)
            row_draw += row_bootstrap(y, a, b, 34, rare, rng) / 6
            block_draw += block_bootstrap(y, a, b, current["row_id"], 34, rare, rng) / 6
            correct_a, correct_b = a == y, b == y
            repairs = int((correct_a & ~correct_b).sum())
            harms = int((~correct_a & correct_b).sum())
            p = stats.binomtest(repairs, repairs + harms, .5).pvalue if repairs + harms else 1.
            mcnemar.append({"seed": current["seed"], "contrast": name, "repairs": repairs, "harms": harms, "discordant": repairs + harms, "exact_p": float(p)})
            for j, metric in enumerate(("macro_f1", "rare_recall")):
                split_rows.append({"seed": current["seed"], "contrast": name, "metric": metric, "difference": delta[j]})
            print(f"AUDITED {name} seed{current['seed']}", flush=True)
        d = np.array(differences)
        for j, metric in enumerate(("macro_f1", "rare_recall")):
            mean, sd = float(d[:, j].mean()), float(d[:, j].std(ddof=1))
            corrected_se = np.sqrt((1 / 6 + .2 / .8) * sd**2)
            half = stats.t.ppf(.975, 5) * corrected_se
            positive, negative = int((d[:, j] > 0).sum()), int((d[:, j] < 0).sum())
            nonzero = positive + negative
            sign_p = stats.binomtest(positive, nonzero, .5).pvalue if nonzero else 1.
            rows.append({"contrast": name, "metric": metric, "mean_difference": mean, "sd_difference": sd,
                         "row_ci_low": np.quantile(row_draw[:, j], .025), "row_ci_high": np.quantile(row_draw[:, j], .975),
                         "block_ci_low": np.quantile(block_draw[:, j], .025), "block_ci_high": np.quantile(block_draw[:, j], .975),
                         "corrected_t_ci_low": mean - half, "corrected_t_ci_high": mean + half,
                         "positive_splits": positive, "negative_splits": negative, "sign_reference_p": sign_p})
    summary = pd.DataFrame(rows)
    summary["sign_reference_holm_p"] = holm(summary.sign_reference_p)
    summary.to_csv(BASE / "paired_uncertainty.csv", index=False)
    pd.DataFrame(split_rows).to_csv(BASE / "split_effects.csv", index=False)
    m = pd.DataFrame(mcnemar)
    m["within_split_four_contrast_holm_p"] = np.nan
    for seed, group in m.groupby("seed"):
        m.loc[group.index, "within_split_four_contrast_holm_p"] = holm(group.exact_p)
    m.to_csv(BASE / "mcnemar_by_split.csv", index=False)
    (BASE / "COMPLETED.json").write_text(json.dumps({"contrasts": 4, "metric_rows": 8, "seeds": 6, "replicates": N_BOOT,
        "block_rows": BLOCK_ROWS, "all_metric_code_checks_passed": True,
        "source_protocol_sha256": file_hash(path)}, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
