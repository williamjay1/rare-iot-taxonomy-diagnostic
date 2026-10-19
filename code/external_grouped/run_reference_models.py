"""Matched classifiers on separately frozen grouped NF-ToN allocations."""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import warnings

sys.dont_write_bytecode = True
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, recall_score
from sklearn.preprocessing import LabelEncoder
from threadpoolctl import threadpool_limits

ROOT = Path(r"legacy_project").resolve()
BASE = Path(r"workspace\external_grouped")
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "working" / "ijismd_v11_mechanical_completion_20260923" / "scripts"))
import run_ciciot_taxonomy_residual_fusion as trf
import run_ciciot_agreement_gated_trf_confirmation as nested
import run_ciciot_calibrated_benefit_gate_confirmation as benefit
import run_ciciot_support_matched_control_v2 as support
import run_v10_grouped_flat_comparators as comparator
import run_v10_grouped_newton_flat_stacker as newton

_xgb, _lgb = trf.make_global_model, trf.make_lgbm
trf.make_global_model = lambda n, t: _xgb(n, t).set_params(n_jobs=2)
trf.make_lgbm = lambda n, t: _lgb(n, t).set_params(n_jobs=2)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def checkpoint(path, builder):
    if path.exists():
        return joblib.load(path)
    print(f"FITTING {path.parent.name}/{path.stem}", flush=True)
    started = time.perf_counter()
    value = builder()
    elapsed = time.perf_counter() - started
    temporary = path.with_suffix(".tmp.joblib")
    joblib.dump((value, elapsed), temporary, compress=3)
    temporary.replace(path)
    return value, elapsed


def align_stacker(model, probability, classes):
    aligned = np.zeros((len(probability), classes), dtype=np.float64)
    aligned[:, model.classes_.astype(int)] = probability
    return aligned


def ten_bin_ece(target, probability):
    probability = comparator.normalise(probability)
    confidence = probability.max(axis=1)
    correct = probability.argmax(axis=1) == target
    bins = np.clip(np.digitize(confidence, np.linspace(0, 1, 11), right=True) - 1, 0, 9)
    return float(sum(np.mean(bins == b) * abs(correct[bins == b].mean() - confidence[bins == b].mean())
                     for b in range(10) if np.any(bins == b)))


def main():
    assert BASE.resolve().drive.upper() == "D:"
    metadata = json.loads((BASE / "data_audit.json").read_text())
    protocol = json.loads((BASE / "protocol.json").read_text())
    frames = {role: pd.read_parquet(BASE / "datasets" / f"{role}.parquet").rename(columns={"attack": "Label"}).set_index("row_id", drop=False) for role in protocol["roles"]}
    features = metadata["features"]
    fit, cal, selection, outer = (frames[x] for x in protocol["roles"])
    fine = LabelEncoder().fit(fit.Label)
    all_labels = sorted(protocol["family_mapping"])
    unseen = sorted(set(outer.Label) - set(fine.classes_))
    known_cal = cal[cal.Label.isin(fine.classes_)].copy()
    if known_cal.empty or known_cal.Label.nunique() < 2:
        raise ValueError("Calibration future has fewer than two fitted classes; cannot fit prespecified meta controls. Do not move cutpoints.")
    y_cal = fine.transform(known_cal.Label)
    support_frame = pd.read_csv(BASE / "role_class_support_full.csv")
    uncapped = support_frame[support_frame.role == "fit"].set_index("attack")["count"]
    rare_labels = [label for label in fine.classes_ if label != "Benign" and uncapped[label] / uncapped.sum() <= .01]
    rare_outer = [label for label in rare_labels if np.any(outer.Label == label)]
    support_rows = []
    for role, frame in frames.items():
        for label in all_labels:
            support_rows.append({"role": role, "label": label, "rows": int((frame.Label == label).sum()), "fit_rare": label in rare_labels, "unseen_at_fit": label not in fine.classes_})
    pd.DataFrame(support_rows).to_csv(BASE / "bounded_role_support.csv", index=False)
    write_json(BASE / "model_audit.json", {"fit_classes": fine.classes_.tolist(), "outer_unseen_classes": unseen,
               "rare_fit_classes": rare_labels, "rare_with_outer_support": rare_outer,
               "calibration_rows_excluded_unseen_class": len(cal) - len(known_cal), "outer_rows": len(outer),
               "role_sizes": {k: len(v) for k, v in frames.items()},
               "absolute_timestamps_and_ips_excluded": True,
               "all_methods_same_test_row_sha256": hashlib.sha256(outer.row_id.to_numpy().tobytes()).hexdigest(),
               "seed_interpretation": protocol["interpretation"]})
    maps = {"real": {k: v for k, v in protocol["family_mapping"].items() if k in fine.classes_}}
    maps["cyclic"], cyclic_shift = trf.cyclic_control_map(maps["real"])
    maps["support"], support_audit = support.support_matched_wrong_map(fit, maps["real"])
    support_audit.to_csv(BASE / "support_map_audit.csv", index=False)
    write_json(BASE / "frozen_maps.json", {"maps": maps, "cyclic_shift": cyclic_shift, "constructed_from_fit_only": True})
    results = []
    class_results = []
    inference_frames = {"calibration": known_cal, "selection": selection, "test": outer}
    known_mask = outer.Label.isin(fine.classes_).to_numpy()
    y_text = outer.Label.to_numpy()

    def save_method(seed, directory, name, probability, train_seconds):
        probability = comparator.normalise(probability)
        prediction = fine.inverse_transform(probability.argmax(axis=1))
        labels_observed = sorted(set(y_text))
        rare_recalls = [np.mean(prediction[y_text == label] == label) for label in rare_outer]
        row = {"seed": seed, "method": name, "n_test": len(outer), "macro_f1_test_supported_labels": f1_score(y_text, prediction, labels=labels_observed, average="macro", zero_division=0),
               "macro_f1_all_ten_labels": f1_score(y_text, prediction, labels=all_labels, average="macro", zero_division=0),
               "macro_f1_known_test_supported_labels": f1_score(y_text[known_mask], prediction[known_mask], labels=sorted(set(y_text[known_mask])), average="macro", zero_division=0) if known_mask.any() else None,
               "accuracy": accuracy_score(y_text, prediction), "rare_recall": float(np.mean(rare_recalls)) if rare_recalls else None,
               "n_test_unseen": int((~known_mask).sum()), "n_rare_test_labels": len(rare_outer), "training_seconds": train_seconds}
        if known_mask.any():
            y_known = fine.transform(y_text[known_mask])
            row["ece_known_rows"] = ten_bin_ece(y_known, probability[known_mask])
            row["brier_known_rows"] = comparator.multiclass_brier(y_known, probability[known_mask])
        directory.mkdir(exist_ok=True)
        np.savez_compressed(directory / f"predictions_{name}.npz", row_id=outer.row_id.to_numpy(), y_true=y_text.astype("U"), classes=fine.classes_.astype("U"), probability=probability.astype(np.float32))
        for label in all_labels:
            mask = y_text == label
            class_results.append({"seed": seed, "method": name, "label": label, "support": int(mask.sum()), "recall": float(np.mean(prediction[mask] == label)) if mask.any() else None, "fit_rare": label in rare_labels, "fit_seen": label in fine.classes_})
        results.append(row)
        pd.DataFrame(results).to_csv(BASE / "temporal_results_partial.csv", index=False)
        print(f"RESULT seed={seed} {name}: F1={row['macro_f1_test_supported_labels']:.6f}; rare={row['rare_recall']}", flush=True)

    for seed in protocol["random_fit_seeds"]:
        directory = BASE / f"seed{seed}"
        directory.mkdir(exist_ok=True)
        trf.RANDOM_STATE = seed
        with threadpool_limits(limits=2), warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            xgb, xt = checkpoint(directory / "flat_xgb.joblib", lambda: nested.fit_weighted_xgb(fit, features, fine, 520)[0])
            xp = {key: xgb.predict_proba(trf.clean_features(frame, features)) for key, frame in inference_frames.items()}
            save_method(seed, directory, "flat_xgb", xp["test"], xt)
            del xgb
            gc.collect()
            lgb, lt = checkpoint(directory / "flat_lgbm.joblib", lambda: nested.fit_flat_lgbm(fit, features, fine, 380)[0])
            lp = {key: lgb.predict_proba(trf.clean_features(frame, features)) for key, frame in inference_frames.items()}
            save_method(seed, directory, "flat_lgbm", lp["test"], lt)
            save_method(seed, directory, "flat_equal_blend", .5 * xp["test"] + .5 * lp["test"], xt + lt)
            tx, _ = comparator.fit_temperature(y_cal, xp["calibration"])
            tl, _ = comparator.fit_temperature(y_cal, lp["calibration"])
            save_method(seed, directory, "temperature_flat_blend", .5 * comparator.temperature_probability(xp["test"], tx) + .5 * comparator.temperature_probability(lp["test"], tl), xt + lt)
            stacker, st = checkpoint(directory / "stacker.joblib", lambda: newton.fit_converged_stacker(comparator.probability_features(xp["calibration"], lp["calibration"]), y_cal, seed, 1., 100))
            stack_probability = align_stacker(stacker, stacker.predict_proba(comparator.probability_features(xp["test"], lp["test"])), len(fine.classes_))
            save_method(seed, directory, "newton_flat_stacker", stack_probability, xt + lt + st)
            write_json(directory / "calibration_audit.json", {"temperature_xgb": tx, "temperature_lgbm": tl, "fit_role": "calibration", "stacker_iterations": stacker.n_iter_.tolist(), "outer_labels_used": False})
            del stacker, lgb, lp
            for kind, mapping in maps.items():
                hierarchy, ht = checkpoint(directory / f"hierarchy_{kind}.joblib", lambda: trf.fit_control_hierarchy(fit, features, mapping, 380)[:3])
                model, family_encoder, locals_ = hierarchy
                hp = benefit.predict_taxonomy(outer, features, model, family_encoder, fine, mapping, locals_)[0]
                save_method(seed, directory, f"{kind}_hierarchy", hp, ht)
                save_method(seed, directory, f"{kind}_blend", .5 * xp["test"] + .5 * hp, xt + ht)
                del hierarchy, model, family_encoder, locals_, hp
                gc.collect()
        write_json(directory / "COMPLETED.json", {"seed": seed, "methods": 11, "source_protocol_sha256": hashlib.sha256((BASE / "protocol.json").read_bytes()).hexdigest()})
    frame = pd.DataFrame(results)
    assert len(frame) == 11 * len(protocol["random_fit_seeds"])
    frame.to_csv(BASE / "temporal_results.csv", index=False)
    pd.DataFrame(class_results).to_csv(BASE / "temporal_classwise.csv", index=False)
    frame.groupby("method").agg(seeds=("seed", "nunique"), macro_f1_mean=("macro_f1_test_supported_labels", "mean"), macro_f1_sd=("macro_f1_test_supported_labels", "std"), rare_recall_mean=("rare_recall", "mean"), rare_recall_sd=("rare_recall", "std"), accuracy_mean=("accuracy", "mean")).to_csv(BASE / "temporal_summary.csv")
    print("TEMPORAL EXPERIMENT COMPLETE", flush=True)


if __name__ == "__main__":
    BASE = Path(sys.argv[1])
    main()
