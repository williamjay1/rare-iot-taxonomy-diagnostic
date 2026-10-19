"""Matched 39-feature capture controls with physical-D caches and checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

sys.dont_write_bytecode = True
import run_matched_extensions as run

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from threadpoolctl import threadpool_limits

import prepare_ciciot2023_official_confirmation_samples as preparation
import run_ciciot2023_official_archive_confirmation as archive

WINDOWS = {"w1_55_56": [55, 56], "w2_57_58": [57, 58],
           "w3_59_60": [59, 60], "w4_61_63": [61, 62, 63]}
FROZEN_METHODS = {"legacy_flat_xgb": "flat_weighted_xgboost",
                  "taxonomy_blend": "equal_flat_taxonomy_blend",
                  "cyclic_blend": "calibrated_cyclic_taxonomy_control",
                  "support_matched_blend": "calibrated_support_matched_control"}
SEED = 20260729
OUTPUT = run.OUTPUT / "capture"
DEFER_SHARED_SUMMARY = False


def cached_input(source, directory, expected=None):
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / source.name
    manifest = destination.with_name(destination.name + ".copy.json")
    if manifest.exists():
        metadata = json.loads(manifest.read_text())
        if (source.stat().st_size != metadata["source_size"]
                or source.stat().st_mtime_ns != metadata["source_mtime_ns"]
                or not destination.exists() or run.file_hash(destination) != metadata["sha256"]):
            raise ValueError(f"Changed capture input: {source}")
        if expected is not None and metadata["sha256"] != expected:
            raise ValueError("Capture input differs from original archive manifest")
        return destination
    print(f"COPY {source.name} to physical D", flush=True)
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copyfile(source, temporary)
    checksum = run.file_hash(temporary)
    if checksum != (expected or run.file_hash(source)):
        raise ValueError(f"Capture checksum mismatch: {source}")
    temporary.replace(destination)
    run.write_json(manifest, {"source": str(source), "source_size": source.stat().st_size,
                              "source_mtime_ns": source.stat().st_mtime_ns, "sha256": checksum})
    return destination


def setup():
    root = run.ROOT / "results"
    sample_source = root / "jit_ciciot2023_official_confirmation_samples_final"
    sample_files = {name: cached_input(sample_source / name, OUTPUT / "datasets") for name in
                    ("base.parquet", "gate_train.parquet", "gate_select.parquet",
                     "sample_metadata.json", "selected_label_counts.csv")}
    frozen_source = root / "jit_sequential_window_drift_study_v2"
    frozen = {name: cached_input(frozen_source / name, OUTPUT / "frozen") for name in
              ("run_metadata.json", "window_dedup_audit.csv",
               *(f"window_predictions_{window}.npz" for window in WINDOWS))}
    metadata = json.loads(sample_files["sample_metadata.json"].read_text())
    previous = json.loads(frozen["run_metadata.json"].read_text())
    features = metadata["feature_columns"]
    mapping = previous["label_to_family"]
    encoder = LabelEncoder().fit(previous["fine_labels"])
    roles = {role: archive.add_family(pd.read_parquet(sample_files[f"{role}.parquet"]), mapping)
             for role in ("base", "gate_train", "gate_select")}
    groups = {name: np.unique(pd.util.hash_pandas_object(frame[features], index=False,
                                                       categorize=True).to_numpy(dtype=np.uint64))
              for name, frame in roles.items()}
    overlaps = {f"{left}_{right}": len(np.intersect1d(groups[left], groups[right], assume_unique=True))
                for left, right in (("base", "gate_train"), ("base", "gate_select"),
                                    ("gate_train", "gate_select"))}
    if any(overlaps.values()):
        raise ValueError(f"Capture development predictor groups overlap: {overlaps}")
    duplicate_source = root / "jit_ciciot_official_split_duplicate_audit_final"
    references = [np.load(cached_input(duplicate_source / f"{role}_unique_feature_hashes.npy",
                                      OUTPUT / "datasets")) for role in roles]
    conflicting = np.load(cached_input(duplicate_source / "global_conflicting_feature_hashes.npy",
                                      OUTPUT / "datasets"))
    manifest = pd.read_csv(cached_input(root / "jit_ciciot_official_archive_audit_final" /
                                        "official_member_manifest.csv", OUTPUT / "datasets"))
    rare, _, threshold = archive.rare_ids_from_counts(
        pd.read_csv(sample_files["selected_label_counts.csv"]), encoder)
    protocol = {"seed": SEED, "feature_columns": features, "n_jobs": 2,
                "base_trees": 520, "lgbm_trees": 380, "stacker_C": 1.0, "stacker_max_iter": 100,
                "calibrator_fit_role": "gate_train (utility/calibration), files 46-50",
                "selection_role": "gate_select, files 51-54; report/audit only, not fitting or tuning",
                "base_role": "frozen base sample, files 1-45", "test_windows": WINDOWS,
                "group_overlap_counts": overlaps, "fine_labels": encoder.classes_.tolist(),
                "rare_ids": rare.tolist(), "rare_threshold_source_rows": threshold,
                "role_rows": {name: len(frame) for name, frame in roles.items()},
                "sample_sha256": {name: run.file_hash(path) for name, path in sample_files.items()},
                "comparison_scope": "Capture-order proxy, not timestamp or temporal generalization proof.",
                "frozen_taxonomy_reuse": "Original hard predictions only, after identical source SHA, dedup counts and exact ordered y verification. No cross-representation probability reuse.",
                "test_used_for_fitting_selection_or_tuning": False,
                "package_versions": archive.package_versions()}
    protocol_hash = run.hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    existing = OUTPUT / "protocol.json"
    if existing.exists() and json.loads(existing.read_text())["protocol_sha256"] != protocol_hash:
        raise ValueError("Capture checkpoint protocol changed")
    protocol["protocol_sha256"] = protocol_hash
    run.write_json(existing, protocol)
    return features, encoder, roles, references, conflicting, manifest, frozen, rare, protocol_hash


def fit_models(features, encoder, roles):
    run.trf.RANDOM_STATE = SEED
    base = roles["base"]
    xgb = run.checkpoint(OUTPUT / "xgb.joblib", lambda: run.nested.fit_weighted_xgb(base, features, encoder, 520)[0])
    lgbm = run.checkpoint(OUTPUT / "lgbm.joblib", lambda: run.nested.fit_flat_lgbm(base, features, encoder, 380)[0])
    calibration = roles["gate_train"]
    y = encoder.transform(calibration["Label"].astype(str))
    path = OUTPUT / "calibration_inputs.npz"
    if path.exists():
        with np.load(path) as data:
            if not np.array_equal(y, data["y_true"]):
                raise ValueError("Capture calibration row mismatch")
            p_xgb, p_lgbm = data["xgb"], data["lgbm"]
    else:
        X = run.trf.clean_features(calibration, features)
        p_xgb = run.flat.normalise(xgb.predict_proba(X))
        p_lgbm = run.flat.normalise(lgbm.predict_proba(X))
        run.save_npz(path, y_true=y, xgb=p_xgb, lgbm=p_lgbm)
    meta = run.checkpoint(OUTPUT / "calibrators.joblib", lambda: run.fit_meta(
        p_xgb, p_lgbm, y, SEED, {"stacker_C": 1.0, "stacker_max_iter": 100}))
    return xgb, lgbm, meta


def prepare_window(window, features, encoder, references, conflicting, manifest, frozen, rare, protocol_hash):
    folder = OUTPUT / window
    folder.mkdir(exist_ok=True)
    if (folder / "cache_complete.json").exists():
        if json.loads((folder / "cache_complete.json").read_text())["protocol_sha256"] != protocol_hash:
            raise ValueError("Capture window cache protocol mismatch")
        return sorted(folder.glob("part_*.parquet"))
    canonical = {preparation.canonical_key(label): label for label in encoder.classes_}
    canonical["benign"] = "BenignTraffic"
    seen = np.empty(0, dtype=np.uint64)
    audit = dict.fromkeys(("raw_rows", "crossed_rows", "conflict_rows", "repeated_rows", "kept_rows"), 0)
    parts, labels = [], []
    source = run.ROOT / "data" / "processed" / "ciciot2023_official_validated" / "MERGED_CSV"
    members = manifest.loc[manifest.sequence.isin(WINDOWS[window])].sort_values("sequence")
    if sorted(members.sequence.tolist()) != WINDOWS[window]:
        raise ValueError("Capture member manifest has missing or duplicate sequences")
    for member in members.itertuples():
        name = Path(member.member).name
        path = cached_input(source / name, OUTPUT / "bytes", str(member.sha256))
        offset = 0
        for number, chunk in enumerate(pd.read_csv(path, chunksize=250000, low_memory=False)):
            if chunk.columns.tolist() != features + ["Label"]:
                raise ValueError(f"Capture schema changed: {name}")
            chunk = archive.canonicalise_labels(chunk, canonical)
            hashes = pd.util.hash_pandas_object(chunk[features], index=False, categorize=True).to_numpy(dtype=np.uint64)
            crossed = np.zeros(len(chunk), dtype=bool)
            for reference in references:
                crossed |= archive.membership_mask(hashes, reference)
            conflict = archive.membership_mask(hashes, conflicting)
            unique, first = np.unique(hashes, return_index=True)
            first_mask = np.zeros(len(chunk), dtype=bool)
            first_mask[first] = True
            new = ~archive.membership_mask(hashes, seen) & first_mask
            seen = np.union1d(seen, unique)
            keep = ~crossed & ~conflict & new
            audit["raw_rows"] += len(chunk)
            audit["crossed_rows"] += int(crossed.sum())
            audit["conflict_rows"] += int(conflict.sum())
            audit["repeated_rows"] += int((~new).sum())
            audit["kept_rows"] += int(keep.sum())
            kept = chunk.loc[keep].copy()
            kept["source_sequence"] = member.sequence
            kept["source_row"] = np.arange(offset, offset + len(chunk))[keep]
            kept["feature_hash"] = hashes[keep]
            part = folder / f"part_{member.sequence:02d}_{number:03d}.parquet"
            temporary = part.with_suffix(".tmp.parquet")
            kept.to_parquet(temporary, index=False)
            temporary.replace(part)
            parts.append(part)
            labels.append(encoder.transform(kept["Label"].astype(str)))
            offset += len(chunk)
        print(f"CACHED {window} {name}: {audit['kept_rows']} kept so far", flush=True)
    old_audit = pd.read_csv(frozen["window_dedup_audit.csv"]).set_index("window").loc[window]
    if any(int(old_audit[key]) != value for key, value in audit.items()):
        raise ValueError(f"Capture dedup differs from frozen window: {audit}")
    with np.load(frozen[f"window_predictions_{window}.npz"]) as old:
        if not np.array_equal(np.concatenate(labels), old["y_true"]) or not np.array_equal(rare, old["rare_ids"]):
            raise ValueError("Capture ordered labels or rare IDs differ from original window")
    run.write_json(folder / "cache_complete.json", {"protocol_sha256": protocol_hash, "audit": audit,
                                                   "ordered_y_sha256": run.array_hash(np.concatenate(labels)),
                                                   "parts": [part.name for part in parts]})
    return parts


def evaluate_window(window, parts, features, encoder, models, frozen, rare, protocol_hash):
    folder = OUTPUT / window
    if (folder / "completed.json").exists():
        if json.loads((folder / "completed.json").read_text())["protocol_sha256"] != protocol_hash:
            raise ValueError("Capture completed protocol mismatch")
        return
    xgb, lgbm, meta = models
    for part in parts:
        prediction_path = part.with_suffix(".npz")
        if prediction_path.exists():
            continue
        frame = pd.read_parquet(part)
        X = run.trf.clean_features(frame, features)
        p_xgb = run.flat.normalise(xgb.predict_proba(X))
        p_lgbm = run.flat.normalise(lgbm.predict_proba(X))
        p_temperature = run.flat.normalise(run.flat.temperature_probability(p_xgb, meta["xgb_temperature"]) +
                                            run.flat.temperature_probability(p_lgbm, meta["lgbm_temperature"]))
        p_stacker = run.flat.normalise(meta["stacker"].predict_proba(run.flat.probability_features(p_xgb, p_lgbm)))
        run.save_npz(prediction_path, y_true=encoder.transform(frame["Label"].astype(str)),
                     feature_hash=frame["feature_hash"].to_numpy(), source_sequence=frame["source_sequence"].to_numpy(),
                     source_row=frame["source_row"].to_numpy(), flat_xgb=p_xgb.argmax(axis=1),
                     flat_lgbm=p_lgbm.argmax(axis=1), flat_equal_blend=(p_xgb + p_lgbm).argmax(axis=1),
                     temperature_flat_blend_probability=p_temperature.astype(np.float32),
                     newton_flat_stacker_probability=p_stacker.astype(np.float32))
        print(f"INFERRED {window} {part.name}", flush=True)
    y_parts, predictions = [], {}
    for part in parts:
        with np.load(part.with_suffix(".npz")) as data:
            y_parts.append(data["y_true"])
            for method in ("flat_xgb", "flat_lgbm", "flat_equal_blend", *run.STRONG):
                values = data[f"{method}_probability"].argmax(axis=1) if method in run.STRONG else data[method]
                predictions.setdefault(method, []).append(values)
    y = np.concatenate(y_parts)
    predictions = {method: np.concatenate(values) for method, values in predictions.items()}
    with np.load(frozen[f"window_predictions_{window}.npz"]) as old:
        if not np.array_equal(y, old["y_true"]):
            raise ValueError("Capture predictions are not aligned with frozen window")
        for method, key in FROZEN_METHODS.items():
            predictions[method] = old[key].copy()
    rows = []
    for method, prediction in predictions.items():
        matrix = confusion_matrix(y, prediction, labels=np.arange(len(encoder.classes_)))
        metrics, _ = archive.confusion_metrics(matrix, rare)
        metrics["macro_f1"] = metrics.pop("f1_macro")
        rows.append({"endpoint": "capture_windows", "window": window, "seed": SEED, "method": method,
                     "test_rows": len(y), **metrics})
    run.write_csv(folder / "results.csv", pd.DataFrame(rows))
    run.write_json(folder / "completed.json", {"protocol_sha256": protocol_hash, "window": window,
                                               "rows": len(y), "method_rows": len(rows),
                                               "ordered_y_sha256": run.array_hash(y),
                                               "flat_xgb_agreement_with_original": float(np.mean(predictions['flat_xgb'] == predictions['legacy_flat_xgb'])),
                                               "stacker_iterations": meta["stacker_iterations"],
                                               "xgb_temperature": meta["xgb_temperature"],
                                               "lgbm_temperature": meta["lgbm_temperature"]})
    summarize()
    print(f"DONE capture {window}: {len(y)} rows", flush=True)


def summarize():
    frames = [pd.read_csv(path) for path in sorted(OUTPUT.glob("w*/results.csv"))
              if path.with_name("completed.json").exists()]
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        run.write_csv(OUTPUT / "windows_results.csv", combined)
        pivot = combined.pivot(index="window", columns="method", values=["macro_f1", "rare_recall"])
        deltas = [{"window": window, "control": control,
                   **{f"taxonomy_blend_minus_control_{metric}": row[(metric, "taxonomy_blend")] - row[(metric, control)]
                      for metric in ("macro_f1", "rare_recall")}}
                  for window, row in pivot.iterrows() for control in run.STRONG]
        run.write_csv(OUTPUT / "paired_deltas.csv", pd.DataFrame(deltas))
        if not DEFER_SHARED_SUMMARY:
            run.write_csv(OUTPUT.parent / "capture_windows_results.csv", combined)
            run.write_csv(OUTPUT.parent / "capture_paired_deltas.csv", pd.DataFrame(deltas))
    if not DEFER_SHARED_SUMMARY:
        run.summarize(json.loads(run.CONFIG_PATH.read_text()))


def main():
    global DEFER_SHARED_SUMMARY
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--defer-shared-summary", action="store_true",
                        help="Only write capture-local summaries while another queue owns shared aggregate files")
    parser.add_argument("--windows", nargs="+", choices=tuple(WINDOWS), default=list(WINDOWS))
    args = parser.parse_args()
    DEFER_SHARED_SUMMARY = args.defer_shared_summary
    run.require_physical_d(OUTPUT)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.summary_only:
        summarize()
        return
    if not args.defer_shared_summary and list((OUTPUT.parent / "dispatch" / "leases").glob("*.lock")):
        parser.error("Seed workers are active; capture must use --defer-shared-summary")
    fit_lease = None
    if not args.cache_only:
        fit_lease = OUTPUT.parent / "dispatch" / "capture_fit.lock"
        fit_lease.parent.mkdir(parents=True, exist_ok=True)
        with fit_lease.open("x", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "output": str(OUTPUT), "n_jobs": 2}, stream)
    (OUTPUT / "temp").mkdir(exist_ok=True)
    for name in ("TEMP", "TMP", "TMPDIR", "JOBLIB_TEMP_FOLDER"):
        os.environ[name] = str(OUTPUT / "temp")
    started = time.perf_counter()
    run.write_json(OUTPUT / "run_status.json", {"status": "running", "pid": os.getpid(),
                                               "command": sys.executable + " -B " + " ".join(sys.argv)})
    try:
        features, encoder, roles, references, conflicting, manifest, frozen, rare, protocol_hash = setup()
        models = None if args.cache_only else fit_models(features, encoder, roles)
        for window in args.windows:
            parts = prepare_window(window, features, encoder, references, conflicting, manifest, frozen, rare, protocol_hash)
            if models is not None:
                evaluate_window(window, parts, features, encoder, models, frozen, rare, protocol_hash)
            gc.collect()
        summarize()
        run.write_json(OUTPUT / "run_status.json", {"status": "cache_completed" if args.cache_only else "completed",
                                                   "windows": args.windows, "elapsed_seconds": time.perf_counter() - started,
                                                   "shared_summary_deferred": DEFER_SHARED_SUMMARY})
    except BaseException as error:
        run.write_json(OUTPUT / "run_status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                                   "error": repr(error), "traceback": traceback.format_exc()})
        raise
    finally:
        if fit_lease is not None:
            fit_lease.unlink()


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
