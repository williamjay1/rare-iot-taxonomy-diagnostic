"""Authorized capture hierarchy refits sharing the existing fresh flat model."""

from __future__ import annotations

import ctypes
from datetime import datetime, timezone
import gc
import json
import msvcrt
import os
from pathlib import Path
import sys
import time
import traceback
import warnings

sys.dont_write_bytecode = True
import run_matched_extensions as run
import run_capture_controls as capture

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from threadpoolctl import threadpool_limits
import run_ciciot_support_matched_control_v2 as support_control

OUTPUT = run.OUTPUT / "capture_common_flat_v2"
FRESH = ("flat_xgb", "flat_lgbm", "flat_equal_blend", *run.STRONG)
HIERARCHIES = {"real": "real_taxonomy", "cyclic": "cyclic_taxonomy",
               "support": "support_matched_taxonomy"}
BLENDS = {"real": "taxonomy_blend", "cyclic": "cyclic_blend", "support": "support_matched_blend"}
METHODS = (*FRESH, "real_taxonomy", *BLENDS.values())
ALL_METHODS = (*FRESH, *HIERARCHIES.values(), *BLENDS.values())
MIN_FREE_BYTES = 8 * 1024 ** 3


def now():
    return datetime.now(timezone.utc).isoformat()


def signature(value):
    return run.hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def frozen_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Frozen manifest changed: {path}")
    else:
        run.write_json(path, value)


def available_bytes():
    class Memory(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                    *((name, ctypes.c_ulonglong) for name in ("total_phys", "avail_phys", "total_page",
                       "avail_page", "total_virtual", "avail_virtual", "extended"))]
    value = Memory()
    value.length = ctypes.sizeof(value)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
        raise OSError("Cannot verify physical memory admission")
    return int(value.avail_phys)


def progress(phase, **extra):
    state = {"phase": phase, "pid": os.getpid(), "updated_utc": now(), "n_jobs": 2,
             "fitted_components_checkpointed": len(list((OUTPUT / "models").glob("*/*.completed.json"))),
             "prediction_parts_checkpointed": len(list(OUTPUT.glob("w*/part_*.completed.json"))),
             "completed_windows": len(list(OUTPUT.glob("w*/completed.json"))), **extra}
    run.write_json(OUTPUT / "progress.json", state)
    run.write_json(run.OUTPUT / "CURRENT_RUN_STATUS.json", {
        "state": "capture_shared_flat_refit_running" if phase != "complete" else "completed_and_verified",
        "original_computed_units": 55, "accepted_matched_units": 55 if phase == "complete" else 51,
        "capture_shared_flat_windows_complete": state["completed_windows"],
        "capture_shared_flat_progress": str(OUTPUT / "progress.json"), **state})


def admit(phase):
    free = available_bytes()
    if free < MIN_FREE_BYTES:
        print(f"WAIT_MEMORY {phase}: {free / 1024**3:.3f} GiB available; require 8", flush=True)
        while free < MIN_FREE_BYTES:
            progress("waiting_memory", waiting_for=phase, available_bytes=free, minimum_bytes=MIN_FREE_BYTES)
            time.sleep(60)
            free = available_bytes()
    print(f"MEMORY_ADMITTED {phase}: {free / 1024**3:.3f} GiB", flush=True)
    return free


def probability_audit(probability):
    if (probability.ndim != 2 or probability.shape[1] != 34
            or not np.isfinite(probability).all() or probability.min() < 0
            or probability.max() > 1 + 2e-6):
        raise ValueError("Invalid 34-class probability matrix")
    error = float(np.max(np.abs(probability.sum(axis=1) - 1))) if len(probability) else 0.0
    if error > 2e-6:
        raise ValueError(f"Probability mass error: {error}")
    return {"shape": list(probability.shape), "dtype": str(probability.dtype),
            "max_row_sum_error": error, "sha256": run.array_hash(probability),
            "hard_sha256": run.array_hash(probability.argmax(axis=1).astype(np.int16))}


def equal_blend(flat, hierarchy):
    if flat.shape != hierarchy.shape:
        raise ValueError("Blend components are not aligned")
    return 0.5 * flat + 0.5 * hierarchy


def prepare():
    admit("initialization")
    folder = capture.OUTPUT
    protocol = json.loads((folder / "protocol.json").read_text())
    old = json.loads((folder / "frozen" / "run_metadata.json").read_text())
    features = protocol["feature_columns"]
    if len(features) != 39 or protocol["seed"] != capture.SEED or protocol["n_jobs"] != 2:
        raise ValueError("Frozen capture representation changed")
    from diagnose_capture_component_scope import audit_inputs
    role_fingerprints = audit_inputs(folder, protocol, old)
    frozen_json(OUTPUT / "role_fingerprints.json", role_fingerprints)
    base = capture.archive.add_family(pd.read_parquet(folder / "datasets" / "base.parquet"), old["label_to_family"])
    encoder = LabelEncoder().fit(protocol["fine_labels"])
    real = old["label_to_family"]
    cyclic, shift = run.trf.cyclic_control_map(real)
    support, support_audit = support_control.support_matched_wrong_map(base, real)
    support_source = run.ROOT / "results" / "jit_ciciot2023_official_archive_confirmation" / "support_matched_map.json"
    if support != json.loads(support_source.read_text()) or shift != old["cyclic_control_shift"]:
        raise ValueError("Control maps differ from the frozen original maps")
    maps = {"real": real, "cyclic": cyclic, "support": support}
    frozen_json(OUTPUT / "frozen_maps.json", maps)
    run.write_csv(OUTPUT / "support_map_audit.csv", support_audit)
    models = {name: {"path": str(folder / name), "sha256": run.file_hash(folder / name)}
              for name in ("xgb.joblib", "lgbm.joblib", "calibrators.joblib")}
    parts = {}
    for window in capture.WINDOWS:
        marker = json.loads((folder / window / "cache_complete.json").read_text())
        paths = sorted((folder / window).glob("part_*.parquet"))
        if [p.name for p in paths] != marker["parts"]:
            raise ValueError("Cached part inventory changed")
        parts[window] = [{"parquet": str(p), "sha256": run.file_hash(p),
                          "fresh_predictions": str(p.with_suffix(".npz")),
                          "fresh_prediction_sha256": run.file_hash(p.with_suffix(".npz"))} for p in paths]
    sources = {str(path): run.file_hash(path) for path in (
        Path(run.trf.__file__), Path(run.nested.__file__), Path(support_control.__file__),
        Path(capture.__file__), Path(__file__))}
    specification = {"version": "capture_common_flat_v2", "authorization": "New hierarchy family/local refits explicitly authorized after the bounded diagnostic; fresh flat models and calibrators reused without refitting",
        "seed": capture.SEED, "n_jobs": 2, "minimum_free_memory_bytes": MIN_FREE_BYTES,
        "original_capture_protocol_sha256": protocol["protocol_sha256"],
        "features": features, "fine_labels": encoder.classes_.tolist(), "windows": capture.WINDOWS,
        "base_role": "frozen files 1-45; same 720747 rows, no extra cap",
        "calibrator_role": "existing utility files 46-50; no new calibrator fit",
        "selection_role": "files 51-54 excluded from all new fits and map selection",
        "test_role": "all four windows; prediction/metrics only, no fitting or tuning",
        "role_rows": protocol["role_rows"], "role_group_overlap": protocol["group_overlap_counts"],
        "base_fingerprint": role_fingerprints[0], "model_checkpoints": models,
        "flat_xgb_trees": 520, "flat_lgbm_trees": 380, "hierarchy_family_and_local_lgbm_trees": 380,
        "family_weight_exponent": .25, "local_weight_exponent": .50,
        "fixed_blend_weights": [.5, .5], "maps": maps, "cyclic_shift": shift,
        "support_map_source": str(support_source), "support_map_source_sha256": run.file_hash(support_source),
        "rare_ids": protocol["rare_ids"], "rare_threshold_source_rows": protocol["rare_threshold_source_rows"],
        "rarity_definition": "inclusive one-percent threshold in original base-file source counts; not selected-base support",
        "formal_method_ids": list(METHODS), "additional_component_audit_ids": [HIERARCHIES[k] for k in ("cyclic", "support")],
        "removed_from_formal_competitors": "legacy_flat_xgb is retained only in original audit CSVs; real_taxonomy is the ninth formal arm",
        "ece_nbins": 10, "ece_bins": "equal-width right-closed, zero in first bin, row-share weighted",
        "prediction_parts": parts, "implementation_sources_sha256": sources,
        "all_arms_retrained": False, "old_hybrid_probabilities_reused": False,
        "real_edge_hardware": False, "test_tuning": False}
    specification["protocol_sha256"] = signature(specification)
    # Script changes after a checkpoint are audited separately, not silently accepted.
    frozen_json(OUTPUT / "protocol.json", specification)
    run.trf.RANDOM_STATE = capture.SEED
    return base, features, encoder, maps, specification


def fit_component(path, X, y, fingerprint, protocol):
    path.parent.mkdir(parents=True, exist_ok=True)
    spec = {"protocol_sha256": protocol["protocol_sha256"], "rows": len(y),
            "n_features": X.shape[1], "n_classes": len(np.unique(y)),
            "ordered_X_sha256": run.array_hash(X), "ordered_y_sha256": run.array_hash(y), **fingerprint}
    weights = run.trf.inverse_frequency_weights(y, spec["weight_exponent"])
    spec["ordered_weight_sha256"] = run.array_hash(weights)
    model = run.trf.make_lgbm(spec["n_classes"], 380)
    if model.n_jobs != 2 or model.random_state != capture.SEED:
        raise ValueError("Hierarchy factory lost thread/seed constraints")
    spec["parameters_repr"] = {name: repr(value) for name, value in model.get_params().items()}
    spec["input_sha256"] = signature(spec)
    frozen_json(path.with_suffix(".input.json"), spec)
    def build():
        admit(path.stem)
        progress("fitting", component=str(path), input_sha256=spec["input_sha256"])
        started = time.perf_counter()
        model.fit(X, y, sample_weight=weights)
        return {"model": model, "fit_seconds": time.perf_counter() - started,
                "input_sha256": spec["input_sha256"], "fitted_utc": now()}
    saved = run.checkpoint(path, build)
    if saved["input_sha256"] != spec["input_sha256"] or saved["model"].n_jobs != 2:
        raise ValueError("Hierarchy checkpoint input/threads differ")
    run.write_json(path.with_suffix(".completed.json"), {**spec, "model_sha256": run.file_hash(path),
        "fit_seconds": saved["fit_seconds"], "fitted_utc": saved["fitted_utc"]})
    print(f"COMPONENT_DONE {path.parent.name}/{path.stem}: {saved['fit_seconds']:.3f}s", flush=True)
    return saved["model"]


def fit_hierarchies(base, features, maps, protocol):
    X = run.trf.clean_features(base, features)
    bundles = {}
    for name, mapping in maps.items():
        start = time.perf_counter()
        values = base.Label.astype(str).map(mapping)
        encoder = LabelEncoder().fit(values)
        if values.isna().any():
            raise ValueError("Map does not cover all fitting labels")
        family = fit_component(OUTPUT / "models" / name / "family.joblib", X, encoder.transform(values),
                               {"kind": "family", "map": name, "classes": encoder.classes_.tolist(),
                                "weight_exponent": .25}, protocol)
        locals_ = {}
        for index, family_name in enumerate(encoder.classes_):
            mask = values.to_numpy() == family_name
            labels = base.loc[mask, "Label"].astype(str)
            local_encoder = LabelEncoder().fit(labels)
            if len(local_encoder.classes_) == 1:
                locals_[str(family_name)] = (None, None, local_encoder.classes_.tolist())
                continue
            local = fit_component(OUTPUT / "models" / name / f"local_{index:02d}.joblib",
                X[mask], local_encoder.transform(labels),
                {"kind": "local", "map": name, "family": str(family_name),
                 "classes": local_encoder.classes_.tolist(), "weight_exponent": .50,
                 "ordered_base_positions_sha256": run.array_hash(np.flatnonzero(mask))}, protocol)
            locals_[str(family_name)] = (local, local_encoder, local_encoder.classes_.tolist())
        bundles[name] = (family, encoder, locals_)
        run.write_json(OUTPUT / "models" / name / "bundle_complete.json", {
            "protocol_sha256": protocol["protocol_sha256"], "family_classes": encoder.classes_.tolist(),
            "local_label_sets": {k: v[2] for k, v in locals_.items()},
            "completed_components": len(list((OUTPUT / "models" / name).glob("*.completed.json"))),
            "this_invocation_seconds": time.perf_counter() - start})
        print(f"HIERARCHY_DONE {name}: this invocation {time.perf_counter() - start:.3f}s", flush=True)
        progress("hierarchy_complete", hierarchy=name)
        gc.collect()
    return bundles


def infer_part(window, source, features, encoder, flat_models, bundles, maps, protocol):
    part = Path(source["parquet"])
    folder = OUTPUT / window
    folder.mkdir(exist_ok=True)
    path = folder / part.with_suffix(".npz").name
    marker = path.with_suffix(".completed.json")
    if marker.exists():
        done = json.loads(marker.read_text())
        if (done["protocol_sha256"] != protocol["protocol_sha256"]
                or done["prediction_file_sha256"] != run.file_hash(path)):
            raise ValueError("Prediction checkpoint mismatch")
        return
    admit(f"inference/{window}/{part.stem}")
    progress("predicting", window=window, part=part.name)
    start = time.perf_counter()
    if run.file_hash(part) != source["sha256"] or run.file_hash(Path(source["fresh_predictions"])) != source["fresh_prediction_sha256"]:
        raise ValueError("Cached source/prediction bytes changed")
    frame = pd.read_parquet(part)
    X = run.trf.clean_features(frame, features)
    y = encoder.transform(frame.Label.astype(str))
    with np.load(source["fresh_predictions"]) as saved:
        np.testing.assert_array_equal(y, saved["y_true"])
        for key in ("feature_hash", "source_sequence", "source_row"):
            np.testing.assert_array_equal(frame[key].to_numpy(), saved[key])
        probabilities = {"flat_xgb": run.flat.normalise(flat_models[0].predict_proba(X)),
                         "flat_lgbm": run.flat.normalise(flat_models[1].predict_proba(X))}
        probabilities["flat_equal_blend"] = equal_blend(probabilities["flat_xgb"], probabilities["flat_lgbm"])
        for method in FRESH:
            if method in run.STRONG:
                probabilities[method] = saved[f"{method}_probability"].copy()
            else:
                np.testing.assert_array_equal(probabilities[method].argmax(axis=1), saved[method])
    for name, (family, family_encoder, local) in bundles.items():
        hierarchy, _ = run.trf.hierarchy_probabilities(X, family, family_encoder, encoder, maps[name], local)
        probabilities[HIERARCHIES[name]] = hierarchy
        probabilities[BLENDS[name]] = equal_blend(probabilities["flat_xgb"], hierarchy)
    audits = {method: probability_audit(p) for method, p in probabilities.items()}
    arrays = {"y_true": y, "rare_ids": np.asarray(protocol["rare_ids"], dtype=np.int16),
              **{key: frame[key].to_numpy() for key in ("feature_hash", "source_sequence", "source_row")}}
    for method, p in probabilities.items():
        arrays[f"{method}_probability"] = p
        arrays[method] = p.argmax(axis=1).astype(np.int16)
    run.save_npz(path, **arrays)
    run.write_json(marker, {"protocol_sha256": protocol["protocol_sha256"], "rows": len(y),
        "ordered_y_sha256": run.array_hash(y), "input_source": source,
        "prediction_file_sha256": run.file_hash(path), "probabilities": audits,
        "elapsed_seconds": time.perf_counter() - start, "finished_utc": now()})
    print(f"PART_DONE {window}/{part.stem}: {len(y)} rows, {time.perf_counter() - start:.3f}s", flush=True)
    progress("part_complete", window=window, part=part.name)


def evaluate_window(window, protocol):
    folder = OUTPUT / window
    labels, predictions = [], {method: [] for method in ALL_METHODS}
    for source in protocol["prediction_parts"][window]:
        path = folder / Path(source["parquet"]).with_suffix(".npz").name
        with np.load(path) as saved:
            labels.append(saved["y_true"])
            for method in ALL_METHODS:
                predictions[method].append(saved[method])
    y = np.concatenate(labels)
    with np.load(capture.OUTPUT / "frozen" / f"window_predictions_{window}.npz") as original:
        np.testing.assert_array_equal(y, original["y_true"])
        np.testing.assert_array_equal(protocol["rare_ids"], original["rare_ids"])
    rows = []
    for method, parts in predictions.items():
        matrix = confusion_matrix(y, np.concatenate(parts), labels=np.arange(len(protocol["fine_labels"])))
        metrics, _ = capture.archive.confusion_metrics(matrix, np.asarray(protocol["rare_ids"]))
        metrics["macro_f1"] = metrics.pop("f1_macro")
        rows.append({"endpoint": "capture_windows", "window": window, "seed": capture.SEED,
                     "method": method, "test_rows": len(y), **metrics})
    all_results = pd.DataFrame(rows)
    run.write_csv(folder / "component_results.csv", all_results)
    run.write_csv(folder / "results.csv", all_results.loc[all_results.method.isin(METHODS)])
    run.write_json(folder / "completed.json", {"protocol_sha256": protocol["protocol_sha256"],
        "window": window, "rows": len(y), "method_rows": len(METHODS), "component_audit_rows": len(ALL_METHODS),
        "ordered_y_sha256": run.array_hash(y), "finished_utc": now()})
    print(f"WINDOW_DONE {window}: {len(y)} rows, {len(METHODS)} formal methods", flush=True)
    progress("window_complete", window=window)


def main():
    run.require_physical_d(OUTPUT)
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "temp").mkdir(exist_ok=True)
    for name in ("TEMP", "TMP", "TMPDIR", "JOBLIB_TEMP_FOLDER"):
        os.environ[name] = str(OUTPUT / "temp")
    with (OUTPUT / "worker.lock").open("a+b") as lease:
        if lease.tell() == 0:
            lease.write(b"0")
            lease.flush()
        lease.seek(0)
        msvcrt.locking(lease.fileno(), msvcrt.LK_NBLCK, 1)
        started = time.perf_counter()
        try:
            base, features, encoder, maps, protocol = prepare()
            bundles = fit_hierarchies(base, features, maps, protocol)
            del base
            gc.collect()
            flat_models = tuple(joblib.load(capture.OUTPUT / name) for name in ("xgb.joblib", "lgbm.joblib"))
            if any(model.n_jobs != 2 for model in flat_models):
                raise ValueError("Fresh flat checkpoints do not respect two threads")
            for window, sources in protocol["prediction_parts"].items():
                for source in sources:
                    infer_part(window, source, features, encoder, flat_models, bundles, maps, protocol)
                evaluate_window(window, protocol)
            del bundles, flat_models
            gc.collect()
            from publish_capture_common_flat import verify_and_publish
            verify_and_publish(protocol)
            progress("complete", elapsed_seconds=time.perf_counter() - started)
        except BaseException as error:
            progress("failed", error=repr(error), traceback=traceback.format_exc(),
                     elapsed_seconds=time.perf_counter() - started)
            raise
        finally:
            lease.seek(0)
            msvcrt.locking(lease.fileno(), msvcrt.LK_UNLCK, 1)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)
    with threadpool_limits(limits=2):
        main()
