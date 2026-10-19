from __future__ import annotations

import argparse
import gc
import hashlib
import html.parser
import json
import os
import re
import sys
import time
import warnings
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "ijismd_v15_standards_20261018"
CACHE = ROOT / "cache"
FROZEN = ROOT / "frozen"
for variable in ("TEMP", "TMP", "TMPDIR", "JOBLIB_TEMP_FOLDER"):
    os.environ[variable] = str(ROOT / "temp")
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
sys.dont_write_bytecode = True

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, recall_score
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore", message="X does not have valid feature names.*")


def sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def physical_d(path: Path) -> None:
    assert path.drive.upper() == "D:"
    for parent in (path, *path.parents):
        assert not (getattr(parent.lstat(), "st_file_attributes", 0) & 0x400), parent


def write_json(path: Path, value: object) -> None:
    physical_d(path.parent)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


class PlainHTML(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def evidence() -> dict:
    physical_d(ROOT)
    sources = json.loads((ROOT / "source_manifest.json").read_text())
    version_rows = []
    xml_source = next(item for item in sources if item["path"].endswith(".xml"))
    root = ET.parse(xml_source["path"]).getroot()
    assert root.attrib["Version"] == "3.9"
    citation = json.loads((OUT / "CAPEC_RELEASE_CITATION.json").read_text())
    assert citation["catalog_date_and_official_release_date"] == root.attrib["Date"] == "2023-01-24"
    assert sha(Path(citation["release_snapshot"])) == citation["release_snapshot_sha256"]
    assert sha(Path(citation["catalog_path"])) == citation["catalog_sha256"] == xml_source["sha256"]
    parser = PlainHTML()
    parser.feed(Path(citation["release_snapshot"]).read_text(encoding="utf-8"))
    release_text = " ".join(" ".join(parser.parts).split())
    start = release_text.index(citation["release_announcement_title"])
    assert "January 24, 2023" in release_text[start : start + 200]
    ns = {"c": "http://capec.mitre.org/capec-3"}
    patterns = {p.attrib["ID"]: p for p in root.findall("c:Attack_Patterns/c:Attack_Pattern", ns)}
    for item in sources:
        path = Path(item["path"])
        assert sha(path) == item["sha256"], path
        version, created, modified = "", "", ""
        if path.suffix == ".xml":
            version = root.attrib["Version"]
            modified = root.attrib.get("Date", "")
        elif path.name.startswith("attack_"):
            parser = PlainHTML()
            parser.feed(path.read_text(encoding="utf-8"))
            text = " ".join(" ".join(parser.parts).split())
            match = re.search(r"Version:\s*([0-9.]+)", text)
            assert match, path
            version = match.group(1)
            for field in ("Created", "Last Modified"):
                match = re.search(field + r":\s*([0-9]{1,2} [A-Za-z]+ [0-9]{4})", text)
                assert match, (path, field)
                if field == "Created":
                    created = match.group(1)
                else:
                    modified = match.group(1)
        else:
            version = "unversioned official HTML; content fixed by SHA256"
        version_rows.append({**item, "official_version": version, "official_created": created,
                             "official_last_modified": modified, "raw_readonly": not bool(path.stat().st_mode & 0o200)})
    pd.DataFrame(version_rows).to_csv(OUT / "source_versions.csv", index=False)
    mapping = pd.read_csv(FROZEN / "standard_mapping.csv", keep_default_na=False)
    records = []
    for row in mapping.to_dict("records"):
        for order, capec_id in enumerate(filter(None, row["capec_ids"].split(";"))):
            pattern = patterns[capec_id.removeprefix("CAPEC-")]
            description = " ".join(" ".join(pattern.find("c:Description", ns).itertext()).split())
            records.append({**row, "capec_id": capec_id, "official_name": pattern.attrib["Name"],
                            "official_status": pattern.attrib["Status"],
                            "official_abstraction": pattern.attrib["Abstraction"],
                            "official_description": description, "candidate_order": order + 1,
                            "source_sha256": xml_source["sha256"]})
    pd.DataFrame(records).to_csv(OUT / "per_label_definition_evidence.csv", index=False)
    attack = mapping[mapping.fine_label != "BenignTraffic"]
    ambiguous = attack[attack.ambiguity.astype(str).str.lower() == "true"]
    coverage = {
        "fine_labels": len(mapping), "attack_labels": len(attack),
        "groups_including_benign_and_unresolved": mapping.mechanism_group.nunique(),
        "labels_with_some_CAPEC_reference": int(attack.capec_ids.ne("").sum()),
        "attack_labels_without_primary_ambiguity": len(attack) - len(ambiguous),
        "ambiguous_attack_labels": ambiguous.fine_label.tolist(),
        "primary_ambiguity_count": len(ambiguous),
        "match_counts_attack_labels": {str(k): int(v) for k, v in attack.match.value_counts().items()},
        "unresolved_fallback_labels": attack[attack.mechanism_group == "Unresolved_Web_Compromise"].fine_label.tolist(),
        "official_pattern_status_counts_unique": dict(Counter(patterns[i.removeprefix("CAPEC-")].attrib["Status"] for i in {r["capec_id"] for r in records})),
        "unique_CAPEC_patterns": len({r["capec_id"] for r in records}),
        "claim_boundary": "Reference coverage is not exact mechanistic validity, expert agreement, per-flow evidence, or official dataset-to-CAPEC annotation. Conditional secondary candidates remain explicitly conditional.",
        "expert_validation_performed": False,
    }
    write_json(OUT / "coverage.json", coverage)
    standard = dict(zip(mapping.fine_label, mapping.mechanism_group))
    control_rows = []
    for seed in range(42, 48):
        wrong_path = FROZEN / f"wrong_map_seed_{seed}.json"
        if not wrong_path.exists():
            continue
        wrong = json.loads(wrong_path.read_text())
        roles = np.load(FROZEN / f"roles_seed_{seed}.npz")
        labels = json.loads((FROZEN / "prepared.json").read_text())["fine_labels"]
        rare = {labels[i] for i in roles["rare_ids"]}
        roles.close()
        for label in sorted(standard):
            control_rows.append({"seed": seed, "fine_label": label, "standard_group": standard[label],
                                 "wrong_group": wrong[label], "named_membership_retained": standard[label] == wrong[label],
                                 "rare_class": label in rare,
                                 "original_group_entirely_unchanged": {l for l in standard if standard[l] == standard[label]} == {l for l in wrong if wrong[l] == standard[label]}})
    pd.DataFrame(control_rows).to_csv(OUT / "wrong_control_label_audit.csv", index=False)
    return coverage


def classifications(y: np.ndarray, pred: np.ndarray, rare: np.ndarray, classes: int) -> dict:
    recalls = recall_score(y, pred, labels=np.arange(classes), average=None, zero_division=0)
    return {"accuracy": float(accuracy_score(y, pred)),
            "f1_macro": float(f1_score(y, pred, labels=np.arange(classes), average="macro", zero_division=0)),
            "recall_macro": float(np.mean(recalls)), "rare_recall": float(np.mean(recalls[rare])),
            "worst_recall": float(np.min(recalls))}


def results() -> dict:
    started = time.perf_counter()
    coverage = evidence()
    state = json.loads((OUT / "run_status.json").read_text())
    assert state["status"] == "COMPLETE" and state["completed_seeds"] == list(range(42, 48))
    prepared = json.loads((FROZEN / "prepared.json").read_text())
    lock = json.loads((FROZEN / "design_lock.json").read_text())
    for key, path in (("runner_sha256", ROOT / "run_standards_control.py"),
                      ("mechanism_spec_sha256", ROOT / "mechanism_spec.json"),
                      ("protocol_sha256", ROOT / "PROTOCOL.md")):
        assert sha(path) == lock[key], path
    for path, expected in lock["helper_hashes"].items():
        assert sha(Path(path)) == expected, path
    for relative, expected in prepared["files"].items():
        assert sha(ROOT / relative) == expected, relative
    for item in prepared["cache_copies"]:
        assert sha(Path(item["cache"])) == item["sha256"], item["cache"]
    metrics = pd.read_csv(OUT / "model_results.csv")
    costs = pd.read_csv(OUT / "costs.csv")
    support = pd.read_csv(FROZEN / "support_match_audit.csv")
    roles = pd.read_csv(FROZEN / "role_audit.csv")
    canonical_source = Path(next(item["source"] for item in prepared["cache_copies"]
                                 if Path(item["cache"]).name == "core_results.csv")).parent / "nested_split_audit.csv"
    canonical = pd.read_csv(canonical_source).set_index("seed")
    canonical.to_csv(OUT / "canonical_role_counts_reference.csv")
    assert len(metrics) == 24 and not metrics.duplicated(["seed", "run_id"]).any()
    assert len(costs) == 12 and costs.n_jobs.eq(2).all()
    assert len(support) == 72
    assert support.standard_label_count.eq(support.wrong_label_count).all()
    assert support.standard_fit_rows.eq(support.wrong_fit_rows).all()
    assert roles.filter(like="overlap").eq(0).all().all()
    prediction_checks, model_checks, class_rows, local_support_rows, quantization_rows = [], [], [], [], []
    expected_parameters = {"n_jobs": 2, "n_estimators": 380, "learning_rate": 0.055,
                           "num_leaves": 127, "max_depth": -1, "min_child_samples": 25,
                           "subsample": 0.9, "colsample_bytree": 0.95,
                           "reg_lambda": 1.5, "reg_alpha": 0.02}
    label_names = prepared["fine_labels"]
    label_ids = {label: index for index, label in enumerate(label_names)}
    columns = prepared["features"] + ["Label"]
    full = pd.concat([pd.read_parquet(CACHE / "train.parquet", columns=columns),
                      pd.read_parquet(CACHE / "test.parquet", columns=columns)], ignore_index=True)
    standard = json.loads((FROZEN / "standard_map.json").read_text())
    mapping = pd.read_csv(FROZEN / "standard_mapping.csv").set_index("fine_label")
    for seed in range(42, 48):
        print(f"Independent audit seed {seed}", flush=True)
        own = np.load(OUT / f"paired_predictions_seed_{seed}.npz")
        core = np.load(CACHE / f"core_seed_{seed}.npz")
        role = np.load(FROZEN / f"roles_seed_{seed}.npz")
        reference = canonical.loc[seed]
        for key, field in (("base_id", "base_pool_rows_before_cap"), ("utility_id", "gate_train_rows"),
                           ("selection_id", "gate_select_rows"), ("outer_id", "outer_test_rows")):
            assert len(role[key]) == reference[field], (seed, key)
        assert roles.set_index("seed").loc[seed, "fit_rows"] == reference.base_fit_rows_after_cap
        assert [label_names[i] for i in role["rare_ids"]] == json.loads(reference.rare_labels)
        assert np.array_equal(own["row_id"], core["row_id"])
        assert np.array_equal(own["y_true"], core["y_true"])
        assert np.array_equal(own["rare_ids"], core["rare_ids"])
        assert np.array_equal(own["coreflat_prediction"], core["flat_weighted_xgboost"])
        assert set(own["row_id"]) == set(role["outer_id"])
        assert np.array_equal(full.loc[own["row_id"], "Label"].astype(str).map(label_ids).to_numpy(), own["y_true"])
        assert sha(CACHE / f"fit_seed_{seed}.parquet") == roles.set_index("seed").loc[seed, "fit_sha256"]
        fit = pd.read_parquet(CACHE / f"fit_seed_{seed}.parquet", columns=["Label"])
        counts = {str(label): int(count) for label, count in fit.Label.astype(str).value_counts().items()}
        y, rare = own["y_true"], own["rare_ids"]
        cached_flat = core["flat_weighted_xgboost_probability"]
        assert cached_flat.dtype == np.float16
        assert cached_flat.shape == (len(y), len(label_names))
        assert np.isfinite(cached_flat).all() and cached_flat.min() >= 0
        cached_argmax = cached_flat.argmax(axis=1)
        original_hard = own["coreflat_prediction"]
        changed = cached_argmax != original_hard
        top_tie = (cached_flat == cached_flat.max(axis=1, keepdims=True)).sum(axis=1) > 1
        original_values = classifications(y, original_hard, rare, len(label_names))
        cached_values = classifications(y, cached_argmax, rare, len(label_names))
        cached_recorded = metrics[(metrics.seed == seed) & (metrics.run_id == "coreflat_cached_probability")].iloc[0]
        assert max(abs(cached_values[name] - cached_recorded[name]) for name in cached_values) < 1e-12
        binary = label_binarize(y, classes=np.arange(len(label_names)))
        cached_ap = float(average_precision_score(binary, cached_flat, average="macro"))
        for flat_arm in ("coreflat", "coreflat_cached_probability"):
            flat_recorded = metrics[(metrics.seed == seed) & (metrics.run_id == flat_arm)].iloc[0]
            assert abs(cached_ap - flat_recorded.pr_auc_macro) < 1e-12
        quantization_rows.append({
            "seed": seed, "rows": len(y), "cached_probability_dtype": str(cached_flat.dtype),
            "original_hard_exact_canonical": True,
            "original_hard_vs_cached_argmax_changes": int(changed.sum()),
            "original_hard_vs_cached_argmax_fraction": float(changed.mean()),
            "cached_top_tie_rows": int(top_tie.sum()),
            "changed_rows_with_cached_top_tie": int(np.sum(changed & top_tie)),
            "cached_rowsum_max_error": float(np.max(np.abs(cached_flat.astype(np.float64).sum(axis=1) - 1))),
            "cached_probability_ap_exact_recorded": True,
            **{f"original_hard_{name}": value for name, value in original_values.items()},
            **{f"cached_argmax_{name}": value for name, value in cached_values.items()},
            **{f"cached_minus_original_{name}": cached_values[name] - original_values[name] for name in cached_values},
        })
        sample = np.unique(np.concatenate([np.linspace(0, len(y) - 1, 512, dtype=int)] +
                                          [np.flatnonzero(y == index)[:8] for index in range(len(label_names))]))
        X = (full.loc[own["row_id"][sample], prepared["features"]]
             .replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=np.float32))
        for arm in ("coreflat", "standardblend", "wrongstandardblend"):
            predicted = own[f"{arm}_prediction"]
            values = classifications(y, predicted, rare, len(label_names))
            recorded = metrics[(metrics.seed == seed) & (metrics.run_id == arm)].iloc[0]
            max_error = max(abs(values[name] - recorded[name]) for name in values)
            assert max_error < 1e-12, (seed, arm, max_error)
            row = {"seed": seed, "run_id": arm, "classification_metrics_max_error": max_error,
                   "row_order_exact": True, "labels_exact": True, "n": len(y)}
            for index, label in enumerate(label_names):
                mask = y == index
                class_rows.append({"seed": seed, "run_id": arm, "fine_label": label,
                                   "support": int(mask.sum()), "correct": int(np.sum(predicted[mask] == index)),
                                   "recall": float(np.mean(predicted[mask] == index)),
                                   "ambiguity": bool(mapping.loc[label, "ambiguity"]),
                                   "mechanism_group": standard[label]})
            if arm != "coreflat":
                probability = own[arm]
                assert probability.shape == (len(y), len(label_names))
                assert probability.dtype == np.float32 and np.isfinite(probability).all()
                assert probability.min() >= 0
                ap = float(average_precision_score(binary, probability, average="macro"))
                row.update(float32_argmax_disagreements=int(np.sum(predicted != probability.argmax(axis=1))),
                           ap_from_float32=ap, ap_difference_from_float64_metric=ap - recorded.pr_auc_macro,
                           probability_rowsum_max_error=float(np.max(np.abs(probability.sum(axis=1) - 1))))
                bundle_path = CACHE / f"models_{arm}_seed_{seed}.joblib"
                physical_d(bundle_path)
                bundle = joblib.load(bundle_path)
                group_map = standard if arm == "standardblend" else json.loads((FROZEN / f"wrong_map_seed_{seed}.json").read_text())
                assert bundle["mapping"] == group_map
                assert bundle["features"] == prepared["features"]
                family_labels = sorted(set(group_map.values()))
                assert list(bundle["family_encoder"].classes_) == family_labels
                for group in family_labels:
                    sublabels = sorted(label for label in group_map if group_map[label] == group)
                    support_row = support[(support.seed == seed) & (support.group == group)].iloc[0]
                    assert len(sublabels) == support_row.standard_label_count
                    assert sum(counts[label] for label in sublabels) == support_row.standard_fit_rows
                    standard_counts = sorted(counts[label] for label in standard if standard[label] == group)
                    actual_counts = sorted(counts[label] for label in sublabels)
                    local_support_rows.append({"seed": seed, "run_id": arm, "group": group,
                                               "standard_sorted_fine_class_counts": json.dumps(standard_counts),
                                               "actual_sorted_fine_class_counts": json.dumps(actual_counts),
                                               "fine_class_support_multiset_exact": standard_counts == actual_counts})
                    model, encoder, names = bundle["local_models"][group]
                    assert sorted(names) == sublabels
                    assert (model is None) == (len(sublabels) == 1)
                    if model is not None:
                        assert list(encoder.classes_) == sublabels
                models = [("family", bundle["family_model"])] + [(group, item[0]) for group, item in bundle["local_models"].items() if item[0] is not None]
                assert len(models) == 10
                for name, model in models:
                    params = model.get_params()
                    for key, value in {**expected_parameters, "random_state": seed}.items():
                        assert params[key] == value, (seed, arm, name, key, params[key])
                    assert model.n_features_in_ == 46
                    model_checks.append({"seed": seed, "run_id": arm, "component": name,
                                         "n_jobs": params["n_jobs"], "n_estimators_limit": params["n_estimators"],
                                         "fitted_iterations": model.booster_.current_iteration(),
                                         "fitted_trees": model.booster_.num_trees(), "features": model.n_features_in_,
                                         "random_state": params["random_state"], "fixed_parameters_match": True,
                                         "bundle_sha256": sha(bundle_path) if name == "family" else ""})
                family_probability = bundle["family_model"].predict_proba(X)
                replay = np.zeros((len(sample), len(label_names)), dtype=np.float64)
                for group_id, group in enumerate(bundle["family_encoder"].classes_):
                    model, local_encoder, names = bundle["local_models"][str(group)]
                    conditional = np.ones((len(sample), 1)) if model is None else model.predict_proba(X)
                    for local_id, label in enumerate(names):
                        replay[:, label_ids[label]] = family_probability[:, group_id] * conditional[:, local_id]
                replay /= np.clip(replay.sum(axis=1, keepdims=True), 1e-12, None)
                replay = 0.5 * cached_flat[sample].astype(np.float64) + 0.5 * replay
                replay_error = float(np.max(np.abs(replay - probability[sample])))
                assert replay_error < 6e-8, (seed, arm, replay_error)
                assert np.array_equal(replay.argmax(axis=1), predicted[sample]), (seed, arm, "replay hard predictions")
                row.update(replay_rows=len(sample), replay_probability_max_error=replay_error,
                           replay_hard_predictions_exact=True)
                del bundle, models, probability
                gc.collect()
            prediction_checks.append(row)
        own.close(); core.close(); role.close()
    pd.DataFrame(prediction_checks).to_csv(OUT / "independent_prediction_audit.csv", index=False)
    pd.DataFrame(model_checks).to_csv(OUT / "fitted_model_parameter_audit.csv", index=False)
    pd.DataFrame(class_rows).to_csv(OUT / "independent_classwise_recall.csv", index=False)
    pd.DataFrame(local_support_rows).to_csv(OUT / "local_class_support_vector_audit.csv", index=False)
    pd.DataFrame(quantization_rows).to_csv(OUT / "flat_float16_hard_audit.csv", index=False)
    costs["workstation_batch_rows_per_second"] = costs.outer_rows / costs.hierarchy_inference_seconds
    cost_summary = costs.groupby("run_id").agg(
        training_seconds_mean=("training_seconds", "mean"), training_seconds_sd=("training_seconds", "std"),
        inference_seconds_mean=("hierarchy_inference_seconds", "mean"), inference_seconds_sd=("hierarchy_inference_seconds", "std"),
        model_bytes_mean=("model_bytes", "mean"), model_bytes_sd=("model_bytes", "std"),
        workstation_batch_rows_per_second_mean=("workstation_batch_rows_per_second", "mean"),
        rss_peak_mib_max=("rss_peak_mib", "max"))
    cost_summary.to_csv(OUT / "cost_summary_audited.csv")
    audit = {"status": "PASS", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
             "audit_seconds": time.perf_counter() - started,
             "seeds": list(range(42, 48)), "result_rows": len(metrics), "new_hierarchy_fits": len(costs),
             "audited_fitted_estimators": len(model_checks), "prediction_checks": len(prediction_checks),
             "all_source_and_frozen_hashes_verified": True, "physical_D_outputs_verified": True,
             "official_CAPEC_catalog_and_release_date_verified": "2023-01-24",
             "all_group_sizes_and_capped_fit_support_exact": True, "all_saved_predictions_order_labels_exact": True,
             "additional_audit_all_local_fine_class_support_multisets_exact": all(item["fine_class_support_multiset_exact"] for item in local_support_rows),
             "canonical_role_counts_and_rare_labels_exact": True, "canonical_role_audit_sha256": sha(canonical_source),
             "all_hard_prediction_classification_metrics_recomputed": True, "all_models_n_jobs_2": True,
             "flat_float16_quantization_seeds_audited": len(quantization_rows),
             "flat_original_hard_preserved_not_replaced_by_float16_argmax": True,
             "all_models_original_fixed_hyperparameters": True,
             "ap_precision_note": "Run metrics use float64 in-memory blends; saved blends are float32. Independent AP differences are reported, not hidden or silently overwritten.",
             "cost_scope": "Original helper timer measures hierarchy fitting after initial feature conversion and label encoding. Prediction timer measures full outer-batch hierarchy prediction including feature conversion. Cached flat fit/inference excluded. Rows per second is workstation batch prediction only, not edge throughput or per-event latency.",
             "serialization_field_warning": "Runner serialization_seconds includes joblib serialization AND metric computation; do not interpret as isolated serialization time. It is excluded from audited cost summary.",
             "rss_scope": "Peak resident set is cumulative peak of the whole runner process, not incremental per-model memory.",
             "coverage": coverage}
    write_json(OUT / "AUDIT.json", audit)
    return audit


def manifest() -> dict:
    state = json.loads((OUT / "AUDIT.json").read_text())
    assert state["status"] == "PASS"
    entries = []
    destination = OUT / "artifact_manifest.json"
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == destination or "temp" in path.relative_to(ROOT).parts:
            continue
        physical_d(path)
        entries.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha(path)})
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "root": str(ROOT),
              "files": entries, "file_count": len(entries), "total_bytes": sum(item["bytes"] for item in entries),
              "scope": "Physical-D standards work only. This manifest excludes itself and temporary scratch files. Raw official source hashes are recorded separately in source_versions.csv."}
    write_json(destination, result)
    return {"file_count": result["file_count"], "total_bytes": result["total_bytes"], "manifest_sha256": sha(destination)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["evidence", "results", "manifest"])
    arguments = parser.parse_args()
    result = {"evidence": evidence, "results": results, "manifest": manifest}[arguments.mode]()
    print(json.dumps(result, indent=2))
