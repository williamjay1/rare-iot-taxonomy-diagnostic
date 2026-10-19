from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "ijismd_v15_standards_20261018"
FROZEN = ROOT / "frozen"
SEEDS = list(range(42, 48))
ARMS = ["coreflat", "standardblend", "wrongstandardblend"]
METRICS = ["accuracy", "f1_macro", "recall_macro", "rare_recall", "worst_recall", "pr_auc_macro"]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def descriptive(values: pd.Series) -> dict:
    return {"n": len(values), "mean": float(values.mean()), "sample_sd": float(values.std(ddof=1)),
            "min": float(values.min()), "max": float(values.max()),
            "positive_seeds": int(values.gt(0).sum()), "negative_seeds": int(values.lt(0).sum()),
            "zero_seeds": int(values.eq(0).sum())}


def table(path: Path, key: list[str]) -> dict:
    frame = pd.read_csv(path)
    assert not frame.duplicated(key).any(), path
    return {"path": str(path), "sha256": sha(path), "rows": len(frame),
            "columns": frame.columns.tolist(), "unique_key": key}


def main() -> None:
    for parent in (ROOT, *ROOT.parents):
        assert parent.drive.upper() == "D:"
        assert not (getattr(parent.lstat(), "st_file_attributes", 0) & 0x400)
    state = read_json(OUT / "run_status.json")
    audit = read_json(OUT / "AUDIT.json")
    assert state["status"] == "COMPLETE" and state["completed_seeds"] == SEEDS
    assert audit["status"] == "PASS" and audit["seeds"] == SEEDS
    assert audit["new_hierarchy_fits"] == 12 and audit["audited_fitted_estimators"] == 120
    assert audit["all_hard_prediction_classification_metrics_recomputed"]
    assert audit["flat_original_hard_preserved_not_replaced_by_float16_argmax"]
    assert "FROZEN INPUTS VERIFIED" in (ROOT / "verify.log").read_text(encoding="utf-8-sig")
    assert (OUT / "REPORT_CN.md").is_file()
    metrics = pd.read_csv(OUT / "model_results.csv")
    assert len(metrics) == 24 and not metrics.duplicated(["seed", "run_id"]).any()
    costs = pd.read_csv(OUT / "costs.csv")
    assert len(costs) == 12 and costs.n_jobs.eq(2).all()
    assert set(zip(costs.seed, costs.run_id)) == {(seed, arm) for seed in SEEDS for arm in ARMS[1:]}
    prepared = read_json(FROZEN / "prepared.json")
    original_metadata_path = ROOT / "cache" / "core_metadata.json"
    original_mapping = read_json(original_metadata_path)["label_to_family"]
    assert len(set(original_mapping.values())) == 8
    bundles, predictions = [], []
    for seed in SEEDS:
        prediction = OUT / f"paired_predictions_seed_{seed}.npz"
        assert prediction.is_file()
        with np.load(prediction) as data:
            schema = {key: {"dtype": str(data[key].dtype), "shape": list(data[key].shape)} for key in data.files}
        predictions.append({"seed": seed, "path": str(prediction), "sha256": sha(prediction), "arrays": schema})
        for arm in ARMS[1:]:
            bundle = ROOT / "cache" / f"models_{arm}_seed_{seed}.joblib"
            assert bundle.is_file()
            bundles.append({"seed": seed, "run_id": arm, "path": str(bundle), "sha256": sha(bundle),
                            "bytes": bundle.stat().st_size})
    summary = {arm: {name: descriptive(group[name]) for name in METRICS}
               for arm, group in metrics[metrics.run_id.isin(ARMS)].groupby("run_id")}
    contrasts = {}
    indexed = metrics.set_index(["seed", "run_id"])
    original = pd.read_csv(ROOT / "cache" / "core_results.csv")
    original_flat = original[original.run_id.eq("flat_weighted_xgboost")].sort_values("seed")
    assert original_flat.seed.tolist() == SEEDS
    for row in original_flat.to_dict("records"):
        for name in METRICS[:-1]:
            assert abs(row[name] - indexed.loc[(row["seed"], "coreflat"), name]) < 1e-12
    for left, right in (("standardblend", "coreflat"), ("wrongstandardblend", "coreflat"),
                        ("standardblend", "wrongstandardblend")):
        rows = [{"seed": seed, **{name: float(indexed.loc[(seed, left), name] - indexed.loc[(seed, right), name])
                                  for name in METRICS}} for seed in SEEDS]
        frame = pd.DataFrame(rows)
        contrasts[f"{left}_minus_{right}"] = {"direction": f"{left} minus {right}",
                                               "flat_classification_reference": "original canonical hard predictions",
                                               "flat_AP_reference": "recomputed cached float16 probability AP, not the historical full-precision core AP",
                                               "per_seed": rows,
                                               "summary": {name: descriptive(frame[name]) for name in METRICS}}
    mapping = pd.read_csv(FROZEN / "standard_mapping.csv", keep_default_na=False)
    definitions = {
        "Benign": "Nonattack singleton; no CAPEC attack annotation.",
        "TCP_Flood": "TCP flooding at protocol level; resource subtype not inferred from DoS versus DDoS.",
        "UDP_Flood": "UDP flooding; neither reflection nor amplification inferred.",
        "ICMP_Flood": "ICMP flooding without a specific message subtype claim.",
        "Fragmentation_Abuse": "Protocol fragmentation; ACK flooding is only a partial CAPEC match.",
        "HTTP_Resource_Exhaustion": "HTTP request flooding or low-rate session holding, with distinct CAPEC references.",
        "Other_Network_Flood": "Broad flooding fallback for generator labels lacking a supported exact CAPEC child.",
        "Active_Probing": "Host, port, OS or vulnerability discovery; no unsupported probe subtype or attack stage.",
        "Input_Injection": "Command, SQL and browser-script injection; conditional subtypes remain conditional.",
        "Unresolved_Web_Compromise": "Explicit unresolved fallback, not an official mechanism or validated web-shell/backdoor technique.",
        "Credential_Guessing": "Dictionary-based password guessing; not password spraying or credential stuffing.",
        "Traffic_Redirection": "DNS spoofing or ARP adversary-in-the-middle; no inferred credential-theft outcome.",
    }
    groups = [{"group": name, "definition": definitions[name], "label_count": len(group),
               "labels": group.fine_label.tolist(),
               "ambiguity_labels": group[group.ambiguity.astype(str).str.lower().eq("true")].fine_label.tolist()}
              for name, group in mapping.groupby("mechanism_group", sort=True)]
    assert len(groups) == 12
    map_hashes = [sha(FROZEN / f"wrong_map_seed_{seed}.json") for seed in SEEDS]
    wrong_maps = [{"seed": seed, "path": str(FROZEN / f"wrong_map_seed_{seed}.json"),
                   "sha256": map_hashes[index],
                   "mapping": read_json(FROZEN / f"wrong_map_seed_{seed}.json")}
                  for index, seed in enumerate(SEEDS)]
    standard_map = dict(zip(mapping.fine_label, mapping.mechanism_group))
    attack_labels = sorted(label for label in standard_map if label != "BenignTraffic")
    for item in wrong_maps:
        wrong = item["mapping"]
        item["retained_named_attack_labels"] = [label for label in attack_labels if wrong[label] == standard_map[label]]
        item["retained_original_attack_copairs"] = [[a, b] for a, b in combinations(attack_labels, 2)
                                                     if standard_map[a] == standard_map[b] and wrong[a] == wrong[b]]
        assert len(item["retained_named_attack_labels"]) == 11
        assert len(item["retained_original_attack_copairs"]) == 8
    control = read_json(FROZEN / "wrong_control_audit.json")
    coverage = read_json(OUT / "coverage.json")
    control_labels = pd.read_csv(OUT / "wrong_control_label_audit.csv")
    citation = read_json(OUT / "CAPEC_RELEASE_CITATION.json")
    assert sha(Path(citation["release_snapshot"])) == citation["release_snapshot_sha256"]
    assert sha(Path(citation["catalog_path"])) == citation["catalog_sha256"]
    classwise = pd.read_csv(OUT / "independent_classwise_recall.csv")
    class_means = classwise.pivot_table(index="fine_label", columns="run_id", values="recall", aggfunc="mean")
    class_deltas = (class_means.standardblend - class_means.coreflat).sort_values()
    entries = {
        "model_results": table(OUT / "model_results.csv", ["seed", "run_id"]),
        "paired_differences": table(OUT / "paired_differences.csv", ["seed", "metric"]),
        "classwise": table(OUT / "independent_classwise_recall.csv", ["seed", "run_id", "fine_label"]),
        "fit_costs": table(OUT / "costs.csv", ["seed", "run_id"]),
        "audited_cost_summary": table(OUT / "cost_summary_audited.csv", ["run_id"]),
        "mapping": table(FROZEN / "standard_mapping.csv", ["fine_label"]),
        "official_definitions": table(OUT / "per_label_definition_evidence.csv", ["fine_label", "capec_id"]),
        "source_versions": table(OUT / "source_versions.csv", ["filename"]),
        "support_match": table(FROZEN / "support_match_audit.csv", ["seed", "group"]),
        "float16_hard_audit": table(OUT / "flat_float16_hard_audit.csv", ["seed"]),
        "prediction_audit": table(OUT / "independent_prediction_audit.csv", ["seed", "run_id"]),
        "model_parameter_audit": table(OUT / "fitted_model_parameter_audit.csv", ["seed", "run_id", "component"]),
        "wrong_control_labels": table(OUT / "wrong_control_label_audit.csv", ["seed", "fine_label"]),
        "local_support_vectors": table(OUT / "local_class_support_vector_audit.csv", ["seed", "run_id", "group"]),
    }
    result = {
        "schema_version": "1.0", "complete": True,
        "experiment_id": "ijismd_v15_standards_20261018",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_seeds": SEEDS, "completed_matched_cells": 12,
        "audited_fitted_estimators": 120, "physical_work_root": str(ROOT),
        "manuscript_role": "Additional retrospective standards control; does not replace or rename the manuscript primary method or its original primary results.",
        "new_statistics_original_reference": "coreflat uses the original canonical flat hard predictions, not the float16 argmax sensitivity arm. Classification metrics match the canonical core exactly; cached-probability AP is separately scoped.",
        "original_flat_baseline_reference": {
            "run_id": "flat_weighted_xgboost", "per_seed": original_flat[["seed", "n", *METRICS]].to_dict("records"),
            "classification_metrics_exact_in_new_coreflat": True,
            "historical_AP_not_substituted_for_cached_AP": True,
            "original_primary_manuscript_method_not_redefined_by_this_handoff": True,
        },
        "frozen_control_primary_contrast": "standardblend minus wrongstandardblend macro-F1",
        "required_arms": ARMS, "summary": summary, "paired_contrasts": contrasts,
        "mapping_status": "Researcher-constructed CAPEC-informed 12-group partition; neither expert certified nor an official CICIoT-to-CAPEC crosswalk.",
        "original_vs_new_taxonomy": {
            "original_dataset_family_groups": 8, "new_CAPEC_informed_groups": 12,
            "original_mapping_source": str(original_metadata_path),
            "original_mapping_source_sha256": sha(original_metadata_path),
            "original_label_to_family": original_mapping,
            "distinct_experiments": True,
            "scope": "Keep the original eight-family primary method separate from this twelve-group standards control. The older broad ATT&CK tactic control is also not this CAPEC mechanism partition.",
        },
        "expert_validation_performed": False, "coverage": coverage,
        "mechanism_groups": groups, "per_label_mapping": mapping.to_dict("records"),
        "matching_constraints": {
            "per_group_label_count_exact": True, "per_group_capped_fit_rows_exact": True,
            "benign_singleton_fixed": True, "attack_labels_may_not_enter_benign": True,
            "wrong_control_audit": control,
            "all_wrong_maps_and_retention_lists": wrong_maps,
            "wrong_partition_count_across_seeds": len(set(map_hashes)),
            "same_wrong_partition_all_seeds": len(set(map_hashes)) == 1,
            "no_claim_of_zero_retained_pairs": True,
            "wholly_unchanged_nonbenign_groups": ["Credential_Guessing", "Input_Injection", "Unresolved_Web_Compromise"],
            "unresolved_and_injection_semantics_not_isolated_by_control": True,
            "all_rare_named_memberships_retained": bool(control_labels.loc[control_labels.rare_class, "named_membership_retained"].all()),
            "rare_class_control_limit": "All eight rare classes keep named group membership; PingSweep co-members change. This is not independent validation of every rare-class mechanism.",
            "not_matched_on": ["uncapped base support", "test support", "runtime", "serialized model bytes"],
        },
        "source_citation": citation,
        "parameters_and_roles": read_json(FROZEN / "design_lock.json")["parameters"],
        "features_ordered": prepared["features"], "labels_ordered": prepared["fine_labels"],
        "float16_hard_audit": pd.read_csv(OUT / "flat_float16_hard_audit.csv").to_dict("records"),
        "negative_results_not_to_omit": {
            "do_not_claim_all_rare_classes_improve": True,
            "worst_recall_standard_minus_coreflat": contrasts["standardblend_minus_coreflat"]["summary"]["worst_recall"],
            "Uploading_Attack_by_seed": classwise[classwise.fine_label.eq("Uploading_Attack")].to_dict("records"),
            "negative_class_mean_recall_deltas_standard_minus_coreflat": [
                {"fine_label": label, "mean_recall_delta": float(delta)} for label, delta in class_deltas.items() if delta < 0
            ],
        },
        "cost_summary": pd.read_csv(OUT / "cost_summary_audited.csv").to_dict("records"),
        "run_metadata": read_json(OUT / "run_metadata.json"),
        "audit_status": "PASS", "audit_path": str(OUT / "AUDIT.json"), "audit_sha256": sha(OUT / "AUDIT.json"),
        "report_path": str(OUT / "REPORT_CN.md"), "tables_and_schemas": entries,
        "paired_prediction_packages": predictions, "model_bundles": bundles,
        "original_core_results_readonly_reference": str(ROOT / "cache" / "core_results.csv"),
        "interpretation_limits": [
            "Six overlapping grouped random splits; mean and sample SD are descriptive, not independent-sample confidence intervals.",
            "Control selected and frozen using capped-fit counts, never test-based map selection.",
            "CAPEC reference coverage is not mechanistic validity, expert agreement, or per-flow proof.",
            "New hierarchy CPU cost excludes reused flat training/inference and is not physical-edge latency, feature-extraction throughput, or energy.",
            "Runner serialization_seconds also includes scoring and must not be interpreted as isolated serialization cost.",
            "NF-ToN-IoT-v3 disjoint chronological attack support is a separate extrapolation boundary, not rare-class replication evidence; no NF experiment is performed in this subtask.",
            "No physical edge device was available and no physical-edge experiment was performed.",
        ],
    }
    destination = OUT / "MANUSCRIPT_HANDOFF.json"
    assert not destination.exists(), "Do not silently overwrite an existing final handoff"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"complete": True, "path": str(destination), "sha256": sha(destination),
                      "cells": 12, "seeds": SEEDS, "audit_status": "PASS"}, indent=2))


if __name__ == "__main__":
    main()
