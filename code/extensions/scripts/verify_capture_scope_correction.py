"""Independently check numeric preservation and capture publication scope."""

from pathlib import Path
import json
import sys

sys.dont_write_bytecode = True
import pandas as pd

import diagnose_capture_component_scope as audit


def main():
    root = audit.run.OUTPUT
    folder = root / audit.VERSION
    correction = json.loads((root / "CAPTURE_CORRECTION_HANDOFF.json").read_text())
    current = json.loads((root / "MANUSCRIPT_HANDOFF.json").read_text())
    original = json.loads((folder / "original_MANUSCRIPT_HANDOFF.json").read_text())
    preserve = json.loads((folder / "preservation_audit.json").read_text())
    checks = []

    def check(name, value):
        checks.append({"check": name, "passed": bool(value)})
        if not value:
            raise AssertionError(name)

    check("original_55_unit_handoff_preserved", original["completion"] and original["completed_units"] == 55)
    check("corrected_handoff_fails_closed", not current["completion"] and not correction["completion"])
    check("counts_distinguish_computed_from_matched", current["computed_units"] == 55
          and current["completed_units"] == 51 and correction["common_flat_capture_windows_complete"] == 0)
    check("all_original_artifacts_unchanged", all(audit.run.file_hash(Path(path)) == digest
          for path, digest in preserve["original_artifacts_unchanged"].items()))
    raw = pd.read_csv(root / "capture_windows_results.csv")
    expected_fresh, expected_archived = audit.split_results(raw)
    fresh = pd.read_csv(folder / "capture_fresh_flat_results.csv")
    archived = pd.read_csv(folder / "capture_original_hybrid_audit.csv")
    for name, actual, expected in (("fresh", fresh, expected_fresh), ("archived", archived, expected_archived)):
        pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True),
                                      check_exact=False, rtol=1e-14, atol=1e-15)
        check(f"{name}_numeric_rows_preserved", True)
    check("all_four_fresh_windows_present", fresh.groupby("window").size().to_dict()
          == dict.fromkeys(audit.capture.WINDOWS, 5))
    check("all_four_archived_windows_preserved", archived.groupby("window").size().to_dict()
          == dict.fromkeys(audit.capture.WINDOWS, 4))
    check("no_archived_control_renamed_to_an_unproven_gate", not archived.method.str.contains("gate").any())
    check("no_cross_generation_capture_comparison_in_handoff", not current["capture_comparisons_by_window"])
    pairs = pd.read_csv(folder / "all_endpoint_per_seed_paired_results.csv")
    check("taxonomy_paired_inference_excludes_capture", not pairs.endpoint.eq("capture_windows").any())
    summary = pd.read_csv(folder / "all_endpoint_summary.csv")
    check("summary_keys_unique", not summary.duplicated(["endpoint", "facet", "method", "metric"]).any())
    check("capture_summary_fresh_only", set(summary.loc[summary.endpoint.eq("capture_windows"), "method"])
          == set(audit.FRESH))
    best = pd.read_csv(folder / "best_competitor_by_endpoint_metric.csv")
    check("no_mixed_capture_best_competitor", not best.endpoint.eq("capture_windows").any())
    requested = pd.read_csv(folder / "common_flat_repair_coverage.csv")
    check("four_missing_reblends_explicit", len(requested) == 4
          and not requested.common_flat_repair_complete.any() and requested.incomplete_reason.notna().all())
    check("all_current_contract_hashes_valid", all(audit.run.file_hash(Path(item["path"])) == item["sha256"]
          and len(pd.read_csv(item["path"])) == item["rows"] for item in current["csv_contracts"].values()))
    for name in ("lofo_results.csv", "scarcity_results.csv", "calibration_results.csv", "alert_budget_results.csv",
                 "table4_nine_arm_ece10.csv"):
        check(f"unaffected_contract_{name}", current["csv_contracts"][name] == original["csv_contracts"][name])
    report = {"status": "passed", "checks": checks, "checks_passed": len(checks),
              "common_flat_repair_complete": False, "no_additional_fit": True,
              "root_handoff_sha256": audit.run.file_hash(root / "MANUSCRIPT_HANDOFF.json"),
              "correction_handoff_sha256": audit.run.file_hash(root / "CAPTURE_CORRECTION_HANDOFF.json")}
    audit.run.write_json(folder / "verification_report.json", report)
    print(json.dumps({"status": "passed", "checks": len(checks), "matched_units": 51,
                      "unrepaired_capture_windows": 4}), flush=True)


if __name__ == "__main__":
    main()
