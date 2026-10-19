"""Check the completion-only manuscript handoff without fitting any model."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
import build_experiment_report as report
import pandas as pd


class HandoffTests(unittest.TestCase):
    def test_incomplete_verification_cannot_write_handoff(self):
        with tempfile.TemporaryDirectory(dir=report.run.WORK / "results" / "paired46" / "temp") as folder:
            root = Path(folder)
            (root / "verification_report.json").write_text(json.dumps({"results": [
                {"protocol": "paired46", "status": "INCOMPLETE"},
                {"protocol": "grouped37", "status": "COMPLETE_PASS"}]}))
            with patch.object(report.run, "WORK", root):
                with self.assertRaises(AssertionError):
                    report.main()
            self.assertFalse((root / "MANUSCRIPT_HANDOFF.json").exists())

    def test_incomplete_ece_cannot_write_handoff(self):
        with tempfile.TemporaryDirectory(dir=report.run.WORK / "results" / "paired46" / "temp") as folder:
            root = Path(folder)
            (root / "verification_report.json").write_text(json.dumps({"results": [
                {"protocol": "paired46", "status": "COMPLETE_PASS"},
                {"protocol": "grouped37", "status": "COMPLETE_PASS"}]}))
            (root / "ece10_derivation_report.json").write_text(json.dumps({"results": [
                {"protocol": "paired46", "status": "COMPLETE_PASS"},
                {"protocol": "grouped37", "status": "INCOMPLETE"}]}))
            with patch.object(report.run, "WORK", root):
                with self.assertRaises(AssertionError):
                    report.main()
            self.assertFalse((root / "MANUSCRIPT_HANDOFF.json").exists())

    def test_schema_and_ranges_preserve_both_ece_endpoints(self):
        with tempfile.TemporaryDirectory(dir=report.run.WORK / "results" / "paired46" / "temp") as folder:
            root = Path(folder)
            output = root / "results" / "paired46"
            output.mkdir(parents=True)
            (output / "run_contract.json").write_text(json.dumps({"config": {"seeds": report.run.SEEDS}}))
            rows = pd.DataFrame({"seed": report.run.SEEDS, "run_id": ["flat_weighted_xgboost"] * 6,
                                 "f1_macro": [.8, .81, .82, .83, .84, .85],
                                 "ece_10": [.01] * 6, "ece_15": [.02] * 6,
                                 "rare_ece_10": [.03] * 6, "rare_ece_15": [.04] * 6})
            with patch.object(report.run, "WORK", root):
                value = report.manuscript_protocol_metadata("paired46", rows, pd.DataFrame(), pd.DataFrame())
            fields = {item["field"]: item for item in value["metric_schema"]}
            self.assertIn("right-closed", fields["ece_10"]["definition"])
            self.assertIn("15-bin", fields["ece_15"]["definition"])
            ranges = pd.read_csv(output / "metric_ranges.csv")
            f1 = ranges.loc[ranges.metric == "f1_macro"].iloc[0]
            self.assertAlmostEqual(f1.minimum, .8)
            self.assertAlmostEqual(f1.maximum, .85)
            self.assertEqual(value["seeds"], report.run.SEEDS)
            self.assertFalse((root / "MANUSCRIPT_HANDOFF.json").exists())


if __name__ == "__main__":
    unittest.main()
