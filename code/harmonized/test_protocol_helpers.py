"""No model fitting and no file writes: protocol and constructor regression tests."""
import sys
sys.dont_write_bytecode = True
import unittest
import numpy as np
import pandas as pd
import run_harmonized_refit as run
import run_grouped37_confirmation as grouped


class ProtocolTests(unittest.TestCase):
    def test_estimators_change_only_worker_limit(self):
        run.trf.RANDOM_STATE = 47
        for original, limited, trees in ((run._global_factory, run.limited_global, 520),
                                         (run._lgbm_factory, run.limited_lgbm, 380)):
            expected = original(34, trees).get_params()
            observed = limited(34, trees).get_params()
            self.assertEqual(observed["n_jobs"], 2)
            self.assertEqual(observed["random_state"], 47)
            observed.pop("n_jobs")
            expected.pop("n_jobs")
            self.assertEqual(observed, expected)

    def test_caps_preserve_source_membership(self):
        frame = pd.DataFrame({"Label": ["BenignTraffic"] * 20 + ["X"] * 20,
                              "__source_row_id": np.arange(40)})
        fit = run.nested.cap_fit_rows(frame, 42, 7, 11)
        self.assertEqual(fit["Label"].value_counts().to_dict(), {"BenignTraffic": 11, "X": 7})
        pd.testing.assert_frame_equal(fit, run.nested.cap_fit_rows(frame, 42, 7, 11))
        self.assertTrue(fit["__source_row_id"].isin(frame.index).all())

    def test_zero_fill_float32(self):
        values = run.trf.clean_features(pd.DataFrame({"a": [np.inf, np.nan, 2.5]}), ["a"])
        self.assertEqual(values.dtype, np.float32)
        np.testing.assert_array_equal(values[:, 0], [0, 0, 2.5])

    def test_projected_hashes_stay_together(self):
        column = run.benefit.FEATURE_GROUP_COLUMN
        hashes = np.repeat(np.arange(100, dtype=np.uint64), 3)
        frame = pd.DataFrame({column: hashes, "Label": ["a"] * 150 + ["b"] * 150})
        representative = frame.drop_duplicates(column).set_index(column)["Label"]
        left, right = grouped.split_groups(frame, representative, 42, test_size=.20)
        self.assertFalse(set(left[column]).intersection(right[column]))
        self.assertEqual(len(left) + len(right), len(frame))

    def test_primary_configuration(self):
        primary = run.json.loads((run.CORE / "run_metadata.json").read_text())
        for key in ("seeds", "outer_test_frac", "base_fraction", "gate_train_fraction", "gate_select_fraction", "majority_cap", "benign_cap", "global_trees", "local_trees", "rare_lambda"):
            self.assertEqual(run.CONFIG[key], primary["arguments"][key])
        self.assertEqual(run.benefit.package_versions(), primary["package_versions"])
        self.assertEqual(run.WORK.resolve().drive.upper(), "D:")


if __name__ == "__main__":
    unittest.main()
