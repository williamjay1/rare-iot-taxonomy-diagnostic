"""No model fitting; protect the explicitly requested ECE bin boundaries."""
import sys
sys.dont_write_bytecode = True
import unittest
import numpy as np
import pandas as pd
from derive_ece10 import ece10, bin_audit, validate_metric_keys


class Ece10Tests(unittest.TestCase):
    def test_exact_euclid_boundary(self):
        confidence = np.array([0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])
        bins = np.clip(np.digitize(confidence, np.linspace(0, 1, 11), right=True) - 1, 0, 9)
        # Decimal .3/.6/.7 can differ from linspace edges by a final bit.
        edges = np.linspace(0, 1, 11)
        exact_bins = np.clip(np.digitize(edges, edges, right=True) - 1, 0, 9)
        np.testing.assert_array_equal(exact_bins, [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(bins[0], 0)
        self.assertEqual(bins[-1], 9)

    def test_boundary_changes_ece(self):
        probability = np.array([[.8, .2], [.81, .19]])
        y = np.array([0, 1])
        self.assertAlmostEqual(ece10(y, probability), .505)
        reference, rows = bin_audit(y, probability, 42, "test", "all")
        self.assertAlmostEqual(reference, .505)
        self.assertEqual(rows[7]["rows"], 1)
        self.assertEqual(rows[8]["rows"], 1)

    def test_row_share_not_number_of_masks(self):
        probability = np.array([[.6, .4], [.6, .4], [.95, .05]])
        y = np.array([0, 0, 1])
        self.assertAlmostEqual(ece10(y, probability), 2 / 3 * .4 + 1 / 3 * .95)

    def test_complete_unique_frozen_keys(self):
        frame = pd.DataFrame({"seed": [42, 42, 43, 43], "run_id": ["a", "b", "a", "b"]})
        self.assertTrue(validate_metric_keys(frame, [42, 43], ["a", "b"], False))
        self.assertFalse(validate_metric_keys(frame.iloc[:-1], [42, 43], ["a", "b"], True))
        with self.assertRaises(AssertionError):
            validate_metric_keys(frame.iloc[:-1], [42, 43], ["a", "b"], False)
        with self.assertRaises(AssertionError):
            validate_metric_keys(pd.concat([frame, frame.iloc[:1]]), [42, 43], ["a", "b"], True)
        with self.assertRaises(AssertionError):
            validate_metric_keys(frame.assign(run_id="wrong"), [42, 43], ["a", "b"], True)


if __name__ == "__main__":
    unittest.main()
