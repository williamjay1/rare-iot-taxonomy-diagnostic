"""Pure arithmetic and scope tests; no estimator fitting."""

import unittest

import numpy as np

import run_capture_common_flat as common
import publish_capture_common_flat as publisher


class CommonFlatTest(unittest.TestCase):
    def test_fixed_half_blend(self):
        flat = np.full((3, 34), 1 / 34)
        hierarchy = np.eye(34)[:3]
        actual = common.equal_blend(flat, hierarchy)
        np.testing.assert_array_equal(actual, .5 * flat + .5 * hierarchy)
        self.assertLess(common.probability_audit(actual)["max_row_sum_error"], 1e-14)

    def test_misaligned_components_rejected(self):
        with self.assertRaises(ValueError):
            common.equal_blend(np.ones((2, 34)), np.ones((3, 34)))

    def test_probability_guard_rejects_invalid_mass(self):
        with self.assertRaises(ValueError):
            common.probability_audit(np.ones((2, 34)))

    def test_formal_nine_arms_have_no_legacy_flat_or_unfitted_gates(self):
        self.assertEqual(len(common.METHODS), 9)
        self.assertEqual(len(set(common.METHODS)), 9)
        self.assertIn("real_taxonomy", common.METHODS)
        self.assertNotIn("legacy_flat_xgb", common.METHODS)
        self.assertFalse(any("gate" in method for method in common.METHODS))
        self.assertEqual(set(common.BLENDS.values()), {"taxonomy_blend", "cyclic_blend", "support_matched_blend"})

    def test_streaming_ece_matches_explicit_right_closed_ten_bins(self):
        p = np.full((4, 34), .01 / 33)
        p[:, 0] = [.1, .2, .8, 1.]
        p[:, 1:] = (1 - p[:, 0, None]) / 33
        y = np.array([0, 1, 0, 1])
        state = publisher.new_quality_state()
        publisher.add_quality(state, y[:2], p[:2])
        publisher.add_quality(state, y[2:], p[2:])
        actual = publisher.finish_quality(state)
        self.assertAlmostEqual(actual["ece"], common.run.ece(y, p, nbins=10), places=14)
        losses = np.square(p).sum(axis=1) - 2 * p[np.arange(4), y] + 1
        self.assertAlmostEqual(actual["brier"], float(losses.mean()), places=14)

    def test_native_float32_brier_uses_float64_arithmetic(self):
        p = np.full((3, 34), 1 / 34, dtype=np.float32)
        p[1, :2] = [.061, .001]
        y = np.array([0, 1, 2])
        state = publisher.new_quality_state()
        publisher.add_quality(state, y, p)
        actual = publisher.finish_quality(state)
        probability = p.astype(np.float64)
        target = np.eye(34)[y]
        expected = np.square(probability - target).sum(axis=1).mean()
        self.assertAlmostEqual(actual["brier"], float(expected), places=14)


if __name__ == "__main__":
    unittest.main()
