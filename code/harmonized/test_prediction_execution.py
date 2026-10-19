"""Execution-only batching tests; no estimator fitting or data-role changes."""
import sys
sys.dont_write_bytecode = True
import run_harmonized_refit  # Sets the physical-D runtime and thread limits.
import unittest
import numpy as np
from prediction_execution import bounded_predict


class Model:
    def __init__(self, n_jobs=2):
        self.n_jobs = n_jobs

    def get_params(self, deep=False):
        return {"n_jobs": self.n_jobs}


class PredictionExecutionTests(unittest.TestCase):
    def test_batches_keep_order_dtype_values_and_arguments(self):
        values = np.arange(123, dtype=np.float32).reshape(41, 3)
        calls = []

        def original(model, part, offset, *, num_threads, raw_score):
            calls.append((len(part), num_threads, raw_score))
            return part.astype(np.float64) + offset

        actual = bounded_predict(original, Model(), values, 0.25,
                                 batch_rows=8, raw_score=True)
        np.testing.assert_array_equal(actual, values.astype(np.float64) + 0.25)
        self.assertEqual(actual.dtype, np.float64)
        self.assertEqual([part[0] for part in calls], [8, 8, 8, 8, 8, 1])
        self.assertTrue(all(part[1:] == (2, True) for part in calls))

    def test_unchanged_two_thread_limit_required(self):
        values = np.ones((3, 2), dtype=np.float32)
        with self.assertRaises(RuntimeError):
            bounded_predict(None, Model(4), values)
        with self.assertRaises(RuntimeError):
            bounded_predict(None, Model(), values, num_threads=4)

    def test_single_batch_does_not_copy_return_value(self):
        result = np.array([[0.3, 0.7]], dtype=np.float64)
        actual = bounded_predict(lambda *a, **k: result, Model(), np.zeros((1, 3)))
        self.assertIs(actual, result)


if __name__ == "__main__":
    unittest.main()
