"""Guard the metadata-only recovery exception without fitting estimators."""
import sys
sys.dont_write_bytecode = True
import unittest
import numpy as np
import resume_paired_execution as resume
import verify_harmonized_outputs as verify
from sklearn.linear_model import LogisticRegression


class ResumeAuditTests(unittest.TestCase):
    def test_frozen_newton_n_jobs_none_is_preserved(self):
        model = LogisticRegression(C=1.0, max_iter=100, solver="newton-cholesky", random_state=42)
        metadata = resume.audited_model_parameters(model)
        self.assertIsNone(metadata["params"]["n_jobs"])
        verify.check_workers(metadata)

    def test_altered_newton_parameters_rejected(self):
        for changed in ({"n_jobs": 2}, {"solver": "lbfgs"}, {"max_iter": 10}):
            parameters = {"C": 1.0, "max_iter": 100, "solver": "newton-cholesky", "random_state": 42, **changed}
            model = LogisticRegression(**parameters)
            with self.assertRaises(AssertionError):
                resume.audited_model_parameters(model)

    def test_tree_thread_limit_still_enforced(self):
        with self.assertRaises(AssertionError):
            verify.check_workers({"type": "LGBMClassifier", "params": {"n_jobs": 4}})

    def test_unknown_timing_is_not_fabricated(self):
        original = resume._checkpoint
        sentinel = object()
        try:
            resume._checkpoint = lambda path, builder: (sentinel, None)
            model, seconds = resume.checkpoint_with_unknown_timing(None, None)
        finally:
            resume._checkpoint = original
        self.assertIs(model, sentinel)
        self.assertTrue(np.isnan(seconds))


if __name__ == "__main__":
    unittest.main()
