"""Scope guards: never promote archived composites to common-flat controls."""

import tempfile
from pathlib import Path
import unittest

import pandas as pd

import diagnose_capture_component_scope as audit
import build_manuscript_handoff as builder


class ScopeTest(unittest.TestCase):
    def frame(self):
        return pd.DataFrame([{"window": window, "method": method, "macro_f1": 0.5,
                              "rare_recall": 0.4} for window in audit.capture.WINDOWS
                             for method in (*audit.FRESH, *audit.ARCHIVED)])

    def test_split_preserves_all_four_windows_without_cross_generation_methods(self):
        fresh, archived = audit.split_results(self.frame())
        self.assertEqual(len(fresh), 20)
        self.assertEqual(len(archived), 16)
        self.assertEqual(set(fresh.method), set(audit.FRESH))
        self.assertEqual(set(archived.method), set(audit.ARCHIVED.values()))
        self.assertFalse(any("gate" in method for method in archived.method))

    def test_incomplete_window_is_rejected(self):
        with self.assertRaises(ValueError):
            audit.split_results(self.frame().iloc[:-1])

    def test_method_collision_is_rejected(self):
        frame = self.frame()
        frame.loc[0, "method"] = "cyclic_gate"
        with self.assertRaises(ValueError):
            audit.split_results(frame)

    def test_old_handoff_cannot_be_republished(self):
        with tempfile.TemporaryDirectory(dir=audit.run.OUTPUT / "temp") as folder:
            root = Path(folder)
            audit.run.write_json(root / "CAPTURE_CORRECTION_HANDOFF.json", {"completion": False})
            with self.assertRaisesRegex(RuntimeError, "obsolete 55-unit"):
                builder.require_ready(root)


if __name__ == "__main__":
    unittest.main()
