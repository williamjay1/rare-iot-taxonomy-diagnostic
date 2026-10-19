import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import run_matched_extensions as run
import run_capture_controls as capture
import build_manuscript_handoff as handoff
import verify_extension_outputs as verify
import audit_frozen_sources as frozen_audit


class ExtensionTests(unittest.TestCase):
    def test_canonical_hard_ece_sensitivity_does_not_switch_declared_correctness(self):
        p = np.array([[.8, .2]])
        y = np.array([0])
        self.assertAlmostEqual(run.ece(y, p, nbins=10), .2)
        self.assertAlmostEqual(frozen_audit.ece_with_hard_labels(y, p, np.array([1])), .8)

    def test_nested_audit_uses_three_seed_sample_sd_not_twenty_one_cases(self):
        with tempfile.TemporaryDirectory(dir=run.OUTPUT / "temp") as directory:
            root = Path(directory)
            metrics = ["macro_f1", "rare_recall", "msp_unknown_auroc",
                       "msp_selection_fpr_0.05_unknown_recall", "msp_selection_fpr_0.05_outer_known_fpr"]
            frame = pd.DataFrame([
                {"seed": seed, "case": case, "method": method,
                 **dict.fromkeys(metrics, value)}
                for seed, value in ((42, .1), (43, .3), (44, .9))
                for case in range(7) for method in run.METHODS
            ])
            per_seed = frame.groupby(["seed", "method"])[metrics].mean().reset_index()
            per_seed["nested_cases_completed"] = 7
            per_seed["nested_cases_expected"] = 7
            run.write_csv(root / "lofo_results.csv", frame)
            run.write_csv(root / "lofo_per_seed_summary.csv", per_seed)
            audit = verify.verify_nested_aggregation(root, "lofo", 7)
            self.assertTrue(audit.seeds.eq(3).all())
            np.testing.assert_allclose(audit.sample_sd_of_three_seed_means, np.std([.1, .3, .9], ddof=1))
            per_seed.loc[0, "macro_f1"] = .99
            run.write_csv(root / "lofo_per_seed_summary.csv", per_seed)
            with self.assertRaises(AssertionError):
                verify.verify_nested_aggregation(root, "lofo", 7)

    def test_manuscript_handoff_refuses_incomplete_coverage(self):
        with tempfile.TemporaryDirectory(dir=run.OUTPUT / "temp") as directory:
            root = Path(directory)
            with self.assertRaises(RuntimeError):
                handoff.require_ready(root)
            self.assertFalse((root / "MANUSCRIPT_HANDOFF.json").exists())

    def test_handoff_calibration_reversal_is_lower_better(self):
        frame = pd.DataFrame([
            {"seed": 42, "method": method, "ece_all": .01 if method == "taxonomy_blend" else .002,
             "brier_all": .1, "ece_rare_rows": .03, "brier_rare_rows": .2}
            for method in ("taxonomy_blend", *run.STRONG)
        ])
        result = handoff.paired_comparisons({"calibration": frame})
        ece = result.loc[result.metric == "ece_all"]
        self.assertTrue(ece.direction.eq("control_better").all())
        self.assertTrue(ece.taxonomy_benefit.lt(0).all())

    def test_handoff_retrieval_is_not_class_balanced_recall(self):
        frame = pd.DataFrame([
            {"seed": 42, "budget_id": "absolute_1000", "method": method,
             "rare_row_retrieval_recall": .2, "rare_row_retrieval_precision": .8, "rare_hits": 800}
            for method in ("taxonomy_blend", *run.STRONG)
        ])
        result = handoff.paired_comparisons({"alert_budget": frame})
        self.assertNotIn("rare_recall", set(result.metric))
        self.assertIn("rare_row_retrieval_recall", set(result.metric))

    def test_final_summary_uses_seed_means_and_reports_best_nonflat(self):
        frames = {}
        methods = ("taxonomy_blend", *run.STRONG, "cyclic_blend")
        values = dict(zip(methods, (.6, .59, .61, .65)))
        for endpoint, cases in (("lofo", range(7)), ("scarcity", range(6))):
            frames[endpoint] = pd.DataFrame([
                {"seed": seed, "case": case, "method": method,
                 "macro_f1": values[method], "rare_recall": values[method]}
                for seed in (42, 43, 44) for case in cases for method in methods
            ])
        frames["calibration"] = pd.DataFrame([
            {"seed": seed, "method": method, "ece_all": .01, "brier_all": .1,
             "ece_rare_rows": .05, "brier_rare_rows": .2}
            for seed in range(42, 48) for method in methods
        ])
        frames["alert_budget"] = pd.DataFrame([
            {"seed": seed, "budget_id": "absolute_1000", "method": method,
             "rare_row_retrieval_recall": .2, "rare_row_retrieval_precision": .8, "rare_hits": 800}
            for seed in range(42, 48) for method in methods
        ])
        frames["capture_windows"] = pd.DataFrame([
            {"seed": 20260729, "window": f"w{window}", "method": method,
             "macro_f1": values[method], "rare_recall": values[method]}
            for window in range(4) for method in methods
        ])
        comparisons = handoff.paired_comparisons(frames)
        summary, best, pairs = handoff.all_endpoint_tables(frames, comparisons)
        lofo = summary.loc[summary.endpoint == "lofo"]
        self.assertTrue(lofo.n_summary_units.eq(3).all())
        self.assertTrue(lofo.n_seeds.eq(3).all())
        self.assertTrue(best.loc[best.endpoint == "lofo", "best_observed_competitor"].eq("cyclic_blend").all())
        self.assertTrue(pairs.loc[pairs.endpoint == "lofo", "within_seed_cases"].eq(7).all())
        self.assertTrue(pairs.loc[pairs.endpoint == "scarcity", "within_seed_cases"].eq(6).all())

    def test_group_overlap_is_fatal(self):
        with self.assertRaises(ValueError):
            run.role_audit({"fit": pd.DataFrame({run.GROUP: [1, 2]}),
                            "outer": pd.DataFrame({run.GROUP: [2, 3]})})

    def test_frozen_misalignment_is_fatal(self):
        anchor = dict(row_id=np.array([1, 2]), y_true=np.array([0, 1]), rare_ids=np.array([1]))
        source = dict(anchor, row_id=np.array([2, 1]), p=np.eye(2))
        with self.assertRaises(ValueError):
            run.aligned_probability(source, anchor, "p")

    def test_ece_has_row_share_weights(self):
        probability = np.array([[.9, .1], [.9, .1], [.3, .7]])
        expected = 2 / 3 * .1 + 1 / 3 * .7
        self.assertAlmostEqual(run.ece(np.array([0, 0, 0]), probability, nbins=10), expected)

    def test_ece_boundary_is_right_closed(self):
        probability = np.array([[.2, .8], [.19, .81]])
        self.assertAlmostEqual(run.ece(np.array([1, 0]), probability, nbins=10), .505)

    def test_ece_rejects_old_fifteen_bin_default(self):
        with self.assertRaises(ValueError):
            run.ece(np.array([0]), np.array([[.9, .1]]), nbins=15)

    def test_rare_recall_uses_declared_ids(self):
        y = np.array([0, 1, 2, 2])
        p = np.eye(3)[[0, 0, 2, 2]]
        self.assertEqual(run.classification_metrics(y, p, np.array([1]))["rare_recall"], 0)

    def test_novelty_threshold_never_uses_outer_rows(self):
        selection = np.array([[.95, .05], [.9, .1], [.8, .2]])
        first = run.novelty_metrics(selection, np.array([[.99, .01]]), np.array([[.5, .5]]))
        second = run.novelty_metrics(selection, np.array([[.51, .49]]), np.array([[.6, .4]]))
        self.assertEqual(first["msp_selection_fpr_0.05_threshold"], second["msp_selection_fpr_0.05_threshold"])

    def test_auroc_ties_are_half_credit(self):
        p = np.array([[.5, .5], [.5, .5]])
        self.assertEqual(run.novelty_metrics(p, p, p)["msp_unknown_auroc"], .5)

    def test_streamed_novelty_scores_match_legacy_exactly(self):
        rng = np.random.default_rng(187)
        selection, known, unknown = [run.flat.normalise(rng.random((n, 7))) for n in (101, 37, 43)]
        expected = {}
        y = np.r_[np.zeros(len(known)), np.ones(len(unknown))]
        for name, score in (
            ("msp", lambda p: 1 - p.max(axis=1)),
            ("entropy", lambda p: -(p * np.log(np.clip(p, 1e-12, 1))).sum(axis=1)),
        ):
            s, k, u = map(score, (selection, known, unknown))
            expected[f"{name}_unknown_auroc"] = float(run.roc_auc_score(y, np.r_[k, u]))
            for fpr in (.01, .05, .10):
                threshold = float(np.quantile(s, 1 - fpr, method="higher"))
                suffix = f"{name}_selection_fpr_{fpr:g}"
                expected.update({f"{suffix}_threshold": threshold,
                                 f"{suffix}_selection_realized_fpr": float(np.mean(s > threshold)),
                                 f"{suffix}_outer_known_fpr": float(np.mean(k > threshold)),
                                 f"{suffix}_unknown_recall": float(np.mean(u > threshold))})
        actual = run.novelty_metrics_from_scores(*(run.novelty_scores(p) for p in (selection, known, unknown)))
        self.assertEqual(actual, expected)

    def test_factories_cap_threads(self):
        self.assertEqual(run.trf.make_global_model(3, 2).get_params()["n_jobs"], 2)
        self.assertEqual(run.trf.make_lgbm(3, 2).get_params()["n_jobs"], 2)

    def test_temperature_and_newton_meta_smoke(self):
        y = np.tile(np.arange(3), 20)
        xgb = .15 + .55 * np.eye(3)[y]
        lgbm = .2 + .4 * np.eye(3)[y]
        meta = run.fit_meta(xgb, lgbm, y, 42, {"stacker_C": 1.0, "stacker_max_iter": 100})
        self.assertLess(meta["stacker_iterations"], 100)
        self.assertGreaterEqual(meta["xgb_temperature"], .25)
        self.assertLessEqual(meta["lgbm_temperature"], 5)
        p = meta["stacker"].predict_proba(run.flat.probability_features(xgb, lgbm))
        np.testing.assert_allclose(p.sum(axis=1), 1)
        np.testing.assert_array_equal(p.argmax(axis=1), y)

    def test_junction_cannot_redirect_training_to_f(self):
        with self.assertRaises(ValueError):
            run.require_physical_d(run.ROOT / "results" / "ijismd_v15_extensions_20261018")

    def test_deferred_capture_does_not_write_shared_summaries(self):
        with tempfile.TemporaryDirectory(dir=run.OUTPUT / "temp") as directory:
            root = Path(directory)
            window = root / "w1_55_56"
            window.mkdir()
            run.write_json(window / "completed.json", {})
            run.write_csv(window / "results.csv", pd.DataFrame([
                {"window": window.name, "method": method, "macro_f1": .5, "rare_recall": .4}
                for method in ("taxonomy_blend", *run.STRONG)
            ]))
            with patch.object(capture, "OUTPUT", root), patch.object(capture, "DEFER_SHARED_SUMMARY", True), \
                    patch.object(run, "write_csv") as write, patch.object(run, "summarize") as shared:
                capture.summarize()
                self.assertEqual(write.call_count, 2)
                self.assertTrue(all(call.args[0].parent == root for call in write.call_args_list))
                shared.assert_not_called()

    def test_direction_consistency_excludes_partial_seed(self):
        families = ["BruteForce", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web-based"]
        with tempfile.TemporaryDirectory(dir=run.OUTPUT / "temp") as directory:
            root = Path(directory)
            for seed in (42, 43, 44):
                for family in families:
                    if seed == 44 and family == families[-1]:
                        continue
                    name = f"lofo_seed{seed}_{family.replace('-', '_')}"
                    folder = root / "cells" / name
                    folder.mkdir(parents=True)
                    run.write_json(folder / "completed.json", {})
                    run.write_csv(folder / "results.csv", pd.DataFrame([
                        {"cell_id": name, "endpoint": "lofo", "case": family, "seed": seed,
                         "method": method, "macro_f1": .55 if method == "taxonomy_blend" else .5,
                         "rare_recall": .45 if method == "taxonomy_blend" else .4}
                        for method in run.METHODS
                    ]))
            with patch.object(run, "OUTPUT", root):
                coverage = run.summarize({})
            self.assertEqual(next(row for row in coverage if row["endpoint"] == "lofo")["completed_units"], 20)
            direction = pd.read_csv(root / "paired_seed_direction_consistency.csv")
            self.assertTrue((direction.seeds_available == 2).all())
            self.assertTrue((direction.seed_means_positive == 2).all())
            per_seed = pd.read_csv(root / "paired_deltas_per_seed.csv")
            self.assertFalse(per_seed.loc[per_seed.seed == 44, "seed_summary_complete"].any())
            self.assertTrue(per_seed.loc[per_seed.seed.isin([42, 43]), "seed_summary_complete"].all())


if __name__ == "__main__":
    unittest.main()
