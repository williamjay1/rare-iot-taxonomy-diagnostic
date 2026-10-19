"""Score unknown flagging from frozen chronology predictions, without refits."""
from pathlib import Path
import hashlib
import json
import os

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
METHODS = [
    'flat_xgb', 'flat_lgbm', 'flat_equal_blend', 'temperature_flat_blend',
    'newton_flat_stacker', 'real_hierarchy', 'real_blend',
    'cyclic_hierarchy', 'cyclic_blend', 'support_hierarchy', 'support_blend',
]


def main():
    assert (HERE / 'unknown_flagging_protocol.md').exists()
    original = pd.read_csv(HERE / 'temporal_results.csv')
    assert len(original) == 33
    rows = []
    base = None
    for seed in (42, 43, 44):
        for method in METHODS:
            path = HERE / f'seed{seed}' / f'predictions_{method}.npz'
            with np.load(path) as saved:
                rid, y, classes, p = (saved[k] for k in
                                      ('row_id', 'y_true', 'classes', 'probability'))
            assert len(rid) == len(y) == len(p) == 300000
            assert p.shape[1] == len(classes)
            assert np.isfinite(p).all() and p.min() >= 0
            assert np.allclose(p.sum(1), 1., atol=2e-6)
            if base is None:
                base = (rid.copy(), y.copy(), classes.copy())
            assert all(np.array_equal(a, b) for a, b in zip((rid, y, classes), base))
            unknown = ~np.isin(y, classes)
            assert unknown.sum() == 129183
            entry = original[(original.seed == seed) & (original.method == method)].iloc[0]
            assert int(entry.n_test_unseen) == unknown.sum()
            score = 1. - p.max(1).astype(np.float64)
            rows.append({
                'seed': seed, 'method': method, 'test_rows': len(y),
                'unknown_rows': int(unknown.sum()),
                'unknown_prevalence': float(unknown.mean()),
                'unknown_auroc': roc_auc_score(unknown, score),
                'unknown_average_precision': average_precision_score(unknown, score),
                'prediction_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'role': 'one frozen chronological future period',
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(HERE / 'unknown_flagging_results.csv', index=False)
    summary = frame.groupby('method').agg(
        model_seeds=('seed', 'nunique'),
        auroc_mean=('unknown_auroc', 'mean'),
        auroc_model_seed_sd=('unknown_auroc', 'std'),
        average_precision_mean=('unknown_average_precision', 'mean'),
        average_precision_model_seed_sd=('unknown_average_precision', 'std'),
        unknown_prevalence=('unknown_prevalence', 'first'),
    ).reset_index()
    summary.to_csv(HERE / 'unknown_flagging_summary.csv', index=False)
    audit = {
        'status': 'COMPLETE_PASS', 'rows': len(frame),
        'protocol_sha256': hashlib.sha256((HERE / 'unknown_flagging_protocol.md').read_bytes()).hexdigest(),
        'post_review_secondary': True, 'fit_changed': False,
        'time_roles_changed': False, 'assessment_selected_threshold': False,
        'scope': 'Three model seeds share one future period; not rare class recall.',
    }
    (HERE / 'UNKNOWN_FLAGGING_AUDIT.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
