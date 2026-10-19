"""Bound LightGBM prediction memory without changing estimators or row values."""
from __future__ import annotations

import functools
import numpy as np

BATCH_ROWS = 8192
_installed = False


def bounded_predict(original, model, values, *args, batch_rows=BATCH_ROWS, **kwargs):
    if model.get_params(deep=False).get("n_jobs") != 2:
        raise RuntimeError("Prediction execution requires the frozen n_jobs=2 estimator")
    if "num_threads" in kwargs and kwargs["num_threads"] != 2:
        raise RuntimeError("Prediction num_threads must remain 2")
    kwargs["num_threads"] = 2
    if not isinstance(values, np.ndarray) or len(values) <= batch_rows:
        return original(model, values, *args, **kwargs)
    result = None
    for start in range(0, len(values), batch_rows):
        part = original(model, values[start:start + batch_rows], *args, **kwargs)
        if result is None:
            result = np.empty((len(values), *part.shape[1:]), dtype=part.dtype)
        result[start:start + len(part)] = part
        del part
    return result


def install():
    global _installed
    if _installed:
        return
    from lightgbm import LGBMClassifier
    original = LGBMClassifier.predict_proba

    @functools.wraps(original)
    def predict(self, values, *args, **kwargs):
        return bounded_predict(original, self, values, *args, **kwargs)

    LGBMClassifier.predict_proba = predict
    _installed = True
