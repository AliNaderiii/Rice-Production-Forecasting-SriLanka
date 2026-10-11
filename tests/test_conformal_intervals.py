import numpy as np
import pandas as pd

from rice_forecasting.conformal import calibrate, finite_sample_quantile


def test_finite_sample_quantile_uses_conformal_rank():
    assert finite_sample_quantile(np.arange(1.0, 10.0), coverage=0.8) == 8.0
    assert np.isinf(finite_sample_quantile(np.arange(1.0, 4.0), coverage=0.95))


def test_calibration_never_uses_not_yet_observed_labels():
    rows = []
    for origin, target, error in [(0, 1, 1), (2, 3, 2), (4, 5, 3), (6, 7, 4),
                                  (8, 9, 5), (10, 11, 6), (12, 13, 7),
                                  (14, 15, 8), (16, 17, 9), (18, 19, 10),
                                  (20, 21, 11)]:
        rows.append({"model": "ETS damped trend seasonal", "origin_idx": origin,
            "target_idx": target, "horizon": 1, "forecast": 100.0,
            "actual": 100.0 + error, "error": float(error)})
    result = calibrate(pd.DataFrame(rows), coverage=0.8, min_calibration=3)
    final = result[result.origin_idx == 20].iloc[0]
    assert final.n_calibration == 10
    assert (result.loc[result.origin_idx == 0, "n_calibration"] == 0).all()
