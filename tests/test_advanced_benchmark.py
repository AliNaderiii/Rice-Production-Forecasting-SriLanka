import numpy as np

from rice_forecasting.advanced_benchmark import features, mase_scale


def test_features_use_only_history_and_known_calendar():
    result = features(np.array([10, 11, 12, 13, 14, 15, 16, 17], dtype=float), 2, "Maha")
    assert result["lag_1"] == 17
    assert result["lag_8"] == 10
    assert result["horizon"] == 2
    assert result["target_is_maha"] == 1
    assert "future_actual" not in result


def test_mase_scale_is_positive():
    assert mase_scale(np.array([1.0, 2.0, 3.0, 5.0])) == 2.5
    assert mase_scale(np.array([1.0, 1.0, 1.0, 1.0])) > 0
