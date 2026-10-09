import numpy as np
import pytest

from rice_forecasting.models import lag_features, seasonal_naive_forecast


def test_seasonal_naive_uses_latest_matching_season_recursively() -> None:
    history = np.array([10.0, 20.0, 11.0, 21.0])
    np.testing.assert_allclose(seasonal_naive_forecast(history, 6), [11, 21, 11, 21, 11, 21])


def test_lag_features_are_past_only_and_include_known_season() -> None:
    features = lag_features(np.array([1.0, 2.0, 3.0, 4.0]), 2, target_is_maha=True)
    assert features["last_value"] == 4.0
    assert features["target_is_maha"] == 1.0
    assert features["horizon"] == 2.0
    assert "future_actual" not in features


def test_seasonal_naive_rejects_short_history() -> None:
    with pytest.raises(ValueError):
        seasonal_naive_forecast(np.array([1.0]), 1)
