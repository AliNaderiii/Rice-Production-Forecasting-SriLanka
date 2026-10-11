from rice_forecasting.scenario_benchmark import FEATURES
from rice_forecasting.data import TARGET


def test_nowcast_predictors_exclude_target_and_are_explicit():
    assert TARGET not in FEATURES
    assert "Harvested (*000  Acres)" in FEATURES
    assert "Rainfall(mm)" in FEATURES
    assert "Production (*000  Mt.)" not in FEATURES
