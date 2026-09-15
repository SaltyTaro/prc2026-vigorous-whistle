"""Matched settings and exhaustive disjoint airport routing are required."""
from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from prc2026.features import ID
from prc2026_surface_airports import inference
from prc2026_surface_airports.training import validate_config


@pytest.mark.parametrize("change", ["airports", "settings", "margin", "december"])
def test_surface_airports_keeps_the_declared_complete_comparison(change):
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/surface_airports_v1.json").read_text())
    protocol = json.loads((root / "configs/validation_v1.json").read_text())
    validate_config(root, config, protocol)
    altered = deepcopy(config)
    if change == "airports":
        altered["airports"].remove("LIRF")
    elif change == "settings":
        altered["settings"]["lightgbm"]["parameters"]["num_leaves"] = 63
    elif change == "margin":
        altered["minimum_improvement_fraction_each_fold"] = .004
    else:
        altered["refit_months"].append("2025-12")
    with pytest.raises(ValueError):
        validate_config(root, altered, protocol)


def test_airport_routing_preserves_shuffled_query_order(monkeypatch):
    frame = pd.DataFrame({ID: [1., 2., 3.], "airport": ["LFPG", "EDDF", "LFPG"], "nmclock_available": [1., 1., 1.], "nmclock_proxy_s": [10., 20., 30.]})
    monkeypatch.setattr(inference, "node_prediction", lambda run, node, features, threads: features.nmclock_proxy_s.to_numpy() + node["constant"])
    values = inference.airport_prediction(Path("."), {"EDDF": {"constant": 100.}, "LFPG": {"constant": 200.}}, frame, 1)
    np.testing.assert_array_equal(values, [210., 120., 230.])


def test_airport_routing_rejects_uncovered_and_missing_clock_rows(monkeypatch):
    frame = pd.DataFrame({ID: [1.], "airport": ["LFPG"], "nmclock_available": [1.], "nmclock_proxy_s": [10.]})
    with pytest.raises(ValueError):
        inference.airport_prediction(Path("."), {}, frame, 1)
    frame["nmclock_available"], frame["nmclock_proxy_s"] = 0., np.nan
    with pytest.raises(ValueError):
        inference.airport_prediction(Path("."), {"LFPG": {}}, frame, 1)
