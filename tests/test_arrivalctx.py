"""Protected-route, native-fit and context checks for the separate arrival batch."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prc2026.artifacts import read_json
from prc2026.features import ID, TARGET
from prc2026.validation import identity_hash
from prc2026_models.experiments import load_model, predict
from prc2026_arrivalctx.encoding import contract, transform
from prc2026_arrivalctx.native import fit_node
from prc2026_arrivalctx.inference import node_prediction
from prc2026_arrivalctx.features import context
from prc2026_arrivalctx.training import validate_config
from test_arrival_context_audit import frames
from audit_arrival_context import direct_check


def test_arrivalctx_features_match_independent_strict_past_scan():
    query, arrivals = frames()
    result = context(query, arrivals)
    assert result.iloc[0].arrival_airport_3600s_count == 2
    assert result.iloc[0].arrival_airport_3600s_raw_mean == 900
    assert direct_check(query, arrivals, result)["maximum_absolute_difference"] == 0


@pytest.mark.parametrize("kind", ["lightgbm", "catboost"])
def test_arrivalctx_complete_native_fit_records_ids_and_restores_offset(tmp_path, kind):
    spec = {"kind": kind, "features": "ground_weather_arrival", "target": "conditional_schedule_residual", "route": "clock_missing"}
    names, cats = contract(spec)
    rng = np.random.default_rng(89)
    frame = pd.DataFrame({name: (pd.Series(["known"] * 80, dtype="string") if name in cats else rng.normal(size=80)) for name in names})
    frame.insert(0, ID, np.arange(80, dtype=float))
    frame["nmclock_available"], frame["nmclock_proxy_s"], frame["conditional_schedule_offset_s"] = 0., np.nan, 350.
    truth = pd.DataFrame({ID: frame[ID], TARGET: 500. + 30 * frame[names[-1]]})
    parameters = ({"objective": "regression", "metric": "rmse", "verbosity": -1, "num_leaves": 3, "min_data_in_leaf": 2, "force_col_wise": True}
                  if kind == "lightgbm" else {"loss_function": "RMSE", "task_type": "CPU", "depth": 2, "allow_writing_files": False})
    settings = {"threads": 1, "seed": 89, kind: {"rounds": 12, "early_stopping_rounds": 3, "parameters": parameters}}
    node = fit_node(tmp_path, "test", spec, settings, frame.iloc[:60], truth.iloc[:60], frame.iloc[60:], truth.iloc[60:])
    assert node["schema"] == "arrivalctx" and node["fit_rows"] == 60 and node["early_stop_rows"] == 20
    assert node["fit_ids_sha256"] == identity_hash(frame.iloc[:60][ID])
    assert node["early_stop_ids_sha256"] == identity_hash(frame.iloc[60:][ID])
    assert node["max_model_reload_difference_seconds"] == 0
    raw = predict(load_model(tmp_path / node["model_path"], kind), transform(frame, read_json(tmp_path / node["encoder_path"])), kind, 1)
    np.testing.assert_array_equal(node_prediction(tmp_path, node, frame, 1), raw + 350.)
    changed = frame.copy()
    changed["nmclock_available"], changed["nmclock_proxy_s"] = 1., 900.
    with pytest.raises(ValueError):
        node_prediction(tmp_path, node, changed, 1)


@pytest.mark.parametrize("change", ["margin", "december", "route", "settings"])
def test_arrivalctx_keeps_matched_learners_and_declared_selection(change):
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/arrivalctx_v1.json").read_text())
    protocol = json.loads((root / "configs/validation_v1.json").read_text())
    validate_config(root, config, protocol)
    altered = deepcopy(config)
    if change == "margin":
        altered["minimum_improvement_fraction_each_fold"] = .004
    elif change == "december":
        altered["refit_months"].append("2025-12")
    elif change == "route":
        altered["variants"]["lgb_arrival"]["route"] = "all"
    else:
        altered["variants"]["lgb_arrival"]["settings"]["lightgbm"]["parameters"]["num_leaves"] = 63
    with pytest.raises(ValueError):
        validate_config(root, altered, protocol)
