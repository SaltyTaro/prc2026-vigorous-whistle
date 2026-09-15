from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from prc2026.features import ID, TIME, TARGET, BLOCK
from prc2026_weather.features import (CATEGORICAL_COLUMNS, FEATURE_NAMES, INPUT_COLUMNS, OBS_NAMES,
    WEATHER_NAMES, observation_features, range_maximum, runway_geometry, weather_features)
from prc2026_weather.encoding import fit_encoder, transform
from prc2026_weather.inference import node_prediction
from prc2026_weather.training import fit_node, settings_for, validate_config
from prc2026_phase4.features import prediction_offset


def raw_weather(times):
    n = len(times)
    out = pd.DataFrame({"station": ["EGLL"]*n, "valid": pd.to_datetime(times, utc=True),
                        "tmpf": 32., "dwpf": 23., "relh": 80., "drct": 270., "sknt": 10., "gust": 20.,
                        "alti": 29.92, "vsby": 1., "wxcodes": ["-FZRASN BR"]*n})
    for i in range(1, 5):
        out[f"skyc{i}"] = "BKN" if i == 1 else pd.NA
        out[f"skyl{i}"] = 1000. if i == 1 else np.nan
    return out


def queries(times, airport="EGLL"):
    return pd.DataFrame({ID: np.arange(len(times), dtype=float), TIME: pd.to_datetime(times, utc=True),
                         "ADEP_mvt": airport, "RUNWAY_mvt": "27R"})


def geometry():
    return pd.DataFrame({"airport": ["EGLL"], "runway": ["27R"], "bearing": [270.], "length_m": [3900.]})


def test_weather_units_wind_signs_and_codes():
    obs = observation_features(raw_weather(["2025-03-01T09:30Z"]))
    x = weather_features(queries(["2025-03-01T10:00Z"]), obs, geometry()).iloc[0]
    assert x.wx_temperature_c == 0
    assert x.wx_dewpoint_spread_c == 5
    assert x.wx_speed_mps == pytest.approx(10*1852/3600)
    assert x.wx_headwind_mps == pytest.approx(x.wx_speed_mps)
    assert x.wx_tailwind_mps == 0
    assert x.wx_abs_crosswind_mps == pytest.approx(0, abs=1e-6)
    assert x.wx_pressure_hpa == pytest.approx(1013.2075, abs=.001)
    assert x.wx_visibility_km == pytest.approx(1.609344)
    assert x.wx_ceiling_m == pytest.approx(304.8)
    assert x.wx_rain == x.wx_snow == x.wx_mist == x.wx_freezing_weather == 1
    assert x.wx_fog == x.wx_thunder == 0


def test_publication_delay_and_maximum_age_boundaries_keep_every_row():
    obs = observation_features(raw_weather(["2025-03-01T09:30Z"]))
    query = queries(["2025-03-01T09:44Z", "2025-03-01T09:45Z", "2025-03-01T11:00Z", "2025-03-01T11:01Z"])
    x = weather_features(query, obs, geometry())
    assert x.wx_missing.tolist() == [1, 0, 0, 1]
    assert x.loc[1, "wx_age_minutes"] == 15
    assert x.loc[2, "wx_age_minutes"] == 90
    assert x[ID].tolist() == query[ID].tolist()
    assert x.loc[[0, 3], "wx_temperature_c"].isna().all()
    assert weather_features(queries(["2025-03-01T10:00Z"], "ZZZZ"), obs, geometry()).wx_missing.iloc[0] == 1


def test_future_reports_and_forbidden_values_cannot_change_features():
    raw = raw_weather(["2025-03-01T08:30Z", "2025-03-01T09:30Z", "2025-03-01T09:50Z"])
    query = queries(["2025-03-01T10:00Z", "2025-03-01T09:45Z"])
    original = weather_features(query, observation_features(raw), geometry())
    raw.loc[2, ["tmpf", "sknt", "gust"]] = [120., 150., 180.]
    for name in [TARGET, BLOCK, "AOBT_3_flt", "LOBT_flt", "ARVT_3_flt"]:
        query[name] = 100000000
        assert name not in INPUT_COLUMNS
    actual = weather_features(query, observation_features(raw.sample(frac=1, random_state=2)), geometry())
    pd.testing.assert_frame_equal(original, actual, check_exact=True)
    pd.testing.assert_frame_equal(original.iloc[::-1].reset_index(drop=True),
                                  weather_features(query.iloc[::-1], observation_features(raw), geometry()), check_exact=True)
    assert original.wx_change60_temperature_c.eq(0).all()
    assert original.wx_reports_3h.eq(2).all()


def test_missing_wind_sky_and_duplicate_reports_are_explicit():
    raw = raw_weather(["2025-03-01T09:30Z"])
    raw.loc[0, ["sknt", "drct", "gust", "wxcodes", "skyc1", "skyl1"]] = [0, np.nan, np.nan, pd.NA, pd.NA, np.nan]
    x = weather_features(queries(["2025-03-01T10:00Z"]), observation_features(raw), geometry()).iloc[0]
    assert x.wx_direction_missing == x.wx_sky_missing == x.wx_no_weather_code == 1
    assert x.wx_gust_reported == x.wx_ceiling_reported == x.wx_headwind_mps == 0
    assert np.isnan(x.wx_gust_mps) and np.isnan(x.wx_cloud_fraction)
    with pytest.raises(ValueError, match="duplicate"):
        observation_features(pd.concat([raw, raw], ignore_index=True))


def test_maxima_use_exact_query_window_with_irregular_reports():
    rng = np.random.default_rng(4)
    values = rng.normal(size=57)
    values[2:6] = np.nan
    left = rng.integers(0, len(values)+1, size=160)
    right = np.array([rng.integers(x, len(values)+1) for x in left])
    actual = range_maximum(values, left, right)
    expected = np.array([np.nanmax(values[a:b]) if a < b and np.isfinite(values[a:b]).any() else np.nan for a, b in zip(left, right)])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=0, equal_nan=True)


def test_geodesic_bearing_and_unknown_runway_fallback():
    raw = pd.DataFrame({"airport_ident": ["EGLL"], "le_ident": ["09R"], "he_ident": ["27L"],
                        "le_latitude_deg": [0.], "le_longitude_deg": [0.], "he_latitude_deg": [0.],
                        "he_longitude_deg": [1.], "length_ft": [10000.]})
    geo = runway_geometry(raw)
    assert geo.bearing.tolist() == [90., 270.]
    query = queries(["2025-03-01T10:00Z"])
    query["RUNWAY_mvt"] = "9L"
    x = weather_features(query, observation_features(raw_weather(["2025-03-01T09:30Z"])), geo).iloc[0]
    assert x.wx_runway_external_bearing == 0
    assert x.wx_tailwind_mps == pytest.approx(10*1852/3600)
    assert np.isnan(x.wx_runway_length_m)


def test_december_and_unapproved_protocol_changes_fail():
    obs = observation_features(raw_weather(["2025-03-01T09:30Z"]))
    with pytest.raises(ValueError, match="Reserved audit"):
        weather_features(queries(["2025-12-01T10:00Z"]), obs, geometry())
    cfg = json.loads((Path(__file__).resolve().parents[1]/"configs/weather_v1.json").read_text())
    protocol = json.loads((Path(__file__).resolve().parents[1]/"configs/validation_v1.json").read_text())
    validate_config(cfg, protocol)
    changed = deepcopy(cfg)
    changed["refit_months"].append("2025-12")
    with pytest.raises(ValueError, match="reserved audit"):
        validate_config(changed, protocol)
    changed = deepcopy(cfg)
    changed["candidate_families"]["invalid"] = {"lgb_weather": .5}
    with pytest.raises(ValueError, match="sum to one"):
        validate_config(changed, protocol)


@pytest.mark.parametrize("kind", ["lightgbm", "catboost"])
def test_native_weather_models_preserve_labels_fit_vocab_and_reload(tmp_path, kind):
    rng = np.random.default_rng(7)
    n = 240
    frame = pd.DataFrame({ID: np.arange(n, dtype=float), **{
        name: rng.choice(["A", "B"], n) if name in CATEGORICAL_COLUMNS else rng.normal(size=n).astype("float32")
        for name in FEATURE_NAMES}})
    frame["airport"] = "LIRF"
    frame["lirf_missing_record"] = 1.
    frame["conditional_schedule_offset_s"] = 90000.
    target = prediction_offset(frame, "lirf_missing_schedule")+50*frame["wx_speed_mps"].to_numpy()+800
    target[0], target[1] = -250., 120000.
    truth = pd.DataFrame({ID: frame[ID], TARGET: target, "FLIGHT_ID_mvt": np.arange(n, dtype=float)})
    spec = {"kind": kind, "features": "ground_weather", "route": "all"}
    cfg = json.loads((Path(__file__).resolve().parents[1]/"configs/weather_v1.json").read_text())
    settings = settings_for(cfg, spec)
    settings["threads"] = 1
    settings[kind]["rounds"] = 12
    settings[kind]["early_stopping_rounds"] = 3
    settings[kind]["parameters"].update({"min_data_in_leaf": 10, "num_leaves": 7} if kind == "lightgbm" else {"depth": 3})
    fit, stop = frame.iloc[:180], frame.iloc[180:].copy()
    stop["operator"] = "STOP_ONLY"
    encoder = fit_encoder(fit, spec)
    assert "STOP_ONLY" not in encoder["vocabularies"]["operator"]
    assert transform(stop, encoder)["operator"].astype(str).eq("__MISSING__").all()
    assert ID not in encoder["feature_names"]
    node = fit_node(tmp_path, "weather", spec, settings, fit, truth.iloc[:180], stop, truth.iloc[180:])
    assert node["fit_rows"] == 180 and node["max_model_reload_difference_seconds"] == 0
    assert np.isfinite(node_prediction(tmp_path, node, stop, 1)).all()
    with pytest.raises(ValueError, match="fixed refit"):
        fit_node(tmp_path, "invalid", spec, settings, fit, truth.iloc[:180], stop, truth.iloc[180:], iterations=3)
