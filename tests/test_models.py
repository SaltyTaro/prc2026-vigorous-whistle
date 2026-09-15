import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from prc2026.features import CATEGORY_SOURCES, FEATURE_NAMES, ID, NUMERIC_FEATURES, TARGET
from prc2026_models.encoding import aligned_target, fit_encoder, transform
from prc2026_models.experiments import fit_model, load_model, predict, save_model, validate_config


@pytest.fixture
def baseline_config():
    return json.loads((Path(__file__).resolve().parents[1]
                       / "configs/baselines_v1.json").read_text(encoding="utf-8"))


def sample_features(n=400):
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({ID: np.arange(n, dtype=float)})
    for name in CATEGORY_SOURCES:
        frame[name] = rng.choice(["A", "B", "C"], n)
    for name in NUMERIC_FEATURES:
        frame[name] = rng.normal(size=n).astype("float32")
    return frame[[ID, *FEATURE_NAMES]]


def test_vocabularies_are_fit_only_and_unknowns_are_missing():
    fit = sample_features(20)
    stop = sample_features(10)
    stop["stand"] = "not-in-fit"
    saved = stop.copy()
    encoder = fit_encoder(fit, "lightgbm")
    encoded = transform(stop, encoder)
    assert "not-in-fit" not in encoder["vocabularies"]["stand"]
    assert encoded["stand"].isna().all()
    assert ID not in encoded
    assert_frame_equal(stop, saved)


def test_alignment_uses_ids_not_row_order_and_keeps_extremes():
    features = sample_features(4)
    truth = pd.DataFrame({ID: [3., 0., 2., 1.], TARGET: [20000., -2., 0., 500.]})
    assert aligned_target(features, truth).tolist() == [-2., 500., 0., 20000.]
    truth.loc[0, ID] = 10.
    with pytest.raises(ValueError, match="Unaligned"):
        aligned_target(features, truth)


def test_target_cannot_be_passed_as_a_feature():
    frame = sample_features(10)
    frame[TARGET] = 123
    with pytest.raises(ValueError, match="feature contract"):
        fit_encoder(frame, "lightgbm")


def test_refit_cannot_use_december(baseline_config, protocol):
    baseline_config["refit_months"].append("2025-12")
    with pytest.raises(ValueError, match="reserved audit"):
        validate_config(baseline_config, protocol)


@pytest.mark.parametrize("kind", ["lightgbm", "catboost"])
def test_native_training_and_saved_model_predictions(kind, baseline_config, tmp_path):
    config = baseline_config
    config["threads"] = 2
    config[kind]["rounds"] = 35
    config[kind]["early_stopping_rounds"] = 8
    config[kind]["parameters"]["learning_rate"] = 0.2
    if kind == "lightgbm":
        config[kind]["parameters"].update(min_data_in_leaf=5, num_leaves=15)
    else:
        config[kind]["parameters"]["depth"] = 4
    raw = sample_features()
    y = 300 + raw["runway"].eq("B").to_numpy() * 600 + raw["utc_hour"].to_numpy() * 10
    fit, stop = raw.iloc[:320].copy(), raw.iloc[320:].copy()
    encoder = fit_encoder(fit, kind)
    x, x_stop = transform(fit, encoder), transform(stop, encoder)
    model, best, _, _ = fit_model(kind, config, x, y[:320], x_stop, y[320:])
    values = predict(model, x_stop, kind, 2)
    assert best >= 1
    assert np.mean((values - y[320:])**2) < np.mean((y[:320].mean() - y[320:])**2)
    path = save_model(model, kind, tmp_path, best)
    saved_encoder = json.loads(json.dumps(encoder))
    restored = predict(load_model(path, kind), transform(stop, saved_encoder), kind, 2)
    np.testing.assert_allclose(values, restored, rtol=0, atol=1e-9)


def test_fixed_refit_rejects_stopping_data(baseline_config):
    raw = sample_features(10)
    x = transform(raw, fit_encoder(raw, "lightgbm"))
    with pytest.raises(ValueError, match="fixed refit"):
        fit_model("lightgbm", baseline_config, x, np.ones(10), x, np.ones(10),
                  fixed_iterations=10)
