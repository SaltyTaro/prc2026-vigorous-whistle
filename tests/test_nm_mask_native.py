"""Exercise weighting, fit-only categories and native artifact reloads."""
from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from prc2026.features import ID
from prc2026_nm_mask.features import FEATURE_NAMES, CATEGORICAL_COLUMNS, offset
from prc2026_nm_mask.common import sample_weights
from prc2026_nm_mask.native import fit_native, save_native, predict_native, load_native
from prc2026_weather.encoding import transform


def matrix(n=100, start=0):
    frame = pd.DataFrame({name: (pd.Series(["__MISSING__"] * n, dtype="string")
                                if name in CATEGORICAL_COLUMNS else np.zeros(n, dtype="float32"))
                          for name in FEATURE_NAMES})
    frame.insert(0, ID, np.arange(start, start + n, dtype=float))
    frame["utc_hour"] = np.arange(n, dtype="float32") % 24
    frame["schedule_to_movement_s"] = np.arange(n, dtype="float32") * 20
    frame["conditional_schedule_offset_s"] = np.where(np.arange(n) % 3 == 0, -5000., 0.).astype("float32")
    frame["airport"] = "LIRF"
    frame["stand"] = "known"
    return frame


def specification(kind):
    root = Path(__file__).resolve().parents[1]
    variants = json.loads((root / "configs/nm_mask_v1.json").read_text())["variants"]
    variant = deepcopy(variants["lgb_augmented" if kind == "lightgbm" else "cat_augmented"])
    variant["settings"][kind]["rounds"] = 18
    variant["settings"][kind]["early_stopping_rounds"] = 4
    return variant


@pytest.mark.parametrize("kind", ["lightgbm", "catboost"])
def test_weighted_native_fit_uses_fit_only_vocab_and_reloads_exactly(tmp_path, kind):
    fit, stop = matrix(), matrix(25, 200)
    stop["stand"] = "unseen_in_stopping"
    target = fit.utc_hour.to_numpy(dtype=float) * 25 + 100
    stop_target = stop.utc_hour.to_numpy(dtype=float) * 25 + 100
    natural = np.arange(len(fit)) % 5 == 0
    weights = sample_weights(natural, "augmented")
    model, encoder, history, iterations, parameters = fit_native(fit, target, weights, stop, stop_target, specification(kind))
    assert "unseen_in_stopping" not in encoder["vocabularies"]["stand"]
    assert transform(stop, encoder).stand.astype(str).eq("__MISSING__").all()
    before = predict_native(model, encoder, stop, iterations)
    assert np.isfinite(before).all()
    assert (before < 0).any()  # Signed restoration is not clipped.
    save_native(tmp_path, model, encoder, history, iterations, parameters)
    restored, saved_encoder, node = load_native(tmp_path)
    np.testing.assert_array_equal(before, predict_native(restored, saved_encoder, stop, node["iterations"]))
    assert node["iterations"] == iterations and history


@pytest.mark.parametrize("bad_weight", [0., -1., np.nan, np.inf])
def test_invalid_native_weights_fail_before_fit(bad_weight):
    fit, stop = matrix(), matrix(25, 200)
    weights = np.ones(len(fit))
    weights[0] = bad_weight
    with pytest.raises(ValueError):
        fit_native(fit, np.ones(len(fit)), weights, stop, np.ones(len(stop)), specification("lightgbm"))
