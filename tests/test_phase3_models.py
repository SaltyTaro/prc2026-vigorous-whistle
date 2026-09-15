import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prc2026.features import ID, TARGET
from prc2026_models.experiments import load_model, predict, save_model
from prc2026_phase3.encoding import aligned_target, fit_encoder, transform
from prc2026_phase3.features import feature_contract
from prc2026_phase3.training import fit_model, validate_config


def sample(n=400, families=None):
    families = ["layout"] if families is None else families
    names, categories = feature_contract(families)
    rng = np.random.default_rng(54)
    features = pd.DataFrame({ID: np.arange(n, dtype=float)})
    for name in names:
        features[name] = rng.choice(["A", "B", "C"], n) if name in categories else rng.normal(size=n).astype("float32")
    return features[[ID, *names]]


def config():
    return json.loads((Path(__file__).resolve().parents[1] / "configs/phase3_v1.json").read_text())


def test_new_categories_are_fit_only_and_contract_is_strict():
    features = sample(20)
    encoder = fit_encoder(features, ["layout"])
    unseen = features.copy()
    unseen["airport_stand_runway"] = "new stand at ranking time"
    assert transform(unseen, encoder)["airport_stand_runway"].isna().all()
    assert "new stand at ranking time" not in encoder["vocabularies"]["airport_stand_runway"]
    unseen[TARGET] = 123
    with pytest.raises(ValueError, match="feature contract"):
        transform(unseen, encoder)


def test_new_encoder_aligns_and_preserves_all_labels():
    features = sample(4)
    labels = pd.DataFrame({ID: [3., 2., 1., 0.], TARGET: [150000., 0., -12., 1000.]})
    assert aligned_target(features, labels, ["layout"]).tolist() == [1000., -12., 0., 150000.]
    labels.loc[0, ID] = 999
    with pytest.raises(ValueError, match="Unaligned"):
        aligned_target(features, labels, ["layout"])


def test_phase3_keeps_model_settings_and_audit_exclusion(protocol):
    current = config()
    root = Path(__file__).resolve().parents[1]
    baseline = json.loads((root / "configs/baselines_v1.json").read_text())
    assert current["lightgbm"] == baseline["lightgbm"]
    validate_config(current, protocol)
    current["refit_months"].append("2025-12")
    with pytest.raises(ValueError, match="reserved audit"):
        validate_config(current, protocol)


def test_native_extended_model_and_saved_encoder_predictions(tmp_path):
    current = config()
    current["threads"] = 2
    current["lightgbm"].update(rounds=40, early_stopping_rounds=8)
    current["lightgbm"]["parameters"].update(num_leaves=15, min_data_in_leaf=5, learning_rate=0.2)
    raw = sample()
    labels = 300 + 700*raw["airport_stand_runway"].eq("A").to_numpy() + raw["stand_number"].to_numpy()*20
    encoder = fit_encoder(raw.iloc[:320], ["layout"])
    x, stop = transform(raw.iloc[:320], encoder), transform(raw.iloc[320:], encoder)
    model, best, _, _ = fit_model(current, encoder, x, labels[:320], stop, labels[320:])
    predictions = predict(model, stop, "lightgbm", 2)
    assert np.mean((predictions-labels[320:])**2) < np.mean((labels[:320].mean()-labels[320:])**2)
    path = save_model(model, "lightgbm", tmp_path, best)
    restored = predict(load_model(path, "lightgbm"), transform(raw.iloc[320:], json.loads(json.dumps(encoder))), "lightgbm", 2)
    np.testing.assert_allclose(predictions, restored, rtol=0, atol=1e-9)
    with pytest.raises(ValueError, match="fixed refit"):
        fit_model(current, encoder, x, labels[:320], stop, labels[320:], iterations=5)
