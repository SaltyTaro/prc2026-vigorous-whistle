import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from prc2026_nn.model import NumericEncoder, TabularNet
from prc2026_nn.preprocess import MISSING_INDEX, MISSING_TOKEN, RARE_INDEX, Preprocessor
from prc2026_nn.train import build_model, load, predict, save, train

ARCH = {"numeric_mode": "ple", "d_num": 4, "cat_cap": 4, "width": 32, "depth": 2, "dropout": 0.0, "k": 1}
TRAINING = {"epochs": 6, "batch_size": 128, "lr": 3e-3, "weight_decay": 1e-5, "warmup_epochs": 0.5, "min_lr_fraction": 0.05, "grad_clip": 5.0, "ema_decay": 0.0}


def synthetic(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.exponential(size=n)
    b[rng.random(n) < 0.2] = np.nan
    cat = rng.choice(["x", "y", "z"], size=n)
    cat[:5] = "rare"
    y = 300 + 80 * np.sin(2 * a) + 50 * np.nan_to_num(b, nan=-1.0) + np.where(cat == "y", 60.0, 0.0) + rng.normal(scale=5, size=n)
    return pd.DataFrame({"a": a, "b": b, "c": pd.Series(cat, dtype="string")}), y


def test_encoder_codes_follow_the_edges():
    edges = np.array([[0.0, 1.0, 2.0, 4.0, 4.0], [5.0, 6.0, 6.0, 6.0, 6.0]])
    enc = NumericEncoder(edges, np.array([3, 1]), 0)
    codes, nan = enc.encode(torch.tensor([[1.5, 5.5], [-1.0, 9.0], [10.0, float("nan")]]))
    assert torch.allclose(codes[0, 0], torch.tensor([1.0, 0.5, 0.0, 0.0]))
    assert torch.allclose(codes[1, 0], torch.zeros(4))
    assert torch.allclose(codes[2, 0], torch.tensor([1.0, 1.0, 1.0, 0.0]))  # padded bin stays zero
    assert torch.allclose(codes[0, 1], torch.tensor([0.5, 0.0, 0.0, 0.0]))
    assert torch.allclose(codes[2, 1], torch.zeros(4)) and bool(nan[2, 1]) and not bool(nan[2, 0])


def test_preprocessor_maps_missing_rare_and_unseen(tmp_path):
    frame, y = synthetic()
    frame.loc[10, "c"] = pd.NA
    pre = Preprocessor.fit(frame, ["a", "b"], ["c"], y, bins=8, min_count=10)
    assert pre.vocab["c"] == ["x", "y", "z"] and pre.cardinalities() == [5]
    probe = pd.DataFrame({"a": [0.0, np.nan], "b": [1.0, 2.0], "c": pd.Series(["y", "never"], dtype="string")})
    x_num, x_cat = pre.transform(probe)
    assert x_cat[:, 0].tolist() == [2 + 1, RARE_INDEX] and np.isnan(x_num[1, 0])
    _, full = pre.transform(frame)
    assert full[10, 0] == MISSING_INDEX and full[0, 0] == RARE_INDEX
    pre.save(tmp_path)
    again = Preprocessor.load(tmp_path)
    assert np.array_equal(again.edges, pre.edges) and again.vocab == pre.vocab and again.target_scale == pre.target_scale
    assert MISSING_TOKEN not in pre.vocab["c"]


@pytest.mark.parametrize("mode,k", [("ple", 1), ("rank", 1), ("ple", 2)])
def test_training_learns_and_reloads_exactly(tmp_path, mode, k):
    torch.set_num_threads(1)
    frame, y = synthetic()
    fit, hold = slice(0, 2400), slice(2400, None)
    pre = Preprocessor.fit(frame.iloc[fit], ["a", "b"], ["c"], y[fit], bins=16, min_count=10)
    x_num, x_cat = pre.transform(frame)
    arch = {**ARCH, "numeric_mode": mode, "k": k}
    model = build_model(pre, arch, seed=7)
    history = train(model, pre, x_num[fit], x_cat[fit], pre.scale_target(y[fit]), TRAINING, seed=7,
                     monitor=(x_num[hold], x_cat[hold], y[hold]), log=lambda _: None)
    baseline = float(np.sqrt(np.mean((y[hold] - y[fit].mean()) ** 2)))
    assert history[-1]["monitor_rmse"] < (0.5 if mode == "ple" else 0.7) * baseline  # a scalar rank input learns the sine more slowly
    values = predict(model, pre, x_num[hold], x_cat[hold])
    save(model, pre, arch, tmp_path)
    reloaded, pre2 = load(tmp_path)
    assert np.array_equal(predict(reloaded, pre2, x_num[hold], x_cat[hold]), values)


def test_same_seed_same_model():
    torch.set_num_threads(1)
    frame, y = synthetic(n=800)
    pre = Preprocessor.fit(frame, ["a", "b"], ["c"], y, bins=8)
    x_num, x_cat = pre.transform(frame)
    runs = []
    for _ in range(2):
        model = build_model(pre, {**ARCH, "dropout": 0.1}, seed=3)
        train(model, pre, x_num, x_cat, pre.scale_target(y), {**TRAINING, "epochs": 2, "ema_decay": 0.99}, seed=3, log=lambda _: None)
        runs.append(predict(model, pre, x_num, x_cat))
    assert np.array_equal(runs[0], runs[1])
