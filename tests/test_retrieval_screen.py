"""Retrieval screen helpers (tools/retrieval_screen.py): the neighbour search, the neighbour means and the embedding path."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
torch = pytest.importorskip("torch")
import retrieval_screen as screen  # noqa: E402


def brute(queries, candidates, k, allowed=None):
    d = np.sqrt(((queries[:, None, :].astype(np.float64) - candidates[None, :, :].astype(np.float64)) ** 2).sum(axis=2))
    if allowed is not None:
        d = np.where(allowed, d, np.inf)
    index = np.argsort(d, axis=1, kind="stable")[:, :k]
    return index, np.take_along_axis(d, index, axis=1)


def test_nearest_matches_brute_force_in_both_scopes():
    rng = np.random.default_rng(3)
    queries, candidates = rng.normal(size=(37, 6)).astype(np.float32), rng.normal(size=(400, 6)).astype(np.float32)
    qg, cg = rng.integers(0, 3, size=37), rng.integers(0, 3, size=400)
    found = screen.nearest(queries, candidates, qg, cg, k=8, block=10)
    index, distance = brute(queries, candidates, 8)
    assert np.allclose(found["any"][1], distance, atol=1e-3)
    assert np.allclose(np.take_along_axis(np.sqrt(((queries[:, None] - candidates[None]) ** 2).sum(axis=2)), found["any"][0].astype(np.int64), axis=1), distance, atol=1e-3)
    index, distance = brute(queries, candidates, 8, allowed=qg[:, None] == cg[None, :])
    assert np.allclose(found["same"][1], distance, atol=1e-3)
    assert (cg[found["same"][0]] == qg[:, None]).all()
    assert (np.diff(found["any"][1], axis=1) >= -1e-6).all() and (np.diff(found["same"][1], axis=1) >= -1e-6).all()


def test_nearest_refuses_a_group_with_too_few_candidates():
    queries, candidates = np.zeros((2, 3), dtype=np.float32), np.ones((5, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        screen.nearest(queries, candidates, np.array([0, 1]), np.array([0, 0, 0, 0, 1]), k=2)


def test_summarise_takes_the_nearest_k_and_clips_each_residual_first():
    index = np.tile(np.arange(64, dtype=np.int32), (2, 1))
    distance = np.tile(np.arange(64, dtype=np.float32), (2, 1))
    residual = np.zeros(64)
    residual[0], residual[1] = 90000.0, -100.0
    prediction = np.arange(64, dtype=float)
    same = np.ones((2, 64), dtype=bool)
    out = screen.summarise(index, distance, residual, prediction, same, "l2_any")
    assert out["res_l2_any_4"][0] == pytest.approx((90000.0 - 100.0) / 4)
    assert out["resc_l2_any_4"][0] == pytest.approx((1800.0 - 100.0) / 4)
    assert out["res_l2_any_64"][0] == pytest.approx((90000.0 - 100.0) / 64)
    assert out["prednb_l2_any_16"][1] == pytest.approx(7.5)
    assert out["dist_l2_any_4"][0] == pytest.approx(1.5)
    assert set(out) == {f"{name}_l2_any_{k}" for name in ("res", "resc", "prednb", "dist", "same") for k in screen.KS}


def test_embed_reproduces_the_network_prediction():
    from prc2026_nn.model import TabularNet
    torch.manual_seed(0)
    edges = np.tile(np.linspace(0.0, 1.0, 5), (3, 1))
    model = TabularNet(edges, np.array([4, 4, 4]), [5, 7], d_num=4, width=32, depth=2, dropout=0.1, k=1)
    rng = np.random.default_rng(1)
    x_num = rng.random((50, 3)).astype(np.float32)
    x_num[3, 1] = np.nan
    x_cat = np.stack([rng.integers(0, 5, 50), rng.integers(0, 7, 50)], axis=1).astype(np.int64)
    embedding, scaled = screen.embed(model, x_num, x_cat, batch=16)
    model.eval()
    with torch.no_grad():
        expected = model(torch.from_numpy(x_num), torch.from_numpy(x_cat))[:, 0].double().numpy()
    assert embedding.shape == (50, 32) and embedding.dtype == np.float32
    assert np.allclose(scaled, expected, atol=1e-6)


def test_the_screen_never_names_a_scoring_or_ranking_month():
    assert not set(screen.FIT_MONTHS) & {"2025-01", "2025-07", "2025-11", "2025-12", "2026-01", "2026-07"}
    assert screen.OCTOBER == "2025-10"
