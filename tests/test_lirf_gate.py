"""configs/lirf_gate_v1.json: the LIRF disagreement gate's arithmetic and evaluation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prc2026_final import lirf_gate as lg  # noqa: E402


def test_gate_replaces_only_disagreeing_rows():
    v9 = np.array([1000.0, 1000.0, 1000.0, 1000.0])
    net = np.array([900.0, 900.0, 900.0, -500.0])
    tree = np.array([1000.0, 1600.0, 2200.0, -400.0])
    base, none = lg.gated(v9, net, tree, "unmatched", None)
    assert not none.any()
    assert base == pytest.approx(np.maximum(lg.V34["unmatched"]["net"] * net + lg.V34["unmatched"]["tree"] * tree, lg.FLOOR), abs=1e-9)
    values, fire = lg.gated(v9, net, tree, "unmatched", {"T": 600, "lean": {"v9": 0.0, "net": 0.2, "tree": 0.8}})
    assert fire.tolist() == [False, True, True, False]
    assert values[0] == base[0] and values[3] == base[3]
    assert values[1] == pytest.approx(0.2 * 900 + 0.8 * 1600) and values[2] == pytest.approx(0.2 * 900 + 0.8 * 2200)
    # the floor applies to the leaning blend as well
    values, fire = lg.gated(np.array([-100.0]), np.array([-2000.0]), np.array([-30.0]), "matched", {"T": 600, "lean": {"v9": 0.0, "net": 0.5, "tree": 0.5}})
    assert fire[0] and values[0] == lg.FLOOR
    with pytest.raises(ValueError):
        lg.blend(v9, net, tree, {"v9": 0.5, "net": 0.5, "tree": 0.5})


def test_evaluate_reports_primary_and_unfiltered_changes():
    rng = np.random.default_rng(0)
    n = 400
    v9 = rng.normal(1000, 100, n); net = v9 - 50; tree = v9.copy()
    delayed = rng.choice(n, 40, replace=False)  # the tree sees a delay on 40 rows spread over the days
    tree[delayed] += 1500
    labels = v9.copy(); labels[delayed] += 1400  # and it is right
    labels[delayed[0]] = 90000.0  # one one-day label
    days = np.repeat(np.arange(20), 20)
    out = lg.evaluate(v9, net, tree, labels, days, "unmatched", {"T": 600, "lean": {"v9": 0.0, "net": 0.2, "tree": 0.8}}, 20000.0, 500, 1)
    assert out["gated_rows"] == 40 and out["primary"]["rows_kept"] == n - 1
    assert out["primary"]["relative_change"] < 0 and out["primary"]["interval_95"][1] < 0
    assert out["primary"]["gated_rows_rmse_gate"] < out["primary"]["gated_rows_rmse_control"]
    assert out["unfiltered"]["sse_control"] > out["primary"]["sse_control"]
