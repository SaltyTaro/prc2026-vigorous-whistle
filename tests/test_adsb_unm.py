"""Unmatched-row specialist design (configs/adsb_unm_design_v1.json): masks, weighted fitting, routing and the declaration's bookkeeping."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
lgb = pytest.importorskip("lightgbm")

from run_adsb_learner import fit  # noqa: E402
from run_adsb_unm import design_masks, fit_weighted  # noqa: E402
from unm_design_report import bag, expand, route  # noqa: E402
from launch_adsb_unm import chain_script  # noqa: E402

DECLARATION = ROOT / "configs/adsb_unm_design_v1.json"


def frame():
    return pd.DataFrame({"airport": ["LTFM", "LFPG", "LIRF", "EGLL", "LIRF", "EDDF"], "adsb_matched": [0.0, 1.0, 0.0, 0.0, 1.0, 1.0]})


def test_masks_by_scope():
    keep, w = design_masks(frame(), "all", 1.0, ["LIRF"])
    assert keep.tolist() == [True, True, False, True, False, True] and w is None
    keep, w = design_masks(frame(), "unmatched", 1.0, ["LIRF"])
    assert keep.tolist() == [True, False, False, True, False, False] and w is None
    keep, w = design_masks(frame(), "weighted", 3.0, ["LIRF"])
    assert keep.tolist() == [True, True, False, True, False, True]
    assert w.tolist() == [3.0, 1.0, 3.0, 1.0]
    keep, w = design_masks(frame(), "weighted", 1.0, ["LIRF"])
    assert w is None  # weight 1 takes the members' exact unweighted path
    with pytest.raises(ValueError):
        design_masks(frame(), "everything", 1.0, ["LIRF"])


def synthetic(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({"a": rng.normal(size=n).astype("float32"), "b": rng.normal(size=n).astype("float32")})
    group = rng.random(n) < 0.4
    y = 600 + 100 * x["a"].to_numpy() + np.where(group, 250 * x["b"].to_numpy(), 0) + rng.normal(0, 20, n)
    return x, y, group


PARAMS = {"objective": "regression", "learning_rate": 0.1, "num_leaves": 7, "min_data_in_leaf": 20, "verbose": -1, "num_threads": 1, "seed": 1,
          "deterministic": True, "metric": "rmse"}


def test_unweighted_path_is_the_members_fit():
    x, y, _ = synthetic()
    xs, ys, _ = synthetic(seed=1)
    a, _ = fit("lightgbm", dict(PARAMS), x, y, 200, xs, ys, 20)
    b, _ = fit_weighted("lightgbm", dict(PARAMS), x, y, None, 200, xs, ys, None, 20)
    assert a.best == b.best
    assert np.array_equal(a.predict(xs), b.predict(xs))
    with pytest.raises(ValueError):
        fit_weighted("lightgbm", dict(PARAMS), x, y, np.ones(len(y)), 200, xs, ys, None, 20)


def test_weights_move_the_fit_towards_the_weighted_group():
    x, y, group = synthetic()
    xs, ys, gs = synthetic(seed=1)
    plain, _ = fit_weighted("lightgbm", dict(PARAMS), x, y, None, 200, xs, ys, None, 20)
    heavy, _ = fit_weighted("lightgbm", dict(PARAMS), x, y, np.where(group, 5.0, 1.0), 200, xs, ys, np.where(gs, 5.0, 1.0), 20)
    err = lambda m: float(np.sqrt(np.mean((m.predict(xs)[gs] - ys[gs]) ** 2)))
    assert err(heavy) <= err(plain) + 1e-9


def test_route_changes_only_the_selected_rows():
    base = np.array([1.0, 2.0, 3.0, 4.0])
    x = np.array([10.0, 20.0, 30.0, 40.0])
    rows = np.array([False, True, False, True])
    out = route(base, x, rows, 0.5)
    assert out.tolist() == [1.0, 11.0, 3.0, 22.0]
    assert base.tolist() == [1.0, 2.0, 3.0, 4.0]
    assert np.array_equal(route(base, x, rows, 1.0)[rows], x[rows])


def test_expand_and_bag():
    assert expand("runs/a_s{3..5}/p.parquet") == ["runs/a_s3/p.parquet", "runs/a_s4/p.parquet", "runs/a_s5/p.parquet"]
    assert expand("runs/plain.parquet") == ["runs/plain.parquet"]
    values = {"l": np.array([4.0, 8.0]), "c": np.array([0.0, 4.0])}
    assert bag(values, {"lightgbm": "l", "catboost": "c", "weights": {"lightgbm": 0.25, "catboost": 0.75}}).tolist() == [1.0, 5.0]


@pytest.mark.skipif(not DECLARATION.exists(), reason="declaration not present")
def test_declaration_is_consistent():
    decl = json.loads(DECLARATION.read_text(encoding="utf-8"))
    runs = decl["runs"]
    assigned = [r for host in decl["execution"]["assignment"].values() for r in host]
    assert sorted(assigned) == sorted(runs), "every declared run is assigned to exactly one worker"
    for item, members in decl["items"].items():
        assert all(r in runs for r in members)
        assert len({runs[r]["kind"] for r in members}) == 1 and len({(runs[r]["scope"], runs[r]["unmatched_weight"]) for r in members}) == 1
    for spec in decl["bags"].values():
        assert abs(sum(spec["weights"].values()) - 1.0) < 1e-12
        assert decl["runs"][decl["items"][spec["lightgbm"]][0]]["kind"] == "lightgbm" and decl["runs"][decl["items"][spec["catboost"]][0]]["kind"] == "catboost"
    assert set(decl["protocol"]["decision_items"]) <= set(decl["bags"])
    assert decl["reproduction_check"]["run"] in runs and runs[decl["reproduction_check"]["run"]]["scope"] == "all"
    for name, spec in runs.items():
        record = ROOT / spec["control_member"] / "seasonal" / "run.json"
        if record.exists():
            inputs = json.loads(record.read_text(encoding="utf-8"))["inputs"]
            assert inputs["seed"] == spec["seed"], name
            assert inputs["kind"] == spec["kind"], name
            assert inputs.get("parameter_overrides", {}) == spec.get("parameters", {}), name


@pytest.mark.skipif(not DECLARATION.exists(), reason="declaration not present")
def test_chain_script_runs_each_assigned_run_once():
    decl = json.loads(DECLARATION.read_text(encoding="utf-8"))
    for host, runs in decl["execution"]["assignment"].items():
        script = chain_script(decl, host)
        for run in runs:
            assert script.count(f"--run {run} ") == 1
            assert f"FAILED {run}" in script
        assert script.rstrip().endswith("chain done $(date -u +%FT%TZ)\" >> $log")
