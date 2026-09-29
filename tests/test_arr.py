"""Arrival second task: the two-task network nests the plain network, and the arrival rows follow their declaration."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026_nn.preprocess import Preprocessor  # noqa: E402
from prc2026_nn import train as base_train  # noqa: E402
from prc2026_arr import train as arr_train  # noqa: E402

ARCHITECTURE = {"numeric_mode": "ple", "d_num": 4, "cat_cap": 8, "width": 32, "depth": 2, "dropout": 0.1, "k": 1}
TRAINING = {"epochs": 3, "batch_size": 64, "lr": 0.001, "weight_decay": 0.0001, "warmup_epochs": 1.0, "min_lr_fraction": 0.02, "grad_clip": 5.0, "ema_decay": 0.0}


def frame(rows: int, seed: int, arrival: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    table = pd.DataFrame({"a": rng.normal(size=rows), "b": rng.normal(size=rows), "c": rng.normal(size=rows),
                          "stand": rng.choice(["S1", "S2", "S3", "S9" if arrival else "S4"], size=rows), "type": rng.choice(["A320", "B738"], size=rows)})
    target = 300 + 100 * table["a"].to_numpy() + 50 * (table["stand"] == "S1").to_numpy() + rng.normal(scale=20, size=rows)
    if arrival:
        table["b"] = np.nan  # a departure-only column
    return table, target


def fitted(seed: int = 1):
    table, target = frame(640, 0)
    pre = Preprocessor.fit(table, ["a", "b", "c"], ["stand", "type"], target, bins=8, min_count=5)
    x_num, x_cat = pre.transform(table)
    return table, target, pre, x_num, x_cat


def test_without_arrivals_the_run_is_the_plain_network_bit_for_bit():
    _, target, pre, x_num, x_cat = fitted()
    plain = base_train.build_model(pre, ARCHITECTURE, 3)
    base_train.train(plain, pre, x_num, x_cat, pre.scale_target(target), TRAINING, 3, log=lambda e: None)
    two = arr_train.build_model(pre, ARCHITECTURE, 3)
    initial_head = two.arr_weight.detach().clone()
    arr_train.train(two, pre, x_num, x_cat, pre.scale_target(target), TRAINING, 3, arrivals=None, log=lambda e: None)
    state = two.state_dict()
    for name, value in plain.state_dict().items():
        assert torch.equal(value, state[name]), name
    assert torch.equal(two.arr_weight, initial_head)  # the unused head is never updated, not even by weight decay
    assert np.array_equal(base_train.predict(plain, pre, x_num, x_cat), base_train.predict(two, pre, x_num, x_cat))


def test_arrival_rows_train_both_heads_and_the_departure_batches_stay_base():
    _, target, pre, x_num, x_cat = fitted()
    arrivals_table, arrival_target = frame(512, 5, arrival=True)
    pre.vocab = arr_train.pooled_vocabulary(pre, [frame(640, 0)[0], arrivals_table], 5)
    assert "S9" in pre.vocab["stand"] and "S4" in pre.vocab["stand"]  # a category seen in arrivals only enters the vocabulary
    x_num, x_cat = pre.transform(frame(640, 0)[0])
    an, ac = pre.transform(arrivals_table)
    assert np.isnan(an[:, 1]).all()
    scaled = ((arrival_target - arrival_target.mean()) / arrival_target.std()).astype(np.float32)
    model = arr_train.build_model(pre, ARCHITECTURE, 3)
    head = model.arr_weight.detach().clone()
    history = arr_train.train(model, pre, x_num, x_cat, pre.scale_target(target), {**TRAINING, "epochs": 6}, 3, arrivals=(an, ac, scaled), lam=0.5, arrival_batch=64,
                              log=lambda e: None)
    assert not torch.equal(model.arr_weight, head)
    assert history[-1]["arrival_loss_scaled"] < history[0]["arrival_loss_scaled"]
    assert history[-1]["train_loss_scaled"] < history[0]["train_loss_scaled"]
    values = arr_train.predict_arrival(model, an, ac, float(arrival_target.mean()), float(arrival_target.std()))
    assert np.isfinite(values).all() and values.shape == (512,)


def test_heads_split_a_joint_batch():
    _, _, pre, x_num, x_cat = fitted()
    model = arr_train.build_model(pre, ARCHITECTURE, 1).eval()
    xn, xc = torch.from_numpy(x_num[:10]), torch.from_numpy(x_cat[:10])
    departures, arrivals = model.heads(xn, xc, 6)
    assert departures.shape == (6, 1) and arrivals.shape == (4, 1)
    assert torch.allclose(departures, model(xn[:6], xc[:6])) and torch.allclose(arrivals, model.arrival(xn[6:], xc[6:]))


def test_arrival_batches_cover_every_row_before_repeating():
    draw = arr_train.arrival_batches(10, 4, torch.Generator().manual_seed(0))
    first = torch.cat([next(draw), next(draw)])
    assert len(set(first.tolist())) == 8
    assert len(next(draw)) == 4
    with pytest.raises(ValueError):
        next(arr_train.arrival_batches(3, 4, torch.Generator().manual_seed(0)))


def test_save_and_load(tmp_path):
    _, _, pre, x_num, x_cat = fitted()
    model = arr_train.build_model(pre, ARCHITECTURE, 1)
    arr_train.save(model, pre, ARCHITECTURE, {"mean": 500.0, "scale": 300.0, "clip": [0.0, 3600.0]}, tmp_path)
    again, pre2, scaling = arr_train.load(tmp_path)
    assert scaling["scale"] == 300.0
    assert np.array_equal(base_train.predict(model.eval(), pre, x_num, x_cat), base_train.predict(again, pre2, x_num, x_cat))


def test_declaration_and_populated_columns():
    config = json.loads((ROOT / "configs/adsb_arr_design_v1.json").read_text(encoding="utf-8"))
    assert config["audit_used"] is False and config["base_variant"] == "nnb"
    assert set(config["variants"]) == {"mt_l025", "mt_l05", "mt_l1", "mt_l05_notarget"}
    assert config["variants"]["mt_l05_notarget"]["arrival_months"] == "fit"
    import build_arr_rows as builder
    names = builder.contract()
    declared = config["protocol"]["arrival_rows"]["populated"]
    assert declared["categorical"] == [*builder.ROW_CATEGORICAL, *builder.QUALITY_CATEGORICAL]
    assert declared["calendar"] == builder.CALENDAR and declared["record"] == [*builder.RECORD_ROW, *builder.RECORD_QUALITY]
    assert len(builder.POPULATED) == 161 and len(names) == 314
    forbidden = ("schedule", "eobt", "iobt", "arvt", "nmclock", "flight_arrival", "flight_duration", "adsb_", "sfc_", "nb_", "offblock", "interval", "stand_")
    for name in builder.POPULATED_NUMERIC:  # no clock of the arrival itself, no departure-only column
        if name.startswith(("dep_schedule_delay", "arr_schedule_delay")):
            continue  # airport histories of other movements
        assert not any(token in name for token in forbidden), name


def test_role_swap():
    import build_arr_rows as builder
    raw = pd.DataFrame({"PHASE_mvt": ["ARR", "DEP"], "ADEP_mvt": ["KJFK", "EGLL"], "ADES_mvt": ["EGLL", "KJFK"], "ADEP_flt": ["KJFK", "EGLL"], "ADES_flt": ["EGLL", None],
                        "MVT_ID_mvt": [1, 2]})
    rows = builder.role_rows(raw, swap=True)
    assert rows["MVT_ID_mvt"].tolist() == [1] and rows["PHASE_mvt"].tolist() == ["DEP"]
    assert rows["ADEP_mvt"].tolist() == ["EGLL"] and rows["ADES_mvt"].tolist() == ["KJFK"] and rows["ADEP_flt"].tolist() == ["EGLL"] and rows["ADES_flt"].tolist() == ["KJFK"]
    assert builder.role_rows(raw, swap=False)["MVT_ID_mvt"].tolist() == [2]
    assert builder.context_months("2025-01")[1] == ["2025-01", "2025-02"] and builder.context_months("2025-11")[1] == ["2025-10", "2025-11"]
    assert builder.context_months("2025-12")[1] == ["2025-11", "2025-12"] and builder.context_months("2026-07") == (["ranking.parquet"], ["2026-07"])


def test_no_scoring_month_is_named_by_the_design_fold():
    import run_adsb_arr as runner
    months = [*runner.FIT_MONTHS["design"], *runner.TARGET_MONTHS["design"]]
    assert not set(months) & {"2025-01", "2025-07", "2025-11", "2025-12", "2026-01", "2026-07"}
