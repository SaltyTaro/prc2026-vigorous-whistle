"""Final-file runner (src/prc2026_final, tools/final_predict.py): input contract, receipts, routing helpers and v30's arithmetic."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from prc2026_final.common import RAW_COLUMNS, Run, inspect, load_movements  # noqa: E402


def movement_frame() -> pd.DataFrame:
    t0 = pd.Timestamp("2026-03-02 10:00", tz="UTC")
    rows = []
    for i, (phase, airport) in enumerate([("DEP", "EDDF"), ("ARR", "EDDF"), ("DEP", "LIRF"), ("DEP", "EGLL")]):
        row = {c: None for c in RAW_COLUMNS}
        row.update({"MVT_ID_mvt": float(100 + i), "PHASE_mvt": phase, "ADEP_mvt": airport if phase == "DEP" else "LFPG", "ADES_mvt": airport if phase == "ARR" else "LFPG",
                    "MVT_TIME_UTC_mvt": t0 + pd.Timedelta(minutes=10 * i), "SCHED_TIME_UTC_mvt": t0, "BLOCK_TIME_UTC_mvt": t0 - pd.Timedelta(minutes=5),
                    "TAXITIME_SEC_mvt": 600.0})
        rows.append(row)
    frame = pd.DataFrame(rows)
    for c in ["MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt"]:
        frame[c] = pd.to_datetime(frame[c], utc=True)
    return frame


def write_inputs(tmp_path: Path, frame: pd.DataFrame, template_ids=None) -> tuple[Path, Path]:
    raw = tmp_path / "file.parquet"
    frame.to_parquet(raw, index=False)
    ids = frame.loc[frame["PHASE_mvt"].eq("DEP"), "MVT_ID_mvt"].to_numpy() if template_ids is None else np.asarray(template_ids, dtype=float)
    template = tmp_path / "template.parquet"
    pd.DataFrame({"MVT_ID_mvt": ids, "TAXITIME_SEC_mvt": pd.array([None] * len(ids), dtype="Int32")}).to_parquet(template, index=False)
    return raw, template


def test_inspect_reports_months_arrivals_and_template(tmp_path):
    raw, template = write_inputs(tmp_path, movement_frame())
    record = inspect(raw, template)
    assert record["months"] == ["2026-03"] and record["departures"] == 3 and record["arrivals"] == 1
    assert record["arrivals_with_in_block"] == 1 and record["departures_in_template_order"]
    assert record["departures_with_label_or_block_in_file"] == 3  # present in the file; masked before any stage


def test_inspect_refuses_missing_columns_and_foreign_templates(tmp_path):
    frame = movement_frame()
    raw, template = write_inputs(tmp_path, frame.drop(columns=["AOBT_3_flt"]))
    with pytest.raises(ValueError, match="lacks"):
        inspect(raw, template)
    raw, template = write_inputs(tmp_path, frame, template_ids=[100.0, 102.0])
    with pytest.raises(ValueError, match="template"):
        inspect(raw, template)


def test_departure_labels_and_blocks_are_masked(tmp_path):
    raw, _ = write_inputs(tmp_path, movement_frame())
    frame = load_movements(raw)
    dep = frame["PHASE_mvt"].eq("DEP")
    assert frame.loc[dep, "TAXITIME_SEC_mvt"].isna().all() and frame.loc[dep, "BLOCK_TIME_UTC_mvt"].isna().all()
    assert frame.loc[~dep, "TAXITIME_SEC_mvt"].notna().all() and frame.loc[~dep, "BLOCK_TIME_UTC_mvt"].notna().all()


def test_stage_receipts_detect_changed_outputs(tmp_path):
    import time
    run = Run(tmp_path / "run")
    out = run.path("stage", "a.parquet")
    pd.DataFrame({"x": [1, 2]}).to_parquet(out)
    run.finish("demo", {"k": 1}, [out], time.time())
    assert run.done("demo")
    run.verify_outputs("demo")
    pd.DataFrame({"x": [3]}).to_parquet(out)
    with pytest.raises(ValueError, match="changed"):
        run.verify_outputs("demo")


def test_weather_coverage_flags_uncovered_station_months():
    from prc2026_final.features import weather_coverage
    inventory = {"objects": [{"station": "EDDF", "start_inclusive": "2026-01-01", "end_exclusive": "2026-02-01"}]}
    departures = pd.DataFrame({"ADEP_mvt": ["EDDF", "EDDF", "LIRF"],
                               "MVT_TIME_UTC_mvt": pd.to_datetime(["2026-01-05", "2026-03-01", "2026-01-05"], utc=True)})
    assert weather_coverage([inventory], departures) == {"EDDF": ["2026-03"], "LIRF": ["2026-01"]}


def test_features_refuse_overlap_with_the_frozen_training_context():
    from prc2026_final.features import build
    with pytest.raises(ValueError, match="overlaps"):
        build(ROOT, movement_frame(), ["2025-07"])


def test_table_mean_keeps_multiplicity_and_ignores_nan_members():
    from prc2026_final.leaves import table_mean
    predictions = {"a": np.array([1.0, np.nan]), "b": np.array([4.0, 2.0])}
    out = table_mean(["a", "a", "b"], predictions)
    assert out[0] == pytest.approx(2.0) and out[1] == pytest.approx(2.0)


def test_network_kind_routes_by_record():
    from prc2026_final.networks import kind_of
    assert kind_of({"variant": "ctx_e32", "inputs": {"device": "cuda", "tokens": {}}}) == "ctx3"
    assert kind_of({"variant": "ctx_e24", "inputs": {"tokens": {}}}) == "ctx"
    assert kind_of({"variant": "nnb", "inputs": {}}) == "nn"


def test_vocabulary_cache_is_seeded_once_and_never_overwrites(tmp_path):
    import json
    from prc2026_final.leaves import seed_vocabularies
    source = tmp_path / "old" / "vocabularies.json"
    source.parent.mkdir()
    source.write_text(json.dumps({"k": {"airport": ["EDDF"]}}), encoding="utf-8")
    target = tmp_path / "new" / "leaves" / "vocabularies.json"
    first = seed_vocabularies(target, source)
    assert first["seeded"] and first["keys"] == 1 and json.loads(target.read_text(encoding="utf-8")) == {"k": {"airport": ["EDDF"]}}
    target.write_text(json.dumps({"own": {}}), encoding="utf-8")
    second = seed_vocabularies(target, source)
    assert not second["seeded"] and json.loads(target.read_text(encoding="utf-8")) == {"own": {}}
    source.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="not a vocabulary cache"):
        seed_vocabularies(tmp_path / "third.json", source)


def test_adj2_needs_an_explicit_refit_for_other_months(tmp_path):
    from prc2026_final.residual import adj2_model
    with pytest.raises(ValueError, match="refit-adj2"):
        adj2_model("2026-02", tmp_path)


def test_recipe_arithmetic_is_v30s():
    """Hand-computed v30 on the five routes: non-LIRF matched / unmatched, LIRF non-population matched / unmatched, fill population."""
    from prc2026_final.recipe import compute
    table = pd.DataFrame({
        "MVT_ID_mvt": [1.0, 2.0, 3.0, 4.0, 5.0], "airport": ["EDDF", "EGLL", "LIRF", "LIRF", "LIRF"], "month": "2026-01",
        "adsb_matched": [1.0, 0.0, 1.0, 0.0, 0.0], "population": [False, False, False, False, True],
        "proxy_s": [600.0, 600.0, 600.0, 600.0, np.nan], "schedule_interval_s": [900.0, 900.0, 900.0, 900.0, 7000.0],
        "v9": [700.0, 800.0, 900.0, 1000.0, 5000.0], "residual": [10.0, 0.0, 0.0, 0.0, 0.0],
        "ens_v20_bag_q_all": [600.0, 650.0, 700.0, 750.0, 800.0], "ens_v22_lirf_q": [1.0, 2.0, 950.0, 1100.0, 3.0],
        "ens_ctx_e24": [620.0, 660.0, 1.0, 1.0, 1.0], "ens_ctx_e24b": [640.0, 680.0, 1.0, 1.0, 1.0], "ens_ctx_e32": [660.0, 700.0, 1.0, 1.0, 1.0],
        "ens_ctx_lirf": [0.0, 0.0, 980.0, 1200.0, 7.0]})
    detail, record = compute(table, -12.0)
    net = 0.25 * table["ens_ctx_e24"] + 0.25 * table["ens_ctx_e24b"] + 0.5 * table["ens_ctx_e32"]
    mix = 0.25 * table["ens_v20_bag_q_all"] + 0.75 * net
    expected = [0.3 * (700 + 10) + 0.7 * mix[0], 0.3 * 800 + 0.7 * mix[1],
                0.15 * 900 + 0.45 * 950 + 0.40 * 980, 0.40 * 1000 + 0.25 * 1100 + 0.35 * 1200, 5000.0]
    assert np.allclose(detail["v30"].to_numpy(), expected, rtol=0, atol=1e-9)
    assert list(detail["scope"]) == ["matched_no_lirf", "all_no_lirf", "lirf_nonpop_matched", "lirf_nonpop_unmatched", "fixed"]
    assert record["candidate"] == "c_e24x32_e32h" and record["network_share"] == 0.75


@pytest.mark.skipif(not (ROOT / "data/raw/ranking.parquet").exists(), reason="needs the competition data")
def test_v9_stage_reproduces_the_recorded_ranking_v9_on_a_sample():
    """The v9 stage on frozen features for 300 ranking departures (every fill-population row among them included) equals v30's v9."""
    from prc2026.artifacts import read_json
    from prc2026_final.v9 import build
    base = ROOT / read_json(ROOT / "data/timeline/latest.json")["bundle"]
    features = pd.read_parquet(base / "features/ranking.parquet")
    movements = load_movements(ROOT / "data/raw/ranking.parquet")
    detail = pd.read_parquet(ROOT / "runs/adsb_ctx/export_v30/ranking_values.parquet").set_index("MVT_ID_mvt")
    fixed = detail.index[detail["scope"].eq("fixed")].to_numpy()[:40]
    rng = np.random.default_rng(1)
    pick = np.unique(np.concatenate([rng.choice(features["MVT_ID_mvt"].to_numpy(), 260, replace=False), fixed]))
    sample = features.set_index("MVT_ID_mvt").loc[pick].reset_index()
    # every arrival stays (the mixture's turnaround features read arrival in-blocks); only the sampled departures are scored
    movements = movements.loc[movements["PHASE_mvt"].eq("ARR") | movements["MVT_ID_mvt"].isin(pick)].reset_index(drop=True)
    frame, record = build(sample, movements, threads=4)
    v7 = pd.read_parquet(ROOT / "runs/azure_arcw0_surface_refit/20260914T185207842456Z_refit_full_timeline_v6_100/ranking_predictions.parquet").set_index("MVT_ID_mvt")
    assert np.max(np.abs(frame["v7"].to_numpy() - v7.loc[pick, "prediction"].to_numpy())) <= 1e-9
    assert np.max(np.abs(frame["v9"].to_numpy() - detail.loc[pick, "v9"].to_numpy())) <= 1e-9


def test_adj2_refit_for_a_new_month_is_saved_and_loadable(tmp_path, monkeypatch):
    """refit-adj2 wiring for a month without a saved booster (the fit itself is the frozen run_adsb_v25.fit_variant, stubbed here)."""
    import lightgbm as lgb
    import run_adsb_v24
    from prc2026_final import residual
    x = pd.DataFrame({"v9": np.linspace(0, 1, 200), "hour": np.linspace(0, 23, 200)})
    booster = lgb.train({"objective": "regression", "verbose": -1, "num_threads": 1}, lgb.Dataset(x, np.sin(x["v9"])), 5)
    calls = {}

    def fake_fit(train, config, variant, target_month, threads):
        calls["target"] = target_month
        return {"model": booster, "rounds": 5, "inner_best_iterations": [5], "rows": len(train), "weights": variant.get("weights"), "weight_sum": 1.0,
                "categories": {"airport": ["EDDF"]}, "feature_set": "nb", "scope": "matched_no_lirf", "seconds": 0.0}

    monkeypatch.setattr(residual, "fit_variant", fake_fit)
    monkeypatch.setattr(run_adsb_v24, "ALL_MONTHS", ["2025-01"])
    monkeypatch.setattr(run_adsb_v24, "load_frame", lambda config, month, source: pd.DataFrame({"x": [1]}))
    monkeypatch.setattr(run_adsb_v24, "adsb_features", lambda month, root, cache: pd.DataFrame({"x": [1]}))
    monkeypatch.setattr(run_adsb_v24, "month_table", lambda frame, feats, month: pd.DataFrame({"month": [month]}))
    record = residual.refit_adj2("2026-10", tmp_path, threads=1)
    assert calls["target"] == "2026-10" and record["target_month"] == "2026-10"
    model = residual.adj2_model("2026-10", tmp_path)
    assert model["rounds"] == 5 and model["scope"] == "matched_no_lirf"
