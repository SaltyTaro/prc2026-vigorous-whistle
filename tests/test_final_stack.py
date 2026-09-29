"""Final-file runner, v31 -> v35 .. v38 stages (src/prc2026_final/stack.py): EOBT_1 detector, the reset edit and the stack arithmetic."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from prc2026_final import stack  # noqa: E402


def departures(airport: str, month: str, n: int, later: int) -> pd.DataFrame:
    t = pd.Timestamp(f"{month}-10 08:00", tz="UTC")
    iobt = pd.Series([t] * n)
    eobt = iobt + pd.to_timedelta([600 if i < later else 0 for i in range(n)], unit="s")
    return pd.DataFrame({"MVT_ID_mvt": np.arange(n, dtype=float), "PHASE_mvt": "DEP", "ADEP_mvt": airport, "MVT_TIME_UTC_mvt": iobt + pd.Timedelta(minutes=20),
                         "EOBT_1_flt": eobt, "IOBT_flt": iobt})


def test_detector_flags_only_shares_more_than_one_span_above_the_2025_maximum():
    reference = pd.DataFrame({"airport": ["LFPG", "LEMD"], "reference_min": [0.04, 0.17], "reference_max": [0.07, 0.21], "reference_months": [12, 12]})
    reference["span"] = reference["reference_max"] - reference["reference_min"]
    reference["threshold"] = reference["reference_max"] + reference["span"]
    frame = pd.concat([departures("LFPG", "2026-07", 100, 35), departures("LFPG", "2026-01", 100, 6),
                       departures("LEMD", "2026-01", 100, 22),  # above the 2025 maximum but within one span: not flagged
                       departures("LEMD", "2026-07", 100, 1),   # below the range: never flagged
                       departures("XXXX", "2026-07", 100, 90)], ignore_index=True)  # no 2025 reference: never flagged
    flagged, table = stack.detect(frame, reference)
    assert flagged == [("LFPG", "2026-07")]
    assert table.set_index(["airport", "month"]).loc[("LFPG", "2026-07"), "share"] == pytest.approx(0.35)


def test_reset_edits_only_later_eobt1_departures_of_flagged_airport_months():
    frame = pd.concat([departures("LFPG", "2026-07", 4, 2), departures("LFPG", "2026-01", 4, 2), departures("EDDF", "2026-07", 4, 2)], ignore_index=True)
    arrival = frame.iloc[[0]].assign(PHASE_mvt="ARR", MVT_ID_mvt=99.0)
    frame = pd.concat([frame, arrival], ignore_index=True)
    edited, mask = stack.reset_input(frame, [("LFPG", "2026-07")])
    assert mask.tolist() == [True, True] + [False] * 11
    assert (edited.loc[mask, "EOBT_1_flt"] == edited.loc[mask, "IOBT_flt"]).all()
    assert edited.loc[~mask].equals(frame.loc[~mask]) and edited.dtypes.equals(frame.dtypes)
    assert stack.same_content(frame, frame.copy()) and not stack.same_content(frame, edited)


def stack_inputs(tmp_path, edit=None):
    """The twelve-route fixture of test_stack_arithmetic_on_the_routes; `edit(detail)` may change rows before v30 is derived by the caller."""
    ids = np.arange(1.0, 13.0)
    committee = np.array([1100.0, 700.0, 900.0, 1.0, 1.0, 1.0, 820.0, 1000.0, 400.0, 900.0, 2000.0, 3000.0])
    nets = np.array([1100.0, 700.0, 900.0, 1.0, 1.0, 1.0, 820.0, 1000.0, 1200.0, 900.0, 2000.0, 3000.0])
    detail = pd.DataFrame({"MVT_ID_mvt": ids, "airport": ["LTFM", "EDDF", "EDDF", "LIRF", "LIRF", "LIRF", "EHAM", "EHAM", "EGLL", "EGLL", "LSZH", "LSZH"], "month": "2026-01",
                           "scope": ["all_no_lirf", "matched_no_lirf", "all_no_lirf", "lirf_nonpop_unmatched", "lirf_nonpop_matched", "fixed", "all_no_lirf",
                                     "matched_no_lirf", "all_no_lirf", "all_no_lirf", "all_no_lirf", "matched_no_lirf"],
                           "v9": [1000.0, 600.0, 500.0, 1000.0, 900.0, 5000.0, 800.0, 900.0, 1000.0, 900.0, 2000.0, 3000.0], "residual": [0.0, 20.0] + [0.0] * 10,
                           "committee": committee, "ens_ctx_e24": nets, "ens_ctx_e24b": nets, "ens_ctx_e32": nets, "committee_mix": 0.25 * committee + 0.75 * nets,
                           "ens_ctx_lirf": [0.0, 0.0, 0.0, 1200.0, 980.0, 7.0] + [0.0] * 6, "ens_v22_lirf_q": [0.0, 0.0, 0.0, 1900.0, 950.0, 3.0] + [0.0] * 6})
    detail["v30"] = [0.3 * 1000 + 0.7 * 1100, 0.3 * 620 + 0.7 * 700, 0.3 * 500 + 0.7 * 900, 0.40 * 1000 + 0.35 * 1200 + 0.25 * 1900,
                     0.15 * 900 + 0.40 * 980 + 0.45 * 950, 5000.0, 0.3 * 800 + 0.7 * 820, 0.3 * 900 + 0.7 * 1000, 1000.0, 900.0, 2000.0, 3000.0]
    repaired = detail["v30"].to_numpy() + np.array([0, 100.0] + [0.0] * 10)
    takeoff = 1.8e9
    rematch = pd.DataFrame({"MVT_ID_mvt": [7.0], "airport": ["EHAM"], "flight": ["KL1"], "mapped": ["KLM1"], "takeoff": [takeoff], "key_icao": [None],
                            "iata_mapped_icao": ["abc123"], "iata_mapped_gate": [True], "iata_mapped_motion_start": [takeoff - 2000.0],
                            "iata_mapped_taxi_stops": [1.0], "iata_mapped_taxi_stopped_s": [100.0], "iata_mapped_taxi_path_m": [900.0]})
    path = tmp_path / "rematch.parquet"
    rematch.to_parquet(path, index=False)
    regular = pd.DataFrame({"MVT_ID_mvt": [8.0, 2.0], "airport": ["EHAM", "EDDF"], "gate": [True, True], "takeoff": [takeoff, takeoff],
                            "motion_start": [takeoff - 4000.0, takeoff - 700.0], "taxi_stops": [6.0, 0.0], "taxi_stopped_s": [1000.0, 0.0], "taxi_path_m": [6000.0, 2000.0]})
    # congested hours (peer_c > 600, >= 5 peers) on rows 2 (tracked), 3 (untracked, clock), 4 (LIRF) and 10 (no clock); row 1's hour is below
    peer = pd.DataFrame({"MVT_ID_mvt": ids, "no_clock": [False, False, False, True, False, False, False, False, False, True, False, False],
                         "peer_c": [500.0, 900.0, 650.0, 900.0, 0.0, 0.0, 0.0, 0.0, 0.0, 700.0, 0.0, 0.0], "peer_n": [9, 9, 5, 10, 0, 0, 0, 0, 0, 6, 0, 0]})
    fill = pd.DataFrame({"MVT_ID_mvt": [6.0], "pin_s": [50000.0], "scope": ["population"], "interval_s": [50000.0], "proxy_s": [np.nan]})
    return ids, detail, repaired, path, regular, peer, fill


def test_stack_arithmetic_on_the_routes(tmp_path, monkeypatch):
    """Hand-computed v31 -> v35, v36, v37 and v38 on: LTFM, matched EDDF (+ repair effect), unmatched EDDF above the u300 gate, LIRF unmatched /
    matched / fill population, an EHAM callsign-fix row, an EHAM regular queue-shaped match, an EGLL row whose nets exceed the committee
    by 800 s (mu600), an EGLL no-clock row in a congested hour (peer lift) and an untracked / a tracked LSZH row above 1,500 s (stretch)."""
    import export_stack
    for module in (stack, export_stack):
        monkeypatch.setattr(module, "offsets", lambda: {"EDDF": 0.0, "EHAM": 10.0})
        monkeypatch.setattr(module, "offsets_all", lambda: {"EHAM": 5.0})
    ids, detail, repaired, path, regular, peer, fill = stack_inputs(tmp_path)
    frame, record = stack.compute(detail, repaired, path, regular, peer, fill)
    v35 = [1100.0,                                   # LTFM: outer weight 1.0
           0.4 * 620 + 0.6 * 700 + 0.5 * 100,        # matched EDDF: outer 0.6, plus half the repair effect
           900.0,                                    # unmatched EDDF, mix - base = 400 > 300: the mix
           (0.35 * 1200 + 0.25 * 1900) / 0.60,       # LIRF unmatched: v9 share 0, net:tree 0.35:0.25 (tree 700 s above the net)
           0.15 * 900 + 0.40 * 980 + 0.45 * 950,     # LIRF matched: unchanged
           5000.0,                                   # fill population: v30
           0.5 * (2000 + 10) + 0.5 * (0.3 * 800 + 0.7 * 820),   # callsign fix 50%
           0.75 * (4000 + 5) + 0.25 * (0.4 * 900 + 0.6 * 1000),  # queue trust 0.75 on a regular match
           1000.0, 900.0, 2000.0, 3000.0]
    v36 = list(v35)
    v36[2] += 50.0                                   # untracked, own clock, congested hour
    v36[8] = 0.3 * 1000 + 0.7 * 1200                 # nets - committee = 800 > 600: network share 1.0
    v36[9] += 100.0                                  # no clock, congested hour
    from prc2026.artifacts import read_json
    k = read_json(ROOT / "docs/disagreement/tail_stretch_v1_results.json")["candidates"]["v35_t1500"]["k_all_months"]
    v37 = list(v35)
    v37[10] += k["untracked"] * (2000 - 1500)          # untracked above 1,500 s (the queue row, 3,244 s, is anchored; LIRF 5,000 s excluded)
    v37[11] += k["tracked"] * (3000 - 1500)
    assert np.allclose(frame["v35"].to_numpy(), v35, rtol=0, atol=1e-9) and np.allclose(frame["v36"].to_numpy(), v36, rtol=0, atol=1e-9)
    assert np.allclose(frame["v37"].to_numpy(), v37, rtol=0, atol=1e-9)
    assert frame["v37_stretch_untracked"].tolist() == [False] * 10 + [True, False] and frame["v37_stretch_tracked"].tolist() == [False] * 11 + [True]
    two = read_json(ROOT / "docs/disagreement/tail_stretch_v2_results.json")["candidates"]["two_sided_900"]["k_all_months"]
    v38 = list(v37)                                  # the upper k is v37's; the lower stretch acts below 900 s on every unanchored non-LIRF row
    v38[1] += two["tracked"][1] * (900 - v35[1])     # matched EDDF at 718 s
    assert two["untracked"][0] == k["untracked"] and np.allclose(frame["v38"].to_numpy(), v38, rtol=0, atol=1e-9)
    assert frame["v38_stretch_untracked"].tolist() == [True, False, True] + [False] * 5 + [True, True, True, False]
    assert frame["v38_stretch_tracked"].tolist() == [False, True] + [False] * 9 + [True]  # the tracked queue row is anchored
    assert frame["v35_u300"].tolist() == [False, False, True] + [False] * 9 and frame["v36_u300"].tolist() == frame["v35_u300"].tolist()
    assert frame["v35_callsign"].tolist() == [False] * 6 + [True] + [False] * 5 and frame["v35_queue"].tolist() == [False] * 7 + [True] + [False] * 4
    assert frame["v36_mix_gate"].tolist() == [False] * 8 + [True, False, False, False] and not frame["v35_mix_gate"].any()
    assert frame["v36_peer_noclock"].tolist() == [False] * 9 + [True, False, False] and frame["v36_peer_clock_untracked"].tolist() == [False, False, True] + [False] * 9
    assert record["rows"]["repair_effect_nonzero"] == 1 and record["layer_sets"]["v36"]["mix_up"] == 600.0
    v39 = list(v38)                                  # the fill pin: the LIRF population row (5,000 s at v30..v38) takes its schedule interval
    v39[5] = 50000.0
    assert np.allclose(frame["v39"].to_numpy(), v39, rtol=0, atol=1e-9) and frame["v39_fill_pin"].tolist() == [False] * 5 + [True] + [False] * 6
    assert record["rows"]["v39"] == {"fill_pin": 1, "fill_pin_population": 1, "fill_pin_clock_day": 0, "floored": 0}
    assert record["layer_sets"]["v39"]["base"] == "v38" and record["shipped_as"]["v39"]["fill_pin"] == "all_twelve"
    # v40 (configs/fill_pin_v2.json evidenced_nine): the same pin lies in the [43,200, 58,000] band, so v40 == v39 here
    assert np.allclose(frame["v40"].to_numpy(), v39, rtol=0, atol=1e-9) and frame["v40_fill_pin"].tolist() == frame["v39_fill_pin"].tolist()
    assert record["rows"]["v40"] == {"fill_pin": 1, "floored": 0} and record["layer_sets"]["v40"]["candidate"] == "evidenced_nine"
    # v41 (configs/fill_pin_v3.json): the pin's interval (50,000 s) is below the 100,000 s band, so nothing moves to the one-day value
    assert np.allclose(frame["v41"].to_numpy(), v39, rtol=0, atol=1e-9) and not frame["v41_one_day"].any()
    assert record["rows"]["v41"] == {"fill_pin": 1, "one_day": 0, "floored": 0} and record["layer_sets"]["v41"]["base"] == "v40"
    # v45 (configs/lirf_gate_v1.json u600_m600): the LIRF unmatched row's tree sits 700 s above its network -> the leaning blend 0.2 net + 0.8 tree,
    # added as a delta against v34's blend; the matched row (30 s apart) and the fixed row do not fire
    v45 = list(v39)
    v45[3] = v39[3] + (0.2 * 1200 + 0.8 * 1900 - (0.35 * 1200 + 0.25 * 1900) / 0.60)
    assert np.allclose(frame["v45"].to_numpy(), v45, rtol=0, atol=1e-9) and frame["v45_lirf_gate"].tolist() == [False] * 3 + [True] + [False] * 8
    assert record["rows"]["v45"] == {"fill_pin": 1, "lirf_gate": 1, "floored": 0} and record["layer_sets"]["v45"]["base"] == "v40"
    assert record["layer_sets"]["v45"]["spec"]["unmatched"]["T"] == 600 and record["shipped_as"]["v45"]["lirf_gate"] == "u600_m600"
    # v46 (configs/thinned_regime_round_v1.json n85_nd1200): only the EGLL row whose networks differ from the committee moves with the share
    # (mix 0.15 x 400 + 0.85 x 1200 = 1,080 against 1,000; outer weight 0.7 on base 1,000); no row has its committee 1,200 s above the networks
    v46 = list(v45)
    v46[8] = 0.3 * 1000 + 0.7 * (0.15 * 400 + 0.85 * 1200)
    assert np.allclose(frame["v46"].to_numpy(), v46, rtol=0, atol=1e-9) and frame["v46_share"].tolist() == [False] * 8 + [True] + [False] * 3
    assert not frame["v46_share_gate"].any() and record["rows"]["v46"] == {"fill_pin": 1, "lirf_gate": 1, "share": 1, "share_gate": 0, "floored": 0}
    assert record["layer_sets"]["v46"]["base"] == "v45" and record["layer_sets"]["v46"]["network_share"] == 0.85
    assert record["layer_sets"]["v46"]["gate"] == {"threshold_s": 1200, "share_inside": 0.5} and record["shipped_as"]["v46"]["share_gate"] == "n85_nd1200"
    assert "v35_peer_noclock" not in frame.columns and "v36_stretch_tracked" not in frame.columns
    assert export_stack.scope_table is not None and export_stack.ranking_peers.__module__ == "peer_congestion"  # readers restored


def test_stack_refuses_a_detail_that_is_not_v30s_recipe(tmp_path):
    detail = pd.DataFrame({"MVT_ID_mvt": [1.0], "airport": ["EDDF"], "month": "2026-01", "scope": ["all_no_lirf"], "v9": [500.0], "residual": [0.0],
                           "committee_mix": [600.0], "ens_ctx_lirf": [0.0], "ens_v22_lirf_q": [0.0], "v30": [999.0]})
    with pytest.raises(ValueError, match="not v30's recipe"):
        stack.compute(detail, None, tmp_path / "unused.parquet", pd.DataFrame(), pd.DataFrame(), pd.DataFrame())


def test_rematch_scope_is_the_declared_no_nm_eddf_eham_departures():
    frame = pd.DataFrame({c: [None] * 4 for c in stack.REMATCH_COLUMNS})
    frame["ADEP_mvt"] = ["EDDF", "EHAM", "EHAM", "LFPG"]
    frame["FLIGHT_ID_mvt"] = [np.nan, np.nan, 5.0, np.nan]
    frame["MVT_ID_mvt"] = [1.0, 2.0, 3.0, 4.0]
    assert stack.rematch_departures(frame)["MVT_ID_mvt"].tolist() == [1.0, 2.0]


def test_v46_gate_row_takes_the_half_share_through_the_whole_stack(tmp_path, monkeypatch):
    """An unmatched EGLL row whose committee (3,000 s) sits 1,500 s above the networks: the control mix is 1,875 s, the gated mix 2,250 s; both
    pass the u300 gate (the mix itself) and the upper stretch, so v46 = 2,250 + k x 750 against v45 = 1,875 + k x 375."""
    import export_stack
    from prc2026.artifacts import read_json
    for module in (stack, export_stack):
        monkeypatch.setattr(module, "offsets", lambda: {"EDDF": 0.0, "EHAM": 10.0})
        monkeypatch.setattr(module, "offsets_all", lambda: {"EHAM": 5.0})
    ids, detail, repaired, path, regular, peer, fill = stack_inputs(tmp_path)
    row = 9
    effect = repaired - detail["v30"].to_numpy()
    detail.loc[row, "committee"] = 3000.0
    detail.loc[row, ["ens_ctx_e24", "ens_ctx_e24b", "ens_ctx_e32"]] = 1500.0
    detail.loc[row, "committee_mix"] = 0.25 * 3000 + 0.75 * 1500
    detail.loc[row, "v30"] = 0.3 * 900 + 0.7 * 1875
    frame, record = stack.compute(detail, detail["v30"].to_numpy() + effect, path, regular, peer, fill)
    k = read_json(ROOT / "docs/disagreement/tail_stretch_v2_results.json")["candidates"]["two_sided_900"]["k_all_months"]["untracked"][0]
    assert frame.loc[row, "v38"] == pytest.approx(1875 + k * 375) and frame.loc[row, "v45"] == pytest.approx(1875 + k * 375)
    assert frame.loc[row, "v46"] == pytest.approx(2250 + k * 750)
    assert frame["v46_share_gate"].tolist() == [False] * 9 + [True, False, False] and frame["v46_share"].tolist() == [False] * 8 + [True, True, False, False]
    assert record["rows"]["v46"]["share_gate"] == 1 and record["rows"]["v46"]["share"] == 2
    lirf = detail["airport"].eq("LIRF").to_numpy()
    assert np.array_equal(frame.loc[lirf, "v46"].to_numpy(), frame.loc[lirf, "v45"].to_numpy())
