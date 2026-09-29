import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from prc2026_final import network_member, share_gate  # noqa: E402


def test_seed_mean_needs_every_seed():
    a, b = np.array([100.0, 200.0, np.nan]), np.array([300.0, np.nan, 50.0])
    out = network_member.seed_mean([a, b])
    assert out[0] == 200.0 and np.isnan(out[1]) and np.isnan(out[2])
    with pytest.raises(ValueError):
        network_member.seed_mean([])


def test_member_enters_only_where_both_are_finite():
    table, member = np.array([1000.0, 1000.0, np.nan, 800.0]), np.array([600.0, np.nan, 500.0, 900.0])
    out, inside = network_member.table_with_member(table, member, 0.25)
    assert inside.tolist() == [True, False, False, True]
    assert out[0] == pytest.approx(0.75 * 1000 + 0.25 * 600) and out[1] == 1000.0 and np.isnan(out[2]) and out[3] == pytest.approx(0.75 * 800 + 0.25 * 900)
    same, _ = network_member.table_with_member(table, member, 0.0)
    assert np.array_equal(same, table, equal_nan=True)
    with pytest.raises(ValueError):
        network_member.table_with_member(table, member, 1.5)


def test_candidate_mix_is_v46s_mix_on_the_new_table():
    committee = np.array([3000.0, 3000.0, 1000.0])
    table, member = np.array([1900.0, 1900.0, 900.0]), np.array([1500.0, np.nan, 1100.0])
    new, _ = network_member.table_with_member(table, member, 0.5)
    mix, fire = share_gate.mix(committee, new, network_member.V46_SHARE, network_member.V46_GATE)
    assert fire.tolist() == [True, False, False]  # the gate reads the candidate's own table: 3000 - 1700 > 1200, 3000 - 1900 is not
    assert mix[0] == pytest.approx(0.5 * 3000 + 0.5 * 1700) and mix[1] == pytest.approx(0.15 * 3000 + 0.85 * 1900) and mix[2] == pytest.approx(0.15 * 1000 + 0.85 * 1000)


def test_candidate_weight_refuses_unwired_keys():
    decl = {"candidates": {"w25": {"member_weight": 0.25}, "odd": {"member_weight": 0.25, "network_share": 1.0}}}
    assert network_member.candidate_weight(decl, "w25") == 0.25
    with pytest.raises(ValueError):
        network_member.candidate_weight(decl, "odd")


def test_selection_rule():
    from run_network_member import select

    def month(ratio, upper):
        return {"ratio_to_control": ratio, "paired_days_vs_control": {"rmse_difference_95pct_interval_seconds": [-1.0, upper]}}

    equal = ["2025-01", "2025-07", "2025-12"]
    entries = {"good": {"months": {m: month(0.999, -0.01) for m in equal}}, "better": {"months": {m: month(0.998, -0.01) for m in equal}},
               "one_worse": {"months": {"2025-01": month(1.0001, -0.01), "2025-07": month(0.99, -0.01), "2025-12": month(0.99, -0.01)}},
               "interval": {"months": {"2025-01": month(0.999, 0.02), "2025-07": month(0.999, -0.01), "2025-12": month(0.999, -0.01)}},
               "small": {"months": {m: month(0.9999, -0.01) for m in equal}}}
    eligible, selected = select(entries, equal, 0.0002)
    assert eligible == ["good", "better"] and selected == "better"
    assert entries["one_worse"]["months_worse"] == ["2025-01"] and entries["interval"]["intervals_below_zero"] == ["2025-07", "2025-12"]
    assert not entries["small"]["eligible"]
    assert select({"small": entries["small"]}, equal, 0.0002) == ([], None)
