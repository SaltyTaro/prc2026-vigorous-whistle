"""Final-file runner, LIRF layer sets v53m / v54m / v55m (src/prc2026_final/lirf_members.py, stack.compute): hand-computed values on the
stack fixture's LIRF rows, the dark-month rule and the route guards."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

from prc2026_final import lirf_members, stack  # noqa: E402
from test_final_stack import stack_inputs  # noqa: E402


def member_frame(ids: np.ndarray, lirf_mask: np.ndarray | None) -> pd.DataFrame:
    """No v51 / v52 member value anywhere (v51m = v52m = v46), the masked LIRF network as given."""
    frame = pd.DataFrame({"MVT_ID_mvt": ids, "adsb_matched": False, "combo_all": np.nan, "mask_unmatched": np.nan})
    if lirf_mask is not None:
        frame["lirf_mask"] = lirf_mask
    return frame


def run(tmp_path, monkeypatch, months: list[str] | None, lirf_mask: np.ndarray | None):
    import export_stack
    for module in (stack, export_stack):
        monkeypatch.setattr(module, "offsets", lambda: {"EDDF": 0.0, "EHAM": 10.0})
        monkeypatch.setattr(module, "offsets_all", lambda: {"EHAM": 5.0})
    ids, detail, repaired, path, regular, peer, fill = stack_inputs(tmp_path)
    if months is not None:
        detail["month"] = months
    return stack.compute(detail, repaired, path, regular, peer, fill, member_frame(ids, lirf_mask))


def test_selections_and_learners_are_the_shipped_ones():
    chosen = lirf_members.selections()
    assert {p: n for p, (n, _) in chosen.items()} == {lirf_members.MASK_DEV: "w55", lirf_members.DARK: "w70_s750", lirf_members.COMBO: "mw40"}
    found = lirf_members.learners()
    assert len(found) == 16 and found[0].endswith("lm25_32_s801") and found[-1].endswith("lm25_32_s816")


def test_dark_months_use_the_match_share_of_the_lirf_scopes():
    scope = np.array(["lirf_nonpop_matched"] * 4 + ["lirf_nonpop_unmatched"] * 96 + ["lirf_nonpop_matched"] * 5 + ["lirf_nonpop_unmatched"] * 95
                     + ["all_no_lirf", "fixed"])
    month = np.array(["2026-07"] * 100 + ["2026-01"] * 100 + ["2026-07", "2026-07"])
    shares = lirf_members.dark_months(scope, month)
    assert shares["2026-07"]["dark"] and not shares["2026-01"]["dark"]  # 4% is dark, 5% is not
    assert shares["2026-07"]["matched"] == 4 and shares["2026-07"]["unmatched"] == 96


def test_lirf_layers_on_the_stack_fixture_covered_month(tmp_path, monkeypatch):
    """One matched and one unmatched LIRF row in January (match share 0.5): v53m moves the unmatched row, v54m adds nothing, v55m moves the
    matched row; every other row keeps v52m."""
    lirf_mask = np.full(12, 9999.0)
    lirf_mask[3], lirf_mask[4] = 1500.0, 800.0
    frame, record = run(tmp_path, monkeypatch, None, lirf_mask)
    v52m = frame["v52m"].to_numpy()
    # row 3 (unmatched; v9 1000, net 1200, tree 1900): v45's gate fires at the control (tree - net = 700): 0.2 x 1200 + 0.8 x 1900 = 1760;
    # network' = 0.45 x 1200 + 0.55 x 1500 = 1365 (tree - network' = 535, no gate): v34's blend (0.35 x 1365 + 0.25 x 1900) / 0.60
    assert v52m[3] == pytest.approx(1760.0)
    v53 = (0.35 * 1365 + 0.25 * 1900) / 0.60
    expect53 = v52m.copy()
    expect53[3] = v53
    assert np.allclose(frame["v53m"].to_numpy(), expect53, rtol=0, atol=1e-9)
    assert np.allclose(frame["v54m"].to_numpy(), expect53, rtol=0, atol=1e-9) and not frame["v54m_lirf"].any()
    # row 4 (matched; v9 900, net 980, tree 950): network' = 0.6 x 980 + 0.4 x 800 = 908, v34's matched blend, no gate either way
    expect55 = expect53.copy()
    expect55[4] = expect53[4] + (0.15 * 900 + 0.40 * 908 + 0.45 * 950) - (0.15 * 900 + 0.40 * 980 + 0.45 * 950)
    assert np.allclose(frame["v55m"].to_numpy(), expect55, rtol=0, atol=1e-9)
    assert frame["v53m_lirf"].tolist() == [False] * 3 + [True] + [False] * 8 and frame["v55m_lirf"].tolist() == [False] * 4 + [True] + [False] * 7
    assert record["rows"]["v53m"] == {"route_rows": 1, "moved": 1, "gate_rows_control": 1, "gate_rows": 0, "floored": 0}
    assert record["layer_sets"]["v54m"]["dark_months"] == [] and record["layer_sets"]["v55m"]["candidate"][0] == "mw40"
    assert record["shipped_as"]["v54m"]["lirf_dark"] == "configs/lirf_dark_reread_v1.json w70_s750"


def test_dark_month_takes_the_reread(tmp_path, monkeypatch):
    """The matched LIRF row moved to July: January's LIRF rows are all unmatched (dark), so v54m re-reads row 3 at w 0.70, share 0.75."""
    lirf_mask = np.full(12, 9999.0)
    lirf_mask[3], lirf_mask[4] = 1500.0, 800.0
    months = ["2026-01"] * 4 + ["2026-07"] + ["2026-01"] * 7
    frame, record = run(tmp_path, monkeypatch, months, lirf_mask)
    assert record["layer_sets"]["v54m"]["dark_months"] == ["2026-01"]
    # network' = 0.3 x 1200 + 0.7 x 1500 = 1410 (tree - network' = 490, no gate): 0.75 x 1410 + 0.25 x 1900
    assert frame.loc[3, "v54m"] == pytest.approx(0.75 * 1410 + 0.25 * 1900)
    assert frame["v54m_lirf"].tolist() == [False] * 3 + [True] + [False] * 8
    others = np.arange(12) != 3
    assert np.array_equal(frame["v54m"].to_numpy()[others], frame["v53m"].to_numpy()[others])


def test_no_lirf_layer_without_the_lirf_member(tmp_path, monkeypatch):
    frame, record = run(tmp_path, monkeypatch, None, None)
    assert "v52m" in frame and not any(c.startswith(("v53m", "v54m", "v55m")) for c in frame.columns) and "v53m" not in record["layer_sets"]


def test_route_guards(tmp_path, monkeypatch):
    lirf_mask = np.full(12, 1000.0)
    real = lirf_members.deltas

    def leaky(*args):
        out, record = real(*args)
        rows, delta, control = out["v53m"]
        rows = rows.copy()
        rows[0] = True  # an LTFM row on the route
        out["v53m"] = (rows, np.append(delta, 0.0), np.append(control, 0.0))
        return out, record

    monkeypatch.setattr(lirf_members, "deltas", leaky)
    with pytest.raises(ValueError, match="non-LIRF or pinned"):
        run(tmp_path, monkeypatch, None, lirf_mask)
    monkeypatch.setattr(lirf_members, "deltas", real)
    missing = lirf_mask.copy()
    missing[4] = np.nan
    with pytest.raises(ValueError, match="missing on a route row"):
        run(tmp_path, monkeypatch, None, missing)
