import numpy as np
import pandas as pd
import pytest

from prc2026.features import ID
from prc2026_final import members, stack


def test_learners_follow_the_declarations():
    found = members.learners()
    assert sorted(found) == ["combo_dl32", "combo_m25dl32", "lirf_lm25_32", "mask_m25_32"]
    assert all(len(v) == 16 for v in found.values())
    assert found["combo_dl32"][0] == "runs/adsb_ctx3_combo/nets/dl32_s601" and found["mask_m25_32"][-1] == "runs/adsb_ctx3_mask/nets/m25_32_s716"
    assert found["lirf_lm25_32"][0] == "runs/adsb_ctx3_lirf_mask/nets/lm25_32_s801" and found["lirf_lm25_32"][-1] == "runs/adsb_ctx3_lirf_mask/nets/lm25_32_s816"
    assert all(name in stack.LAYER_SETS for name in ["v51m", "v52m", "v53m", "v54m", "v55m"])


def test_members_route_by_the_match_flag():
    ids = np.array([1, 2, 3])
    ens = pd.DataFrame({ID: ids, "combo_dl32": [100.0, 200.0, 300.0], "combo_m25dl32": [110.0, 210.0, 310.0], "mask_m25_32": [120.0, 220.0, 320.0]})
    out = members.members(ids, ens, np.array([True, False, True]))
    assert out["combo_all"].tolist() == [100.0, 210.0, 300.0]
    assert np.isnan(out["mask_unmatched"].to_numpy()[[0, 2]]).all() and out["mask_unmatched"].to_numpy()[1] == 220.0
    assert "lirf_mask" not in out  # a table without the LIRF ensemble builds no LIRF layer
    with pytest.raises(ValueError):
        members.members(ids[::-1], ens, np.array([True, False, True]))
    lirf = members.members(ids, ens.assign(lirf_lm25_32=[130.0, 230.0, 330.0]), np.array([True, False, True]))
    assert lirf["lirf_mask"].tolist() == [130.0, 230.0, 330.0]  # unrouted: matched and unmatched rows alike


def test_tables_weights_and_lirf():
    frame = pd.DataFrame({"combo_all": [600.0, 600.0, 600.0], "mask_unmatched": [np.nan, 300.0, 300.0]})
    nets = np.array([1000.0, 1000.0, 1000.0])
    t51, in51, t52, in52 = members.tables(nets, frame, np.array([False, False, True]))
    assert t51[:2] == pytest.approx([0.6 * 1000 + 0.4 * 600] * 2) and t51[2] == 1000.0  # LIRF keeps the shipped table
    assert in51.tolist() == [True, True, False] and in52.tolist() == [False, True, False]
    assert t52[0] == t51[0] and t52[1] == pytest.approx(0.6 * t51[1] + 0.4 * 300) and t52[2] == 1000.0
