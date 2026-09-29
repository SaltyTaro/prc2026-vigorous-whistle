import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from prc2026_final import share_gate  # noqa: E402


def test_mix_equals_the_committee_mixing_rule():
    from run_adsb_nn_dev import mixed
    rng = np.random.default_rng(0)
    committee, nets = rng.uniform(300, 3000, 500), rng.uniform(300, 3000, 500)
    nets[::7] = np.nan
    for share in (0.75, 0.85, 1.0):
        values, fire = share_gate.mix(committee, nets, share)
        assert np.allclose(values, mixed(committee, nets, share), rtol=0, atol=1e-12)
        assert not fire.any()
    assert np.array_equal(share_gate.mix(committee, nets, 0.85)[0][::7], committee[::7])  # no network value: the committee alone


def test_gate_fires_only_where_the_committee_is_far_above():
    committee = np.array([1000.0, 3000.0, 3000.0, 3000.0, 500.0])
    nets = np.array([900.0, 1800.0, 1799.0, np.nan, 2500.0])
    values, fire = share_gate.mix(committee, nets, 0.85, {"threshold_s": 1200, "share_inside": 0.5})
    assert fire.tolist() == [False, False, True, False, False]  # strictly above the threshold, both finite
    assert values[0] == pytest.approx(0.15 * 1000 + 0.85 * 900)
    assert values[1] == pytest.approx(0.15 * 3000 + 0.85 * 1800)
    assert values[2] == pytest.approx(0.5 * 3000 + 0.5 * 1799)
    assert values[3] == 3000.0
    assert values[4] == pytest.approx(0.15 * 500 + 0.85 * 2500)  # networks above the committee: never gated


def test_network_table_and_candidate_spec():
    detail = pd.DataFrame({"ens_ctx_e24": [100.0], "ens_ctx_e24b": [200.0], "ens_ctx_e32": [400.0]})
    assert share_gate.network_table(detail)[0] == pytest.approx(0.25 * 100 + 0.25 * 200 + 0.5 * 400)
    decl = {"control": {"network_share": 0.75},
            "candidates": {"n85_nd1200": {"network_share": 0.85, "gate": {"threshold_s": 1200, "share_inside": 0.5}}, "nd1200": {"gate": {"threshold_s": 1200, "share_inside": 0.5}},
                           "u85": {"network_share": 0.85, "scope": "unmatched"}}}
    assert share_gate.candidate_spec(decl, "n85_nd1200") == (0.85, {"threshold_s": 1200, "share_inside": 0.5})
    assert share_gate.candidate_spec(decl, "nd1200")[0] == 0.75
    with pytest.raises(ValueError):
        share_gate.candidate_spec(decl, "u85")


def test_round_and_ranking_semantics_agree():
    """tools/run_thinned_regime_round.values_for (the declared round) and share_gate.mix (the export) give the same non-LIRF values."""
    from run_thinned_regime_round import values_for
    rng = np.random.default_rng(1)
    n = 400
    committee, nets = rng.uniform(300, 4000, n), rng.uniform(300, 4000, n)
    airport = rng.choice(["EHAM", "LTFM", "EGLL"], size=n)
    f = pd.DataFrame({"airport": airport, "matched": rng.random(n) < 0.5, "floor": -12.0, "base": rng.uniform(300, 4000, n), "committee": committee, "nets": nets,
                      "mix": 0.25 * committee + 0.75 * nets, "v9": 0.0, "v33": 0.0, "lirf_m": False, "lirf_u": False, "lirf_net": 0.0, "lirf_tree": 0.0})
    control = {"outer": {"ltfm": 1.0, "matched_other": 0.6, "unmatched_other": 0.7}, "network_share": 0.75,
               "lirf": {"lirf_nonpop_matched": {"v9": 0.15, "net": 0.40, "tree": 0.45}, "lirf_nonpop_unmatched": {"v9": 0.0, "net": 0.5833333333333334, "tree": 0.4166666666666667}}}
    gate = {"threshold_s": 1200, "share_inside": 0.5}
    expected_mix, fire = share_gate.mix(committee, nets, 0.85, gate)
    assert fire.any()
    w = np.where(airport == "LTFM", 1.0, np.where(f["matched"].to_numpy(), 0.6, 0.7))
    expected = np.maximum((1 - w) * f["base"].to_numpy() + w * expected_mix, -12.0)
    assert np.allclose(values_for(f, control, {"network_share": 0.85, "gate": gate}), expected, rtol=0, atol=1e-9)
