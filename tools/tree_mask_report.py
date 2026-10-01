"""Design reading of configs/tree_mask_design_v1.json: October non-LIRF rows only; no scoring month, ranking row or official score is read.

Table as tools/learner_family_report.py (RecipeProxy on the committee's October table), with the network pool P1 = P0 on matched rows and
0.6 * P0 + 0.4 * M on unmatched rows (P0 = 16 ctx + 16 ctx_e24 seed means, M = m25_32 seeds 9-12, standing in for v52's masked member).
Table = (1 - share) * committee' + share * P1'. Unmatched = adsb_matched 0 in the proxy table; only unmatched non-LIRF rows ever change.

- standalone: unmatched non-LIRF RMSE of the variant's four-seed mean against its control's; paired-day interval of the difference.
- swap proxy: committee' = committee + w * (X_variant - X_control) on unmatched rows (w 0.5, 1); read against the unchanged table.
- member proxy: P1' = (1 - b) * P1 + b * X on unmatched rows (b 0.25, 0.4); variant against the same with the control's mean.
GO (declared): standalone at least 0.3% below the control with the interval below zero, AND (swap at w = 1 at most -0.05 points with its
interval below zero, OR member at b = 0.4 at least 0.05 points below the control's member reading with that interval below zero).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import TIME  # noqa: E402
from nn_design_report import RecipeProxy, day_interval, rmse  # noqa: E402
from ctx_design_report import october  # noqa: E402

DECL = "configs/tree_mask_design_v1.json"
FAMILIES = {"lgb": {"control": "ref_lgb", "variants": ["lgb_m25", "lgb_m50"]}, "cat": {"control": "cat_ref", "variants": ["cat_m25"]}}
SHARES, SWAP_W, MEMBER_B = [0.75, 0.85], [0.5, 1.0], [0.25, 0.4]
STANDALONE_BAR, BAR = 0.3, 0.05


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", required=True, choices=list(FAMILIES))
    p.add_argument("--runs", type=Path, default=Path("runs/tree_mask/design"))
    p.add_argument("--control-runs", type=Path, default=Path("runs/learner_family/design"), help="ref_lgb seeds 1-4 (LightGBM family)")
    p.add_argument("--pool-runs", type=Path, default=Path("runs/adsb_ctx"))
    p.add_argument("--masked-runs", type=Path, default=Path("runs/adsb_ctx3_combo/design"))
    p.add_argument("--committee", type=Path, default=Path("runs/adsb_nn/design/committee_october.parquet"))
    p.add_argument("--report", type=Path)
    a = p.parse_args()
    decl = read_json(ROOT / DECL)
    if not decl.get("declared_at_utc"):
        raise ValueError("The declaration is a draft.")
    fam = FAMILIES[a.family]
    proxy = RecipeProxy(pd.read_parquet(ROOT / a.committee))
    ids, label, nl, committee, table = proxy.ids, proxy.label, proxy.non_lirf, proxy.committee, proxy.table
    days = table[TIME].dt.strftime("%Y-%m-%d").to_numpy()
    unm = table["adsb_matched"].ne(1).to_numpy() & nl
    ctx = [ROOT / a.pool_runs / f"ctx_s{s}/seasonal/stop_predictions.parquet" for s in range(101, 117)]
    e24 = [ROOT / a.pool_runs / f"ctx_e24_s{s}/seasonal/stop_predictions.parquet" for s in range(117, 133)]
    p0 = 0.5 * np.mean([october(q, ids) for q in ctx], axis=0) + 0.5 * np.mean([october(q, ids) for q in e24], axis=0)
    masked = np.mean([october(ROOT / a.masked_runs / f"m25_32_s{s}/design/stop_predictions.parquet", ids) for s in range(9, 13)], axis=0)
    p1 = np.where(unm, 0.6 * p0 + 0.4 * masked, p0)

    def load(variant):
        root = ROOT / (a.control_runs if variant == "ref_lgb" else a.runs)
        out = {}
        for s in decl["seeds"].get(variant, [1, 2, 3, 4]):
            d = root / f"{variant}_s{s}"
            rec = read_json(d / "run.json")
            if rec["fold"] != "design" or rec["audit_used"]:
                raise ValueError(d)
            out[s] = {"values": october(d / "stop_predictions.parquet", ids), "best_iteration": rec["best_iteration"], "seconds": rec["seconds"],
                      "sha256": sha256_file(d / "stop_predictions.parquet")}
        return out

    runs = {v: load(v) for v in [fam["control"], *fam["variants"]]}
    mean = {v: np.mean([r["values"] for r in runs[v].values()], axis=0) for v in runs}
    values = lambda comm, pool, sh: proxy.values((1 - sh) * comm + sh * pool)
    base = {sh: values(committee, p1, sh) for sh in SHARES}
    base_rmse = {sh: rmse(label, base[sh]) for sh in SHARES}
    r2 = lambda v, sh: 100 * (rmse(label, v) / base_rmse[sh] - 1)
    swap = lambda x, xc, w, sh: values(np.where(unm, committee + w * (x - xc), committee), p1, sh)
    member = lambda x, b, sh: values(committee, np.where(unm, (1 - b) * p1 + b * x, p1), sh)
    xc = mean[fam["control"]]
    out = {"control": {"name": fam["control"], "rmse_unmatched": rmse(label[unm], xc[unm]), "rmse_matched_non_lirf": rmse(label[nl & ~unm], xc[nl & ~unm]),
                       "per_seed_rmse_unmatched": {str(s): rmse(label[unm], r["values"][unm]) for s, r in runs[fam["control"]].items()},
                       "member_r2_pct": {f"b{b}_share{sh}": r2(member(xc, b, sh), sh) for b in MEMBER_B for sh in SHARES},
                       "best_iterations": {str(s): r["best_iteration"] for s, r in runs[fam["control"]].items()}}}
    for v in fam["variants"]:
        x = mean[v]
        e = {"rmse_unmatched": rmse(label[unm], x[unm]), "rmse_matched_non_lirf": rmse(label[nl & ~unm], x[nl & ~unm]),
             "per_seed_rmse_unmatched": {str(s): rmse(label[unm], r["values"][unm]) for s, r in runs[v].items()},
             "best_iterations": {str(s): r["best_iteration"] for s, r in runs[v].items()}, "seconds": {str(s): r["seconds"] for s, r in runs[v].items()},
             "predictions_sha256": {str(s): r["sha256"] for s, r in runs[v].items()}}
        e["standalone_vs_control_pct"] = 100 * (e["rmse_unmatched"] / out["control"]["rmse_unmatched"] - 1)
        e["standalone_interval_s"] = day_interval(days[unm], label[unm], x[unm], xc[unm])
        e["swap_r2_pct"] = {f"w{w}_share{sh}": r2(swap(x, xc, w, sh), sh) for w in SWAP_W for sh in SHARES}
        e["swap_interval_w1_s"] = day_interval(days, label, swap(x, xc, 1.0, 0.75), base[0.75])
        e["member_r2_pct"] = {f"b{b}_share{sh}": r2(member(x, b, sh), sh) for b in MEMBER_B for sh in SHARES}
        e["member_gain_vs_control_b0.4_pts"] = out["control"]["member_r2_pct"]["b0.4_share0.75"] - e["member_r2_pct"]["b0.4_share0.75"]
        e["member_interval_b0.4_vs_control_s"] = day_interval(days, label, member(x, 0.4, 0.75), member(xc, 0.4, 0.75))
        standalone_ok = e["standalone_vs_control_pct"] <= -STANDALONE_BAR and e["standalone_interval_s"][1] < 0
        swap_ok = e["swap_r2_pct"]["w1.0_share0.75"] <= -BAR and e["swap_interval_w1_s"][1] < 0
        member_ok = e["member_gain_vs_control_b0.4_pts"] >= BAR and e["member_interval_b0.4_vs_control_s"][1] < 0
        e["checks"] = {"standalone": standalone_ok, "swap": swap_ok, "member": member_ok}
        e["go"] = bool(standalone_ok and (swap_ok or member_ok))
        out[v] = e
    go = [v for v in fam["variants"] if out[v]["go"]]
    rep = {"created_at_utc": utc_now(), "family": a.family, "declaration_sha256": sha256_file(ROOT / DECL), "unmatched_rows": int(unm.sum()),
           "base_table_rmse": base_rmse, "committee_rmse_unmatched": rmse(label[unm], committee[unm]), "p0_rmse_unmatched": rmse(label[unm], p0[unm]),
           "p1_rmse_unmatched": rmse(label[unm], p1[unm]), "variants": out, "go": go, "audit_used": False,
           "note": "October only; no scoring month, ranking row or official score read."}
    report = a.report or Path(f"docs/tree_mask/design_report_{a.family}.json")
    (ROOT / report).parent.mkdir(parents=True, exist_ok=True)
    write_json(ROOT / report, rep)
    print(f"unmatched rows {rep['unmatched_rows']}: committee {rep['committee_rmse_unmatched']:.3f}, P0 {rep['p0_rmse_unmatched']:.3f}, P1 {rep['p1_rmse_unmatched']:.3f}; "
          f"table {base_rmse[0.75]:.3f}")
    c = out["control"]
    print(f"{c['name']:8s} unmatched {c['rmse_unmatched']:.3f} matched {c['rmse_matched_non_lirf']:.3f} member b.4 {c['member_r2_pct']['b0.4_share0.75']:+.4f}%")
    for v in fam["variants"]:
        e = out[v]
        print(f"{v:8s} unmatched {e['rmse_unmatched']:.3f} ({e['standalone_vs_control_pct']:+.3f}%, {e['standalone_interval_s'][0]:+.2f}..{e['standalone_interval_s'][1]:+.2f}) "
              f"swap w1 {e['swap_r2_pct']['w1.0_share0.75']:+.4f}% ({e['swap_interval_w1_s'][0]:+.2f}..{e['swap_interval_w1_s'][1]:+.2f}) "
              f"member b.4 {e['member_r2_pct']['b0.4_share0.75']:+.4f}% gain {e['member_gain_vs_control_b0.4_pts']:+.4f} "
              f"({e['member_interval_b0.4_vs_control_s'][0]:+.2f}..{e['member_interval_b0.4_vs_control_s'][1]:+.2f}) -> {'GO' if e['go'] else 'no'}")
    print("GO:", go)


if __name__ == "__main__":
    main()
