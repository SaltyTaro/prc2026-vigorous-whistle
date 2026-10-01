"""Design reading of configs/learner_family_design_v1.json: October non-LIRF only; no scoring month, ranking row or official score is read.

Recipe proxy as tools/ctx3_combo_report.py: table = (1 - share) * committee' + share * P (P = the 16 ctx + 16 ctx_e24 network seed means,
share 0.75, 0.85 reported), committee' = (1 - c) * committee + c * X, X = a variant's four-seed mean, c in {0.25, 0.5}. The control is
ref_lgb in the same slot. GO (declared): R2 at c = 0.5 at least 0.05 points below ref_lgb's AND standalone RMSE at most 0.5% above
ref_lgb's AND the paired-day 95% interval of (variant - ref_lgb) at c = 0.5 below zero.
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

DECL = "configs/learner_family_design_v1.json"
VARIANTS = ["ref_lgb", "dart", "linear_tree", "huber_tail"]
C_VALUES, SHARES, BAR = [0.25, 0.5], [0.75, 0.85], 0.05


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, default=Path("runs/learner_family/design"))
    p.add_argument("--pool-runs", type=Path, default=Path("runs/adsb_ctx"))
    p.add_argument("--committee", type=Path, default=Path("runs/adsb_nn/design/committee_october.parquet"))
    p.add_argument("--report", type=Path, default=Path("docs/learner_family/design_report.json"))
    a = p.parse_args()
    proxy = RecipeProxy(pd.read_parquet(ROOT / a.committee))
    ids, label, nl, committee, table = proxy.ids, proxy.label, proxy.non_lirf, proxy.committee, proxy.table
    days = table[TIME].dt.strftime("%Y-%m-%d").to_numpy()
    matched, airport = table["adsb_matched"].eq(1).to_numpy(), table["airport"].to_numpy()
    ctx = [ROOT / a.pool_runs / f"ctx_s{s}/seasonal/stop_predictions.parquet" for s in range(101, 117)]
    e24 = [ROOT / a.pool_runs / f"ctx_e24_s{s}/seasonal/stop_predictions.parquet" for s in range(117, 133)]
    pool = 0.5 * np.mean([october(q, ids) for q in ctx], axis=0) + 0.5 * np.mean([october(q, ids) for q in e24], axis=0)
    runs = {}
    for v in VARIANTS:
        for s in range(1, 5):
            d = ROOT / a.runs / f"{v}_s{s}"
            rec = read_json(d / "run.json")
            if rec["fold"] != "design" or rec["audit_used"]:
                raise ValueError(d)
            runs.setdefault(v, {})[s] = (october(d / "stop_predictions.parquet", ids), rec["best_iteration"], rec["october_rmse_non_lirf"], rec["seconds"])
    net = lambda x, k=None: rmse(label[nl if k is None else nl & k], x[nl if k is None else nl & k])
    prox = lambda x, c, share: proxy.values((1 - share) * ((1 - c) * committee + c * x) + share * pool)
    control_table = {sh: rmse(label, proxy.values((1 - sh) * committee + sh * pool)) for sh in SHARES}
    mean = {v: np.mean([runs[v][s][0] for s in runs[v]], axis=0) for v in VARIANTS}
    r2 = lambda x, c, sh: 100 * (rmse(label, prox(x, c, sh)) / control_table[sh] - 1)
    ref = mean["ref_lgb"]
    ref_prox = prox(ref, 0.5, 0.75)
    out = {}
    for v in VARIANTS:
        x = mean[v]
        e = {"rmse_non_lirf": net(x), "rmse_matched": net(x, matched), "rmse_unmatched": net(x, ~matched),
             "per_seed_rmse": {str(s): net(runs[v][s][0]) for s in runs[v]}, "best_iterations": {str(s): runs[v][s][1] for s in runs[v]},
             "r2_pct": {f"c{c}_share{sh}": r2(x, c, sh) for c in C_VALUES for sh in SHARES}, "seconds": {str(s): runs[v][s][3] for s in runs[v]},
             "error_correlation_with_ref": float(np.corrcoef(label[nl] - x[nl], label[nl] - ref[nl])[0, 1])}
        if v != "ref_lgb":
            e["gain_points_vs_ref_c0.5"] = r2(ref, 0.5, 0.75) - r2(x, 0.5, 0.75)
            e["standalone_vs_ref_pct"] = 100 * (e["rmse_non_lirf"] / net(ref) - 1)
            e["interval_vs_ref_c0.5_s"] = day_interval(days, label, prox(x, 0.5, 0.75), ref_prox)
            e["go"] = bool(e["gain_points_vs_ref_c0.5"] >= BAR and e["standalone_vs_ref_pct"] <= 0.5 and e["interval_vs_ref_c0.5_s"][1] < 0)
        out[v] = e
    go = [v for v in VARIANTS if out[v].get("go")]
    rep = {"created_at_utc": utc_now(), "declaration_sha256": sha256_file(ROOT / DECL), "control_table_proxy_rmse": control_table, "pool_rmse_non_lirf": net(pool),
           "committee_rmse_non_lirf": net(committee), "variants": out, "go": go, "bar_points": BAR, "audit_used": False,
           "note": "October only; no scoring month, ranking row or official score read."}
    (ROOT / a.report).parent.mkdir(parents=True, exist_ok=True)
    write_json(ROOT / a.report, rep)
    print(f"committee {net(committee):.3f}, pool {net(pool):.3f}, control proxy {control_table[0.75]:.3f}")
    for v, e in out.items():
        extra = f" gain {e['gain_points_vs_ref_c0.5']:+.4f} pts, standalone {e['standalone_vs_ref_pct']:+.3f}%, interval {e['interval_vs_ref_c0.5_s'][0]:+.2f}..{e['interval_vs_ref_c0.5_s'][1]:+.2f} -> {'GO' if e['go'] else 'no'}" if v != "ref_lgb" else ""
        print(f"{v:12s} R1 {e['rmse_non_lirf']:.3f} R2(c.5) {e['r2_pct']['c0.5_share0.75']:+.4f}%{extra}")
    print("GO:", go)


if __name__ == "__main__":
    main()
