"""Design readings for the GPU sprint's second batch (configs/adsb_ctx3_design_v2.json): October only.

As tools/ctx3_design_report.py (pool P, share 0.75, R2 at b in {0.25, 0.5, 1.0}, paired-day interval at b = 0.5), with ref32 as the
reference and four declared items: route_mask25/50/100 (ref32 on rows with adsb_matched = 1, the masked variant elsewhere) and e40.
GO (declared): R2 at b = 0.5 at least 0.05 points below ref32's, 4-seed standalone RMSE below ref32's, interval below zero; of the routed
items only the lowest R2 is carried forward. No scoring month is read.
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

DECLARATION = "configs/adsb_ctx3_design_v2.json"
B_VALUES = [0.25, 0.5, 1.0]
SHARE = 0.75
BAR_POINTS = 0.05
ROUTED = {"route_mask25": "mask25_32", "route_mask50": "mask50_32", "route_mask100": "mask100_32"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path)
    parser.add_argument("--pool-runs", type=Path, default=Path("runs/adsb_ctx"))
    parser.add_argument("--committee", type=Path, default=Path("runs/adsb_nn/design/committee_october.parquet"))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    decl = read_json(ROOT / DECLARATION)
    decl_sha = sha256_file(ROOT / DECLARATION)
    proxy = RecipeProxy(pd.read_parquet(ROOT / args.committee))
    ids, label, non_lirf, table = proxy.ids, proxy.label, proxy.non_lirf, proxy.table
    days = table[TIME].dt.strftime("%Y-%m-%d").to_numpy()
    matched = table["adsb_matched"].eq(1).to_numpy()
    airport = table["airport"].to_numpy()
    committee = proxy.committee
    ctx = [ROOT / args.pool_runs / f"ctx_s{s}/seasonal/stop_predictions.parquet" for s in range(101, 117)]
    e24 = [ROOT / args.pool_runs / f"ctx_e24_s{s}/seasonal/stop_predictions.parquet" for s in range(117, 133)]
    pool = 0.5 * np.mean([october(p, ids) for p in ctx], axis=0) + 0.5 * np.mean([october(p, ids) for p in e24], axis=0)
    control_rmse = rmse(label, proxy.values((1 - SHARE) * committee + SHARE * pool))

    def net(values, rows=None):
        k = non_lirf if rows is None else (non_lirf & rows)
        return rmse(label[k], values[k])

    def proxied(x, b):
        return proxy.values((1 - SHARE) * committee + SHARE * ((1 - b) * pool + b * x))

    def r2(x, b):
        return 100 * (rmse(label, proxied(x, b)) / control_rmse - 1)

    runs = {}
    for path in sorted((ROOT / args.runs).glob("*/design/run.json")):
        record = read_json(path)
        if record["config_sha256"] != decl_sha or record.get("smoke_limit_rows") is not None:
            raise ValueError(f"{path} was not fitted under the batch-2 declaration.")
        runs.setdefault(record["variant"], {})[record["seed"]] = {"values": october(path.parent / "stop_predictions.parquet", ids),
                                                                   "fit_seconds": record["fit_seconds"], "curve_last": [round(e["monitor_rmse"], 3) for e in record["history"][-3:]],
                                                                   "predictions_sha256": sha256_file(path.parent / "stop_predictions.parquet")}
    missing = {v: sorted(set(s) - set(runs.get(v, {}))) for v, s in decl["runs"].items() if set(s) - set(runs.get(v, {}))}
    if missing:
        raise ValueError(f"Declared runs missing: {missing}")
    mean = {v: np.mean([runs[v][s]["values"] for s in decl["runs"][v]], axis=0) for v in decl["runs"]}
    ref = mean["ref32"]
    ref_proxy, ref_r2 = proxied(ref, 0.5), r2(ref, 0.5)
    tables = {"ref32": ref, "e40": mean["e40"], **{name: np.where(matched, ref, mean[v]) for name, v in ROUTED.items()},
              **{v: mean[v] for v in ROUTED.values()}}
    items = {}
    for name, x in tables.items():
        entry = {"rmse_non_lirf": net(x), "rmse_matched": net(x, matched), "rmse_unmatched": net(x, ~matched),
                 "rmse_unmatched_by_airport": {a: net(x, (~matched) & (airport == a)) for a in ("LFPG", "EGLL", "LEMD", "LTFM", "EDDM", "EDDF")},
                 "r2_pct": {str(b): r2(x, b) for b in B_VALUES}, "decision_item": name in ROUTED or name == "e40"}
        if name in runs:
            entry["per_seed_rmse_non_lirf"] = {str(s): net(runs[name][s]["values"]) for s in decl["runs"][name]}
        if name != "ref32":
            v = proxied(x, 0.5)
            entry.update({"gain_points_vs_ref32_b0.5": ref_r2 - entry["r2_pct"]["0.5"], "standalone_below_ref32": bool(entry["rmse_non_lirf"] < net(ref)),
                          "interval_vs_ref32_b0.5_s": day_interval(days, label, v, ref_proxy),
                          "unmatched_change_vs_ref32_pct": 100 * (net(v, ~matched) / net(ref_proxy, ~matched) - 1),
                          "matched_change_vs_ref32_pct": 100 * (net(v, matched) / net(ref_proxy, matched) - 1)})
            if entry["decision_item"]:
                entry["go"] = bool(entry["gain_points_vs_ref32_b0.5"] >= BAR_POINTS and entry["standalone_below_ref32"] and entry["interval_vs_ref32_b0.5_s"][1] < 0)
        items[name] = entry
    routed_go = [n for n in ROUTED if items[n].get("go")]
    carried = min(routed_go, key=lambda n: items[n]["r2_pct"]["0.5"]) if routed_go else None
    go = sorted([carried] if carried else []) + (["e40"] if items["e40"].get("go") else [])
    report = {"created_at_utc": utc_now(), "declaration_sha256": decl_sha, "control_proxy_rmse": control_rmse, "pool_rmse_non_lirf": net(pool), "share": SHARE,
              "bar_points": BAR_POINTS, "items": items, "routed_go": routed_go, "go": go,
              "runs": {v: {str(s): {k: r[k] for k in ("fit_seconds", "curve_last", "predictions_sha256")} for s, r in seeds.items()} for v, seeds in runs.items()},
              "decision": decl["protocol"]["decision"], "audit_used": False, "note": "October only; no scoring month read."}
    write_json(ROOT / args.report, report)
    print(f"pool October non-LIRF {report['pool_rmse_non_lirf']:.3f}; control proxy {control_rmse:.3f}; ref32 R2(b=0.5) {ref_r2:+.4f}%")
    for name, e in items.items():
        flag = ("GO" if e.get("go") else "no") if e["decision_item"] else "  "
        extra = (f" gain {e['gain_points_vs_ref32_b0.5']:+.4f} pts, interval [{e['interval_vs_ref32_b0.5_s'][0]:+.2f}, {e['interval_vs_ref32_b0.5_s'][1]:+.2f}] s, "
                 f"unmatched {e['unmatched_change_vs_ref32_pct']:+.3f}%, matched {e['matched_change_vs_ref32_pct']:+.3f}%") if name != "ref32" else ""
        print(f"{flag:3s} {name:14s} R1 {e['rmse_non_lirf']:.3f} (unm {e['rmse_unmatched']:.2f}) R2 b0.5 {e['r2_pct']['0.5']:+.4f}%{extra}")
    print("GO:", go)


if __name__ == "__main__":
    main()
