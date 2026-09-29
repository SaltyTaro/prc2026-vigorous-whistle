"""Design readings of the context network's combination batch (configs/adsb_ctx3_combo_v1.json): October only.

As tools/ctx3_design_report2.py (pool P, share 0.75, R2 at b in {0.25, 0.5, 1.0}, paired-day interval at b = 0.5 against ref32's
table), with three declared items on fresh CPU seeds:
  combo_dl    dl32 (numeric embeddings 16 wide, learning rate doubled) on every row
  route_m25   ref32 on rows with adsb_matched = 1, m25_32 (masking 0.25) on the other rows
  combo_all   dl32 on rows with adsb_matched = 1, m25dl32 on the other rows
GO (declared): R2 at b = 0.5 at least 0.05 points below ref32's, four-seed standalone RMSE below ref32's, interval below zero; of
several GO items the lowest R2 is carried forward. Share 0.85 is reported beside the governing 0.75. No scoring month is read.
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

DECLARATION = "configs/adsb_ctx3_combo_v1.json"
B_VALUES = [0.25, 0.5, 1.0]
SHARE, OTHER_SHARE = 0.75, 0.85
BAR_POINTS = 0.05
ITEMS = {"combo_dl": ("dl32", "dl32"), "route_m25": ("ref32", "m25_32"), "combo_all": ("dl32", "m25dl32")}  # (matched rows, other rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=Path("runs/adsb_ctx3_combo/design"))
    parser.add_argument("--pool-runs", type=Path, default=Path("runs/adsb_ctx"))
    parser.add_argument("--committee", type=Path, default=Path("runs/adsb_nn/design/committee_october.parquet"))
    parser.add_argument("--report", type=Path, default=Path("docs/adsb_ctx3_combo/design_report.json"))
    parser.add_argument("--allow-missing", action="store_true", help="interim reading: variants with missing seeds are read on the seeds every variant has")
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

    def net(values, rows=None):
        k = non_lirf if rows is None else (non_lirf & rows)
        return rmse(label[k], values[k])

    def proxied(x, b, share=SHARE):
        return proxy.values((1 - share) * committee + share * ((1 - b) * pool + b * x))

    control = {share: rmse(label, proxy.values((1 - share) * committee + share * pool)) for share in (SHARE, OTHER_SHARE)}

    def r2(x, b, share=SHARE):
        return 100 * (rmse(label, proxied(x, b, share)) / control[share] - 1)

    runs = {}
    for path in sorted((ROOT / args.runs).glob("*/design/run.json")):
        record = read_json(path)
        if record["config_sha256"] != decl_sha or record.get("smoke_limit_rows") is not None or record["inputs"]["device"] != "cpu":
            raise ValueError(f"{path} was not fitted under this declaration.")
        if record["variant"] not in decl["runs"] or record["seed"] not in decl["runs"][record["variant"]]:
            raise ValueError(f"{path}: undeclared run.")
        runs.setdefault(record["variant"], {})[record["seed"]] = {"values": october(path.parent / "stop_predictions.parquet", ids), "fit_seconds": record["fit_seconds"],
                                                                   "curve_last": [round(e["monitor_rmse"], 3) for e in record["history"][-3:]],
                                                                   "predictions_sha256": sha256_file(path.parent / "stop_predictions.parquet")}
    missing = {v: sorted(set(s) - set(runs.get(v, {}))) for v, s in decl["runs"].items() if set(s) - set(runs.get(v, {}))}
    if missing and not args.allow_missing:
        raise ValueError(f"Declared runs missing: {missing}")
    seeds = sorted(set.intersection(*[set(runs.get(v, {})) for v in decl["runs"]]))
    if not seeds:
        raise ValueError("No seed is complete over the four variants.")
    mean = {v: np.mean([runs[v][s]["values"] for s in seeds], axis=0) for v in decl["runs"]}
    ref = mean["ref32"]
    ref_proxy, ref_r2 = proxied(ref, 0.5), r2(ref, 0.5)
    tables = {"ref32": ref, **{name: np.where(matched, mean[a], mean[b]) for name, (a, b) in ITEMS.items()}, **{v: mean[v] for v in ("dl32", "m25_32", "m25dl32")}}
    items = {}
    for name, x in tables.items():
        entry = {"rmse_non_lirf": net(x), "rmse_matched": net(x, matched), "rmse_unmatched": net(x, ~matched),
                 "rmse_by_airport": {a: net(x, airport == a) for a in sorted(set(airport[non_lirf]))},
                 "r2_pct": {str(b): r2(x, b) for b in B_VALUES}, "r2_pct_share_085": {str(b): r2(x, b, OTHER_SHARE) for b in B_VALUES}, "decision_item": name in ITEMS}
        if name in runs:
            entry["per_seed_rmse_non_lirf"] = {str(s): net(runs[name][s]["values"]) for s in seeds}
        if name != "ref32":
            v = proxied(x, 0.5)
            entry.update({"gain_points_vs_ref32_b0.5": ref_r2 - entry["r2_pct"]["0.5"], "standalone_below_ref32": bool(entry["rmse_non_lirf"] < net(ref)),
                          "standalone_vs_ref32_pct": 100 * (entry["rmse_non_lirf"] / net(ref) - 1),
                          "interval_vs_ref32_b0.5_s": day_interval(days, label, v, ref_proxy),
                          "gain_points_vs_ref32_b1.0": r2(ref, 1.0) - entry["r2_pct"]["1.0"],
                          "unmatched_change_vs_ref32_pct": 100 * (net(v, ~matched) / net(ref_proxy, ~matched) - 1),
                          "matched_change_vs_ref32_pct": 100 * (net(v, matched) / net(ref_proxy, matched) - 1)})
            if entry["decision_item"]:
                entry["go"] = bool(not missing and entry["gain_points_vs_ref32_b0.5"] >= BAR_POINTS and entry["standalone_below_ref32"] and entry["interval_vs_ref32_b0.5_s"][1] < 0)
        items[name] = entry
    go = [n for n in ITEMS if items[n].get("go")]
    carried = min(go, key=lambda n: items[n]["r2_pct"]["0.5"]) if go else None
    report = {"created_at_utc": utc_now(), "declaration_sha256": decl_sha, "seeds_read": seeds, "missing": missing, "control_proxy_rmse": control[SHARE],
              "control_proxy_rmse_share_085": control[OTHER_SHARE], "pool_rmse_non_lirf": net(pool), "share": SHARE, "bar_points": BAR_POINTS, "items": items, "go": go,
              "carried_forward": carried,
              "runs": {v: {str(s): {k: r[k] for k in ("fit_seconds", "curve_last", "predictions_sha256")} for s, r in by_seed.items()} for v, by_seed in runs.items()},
              "decision": decl["protocol"]["decision"], "tool_sha256": sha256_file(ROOT / "tools/ctx3_combo_report.py"), "audit_used": False,
              "note": "October only; no scoring month, ranking row or official score read."}
    (ROOT / args.report).parent.mkdir(parents=True, exist_ok=True)
    write_json(ROOT / args.report, report)
    print(f"pool October non-LIRF {report['pool_rmse_non_lirf']:.3f}; control proxy {control[SHARE]:.3f}; seeds {seeds}; ref32 R2(b=0.5) {ref_r2:+.4f}%")
    for name, e in items.items():
        flag = ("GO" if e.get("go") else "no") if e["decision_item"] else "  "
        extra = (f" standalone {e['standalone_vs_ref32_pct']:+.3f}%, gain {e['gain_points_vs_ref32_b0.5']:+.4f} pts (b=1: {e['gain_points_vs_ref32_b1.0']:+.4f}), "
                 f"interval [{e['interval_vs_ref32_b0.5_s'][0]:+.2f}, {e['interval_vs_ref32_b0.5_s'][1]:+.2f}] s, "
                 f"unmatched {e['unmatched_change_vs_ref32_pct']:+.3f}%, matched {e['matched_change_vs_ref32_pct']:+.3f}%") if name != "ref32" else ""
        seeds_text = f" seeds {[round(v, 2) for v in e['per_seed_rmse_non_lirf'].values()]}" if "per_seed_rmse_non_lirf" in e else ""
        print(f"{flag:3s} {name:10s} R1 {e['rmse_non_lirf']:.3f} (m {e['rmse_matched']:.2f} / u {e['rmse_unmatched']:.2f}) R2 b0.5 {e['r2_pct']['0.5']:+.4f}%{extra}{seeds_text}")
    print("GO:", go, "carried forward:", carried)


if __name__ == "__main__":
    main()
