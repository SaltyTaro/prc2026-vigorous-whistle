"""Design readings for the GPU sprint (configs/adsb_ctx3_design_v1.json): October only.

P   the incumbent network table on October: 0.5 x mean(ctx seeds 101-116) + 0.5 x mean(ctx_e24 seeds 117-132), seasonal-fold stop predictions.
R1  October non-LIRF RMSE of every run and of each item's 4-seed mean (final weights and readouts); matched / unmatched; unmatched by airport.
R2  recipe proxy (tools/nn_design_report.RecipeProxy) of table = 0.25 committee + 0.75 [(1 - b) P + b X], b in {0.25, 0.5, 1.0}, % against X = P.
Items: every variant but ref (final weights) against ref (final weights), and ref's readouts (ema, tail) against ref's final weights.
GO (declared): R2 at b = 0.5 at least 0.05 points below ref's, 4-seed standalone RMSE below ref's, and the paired-day 95% interval of the proxy
RMSE difference at b = 0.5 below zero. Output: a JSON report and a printed table. No scoring month is read.
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

DECLARATION = "configs/adsb_ctx3_design_v1.json"
B_VALUES = [0.25, 0.5, 1.0]
SHARE = 0.75
BAR_POINTS = 0.05
READOUTS = ["", "_ema", "_tail"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/adsb_ctx3/design"))
    parser.add_argument("--pool-runs", type=Path, default=Path("runs/adsb_ctx"))
    parser.add_argument("--round2-runs", type=Path, default=Path("runs/adsb_ctx2/design"))
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
    control = proxy.values((1 - SHARE) * committee + SHARE * pool)
    control_rmse = rmse(label, control)

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
        if record["config_sha256"] != decl_sha:
            raise ValueError(f"{path} was not fitted under the sprint declaration.")
        if record.get("smoke_limit_rows") is not None:
            raise ValueError(f"{path} is a smoke run.")
        entry = {"curve": [round(e["monitor_rmse"], 3) for e in record["history"]], "fit_seconds": record["fit_seconds"], "parameters": record["parameters"],
                 "device": record["inputs"].get("cuda_device"), "values": {}}
        for suffix in READOUTS:
            file = path.parent / f"stop_predictions{suffix}.parquet"
            entry["values"][suffix] = october(file, ids)
            entry.setdefault("predictions_sha256", {})[suffix or "final"] = sha256_file(file)
        runs.setdefault(record["variant"], {})[record["seed"]] = entry
    missing = {v: sorted(set(s) - set(runs.get(v, {}))) for v, s in decl["runs"].items() if set(s) - set(runs.get(v, {}))}
    if missing:
        raise ValueError(f"Declared runs missing: {missing}")

    def mean_of(variant, suffix=""):
        seeds = decl["runs"][variant]
        return np.mean([runs[variant][s]["values"][suffix] for s in seeds], axis=0)

    ref = mean_of("ref")
    ref_proxy = proxied(ref, 0.5)
    ref_r2 = r2(ref, 0.5)

    def describe(x):
        out = {"rmse_non_lirf": net(x), "rmse_matched": net(x, matched), "rmse_unmatched": net(x, ~matched),
               "rmse_unmatched_by_airport": {a: net(x, (~matched) & (airport == a)) for a in ("LFPG", "EGLL", "LEMD", "LTFM", "EDDM", "EDDF")},
               "r2_pct": {str(b): r2(x, b) for b in B_VALUES}}
        return out

    items = {}
    for variant in sorted(runs):
        for suffix in READOUTS:
            x = mean_of(variant, suffix)
            name = variant + suffix
            entry = describe(x)
            entry["per_seed_rmse_non_lirf"] = {str(s): net(runs[variant][s]["values"][suffix]) for s in decl["runs"][variant]}
            decision_item = (variant != "ref" and suffix == "") or (variant == "ref" and suffix != "")
            entry["decision_item"] = decision_item
            if name != "ref":
                v = proxied(x, 0.5)
                entry["gain_points_vs_ref_b0.5"] = ref_r2 - entry["r2_pct"]["0.5"]
                entry["standalone_below_ref"] = bool(entry["rmse_non_lirf"] < net(ref))
                entry["interval_vs_ref_b0.5_s"] = day_interval(days, label, v, ref_proxy)
                entry["unmatched_change_vs_ref_pct"] = 100 * (net(v, ~matched) / net(ref_proxy, ~matched) - 1)
                entry["matched_change_vs_ref_pct"] = 100 * (net(v, matched) / net(ref_proxy, matched) - 1)
                if decision_item:
                    entry["go"] = bool(entry["gain_points_vs_ref_b0.5"] >= BAR_POINTS and entry["standalone_below_ref"] and entry["interval_vs_ref_b0.5_s"][1] < 0)
            items[name] = entry
    hardware = {}
    round2 = {s: ROOT / args.round2_runs / f"g_e24_s{s}/design/stop_predictions.parquet" for s in (1, 2)}
    if all(p.exists() for p in round2.values()):
        for s, p in round2.items():
            hardware[str(s)] = {"sprint_ref": net(runs["ref"][s]["values"][""]), "round2_g_e24": net(october(p, ids))}
    go = sorted(n for n, e in items.items() if e.get("go"))
    report = {"created_at_utc": utc_now(), "declaration_sha256": decl_sha, "control_proxy_rmse": control_rmse, "pool_rmse_non_lirf": net(pool),
              "share": SHARE, "bar_points": BAR_POINTS, "items": items, "go": go, "hardware_check_vs_round2_g_e24": hardware,
              "runs": {v: {str(s): {"curve_last": r["curve"][-3:], "fit_seconds": r["fit_seconds"], "parameters": r["parameters"], "device": r["device"],
                                    "predictions_sha256": r["predictions_sha256"]} for s, r in seeds.items()} for v, seeds in runs.items()},
              "decision": decl["protocol"]["decision"], "audit_used": False, "note": "October only; no scoring month read."}
    write_json(ROOT / args.report, report)
    print(f"pool (v28/v29 network table) October non-LIRF {report['pool_rmse_non_lirf']:.3f}; control proxy {control_rmse:.3f}; ref R2(b=0.5) {ref_r2:+.4f}%")
    print("hardware check (sprint ref vs round-2 g_e24, same seeds):", {k: (round(v["sprint_ref"], 3), round(v["round2_g_e24"], 3)) for k, v in hardware.items()})
    for name, e in items.items():
        flag = ("GO" if e.get("go") else "no") if e["decision_item"] else "  "
        extra = (f" gain {e['gain_points_vs_ref_b0.5']:+.4f} pts, interval [{e['interval_vs_ref_b0.5_s'][0]:+.2f}, {e['interval_vs_ref_b0.5_s'][1]:+.2f}] s, "
                 f"unmatched {e['unmatched_change_vs_ref_pct']:+.3f}%, matched {e['matched_change_vs_ref_pct']:+.3f}%") if name != "ref" else ""
        print(f"{flag:3s} {name:12s} R1 {e['rmse_non_lirf']:.3f} (unm {e['rmse_unmatched']:.2f}) R2 b0.5 {e['r2_pct']['0.5']:+.4f}% b1 {e['r2_pct']['1.0']:+.4f}%{extra}")
    print("GO:", go)


if __name__ == "__main__":
    main()
