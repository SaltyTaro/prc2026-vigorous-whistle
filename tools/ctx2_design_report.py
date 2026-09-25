"""Design readings for the round-2 context network (configs/adsb_ctx2_design_v1.json): October only.

R1  October non-LIRF RMSE of every run and variant seed mean, each variant against g_ctx at equal seed counts; g_ctx against the
    round-1 CPU ctx seeds at equal counts (hardware check, descriptive).
R2  Incremental over the incumbent pool on the recipe proxy, as round 1 (tools/ctx_design_report.py):
    table = 0.5 x committee + 0.5 x [(1 - b) x nnb32 + b x variant_mean], b in {0.25, 0.5, 0.75, 1.0}, against b = 0.
Decision (declared): a variant replaces the round-1 settings if its R2 at b = 1 beats g_ctx's at equal seed count by >= 0.10 points and
it is better standalone. Output: a JSON report and a printed table. No scoring month is read.
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
from prc2026.features import ID, TIME  # noqa: E402
from nn_design_report import RecipeProxy, day_interval, rmse  # noqa: E402
from ctx_design_report import B_VALUES, october  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/adsb_ctx2/design"))
    parser.add_argument("--round1-runs", type=Path, default=Path("runs/adsb_ctx/design"))
    parser.add_argument("--pool", type=Path, default=Path("runs/adsb_nn"))
    parser.add_argument("--committee", type=Path, default=Path("runs/adsb_nn/design/committee_october.parquet"))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    decl = read_json(ROOT / "configs/adsb_ctx2_design_v1.json")
    proxy = RecipeProxy(pd.read_parquet(ROOT / args.committee))
    ids, label, non_lirf = proxy.ids, proxy.label, proxy.non_lirf
    table = proxy.table
    days = table[TIME].dt.strftime("%Y-%m-%d").to_numpy()
    matched = table["adsb_matched"].eq(1).to_numpy()
    airport = table["airport"].to_numpy()
    committee = proxy.committee
    pool_paths = [p for p in sorted((ROOT / args.pool).glob("nnb_s1[0-3][0-9]/seasonal/stop_predictions.parquet"))
                  if 101 <= int(p.parent.parent.name.split("_s")[1]) <= 132]
    if len(pool_paths) != 32:
        raise ValueError("Expected the 32 nnb pool seeds.")
    pool = np.mean([october(p, ids) for p in pool_paths], axis=0)
    control = proxy.values(0.5 * committee + 0.5 * pool)

    def net(values):
        return rmse(label[non_lirf], values[non_lirf])

    def r2(values, b):
        v = proxy.values(0.5 * committee + 0.5 * ((1 - b) * pool + b * values))
        return 100 * (rmse(label, v) / rmse(label, control) - 1), v

    runs = {}
    for path in sorted((ROOT / args.runs).glob("*/design/run.json")):
        record = read_json(path)
        if record["config_sha256"] != sha256_file(ROOT / "configs/adsb_ctx2_design_v1.json"):
            raise ValueError(f"{path} was not fitted under the round-2 declaration.")
        runs.setdefault(record["variant"], {})[record["seed"]] = {"values": october(path.parent / "stop_predictions.parquet", ids),
                                                                   "curve": [round(e["monitor_rmse"], 3) for e in record["history"]],
                                                                   "fit_seconds": record["fit_seconds"], "parameters": record["parameters"],
                                                                   "device": record["inputs"].get("cuda_device"),
                                                                   "predictions_sha256": sha256_file(path.parent / "stop_predictions.parquet")}
    round1 = {read_json(p)["seed"]: october(p.parent / "stop_predictions.parquet", ids) for p in sorted((ROOT / args.round1_runs).glob("ctx_s*/design/run.json"))}

    def mean_of(variant, n):
        seeds = sorted(runs[variant])[:n]
        return np.mean([runs[variant][s]["values"] for s in seeds], axis=0)

    ref = "g_ctx"
    report_variants = {}
    for variant, seeds in runs.items():
        n = len(seeds)
        values = mean_of(variant, n)
        entry = {"seeds": sorted(seeds), "runs": {str(s): {"october_rmse": net(r["values"]), "curve_last": r["curve"][-3:], "fit_seconds": r["fit_seconds"],
                                                           "parameters": r["parameters"], "device": r["device"]} for s, r in seeds.items()},
                 "mean_rmse": net(values), "r2_pct": {str(b): r2(values, b)[0] for b in B_VALUES}}
        if ref in runs and variant != ref:
            m = min(n, len(runs[ref]))
            mine, theirs = mean_of(variant, m), mean_of(ref, m)
            entry["vs_g_ctx_equal_seeds"] = {"seeds": m, "rmse": net(mine), "g_ctx_rmse": net(theirs), "r2_b1_pct": r2(mine, 1.0)[0], "g_ctx_r2_b1_pct": r2(theirs, 1.0)[0]}
            entry["vs_g_ctx_equal_seeds"]["r2_gain_points"] = entry["vs_g_ctx_equal_seeds"]["g_ctx_r2_b1_pct"] - entry["vs_g_ctx_equal_seeds"]["r2_b1_pct"]
            entry["qualifies"] = bool(entry["vs_g_ctx_equal_seeds"]["r2_gain_points"] >= 0.10 and net(mine) < net(theirs))
        best_b = min(B_VALUES, key=lambda b: entry["r2_pct"][str(b)])
        _, v = r2(values, best_b)
        entry["best_b"], entry["best_interval_s"] = best_b, day_interval(days, label, v, control)
        entry["best_matched_pct"] = 100 * (rmse(label[matched & non_lirf], v[matched & non_lirf]) / rmse(label[matched & non_lirf], control[matched & non_lirf]) - 1)
        entry["best_unmatched_pct"] = 100 * (rmse(label[~matched & non_lirf], v[~matched & non_lirf]) / rmse(label[~matched & non_lirf], control[~matched & non_lirf]) - 1)
        entry["best_airports_pct"] = {a: 100 * (rmse(label[airport == a], v[airport == a]) / rmse(label[airport == a], control[airport == a]) - 1) for a in sorted(set(airport))}
        report_variants[variant] = entry
    hardware = {}
    if ref in runs:
        for n in range(1, len(runs[ref]) + 1):
            if n <= len(round1):
                cpu = np.mean([round1[s] for s in sorted(round1)[:n]], axis=0)
                hardware[str(n)] = {"gpu_g_ctx": net(mean_of(ref, n)), "cpu_round1_ctx": net(cpu)}
    qualifying = [v for v, e in report_variants.items() if e.get("qualifies")]
    chosen = min(qualifying, key=lambda v: report_variants[v]["vs_g_ctx_equal_seeds"]["r2_b1_pct"]) if qualifying else None
    report = {"created_at_utc": utc_now(), "declaration_sha256": sha256_file(ROOT / "configs/adsb_ctx2_design_v1.json"), "pool32_rmse": net(pool),
              "variants": report_variants, "hardware_check": hardware, "qualifying": qualifying, "chosen": chosen,
              "decision": decl["protocol"]["decision"], "audit_used": False, "note": "October only; no scoring month read."}
    write_json(ROOT / args.report, report)
    print(f"pool32 {report['pool32_rmse']:.3f}; hardware check (GPU g_ctx vs CPU round-1 ctx):", {k: (round(v["gpu_g_ctx"], 3), round(v["cpu_round1_ctx"], 3)) for k, v in hardware.items()})
    for v, e in report_variants.items():
        cmp = e.get("vs_g_ctx_equal_seeds")
        print(f"{v:12s} seeds {len(e['seeds'])} mean {e['mean_rmse']:.3f} R2 b1 {e['r2_pct']['1.0']:+.4f}% "
              + (f"| vs g_ctx at {cmp['seeds']}: {cmp['rmse']:.3f} vs {cmp['g_ctx_rmse']:.3f}, R2 gain {cmp['r2_gain_points']:+.4f} pts qualifies={e['qualifies']}" if cmp else "")
              + f" | fit {np.mean([r['fit_seconds'] for r in e['runs'].values()]):.0f}s")
    print("chosen:", chosen)


if __name__ == "__main__":
    main()
