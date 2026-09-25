"""Design readings for the context network (configs/adsb_ctx_design_v1.json protocol): October only.

R1  October RMSE (non-LIRF monitored rows) of every context run and of seed means, against the plain network's design runs
    (runs/adsb_nn/design/base_s*) at equal seed counts.
R2  Incremental over the incumbent network pool on the recipe proxy (tools/nn_design_report.RecipeProxy, all October rows):
    table(b) = 0.5 x committee + 0.5 x [(1 - b) x nnb32 + b x ctx_mean], b in {0.25, 0.5, 0.75, 1.0}, against b = 0; with the
    non-LIRF ratio, matched / unmatched and per-airport changes and the paired-day 95% interval at the best b.
R3  Ablations (one seed each) against ctx seed 1: standalone RMSE and the single-seed R2 reading at b = 0.5.
No scoring month is read. Output: a JSON report and a printed summary.
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
from prc2026.features import ID, TARGET, TIME  # noqa: E402
from nn_design_report import RecipeProxy, day_interval, rmse  # noqa: E402

B_VALUES = [0.25, 0.5, 0.75, 1.0]


def october(path: Path, ids: pd.Series) -> np.ndarray:
    values = pd.read_parquet(path).set_index(ID)["adsb_full"].reindex(ids).to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{path} does not cover the October rows.")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/adsb_ctx/design"))
    parser.add_argument("--base-runs", type=Path, default=Path("runs/adsb_nn/design"))
    parser.add_argument("--pool", type=Path, default=Path("runs/adsb_nn"), help="nnb_s101..s132/seasonal/stop_predictions.parquet")
    parser.add_argument("--committee", type=Path, default=Path("runs/adsb_nn/design/committee_october.parquet"))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    ref = pd.read_parquet(ROOT / args.committee)
    proxy = RecipeProxy(ref)
    ids = proxy.ids
    table = proxy.table
    label, airport = proxy.label, table["airport"].to_numpy()
    non_lirf = proxy.non_lirf
    matched = table["adsb_matched"].eq(1).to_numpy()
    days = table[TIME].dt.strftime("%Y-%m-%d").to_numpy()
    committee = proxy.committee

    pool_paths = sorted((ROOT / args.pool).glob("nnb_s1[0-3][0-9]/seasonal/stop_predictions.parquet"))
    pool_paths = [p for p in pool_paths if 101 <= int(p.parent.parent.name.split("_s")[1]) <= 132]
    if len(pool_paths) != 32:
        raise ValueError(f"Expected the 32 nnb pool seeds, found {len(pool_paths)}.")
    pool = np.mean([october(p, ids) for p in pool_paths], axis=0)
    control_table = 0.5 * committee + 0.5 * pool
    control = proxy.values(control_table)

    runs = {}
    for path in sorted((ROOT / args.runs).glob("*/design/run.json")):
        record = read_json(path)
        runs[path.parent.parent.name] = {"variant": record["variant"], "seed": record["seed"], "values": october(path.parent / "stop_predictions.parquet", ids),
                                         "curve": [round(e["monitor_rmse"], 3) for e in record["history"]], "fit_seconds": record["fit_seconds"],
                                         "parameters": record["parameters"], "predictions_sha256": sha256_file(path.parent / "stop_predictions.parquet")}
    base = {p.parent.parent.name: october(p.parent / "stop_predictions.parquet", ids) for p in sorted((ROOT / args.base_runs).glob("base_s*/design/run.json"))}

    def net_rmse(values):
        return rmse(label[non_lirf], values[non_lirf])

    def incremental(ctx_values, b) -> tuple[float, np.ndarray]:
        values = proxy.values(0.5 * committee + 0.5 * ((1 - b) * pool + b * ctx_values))
        return rmse(label, values) / rmse(label, control), values

    ctx_names = sorted((n for n, r in runs.items() if r["variant"] == "ctx"), key=lambda n: runs[n]["seed"])
    base_names = sorted(base, key=lambda n: int(n.split("_s")[1]))
    r1 = {"runs": {n: {"variant": r["variant"], "seed": r["seed"], "october_rmse": net_rmse(r["values"]), "curve": r["curve"], "fit_seconds": r["fit_seconds"],
                       "parameters": r["parameters"]} for n, r in runs.items()},
          "base_runs": {n: net_rmse(v) for n, v in base.items()}, "seed_means": {}}
    for n in range(1, len(ctx_names) + 1):
        entry = {"ctx": net_rmse(np.mean([runs[c]["values"] for c in ctx_names[:n]], axis=0))}
        if n <= len(base_names):
            entry["base"] = net_rmse(np.mean([base[b] for b in base_names[:n]], axis=0))
            entry["ctx_minus_base_pct"] = 100 * (entry["ctx"] / entry["base"] - 1)
        r1["seed_means"][str(n)] = entry
    r1["pool32_rmse"] = net_rmse(pool)

    r2 = {"control_rmse": rmse(label, control), "rows": int(len(label)), "ctx_seeds": len(ctx_names), "by_b": {}}
    if ctx_names:
        ctx_mean = np.mean([runs[c]["values"] for c in ctx_names], axis=0)
        results = {b: incremental(ctx_mean, b) for b in B_VALUES}
        for b, (ratio, values) in results.items():
            r2["by_b"][str(b)] = {"ratio_all": ratio, "pct_all": 100 * (ratio - 1),
                                  "pct_non_lirf": 100 * (rmse(label[non_lirf], values[non_lirf]) / rmse(label[non_lirf], control[non_lirf]) - 1)}
        best = min(B_VALUES, key=lambda b: results[b][0])
        values = results[best][1]
        r2["best_b"] = best
        r2["best_pct_all"] = 100 * (results[best][0] - 1)
        r2["best_day_interval_seconds"] = day_interval(days, label, values, control)
        r2["best_matched_pct"] = 100 * (rmse(label[matched & non_lirf], values[matched & non_lirf]) / rmse(label[matched & non_lirf], control[matched & non_lirf]) - 1)
        r2["best_unmatched_pct"] = 100 * (rmse(label[~matched & non_lirf], values[~matched & non_lirf]) / rmse(label[~matched & non_lirf], control[~matched & non_lirf]) - 1)
        r2["best_airports_pct"] = {a: 100 * (rmse(label[airport == a], values[airport == a]) / rmse(label[airport == a], control[airport == a]) - 1) for a in sorted(set(airport))}
        r2["seed_scaling_pct_at_best_b"] = {str(n): 100 * (incremental(np.mean([runs[c]["values"] for c in ctx_names[:n]], axis=0), best)[0] - 1)
                                            for n in range(1, len(ctx_names) + 1)}
        r2["go"] = bool(r2["best_pct_all"] <= -0.10 and r2["best_day_interval_seconds"][1] < 0)

    r3 = {}
    seed1 = next((n for n in ctx_names if runs[n]["seed"] == 1), None)
    if seed1:
        r3[seed1] = {"october_rmse": net_rmse(runs[seed1]["values"]), "pct_all_b050": 100 * (incremental(runs[seed1]["values"], 0.5)[0] - 1)}
    for name, r in runs.items():
        if r["variant"] != "ctx":
            r3[name] = {"october_rmse": net_rmse(r["values"]), "pct_all_b050": 100 * (incremental(r["values"], 0.5)[0] - 1)}

    report = {"created_at_utc": utc_now(), "config_sha256": sha256_file(ROOT / "configs/adsb_ctx_design_v1.json"), "committee_sha256": sha256_file(ROOT / args.committee),
              "pool_seeds": [p.parent.parent.name for p in pool_paths], "recipe_proxy_inputs": proxy.inputs, "R1": r1, "R2": r2, "R3": r3,
              "predictions_sha256": {n: r["predictions_sha256"] for n, r in runs.items()}, "audit_used": False,
              "note": "October only (the committee's early-stopping month); no scoring month read."}
    write_json(ROOT / args.report, report)
    print(f"R1 pool32 {r1['pool32_rmse']:.3f}; seed means (ctx vs base):", {k: (round(v["ctx"], 3), round(v.get("base", float("nan")), 3)) for k, v in r1["seed_means"].items()})
    for n, v in r1["runs"].items():
        print(f"  {n:16s} {v['october_rmse']:.3f}  last {v['curve'][-3:]}  {v['fit_seconds']:.0f}s")
    if r2["by_b"]:
        print("R2 recipe proxy % by b:", {b: round(v["pct_all"], 4) for b, v in r2["by_b"].items()}, "best", r2["best_b"], "interval", [round(x, 3) for x in r2["best_day_interval_seconds"]],
              "matched", round(r2["best_matched_pct"], 3), "unmatched", round(r2["best_unmatched_pct"], 3), "GO" if r2["go"] else "no-go")
        print("   airports:", {a: round(v, 3) for a, v in r2["best_airports_pct"].items()}, "seed scaling:", {k: round(v, 4) for k, v in r2["seed_scaling_pct_at_best_b"].items()})
    print("R3:", {n: {k: round(v, 4) for k, v in e.items()} for n, e in r3.items()})


if __name__ == "__main__":
    main()
