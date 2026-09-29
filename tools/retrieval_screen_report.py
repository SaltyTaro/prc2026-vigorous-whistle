"""Readings of the retrieval screen (tools/retrieval_screen.py): October only, no scoring month read.

For every October row the screen found, in each of seven networks, the nearest departures of that network's held-out month and
their out-of-sample residuals. A correction c is the mean over the seven held-out months of the neighbours' mean residual
(each residual clipped to +-1800 s first). Corrections read:
  knn_<scope>_<k>    neighbours in the embedding (l2), scope any / same airport, k in 4, 16, 64
  knnc_<scope>_<k>   the same after removing every candidate's airport-month mean residual (what is left is local, not the month's bias)
  knnw_<scope>_<k>   neighbours under the head-weighted metric (l2w), not centred
  bias_airport       the airport-month mean residual alone (no neighbours): the plain calibration a retrieval member must beat
Readings, each for w in W_VALUES:
  own     RMSE of (seven-network mean + w c) against the seven-network mean, non-LIRF October rows
  pool    RMSE of (P + w c) against P, the incumbent context pool (ctx seeds 101-116 and ctx_e24 seeds 117-132, seasonal-fold stop
          predictions), non-LIRF October rows
  proxy   recipe proxy (tools/nn_design_report.RecipeProxy, all October rows) of the table 0.25 committee + 0.75 (P + w c) against w = 0,
          with the paired-day 95% interval
  corr    correlation of c with the October residual of the seven-network mean and of P (residuals clipped to +-1800 s, and raw)
Screen rule, fixed before any reading: GO for a declared retrieval-network design round if the best proxy reading over the grid is
at or below -0.05% with its paired-day interval below zero. The design round would keep its own gate. Otherwise the lever is
recorded as closed by the screen (a trained retrieval network could still differ; the screen measures whether the residuals of
similar departures of other months carry anything at all).
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
from ctx_design_report import october  # noqa: E402
from retrieval_screen import CLIP, FIT_MONTHS, KS, SCOPES  # noqa: E402

W_VALUES = [0.25, 0.5, 1.0]
SHARE = 0.75
BAR_PCT = -0.05


def corrections(directory: Path) -> tuple[pd.DataFrame, dict]:
    """Per October row: every correction of one held-out month's run, and the run's own prediction."""
    record = read_json(directory / "run.json")
    if record["smoke"]:
        raise ValueError(f"{directory} is a smoke run.")
    queries = pd.read_parquet(directory / "queries.parquet")
    if sha256_file(directory / "queries.parquet") != record["outputs"]["queries_sha256"]:
        raise ValueError(f"{directory}: the query table changed after the run.")
    candidates = pd.read_parquet(directory / "candidates.parquet")
    out = pd.DataFrame({ID: queries[ID].to_numpy(), "pred": queries["pred"].to_numpy(dtype=float)})
    for scope in SCOPES:
        for k in KS:
            out[f"knn_{scope}_{k}"] = queries[f"resc_l2_{scope}_{k}"].to_numpy(dtype=float)
            out[f"knnw_{scope}_{k}"] = queries[f"resc_l2w_{scope}_{k}"].to_numpy(dtype=float)
    clipped = np.clip(candidates["label"].to_numpy(dtype=float) - candidates["pred"].to_numpy(dtype=float), -CLIP, CLIP)
    bias = pd.Series(clipped).groupby(candidates["airport"].to_numpy()).mean()
    centred = clipped - candidates["airport"].map(bias).to_numpy(dtype=float)
    neighbours = np.load(directory / "neighbours.npz")
    for scope in SCOPES:
        index = neighbours[f"index_{scope}"].astype(np.int64)
        for k in KS:
            plain = clipped[index[:, :k]].mean(axis=1)
            if np.max(np.abs(plain - out[f"knn_{scope}_{k}"].to_numpy())) > 1e-2:
                raise ValueError(f"{directory}: the neighbour file does not reproduce the recorded means ({scope}, {k}).")
            out[f"knnc_{scope}_{k}"] = centred[index[:, :k]].mean(axis=1)
    info = {"held_out": record["held_out"], "october_rmse": record["october_rmse"], "candidate_rmse": record["candidate_rmse"], "fitted_rows": record["fitted_rows"],
            "candidate_rows": record["candidate_rows"], "fit_seconds": record["fit_seconds"], "curve_last": [round(e["monitor_rmse"], 3) for e in record["history"][-3:]],
            "same_airport_share_k16": float(queries["same_l2_any_16"].mean()), "airport_bias_clipped": {a: float(v) for a, v in bias.items()},
            "code_sha256": record["inputs"]["code"]["tools/retrieval_screen.py"], "queries_sha256": record["outputs"]["queries_sha256"]}
    return out, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/retrieval/screen"))
    parser.add_argument("--pool-runs", type=Path, default=Path("runs/adsb_ctx"))
    parser.add_argument("--committee", type=Path, default=Path("runs/adsb_nn/design/committee_october.parquet"))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    reference = pd.read_parquet(ROOT / args.committee)
    proxy = RecipeProxy(reference)
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

    runs, tables = {}, []
    for month in FIT_MONTHS:
        frame, info = corrections(ROOT / args.runs / month)
        runs[month] = info
        tables.append(frame.set_index(ID).reindex(ids))
    names = [c for c in tables[0].columns if c != "pred"]
    covered = np.all([t["pred"].notna().to_numpy() for t in tables], axis=0)
    if not covered[non_lirf].all() or covered[~non_lirf].any():
        raise ValueError("The screen's October rows are not the proxy table's non-LIRF rows.")
    own = np.where(covered, np.mean([t["pred"].to_numpy(dtype=float) for t in tables], axis=0), np.nan)
    stack = {n: np.where(covered, np.mean([t[n].to_numpy(dtype=float) for t in tables], axis=0), 0.0) for n in names}
    bias_by_airport = pd.DataFrame({m: runs[m]["airport_bias_clipped"] for m in FIT_MONTHS}).mean(axis=1)
    stack["bias_airport"] = np.where(covered, pd.Series(airport).map(bias_by_airport).to_numpy(dtype=float), 0.0)

    k = non_lirf

    def clip(v):
        return np.clip(v, -CLIP, CLIP)

    def reading(c: np.ndarray) -> dict:
        out = {"mean": float(c[k].mean()), "sd": float(c[k].std()),
               "corr_own_clipped": float(np.corrcoef(c[k], clip(label[k] - own[k]))[0, 1]), "corr_own_raw": float(np.corrcoef(c[k], label[k] - own[k])[0, 1]),
               "corr_pool_clipped": float(np.corrcoef(c[k], clip(label[k] - pool[k]))[0, 1]), "corr_pool_raw": float(np.corrcoef(c[k], label[k] - pool[k])[0, 1]),
               "corr_proxy_clipped": float(np.corrcoef(c[k], clip(label[k] - control[k]))[0, 1]), "by_w": {}}
        for w in W_VALUES:
            values = proxy.values((1 - SHARE) * committee + SHARE * (pool + w * c))
            entry = {"own_pct": 100 * (rmse(label[k], own[k] + w * c[k]) / rmse(label[k], own[k]) - 1),
                     "pool_pct": 100 * (rmse(label[k], pool[k] + w * c[k]) / rmse(label[k], pool[k]) - 1),
                     "proxy_pct": 100 * (rmse(label, values) / control_rmse - 1),
                     "proxy_interval_s": day_interval(days, label, values, control),
                     "proxy_matched_pct": 100 * (rmse(label[k & matched], values[k & matched]) / rmse(label[k & matched], control[k & matched]) - 1),
                     "proxy_unmatched_pct": 100 * (rmse(label[k & ~matched], values[k & ~matched]) / rmse(label[k & ~matched], control[k & ~matched]) - 1)}
            entry["go"] = bool(entry["proxy_pct"] <= BAR_PCT and entry["proxy_interval_s"][1] < 0)
            out["by_w"][str(w)] = entry
        return out

    readings = {n: reading(c) for n, c in stack.items()}
    grid = [(n, w, e) for n, r in readings.items() if n != "bias_airport" for w, e in r["by_w"].items()]
    best = min(grid, key=lambda g: g[2]["proxy_pct"])
    best_values = proxy.values((1 - SHARE) * committee + SHARE * (pool + float(best[1]) * stack[best[0]]))
    report = {"created_at_utc": utc_now(), "rows_non_lirf": int(k.sum()), "rows_proxy": int(len(label)), "share": SHARE, "bar_pct": BAR_PCT, "clip": CLIP,
              "own_rmse_non_lirf": rmse(label[k], own[k]), "pool_rmse_non_lirf": rmse(label[k], pool[k]), "committee_rmse_non_lirf": rmse(label[k], committee[k]),
              "control_proxy_rmse": control_rmse, "runs": runs, "readings": readings,
              "best": {"correction": best[0], "w": float(best[1]), **best[2],
                       "airports_pct": {a: 100 * (rmse(label[airport == a], best_values[airport == a]) / rmse(label[airport == a], control[airport == a]) - 1)
                                        for a in sorted(set(airport[k]))}},
              "go": bool(any(e["go"] for _, _, e in grid)), "grid_entries": len(grid),
              "inputs": {"committee_sha256": sha256_file(ROOT / args.committee), "recipe_proxy_inputs": proxy.inputs, "tool_sha256": sha256_file(ROOT / "tools/retrieval_screen_report.py")},
              "audit_used": False, "note": "October only; candidates are fitting rows of the seasonal fold (2025-02..06, 08, 09); no scoring month or ranking row read."}
    write_json(ROOT / args.report, report)
    print(f"October non-LIRF RMSE: seven-network mean {report['own_rmse_non_lirf']:.3f}, context pool {report['pool_rmse_non_lirf']:.3f}, committee {report['committee_rmse_non_lirf']:.3f}; "
          f"control proxy {control_rmse:.3f}")
    print("runs:", {m: (round(r["october_rmse"], 2), round(r["candidate_rmse"], 2)) for m, r in runs.items()})
    print(f"{'correction':16s} {'sd':>7s} {'r_own':>7s} {'r_pool':>7s} {'r_prx':>7s} | own% w.25 w.5 w1 | pool% w.25 w.5 w1 | proxy% w.25 w.5 w1")
    for n, r in readings.items():
        by = r["by_w"]
        print(f"{n:16s} {r['sd']:7.2f} {r['corr_own_clipped']:+7.4f} {r['corr_pool_clipped']:+7.4f} {r['corr_proxy_clipped']:+7.4f} | "
              + " ".join(f"{by[str(w)]['own_pct']:+.3f}" for w in W_VALUES) + " | " + " ".join(f"{by[str(w)]['pool_pct']:+.3f}" for w in W_VALUES) + " | "
              + " ".join(f"{by[str(w)]['proxy_pct']:+.4f}" for w in W_VALUES))
    print("best:", best[0], "w", best[1], {key: (round(v, 4) if isinstance(v, float) else v) for key, v in best[2].items()})
    print("GO" if report["go"] else "NO-GO", f"(bar {BAR_PCT}% on the proxy with the interval below zero)")


if __name__ == "__main__":
    main()
