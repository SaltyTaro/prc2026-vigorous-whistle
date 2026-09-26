"""Descriptive error budget of v28 (adsb_ctx_v2 pool_s75) on the four recipe months: where the remaining squared error sits.

Rebuilds v28 and v27 per row with the development scorer's code path (tools/run_adsb_ctx_dev2.py merged configuration, the v26 residual
via combine(rnet_w90), the v28 committee mix), checks each month against the recorded development RMSE, and splits the SSE into
mutually exclusive row classes (first match wins):
  one_day            label >= 80,000 s
  long               10,800 s < label < 80,000 s
  lirf_population    LIRF fill population (the v9 mixture's rows)
  lirf_matched       other LIRF rows with an ADS-B match
  lirf_unmatched     other LIRF rows without one
  matched            non-LIRF rows with an ADS-B match
  unmatched          non-LIRF rows without one
per month and pooled: rows, RMSE, bias, SSE share, and the month-RMSE change if the class were predicted perfectly (a ceiling, not a
target). Ordinary non-LIRF rows are also split by airport. Nothing is selected or exported; the report is docs/adsb_ctx/error_budget_v28.json.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import read_json, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET  # noqa: E402
from run_adsb_ctx_dev2 import merged_config  # noqa: E402
from run_adsb_nn_dev import mixed  # noqa: E402
from run_adsb_nn2_dev import lirf_blends, spec_for, weighted  # noqa: E402
from run_adsb_rnn_dev import combine, residual_columns  # noqa: E402
from run_adsb_v11 import candidate_values, rmse  # noqa: E402
from run_adsb_v24 import adsb_features, extra_columns, load_frame, month_table  # noqa: E402

CLASSES = ["one_day", "long", "lirf_population", "lirf_matched", "lirf_unmatched", "matched", "unmatched"]


def classes(table: pd.DataFrame) -> np.ndarray:
    y = table[TARGET].to_numpy(dtype=float)
    lirf = table["airport"].eq("LIRF").to_numpy()
    pop = table["population"].astype(bool).to_numpy()
    matched = table["adsb_matched"].eq(1).to_numpy()
    out = np.where(matched, "matched", "unmatched").astype(object)
    out[lirf & matched] = "lirf_matched"
    out[lirf & ~matched] = "lirf_unmatched"
    out[lirf & pop] = "lirf_population"
    out[(y > 10800) & (y < 80000)] = "long"
    out[y >= 80000] = "one_day"
    return out


def main():
    decl = read_json(ROOT / "configs/adsb_ctx_v2.json")
    config = merged_config(decl)
    v26, v25 = config["candidates"]["rnet_w90"], config["v25_recipe"]
    recorded = {m: e["rmse"] for m, e in read_json(ROOT / "docs/adsb_ctx/development_results_v2.json")["candidates"]["pool_s75"]["months"].items()}
    frames = []
    for month in decl["recipe_months"]:
        table = month_table(load_frame(config, month, "cross_fit"), adsb_features(month, ROOT / config["matches_dirs"]["v11"], ROOT / "runs/adsb_v24/features"), month)
        ids = table[ID]
        v9 = table["v9"].to_numpy(dtype=float)
        residual = combine(v26, residual_columns(config, month, ids, smoke=False), "rnet_w90")
        floor = float(table["floor"].iloc[0])
        for name, values in extra_columns(config, month, ids.to_numpy()).items():
            table[name] = values
        committee = table[config["committee_table"]].to_numpy(dtype=float)
        values = {}
        for label, network, share in [("v27", {"ens_ctx": 1.0}, 0.65), ("v28", {"ens_ctx": 0.5, "ens_ctx_e24": 0.5}, 0.75)]:
            table[f"mix:{label}"] = mixed(committee, weighted(table, network), float(share))
            values[label], _ = candidate_values(table, v9, residual, spec_for(config, f"mix:{label}", lirf_blends(config, table, v25, "v26")), floor)
        y = table[TARGET].to_numpy(dtype=float)
        if abs(rmse(y, values["v28"]) - recorded[month]) > 1e-9:
            raise ValueError(f"v28 does not reproduce its development RMSE on {month}.")
        frames.append(pd.DataFrame({ID: ids.to_numpy(), "month": month, "airport": table["airport"].to_numpy(), "label": y, "v27": values["v27"], "v28": values["v28"],
                                    "cls": classes(table)}))
    rows = pd.concat(frames, ignore_index=True)

    def budget(frame: pd.DataFrame) -> dict:
        err = frame["v28"] - frame["label"]
        total = float((err ** 2).sum())
        month_rmse = float(np.sqrt(total / len(frame)))
        out = {"rows": int(len(frame)), "rmse": month_rmse, "classes": {}}
        for c in CLASSES:
            k = frame["cls"].eq(c).to_numpy()
            if not k.any():
                continue
            sse = float((err[k] ** 2).sum())
            sse27 = float(((frame["v27"] - frame["label"])[k] ** 2).sum())
            out["classes"][c] = {"rows": int(k.sum()), "rmse": float(np.sqrt(sse / k.sum())), "bias": float(err[k].mean()), "sse_share": sse / total,
                                 "month_rmse_if_perfect_pct": 100 * (np.sqrt((total - sse) / len(frame)) / month_rmse - 1),
                                 "v28_vs_v27_sse_change": sse - sse27}
        return out

    report = {"created_at_utc": utc_now(), "recipe": "v28 = adsb_ctx_v2 pool_s75", "months": {m: budget(f) for m, f in rows.groupby("month")},
              "pooled": budget(rows), "note": "descriptive; a class's 'if perfect' figure is a ceiling, not a target"}
    ordinary = rows[rows["cls"].isin(["matched", "unmatched"])]
    report["ordinary_non_lirf_by_airport"] = {}
    total = float(((rows["v28"] - rows["label"]) ** 2).sum())
    for (a, c), f in ordinary.groupby(["airport", "cls"]):
        sse = float(((f["v28"] - f["label"]) ** 2).sum())
        report["ordinary_non_lirf_by_airport"][f"{a}:{c}"] = {"rows": int(len(f)), "rmse": float(np.sqrt(sse / len(f))), "sse_share_pooled": sse / total}
    write_json(ROOT / "docs/adsb_ctx/error_budget_v28.json", report)
    for scope, b in [*report["months"].items(), ("pooled", report["pooled"])]:
        print(f"{scope:8s} rmse {b['rmse']:.2f} | " + " ".join(f"{c}:{e['sse_share'] * 100:.1f}%({e['rows']})" for c, e in b["classes"].items()))
    print("pooled 'if perfect' month-RMSE change:", {c: round(e["month_rmse_if_perfect_pct"], 2) for c, e in report["pooled"]["classes"].items()})
    top = sorted(report["ordinary_non_lirf_by_airport"].items(), key=lambda kv: -kv[1]["sse_share_pooled"])[:10]
    print("largest ordinary non-LIRF cells:", [(k, round(v["sse_share_pooled"] * 100, 1), round(v["rmse"], 1)) for k, v in top])


if __name__ == "__main__":
    main()
