"""Descriptive ceiling for any airport-time covariate (ERA5 / open weather) on v30's development residuals (gap plan step 3).

Reads runs/error_map/dev_rows.parquet (v30 rebuilt per row on 2025-01/07/11/12 by tools/v30_error_map.py) and the frozen METAR weather
columns of the timeline bundle (December from the December build). Nothing is fitted, selected or downloaded.

1. Group-bias ceiling: the squared error a perfect per-group bias correction would remove, noise-adjusted
   (sum over groups of n * mean^2 - within-group variance, since E[n * mean^2] = n * mu^2 + sigma^2), for airport x day,
   airport x day x 3-hour block and airport x day x hour. Any covariate that varies only by airport and time, weather included,
   cannot remove more than this.
2. Weather-state bias: v30's mean residual in METAR states (snow, snow in the last 3 h, freezing weather, frost, thunder, fog, ...),
   where ERA5 would add intensity or snow on the ground. A state whose bias removal is worth ~0 leaves ERA5 only within-state variance.
Ordinary labels (<= 3 h) as in the error map; all labels also reported for the ceiling.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402

DEV_ROWS = "runs/error_map/dev_rows.parquet"
MONTH_FILES = ["training_2025-01-01_2025-02-01.parquet", "training_2025-07-01_2025-08-01.parquet", "training_2025-11-01_2025-12-01.parquet"]
DECEMBER = "data/december/d198dea10625b0a75977/features/training_2025-12-01_2026-01-01.parquet"
WX = ["wx_snow", "wx_mean3h_snow", "wx_freezing_weather", "wx_temperature_c", "wx_thunder", "wx_mean3h_thunder", "wx_fog", "wx_rain", "wx_heavy_weather"]
ORDINARY = 10800


def group_ceiling(d: pd.DataFrame, keys: list[str], total: float) -> dict:
    g = d.groupby(keys)["r"]
    n, m, v = g.size(), g.mean(), g.var(ddof=1).fillna(0.0)
    raw, noise = float((n * m ** 2).sum()), float(v.sum())
    return {"groups": int(len(n)), "in_sample_share": raw / total, "noise_adjusted_share": (raw - noise) / total}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("docs/weather_ceiling/weather_ceiling.json"))
    args = parser.parse_args()
    bundle = ROOT / read_json(ROOT / "data/timeline/latest.json")["bundle"] / "features"
    feats = pd.concat([pd.read_parquet(bundle / f, columns=["MVT_ID_mvt", *WX]) for f in MONTH_FILES]
                      + [pd.read_parquet(ROOT / DECEMBER, columns=["MVT_ID_mvt", *WX])], ignore_index=True).drop_duplicates("MVT_ID_mvt")
    rows = pd.read_parquet(ROOT / DEV_ROWS)
    rows["r"] = rows["label"] - rows["v30"]
    rows["block"] = rows["hour"] // 3
    report = {"created_at_utc": utc_now(), "inputs": {DEV_ROWS: sha256_file(ROOT / DEV_ROWS), DECEMBER: sha256_file(ROOT / DECEMBER),
                                                      **{f: sha256_file(bundle / f) for f in MONTH_FILES}},
              "months": sorted(rows["month"].unique().tolist()), "ceiling": {}, "weather_states": {}}
    for name, sub in [("all_labels", rows), ("ordinary_labels", rows[rows["label"] <= ORDINARY])]:
        total = float((sub["r"] ** 2).sum())
        report["ceiling"][name] = {"rows": int(len(sub)), "rmse": float(np.sqrt(total / len(sub))),
                                   **{"x".join(k): group_ceiling(sub, k, total) for k in [["airport", "day"], ["airport", "day", "block"], ["airport", "day", "hour"]]},
                                   "by_month_airport_day": {mo: group_ceiling(s, ["airport", "day"], float((s["r"] ** 2).sum()))["noise_adjusted_share"] for mo, s in sub.groupby("month")}}
    d = rows[rows["label"] <= ORDINARY].merge(feats, on="MVT_ID_mvt", how="left")
    total = float((d["r"] ** 2).sum())
    report["feature_coverage"] = float(d["wx_snow"].notna().mean())
    states = {"snow_now": d.wx_snow > 0, "snow_last_3h_not_now": (d.wx_mean3h_snow > 0) & (d.wx_snow == 0),
              "freezing_weather": d.wx_freezing_weather > 0, "temperature_le_0c": d.wx_temperature_c <= 0,
              "frost_no_snow_3h": (d.wx_temperature_c <= 0) & (d.wx_snow == 0) & (d.wx_mean3h_snow == 0),
              "thunder_now": d.wx_thunder > 0, "thunder_last_3h": d.wx_mean3h_thunder > 0, "fog": d.wx_fog > 0,
              "heavy_weather": d.wx_heavy_weather > 0, "rain": d.wx_rain > 0}
    for name, mask in states.items():
        s = d.loc[mask.fillna(False), "r"]
        report["weather_states"][name] = {"rows": int(len(s)), "mean_residual_s": float(s.mean()), "se_s": float(s.std() / np.sqrt(len(s))),
                                          "rmse_s": float(np.sqrt((s ** 2).mean())), "sse_share": float((s ** 2).sum() / total),
                                          "bias_removal_share": float(len(s) * s.mean() ** 2 / total)}
    winter = d[d["month"].isin(["2025-01", "2025-11", "2025-12"])]
    report["winter_cold_vs_warm_by_airport"] = {ap: {"cold_rows": int((s.wx_temperature_c <= 1).sum()), "cold_mean_residual_s": float(s.loc[s.wx_temperature_c <= 1, "r"].mean()),
                                                     "warm_mean_residual_s": float(s.loc[s.wx_temperature_c > 1, "r"].mean()),
                                                     "cold_bias_removal_share": float((s.wx_temperature_c <= 1).sum() * s.loc[s.wx_temperature_c <= 1, "r"].mean() ** 2 / total)}
                                                for ap, s in winter.groupby("airport")}
    out = args.output if args.output.is_absolute() else ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, report)
    c = report["ceiling"]["ordinary_labels"]
    print({"airport_day": round(c["airportxday"]["noise_adjusted_share"] * 100, 2), "airport_day_block": round(c["airportxdayxblock"]["noise_adjusted_share"] * 100, 2),
           "airport_day_hour": round(c["airportxdayxhour"]["noise_adjusted_share"] * 100, 2),
           "max_state_bias_removal_pct": round(max(v["bias_removal_share"] for v in report["weather_states"].values()) * 100, 4)})


if __name__ == "__main__":
    main()
