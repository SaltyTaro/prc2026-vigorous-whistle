"""Snow-day scan (28 September 2026). Label-free on the ranking; labels only on the 2025 months. Nothing is fitted or selected.

A snow airport-day has at least 8 METAR reports with SN in the present-weather codes, or at least 4 with FZ (IEM archive,
data/external/iem/<airport>_{train,ranking_january,ranking_july}.csv). Per airport-day it compares the day's lift over the
airport-month's median non-snow day (label lift against prediction lift) for
  - the 12-month cross-fit v9 (scratch/v24, 4 folds of 3 months) on every 2025 month, and
  - the full v33 recipe on the fair development rows (runs/fair_dev/rows.parquet: 2025-01/07/11 folds, December refit leaves),
and lists the ranking's snow airport-days with v35's lift and untracked-row counts.

  --report docs/disagreement/snow_days.json
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET  # noqa: E402

AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
SN_REPORTS, FZ_REPORTS = 8, 4
V35 = "runs/submissions/vigorous-whistle_v35.parquet"
RANKING = "runs/adsb_ctx/export_v30/ranking_values.parquet"


def weather_days() -> pd.DataFrame:
    parts = []
    for airport in AIRPORTS:
        w = pd.concat([pd.read_csv(p, usecols=["valid", "wxcodes"], low_memory=False)
                       for p in sorted((ROOT / "data/external/iem").glob(f"{airport}_*.csv")) if "probe" not in p.name]).drop_duplicates("valid")
        codes = w["wxcodes"].fillna("").astype(str)
        day = pd.to_datetime(w["valid"]).dt.strftime("%Y-%m-%d")
        parts.append(pd.DataFrame({"airport": airport, "day": day, "sn": codes.str.contains("SN"), "fz": codes.str.contains("FZ")})
                     .groupby(["airport", "day"]).agg(reports=("sn", "size"), sn=("sn", "sum"), fz=("fz", "sum")).reset_index())
    out = pd.concat(parts, ignore_index=True)
    out["snow"] = (out["sn"] >= SN_REPORTS) | (out["fz"] >= FZ_REPORTS)
    return out


def day_table(rows: pd.DataFrame, pred: str, weather: pd.DataFrame) -> pd.DataFrame:
    rows = rows.merge(weather[["airport", "day", "snow"]], on=["airport", "day"], how="left").assign(snow=lambda f: f["snow"].fillna(False).astype(bool))
    days = rows.groupby(["airport", "month", "day", "snow"]).agg(n=("label", "size"), label=("label", "mean"), pred=(pred, "mean")).reset_index()
    normal = days[~days["snow"]].groupby(["airport", "month"])[["label", "pred"]].median().rename(columns=lambda c: c + "_normal")
    days = days.join(normal, on=["airport", "month"])
    return days.assign(lift_label=days["label"] - days["label_normal"], lift_pred=days["pred"] - days["pred_normal"], residual=days["label"] - days["pred"])


def summary(days: pd.DataFrame) -> dict:
    s = days[days["snow"]]
    w = s["n"].to_numpy(float)
    out = {"snow_days": int(len(s)), "snow_rows": int(w.sum()),
           "beta_label_lift_on_pred_lift": float(np.sum(w * s["lift_pred"] * s["lift_label"]) / np.sum(w * s["lift_pred"] ** 2)),
           "mean_residual_snow_days": float(np.average(s["residual"], weights=w)),
           "mean_residual_other_days": float(np.average(days.loc[~days["snow"], "residual"], weights=days.loc[~days["snow"], "n"])),
           "by_airport": {a: {"days": int(len(g)), "mean_residual": float(np.average(g["residual"], weights=g["n"])),
                              "max_day_residual": float(g["residual"].max())} for a, g in s.groupby("airport")},
           "days": s.sort_values("lift_label", ascending=False)[["airport", "day", "n", "label", "pred", "lift_label", "lift_pred", "residual"]]
                    .round(1).to_dict(orient="records")}
    return out


def main() -> None:
    weather = weather_days()
    cross = []
    for m in [f"2025-{i:02d}" for i in range(1, 13)]:
        f = pd.read_parquet(ROOT / "scratch/v24" / f"{m}.parquet", columns=[ID, "airport", "v9", TARGET, "MVT_TIME_UTC_mvt"])
        cross.append(f.assign(month=m, day=pd.to_datetime(f["MVT_TIME_UTC_mvt"]).dt.strftime("%Y-%m-%d")).rename(columns={TARGET: "label"}))
    cross = pd.concat(cross, ignore_index=True)
    fair = pd.read_parquet(ROOT / "runs/fair_dev/rows.parquet", columns=[ID, "month", "day", "airport", "matched", "label", "v33"])
    fair_days = day_table(fair, "v33", weather)
    fair_snow = fair.merge(weather[["airport", "day", "snow"]], on=["airport", "day"], how="left")
    fair_snow = fair_snow[fair_snow["snow"].fillna(False).astype(bool)]
    r = pd.read_parquet(ROOT / RANKING, columns=[ID, "airport", "month", "scope"])
    times = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=[ID, "MVT_TIME_UTC_mvt"]).drop_duplicates(ID).set_index(ID)["MVT_TIME_UTC_mvt"]
    r["day"] = pd.to_datetime(times.reindex(r[ID])).dt.strftime("%Y-%m-%d").to_numpy()
    r["v35"] = pd.read_parquet(ROOT / V35).set_index(ID)[TARGET].reindex(r[ID]).to_numpy(float)
    r["untracked"] = ~r["scope"].isin(["matched_no_lirf", "lirf_nonpop_matched"])
    typical = r.groupby("airport")["v35"].median()
    rk = r.merge(weather.loc[weather["snow"], ["airport", "day", "sn", "fz"]], on=["airport", "day"])
    ranking_days = rk.groupby(["airport", "day"]).agg(rows=("v35", "size"), untracked=("untracked", "sum"), v35_mean=("v35", "mean"),
                                                    sn=("sn", "first"), fz=("fz", "first")).reset_index()
    ranking_days["v35_lift_over_airport_median"] = ranking_days["v35_mean"] - ranking_days["airport"].map(typical)
    snow = weather[weather["snow"]]
    report = {"created_at_utc": utc_now(), "code": sha256_file(Path(__file__)), "rule": {"sn_reports": SN_REPORTS, "fz_reports": FZ_REPORTS},
              "labels_read": "2025 months only; no ranking label exists",
              "snow_airport_days": {y: snow[snow["day"].str[:4].eq(y)].groupby("airport").size().to_dict() for y in ["2025", "2026"]},
              "cross_fit_v9_12_months": summary(day_table(cross, "v9", weather)),
              "fair_v33_4_months": {**summary(fair_days),
                                    "row_residual_snow_days": {"tracked": float((fair_snow["label"] - fair_snow["v33"])[fair_snow["matched"]].mean()),
                                                               "untracked": float((fair_snow["label"] - fair_snow["v33"])[~fair_snow["matched"]].mean())}},
              "ranking": {"rows": int(ranking_days["rows"].sum()), "untracked": int(ranking_days["untracked"].sum()),
                          "days": ranking_days.round(1).to_dict(orient="records")}}
    out = ROOT / (sys.argv[sys.argv.index("--report") + 1] if "--report" in sys.argv else "docs/disagreement/snow_days.json")
    write_json(out, report)
    for k in ["cross_fit_v9_12_months", "fair_v33_4_months"]:
        print(k, {x: (round(v, 3) if isinstance(v, float) else v) for x, v in report[k].items() if x not in ("days", "by_airport")})
        print("  ", {a: round(v["mean_residual"], 1) for a, v in report[k]["by_airport"].items()})
    print("ranking snow rows", report["ranking"]["rows"], "untracked", report["ranking"]["untracked"], report["snow_airport_days"])


if __name__ == "__main__":
    main()
