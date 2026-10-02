"""What EUROCONTROL's published daily pre-departure delay is, exactly: DLY_ALL_PRE_2 against the supplied 2025 clocks (labels' clocks of
the training months only; nothing from the scored months except their published counts and the ranking file's own columns).

Per airport and UTC off-block day, for the flight sets {all departures, IFR departures (FLIGHT_RULE_mvt == "I")} and the hinge
conventions {exact seconds, per-flight minute rounding}: the published total minus our total (median, robust sd, days within 1 s / 60 s,
share of days off by more than 600 s), the count difference dn = FLT_DEP_IFR_2 - our rows, and, where counts differ, the implied delay per
missing flight. The ranking side: per airport-day, the ranking file's departure count (IFR-only at EHAM / LSZH / LEBL) against the
published count, with the off-block day taken from a base file's predictions.

  --base runs/lirf_dark/export_dry/vigorous-whistle_v54.parquet     -> docs/scout/published_definition.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET  # noqa: E402
from prc2026_scout.published_figure import IFR_ONLY, robust_sd  # noqa: E402

AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
TABLE = ROOT / "data/external/eurocontrol_aiu/All_Pre-Departure_Delay.parquet"
WORKBOOK = ROOT / "data/external/eurocontrol_aiu/All_Pre-Departure_Delay.xlsx"
OUT = ROOT / "docs/scout/published_definition.json"


def published() -> tuple[pd.Series, pd.Series]:
    p = pd.read_parquet(TABLE)
    p["day"] = pd.to_datetime(p["FLT_DATE"]).dt.strftime("%Y-%m-%d")
    p = p.set_index(["APT_ICAO", "day"])
    return (p["DLY_ALL_PRE_2"].astype(float) * 60).rename("pub"), p["FLT_DEP_IFR_2"].astype(float).rename("n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="runs/lirf_dark/export_dry/vigorous-whistle_v54.parquet")
    args = parser.parse_args()
    P, N = published()
    files = sorted((ROOT / "data/raw").glob("training_2025-*.parquet"))
    cols = [ID, "ADEP_mvt", "PHASE_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "FLIGHT_RULE_mvt"]
    raw = pd.concat([pd.read_parquet(p, columns=cols) for p in files], ignore_index=True)
    d = raw[raw["PHASE_mvt"].eq("DEP")].drop_duplicates(ID).copy()
    d["day"] = d["BLOCK_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d")
    delay = (d["BLOCK_TIME_UTC_mvt"] - d["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    d["exact"], d["round"] = delay.clip(lower=0), (np.round(delay / 60) * 60).clip(lower=0)
    d["ifr"] = d["FLIGHT_RULE_mvt"].eq("I")
    out = {"created_at_utc": utc_now(), "workbook_sha256": sha256_file(WORKBOOK), "training_files": {p.name: sha256_file(p) for p in files},
           "official_scores_used": "none", "labels_used": "the supplied 2025 training clocks (block and schedule) only", "conventions": {}, "missing_flights": {}, "ranking_counts": {}}
    for a in AIRPORTS:
        x = d[d["ADEP_mvt"].eq(a)]
        rows = {}
        for subset in ["all", "ifr"]:
            y = x if subset == "all" else x[x["ifr"]]
            for kind in ["exact", "round"]:
                g = y.groupby("day")
                j = pd.concat([g[kind].sum().rename("own"), g.size().rename("cnt"), P.xs(a, level=0), N.xs(a, level=0)], axis=1).dropna()
                j = j[(j.index >= "2025-01-01") & (j.index < "2026-01-01")]
                diff, eq = j["pub"] - j["own"], j["cnt"] == j["n"]
                rows[f"{subset}|{kind}"] = {"days": int(len(j)), "median_s": float(diff.median()), "robust_sd_s": robust_sd(diff.to_numpy()), "days_within_1s": int((diff.abs() < 1).sum()),
                                            "days_within_60s": int((diff.abs() <= 60).sum()), "share_days_off_over_600s": float((diff.abs() > 600).mean()),
                                            "days_counts_equal": int(eq.sum()), "days_counts_equal_within_60s": int((diff[eq].abs() <= 60).sum()),
                                            "days_published_count_higher": int((j["n"] > j["cnt"]).sum()), "days_published_count_lower": int((j["n"] < j["cnt"]).sum())}
        out["conventions"][a] = rows
        y = x[x["ifr"]] if a in IFR_ONLY else x
        g = y.groupby("day")
        kind = "round"
        j = pd.concat([g[kind].sum().rename("own"), g.size().rename("cnt"), P.xs(a, level=0), N.xs(a, level=0)], axis=1).dropna()
        j = j[(j.index >= "2025-01-01") & (j.index < "2026-01-01")]
        j["diff"], j["dn"] = j["pub"] - j["own"], j["n"] - j["cnt"]
        out["missing_flights"][a] = {"flight_set": "IFR" if a in IFR_ONLY else "all", "hinge": kind, "dn_days": {str(int(k)): int(v) for k, v in j["dn"].value_counts().sort_index().items()},
                                     "by_dn": {str(int(k)): {"days": int(len(s)), "median_s": float(s.median()), "mean_s": float(s.mean()), "q10_s": float(s.quantile(0.1)), "q90_s": float(s.quantile(0.9))}
                                               for k, s in j.groupby("dn")["diff"] if abs(k) <= 4}}
    base = pd.read_parquet(ROOT / args.base).set_index(ID)[TARGET]
    rk = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=[ID, "ADEP_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "FLIGHT_RULE_mvt"])
    rk = rk[rk["PHASE_mvt"].eq("DEP")].drop_duplicates(ID)
    rk["bday"] = (rk["MVT_TIME_UTC_mvt"] - pd.to_timedelta(base.reindex(rk[ID]).to_numpy(dtype=float), unit="s")).dt.strftime("%Y-%m-%d")
    out["ranking_counts"]["base"] = {"path": args.base, "sha256": sha256_file(ROOT / args.base)}
    out["ranking_counts"]["flight_rules"] = {a: {k: int(v) for k, v in rk.loc[rk["ADEP_mvt"].eq(a), "FLIGHT_RULE_mvt"].fillna("NA").value_counts().items()} for a in AIRPORTS}
    for a in AIRPORTS:
        for subset in ["all", "ifr"]:
            z = rk[rk["ADEP_mvt"].eq(a)]
            z = z if subset == "all" else z[z["FLIGHT_RULE_mvt"].eq("I")]
            j = pd.concat([z.groupby("bday").size().rename("cnt"), N.xs(a, level=0)], axis=1).dropna()
            j = j[j.index.str[:7].isin(["2026-01", "2026-07"])]
            dn = j["n"] - j["cnt"]
            out["ranking_counts"].setdefault(a, {})[subset] = {"days": int(len(j)), "dn_days": {str(int(k)): int(v) for k, v in dn.value_counts().sort_index().items()}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_json(OUT, out)
    for a in AIRPORTS:
        c = out["conventions"][a]
        best = min(c, key=lambda k: (c[k]["robust_sd_s"], -c[k]["days_within_60s"]))
        print(a, "best convention", best, {k: round(v, 1) if isinstance(v, float) else v for k, v in c[best].items()})
    print("->", OUT.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
