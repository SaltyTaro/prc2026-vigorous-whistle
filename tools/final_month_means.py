"""Label-free plausibility check of a final-file export's new months against EUROCONTROL's published monthly taxi-out times.

For every airport-month the workbook (Taxi-Out_Additional_Time.xlsx, sheet DATA) gives the mean taxi-out of its reference flights,
(TOTAL_REF_TIME_MIN + TOTAL_ADD_TIME_MIN) / TOTAL_REF_NB_FL. The indicator filters flights, so it is not the mean of our target; the
check therefore calibrates on 2025: for each airport, the gap "mean of the 2025 labels (clipped at CLIP s) - published mean" over the
twelve 2025 months gives a range. For each 2026 month of the export, the gap "mean of our predictions (same clip) - published mean" is
reported and flagged when it lies more than MARGIN s outside that airport's 2025 range. A flag is a prompt to look, not a verdict.
The published figures are read only here, as a check; nothing is fitted or changed.

  python tools/final_month_means.py --final F [--raw data/raw/final_ranking.parquet] [--out OUT.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import sha256_file, utc_now, write_json  # noqa: E402

ID, TARGET = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
WORKBOOK = ROOT / "data/external/eurocontrol_aiu/Taxi-Out_Additional_Time.xlsx"
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
CLIP, MARGIN = 3600.0, 30.0


def published() -> pd.Series:
    d = pd.read_excel(WORKBOOK, sheet_name="DATA")
    d = d[d["APT_ICAO"].isin(AIRPORTS)].copy()
    d["month"] = d["YEAR"].astype(int).astype(str) + "-" + d["MONTH_NUM"].astype(int).map("{:02d}".format)
    d["pub_s"] = 60.0 * (d["TOTAL_REF_TIME_MIN"] + d["TOTAL_ADD_TIME_MIN"]) / d["TOTAL_REF_NB_FL"]
    return d.set_index(["APT_ICAO", "month"])["pub_s"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", required=True)
    ap.add_argument("--raw", default="data/raw/final_ranking.parquet")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    pub = published()
    cols = [ID, "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"]
    lab = pd.concat([pd.read_parquet(f, columns=cols + [TARGET]) for f in sorted(glob.glob(str(ROOT / "data/raw/training_2025-*.parquet")))])
    lab = lab[(lab["PHASE_mvt"] == "DEP") & lab["ADEP_mvt"].isin(AIRPORTS)]
    lab["month"] = lab["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m")
    truth = lab.assign(v=lab[TARGET].clip(upper=CLIP)).groupby(["ADEP_mvt", "month"])["v"].mean()
    gap25 = (truth - pub.reindex(truth.index)).rename("gap").reset_index()
    rng = gap25.groupby("ADEP_mvt")["gap"].agg(["min", "max", "mean"])
    raw = pd.read_parquet(ROOT / a.raw, columns=cols).query("PHASE_mvt == 'DEP'")
    raw["month"] = raw["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m")
    pred = pd.read_parquet(ROOT / a.final).merge(raw[[ID, "ADEP_mvt", "month"]], on=ID, how="left", validate="one_to_one")
    ours = pred.assign(v=pred[TARGET].astype(float).clip(upper=CLIP)).groupby(["ADEP_mvt", "month"])["v"].mean()
    rows = []
    for (apt, month), v in ours.items():
        p = pub.get((apt, month), np.nan)
        g = v - p
        lo, hi = rng.loc[apt, "min"], rng.loc[apt, "max"]
        rows.append({"airport": apt, "month": month, "ours_clipped_mean_s": round(float(v), 1), "published_mean_s": round(float(p), 1),
                     "gap_s": round(float(g), 1), "gap_2025_min_s": round(float(lo), 1), "gap_2025_max_s": round(float(hi), 1),
                     "gap_2025_mean_s": round(float(rng.loc[apt, "mean"]), 1),
                     "flag": bool(np.isfinite(g) and (g < lo - MARGIN or g > hi + MARGIN))})
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    print("flags:", int(t["flag"].sum()), "of", len(t))
    if a.out:
        write_json(ROOT / a.out, {"created_at_utc": utc_now(), "final": a.final, "final_sha256": sha256_file(ROOT / a.final),
                                  "workbook_sha256": sha256_file(WORKBOOK), "clip_s": CLIP, "margin_s": MARGIN, "rows": rows})


if __name__ == "__main__":
    main()
