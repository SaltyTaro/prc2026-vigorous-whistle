"""Label-free sanity check of a final-file export against the shipped ranking file of the same chain step.

The final dataset repeats every January / July 2026 departure of ranking.parquet (same MVT_ID_mvt; times jittered by up to 12 s,
docs/final_file/final_dataset.md). A final-file export and the shipped file of the same step (v69 for the early chain's step 10,
v76 for the late chain's last step) should therefore agree on those rows apart from: the jitter, v40's nine LIRF fill pins (off in
the final file), the CPU-predicted members (about 0.06 s RMS against the GPU tables) and refits on the final file's own rows.

  python tools/final_vs_shipped.py --final F --shipped S [--raw data/raw/final_ranking.parquet] [--out OUT.json]

Prints and writes: rows compared, RMS / percentiles / max of the difference overall and per airport-month, and the rows moving more
than 300 s (the fill-pin rows should be among them). No label is read; no official score.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import sha256_file, utc_now, write_json  # noqa: E402

ID, TARGET = "MVT_ID_mvt", "TAXITIME_SEC_mvt"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", required=True)
    ap.add_argument("--shipped", required=True)
    ap.add_argument("--raw", default="data/raw/final_ranking.parquet")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    f = pd.read_parquet(ROOT / a.final).rename(columns={TARGET: "final"})
    s = pd.read_parquet(ROOT / a.shipped).rename(columns={TARGET: "shipped"})
    raw = pd.read_parquet(ROOT / a.raw, columns=[ID, "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"]).query("PHASE_mvt == 'DEP'")
    raw["month"] = raw["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m")
    d = s.merge(f, on=ID, how="inner", validate="one_to_one").merge(raw[[ID, "ADEP_mvt", "month"]], on=ID, how="left")
    d["diff"] = d["final"].astype(float) - d["shipped"].astype(float)
    ad = d["diff"].abs()
    by = d.groupby(["ADEP_mvt", "month"])["diff"].agg(rows="size", rms=lambda x: float(np.sqrt(np.mean(np.square(x)))),
                                                       mean="mean", changed=lambda x: int((x != 0).sum()), max_abs=lambda x: float(np.abs(x).max()))
    big = d.loc[ad > 300, [ID, "ADEP_mvt", "month", "shipped", "final", "diff"]].sort_values("diff", key=np.abs, ascending=False)
    out = {"created_at_utc": utc_now(), "final": a.final, "final_sha256": sha256_file(ROOT / a.final), "shipped": a.shipped,
           "shipped_sha256": sha256_file(ROOT / a.shipped), "rows_final": int(len(f)), "rows_shipped": int(len(s)), "rows_compared": int(len(d)),
           "rms_diff_s": float(np.sqrt(np.mean(np.square(d["diff"])))), "mean_diff_s": float(d["diff"].mean()),
           "changed_rows": int((d["diff"] != 0).sum()), "abs_diff_percentiles_s": {str(q): float(np.percentile(ad, q)) for q in (50, 90, 99, 99.9)},
           "max_abs_diff_s": float(ad.max()), "rows_over_300_s": int(len(big)),
           "rms_contribution_of_rows_over_300_s": float(np.sqrt(np.sum(np.square(big["diff"])) / len(d))) if len(big) else 0.0,
           "by_airport_month": {f"{k[0]} {k[1]}": {kk: (float(vv) if isinstance(vv, (float, np.floating)) else int(vv)) for kk, vv in r.items()}
                                for k, r in by.iterrows()},
           "largest": big.head(25).astype({ID: "float64"}).to_dict(orient="records")}
    print(json.dumps({k: v for k, v in out.items() if k not in ("by_airport_month", "largest")}, indent=1))
    print(by.round(2).to_string())
    print(big.head(25).to_string(index=False))
    if a.out:
        write_json(ROOT / a.out, out)


if __name__ == "__main__":
    main()
