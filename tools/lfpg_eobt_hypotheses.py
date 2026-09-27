"""Bounds for the LFPG EOBT_1 decision (docs/input_shift/lfpg_eobt_sensitivity.md): which of shipped v30 (2026 EOBT_1 as given) and
the EOBT_1 := IOBT rebuild (runs/final_lfpg_eobt/run), or a blend, is closer to 2025's delay-conditional expectation?

The NM proxy (MVT - AOBT_3) drives LFPG. For July-2026 LFPG departures, the expected label is proxy + the 2025 mean of
(label - proxy) in the row's delay bin, from the four recipe months (development labels, v30 dev rows). Two things define it:
- which 2025 rows supply the mean: EOBT_1 > IOBT ("announced" delay) or EOBT_1 == IOBT;
- what is binned: total delay AOBT - IOBT, or unannounced delay AOBT - EOBT_1.
The bias SSE (expected label - prediction) is compared for blends w in {0, 0.5, 0.75, 1} of the rebuild; row noise is common to
all blends and cancels. Descriptive; no ranking label exists or is inferred.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import utc_now, write_json  # noqa: E402

COLS = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "EOBT_1_flt", "IOBT_flt", "AOBT_3_flt", "MVT_TIME_UTC_mvt"]
RECIPE_MONTH_FILES = ["training_2025-01-01_2025-02-01.parquet", "training_2025-07-01_2025-08-01.parquet", "training_2025-11-01_2025-12-01.parquet",
                      "training_2025-12-01_2026-01-01.parquet"]
BINS = [-1e9, -300, 0, 150, 300, 450, 600, 750, 900, 1200, 1500, 1800, 2700, 3600, 1e9]
RANKING_SSE = 344841 * 229.6009 ** 2  # v30 official RMSE on all rows


def load(files: list[str]) -> pd.DataFrame:
    parts = []
    for f in files:
        t = pq.read_table(ROOT / "data/raw" / f, columns=COLS).to_pandas()
        parts.append(t[(t.PHASE_mvt == "DEP") & (t.ADEP_mvt == "LFPG")])
    r = pd.concat(parts).drop_duplicates("MVT_ID_mvt")
    r["ei"] = (r.EOBT_1_flt - r.IOBT_flt).dt.total_seconds()
    r["delay"] = (r.AOBT_3_flt - r.IOBT_flt).dt.total_seconds()
    r["unann"] = (r.AOBT_3_flt - r.EOBT_1_flt).dt.total_seconds()
    r["proxy"] = (r.MVT_TIME_UTC_mvt - r.AOBT_3_flt).dt.total_seconds()
    r["dbin"], r["ubin"] = pd.cut(r.delay, BINS), pd.cut(r.unann, BINS)
    return r


def main() -> None:
    dev = pd.read_parquet(ROOT / "runs/error_map/dev_rows.parquet", columns=["MVT_ID_mvt", "label", "v30", "month"])
    d = load(RECIPE_MONTH_FILES).merge(dev, on="MVT_ID_mvt")
    d = d[(d.label <= 10800) & d.proxy.notna()].copy()
    d["lp"] = d.label - d.proxy
    july = d[d.month == "2025-07"]
    hypotheses = {  # name: (2025 rows supplying the mean, bin key)
        "unannounced_total_delay": (d[d.ei == 0], "dbin"),
        "unannounced_total_delay_july_only": (july[july.ei == 0], "dbin"),
        "announced_total_delay": (d[d.ei > 0], "dbin"),
        "announced_unannounced_part": (d[d.ei > 0], "ubin"),
        "all_rows_unannounced_part": (d, "ubin"),
    }
    r = load(["ranking.parquet"])
    r = r[r.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m") == "2026-07"]
    orig = pd.read_parquet(ROOT / "runs/final_dryrun/run/recipe/detail.parquet", columns=["MVT_ID_mvt", "v30"]).rename(columns={"v30": "orig"})
    reset = pd.read_parquet(ROOT / "runs/final_lfpg_eobt/run/recipe/detail.parquet", columns=["MVT_ID_mvt", "v30"]).rename(columns={"v30": "reset"})
    r = r.merge(orig, on="MVT_ID_mvt").merge(reset, on="MVT_ID_mvt")
    r = r[r.proxy.notna()].copy()
    report = {"created_at_utc": utc_now(), "bins": BINS, "ranking_sse_reference": RANKING_SSE, "rows": {}, "hypotheses": {}}
    for name, (rows, key) in hypotheses.items():
        table = rows.groupby(key, observed=True).lp.mean()
        out = {"bin": "total delay AOBT-IOBT" if key == "dbin" else "unannounced delay AOBT-EOBT_1 (as given in 2026)", "rows_2025": int(len(rows))}
        for group, mask in [("edited", r.ei > 0), ("unedited", ~(r.ei > 0))]:
            e = r[mask]
            expected = e.proxy + e[key].map(table).astype(float)
            report["rows"][group] = int(len(e))
            sse = {w: float(((expected - (e.orig + w * (e.reset - e.orig))) ** 2).sum()) for w in [0.0, 0.5, 0.75, 1.0]}
            out[group] = {"mean_expected_minus_shipped": float((expected - e.orig).mean()), "mean_expected_minus_reset": float((expected - e.reset).mean()),
                          "bias_sse_change_vs_shipped_pct_of_ranking_sse": {f"w{w}": (v - sse[0.0]) / RANKING_SSE * 100 for w, v in sse.items()}}
        report["hypotheses"][name] = out
        print(name, {g: {k: round(v, 3) for k, v in out[g]["bias_sse_change_vs_shipped_pct_of_ranking_sse"].items()} for g in ["edited", "unedited"]})
    write_json(ROOT / "docs/input_shift/lfpg_eobt_hypotheses.json", report)


if __name__ == "__main__":
    main()
