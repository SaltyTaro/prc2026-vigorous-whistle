"""Label-free ADS-B feature-shift audit, 2025 -> 2026 (docs/adsb_shift/): did the surface-trace features change distribution on
matched rows in a way that looks like a receiver / coverage / processing change rather than real operations?

Inputs: the committee learners' ADS-B tables (runs/adsb_ctx2/inputs/adsb_features_arcw0: every 2025 month and the ranking; the
table every context seed used) and the v19 extras (data/external/adsblol/features_v19). Status as in tools/v30_error_map.py: gate =
matched with adsb_gate 1, dark = matched without it, unmatched otherwise. Per airport, January 2026 vs January 2025 and July 2026 vs
July 2025, gate and dark rows separately: p10/p50/p90, missing share, standardized mean shift, and the 2026 median against the 2025
monthly medians (range and typical month-to-month step). Weights: mean normalized gain over two committee LightGBMs (all non-LIRF
rows / LIRF) and the residual stage's adj2 booster. Read-only; writes --output and a CSV.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import utc_now, write_json  # noqa: E402

ID = "MVT_ID_mvt"
ADSB = ROOT / "runs/adsb_ctx2/inputs/adsb_features_arcw0"
V19 = ROOT / "data/external/adsblol/features_v19"
MONTHS_2025 = [f"2025-{m:02}" for m in range(1, 13)]
PAIRS = {"2026-01": "2025-01", "2026-07": "2025-07"}
IMPORTANCE = ["runs/adsb_v23/adsb_v20__lgb_x19_s24/refit/model.txt", "runs/adsb_v23/adsb_v21__lgb_lirf_s31/refit/model.txt",
              "runs/adsb_v25/export_v23/residual_adj2_2026-07.txt"]


def training_file(month: str) -> str:
    y, m = int(month[:4]), int(month[5:7])
    return f"training_{y}-{m:02}-01_{y + (m == 12)}-{(m % 12) + 1:02}-01.parquet"


def departures(files: list[str]) -> pd.DataFrame:
    parts = []
    for f in files:
        t = pq.read_table(ROOT / "data/raw" / f, columns=[ID, "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"]).to_pandas()
        t = t[t["PHASE_mvt"].eq("DEP")]
        parts.append(pd.DataFrame({ID: t[ID].to_numpy(), "airport": t["ADEP_mvt"].to_numpy(), "month": t["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m").to_numpy()}))
    return pd.concat(parts, ignore_index=True).drop_duplicates(ID)


def load() -> pd.DataFrame:
    frames = []
    for month in MONTHS_2025:
        dep = departures([training_file(month)])
        dep = dep[dep["month"] == month]
        frames.append(dep.merge(pd.read_parquet(ADSB / f"adsb_{training_file(month)}"), on=ID, how="left")
                      .merge(pd.read_parquet(V19 / f"features_v19_{month}.parquet"), on=ID, how="left"))
    dep = departures(["ranking.parquet"])
    v19 = pd.concat([pd.read_parquet(V19 / f"features_v19_{m}.parquet") for m in PAIRS], ignore_index=True)
    frames.append(dep.merge(pd.read_parquet(ADSB / "adsb_ranking.parquet"), on=ID, how="left").merge(v19, on=ID, how="left"))
    frame = pd.concat(frames, ignore_index=True)
    matched = frame["adsb_matched"].eq(1)
    frame["status"] = np.where(matched & frame["adsb_gate"].eq(1), "gate", np.where(matched, "dark", "unmatched"))
    return frame


def importance() -> dict[str, float]:
    import lightgbm as lgb
    shares = []
    for path in IMPORTANCE:
        booster = lgb.Booster(model_file=str(ROOT / path))
        gain = pd.Series(booster.feature_importance("gain"), index=booster.feature_name(), dtype=float)
        shares.append(gain / gain.sum())
    return pd.concat(shares, axis=1).fillna(0.0).mean(axis=1).to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("docs/adsb_shift/adsb_shift.json"))
    args = parser.parse_args()
    imp = importance()
    frame = load()
    features = [c for c in frame.columns if c.startswith(("adsb_", "nb_", "sfc_")) and c not in ("adsb_matched", "adsb_gate")]
    mix = frame.groupby(["airport", "month", "status"]).size().unstack(fill_value=0)
    mix = mix.div(mix.sum(axis=1), axis=0)
    rows = []
    for airport, g in frame.groupby("airport"):
        for status in ["gate", "dark"]:
            s = g[g["status"] == status]
            monthly = s[s["month"].isin(MONTHS_2025)].groupby("month")[features].median()
            steps = monthly.sort_index().diff().abs().median()
            for new, old in PAIRS.items():
                a, b = s[s["month"] == new], s[s["month"] == old]
                if len(a) < 200 or len(b) < 200:
                    continue
                for c in features:
                    x, y = a[c].astype(float), b[c].astype(float)
                    sd = float(y.std())
                    med = float(x.median()) if x.notna().any() else np.nan
                    rows.append({"airport": airport, "status": status, "month": new, "rows_2026": int(len(a)), "rows_2025": int(len(b)), "feature": c,
                                 "p10_2026": float(x.quantile(0.1)), "p50_2026": med, "p90_2026": float(x.quantile(0.9)),
                                 "p10_2025": float(y.quantile(0.1)), "p50_2025": float(y.median()), "p90_2025": float(y.quantile(0.9)),
                                 "na_2026": float(x.isna().mean()), "na_2025": float(y.isna().mean()),
                                 "smd": (float(x.mean()) - float(y.mean())) / sd if sd > 0 else np.nan,
                                 "median_min_2025": float(monthly[c].min()), "median_max_2025": float(monthly[c].max()),
                                 "median_outside_2025": bool(med < monthly[c].min() or med > monthly[c].max()) if np.isfinite(med) else False,
                                 "median_shift_over_step": float((med - float(y.median())) / steps[c]) if steps[c] and steps[c] > 0 else np.nan,
                                 "importance_share": float(imp.get(c, 0.0))})
    table = pd.DataFrame(rows)
    table["impact"] = table["importance_share"] * table["rows_2026"] / 344841 * (np.minimum(table["smd"].abs().fillna(0), 5) + 5 * (table["na_2026"] - table["na_2025"]).abs())
    out = args.output if args.output.is_absolute() else ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out.with_suffix(".csv"), index=False)
    report = {"created_at_utc": utc_now(), "importance_sources": IMPORTANCE,
              "importance_adsb": {k: v for k, v in sorted(imp.items(), key=lambda kv: -kv[1]) if k.startswith(("adsb_", "nb_", "sfc_"))},
              "status_mix": {f"{a}:{m}": {k: float(v) for k, v in r.items()} for (a, m), r in mix.iterrows() if m in (*PAIRS, *PAIRS.values())},
              "top_impact": table.sort_values("impact", ascending=False).head(40).to_dict("records")}
    write_json(out, json.loads(json.dumps(report, default=float).replace("NaN", "null")))
    print("written", out)


if __name__ == "__main__":
    main()
