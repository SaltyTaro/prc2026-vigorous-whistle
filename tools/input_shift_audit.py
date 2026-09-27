"""Label-free input-shift audit, 2025 -> 2026 (docs/input_shift/): did any model input change meaning, format or coverage?

Compares the ranking months (January and July 2026) with the 2025 training months per airport. Labels and departure block times
are never read (departure TAXITIME/BLOCK are only counted as present/absent; arrival in-blocks and taxi-in are inputs).
1. Categorical novelty: share of 2026 departures whose stand / runway / operator / aircraft type / flight prefix / destination /
   stand-name pattern never appears at that airport in 2025, against a leave-one-month-out 2025 baseline (Jan/Jul/Dec 2025).
2. NM clock semantics per airport x month: missingness (AOBT_3, LOBT, EOBT_1, NM origin, no clock), MVT-AOBT_3, AOBT_3-SCHED,
   MVT-LOBT, EOBT_1-SCHED, IOBT-SCHED quantiles, seconds==0 shares.
3. Schedule / movement fields: rounding, missingness, departure label/block presence, arrival in-block / taxi-in distributions.
4. The frozen timeline bundle's ranking rows against the same calendar month of 2025 per airport: standardized mean shift and
   missingness shift of every numeric feature.
Impact = ranking rows affected x the features' importance share (saved gain importances of the twelve-month v7 nodes and two
committee LightGBMs). Read-only; writes only --output.
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
from prc2026.artifacts import sha256_file, utc_now, write_json  # noqa: E402

RAW = ROOT / "data/raw"
TRAINING = [f"training_2025-{m:02}-01_{2025 + (m == 12)}-{(m % 12) + 1:02}-01.parquet" for m in range(1, 13)]
COLUMNS = ["MVT_ID_mvt", "FLIGHT_mvt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt",
           "AIRCRAFT_TYPE_mvt", "RUNWAY_mvt", "STAND_mvt", "TAXITIME_SEC_mvt", "LOBT_flt", "ADEP_flt", "IOBT_flt", "AIRCRAFT_OPERATOR_flt",
           "EOBT_1_flt", "AOBT_3_flt", "ARVT_3_flt"]
PAIRS = {"2026-01": "2025-01", "2026-07": "2025-07"}
BASELINE_MONTHS = ["2025-01", "2025-07", "2025-12"]
CATEGORICALS = {"stand": "STAND_mvt", "runway": "RUNWAY_mvt", "operator": "AIRCRAFT_OPERATOR_flt", "aircraft_type": "AIRCRAFT_TYPE_mvt",
                "flight_prefix": "flight_prefix", "destination": "ADES_mvt", "stand_pattern": "stand_pattern"}
FEATURE_OF = {"stand": ["stand"], "runway": ["runway"], "operator": ["operator"], "aircraft_type": ["aircraft_type"],
              "flight_prefix": ["record_flight_prefix"], "destination": ["destination"], "stand_pattern": ["stand"]}
TIMELINE = "data/timeline/ce42240e5e43eee1713f/features"
IMPORTANCE = {"v7_twelve": "runs/adsb_v24/v7/refit", "committee_lgb": "runs/adsb_v23/adsb_v20__lgb_x19_s24/refit/model.txt",
              "lirf_lgb": "runs/adsb_v23/adsb_v21__lgb_lirf_s31/refit/model.txt"}


def pattern(values: pd.Series) -> pd.Series:
    s = values.astype("string").str.strip()
    return s.str.replace(r"[A-Za-z]", "A", regex=True).str.replace(r"\d", "9", regex=True)


def load() -> pd.DataFrame:
    parts = []
    for name in [*TRAINING, "ranking.parquet"]:
        frame = pq.read_table(RAW / name, columns=COLUMNS).to_pandas()
        frame["source"] = "ranking" if name == "ranking.parquet" else "training"
        parts.append(frame)
    raw = pd.concat(parts, ignore_index=True).drop_duplicates("MVT_ID_mvt")
    raw["month"] = raw["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m")
    dep = raw["PHASE_mvt"].eq("DEP")
    raw["airport"] = np.where(dep, raw["ADEP_mvt"], raw["ADES_mvt"])
    flight = raw["FLIGHT_mvt"].astype("string").str.strip().str.upper()
    raw["flight_prefix"] = flight.str.extract(r"^([A-Z]+)", expand=False).str[:3]
    raw["stand_pattern"] = pattern(raw["STAND_mvt"])
    for c in ["PHASE_mvt", "airport", "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt", "ADEP_flt",
              "flight_prefix", "stand_pattern", "source", "month"]:
        raw[c] = raw[c].astype("category")
    return raw.drop(columns=["FLIGHT_mvt"])


def importance() -> dict[str, float]:
    """Mean normalized gain share per feature over the twelve-month v7 nodes (globals + airports) and two committee LightGBMs."""
    import lightgbm as lgb
    shares = []
    node_dir = ROOT / IMPORTANCE["v7_twelve"]
    for path in sorted(node_dir.glob("*_importance.json")):
        gain = pd.Series(json.loads(path.read_text(encoding="utf-8")), dtype=float)
        shares.append(gain / gain.sum())
    for key in ["committee_lgb", "lirf_lgb"]:
        booster = lgb.Booster(model_file=str(ROOT / IMPORTANCE[key]))
        gain = pd.Series(booster.feature_importance("gain"), index=booster.feature_name(), dtype=float)
        shares.append(gain / gain.sum())
        del booster
    table = pd.concat(shares, axis=1).fillna(0.0)
    return table.mean(axis=1).sort_values(ascending=False).to_dict()


def novelty(dep: pd.DataFrame, imp: dict) -> dict:
    out = {}
    total = int((dep["source"] == "ranking").sum())
    for var, col in CATEGORICALS.items():
        weight = sum(imp.get(f, 0.0) for f in FEATURE_OF[var])
        rows = []
        for airport, g in dep.groupby("airport", observed=True):
            by_month = {m: set(s.dropna().astype(str)) for m, s in g.groupby("month", observed=True)[col]}
            seen_2025 = set().union(*[v for m, v in by_month.items() if m.startswith("2025")]) if by_month else set()
            rec = {"airport": airport}
            for month in [*PAIRS, *BASELINE_MONTHS]:
                s = g.loc[g["month"] == month, col]
                if not len(s):
                    continue
                known = seen_2025 if month.startswith("2026") else set().union(*[v for m, v in by_month.items() if m.startswith("2025") and m != month])
                values = s.dropna().astype(str)
                novel = ~values.isin(known)
                rec[f"{month}_novel_share"] = float(novel.sum() / len(s))
                rec[f"{month}_null_share"] = float(s.isna().mean())
                if month.startswith("2026"):
                    rec[f"{month}_rows"] = int(len(s))
                    rec[f"{month}_top_novel"] = values[novel].value_counts().head(5).to_dict()
            rec["excess_novel_rows"] = float(sum(max(0.0, rec.get(f"{m}_novel_share", 0) - max(rec.get(f"{b}_novel_share", 0) for b in BASELINE_MONTHS if f"{b}_novel_share" in rec))
                                                 * rec.get(f"{m}_rows", 0) for m in PAIRS))
            rec["excess_null_rows"] = float(sum(max(0.0, rec.get(f"{m}_null_share", 0) - rec.get(f"{PAIRS[m]}_null_share", 0)) * rec.get(f"{m}_rows", 0) for m in PAIRS))
            rec["impact"] = (rec["excess_novel_rows"] + rec["excess_null_rows"]) / total * weight
            rows.append(rec)
        out[var] = {"importance_share": weight, "airports": rows}
    return out


def quantiles(x: pd.Series) -> dict:
    x = x.dropna()
    if not len(x):
        return {"n": 0}
    q = x.quantile([0.1, 0.5, 0.9]).to_numpy()
    return {"n": int(len(x)), "p10": float(q[0]), "p50": float(q[1]), "p90": float(q[2]), "neg_share": float((x < 0).mean())}


def seconds_zero(t: pd.Series) -> float | None:
    t = t.dropna()
    return float((t.dt.second == 0).mean()) if len(t) else None


def clock_table(dep: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sec = lambda a, b: (a - b).dt.total_seconds()  # noqa: E731
    for (airport, month), g in dep.groupby(["airport", "month"], observed=True):
        if len(g) < 500:
            continue
        rec = {"airport": airport, "month": month, "departures": int(len(g)),
               "aobt3_missing": float(g["AOBT_3_flt"].isna().mean()), "lobt_missing": float(g["LOBT_flt"].isna().mean()),
               "eobt1_missing": float(g["EOBT_1_flt"].isna().mean()), "nm_origin_missing": float(g["ADEP_flt"].isna().mean()),
               "no_clock": float((g["AOBT_3_flt"].isna() & g["LOBT_flt"].isna()).mean()), "sched_missing": float(g["SCHED_TIME_UTC_mvt"].isna().mean()),
               "arvt3_missing": float(g["ARVT_3_flt"].isna().mean()),
               "aobt3_sec0": seconds_zero(g["AOBT_3_flt"]), "lobt_sec0": seconds_zero(g["LOBT_flt"]), "eobt1_sec0": seconds_zero(g["EOBT_1_flt"]),
               "sched_sec0": seconds_zero(g["SCHED_TIME_UTC_mvt"]), "mvt_sec0": seconds_zero(g["MVT_TIME_UTC_mvt"]),
               "stand_missing": float(g["STAND_mvt"].isna().mean()), "runway_missing": float(g["RUNWAY_mvt"].isna().mean()),
               "aircraft_missing": float(g["AIRCRAFT_TYPE_mvt"].isna().mean()), "operator_missing": float(g["AIRCRAFT_OPERATOR_flt"].isna().mean())}
        for name, x in {"mvt_minus_aobt3": sec(g["MVT_TIME_UTC_mvt"], g["AOBT_3_flt"]), "aobt3_minus_sched": sec(g["AOBT_3_flt"], g["SCHED_TIME_UTC_mvt"]),
                        "mvt_minus_lobt": sec(g["MVT_TIME_UTC_mvt"], g["LOBT_flt"]), "eobt1_minus_sched": sec(g["EOBT_1_flt"], g["SCHED_TIME_UTC_mvt"]),
                        "iobt_minus_sched": sec(g["IOBT_flt"], g["SCHED_TIME_UTC_mvt"]), "mvt_minus_sched": sec(g["MVT_TIME_UTC_mvt"], g["SCHED_TIME_UTC_mvt"]),
                        "aobt3_minus_lobt": sec(g["AOBT_3_flt"], g["LOBT_flt"])}.items():
            for k, v in quantiles(x).items():
                rec[f"{name}_{k}"] = v
        rows.append(rec)
    return pd.DataFrame(rows)


def arrival_table(arr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (airport, month), g in arr.groupby(["airport", "month"], observed=True):
        if len(g) < 500:
            continue
        rec = {"airport": airport, "month": month, "arrivals": int(len(g)), "in_block_present": float(g["BLOCK_TIME_UTC_mvt"].notna().mean()),
               "taxi_in_present": float(g["TAXITIME_SEC_mvt"].notna().mean()), "in_block_sec0": seconds_zero(g["BLOCK_TIME_UTC_mvt"]),
               "landing_sec0": seconds_zero(g["MVT_TIME_UTC_mvt"]), "stand_missing": float(g["STAND_mvt"].isna().mean()),
               "sched_missing": float(g["SCHED_TIME_UTC_mvt"].isna().mean())}
        for k, v in quantiles(g["TAXITIME_SEC_mvt"].astype(float)).items():
            rec[f"taxi_in_{k}"] = v
        rows.append(rec)
    return pd.DataFrame(rows)


def jumps(table: pd.DataFrame, metrics: list[str], rows_col: str) -> list[dict]:
    """2026 month vs the same 2025 month, scaled by the 2025 month-to-month spread (robust: MAD of consecutive differences)."""
    out = []
    for airport, g in table.groupby("airport"):
        g = g.set_index("month")
        base = g[g.index.str.startswith("2025")].sort_index()
        for metric in metrics:
            if metric not in g or base[metric].isna().all():
                continue
            steps = base[metric].astype(float).diff().abs().dropna()
            spread = float(np.median(steps)) if len(steps) else np.nan
            for new, old in PAIRS.items():
                if new not in g.index or old not in g.index:
                    continue
                a, b = g.at[new, metric], g.at[old, metric]
                if a is None or b is None or pd.isna(a) or pd.isna(b):
                    continue
                delta = float(a) - float(b)
                lo, hi = float(base[metric].min()), float(base[metric].max())
                outside = float(a) < lo or float(a) > hi
                z = abs(delta) / spread if spread and spread > 0 else (np.inf if delta else 0.0)
                out.append({"airport": airport, "month": new, "metric": metric, "value_2026": float(a), "value_2025_same_month": float(b), "delta": delta,
                            "range_2025": [lo, hi], "outside_2025_range": bool(outside), "delta_over_typical_monthly_step": float(z), "rows": int(g.at[new, rows_col])})
    return out


def feature_shift(imp: dict, names: list[str], categorical: set[str]) -> pd.DataFrame:
    base = ROOT / TIMELINE
    numeric = [c for c in names if c not in categorical and c not in ("airport", "month")]
    keys = ["MVT_ID_mvt", "airport"]
    files = {"2026": ["ranking.parquet"], "2025-01": ["training_2025-01-01_2025-02-01.parquet"], "2025-07": ["training_2025-07-01_2025-08-01.parquet"]}
    times = pd.read_parquet(RAW / "ranking.parquet", columns=["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "PHASE_mvt"])
    times = times.loc[times["PHASE_mvt"].eq("DEP")]
    month_of = times.set_index("MVT_ID_mvt")["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m")
    rows = []
    for start in range(0, len(numeric), 40):
        cols = numeric[start:start + 40]
        frames = {k: pd.concat([pd.read_parquet(base / f, columns=[*keys, *cols]) for f in v], ignore_index=True) for k, v in files.items()}
        ranking = frames["2026"]
        ranking["m"] = ranking["MVT_ID_mvt"].map(month_of)
        for new, old in PAIRS.items():
            a_all, b_all = ranking.loc[ranking["m"] == new], frames[old]
            for airport, a in a_all.groupby("airport", observed=True):
                b = b_all.loc[b_all["airport"] == airport]
                if len(b) < 500 or len(a) < 500:
                    continue
                for c in cols:
                    x, y = pd.to_numeric(a[c], errors="coerce").astype(float), pd.to_numeric(b[c], errors="coerce").astype(float)
                    sd = float(y.std())
                    smd = (float(x.mean()) - float(y.mean())) / sd if sd > 0 else (0.0 if x.mean() == y.mean() or (np.isnan(x.mean()) and np.isnan(y.mean())) else np.inf)
                    rows.append({"feature": c, "airport": airport, "month": new, "rows": int(len(a)), "mean_2026": float(x.mean()), "mean_2025": float(y.mean()),
                                 "sd_2025": sd, "smd": smd, "na_2026": float(x.isna().mean()), "na_2025": float(y.isna().mean()),
                                 "importance_share": float(imp.get(c, 0.0))})
        del frames, ranking
    table = pd.DataFrame(rows)
    total = int(table.groupby(["airport", "month"])["rows"].first().sum())
    table["na_shift"] = table["na_2026"] - table["na_2025"]
    table["impact"] = table["importance_share"] * table["rows"] / total * (np.minimum(table["smd"].abs().fillna(0.0), 5.0) + 5.0 * table["na_shift"].abs())
    return table


def clean(value):
    """JSON-safe: non-finite floats become strings ('inf', 'nan')."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("docs/input_shift/input_shift.json"))
    parser.add_argument("--tables", type=Path, default=Path("docs/input_shift/tables"))
    args = parser.parse_args()
    imp = importance()
    raw = load()
    dep = raw.loc[raw["PHASE_mvt"] == "DEP"].reset_index(drop=True)
    arr = raw.loc[raw["PHASE_mvt"] == "ARR"].reset_index(drop=True)
    del raw
    report = {"created_at_utc": utc_now(), "inputs": {f: sha256_file(RAW / f) for f in [*TRAINING, "ranking.parquet"]},
              "importance_sources": IMPORTANCE, "top_importance": dict(list(imp.items())[:40])}
    report["departure_label_block_present_in_ranking"] = {"taxi": int(dep.loc[dep["source"] == "ranking", "TAXITIME_SEC_mvt"].notna().sum()),
                                                          "block": int(dep.loc[dep["source"] == "ranking", "BLOCK_TIME_UTC_mvt"].notna().sum())}
    report["novelty"] = novelty(dep, imp)
    clocks = clock_table(dep)
    arrivals = arrival_table(arr)
    clock_metrics = [c for c in clocks.columns if c not in ("airport", "month", "departures") and not c.endswith("_n")]
    arrival_metrics = [c for c in arrivals.columns if c not in ("airport", "month", "arrivals") and not c.endswith("_n")]
    report["clock_jumps"] = sorted(jumps(clocks, clock_metrics, "departures"), key=lambda r: -r["delta_over_typical_monthly_step"] if np.isfinite(r["delta_over_typical_monthly_step"]) else -1e9)
    report["arrival_jumps"] = sorted(jumps(arrivals, arrival_metrics, "arrivals"), key=lambda r: -r["delta_over_typical_monthly_step"] if np.isfinite(r["delta_over_typical_monthly_step"]) else -1e9)
    del dep, arr
    sys.path.insert(0, str(ROOT / "src"))
    from prc2026_timeline.features import CATEGORICAL_COLUMNS, FEATURE_NAMES
    shift = feature_shift(imp, list(FEATURE_NAMES), set(CATEGORICAL_COLUMNS))
    report["feature_shift_top_impact"] = shift.sort_values("impact", ascending=False).head(60).to_dict("records")
    report["feature_shift_top30_by_airport"] = {ap: g.assign(a=g["smd"].abs()).sort_values("a", ascending=False).drop(columns="a").head(30).to_dict("records")
                                                for ap, g in shift.groupby("airport")}
    tables = args.tables if args.tables.is_absolute() else ROOT / args.tables
    tables.mkdir(parents=True, exist_ok=True)
    clocks.to_csv(tables / "clocks_by_airport_month.csv", index=False)
    arrivals.to_csv(tables / "arrivals_by_airport_month.csv", index=False)
    shift.to_csv(tables / "feature_shift.csv", index=False)
    out = args.output if args.output.is_absolute() else ROOT / args.output
    write_json(out, clean(json.loads(json.dumps(report, default=lambda o: o.item() if hasattr(o, "item") else str(o)))))
    print("written", out)


if __name__ == "__main__":
    main()
