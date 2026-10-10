"""Label-free drift screen of the final file's new months (docs/final_file/drift_screen.md).

For every airport and quantity, February / June 2026 (the final file's unscored months) are compared with February / June 2025, and
January / July 2026 (board months, already scored, "drift we survived") with January / July 2025. Each 2026 value is also placed in the
range of the twelve 2025 months. A February / June 2026 value is FLAGGED when it lies outside the 2025 range AND further from its 2025
analogue month than both board months moved from theirs (same airport, same quantity). Flags are prompts to look, not verdicts.

Quantities per airport-month (2026 side label-free):
  dep_per_day           departures per calendar day
  adsb_match            share of departures with an ADS-B surface match (last_ground present; the runner's adsb_matched)
  adsb_gate             share of matched departures traced from the gate (the runner's adsb_gate)
  nm_record             share with an NM flight record (ADEP_flt present)
  nm_clock              share with an NM off-block clock (AOBT_3_flt or LOBT_flt present)
  eobt_later            share with EOBT_1 > IOBT (the repair detector's statistic, src/prc2026_final/stack.py eobt_shares)
  lirf_pop_per_day      LIRF only: departures per day of the fill population (src/prc2026_lirf_mixture/features.py population_rows:
                        no NM clock, no airport aircraft type)
  lirf_pop_long_per_day LIRF only: of those, schedule interval (take-off - schedule) > 43,200 s (src/prc2026_final/fill_pin.py)
  unseen_stand          share of departures at a stand never used by a departure of that airport in 2025 (for a 2025 month: in the
                        other eleven months)
  unseen_type           the same for the aircraft type (AIRCRAFT_TYPE_flt, else AIRCRAFT_TYPE_mvt)
  runway_tvd            total-variation distance of the runway shares from the 2025 analogue month (for 2025 months: from the
                        pooled 2025 distribution); range check uses the distance from the pooled 2025 distribution
  pred_* / label_*      the final candidate's predictions (2026) against the 2025 labels of the same calendar month: mean, p50, p90,
                        p99 and share above 3,600 s

  python tools/final_drift_screen.py [--final runs/final_blind/final_candidate/chain_out_km.parquet] [--out docs/final_file/drift_screen_results.json]
No 2026 label exists or is read. No official score.
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

ID, TARGET, TIME = "MVT_ID_mvt", "TAXITIME_SEC_mvt", "MVT_TIME_UTC_mvt"
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
COLS = [ID, "PHASE_mvt", "ADEP_mvt", TIME, "SCHED_TIME_UTC_mvt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_TYPE_flt",
        "ADEP_flt", "AOBT_3_flt", "LOBT_flt", "EOBT_1_flt", "IOBT_flt"]
PAIRS = {"2026-02": "2025-02", "2026-06": "2025-06", "2026-01": "2025-01", "2026-07": "2025-07"}
NEW, BOARD = ["2026-02", "2026-06"], ["2026-01", "2026-07"]
LONG_S = 43200.0


def departures(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame[frame["PHASE_mvt"].eq("DEP") & frame["ADEP_mvt"].isin(AIRPORTS)].copy()
    d["month"] = d[TIME].dt.strftime("%Y-%m")
    d["type"] = d["AIRCRAFT_TYPE_flt"].fillna(d["AIRCRAFT_TYPE_mvt"])
    return d


def month_stats(d: pd.DataFrame, matched: pd.Series, gate: pd.Series) -> dict:
    """Scalar quantities for one airport-month's departures (index-aligned matched / gate flags)."""
    days = d[TIME].dt.normalize().nunique()
    out = {"departures": int(len(d)), "dep_per_day": len(d) / max(days, 1),
           "adsb_match": float(matched.mean()), "adsb_gate": float(gate[matched].mean()) if matched.any() else np.nan,
           "nm_record": float(d["ADEP_flt"].notna().mean()),
           "nm_clock": float((d["AOBT_3_flt"].notna() | d["LOBT_flt"].notna()).mean()),
           "eobt_later": float((d["EOBT_1_flt"] > d["IOBT_flt"]).fillna(False).mean())}
    if d["ADEP_mvt"].iat[0] == "LIRF":
        pop = d["AOBT_3_flt"].isna() & d["LOBT_flt"].isna() & d["AIRCRAFT_TYPE_mvt"].isna()
        interval = (d[TIME] - d["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
        out["lirf_pop_per_day"] = float(pop.sum()) / max(days, 1)
        out["lirf_pop_long_per_day"] = float((pop & (interval > LONG_S)).sum()) / max(days, 1)
    return out


def dist(values: np.ndarray, prefix: str) -> dict:
    v = values[np.isfinite(values)]
    return {f"{prefix}_mean": float(v.mean()), f"{prefix}_p50": float(np.percentile(v, 50)), f"{prefix}_p90": float(np.percentile(v, 90)),
            f"{prefix}_p99": float(np.percentile(v, 99)), f"{prefix}_over3600": float((v > 3600).mean())}


def tvd(a: pd.Series, b: pd.Series) -> float:
    i = a.index.union(b.index)
    return float(0.5 * np.abs(a.reindex(i, fill_value=0) - b.reindex(i, fill_value=0)).sum())


def clean(o):
    """NaN / inf -> None, numpy scalars -> Python (strict JSON)."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", default="runs/final_blind/final_candidate/chain_out_km.parquet")
    ap.add_argument("--raw", default="data/raw/final_ranking.parquet")
    ap.add_argument("--learner-adsb", default="runs/final_blind/run_final4m/adsb/learner_adsb.parquet")
    ap.add_argument("--shares", default="runs/final_blind/run_final4m/repair/shares.parquet")
    ap.add_argument("--out", default="docs/final_file/drift_screen_results.json")
    a = ap.parse_args()

    stats: dict[tuple[str, str], dict] = {}
    stands: dict[tuple[str, str], set] = {}
    types: dict[tuple[str, str], set] = {}
    runways: dict[tuple[str, str], pd.Series] = {}
    # 2025, one month file at a time (memory)
    for f in sorted(glob.glob(str(ROOT / "data/raw/training_2025-*.parquet"))):
        d = departures(pd.read_parquet(f, columns=COLS + [TARGET]))
        for month in [Path(f).name[9:16]]:   # the file's own month (a take-off after midnight UTC of the last day spills into the next file)
            mfile = ROOT / f"data/external/adsblol/matches/matches2_{month}.parquet"
            m = pd.read_parquet(mfile, columns=[ID, "gate", "last_ground"]) if mfile.exists() else pd.DataFrame(columns=[ID, "gate", "last_ground"])
            m = m[m["last_ground"].notna()].drop_duplicates(ID).set_index(ID)
            for apt, g in d[d["month"].eq(month)].groupby("ADEP_mvt"):
                matched = g[ID].isin(m.index).reset_index(drop=True)
                gate = pd.Series(m["gate"].reindex(g[ID]).fillna(False).astype(bool).to_numpy())
                s = month_stats(g.reset_index(drop=True), matched, gate)
                s.update(dist(g[TARGET].to_numpy(dtype=float), "y"))
                key = (apt, month)
                stats[key] = s
                stands[key] = set(g["STAND_mvt"].dropna())
                types[key] = set(g["type"].dropna())
                runways[key] = g["RUNWAY_mvt"].value_counts(normalize=True)
        del d
    months25 = sorted({m for _, m in stats})

    # 2026: the final file, the runner's ADS-B table, the final candidate
    d = departures(pd.read_parquet(ROOT / a.raw, columns=COLS))
    lt = pd.read_parquet(ROOT / a.learner_adsb, columns=[ID, "adsb_matched", "adsb_gate"]).set_index(ID)
    pred = pd.read_parquet(ROOT / a.final).set_index(ID)[TARGET]
    for (apt, month), g in d.groupby(["ADEP_mvt", "month"]):
        g = g.reset_index(drop=True)
        matched = pd.Series(lt["adsb_matched"].reindex(g[ID]).to_numpy() == 1)
        gate = pd.Series(lt["adsb_gate"].reindex(g[ID]).fillna(0).to_numpy() == 1)
        s = month_stats(g, matched, gate)
        s.update(dist(pred.reindex(g[ID]).to_numpy(dtype=float), "y"))
        stats[(apt, month)] = s
        stands[(apt, month)] = set(g["STAND_mvt"].dropna())
        types[(apt, month)] = set(g["type"].dropna())
        runways[(apt, month)] = g["RUNWAY_mvt"].value_counts(normalize=True)
        stats[(apt, month)]["_stands"] = g["STAND_mvt"]
        stats[(apt, month)]["_types"] = g["type"]
    # unseen shares (2025 month: against the other eleven; 2026: against all twelve) and runway distances
    raw25 = {}
    for f in sorted(glob.glob(str(ROOT / "data/raw/training_2025-*.parquet"))):
        dd = departures(pd.read_parquet(f, columns=COLS))
        own = Path(f).name[9:16]
        for apt, g in dd[dd["month"].eq(own)].groupby("ADEP_mvt"):
            raw25[(apt, own)] = (g["STAND_mvt"], g["type"])
        del dd
    pooled_rw = {}
    for apt in AIRPORTS:
        rows = [runways[(apt, m)] * stats[(apt, m)]["departures"] for m in months25 if (apt, m) in runways]
        tot = pd.concat(rows, axis=1).fillna(0).sum(axis=1)
        pooled_rw[apt] = tot / tot.sum()
        for m in months25:
            st_other = set().union(*[stands[(apt, o)] for o in months25 if o != m and (apt, o) in stands])
            ty_other = set().union(*[types[(apt, o)] for o in months25 if o != m and (apt, o) in types])
            s_col, t_col = raw25[(apt, m)]
            stats[(apt, m)]["unseen_stand"] = float((~s_col.dropna().isin(st_other)).sum() / len(s_col))
            stats[(apt, m)]["unseen_type"] = float((~t_col.dropna().isin(ty_other)).sum() / len(t_col))
            stats[(apt, m)]["runway_tvd_pooled"] = tvd(runways[(apt, m)], pooled_rw[apt])
        st_all = set().union(*[stands[(apt, o)] for o in months25 if (apt, o) in stands])
        ty_all = set().union(*[types[(apt, o)] for o in months25 if (apt, o) in types])
        for m26 in PAIRS:
            s = stats[(apt, m26)]
            s_col, t_col = s.pop("_stands"), s.pop("_types")
            s["unseen_stand"] = float((~s_col.dropna().isin(st_all)).sum() / len(s_col))
            s["unseen_type"] = float((~t_col.dropna().isin(ty_all)).sum() / len(t_col))
            s["runway_tvd_pooled"] = tvd(runways[(apt, m26)], pooled_rw[apt])
            s["runway_tvd_analogue"] = tvd(runways[(apt, m26)], runways[(apt, PAIRS[m26])])

    # cross-check: the repair detector's 2026 shares
    sh = pd.read_parquet(ROOT / a.shares).set_index(["airport", "month"])["share"]
    eobt_check = max(abs(stats[k]["eobt_later"] - sh[k]) for k in sh.index if k in stats)

    quantities = ["dep_per_day", "adsb_match", "adsb_gate", "nm_record", "nm_clock", "eobt_later", "lirf_pop_per_day",
                  "lirf_pop_long_per_day", "unseen_stand", "unseen_type", "runway_tvd_pooled", "y_mean", "y_p50", "y_p90", "y_p99",
                  "y_over3600"]
    rows, flags = [], []
    for apt in AIRPORTS:
        for q in quantities:
            if q not in stats[(apt, "2026-02")]:
                continue
            v25 = np.array([stats[(apt, m)][q] for m in months25 if q in stats[(apt, m)] and np.isfinite(stats[(apt, m)][q])])
            if len(v25) == 0:   # e.g. the gate share at LTFM, which has no ADS-B match in any month
                continue
            lo, hi = float(v25.min()), float(v25.max())
            board = np.nanmax([abs(stats[(apt, b)][q] - stats[(apt, PAIRS[b])][q]) for b in BOARD])
            for m26 in [*NEW, *BOARD]:
                v, a25 = stats[(apt, m26)][q], stats[(apt, PAIRS[m26])][q]
                moved = abs(v - a25)
                outside = bool(np.isfinite(v) and (v < lo or v > hi))
                flag = bool(m26 in NEW and outside and np.isfinite(moved) and moved > board)
                row = {"airport": apt, "quantity": q, "month": m26, "value": v, "analogue_2025": a25, "moved": moved,
                       "range_2025": [lo, hi], "outside_2025_range": outside, "board_months_moved_max": board, "flag": flag}
                if q == "runway_tvd_pooled":
                    row["runway_tvd_analogue"] = stats[(apt, m26)]["runway_tvd_analogue"]
                rows.append(row)
                if flag:
                    flags.append(row)
    for r in flags:
        print(f"FLAG {r['airport']} {r['month']} {r['quantity']}: {r['value']:.4g} (2025 {r['analogue_2025']:.4g}, range "
              f"{r['range_2025'][0]:.4g}..{r['range_2025'][1]:.4g}; moved {r['moved']:.4g} vs board max {r['board_months_moved_max']:.4g})")
    print("flags:", len(flags), "| eobt_later vs repair/shares.parquet max |diff|:", eobt_check)
    write_json(ROOT / a.out, clean({"created_at_utc": utc_now(), "final": a.final, "final_sha256": sha256_file(ROOT / a.final),
                              "raw_sha256": sha256_file(ROOT / a.raw), "learner_adsb_sha256": sha256_file(ROOT / a.learner_adsb),
                              "rule": "flag a Feb/Jun 2026 value outside the twelve 2025 months' range AND further from its 2025 analogue "
                                      "than both board months (Jan/Jul 2026) moved from theirs",
                              "eobt_later_check_vs_repair_shares": eobt_check, "flags": flags, "rows": rows,
                              "per_airport_month": {f"{k[0]} {k[1]}": {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                                                    for k, v in sorted(stats.items())}}))


if __name__ == "__main__":
    main()
