"""Fork U probe: does the ADS-B squawk / callsign timeline at the stand (extracted by tools/adsb_extract_day.py, never used) carry
off-block timing beyond the motion start? Label-free event extraction for given days; labels read only for the 2025 diagnostic.

  python tools/unused_squawk_probe.py --days 20250115 20250715 --out runs/unused/squawk_probe.parquet

Per departure (identified aircraft, ids.parquet): points of its ICAO address at the departure airport in [take-off - 4 h, take-off];
first point, first moving point (gs >= 3 kt, on ground), last parked point before it, the time the squawk last changes before the first
motion (and the codes), and the time the broadcast callsign first equals CALLSIGN_flt.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ID = "MVT_ID_mvt"


def departures(day: str) -> pd.DataFrame:
    ym = f"{day[:4]}-{day[4:6]}"
    files = [f for f in glob.glob(str(ROOT / "data/raw/training_2025-*.parquet")) if f"_{ym}-01_" in f.replace("\\", "/")]
    src = files[0] if files else str(ROOT / "data/raw/ranking.parquet")
    cols = [ID, "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "CALLSIGN_flt", "STAND_mvt", "AOBT_3_flt"]
    d = pd.read_parquet(src, columns=cols)
    d = d[d["PHASE_mvt"].eq("DEP")]
    t = pd.to_datetime(d["MVT_TIME_UTC_mvt"], utc=True)
    return d[t.dt.strftime("%Y%m%d").eq(day)].copy()


def events(day: str) -> pd.DataFrame:
    dep = departures(day)
    ids = pd.read_parquet(ROOT / "runs/intruder_identity/ids.parquet", columns=[ID, "PHASE_mvt", "icao"])
    ids = ids[ids["PHASE_mvt"].eq("DEP")].drop_duplicates(ID)
    dep = dep.merge(ids[[ID, "icao"]], on=ID, how="inner")
    pts = []
    for d in {day, (pd.Timestamp(day) - pd.Timedelta(days=1)).strftime("%Y%m%d")}:
        p = ROOT / f"data/external/adsblol/surface/surface_{d}.parquet"
        if p.exists():
            pts.append(pd.read_parquet(p, columns=["icao", "airport", "time", "ground", "gs", "callsign", "squawk", "on_ground_flag"]))
    pts = pd.concat(pts, ignore_index=True)
    pts = pts[pts["icao"].isin(set(dep["icao"]))].sort_values("time")
    by = {k: g for k, g in pts.groupby(["icao", "airport"], sort=False)}
    rows = []
    for r in dep.itertuples(index=False):
        tk = pd.Timestamp(r.MVT_TIME_UTC_mvt).timestamp()
        g = by.get((r.icao, r.ADEP_mvt))
        out = {ID: getattr(r, ID), "airport": r.ADEP_mvt, "takeoff": tk,
               "label_offblock": pd.Timestamp(r.BLOCK_TIME_UTC_mvt).timestamp() if pd.notna(r.BLOCK_TIME_UTC_mvt) else np.nan,
               "aobt3": pd.Timestamp(r.AOBT_3_flt).timestamp() if pd.notna(r.AOBT_3_flt) else np.nan}
        if g is not None:
            g = g[(g["time"] >= tk - 4 * 3600) & (g["time"] <= tk)]
        if g is None or g.empty:
            rows.append(out)
            continue
        ground = g["ground"].fillna(False).to_numpy() | g["on_ground_flag"].fillna(False).to_numpy()
        gs = g["gs"].to_numpy(dtype=float)
        tm = g["time"].to_numpy(dtype=float)
        mv = np.flatnonzero(ground & (gs >= 3))
        out.update({"n_points": len(g), "t_first": tm[0], "t_motion": tm[mv[0]] if len(mv) else np.nan})
        before = tm < (out["t_motion"] if len(mv) else tk)
        park = before & ground & ~(gs >= 3)
        out["n_parked"] = int(park.sum())
        out["t_last_park"] = tm[park][-1] if park.any() else np.nan
        sq = g["squawk"].to_numpy(dtype=object)
        sqb = [(t, s) for t, s, b in zip(tm, sq, before) if b and isinstance(s, str) and s]
        if sqb:
            out["sq_first"] = sqb[0][1]
            out["sq_last"] = sqb[-1][1]
            ch = [sqb[i][0] for i in range(1, len(sqb)) if sqb[i][1] != sqb[i - 1][1]]
            out["t_sq_change"] = ch[-1] if ch else np.nan
            out["n_sq_codes"] = len({s for _, s in sqb})
        cs = g["callsign"].to_numpy(dtype=object)
        want = r.CALLSIGN_flt.strip().upper() if isinstance(r.CALLSIGN_flt, str) else ""
        hit = [t for t, c in zip(tm, cs) if isinstance(c, str) and want and c.strip().upper() == want]
        out["t_callsign_set"] = hit[0] if hit else np.nan
        csb = [(t, c.strip().upper()) for t, c, b in zip(tm, cs, before) if b and isinstance(c, str) and c.strip()]
        chc = [csb[i][0] for i in range(1, len(csb)) if csb[i][1] != csb[i - 1][1]]
        out["t_cs_change"] = chc[-1] if chc else np.nan
        rows.append(out)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    res = pd.concat([events(d).assign(day=d) for d in a.days], ignore_index=True)
    res.to_parquet(out, index=False)
    print(len(res), "rows ->", out)


if __name__ == "__main__":
    main()
