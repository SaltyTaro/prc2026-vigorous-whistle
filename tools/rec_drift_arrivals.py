"""Recording-drift screen, step 1 (docs/rec_drift/): every supplied arrival matched to its adsb.lol surface track, label-free.

For each ARR row (2025 training files and the ranking file) the aircraft is identified by callsign (CALLSIGN_flt, else FLIGHT_mvt, as
the departure matcher) at the arrival airport: the candidate ICAO whose air-to-ground transition lies nearest the recorded landing
time (within 300 s) is kept. From that touchdown to the next take-off of the same ICAO (at most 2 h later) two ADS-B stop times are
derived:

  A (known stand)  first ground point within 40 m of the learned stand position (data/external/adsblol/matches/stands.parquet,
                   the departure matcher's anchors) with ground speed <= 1 kt;
  B (parked run)   first ground point with gs <= 1 kt from which the aircraft stays within 30 m for the next 900 s (at least one
                   later point, >= 60 s later) - a taxiway hold moves on, a parked aircraft does not.

For each stop the gap to the previous point is kept (the stop is only sharp when the aircraft was seen moving just before). The
recorded in-block (BLOCK_TIME_UTC_mvt of the ARR row) minus the stop is the arrival recording offset. No departure label is read.

Runs on an Azure worker over the day files it holds:  rec_drift_arrivals.py --out runs/rec_drift/arr_<host>.parquet [--workers 14]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import multiprocessing as mp
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SURFACE = ROOT / "data/external/adsblol/surface"
STANDS = ROOT / "runs/rec_drift/stands.parquet"   # copy of data/external/adsblol/matches/stands.parquet (hash in the receipt)
EPOCH = pd.Timestamp(0, tz="UTC")
COLS = ["MVT_ID_mvt", "PHASE_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "STAND_mvt", "CALLSIGN_flt", "FLIGHT_mvt",
        "AIRCRAFT_TYPE_mvt"]
TD_TOL, WINDOW = 300.0, 7200.0
A_RADIUS, B_RADIUS, B_HOLD, GS_STOP = 40.0, 30.0, 900.0, 1.0


def metres(lat1, lon1, lat2, lon2):
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1) * np.cos(np.radians((lat1 + lat2) / 2))
    return 6371000.0 * np.sqrt(dlat ** 2 + dlon ** 2)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 23), b""):
            h.update(b)
    return h.hexdigest()


def arrivals() -> pd.DataFrame:
    files = sorted(glob.glob(str(ROOT / "data/raw/training_2025-*.parquet"))) + [str(ROOT / "data/raw/ranking.parquet")]
    parts = []
    for f in files:
        t = pd.read_parquet(f, columns=COLS)
        parts.append(t[t["PHASE_mvt"].eq("ARR")])
    a = pd.concat(parts, ignore_index=True).drop_duplicates("MVT_ID_mvt")
    a["L"] = (a["MVT_TIME_UTC_mvt"] - EPOCH).dt.total_seconds()
    a["inblk"] = (a["BLOCK_TIME_UTC_mvt"] - EPOCH).dt.total_seconds()
    a["day"] = a["MVT_TIME_UTC_mvt"].dt.strftime("%Y%m%d")
    a["cs"] = a["CALLSIGN_flt"].fillna(a["FLIGHT_mvt"]).fillna("").str.strip().str.upper()
    return a.rename(columns={"ADES_mvt": "airport", "STAND_mvt": "stand", "AIRCRAFT_TYPE_mvt": "type_mvt"})[
        ["MVT_ID_mvt", "airport", "day", "L", "inblk", "stand", "cs", "type_mvt"]]


def track_stops(t, lat, lon, gs, ground, L, stand):
    """Touchdown and stops on one ICAO track (arrays sorted by time). Returns dict or None when no touchdown near L."""
    air = ~ground
    idx = np.flatnonzero(ground[1:] & air[:-1]) + 1                       # air -> ground transitions
    idx = idx[(t[idx] >= L - TD_TOL) & (t[idx] <= L + TD_TOL) & (t[idx] - t[idx - 1] <= 60)]
    if not len(idx):
        return None
    i0 = int(idx[np.argmin(np.abs(t[idx] - L))])
    td = float(t[i0])
    later_air = np.flatnonzero(air & (t > td + 120))
    end = min(td + WINDOW, float(t[later_air[0]]) if len(later_air) else np.inf)
    sel = np.flatnonzero(ground & (t >= td) & (t < end))
    out = {"td": td, "seg_points": int(len(sel)), "seg_last": float(t[sel[-1]]) if len(sel) else np.nan,
           "next_air": float(t[later_air[0]]) if len(later_air) else np.nan}
    st, sla, slo, sgs = t[sel], lat[sel], lon[sel], np.nan_to_num(gs[sel], nan=0.0)
    prev = np.concatenate([[np.nan], np.diff(st)])
    # A: known stand
    out.update(stopA=np.nan, stopA_gap=np.nan, enterA=np.nan, stopA_dist=np.nan)
    if stand is not None and len(st):
        d = metres(stand[0], stand[1], sla, slo)
        k = np.flatnonzero((d <= A_RADIUS) & (st >= td + 30))
        if len(k):
            out["enterA"] = float(st[k[0]])
            k2 = k[sgs[k] <= GS_STOP]
            if len(k2):
                out["stopA"], out["stopA_gap"], out["stopA_dist"] = float(st[k2[0]]), float(prev[k2[0]]), float(d[k2[0]])
    # B: parked run
    out.update(stopB=np.nan, stopB_gap=np.nan, stopB_lat=np.nan, stopB_lon=np.nan, stopB_span=np.nan)
    cand = np.flatnonzero((sgs <= GS_STOP) & (st >= td + 30))
    for i in cand:
        w = np.flatnonzero((st > st[i]) & (st <= st[i] + B_HOLD))
        if not len(w):
            continue
        if np.all(metres(sla[i], slo[i], sla[w], slo[w]) <= B_RADIUS) and st[w[-1]] - st[i] >= 60:
            out.update(stopB=float(st[i]), stopB_gap=float(prev[i]), stopB_lat=float(sla[i]), stopB_lon=float(slo[i]),
                       stopB_span=float(st[w[-1]] - st[i]))
            break
    return out


def one_day(job):
    day, arr, stands = job
    nxt = (datetime.strptime(day, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")
    frames = [pd.read_parquet(p, columns=["icao", "airport", "time", "lat", "lon", "ground", "gs", "callsign", "type"])
              for p in (SURFACE / f"surface_{day}.parquet", SURFACE / f"surface_{nxt}.parquet") if p.exists()]
    have_next = (SURFACE / f"surface_{nxt}.parquet").exists()
    s = pd.concat(frames, ignore_index=True)
    s["callsign"] = s["callsign"].fillna("").str.strip().str.upper()
    rows = []
    for airport, g in arr.groupby("airport"):
        sa = s[s["airport"].eq(airport)]
        if sa.empty:
            continue
        by_cs = {c: f["icao"].unique() for c, f in sa[sa["callsign"].ne("")].groupby("callsign")}
        tracks = {}
        for icao, f in sa.groupby("icao"):
            f = f.sort_values("time")
            tracks[icao] = (f["time"].to_numpy(float), f["lat"].to_numpy(float), f["lon"].to_numpy(float), f["gs"].to_numpy(float),
                            f["ground"].to_numpy(bool), f["type"].iloc[-1])
        for r in g.itertuples(index=False):
            icaos = by_cs.get(r.cs)
            if icaos is None or not r.cs:
                continue
            stand = stands.get((airport, r.stand)) if isinstance(r.stand, str) else None
            best = None
            for icao in icaos:
                t, la, lo, gs, gr, typ = tracks[icao]
                info = track_stops(t, la, lo, gs, gr, r.L, stand)
                if info is not None and (best is None or abs(info["td"] - r.L) < abs(best["td"] - r.L)):
                    best = {**info, "icao": icao, "adsb_type": typ, "n_icao": len(icaos)}
            if best is not None:
                rows.append({"MVT_ID_mvt": r.MVT_ID_mvt, "airport": airport, "day": day, "L": r.L, "inblk": r.inblk, "stand": r.stand,
                             "type_mvt": r.type_mvt, "stand_known": stand is not None, "next_day_file": have_next, **best})
    return day, len(arr), pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    started = time.time()
    days = sorted(Path(p).name[8:16] for p in glob.glob(str(SURFACE / "surface_*.parquet")))
    arr = arrivals()
    arr = arr[arr["day"].isin(days)]
    st = pd.read_parquet(STANDS)
    stands = {(r.airport, r.stand): (r.lat, r.lon) for r in st.itertuples(index=False)}
    jobs = [(d, g.reset_index(drop=True), stands) for d, g in arr.groupby("day")]
    with mp.Pool(a.workers) as pool:
        res = pool.map(one_day, jobs, chunksize=1)
    out = pd.concat([r[2] for r in res if len(r[2])], ignore_index=True)
    target = ROOT / a.out
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(target, index=False)
    rec = {"written_at_utc": datetime.now(timezone.utc).isoformat(), "output": a.out, "output_sha256": sha256(target), "rows": int(len(out)),
           "arrivals_on_days": int(len(arr)), "days": len(jobs), "per_day": {r[0]: [r[1], int(len(r[2]))] for r in res},
           "stands_sha256": sha256(STANDS), "code_sha256": sha256(Path(__file__)), "seconds": round(time.time() - started, 1),
           "surface_sha256": {d: sha256(SURFACE / f"surface_{d}.parquet") for d in days}}
    Path(str(target).replace(".parquet", ".receipt.json")).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print({k: rec[k] for k in ["rows", "arrivals_on_days", "days", "seconds"]})


if __name__ == "__main__":
    main()
