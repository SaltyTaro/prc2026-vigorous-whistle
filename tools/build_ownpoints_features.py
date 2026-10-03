"""Own-aircraft turnaround ADS-B features for the context network (dig tunnel 21; the label-free features of dig tunnel 18,
docs/dig/pregap.md, built for every month and the ranking file).

Identity: nearest adsb.lol point of the same callsign (CALLSIGN_flt, else the no-NM rematch key map runs/rematch_nonm/key_map.parquet)
at the same airport within 600 s of the take-off / landing. Inbound = the latest identified arrival of the same ICAO address at the
same airport with in-block 600 s .. 18 h before take-off. Own points in [in-block + 120 s, take-off + 120 s]; times are seconds
before take-off. No departure label or departure block time is read (BLOCK_TIME_UTC_mvt is read for ARRIVAL rows only, as supplied in
the ranking file).

Columns (15): p_has_inb, p_inb_rel, p_tow, p_n, p_first_any, p_first_mv, p_last_park, p_park10, p_park20, p_park40, p_pregap_n,
p_pregap_last, p_pregap_maxdist, p_gap_len, p_seg_start.

Writes OUT/features_v19_<month>.parquet = the v19 extra table of that month (data/external/adsblol/features_v19) with the 15 columns
appended (left join on MVT_ID_mvt; rows without an identified inbound: p_has_inb 0, the rest NaN), and OUT/manifest.json.
Usage: tools/build_ownpoints_features.py [--out data/external/adsblol/features_v19own]
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

ID = "MVT_ID_mvt"
SURF = ROOT / "data/external/adsblol/surface"
TOL = 600
COLS = [ID, "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "STAND_mvt", "CALLSIGN_flt", "FLIGHT_mvt"]
OWN = ["p_has_inb", "p_inb_rel", "p_tow", "p_n", "p_first_any", "p_first_mv", "p_last_park", "p_park10", "p_park20", "p_park40",
       "p_pregap_n", "p_pregap_last", "p_pregap_maxdist", "p_gap_len", "p_seg_start"]


def load(paths, key_map):
    mv = pd.concat([pd.read_parquet(p, columns=COLS) for p in paths], ignore_index=True)
    arr = mv.PHASE_mvt.eq("ARR")
    mv.loc[~arr, "BLOCK_TIME_UTC_mvt"] = pd.NaT  # departure block times are never read
    mv["airport"] = np.where(arr, mv.ADES_mvt, mv.ADEP_mvt)
    mv = mv.merge(key_map, on=["airport", "FLIGHT_mvt"], how="left")
    mv["cs"] = mv.CALLSIGN_flt.fillna(mv.key).str.upper().str.strip()
    mv["t"] = (mv.MVT_TIME_UTC_mvt - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
    mv["day"] = mv.MVT_TIME_UTC_mvt.dt.strftime("%Y%m%d")
    return mv


def surf(day, cols):
    path = SURF / f"surface_{day}.parquet"
    return pd.read_parquet(path, columns=cols) if path.exists() else None


def identify(mv):
    out = []
    for day, g in mv[mv.cs.notna()].groupby("day"):
        x = surf(day, ["icao", "airport", "time", "callsign"])
        if x is None:
            continue
        x = x[x.callsign.notna()].copy()
        x["cs"] = x.callsign.str.upper().str.strip()
        m = pd.merge_asof(g.sort_values("t")[[ID, "PHASE_mvt", "airport", "cs", "t"]], x.sort_values("time")[["airport", "cs", "time", "icao"]],
                          left_on="t", right_on="time", by=["airport", "cs"], direction="nearest", tolerance=TOL)
        out.append(m[[ID, "PHASE_mvt", "icao"]])
    return mv.merge(pd.concat(out), on=[ID, "PHASE_mvt"], how="left")


def dist_m(lat1, lon1, lat2, lon2):
    k = np.pi / 180
    x = (lon2 - lon1) * k * np.cos(0.5 * (lat1 + lat2) * k)
    y = (lat2 - lat1) * k
    return 6371000.0 * np.sqrt(x * x + y * y)


def features(mv):
    arr = mv[(mv.PHASE_mvt == "ARR") & mv.icao.notna() & mv.BLOCK_TIME_UTC_mvt.notna()].copy()
    arr["inb"] = (arr.BLOCK_TIME_UTC_mvt - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
    arr = arr.rename(columns={"STAND_mvt": "inb_stand"})[["airport", "icao", "inb", "inb_stand"]].sort_values("inb")
    dep = mv[(mv.PHASE_mvt == "DEP") & mv.icao.notna()].copy()
    dep["ref"] = dep.t - 600.0
    dep = pd.merge_asof(dep.sort_values("ref"), arr, left_on="ref", right_on="inb", by=["airport", "icao"], direction="backward")
    dep["p_inb_rel"] = dep.t - dep.inb
    dep = dep[dep.p_inb_rel.between(600, 18 * 3600)].copy()
    dep["p_tow"] = (dep.inb_stand.astype(str) != dep.STAND_mvt.astype(str)).astype(float)
    dep["lo"] = dep.inb + 120.0
    res = []
    for day, g in dep.groupby("day"):
        prev = (pd.Timestamp(day) - pd.Timedelta(days=1)).strftime("%Y%m%d")
        cols = ["icao", "airport", "time", "ground", "gs", "lat", "lon"]
        pts = [p for p in (surf(prev, cols), surf(day, cols)) if p is not None]
        if not pts:
            continue
        p = pd.concat(pts)
        p = p[p.icao.isin(set(g.icao))]
        j = g[[ID, "airport", "icao", "t", "lo"]].merge(p, on=["airport", "icao"])
        j = j[(j.time > j.lo) & (j.time <= j.t + 120)].sort_values([ID, "time"])
        if j.empty:
            continue
        j["rel"] = j.t - j.time
        j["park"] = j.ground & (j.gs.fillna(0) < 1)
        j["mv"] = j.ground & (j.gs.fillna(0) >= 3)
        pk = j[j.park & (j.time <= j.lo + 1800)].groupby(ID)[["lat", "lon"]].median()
        fp = j.groupby(ID)[["lat", "lon"]].first()
        P = fp.copy()
        P.loc[pk.index] = pk
        j = j.merge(P.rename(columns={"lat": "plat", "lon": "plon"}), left_on=ID, right_index=True)
        j["d"] = dist_m(j.plat, j.plon, j.lat, j.lon)
        pre = j.rel >= 60
        j["dt"] = j.groupby(ID).time.diff()
        j["gap"] = (j.dt > 600).astype(int)
        j["gk"] = j.groupby(ID).gap.cumsum()
        j["gmax"] = j.groupby(ID).gk.transform("max")
        j["pregap"] = (j.gk < j.gmax) & (j.gmax > 0)
        a = pd.DataFrame(index=j[ID].unique())
        jp = j[pre]
        a["p_n"] = jp.groupby(ID).size()
        a["p_first_any"] = jp.groupby(ID).rel.max()
        a["p_first_mv"] = jp[jp.mv].groupby(ID).rel.max()
        a["p_last_park"] = jp[jp.park].groupby(ID).rel.min()
        for T in (600, 1200, 2400):
            q = jp[jp.park & (jp.d < 150) & (jp.rel <= T)]
            a[f"p_park{T // 60}"] = 0.0
            a.loc[q[ID].unique(), f"p_park{T // 60}"] = 1.0
        jg = j[j.pregap]
        a["p_pregap_n"] = jg.groupby(ID).size()
        a["p_pregap_last"] = jg.groupby(ID).rel.min()
        a["p_pregap_maxdist"] = jg.groupby(ID).d.max()
        a["p_gap_len"] = j[(j.gap == 1) & (j.gk == j.gmax)].groupby(ID).dt.first()
        a["p_seg_start"] = j[~j.pregap].groupby(ID).rel.max()
        a["p_pregap_n"] = a.p_pregap_n.fillna(0.0)
        res.append(a)
    f = pd.concat(res)
    f.index.name = ID
    out = dep.set_index(ID)[["p_inb_rel", "p_tow"]].join(f, how="left")
    out["p_n"] = out.p_n.fillna(0.0)
    out["p_has_inb"] = 1.0
    return out.reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/external/adsblol/features_v19own"))
    parser.add_argument("--v19", type=Path, default=Path("data/external/adsblol/features_v19"))
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    key_map = pd.read_parquet(ROOT / "runs/rematch_nonm/key_map.parquet")[["ADEP_mvt", "FLIGHT_mvt", "key"]].rename(columns={"ADEP_mvt": "airport"})
    blocks = {"training": sorted(glob.glob(str(ROOT / "data/raw/training_2025-*.parquet"))), "ranking": [str(ROOT / "data/raw/ranking.parquet")]}
    own = []
    stats = {}
    for name, paths in blocks.items():
        mv = identify(load(paths, key_map))
        f = features(mv)
        dep = mv[mv.PHASE_mvt == "DEP"]
        stats[name] = {"departures": int(len(dep)), "identified_share_by_airport": dep.groupby("airport").icao.apply(lambda s: round(float(s.notna().mean()), 4)).to_dict(),
                       "inbound_rows": int(len(f)), "nonnull_share_among_inbound": {c: round(float(f[c].notna().mean()), 4) for c in OWN if c in f}}
        print(name, stats[name], flush=True)
        own.append(f[[ID, *OWN]])
        del mv
    own = pd.concat(own, ignore_index=True)
    if own[ID].duplicated().any():
        raise ValueError("Duplicate departures across blocks.")
    files = {}
    for path in sorted((ROOT / args.v19).glob("features_v19_*.parquet")):
        v19 = pd.read_parquet(path)
        t = v19.merge(own, on=ID, how="left", validate="one_to_one")
        t["p_has_inb"] = t["p_has_inb"].fillna(0.0)
        t[OWN] = t[OWN].astype("float64")
        target = out / path.name
        t.to_parquet(target, index=False, compression="zstd")
        files[path.name] = {"rows": int(len(t)), "sha256": sha256_file(target), "v19_sha256": sha256_file(path),
                            "inbound_share": round(float(t["p_has_inb"].mean()), 4)}
        print(path.name, files[path.name], flush=True)
    write_json(out / "manifest.json", {"created_at_utc": utc_now(), "tool": "tools/build_ownpoints_features.py", "tool_sha256": sha256_file(Path(__file__)),
                                        "own_columns": OWN, "v19_columns": [c for c in pd.read_parquet(sorted((ROOT / args.v19).glob('*.parquet'))[0]).columns if c != ID],
                                        "key_map_sha256": sha256_file(ROOT / "runs/rematch_nonm/key_map.parquet"), "stats": stats, "files": files,
                                        "label_free": True, "note": "departure labels and departure block times are never read"})


if __name__ == "__main__":
    main()
