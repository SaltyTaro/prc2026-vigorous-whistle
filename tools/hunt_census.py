"""Fork M (rank-1 hunt): label-free class census of departures by ADS-B match state and the shipped layers that pin them.
Light table joins only (matches2 tables, raw files, chain step files). Writes runs/hunt/census_{rank,dev}.parquet."""
from __future__ import annotations
import glob, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
OUT = ROOT / "runs/hunt"
STEPS = ROOT / "runs/ctx_ownpoints_weight/rank/dry_v66/steps"
COLS = [ID, "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "CALLSIGN_flt", "FLIGHT_mvt", "STAND_mvt", "ADEP_flt", "AOBT_3_flt", "LOBT_flt"]


def deps(paths):
    raw = pd.concat([pd.read_parquet(p, columns=COLS) for p in paths], ignore_index=True)
    raw = raw[raw["PHASE_mvt"].eq("DEP")].drop_duplicates(ID)
    raw["month"] = raw["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m")
    raw["nm"] = raw["ADEP_flt"].notna()
    raw["cs"] = raw["CALLSIGN_flt"].notna()
    raw["clock"] = raw["AOBT_3_flt"].notna() | raw["LOBT_flt"].notna()
    return raw


def matches(months):
    m = pd.concat([pd.read_parquet(ROOT / f"data/external/adsblol/matches/matches2_{x}.parquet", columns=[ID, "gate", "icao", "segment_start", "motion_start", "takeoff", "at_known_stand", "first_dist_stand_m", "points"]) for x in months])
    return m.drop_duplicates(ID)


def rank():
    d = deps([ROOT / "data/raw/ranking.parquet"])
    m = matches(["2026-01", "2026-07"])
    d = d.merge(m, on=ID, how="left")
    f = {k: pd.read_parquet(STEPS / p).set_index(ID)[T].reindex(d[ID]).to_numpy(float) for k, p in
         [("mp", "model_part/vigorous-whistle_v66.parquet"), ("rm", "rematch/base/vigorous-whistle_v55m_rematch.parquet"),
          ("st", "stack/vigorous-whistle_v66.parquet"), ("id", "identity/vigorous-whistle_v66.parquet"), ("R", "R/vigorous-whistle_v66.parquet"),
          ("own", "own/vigorous-whistle_v66.parquet"), ("om2", "om2/vigorous-whistle_v66.parquet")]}
    d["L_rematch"] = f["mp"] != f["rm"]
    # chain order (docs/final_file/v66_chain.md): R = v48, X = extreme-day, Y = var-alloc, stack, identity = intruder bound +
    # identity extension, own = own-motion v1, om2 = own-motion v2
    d["L_intruder"] = f["id"] != f["st"]
    d["L_own1"] = f["own"] != f["id"]
    d["L_own2"] = f["om2"] != f["own"]
    own = pd.read_parquet(ROOT / "runs/stack_resid/v3_features/rank.parquet").set_index(ID).reindex(d[ID])
    d["own_n"] = own["p_n"].to_numpy(); d["own_inb"] = own["p_inb_rel"].notna().to_numpy()
    return d


def dev():
    files = sorted(glob.glob(str(ROOT / "data/raw/training_2025-*.parquet")))
    d = deps([Path(f) for f in files])
    p = pd.read_parquet(ROOT / "runs/opdi_stack_lszh/dev_v68_proxy.parquet")
    d = p.merge(d, on=ID, how="left", suffixes=("", "_raw"))
    m = matches(["2025-01", "2025-07", "2025-11", "2025-12"])
    d = d.merge(m, on=ID, how="left")
    own = pd.read_parquet(ROOT / "runs/stack_resid/v3_features/dev.parquet").set_index(ID).reindex(d[ID])
    d["own_n"] = own["p_n"].to_numpy(); d["own_inb"] = own["p_inb_rel"].notna().to_numpy()
    d["se"] = (d["label"] - d["value"]) ** 2
    return d


def klass(d):
    k = np.where(d["gate"].eq(True), "gate", np.where(d["gate"].eq(False), "dark", "unm"))
    return pd.Series(k, index=d.index)


if __name__ == "__main__":
    r, v = rank(), dev()
    for x in (r, v):
        x["k"] = klass(x)
    r.to_parquet(OUT / "census_rank.parquet", index=False)
    v.to_parquet(OUT / "census_dev.parquet", index=False)
    print("rank", len(r), "dev", len(v))
