"""Peer congestion per departure (28 September 2026): label-free, from the NM clocks of the other departures in the same airport-hour.

peer_c = mean over the airport-hour's departures of the NM proxy (MVT_TIME - AOBT_3, kept when 0-7200 s) minus the median of those hourly
means over the airport's calendar month; peer_n = the number of proxies in the hour. no_clock = the row's own AOBT_3 is missing.
Only the movement file's own fields are read (no label, no block time), so the same function serves the 2025 files and the ranking.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.features import ID  # noqa: E402

COLUMNS = [ID, "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"]
PROXY_RANGE = (0.0, 7200.0)


def peer_table(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per departure: ID, airport, hour, no_clock, peer_c, peer_n."""
    dep = raw[raw["PHASE_mvt"].eq("DEP")].drop_duplicates(ID)
    t = pd.to_datetime(dep["MVT_TIME_UTC_mvt"])
    out = pd.DataFrame({ID: dep[ID].to_numpy(), "airport": dep["ADEP_mvt"].to_numpy(), "hour": t.dt.floor("h").to_numpy(),
                        "month": t.dt.strftime("%Y-%m").to_numpy(), "proxy": (t - pd.to_datetime(dep["AOBT_3_flt"])).dt.total_seconds().to_numpy()})
    out["no_clock"] = np.isnan(out["proxy"].to_numpy())
    ok = out["proxy"].between(*PROXY_RANGE)
    hourly = out[ok].groupby(["airport", "month", "hour"])["proxy"].agg(peer_mean="mean", peer_n="size").reset_index()
    hourly["peer_c"] = hourly["peer_mean"] - hourly.groupby(["airport", "month"])["peer_mean"].transform("median")
    out = out.merge(hourly[["airport", "month", "hour", "peer_c", "peer_n"]], on=["airport", "month", "hour"], how="left")
    out["peer_n"] = out["peer_n"].fillna(0).astype(int)
    return out[[ID, "airport", "hour", "no_clock", "peer_c", "peer_n"]]


def training_peers() -> pd.DataFrame:
    return peer_table(pd.concat([pd.read_parquet(p, columns=COLUMNS) for p in sorted(glob.glob(str(ROOT / "data/raw/training_*.parquet")))], ignore_index=True))


def ranking_peers() -> pd.DataFrame:
    return peer_table(pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=COLUMNS))
