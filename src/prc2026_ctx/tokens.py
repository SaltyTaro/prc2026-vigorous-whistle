"""Context tokens: every movement of a supplied file, and each departure's neighbours at the same airport.

A token is one movement (DEP or ARR) at its airport (ADEP for departures, ADES for arrivals). Label-free contract:
- departures carry their takeoff, schedule and NM clocks (AOBT_3, EOBT_1) and, when matched, their own ADS-B surface segment
  (motion start, first point, stopped time, gate class); a departure token NEVER carries BLOCK_TIME_UTC_mvt or TAXITIME_SEC_mvt
  (those are the hidden labels in the ranking file and are dropped for every departure, in every file);
- arrivals carry their landing, schedule, in-block time and taxi-in, which the supplied ranking file retains for arrivals.
Neighbours of a departure: movements at the same airport whose movement time lies in [t0 - 60 min, t0 + 30 min] (t0 = the
departure's takeoff), the departure itself excluded, at most K, a contiguous run of the time-sorted tokens centred on t0 - 15 min.
Later movements are included (retrospective task; docs/next_lever/timeline_availability.json reviewed later peer events).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ID = "MVT_ID_mvt"
EPOCH = pd.Timestamp(0, tz="UTC")
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
WAKE = ["H", "M", "L", "J"]
TIME_COLUMNS = ["t_mvt", "t_aobt", "t_eobt", "t_sched", "t_inblock", "t_motion", "t_seg"]
VALUE_COLUMNS = ["is_arr", "taxi_in", "adsb_matched", "adsb_gate", "adsb_stopped"]
CODE_COLUMNS = ["runway", "stand", "group", "wake"]
RAW_COLUMNS = [ID, "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt",
               "RUNWAY_mvt", "STAND_mvt", "AOBT_3_flt", "EOBT_1_flt", "WK_TBL_CAT_flt"]
BEFORE_S, AFTER_S, CENTRE_S = 3600.0, 1800.0, -900.0
STAND_GROUP = re.compile(r"^([A-Za-z]+)|^(\d)")


def seconds(series: pd.Series) -> np.ndarray:
    return (pd.to_datetime(series, utc=True) - EPOCH).dt.total_seconds().to_numpy(dtype=float)


def stand_group(stand) -> str | None:
    if not isinstance(stand, str):
        return None
    match = STAND_GROUP.match(stand.strip())
    return (match.group(1) or match.group(2)).upper() if match else None


def build_tokens(raw: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """One token per movement, sorted by (airport, t_mvt, id). `raw` has RAW_COLUMNS; `matches` is a pass-2 match table."""
    dep = raw["PHASE_mvt"].eq("DEP").to_numpy()
    arr = raw["PHASE_mvt"].eq("ARR").to_numpy()
    keep = dep | arr
    raw, dep, arr = raw.loc[keep].reset_index(drop=True), dep[keep], arr[keep]
    out = pd.DataFrame({ID: raw[ID].to_numpy(dtype=float), "airport": np.where(dep, raw["ADEP_mvt"], raw["ADES_mvt"])})
    out["is_dep"] = dep
    out["t_mvt"] = seconds(raw["MVT_TIME_UTC_mvt"])
    out["t_sched"] = seconds(raw["SCHED_TIME_UTC_mvt"])
    out["t_aobt"] = np.where(dep, seconds(raw["AOBT_3_flt"]), np.nan)
    out["t_eobt"] = np.where(dep, seconds(raw["EOBT_1_flt"]), np.nan)
    # Labels never enter a departure token: block time and taxi time are kept for arrivals only.
    out["t_inblock"] = np.where(arr, seconds(raw["BLOCK_TIME_UTC_mvt"]), np.nan)
    out["taxi_in"] = np.where(arr, raw["TAXITIME_SEC_mvt"].to_numpy(dtype=float), np.nan)
    out["is_arr"] = arr.astype(np.float32)
    m = matches[[ID, "gate", "motion_start", "segment_start", "taxi_stopped_s"]].drop_duplicates(ID)
    out = out.merge(m, on=ID, how="left", validate="many_to_one")
    matched = out["motion_start"].notna().to_numpy() & out["is_dep"].to_numpy()
    out["t_motion"] = np.where(matched, out["motion_start"], np.nan)
    out["t_seg"] = np.where(matched, out["segment_start"], np.nan)
    out["adsb_matched"] = np.where(out["is_dep"], matched.astype(float), np.nan)
    out["adsb_gate"] = np.where(matched, out["gate"].astype(float), np.nan)
    out["adsb_stopped"] = np.where(matched, out["taxi_stopped_s"], np.nan)
    out["runway_name"] = raw["RUNWAY_mvt"].to_numpy()
    out["stand_name"] = raw["STAND_mvt"].to_numpy()
    out["wake"] = pd.Categorical(raw["WK_TBL_CAT_flt"].to_numpy(), categories=WAKE).codes.astype(np.int32) + 1  # 0 = missing/other
    out = out.drop(columns=["gate", "motion_start", "segment_start", "taxi_stopped_s"])
    return out.sort_values(["airport", "t_mvt", ID], kind="mergesort").reset_index(drop=True)


def encode_codes(tokens: pd.DataFrame) -> pd.DataFrame:
    """Per-airport integer codes for runway, stand and stand group (-1 = missing); only equality with the focal row is ever used."""
    out = tokens.copy()
    for column, source in [("runway", out["runway_name"]), ("stand", out["stand_name"]), ("group", out["stand_name"].map(stand_group))]:
        key = out["airport"].astype(str) + "|" + source.astype("string").fillna("")
        codes = pd.Categorical(key).codes.astype(np.int32)
        out[column] = np.where(source.notna().to_numpy(), codes, -1).astype(np.int32)
    return out


def neighbours(tokens: pd.DataFrame, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(departure ids, their token rows, (n_dep, k) int32 neighbour token rows padded with -1)."""
    rows = np.flatnonzero(tokens["is_dep"].to_numpy())
    t = tokens["t_mvt"].to_numpy()
    airport = tokens["airport"].to_numpy()
    index = np.full((len(rows), k), -1, dtype=np.int32)
    for span in (np.flatnonzero(airport == a) for a in pd.unique(airport)):
        lo_a, hi_a = span[0], span[-1] + 1  # tokens are sorted by airport, so each airport is one contiguous block
        mine = rows[(rows >= lo_a) & (rows < hi_a)]
        if len(mine) == 0:
            continue
        ta = t[lo_a:hi_a]
        t0 = t[mine]
        lo = lo_a + np.searchsorted(ta, t0 - BEFORE_S, side="left")
        hi = lo_a + np.searchsorted(ta, t0 + AFTER_S, side="right")
        centre = lo_a + np.searchsorted(ta, t0 + CENTRE_S, side="left")
        start = np.clip(centre - (k + 1) // 2, lo, np.maximum(lo, hi - (k + 1)))
        cand = start[:, None] + np.arange(k + 1)[None, :]
        ok = (cand < hi[:, None]) & (cand != mine[:, None])
        order = np.argsort(~ok, axis=1, kind="stable")[:, :k]  # valid candidates first, in time order
        picked = np.take_along_axis(cand, order, axis=1)
        valid = np.take_along_axis(ok, order, axis=1)
        where = np.searchsorted(rows, mine)
        index[where] = np.where(valid, picked, -1).astype(np.int32)
    return tokens[ID].to_numpy()[rows], rows.astype(np.int64), index


def save_file(directory: Path, name: str, tokens: pd.DataFrame, dep_ids: np.ndarray, self_rows: np.ndarray, index: np.ndarray) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / f"{name}.npz", ids=tokens[ID].to_numpy(dtype=float),
             times=tokens[TIME_COLUMNS].to_numpy(dtype=np.float64), values=tokens[VALUE_COLUMNS].to_numpy(dtype=np.float32),
             codes=tokens[CODE_COLUMNS].to_numpy(dtype=np.int32), dep_ids=dep_ids.astype(float), self_rows=self_rows, neighbours=index)


class TokenStore:
    """Concatenated tokens of several files, with a departure-id -> (self row, neighbour rows) lookup in global token rows."""

    def __init__(self, directory: Path, names: list[str]):
        times, values, codes, dep_ids, self_rows, index = [], [], [], [], [], []
        offset = 0
        for name in names:
            z = np.load(directory / f"{name}.npz")
            times.append(z["times"]); values.append(z["values"]); codes.append(z["codes"])
            dep_ids.append(z["dep_ids"]); self_rows.append(z["self_rows"] + offset)
            nb = z["neighbours"]
            index.append(np.where(nb >= 0, nb + offset, -1).astype(np.int32))
            offset += len(z["ids"])
        self.times = np.concatenate(times)
        self.values = np.concatenate(values)
        self.codes = np.concatenate(codes)
        self.dep_index = pd.Index(np.concatenate(dep_ids))
        if self.dep_index.has_duplicates:
            raise ValueError("A departure id occurs in two token files.")
        self.self_rows = np.concatenate(self_rows)
        self.neighbour_rows = np.concatenate(index)

    def lookup(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(self token rows, neighbour token rows) for the given departure ids, in their order."""
        where = self.dep_index.get_indexer(np.asarray(ids, dtype=float))
        if (where < 0).any():
            raise ValueError(f"{int((where < 0).sum())} departures have no context tokens.")
        return self.self_rows[where], self.neighbour_rows[where]
