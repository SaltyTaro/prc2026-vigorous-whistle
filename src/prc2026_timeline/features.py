"""Supplied flight clocks and other-flight surface events; no departure labels."""
from __future__ import annotations

import numpy as np
import pandas as pd

from prc2026.features import ID, PHASE, TIME
from prc2026_phase3.features import MISSING, clean_codes, history_names, _seconds
from prc2026_nmclock.features import (
    FEATURE_NAMES as PREVIOUS_NAMES, MODEL_NAMES as CORE_NAMES, CATEGORICAL_COLUMNS,
)

POLICY = "flown-arrival-and-past-offblock-context-v1"
INPUT_COLUMNS = [ID, PHASE, TIME, "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
                 "AOBT_3_flt", "LOBT_flt", "ARVT_3_flt", "ARVT_1_flt", "EOBT_1_flt", "IOBT_flt"]
FLOWN_NAMES = ["flight_arrival_missing", "flight_arrival_minus_movement_s",
               "flight_arrival_minus_plan_s", "flight_duration_aobt_s", "flight_duration_lobt_s",
               "flight_duration_minus_plan_s", "flight_duration_minus_initial_s",
               "flight_arrival_before_movement", "flight_duration_nonpositive", "flight_duration_over24h",
               "flight_arrival_day_delta", "flight_arrival_second", "flight_arrival_hour_sin", "flight_arrival_hour_cos"]
SURFACE_NAMES = ["surface_anchor_nm_used", "surface_anchor_lookback_s"]
for anchor in ("takeoff", "offblock"):
    for scope in ("airport", "runway"):
        SURFACE_NAMES += history_names(f"surface_{anchor}_{scope}_proxy", (15, 60), True)
        SURFACE_NAMES += history_names(f"surface_{anchor}_{scope}_push", (15, 30, 60))
        SURFACE_NAMES += [f"surface_{anchor}_{scope}_active"]
    SURFACE_NAMES += history_names(f"surface_{anchor}_stand_push", (60,))
SURFACE_NAMES += history_names("surface_offblock_stand_arr", (60,))
for scope in ("airport", "runway"):
    for phase in ("dep", "arr"):
        SURFACE_NAMES += [f"surface_interval_{scope}_{phase}_count"]
EXTRA_NAMES = FLOWN_NAMES + SURFACE_NAMES
FEATURE_NAMES = list(PREVIOUS_NAMES) + EXTRA_NAMES


def checked_context(raw):
    if set(INPUT_COLUMNS) - set(raw):
        raise ValueError("Missing declared timeline inputs.")
    frame = raw[INPUT_COLUMNS].reset_index(drop=True).copy()
    ids = frame[ID].to_numpy(dtype=float)
    if not np.isfinite(ids).all() or np.any(ids % 1) or frame[ID].duplicated().any():
        raise ValueError("Unique integral movement IDs are required.")
    if not frame[PHASE].isin(["DEP", "ARR"]).all():
        raise ValueError("Unknown movement phase.")
    for name in [TIME, *INPUT_COLUMNS[7:]]:
        if not isinstance(frame[name].dtype, pd.DatetimeTZDtype) or str(frame[name].dt.tz) != "UTC":
            raise ValueError("All clocks require explicit UTC timestamps.")
    allowed = [f"2025-{m:02}" for m in range(1, 12)] + ["2026-01", "2026-07"]
    if frame[TIME].isna().any() or not frame[TIME].dt.strftime("%Y-%m").isin(allowed).all():
        raise ValueError("Reserved or undeclared movement period.")
    return frame


def flown_features(queries):
    stamp = queries["ARVT_3_flt"]
    aobt_duration = (stamp-queries["AOBT_3_flt"]).dt.total_seconds()
    movement = (stamp-queries[TIME]).dt.total_seconds()
    hour = stamp.dt.hour+stamp.dt.minute/60+stamp.dt.second/3600
    values = {
        "flight_arrival_missing": stamp.isna().astype(float),
        "flight_arrival_minus_movement_s": movement,
        "flight_arrival_minus_plan_s": (stamp-queries["ARVT_1_flt"]).dt.total_seconds(),
        "flight_duration_aobt_s": aobt_duration,
        "flight_duration_lobt_s": (stamp-queries["LOBT_flt"]).dt.total_seconds(),
        "flight_duration_minus_plan_s": aobt_duration-(queries["ARVT_1_flt"]-queries["EOBT_1_flt"]).dt.total_seconds(),
        "flight_duration_minus_initial_s": aobt_duration-(queries["ARVT_1_flt"]-queries["IOBT_flt"]).dt.total_seconds(),
        "flight_arrival_before_movement": np.where(movement.notna(), movement.lt(0), np.nan),
        "flight_duration_nonpositive": np.where(aobt_duration.notna(), aobt_duration.le(0), np.nan),
        "flight_duration_over24h": np.where(aobt_duration.notna(), aobt_duration.gt(86400), np.nan),
        "flight_arrival_day_delta": (stamp.dt.normalize()-queries[TIME].dt.normalize()).dt.total_seconds()/86400,
        "flight_arrival_second": stamp.dt.second+stamp.dt.microsecond/1e6,
        "flight_arrival_hour_sin": np.sin(hour*2*np.pi/24),
        "flight_arrival_hour_cos": np.cos(hour*2*np.pi/24),
    }
    return pd.DataFrame({name: np.asarray(values[name], dtype="float32") for name in FLOWN_NAMES})


def peer_history(events, queries, keys, prefix, windows, value=None):
    """[t-window,t), excluding all equal-time/future events and the focal ID."""
    names = history_names(prefix, windows, value is not None)
    result = {name: np.full(len(queries), 0. if "_count_" in name else np.nan, dtype="float32") for name in names}
    groups = {key: group.sort_values([TIME, ID], kind="stable") for key, group in events.groupby(keys, sort=False, dropna=False)}
    for key, q in queries.groupby(keys, sort=False, dropna=False):
        e = groups.get(key)
        if e is None or e.empty:
            continue
        indexes = q.index.to_numpy()
        times, qt = _seconds(e[TIME]), _seconds(q[TIME])
        hi = np.searchsorted(times, qt, side="left")
        own = q[ID].map(pd.Series(np.arange(len(e)), index=e[ID])).fillna(-1).to_numpy(dtype=int)
        last = hi-1
        last -= (last >= 0) & (last == own)
        gap = qt-times[np.maximum(last, 0)]
        gap[(last < 0) | (gap > 21600)] = np.nan
        result[f"{prefix}_last_s"][indexes] = gap
        if value is not None:
            values = e[value].to_numpy(dtype=float)
            valid = np.isfinite(values)
            total = np.r_[0., np.cumsum(np.where(valid, values, 0.))]
            squared = np.r_[0., np.cumsum(np.where(valid, values*values, 0.))]
            count = np.r_[0, np.cumsum(valid)]
            own_valid = (own >= 0) & valid[np.maximum(own, 0)]
            own_values = np.where(own_valid, values[np.maximum(own, 0)], 0.)
        for window in windows:
            lo = np.searchsorted(times, qt-window*60, side="left")
            own_inside = (own >= lo) & (own < hi)
            result[f"{prefix}_count_{window}m"][indexes] = hi-lo-own_inside
            if value is not None:
                remove = own_inside & own_valid
                n = count[hi]-count[lo]-remove
                sums = total[hi]-total[lo]-np.where(remove, own_values, 0.)
                sq = squared[hi]-squared[lo]-np.where(remove, own_values*own_values, 0.)
                mean = np.divide(sums, n, out=np.full(len(q), np.nan), where=n > 0)
                second = np.divide(sq, n, out=np.full(len(q), np.nan), where=n > 0)
                result[f"{prefix}_mean_{window}m"][indexes] = mean
                result[f"{prefix}_std_{window}m"][indexes] = np.sqrt(np.maximum(0., second-mean*mean))
    return pd.DataFrame(result)


def active_counts(departures, queries, keys):
    """Other flights with offblock < t <= takeoff and a 0..6h NM interval."""
    valid = departures.loc[departures["proxy_s"].gt(0) & departures["proxy_s"].le(21600)]
    result = np.zeros(len(queries), dtype="float32")
    groups = {key: group for key, group in valid.groupby(keys, sort=False, dropna=False)}
    for key, q in queries.groupby(keys, sort=False, dropna=False):
        e = groups.get(key)
        if e is None or e.empty:
            continue
        qt = _seconds(q[TIME])
        starts, ends = np.sort(_seconds(e["offblock"])), np.sort(_seconds(e["movement"]))
        count = np.searchsorted(starts, qt, side="left")-np.searchsorted(ends, qt, side="left")
        own_valid = q[ID].isin(e[ID])
        own_active = own_valid & q.offblock.lt(q[TIME]) & q.movement.ge(q[TIME])
        count -= own_active.to_numpy()
        if np.any(count < 0):
            raise ValueError("Negative number of active peer departures.")
        result[q.index] = count
    return result


def interval_counts(events, queries, anchors, keys):
    result = np.zeros(len(queries), dtype="float32")
    groups = {key: np.sort(_seconds(g[TIME])) for key, g in events.groupby(keys, sort=False, dropna=False)}
    for key, q in queries.groupby(keys, sort=False, dropna=False):
        times = groups.get(key)
        if times is None:
            continue
        upper, lower = _seconds(q[TIME]), _seconds(anchors.loc[q.index, TIME])
        result[q.index] = np.searchsorted(times, upper, side="left")-np.searchsorted(times, lower, side="left")
    return result


def surface_features(raw_context, query_ids):
    context = checked_context(raw_context)
    dep_mask = context[PHASE].eq("DEP")
    context["airport"] = clean_codes(context.ADEP_mvt.where(dep_mask, context.ADES_mvt))
    context["runway"], context["stand"] = clean_codes(context.RUNWAY_mvt), clean_codes(context.STAND_mvt)
    context["movement"] = context[TIME]
    context["offblock"] = context.AOBT_3_flt.fillna(context.LOBT_flt)
    context["proxy_s"] = (context[TIME]-context.offblock).dt.total_seconds()
    dep, arr = context.loc[dep_mask], context.loc[~dep_mask]
    queries = pd.DataFrame({ID: query_ids.to_numpy()}).merge(dep, on=ID, how="left", validate="one_to_one", sort=False)
    if queries[TIME].isna().any():
        raise ValueError("Queries must be unique supplied departures.")
    anchor = queries.copy()
    valid_anchor = queries.proxy_s.between(0, 21600)
    anchor[TIME] = queries.offblock.where(valid_anchor, queries[TIME]-pd.Timedelta(minutes=20))
    frames = [pd.DataFrame({"surface_anchor_nm_used": valid_anchor.astype("float32"),
                            "surface_anchor_lookback_s": (queries[TIME]-anchor[TIME]).dt.total_seconds().astype("float32")})]
    present = dep.loc[dep.offblock.notna()]
    push = present.copy()
    push[TIME] = push.offblock
    for label, q in [("takeoff", queries), ("offblock", anchor)]:
        for scope, keys in [("airport", ["airport"]), ("runway", ["airport", "runway"])]:
            peers = present if scope == "airport" else present.loc[present.runway.ne(MISSING)]
            offblocks = push if scope == "airport" else push.loc[push.runway.ne(MISSING)]
            frames.append(peer_history(peers, q, keys, f"surface_{label}_{scope}_proxy", (15, 60), "proxy_s"))
            frames.append(peer_history(offblocks, q, keys, f"surface_{label}_{scope}_push", (15, 30, 60)))
            frames.append(pd.DataFrame({f"surface_{label}_{scope}_active": active_counts(peers, q, keys)}))
        frames.append(peer_history(push.loc[push.stand.ne(MISSING)], q, ["airport", "stand"], f"surface_{label}_stand_push", (60,)))
    frames.append(peer_history(arr.loc[arr.stand.ne(MISSING)], anchor, ["airport", "stand"], "surface_offblock_stand_arr", (60,)))
    for scope, keys in [("airport", ["airport"]), ("runway", ["airport", "runway"])]:
        for label, events in [("dep", dep), ("arr", arr)]:
            peers = events if scope == "airport" else events.loc[events.runway.ne(MISSING)]
            frames.append(pd.DataFrame({f"surface_interval_{scope}_{label}_count": interval_counts(peers, queries, anchor, keys)}))
    result = pd.concat(frames, axis=1)[SURFACE_NAMES].astype("float32")
    if np.isinf(result.to_numpy()).any():
        raise ValueError("Infinite surface feature.")
    return result


def timeline_features(raw_context, query_ids):
    context = checked_context(raw_context)
    queries = pd.DataFrame({ID: query_ids.to_numpy()}).merge(context, on=ID, how="left", validate="one_to_one", sort=False)
    if queries[TIME].isna().any() or not queries[PHASE].eq("DEP").all():
        raise ValueError("Every query must be a supplied departure.")
    result = pd.concat([queries[[ID]], flown_features(queries), surface_features(context, query_ids)], axis=1)
    return result[[ID, *EXTRA_NAMES]]
