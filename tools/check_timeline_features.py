"""Independent direct-scan checks of the new timeline features, without labels."""
from pathlib import Path
import sys
import math
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from prc2026.features import ID, PHASE, TIME
from prc2026_phase3.features import clean_codes, MISSING
from prc2026_timeline.features import FLOWN_NAMES, SURFACE_NAMES, EXTRA_NAMES, PREVIOUS_NAMES
from prc2026_timeline.bundle import read_context
from prc2026.artifacts import read_json


def timestamp_seconds(series):
    values = series.dt.as_unit("us").astype("int64").to_numpy(dtype=float)/1_000_000
    values[series.isna().to_numpy()] = np.nan
    return values


def direct_flown(row):
    def seconds(left, right):
        return float((row[left]-row[right]).total_seconds()) if pd.notna(row[left]) and pd.notna(row[right]) else np.nan
    arrival = row["ARVT_3_flt"]
    movement = seconds("ARVT_3_flt", TIME)
    duration = seconds("ARVT_3_flt", "AOBT_3_flt")
    hour = arrival.hour+arrival.minute/60+arrival.second/3600 if pd.notna(arrival) else np.nan
    values = [
        float(pd.isna(arrival)), movement, seconds("ARVT_3_flt", "ARVT_1_flt"), duration,
        seconds("ARVT_3_flt", "LOBT_flt"), duration-seconds("ARVT_1_flt", "EOBT_1_flt"),
        duration-seconds("ARVT_1_flt", "IOBT_flt"), float(movement < 0) if np.isfinite(movement) else np.nan,
        float(duration <= 0) if np.isfinite(duration) else np.nan,
        float(duration > 86400) if np.isfinite(duration) else np.nan,
        (arrival.normalize()-row[TIME].normalize()).total_seconds()/86400 if pd.notna(arrival) else np.nan,
        arrival.second+arrival.microsecond/1e6 if pd.notna(arrival) else np.nan,
        math.sin(2*math.pi*hour/24), math.cos(2*math.pi*hour/24),
    ]
    return dict(zip(FLOWN_NAMES, values))


def direct_history(frame, event_clock, query_time, prefix, windows, proxy_values=False):
    prior = frame.loc[frame[event_clock].lt(query_time)]
    last = query_time-prior[event_clock].max() if len(prior) else np.nan
    output = {f"{prefix}_last_s": last if last <= 21600 else np.nan}
    for window in windows:
        chosen = prior.loc[prior[event_clock].ge(query_time-window*60)]
        output[f"{prefix}_count_{window}m"] = len(chosen)
        if proxy_values:
            values = chosen["proxy"].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            output[f"{prefix}_mean_{window}m"] = float(np.mean(values)) if len(values) else np.nan
            output[f"{prefix}_std_{window}m"] = float(np.std(values)) if len(values) else np.nan
    return output


def direct_surface(row, peers):
    clock = row["AOBT_3_flt"] if pd.notna(row["AOBT_3_flt"]) else row["LOBT_flt"]
    movement = row[TIME].timestamp()
    offblock = clock.timestamp() if pd.notna(clock) else np.nan
    duration = movement-offblock
    valid = np.isfinite(duration) and 0 <= duration <= 21600
    anchor = offblock if valid else movement-1200
    output = {"surface_anchor_nm_used": float(valid), "surface_anchor_lookback_s": movement-anchor}
    peers = peers.loc[peers[ID].ne(row[ID])]
    dep = peers.loc[peers.phase.eq("DEP")]
    arr = peers.loc[peers.phase.eq("ARR")]
    present = dep.loc[dep.push.notna()]
    runway = clean_codes(pd.Series([row["RUNWAY_mvt"]])).iloc[0]
    stand = clean_codes(pd.Series([row["STAND_mvt"]])).iloc[0]
    for label, query_time in [("takeoff", movement), ("offblock", anchor)]:
        for scope in ["airport", "runway"]:
            flights = present if scope == "airport" else present.loc[present.runway.eq(runway) & present.runway.ne(MISSING)]
            output.update(direct_history(flights, "movement", query_time, f"surface_{label}_{scope}_proxy", (15, 60), True))
            output.update(direct_history(flights, "push", query_time, f"surface_{label}_{scope}_push", (15, 30, 60)))
            active = flights.push.lt(query_time) & flights.movement.ge(query_time) & flights.proxy.gt(0) & flights.proxy.le(21600)
            output[f"surface_{label}_{scope}_active"] = int(active.sum())
        same_stand = present.loc[present.stand.eq(stand) & present.stand.ne(MISSING)]
        output.update(direct_history(same_stand, "push", query_time, f"surface_{label}_stand_push", (60,)))
    same_stand_arr = arr.loc[arr.stand.eq(stand) & arr.stand.ne(MISSING)]
    output.update(direct_history(same_stand_arr, "movement", anchor, "surface_offblock_stand_arr", (60,)))
    for scope in ["airport", "runway"]:
        for phase, flights in [("dep", dep), ("arr", arr)]:
            events = flights if scope == "airport" else flights.loc[flights.runway.eq(runway) & flights.runway.ne(MISSING)]
            output[f"surface_interval_{scope}_{phase}_count"] = int((events.movement.ge(anchor) & events.movement.lt(movement)).sum())
    return output


def verify_feature_samples(root, base, manifest, original):
    parent = root/manifest["previous_bundle"]
    queries, saved_parts, inherited_rows = [], [], 0
    for source in manifest["rows"]:
        saved = pd.read_parquet(base/"features"/source)
        previous = pd.read_parquet(parent/"features"/source, columns=[ID, *PREVIOUS_NAMES])
        pd.testing.assert_frame_equal(saved[[ID, *PREVIOUS_NAMES]], previous, check_exact=True)
        inherited_rows += len(saved)
        exceptional = saved.nmclock_proxy_s.lt(0) | saved.nmclock_proxy_s.gt(10800)
        indexes = np.unique(np.r_[np.linspace(0, len(saved)-1, min(4, len(saved)), dtype=int),
                                  np.flatnonzero(exceptional)[:4], np.flatnonzero(saved.nmclock_available.eq(0))[:4]])
        chosen = saved.iloc[indexes][[ID, *EXTRA_NAMES]].reset_index(drop=True)
        queries.extend(chosen[ID].tolist())
        saved_parts.append(chosen)
    saved = pd.concat(saved_parts, ignore_index=True)
    context = read_context(root, original, manifest["descriptor"]["config"])
    raw_queries = pd.DataFrame({ID: queries}).merge(context, on=ID, how="left", validate="one_to_one", sort=False)
    is_dep = context[PHASE].eq("DEP")
    airport = clean_codes(context.ADEP_mvt.where(is_dep, context.ADES_mvt))
    push = context.AOBT_3_flt.fillna(context.LOBT_flt)
    flat = pd.DataFrame({ID: context[ID].to_numpy(), "airport": airport, "phase": context[PHASE],
                         "runway": clean_codes(context.RUNWAY_mvt), "stand": clean_codes(context.STAND_mvt),
                         "movement": timestamp_seconds(context[TIME]), "push": timestamp_seconds(push)})
    flat["proxy"] = flat.movement-flat.push
    groups = {name: group for name, group in flat.groupby("airport", sort=False)}
    rebuilt = []
    for _, row in raw_queries.iterrows():
        airport_code = clean_codes(pd.Series([row["ADEP_mvt"]])).iloc[0]
        rebuilt.append({**direct_flown(row), **direct_surface(row, groups[airport_code])})
    expected = pd.DataFrame(rebuilt)[EXTRA_NAMES].astype("float32")
    actual = saved[EXTRA_NAMES]
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=.002, equal_nan=True)
    exact = [name for name in EXTRA_NAMES if "_count" in name or name.endswith("_active") or name in ["surface_anchor_nm_used", "flight_arrival_missing", "flight_arrival_before_movement", "flight_duration_nonpositive", "flight_duration_over24h"]]
    np.testing.assert_allclose(actual[exact], expected[exact], rtol=0., atol=0., equal_nan=True)
    differences = np.abs(actual.to_numpy(dtype=float)-expected.to_numpy(dtype=float))
    return {"inherited_rows_checked_exactly": inherited_rows, "sample_rows": len(saved), "feature_values_reconstructed": len(saved)*len(EXTRA_NAMES),
            "maximum_absolute_numeric_difference": float(np.nanmax(differences)), "maximum_count_flag_difference": 0.,
            "numerical_tolerance": {"rtol": 2e-6, "atol": .002}, "audit_used": False, "departure_labels_read": False}
