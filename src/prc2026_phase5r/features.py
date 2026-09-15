"""Planned ground demand and strictly past observed runway service.

The challenge is retrospective. Planned-demand events use the supplied schedule
or first-filed EOBT, including plans for movements that depart later. They do not
use those movements' eventual departure times as the demand event clock. This
channel is distinct from observed service, which excludes all simultaneous and
future movements. Neither channel reads departure labels or actual off-blocks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prc2026.features import ID, PHASE, TIME
from prc2026_phase3.features import MISSING, clean_codes, history_names, past_history, _seconds
from prc2026_phase4.features import FEATURE_NAMES as CORE_NAMES, CATEGORICAL_COLUMNS as CORE_CATEGORIES

POLICY = "planned-demand-past-service-v1"
INPUT_COLUMNS = [ID, PHASE, TIME, "ADEP_mvt", "ADES_mvt", "RUNWAY_mvt", "STAND_mvt",
                 "SCHED_TIME_UTC_mvt", "EOBT_1_flt", "WK_TBL_CAT_flt"]
GROUND_CATEGORIES = ["service_previous_phase", "service_wake_pair"]
GROUND_NUMERIC = []
for lag in (0, 10):
    for plan in ("schedule", "eobt"):
        GROUND_NUMERIC += [f"gd_{lag}_{plan}_count_{w}m" for w in (15, 30, 60)]
        GROUND_NUMERIC += [f"gd_{lag}_{plan}_balance_{w}m" for w in (15, 30)]
for lag in (10, 20):
    for phase in ("dep", "arr"):
        GROUND_NUMERIC += history_names(f"gs_{lag}_{phase}", (5, 15, 30))
for scope in ("airport", "runway"):
    for phase in ("dep", "all"):
        for n in (5, 10):
            GROUND_NUMERIC += [f"service_{scope}_{phase}_gap{n}_{stat}_s" for stat in ("mean", "std")]
GROUND_NUMERIC += ["service_active_runways_60m", "service_runway_dep_share_30m",
                   "service_runway_movement_share_30m"]
GROUND_NUMERIC += history_names("service_stand_dep", (60,))
GROUND_NAMES = GROUND_CATEGORIES + GROUND_NUMERIC
FEATURE_NAMES = CORE_NAMES + GROUND_NAMES
CATEGORICAL_COLUMNS = CORE_CATEGORIES + GROUND_CATEGORIES


def _headways(events: pd.DataFrame, queries: pd.DataFrame, keys: list[str], prefix: str) -> pd.DataFrame:
    arrays = {f"{prefix}_gap{n}_{stat}_s": np.full(len(queries), np.nan, dtype="float32")
              for n in (5, 10) for stat in ("mean", "std")}
    groups = {k: g.sort_values([TIME, ID], kind="stable")
              for k, g in events.groupby(keys, sort=False, dropna=False)}
    for key, q in queries.groupby(keys, sort=False, dropna=False):
        e = groups.get(key)
        if e is None or len(e) < 2:
            continue
        t, qt = _seconds(e[TIME]), _seconds(q[TIME])
        hi = np.searchsorted(t, qt, side="left")
        fresh = (hi > 0) & (qt-t[np.maximum(hi-1, 0)] <= 1800)
        for n in (5, 10):
            positions = hi[:, None]-np.arange(1, n+1)[None, :]
            observed = t[np.maximum(positions, 0)]
            gaps = observed[:, :-1]-observed[:, 1:]
            valid = (positions[:, 1:] >= 0) & (gaps <= 3600) & fresh[:, None]
            count = valid.sum(axis=1)
            total = np.where(valid, gaps, 0).sum(axis=1)
            squared = np.where(valid, gaps*gaps, 0).sum(axis=1)
            mean = np.divide(total, count, out=np.full(len(q), np.nan), where=count > 0)
            second = np.divide(squared, count, out=np.full(len(q), np.nan), where=count > 0)
            arrays[f"{prefix}_gap{n}_mean_s"][q.index] = mean
            arrays[f"{prefix}_gap{n}_std_s"][q.index] = np.sqrt(np.maximum(0, second-mean*mean))
    return pd.DataFrame(arrays)


def _previous_service(events: pd.DataFrame, queries: pd.DataFrame) -> pd.DataFrame:
    phase, wake = np.full(len(queries), MISSING, dtype=object), np.full(len(queries), MISSING, dtype=object)
    keys = ["airport", "runway"]
    groups = {k: g.sort_values([TIME, ID], kind="stable")
              for k, g in events.groupby(keys, sort=False, dropna=False)}
    for key, q in queries.groupby(keys, sort=False, dropna=False):
        e = groups.get(key)
        if e is None:
            continue
        t, qt = _seconds(e[TIME]), _seconds(q[TIME])
        hi = np.searchsorted(t, qt, side="left")
        pos = np.maximum(hi-1, 0)
        valid = (hi > 0) & (qt-t[pos] <= 3600)
        indexes = q.index.to_numpy()[valid]
        phase[indexes] = e[PHASE].to_numpy()[pos[valid]]
        wake[indexes] = e["wake"].to_numpy()[pos[valid]]
    return pd.DataFrame({"service_previous_phase": pd.Series(phase, dtype="string"),
                         "service_wake_pair": pd.Series(wake, dtype="string") + "|" + queries["wake"]})


def _active_runways(events: pd.DataFrame, queries: pd.DataFrame) -> np.ndarray:
    counts = np.zeros(len(queries), dtype="float32")
    for airport, q in queries.groupby("airport", sort=False):
        qt = _seconds(q[TIME])
        e = events.loc[events["airport"].eq(airport) & events["runway"].ne(MISSING)]
        for _, runway in e.groupby("runway", sort=False):
            times = np.sort(_seconds(runway[TIME]))
            n = np.searchsorted(times, qt, side="left")-np.searchsorted(times, qt-3600, side="left")
            counts[q.index] += n > 0
    return counts


def ground_features(raw_context: pd.DataFrame, query_ids: pd.Series) -> pd.DataFrame:
    if set(INPUT_COLUMNS)-set(raw_context):
        raise ValueError("Missing declared ground-demand inputs.")
    context = raw_context[INPUT_COLUMNS].copy().reset_index(drop=True)
    if context[ID].isna().any() or context[ID].duplicated().any() or not context[PHASE].isin(["DEP", "ARR"]).all():
        raise ValueError("Invalid movement context identity/phase.")
    if context[TIME].dt.strftime("%Y-%m").eq("2025-12").any():
        raise ValueError("Reserved audit context cannot enter this campaign.")
    _seconds(context[TIME])
    is_dep = context[PHASE].eq("DEP")
    context["airport"] = clean_codes(context["ADEP_mvt"].where(is_dep, context["ADES_mvt"]))
    for target, source in [("runway", "RUNWAY_mvt"), ("stand", "STAND_mvt"), ("wake", "WK_TBL_CAT_flt")]:
        context[target] = clean_codes(context[source])
    dep, arr = context.loc[is_dep], context.loc[~is_dep]
    queries = pd.DataFrame({ID: query_ids.to_numpy()}).merge(dep, on=ID, how="left", validate="one_to_one")
    if queries[TIME].isna().any():
        raise ValueError("Every query must be a unique supplied departure.")
    frames = [queries[[ID]], _previous_service(context, queries)]
    for lag in (0, 10, 20):
        q = queries.copy()
        q[TIME] = q[TIME]-pd.Timedelta(minutes=lag)
        service = {}
        for label, events in [("dep", dep), ("arr", arr)]:
            service[label] = past_history(events, q, ["airport"], f"gs_{lag}_{label}", (5, 15, 30))
            if lag in (10, 20):
                frames.append(service[label])
        if lag not in (0, 10):
            continue
        for label, clock in [("schedule", "SCHED_TIME_UTC_mvt"), ("eobt", "EOBT_1_flt")]:
            planned = dep.loc[dep[clock].notna(), [ID, "airport", clock]].rename(columns={clock: TIME})
            counts = past_history(planned, q, ["airport"], f"gd_{lag}_{label}", (15, 30, 60))
            values = {}
            for window in (15, 30, 60):
                # The focal flight's own plan is already a row feature; it is
                # excluded from other-flight demand even at a window boundary.
                own = queries[clock].ge(q[TIME]-pd.Timedelta(minutes=window)) & queries[clock].lt(q[TIME])
                name = f"gd_{lag}_{label}_count_{window}m"
                values[name] = counts[name]-own.to_numpy(dtype="float32")
                if window != 60:
                    values[f"gd_{lag}_{label}_balance_{window}m"] = values[name]-service["dep"][f"gs_{lag}_dep_count_{window}m"]
            frames.append(pd.DataFrame(values))
    for scope, keys in [("airport", ["airport"]), ("runway", ["airport", "runway"])]:
        for label, events in [("dep", dep), ("all", context)]:
            frames.append(_headways(events, queries, keys, f"service_{scope}_{label}"))
    shares = {"service_active_runways_60m": _active_runways(context, queries)}
    for label, events in [("dep", dep), ("movement", context)]:
        at_airport = past_history(events, queries, ["airport"], "a", (30,))["a_count_30m"].to_numpy()
        at_runway = past_history(events, queries, ["airport", "runway"], "r", (30,))["r_count_30m"].to_numpy()
        shares[f"service_runway_{label}_share_30m"] = np.divide(
            at_runway, at_airport, out=np.full(len(queries), np.nan), where=at_airport > 0)
    frames.append(pd.DataFrame(shares))
    frames.append(past_history(dep, queries, ["airport", "stand"], "service_stand_dep", (60,)))
    out = pd.concat(frames, axis=1)[[ID, *GROUND_NAMES]]
    out[GROUND_NUMERIC] = out[GROUND_NUMERIC].astype("float32")
    if np.isinf(out[GROUND_NUMERIC].to_numpy()).any():
        raise ValueError("Infinite ground-demand/service feature.")
    return out
