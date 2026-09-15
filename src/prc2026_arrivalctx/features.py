"""Supplied completed-arrival context, excluding every focal flight ID."""
import numpy as np
import pandas as pd
from prc2026.features import ID, PHASE, TIME, BLOCK, TARGET
from prc2026_phase3.features import clean_codes, MISSING
from prc2026_nmclock.features import FEATURE_NAMES as PREVIOUS_NAMES
from prc2026_weather.features import FEATURE_NAMES as WEATHER_NAMES, CATEGORICAL_COLUMNS

POLICY = "past-completed-arrival-context-v1"
SPECS = [
    ("airport", 3600, ["count", "raw_mean", "normal_mean", "negative_fraction", "over3h_fraction"]),
    ("airport", 10800, ["count", "raw_mean", "normal_mean", "negative_fraction", "over3h_fraction"]),
    ("runway", 10800, ["count", "normal_mean"]),
    ("stand", 604800, ["count", "normal_mean"]),
]
QUERY_COLUMNS = [ID, "FLIGHT_ID_mvt", PHASE, TIME, "ADEP_mvt", "RUNWAY_mvt", "STAND_mvt"]
ARRIVAL_COLUMNS = [ID, "FLIGHT_ID_mvt", PHASE, TIME, BLOCK, TARGET, "ADES_mvt", "RUNWAY_mvt", "STAND_mvt"]


def seconds(series):
    return series.astype("datetime64[ns, UTC]").astype("int64").to_numpy() / 1e9


def key(frame, group):
    if group == "airport":
        return frame.airport
    return (frame.airport + "|" + frame[group]).where(frame[group].ne(MISSING))


def moments(values):
    values = np.asarray(values, dtype=float)
    normal = (values >= 0) & (values <= 10800)
    return np.column_stack([np.ones(len(values)), values, normal, np.where(normal, values, 0), values < 0, values > 10800])


def finish(sums):
    count, total, normal_count, normal_sum, negative, long = sums.T
    def divide(a, b):
        return np.divide(a, b, out=np.full(len(a), np.nan), where=b > 0)
    return {"count": count, "raw_mean": divide(total, count), "normal_mean": divide(normal_sum, normal_count),
            "negative_fraction": divide(negative, count), "over3h_fraction": divide(long, count)}


def context(queries, arrivals):
    queries, arrivals = queries.reset_index(drop=True), arrivals.reset_index(drop=True)
    t, event_t = seconds(queries[TIME]), seconds(arrivals.completion)
    event_values = moments(arrivals[TARGET])
    # Exclude every occurrence of the query's own flight ID, independently of movement ID.
    own = queries[["FLIGHT_ID_mvt"]].reset_index(names="query_pos").dropna().merge(
        arrivals[["FLIGHT_ID_mvt"]].reset_index(names="arrival_pos").dropna(), on="FLIGHT_ID_mvt", validate="many_to_many")
    qi, ai = own.query_pos.to_numpy(), own.arrival_pos.to_numpy()
    result = queries[[ID]].copy()
    for group, window, names in SPECS:
        qkey, akey = key(queries, group), key(arrivals, group)
        sums = np.zeros((len(queries), 6))
        a_groups = pd.DataFrame({"key": akey}).groupby("key", sort=False).indices
        q_groups = pd.DataFrame({"key": qkey}).groupby("key", sort=False).indices
        for label, indexes in q_groups.items():
            if label not in a_groups:
                continue
            events = a_groups[label]
            events = events[np.argsort(event_t[events], kind="stable")]
            times = event_t[events]
            prefix = np.vstack([np.zeros(6), np.cumsum(event_values[events], axis=0)])
            lower = np.searchsorted(times, t[indexes] - window, side="left")
            upper = np.searchsorted(times, t[indexes], side="left")
            sums[indexes] = prefix[upper] - prefix[lower]
        same = qkey.iloc[qi].reset_index(drop=True).eq(akey.iloc[ai].reset_index(drop=True)).fillna(False).to_numpy(dtype=bool)
        in_window = same & (event_t[ai] < t[qi]) & (event_t[ai] >= t[qi] - window)
        np.add.at(sums, qi[in_window], -event_values[ai[in_window]])
        if (sums[:, 0] < 0).any() or (sums[:, 2] < 0).any():
            raise ValueError("Arrival self-exclusion produced negative counts.")
        for name, values in finish(sums).items():
            if name in names:
                result[f"arrival_{group}_{window}s_{name}"] = values
    return result


EXTRA_NAMES = [f"arrival_{group}_{window}s_{name}" for group, window, names in SPECS for name in names]
FEATURE_NAMES = list(PREVIOUS_NAMES) + EXTRA_NAMES
MODEL_NAMES = list(WEATHER_NAMES) + EXTRA_NAMES


def read_context(root, original, config):
    queries, arrivals = [], []
    for source in original["sources"]:
        name = source["source"]
        periods = [("2026-01-01", "2026-02-01"), ("2026-07-01", "2026-08-01")] if name == "ranking.parquet" else [("2025-01-01", "2025-12-01")]
        path = root / "data/raw" / name
        def filters(phase):
            return [[(PHASE, "==", phase), (TIME, ">=", pd.Timestamp(start, tz="UTC")), (TIME, "<", pd.Timestamp(end, tz="UTC"))] for start, end in periods]
        query = pd.read_parquet(path, columns=QUERY_COLUMNS, filters=filters("DEP"))
        arrival = pd.read_parquet(path, columns=ARRIVAL_COLUMNS, filters=filters("ARR"))
        if not query[PHASE].eq("DEP").all() or not arrival[PHASE].eq("ARR").all() or arrival[TIME].dt.month.eq(12).any():
            raise ValueError("Protected phase or month entered arrival context.")
        queries.append(query)
        arrivals.append(arrival)
    queries, arrivals = pd.concat(queries, ignore_index=True), pd.concat(arrivals, ignore_index=True)
    if queries[ID].duplicated().any() or arrivals[ID].duplicated().any():
        raise ValueError("Duplicate movement identity in arrival context.")
    queries["airport"], arrivals["airport"] = clean_codes(queries.ADEP_mvt), clean_codes(arrivals.ADES_mvt)
    for frame in [queries, arrivals]:
        for name in ["runway", "stand"]:
            frame[name] = clean_codes(frame[name.upper() + "_mvt"])
    valid = arrivals[BLOCK].notna() & arrivals[TARGET].notna()
    arrivals = arrivals.loc[valid].reset_index(drop=True)
    if not np.isfinite(arrivals[TARGET].to_numpy(dtype=float)).all():
        raise ValueError("Arrival context contains non-finite labels.")
    arrivals["completion"] = arrivals[TIME].where(arrivals[TIME].ge(arrivals[BLOCK]), arrivals[BLOCK])
    return queries, arrivals
