"""Population signals (configs/pop_signals_v1.json): new label-free classifier inputs for the v9 LIRF schedule-fill mixture.

The v9 mixture (src/prc2026_lirf_mixture) predicts P(fill) for the LIRF population (departures without any NM clock and without an
airport aircraft type) from the row's own clocks and the timing of the previous arrival in-block at the same stand. This module adds
what that arrival *is*, again from supplied movement columns only:

- ``pa_same_prefix``: the previous arrival's flight prefix (first three characters of FLIGHT_mvt) equals the departure's;
- ``pa_sched_gap``: the departure's scheduled time minus the previous arrival's scheduled time (the planned turnaround);
- ``pa_fill``: the previous arrival's in-block equals its scheduled arrival within the declared fill tolerance;
- ``pa_nonm``: the previous arrival has no NM flight record (ADEP_flt missing);
- ``cs_rate``: the departure callsign's schedule-fill share among the fitting-month population rows of *other* months (smoothed towards
  the population's overall share), a leave-one-month-out target encoding; for a file without labels, the share over every fitting month.

No departure label or departure block time is read for any feature. Fill modes come from fitting labels only (``label_modes``).
The classifier protocol is v9's (three classes, seven leaves, date-grouped inner folds for the round count), bagged over declared seeds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

from prc2026.features import ID, PHASE, TIME
from prc2026_lirf_mixture.features import (AIRPORT, ARRIVAL_COLUMNS, CATEGORICAL, DEPARTURE_COLUMNS, FEATURE_NAMES, classifier_features,
                                            label_modes, population_rows)
from prc2026_lirf_mixture.model import band_values, mixture_values, normal_component

PA_NUMERIC = ["pa_same_prefix", "pa_sched_gap", "pa_fill", "pa_nonm"]
CS_NUMERIC = ["cs_rate"]
ARRIVAL_EXTRA = ["SCHED_TIME_UTC_mvt", "FLIGHT_mvt", "ADEP_flt"]
RAW_COLUMNS = sorted(set(DEPARTURE_COLUMNS + ARRIVAL_COLUMNS + ARRIVAL_EXTRA + ["TAXITIME_SEC_mvt"]))
FEATURE_SETS = {"base": [], "pa": PA_NUMERIC, "pa2": ["pa_same_prefix", "pa_sched_gap"], "cs": CS_NUMERIC, "cs_pa": CS_NUMERIC + PA_NUMERIC}
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def _seconds(stamps: pd.Series) -> np.ndarray:
    return (pd.to_datetime(stamps, utc=True) - EPOCH).dt.total_seconds().to_numpy(dtype=float)


def _prefix(flight: pd.Series) -> pd.Series:
    flight = flight.fillna("").astype(str)
    return flight.str[:3].where(flight.ne(""), "?")


def previous_arrival_features(population: pd.DataFrame, movements: pd.DataFrame, margin_seconds: float, fill_tolerance_seconds: float) -> pd.DataFrame:
    """[ID] + PA_NUMERIC for the population rows: identity and block source of the previous arrival in-block at the departure's stand
    (the same arrival the v9 turnaround features use: the latest in-block at or before takeoff minus the margin)."""
    needed = set(ARRIVAL_COLUMNS + ARRIVAL_EXTRA)
    if needed - set(movements):
        raise ValueError("Missing arrival columns for previous-arrival features.")
    arrivals = movements.loc[movements[PHASE].eq("ARR") & movements["ADES_mvt"].eq(AIRPORT)
                             & movements["STAND_mvt"].notna() & movements["BLOCK_TIME_UTC_mvt"].notna()]
    arrivals = arrivals.assign(_b=_seconds(arrivals["BLOCK_TIME_UTC_mvt"]), _s=_seconds(arrivals["SCHED_TIME_UTC_mvt"]),
                               _prefix=_prefix(arrivals["FLIGHT_mvt"]).to_numpy(),
                               _nonm=arrivals["ADEP_flt"].isna().to_numpy(dtype=float)).sort_values("_b")
    arrivals = arrivals.assign(_fill=(np.abs(arrivals["_b"] - arrivals["_s"]) <= fill_tolerance_seconds).astype(float))
    per_stand = {stand: (g["_b"].to_numpy(), g[["_s", "_prefix", "_fill", "_nonm"]].to_numpy(dtype=object)) for stand, g in arrivals.groupby("STAND_mvt")}
    takeoff, schedule = _seconds(population[TIME]), _seconds(population["SCHED_TIME_UTC_mvt"])
    prefix = _prefix(population["FLIGHT_mvt"]).to_numpy()
    same, gap, fill, nonm = (np.full(len(population), np.nan) for _ in range(4))
    for i, (stand, t) in enumerate(zip(population["STAND_mvt"].to_numpy(), takeoff)):
        if not isinstance(stand, str) or stand not in per_stand:
            continue
        times, values = per_stand[stand]
        k = int(np.searchsorted(times, t - margin_seconds, side="left"))
        if k == 0:
            continue
        arr_s, arr_prefix, arr_fill, arr_nonm = values[k - 1]
        same[i] = float(arr_prefix == prefix[i])
        gap[i] = schedule[i] - float(arr_s)
        fill[i], nonm[i] = float(arr_fill), float(arr_nonm)
    return pd.DataFrame({ID: population[ID].to_numpy(), "pa_same_prefix": same, "pa_sched_gap": gap, "pa_fill": fill, "pa_nonm": nonm})


def callsign_rates(flights: pd.Series, months: pd.Series, fill: np.ndarray, target_flights: pd.Series, target_months: pd.Series | None,
                   smoothing: float) -> np.ndarray:
    """Schedule-fill share of each target callsign among fitting population rows of other months (target_months given) or of every
    fitting month (None), smoothed towards the fitting population's overall share with weight `smoothing`."""
    table = pd.DataFrame({"flight": flights.fillna("").astype(str).to_numpy(), "month": months.to_numpy(), "fill": np.asarray(fill, dtype=float)})
    prior = float(table["fill"].mean())
    total = table.groupby("flight")["fill"].agg(["sum", "size"])
    by_month = table.groupby(["flight", "month"])["fill"].agg(["sum", "size"])
    flight = target_flights.fillna("").astype(str).to_numpy()
    tot = total.reindex(flight).fillna(0.0).to_numpy()
    if target_months is not None:
        own = by_month.reindex(list(zip(flight, target_months.to_numpy()))).fillna(0.0).to_numpy()
        tot = tot - own
    return (tot[:, 0] + smoothing * prior) / (tot[:, 1] + smoothing)


def population_table(movements: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Population rows with v9's classifier features, the previous-arrival features, the schedule interval, month and date."""
    population = population_rows(movements)
    margin = float(config["turnaround_features"]["previous_arrival_margin_seconds"])
    base = classifier_features(population, movements, margin)
    extra = previous_arrival_features(population, movements, margin, float(config["fill_tolerance_seconds"]))
    out = base.merge(extra, on=ID, how="left", validate="one_to_one")
    stamps = pd.to_datetime(population[TIME], utc=True)
    out["lomo_month"] = stamps.dt.strftime("%Y-%m").to_numpy()  # v9's numeric "month" feature keeps its column
    out["movement_date_utc"] = stamps.dt.strftime("%Y-%m-%d").to_numpy()
    out["flight"] = population["FLIGHT_mvt"].to_numpy()
    out["label"] = population["TAXITIME_SEC_mvt"].to_numpy(dtype=float) if "TAXITIME_SEC_mvt" in population else np.nan
    return out


def modes_of(table: pd.DataFrame, config: dict) -> np.ndarray:
    return label_modes(table["label"].to_numpy(dtype=float), table["schedule_interval_s"].to_numpy(dtype=float), config)


def _matrix(features: pd.DataFrame, names: list[str], categories: dict | None = None) -> tuple[pd.DataFrame, dict]:
    frame = features[names].copy()
    categories = categories or {}
    for name in CATEGORICAL:
        levels = categories.get(name)
        if levels is None:
            levels = sorted(frame[name].astype(str).unique().tolist())
            categories[name] = levels
        frame[name] = pd.Categorical(frame[name].astype(str), categories=levels)
    return frame, categories


def fit_bag(features: pd.DataFrame, modes: np.ndarray, dates: np.ndarray, names: list[str], classifier: dict, seeds: list[int]) -> dict:
    """v9's protocol per seed (rounds = median best inner date-grouped iteration, at least 50), one booster per declared seed."""
    frame, categories = _matrix(features, names)
    modes = np.asarray(modes, dtype=int)
    if len(frame) != len(modes) or len(dates) != len(modes) or set(np.unique(modes)) - {0, 1, 2}:
        raise ValueError("Classifier inputs must align and use the three declared modes.")
    members = []
    for seed in seeds:
        params = {**classifier["params"], "seed": int(seed)}
        best = []
        for train, valid in GroupKFold(int(classifier["inner_folds"])).split(frame, modes, groups=dates):
            booster = lgb.train(params, lgb.Dataset(frame.iloc[train], modes[train], categorical_feature=CATEGORICAL), int(classifier["max_rounds"]),
                                valid_sets=[lgb.Dataset(frame.iloc[valid], modes[valid], categorical_feature=CATEGORICAL)],
                                callbacks=[lgb.early_stopping(100, verbose=False)])
            best.append(booster.best_iteration)
        rounds = max(50, int(np.rint(np.median(best))))
        booster = lgb.train(params, lgb.Dataset(frame, modes, categorical_feature=CATEGORICAL), rounds)
        members.append({"seed": int(seed), "booster": booster, "rounds": rounds, "inner_best_iterations": [int(b) for b in best]})
    return {"names": list(names), "categories": categories, "members": members, "fitting_rows": int(len(frame)),
            "mode_counts": {str(k): int((modes == k).sum()) for k in range(3)}}


def predict_bag(bag: dict, features: pd.DataFrame) -> np.ndarray:
    frame, _ = _matrix(features, bag["names"], bag["categories"])
    probabilities = np.zeros((len(frame), 3))
    for member in bag["members"]:
        probabilities += member["booster"].predict(frame, num_iteration=member["rounds"]) / len(bag["members"])
    if not np.isfinite(probabilities).all():
        raise ValueError("Classifier produced invalid probabilities.")
    return probabilities


def candidate_names(candidate: dict) -> list[str]:
    return FEATURE_NAMES + FEATURE_SETS[candidate["feature_set"]]


def mixture_for(bag: dict, component: dict, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(mixture value, P(fill) renormalised over fill/normal) for `features` rows."""
    probabilities = predict_bag(bag, features)
    interval = features["schedule_interval_s"].to_numpy(dtype=float)
    fill = probabilities[:, 0] / (probabilities[:, 0] + probabilities[:, 1])
    return mixture_values(probabilities, interval, component), fill


__all__ = ["PA_NUMERIC", "CS_NUMERIC", "RAW_COLUMNS", "FEATURE_SETS", "previous_arrival_features", "callsign_rates", "population_table", "modes_of",
           "fit_bag", "predict_bag", "candidate_names", "mixture_for", "normal_component", "band_values"]
