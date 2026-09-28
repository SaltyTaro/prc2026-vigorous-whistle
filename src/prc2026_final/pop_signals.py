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
ARR_NUMERIC = ["arr_fill_m", "arr_fill_d"]  # the airport's arrival fill share (in-block = scheduled arrival) in the row's month / UTC day
OP_NUMERIC = ["op_arr_fill_m", "op_arr_n_m"]  # the operator prefix's LIRF arrival fill share in the row's month (smoothed) and its log1p arrival count
OPD_NUMERIC = ["prefix_nonm_m", "prefix_notype_m", "prefix_n_m"]  # the prefix's LIRF departures in the month: no-NM-record share, no-type share, log1p count
OP_SMOOTHING = 5.0
ARRIVAL_EXTRA = ["SCHED_TIME_UTC_mvt", "FLIGHT_mvt", "ADEP_flt", "MVT_TIME_UTC_mvt"]
RAW_COLUMNS = sorted(set(DEPARTURE_COLUMNS + ARRIVAL_COLUMNS + ARRIVAL_EXTRA + ["TAXITIME_SEC_mvt"]))
FEATURE_SETS = {"base": [], "pa": PA_NUMERIC, "pa2": ["pa_same_prefix", "pa_sched_gap"], "cs": CS_NUMERIC, "cs_pa": CS_NUMERIC + PA_NUMERIC,
                "pa_arrm": PA_NUMERIC + ["arr_fill_m"], "pa_arrd": PA_NUMERIC + ARR_NUMERIC, "arrm": ["arr_fill_m"],
                "pa_op": PA_NUMERIC + OP_NUMERIC, "pa_opd": PA_NUMERIC + OPD_NUMERIC, "pa_opall": PA_NUMERIC + OP_NUMERIC + OPD_NUMERIC}


def arrival_fill_shares(movements: pd.DataFrame, fill_tolerance_seconds: float) -> tuple[pd.Series, pd.Series]:
    """Share of LIRF arrivals whose in-block equals their scheduled arrival within the tolerance, per month ('YYYY-MM') and per UTC day
    ('YYYY-MM-DD') of the arrival's movement time. Arrival block times are supplied in every file, including the ranking's."""
    arrivals = movements.loc[movements[PHASE].eq("ARR") & movements["ADES_mvt"].eq(AIRPORT) & movements["BLOCK_TIME_UTC_mvt"].notna()
                             & movements["SCHED_TIME_UTC_mvt"].notna()]
    fill = pd.Series(np.abs(_seconds(arrivals["BLOCK_TIME_UTC_mvt"]) - _seconds(arrivals["SCHED_TIME_UTC_mvt"])) <= fill_tolerance_seconds,
                     index=arrivals.index, dtype=float)
    stamps = pd.to_datetime(arrivals[TIME], utc=True)
    return fill.groupby(stamps.dt.strftime("%Y-%m").to_numpy()).mean(), fill.groupby(stamps.dt.strftime("%Y-%m-%d").to_numpy()).mean()
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def operator_descriptors(movements: pd.DataFrame, fill_tolerance_seconds: float, smoothing: float = OP_SMOOTHING) -> pd.DataFrame:
    """Label-free descriptors of each (calendar month, flight prefix) at LIRF, over every LIRF movement of the frame (not only the
    population): the prefix's arrival fill share smoothed towards the month's arrival fill share with weight `smoothing`, its log1p
    arrival count, and over the prefix's departures the share without an NM record (ADEP_flt missing), the share without an airport
    aircraft type and the log1p departure count. A prefix with departures but no arrivals in the month gets the month's share and count 0."""
    month = pd.to_datetime(movements[TIME], utc=True).dt.strftime("%Y-%m")
    prefix = _prefix(movements["FLIGHT_mvt"])
    arr = (movements[PHASE].eq("ARR") & movements["ADES_mvt"].eq(AIRPORT) & movements["BLOCK_TIME_UTC_mvt"].notna()
           & movements["SCHED_TIME_UTC_mvt"].notna())
    arrivals = pd.DataFrame({"month": month[arr].to_numpy(), "prefix": prefix[arr].to_numpy(),
                             "fill": (np.abs(_seconds(movements.loc[arr, "BLOCK_TIME_UTC_mvt"]) - _seconds(movements.loc[arr, "SCHED_TIME_UTC_mvt"]))
                                      <= fill_tolerance_seconds).astype(float)})
    month_share = arrivals.groupby("month")["fill"].mean()
    by_prefix = arrivals.groupby(["month", "prefix"])["fill"].agg(["sum", "size"])
    share = by_prefix.index.get_level_values("month").map(month_share).to_numpy(dtype=float)
    by_prefix["op_arr_fill_m"] = (by_prefix["sum"] + smoothing * share) / (by_prefix["size"] + smoothing)
    by_prefix["op_arr_n_m"] = np.log1p(by_prefix["size"].to_numpy(dtype=float))
    dep = movements[PHASE].eq("DEP") & movements["ADEP_mvt"].eq(AIRPORT)
    departures = pd.DataFrame({"month": month[dep].to_numpy(), "prefix": prefix[dep].to_numpy(),
                               "nonm": movements.loc[dep, "ADEP_flt"].isna().to_numpy(dtype=float),
                               "notype": movements.loc[dep, "AIRCRAFT_TYPE_mvt"].isna().to_numpy(dtype=float)})
    out = departures.groupby(["month", "prefix"]).agg(prefix_nonm_m=("nonm", "mean"), prefix_notype_m=("notype", "mean"), prefix_n_m=("nonm", "size"))
    out["prefix_n_m"] = np.log1p(out["prefix_n_m"].to_numpy(dtype=float))
    out = out.join(by_prefix[OP_NUMERIC], how="left")
    missing = out["op_arr_fill_m"].isna().to_numpy()
    out.loc[missing, "op_arr_fill_m"] = out.index.get_level_values("month")[missing].map(month_share).to_numpy(dtype=float)
    out.loc[missing, "op_arr_n_m"] = 0.0
    return out[OPD_NUMERIC + OP_NUMERIC]


def stake_weights(table: pd.DataFrame, component: dict, spec: dict) -> np.ndarray:
    """|schedule interval - the normal component's band value| clipped to [floor_s, cap_s] and scaled to mean 1: the squared-error stake
    of the fill/normal decision on each fitting row, used as LightGBM row weights."""
    interval = table["schedule_interval_s"].to_numpy(dtype=float)
    stake = np.clip(np.abs(interval - band_values(component, interval)), float(spec["floor_s"]), float(spec["cap_s"]))
    return stake / stake.mean()


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
    by_month, by_day = arrival_fill_shares(movements, float(config["fill_tolerance_seconds"]))
    out["arr_fill_m"] = pd.Series(out["lomo_month"]).map(by_month).to_numpy(dtype=float)
    out["arr_fill_d"] = pd.Series(stamps.dt.strftime("%Y-%m-%d").to_numpy()).map(by_day).to_numpy(dtype=float)
    descriptors = operator_descriptors(movements, float(config["fill_tolerance_seconds"]))
    keys = pd.MultiIndex.from_arrays([out["lomo_month"].to_numpy(), _prefix(population["FLIGHT_mvt"]).to_numpy()])
    for name in OPD_NUMERIC + OP_NUMERIC:
        out[name] = descriptors[name].reindex(keys).to_numpy(dtype=float)
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


def fit_bag(features: pd.DataFrame, modes: np.ndarray, dates: np.ndarray, names: list[str], classifier: dict, seeds: list[int],
            weight: np.ndarray | None = None) -> dict:
    """v9's protocol per seed (rounds = median best inner date-grouped iteration, at least 50), one booster per declared seed; optional
    row weights (stake_weights) enter every inner fit, its early-stopping metric and the final booster."""
    frame, categories = _matrix(features, names)
    modes = np.asarray(modes, dtype=int)
    if len(frame) != len(modes) or len(dates) != len(modes) or set(np.unique(modes)) - {0, 1, 2}:
        raise ValueError("Classifier inputs must align and use the three declared modes.")
    if weight is not None:
        weight = np.asarray(weight, dtype=float)
        if len(weight) != len(frame) or not np.isfinite(weight).all() or (weight <= 0).any():
            raise ValueError("Row weights must align with the fitting rows and be finite and positive.")
    members = []
    for seed in seeds:
        params = {**classifier["params"], "seed": int(seed)}
        best = []
        for train, valid in GroupKFold(int(classifier["inner_folds"])).split(frame, modes, groups=dates):
            booster = lgb.train(params, lgb.Dataset(frame.iloc[train], modes[train], weight=None if weight is None else weight[train],
                                                    categorical_feature=CATEGORICAL), int(classifier["max_rounds"]),
                                valid_sets=[lgb.Dataset(frame.iloc[valid], modes[valid], weight=None if weight is None else weight[valid],
                                                        categorical_feature=CATEGORICAL)],
                                callbacks=[lgb.early_stopping(100, verbose=False)])
            best.append(booster.best_iteration)
        rounds = max(50, int(np.rint(np.median(best))))
        booster = lgb.train(params, lgb.Dataset(frame, modes, weight=weight, categorical_feature=CATEGORICAL), rounds)
        members.append({"seed": int(seed), "booster": booster, "rounds": rounds, "inner_best_iterations": [int(b) for b in best]})
    return {"names": list(names), "categories": categories, "members": members, "fitting_rows": int(len(frame)),
            "mode_counts": {str(k): int((modes == k).sum()) for k in range(3)}, "weighted": weight is not None,
            "weight_summary": None if weight is None else {"min": float(weight.min()), "max": float(weight.max()), "mean": float(weight.mean())}}


def predict_bag(bag: dict, features: pd.DataFrame) -> np.ndarray:
    frame, _ = _matrix(features, bag["names"], bag["categories"])
    probabilities = np.zeros((len(frame), 3))
    for member in bag["members"]:
        probabilities += member["booster"].predict(frame, num_iteration=member["rounds"]) / len(bag["members"])
    if not np.isfinite(probabilities).all():
        raise ValueError("Classifier produced invalid probabilities.")
    return probabilities


def candidate_names(candidate: dict) -> list[str]:
    """v9's classifier features (minus any the candidate declares in `drop`, e.g. the calendar month) plus the candidate's feature set."""
    drop = set(candidate.get("drop", []))
    if drop - set(FEATURE_NAMES):
        raise KeyError(f"drop names not in v9's features: {sorted(drop - set(FEATURE_NAMES))}")
    return [n for n in FEATURE_NAMES if n not in drop] + FEATURE_SETS[candidate["feature_set"]]


def mixture_for(bag: dict, component: dict, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(mixture value, P(fill) renormalised over fill/normal) for `features` rows."""
    probabilities = predict_bag(bag, features)
    interval = features["schedule_interval_s"].to_numpy(dtype=float)
    fill = probabilities[:, 0] / (probabilities[:, 0] + probabilities[:, 1])
    return mixture_values(probabilities, interval, component), fill


__all__ = ["PA_NUMERIC", "CS_NUMERIC", "OP_NUMERIC", "OPD_NUMERIC", "RAW_COLUMNS", "FEATURE_SETS", "previous_arrival_features", "callsign_rates",
           "operator_descriptors", "stake_weights", "population_table", "modes_of", "fit_bag", "predict_bag", "candidate_names", "mixture_for",
           "normal_component", "band_values"]
