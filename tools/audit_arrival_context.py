"""Audit past supplied arrival context on protected fitting rows only."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json
from prc2026.features import ID, PHASE, TIME, BLOCK, TARGET
from prc2026.validation import identity_hash
from prc2026_nmclock.bundle import load_bundle
from prc2026_phase3.bundle import load_partition
from prc2026_phase3.features import clean_codes, MISSING

SPECS = [
    ("airport", 3600, ["count", "raw_mean", "normal_mean", "negative_fraction", "over3h_fraction"]),
    ("airport", 10800, ["count", "raw_mean", "normal_mean", "negative_fraction", "over3h_fraction"]),
    ("runway", 10800, ["count", "normal_mean"]),
    ("stand", 604800, ["count", "normal_mean"]),
]
QUERY_COLUMNS = [ID, "FLIGHT_ID_mvt", PHASE, TIME, "ADEP_mvt", "RUNWAY_mvt", "STAND_mvt", "AOBT_3_flt", "LOBT_flt"]
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


def direct_check(queries, arrivals, saved):
    indexes = np.unique(np.linspace(0, len(queries) - 1, min(64, len(queries)), dtype=int))
    airport_groups = {name: frame for name, frame in arrivals.groupby("airport", sort=False)}
    delta = 0.0
    for index in indexes:
        row = queries.iloc[index]
        pool = airport_groups[row.airport]
        for group, window, names in SPECS:
            valid = pool.completion.lt(row[TIME]) & pool.completion.ge(row[TIME] - pd.Timedelta(seconds=window))
            if group != "airport":
                valid &= pool[group].eq(row[group]) & pool[group].ne(MISSING)
            if pd.notna(row.FLIGHT_ID_mvt):
                valid &= pool.FLIGHT_ID_mvt.ne(row.FLIGHT_ID_mvt)
            actual = finish(moments(pool.loc[valid, TARGET]).sum(axis=0, keepdims=True))
            for name in names:
                expected = float(actual[name][0])
                observed = float(saved.iloc[index][f"arrival_{group}_{window}s_{name}"])
                if np.isnan(expected) and np.isnan(observed):
                    continue
                if not np.isfinite(expected) or not np.isfinite(observed):
                    raise ValueError("Direct arrival context missingness differs.")
                delta = max(delta, abs(expected - observed))
    if delta > 1e-8:
        raise ValueError("Direct event scans do not reproduce arrival context.")
    return {"query_rows": len(indexes), "feature_values": len(indexes) * (len(saved.columns) - 1), "maximum_absolute_difference": delta}


def describe(frame, names, labels=False):
    result = {"rows": len(frame), "features": {}}
    groups = [frame.airport, frame.nmclock_available]
    residual = frame.residual - frame.groupby(["airport", "nmclock_available"], observed=True).residual.transform("mean") if labels else None
    for name in names:
        x = frame[name]
        row = {"finite_rows": int(np.isfinite(x).sum()), "nonzero_rows": int((x.notna() & x.ne(0)).sum())}
        if labels:
            centered = x - x.groupby(groups, observed=True).transform("mean")
            valid = centered.notna() & residual.notna()
            if valid.sum() > 20 and centered.loc[valid].std() > 0 and residual.loc[valid].std() > 0:
                row["pearson_centered_raw_proxy_residual"] = float(centered.loc[valid].corr(residual.loc[valid]))
                row["spearman_centered_raw_proxy_residual"] = float(centered.loc[valid].corr(residual.loc[valid], method="spearman"))
        result["features"][name] = row
    return result


def main():
    destination = ROOT / "docs/next_lever/arrival_context_audit.json"
    declaration_path = ROOT / "docs/next_lever/arrival_context_audit_declaration.json"
    if destination.exists():
        raise ValueError("Preserve the original arrival-context audit.")
    declaration = read_json(declaration_path)
    assert not declaration["audit_used"] and declaration["models_fitted"] == 0
    base = ROOT / "data/nmclock/72e86725e23367ede5b5"
    manifest, parent, original, catalog = load_bundle(ROOT, base)
    folds = {fold: pd.read_parquet(parent / "splits" / f"{fold}.parquet") for fold in ["seasonal", "forward"]}
    fitting = set.intersection(*[set(frame.loc[frame.partition.eq("fit"), ID]) for frame in folds.values()])
    protected = set.union(*[set(frame.loc[~frame.partition.eq("fit"), ID]) for frame in folds.values()])
    assert not fitting.intersection(protected)
    queries, arrivals, hashes = [], [], {}
    ranges = [("2025-01-01", "2025-12-01"), ("2026-01-01", "2026-02-01"), ("2026-07-01", "2026-08-01")]
    for record in read_json(ROOT / "docs/competition/download_manifest.json")["objects"]:
        if record["key"] == "submitting.parquet":
            continue
        path = ROOT / "data/raw" / record["key"]
        if sha256_file(path) != record["sha256"]:
            raise ValueError("Original supplied input changed.")
        hashes[record["key"]] = record["sha256"]
        filters = [[(PHASE, "==", "ARR"), (TIME, ">=", pd.Timestamp(start, tz="UTC")), (TIME, "<", pd.Timestamp(end, tz="UTC"))] for start, end in ranges]
        arrival = pd.read_parquet(path, columns=ARRIVAL_COLUMNS, filters=filters)
        assert arrival[PHASE].eq("ARR").all() and not arrival[TIME].dt.month.eq(12).any()
        arrivals.append(arrival)
        query = pd.read_parquet(path, columns=QUERY_COLUMNS, filters=[(PHASE, "==", "DEP")])
        queries.append(query.loc[query[ID].isin(fitting) | (query[TIME].dt.year.eq(2026) & query[TIME].dt.month.isin([1, 7]))])
    queries, arrivals = pd.concat(queries, ignore_index=True), pd.concat(arrivals, ignore_index=True)
    assert not queries[ID].duplicated().any() and not arrivals[ID].duplicated().any()
    queries["airport"], arrivals["airport"] = clean_codes(queries.ADEP_mvt), clean_codes(arrivals.ADES_mvt)
    for frame in [queries, arrivals]:
        for name in ["runway", "stand"]:
            frame[name] = clean_codes(frame[name.upper() + "_mvt"])
    queries["nmclock_available"] = (queries.AOBT_3_flt.notna() | queries.LOBT_flt.notna()).astype(int)
    training = queries.loc[queries[ID].isin(fitting)]
    samples = [training.loc[training.nmclock_available.eq(0)]]
    for _, group in training.groupby("airport", sort=True):
        group = group.sort_values(ID)
        samples.append(group.iloc[np.linspace(0, len(group) - 1, min(10000, len(group)), dtype=int)])
    sampled = pd.concat(samples).drop_duplicates(ID)
    ranking = queries.loc[queries[TIME].dt.year.eq(2026)]
    queries = pd.concat([sampled, ranking], ignore_index=True)
    valid = arrivals[BLOCK].notna() & arrivals[TARGET].notna()
    excluded_context_rows = int((~valid).sum())
    arrivals = arrivals.loc[valid].reset_index(drop=True)
    arrivals["completion"] = arrivals[TIME].where(arrivals[TIME].ge(arrivals[BLOCK]), arrivals[BLOCK])
    features = context(queries, arrivals)
    checked = direct_check(queries, arrivals, features)
    assignment = catalog[[ID]].copy()
    assignment["partition"] = np.where(assignment[ID].isin(sampled[ID]), "fit", "excluded")
    base_features, truth = load_partition(base, parent, catalog, assignment, "fit", ["conditional_schedule_offset_s", "nmclock_available", "nmclock_proxy_s"])
    frame = queries[[ID, TIME, "airport", "nmclock_available"]].merge(features, on=ID, validate="one_to_one")
    fitting_frame = frame.loc[frame[ID].isin(sampled[ID])].merge(truth[[ID, TARGET]], on=ID, validate="one_to_one").merge(base_features.drop(columns="nmclock_available"), on=ID, validate="one_to_one")
    assert set(fitting_frame[ID]).issubset(fitting) and not set(fitting_frame[ID]).intersection(protected)
    fitting_frame["residual"] = fitting_frame[TARGET] - np.where(fitting_frame.nmclock_available.eq(1), fitting_frame.nmclock_proxy_s, fitting_frame.conditional_schedule_offset_s)
    names = list(features.columns[1:])
    results = {"all": describe(fitting_frame, names, labels=True)}
    for value in [0, 1]:
        results[f"clock_available_{value}"] = describe(fitting_frame.loc[fitting_frame.nmclock_available.eq(value)], names, labels=True)
    artifact = ROOT / "runs/research_next_lever/arrival_context_features.parquet"
    if artifact.exists():
        raise ValueError("Preserve original audit features.")
    features.to_parquet(artifact, index=False, compression="zstd")
    report = {
        "created_at_utc": utc_now(), "code_sha256": sha256_file(Path(__file__)), "declaration_sha256": sha256_file(declaration_path),
        "input_sha256": hashes, "query_raw_columns": QUERY_COLUMNS, "arrival_only_raw_columns": ARRIVAL_COLUMNS,
        "fitting_rows": len(fitting_frame), "fitting_ids_sha256": identity_hash(fitting_frame[ID]),
        "fitting_months": sorted(fitting_frame[TIME].dt.strftime("%Y-%m").unique()),
        "arrival_context_rows": len(arrivals), "arrival_rows_excluded_missing_context": excluded_context_rows,
        "independent_direct_scan": checked, "fitting_signal": results,
        "ranking_covariate_coverage": describe(frame.loc[frame[TIME].dt.year.eq(2026)], names),
        "feature_artifact": artifact.relative_to(ROOT).as_posix(), "feature_artifact_sha256": sha256_file(artifact),
        "audit_used": False, "models_fitted": 0, "predictions_changed": False,
        "interpretation": "Only supplied past completed arrivals enter context. Departure labels are restricted to a common-fitting sample; these are raw offset-residual correlations, not incremental v6 improvement or complete validation scores. Arrival quality means have explicit population flags; no departure is clipped or filtered. A model experiment needs a new declaration.",
    }
    write_json(destination, report)
    print({"fitting_rows": len(fitting_frame), "ranking_rows": report["ranking_covariate_coverage"]["rows"], "independent_check": checked,
           "signals": {cohort: {name: row.get("spearman_centered_raw_proxy_residual") for name, row in result["features"].items() if name.endswith("mean")} for cohort, result in results.items()}})


if __name__ == "__main__":
    main()
