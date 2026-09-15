"""Check scheduled-clock agreement and fitting-only value of arrival-record linkage."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json
from prc2026.features import ID, TIME, TARGET
from prc2026.validation import identity_hash
from prc2026_nmclock.bundle import load_bundle
from prc2026_phase3.bundle import load_partition


def loss(values):
    values = np.asarray(values, dtype=float)
    return {"rows": len(values), "rmse_seconds": float(np.sqrt(np.mean(values*values))) if len(values) else None,
            "mae_seconds": float(np.mean(np.abs(values))) if len(values) else None,
            "absolute_error_p95_seconds": float(np.quantile(np.abs(values), .95)) if len(values) else None}


def main():
    destination = ROOT/"docs/next_lever/clock_relink_quality.json"
    if destination.exists():
        raise ValueError("Preserve the original linkage quality audit.")
    prior = read_json(ROOT/"docs/next_lever/clock_relink_coverage.json")
    pairs = pd.read_parquet(ROOT/prior["candidate_file"])
    ids = set(pairs.query_id) | set(pairs.peer_id)
    parts = []
    columns = [ID, TIME, "SCHED_TIME_UTC_mvt", "EOBT_1_flt", "IOBT_flt"]
    for name, digest in prior["input_sha256"].items():
        path = ROOT/"data/raw"/name
        if sha256_file(path) != digest:
            raise ValueError("Raw supplied input changed.")
        part = pd.read_parquet(path, columns=columns)
        parts.append(part.loc[part[ID].isin(ids)])
    clocks = pd.concat(parts, ignore_index=True)
    own = clocks[[ID, "SCHED_TIME_UTC_mvt"]].rename(columns={ID: "query_id"})
    peer = clocks[[ID, "EOBT_1_flt", "IOBT_flt"]].rename(columns={ID: "peer_id"})
    pairs = pairs.merge(own, on="query_id", validate="one_to_one").merge(peer, on="peer_id", validate="many_to_one")
    agreement = pd.concat([(pairs.SCHED_TIME_UTC_mvt-pairs.EOBT_1_flt).dt.total_seconds().abs(),
                           (pairs.SCHED_TIME_UTC_mvt-pairs.IOBT_flt).dt.total_seconds().abs()], axis=1).min(axis=1)
    pairs["schedule_difference_s"] = agreement
    pairs["schedule_agrees_30m"] = agreement.le(1800)
    quality = {}
    for year in [2025, 2026]:
        quality[str(year)] = {}
        for cohort in ["missing", "known_control"]:
            population = pairs.loc[pairs.query_time.dt.year.eq(year) & pairs.cohort.eq(cohort)]
            row = {}
            for rule, mask in [("unique", np.ones(len(population), dtype=bool)), ("schedule_30m", population.schedule_agrees_30m),
                               ("schedule_30m_and_aircraft", population.schedule_agrees_30m & population.aircraft_confirmed)]:
                group = population.loc[mask]
                row[rule] = {"rows": len(group), "by_airport": group.groupby("airport").size().astype(int).to_dict()}
                if cohort == "known_control":
                    row[rule]["matching_original_flight_ids"] = int(group.peer_flight_id.eq(group.original_flight_id).sum())
                    row[rule]["different_original_flight_ids"] = int(group.peer_flight_id.ne(group.original_flight_id).sum())
            quality[str(year)][cohort] = row
    base = ROOT/"data/nmclock/72e86725e23367ede5b5"
    manifest, parent, original, catalog = load_bundle(ROOT, base)
    folds = {fold: pd.read_parquet(parent/"splits"/f"{fold}.parquet") for fold in ["seasonal", "forward"]}
    fitting = set.intersection(*[set(f.loc[f.partition.eq("fit"), ID]) for f in folds.values()])
    protected = set.union(*[set(f.loc[~f.partition.eq("fit"), ID]) for f in folds.values()])
    assert not fitting.intersection(protected)
    candidates = pairs.loc[pairs.cohort.eq("missing") & pairs.query_id.isin(fitting)].copy()
    assignment = catalog[[ID]].copy()
    assignment["partition"] = np.where(assignment[ID].isin(candidates.query_id), "fit", "excluded")
    features, truth = load_partition(base, parent, catalog, assignment, "fit", ["airport", "nmclock_available", "conditional_schedule_offset_s"])
    assert features.nmclock_available.eq(0).all()
    frame = features.merge(truth[[ID, TIME, TARGET]], on=ID, validate="one_to_one").merge(candidates, left_on=ID, right_on="query_id", validate="one_to_one", suffixes=("", "_linked"))
    groups = {}
    for rule, mask in [("unique", np.ones(len(frame), dtype=bool)), ("schedule_30m", frame.schedule_agrees_30m),
                       ("schedule_30m_and_aircraft", frame.schedule_agrees_30m & frame.aircraft_confirmed)]:
        selected = frame.loc[mask]
        groups[rule] = {}
        for name, group in [("all", selected), *[(str(airport), part) for airport, part in selected.groupby("airport", sort=True)]]:
            groups[rule][name] = {"relinked_nm_proxy_error": loss(group[TARGET]-group.relinked_proxy_s),
                                  "original_conditional_schedule_offset_error": loss(group[TARGET]-group.conditional_schedule_offset_s),
                                  "target_over_3h_rows": int(group[TARGET].gt(10800).sum())}
    qualified = ROOT/"runs/research_next_lever/clock_relink_reviewed.parquet"
    if qualified.exists():
        raise ValueError("Preserve reviewed linkage candidates.")
    pairs.to_parquet(qualified, index=False, compression="zstd")
    report = {"created_at_utc": utc_now(), "code_sha256": sha256_file(Path(__file__)), "prior_audit_sha256": sha256_file(ROOT/"docs/next_lever/clock_relink_coverage.json"),
              "covariate_control_quality": quality, "fitting_only_errors": groups, "fitting_rows": len(frame),
              "fitting_ids_sha256": identity_hash(frame[ID]), "fitting_months": sorted(frame[TIME].dt.strftime("%Y-%m").unique()),
              "reviewed_candidates": qualified.relative_to(ROOT).as_posix(), "reviewed_candidates_sha256": sha256_file(qualified),
              "labels_scope": "Only linked missing-clock rows within the intersection of both original fitting sets; no scoring, stopping, December or ranking labels.",
              "audit_used": False, "models_fitted": 0, "predictions_changed": False,
              "interpretation": "Arithmetic raw-proxy errors do not measure improvement over v6. Schedule agreement is a pre-specified 30-minute quality condition; remaining identifier mismatches make automatic substitution unjustified without a declared full-fold check."}
    write_json(destination, report)
    print({"quality": quality, "fitting_all": {name: value["all"] for name, value in groups.items()}})


if __name__ == "__main__":
    main()
