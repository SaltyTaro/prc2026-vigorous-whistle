"""Label-free, conservative callsign/route linkage to supplied arrival records."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json
from prc2026.features import ID, PHASE, TIME
from prc2026_phase3.features import clean_codes, MISSING

COLUMNS = [ID, PHASE, TIME, "FLIGHT_ID_mvt", "FLIGHT_mvt", "CALLSIGN_flt", "ADEP_mvt", "ADES_mvt",
           "ADEP_flt", "ADES_flt", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_TYPE_flt", "AOBT_3_flt", "LOBT_flt", "ARVT_3_flt"]
ALLOWED = [f"2025-{m:02}" for m in range(1, 12)]+["2026-01", "2026-07"]


def flight_codes(series):
    return series.astype("string").fillna("").str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)


def main():
    destination = ROOT/"docs/next_lever/clock_relink_coverage.json"
    if destination.exists():
        raise ValueError("Preserve the original linkage audit.")
    parts, hashes = [], {}
    for record in read_json(ROOT/"docs/competition/download_manifest.json")["objects"]:
        if record["key"] == "submitting.parquet":
            continue
        path = ROOT/"data/raw"/record["key"]
        digest = sha256_file(path)
        if digest != record["sha256"]:
            raise ValueError("Supplied inputs changed.")
        raw = pd.read_parquet(path, columns=COLUMNS)
        parts.append(raw.loc[raw[TIME].dt.strftime("%Y-%m").isin(ALLOWED)])
        hashes[record["key"]] = digest
    pool = pd.concat(parts, ignore_index=True)
    pool["airport"], pool["destination"] = clean_codes(pool.ADEP_mvt), clean_codes(pool.ADES_mvt)
    pool["aircraft"] = clean_codes(pool.AIRCRAFT_TYPE_mvt)
    pool["clock"] = pool.AOBT_3_flt.fillna(pool.LOBT_flt)
    dep = pool.loc[pool[PHASE].eq("DEP")].copy()
    missing = dep.loc[dep.clock.isna()].copy()
    # Positive controls use only movement-side query fields in the matcher.
    # Their original NM IDs and clocks are consulted afterward to measure linkage fidelity.
    control_parts = []
    known = dep.loc[dep.clock.notna() & dep.FLIGHT_ID_mvt.notna()].copy()
    known["year"] = known[TIME].dt.year
    for _, group in known.groupby(["year", "airport"], sort=True):
        group = group.sort_values(ID)
        control_parts.append(group.iloc[np.linspace(0, len(group)-1, min(2000, len(group)), dtype=int)])
    control = pd.concat(control_parts, ignore_index=True)
    queries = pd.concat([missing.assign(cohort="missing"), control.assign(cohort="known_control")], ignore_index=True)
    q = queries[[ID, TIME, "airport", "destination", "aircraft", "cohort"]].rename(columns={ID: "query_id", TIME: "query_time", "aircraft": "query_aircraft"})
    q["code"] = flight_codes(queries.FLIGHT_mvt).to_numpy()
    q["day"] = q.query_time.dt.floor("D")
    q = pd.concat([q, q.assign(day=q.day+pd.Timedelta(days=1))], ignore_index=True)
    q = q.loc[q.code.str.len().ge(3) & q.destination.ne(MISSING)]
    arrivals = pool.loc[pool[PHASE].eq("ARR") & pool.clock.notna() & pool.FLIGHT_ID_mvt.notna() & pool.ARVT_3_flt.notna()].copy()
    consistent_route = arrivals.airport.eq(clean_codes(arrivals.ADEP_flt)) & arrivals.destination.eq(clean_codes(arrivals.ADES_flt))
    consistent_arrival = (arrivals[TIME]-arrivals.ARVT_3_flt).dt.total_seconds().abs().le(1800)
    arrivals = arrivals.loc[consistent_route & consistent_arrival]
    p = arrivals[[ID, TIME, "FLIGHT_ID_mvt", "clock", "airport", "destination"]].rename(columns={ID: "peer_id", TIME: "peer_time", "FLIGHT_ID_mvt": "peer_flight_id"})
    p["peer_aircraft"] = clean_codes(arrivals.AIRCRAFT_TYPE_flt).to_numpy()
    p["day"] = p.peer_time.dt.floor("D")
    aliases = []
    for column in ["FLIGHT_mvt", "CALLSIGN_flt"]:
        alias = p.copy()
        alias["code"] = flight_codes(arrivals[column]).to_numpy()
        aliases.append(alias.loc[alias.code.str.len().ge(3)])
    peers = pd.concat(aliases, ignore_index=True).drop_duplicates(["peer_id", "code"])
    pairs = q.merge(peers, on=["airport", "destination", "code", "day"], how="inner", validate="many_to_many")
    elapsed = (pairs.peer_time-pairs.query_time).dt.total_seconds()
    taxi_proxy = (pairs.query_time-pairs.clock).dt.total_seconds()
    aircraft_known = pairs.query_aircraft.ne(MISSING) & pairs.peer_aircraft.ne(MISSING)
    aircraft_agrees = pairs.query_aircraft.eq(pairs.peer_aircraft)
    pairs = pairs.loc[elapsed.between(0, 86400) & taxi_proxy.between(0, 21600) & (~aircraft_known | aircraft_agrees)].copy()
    pairs["aircraft_confirmed"] = aircraft_known.loc[pairs.index] & aircraft_agrees.loc[pairs.index]
    pairs["relinked_proxy_s"] = taxi_proxy.loc[pairs.index]
    pairs = pairs.drop_duplicates(["query_id", "peer_flight_id", "clock"])
    multiplicity = pairs.groupby("query_id").size()
    unique = pairs.loc[pairs.query_id.isin(multiplicity.index[multiplicity.eq(1)])].copy()
    original = queries[[ID, "FLIGHT_ID_mvt", "clock"]].rename(columns={ID: "query_id", "FLIGHT_ID_mvt": "original_flight_id", "clock": "original_clock"})
    unique = unique.merge(original, on="query_id", how="left", validate="one_to_one")
    summaries = {}
    for year in [2025, 2026]:
        summaries[str(year)] = {}
        for cohort in ["missing", "known_control"]:
            population = queries.loc[queries[TIME].dt.year.eq(year) & queries.cohort.eq(cohort)]
            result = unique.loc[unique.query_time.dt.year.eq(year) & unique.cohort.eq(cohort)]
            row = {"query_rows": len(population), "unique_candidate_rows": len(result), "unique_with_confirmed_aircraft_rows": int(result.aircraft_confirmed.sum()),
                   "ambiguous_query_rows": int(population[ID].isin(multiplicity.index[multiplicity.gt(1)]).sum())}
            if cohort == "known_control":
                row["same_original_flight_id_rows"] = int(result.peer_flight_id.eq(result.original_flight_id).sum())
                row["different_original_flight_id_rows"] = int(result.peer_flight_id.ne(result.original_flight_id).sum())
                delta = (result.clock-result.original_clock).dt.total_seconds().abs()
                row["identical_original_clock_rows"] = int(delta.eq(0).sum())
                row["maximum_clock_difference_seconds"] = float(delta.max()) if len(result) else None
            else:
                row["by_airport"] = result.groupby("airport").size().astype(int).to_dict()
            summaries[str(year)][cohort] = row
    output = ROOT/"runs/research_next_lever"
    output.mkdir(exist_ok=True)
    candidate_path = output/"clock_relink_candidates.parquet"
    if candidate_path.exists():
        raise ValueError("Preserve the existing linkage candidate artifact.")
    unique.to_parquet(candidate_path, index=False, compression="zstd")
    report = {"created_at_utc": utc_now(), "code_sha256": sha256_file(Path(__file__)), "input_sha256": hashes,
              "input_columns": COLUMNS, "summaries": summaries, "candidate_file": candidate_path.relative_to(ROOT).as_posix(),
              "candidate_sha256": sha256_file(candidate_path), "labels_read": False, "airport_block_times_read": False,
              "audit_used": False, "models_fitted": 0, "predictions_changed": False,
              "matching_rule": "Exact normalized movement flight code against arrival movement-code or NM-callsign; exact origin/destination; unique NM flight and clock; arrival 0..24h after query; NM offblock 0..6h before query; peer airport/NM routes agree; flown arrival and recorded landing within 30m; reject known aircraft mismatch. No query NM fields enter matching.",
              "interpretation": "Potential covariate recovery and positive-control identity fidelity only. Unique matches may still be false; no clock is substituted and no model or submission is changed. Further review is required before a new feature experiment."}
    write_json(destination, report)
    print(summaries)


if __name__ == "__main__":
    main()
