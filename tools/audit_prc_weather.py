"""Measure label-free coverage of past IEM observations for competition inputs."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json
from prc2026.features import ID, PHASE, TIME
from prc2026.phase1 import validate_inputs


def main():
    root = Path(__file__).resolve().parents[1]
    inventory_path = root/"docs/competition/weather_20260914/inventory.json"
    inventory = read_json(inventory_path)
    if inventory["failures"] or len(inventory["objects"]) != 30:
        raise ValueError("Complete the declared 30 weather inputs before the coverage audit.")
    input_receipts = validate_inputs(root)
    weather = []
    for item in inventory["objects"]:
        path = root/item["path"]
        if sha256_file(path) != item["sha256"]:
            raise ValueError("Weather input changed.")
        frame = pd.read_csv(path, na_values=["M"])
        frame["valid"] = pd.to_datetime(frame["valid"], utc=True)
        begin, end = pd.Timestamp(item["start_inclusive"], tz="UTC"), pd.Timestamp(item["end_exclusive"], tz="UTC")
        frame = frame.loc[frame["valid"].ge(begin) & frame["valid"].lt(end)]
        weather.append(frame[["station", "valid"]])
    observation_times = pd.concat(weather, ignore_index=True).drop_duplicates().sort_values(["station", "valid"])
    original = read_json(root/"data/phase1/58d621b245de3f5e12dd/manifest.json")
    summaries, total_rows, covered_rows = {}, {}, {}
    for item in original["sources"]:
        source = item["source"]
        # The projection cannot expose departure labels or block clocks.
        frame = pd.read_parquet(root/"data/raw"/source, columns=[ID, PHASE, TIME, "ADEP_mvt"])
        frame = frame.loc[frame[PHASE].eq("DEP")]
        months = ["2026-01", "2026-07"] if source == "ranking.parquet" else [f"2025-{m:02}" for m in range(1, 12)]
        frame = frame.loc[frame[TIME].dt.strftime("%Y-%m").isin(months)]
        population = "ranking" if source == "ranking.parquet" else "january_november_training"
        for station, query in frame.groupby("ADEP_mvt", sort=True):
            observations = observation_times.loc[observation_times["station"].eq(station), "valid"]
            ts = observations.dt.as_unit("us").astype("int64").to_numpy()/1e6
            qt = query[TIME].dt.as_unit("us").astype("int64").to_numpy()/1e6
            hi = np.searchsorted(ts, qt-15*60, side="right")
            age = qt-ts[np.maximum(hi-1, 0)] if len(ts) else np.full(len(qt), np.inf)
            available = (hi > 0) & (age <= 90*60)
            key = population+"/"+station
            summary = summaries.setdefault(key, {"population": population, "airport": station, "rows": 0, "covered_rows": 0})
            summary["rows"] += len(query)
            summary["covered_rows"] += int(available.sum())
            total_rows[population] = total_rows.get(population, 0)+len(query)
            covered_rows[population] = covered_rows.get(population, 0)+int(available.sum())
    for summary in summaries.values():
        summary["coverage_fraction"] = summary["covered_rows"]/summary["rows"]
    if total_rows != {"ranking": 344841, "january_november_training": 1919370}:
        raise ValueError("Weather audit changed the audited departure population.")
    report = {"created_at_utc": utc_now(), "inventory_sha256": sha256_file(inventory_path),
              "license": inventory["license"], "official_input_receipts": input_receipts,
              "weather_join_for_coverage_only": "last observation at least 15 minutes before takeoff; maximum observation age 90 minutes",
              "publication_time_limitation": "METAR publication timestamps are absent; the 15-minute delay is an explicit conservative convention",
              "unique_station_times": len(observation_times), "departure_labels_read": False,
              "audit_labels_used": False, "model_trained": False,
              "overall": {p: {"rows": total_rows[p], "covered_rows": covered_rows[p],
                               "coverage_fraction": covered_rows[p]/total_rows[p]} for p in total_rows},
              "by_population_airport": summaries, "code_sha256": sha256_file(Path(__file__))}
    output = root/"docs/competition/weather_20260914/coverage.json"
    write_json(output, report)
    print(report["overall"])


if __name__ == "__main__":
    main()
