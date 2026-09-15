"""Produce a local data dictionary and factual audit of the supplied PRC files.

No model fitting or submissions are performed. Run after prc_data.py download.
"""

from __future__ import annotations

import json
import platform
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw"
DOCS = ROOT / "docs/competition"
ID = "MVT_ID_mvt"
TARGET = "TAXITIME_SEC_mvt"
PHASE = "PHASE_mvt"
MVT_TIME = "MVT_TIME_UTC_mvt"
BLOCK_TIME = "BLOCK_TIME_UTC_mvt"


def target_summary(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    result = {"rows": len(values), "missing_or_nonfinite": int((~np.isfinite(values)).sum())}
    if not len(finite):
        return result
    result.update({
        "mean": float(finite.mean()), "std": float(finite.std()),
        "min": float(finite.min()), "max": float(finite.max()),
        "zero": int((finite == 0).sum()), "negative": int((finite < 0).sum()),
        "over_1_hour": int((finite > 3600).sum()),
        "over_3_hours": int((finite > 10800).sum()),
        "quantiles": {str(q): float(np.quantile(finite, q)) for q in [0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999]},
    })
    return result


def main():
    training_files = sorted(DATA.glob("training_*.parquet"))
    if len(training_files) != 12:
        raise SystemExit(f"Expected 12 monthly training files, found {len(training_files)}.")
    files = training_files + [DATA / "ranking.parquet", DATA / "submitting.parquet"]
    profile = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), **{name: version(name) for name in ["pyarrow", "pandas", "numpy", "boto3", "beautifulsoup4", "python-dotenv"]}},
        "files": [], "training": {}, "ranking": {}, "submission": {},
    }
    training_rows = 0
    training_dep_rows = 0
    phase_counts = Counter()
    dep_missing = Counter()
    movement_counts = Counter()
    dep_targets = defaultdict(list)
    all_targets = []
    train_ids = []
    target_equation_count = 0
    target_equation_mismatches = 0
    proxy_errors = defaultdict(list)
    for path in files:
        parquet = pq.ParquetFile(path)
        record = {
            "filename": path.name, "bytes": path.stat().st_size,
            "rows": parquet.metadata.num_rows, "columns": len(parquet.schema_arrow),
            "row_groups": parquet.metadata.num_row_groups,
            "schema": [{"name": field.name, "type": str(field.type), "nullable": field.nullable} for field in parquet.schema_arrow],
        }
        frame = parquet.read().to_pandas()
        record["null_counts"] = {col: int(count) for col, count in frame.isna().sum().items()}
        record["duplicate_movement_ids"] = int(frame[ID].duplicated().sum())
        if path.name == "submitting.parquet":
            profile["submission"] = {
                "rows": len(frame), "columns": list(frame.columns),
                "duplicate_ids": int(frame[ID].duplicated().sum()),
                "null_ids": int(frame[ID].isna().sum()),
                "blank_targets": int(frame[TARGET].isna().sum()),
                "all_ids_integer_valued": bool((frame[ID] % 1 == 0).all()),
                "ranking_departure_ids_match_as_set": bool(set(frame[ID]) == set(ranking_departure_ids)),
                "ranking_departure_ids_match_in_order": bool(np.array_equal(frame[ID].to_numpy(), ranking_departure_ids)),
            }
        else:
            phases = frame[PHASE].value_counts(dropna=False).to_dict()
            record["phase_counts"] = {str(k): int(v) for k, v in phases.items()}
            record["movement_time_min_utc"] = str(frame[MVT_TIME].min())
            record["movement_time_max_utc"] = str(frame[MVT_TIME].max())
            dep = frame.loc[frame[PHASE] == "DEP"]
            airports = frame["ADEP_mvt"].where(frame[PHASE] == "DEP", frame["ADES_mvt"])
            counts = pd.DataFrame({"airport": airports, "phase": frame[PHASE]}).value_counts().to_dict()
            record["movements_by_airport_phase"] = {f"{k[0]}/{k[1]}": int(v) for k, v in sorted(counts.items())}
            record["departures_target_seconds"] = target_summary(dep[TARGET].to_numpy())
            if path.name.startswith("training_"):
                training_rows += len(frame)
                training_dep_rows += len(dep)
                phase_counts.update(phases)
                dep_missing.update({k: int(v) for k, v in dep.isna().sum().items()})
                movement_counts.update(record["movements_by_airport_phase"])
                all_targets.append(dep[TARGET].to_numpy(dtype=float))
                train_ids.append(frame[ID].to_numpy())
                equation = (dep[MVT_TIME] - dep[BLOCK_TIME]).dt.total_seconds()
                comparable = equation.notna() & dep[TARGET].notna()
                target_equation_count += int(comparable.sum())
                target_equation_mismatches += int((equation[comparable] != dep.loc[comparable, TARGET]).sum())
                for airport, group in dep.groupby("ADEP_mvt"):
                    dep_targets[airport].append(group[TARGET].to_numpy(dtype=float))
                    raw_proxy = (group[MVT_TIME] - group["AOBT_3_flt"]).dt.total_seconds()
                    errors = (raw_proxy - group[TARGET]).dropna().to_numpy(dtype=float)
                    proxy_errors[airport].append(errors)
            else:
                ranking_departure_ids = dep[ID].to_numpy()
                profile["ranking"] = {
                    "rows": len(frame), "phase_counts": record["phase_counts"],
                    "months": {str(k): int(v) for k, v in frame[MVT_TIME].dt.strftime("%Y-%m").value_counts().sort_index().items()},
                    "departure_null_counts": {col: int(count) for col, count in dep.isna().sum().items()},
                    "arrivals_with_labels": int(frame.loc[frame[PHASE] == "ARR", TARGET].notna().sum()),
                    "movements_by_airport_phase": record["movements_by_airport_phase"],
                    "departure_ids_shared_with_training": int(np.isin(ranking_departure_ids, np.concatenate(train_ids)).sum()),
                }
                profile["ranking"]["schema_matches_training"] = record["schema"] == profile["files"][0]["schema"]
        profile["files"].append(record)
        print(f"Profiled {path.name}: {len(frame):,} rows", flush=True)
    ids = np.concatenate(train_ids)
    profile["training"] = {
        "rows": training_rows, "departure_rows": training_dep_rows,
        "phase_counts": dict(phase_counts), "departure_null_counts": dict(dep_missing),
        "movements_by_airport_phase": dict(sorted(movement_counts.items())),
        "departure_target_seconds": target_summary(np.concatenate(all_targets)),
        "departure_target_seconds_by_airport": {airport: target_summary(np.concatenate(values)) for airport, values in sorted(dep_targets.items())},
        "duplicate_movement_ids_across_months": int(len(ids) - len(np.unique(ids))),
        "target_equation_comparable_rows": target_equation_count,
        "target_equation_mismatch_rows": target_equation_mismatches,
        "schemas_consistent_across_training_months": all(row["schema"] == profile["files"][0]["schema"] for row in profile["files"][:12]),
    }
    raw_proxy_report = {}
    for airport, chunks in sorted(proxy_errors.items()):
        errors = np.concatenate(chunks)
        raw_proxy_report[airport] = {
            "comparable_rows": len(errors), "rmse_seconds": float(np.sqrt(np.mean(errors ** 2))),
            "mae_seconds": float(np.mean(np.abs(errors))), "median_error_seconds": float(np.median(errors)),
            "exact_matches": int((errors == 0).sum()),
        }
    profile["supplied_aobt3_diagnostic"] = {
        "description": "Unfitted arithmetic diagnostic on training departures only: (MVT_TIME_UTC_mvt - AOBT_3_flt) in seconds versus the supplied taxi-out target. This is not a validation or leaderboard result.",
        "by_airport": raw_proxy_report,
    }
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "data_profile.json").write_text(json.dumps(profile, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    schema = profile["files"][0]["schema"]
    lines = ["# Observed data schema", "", "Generated from the downloaded Parquet files. Field meanings are documented in [the official data page](pages/data.md).", "", "| Column | Arrow type | Missing in training departures | Missing in ranking departures |", "| --- | --- | ---: | ---: |"]
    for field in schema:
        col = field["name"]
        trn = dep_missing[col]
        rank = profile["ranking"]["departure_null_counts"][col]
        lines.append(f"| `{col}` | `{field['type']}` | {trn:,} / {training_dep_rows:,} | {rank:,} / {len(ranking_departure_ids):,} |")
    (DOCS / "DATA_SCHEMA.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"training_rows": training_rows, "training_departures": training_dep_rows, "ranking": profile["ranking"]["phase_counts"], "submission": profile["submission"]}, indent=2))


if __name__ == "__main__":
    main()
