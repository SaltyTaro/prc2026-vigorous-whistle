"""Arrival rows on the committee networks' 314-column contract, for the arrival second task (configs/adsb_arr_design_v1.json).

One table per month: every arrival of the month as a role-swapped row. The reporting airport is ADES_mvt, the other end is ADEP_mvt,
and the NM own-airport / other-end fields are ADES_flt / ADEP_flt. The populated columns come from the FROZEN feature functions
(phase 1, 3, 4, weather, timeline; the December guards lifted as in src/prc2026_december), called on the arrival rows in the
departure role; every other contract column is missing (NaN). The arrival's own clocks (schedule, EOBT, IOBT, ARVT, NM off-block)
never enter a populated column. The arrival taxi-in (TAXITIME_SEC_mvt) is written as `taxi_in`: it is the second task's target and
never a feature. No departure label and no departure block time is read.

Context months: the month, its neighbours inside the same build (2025-01..11 as the frozen bundle; December with November; a
ranking month with the ranking file only). Histories look back six hours at most.

--verify-departures builds the same populated columns for the month's REAL departures through the same code path (no swap) and
requires exact equality with the frozen timeline bundle (or the December build): the check that this tool's use of the frozen
functions reproduces the contract. Nothing is written in that mode.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, PHASE, TARGET, TIME  # noqa: E402
from prc2026_phase3.features import CONTEXT_COLUMNS, INPUT_COLUMNS as PHASE3_INPUTS, row_features, traffic_features  # noqa: E402
from prc2026_phase4.features import INPUT_COLUMNS as PHASE4_INPUTS  # noqa: E402
from prc2026_weather.features import INPUT_COLUMNS as WEATHER_INPUTS, WEATHER_NAMES, observation_features, runway_geometry, verify_external  # noqa: E402
from prc2026_timeline.features import INPUT_COLUMNS as TIMELINE_INPUTS, SURFACE_NAMES, CATEGORICAL_COLUMNS  # noqa: E402
from prc2026_timeline.bundle import load_bundle  # noqa: E402
from prc2026_december import patched  # noqa: E402
from prc2026_december.features import in_months, observation_parts, weather_config  # noqa: E402
from run_adsb_learner import ADSB_NAMES, BUNDLE_NAMES, source_file  # noqa: E402

quality_features = patched.load("prc2026_phase4").quality_features
weather_features = patched.load("prc2026_weather").weather_features
surface_features = patched.load("prc2026_timeline").surface_features

FROZEN_MONTHS = [f"2025-{m:02}" for m in range(1, 12)]
DECEMBER = "2025-12"
RANKING_MONTHS = ["2026-01", "2026-07"]
SWAPS = [("ADEP_mvt", "ADES_mvt"), ("ADEP_flt", "ADES_flt")]
CLOCKS = ["AOBT_3_flt", "LOBT_flt"]
CALENDAR = ["utc_hour", "utc_weekday", "month", "minute_of_day", "local_hour", "local_weekday", "local_minute_of_day", "local_time_sin", "local_time_cos",
            "local_weekday_sin", "local_weekday_cos", "local_weekend", "utc_offset_hours", "day_of_year_sin", "day_of_year_cos", "movement_second"]
RECORD_ROW = ["nm_missing", "nm_origin_agrees", "nm_destination_agrees", "nm_aircraft_agrees", "nm_rule_agrees"]
RECORD_QUALITY = ["aircraft_missing", "flight_missing", "flight_designator_length", "flight_prefix_length", "flight_suffix_length", "flight_digit_count",
                  "flight_prefix_unique_letters", "flight_prefix_adjacent_repeats", "flight_prefix_tail_repeat", "flight_standard_shape"]
TRAFFIC = [f"{p}_{s}" for p, tail in [("airport_dep", ["last_s", "count_5m", "count_15m", "count_30m", "count_60m"]),
                                      ("airport_arr", ["last_s", "count_5m", "count_15m", "count_30m", "count_60m"]),
                                      ("runway_dep", ["last_s", "count_15m", "count_60m"]), ("runway_arr", ["last_s", "count_15m", "count_60m"]),
                                      ("dep_schedule_delay", ["last_s", "count_30m", "mean_30m", "std_30m", "count_60m", "mean_60m", "std_60m"]),
                                      ("arr_schedule_delay", ["last_s", "count_30m", "mean_30m", "std_30m", "count_60m", "mean_60m", "std_60m"])] for s in tail]
SURFACE = [n for n in SURFACE_NAMES if n.startswith(("surface_takeoff_airport_", "surface_takeoff_runway_"))]
ROW_CATEGORICAL = ["airport", "destination", "runway", "stand", "aircraft_type", "movement_rule", "nm_origin", "nm_destination", "nm_aircraft_type",
                   "market_segment", "flight_type", "wake_category", "operator"]
QUALITY_CATEGORICAL = ["record_flight_prefix", "record_flight_shape"]
POPULATED_NUMERIC = [*CALENDAR, *RECORD_ROW, *RECORD_QUALITY, *TRAFFIC, *WEATHER_NAMES, *SURFACE]
POPULATED = [*ROW_CATEGORICAL, *QUALITY_CATEGORICAL, *POPULATED_NUMERIC]
CODE_FILES = ["tools/build_arr_rows.py", "src/prc2026/features.py", "src/prc2026_phase3/features.py", "src/prc2026_phase4/features.py",
              "src/prc2026_weather/features.py", "src/prc2026_timeline/features.py", "src/prc2026_december/patched.py"]


def contract() -> list[str]:
    extra = read_json(ROOT / "configs/adsb_nn_network_v1.json")["defaults"]["features"]["extra_features"]
    names = [*BUNDLE_NAMES, *ADSB_NAMES, *extra]
    if len(names) != 314 or len(set(names)) != 314 or set(POPULATED) - set(names) or len(SURFACE) != 24 or len(TRAFFIC) != 30:
        raise ValueError("The contract or the populated column list changed.")
    if sorted(n for n in names if n in CATEGORICAL_COLUMNS) != sorted([*ROW_CATEGORICAL, *QUALITY_CATEGORICAL]):
        raise ValueError("Every categorical contract column must be populated.")
    return names


def context_months(month: str) -> tuple[list[str], list[str]]:
    """(raw files, months) of the movement context for one month's rows."""
    if month in RANKING_MONTHS:
        return ["ranking.parquet"], [month]
    y, m = int(month[:4]), int(month[5:7])
    near = [f"{y}-{k:02}" for k in (m - 1, m, m + 1) if 1 <= k <= 12]
    allowed = [*FROZEN_MONTHS, DECEMBER] if month == DECEMBER else FROZEN_MONTHS
    months = [k for k in near if k in allowed]
    return [source_file(k) for k in months], months


def role_rows(raw: pd.DataFrame, swap: bool) -> pd.DataFrame:
    """The month's arrivals in the departure role (swap) or its real departures (no swap)."""
    rows = raw.loc[raw[PHASE].eq("ARR" if swap else "DEP")].reset_index(drop=True).copy()
    if swap:
        for left, right in SWAPS:
            a, b = rows[left].copy(), rows[right].copy()
            rows[left], rows[right] = b, a
        rows[PHASE] = "DEP"
    return rows


def read_raw(files: list[str], columns: list[str], months: list[str]) -> pd.DataFrame:
    return in_months(pd.concat([pd.read_parquet(ROOT / "data/raw" / f, columns=columns) for f in files], ignore_index=True), months)


def observations(manifest: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    config = weather_config(ROOT, manifest)
    external = verify_external(ROOT, config)
    inventories = [external["weather"]]
    december = ROOT / "docs/december/weather_inventory.json"
    if december.exists():
        inventories.append(read_json(december))
    obs = observation_features(pd.concat([p for inv in inventories for p in observation_parts(ROOT, inv)], ignore_index=True))
    runway_path = next(x["path"] for x in external["runways"]["objects"] if x["path"].endswith("runways.csv"))
    return obs, runway_geometry(pd.read_csv(ROOT / runway_path)), {"weather_objects": sum(len(i["objects"]) for i in inventories)}


def build(month: str, swap: bool, phase1: Path, obs: pd.DataFrame, geometry: pd.DataFrame) -> pd.DataFrame:
    """ID, the populated columns, and (swap only) the arrival taxi-in, for one month's rows."""
    source = "ranking.parquet" if month in RANKING_MONTHS else source_file(month)
    files, months = context_months(month)
    columns = list(dict.fromkeys([*PHASE3_INPUTS, *PHASE4_INPUTS, *WEATHER_INPUTS, *TIMELINE_INPUTS, *([TARGET] if swap else [])]))
    raw = in_months(pd.read_parquet(ROOT / "data/raw" / source, columns=columns), [month])
    rows = role_rows(raw, swap)
    if rows[ID].duplicated().any() or not len(rows):
        raise ValueError(f"{month}: no rows or duplicate identities.")
    table = row_features(rows[PHASE3_INPUTS])
    if not np.array_equal(table[ID].to_numpy(), rows[ID].to_numpy()):
        raise ValueError("Row features are misaligned.")
    context3 = in_months(pd.concat([pd.read_parquet(phase1 / "context" / f, columns=CONTEXT_COLUMNS) for f in files], ignore_index=True), months)
    traffic = traffic_features(context3, rows[[ID, TIME, "ADEP_mvt", "RUNWAY_mvt", "STAND_mvt"]])
    quality = quality_features(rows[PHASE4_INPUTS])
    weather = weather_features(rows[WEATHER_INPUTS], obs, geometry)
    context = read_raw(files, TIMELINE_INPUTS, months)
    if swap:
        mine = rows[TIMELINE_INPUTS].copy()
        for name in CLOCKS:  # arrival rows are queries only: without an off-block clock they are nobody's peer
            mine[name] = pd.Series(pd.NaT, index=mine.index, dtype=context[name].dtype)
        context = pd.concat([context.loc[context[PHASE].eq("DEP")], mine], ignore_index=True)
    surface = surface_features(context, rows[ID])
    for name, part in [("traffic", traffic), ("quality", quality), ("weather", weather)]:
        if not np.array_equal(part[ID].to_numpy(), rows[ID].to_numpy()):
            raise ValueError(f"{name} features are misaligned.")
    if len(surface) != len(rows):
        raise ValueError("surface features are misaligned.")
    sources = {**{n: table for n in [*ROW_CATEGORICAL, *CALENDAR, *RECORD_ROW]}, **{n: quality for n in [*QUALITY_CATEGORICAL, *RECORD_QUALITY]},
               **{n: traffic for n in TRAFFIC}, **{n: weather for n in WEATHER_NAMES}, **{n: surface for n in SURFACE}}
    out = pd.DataFrame({ID: rows[ID].to_numpy(), **{name: (sources[name][name].astype("string").to_numpy() if name in CATEGORICAL_COLUMNS
                                                          else sources[name][name].to_numpy(dtype="float32")) for name in POPULATED}})
    if swap:
        out["taxi_in"] = rows[TARGET].to_numpy(dtype=float)
        out["movement_time"] = rows[TIME].to_numpy()
    return out


def frozen_rows(month: str, base: Path, december: Path) -> pd.DataFrame:
    if month == DECEMBER:
        return pd.read_parquet(december / "features" / source_file(DECEMBER), columns=[ID, *POPULATED])
    source = "ranking.parquet" if month in RANKING_MONTHS else source_file(month)
    return pd.read_parquet(base / "features" / source, columns=[ID, *POPULATED])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--month", action="append", required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/adsb_arr/rows"))
    parser.add_argument("--december", type=Path, default=Path("data/december/d198dea10625b0a75977"))
    parser.add_argument("--verify-departures", action="store_true")
    args = parser.parse_args()
    names = contract()
    base = ROOT / read_json(ROOT / "data/timeline/latest.json")["bundle"]
    manifest, phase1, _, _ = load_bundle(ROOT, base)
    obs, geometry, weather_info = observations(manifest)
    code = {p: sha256_file(ROOT / p) for p in CODE_FILES}
    for month in args.month:
        started = time.time()
        table = build(month, not args.verify_departures, phase1, obs, geometry)
        if args.verify_departures:
            frozen = frozen_rows(month, base, ROOT / args.december).set_index(ID).reindex(table[ID]).reset_index()
            if frozen[POPULATED[0]].isna().any():
                raise ValueError(f"{month}: the frozen build does not hold every departure of the month.")
            rounding = {}
            for name in POPULATED:
                if name in CATEGORICAL_COLUMNS:
                    same = bool(np.all(frozen[name].astype("string").fillna("__MISSING__").to_numpy() == table[name].astype("string").fillna("__MISSING__").to_numpy()))
                else:
                    a, b = frozen[name].to_numpy(dtype="float32"), table[name].to_numpy(dtype="float32")
                    same = np.array_equal(a, b, equal_nan=True)
                    if not same and name.startswith("wx_") and np.array_equal(np.isnan(a), np.isnan(b)):
                        # the frozen bundle was prepared on Linux: wind components pass through libm sin / cos in float64, one ulp apart across platforms
                        gap = float(np.nanmax(np.abs(a.astype(float) - b.astype(float))))
                        same = gap <= 1e-5
                        rounding[name] = {"max_abs_difference": gap, "rows": int(np.sum(~((a == b) | (np.isnan(a) & np.isnan(b)))))}
                if not same:
                    raise ValueError(f"{month}: column {name} differs from the frozen build.")
            print({"month": month, "verified_departures": int(len(table)), "columns": len(POPULATED), "exact_columns": len(POPULATED) - len(rounding), "platform_rounding": rounding, "seconds": round(time.time() - started, 1)}, flush=True)
            continue
        full = pd.DataFrame({ID: table[ID].to_numpy(), **{name: (table[name].to_numpy() if name in table else np.full(len(table), np.nan, dtype="float32")) for name in names}})
        full["taxi_in"], full["movement_time"] = table["taxi_in"].to_numpy(), table["movement_time"].to_numpy()
        out = ROOT / args.output
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"arr_{month}.parquet"
        full.to_parquet(path, index=False, compression="zstd")
        record = {"created_at_utc": utc_now(), "month": month, "rows": int(len(full)), "columns": len(names), "populated": POPULATED, "finite_taxi_in": int(np.isfinite(full["taxi_in"]).sum()),
                  "airports": {k: int(v) for k, v in full["airport"].value_counts().items()}, "context_months": context_months(month)[1], **weather_info,
                  "weather_missing_rows": int(full["wx_missing"].sum()), "code": code, "sha256": sha256_file(path), "audit_used": False,
                  "note": "arrival rows only; taxi_in is the second task's target, never a feature; no departure label or departure block time read"}
        write_json(out / f"arr_{month}.json", record)
        print({k: record[k] for k in ["month", "rows", "finite_taxi_in", "weather_missing_rows"]}, {"seconds": round(time.time() - started, 1)}, flush=True)


if __name__ == "__main__":
    main()
