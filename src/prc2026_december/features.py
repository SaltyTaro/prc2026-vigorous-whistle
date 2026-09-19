"""Six feature layers for one month's departures on top of the Phase-1 bundle, with the frozen functions (guards lifted).

`build(root, month, ...)` writes features/<source> with the timeline bundle's 335 columns (and the month's labels)
under data/december/<id>/; `month="2025-11"` against the frozen timeline bundle is the reproduction check.
The context is every 2025 month up to and including the month, read from the raw files with each layer's own
input projection (never a label or a departure block time): the windows are at most one hour, but the frozen
headway features look at the previous n same-runway departures wherever they are, so a rare runway's value can
depend on events months earlier and only the full history reproduces the frozen bundle exactly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from prc2026.artifacts import environment, object_hash, read_json, sha256_file, utc_now, write_json
from prc2026.features import FEATURE_NAMES as PHASE1_NAMES, ID, PHASE, TARGET, TIME
from prc2026.validation import identity_hash
from prc2026_phase3.features import ALL_FEATURE_NAMES as PHASE3_NAMES, CONTEXT_COLUMNS, INPUT_COLUMNS as PHASE3_INPUTS, row_features, traffic_features
from prc2026_phase4.features import FEATURE_NAMES as PHASE4_NAMES, INPUT_COLUMNS as PHASE4_INPUTS
from prc2026_phase5r.features import FEATURE_NAMES as PHASE5R_NAMES, INPUT_COLUMNS as PHASE5R_INPUTS
from prc2026_weather.features import FEATURE_NAMES as WEATHER_NAMES, INPUT_COLUMNS as WEATHER_INPUTS, observation_features, runway_geometry, verify_external
from prc2026_nmclock.features import FEATURE_NAMES as NMCLOCK_NAMES, INPUT_COLUMNS as NMCLOCK_INPUTS
from prc2026_timeline.features import FEATURE_NAMES as TIMELINE_NAMES, INPUT_COLUMNS as TIMELINE_INPUTS
from prc2026_timeline.bundle import load_bundle as load_timeline
from . import patched

quality_features = patched.load("prc2026_phase4").quality_features
ground_features = patched.load("prc2026_phase5r").ground_features
weather_features = patched.load("prc2026_weather").weather_features
clock_features = patched.load("prc2026_nmclock").clock_features
timeline_features = patched.load("prc2026_timeline").timeline_features


def source_file(month: str) -> str:
    y, m = int(month[:4]), int(month[5:7])
    return f"training_{y}-{m:02}-01_{y + (m == 12)}-{(m % 12) + 1:02}-01.parquet"


def previous_month(month: str) -> str:
    y, m = int(month[:4]), int(month[5:7])
    return f"{y - (m == 1)}-{(m - 2) % 12 + 1:02}"


def next_month(month: str) -> str:
    y, m = int(month[:4]), int(month[5:7])
    return f"{y + (m == 12)}-{m % 12 + 1:02}"


def in_months(frame: pd.DataFrame, months: list[str]) -> pd.DataFrame:
    return frame.loc[frame[TIME].dt.strftime("%Y-%m").isin(months)].reset_index(drop=True)


def raw_context(root: Path, files: list[str], columns: list[str], months: list[str]) -> pd.DataFrame:
    parts = [pd.read_parquet(root / "data/raw" / name, columns=columns) for name in files]
    return in_months(pd.concat(parts, ignore_index=True), months)


def observation_parts(root: Path, inventory: dict) -> list[pd.DataFrame]:
    """The frozen loader's per-object reading (prc2026_weather.features.load_observations), returned before the single feature pass."""
    parts = []
    for item in inventory["objects"]:
        raw = pd.read_csv(root / item["path"], na_values=["M"])
        raw["valid"] = pd.to_datetime(raw["valid"], utc=True)
        begin, end = pd.Timestamp(item["start_inclusive"], tz="UTC"), pd.Timestamp(item["end_exclusive"], tz="UTC")
        raw = raw.loc[raw["valid"].ge(begin) & raw["valid"].lt(end)].copy()
        if len(raw) != item["rows_in_period"] or not raw["station"].eq(item["station"]).all():
            raise ValueError("Weather source station or bounded population changed.")
        parts.append(raw)
    return parts


def weather_config(root: Path, timeline_manifest: dict) -> dict:
    """The frozen weather campaign's external-data configuration, reached through the bundle chain."""
    nmclock = read_json(root / timeline_manifest["previous_bundle"] / "manifest.json")
    weather = read_json(root / nmclock["previous_bundle"] / "manifest.json")
    return weather["descriptor"]["config"]


def aligned(previous: pd.DataFrame, extra: pd.DataFrame, names: list[str], layer: str) -> pd.DataFrame:
    if not np.array_equal(previous[ID].to_numpy(), extra[ID].to_numpy()):
        raise ValueError(f"{layer}: feature rows are misaligned.")
    return pd.concat([previous, extra.drop(columns=ID)], axis=1)[[ID, *names]]


def build(root: Path, month: str, output: Path, extra_weather_inventories: list[dict] | None = None, verify: bool = False) -> dict:
    manifest, phase1, original, catalog = load_timeline(root, root / read_json(root / "data/timeline/latest.json")["bundle"])
    source = source_file(month)
    months = [f"2025-{m:02}" for m in range(1, int(month[5:7]) + 1)]  # the frozen builds used every allowed month as context
    files = [source_file(f"2025-{m:02}") for m in range(1, 13)]  # every 2025 file, month-filtered (files hold a few rows across their boundaries)
    # Phase 1 (frozen bundle rows of this file) and the month's query population
    base1 = pd.read_parquet(phase1 / "features" / source)
    raw3 = in_months(pd.read_parquet(root / "data/raw" / source, columns=PHASE3_INPUTS), [month])
    features3 = row_features(raw3)
    baseline = base1.loc[base1[ID].isin(features3[ID])].reset_index(drop=True)
    pd.testing.assert_frame_equal(features3[[ID, *PHASE1_NAMES]], baseline, check_exact=True)
    # Phase 3 traffic: Phase-1 movement context of the surrounding files, month-filtered
    context3 = in_months(pd.concat([pd.read_parquet(phase1 / "context" / f, columns=CONTEXT_COLUMNS) for f in files], ignore_index=True), months)
    queries3 = raw3.loc[raw3[PHASE].eq("DEP"), [ID, TIME, "ADEP_mvt", "RUNWAY_mvt", "STAND_mvt"]].reset_index(drop=True)
    table = aligned(features3, traffic_features(context3, queries3), PHASE3_NAMES, "phase3")
    # Phase 4 row quality
    table = aligned(table, quality_features(in_months(pd.read_parquet(root / "data/raw" / source, columns=PHASE4_INPUTS), [month])), PHASE4_NAMES, "phase4")
    # Phase 5R ground demand / service
    table = aligned(table, ground_features(raw_context(root, files, PHASE5R_INPUTS, months), table[ID]), PHASE5R_NAMES, "phase5r")
    # Weather: the frozen thirty objects (2025 training year and the ranking months) plus any December objects, one feature pass
    config = weather_config(root, manifest)
    external = verify_external(root, config)
    inventories = [external["weather"], *(extra_weather_inventories or [])]
    observations = observation_features(pd.concat([p for inv in inventories for p in observation_parts(root, inv)], ignore_index=True))
    runway_path = next(x["path"] for x in external["runways"]["objects"] if x["path"].endswith("runways.csv"))
    geometry = runway_geometry(pd.read_csv(root / runway_path))
    raw_w = in_months(pd.read_parquet(root / "data/raw" / source, columns=WEATHER_INPUTS), [month])
    queries_w = table[[ID]].merge(raw_w, on=ID, how="left", validate="one_to_one", sort=False)
    table = aligned(table, weather_features(queries_w, observations, geometry), WEATHER_NAMES, "weather")
    # NM clocks
    raw_n = in_months(pd.read_parquet(root / "data/raw" / source, columns=NMCLOCK_INPUTS), [month])
    table = aligned(table, clock_features(table[[ID]].merge(raw_n, on=ID, how="left", validate="one_to_one", sort=False)), NMCLOCK_NAMES, "nmclock")
    # Timeline: flown arrival and surface context
    table = aligned(table, timeline_features(raw_context(root, files, TIMELINE_INPUTS, months), table[ID]), TIMELINE_NAMES, "timeline")
    if list(table.columns) != [ID, *manifest["feature_names"]]:
        raise ValueError("December feature columns differ from the timeline bundle's.")
    labels = pd.read_parquet(phase1 / "labels" / source)
    labels = labels.loc[labels[ID].isin(table[ID])].reset_index(drop=True)
    if not np.array_equal(labels[ID].to_numpy(), table[ID].to_numpy()):
        raise ValueError("Label rows do not align with the feature rows.")
    record = {"month": month, "source": source, "context_months": months, "context_files": files, "rows": int(len(table)),
              "ids_sha256": identity_hash(table[ID]), "phase1_bundle": phase1.relative_to(root).as_posix(), "phase1_bundle_id": original["bundle_id"],
              "timeline_bundle_id": manifest["bundle_id"], "columns": len(table.columns) - 1, "patched_sources": patched.source_hashes(),
              "guard_replacement": patched.REPLACEMENT, "weather_inventories": [{"objects": len(inv["objects"]), "first": inv["objects"][0]["path"]} for inv in inventories],
              "label_rows": int(len(labels)), "audit_used": False, "created_at_utc": utc_now(), "environment": environment(root)}
    if verify:
        # The frozen bundle was prepared on Linux; the four cyclic NM-clock columns are float64 libm outputs that differ by one
        # ulp across platforms (recorded for the NM-clock campaign: at most 1.11e-16, float32 model inputs identical).
        frozen_path = root / read_json(root / "data/timeline/latest.json")["bundle"] / "features" / source
        frozen = pd.read_parquet(frozen_path)
        cyclic = [c for c in table.columns if c.startswith("nmclock_") and c.endswith(("_utc_hour_sin", "_utc_hour_cos"))]
        others = [c for c in table.columns if c not in cyclic]
        pd.testing.assert_frame_equal(table[others], frozen[others], check_exact=True)
        gap = max(float(np.nanmax(np.abs(table[c].to_numpy(dtype=float) - frozen[c].to_numpy(dtype=float)))) for c in cyclic)
        if gap > 2e-16 or any(not np.array_equal(table[c].to_numpy(dtype="float32"), frozen[c].to_numpy(dtype="float32"), equal_nan=True) for c in cyclic):
            raise ValueError("Cyclic NM-clock columns differ beyond platform rounding.")
        record["reproduces_frozen_timeline_rows_exactly"] = True
        record["reproduction"] = {"exact_columns": len(others), "cyclic_float64_columns_platform_rounding": cyclic, "cyclic_max_abs_difference": gap,
                                  "cyclic_float32_identical": True, "frozen_file_sha256": sha256_file(frozen_path)}
    out = output / record["ids_sha256"][:20]
    (out / "features").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    table.to_parquet(out / "features" / source, index=False)
    labels.to_parquet(out / "labels" / source, index=False)
    record["artifacts"] = {f"features/{source}": sha256_file(out / "features" / source), f"labels/{source}": sha256_file(out / "labels" / source)}
    record["build_id"] = object_hash({k: v for k, v in record.items() if k not in ("created_at_utc", "environment")})
    write_json(out / "manifest.json", record)
    return record
