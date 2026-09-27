"""S1-S2: the timeline bundle's 335 feature columns for a movement file's departures, with the frozen feature functions.

The frozen bundles computed the ranking rows once, with the context "every January-November 2025 movement plus the ranking file's
own months" (each layer reads it with its own input projection; December was never in it). This builder applies the same functions
(the five guarded modules loaded through prc2026_december.patched, unchanged on disk) to any file with the ranking schema, with the
same context convention: the frozen 2025 training files filtered to January-November plus the new file filtered to its own departure
months. On a copy of data/raw/ranking.parquet it must reproduce the frozen timeline bundle's features/ranking.parquet (`compare_frozen`).
Label-free: departure labels and block times are masked before anything reads the file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from prc2026.artifacts import read_json, sha256_file
from prc2026.features import ID, PHASE, TIME, movement_context
from prc2026_phase3.features import ALL_FEATURE_NAMES as PHASE3_NAMES, CONTEXT_COLUMNS, row_features, traffic_features
from prc2026_phase4.features import FEATURE_NAMES as PHASE4_NAMES, INPUT_COLUMNS as PHASE4_INPUTS
from prc2026_phase5r.features import FEATURE_NAMES as PHASE5R_NAMES, INPUT_COLUMNS as PHASE5R_INPUTS
from prc2026_weather.features import FEATURE_NAMES as WEATHER_NAMES, INPUT_COLUMNS as WEATHER_INPUTS, observation_features, runway_geometry, verify_external
from prc2026_nmclock.features import FEATURE_NAMES as NMCLOCK_NAMES, INPUT_COLUMNS as NMCLOCK_INPUTS
from prc2026_timeline.features import FEATURE_NAMES as TIMELINE_NAMES, INPUT_COLUMNS as TIMELINE_INPUTS
from prc2026_timeline.bundle import load_bundle as load_timeline
from prc2026_december import patched
from prc2026_december.features import aligned, observation_parts, weather_config

from .common import TRAINING_MONTHS, source_file

quality_features = patched.load("prc2026_phase4").quality_features
ground_features = patched.load("prc2026_phase5r").ground_features
weather_features = patched.load("prc2026_weather").weather_features
clock_features = patched.load("prc2026_nmclock").clock_features
timeline_features = patched.load("prc2026_timeline").timeline_features

TRAINING_FILES = [source_file(f"2025-{m:02}") for m in range(1, 13)]  # the December file holds a few November movements across its boundary


def in_months(frame: pd.DataFrame, months: list[str]) -> pd.DataFrame:
    return frame.loc[frame[TIME].dt.strftime("%Y-%m").isin(months)].reset_index(drop=True)


def training_context(root: Path, columns: list[str]) -> pd.DataFrame:
    parts = [pd.read_parquet(root / "data/raw" / name, columns=columns) for name in TRAINING_FILES]
    return in_months(pd.concat(parts, ignore_index=True), TRAINING_MONTHS)


def timeline_bundle(root: Path) -> tuple[dict, Path, dict, Path]:
    base = root / read_json(root / "data/timeline/latest.json")["bundle"]
    manifest, phase1, original, _ = load_timeline(root, base)
    return manifest, phase1, original, base


def weather_coverage(inventories: list[dict], departures: pd.DataFrame) -> dict:
    """Departures whose station-month has no weather object in the inventories (the feature layer would silently flag them missing)."""
    covered = {}
    for inv in inventories:
        for item in inv["objects"]:
            covered.setdefault(item["station"], []).append((pd.Timestamp(item["start_inclusive"], tz="UTC"), pd.Timestamp(item["end_exclusive"], tz="UTC")))
    uncovered = {}
    for airport, group in departures.groupby("ADEP_mvt"):
        spans = covered.get(airport, [])
        t = group[TIME]
        inside = np.zeros(len(group), dtype=bool)
        for begin, end in spans:
            inside |= ((t >= begin) & (t < end)).to_numpy()
        if (~inside).any():
            uncovered[airport] = sorted(t[~inside].dt.strftime("%Y-%m").unique().tolist())
    return uncovered


def build(root: Path, movements: pd.DataFrame, months: list[str], extra_weather_inventories: list[dict] | None = None,
          allow_missing_weather: bool = False) -> tuple[pd.DataFrame, dict]:
    """The 335-column timeline table for the file's departures (in the file's departure order) and a record."""
    if set(months) & set(TRAINING_MONTHS):
        raise ValueError("The file overlaps the frozen January-November 2025 context; a re-cut of fitting months needs its own context declaration.")
    manifest, phase1, original, _ = timeline_bundle(root)
    new = in_months(movements, months)
    # Phase 1 + Phase 3 row features and traffic context (Phase-1 context of the frozen training files + this file's)
    features3 = row_features(new)
    context_files = [f for f in TRAINING_FILES if (phase1 / "context" / f).exists()]
    frozen_context = in_months(pd.concat([pd.read_parquet(phase1 / "context" / f, columns=CONTEXT_COLUMNS) for f in context_files], ignore_index=True), TRAINING_MONTHS)
    context3 = pd.concat([movement_context(new)[CONTEXT_COLUMNS], frozen_context], ignore_index=True)  # the frozen order: the ranking file's context came first
    queries3 = new.loc[new[PHASE].eq("DEP"), [ID, TIME, "ADEP_mvt", "RUNWAY_mvt", "STAND_mvt"]].reset_index(drop=True)
    table = aligned(features3, traffic_features(context3, queries3), PHASE3_NAMES, "phase3")
    del frozen_context, context3
    # Phase 4 row quality
    table = aligned(table, quality_features(new[PHASE4_INPUTS]), PHASE4_NAMES, "phase4")
    # Phase 5R ground demand / service
    context5 = pd.concat([new[PHASE5R_INPUTS], training_context(root, PHASE5R_INPUTS)], ignore_index=True)
    table = aligned(table, ground_features(context5, table[ID]), PHASE5R_NAMES, "phase5r")
    del context5
    # Weather: the frozen thirty objects plus any declared extra inventories, one feature pass
    config = weather_config(root, manifest)
    external = verify_external(root, config)
    inventories = [external["weather"], *(extra_weather_inventories or [])]
    departures = new.loc[new[PHASE].eq("DEP")]
    uncovered = weather_coverage(inventories, departures)
    if uncovered and not allow_missing_weather:
        raise ValueError(f"No weather object covers these station-months: {uncovered}. Pull them first (final_predict.py external) "
                         "and pass --weather-inventory, or pass --allow-missing-weather to accept the frozen missing-weather route.")
    observations = observation_features(pd.concat([p for inv in inventories for p in observation_parts(root, inv)], ignore_index=True))
    runway_path = next(x["path"] for x in external["runways"]["objects"] if x["path"].endswith("runways.csv"))
    geometry = runway_geometry(pd.read_csv(root / runway_path))
    queries_w = table[[ID]].merge(new[WEATHER_INPUTS], on=ID, how="left", validate="one_to_one", sort=False)
    table = aligned(table, weather_features(queries_w, observations, geometry), WEATHER_NAMES, "weather")
    # NM clocks
    table = aligned(table, clock_features(table[[ID]].merge(new[NMCLOCK_INPUTS], on=ID, how="left", validate="one_to_one", sort=False)), NMCLOCK_NAMES, "nmclock")
    # Timeline: flown arrival and surface context
    contextt = pd.concat([new[TIMELINE_INPUTS], training_context(root, TIMELINE_INPUTS)], ignore_index=True)
    table = aligned(table, timeline_features(contextt, table[ID]), TIMELINE_NAMES, "timeline")
    if list(table.columns) != [ID, *manifest["feature_names"]]:
        raise ValueError("Feature columns differ from the timeline bundle's.")
    dep_ids = departures[ID].to_numpy()
    if not np.array_equal(table[ID].to_numpy(), dep_ids):
        table = table.set_index(ID).reindex(dep_ids).reset_index()
    record = {"months": months, "context": "this file filtered to its departure months, then the frozen training files filtered to January-November 2025 (the frozen source order)",
              "context_files": TRAINING_FILES, "rows": int(len(table)), "columns": len(table.columns) - 1, "timeline_bundle_id": manifest["bundle_id"],
              "phase1_bundle_id": original["bundle_id"], "patched_sources": patched.source_hashes(), "guard_replacement": patched.REPLACEMENT,
              "weather_inventories": [{"objects": len(inv["objects"]), "first": inv["objects"][0]["path"]} for inv in inventories],
              "weather_uncovered_station_months": uncovered}
    return table, record


def compare_frozen(root: Path, table: pd.DataFrame) -> dict:
    """Against the frozen timeline bundle's ranking rows: exact except the four cyclic NM-clock float64 columns (platform rounding <= 2e-16)."""
    _, _, _, base = timeline_bundle(root)
    frozen_path = base / "features" / "ranking.parquet"
    frozen = pd.read_parquet(frozen_path).set_index(ID).reindex(table[ID]).reset_index()
    if frozen.drop(columns=ID).isna().all(axis=1).any():
        raise ValueError("Some rows are not ranking rows of the frozen bundle.")
    cyclic = [c for c in table.columns if c.startswith("nmclock_") and c.endswith(("_utc_hour_sin", "_utc_hour_cos"))]
    others = [c for c in table.columns if c not in cyclic]
    differing = {}
    for c in others:
        a, b = table[c], frozen[c]
        same = (a.astype("object").where(a.notna(), None) == b.astype("object").where(b.notna(), None)) | (a.isna() & b.isna())
        if not bool(same.all()):
            differing[c] = int((~same).sum())
    gap = max(float(np.nanmax(np.abs(table[c].to_numpy(dtype=float) - frozen[c].to_numpy(dtype=float)))) for c in cyclic) if cyclic else 0.0
    return {"frozen_file": frozen_path.relative_to(root).as_posix(), "frozen_sha256": sha256_file(frozen_path), "exact_columns": len(others) - len(differing),
            "differing_columns": differing, "cyclic_columns": cyclic, "cyclic_max_abs_difference": gap,
            "passes": not differing and gap <= 2e-16}
