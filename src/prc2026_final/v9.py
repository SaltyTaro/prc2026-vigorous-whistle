"""S6: v7 and v9 for a movement file, as the v23 export built them for the ranking (tools/export_adsb_v25.py step 1).

- v7 frozen: the surface-refit pipeline (prc2026_surface_refit.inference.pipeline_prediction; v6/v5 fallback inside) on the timeline
  features; on the ranking this is the recorded v7 of runs/azure_arcw0_surface_refit/.../ranking_predictions.parquet.
- v9 frozen: v7 frozen with the January-November LIRF mixture (refitted from 2025 labels in seconds, byte-identical on Windows) on the
  fill population, weight 0.75, schedule-interval threshold 43,200 s, floor -12 (tools/run_lirf_mixture.py, imported unchanged).
- v7 twelve-month: the twelve saved nodes of runs/adsb_v24/v7/refit (tools/run_v7_dec.py::predict) on clock-present rows.
- v9 = max(v7 twelve-month, floor) on clock-present rows, v9 frozen elsewhere.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from prc2026.artifacts import object_hash, read_json, sha256_file
from prc2026.features import ID, PHASE, TIME
from prc2026_lirf_mixture.model import apply_candidate
from prc2026_nmclock.features import route_mask
from prc2026_postprocess.rules import row_context
from prc2026_surface_refit.inference import pipeline_prediction
from prc2026_timeline.features import FEATURE_NAMES as TIMELINE_NAMES

from .common import ROOT, TRAINING_MONTHS

sys.path.insert(0, str(ROOT / "tools"))
from run_lirf_mixture import MOVEMENT_COLUMNS, fit_population_model, movements, score_population  # noqa: E402
from run_v7_dec import predict as v7_nodes_predict  # noqa: E402

SURFACE_REFIT = Path("runs/azure_arcw0_surface_refit/20260914T185207842456Z_refit_full_timeline_v6_100")
V7_TWELVE = Path("runs/adsb_v24/v7/refit")


def build(features: pd.DataFrame, movements_frame: pd.DataFrame, threads: int) -> tuple[pd.DataFrame, dict]:
    """features: the timeline table in departure order; movements_frame: the file (masked) with every phase."""
    ids = features[ID].to_numpy()
    run = ROOT / SURFACE_REFIT
    frame = features[[ID, *TIMELINE_NAMES]]
    v7 = np.empty(len(ids))
    months = movements_frame.set_index(ID).loc[ids, TIME].dt.strftime("%Y-%m").to_numpy()
    for month in sorted(set(months)):  # month chunks bound the memory; every row's prediction is independent of the others
        rows = months == month
        v7[rows] = pipeline_prediction(run, read_json(run / "pipeline.json"), frame.loc[rows].reset_index(drop=True), threads)
    config = read_json(ROOT / "configs/lirf_mixture_v1.json")
    selection = read_json(ROOT / "runs/lirf_mixture/selection.json")
    if selection["config_sha256"] != object_hash(config) or selection["selected_candidate"] != "mixture_75":
        raise ValueError("The v9 declaration does not match its selection receipt.")
    model, component, summary = fit_population_model(movements(TRAINING_MONTHS), config)
    floor = summary["floor_seconds"]
    scoring = movements_frame[MOVEMENT_COLUMNS]
    _, _, _, interval, mixture, index = score_population(model, component, scoring, ids, config)
    spec = config["candidates"]["mixture_75"]
    v9_frozen, applied = apply_candidate(v7, index, interval, mixture, spec["mixture_weight"], config["mixture"]["schedule_interval_threshold_seconds"], floor)
    population = np.zeros(len(ids), dtype=bool)
    population[index] = True
    nodes_run = ROOT / V7_TWELVE
    nodes = read_json(nodes_run / "nodes.json")
    clock = route_mask(frame)
    v7_twelve = np.full(len(ids), np.nan)
    for month in sorted(set(months)):
        rows = (months == month) & clock
        if rows.any():
            v7_twelve[rows] = v7_nodes_predict(nodes_run, nodes, frame.loc[rows].reset_index(drop=True), threads)
    if (clock & population).any():
        raise ValueError("A fill-population row has an NM clock; the v9 routes overlap.")
    v9 = np.where(clock, np.maximum(v7_twelve, floor), v9_frozen)
    departures = movements_frame.loc[movements_frame[PHASE].eq("DEP")].set_index(ID).loc[ids].reset_index()
    context = row_context(departures).set_index(ID).reindex(ids)
    out = pd.DataFrame({ID: ids, "v7": v7, "v9_frozen": v9_frozen, "v7_twelve": v7_twelve, "v9": v9, "population": population, "clock": clock,
                        "floor": floor, "proxy_s": context["proxy_s"].to_numpy(), "schedule_interval_s": context["schedule_interval_s"].to_numpy(),
                        "clock_available": context["clock_available"].to_numpy(), "hedge_population": context["hedge_population"].to_numpy()})
    if not np.array_equal(out["population"].to_numpy(), out["hedge_population"].to_numpy()):
        raise ValueError("The mixture's population differs from the routing rule's fill population.")
    record = {"surface_refit_run": SURFACE_REFIT.as_posix(), "pipeline_sha256": sha256_file(run / "pipeline.json"), "mixture": applied,
              "mixture_classifier": {k: summary[k] for k in ["fitting_population_rows", "fitting_population_ids_sha256", "rounds", "mode_counts"]},
              "floor_seconds": floor, "v7_twelve_nodes_sha256": sha256_file(nodes_run / "nodes.json"), "clock_rows": int(clock.sum()),
              "population_rows": int(population.sum()), "v9_finite": bool(np.isfinite(v9).all())}
    if not record["v9_finite"]:
        raise ValueError("v9 is not finite on every departure.")
    return out, record
