"""Rebuild the fold tables that the ADS-B residual tools read from scratch/ (v9 out-of-sample predictions with labels).

For each development fold the frozen v7 is reconstructed from hash-checked runs (its
score must reproduce exactly), the declared LIRF turnaround mixture is fitted on the
fold's fitting months and applied to the fold's scoring months (exactly v9), and the
table [ID, TIME, airport, label, v9, floor, proxy_s, schedule_interval_s,
clock_available, population] is written to scratch/<fold>.parquet. The v9 scores must
reproduce the frozen `mixture_75` fold scores. December is never read. Nothing is uploaded.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import object_hash, read_json, utc_now, write_json
from prc2026.features import ID, TARGET, TIME
from prc2026.validation import evaluate
from prc2026_lirf_mixture.model import apply_candidate
from prc2026_postprocess.rules import row_context
from run_lirf_mixture import fit_population_model, movements, score_population
from run_postprocess import PROTOCOL, check_parents, reconstruct_fold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, default=Path("scratch"))
    args = parser.parse_args()
    config = read_json(ROOT / "configs/lirf_mixture_v1.json")
    selection = read_json(ROOT / "runs/lirf_mixture/selection.json")
    if selection["config_sha256"] != object_hash(config) or selection["selected_candidate"] != "mixture_75":
        raise ValueError("The v9 declaration does not match its selection receipt.")
    check_parents(config)
    protocol = read_json(PROTOCOL)["folds"]
    out = ROOT / args.scratch
    out.mkdir(parents=True, exist_ok=True)
    record = {"created_at_utc": utc_now(), "folds": {}}
    for fold in config["development_folds"]:
        scoring, truth, v7, _, _, _ = reconstruct_fold(fold, config)
        ids = scoring[ID].to_numpy()
        fitting = movements(protocol[fold]["fit_months"])
        model, component, summary = fit_population_model(fitting, config)
        scoring_movements = movements(protocol[fold]["score_months"])
        _, _, _, interval, mixture, index = score_population(model, component, scoring_movements, ids, config)
        spec = config["candidates"]["mixture_75"]
        v9, _ = apply_candidate(v7, index, interval, mixture, spec["mixture_weight"], config["mixture"]["schedule_interval_threshold_seconds"], summary["floor_seconds"])
        rmse = evaluate(truth, pd.DataFrame({ID: ids, "prediction": v9}))["overall"]["rmse_seconds"]
        expected = selection["candidates"]["mixture_75"]["folds"][fold]["metrics"]["overall"]["rmse_seconds"]
        if abs(rmse - expected) > 1e-9:
            raise ValueError(f"Reconstructed v9 does not reproduce the frozen {fold} score ({rmse} vs {expected}).")
        context = row_context(scoring).set_index(ID).reindex(ids)
        population = np.zeros(len(ids), dtype=bool)
        population[index] = True
        table = pd.DataFrame({ID: ids, TIME: scoring[TIME].to_numpy(), "airport": scoring["ADEP_mvt"].to_numpy(), TARGET: scoring[TARGET].to_numpy(dtype=float),
                              "v9": v9, "floor": summary["floor_seconds"], "proxy_s": context["proxy_s"].to_numpy(), "schedule_interval_s": context["schedule_interval_s"].to_numpy(),
                              "clock_available": context["clock_available"].to_numpy(), "population": population})
        table.to_parquet(out / f"{fold}.parquet", index=False)
        record["folds"][fold] = {"rows": int(len(table)), "v9_rmse": rmse, "floor_seconds": summary["floor_seconds"], "population_rows": int(population.sum())}
        print(fold, record["folds"][fold])
    write_json(out / "build.json", record)


if __name__ == "__main__":
    main()
