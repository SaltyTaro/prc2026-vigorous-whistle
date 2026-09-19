"""How much do adjacent fitting months buy on December? The forward-fold learners (January-September, October stopping) versus the
refit learners (January-November) applied to December, inside the incumbent recipe; the analog of adding December for January 2026.

Usage: december_month_value.py --config configs/adsb_v22_v1.json --features-dir data/december/d198dea10625b0a75977 --report docs/december/month_value.json
Uses the cached refit predictions of tools/score_december.py and predicts the same leaf learners from their forward-fold models
(runs/december/learners_forward/). The residual stage, v9 and the blend recipe are held fixed; December is out-of-sample for every model.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import read_json, utc_now, write_json
from prc2026.features import ID, TARGET
from prc2026_december.scoring import BUNDLE_NAMES, VocabularyCache, adsb_table, december_departures, ensemble_prediction, leaf_members, learner_prediction, load_residual, residual_table
from run_adsb_v11 import candidate_values, predict_residual, rmse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--residual-export", type=Path, default=Path("runs/adsb_v21_v2/export_v19"))
    parser.add_argument("--candidates", nargs="*", default=["lirf_ens3_70", "lirf_q_85"])
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=12)
    args = parser.parse_args()
    config = read_json(ROOT / args.config)
    features = pd.read_parquet(ROOT / args.features_dir / "features/training_2025-12-01_2026-01-01.parquet")
    scratch = pd.read_parquet(ROOT / "scratch/december.parquet")
    departures = december_departures(ROOT).set_index(ID).reindex(features[ID]).reset_index()
    frame = features[[ID, *BUNDLE_NAMES]].merge(adsb_table(ROOT, departures, Path("data/external/adsblol/matches")), on=ID, how="left", validate="one_to_one")
    frame = frame.merge(pd.read_parquet(ROOT / "data/external/adsblol/features_v19/features_v19_2025-12.parquet"), on=ID, how="left", validate="one_to_one")
    tables = config["extra_feature_tables"]
    members = {name: leaf_members(ROOT, Path(spec["development"][0]).parent.parent.as_posix()) for name, spec in tables.items()}
    leaves = sorted({leaf for group in members.values() for leaf in group})
    vocab = VocabularyCache(ROOT)
    predictions, records = {"refit": {}, "forward": {}}, {}
    for leaf in leaves:
        cache = ROOT / "runs/december/learners" / (leaf.replace("/", "__") + ".parquet")
        predictions["refit"][leaf] = pd.read_parquet(cache).set_index(ID)["adsb_full"].reindex(features[ID]).to_numpy(dtype=float)
        fcache = ROOT / "runs/december/learners_forward" / (leaf.replace("/", "__") + ".parquet")
        fcache.parent.mkdir(parents=True, exist_ok=True)
        if fcache.exists():
            predictions["forward"][leaf] = pd.read_parquet(fcache).set_index(ID)["adsb_full"].reindex(features[ID]).to_numpy(dtype=float)
            records[leaf] = read_json(fcache.with_suffix(".json"))
            continue
        t0 = time.time()
        predictions["forward"][leaf], records[leaf] = learner_prediction(ROOT, Path(leaf), frame, vocab, args.threads, fold="forward")
        pd.DataFrame({ID: features[ID].to_numpy(), "adsb_full": predictions["forward"][leaf]}).to_parquet(fcache, index=False, compression="zstd")
        write_json(fcache.with_suffix(".json"), records[leaf])
        print(leaf, records[leaf]["fitted_rows"], records[leaf]["rounds"], round(time.time() - t0, 1), "s", flush=True)
    model = load_residual(ROOT, args.residual_export)
    label, v9, floor = scratch[TARGET].to_numpy(dtype=float), scratch["v9"].to_numpy(dtype=float), float(scratch["floor"].iloc[0])
    report = {"created_at_utc": utc_now(), "config": args.config.as_posix(), "learners": {}, "recipes": {}, "note": "forward-fold models: fitted January-September with October stopping (1.45M-row fitting set); refit models: January-November (1.77M / 1.92M rows). December is out-of-sample for both."}
    for leaf in leaves:
        k = np.isfinite(predictions["refit"][leaf]) & np.isfinite(predictions["forward"][leaf])
        report["learners"][leaf] = {"refit_rmse": rmse(label[k], predictions["refit"][leaf][k]), "forward_rmse": rmse(label[k], predictions["forward"][leaf][k]), "rows": int(k.sum()), "forward_fitted_rows": records[leaf]["fitted_rows"]}
    for stage in ["refit", "forward"]:
        extra = [pd.DataFrame({ID: features[ID].to_numpy(), name: ensemble_prediction(members[name], predictions[stage])}) for name in tables]
        table = residual_table(ROOT, config, scratch, departures, extra)
        residual = predict_residual(model, table, config, model["scope"])
        for name in args.candidates:
            item = config["candidates"][name]
            spec = {"weight": item["weight"], "scope": item.get("scope", "matched"), "blend_with": item.get("blend_with"), "extra_blends": item.get("extra_blends", [])}
            values, _ = candidate_values(table, v9, residual, spec, floor)
            report["recipes"].setdefault(name, {})[stage] = rmse(label, values)
            by_airport = {a: rmse(label[table["airport"].eq(a).to_numpy()], values[table["airport"].eq(a).to_numpy()]) for a in sorted(table["airport"].unique())}
            report["recipes"][name][stage + "_by_airport"] = by_airport
    for name, entry in report["recipes"].items():
        entry["refit_vs_forward_pct"] = 100 * (entry["refit"] / entry["forward"] - 1)
    write_json(ROOT / args.report, report)
    print(json.dumps({n: {k: (round(v, 4) if isinstance(v, float) else None) for k, v in e.items() if not k.endswith("by_airport")} for n, e in report["recipes"].items()}, indent=1))
    worst = sorted(report["learners"].items(), key=lambda kv: kv[1]["refit_rmse"] / kv[1]["forward_rmse"])
    for leaf, e in worst:
        print(f"{leaf:36s} forward {e['forward_rmse']:.3f}  refit {e['refit_rmse']:.3f}  {100 * (e['refit_rmse'] / e['forward_rmse'] - 1):+.3f}%  rows {e['rows']}")


if __name__ == "__main__":
    main()
