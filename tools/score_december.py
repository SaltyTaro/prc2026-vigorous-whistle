"""Score a declared ADS-B recipe family on December 2025 with the shipped January-November models (docs/december/).

Usage: score_december.py --config configs/adsb_v22_v1.json --features-dir data/december/d198dea10625b0a75977

Builds scratch/december.parquet (v7 and v9 on December) once, predicts every leaf learner the declaration's tables need
(cached under runs/december/learners/), rebuilds the ensembles from their recorded member lists, applies the v19 export's
residual booster, and scores every candidate of the declaration against its development baseline recipe, on December.
December is out-of-sample for every model involved. Nothing is uploaded and no selection rule is applied here.
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
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, write_json
from prc2026.features import ID, TARGET
from prc2026.validation import identity_hash
from prc2026_december.scoring import (BUNDLE_NAMES, VocabularyCache, adsb_table, build_scratch, december_departures, ensemble_prediction, evaluate_candidates,
                                      leaf_members, learner_prediction, load_residual, residual_table)
from run_adsb_v11 import candidate_values, predict_residual, rmse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--residual-export", type=Path, default=Path("runs/adsb_v21_v2/export_v19"), help="Saved residual booster of the incumbent export")
    parser.add_argument("--scratch", type=Path, default=Path("scratch/december.parquet"))
    parser.add_argument("--learners", type=Path, default=Path("runs/december/learners"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    started = time.time()
    config = read_json(ROOT / args.config)
    features = pd.read_parquet(ROOT / args.features_dir / "features/training_2025-12-01_2026-01-01.parquet")
    labels = pd.read_parquet(ROOT / args.features_dir / "labels/training_2025-12-01_2026-01-01.parquet")
    manifest = read_json(ROOT / args.features_dir / "manifest.json")
    if manifest["ids_sha256"] != identity_hash(features[ID]) or manifest["month"] != "2025-12":
        raise ValueError("December feature bundle identity mismatch.")
    departures = december_departures(ROOT).set_index(ID).reindex(features[ID]).reset_index()
    # v7 and v9 on December (once)
    scratch_path = ROOT / args.scratch
    if scratch_path.exists():
        scratch = pd.read_parquet(scratch_path)
        stack = read_json(scratch_path.with_suffix(".json"))
    else:
        scratch, stack = build_scratch(ROOT, features, labels, args.threads)
        scratch_path.parent.mkdir(parents=True, exist_ok=True)
        scratch.to_parquet(scratch_path, index=False)
        write_json(scratch_path.with_suffix(".json"), {**stack, "created_at_utc": utc_now(), "features_dir": args.features_dir.as_posix(), "features_manifest_sha256": sha256_file(ROOT / args.features_dir / "manifest.json")})
    if not np.array_equal(scratch[ID].to_numpy(), features[ID].to_numpy()):
        raise ValueError("Scratch rows do not align with the December features.")
    print({"v7": round(stack["v7_rmse"], 4), "v9": round(stack["v9_rmse"], 4), "rows": len(scratch)}, flush=True)
    # Learner frame: 276 bundle features + 27 ADS-B features (+ the eleven v19 features where a learner asks for them)
    frame = features[[ID, *BUNDLE_NAMES]].merge(adsb_table(ROOT, departures, Path("data/external/adsblol/matches")), on=ID, how="left", validate="one_to_one")
    v19 = pd.read_parquet(ROOT / "data/external/adsblol/features_v19/features_v19_2025-12.parquet")
    frame = frame.merge(v19, on=ID, how="left", validate="one_to_one")
    if frame["adsb_matched"].isna().any():
        raise ValueError("ADS-B table does not cover every December row.")
    # Leaf learners behind every declared table
    tables = config["extra_feature_tables"]
    run_dirs = {name: Path(spec["development"][0]).parent.parent.as_posix() for name, spec in tables.items()}
    members = {name: leaf_members(ROOT, run_dir) for name, run_dir in run_dirs.items()}
    leaves = sorted({leaf for group in members.values() for leaf in group})
    (ROOT / args.learners).mkdir(parents=True, exist_ok=True)
    vocab, predictions, records = VocabularyCache(ROOT), {}, {}
    for leaf in leaves:
        cache = ROOT / args.learners / (leaf.replace("/", "__") + ".parquet")
        if cache.exists() and cache.with_suffix(".json").exists():
            predictions[leaf] = pd.read_parquet(cache).set_index(ID)["adsb_full"].reindex(features[ID]).to_numpy(dtype=float)
            records[leaf] = read_json(cache.with_suffix(".json"))
            continue
        t0 = time.time()
        predictions[leaf], records[leaf] = learner_prediction(ROOT, Path(leaf), frame, vocab, args.threads)
        pd.DataFrame({ID: features[ID].to_numpy(), "adsb_full": predictions[leaf]}).to_parquet(cache, index=False, compression="zstd")
        write_json(cache.with_suffix(".json"), {**records[leaf], "seconds": round(time.time() - t0, 1)})
        print(leaf, {k: records[leaf][k] for k in ["kind", "target", "rounds", "finite_rows"]}, round(time.time() - t0, 1), "s", flush=True)
    extra = [pd.DataFrame({ID: features[ID].to_numpy(), name: ensemble_prediction(members[name], predictions)}) for name in tables]
    # Residual stage (the incumbent export's saved booster) on December's declared matches directory
    table = residual_table(ROOT, config, scratch, departures, extra)
    model = load_residual(ROOT, args.residual_export)
    if model["feature_set"] != "nb" and model["feature_set"] is not None:
        raise ValueError("Unexpected residual feature set.")
    residual = predict_residual(model, table, config, model["scope"])
    label, v9, floor = table[TARGET].to_numpy(dtype=float), table["v9"].to_numpy(dtype=float), float(table["floor"].iloc[0])
    base_spec = config["development_baseline"]
    baseline, _ = candidate_values(table, v9, residual, {"weight": base_spec["weight"], "scope": base_spec["scope"], "blend_with": base_spec["blend_with"], "extra_blends": base_spec.get("extra_blends", [])}, floor)
    references = {"v7": rmse(label, table["v7"].to_numpy(dtype=float)), "v9": rmse(label, v9), "baseline": rmse(label, baseline),
                  "v9_plus_residual_only": rmse(label, np.maximum(np.where(SCOPES_mask(table, base_spec["scope"]), v9 + residual, v9), floor))}
    if base_spec.get("extra_blends"):
        without, _ = candidate_values(table, v9, residual, {"weight": base_spec["weight"], "scope": base_spec["scope"], "blend_with": base_spec["blend_with"], "extra_blends": []}, floor)
        references["baseline_without_extra_blends"] = rmse(label, without)
    candidates = evaluate_candidates(config, table, residual, baseline)
    matched = table["adsb_matched"].eq(1).to_numpy()
    report = {"created_at_utc": utc_now(), "month": "2025-12", "rows": int(len(table)), "ids_sha256": identity_hash(table[ID]), "config": args.config.as_posix(), "config_sha256": object_hash(config),
              "features_dir": args.features_dir.as_posix(), "features_manifest_sha256": sha256_file(ROOT / args.features_dir / "manifest.json"), "stack": stack,
              "matched_share": float(matched.mean()), "gate_share_of_matched": float(table.loc[matched, "adsb_gate"].mean()),
              "baseline": {"name": base_spec["name"], "recipe": base_spec["recipe"], "rmse": references["baseline"]}, "references": references,
              "residual": {"export": args.residual_export.as_posix(), "models_sha256": model["models_sha256"], "rounds": model["rounds"], "scope": model["scope"], "rows_in_scope": int(SCOPES_mask(table, model["scope"]).sum())},
              "tables": {name: {"run_dir": run_dirs[name], "leaf_members": members[name]} for name in tables}, "learners": records,
              "candidates": candidates, "audit_used": False, "note": "December is out-of-sample for every model here; this is an honest single-month read, not a selection under the declared rule.",
              "seconds": round(time.time() - started, 1)}
    (ROOT / args.report).parent.mkdir(parents=True, exist_ok=True)
    write_json(ROOT / args.report, report)
    ranked = sorted(candidates.items(), key=lambda kv: kv[1]["rmse"])
    print(json.dumps({"references": {k: round(v, 4) for k, v in references.items()}}))
    for name, entry in ranked:
        print(f"{name:28s} {entry['rmse']:.4f}  vs baseline {100 * (entry['ratio_to_baseline'] - 1):+.3f}%  rows differing {entry['rows_differing_from_baseline']}")


def SCOPES_mask(table: pd.DataFrame, scope: str) -> np.ndarray:
    from run_adsb_v11 import scope_mask
    return scope_mask(table, scope)


if __name__ == "__main__":
    main()
