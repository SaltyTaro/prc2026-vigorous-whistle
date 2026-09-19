"""Twelve-month refit of a banked full-data ADS-B learner: the same recipe, the same rounds, January-December 2025 fitting rows.

Reuses tools/run_adsb_learner.py (frozen, imported) for features, encoding and fitting. `--source-run` names the learner
whose seasonal / forward fold records fix the recipe (kind, seed, parameter overrides, exclusions, target, extra features)
and the rounds (median best iteration of the two folds, the frozen refit rule; no fold saw December, so the rule is
unchanged). December's rows come from the opened December feature build (data/december/<id>/features + labels; the
timeline bundle's 335-column layout) and its ADS-B match table. Output: <output>/refit/{model, ranking_predictions.parquet,
run.json}. Nothing is uploaded.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import environment, object_hash, read_json, sha256_file, utc_now, write_json
from prc2026.features import ID, TARGET
from prc2026.validation import identity_hash
from prc2026_phase3.bundle import load_partition
from prc2026_phase5r.baseline import aligned_truth
from prc2026_timeline.bundle import load_bundle
from run_adsb_learner import ADSB_NAMES, BUNDLE_NAMES, RANKING_MONTHS, TRAINING_MONTHS, adsb_table, attach, encode, fit, source_file

DECEMBER = "2025-12"
DECEMBER_FILE = source_file(DECEMBER)


def code_receipt() -> dict:
    paths = [ROOT / "src/prc2026_adsb/features.py", ROOT / "tools/run_adsb_learner.py", ROOT / "tools/run_adsb_learner_dec.py"]
    hashes = {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths}
    return {"files": hashes, "sha256": object_hash(hashes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True, help="Directory with the learner's seasonal/ and forward/ run.json (the frozen recipe and rounds)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--december", type=Path, required=True, help="December feature build directory (features/ and labels/ with the December file)")
    parser.add_argument("--matches-dir", type=Path, default=Path("data/external/adsblol/matches"))
    parser.add_argument("--settings", type=Path, default=Path("configs/timeline_v1.json"))
    parser.add_argument("--extra-features", type=Path, default=Path("data/external/adsblol/features_v19"))
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    started = time.time()
    out = ROOT / args.output / "refit"
    if (out / "run.json").exists():
        raise ValueError(f"Preserve the existing run: {out}")
    folds = {f: read_json(ROOT / args.source_run / f / "run.json") for f in ["seasonal", "forward"]}
    src = folds["seasonal"]["inputs"]
    for f, rec in folds.items():
        for key in ["kind", "target", "parameters", "lightgbm_parameters", "excluded_airports", "max_label", "specialist_airports", "exclude_population", "extra_features", "adsb_features"]:
            if rec["inputs"].get(key) != src.get(key):
                raise ValueError(f"Source folds disagree on {key}.")
    kind, target = src.get("kind") or "lightgbm", src.get("target") or "label"
    params = dict(src.get("parameters") or src["lightgbm_parameters"])  # tools/run_adsb_full.py (the first learner) recorded lightgbm_parameters
    params.update({"num_threads": args.threads} if kind == "lightgbm" else {"thread_count": args.threads})
    excluded, exclude_population, max_label, specialists = src.get("excluded_airports") or [], bool(src.get("exclude_population")), src.get("max_label"), src.get("specialist_airports") or []
    extra_names = src.get("extra_features") or []
    rounds = int(np.rint(np.median([rec["best_iteration"] for rec in folds.values()])))
    matches_root = ROOT / args.matches_dir
    stands = pd.read_parquet(matches_root / "stands.parquet")
    bundle = ROOT / read_json(ROOT / "data/timeline/latest.json")["bundle"]
    manifest, parent, original, catalog = load_bundle(ROOT, bundle)
    december = read_json(ROOT / args.december / "manifest.json")
    if december["month"] != DECEMBER or december["timeline_bundle_id"] != manifest["bundle_id"]:
        raise ValueError("December build does not belong to this timeline bundle.")
    cache = ROOT / args.output / "features"
    months = [*TRAINING_MONTHS, DECEMBER]
    jobs = [([m], source_file(m), matches_root, stands, cache) for m in months] + [(RANKING_MONTHS, "ranking.parquet", matches_root, stands, cache)]
    with mp.Pool(min(len(jobs), max(1, args.threads // 2))) as pool:
        built = pool.starmap(adsb_table, jobs)
    tables = {job[1]: table for job, table in zip(jobs, built)}
    extra_hashes = {}
    if extra_names:
        for ms, raw_name in [([m], source_file(m)) for m in months] + [(RANKING_MONTHS, "ranking.parquet")]:
            extra = pd.concat([pd.read_parquet(ROOT / args.extra_features / f"features_v19_{m}.parquet") for m in ms], ignore_index=True)
            if [c for c in extra.columns if c != ID] != extra_names:
                raise ValueError(f"Extra feature columns differ in {raw_name}.")
            tables[raw_name] = tables[raw_name].merge(extra, on=ID, how="left", validate="one_to_one")
            extra_hashes.update({f"features_v19_{m}.parquet": sha256_file(ROOT / args.extra_features / f"features_v19_{m}.parquet") for m in ms})
    names = [*BUNDLE_NAMES, *ADSB_NAMES, *extra_names]
    # January-November from the frozen bundle (the frozen refit population), December from the opened build
    training = catalog[catalog["source_file"].isin([source_file(m) for m in TRAINING_MONTHS])]
    assignment = pd.DataFrame({ID: training[ID].to_numpy(), "partition": "fit"})
    features, truth = load_partition(bundle, parent, catalog, assignment, "fit", BUNDLE_NAMES)
    frame = attach(features, tables, catalog)
    y = aligned_truth(frame, truth)[TARGET].to_numpy(dtype=float)
    if len(frame) != 1919370:
        raise ValueError("The January-November refit population changed.")
    dec_features = pd.read_parquet(ROOT / args.december / "features" / DECEMBER_FILE, columns=[ID, *BUNDLE_NAMES])
    dec_labels = pd.read_parquet(ROOT / args.december / "labels" / DECEMBER_FILE).set_index(ID)[TARGET].reindex(dec_features[ID]).to_numpy(dtype=float)
    if identity_hash(dec_features[ID]) != december["ids_sha256"] or np.isnan(dec_labels).any():
        raise ValueError("December feature/label identity changed.")
    dec_frame = dec_features.merge(tables[DECEMBER_FILE], on=ID, how="left", validate="one_to_one")
    if dec_frame["adsb_matched"].isna().any():
        raise ValueError("December ADS-B table does not cover every December row.")
    frame = pd.concat([frame, dec_frame], ignore_index=True)
    y = np.concatenate([y, dec_labels])
    del features, dec_features, dec_frame
    if frame[ID].duplicated().any():
        raise ValueError("Duplicate identities across the twelve months.")

    def clock_mask(fr: pd.DataFrame) -> np.ndarray:
        mask = fr["nmclock_available"].to_numpy(dtype=float) == 1 if target == "nm_residual" else np.ones(len(fr), dtype=bool)
        if specialists:
            mask &= fr["airport"].isin(specialists).to_numpy()
        return mask

    def keep(fr: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
        mask = clock_mask(fr) & ~fr["airport"].isin(excluded).to_numpy()
        if exclude_population:
            mask &= ~(fr["airport"].eq("LIRF") & fr["nmclock_available"].eq(0) & fr["aircraft_missing"].eq(1)).to_numpy()
        if max_label is not None:
            mask &= labels <= max_label
        return mask

    def offset(fr: pd.DataFrame) -> np.ndarray:
        if target == "nm_residual":
            return fr["nmclock_proxy_s"].to_numpy(dtype=float)
        if target == "adsb_residual":
            motion = fr["adsb_motion_s"].to_numpy(dtype=float)
            return np.where(np.isfinite(motion), motion, 0.0)
        return np.zeros(len(fr))

    fit_keep = keep(frame, y)
    x, vocab = encode(frame[fit_keep].reset_index(drop=True), names)
    booster, _ = fit(kind, params, x, y[fit_keep] - offset(frame)[fit_keep], rounds)
    ranking = pd.read_parquet(bundle / "features" / "ranking.parquet", columns=[ID, *BUNDLE_NAMES])
    if identity_hash(ranking[ID]) != original["ranking_ids_sha256"]:
        raise ValueError("Ranking identities changed.")
    rank = ranking.merge(tables["ranking.parquet"], on=ID, how="left", validate="one_to_one")
    if rank["adsb_matched"].isna().any():
        raise ValueError("ADS-B table does not cover every ranking row.")
    xr, _ = encode(rank, names, vocab)
    clock = clock_mask(rank)
    prediction = np.where(clock, offset(rank) + booster.predict(xr), np.nan)
    out.mkdir(parents=True, exist_ok=True)
    booster.save(out / ("model.txt" if kind == "lightgbm" else "model.cbm"))
    pd.DataFrame({ID: rank[ID].to_numpy(), "adsb_full": prediction}).to_parquet(out / "ranking_predictions.parquet", index=False, compression="zstd")
    inputs = {"bundle_id": manifest["bundle_id"], "december_build_id": december["build_id"], "december_ids_sha256": december["ids_sha256"], "stands_sha256": sha256_file(matches_root / "stands.parquet"),
              "matches_sha256": {m: sha256_file(matches_root / f"matches2_{m}.parquet") for m in months + RANKING_MONTHS}, "code": code_receipt(), "kind": kind, "parameters": params,
              "features": len(names), "adsb_features": ADSB_NAMES, "extra_features": extra_names, "extra_feature_tables_sha256": extra_hashes, "excluded_airports": sorted(excluded), "target": target,
              "parameter_overrides": src.get("parameter_overrides"), "max_label": max_label, "seed": src.get("seed"), "specialist_airports": sorted(specialists), "exclude_population": exclude_population,
              "source_run": args.source_run.as_posix(), "source_fold_records_sha256": {f: sha256_file(ROOT / args.source_run / f / "run.json") for f in folds}}
    record = {"created_at_utc": utc_now(), "fold": "refit", "stage": "refit_twelve_months", "rounds": rounds, "rounds_rule": "median best iteration of the source learner's seasonal and forward folds",
              "development_best_iterations": {f: rec["best_iteration"] for f, rec in folds.items()}, "rows": int(len(frame)), "fitted_rows": int(fit_keep.sum()), "december_rows": int(len(dec_labels)),
              "months": months, "ranking_rows": int(len(rank)), "ranking_ids_sha256": identity_hash(rank[ID]), "ranking_matched_share": float(rank["adsb_matched"].mean()),
              "prediction_quantiles": {str(q): float(np.nanquantile(prediction, q)) for q in [0.01, 0.1, 0.5, 0.9, 0.99]}, "clock_share": float(clock.mean()), "inputs": inputs, "audit_used": False,
              "seconds": round(time.time() - started, 1), "environment": environment(ROOT)}
    write_json(out / "run.json", record)
    print({k: record[k] for k in ["fold", "rounds", "rows", "fitted_rows", "december_rows", "ranking_rows", "prediction_quantiles", "seconds"]})


if __name__ == "__main__":
    main()
