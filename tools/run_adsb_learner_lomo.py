"""Leave-one-month-out (LOMO) fit of a banked full-data ADS-B learner (configs/lomo_dev_v1.json).

The same recipe and rounds as the learner's twelve-month refit (tools/run_adsb_learner_dec.py, unchanged and not imported for its main):
the recipe (kind, seed, parameters, exclusions, target, extra features) from the source learner's seasonal / forward fold records and the
rounds as the median best iteration of the two folds. The fitting population is the twelve-month refit population -- the frozen
January-November refit rows (1,919,370) plus the opened December build (165,677) -- with every row whose movement month (MVT time, the
validation protocol's assignment clock) is the held month removed; every departure of the held month is predicted. The tool refuses if a
held-month row is in the fit, and checks that the predicted rows are exactly the held month's rows of the learner's seasonal predictions
(2025-01 / 2025-07) or the December build (2025-12).

  --hold-out 2025-01 | 2025-07   the development months made ranking-like (eleven fitting months, as December's January-November refit)
  --hold-out 2025-02 | 2025-06   Tier B of docs/final_file/febjun_check.md: the final file's February / June months' 2025 analogues; the
                                 held-row reference is the training catalog's rows of that month (no fold predicts them)
  --hold-out 2025-12             reproduction check: the fit is then exactly the January-November refit population, so the model and its
                                 December predictions must equal the leaf's refit/ model and runs/december/learners/<leaf>.parquet
  --smoke N                      code-path test: N sampled rows per month (the held month included) and at most 20 rounds; writes only outside runs/

Output: <output>/lomo_<month>/{model.txt | model.cbm, predictions.parquet (MVT_ID_mvt, adsb_full), run.json}. Nothing is uploaded.
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
from prc2026.artifacts import environment, object_hash, read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET, TIME  # noqa: E402
from prc2026.validation import identity_hash  # noqa: E402
from prc2026_phase3.bundle import load_partition  # noqa: E402
from prc2026_phase5r.baseline import aligned_truth  # noqa: E402
from prc2026_timeline.bundle import load_bundle  # noqa: E402
from run_adsb_learner import ADSB_NAMES, BUNDLE_NAMES, TRAINING_MONTHS, adsb_table, attach, encode, fit, source_file  # noqa: E402

DECEMBER = "2025-12"
DECEMBER_FILE = source_file(DECEMBER)
FEBJUN = ["2025-02", "2025-06"]  # Tier B of docs/final_file/febjun_check.md (10 October 2026): the final file's new months' 2025 analogues
HOLD_OUT = ["2025-01", "2025-07", DECEMBER, *FEBJUN]


def code_receipt() -> dict:
    paths = [ROOT / "src/prc2026_adsb/features.py", ROOT / "tools/run_adsb_learner.py", ROOT / "tools/run_adsb_learner_lomo.py"]
    hashes = {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths}
    return {"files": hashes, "sha256": object_hash(hashes)}


def recipe(source_run: Path) -> dict:
    """The frozen recipe and rounds of the twelve-month refit (tools/run_adsb_learner_dec.py's rule, repeated verbatim)."""
    folds = {f: read_json(ROOT / source_run / f / "run.json") for f in ["seasonal", "forward"]}
    src = folds["seasonal"]["inputs"]
    for f, rec in folds.items():
        for key in ["kind", "target", "parameters", "lightgbm_parameters", "excluded_airports", "max_label", "specialist_airports", "exclude_population", "extra_features", "adsb_features"]:
            if rec["inputs"].get(key) != src.get(key):
                raise ValueError(f"Source folds disagree on {key}.")
    return {"folds": folds, "src": src, "kind": src.get("kind") or "lightgbm", "target": src.get("target") or "label",
            "params": dict(src.get("parameters") or src["lightgbm_parameters"]), "excluded": src.get("excluded_airports") or [],
            "exclude_population": bool(src.get("exclude_population")), "max_label": src.get("max_label"), "specialists": src.get("specialist_airports") or [],
            "extra_names": src.get("extra_features") or [], "rounds": int(np.rint(np.median([rec["best_iteration"] for rec in folds.values()])))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-run", type=Path, required=True, help="the learner directory with seasonal/ and forward/ run.json")
    parser.add_argument("--hold-out", required=True, choices=HOLD_OUT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--december", type=Path, default=Path("data/december/d198dea10625b0a75977"))
    parser.add_argument("--matches-dir", type=Path, default=Path("data/external/adsblol/matches"))
    parser.add_argument("--extra-features", type=Path, default=Path("data/external/adsblol/features_v19"))
    parser.add_argument("--cache", type=Path, default=Path("runs/lomo_dev/_features"), help="ADS-B feature table cache (label-free, shared by jobs)")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--smoke", type=int, default=0, metavar="N")
    args = parser.parse_args()
    started = time.time()
    held_month = args.hold_out
    out = ROOT / args.output / f"lomo_{held_month}"
    if args.smoke and ("runs" in args.output.parts or not args.output.is_absolute()):
        raise ValueError("A smoke test writes only to an absolute path outside runs/.")
    if (out / "run.json").exists():
        raise ValueError(f"Preserve the existing run: {out}")
    spec = recipe(args.source_run)
    kind, target, rounds = spec["kind"], spec["target"], spec["rounds"]
    if args.smoke:
        rounds = min(rounds, 20)
    params = dict(spec["params"])
    params.update({"num_threads": args.threads} if kind == "lightgbm" else {"thread_count": args.threads})
    excluded, exclude_population, max_label, specialists, extra_names = (spec[k] for k in ["excluded", "exclude_population", "max_label", "specialists", "extra_names"])
    matches_root = ROOT / args.matches_dir
    stands = pd.read_parquet(matches_root / "stands.parquet")
    bundle = ROOT / read_json(ROOT / "data/timeline/latest.json")["bundle"]
    manifest, parent, original, catalog = load_bundle(ROOT, bundle)
    december = read_json(ROOT / args.december / "manifest.json")
    if december["month"] != DECEMBER or december["timeline_bundle_id"] != manifest["bundle_id"]:
        raise ValueError("December build does not belong to this timeline bundle.")
    months = [*TRAINING_MONTHS, DECEMBER]
    cache = ROOT / args.cache

    # the frozen January-November refit rows (as run_adsb_learner_dec.py), optionally sampled for a smoke test
    training = catalog[catalog["source_file"].isin([source_file(m) for m in TRAINING_MONTHS])]
    if args.smoke:
        training = training.groupby("source_file", group_keys=False).apply(lambda g: g.sample(n=min(args.smoke, len(g)), random_state=0))
    keep_ids = None if not args.smoke else set(training[ID])
    jobs = [([m], source_file(m), matches_root, stands, cache) for m in months]
    if args.smoke:  # sequential and filtered at once, to stay small
        tables = {}
        for job in jobs:
            t = adsb_table(*job)
            tables[job[1]] = t if job[1] == DECEMBER_FILE else t[t[ID].isin(keep_ids)].reset_index(drop=True)
    else:
        with mp.Pool(min(len(jobs), max(1, args.threads // 2))) as pool:
            built = pool.starmap(adsb_table, jobs)
        tables = {job[1]: table for job, table in zip(jobs, built)}
    extra_hashes = {}
    if extra_names:
        for m in months:
            raw_name = source_file(m)
            extra = pd.read_parquet(ROOT / args.extra_features / f"features_v19_{m}.parquet")
            if [c for c in extra.columns if c != ID] != extra_names:
                raise ValueError(f"Extra feature columns differ in {raw_name}.")
            tables[raw_name] = tables[raw_name].merge(extra, on=ID, how="left", validate="one_to_one")
            extra_hashes[f"features_v19_{m}.parquet"] = sha256_file(ROOT / args.extra_features / f"features_v19_{m}.parquet")
    names = [*BUNDLE_NAMES, *ADSB_NAMES, *extra_names]
    assignment = pd.DataFrame({ID: training[ID].to_numpy(), "partition": "fit"})
    features, truth = load_partition(bundle, parent, catalog, assignment, "fit", BUNDLE_NAMES)
    frame = attach(features, tables, catalog)
    y = aligned_truth(frame, truth)[TARGET].to_numpy(dtype=float)
    if not args.smoke and len(frame) != 1919370:
        raise ValueError("The January-November refit population changed.")
    month = catalog.set_index(ID)[TIME].reindex(frame[ID]).dt.strftime("%Y-%m").to_numpy()  # never a frame column: `month` is a frozen predictor
    dec_features = pd.read_parquet(ROOT / args.december / "features" / DECEMBER_FILE, columns=[ID, *BUNDLE_NAMES])
    dec_labels = pd.read_parquet(ROOT / args.december / "labels" / DECEMBER_FILE).set_index(ID)[TARGET].reindex(dec_features[ID]).to_numpy(dtype=float)
    if identity_hash(dec_features[ID]) != december["ids_sha256"] or np.isnan(dec_labels).any():
        raise ValueError("December feature/label identity changed.")
    if args.smoke:
        pick = np.random.default_rng(0).choice(len(dec_features), size=min(args.smoke, len(dec_features)), replace=False)
        dec_features, dec_labels = dec_features.iloc[np.sort(pick)].reset_index(drop=True), dec_labels[np.sort(pick)]
    dec_frame = dec_features.merge(tables[DECEMBER_FILE], on=ID, how="left", validate="one_to_one")
    if dec_frame["adsb_matched"].isna().any():
        raise ValueError("December ADS-B table does not cover every December row.")
    frame = pd.concat([frame, dec_frame], ignore_index=True)
    y = np.concatenate([y, dec_labels])
    month = np.concatenate([month, np.full(len(dec_frame), DECEMBER)])
    del features, dec_features, dec_frame
    if frame[ID].duplicated().any():
        raise ValueError("Duplicate identities across the twelve months.")

    held = month == held_month
    held_ids = frame.loc[held, ID].to_numpy()
    if not held.any():
        raise ValueError(f"No row of {held_month} in the population.")
    if held_month == DECEMBER:
        reference = pd.read_parquet(ROOT / args.december / "features" / DECEMBER_FILE, columns=[ID])[ID].to_numpy()
    elif held_month in FEBJUN:  # fitting months of both folds: no seasonal predictions; the reference is the training catalog's month
        cat_train = catalog[catalog["source_file"].isin([source_file(m) for m in TRAINING_MONTHS])]
        reference = cat_train.loc[cat_train[TIME].dt.strftime("%Y-%m").to_numpy() == held_month, ID].to_numpy()
    else:
        seasonal = pd.read_parquet(ROOT / args.source_run / "seasonal" / "predictions.parquet", columns=[ID])[ID]
        seasonal_month = catalog.set_index(ID)[TIME].reindex(seasonal).dt.strftime("%Y-%m").to_numpy()
        reference = seasonal.to_numpy()[seasonal_month == held_month]
    if args.smoke:
        if not set(held_ids) <= set(reference):
            raise ValueError("Held rows outside the reference rows.")
    elif set(held_ids) != set(reference):
        raise ValueError(f"The held rows differ from the reference rows of {held_month}: {len(held_ids)} vs {len(reference)}.")

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

    fit_keep = keep(frame, y) & ~held
    if np.intersect1d(frame.loc[fit_keep, ID].to_numpy(), held_ids).size or (month[fit_keep] == held_month).any():
        raise ValueError("A held-month row is in the fit.")
    fit_months = sorted(set(month[fit_keep]))
    x, vocab = encode(frame[fit_keep].reset_index(drop=True), names)
    booster, _ = fit(kind, params, x, y[fit_keep] - offset(frame)[fit_keep], rounds)
    del x
    held_frame = frame.loc[held].reset_index(drop=True)
    xh, _ = encode(held_frame, names, vocab)
    clock = clock_mask(held_frame)
    prediction = np.where(clock, offset(held_frame) + booster.predict(xh), np.nan)
    out.mkdir(parents=True, exist_ok=True)
    model_file = out / ("model.txt" if kind == "lightgbm" else "model.cbm")
    booster.save(model_file)
    pd.DataFrame({ID: held_frame[ID].to_numpy(), "adsb_full": prediction}).to_parquet(out / "predictions.parquet", index=False, compression="zstd")
    check = None
    if held_month == DECEMBER and not args.smoke:  # reproduction of the leaf's January-November refit
        leaf_model = ROOT / args.source_run / "refit" / model_file.name
        cached = ROOT / "runs/december/learners" / (args.source_run.as_posix().replace("/", "__") + ".parquet")
        check = {"refit_model_sha256": sha256_file(leaf_model) if leaf_model.exists() else None, "lomo_model_sha256": sha256_file(model_file)}
        check["model_identical"] = check["refit_model_sha256"] == check["lomo_model_sha256"]
        if cached.exists():
            ref = pd.read_parquet(cached).set_index(ID)["adsb_full"].reindex(held_frame[ID]).to_numpy(dtype=float)
            both = np.isfinite(ref) & np.isfinite(prediction)
            check.update({"december_cache": cached.relative_to(ROOT).as_posix(), "max_abs_difference_s": float(np.max(np.abs(ref[both] - prediction[both]))) if both.any() else None,
                          "finite_pattern_equal": bool((np.isfinite(ref) == np.isfinite(prediction)).all())})
    inputs = {"bundle_id": manifest["bundle_id"], "december_build_id": december["build_id"], "december_ids_sha256": december["ids_sha256"], "stands_sha256": sha256_file(matches_root / "stands.parquet"),
              "matches_sha256": {m: sha256_file(matches_root / f"matches2_{m}.parquet") for m in months}, "code": code_receipt(), "kind": kind, "parameters": params,
              "features": len(names), "adsb_features": ADSB_NAMES, "extra_features": extra_names, "extra_feature_tables_sha256": extra_hashes, "excluded_airports": sorted(excluded), "target": target,
              "parameter_overrides": spec["src"].get("parameter_overrides"), "max_label": max_label, "seed": spec["src"].get("seed"), "specialist_airports": sorted(specialists),
              "exclude_population": exclude_population, "source_run": args.source_run.as_posix(),
              "source_fold_records_sha256": {f: sha256_file(ROOT / args.source_run / f / "run.json") for f in spec["folds"]}}
    record = {"created_at_utc": utc_now(), "fold": f"lomo_{held_month}", "stage": "leave_one_month_out", "held_month": held_month, "fit_months": fit_months,
              "declaration": "configs/febjun_tierb_v1.json" if held_month in FEBJUN else "configs/lomo_dev_v1.json", "smoke": args.smoke,
              "rounds": rounds, "rounds_rule": "median best iteration of the source learner's seasonal and forward folds (the twelve-month refit's rule)",
              "development_best_iterations": {f: rec["best_iteration"] for f, rec in spec["folds"].items()}, "population_rows": int(len(frame)), "fitted_rows": int(fit_keep.sum()),
              "fit_ids_sha256": identity_hash(pd.Series(frame.loc[fit_keep, ID].to_numpy())), "held_rows": int(held.sum()), "held_ids_sha256": identity_hash(pd.Series(held_ids)),
              "held_rows_equal_reference": not args.smoke, "predicted_finite_rows": int(np.isfinite(prediction).sum()),
              "prediction_quantiles": {str(q): float(np.nanquantile(prediction, q)) for q in [0.01, 0.1, 0.5, 0.9, 0.99]}, "clock_share": float(clock.mean()),
              "reproduction_check": check, "inputs": inputs, "audit_used": False, "seconds": round(time.time() - started, 1), "environment": environment(ROOT)}
    write_json(out / "run.json", record)
    print({k: record[k] for k in ["fold", "rounds", "population_rows", "fitted_rows", "held_rows", "predicted_finite_rows", "prediction_quantiles", "reproduction_check", "seconds"]}, flush=True)


if __name__ == "__main__":
    main()
