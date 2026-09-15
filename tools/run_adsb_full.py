"""Full-data ADS-B learner: LightGBM on the timeline bundle model features (the combined contract) plus the ADS-B surface and neighbour features.

`--fold seasonal|forward`: fit on the fold's fit partition (early stopping on its
early-stop partition), predict the fold's score partition, and save the
out-of-sample predictions (`predictions.parquet`: ID, adsb_full) with a run
record. `--fold refit`: fit on all eleven 2025 months at the median best
iteration of the two development folds and predict the ranking departures
(`ranking_predictions.parquet`). The ADS-B feature tables come from the pass-2
matches and stands of the given matches directory (departure and neighbour
features; label-free) and are cached under the output directory. Target: the
departure label. December is never read; nothing is uploaded.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import environment, object_hash, read_json, sha256_file, utc_now, write_json
from prc2026.features import ID, PHASE, TARGET, TIME
from prc2026.validation import identity_hash
from prc2026_adsb.features import DEPARTURE_COLUMNS, FEATURE_NAMES, NEIGHBOUR_NAMES, departure_features, neighbour_features
from prc2026_phase3.bundle import load_partition
from prc2026_phase5r.baseline import aligned_truth
from prc2026_timeline.bundle import load_bundle
from prc2026_timeline.features import CATEGORICAL_COLUMNS, CORE_NAMES, FLOWN_NAMES, SURFACE_NAMES

MISSING = "__MISSING__"
BUNDLE_NAMES = [*CORE_NAMES, *FLOWN_NAMES, *SURFACE_NAMES]  # the timeline campaign's 'combined' model contract (276 features)
ADSB_NAMES = [*FEATURE_NAMES, *NEIGHBOUR_NAMES]
TRAINING_MONTHS = [f"2025-{m:02}" for m in range(1, 12)]
RANKING_MONTHS = ["2026-01", "2026-07"]


def code_receipt() -> dict:
    paths = [ROOT / "src/prc2026_adsb/features.py", ROOT / "tools/run_adsb_full.py"]
    hashes = {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths}
    return {"files": hashes, "sha256": object_hash(hashes)}


def source_file(month: str) -> str:
    y, m = int(month[:4]), int(month[5:7])
    return f"training_{y}-{m:02}-01_{y + (m == 12)}-{(m % 12) + 1:02}-01.parquet"


def adsb_table(months: list[str], raw_name: str, matches_root: Path, stands: pd.DataFrame, cache: Path) -> pd.DataFrame:
    """Label-free ADS-B departure and neighbour features for every departure of one raw file (cached)."""
    target = cache / f"adsb_{raw_name.replace('.parquet', '')}.parquet"
    if target.exists():
        return pd.read_parquet(target)
    raw = pd.read_parquet(ROOT / "data/raw" / raw_name, columns=DEPARTURE_COLUMNS + ["RUNWAY_mvt"])
    dep = raw[raw[PHASE].eq("DEP")].reset_index(drop=True)
    matches = pd.concat([pd.read_parquet(matches_root / f"matches2_{month}.parquet") for month in months], ignore_index=True)
    feats = departure_features(dep, matches, stands).merge(neighbour_features(dep, matches), on=ID, validate="one_to_one")
    if list(feats.columns) != [ID, *ADSB_NAMES]:
        raise ValueError("Unexpected ADS-B feature columns.")
    cache.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(target, index=False, compression="zstd")
    return feats


def attach(features: pd.DataFrame, tables: dict[str, pd.DataFrame], catalog: pd.DataFrame) -> pd.DataFrame:
    """Join the ADS-B table of each row's source file onto the bundle features (one-to-one on ID)."""
    sources = catalog.set_index(ID)["source_file"].reindex(features[ID]).to_numpy()
    parts = []
    for name in sorted(set(sources)):
        part = features.loc[sources == name].merge(tables[name], on=ID, how="left", validate="one_to_one")
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    out = out.set_index(ID).reindex(features[ID]).reset_index()
    if out["adsb_matched"].isna().any():
        raise ValueError("ADS-B table does not cover every bundle row.")
    return out


def encode(frame: pd.DataFrame, names: list[str], vocab: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Fixed categorical vocabularies from the fitting rows; float32 elsewhere."""
    vocab = vocab or {}
    out = {}
    for name in names:
        if name in CATEGORICAL_COLUMNS:
            values = frame[name].astype("string").fillna(MISSING)
            if name not in vocab:
                vocab[name] = sorted(set(values) | {MISSING})
            values = values.where(values.isin(vocab[name]), MISSING)
            out[name] = pd.Categorical(values, categories=vocab[name])
        else:
            out[name] = frame[name].to_numpy(dtype="float32", na_value=np.nan)
    return pd.DataFrame(out), vocab


def fit(params: dict, x: pd.DataFrame, y: np.ndarray, rounds: int, xs: pd.DataFrame | None = None, ys: np.ndarray | None = None, patience: int = 200) -> tuple[lgb.Booster, dict]:
    cats = [c for c in x.columns if c in CATEGORICAL_COLUMNS]
    train = lgb.Dataset(x, label=y, categorical_feature=cats, free_raw_data=False)
    history, kwargs = {}, {}
    callbacks = [lgb.log_evaluation(period=100), lgb.record_evaluation(history)]
    if xs is not None:
        kwargs = {"valid_sets": [lgb.Dataset(xs, label=ys, reference=train, categorical_feature=cats)], "valid_names": ["early_stop"]}
        callbacks.append(lgb.early_stopping(patience, first_metric_only=True, verbose=False))
    booster = lgb.train(params, train, num_boost_round=rounds, callbacks=callbacks, **kwargs)
    return booster, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", required=True, choices=["seasonal", "forward", "refit"])
    parser.add_argument("--matches-dir", type=Path, default=Path("data/external/adsblol/matches"))
    parser.add_argument("--output", type=Path, default=Path("runs/adsb_full"))
    parser.add_argument("--settings", type=Path, default=Path("configs/timeline_v1.json"), help="LightGBM parameters are taken from this declaration's lightgbm block")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    started = time.time()
    out = ROOT / args.output / args.fold
    if (out / "run.json").exists():
        raise ValueError(f"Preserve the existing run: {out}")
    settings = read_json(ROOT / args.settings)["lightgbm"]
    params = dict(settings["parameters"])
    params.update(num_threads=args.threads, seed=args.seed, feature_fraction_seed=args.seed, bagging_seed=args.seed, data_random_seed=args.seed, metric="rmse")
    matches_root = ROOT / args.matches_dir
    stands = pd.read_parquet(matches_root / "stands.parquet")
    base = ROOT / read_json(ROOT / "data/timeline/latest.json")["bundle"]
    manifest, parent, original, catalog = load_bundle(ROOT, base)
    cache = ROOT / args.output / "features"
    jobs = [([m], source_file(m), matches_root, stands, cache) for m in TRAINING_MONTHS] + [(RANKING_MONTHS, "ranking.parquet", matches_root, stands, cache)]
    with mp.Pool(min(len(jobs), max(1, args.threads // 2))) as pool:
        built = pool.starmap(adsb_table, jobs)
    tables = {job[1]: table for job, table in zip(jobs, built)}
    names = [*BUNDLE_NAMES, *ADSB_NAMES]
    inputs = {"bundle_id": manifest["bundle_id"], "stands_sha256": sha256_file(matches_root / "stands.parquet"),
              "matches_sha256": {m: sha256_file(matches_root / f"matches2_{m}.parquet") for m in TRAINING_MONTHS + RANKING_MONTHS},
              "code": code_receipt(), "lightgbm_parameters": params, "features": len(names), "adsb_features": ADSB_NAMES}
    out.mkdir(parents=True, exist_ok=True)
    if args.fold in ("seasonal", "forward"):
        assignment = pd.read_parquet(parent / "splits" / f"{args.fold}.parquet")
        parts, truths = {}, {}
        for role in ["fit", "early_stop", "score"]:
            features, truth = load_partition(base, parent, catalog, assignment, role, BUNDLE_NAMES)
            if identity_hash(features[ID]) != original["folds"][args.fold]["identity_sha256"][role]:
                raise ValueError("Protected split populations changed.")
            parts[role] = attach(features, tables, catalog)
            truths[role] = aligned_truth(parts[role], truth)[TARGET].to_numpy(dtype=float)
        x, vocab = encode(parts["fit"], names)
        xs, _ = encode(parts["early_stop"], names, vocab)
        booster, history = fit(params, x, truths["fit"], settings["rounds"], xs, truths["early_stop"], settings["early_stopping_rounds"])
        best = int(booster.best_iteration or booster.current_iteration())
        xt, _ = encode(parts["score"], names, vocab)
        prediction = booster.predict(xt, num_iteration=best)
        booster.save_model(str(out / "model.txt"), num_iteration=best)
        pd.DataFrame({ID: parts["score"][ID].to_numpy(), "adsb_full": prediction}).to_parquet(out / "predictions.parquet", index=False, compression="zstd")
        score = parts["score"]
        matched = score["adsb_matched"].eq(1).to_numpy()
        rmse = lambda k: float(np.sqrt(np.mean((truths["score"][k] - prediction[k]) ** 2)))
        months = catalog.set_index(ID)[TIME].reindex(score[ID]).dt.strftime("%Y-%m").to_numpy()
        by_month = {m: rmse(months == m) for m in sorted(set(months))}
        record = {"created_at_utc": utc_now(), "fold": args.fold, "stage": "development", "best_iteration": best, "rows": {r: int(len(p)) for r, p in parts.items()},
                  "score_ids_sha256": identity_hash(score[ID]), "score_rmse": rmse(np.ones(len(score), dtype=bool)), "score_rmse_by_month": by_month,
                  "score_rmse_matched": rmse(matched), "score_rmse_unmatched": rmse(~matched), "matched_share": float(matched.mean()),
                  "early_stop_rmse_curve_tail": {k: v["rmse"][-5:] for k, v in history.items()}, "inputs": inputs, "audit_used": False,
                  "feature_importance_gain_top40": dict(sorted(zip(booster.feature_name(), map(float, booster.feature_importance("gain"))), key=lambda kv: -kv[1])[:40]),
                  "seconds": round(time.time() - started, 1), "environment": environment(ROOT)}
        write_json(out / "run.json", record)
        print({k: record[k] for k in ["fold", "best_iteration", "rows", "score_rmse", "score_rmse_by_month", "score_rmse_matched", "score_rmse_unmatched", "seconds"]})
    else:
        folds = {f: read_json(ROOT / args.output / f / "run.json") for f in ["seasonal", "forward"]}
        for f, rec in folds.items():
            if rec["inputs"]["code"]["sha256"] != inputs["code"]["sha256"] or rec["inputs"]["stands_sha256"] != inputs["stands_sha256"] or rec["inputs"]["matches_sha256"] != inputs["matches_sha256"]:
                raise ValueError(f"Development fold {f} used different inputs.")
        rounds = int(np.rint(np.median([rec["best_iteration"] for rec in folds.values()])))
        training = catalog[catalog["source_file"].isin([source_file(m) for m in TRAINING_MONTHS])]
        if training[TIME].dt.strftime("%Y-%m").eq("2025-12").any():
            raise ValueError("December must never enter the refit.")
        assignment = pd.DataFrame({ID: training[ID].to_numpy(), "partition": "fit"})
        features, truth = load_partition(base, parent, catalog, assignment, "fit", BUNDLE_NAMES)
        frame = attach(features, tables, catalog)
        y = aligned_truth(frame, truth)[TARGET].to_numpy(dtype=float)
        x, vocab = encode(frame, names)
        booster, _ = fit(params, x, y, rounds)
        ranking = pd.read_parquet(base / "features" / "ranking.parquet", columns=[ID, *BUNDLE_NAMES])
        if identity_hash(ranking[ID]) != original["ranking_ids_sha256"]:
            raise ValueError("Ranking identities changed.")
        rank = ranking.merge(tables["ranking.parquet"], on=ID, how="left", validate="one_to_one")
        if rank["adsb_matched"].isna().any():
            raise ValueError("ADS-B table does not cover every ranking row.")
        xr, _ = encode(rank, names, vocab)
        prediction = booster.predict(xr, num_iteration=rounds)
        booster.save_model(str(out / "model.txt"), num_iteration=rounds)
        pd.DataFrame({ID: rank[ID].to_numpy(), "adsb_full": prediction}).to_parquet(out / "ranking_predictions.parquet", index=False, compression="zstd")
        record = {"created_at_utc": utc_now(), "fold": "refit", "stage": "refit", "rounds": rounds, "rounds_rule": "median best iteration of the seasonal and forward folds",
                  "development_best_iterations": {f: rec["best_iteration"] for f, rec in folds.items()}, "rows": int(len(frame)), "months": TRAINING_MONTHS,
                  "ranking_rows": int(len(rank)), "ranking_ids_sha256": identity_hash(rank[ID]), "ranking_matched_share": float(rank["adsb_matched"].mean()),
                  "prediction_quantiles": {str(q): float(np.quantile(prediction, q)) for q in [0.01, 0.1, 0.5, 0.9, 0.99]}, "inputs": inputs, "audit_used": False,
                  "seconds": round(time.time() - started, 1), "environment": environment(ROOT)}
        write_json(out / "run.json", record)
        print({k: record[k] for k in ["fold", "rounds", "rows", "ranking_rows", "ranking_matched_share", "prediction_quantiles", "seconds"]})


if __name__ == "__main__":
    main()
