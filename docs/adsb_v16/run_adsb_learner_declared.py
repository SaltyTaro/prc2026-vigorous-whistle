"""Full-data ADS-B learners: LightGBM or CatBoost on the timeline bundle model features plus the ADS-B features, target = label or NM residual.

Generalises run_adsb_full.py (kept unchanged while configs/adsb_v12_v3.json binds its hash): `--kind lightgbm|catboost` and
`--target label|nm_residual|adsb_residual`. With the NM-residual target, rows without the NM clock are neither fitted nor predicted (NaN).
With the ADS-B-residual target the model learns label minus the observed ADS-B taxi (takeoff minus motion start) on matched rows and
the plain label elsewhere; the offset is added back, so every row is predicted.
The output column is always `adsb_full`; the residual stage binds each table under its own name.

`--param key=value` (repeatable) overrides single learner parameters, `--rounds` the
round cap and `--patience` the early-stopping patience; `--max-label` drops fitting and
early-stopping rows whose label exceeds the value (every row is still predicted). All
overrides are recorded in the run's inputs; the defaults reproduce the earlier runs.

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


def parse_value(raw: str):
    """Numbers and booleans from the command line; anything else stays a string."""
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    try:
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


def code_receipt() -> dict:
    paths = [ROOT / "src/prc2026_adsb/features.py", ROOT / "tools/run_adsb_learner.py"]
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


class Model:
    """Uniform wrapper over a LightGBM booster or a CatBoost regressor."""

    def __init__(self, kind: str, native, best: int):
        self.kind, self.native, self.best = kind, native, best

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if self.kind == "lightgbm":
            return self.native.predict(x, num_iteration=self.best)
        return self.native.predict(catboost_frame(x))

    def save(self, path: Path) -> None:
        if self.kind == "lightgbm":
            self.native.save_model(str(path), num_iteration=self.best)
        else:
            self.native.save_model(str(path))

    def feature_names(self) -> list[str]:
        return list(self.native.feature_name()) if self.kind == "lightgbm" else list(self.native.feature_names_)

    def importance(self) -> list[float]:
        return list(map(float, self.native.feature_importance("gain"))) if self.kind == "lightgbm" else list(map(float, self.native.get_feature_importance()))


def catboost_frame(x: pd.DataFrame) -> pd.DataFrame:
    """CatBoost wants categorical columns as strings, not pandas categoricals."""
    out = x.copy()
    for name in out.columns:
        if name in CATEGORICAL_COLUMNS:
            out[name] = out[name].astype(str)
    return out


def fit(kind: str, params: dict, x: pd.DataFrame, y: np.ndarray, rounds: int, xs: pd.DataFrame | None = None, ys: np.ndarray | None = None, patience: int = 200) -> tuple[Model, dict]:
    cats = [c for c in x.columns if c in CATEGORICAL_COLUMNS]
    if kind == "lightgbm":
        train = lgb.Dataset(x, label=y, categorical_feature=cats, free_raw_data=False)
        history, kwargs = {}, {}
        callbacks = [lgb.log_evaluation(period=100), lgb.record_evaluation(history)]
        if xs is not None:
            kwargs = {"valid_sets": [lgb.Dataset(xs, label=ys, reference=train, categorical_feature=cats)], "valid_names": ["early_stop"]}
            callbacks.append(lgb.early_stopping(patience, first_metric_only=True, verbose=False))
        booster = lgb.train(params, train, num_boost_round=rounds, callbacks=callbacks, **kwargs)
        best = int(booster.best_iteration or booster.current_iteration())
        return Model(kind, booster, best), {k: {"rmse": v["rmse"][-5:]} for k, v in history.items()}
    from catboost import CatBoostRegressor, Pool
    model = CatBoostRegressor(iterations=rounds, **params)
    kwargs = {}
    if xs is not None:
        kwargs = {"eval_set": Pool(catboost_frame(xs), ys, cat_features=cats), "early_stopping_rounds": patience, "use_best_model": True}
    model.fit(Pool(catboost_frame(x), y, cat_features=cats), verbose=100, **kwargs)
    best = int(model.get_best_iteration() + 1) if xs is not None and model.get_best_iteration() is not None else int(model.tree_count_)
    history = {"early_stop": {"rmse": list(map(float, model.get_evals_result().get("validation", {}).get("RMSE", [])[-5:]))}} if xs is not None else {}
    return Model(kind, model, best), history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", required=True, choices=["seasonal", "forward", "refit"])
    parser.add_argument("--matches-dir", type=Path, default=Path("data/external/adsblol/matches"))
    parser.add_argument("--output", type=Path, default=Path("runs/adsb_full"))
    parser.add_argument("--settings", type=Path, default=Path("configs/timeline_v1.json"), help="Learner parameters are taken from this declaration's lightgbm / catboost block")
    parser.add_argument("--kind", choices=["lightgbm", "catboost"], default="lightgbm")
    parser.add_argument("--target", choices=["label", "nm_residual", "adsb_residual"], default="label")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--exclude-airports", nargs="*", default=[], help="Airports left out of fitting and early stopping (their rows are still predicted)")
    parser.add_argument("--airports", nargs="*", default=[], help="Specialist: fit, early-stop and predict ONLY these airports (other rows get NaN)")
    parser.add_argument("--param", action="append", default=[], metavar="KEY=VALUE", help="Override one learner parameter (repeatable; numbers and true/false are parsed)")
    parser.add_argument("--rounds", type=int, help="Override the declaration's round cap")
    parser.add_argument("--patience", type=int, help="Override the declaration's early-stopping patience")
    parser.add_argument("--max-label", type=float, help="Fit and early-stop only on rows whose label is at most this value (every row is still predicted)")
    args = parser.parse_args()
    started = time.time()
    out = ROOT / args.output / args.fold
    if (out / "run.json").exists():
        raise ValueError(f"Preserve the existing run: {out}")
    settings = dict(read_json(ROOT / args.settings)[args.kind])
    params = dict(settings["parameters"])
    if args.kind == "lightgbm":
        params.update(num_threads=args.threads, seed=args.seed, feature_fraction_seed=args.seed, bagging_seed=args.seed, data_random_seed=args.seed, metric="rmse")
    else:
        params.update(thread_count=args.threads, random_seed=args.seed)
    overrides = {}
    for item in args.param:
        key, _, raw = item.partition("=")
        overrides[key] = parse_value(raw)
    params.update(overrides)
    if args.rounds is not None:
        settings["rounds"] = args.rounds
    if args.patience is not None:
        settings["early_stopping_rounds"] = args.patience
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
              "code": code_receipt(), "kind": args.kind, "parameters": params, "features": len(names), "adsb_features": ADSB_NAMES, "excluded_airports": sorted(args.exclude_airports), "target": args.target,
              "parameter_overrides": overrides, "rounds_cap": settings["rounds"], "early_stopping_rounds": settings["early_stopping_rounds"], "max_label": args.max_label, "seed": args.seed,
              "specialist_airports": sorted(args.airports)}

    def clock_mask(frame: pd.DataFrame) -> np.ndarray:
        mask = frame["nmclock_available"].to_numpy(dtype=float) == 1 if args.target == "nm_residual" else np.ones(len(frame), dtype=bool)
        if args.airports:
            mask &= frame["airport"].isin(args.airports).to_numpy()
        return mask

    def keep(frame: pd.DataFrame, labels: np.ndarray | None = None) -> np.ndarray:
        mask = clock_mask(frame) & ~frame["airport"].isin(args.exclude_airports).to_numpy()
        if args.max_label is not None and labels is not None:
            mask &= labels <= args.max_label
        return mask

    def offset(frame: pd.DataFrame) -> np.ndarray:
        if args.target == "nm_residual":
            return frame["nmclock_proxy_s"].to_numpy(dtype=float)
        if args.target == "adsb_residual":
            motion = frame["adsb_motion_s"].to_numpy(dtype=float)
            return np.where(np.isfinite(motion), motion, 0.0)
        return np.zeros(len(frame))
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
        fit_keep, stop_keep = keep(parts["fit"], truths["fit"]), keep(parts["early_stop"], truths["early_stop"])
        x, vocab = encode(parts["fit"][fit_keep].reset_index(drop=True), names)
        xs, _ = encode(parts["early_stop"][stop_keep].reset_index(drop=True), names, vocab)
        booster, history = fit(args.kind, params, x, truths["fit"][fit_keep] - offset(parts["fit"])[fit_keep], settings["rounds"], xs,
                               truths["early_stop"][stop_keep] - offset(parts["early_stop"])[stop_keep], settings["early_stopping_rounds"])
        best = booster.best
        xt, _ = encode(parts["score"], names, vocab)
        clock = clock_mask(parts["score"])
        prediction = np.where(clock, offset(parts["score"]) + booster.predict(xt), np.nan)
        booster.save(out / ("model.txt" if args.kind == "lightgbm" else "model.cbm"))
        pd.DataFrame({ID: parts["score"][ID].to_numpy(), "adsb_full": prediction}).to_parquet(out / "predictions.parquet", index=False, compression="zstd")
        score = parts["score"]
        matched = score["adsb_matched"].eq(1).to_numpy()
        rmse = lambda k: float(np.sqrt(np.nanmean((truths["score"][k] - prediction[k]) ** 2)))
        months = catalog.set_index(ID)[TIME].reindex(score[ID]).dt.strftime("%Y-%m").to_numpy()
        by_month = {m: rmse(months == m) for m in sorted(set(months))}
        record = {"created_at_utc": utc_now(), "fold": args.fold, "stage": "development", "best_iteration": best, "rows": {r: int(len(p)) for r, p in parts.items()},
                  "fitted_rows": int(fit_keep.sum()), "early_stop_rows": int(stop_keep.sum()),
                  "score_ids_sha256": identity_hash(score[ID]), "score_rmse": rmse(np.ones(len(score), dtype=bool)), "score_rmse_by_month": by_month,
                  "score_rmse_matched": rmse(matched), "score_rmse_unmatched": rmse(~matched), "matched_share": float(matched.mean()),
                  "early_stop_rmse_curve_tail": history, "inputs": inputs, "audit_used": False,
                  "feature_importance_top40": dict(sorted(zip(booster.feature_names(), booster.importance()), key=lambda kv: -kv[1])[:40]),
                  "seconds": round(time.time() - started, 1), "environment": environment(ROOT)}
        write_json(out / "run.json", record)
        print({k: record[k] for k in ["fold", "best_iteration", "rows", "score_rmse", "score_rmse_by_month", "score_rmse_matched", "score_rmse_unmatched", "seconds"]})
    else:
        folds = {f: read_json(ROOT / args.output / f / "run.json") for f in ["seasonal", "forward"]}
        for f, rec in folds.items():
            if (rec["inputs"]["code"]["sha256"] != inputs["code"]["sha256"] or rec["inputs"]["stands_sha256"] != inputs["stands_sha256"] or rec["inputs"]["matches_sha256"] != inputs["matches_sha256"]
                    or rec["inputs"].get("excluded_airports", []) != inputs["excluded_airports"] or rec["inputs"].get("kind") != args.kind or rec["inputs"].get("target") != args.target
                    or rec["inputs"].get("parameters") != inputs["parameters"] or rec["inputs"].get("max_label") != args.max_label
                    or rec["inputs"].get("specialist_airports", []) != inputs["specialist_airports"]):
                raise ValueError(f"Development fold {f} used different inputs.")
        rounds = int(np.rint(np.median([rec["best_iteration"] for rec in folds.values()])))
        training = catalog[catalog["source_file"].isin([source_file(m) for m in TRAINING_MONTHS])]
        if training[TIME].dt.strftime("%Y-%m").eq("2025-12").any():
            raise ValueError("December must never enter the refit.")
        assignment = pd.DataFrame({ID: training[ID].to_numpy(), "partition": "fit"})
        features, truth = load_partition(base, parent, catalog, assignment, "fit", BUNDLE_NAMES)
        frame = attach(features, tables, catalog)
        y = aligned_truth(frame, truth)[TARGET].to_numpy(dtype=float)
        fit_keep = keep(frame, y)
        x, vocab = encode(frame[fit_keep].reset_index(drop=True), names)
        booster, _ = fit(args.kind, params, x, y[fit_keep] - offset(frame)[fit_keep], rounds)
        ranking = pd.read_parquet(base / "features" / "ranking.parquet", columns=[ID, *BUNDLE_NAMES])
        if identity_hash(ranking[ID]) != original["ranking_ids_sha256"]:
            raise ValueError("Ranking identities changed.")
        rank = ranking.merge(tables["ranking.parquet"], on=ID, how="left", validate="one_to_one")
        if rank["adsb_matched"].isna().any():
            raise ValueError("ADS-B table does not cover every ranking row.")
        xr, _ = encode(rank, names, vocab)
        clock = clock_mask(rank)
        prediction = np.where(clock, offset(rank) + booster.predict(xr), np.nan)
        booster.save(out / ("model.txt" if args.kind == "lightgbm" else "model.cbm"))
        pd.DataFrame({ID: rank[ID].to_numpy(), "adsb_full": prediction}).to_parquet(out / "ranking_predictions.parquet", index=False, compression="zstd")
        record = {"created_at_utc": utc_now(), "fold": "refit", "stage": "refit", "rounds": rounds, "rounds_rule": "median best iteration of the seasonal and forward folds",
                  "development_best_iterations": {f: rec["best_iteration"] for f, rec in folds.items()}, "rows": int(len(frame)), "fitted_rows": int(fit_keep.sum()), "months": TRAINING_MONTHS,
                  "ranking_rows": int(len(rank)), "ranking_ids_sha256": identity_hash(rank[ID]), "ranking_matched_share": float(rank["adsb_matched"].mean()),
                  "prediction_quantiles": {str(q): float(np.nanquantile(prediction, q)) for q in [0.01, 0.1, 0.5, 0.9, 0.99]}, "clock_share": float(clock.mean()), "inputs": inputs, "audit_used": False,
                  "seconds": round(time.time() - started, 1), "environment": environment(ROOT)}
        write_json(out / "run.json", record)
        print({k: record[k] for k in ["fold", "rounds", "rows", "ranking_rows", "ranking_matched_share", "prediction_quantiles", "seconds"]})


if __name__ == "__main__":
    main()
