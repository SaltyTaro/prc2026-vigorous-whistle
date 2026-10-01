"""Masked-tree design fit (configs/tree_mask_design_v1.json): one (variant, seed) on the seasonal fold's fit partition, early stopping on
October, predictions for the October (early_stop) partition ONLY, written as stop_predictions.parquet (ID, adsb_full) with a run record.

The data path is tools/run_learner_family_design.py's (tools/run_adsb_learner.py imported unchanged: timeline bundle 276 + 27 ADS-B +
11 v19 features; label target; LIRF left out of fitting and early stopping, every row predicted). The only change: the fitting rows pass
through prc2026_tree_mask.augment with the variant's p (matched rows weight 1 - p plus a masked copy with weight p). The October rows are
never masked. Variant settings (kind, p, overrides, round cap) come from the declaration. No scoring month and no ranking row is read.
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
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import environment, read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET  # noqa: E402
from prc2026.validation import identity_hash  # noqa: E402
from prc2026_phase3.bundle import load_partition  # noqa: E402
from prc2026_phase5r.baseline import aligned_truth  # noqa: E402
from prc2026_timeline.bundle import load_bundle  # noqa: E402
from prc2026_timeline.features import CATEGORICAL_COLUMNS  # noqa: E402
from prc2026_tree_mask import FOCAL_COLUMNS, augment  # noqa: E402
import run_adsb_learner as L  # noqa: E402

DECL = "configs/tree_mask_design_v1.json"
EXTRA_DIR = "data/external/adsblol/features_v19"


def fit_weighted(kind: str, params: dict, x: pd.DataFrame, y: np.ndarray, w: np.ndarray | None, rounds: int, xs: pd.DataFrame, ys: np.ndarray,
                 patience: int) -> L.Model:
    """tools/run_adsb_learner.fit with a fitting weight (None = its exact unweighted call)."""
    cats = [c for c in x.columns if c in CATEGORICAL_COLUMNS]
    if kind == "lightgbm":
        train = lgb.Dataset(x, label=y, weight=w, categorical_feature=cats, free_raw_data=False)
        valid = lgb.Dataset(xs, label=ys, reference=train, categorical_feature=cats)
        callbacks = [lgb.log_evaluation(period=100), lgb.early_stopping(patience, first_metric_only=True, verbose=False)]
        booster = lgb.train(params, train, num_boost_round=rounds, valid_sets=[valid], valid_names=["early_stop"], callbacks=callbacks)
        return L.Model(kind, booster, int(booster.best_iteration or booster.current_iteration()))
    from catboost import CatBoostRegressor, Pool
    model = CatBoostRegressor(iterations=rounds, **params)
    model.fit(Pool(L.catboost_frame(x), y, cat_features=cats, weight=w), verbose=200,
              eval_set=Pool(L.catboost_frame(xs), ys, cat_features=cats), early_stopping_rounds=patience, use_best_model=True)
    best = model.get_best_iteration()
    return L.Model(kind, model, int(best + 1) if best is not None else int(model.tree_count_))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output", type=Path, default=Path("runs/tree_mask/design"))
    p.add_argument("--matches-dir", type=Path, default=Path("data/external/adsblol/matches"))
    p.add_argument("--extra-features", type=Path, default=Path(EXTRA_DIR))
    p.add_argument("--threads", type=int, default=16)
    a = p.parse_args()
    t0 = time.time()
    decl = read_json(ROOT / DECL)
    if not decl.get("declared_at_utc"):
        raise ValueError("The declaration is a draft.")
    spec = decl["variants"][a.variant]
    if a.seed not in decl["seeds"][a.variant]:
        raise ValueError(f"Seed {a.seed} is not declared for {a.variant}.")
    out = ROOT / a.output / f"{a.variant}_s{a.seed}"
    if (out / "run.json").exists():
        raise ValueError(f"Preserve the existing run: {out}")
    out.mkdir(parents=True, exist_ok=True)
    kind, mask_p = spec["kind"], float(spec["mask_p"])
    settings = dict(read_json(ROOT / "configs/timeline_v1.json")[kind])
    params = dict(settings["parameters"])
    if kind == "lightgbm":
        params.update(num_threads=a.threads, seed=a.seed, feature_fraction_seed=a.seed, bagging_seed=a.seed, data_random_seed=a.seed, metric="rmse")
    else:
        params.update(thread_count=a.threads, random_seed=a.seed)
    params.update(spec.get("overrides", {}))
    rounds = int(spec.get("rounds", settings["rounds"]))
    patience = int(settings["early_stopping_rounds"])
    matches_root = ROOT / a.matches_dir
    stands = pd.read_parquet(matches_root / "stands.parquet")
    base = ROOT / read_json(ROOT / "data/timeline/latest.json")["bundle"]
    manifest, parent, original, catalog = load_bundle(ROOT, base)
    cache = out / "features"
    cache.mkdir(exist_ok=True)
    months = L.TRAINING_MONTHS
    jobs = [([m], L.source_file(m), matches_root, stands, cache) for m in months]
    with mp.Pool(min(len(jobs), max(1, a.threads // 2))) as pool:
        built = pool.starmap(L.adsb_table, jobs)
    tables = {j[1]: t for j, t in zip(jobs, built)}
    extra_root = ROOT / a.extra_features
    extra_names = []
    for m in months:
        extra = pd.read_parquet(extra_root / f"features_v19_{m}.parquet")
        extra_names = extra_names or [c for c in extra.columns if c != ID]
        tables[L.source_file(m)] = tables[L.source_file(m)].merge(extra, on=ID, how="left", validate="one_to_one")
    names = [*L.BUNDLE_NAMES, *L.ADSB_NAMES, *extra_names]
    if set(FOCAL_COLUMNS) - set(names):
        raise ValueError("Focal columns missing from the learner contract.")
    assignment = pd.read_parquet(parent / "splits" / "seasonal.parquet")
    parts, truths = {}, {}
    for role in ["fit", "early_stop"]:
        feats, truth = load_partition(base, parent, catalog, assignment, role, L.BUNDLE_NAMES)
        if identity_hash(feats[ID]) != original["folds"]["seasonal"]["identity_sha256"][role]:
            raise ValueError("Protected split populations changed.")
        parts[role] = L.attach(feats, tables, catalog)
        truths[role] = aligned_truth(parts[role], truth)[TARGET].to_numpy(dtype=float)

    def keep(frame):
        return ~frame["airport"].isin(["LIRF"]).to_numpy()
    fk, sk = keep(parts["fit"]), keep(parts["early_stop"])
    x, vocab = L.encode(parts["fit"][fk].reset_index(drop=True), names)
    xs, _ = L.encode(parts["early_stop"][sk].reset_index(drop=True), names, vocab)
    xa, ya, wa = augment(x, truths["fit"][fk], mask_p)
    model = fit_weighted(kind, params, xa, ya, wa, rounds, xs, truths["early_stop"][sk], patience)
    xt, _ = L.encode(parts["early_stop"], names, vocab)
    pred = model.predict(xt)
    pd.DataFrame({ID: parts["early_stop"][ID].to_numpy(), "adsb_full": pred}).to_parquet(out / "stop_predictions.parquet", index=False, compression="zstd")
    y, k = truths["early_stop"], sk
    unmatched = parts["early_stop"]["adsb_matched"].to_numpy(dtype=float) == 0
    rec = {"created_at_utc": utc_now(), "variant": a.variant, "seed": a.seed, "fold": "design", "kind": kind, "mask_p": mask_p,
           "best_iteration": model.best, "rounds_cap": rounds, "parameters": {k2: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k2, v in params.items()},
           "fitted_rows": int(fk.sum()), "fitted_rows_after_augmentation": int(len(xa)), "masked_copies": int(len(xa) - fk.sum()),
           "october_rows": int(len(pred)), "october_rmse_non_lirf": float(np.sqrt(np.mean((y[k] - pred[k]) ** 2))),
           "october_rmse_non_lirf_unmatched": float(np.sqrt(np.mean((y[k & unmatched] - pred[k & unmatched]) ** 2))),
           "features": len(names), "declaration_sha256": sha256_file(ROOT / DECL),
           "code_sha256": {"runner": sha256_file(Path(__file__)), "augment": sha256_file(ROOT / "src/prc2026_tree_mask/__init__.py")},
           "extra_tables_sha256": {m: sha256_file(extra_root / f"features_v19_{m}.parquet") for m in months},
           "audit_used": False, "seconds": round(time.time() - t0, 1), "environment": environment(ROOT)}
    write_json(out / "run.json", rec)
    print({k2: rec[k2] for k2 in ["variant", "seed", "best_iteration", "october_rmse_non_lirf", "october_rmse_non_lirf_unmatched", "seconds"]})


if __name__ == "__main__":
    main()
