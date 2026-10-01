"""Learner-family design fit (configs/learner_family_design_v1.json): one (variant, seed) on the seasonal fold's fit partition, early
stopping on October (not under DART, which has none: it takes `--rounds` from the control's best iteration), predictions for the
October (early_stop) partition ONLY, written as stop_predictions.parquet (ID, adsb_full) with a run record.

The data path is tools/run_adsb_learner.py's, imported unchanged (timeline bundle 276 + 27 ADS-B + 11 v19 features; label target;
LIRF left out of fitting and early stopping, every row predicted). No scoring month and no ranking row is predicted or read.
Variants: ref_lgb (no override), dart, linear_tree, huber_tail: the override sets are in the declaration and are passed with --param.
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
from prc2026.artifacts import environment, read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET  # noqa: E402
from prc2026.validation import identity_hash  # noqa: E402
from prc2026_phase3.bundle import load_partition  # noqa: E402
from prc2026_phase5r.baseline import aligned_truth  # noqa: E402
from prc2026_timeline.bundle import load_bundle  # noqa: E402
import run_adsb_learner as L  # noqa: E402

EXTRA_DIR = "data/external/adsblol/features_v19"  # the v19 table directory of the committee learners; checked below


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output", type=Path, default=Path("runs/learner_family/design"))
    p.add_argument("--matches-dir", type=Path, default=Path("data/external/adsblol/matches"))
    p.add_argument("--extra-features", type=Path, default=Path(EXTRA_DIR))
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--rounds", type=int)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--param", action="append", default=[])
    a = p.parse_args()
    t0 = time.time()
    out = ROOT / a.output / f"{a.variant}_s{a.seed}"
    if (out / "run.json").exists():
        raise ValueError(f"Preserve the existing run: {out}")
    out.mkdir(parents=True, exist_ok=True)
    settings = dict(read_json(ROOT / "configs/timeline_v1.json")["lightgbm"])
    params = dict(settings["parameters"])
    params.update(num_threads=a.threads, seed=a.seed, feature_fraction_seed=a.seed, bagging_seed=a.seed, data_random_seed=a.seed, metric="rmse")
    overrides = {k: L.parse_value(v) for k, _, v in (i.partition("=") for i in a.param)}
    params.update(overrides)
    rounds = a.rounds if a.rounds is not None else settings["rounds"]
    patience = a.patience if a.patience is not None else settings["early_stopping_rounds"]
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
    dart = str(params.get("boosting_type", "gbdt")) == "dart"
    booster, history = L.fit("lightgbm", params, x, truths["fit"][fk], rounds, None if dart else xs, None if dart else truths["early_stop"][sk], patience)
    xt, _ = L.encode(parts["early_stop"], names, vocab)
    pred = booster.predict(xt)
    pd.DataFrame({ID: parts["early_stop"][ID].to_numpy(), "adsb_full": pred}).to_parquet(out / "stop_predictions.parquet", index=False, compression="zstd")
    y = truths["early_stop"]
    k = sk
    rec = {"created_at_utc": utc_now(), "variant": a.variant, "seed": a.seed, "fold": "design", "best_iteration": booster.best, "rounds_cap": rounds,
           "dart": dart, "overrides": overrides, "parameters": params, "fitted_rows": int(fk.sum()), "october_rows": int(len(pred)),
           "october_rmse_non_lirf": float(np.sqrt(np.mean((y[k] - pred[k]) ** 2))), "features": len(names),
           "extra_tables_sha256": {m: sha256_file(extra_root / f"features_v19_{m}.parquet") for m in months},
           "audit_used": False, "seconds": round(time.time() - t0, 1), "environment": environment(ROOT)}
    write_json(out / "run.json", rec)
    print({k: rec[k] for k in ["variant", "seed", "best_iteration", "october_rmse_non_lirf", "seconds"]})


if __name__ == "__main__":
    main()
