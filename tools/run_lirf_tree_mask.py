"""Masked LIRF tree design fit (configs/lirf_tree_mask_design_v1.json): one variant of one of the seven learners of ens_v22_lirf_q, refitted
on the SEASONAL fold exactly as tools/run_adsb_learner.py fitted the original (same kind, target, seed, parameters, round cap, patience,
label cap, LIRF fill population left out, matches data/external/adsblol/matches), with one change: LIRF non-population ADS-B-matched
FITTING rows keep weight 1 - p and gain a masked copy with weight p (early-stopping and score rows are never masked).

Masking semantics (declared):
  focal  the 15 focal ADS-B columns NaN and adsb_matched 0 (prc2026_tree_mask.augment's semantics; neighbour columns untouched)
  dark   focal as above, plus the neighbour columns set to what LIRF rows carry when the airport has no ADS-B (July 2026 ranking and July
         2025 thinned to it): the three neighbour counts 0, every other neighbour column NaN
p = 0 is the exact unweighted call of the original runner (the control).

Output layout = tools/run_adsb_learner.py's seasonal fold (<output>/<name>/seasonal/{model.txt|model.cbm, predictions.parquet, run.json};
run.json carries the original runner's `inputs` keys), so prc2026_december.scoring.learner_prediction can re-predict it on thinned inputs.
--build-cache-only builds the shared ADS-B feature cache (<output>/features) and exits (run once per worker before parallel lanes).
No scoring label is read beyond the run record's own fold RMSE; no ranking row is predicted; nothing is uploaded.
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
from prc2026.features import ID, TARGET, TIME  # noqa: E402
from prc2026.validation import identity_hash  # noqa: E402
from prc2026_adsb.features import FEATURE_NAMES, NEIGHBOUR_NAMES  # noqa: E402
from prc2026_phase3.bundle import load_partition  # noqa: E402
from prc2026_phase5r.baseline import aligned_truth  # noqa: E402
from prc2026_timeline.bundle import load_bundle  # noqa: E402
from prc2026_timeline.features import CATEGORICAL_COLUMNS  # noqa: E402
import run_adsb_learner as L  # noqa: E402

DECL = "configs/lirf_tree_mask_design_v1.json"
MATCHED = "adsb_matched"
FOCAL = [c for c in FEATURE_NAMES if c != MATCHED]
NB_COUNTS = ["nb_runway_10m_count", "nb_airport_10m_count", "nb_runway_30m_count"]
NB_OTHER = [c for c in NEIGHBOUR_NAMES if c not in NB_COUNTS]


def augment_rows(x: pd.DataFrame, y: np.ndarray, p: float, rows: np.ndarray, semantics: str):
    """(x, y, weight): rows[i] True -> weight 1 - p plus an appended masked copy with weight p (copies in the original order); p == 0 -> unchanged, weight None."""
    if not 0.0 <= p < 1.0:
        raise ValueError("p must lie in [0, 1).")
    if semantics not in ("focal", "dark"):
        raise ValueError(f"Unknown masking semantics {semantics!r}.")
    y = np.asarray(y, dtype=float)
    rows = np.asarray(rows, dtype=bool)
    if len(y) != len(x) or len(rows) != len(x):
        raise ValueError("Frame, label and row-mask lengths differ.")
    if p == 0.0:
        return x, y, None
    if not (x.loc[rows, MATCHED].to_numpy(dtype=float) == 1.0).all():
        raise ValueError("Only matched rows may be masked.")
    copy = x.loc[rows].copy()
    for c in FOCAL:
        copy[c] = np.nan
    copy[MATCHED] = 0.0
    if semantics == "dark":
        for c in NB_COUNTS:
            copy[c] = 0.0
        for c in NB_OTHER:
            copy[c] = np.nan
    copy = copy.astype(x.dtypes.to_dict())
    out = pd.concat([x, copy], ignore_index=True)
    weight = np.concatenate([np.where(rows, 1.0 - p, 1.0), np.full(int(rows.sum()), p)])
    return out, np.concatenate([y, y[rows]]), weight


def fit_weighted(kind: str, params: dict, x: pd.DataFrame, y: np.ndarray, w: np.ndarray | None, rounds: int, xs: pd.DataFrame, ys: np.ndarray,
                 patience: int) -> tuple[L.Model, dict]:
    """tools/run_adsb_learner.fit with a fitting weight (None = its exact unweighted call)."""
    cats = [c for c in x.columns if c in CATEGORICAL_COLUMNS]
    if kind == "lightgbm":
        train = lgb.Dataset(x, label=y, weight=w, categorical_feature=cats, free_raw_data=False)
        history = {}
        callbacks = [lgb.log_evaluation(period=100), lgb.record_evaluation(history), lgb.early_stopping(patience, first_metric_only=True, verbose=False)]
        booster = lgb.train(params, train, num_boost_round=rounds, valid_sets=[lgb.Dataset(xs, label=ys, reference=train, categorical_feature=cats)],
                            valid_names=["early_stop"], callbacks=callbacks)
        return L.Model(kind, booster, int(booster.best_iteration or booster.current_iteration())), {k: {"rmse": v["rmse"][-5:]} for k, v in history.items()}
    from catboost import CatBoostRegressor, Pool
    model = CatBoostRegressor(iterations=rounds, **params)
    model.fit(Pool(L.catboost_frame(x), y, cat_features=cats, weight=w), verbose=200,
              eval_set=Pool(L.catboost_frame(xs), ys, cat_features=cats), early_stopping_rounds=patience, use_best_model=True)
    best = model.get_best_iteration()
    history = {"early_stop": {"rmse": list(map(float, model.get_evals_result().get("validation", {}).get("RMSE", [])[-5:]))}}
    return L.Model(kind, model, int(best + 1) if best is not None else int(model.tree_count_)), history


def tables(output: Path, threads: int) -> dict:
    matches_root = ROOT / "data/external/adsblol/matches"
    stands = pd.read_parquet(matches_root / "stands.parquet")
    cache = ROOT / output / "features"
    cache.mkdir(parents=True, exist_ok=True)
    jobs = [([m], L.source_file(m), matches_root, stands, cache) for m in L.TRAINING_MONTHS]
    with mp.Pool(min(len(jobs), max(1, threads // 2))) as pool:
        built = pool.starmap(L.adsb_table, jobs)
    return {j[1]: t for j, t in zip(jobs, built)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run")
    p.add_argument("--output", type=Path, default=Path("runs/lirf_tree_mask/design"))
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--build-cache-only", action="store_true")
    a = p.parse_args()
    t0 = time.time()
    if a.build_cache_only:
        built = tables(a.output, a.threads)
        print({k: len(v) for k, v in built.items()}, round(time.time() - t0, 1))
        return
    decl = read_json(ROOT / DECL)
    if not decl.get("declared_at_utc"):
        raise ValueError("The declaration is a draft.")
    spec = decl["runs"][a.run]
    learner = decl["learners"][spec["learner"]]
    p_mask, semantics = float(spec["mask_p"]), spec["semantics"]
    out = ROOT / a.output / a.run / "seasonal"
    if (out / "run.json").exists():
        raise ValueError(f"Preserve the existing run: {out}")
    original = read_json(ROOT / learner["original"] / "seasonal" / "run.json")["inputs"]
    kind, target = original["kind"], original.get("target") or "label"
    settings = dict(read_json(ROOT / "configs/timeline_v1.json")[kind])
    params = dict(settings["parameters"])
    seed = int(original["seed"])
    if kind == "lightgbm":
        params.update(num_threads=a.threads, seed=seed, feature_fraction_seed=seed, bagging_seed=seed, data_random_seed=seed, metric="rmse")
    else:
        params.update(thread_count=a.threads, random_seed=seed)
    params.update(original.get("parameter_overrides") or {})
    expected = {k: v for k, v in original["parameters"].items() if k not in ("num_threads", "thread_count")}
    got = {k: v for k, v in params.items() if k not in ("num_threads", "thread_count")}
    if expected != got:
        raise ValueError(f"Parameters differ from the original run: {expected} vs {got}")
    rounds, patience, max_label = int(original["rounds_cap"]), int(original["early_stopping_rounds"]), original.get("max_label")
    if original.get("excluded_airports") or original.get("specialist_airports") or original.get("extra_features") or not original.get("exclude_population"):
        raise ValueError("The original learner is not a plain LIRF-inclusive, population-excluding run.")
    matches_root = ROOT / "data/external/adsblol/matches"
    if sha256_file(matches_root / "stands.parquet") != original["stands_sha256"] or any(
            sha256_file(matches_root / f"matches2_{m}.parquet") != original["matches_sha256"][m] for m in L.TRAINING_MONTHS):
        raise ValueError("Matches or stands differ from the original run's.")
    built = tables(a.output, a.threads)
    base = ROOT / read_json(ROOT / "data/timeline/latest.json")["bundle"]
    manifest, parent, orig_bundle, catalog = load_bundle(ROOT, base)
    if manifest["bundle_id"] != original["bundle_id"]:
        raise ValueError("The timeline bundle differs from the original run's.")
    names = [*L.BUNDLE_NAMES, *L.ADSB_NAMES]
    assignment = pd.read_parquet(parent / "splits" / "seasonal.parquet")
    parts, truths = {}, {}
    for role in ["fit", "early_stop", "score"]:
        feats, truth = load_partition(base, parent, catalog, assignment, role, L.BUNDLE_NAMES)
        if identity_hash(feats[ID]) != orig_bundle["folds"]["seasonal"]["identity_sha256"][role]:
            raise ValueError("Protected split populations changed.")
        parts[role] = L.attach(feats, built, catalog)
        truths[role] = aligned_truth(parts[role], truth)[TARGET].to_numpy(dtype=float)

    def clock(frame):
        return frame["nmclock_available"].to_numpy(dtype=float) == 1 if target == "nm_residual" else np.ones(len(frame), dtype=bool)

    def keep(frame, labels):
        m = clock(frame) & ~(frame["airport"].eq("LIRF") & frame["nmclock_available"].eq(0) & frame["aircraft_missing"].eq(1)).to_numpy()
        if max_label is not None:
            m &= labels <= max_label
        return m

    def offset(frame):
        return frame["nmclock_proxy_s"].to_numpy(dtype=float) if target == "nm_residual" else np.zeros(len(frame))
    fk, sk = keep(parts["fit"], truths["fit"]), keep(parts["early_stop"], truths["early_stop"])
    fit_frame = parts["fit"][fk].reset_index(drop=True)
    x, vocab = L.encode(fit_frame, names)
    xs, _ = L.encode(parts["early_stop"][sk].reset_index(drop=True), names, vocab)
    rows = (fit_frame["airport"].eq("LIRF") & fit_frame[MATCHED].eq(1)).to_numpy()
    yf = truths["fit"][fk] - offset(parts["fit"])[fk]
    xa, ya, wa = augment_rows(x, yf, p_mask, rows, semantics)
    model, history = fit_weighted(kind, params, xa, ya, wa, rounds, xs, truths["early_stop"][sk] - offset(parts["early_stop"])[sk], patience)
    xt, _ = L.encode(parts["score"], names, vocab)
    pred = np.where(clock(parts["score"]), offset(parts["score"]) + model.predict(xt), np.nan)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / ("model.txt" if kind == "lightgbm" else "model.cbm"))
    pd.DataFrame({ID: parts["score"][ID].to_numpy(), "adsb_full": pred}).to_parquet(out / "predictions.parquet", index=False, compression="zstd")
    score = parts["score"]
    months = catalog.set_index(ID)[TIME].reindex(score[ID]).dt.strftime("%Y-%m").to_numpy()
    lirf_u = (score["airport"].eq("LIRF") & score[MATCHED].eq(0)).to_numpy()

    def rmse(k):
        e = (truths["score"][k] - pred[k]) ** 2
        return float(np.sqrt(np.nanmean(e))) if np.isfinite(e).any() else None
    inputs = dict(original)
    inputs.update({"parameters": params, "code": L.code_receipt(), "mask": {"p": p_mask, "semantics": semantics, "rows": "LIRF non-population matched fitting rows",
                                                                             "runner_sha256": sha256_file(Path(__file__))}})
    record = {"created_at_utc": utc_now(), "fold": "seasonal", "stage": "design", "run": a.run, "learner": spec["learner"], "original": learner["original"],
              "best_iteration": model.best, "rows": {r: int(len(v)) for r, v in parts.items()}, "fitted_rows": int(fk.sum()), "early_stop_rows": int(sk.sum()),
              "masked_copies": int(len(xa) - len(x)), "fitted_rows_after_augmentation": int(len(xa)),
              "score_ids_sha256": identity_hash(score[ID]), "score_rmse_by_month": {m: rmse(months == m) for m in sorted(set(months))},
              "score_rmse_lirf_unmatched_by_month": {m: rmse((months == m) & lirf_u) for m in sorted(set(months))},
              "early_stop_rmse_curve_tail": history, "inputs": inputs, "declaration_sha256": sha256_file(ROOT / DECL), "audit_used": False,
              "seconds": round(time.time() - t0, 1), "environment": environment(ROOT)}
    write_json(out / "run.json", record)
    print({k: record[k] for k in ["run", "best_iteration", "masked_copies", "score_rmse_by_month", "score_rmse_lirf_unmatched_by_month", "seconds"]}, flush=True)


if __name__ == "__main__":
    main()
