"""One declared design run of the unmatched-row specialist (configs/adsb_unm_design_v1.json): the seasonal fold, predicting October.

The run's learner is a committee member's recipe (tools/run_adsb_learner.py: the timeline bundle's 276 features, the 27 ADS-B features of
data/external/adsblol/matches and the eleven v19 features; LIRF left out of fitting and early stopping) with one change, its scope:
  all        every non-LIRF row, unweighted (the member's own recipe; the reproduction run)
  unmatched  only non-LIRF rows without an ADS-B match (adsb_matched != 1) are fitted and early-stopped on
  weighted   every non-LIRF row, rows without an ADS-B match carry the run's sample weight in the loss and in the early-stopping metric
Fit partition: the seasonal fold's fit rows (2025-02..06, 08, 09); early stopping: its October rows; output: every October row
(`stop_predictions.parquet`: ID, adsb_full) with a run record. January, July, November and December are never read.
`--smoke-rows N` runs on the first N fit rows of 2025-09 and N October rows with a 30-round cap into a scratch root (the report refuses it).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import environment, object_hash, read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET, TIME  # noqa: E402
from prc2026.validation import identity_hash  # noqa: E402
from prc2026_phase3.bundle import load_partition  # noqa: E402
from prc2026_phase5r.baseline import aligned_truth  # noqa: E402
from prc2026_timeline.bundle import load_bundle  # noqa: E402
from prc2026_timeline.features import CATEGORICAL_COLUMNS  # noqa: E402
from run_adsb_learner import ADSB_NAMES, BUNDLE_NAMES, Model, adsb_table, attach, catboost_frame, encode, fit, source_file  # noqa: E402

SMOKE_FIT_MONTH = "2025-09"
SMOKE_ROUNDS = 30


def code_receipt() -> dict:
    paths = [ROOT / "src/prc2026_adsb/features.py", ROOT / "tools/run_adsb_learner.py", ROOT / "tools/run_adsb_unm.py"]
    hashes = {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths}
    return {"files": hashes, "sha256": object_hash(hashes)}


def design_masks(frame: pd.DataFrame, scope: str, weight: float, excluded: list[str]) -> tuple[np.ndarray, np.ndarray | None]:
    """Rows fitted (or early-stopped on) and their sample weights; None when every kept row has weight 1 (the members' exact path)."""
    unmatched = frame["adsb_matched"].to_numpy(dtype=float) != 1
    keep = ~frame["airport"].isin(excluded).to_numpy()
    if scope == "unmatched":
        keep &= unmatched
    elif scope not in ("all", "weighted"):
        raise ValueError(f"Unknown scope {scope}.")
    if scope != "weighted" or weight == 1.0:
        return keep, None
    if not weight > 0:
        raise ValueError("Sample weights must be positive.")
    return keep, np.where(unmatched[keep], weight, 1.0)


def fit_weighted(kind: str, params: dict, x: pd.DataFrame, y: np.ndarray, w: np.ndarray | None, rounds: int, xs: pd.DataFrame, ys: np.ndarray,
                 ws: np.ndarray | None, patience: int) -> tuple[Model, dict]:
    """run_adsb_learner.fit with sample weights in the loss and the early-stopping metric; without weights it IS run_adsb_learner.fit."""
    if w is None and ws is None:
        return fit(kind, params, x, y, rounds, xs, ys, patience)
    if w is None or ws is None or len(w) != len(y) or len(ws) != len(ys):
        raise ValueError("Weights must be given for both the fitting and the early-stopping rows.")
    cats = [c for c in x.columns if c in CATEGORICAL_COLUMNS]
    if kind == "lightgbm":
        train = lgb.Dataset(x, label=y, weight=w, categorical_feature=cats, free_raw_data=False)
        history = {}
        callbacks = [lgb.log_evaluation(period=100), lgb.record_evaluation(history), lgb.early_stopping(patience, first_metric_only=True, verbose=False)]
        booster = lgb.train(params, train, num_boost_round=rounds, valid_sets=[lgb.Dataset(xs, label=ys, weight=ws, reference=train, categorical_feature=cats)],
                            valid_names=["early_stop"], callbacks=callbacks)
        best = int(booster.best_iteration or booster.current_iteration())
        return Model(kind, booster, best), {k: {"rmse": v["rmse"][-5:]} for k, v in history.items()}
    from catboost import CatBoostRegressor, Pool
    model = CatBoostRegressor(iterations=rounds, **params)
    model.fit(Pool(catboost_frame(x), y, cat_features=cats, weight=w), verbose=100,
              eval_set=Pool(catboost_frame(xs), ys, cat_features=cats, weight=ws), early_stopping_rounds=patience, use_best_model=True)
    best = int(model.get_best_iteration() + 1) if model.get_best_iteration() is not None else int(model.tree_count_)
    history = {"early_stop": {"rmse": list(map(float, model.get_evals_result().get("validation", {}).get("RMSE", [])[-5:]))}}
    return Model(kind, model, best), history


def rmse(label: np.ndarray, values: np.ndarray, rows: np.ndarray) -> float | None:
    return float(np.sqrt(np.mean((label[rows] - values[rows]) ** 2))) if rows.any() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, default=Path("configs/adsb_unm_design_v1.json"))
    parser.add_argument("--run", required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--output-root", type=Path, help="Default: the declaration's output_root")
    parser.add_argument("--cache", type=Path, help="ADS-B table cache (default: <output root>/../features, shared by a worker's runs)")
    parser.add_argument("--smoke-rows", type=int, help="Local smoke only: first N fit rows of 2025-09 and N October rows, 30-round cap")
    args = parser.parse_args()
    started = time.time()
    decl_path = ROOT / args.declaration
    decl = read_json(decl_path)
    if decl["audit_used"]:
        raise ValueError("Declaration does not respect its own contract.")
    if args.run not in decl["runs"]:
        raise SystemExit(f"Unknown run {args.run}.")
    spec, common = decl["runs"][args.run], decl["learner_common"]
    smoke = args.smoke_rows is not None
    if not smoke and not decl.get("declared_at_utc"):
        raise SystemExit("The declaration is still a draft (no declared_at_utc); only --smoke-rows runs are allowed.")
    output_root = ROOT / (args.output_root or decl["output_root"])
    if smoke and output_root.resolve() == (ROOT / decl["output_root"]).resolve():
        raise SystemExit("Smoke runs go to a scratch --output-root.")
    out = output_root / args.run / "design"
    if (out / "run.json").exists():
        raise ValueError(f"Preserve the existing run: {out}")
    if sha256_file(ROOT / common["settings"]) != common["settings_sha256"]:
        raise ValueError("Learner settings changed.")
    settings = dict(read_json(ROOT / common["settings"])[spec["kind"]])
    params = dict(settings["parameters"])
    seed = int(spec["seed"])
    if spec["kind"] == "lightgbm":
        params.update(num_threads=args.threads, seed=seed, feature_fraction_seed=seed, bagging_seed=seed, data_random_seed=seed, metric="rmse")
    else:
        params.update(thread_count=args.threads, random_seed=seed)
    params.update(spec.get("parameters", {}))
    rounds = int(spec.get("rounds", settings["rounds"]))
    patience = int(common["patience"])
    if smoke:
        rounds = min(rounds, SMOKE_ROUNDS)

    matches_root = ROOT / common["matches_dir"]
    stands = pd.read_parquet(matches_root / "stands.parquet")
    base = ROOT / read_json(ROOT / "data/timeline/latest.json")["bundle"]
    manifest, parent, original, catalog = load_bundle(ROOT, base)
    assignment = pd.read_parquet(parent / "splits" / "seasonal.parquet")
    month_of = catalog.set_index(ID)[TIME].dt.strftime("%Y-%m")
    if smoke:
        fit_ids = assignment.loc[assignment["partition"].eq("fit") & assignment[ID].map(month_of).eq(SMOKE_FIT_MONTH).to_numpy(), ID].sort_values().head(args.smoke_rows)
        stop_ids = assignment.loc[assignment["partition"].eq("early_stop"), ID].sort_values().head(args.smoke_rows)
        assignment = assignment[assignment[ID].isin(set(fit_ids) | set(stop_ids))]
    roles = {"fit": None, "early_stop": None}
    months = sorted(set(month_of.reindex(assignment.loc[assignment["partition"].isin(list(roles)), ID]).to_numpy()))
    if set(months) & {"2025-01", "2025-07", "2025-11", "2025-12"}:
        raise ValueError(f"A scoring or reserved month entered the design fold: {months}")
    cache = ROOT / args.cache if args.cache else output_root.parent / "features"
    extra_root = ROOT / common["extra_features"]
    tables, extra_hashes = {}, {}
    for month in months:
        table = adsb_table([month], source_file(month), matches_root, stands, cache)
        extra_file = extra_root / f"features_{common['extra_tag']}_{month}.parquet"
        extra = pd.read_parquet(extra_file)
        tables[source_file(month)] = table.merge(extra, on=ID, how="left", validate="one_to_one")
        extra_hashes[extra_file.name] = sha256_file(extra_file)
    extra_names = [c for c in extra.columns if c != ID]
    names = [*BUNDLE_NAMES, *ADSB_NAMES, *extra_names]
    parts, truths = {}, {}
    for role in roles:
        features, truth = load_partition(base, parent, catalog, assignment, role, BUNDLE_NAMES)
        if not smoke and identity_hash(features[ID]) != original["folds"]["seasonal"]["identity_sha256"][role]:
            raise ValueError("Protected split populations changed.")
        parts[role] = attach(features, tables, catalog)
        truths[role] = aligned_truth(parts[role], truth)[TARGET].to_numpy(dtype=float)
    excluded = list(common["excluded_airports"])
    weight = float(spec["unmatched_weight"])
    fit_keep, fit_w = design_masks(parts["fit"], spec["scope"], weight, excluded)
    stop_keep, stop_w = design_masks(parts["early_stop"], spec["scope"], weight, excluded)
    x, vocab = encode(parts["fit"][fit_keep].reset_index(drop=True), names)
    xs, _ = encode(parts["early_stop"][stop_keep].reset_index(drop=True), names, vocab)
    model, history = fit_weighted(spec["kind"], params, x, truths["fit"][fit_keep], fit_w, rounds, xs, truths["early_stop"][stop_keep], stop_w, patience)
    october = parts["early_stop"]
    xo, _ = encode(october, names, vocab)
    prediction = model.predict(xo)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / ("model.txt" if spec["kind"] == "lightgbm" else "model.cbm"))
    pd.DataFrame({ID: october[ID].to_numpy(), "adsb_full": prediction}).to_parquet(out / "stop_predictions.parquet", index=False, compression="zstd")
    label = truths["early_stop"]
    non_lirf = october["airport"].ne("LIRF").to_numpy()
    unmatched = october["adsb_matched"].to_numpy(dtype=float) != 1
    airports = october["airport"].to_numpy()
    fit_unmatched = parts["fit"]["adsb_matched"].to_numpy(dtype=float)[fit_keep] != 1
    record = {"created_at_utc": utc_now(), "fold": "design", "run": args.run, "kind": spec["kind"], "seed": seed, "scope": spec["scope"],
              "unmatched_weight": weight, "control_member": spec["control_member"], "config_sha256": sha256_file(decl_path),
              "declared_at_utc": decl.get("declared_at_utc"), "smoke_limit_rows": args.smoke_rows, "best_iteration": model.best, "rounds_cap": rounds,
              "rows": {r: int(len(p)) for r, p in parts.items()}, "fitted_rows": int(fit_keep.sum()), "fitted_unmatched_rows": int(fit_unmatched.sum()),
              "unmatched_weight_share": float(np.sum(fit_w[fit_unmatched]) / np.sum(fit_w)) if fit_w is not None else float(fit_unmatched.mean()),
              "early_stop_rows": int(stop_keep.sum()), "months": months,
              "october_rmse": {"non_lirf": rmse(label, prediction, non_lirf), "matched": rmse(label, prediction, non_lirf & ~unmatched),
                               "unmatched": rmse(label, prediction, non_lirf & unmatched),
                               "unmatched_by_airport": {a: rmse(label, prediction, non_lirf & unmatched & (airports == a)) for a in sorted(set(airports[non_lirf]))}},
              "early_stop_curve_tail": history, "feature_importance_top30": dict(sorted(zip(model.feature_names(), model.importance()), key=lambda kv: -kv[1])[:30]),
              "inputs": {"bundle_id": manifest["bundle_id"], "stands_sha256": sha256_file(matches_root / "stands.parquet"),
                         "matches_sha256": {m: sha256_file(matches_root / f"matches2_{m}.parquet") for m in months}, "extra_feature_tables_sha256": extra_hashes,
                         "code": code_receipt(), "parameters": params, "features": len(names), "extra_features": extra_names, "excluded_airports": excluded,
                         "patience": patience, "threads": args.threads},
              "audit_used": False, "seconds": round(time.time() - started, 1), "environment": environment(ROOT)}
    write_json(out / "run.json", record)
    print({k: record[k] for k in ["run", "best_iteration", "fitted_rows", "fitted_unmatched_rows", "unmatched_weight_share", "october_rmse", "seconds"]}, flush=True)


if __name__ == "__main__":
    main()
