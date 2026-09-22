"""Declared development of adsb_v25: the residual stage on a twelve-month leave-one-month-out (sample weights by month distance, a longer
rounds cap, a CatBoost residual and two-member means), everything else exactly the v22 recipe.

Per held-out month (--held-out 2025-01 .. 2025-12) every declared residual variant is fitted on the in-scope rows of the other eleven
months (cross-fit v9 everywhere, the provenance that ships) and predicts the held-out month. Two readings are recorded: (1) the residual
stage's own out-of-sample error on its scope (matched non-LIRF rows: RMSE of v9 + r against the label), available on all twelve months;
(2) on the four recipe months (January / July / November / December, the months with out-of-sample committee tables) every declared
candidate is built with the frozen blend arithmetic and scored on ALL rows against the incumbent's development prediction (the v22
recipe, `runs/adsb_v24/dev/predictions_<month>.parquet`, reproduced exactly). `--select` combines the twelve month reports under the
pre-registered rule. Feature tables are the label-free caches of adsb_v24. Nothing is uploaded.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import environment, object_hash, read_json, sha256_file, utc_now, write_json
from prc2026.features import ID, TARGET, TIME
from run_adsb_v11 import candidate_values, check_parents, paired_days, prepare_matrix, rmse, scope_mask
from run_adsb_v24 import ALL_MONTHS, adsb_features, extra_columns, load_frame, month_table

RECIPE_MONTHS = ["2025-01", "2025-07", "2025-11", "2025-12"]


def month_distance(months: np.ndarray, target: str) -> np.ndarray:
    """Circular calendar-month distance (0..6) between each row's month ('YYYY-MM') and the target month."""
    a = np.array([int(m[5:7]) for m in months])
    b = int(target[5:7])
    d = np.abs(a - b)
    return np.minimum(d, 12 - d)


def month_weights(months: np.ndarray, target: str, scheme: dict | None) -> np.ndarray:
    """Sample weights from the declared distance scheme ({"1": w1, "2": w2, "default": w}); ones without a scheme."""
    if not scheme:
        return np.ones(len(months))
    d = month_distance(months, target)
    return np.array([float(scheme.get(str(k), scheme["default"])) for k in d])


def catboost_matrix(x: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    """CatBoost input: the LightGBM matrix with categorical columns as strings ('NA' for missing)."""
    out = x.copy()
    for name in categorical:
        out[name] = out[name].astype(object).where(out[name].notna(), "NA").astype(str)
    return out


def fit_variant(train: pd.DataFrame, config: dict, variant: dict, target_month: str, threads: int) -> dict:
    """Fit one residual variant on the in-scope rows of `train` for the target month: rounds from the 5-fold day-grouped inner split
    (median best iteration, patience 200, the variant's cap), sample weights from the variant's distance scheme on the training folds only
    (validation unweighted), then the final fit at those rounds on all rows."""
    spec = config["residual_model"]
    scope = config["residual_scope"]
    rows = train[scope_mask(train, scope)].reset_index(drop=True)
    x, categories = prepare_matrix(rows, config, feature_set=variant["feature_set"])
    y = (rows[TARGET] - rows["v9"]).to_numpy(dtype=float)
    dates = rows[TIME].dt.strftime("%Y-%m-%d").to_numpy()
    weights = month_weights(rows["month"].to_numpy(), target_month, variant.get("weights"))
    lgb_weight = (lambda idx: weights[idx]) if variant.get("weights") else (lambda idx: None)  # the unweighted control takes exactly the v24 code path
    cap, patience = int(variant.get("rounds_cap", spec["rounds_cap"])), int(spec["patience"])
    kind = variant.get("kind", "lightgbm")
    best = []
    t0 = time.time()
    if kind == "lightgbm":
        params = {**spec["params"], **variant.get("params", {}), "num_threads": threads}
        for tr, va in GroupKFold(5).split(x, y, groups=dates):
            booster = lgb.train(params, lgb.Dataset(x.iloc[tr], y[tr], weight=lgb_weight(tr), categorical_feature=spec["categorical"]), cap,
                                valid_sets=[lgb.Dataset(x.iloc[va], y[va], categorical_feature=spec["categorical"])], callbacks=[lgb.early_stopping(patience, verbose=False)])
            best.append(int(booster.best_iteration))
        rounds = max(50, int(np.rint(np.median(best))))
        model = lgb.train(params, lgb.Dataset(x, y, weight=lgb_weight(slice(None)), categorical_feature=spec["categorical"]), rounds)
    elif kind == "catboost":
        from catboost import CatBoostRegressor, Pool
        params = {**spec["catboost_params"], **variant.get("params", {}), "thread_count": threads}
        xc = catboost_matrix(x, spec["categorical"])
        for tr, va in GroupKFold(5).split(xc, y, groups=dates):
            model = CatBoostRegressor(iterations=cap, od_type="Iter", od_wait=patience, verbose=False, allow_writing_files=False, **params)
            model.fit(Pool(xc.iloc[tr], y[tr], weight=weights[tr], cat_features=spec["categorical"]), eval_set=Pool(xc.iloc[va], y[va], cat_features=spec["categorical"]), use_best_model=False)
            best.append(int(model.get_best_iteration()) + 1)
        rounds = max(50, int(np.rint(np.median(best))))
        model = CatBoostRegressor(iterations=rounds, verbose=False, allow_writing_files=False, **params)
        model.fit(Pool(xc, y, weight=weights, cat_features=spec["categorical"]))
    else:
        raise ValueError(f"Unknown residual kind {kind}.")
    return {"kind": kind, "model": model, "categories": categories, "rounds": rounds, "inner_best_iterations": best, "rows": int(len(rows)), "scope": scope,
            "feature_set": variant["feature_set"], "target_month": target_month, "weights": variant.get("weights"), "weight_sum": float(weights.sum()),
            "weight_by_distance": {str(int(k)): float(v) for k, v in zip(*np.unique(month_distance(rows["month"].to_numpy(), target_month), return_counts=True))} if variant.get("weights") else None,
            "seconds": round(time.time() - t0, 1)}


def predict_variant(model: dict, table: pd.DataFrame, config: dict) -> np.ndarray:
    """Residual for every row of the table; zero outside the scope."""
    out = np.zeros(len(table))
    inside = scope_mask(table, model["scope"])
    if inside.any():
        x, _ = prepare_matrix(table[inside], config, model["categories"], model["feature_set"])
        if model["kind"] == "lightgbm":
            out[inside] = model["model"].predict(x, num_iteration=model["rounds"])
        else:
            out[inside] = model["model"].predict(catboost_matrix(x, config["residual_model"]["categorical"]))
    return out


def candidate_residual(item: dict, residuals: dict[str, np.ndarray]) -> np.ndarray:
    members = item["residual"] if isinstance(item["residual"], list) else [item["residual"]]
    return np.mean([residuals[m] for m in members], axis=0)


def run_month(config: dict, held: str, cache: Path, out_dir: Path, threads: int, smoke: bool = False) -> None:
    """One held-out month. `smoke` (code-path test only, never a development result): 30,000 subsampled training rows, every cap at 60 rounds."""
    started = time.time()
    matches_root = ROOT / config["matches_dirs"]["v11"]
    scope = config["residual_scope"]
    tables = {m: month_table(load_frame(config, m, "cross_fit"), adsb_features(m, matches_root, cache), m) for m in ALL_MONTHS}
    test = tables[held].copy()
    floor = float(test["floor"].iloc[0])
    label = test[TARGET].to_numpy(dtype=float)
    v9 = test["v9"].to_numpy(dtype=float)
    inside = scope_mask(test, scope)
    train = pd.concat([tables[m] for m in ALL_MONTHS if m != held], ignore_index=True)
    if smoke:
        train = train.sample(n=30000, random_state=config["seed"]).reset_index(drop=True)
        config = {**config, "residual_model": {**config["residual_model"], "rounds_cap": 60}, "residual_variants": {n: {**v, "rounds_cap": 60} for n, v in config["residual_variants"].items()}}
    report = {"held_out": held, "smoke": smoke, "rows": int(len(test)), "rows_in_scope": int(inside.sum()), "floor_seconds": floor, "threads": threads,
              "v9_cross_fit_rmse": rmse(label, v9), "v9_scope_rmse": rmse(label[inside], v9[inside]), "variants": {}, "candidates": {}}
    residuals = {}
    for name, variant in config["residual_variants"].items():
        model = fit_variant(train, config, variant, held, threads)
        residuals[name] = predict_variant(model, test, config)
        report["variants"][name] = {k: model[k] for k in ["kind", "rounds", "inner_best_iterations", "rows", "feature_set", "target_month", "weights", "weight_sum", "weight_by_distance", "seconds"]}
        report["variants"][name]["scope_rmse"] = rmse(label[inside], (v9 + residuals[name])[inside])
        print(held, name, {k: report["variants"][name][k] for k in ["rows", "rounds", "seconds", "scope_rmse"]}, flush=True)
    frame = pd.DataFrame({ID: test[ID].to_numpy()})
    for name in residuals:
        frame[f"residual:{name}"] = residuals[name]
    if held in RECIPE_MONTHS:
        for name, values in extra_columns(config, held, test[ID].to_numpy()).items():
            test[name] = values
        incumbent = pd.read_parquet(ROOT / config["development_baseline"]["predictions"][held]).set_index(ID)[config["development_baseline"]["column"]].reindex(test[ID]).to_numpy(dtype=float)
        expected = config["development_baseline"]["recorded_rmse"][held]
        if abs(rmse(label, incumbent) - expected) > 1e-6:
            raise ValueError(f"The incumbent's development prediction does not reproduce on {held}: {rmse(label, incumbent)} vs {expected}.")
        days = test[TIME].dt.strftime("%Y-%m-%d").to_numpy()
        airports = test["airport"].to_numpy()
        report["baseline_rmse"] = rmse(label, incumbent)
        frame["baseline"] = incumbent
        spec = {"weight": 1.0, "scope": scope, "blend_with": config["blend_with"], "extra_blends": config["extra_blends"]}
        for name, item in config["candidates"].items():
            values, _ = candidate_values(test, v9, candidate_residual(item, residuals), spec, floor)
            score = rmse(label, values)
            entry = {"rmse": score, "ratio_to_baseline": score / report["baseline_rmse"], "rows_differing_from_baseline": int((np.abs(values - incumbent) > 1e-9).sum()),
                     "max_abs_difference_from_baseline": float(np.max(np.abs(values - incumbent))), "paired_days_vs_baseline": paired_days(days, label, values, incumbent, config["seed"]), "airports": {}}
            for airport in sorted(set(airports)):
                k = airports == airport
                entry["airports"][airport] = {"rows": int(k.sum()), "baseline": rmse(label[k], incumbent[k]), "candidate": rmse(label[k], values[k]), "change_pct_vs_baseline": 100 * (rmse(label[k], values[k]) / rmse(label[k], incumbent[k]) - 1)}
            ordinary = label <= 21600
            entry["ordinary_rows"] = {"rows": int(ordinary.sum()), "baseline": rmse(label[ordinary], incumbent[ordinary]), "candidate": rmse(label[ordinary], values[ordinary])}
            report["candidates"][name] = entry
            frame[name] = values
            print(held, name, {"rmse": round(score, 4), "vs_baseline_pct": round(100 * (entry["ratio_to_baseline"] - 1), 4), "interval": [round(v, 2) for v in entry["paired_days_vs_baseline"]["rmse_difference_95pct_interval_seconds"]]}, flush=True)
    report["inputs"] = {"match_tables": {m: sha256_file(matches_root / f"matches2_{m}.parquet") for m in ALL_MONTHS}, "stands": sha256_file(matches_root / "stands.parquet"),
                        "scratch": {m: sha256_file(ROOT / config["cross_fit"]["scratch"] / f"{m}.parquet") for m in ALL_MONTHS},
                        "feature_cache": {m: sha256_file(cache / f"{matches_root.name}_{m}.parquet") for m in ALL_MONTHS}}
    report["seconds"] = round(time.time() - started, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_dir / f"predictions_{held}.parquet", index=False)
    write_json(out_dir / f"month_{held}.json", report)


def select(config: dict, out_dir: Path, report_path: Path, cache: Path) -> None:
    reports = {m: read_json(out_dir / f"month_{m}.json") for m in ALL_MONTHS}
    control = config["control_variant"]
    variants = {}
    for name in config["residual_variants"]:
        months = {m: reports[m]["variants"][name] for m in ALL_MONTHS}
        pooled = float(np.sqrt(sum(months[m]["scope_rmse"] ** 2 * reports[m]["rows_in_scope"] for m in ALL_MONTHS) / sum(reports[m]["rows_in_scope"] for m in ALL_MONTHS)))
        variants[name] = {"months": months, "pooled_scope_rmse": pooled, "scope_rmse_by_month": {m: months[m]["scope_rmse"] for m in ALL_MONTHS}}
    for name in variants:
        variants[name]["pooled_scope_ratio_to_control"] = variants[name]["pooled_scope_rmse"] / variants[control]["pooled_scope_rmse"]
        variants[name]["months_better_than_control"] = int(sum(variants[name]["scope_rmse_by_month"][m] < variants[control]["scope_rmse_by_month"][m] for m in ALL_MONTHS))
    # Pooled scope error of each candidate's residual (means included) from the saved residual columns.
    candidate_scope = {}
    for name, item in config["candidates"].items():
        sse, n = 0.0, 0
        for m in ALL_MONTHS:
            table = load_frame(config, m, "cross_fit")
            preds = pd.read_parquet(out_dir / f"predictions_{m}.parquet").set_index(ID).reindex(table[ID])
            feats = pd.read_parquet(cache / f"matches_v11_{m}.parquet", columns=[ID, "adsb_matched"]).set_index(ID).reindex(table[ID])
            inside = feats["adsb_matched"].eq(1).to_numpy() & table["airport"].ne("LIRF").to_numpy()
            members = item["residual"] if isinstance(item["residual"], list) else [item["residual"]]
            r = np.mean([preds[f"residual:{v}"].to_numpy(dtype=float) for v in members], axis=0)
            err = (table[TARGET].to_numpy(dtype=float) - table["v9"].to_numpy(dtype=float) - r)[inside]
            sse += float(np.square(err).sum())
            n += int(inside.sum())
        candidate_scope[name] = float(np.sqrt(sse / n))
    candidates = {}
    for name in config["candidates"]:
        entry = {"months": {m: reports[m]["candidates"][name] for m in RECIPE_MONTHS}, "pooled_scope_rmse": candidate_scope[name],
                 "pooled_scope_ratio_to_control": candidate_scope[name] / variants[control]["pooled_scope_rmse"]}
        ratios = [entry["months"][m]["ratio_to_baseline"] for m in RECIPE_MONTHS]
        entry["mean_ratio"] = float(np.mean(ratios))
        touched = [m for m in RECIPE_MONTHS if entry["months"][m]["rows_differing_from_baseline"] > 0]
        entry["interval_condition_months"] = touched
        entry["eligible"] = bool(all(r <= 1.0 for r in ratios) and entry["mean_ratio"] <= 1 - config["minimum_mean_improvement_fraction"]
                                 and all(entry["months"][m]["paired_days_vs_baseline"]["rmse_difference_95pct_interval_seconds"][1] < 0 for m in touched)
                                 and entry["pooled_scope_ratio_to_control"] < 1.0 and name not in config["check_candidates"])
        candidates[name] = entry
    eligible = [n for n, e in candidates.items() if e["eligible"]]
    selected = min(eligible, key=lambda n: (candidates[n]["mean_ratio"], n)) if eligible else None
    report = {"created_at_utc": utc_now(), "configuration": config, "config_sha256": object_hash(config), "audit_used": False, "rule": config["selection"],
              "months": {m: {k: reports[m].get(k) for k in ["rows", "rows_in_scope", "baseline_rmse", "v9_cross_fit_rmse", "v9_scope_rmse", "floor_seconds", "threads"]} for m in ALL_MONTHS},
              "residual_variants": variants, "candidates": candidates, "selected_candidate": selected,
              "refit_rounds_by_variant": {name: int(np.rint(np.median([variants[name]["months"][m]["rounds"] for m in ALL_MONTHS]))) for name in variants},
              "model_code": {p: sha256_file(ROOT / p) for p in ["tools/run_adsb_v25.py", "tools/run_adsb_v24.py", "tools/run_adsb_v11.py", "src/prc2026_adsb/features.py"]},
              "environment": environment(ROOT)}
    for path in [out_dir / "selection.json", report_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, report)
    print({"selected": selected, "candidates": {n: {"mean_pct": round(100 * (e["mean_ratio"] - 1), 4), "scope_pct": round(100 * (e["pooled_scope_ratio_to_control"] - 1), 4), "eligible": e["eligible"],
                                                        "months": {m: round(100 * (e["months"][m]["ratio_to_baseline"] - 1), 4) for m in RECIPE_MONTHS}} for n, e in candidates.items()}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/adsb_v25_v1.json"))
    parser.add_argument("--held-out", choices=ALL_MONTHS)
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/adsb_v25/dev"))
    parser.add_argument("--report", type=Path, default=Path("docs/adsb_v25/development_results.json"))
    parser.add_argument("--cache", type=Path, default=Path("runs/adsb_v24/features"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--smoke", action="store_true", help="code-path test on subsampled rows and 60 rounds; refuses to write under runs/")
    args = parser.parse_args()
    if args.smoke and "runs" in Path(args.output).parts:
        raise ValueError("A smoke test must write outside runs/.")
    config = read_json(ROOT / args.config)
    if config["audit_used"]:
        raise ValueError("Declaration does not respect its own contract.")
    check_parents(config)
    if args.select:
        if (ROOT / args.output / "selection.json").exists():
            raise ValueError("Preserve the original selection.")
        select(config, ROOT / args.output, ROOT / args.report, ROOT / args.cache)
    elif args.held_out:
        if (ROOT / args.output / f"month_{args.held_out}.json").exists():
            raise ValueError(f"Preserve the existing month report for {args.held_out}.")
        run_month(config, args.held_out, ROOT / args.cache, ROOT / args.output, args.threads, args.smoke)
    else:
        raise SystemExit("--held-out MONTH or --select")


if __name__ == "__main__":
    main()
