"""Declared development of adsb_v26: further CatBoost residual members (learning rate 0.10, a second seed) on the twelve-month leave-one-month-out,
reusing the adsb_v25 residual columns of the members that ship in v23 (adj2, cat) without refitting them; everything else exactly the v23 recipe.

Stages, so that (month x new variant) jobs can run on separate workers:
  --held-out M --variants V [V ...] --tag T   fit the listed NEW variants for month M (eleven other months, cross-fit v9, tools/run_adsb_v25.fit_variant
                                              unchanged) and save runs/adsb_v26/dev/partial_M_T.{parquet,json} (residual columns + variant records)
  --score --held-out M                        combine the inherited residual columns (runs/adsb_v25/dev/predictions_M.parquet) with every partial of M,
                                              record each variant's residual-scope RMSE and, on the recipe months, score every candidate on ALL rows against
                                              the incumbent's development prediction (v25's mean_adj2_cat, reproduced exactly); write month_M.json
  --select                                    combine the twelve month reports under the pre-registered rule
Nothing is uploaded.
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import environment, object_hash, read_json, sha256_file, utc_now, write_json
from prc2026.features import ID, TARGET, TIME
from run_adsb_v11 import candidate_values, check_parents, paired_days, rmse, scope_mask
from run_adsb_v24 import ALL_MONTHS, adsb_features, extra_columns, load_frame, month_table
from run_adsb_v25 import RECIPE_MONTHS, fit_variant, predict_variant


def members_of(item: dict) -> list[str]:
    return item["residual"] if isinstance(item["residual"], list) else [item["residual"]]


def fit_month(config: dict, held: str, variants: list[str], tag: str, cache: Path, out_dir: Path, threads: int, smoke: bool = False) -> None:
    """One (month, variants) job. `smoke` (code-path test only, never a development result): 30,000 subsampled training rows, every cap at 60 rounds."""
    started = time.time()
    config_sha256 = object_hash(config)
    matches_root = ROOT / config["matches_dirs"]["v11"]
    tables = {m: month_table(load_frame(config, m, "cross_fit"), adsb_features(m, matches_root, cache), m) for m in ALL_MONTHS}
    test = tables[held]
    label = test[TARGET].to_numpy(dtype=float)
    v9 = test["v9"].to_numpy(dtype=float)
    inside = scope_mask(test, config["residual_scope"])
    train = pd.concat([tables[m] for m in ALL_MONTHS if m != held], ignore_index=True)
    if smoke:
        train = train.sample(n=30000, random_state=config["seed"]).reset_index(drop=True)
        config = {**config, "residual_model": {**config["residual_model"], "rounds_cap": 60}, "new_variants": {n: {**v, "rounds_cap": 60} for n, v in config["new_variants"].items()}}
    frame = pd.DataFrame({ID: test[ID].to_numpy()})
    records = {}
    for name in variants:
        if name not in config["new_variants"]:
            raise ValueError(f"{name} is not a declared new variant.")
        model = fit_variant(train, config, config["new_variants"][name], held, threads)
        residual = predict_variant(model, test, config)
        frame[f"residual:{name}"] = residual
        records[name] = {k: model[k] for k in ["kind", "rounds", "inner_best_iterations", "rows", "feature_set", "target_month", "weights", "weight_sum", "weight_by_distance", "seconds"]}
        records[name]["scope_rmse"] = rmse(label[inside], (v9 + residual)[inside])
        print(held, name, {k: records[name][k] for k in ["rows", "rounds", "seconds", "scope_rmse"]}, flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_dir / f"partial_{held}_{tag}.parquet", index=False)
    write_json(out_dir / f"partial_{held}_{tag}.json", {"held_out": held, "tag": tag, "variants": records, "threads": threads, "rows": int(len(test)), "rows_in_scope": int(inside.sum()), "smoke": smoke,
                                                        "config_sha256": config_sha256, "seconds": round(time.time() - started, 1)})


def score_month(config: dict, held: str, cache: Path, out_dir: Path) -> None:
    started = time.time()
    matches_root = ROOT / config["matches_dirs"]["v11"]
    test = month_table(load_frame(config, held, "cross_fit"), adsb_features(held, matches_root, cache), held)
    ids = test[ID].to_numpy()
    floor = float(test["floor"].iloc[0])
    label = test[TARGET].to_numpy(dtype=float)
    v9 = test["v9"].to_numpy(dtype=float)
    inside = scope_mask(test, config["residual_scope"])
    # Inherited residual columns (hash-bound v25 development predictions) and the new variants' partials.
    inherited = pd.read_parquet(ROOT / config["inherited_variants"]["predictions"][held]).set_index(ID).reindex(ids)
    residuals, records = {}, {}
    for name in config["inherited_variants"]["names"]:
        residuals[name] = inherited[f"residual:{name}"].to_numpy(dtype=float)
        if not np.isfinite(residuals[name]).all():
            raise ValueError(f"Inherited residual {name} does not cover {held}.")
        records[name] = {"source": "inherited", "scope_rmse": rmse(label[inside], (v9 + residuals[name])[inside])}
    for path in sorted(glob.glob(str(out_dir / f"partial_{held}_*.json"))):
        part = read_json(Path(path))
        if part["config_sha256"] != object_hash(config):
            raise ValueError(f"Partial {path} belongs to another declaration.")
        values = pd.read_parquet(Path(path).with_suffix(".parquet")).set_index(ID).reindex(ids)
        for name, record in part["variants"].items():
            if name in residuals:
                raise ValueError(f"Variant {name} appears twice for {held}.")
            residuals[name] = values[f"residual:{name}"].to_numpy(dtype=float)
            records[name] = {"source": Path(path).name, **record}
    missing = [n for n in config["new_variants"] if n not in residuals]
    if missing:
        raise ValueError(f"New variants without a partial for {held}: {missing}")
    report = {"held_out": held, "rows": int(len(test)), "rows_in_scope": int(inside.sum()), "floor_seconds": floor, "v9_cross_fit_rmse": rmse(label, v9),
              "v9_scope_rmse": rmse(label[inside], v9[inside]), "variants": records, "candidates": {}}
    frame = pd.DataFrame({ID: ids})
    for name in residuals:
        frame[f"residual:{name}"] = residuals[name]
    if held in RECIPE_MONTHS:
        for name, values in extra_columns(config, held, ids).items():
            test[name] = values
        incumbent = pd.read_parquet(ROOT / config["development_baseline"]["predictions"][held]).set_index(ID)[config["development_baseline"]["column"]].reindex(ids).to_numpy(dtype=float)
        expected = config["development_baseline"]["recorded_rmse"][held]
        if abs(rmse(label, incumbent) - expected) > 1e-6:
            raise ValueError(f"The incumbent's development prediction does not reproduce on {held}: {rmse(label, incumbent)} vs {expected}.")
        days = test[TIME].dt.strftime("%Y-%m-%d").to_numpy()
        airports = test["airport"].to_numpy()
        report["baseline_rmse"] = rmse(label, incumbent)
        frame["baseline"] = incumbent
        spec = {"weight": 1.0, "scope": config["residual_scope"], "blend_with": config["blend_with"], "extra_blends": config["extra_blends"]}
        for name, item in config["candidates"].items():
            residual = np.mean([residuals[m] for m in members_of(item)], axis=0)
            values, _ = candidate_values(test, v9, residual, spec, floor)
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
    report["inputs"] = {"match_table": sha256_file(matches_root / f"matches2_{held}.parquet"), "scratch": sha256_file(ROOT / config["cross_fit"]["scratch"] / f"{held}.parquet"),
                        "feature_cache": sha256_file(cache / f"{matches_root.name}_{held}.parquet"), "inherited_predictions": sha256_file(ROOT / config["inherited_variants"]["predictions"][held]),
                        "partials": {Path(p).name: sha256_file(Path(p)) for p in sorted(glob.glob(str(out_dir / f"partial_{held}_*")))}}
    report["seconds"] = round(time.time() - started, 1)
    frame.to_parquet(out_dir / f"predictions_{held}.parquet", index=False)
    write_json(out_dir / f"month_{held}.json", report)


def select(config: dict, out_dir: Path, report_path: Path, cache: Path) -> None:
    reports = {m: read_json(out_dir / f"month_{m}.json") for m in ALL_MONTHS}
    names = [*config["inherited_variants"]["names"], *config["new_variants"]]
    weights_n = {m: reports[m]["rows_in_scope"] for m in ALL_MONTHS}
    variants = {}
    for name in names:
        months = {m: reports[m]["variants"][name] for m in ALL_MONTHS}
        pooled = float(np.sqrt(sum(months[m]["scope_rmse"] ** 2 * weights_n[m] for m in ALL_MONTHS) / sum(weights_n.values())))
        variants[name] = {"months": months, "pooled_scope_rmse": pooled, "scope_rmse_by_month": {m: months[m]["scope_rmse"] for m in ALL_MONTHS}}
    # Pooled scope error of every candidate's residual mean, and of the control (the incumbent's residual mean).
    tables = {m: load_frame(config, m, "cross_fit") for m in ALL_MONTHS}
    preds = {m: pd.read_parquet(out_dir / f"predictions_{m}.parquet").set_index(ID).reindex(tables[m][ID]) for m in ALL_MONTHS}
    masks = {m: pd.read_parquet(cache / f"matches_v11_{m}.parquet", columns=[ID, "adsb_matched"]).set_index(ID).reindex(tables[m][ID])["adsb_matched"].eq(1).to_numpy() & tables[m]["airport"].ne("LIRF").to_numpy() for m in ALL_MONTHS}

    def pooled_scope(members: list[str]) -> float:
        sse, n = 0.0, 0
        for m in ALL_MONTHS:
            r = np.mean([preds[m][f"residual:{v}"].to_numpy(dtype=float) for v in members], axis=0)
            err = (tables[m][TARGET].to_numpy(dtype=float) - tables[m]["v9"].to_numpy(dtype=float) - r)[masks[m]]
            sse += float(np.square(err).sum())
            n += int(masks[m].sum())
        return float(np.sqrt(sse / n))

    control = pooled_scope(config["control_members"])
    candidates = {}
    for name, item in config["candidates"].items():
        entry = {"members": members_of(item), "months": {m: reports[m]["candidates"][name] for m in RECIPE_MONTHS}, "pooled_scope_rmse": pooled_scope(members_of(item))}
        entry["pooled_scope_ratio_to_control"] = entry["pooled_scope_rmse"] / control
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
              "months": {m: {k: reports[m].get(k) for k in ["rows", "rows_in_scope", "baseline_rmse", "v9_cross_fit_rmse", "v9_scope_rmse", "floor_seconds"]} for m in ALL_MONTHS},
              "control_pooled_scope_rmse": control, "residual_variants": variants, "candidates": candidates, "selected_candidate": selected,
              "refit_rounds_by_variant": {name: int(np.rint(np.median([variants[name]["months"][m]["rounds"] for m in ALL_MONTHS]))) for name in config["new_variants"]},
              "model_code": {p: sha256_file(ROOT / p) for p in ["tools/run_adsb_v26.py", "tools/run_adsb_v25.py", "tools/run_adsb_v24.py", "tools/run_adsb_v11.py", "src/prc2026_adsb/features.py"]},
              "environment": environment(ROOT)}
    for path in [out_dir / "selection.json", report_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, report)
    print({"selected": selected, "control_scope": round(control, 4), "candidates": {n: {"mean_pct": round(100 * (e["mean_ratio"] - 1), 4), "scope_pct": round(100 * (e["pooled_scope_ratio_to_control"] - 1), 4), "eligible": e["eligible"],
                                                                                        "months": {m: round(100 * (e["months"][m]["ratio_to_baseline"] - 1), 4) for m in RECIPE_MONTHS}} for n, e in candidates.items()}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/adsb_v26_v1.json"))
    parser.add_argument("--held-out", choices=ALL_MONTHS)
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/adsb_v26/dev"))
    parser.add_argument("--report", type=Path, default=Path("docs/adsb_v26/development_results.json"))
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
    out_dir = ROOT / args.output
    if args.select:
        if (out_dir / "selection.json").exists():
            raise ValueError("Preserve the original selection.")
        select(config, out_dir, ROOT / args.report, ROOT / args.cache)
    elif args.score and args.held_out:
        if (out_dir / f"month_{args.held_out}.json").exists():
            raise ValueError(f"Preserve the existing month report for {args.held_out}.")
        score_month(config, args.held_out, ROOT / args.cache, out_dir)
    elif args.held_out and args.variants and args.tag:
        if (out_dir / f"partial_{args.held_out}_{args.tag}.json").exists():
            raise ValueError(f"Preserve the existing partial {args.held_out}/{args.tag}.")
        fit_month(config, args.held_out, args.variants, args.tag, ROOT / args.cache, out_dir, args.threads, args.smoke)
    else:
        raise SystemExit("--held-out M --variants V.. --tag T | --score --held-out M | --select")


if __name__ == "__main__":
    main()
