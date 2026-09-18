"""Declared leave-one-month-out development of the ADS-B surface residual on top of v9.

For each of the three scoring months with out-of-sample v9 predictions, fits the
declared residual LightGBM on the in-scope rows of the other two months (rounds
from a day-grouped inner split of those two months), applies it to the held-out
month's in-scope rows, and scores each candidate against the declared baseline
(v9, or the incumbent's development predictions when the declaration names one)
on all rows of the month. A candidate's scope is "matched" (rows with a matched
surface segment) or "context" (matched rows plus rows with a matched runway
neighbour within 30 minutes); the neighbour features are merged when the
declaration asks for them. December is never read. Nothing is uploaded.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import environment, object_hash, read_json, sha256_file, utc_now, write_json
from prc2026.features import ID, PHASE, TARGET, TIME
from prc2026_adsb.features import DEPARTURE_COLUMNS, departure_features, neighbour_features

MATCHES = ROOT / "data/external/adsblol/matches"
SCOPES = {"matched": lambda t: t["adsb_matched"].eq(1),
          "context": lambda t: t["adsb_matched"].eq(1) | t["nb_runway_30m_count"].fillna(0).gt(0),
          "matched_no_lirf": lambda t: t["adsb_matched"].eq(1) & t["airport"].ne("LIRF"),
          "all_no_lirf": lambda t: t["airport"].ne("LIRF"),
          "lirf_nonpop": lambda t: t["airport"].eq("LIRF") & ~t["population"].astype(bool),  # LIRF rows outside the v9 mixture's fill population
          # ... with a matched surface segment and an ordinary NM clock and schedule interval (both under six hours): the label-target learners
          # cannot follow a day-long clock (a one-day LIRF row with proxy 87,181 s was predicted at 2,186 s), and without ADS-B they lose to v9
          "lirf_nonpop_matched": lambda t: (t["airport"].eq("LIRF") & ~t["population"].astype(bool) & t["adsb_matched"].eq(1)
                                            & t["proxy_s"].abs().lt(21600) & t["schedule_interval_s"].abs().lt(21600))}


def model_code_receipt(root: Path) -> dict:
    paths = sorted((root / "src/prc2026_adsb").glob("*.py")) + [root / "tools/run_adsb_v11.py", root / "tools/build_adsb_features.py", root / "tools/adsb_extract_day.py"]
    hashes = {path.relative_to(root).as_posix(): sha256_file(path) for path in paths}
    return {"files": hashes, "sha256": object_hash(hashes)}


def check_parents(config: dict) -> None:
    for name, parent in config["parents"].items():
        if sha256_file(ROOT / parent["path"]) != parent["sha256"]:
            raise ValueError(f"Parent artifact changed: {name}")


def matches_dir(config: dict) -> Path:
    return ROOT / config.get("matches_dir", MATCHES.relative_to(ROOT).as_posix())


def extra_tables(config: dict, stage: str) -> list[pd.DataFrame]:
    """Declared ID-aligned extra feature tables (e.g. the full-data learner's out-of-sample predictions), one column each."""
    frames = []
    for name, spec in config.get("extra_feature_tables", {}).items():
        paths = spec["development"] if stage == "development" else [spec["ranking"]]
        frame = pd.concat([pd.read_parquet(ROOT / path, columns=[ID, spec["column"]]) for path in paths], ignore_index=True)
        if frame[ID].duplicated().any():
            raise ValueError(f"Extra feature table {name} has duplicate identities.")
        frames.append(frame.rename(columns={spec["column"]: name}))
    return frames


def attach_extra(table: pd.DataFrame, frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge the extra tables one-to-one and add the declared derived differences to v9."""
    for frame in frames:
        name = [c for c in frame.columns if c != ID][0]
        if not table[ID].isin(frame[ID]).all():
            raise ValueError(f"Extra feature table {name} does not cover every row.")
        table = table.merge(frame, on=ID, how="left", validate="one_to_one")  # a NaN value (e.g. an NM-residual learner without the clock) leaves the row unblended
        table[f"{name}_minus_v9"] = table[name] - table["v9"]
    return table


def month_table(month: str, fold_frame: pd.DataFrame, stands: pd.DataFrame, matches_root: Path = MATCHES, neighbours: bool = False,
                extra: list[pd.DataFrame] | None = None) -> pd.DataFrame:
    """Fold rows of the month (with v9 and label) joined to ADS-B features; label-free features only."""
    y, m = int(month[:4]), int(month[5:7])
    raw = pd.read_parquet(ROOT / "data/raw" / f"training_{y}-{m:02}-01_{y + (m == 12)}-{(m % 12) + 1:02}-01.parquet", columns=DEPARTURE_COLUMNS + ["RUNWAY_mvt"])
    dep = raw[raw[PHASE].eq("DEP")].reset_index(drop=True)
    matches = pd.read_parquet(matches_root / f"matches2_{month}.parquet")
    feats = departure_features(dep, matches, stands).merge(dep[[ID, "STAND_mvt", "RUNWAY_mvt"]], on=ID, validate="one_to_one")
    if neighbours:
        feats = feats.merge(neighbour_features(dep, matches), on=ID, validate="one_to_one")
    part = fold_frame[fold_frame[TIME].dt.strftime("%Y-%m").eq(month)]
    table = part.merge(feats, on=ID, how="left", validate="one_to_one")
    if table["adsb_matched"].isna().any():
        raise ValueError(f"ADS-B feature table does not cover the fold rows of {month}.")
    table["month"] = month
    table["hour"] = table[TIME].dt.hour + table[TIME].dt.minute / 60
    table = table.rename(columns={"STAND_mvt": "stand", "RUNWAY_mvt": "runway"})
    return attach_extra(table, extra or [])


def scope_mask(table: pd.DataFrame, scope: str) -> np.ndarray:
    return SCOPES[scope](table).to_numpy()


def feature_list(config: dict, feature_set: str | None = None) -> list[str]:
    """The declared feature list, or one of the declared named feature sets."""
    spec = config["residual_model"]
    return list(spec["feature_sets"][feature_set]) if feature_set else list(spec["features"])


def prepare_matrix(table: pd.DataFrame, config: dict, categories: dict | None = None, feature_set: str | None = None) -> tuple[pd.DataFrame, dict]:
    spec = config["residual_model"]
    x = table[feature_list(config, feature_set)].copy()
    categories = categories or {}
    for name in spec["categorical"]:
        levels = categories.get(name)
        if levels is None:
            levels = sorted(x[name].dropna().astype(str).unique().tolist())
            categories[name] = levels
        x[name] = pd.Categorical(x[name].astype(str).where(x[name].notna(), None), categories=levels)
    return x, categories


def seed_params(params: dict, seed: int | None) -> dict:
    """The declared parameters with every LightGBM seed set to `seed` (None keeps the declared seed)."""
    out = dict(params)
    if seed is not None:
        out.update(seed=seed, bagging_seed=seed, feature_fraction_seed=seed, data_random_seed=seed)
    return out


def fit_boosters(spec: dict, x: pd.DataFrame, y: np.ndarray, rounds: int) -> list:
    """One booster per declared seed (`residual_model.seeds`; without the key, the single declared parameter set), all at the same rounds."""
    return [lgb.train(seed_params(spec["params"], seed), lgb.Dataset(x, y, categorical_feature=spec["categorical"]), rounds) for seed in (spec.get("seeds") or [None])]


def fit_residual(train: pd.DataFrame, config: dict, scope: str = "matched", feature_set: str | None = None) -> dict:
    """Fit on in-scope rows; rounds from a 5-fold day-grouped inner split (median best iteration) with the declared parameters; a seed bag if declared."""
    spec = config["residual_model"]
    rows = train[scope_mask(train, scope)].reset_index(drop=True)
    x, categories = prepare_matrix(rows, config, feature_set=feature_set)
    y = (rows[TARGET] - rows["v9"]).to_numpy(dtype=float)
    dates = rows[TIME].dt.strftime("%Y-%m-%d").to_numpy()
    params = dict(spec["params"])
    best = []
    for tr, va in GroupKFold(5).split(x, y, groups=dates):
        booster = lgb.train(params, lgb.Dataset(x.iloc[tr], y[tr], categorical_feature=spec["categorical"]), 4000,
                            valid_sets=[lgb.Dataset(x.iloc[va], y[va], categorical_feature=spec["categorical"])], callbacks=[lgb.early_stopping(200, verbose=False)])
        best.append(int(booster.best_iteration))
    rounds = max(50, int(np.rint(np.median(best))))
    boosters = fit_boosters(spec, x, y, rounds)
    return {"boosters": boosters, "booster": boosters[0], "categories": categories, "rounds": rounds, "inner_best_iterations": best, "rows": int(len(rows)), "scope": scope, "feature_set": feature_set,
            "seeds": list(spec.get("seeds") or [])}


def predict_residual(model: dict, table: pd.DataFrame, config: dict, scope: str = "matched") -> np.ndarray:
    """Residual for every row of the table (mean over the seed bag); zero outside the scope."""
    out = np.zeros(len(table))
    inside = scope_mask(table, scope)
    if inside.any():
        x, _ = prepare_matrix(table[inside], config, model["categories"], model.get("feature_set"))
        out[inside] = np.mean([booster.predict(x, num_iteration=model["rounds"]) for booster in model.get("boosters", [model["booster"]])], axis=0)
    return out


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def candidate_values(table: pd.DataFrame, v9: np.ndarray, residual: np.ndarray, item: dict, floor: float) -> tuple[np.ndarray, np.ndarray]:
    """v9 + weight * residual on the residual scope, then the declared blend(s) with extra tables on their blend scopes, in order; floored.

    `blend_with` is the primary blend; `extra_blends` (optional list of {name, scope, weight}) follow it, each a convex combination of the
    running value with its table on its own scope. Returns the values and the mask of rows that may differ from v9 (residual scope or a
    blend scope with a finite blend value).
    """
    inside = scope_mask(table, item["scope"])
    values = np.where(inside, v9 + item["weight"] * residual, v9)
    touched = inside.copy()
    for blend in [b for b in [item.get("blend_with"), *(item.get("extra_blends") or [])] if b]:
        partner = table[blend["name"]].to_numpy(dtype=float)
        where = scope_mask(table, blend.get("scope", item["scope"])) & np.isfinite(partner)
        values = np.where(where, (1 - blend["weight"]) * values + blend["weight"] * partner, values)
        touched |= where
    return np.maximum(values, floor), touched


def paired_days(days: np.ndarray, label: np.ndarray, candidate: np.ndarray, reference: np.ndarray, seed: int) -> dict:
    """Descriptive paired-day bootstrap of RMSE(candidate) - RMSE(reference) on the same rows."""
    frame = pd.DataFrame({"day": days, "c": (candidate - label) ** 2, "r": (reference - label) ** 2})
    grouped = frame.groupby("day").agg(c=("c", "sum"), r=("r", "sum"), rows=("c", "size"))
    indices = np.random.default_rng(seed).integers(0, len(grouped), size=(2000, len(grouped)))
    rows = grouped["rows"].to_numpy()[indices].sum(axis=1)
    delta = np.sqrt(grouped["c"].to_numpy()[indices].sum(axis=1) / rows) - np.sqrt(grouped["r"].to_numpy()[indices].sum(axis=1) / rows)
    return {"days": int(len(grouped)), "replicates": 2000, "rmse_difference_95pct_interval_seconds": list(map(float, np.quantile(delta, [0.025, 0.975]))),
            "days_with_lower_squared_error": int((grouped["c"] < grouped["r"]).sum()), "interpretation": "descriptive on reused development data"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/adsb_v11_v1.json"))
    parser.add_argument("--scratch", type=Path, default=Path("scratch"))
    parser.add_argument("--output", type=Path, default=Path("runs/adsb_v11/selection.json"))
    parser.add_argument("--report", type=Path, default=Path("docs/adsb_v11/development_results.json"))
    args = parser.parse_args()
    config = read_json(ROOT / args.config)
    for path in [ROOT / args.output, ROOT / args.report]:
        if path.exists():
            raise ValueError(f"Preserve the original selection: {path}")
    if config["audit_used"] or "2025-12" in config["development_months"] or "2025-12" in config["refit_months"]:
        raise ValueError("Declaration does not respect the reserved period.")
    check_parents(config)
    spec = config["residual_model"]
    neighbours = bool(spec.get("neighbour_features", False))
    stands = pd.read_parquet(matches_dir(config) / "stands.parquet")
    seasonal = pd.read_parquet(ROOT / args.scratch / "seasonal.parquet")
    forward = pd.read_parquet(ROOT / args.scratch / "forward.parquet")
    floor = float(min(seasonal["floor"].iloc[0], forward["floor"].iloc[0]))
    extra = extra_tables(config, "development")
    tables = {}
    for month in config["development_months"]:
        fold = seasonal if month in ["2025-01", "2025-07"] else forward
        tables[month] = month_table(month, fold, stands, matches_dir(config), neighbours, extra)
    baseline_spec = config.get("development_baseline")
    baseline_residual = None
    if baseline_spec:
        parent = config["parents"][baseline_spec.get("parent", "v11_development_residuals")]["path"]
        baseline_residual = pd.read_parquet(ROOT / parent).set_index(ID)[baseline_spec.get("column", "residual")]
    candidates = {name: {"weight": item["weight"], "scope": item.get("scope", "matched"), "feature_set": item.get("features"), "blend_with": item.get("blend_with"),
                         "extra_blends": item.get("extra_blends", []), "months": {}}
                  for name, item in config["candidates"].items()}
    variants = sorted({(item["scope"], item["feature_set"]) for item in candidates.values()}, key=lambda v: (v[0], v[1] or ""))
    key_of = lambda scope, feature_set: scope + (f"|{feature_set}" if feature_set else "")
    months_report, residuals = {}, {m: {} for m in config["development_months"]}
    for held in config["development_months"]:
        train = pd.concat([tables[m] for m in config["development_months"] if m != held], ignore_index=True)
        test = tables[held]
        label, v9 = test[TARGET].to_numpy(dtype=float), test["v9"].to_numpy(dtype=float)
        days = test[TIME].dt.strftime("%Y-%m-%d").to_numpy()
        if baseline_residual is not None:
            baseline_values = baseline_residual.reindex(test[ID]).fillna(0.0).to_numpy(dtype=float)
            if baseline_spec.get("blend_with"):
                # The baseline is itself a blended export (e.g. v14): residual on its scope, then the declared blend, then the floor.
                baseline, _ = candidate_values(test, v9, baseline_values, {"weight": baseline_spec["weight"], "scope": baseline_spec["scope"], "blend_with": baseline_spec["blend_with"],
                                                                           "extra_blends": baseline_spec.get("extra_blends", [])}, floor)
            else:
                baseline = np.maximum(v9 + baseline_spec["weight"] * baseline_values, floor)
        else:
            baseline = v9
        matched = test["adsb_matched"].eq(1).to_numpy()
        months_report[held] = {"rows": int(len(test)), "matched_rows": int(matched.sum()), "matched_share": float(matched.mean()),
                               "gate_share_of_matched": float(test.loc[matched, "adsb_gate"].mean()) if matched.any() else None,
                               "v9_rmse": rmse(label, v9), "baseline_rmse": rmse(label, baseline), "baseline": baseline_spec["name"] if baseline_spec else "v9",
                               "scopes": {}, "airports": {}}
        models = {}
        for scope, feature_set in variants:
            key = key_of(scope, feature_set)
            models[key] = fit_residual(train, config, scope, feature_set)
            residuals[held][key] = predict_residual(models[key], test, config, scope)
            inside = scope_mask(test, scope)
            months_report[held]["scopes"][key] = {"scope": scope, "feature_set": feature_set, "rows_in_scope": int(inside.sum()), "share_in_scope": float(inside.mean()),
                                                  "training_rows": models[key]["rows"], "rounds": models[key]["rounds"], "inner_best_iterations": models[key]["inner_best_iterations"],
                                                  "seeds": models[key]["seeds"]}
        for item in candidates.values():
            residual = residuals[held][key_of(item["scope"], item["feature_set"])]
            values, _ = candidate_values(test, v9, residual, item, floor)
            score = rmse(label, values)
            entry = {"rmse": score, "ratio_to_v9": score / months_report[held]["v9_rmse"], "ratio_to_baseline": score / months_report[held]["baseline_rmse"],
                     "paired_days_vs_baseline": paired_days(days, label, values, baseline, config["seed"]), "airports": {}}
            for airport in sorted(test["airport"].unique()):
                k = test["airport"].eq(airport).to_numpy()
                entry["airports"][airport] = {"rows": int(k.sum()), "matched_share": float(matched[k].mean()), "in_scope_share": float(scope_mask(test, item["scope"])[k].mean()),
                                              "v9": rmse(label[k], v9[k]), "baseline": rmse(label[k], baseline[k]), "candidate": rmse(label[k], values[k]),
                                              "change_pct_vs_baseline": 100 * (rmse(label[k], values[k]) / rmse(label[k], baseline[k]) - 1)}
            item["months"][held] = entry
    margin = config.get("minimum_improvement_fraction_each_month")
    mean_margin = config.get("minimum_mean_improvement_fraction")
    for item in candidates.values():
        ratios = [item["months"][m]["ratio_to_baseline"] for m in config["development_months"]]
        item["mean_ratio"] = float(np.mean(ratios))
        item["mean_ratio_to_v9"] = float(np.mean([item["months"][m]["ratio_to_v9"] for m in config["development_months"]]))
        if margin is not None:
            item["eligible"] = bool(all(r <= 1 - margin for r in ratios))
        else:  # never worse on any month, and at least the declared mean improvement
            item["eligible"] = bool(all(r <= 1.0 for r in ratios) and item["mean_ratio"] <= 1 - mean_margin)
        if config.get("require_paired_day_intervals_below_zero"):  # every month's paired-day interval of the RMSE difference must lie below zero
            item["eligible"] = bool(item["eligible"] and all(item["months"][m]["paired_days_vs_baseline"]["rmse_difference_95pct_interval_seconds"][1] < 0
                                                             for m in config["development_months"]))
    eligible = [n for n, item in candidates.items() if item["eligible"]]
    selected = min(eligible, key=lambda n: (candidates[n]["mean_ratio"], n)) if eligible else None
    selected_key = key_of(candidates[selected]["scope"], candidates[selected]["feature_set"]) if selected else key_of(*variants[0])
    report = {"created_at_utc": utc_now(), "policy": config["prediction_postprocess"], "configuration": config, "config_sha256": object_hash(config),
              "model_code": model_code_receipt(ROOT), "audit_used": False, "native_fit_count": 3 * 6 * len(variants), "floor_seconds": floor,
              "stands": int(len(stands)), "months": months_report, "candidates": candidates, "selected_candidate": selected,
              "refit_rounds": int(np.rint(np.median([months_report[m]["scopes"][selected_key]["rounds"] for m in config["development_months"]]))),
              "refit_rounds_by_variant": {key_of(*v): int(np.rint(np.median([months_report[m]["scopes"][key_of(*v)]["rounds"] for m in config["development_months"]]))) for v in variants},
              "rule": config["selection"], "environment": environment(ROOT)}
    for path in [ROOT / args.output, ROOT / args.report]:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, report)
    frames = []
    for m in config["development_months"]:
        frame = pd.DataFrame({ID: tables[m][ID].to_numpy()})
        for scope, feature_set in variants:
            key = key_of(scope, feature_set)
            frame["residual" if key == "matched" and len(variants) == 1 else f"residual_{key}"] = residuals[m][key]
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_parquet(ROOT / args.output.parent / "development_residuals.parquet", index=False)
    print({"selected": selected, "refit_rounds": report["refit_rounds"], "baseline": months_report[config["development_months"][0]]["baseline"],
           "months": {m: {"v9": round(r["v9_rmse"], 3), "baseline": round(r["baseline_rmse"], 3), "matched": round(r["matched_share"], 3),
                          "variants": {s: (v["rows_in_scope"], v["rounds"]) for s, v in r["scopes"].items()}} for m, r in months_report.items()},
           "candidates": {n: {m: (round(e["rmse"], 3), round(100 * (e["ratio_to_baseline"] - 1), 2), [round(v, 2) for v in e["paired_days_vs_baseline"]["rmse_difference_95pct_interval_seconds"]])
                              for m, e in item["months"].items()} for n, item in candidates.items()}})
    best = selected or next(iter(candidates))
    for m in config["development_months"]:
        print(m, {a: f"{r['baseline']:.1f}->{r['candidate']:.1f} ({r['change_pct_vs_baseline']:+.1f}%, scope {r['in_scope_share']:.2f})" for a, r in candidates[best]["months"][m]["airports"].items()})


if __name__ == "__main__":
    main()
