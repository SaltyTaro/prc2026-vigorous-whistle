"""Declared development of adsb_v24: the residual stage on twelve months of cross-fitted v9, v7 from more months, and two scans.

Per held-out development month (--held-out 2025-01 | 2025-07 | 2025-11 | 2025-12), every declared residual variant is fitted on its
declared training rows (the other months; `run_adsb_v11.fit_residual`, unchanged parameters and inner rounds rule) and every declared
candidate is scored on all rows of the month against the incumbent's development prediction (v21 recipe: the recorded v12 development
residual on January / July / November, the v19 export's residual booster on December; both reproduce the recorded scores exactly).
`--select` combines the four month reports under the pre-registered rule. ADS-B feature tables are cached per month and match directory
(label-free). December is a development month since 19 September (docs/december/authorization_20260919.json). Nothing is uploaded.
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
from prc2026.artifacts import environment, object_hash, read_json, sha256_file, utc_now, write_json
from prc2026.features import ID, PHASE, TARGET, TIME
from prc2026_adsb.features import DEPARTURE_COLUMNS, departure_features, neighbour_features
from prc2026_december.scoring import ensemble_prediction, leaf_members
from run_adsb_v11 import SCOPES, candidate_values, check_parents, fit_residual, paired_days, predict_residual, rmse

ALL_MONTHS = [f"2025-{m:02}" for m in range(1, 13)]
FRAME_OF = {"2025-01": "seasonal", "2025-07": "seasonal", "2025-11": "forward", "2025-12": "december"}
# LIRF non-population rows WITHOUT a matched surface segment and with ordinary clocks: the complement of lirf_nonpop_matched inside the
# ordinary-clock LIRF non-population set (registered at runtime; tools/run_adsb_v11.py is a frozen parent and is not edited).
SCOPES["lirf_nonpop_unmatched"] = lambda t: (t["airport"].eq("LIRF") & ~t["population"].astype(bool) & t["adsb_matched"].ne(1)
                                             & t["proxy_s"].abs().lt(21600) & t["schedule_interval_s"].abs().lt(21600))


def raw_file(month: str) -> str:
    y, m = int(month[:4]), int(month[5:7])
    return f"training_{y}-{m:02}-01_{y + (m == 12)}-{(m % 12) + 1:02}-01.parquet"


def adsb_features(month: str, matches_root: Path, cache: Path) -> pd.DataFrame:
    """The residual stage's label-free ADS-B columns (departure + neighbour features, stand, runway) for every departure of the month."""
    path = cache / f"{matches_root.name}_{month}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    raw = pd.read_parquet(ROOT / "data/raw" / raw_file(month), columns=DEPARTURE_COLUMNS + ["RUNWAY_mvt"])
    dep = raw[raw[PHASE].eq("DEP")].reset_index(drop=True)
    matches = pd.read_parquet(matches_root / f"matches2_{month}.parquet")
    stands = pd.read_parquet(matches_root / "stands.parquet")
    feats = departure_features(dep, matches, stands).merge(dep[[ID, "STAND_mvt", "RUNWAY_mvt"]], on=ID, validate="one_to_one")
    feats = feats.merge(neighbour_features(dep, matches), on=ID, validate="one_to_one")
    cache.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(path, index=False)
    return feats


def month_table(frame: pd.DataFrame, feats: pd.DataFrame, month: str) -> pd.DataFrame:
    """run_adsb_v11.month_table on a prepared frame (v9 provenance chosen by the caller) and cached features."""
    table = frame.merge(feats, on=ID, how="left", validate="one_to_one")
    if table["adsb_matched"].isna().any():
        raise ValueError(f"ADS-B feature table does not cover the rows of {month}.")
    table["month"] = month
    table["hour"] = table[TIME].dt.hour + table[TIME].dt.minute / 60
    day = table[TIME].dt.dayofyear.to_numpy(dtype=float)
    table["season_cos"] = np.cos(2 * np.pi * (day - 1) / 365.25)  # 1 at New Year, -1 at the start of July: the ranking months sit at the extremes
    return table.rename(columns={"STAND_mvt": "stand", "RUNWAY_mvt": "runway"})


def extra_columns(config: dict, month: str, ids: np.ndarray) -> dict[str, np.ndarray]:
    """Development values of the declared blend tables: fold predictions for 2025-01/07/11, the cached January-November refit leaves for December."""
    out = {}
    for name, spec in config["extra_feature_tables"].items():
        if month == "2025-12":
            members = leaf_members(ROOT, spec["run_dir"])
            cache = {}
            for leaf in set(members):
                cache[leaf] = pd.read_parquet(ROOT / "runs/december/learners" / (leaf.replace("/", "__") + ".parquet")).set_index(ID)["adsb_full"].reindex(ids).to_numpy(dtype=float)
            out[name] = ensemble_prediction(members, cache)
        else:
            path = ROOT / spec["development"][0 if FRAME_OF[month] == "seasonal" else 1]
            out[name] = pd.read_parquet(path).set_index(ID)[spec["column"]].reindex(ids).to_numpy(dtype=float)
    return out


def load_frame(config: dict, month: str, source: str) -> pd.DataFrame:
    """The month's scratch frame with the requested v9 provenance: 'fold' (development months only) or 'cross_fit' (scratch/v24)."""
    if source == "fold":
        frame = pd.read_parquet(ROOT / "scratch" / f"{FRAME_OF[month]}.parquet")
        frame = frame[frame[TIME].dt.strftime("%Y-%m").eq(month)].reset_index(drop=True)
        return frame[[ID, TIME, "airport", TARGET, "v9", "floor", "proxy_s", "schedule_interval_s", "clock_available", "population"]]
    frame = pd.read_parquet(ROOT / config["cross_fit"]["scratch"] / f"{month}.parquet")
    return frame[[ID, TIME, "airport", TARGET, "v9", "floor", "proxy_s", "schedule_interval_s", "clock_available", "population"]]


def baseline_values(config: dict, month: str, table: pd.DataFrame, floor: float) -> np.ndarray:
    """The incumbent's development prediction on the month (fold v9 frame): recorded residual + the v21 recipe's blends."""
    spec = config["development_baseline"]
    v9 = table["v9"].to_numpy(dtype=float)
    if month == "2025-12":
        meta = read_json(ROOT / config["parents"]["v19_residual_record"]["path"])
        booster = lgb.Booster(model_file=str(ROOT / config["parents"]["v19_residual_model"]["path"]))
        model = {"boosters": [booster], "booster": booster, "categories": meta["categories"], "rounds": meta["rounds"], "feature_set": meta["feature_set"], "scope": meta["scope"]}
        residual = predict_residual(model, table, config, spec["scope"])
    else:
        residual = pd.read_parquet(ROOT / config["parents"]["v12_development_residuals"]["path"]).set_index(ID)[spec["column"]].reindex(table[ID]).fillna(0.0).to_numpy(dtype=float)
    values, _ = candidate_values(table, v9, residual, {"weight": spec["weight"], "scope": spec["scope"], "blend_with": spec["blend_with"], "extra_blends": spec["extra_blends"]}, floor)
    expected = spec["recorded_rmse"][month]
    score = rmse(table[TARGET].to_numpy(dtype=float), values)
    if abs(score - expected) > 1e-6:
        raise ValueError(f"The incumbent's development prediction does not reproduce on {month}: {score} vs {expected}.")
    return values


def run_month(config: dict, held: str, cache: Path, out_dir: Path) -> None:
    started = time.time()
    dirs = {key: ROOT / path for key, path in config["matches_dirs"].items()}
    scope = config["residual_scope"]
    frames = {(m, src): load_frame(config, m, src) for m in ALL_MONTHS for src in (["fold", "cross_fit"] if m in FRAME_OF else ["cross_fit"])}
    floor = float(frames[(held, "fold")]["floor"].iloc[0])
    feats = {}
    tables = {}

    def table(m: str, src: str, dir_key: str) -> pd.DataFrame:
        key = (m, src, dir_key)
        if key not in tables:
            if (m, dir_key) not in feats:
                feats[(m, dir_key)] = adsb_features(m, dirs[dir_key], cache)
            tables[key] = month_table(frames[(m, src)], feats[(m, dir_key)], m)
        return tables[key]

    # Held-out month tables: fold v9 (baseline and 'fold' candidates) and cross-fit v9, on each match directory, with the blend tables attached.
    extras = extra_columns(config, held, frames[(held, "fold")][ID].to_numpy())
    tests = {}
    for src in ["fold", "cross_fit"]:
        for dir_key in dirs:
            t = table(held, src, dir_key).copy()
            if not np.array_equal(t[ID].to_numpy(), frames[(held, "fold")][ID].to_numpy()):
                raise ValueError("Held-out row order differs between v9 sources.")
            for name, values in extras.items():
                t[name] = values
            tests[(src, dir_key)] = t
    label = tests[("fold", "v11")][TARGET].to_numpy(dtype=float)
    days = tests[("fold", "v11")][TIME].dt.strftime("%Y-%m-%d").to_numpy()
    baseline = baseline_values(config, held, tests[("fold", "v11")], floor)
    report = {"held_out": held, "rows": int(len(label)), "floor_seconds": floor, "baseline_rmse": rmse(label, baseline), "v9_fold_rmse": rmse(label, tests[("fold", "v11")]["v9"].to_numpy(dtype=float)),
              "v9_cross_fit_rmse": rmse(label, tests[("cross_fit", "v11")]["v9"].to_numpy(dtype=float)), "variants": {}, "candidates": {}}
    residuals = {}
    for name, variant in config["residual_variants"].items():
        months = [m for m in (ALL_MONTHS if variant["months"] == "all" else config["residual_months_short"]) if m != held]
        parts = []
        for m in months:
            src = variant["v9"] if m in FRAME_OF else "cross_fit"
            parts.append(table(m, src, variant["matches_dir"]))
        train = pd.concat(parts, ignore_index=True)
        t0 = time.time()
        model = fit_residual(train, config, scope, variant["feature_set"])
        residuals[name] = {src: predict_residual(model, tests[(src, variant["matches_dir"])], config, scope) for src in ["fold", "cross_fit"]}
        report["variants"][name] = {"training_months": months, "training_rows": model["rows"], "rounds": model["rounds"], "inner_best_iterations": model["inner_best_iterations"],
                                    "feature_set": variant["feature_set"], "matches_dir": variant["matches_dir"], "v9_provenance": variant["v9"], "seconds": round(time.time() - t0, 1)}
        print(held, name, {k: report["variants"][name][k] for k in ["training_rows", "rounds", "seconds"]}, flush=True)
    frame = pd.DataFrame({ID: tests[("fold", "v11")][ID].to_numpy(), "baseline": baseline})
    for name, item in config["candidates"].items():
        variant = config["residual_variants"][item["residual"]]
        test = tests[(item["v9"], variant["matches_dir"])]
        v9 = test["v9"].to_numpy(dtype=float)
        residual = residuals[item["residual"]][item["v9"]]
        spec = {"weight": 1.0, "scope": scope, "blend_with": config["blend_with"], "extra_blends": [*config["extra_blends"], *item.get("extra_blends", [])]}
        values, _ = candidate_values(test, v9, residual, spec, floor)
        score = rmse(label, values)
        entry = {"rmse": score, "ratio_to_baseline": score / report["baseline_rmse"], "rows_differing_from_baseline": int((np.abs(values - baseline) > 1e-9).sum()),
                 "paired_days_vs_baseline": paired_days(days, label, values, baseline, config["seed"]), "airports": {}}
        airports = test["airport"].to_numpy()
        for airport in sorted(set(airports)):
            k = airports == airport
            entry["airports"][airport] = {"rows": int(k.sum()), "baseline": rmse(label[k], baseline[k]), "candidate": rmse(label[k], values[k]),
                                          "change_pct_vs_baseline": 100 * (rmse(label[k], values[k]) / rmse(label[k], baseline[k]) - 1)}
        ordinary = label <= 21600
        entry["ordinary_rows"] = {"rows": int(ordinary.sum()), "baseline": rmse(label[ordinary], baseline[ordinary]), "candidate": rmse(label[ordinary], values[ordinary])}
        report["candidates"][name] = entry
        frame[name] = values
        print(held, name, {"rmse": round(score, 4), "vs_baseline_pct": round(100 * (entry["ratio_to_baseline"] - 1), 4), "interval": [round(v, 2) for v in entry["paired_days_vs_baseline"]["rmse_difference_95pct_interval_seconds"]]}, flush=True)
    report["inputs"] = {"match_tables": {f"{key}/{m}": sha256_file(dirs[key] / f"matches2_{m}.parquet") for (m, key) in feats}, "stands": {key: sha256_file(dirs[key] / "stands.parquet") for key in dirs},
                        "scratch": {f"{m}/{src}": (sha256_file(ROOT / config["cross_fit"]["scratch"] / f"{m}.parquet") if src == "cross_fit" else sha256_file(ROOT / "scratch" / f"{FRAME_OF[m]}.parquet")) for (m, src) in frames}}
    report["seconds"] = round(time.time() - started, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_dir / f"predictions_{held}.parquet", index=False)
    write_json(out_dir / f"month_{held}.json", report)


def select(config: dict, out_dir: Path, report_path: Path) -> None:
    months = config["development_months"]
    reports = {m: read_json(out_dir / f"month_{m}.json") for m in months}
    candidates = {}
    for name in config["candidates"]:
        entry = {"months": {m: reports[m]["candidates"][name] for m in months}}
        ratios = [entry["months"][m]["ratio_to_baseline"] for m in months]
        entry["mean_ratio"] = float(np.mean(ratios))
        touched = [m for m in months if entry["months"][m]["rows_differing_from_baseline"] > 0]
        entry["interval_condition_months"] = touched
        entry["eligible"] = bool(all(r <= 1.0 for r in ratios) and entry["mean_ratio"] <= 1 - config["minimum_mean_improvement_fraction"]
                                 and all(entry["months"][m]["paired_days_vs_baseline"]["rmse_difference_95pct_interval_seconds"][1] < 0 for m in touched))
        candidates[name] = entry
    eligible = [n for n, e in candidates.items() if e["eligible"]]
    selected = min(eligible, key=lambda n: (candidates[n]["mean_ratio"], n)) if eligible else None
    variants = {name: {m: reports[m]["variants"][name] for m in months} for name in config["residual_variants"]}
    report = {"created_at_utc": utc_now(), "configuration": config, "config_sha256": object_hash(config), "audit_used": False, "rule": config["selection"],
              "months": {m: {k: reports[m][k] for k in ["rows", "baseline_rmse", "v9_fold_rmse", "v9_cross_fit_rmse", "floor_seconds"]} for m in months},
              "residual_variants": variants, "candidates": candidates, "selected_candidate": selected,
              "refit_rounds_by_variant": {name: int(np.rint(np.median([variants[name][m]["rounds"] for m in months]))) for name in variants},
              "model_code": {p: sha256_file(ROOT / p) for p in ["tools/run_adsb_v24.py", "tools/run_adsb_v11.py", "tools/build_v24_scratch.py", "tools/run_v7_dec.py", "src/prc2026_adsb/features.py"]},
              "environment": environment(ROOT)}
    for path in [out_dir / "selection.json", report_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, report)
    print({"selected": selected, "candidates": {n: {"mean_pct": round(100 * (e["mean_ratio"] - 1), 4), "eligible": e["eligible"],
                                                        "months": {m: round(100 * (e["months"][m]["ratio_to_baseline"] - 1), 4) for m in months}} for n, e in candidates.items()}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/adsb_v24_v1.json"))
    parser.add_argument("--held-out", choices=["2025-01", "2025-07", "2025-11", "2025-12"])
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/adsb_v24/dev"))
    parser.add_argument("--report", type=Path, default=Path("docs/adsb_v24/development_results.json"))
    parser.add_argument("--cache", type=Path, default=Path("runs/adsb_v24/features"))
    parser.add_argument("--threads", type=int, default=None, help="override the residual thread count (fitting only; the declared value reproduces exactly)")
    args = parser.parse_args()
    config = read_json(ROOT / args.config)
    if config["audit_used"]:
        raise ValueError("Declaration does not respect its own contract.")
    check_parents(config)
    if args.threads:
        config["residual_model"]["params"]["num_threads"] = args.threads
    if args.select:
        if (ROOT / args.output / "selection.json").exists():
            raise ValueError("Preserve the original selection.")
        select(config, ROOT / args.output, ROOT / args.report)
    elif args.held_out:
        if (ROOT / args.output / f"month_{args.held_out}.json").exists():
            raise ValueError(f"Preserve the existing month report for {args.held_out}.")
        run_month(config, args.held_out, ROOT / args.cache, ROOT / args.output)
    else:
        raise SystemExit("--held-out MONTH or --select")


if __name__ == "__main__":
    main()
