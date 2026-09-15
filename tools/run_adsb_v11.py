"""Declared leave-one-month-out development of the ADS-B surface residual on top of v9.

For each of the three scoring months with out-of-sample v9 predictions, fits the
declared residual LightGBM on the matched rows of the other two months (rounds
from a day-grouped inner split of those two months), applies it to the held-out
month's matched rows, and scores each candidate weight against v9 on all rows of
the month. December is never read. Nothing is uploaded.
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
from prc2026_adsb.features import DEPARTURE_COLUMNS, departure_features

MATCHES = ROOT / "data/external/adsblol/matches"


def model_code_receipt(root: Path) -> dict:
    paths = sorted((root / "src/prc2026_adsb").glob("*.py")) + [root / "tools/run_adsb_v11.py", root / "tools/build_adsb_features.py", root / "tools/adsb_extract_day.py"]
    hashes = {path.relative_to(root).as_posix(): sha256_file(path) for path in paths}
    return {"files": hashes, "sha256": object_hash(hashes)}


def check_parents(config: dict) -> None:
    for name, parent in config["parents"].items():
        if sha256_file(ROOT / parent["path"]) != parent["sha256"]:
            raise ValueError(f"Parent artifact changed: {name}")


def month_table(month: str, fold_frame: pd.DataFrame, stands: pd.DataFrame) -> pd.DataFrame:
    """Fold rows of the month (with v9 and label) joined to ADS-B features; label-free features only."""
    y, m = int(month[:4]), int(month[5:7])
    raw = pd.read_parquet(ROOT / "data/raw" / f"training_{y}-{m:02}-01_{y + (m == 12)}-{(m % 12) + 1:02}-01.parquet", columns=DEPARTURE_COLUMNS + ["RUNWAY_mvt"])
    dep = raw[raw[PHASE].eq("DEP")].reset_index(drop=True)
    matches = pd.read_parquet(MATCHES / f"matches2_{month}.parquet")
    feats = departure_features(dep, matches, stands).merge(dep[[ID, "STAND_mvt", "RUNWAY_mvt"]], on=ID, validate="one_to_one")
    part = fold_frame[fold_frame[TIME].dt.strftime("%Y-%m").eq(month)]
    table = part.merge(feats, on=ID, how="left", validate="one_to_one")
    if table["adsb_matched"].isna().any():
        raise ValueError(f"ADS-B feature table does not cover the fold rows of {month}.")
    table["month"] = month
    table["hour"] = table[TIME].dt.hour + table[TIME].dt.minute / 60
    table = table.rename(columns={"STAND_mvt": "stand", "RUNWAY_mvt": "runway"})
    return table


def prepare_matrix(table: pd.DataFrame, config: dict, categories: dict | None = None) -> tuple[pd.DataFrame, dict]:
    spec = config["residual_model"]
    x = table[spec["features"]].copy()
    categories = categories or {}
    for name in spec["categorical"]:
        levels = categories.get(name)
        if levels is None:
            levels = sorted(x[name].dropna().astype(str).unique().tolist())
            categories[name] = levels
        x[name] = pd.Categorical(x[name].astype(str).where(x[name].notna(), None), categories=levels)
    return x, categories


def fit_residual(train: pd.DataFrame, config: dict) -> dict:
    """Fit on matched rows; rounds from a 5-fold day-grouped inner split (median best iteration)."""
    spec = config["residual_model"]
    matched = train[train["adsb_matched"].eq(1)].reset_index(drop=True)
    x, categories = prepare_matrix(matched, config)
    y = (matched[TARGET] - matched["v9"]).to_numpy(dtype=float)
    dates = matched[TIME].dt.strftime("%Y-%m-%d").to_numpy()
    params = dict(spec["params"])
    best = []
    for tr, va in GroupKFold(5).split(x, y, groups=dates):
        booster = lgb.train(params, lgb.Dataset(x.iloc[tr], y[tr], categorical_feature=spec["categorical"]), 4000,
                            valid_sets=[lgb.Dataset(x.iloc[va], y[va], categorical_feature=spec["categorical"])], callbacks=[lgb.early_stopping(200, verbose=False)])
        best.append(int(booster.best_iteration))
    rounds = max(50, int(np.rint(np.median(best))))
    booster = lgb.train(params, lgb.Dataset(x, y, categorical_feature=spec["categorical"]), rounds)
    return {"booster": booster, "categories": categories, "rounds": rounds, "inner_best_iterations": best, "rows": int(len(matched))}


def predict_residual(model: dict, table: pd.DataFrame, config: dict) -> np.ndarray:
    """Residual for every row of the table; zero where no ADS-B segment matched."""
    out = np.zeros(len(table))
    matched = table["adsb_matched"].eq(1).to_numpy()
    if matched.any():
        x, _ = prepare_matrix(table[matched], config, model["categories"])
        out[matched] = model["booster"].predict(x, num_iteration=model["rounds"])
    return out


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


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
    stands = pd.read_parquet(MATCHES / "stands.parquet")
    seasonal = pd.read_parquet(ROOT / args.scratch / "seasonal.parquet")
    forward = pd.read_parquet(ROOT / args.scratch / "forward.parquet")
    floor = float(min(seasonal["floor"].iloc[0], forward["floor"].iloc[0]))
    tables = {}
    for month in config["development_months"]:
        fold = seasonal if month in ["2025-01", "2025-07"] else forward
        tables[month] = month_table(month, fold, stands)
    candidates = {name: {"weight": spec["weight"], "months": {}} for name, spec in config["candidates"].items()}
    months_report, residuals = {}, {}
    for held in config["development_months"]:
        train = pd.concat([tables[m] for m in config["development_months"] if m != held], ignore_index=True)
        model = fit_residual(train, config)
        test = tables[held]
        residual = predict_residual(model, test, config)
        residuals[held] = residual
        label, v9 = test[TARGET].to_numpy(dtype=float), test["v9"].to_numpy(dtype=float)
        matched = test["adsb_matched"].eq(1).to_numpy()
        months_report[held] = {"rows": int(len(test)), "matched_rows": int(matched.sum()), "matched_share": float(matched.mean()),
                               "gate_share_of_matched": float(test.loc[matched, "adsb_gate"].mean()) if matched.any() else None,
                               "training_matched_rows": model["rows"], "rounds": model["rounds"], "inner_best_iterations": model["inner_best_iterations"],
                               "v9_rmse": rmse(label, v9), "airports": {}}
        for name, spec in config["candidates"].items():
            values = np.maximum(v9 + spec["weight"] * residual, floor)
            score = rmse(label, values)
            entry = {"rmse": score, "ratio_to_v9": score / months_report[held]["v9_rmse"], "airports": {}}
            for airport in sorted(test["airport"].unique()):
                k = test["airport"].eq(airport).to_numpy()
                entry["airports"][airport] = {"rows": int(k.sum()), "matched_share": float(matched[k].mean()), "v9": rmse(label[k], v9[k]), "candidate": rmse(label[k], values[k]),
                                              "change_pct": 100 * (rmse(label[k], values[k]) / rmse(label[k], v9[k]) - 1)}
            candidates[name]["months"][held] = entry
    margin = config["minimum_improvement_fraction_each_month"]
    for name, item in candidates.items():
        ratios = [item["months"][m]["ratio_to_v9"] for m in config["development_months"]]
        item["mean_ratio"] = float(np.mean(ratios))
        item["eligible"] = bool(all(r <= 1 - margin for r in ratios))
    eligible = [n for n, item in candidates.items() if item["eligible"]]
    selected = min(eligible, key=lambda n: (candidates[n]["mean_ratio"], n)) if eligible else None
    report = {"created_at_utc": utc_now(), "policy": config["prediction_postprocess"], "configuration": config, "config_sha256": object_hash(config),
              "model_code": model_code_receipt(ROOT), "audit_used": False, "native_fit_count": 3 * 6, "floor_seconds": floor,
              "stands": int(len(stands)), "months": months_report, "candidates": candidates, "selected_candidate": selected,
              "refit_rounds": int(np.rint(np.median([months_report[m]["rounds"] for m in config["development_months"]]))),
              "rule": config["selection"], "environment": environment(ROOT)}
    for path in [ROOT / args.output, ROOT / args.report]:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, report)
    pd.DataFrame({ID: np.concatenate([tables[m][ID].to_numpy() for m in config["development_months"]]),
                  "residual": np.concatenate([residuals[m] for m in config["development_months"]])}).to_parquet(ROOT / args.output.parent / "development_residuals.parquet", index=False)
    print({"selected": selected, "refit_rounds": report["refit_rounds"],
           "months": {m: {"v9": round(r["v9_rmse"], 3), "matched": round(r["matched_share"], 3), "gate": None if r["gate_share_of_matched"] is None else round(r["gate_share_of_matched"], 3), "rounds": r["rounds"]} for m, r in months_report.items()},
           "candidates": {n: {m: (round(e["rmse"], 3), round(100 * (e["ratio_to_v9"] - 1), 2)) for m, e in item["months"].items()} for n, item in candidates.items()}})
    best = selected or "residual_matched_100"
    for m in config["development_months"]:
        print(m, {a: f"{r['v9']:.1f}->{r['candidate']:.1f} ({r['change_pct']:+.1f}%, matched {r['matched_share']:.2f})" for a, r in candidates[best]["months"][m]["airports"].items()})


if __name__ == "__main__":
    main()
