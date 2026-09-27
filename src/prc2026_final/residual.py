"""S9: the residual stage on the file (matched rows outside LIRF): 0.1 x mean(adj2, cat) + 0.9 x mean(16 rnet seeds), 0 elsewhere.

- Table: tools/export_adsb_v11.py::ranking_table with neighbour features on the residual match directory (matches_v11 stands),
  re-implemented here because the frozen function reads the ranking months' match files by name. Columns: the ADS-B departure and
  neighbour features, stand, runway, airport, v9, proxy_s, schedule_interval_s, population, hour.
- adj2: one LightGBM per target month (circular month-distance weights), saved only for 2026-01 and 2026-07
  (runs/adsb_v25/export_v23/residual_adj2_<month>.txt); any other month needs `refit_adj2` (tools/export_adsb_v25.py --stage fit's
  fit_variant on the twelve 2025 scratch/v24 months, ~5 min on 16 threads), invoked explicitly (final_predict.py refit-adj2).
- cat: the single CatBoost of runs/adsb_v25/export_v23/residual_cat_all.cbm.
- rnet: the 16 twelve-month refits of runs/adsb_rnn/rnb_s5??/refit12 on the nb columns of the in-scope rows.
Arithmetic as tools/export_adsb_v25.py then tools/export_adsb_rnn.py: mean over [adj2, cat], zero outside the scope, then
(1 - 0.9) x that + 0.9 x the seed mean inside the scope.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from prc2026.artifacts import read_json, sha256_file, utc_now, write_json
from prc2026.features import ID, TIME
from prc2026_adsb.features import DEPARTURE_COLUMNS, departure_features, neighbour_features
from prc2026_postprocess.rules import row_context

from .common import ROOT

sys.path.insert(0, str(ROOT / "tools"))
from run_adsb_v11 import scope_mask  # noqa: E402
from run_adsb_v25 import fit_variant, predict_variant  # noqa: E402

V25_CONFIG = "configs/adsb_v25_v1.json"
V23_EXPORT = Path("runs/adsb_v25/export_v23")
RNN_DECLARATION = "configs/adsb_rnn_v1.json"
SCOPE = "matched_no_lirf"


def v25_config() -> dict:
    config = read_json(ROOT / V25_CONFIG)
    config["matches_dir"] = config["matches_dirs"]["v11"]
    config["residual_model"]["neighbour_features"] = True  # the runtime switches tools/export_adsb_v25.py sets after the hash
    return config


def residual_table(departures: pd.DataFrame, matches: pd.DataFrame, v9: pd.DataFrame) -> pd.DataFrame:
    """export_adsb_v11.ranking_table (neighbour features, no extra tables) for the file's departures and its residual-directory matches."""
    config = v25_config()
    stands = pd.read_parquet(ROOT / config["matches_dir"] / "stands.parquet")
    dep = departures.reset_index(drop=True)
    feats = departure_features(dep[DEPARTURE_COLUMNS], matches, stands).merge(dep[[ID, "STAND_mvt", "RUNWAY_mvt"]], on=ID, validate="one_to_one")
    feats = feats.merge(neighbour_features(dep[[*DEPARTURE_COLUMNS, "RUNWAY_mvt"]], matches), on=ID, validate="one_to_one")
    context = row_context(dep)
    table = feats.rename(columns={"STAND_mvt": "stand", "RUNWAY_mvt": "runway"})
    table["airport"] = dep["ADEP_mvt"].to_numpy()
    table["v9"] = v9.set_index(ID).reindex(dep[ID])["v9"].to_numpy(dtype=float)
    table["proxy_s"] = context["proxy_s"].to_numpy()
    table["schedule_interval_s"] = context["schedule_interval_s"].to_numpy()
    table["population"] = context["hedge_population"].to_numpy()
    table["hour"] = dep[TIME].dt.hour + dep[TIME].dt.minute / 60
    if not np.array_equal(table[ID].to_numpy(), dep[ID].to_numpy()):
        raise ValueError("Residual table order differs from the departures.")
    return table


def adj2_model(month: str, refit_dir: Path) -> dict:
    """The saved adj2 booster of the target month: the v23 export's for 2026-01/07, else this run's explicit refit."""
    fitted = read_json(ROOT / V23_EXPORT / "fit_record.json")["models"]["adj2"]
    if month in fitted:
        spec, path = fitted[month], ROOT / V23_EXPORT / fitted[month]["file"]
    else:
        record_path = refit_dir / f"residual_adj2_{month}.json"
        if not record_path.exists():
            raise ValueError(f"No adj2 booster for {month}: run `final_predict.py refit-adj2 --month {month}` first (declared ~5-minute refit).")
        spec = read_json(record_path)
        path = refit_dir / spec["file"]
    if sha256_file(path) != spec["sha256"]:
        raise ValueError(f"adj2 booster for {month} differs from its record.")
    return {"kind": "lightgbm", "model": lgb.Booster(model_file=str(path)), "categories": spec["categories"], "rounds": spec["rounds"],
            "scope": spec["scope"], "feature_set": spec["feature_set"], "sha256": spec["sha256"]}


def cat_model() -> dict:
    from catboost import CatBoostRegressor
    spec = read_json(ROOT / V23_EXPORT / "fit_record.json")["models"]["cat"]["all"]
    path = ROOT / V23_EXPORT / spec["file"]
    if sha256_file(path) != spec["sha256"]:
        raise ValueError("The cat residual differs from its record.")
    model = CatBoostRegressor()
    model.load_model(str(path))
    return {"kind": "catboost", "model": model, "categories": spec["categories"], "rounds": spec["rounds"], "scope": spec["scope"],
            "feature_set": spec["feature_set"], "sha256": spec["sha256"]}


def refit_adj2(month: str, refit_dir: Path, threads: int) -> dict:
    """The declared adj2 refit for a target month without a saved booster: fit_variant on the twelve 2025 cross-fit months (scratch/v24)."""
    sys.path.insert(0, str(ROOT / "tools"))
    from run_adsb_v24 import ALL_MONTHS, adsb_features, load_frame, month_table
    config = v25_config()
    matches_root = ROOT / config["matches_dirs"]["v11"]
    train = pd.concat([month_table(load_frame(config, m, "cross_fit"), adsb_features(m, matches_root, ROOT / "runs/adsb_v24/features"), m) for m in ALL_MONTHS], ignore_index=True)
    model = fit_variant(train, config, config["residual_variants"]["adj2"], month, threads)
    refit_dir.mkdir(parents=True, exist_ok=True)
    path = refit_dir / f"residual_adj2_{month}.txt"
    model["model"].save_model(str(path), num_iteration=model["rounds"])
    record = {"created_at_utc": utc_now(), "file": path.name, "sha256": sha256_file(path), "kind": "lightgbm", "rounds": model["rounds"],
              "inner_best_iterations": model["inner_best_iterations"], "training_rows": model["rows"], "weights": model["weights"], "weight_sum": model["weight_sum"],
              "target_month": month, "categories": model["categories"], "feature_set": model["feature_set"], "scope": model["scope"], "seconds": model["seconds"],
              "threads": threads, "rule": "tools/run_adsb_v25.fit_variant on the twelve 2025 cross-fit months (tools/export_adsb_v25.py --stage fit), unchanged"}
    write_json(refit_dir / f"residual_adj2_{month}.json", record)
    return record


def rnet_predictions(table: pd.DataFrame, threads: int) -> tuple[np.ndarray, dict]:
    import torch
    torch.set_num_threads(threads)
    from prc2026_nn.train import load
    from run_adsb_rnn import columns_for, predict_rows, resolve_variant
    spec = read_json(ROOT / RNN_DECLARATION)["residual_networks"]["rnet"]
    config = read_json(ROOT / spec["config"])
    numeric, categorical, extra = columns_for(config, resolve_variant(config, spec["variant"]))
    if extra:
        raise ValueError("The declared residual network uses the wide feature set; this runner implements the nb set only.")
    inside = scope_mask(table, SCOPE)
    rows = table.loc[inside].reset_index(drop=True)
    columns, records = [], {}
    for learner in spec["learners"]:
        run = ROOT / learner / "refit12"
        record = read_json(run / "run.json")
        if record["model_sha256"] != sha256_file(run / "model.pt"):
            raise ValueError(f"{learner}: saved model differs from its record.")
        model, pre = load(run)
        values = predict_rows(model, pre, rows) if len(rows) else np.empty(0)
        column = np.zeros(len(table))
        column[inside] = values
        columns.append(column)
        records[learner] = {"seed": record["seed"], "model_sha256": record["model_sha256"]}
    return np.mean(columns, axis=0), {"learners": records, "rows_in_scope": int(inside.sum())}


def build(table: pd.DataFrame, months_of_rows: np.ndarray, refit_dir: Path, threads: int) -> tuple[pd.DataFrame, dict]:
    config = v25_config()
    inside = scope_mask(table, SCOPE)
    adj2 = np.zeros(len(table))
    adj2_records = {}
    for month in sorted(set(months_of_rows)):
        model = adj2_model(month, refit_dir)
        rows = months_of_rows == month
        values = predict_variant(model, table, config)
        adj2[rows] = values[rows]
        adj2_records[month] = {"sha256": model["sha256"], "rounds": model["rounds"]}
    cat = predict_variant(cat_model(), table, config)
    residual_v25 = np.mean([adj2, cat], axis=0)
    residual_v25[~inside] = 0.0
    net, net_record = rnet_predictions(table, threads)
    w = 0.9
    residual = np.where(inside, (1 - w) * residual_v25 + w * net, 0.0)
    out = pd.DataFrame({ID: table[ID].to_numpy(), "in_scope": inside, "residual:adj2": adj2, "residual:cat": cat, "residual_v25": residual_v25,
                        "network": net, "residual": residual})
    return out, {"adj2": adj2_records, "rnet": net_record, "rows_in_scope": int(inside.sum()), "weights": {"adj2": 0.05, "cat": 0.05, "rnet": 0.9}}
