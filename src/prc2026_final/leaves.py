"""S7: the 29 twelve-month tree leaves and the two tree tables (committee ens_v20_bag_q_all, LIRF table ens_v22_lirf_q).

Each leaf is its saved twelve-month refit (configs/adsb_v23_v1.json leaf_learners -> refit_output/{model.txt|model.cbm, run.json}),
applied exactly as tools/verify_december_refit.py re-predicts it: the timeline bundle's 276 features + the learners' 27 ADS-B features
(+ the 11 v19 features when the leaf declares them), categorical vocabularies of the leaf's twelve-month fitting rows
(TwelveMonthVocabulary, cached once as JSON), NM-residual leaves predict clock-present rows only with the proxy added back.
A table is the mean of finite leaf predictions over its frozen flattened member list (with multiplicities), as
tools/export_december_refit.py::build_ranking_ensembles built runs/adsb_v23/ens/<table>/refit/ranking_predictions.parquet.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from prc2026.artifacts import read_json, sha256_file
from prc2026_december.scoring import ADSB_NAMES, BUNDLE_NAMES, encode
from prc2026_timeline.features import CATEGORICAL_COLUMNS

from .common import ROOT

sys.path.insert(0, str(ROOT / "tools"))
from verify_december_refit import TwelveMonthVocabulary  # noqa: E402

TABLES = {"ens_v20_bag_q_all": "runs/adsb_v23/ens/ens_v20_bag_q_all/refit/run.json", "ens_v22_lirf_q": "runs/adsb_v23/ens/ens_v22_lirf_q/refit/run.json"}
LEAF_CONFIG = "configs/adsb_v23_v1.json"
DECEMBER_BUILD = Path("data/december/d198dea10625b0a75977")
CHUNK = 60000


def members() -> dict[str, list[str]]:
    return {name: read_json(ROOT / path)["learners"] for name, path in TABLES.items()}


def vocab_key(inputs: dict) -> str:
    return json.dumps([sorted(inputs.get("excluded_airports") or []), bool(inputs.get("exclude_population")), inputs.get("max_label"),
                       inputs.get("target") or "label", sorted(inputs.get("specialist_airports") or [])])


def vocabularies(cache_path: Path, leaves: dict[str, dict]) -> dict[str, dict]:
    """Twelve-month fitting-row vocabularies per distinct leaf recipe, computed once (they depend only on the frozen training data)."""
    cached = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    needed = {vocab_key(spec["inputs"]): spec["inputs"] for spec in leaves.values()}
    missing = {k: v for k, v in needed.items() if k not in cached}
    if missing:
        builder = TwelveMonthVocabulary(ROOT / DECEMBER_BUILD)
        for key, inputs in missing.items():
            cached[key] = builder.vocabulary(inputs)
        del builder
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cached), encoding="utf-8")
    return cached


def leaf_specs() -> dict[str, dict]:
    config = read_json(ROOT / LEAF_CONFIG)
    out = {}
    for leaf, spec in config["leaf_learners"].items():
        run_dir = ROOT / spec["refit_output"]
        record = read_json(run_dir / "run.json")
        if record["stage"] != "refit_twelve_months" or record["inputs"]["source_run"] != leaf:
            raise ValueError(f"{leaf}: not its twelve-month refit.")
        kind = record["inputs"]["kind"]
        model_file = run_dir / ("model.txt" if kind == "lightgbm" else "model.cbm")
        out[leaf] = {"run_dir": run_dir, "inputs": record["inputs"], "kind": kind, "model_file": model_file, "model_sha256": sha256_file(model_file)}
    return out


def predict_leaf(spec: dict, frame: pd.DataFrame, vocab: dict, threads: int) -> np.ndarray:
    inputs = spec["inputs"]
    target = inputs.get("target") or "label"
    names = [*BUNDLE_NAMES, *ADSB_NAMES, *(inputs.get("extra_features") or [])]
    if spec["kind"] == "lightgbm":
        model = lgb.Booster(model_file=str(spec["model_file"]))
        if model.feature_name() != names:
            raise ValueError(f"{spec['run_dir']}: model feature names differ from the frame.")
    else:
        from catboost import CatBoostRegressor
        model = CatBoostRegressor()
        model.load_model(str(spec["model_file"]))
        if list(model.feature_names_) != names:
            raise ValueError(f"{spec['run_dir']}: model feature names differ from the frame.")
    raw = np.empty(len(frame))
    for start in range(0, len(frame), CHUNK):
        x = encode(frame.iloc[start:start + CHUNK].reset_index(drop=True), names, vocab)
        if spec["kind"] == "lightgbm":
            raw[start:start + CHUNK] = model.predict(x, num_threads=threads)
        else:
            for name in x.columns:
                if name in CATEGORICAL_COLUMNS:
                    x[name] = x[name].astype(str)
            raw[start:start + CHUNK] = model.predict(x, thread_count=threads)
    clock = frame["nmclock_available"].to_numpy(dtype=float) == 1 if target == "nm_residual" else np.ones(len(frame), dtype=bool)
    if inputs.get("specialist_airports"):
        clock &= frame["airport"].isin(inputs["specialist_airports"]).to_numpy()
    offset = frame["nmclock_proxy_s"].to_numpy(dtype=float) if target == "nm_residual" else np.zeros(len(frame))
    if target not in ("label", "nm_residual"):
        raise ValueError(f"{spec['run_dir']}: undeclared target {target}.")
    return np.where(clock, offset + raw, np.nan)


def table_mean(member_list: list[str], predictions: dict[str, np.ndarray]) -> np.ndarray:
    wide = np.column_stack([predictions[m] for m in member_list])
    with np.errstate(invalid="ignore"):
        return np.nanmean(wide, axis=1)
