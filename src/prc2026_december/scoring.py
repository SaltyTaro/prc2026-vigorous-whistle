"""Score the shipped January-November models on December: v7 -> v9, every banked learner, the residual stage, and the recipes.

Everything is out-of-sample for December by construction (no December row entered any fit, stopping set or selection).
Outputs (label-free except the final evaluation): scratch/december.parquet (the fold-table layout of tools/build_adsb_scratch.py),
runs/december/learners/<name>.parquet (ID, adsb_full) and the evaluation report.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, write_json
from prc2026.features import ID, PHASE, TARGET, TIME
from prc2026.validation import identity_hash
from prc2026_adsb.features import DEPARTURE_COLUMNS, FEATURE_NAMES as ADSB_DEP_NAMES, NEIGHBOUR_NAMES, departure_features, neighbour_features
from prc2026_lirf_mixture.model import apply_candidate
from prc2026_postprocess.rules import row_context
from prc2026_surface_refit.inference import pipeline_prediction
from prc2026_timeline.features import CATEGORICAL_COLUMNS, CORE_NAMES, FEATURE_NAMES as TIMELINE_NAMES, FLOWN_NAMES, SURFACE_NAMES

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from run_lirf_mixture import MOVEMENT_COLUMNS, fit_population_model, movements, score_population  # noqa: E402
from run_adsb_v11 import SCOPES, attach_extra, candidate_values, predict_residual, rmse  # noqa: E402

DEC_RAW = "training_2025-12-01_2026-01-01.parquet"
SURFACE_REFIT = Path("runs/azure_arcw0_surface_refit/20260914T185207842456Z_refit_full_timeline_v6_100")
BUNDLE_NAMES = [*CORE_NAMES, *FLOWN_NAMES, *SURFACE_NAMES]  # the learners' 276-feature contract (tools/run_adsb_learner.py)
ADSB_NAMES = [*ADSB_DEP_NAMES, *NEIGHBOUR_NAMES]
MISSING = "__MISSING__"
TRAINING_MONTHS = [f"2025-{m:02}" for m in range(1, 12)]


def source_file(month: str) -> str:
    y, m = int(month[:4]), int(month[5:7])
    return f"training_{y}-{m:02}-01_{y + (m == 12)}-{(m % 12) + 1:02}-01.parquet"


def december_departures(root: Path) -> pd.DataFrame:
    raw = pd.read_parquet(root / "data/raw" / DEC_RAW, columns=sorted(set(MOVEMENT_COLUMNS) | set(DEPARTURE_COLUMNS) | {"RUNWAY_mvt"}))
    return raw.loc[raw[PHASE].eq("DEP") & raw[TIME].dt.strftime("%Y-%m").eq("2025-12")].reset_index(drop=True)


def build_scratch(root: Path, features: pd.DataFrame, labels: pd.DataFrame, threads: int) -> tuple[pd.DataFrame, dict]:
    """v7 (the frozen surface-refit pipeline) and v9 (the January-November LIRF mixture) on December, in the scratch-table layout."""
    run = root / SURFACE_REFIT
    v7 = pipeline_prediction(run, read_json(run / "pipeline.json"), features[[ID, *TIMELINE_NAMES]], threads)
    config = read_json(root / "configs/lirf_mixture_v1.json")
    selection = read_json(root / "runs/lirf_mixture/selection.json")
    if selection["config_sha256"] != object_hash(config) or selection["selected_candidate"] != "mixture_75":
        raise ValueError("The v9 declaration does not match its selection receipt.")
    model, component, summary = fit_population_model(movements(config["refit_months"]), config)  # January-November: the v9 export's classifier
    scoring = pd.read_parquet(root / "data/raw" / DEC_RAW, columns=MOVEMENT_COLUMNS)
    departures = scoring.loc[scoring[PHASE].eq("DEP")].set_index(ID).reindex(features[ID]).reset_index()
    if departures[TIME].isna().any():
        raise ValueError("December features do not align with the raw departures.")
    ids = features[ID].to_numpy()
    _, _, _, interval, mixture, index = score_population(model, component, scoring, ids, config)
    spec = config["candidates"]["mixture_75"]
    v9, applied = apply_candidate(v7, index, interval, mixture, spec["mixture_weight"], config["mixture"]["schedule_interval_threshold_seconds"], summary["floor_seconds"])
    context = row_context(departures).set_index(ID).reindex(ids)
    population = np.zeros(len(ids), dtype=bool)
    population[index] = True
    table = pd.DataFrame({ID: ids, TIME: departures[TIME].to_numpy(), "airport": departures["ADEP_mvt"].to_numpy(), TARGET: labels.set_index(ID)[TARGET].reindex(ids).to_numpy(dtype=float),
                          "v9": v9, "floor": summary["floor_seconds"], "proxy_s": context["proxy_s"].to_numpy(), "schedule_interval_s": context["schedule_interval_s"].to_numpy(),
                          "clock_available": context["clock_available"].to_numpy(), "population": population, "v7": v7})
    record = {"v7_rmse": rmse(table[TARGET].to_numpy(), v7), "v9_rmse": rmse(table[TARGET].to_numpy(), v9), "mixture": applied, "floor_seconds": summary["floor_seconds"],
              "population_rows": int(population.sum()), "surface_refit_run": SURFACE_REFIT.as_posix(), "pipeline_sha256": sha256_file(run / "pipeline.json")}
    return table, record


def adsb_table(root: Path, departures: pd.DataFrame, matches_dir: Path) -> pd.DataFrame:
    """The learners' 27 label-free ADS-B features for December from one match table and its stands (tools/run_adsb_learner.py::adsb_table)."""
    stands = pd.read_parquet(root / matches_dir / "stands.parquet")
    matches = pd.read_parquet(root / matches_dir / "matches2_2025-12.parquet")
    feats = departure_features(departures, matches, stands).merge(neighbour_features(departures, matches), on=ID, validate="one_to_one")
    if list(feats.columns) != [ID, *ADSB_NAMES]:
        raise ValueError("Unexpected ADS-B feature columns.")
    return feats


def encode(frame: pd.DataFrame, names: list[str], vocab: dict) -> pd.DataFrame:
    out = {}
    for name in names:
        if name in CATEGORICAL_COLUMNS:
            values = frame[name].astype("string").fillna(MISSING)
            out[name] = pd.Categorical(values.where(values.isin(vocab[name]), MISSING), categories=vocab[name])
        else:
            out[name] = frame[name].to_numpy(dtype="float32", na_value=np.nan)
    return pd.DataFrame(out)


class VocabularyCache:
    """Fitting-row categorical vocabularies of the refit learners, rebuilt from the frozen January-November bundle rows (run_adsb_learner.py::encode/keep)."""

    def __init__(self, root: Path):
        base = root / read_json(root / "data/timeline/latest.json")["bundle"]
        parent = root / read_json(base / "manifest.json")["parent_bundle"]
        cats = [c for c in BUNDLE_NAMES if c in CATEGORICAL_COLUMNS]
        parts, labels = [], []
        for month in TRAINING_MONTHS:
            parts.append(pd.read_parquet(base / "features" / source_file(month), columns=list(dict.fromkeys([ID, *cats, "airport", "nmclock_available", "aircraft_missing"]))))
            labels.append(pd.read_parquet(parent / "labels" / source_file(month)))
        self.frame = pd.concat(parts, ignore_index=True)
        self.labels = pd.concat(labels, ignore_index=True).set_index(ID)[TARGET].reindex(self.frame[ID]).to_numpy(dtype=float)
        if len(self.frame) != 1919370 or np.isnan(self.labels).any():
            raise ValueError("The refit population must be the 1,919,370 January-November departures with labels.")
        self.cats, self.cache, self.parent = cats, {}, parent

    def keep(self, inputs: dict) -> np.ndarray:
        frame = self.frame
        mask = np.ones(len(frame), dtype=bool)
        if inputs.get("target") == "nm_residual":
            mask &= frame["nmclock_available"].to_numpy(dtype=float) == 1
        if inputs.get("specialist_airports"):
            mask &= frame["airport"].isin(inputs["specialist_airports"]).to_numpy()
        mask &= ~frame["airport"].isin(inputs.get("excluded_airports") or []).to_numpy()
        if inputs.get("exclude_population"):
            mask &= ~(frame["airport"].eq("LIRF") & frame["nmclock_available"].eq(0) & frame["aircraft_missing"].eq(1)).to_numpy()
        if inputs.get("max_label") is not None:
            mask &= self.labels <= inputs["max_label"]
        return mask

    def vocabulary(self, inputs: dict, subset: str | None = None) -> dict:
        """`subset` names a frozen fold whose fit partition restricts the rows (the fold models' vocabularies); None is the refit population."""
        key = (tuple(sorted(inputs.get("excluded_airports") or [])), bool(inputs.get("exclude_population")), inputs.get("max_label"), inputs.get("target") or "label", tuple(inputs.get("specialist_airports") or []), subset)
        if key not in self.cache:
            mask = self.keep(inputs)
            if subset is not None:
                assignment = pd.read_parquet(self.parent / "splits" / f"{subset}.parquet")
                mask &= self.frame[ID].isin(assignment.loc[assignment["partition"].eq("fit"), ID]).to_numpy()
            rows = self.frame.loc[mask]
            self.cache[key] = {c: sorted(set(rows[c].astype("string").fillna(MISSING)) | {MISSING}) for c in self.cats}
        return self.cache[key]


def learner_prediction(root: Path, run_dir: Path, frame: pd.DataFrame, vocab: VocabularyCache, threads: int, fold: str = "refit") -> tuple[np.ndarray, dict]:
    """A banked learner applied to December (NaN where its route excludes the row), exactly as tools/run_adsb_learner.py predicts the ranking.

    `fold="refit"` is the shipped January-November model; "seasonal" / "forward" are the development-fold models (fitted on the fold's
    fit partition, early-stopped on October), which December also lies outside of.
    """
    record = read_json(root / run_dir / fold / "run.json")
    inputs = record["inputs"]
    kind, target = inputs.get("kind") or "lightgbm", inputs.get("target") or "label"
    names = [*BUNDLE_NAMES, *ADSB_NAMES, *(inputs.get("extra_features") or [])]
    x = encode(frame, names, vocab.vocabulary(inputs, None if fold == "refit" else fold))
    if kind == "lightgbm":
        booster = lgb.Booster(model_file=str(root / run_dir / fold / "model.txt"))
        if booster.feature_name() != names:
            raise ValueError(f"{run_dir}: feature names differ from the December frame.")
        values = booster.predict(x, num_threads=threads)
    else:
        from catboost import CatBoostRegressor
        model = CatBoostRegressor()
        model.load_model(str(root / run_dir / fold / "model.cbm"))
        if list(model.feature_names_) != names:
            raise ValueError(f"{run_dir}: feature names differ from the December frame.")
        cat_frame = x.copy()
        for name in cat_frame.columns:
            if name in CATEGORICAL_COLUMNS:
                cat_frame[name] = cat_frame[name].astype(str)
        values = model.predict(cat_frame, thread_count=threads)
    clock = frame["nmclock_available"].to_numpy(dtype=float) == 1 if target == "nm_residual" else np.ones(len(frame), dtype=bool)
    if inputs.get("specialist_airports"):
        clock &= frame["airport"].isin(inputs["specialist_airports"]).to_numpy()
    offset = frame["nmclock_proxy_s"].to_numpy(dtype=float) if target == "nm_residual" else np.zeros(len(frame))
    if target == "adsb_residual":
        motion = frame["adsb_motion_s"].to_numpy(dtype=float)
        offset = np.where(np.isfinite(motion), motion, 0.0)
    prediction = np.where(clock, offset + values, np.nan)
    return prediction, {"kind": kind, "target": target, "fold": fold, "features": len(names), "rounds": record.get("rounds", record.get("best_iteration")), "fitted_rows": record.get("fitted_rows"),
                        "model_sha256": sha256_file(root / run_dir / fold / ("model.txt" if kind == "lightgbm" else "model.cbm")),
                        "run_sha256": sha256_file(root / run_dir / fold / "run.json"), "finite_rows": int(np.isfinite(prediction).sum())}


def leaf_members(root: Path, run_dir: str) -> list[str]:
    """Flatten an ensemble directory to its leaf learner directories with multiplicities (tools/ensemble_adsb_full.py records the list it averaged)."""
    if (root / run_dir / "refit" / "run.json").exists():
        return [run_dir]
    return [leaf for member in read_json(root / run_dir / "run.json")["learners"] for leaf in leaf_members(root, member)]


def ensemble_prediction(members: list[str], predictions: dict[str, np.ndarray]) -> np.ndarray:
    wide = np.column_stack([predictions[m] for m in members])
    with np.errstate(invalid="ignore"):
        return np.nanmean(wide, axis=1)


def residual_table(root: Path, config: dict, scratch: pd.DataFrame, departures: pd.DataFrame, extra: list[pd.DataFrame]) -> pd.DataFrame:
    """The residual stage's December feature table (tools/run_adsb_v11.py::month_table with the declared matches directory and neighbour features)."""
    matches_dir = root / config.get("matches_dir", "data/external/adsblol/matches")
    stands = pd.read_parquet(matches_dir / "stands.parquet")
    matches = pd.read_parquet(matches_dir / "matches2_2025-12.parquet")
    feats = departure_features(departures, matches, stands).merge(departures[[ID, "STAND_mvt", "RUNWAY_mvt"]], on=ID, validate="one_to_one")
    if config["residual_model"].get("neighbour_features", False):
        feats = feats.merge(neighbour_features(departures, matches), on=ID, validate="one_to_one")
    table = scratch.merge(feats, on=ID, how="left", validate="one_to_one")
    if table["adsb_matched"].isna().any():
        raise ValueError("ADS-B feature table does not cover every December row.")
    table["month"] = "2025-12"
    table["hour"] = table[TIME].dt.hour + table[TIME].dt.minute / 60
    table = table.rename(columns={"STAND_mvt": "stand", "RUNWAY_mvt": "runway"})
    return attach_extra(table, extra)


def load_residual(root: Path, export_dir: Path) -> dict:
    meta = read_json(root / export_dir / "residual_model.json")
    boosters = [lgb.Booster(model_file=str(root / export_dir / f)) for f in meta["models"]]
    return {"boosters": boosters, "booster": boosters[0], "categories": meta["categories"], "rounds": meta["rounds"], "feature_set": meta["feature_set"], "scope": meta["scope"],
            "models_sha256": [sha256_file(root / export_dir / f) for f in meta["models"]]}


def evaluate_candidates(config: dict, table: pd.DataFrame, residual: np.ndarray, baseline: np.ndarray) -> dict:
    label, v9, floor = table[TARGET].to_numpy(dtype=float), table["v9"].to_numpy(dtype=float), float(table["floor"].iloc[0])
    out = {}
    for name, item in config["candidates"].items():
        spec = {"weight": item["weight"], "scope": item.get("scope", "matched"), "blend_with": item.get("blend_with"), "extra_blends": item.get("extra_blends", [])}
        values, _ = candidate_values(table, v9, residual, spec, floor)
        score = rmse(label, values)
        entry = {"rmse": score, "ratio_to_v9": score / rmse(label, v9), "ratio_to_baseline": score / rmse(label, baseline), "rows_differing_from_baseline": int((np.abs(values - baseline) > 1e-9).sum()), "airports": {}}
        for airport in sorted(table["airport"].unique()):
            k = table["airport"].eq(airport).to_numpy()
            entry["airports"][airport] = {"rows": int(k.sum()), "baseline": rmse(label[k], baseline[k]), "candidate": rmse(label[k], values[k]), "change_pct_vs_baseline": 100 * (rmse(label[k], values[k]) / rmse(label[k], baseline[k]) - 1)}
        out[name] = entry
    return out
