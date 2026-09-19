"""Independently verify a twelve-month-refit export (adsb_v23_v1): learner records, mirrored models, ensembles, residual, blend, floor.

Checks: declaration parents; the export receipt's provenance; every leaf refit record (twelve months, 2,085,047 rows, December's
165,677, rounds equal to the median of its frozen fold best iterations, recipe fields equal to the source folds, code hashes);
every mirrored model re-predicts a sample of ranking rows to its saved predictions; every ranking table equals the mean of finite
leaf predictions over its frozen flattened member list; v9 reproduced from scratch; the residual recomputed from the saved booster
on an independently rebuilt ranking feature table; the blend recomputed and equal to the export after rounding; rows outside every
scope equal to the uploaded v9. Nothing is uploaded.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, write_json
from prc2026.features import ID, PHASE, TARGET
from prc2026.validation import identity_hash
from prc2026_adsb.features import DEPARTURE_COLUMNS
from prc2026_december.scoring import ADSB_NAMES, BUNDLE_NAMES, MISSING, adsb_table, encode
from prc2026_lirf_mixture.model import apply_candidate
from prc2026_timeline.features import CATEGORICAL_COLUMNS
from export_adsb_v11 import ranking_table
from export_december_refit import flattened_members
from run_adsb_v11 import candidate_values, check_parents, matches_dir, predict_residual
from run_lirf_mixture import MOVEMENT_COLUMNS, fit_population_model, movements, score_population

TRAINING = [f"2025-{m:02}" for m in range(1, 12)]


def source_file(month: str) -> str:
    y, m = int(month[:4]), int(month[5:7])
    return f"training_{y}-{m:02}-01_{y + (m == 12)}-{(m % 12) + 1:02}-01.parquet"


class TwelveMonthVocabulary:
    """Categorical vocabularies of the twelve-month fitting rows per learner recipe (run_adsb_learner.py::encode on the kept rows)."""

    def __init__(self, december: Path):
        base = ROOT / read_json(ROOT / "data/timeline/latest.json")["bundle"]
        parent = ROOT / read_json(base / "manifest.json")["parent_bundle"]
        self.cats = [c for c in BUNDLE_NAMES if c in CATEGORICAL_COLUMNS]
        cols = list(dict.fromkeys([ID, *self.cats, "airport", "nmclock_available", "aircraft_missing"]))
        parts = [pd.read_parquet(base / "features" / source_file(m), columns=cols) for m in TRAINING] + [pd.read_parquet(december / "features" / source_file("2025-12"), columns=cols)]
        labels = [pd.read_parquet(parent / "labels" / source_file(m)) for m in TRAINING] + [pd.read_parquet(december / "labels" / source_file("2025-12"))]
        self.frame = pd.concat(parts, ignore_index=True)
        self.labels = pd.concat(labels, ignore_index=True).set_index(ID)[TARGET].reindex(self.frame[ID]).to_numpy(dtype=float)
        if len(self.frame) != 2085047 or np.isnan(self.labels).any():
            raise ValueError("The twelve-month population must be 2,085,047 labelled departures.")
        self.cache = {}

    def vocabulary(self, inputs: dict) -> dict:
        key = (tuple(inputs.get("excluded_airports") or []), bool(inputs.get("exclude_population")), inputs.get("max_label"), inputs.get("target") or "label", tuple(inputs.get("specialist_airports") or []))
        if key not in self.cache:
            f = self.frame
            mask = np.ones(len(f), dtype=bool)
            if key[3] == "nm_residual":
                mask &= f["nmclock_available"].to_numpy(dtype=float) == 1
            if key[4]:
                mask &= f["airport"].isin(key[4]).to_numpy()
            mask &= ~f["airport"].isin(key[0]).to_numpy()
            if key[1]:
                mask &= ~(f["airport"].eq("LIRF") & f["nmclock_available"].eq(0) & f["aircraft_missing"].eq(1)).to_numpy()
            if key[2] is not None:
                mask &= self.labels <= key[2]
            rows = f.loc[mask]
            self.cache[key] = {c: sorted(set(rows[c].astype("string").fillna(MISSING)) | {MISSING}) for c in self.cats}
        return self.cache[key]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/adsb_v23_v1.json"))
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--sample", type=int, default=3000)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--report", type=Path, default=Path("docs/adsb_v23/verification.json"))
    args = parser.parse_args()
    config = read_json(ROOT / args.config)
    check_parents(config)
    export = ROOT / "runs/submissions" / f"vigorous-whistle_v{args.version}.parquet"
    receipt = read_json(export.with_suffix(".receipt.json"))
    if receipt["sha256"] != sha256_file(export) or receipt["provenance"]["config_sha256"] != object_hash(config) or receipt["provenance"]["policy"] != config["prediction_postprocess"]:
        raise ValueError("Export receipt does not belong to this declaration.")
    exported = pd.read_parquet(export)
    ids = exported[ID].to_numpy()
    values_exported = exported[TARGET].to_numpy()
    december = ROOT / Path(config["december_build"]["path"]).parent
    december_manifest = read_json(ROOT / config["december_build"]["path"])
    # Ranking learner frame (independent of the workers' caches)
    ranking_raw = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=sorted(set(MOVEMENT_COLUMNS) | set(DEPARTURE_COLUMNS) | {"RUNWAY_mvt"}))
    departures = ranking_raw.loc[ranking_raw[PHASE].eq("DEP")].reset_index(drop=True)
    if not np.array_equal(departures[ID].to_numpy(), ids):
        raise ValueError("Export rows are not the ranking departures in template order.")
    bundle = ROOT / read_json(ROOT / "data/timeline/latest.json")["bundle"]
    frame = pd.read_parquet(bundle / "features/ranking.parquet", columns=[ID, *BUNDLE_NAMES]).merge(adsb_table(ROOT, departures, Path("data/external/adsblol/matches")), on=ID, how="left", validate="one_to_one")
    frame = frame.merge(pd.concat([pd.read_parquet(ROOT / f"data/external/adsblol/features_v19/features_v19_{m}.parquet") for m in ["2026-01", "2026-07"]], ignore_index=True), on=ID, how="left", validate="one_to_one")
    if not np.array_equal(frame[ID].to_numpy(), ids):
        raise ValueError("Ranking feature frame order differs from the export.")
    vocab = TwelveMonthVocabulary(december)
    rng = np.random.default_rng(config["seed"])
    sample = np.sort(rng.choice(len(ids), size=min(args.sample, len(ids)), replace=False))
    leaves, predictions = {}, {}
    for leaf, spec in config["leaf_learners"].items():
        run_dir = ROOT / spec["refit_output"]
        record = read_json(run_dir / "run.json")
        folds = {f: read_json(ROOT / leaf / f / "run.json") for f in ["seasonal", "forward"]}
        inputs, src = record["inputs"], folds["seasonal"]["inputs"]
        expected_rounds = int(np.rint(np.median([r["best_iteration"] for r in folds.values()])))
        src_params = {k: v for k, v in (src.get("parameters") or src.get("lightgbm_parameters")).items() if k not in ("num_threads", "thread_count")}
        got_params = {k: v for k, v in inputs["parameters"].items() if k not in ("num_threads", "thread_count")}
        problems = []
        if record["stage"] != "refit_twelve_months" or record["months"] != [*TRAINING, "2025-12"] or record["rows"] != 2085047 or record["december_rows"] != 165677:
            problems.append("population")
        if record["rounds"] != expected_rounds or inputs["source_run"] != leaf:
            problems.append("rounds/source")
        if got_params != src_params or inputs["excluded_airports"] != sorted(src.get("excluded_airports") or []) or inputs["target"] != (src.get("target") or "label") \
                or bool(inputs["exclude_population"]) != bool(src.get("exclude_population")) or inputs["max_label"] != src.get("max_label") or (inputs["extra_features"] or []) != (src.get("extra_features") or []):
            problems.append("recipe")
        if inputs["source_fold_records_sha256"] != {f: sha256_file(ROOT / leaf / f / "run.json") for f in folds} or inputs["december_ids_sha256"] != december_manifest["ids_sha256"]:
            problems.append("fold records / December identity")
        if inputs["code"]["files"]["tools/run_adsb_learner_dec.py"] != config["parents"]["runner_twelve_months"]["sha256"] or inputs["code"]["files"]["tools/run_adsb_learner.py"] != config["parents"]["runner_frozen"]["sha256"] \
                or inputs["code"]["files"]["src/prc2026_adsb/features.py"] != config["parents"]["adsb_feature_code"]["sha256"]:
            problems.append("code")
        if problems:
            raise ValueError(f"{leaf}: {problems}")
        saved = pd.read_parquet(run_dir / "ranking_predictions.parquet")
        if not np.array_equal(saved[ID].to_numpy(), ids):
            raise ValueError(f"{leaf}: prediction order.")
        predictions[leaf] = saved["adsb_full"].to_numpy(dtype=float)
        kind, target = inputs["kind"], inputs["target"]
        names = [*BUNDLE_NAMES, *ADSB_NAMES, *(inputs["extra_features"] or [])]
        x = encode(frame.iloc[sample].reset_index(drop=True), names, vocab.vocabulary(inputs))
        if kind == "lightgbm":
            booster = lgb.Booster(model_file=str(run_dir / "model.txt"))
            if booster.feature_name() != names:
                raise ValueError(f"{leaf}: model feature names.")
            raw = booster.predict(x, num_threads=args.threads)
        else:
            from catboost import CatBoostRegressor
            model = CatBoostRegressor()
            model.load_model(str(run_dir / "model.cbm"))
            if list(model.feature_names_) != names:
                raise ValueError(f"{leaf}: model feature names.")
            xc = x.copy()
            for name in xc.columns:
                if name in CATEGORICAL_COLUMNS:
                    xc[name] = xc[name].astype(str)
            raw = model.predict(xc, thread_count=args.threads)
        sub = frame.iloc[sample]
        clock = sub["nmclock_available"].to_numpy(dtype=float) == 1 if target == "nm_residual" else np.ones(len(sub), dtype=bool)
        offset = sub["nmclock_proxy_s"].to_numpy(dtype=float) if target == "nm_residual" else np.zeros(len(sub))
        expected = np.where(clock, offset + raw, np.nan)
        got = predictions[leaf][sample]
        finite = np.isfinite(expected) & np.isfinite(got)
        if not np.array_equal(np.isfinite(expected), np.isfinite(got)):
            raise ValueError(f"{leaf}: route mask differs.")
        delta = float(np.max(np.abs(expected[finite] - got[finite]))) if finite.any() else 0.0
        if delta > 1e-6:
            raise ValueError(f"{leaf}: mirrored model re-prediction differs by {delta}.")
        leaves[leaf] = {"kind": kind, "target": target, "rounds": record["rounds"], "fitted_rows": record["fitted_rows"], "seconds": record["seconds"], "sample_max_abs_difference": delta,
                        "predictions_sha256": sha256_file(run_dir / "ranking_predictions.parquet"), "model_sha256": sha256_file(run_dir / ("model.txt" if kind == "lightgbm" else "model.cbm"))}
        print("leaf ok", leaf, kind, record["rounds"], delta, flush=True)
    # Ranking tables from the frozen flattened member lists
    base = read_json(ROOT / config["parents"]["v22_v1_declaration"]["path"])
    tables = {}
    for name, spec in config["extra_feature_tables"].items():
        members = flattened_members(ROOT, Path(base["extra_feature_tables"][name]["development"][0]).parent.parent.as_posix())
        wide = np.column_stack([predictions[m] for m in members])
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(wide, axis=1)
        saved = pd.read_parquet(ROOT / spec["ranking"])
        if not np.array_equal(saved[ID].to_numpy(), ids) or not np.array_equal(np.isfinite(mean), np.isfinite(saved["adsb_full"].to_numpy())):
            raise ValueError(f"{name}: table population.")
        finite = np.isfinite(mean)
        delta = float(np.max(np.abs(mean[finite] - saved["adsb_full"].to_numpy(dtype=float)[finite])))
        if delta > 1e-9:
            raise ValueError(f"{name}: table differs by {delta}.")
        tables[name] = {"members": len(members), "max_abs_difference": delta, "sha256": sha256_file(ROOT / spec["ranking"])}
    # v9 from scratch, residual from the saved booster on a rebuilt table, the blend, the floor
    mixture_config = read_json(ROOT / "configs/lirf_mixture_v1.json")
    model, component, summary = fit_population_model(movements(TRAINING), mixture_config)
    floor = summary["floor_seconds"]
    v7 = pd.read_parquet(ROOT / config["parents"]["incumbent_ranking_predictions"]["path"]).set_index(ID)["prediction"].reindex(ids).to_numpy(dtype=float)
    _, _, _, interval, mixture, index = score_population(model, component, ranking_raw, ids, mixture_config)
    v9, _ = apply_candidate(v7, index, interval, mixture, mixture_config["candidates"]["mixture_75"]["mixture_weight"], mixture_config["mixture"]["schedule_interval_threshold_seconds"], floor)
    uploaded_v9 = pd.read_parquet(ROOT / config["parents"]["v9_export"]["path"]).set_index(ID)[TARGET].reindex(ids).to_numpy()
    if not np.array_equal(np.rint(v9).astype(np.int32), uploaded_v9):
        raise ValueError("v9 does not reproduce from scratch.")
    stands = pd.read_parquet(matches_dir(config) / "stands.parquet")
    table, _ = ranking_table(config, departures, stands, v9)
    export_dir = ROOT / "runs/adsb_v23" / f"export_v{args.version}"
    meta = read_json(export_dir / "residual_model.json")
    if receipt["provenance"]["residual_model_sha256"] != sha256_file(export_dir / "residual_model.txt"):
        raise ValueError("Residual booster differs from the receipt.")
    residual_model = {"boosters": [lgb.Booster(model_file=str(export_dir / "residual_model.txt"))], "categories": meta["categories"], "rounds": meta["rounds"], "feature_set": meta["feature_set"]}
    residual_model["booster"] = residual_model["boosters"][0]
    residual = predict_residual(residual_model, table, config, meta["scope"])
    saved_features = pd.read_parquet(export_dir / "ranking_features.parquet")
    if not np.array_equal(saved_features[ID].to_numpy(), ids):
        raise ValueError("Saved ranking features order.")
    residual_delta = float(np.max(np.abs(residual - saved_features["residual"].to_numpy(dtype=float))))
    candidate = config["candidates"][config["selected_candidate"]]
    item = {"weight": candidate["weight"], "scope": candidate.get("scope", "matched"), "blend_with": candidate.get("blend_with"), "extra_blends": candidate.get("extra_blends", [])}
    values, inside = candidate_values(table, v9, residual, item, floor)
    recomputed = np.rint(values).astype(np.int32)
    if not np.array_equal(recomputed, values_exported):
        raise ValueError(f"Independent recomputation differs from the export on {int((recomputed != values_exported).sum())} rows.")
    if not np.array_equal(values_exported[~inside], uploaded_v9[~inside]):
        raise ValueError("Rows outside every scope differ from the uploaded v9.")
    if values_exported.min() < floor:
        raise ValueError("Export below the floor.")
    report = {"verified_at_utc": utc_now(), "file": export.relative_to(ROOT).as_posix(), "sha256": receipt["sha256"], "rows": int(len(ids)), "ids_sha256": identity_hash(ids), "config_sha256": object_hash(config),
              "leaves": leaves, "sampled_ranking_rows_per_leaf": int(len(sample)), "tables": tables, "v9_reproduced_from_scratch": True, "residual_max_abs_difference": residual_delta,
              "residual_rounds": meta["rounds"], "residual_training_rows": meta["training_rows"], "residual_months": meta["training_months"], "independent_recomputation_matches": True,
              "rows_outside_scope_equal_v9": True, "rows_in_scope": int(inside.sum()), "floor_seconds": floor, "matched_rows": int(table["adsb_matched"].eq(1).sum())}
    (ROOT / args.report).parent.mkdir(parents=True, exist_ok=True)
    write_json(ROOT / args.report, report)
    print({k: report[k] for k in ["file", "rows", "sampled_ranking_rows_per_leaf", "residual_max_abs_difference", "independent_recomputation_matches", "rows_outside_scope_equal_v9", "rows_in_scope"]}, "leaves", len(leaves), "tables", len(tables))


if __name__ == "__main__":
    main()
