"""Own-aircraft turnaround columns in the tree members (configs/tree_own_screen_v1.json; a development round, if any, declares itself).

One fit of one banked leaf of the committee (ens_v20_bag_q_all) or the LIRF table (ens_v22_lirf_q), in one of two variants:

  ctl   the leaf's own recipe and inputs (kind, seed, parameters, exclusions, target, its extra features from features_v19)
  own   the same recipe plus the 15 own-aircraft turnaround columns of data/external/adsblol/features_v19own (tunnel 18's columns,
        tools/build_ownpoints_features.py; label-free, departure block times never read)

on one fold, each the construction the shipped development tables were built with:

  lomo_2025-01 / lomo_2025-07  the twelve-month refit population (frozen January-November refit rows + the December build) minus the held
                               month, fixed rounds, predicts the held month (tools/run_adsb_learner_lomo.py, its construction repeated)
  jan_nov                      the same construction holding out 2025-12: the January-November refit population, predicts December 2025
  forward                      tools/run_adsb_learner.py's forward fold (fit January-September, early stopping October, score November)
  refit12                      all twelve 2025 months at fixed rounds, predicts the ranking departures (tools/run_adsb_learner_dec.py)

Fixed rounds are the leaf's twelve-month rule (median of its recorded seasonal and forward best iterations) for BOTH variants; only the
forward fold early-stops (cap and patience of the leaf's records). Output: <output>/<leaf key>/<variant>/<fold>/{model.*, predictions.parquet
(MVT_ID_mvt, adsb_full), run.json}. --build-cache-only builds the label-free ADS-B tables (<output>/_features) and exits.
No official score, no leaderboard and no upload; reproduction of the saved leaves is checked after mirroring (tools/run_tree_own_chain.py).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import environment, object_hash, read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET, TIME  # noqa: E402
from prc2026.validation import identity_hash  # noqa: E402
from prc2026_phase3.bundle import load_partition  # noqa: E402
from prc2026_phase5r.baseline import aligned_truth  # noqa: E402
from prc2026_timeline.bundle import load_bundle  # noqa: E402
from run_adsb_learner import ADSB_NAMES, BUNDLE_NAMES, RANKING_MONTHS, TRAINING_MONTHS, adsb_table, attach, encode, fit, source_file  # noqa: E402

DECEMBER = "2025-12"
DECEMBER_FILE = source_file(DECEMBER)
DECEMBER_BUILD = Path("data/december/d198dea10625b0a75977")
MATCHES = Path("data/external/adsblol/matches")
V19 = Path("data/external/adsblol/features_v19")
OWN_DIR = Path("data/external/adsblol/features_v19own")
OWN = ["p_has_inb", "p_inb_rel", "p_tow", "p_n", "p_first_any", "p_first_mv", "p_last_park", "p_park10", "p_park20", "p_park40", "p_pregap_n",
       "p_pregap_last", "p_pregap_maxdist", "p_gap_len", "p_seg_start"]
FOLDS = ["lomo_2025-01", "lomo_2025-07", "jan_nov", "forward", "refit12"]
HELD = {"lomo_2025-01": "2025-01", "lomo_2025-07": "2025-07", "jan_nov": DECEMBER, "refit12": None}


def key_of(source_run: str) -> str:
    return source_run.replace("runs/", "").replace("/", "__")


def code_receipt() -> dict:
    paths = [ROOT / "src/prc2026_adsb/features.py", ROOT / "tools/run_adsb_learner.py", ROOT / "tools/run_tree_own.py"]
    hashes = {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths}
    return {"files": hashes, "sha256": object_hash(hashes)}


def recipe(source_run: str) -> dict:
    """The leaf's frozen recipe from its seasonal / forward fold records (tools/run_adsb_learner_lomo.recipe's rule, repeated)."""
    folds = {f: read_json(ROOT / source_run / f / "run.json") for f in ["seasonal", "forward"]}
    src = folds["seasonal"]["inputs"]
    for f, rec in folds.items():
        for key in ["kind", "target", "parameters", "lightgbm_parameters", "excluded_airports", "max_label", "specialist_airports", "exclude_population", "extra_features", "adsb_features"]:
            if rec["inputs"].get(key) != src.get(key):
                raise ValueError(f"Source folds disagree on {key}.")
    return {"folds": folds, "src": src, "kind": src.get("kind") or "lightgbm", "target": src.get("target") or "label",
            "params": dict(src.get("parameters") or src["lightgbm_parameters"]), "excluded": src.get("excluded_airports") or [],
            "exclude_population": bool(src.get("exclude_population")), "max_label": src.get("max_label"), "specialists": src.get("specialist_airports") or [],
            "extra_names": src.get("extra_features") or [], "rounds": int(np.rint(np.median([rec["best_iteration"] for rec in folds.values()]))),
            "rounds_cap": int(src["rounds_cap"]), "patience": int(src["early_stopping_rounds"])}


def extras(variant: str, base_names: list[str], months: list[str]) -> tuple[pd.DataFrame, list[str], dict]:
    """The variant's extra columns for the given months: ctl = the leaf's own (features_v19), own = those + OWN (features_v19own)."""
    if variant == "ctl":
        if not base_names:
            return pd.DataFrame({ID: pd.Series(dtype="int64")}), [], {}
        directory, names = V19, list(base_names)
    else:
        directory, names = OWN_DIR, [*base_names, *OWN]
    parts, hashes = [], {}
    for m in months:
        path = ROOT / directory / f"features_v19_{m}.parquet"
        table = pd.read_parquet(path)
        if variant == "ctl" and [c for c in table.columns if c != ID] != names:
            raise ValueError(f"{path}: columns differ from the leaf's extra features.")
        missing = [c for c in names if c not in table.columns]
        if missing:
            raise ValueError(f"{path}: missing {missing}.")
        parts.append(table[[ID, *names]])
        hashes[(directory / path.name).as_posix()] = sha256_file(path)
    return pd.concat(parts, ignore_index=True), names, hashes


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--leaf", help="the leaf's source run directory, e.g. runs/adsb_v21/lgb_lirf_s31")
    p.add_argument("--variant", choices=["ctl", "own"])
    p.add_argument("--fold", choices=FOLDS)
    p.add_argument("--output", type=Path, default=Path("runs/tree_own/fits"))
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--build-cache-only", action="store_true")
    a = p.parse_args()
    started = time.time()
    cache = ROOT / a.output / "_features"
    stands = pd.read_parquet(ROOT / MATCHES / "stands.parquet")
    months = [*TRAINING_MONTHS, DECEMBER]
    jobs = [([m], source_file(m), ROOT / MATCHES, stands, cache) for m in months] + [(RANKING_MONTHS, "ranking.parquet", ROOT / MATCHES, stands, cache)]
    if a.build_cache_only:
        with mp.Pool(min(len(jobs), max(1, a.threads // 2))) as pool:
            built = pool.starmap(adsb_table, jobs)
        print({j[1]: len(t) for j, t in zip(jobs, built)}, round(time.time() - started, 1), flush=True)
        return
    if not (a.leaf and a.variant and a.fold):
        raise ValueError("--leaf, --variant and --fold are required.")
    decl = read_json(ROOT / "configs/tree_own_screen_v1.json")
    if not decl.get("declared_at_utc") or a.leaf not in decl["leaves"]:
        raise ValueError("The leaf is not declared (or the declaration is a draft).")
    out = ROOT / a.output / key_of(a.leaf) / a.variant / a.fold
    if (out / "run.json").exists():
        raise ValueError(f"Preserve the existing run: {out}")
    spec = recipe(a.leaf)
    kind, target = spec["kind"], spec["target"]
    if kind != "lightgbm":
        raise ValueError("The screen fits LightGBM leaves only.")
    params = dict(spec["params"])
    params.update({"num_threads": a.threads})
    excluded, exclude_population, max_label, specialists = (spec[k] for k in ["excluded", "exclude_population", "max_label", "specialists"])
    bundle = ROOT / read_json(ROOT / "data/timeline/latest.json")["bundle"]
    manifest, parent, original, catalog = load_bundle(ROOT, bundle)

    def clock_mask(fr: pd.DataFrame) -> np.ndarray:
        mask = fr["nmclock_available"].to_numpy(dtype=float) == 1 if target == "nm_residual" else np.ones(len(fr), dtype=bool)
        if specialists:
            mask &= fr["airport"].isin(specialists).to_numpy()
        return mask

    def keep(fr: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
        mask = clock_mask(fr) & ~fr["airport"].isin(excluded).to_numpy()
        if exclude_population:
            mask &= ~(fr["airport"].eq("LIRF") & fr["nmclock_available"].eq(0) & fr["aircraft_missing"].eq(1)).to_numpy()
        if max_label is not None:
            mask &= labels <= max_label
        return mask

    def offset(fr: pd.DataFrame) -> np.ndarray:
        if target == "nm_residual":
            return fr["nmclock_proxy_s"].to_numpy(dtype=float)
        if target == "adsb_residual":
            motion = fr["adsb_motion_s"].to_numpy(dtype=float)
            return np.where(np.isfinite(motion), motion, 0.0)
        return np.zeros(len(fr))

    with mp.Pool(min(len(jobs), max(1, a.threads // 2))) as pool:
        built = pool.starmap(adsb_table, jobs)
    tables = {job[1]: table for job, table in zip(jobs, built)}
    ext, extra_names, extra_hashes = extras(a.variant, spec["extra_names"], [*months, *RANKING_MONTHS])
    if extra_names:
        for raw_name in list(tables):
            tables[raw_name] = tables[raw_name].merge(ext, on=ID, how="left", validate="one_to_one")
    names = [*BUNDLE_NAMES, *ADSB_NAMES, *extra_names]
    record = {"created_at_utc": None, "declaration": "configs/tree_own_screen_v1.json", "declaration_sha256": sha256_file(ROOT / "configs/tree_own_screen_v1.json"),
              "leaf": a.leaf, "variant": a.variant, "fold": a.fold, "kind": kind, "target": target, "parameters": params, "features": len(names),
              "extra_features": extra_names, "extra_feature_tables_sha256": extra_hashes, "excluded_airports": sorted(excluded), "exclude_population": exclude_population,
              "max_label": max_label, "specialist_airports": sorted(specialists), "seed": spec["src"].get("seed"),
              "source_fold_records_sha256": {f: sha256_file(ROOT / a.leaf / f / "run.json") for f in spec["folds"]},
              "bundle_id": manifest["bundle_id"], "stands_sha256": sha256_file(ROOT / MATCHES / "stands.parquet"),
              "matches_sha256": {m: sha256_file(ROOT / MATCHES / f"matches2_{m}.parquet") for m in months + RANKING_MONTHS}, "code": code_receipt()}

    if a.fold == "forward":
        assignment = pd.read_parquet(parent / "splits" / "forward.parquet")
        parts, truths = {}, {}
        for role in ["fit", "early_stop", "score"]:
            feats, truth = load_partition(bundle, parent, catalog, assignment, role, BUNDLE_NAMES)
            if identity_hash(feats[ID]) != original["folds"]["forward"]["identity_sha256"][role]:
                raise ValueError("Protected split populations changed.")
            parts[role] = attach(feats, tables, catalog)
            truths[role] = aligned_truth(parts[role], truth)[TARGET].to_numpy(dtype=float)
        fit_keep, stop_keep = keep(parts["fit"], truths["fit"]), keep(parts["early_stop"], truths["early_stop"])
        x, vocab = encode(parts["fit"][fit_keep].reset_index(drop=True), names)
        xs, _ = encode(parts["early_stop"][stop_keep].reset_index(drop=True), names, vocab)
        booster, history = fit(kind, params, x, truths["fit"][fit_keep] - offset(parts["fit"])[fit_keep], spec["rounds_cap"], xs,
                               truths["early_stop"][stop_keep] - offset(parts["early_stop"])[stop_keep], spec["patience"])
        del x, xs
        score = parts["score"]
        xt, _ = encode(score, names, vocab)
        prediction = np.where(clock_mask(score), offset(score) + booster.predict(xt), np.nan)
        pred_ids = score[ID].to_numpy()
        record.update({"rounds": int(booster.best), "rounds_rule": "early stopping on the forward fold's October rows (cap and patience of the leaf's records)",
                       "fitted_rows": int(fit_keep.sum()), "early_stop_rows": int(stop_keep.sum()), "predicted_rows": int(len(score)),
                       "early_stop_rmse_curve_tail": history})
    else:
        held_month = HELD[a.fold]
        december = read_json(ROOT / DECEMBER_BUILD / "manifest.json")
        if december["month"] != DECEMBER or december["timeline_bundle_id"] != manifest["bundle_id"]:
            raise ValueError("December build does not belong to this timeline bundle.")
        training = catalog[catalog["source_file"].isin([source_file(m) for m in TRAINING_MONTHS])]
        assignment = pd.DataFrame({ID: training[ID].to_numpy(), "partition": "fit"})
        features, truth = load_partition(bundle, parent, catalog, assignment, "fit", BUNDLE_NAMES)
        frame = attach(features, tables, catalog)
        y = aligned_truth(frame, truth)[TARGET].to_numpy(dtype=float)
        if len(frame) != 1919370:
            raise ValueError("The January-November refit population changed.")
        month = catalog.set_index(ID)[TIME].reindex(frame[ID]).dt.strftime("%Y-%m").to_numpy()
        dec_features = pd.read_parquet(ROOT / DECEMBER_BUILD / "features" / DECEMBER_FILE, columns=[ID, *BUNDLE_NAMES])
        dec_labels = pd.read_parquet(ROOT / DECEMBER_BUILD / "labels" / DECEMBER_FILE).set_index(ID)[TARGET].reindex(dec_features[ID]).to_numpy(dtype=float)
        if identity_hash(dec_features[ID]) != december["ids_sha256"] or np.isnan(dec_labels).any():
            raise ValueError("December feature/label identity changed.")
        dec_frame = dec_features.merge(tables[DECEMBER_FILE], on=ID, how="left", validate="one_to_one")
        if dec_frame["adsb_matched"].isna().any():
            raise ValueError("December ADS-B table does not cover every December row.")
        frame = pd.concat([frame, dec_frame], ignore_index=True)
        y = np.concatenate([y, dec_labels])
        month = np.concatenate([month, np.full(len(dec_frame), DECEMBER)])
        del features, dec_features, dec_frame
        if frame[ID].duplicated().any():
            raise ValueError("Duplicate identities across the twelve months.")
        held = month == held_month if held_month else np.zeros(len(frame), dtype=bool)
        fit_keep = keep(frame, y) & ~held
        if held_month and (month[fit_keep] == held_month).any():
            raise ValueError("A held-month row is in the fit.")
        x, vocab = encode(frame[fit_keep].reset_index(drop=True), names)
        booster, _ = fit(kind, params, x, y[fit_keep] - offset(frame)[fit_keep], spec["rounds"])
        del x
        if held_month:
            target_frame = frame.loc[held].reset_index(drop=True)
        else:
            ranking = pd.read_parquet(bundle / "features" / "ranking.parquet", columns=[ID, *BUNDLE_NAMES])
            if identity_hash(ranking[ID]) != original["ranking_ids_sha256"]:
                raise ValueError("Ranking identities changed.")
            target_frame = ranking.merge(tables["ranking.parquet"], on=ID, how="left", validate="one_to_one")
            if target_frame["adsb_matched"].isna().any():
                raise ValueError("ADS-B table does not cover every ranking row.")
        xh, _ = encode(target_frame, names, vocab)
        prediction = np.where(clock_mask(target_frame), offset(target_frame) + booster.predict(xh), np.nan)
        pred_ids = target_frame[ID].to_numpy()
        record.update({"rounds": spec["rounds"], "rounds_rule": "median best iteration of the leaf's recorded seasonal and forward folds (the twelve-month rule)",
                       "held_month": held_month, "fit_months": sorted(set(month[fit_keep])), "population_rows": int(len(frame)), "fitted_rows": int(fit_keep.sum()),
                       "fit_ids_sha256": identity_hash(pd.Series(frame.loc[fit_keep, ID].to_numpy())), "predicted_rows": int(len(target_frame))})
    out.mkdir(parents=True, exist_ok=True)
    booster.save(out / "model.txt")
    pd.DataFrame({ID: pred_ids, "adsb_full": prediction}).to_parquet(out / "predictions.parquet", index=False, compression="zstd")
    record.update({"created_at_utc": utc_now(), "predicted_ids_sha256": identity_hash(pd.Series(pred_ids)), "predicted_finite_rows": int(np.isfinite(prediction).sum()),
                   "prediction_quantiles": {str(q): float(np.nanquantile(prediction, q)) for q in [0.01, 0.1, 0.5, 0.9, 0.99]},
                   "model_sha256": sha256_file(out / "model.txt"), "predictions_sha256": sha256_file(out / "predictions.parquet"),
                   "feature_importance_top30": dict(sorted(zip(booster.feature_names(), booster.importance()), key=lambda kv: -kv[1])[:30]),
                   "audit_used": False, "seconds": round(time.time() - started, 1), "environment": environment(ROOT)})
    write_json(out / "run.json", record)
    print({k: record[k] for k in ["leaf", "variant", "fold", "rounds", "fitted_rows", "predicted_rows", "predicted_finite_rows", "seconds"]}, flush=True)


if __name__ == "__main__":
    main()
