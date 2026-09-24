"""The incumbent committee (ens_v20_bag_q_all) on October 2025, rebuilt from its members' seasonal-fold models: the design reference.

Every member's seasonal-fold model (fitted on 2025-02..06, 08, 09; early-stopped on October) predicts the seasonal early-stop
partition (October) through prc2026_december.scoring.learner_prediction, the path that reproduced the December tables. The committee
is the member mean with the recorded multiplicities. Before anything is written, one LightGBM and one CatBoost member re-predict
the first 20,000 January rows of the seasonal score partition, which must equal their saved fold predictions exactly (label-free:
no January label is read). Output: runs/adsb_nn/design/committee_october.parquet (ID, label, airport, adsb_matched, one column per
leaf, committee). October labels are read; they are the early-stop labels every member already used. Nothing is uploaded.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import environment, read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET, TIME  # noqa: E402
from prc2026.validation import identity_hash  # noqa: E402
from prc2026_december.scoring import VocabularyCache, ensemble_prediction, leaf_members, learner_prediction  # noqa: E402
from prc2026_phase3.bundle import load_partition  # noqa: E402
from prc2026_phase5r.baseline import aligned_truth  # noqa: E402
from prc2026_timeline.bundle import load_bundle  # noqa: E402
from run_adsb_learner import BUNDLE_NAMES, adsb_table, attach, source_file  # noqa: E402

COMMITTEE = "runs/adsb_v20/ens_v20_bag_q_all"
V19 = ["adsb_appear_s", "adsb_last_stand_s", "adsb_pre_motion_gap_s", "adsb_first_motion_s", "adsb_first_dist_stand_imp_m", "adsb_stand_imputed",
       "sfc_ground_10m", "sfc_moving_10m", "sfc_moving_20m", "sfc_rwy_queue_5m", "sfc_rwy_queue_10m"]


def frame_for(months: list[str], role: str, base, parent, catalog, original, args) -> tuple[pd.DataFrame, np.ndarray | None]:
    matches_root = ROOT / args.matches_dir
    stands = pd.read_parquet(matches_root / "stands.parquet")
    tables = {}
    for m in months:
        table = adsb_table([m], source_file(m), matches_root, stands, ROOT / args.adsb_cache)
        extra = pd.read_parquet(ROOT / args.extra_features / f"features_v19_{m}.parquet")
        tables[source_file(m)] = table.merge(extra, on=ID, how="left", validate="one_to_one")
    assignment = pd.read_parquet(parent / "splits" / "seasonal.parquet")
    ids = set(catalog.loc[catalog["source_file"].isin([source_file(m) for m in months]), ID])
    assignment = assignment[assignment[ID].isin(ids)]
    features, truth = load_partition(base, parent, catalog, assignment, role, BUNDLE_NAMES)
    frame = attach(features, tables, catalog)
    labels = aligned_truth(frame, truth)[TARGET].to_numpy(dtype=float) if role == "early_stop" else None
    return frame, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches-dir", type=Path, default=Path("data/external/adsblol/matches"))
    parser.add_argument("--extra-features", type=Path, default=Path("data/external/adsblol/features_v19"))
    parser.add_argument("--adsb-cache", type=Path, default=Path("runs/adsb_nn/features"))
    parser.add_argument("--output", type=Path, default=Path("runs/adsb_nn/design/committee_october.parquet"))
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    started = time.time()
    if (ROOT / args.output).exists():
        raise ValueError(f"Preserve the existing reference: {args.output}")
    base = ROOT / read_json(ROOT / "data/timeline/latest.json")["bundle"]
    manifest, parent, original, catalog = load_bundle(ROOT, base)
    members = leaf_members(ROOT, COMMITTEE)
    leaves = sorted(set(members))
    vocab = VocabularyCache(ROOT)
    # Label-free reproduction check on January score rows
    january, _ = frame_for(["2025-01"], "score", base, parent, catalog, original, args)
    january = january.iloc[:20000].reset_index(drop=True)
    checks = {}
    for leaf in ["runs/adsb_v20/lgb_x19_s24", "runs/adsb_v20/cat_ctr2_x19_s27"]:
        values, _ = learner_prediction(ROOT, Path(leaf), january, vocab, args.threads, fold="seasonal")
        saved = pd.read_parquet(ROOT / leaf / "seasonal" / "predictions.parquet").set_index(ID)["adsb_full"].reindex(january[ID]).to_numpy(dtype=float)
        checks[leaf] = float(np.max(np.abs(values - saved)))
        if checks[leaf] != 0.0:
            raise ValueError(f"{leaf} does not reproduce its saved seasonal predictions (max difference {checks[leaf]}).")
    print({"reproduction_max_abs_difference": checks}, flush=True)
    del january
    october, labels = frame_for(["2025-10"], "early_stop", base, parent, catalog, original, args)
    if identity_hash(october[ID]) != original["folds"]["seasonal"]["identity_sha256"]["early_stop"]:
        raise ValueError("The October early-stop population changed.")
    out = pd.DataFrame({ID: october[ID].to_numpy(), "label": labels, "airport": october["airport"].to_numpy(), "adsb_matched": october["adsb_matched"].to_numpy(),
                        "day": catalog.set_index(ID)[TIME].reindex(october[ID]).dt.strftime("%Y-%m-%d").to_numpy()})
    predictions, records = {}, {}
    for leaf in leaves:
        t0 = time.time()
        predictions[leaf], records[leaf] = learner_prediction(ROOT, Path(leaf), october, vocab, args.threads, fold="seasonal")
        out[leaf] = predictions[leaf]
        print(leaf, round(time.time() - t0, 1), "s", flush=True)
    out["committee"] = ensemble_prediction(members, predictions)
    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False, compression="zstd")
    keep = out["airport"].ne("LIRF").to_numpy()
    rmse = float(np.sqrt(np.mean((out.loc[keep, "label"] - out.loc[keep, "committee"]) ** 2)))
    write_json(path.with_suffix(".json"), {"created_at_utc": utc_now(), "committee": COMMITTEE, "leaves": leaves, "member_multiplicities": {l: members.count(l) for l in leaves},
                                           "rows": int(len(out)), "non_lirf_rows": int(keep.sum()), "committee_rmse_non_lirf": rmse,
                                           "reproduction_max_abs_difference": checks, "learners": records, "output_sha256": sha256_file(path),
                                           "seconds": round(time.time() - started, 1), "environment": environment(ROOT), "audit_used": False})
    print({"committee_rmse_non_lirf": rmse, "rows": int(keep.sum()), "seconds": round(time.time() - started, 1)}, flush=True)


if __name__ == "__main__":
    main()
