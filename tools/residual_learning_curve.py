"""Learning curve of the v12 residual stage over its training months (read-only diagnostic, no upload).

The shipped residual (v21, `configs/adsb_v23_v1.json`) is fitted on the in-scope rows of the only four months with out-of-sample v9:
January, July, November and December 2025. This diagnostic asks whether that stage is data-limited: for each held-out month it fits
the declared residual (unchanged parameters, scope, feature set and inner rounds rule) on nested subsets of the other three months
and scores (a) the component v9 + residual on the in-scope rows and (b) the incumbent's blend 0.3 * (v9 + residual) + 0.7 * committee
on every non-LIRF row of the held-out month. Row-subsampled controls separate "more rows" from "more months". Labels of the four
months are development labels already used by the shipped selections (December opened on 19 September); no ranking label exists.
Results accumulate in the report after every fit.
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
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json
from prc2026.features import ID, TARGET
from prc2026_december.scoring import ensemble_prediction, leaf_members
from run_adsb_v11 import candidate_values, fit_residual, matches_dir, month_table, predict_residual, rmse, scope_mask

FRAME_OF = {"2025-01": "seasonal", "2025-07": "seasonal", "2025-11": "forward", "2025-12": "december"}
COMMITTEE = "runs/adsb_v20/ens_v20_bag_q_all"


def committee_values(month: str, ids: np.ndarray) -> np.ndarray:
    """The incumbent's committee table for the month: fold predictions for 2025-01/07/11, the cached January-November refit leaves for December."""
    if month == "2025-12":
        members = leaf_members(ROOT, COMMITTEE)
        cache = {}
        for leaf in set(members):
            frame = pd.read_parquet(ROOT / "runs/december/learners" / (leaf.replace("/", "__") + ".parquet")).set_index(ID)["adsb_full"]
            cache[leaf] = frame.reindex(ids).to_numpy(dtype=float)
        return ensemble_prediction(members, cache)
    fold = FRAME_OF[month]
    return pd.read_parquet(ROOT / COMMITTEE / fold / "predictions.parquet").set_index(ID)["adsb_full"].reindex(ids).to_numpy(dtype=float)


def evaluate(table: pd.DataFrame, residual: np.ndarray, config: dict, scope: str) -> dict:
    label, v9, floor = table[TARGET].to_numpy(dtype=float), table["v9"].to_numpy(dtype=float), float(table["floor"].iloc[0])
    inside = scope_mask(table, scope)
    item = {"weight": 1.0, "scope": scope, "blend_with": {"name": "ens_v20_bag_q_all", "scope": "all_no_lirf", "weight": 0.7}, "extra_blends": []}
    blend, _ = candidate_values(table, v9, residual, item, floor)
    blend0, _ = candidate_values(table, v9, np.zeros_like(residual), item, floor)
    no_lirf = scope_mask(table, "all_no_lirf")
    out = {"in_scope_rows": int(inside.sum()), "component_v9": rmse(label[inside], v9[inside]), "component_v9_plus_residual": rmse(label[inside], np.maximum(v9 + residual, floor)[inside]),
           "no_lirf_rows": int(no_lirf.sum()), "blend_without_residual": rmse(label[no_lirf], blend0[no_lirf]), "blend_with_residual": rmse(label[no_lirf], blend[no_lirf])}
    out["component_change_pct"] = 100 * (out["component_v9_plus_residual"] / out["component_v9"] - 1)
    out["blend_change_pct_vs_no_residual"] = 100 * (out["blend_with_residual"] / out["blend_without_residual"] - 1)
    ordinary = no_lirf & (label <= 21600)
    out["blend_with_residual_ordinary"] = rmse(label[ordinary], blend[ordinary])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/adsb_v23_v1.json"))
    parser.add_argument("--report", type=Path, default=Path("docs/adsb_v24/learning_curve.json"))
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--held-out", nargs="*", default=["2025-12", "2025-01", "2025-07", "2025-11"])
    parser.add_argument("--controls", action="store_true", help="add row-subsampled controls of the three-month set (December only)")
    args = parser.parse_args()
    config = read_json(ROOT / args.config)
    config["residual_model"]["params"]["num_threads"] = args.threads
    scope, feature_set = "matched_no_lirf", "nb"
    stands = pd.read_parquet(matches_dir(config) / "stands.parquet")
    frames = {name: pd.read_parquet(ROOT / "scratch" / f"{name}.parquet") for name in ("seasonal", "forward", "december")}
    months = ["2025-01", "2025-07", "2025-11", "2025-12"]
    tables = {}
    for month in months:
        table = month_table(month, frames[FRAME_OF[month]], stands, matches_dir(config), True, [])
        table["ens_v20_bag_q_all"] = committee_values(month, table[ID].to_numpy())
        if not np.isfinite(table.loc[scope_mask(table, "all_no_lirf"), "ens_v20_bag_q_all"]).all():
            raise ValueError(f"Committee table does not cover the non-LIRF rows of {month}.")
        tables[month] = table
    # Nested training sets per held-out month (most adjacent month first), singles and pairs for December.
    plan = {"2025-12": [["2025-11"], ["2025-07"], ["2025-01"], ["2025-07", "2025-11"], ["2025-01", "2025-11"], ["2025-01", "2025-07"], ["2025-01", "2025-07", "2025-11"]],
            "2025-01": [["2025-12"], ["2025-11", "2025-12"], ["2025-07", "2025-11", "2025-12"]],
            "2025-07": [["2025-11"], ["2025-01", "2025-11"], ["2025-01", "2025-11", "2025-12"]],
            "2025-11": [["2025-12"], ["2025-07", "2025-12"], ["2025-01", "2025-07", "2025-12"]]}
    report = {"created_at_utc": utc_now(), "config": str(args.config), "config_sha256": sha256_file(ROOT / args.config), "scope": scope, "feature_set": feature_set, "threads": args.threads,
              "committee": COMMITTEE, "in_scope_rows": {m: int(scope_mask(t, scope).sum()) for m, t in tables.items()}, "runs": []}
    (ROOT / args.report).parent.mkdir(parents=True, exist_ok=True)

    def run(held: str, train_months: list[str], subsample: int | None = None, label: str | None = None):
        train = pd.concat([tables[m] for m in train_months], ignore_index=True)
        if subsample is not None:
            inside = np.flatnonzero(scope_mask(train, scope))
            keep = np.sort(np.random.default_rng(20260920).choice(inside, size=subsample, replace=False))
            train = train.iloc[keep].reset_index(drop=True)
        start = time.time()
        model = fit_residual(train, config, scope, feature_set)
        residual = predict_residual(model, tables[held], config, scope)
        entry = {"held_out": held, "training_months": train_months, "subsample_rows": subsample, "label": label or "+".join(m[5:] for m in train_months), "training_rows": model["rows"],
                 "rounds": model["rounds"], "inner_best_iterations": model["inner_best_iterations"], "seconds": round(time.time() - start, 1), **evaluate(tables[held], residual, config, scope)}
        report["runs"].append(entry)
        write_json(ROOT / args.report, report)
        print(f"{held} <- {entry['label']:<12} rows {entry['training_rows']:>7} rounds {entry['rounds']:>5} component {entry['component_v9']:.3f} -> {entry['component_v9_plus_residual']:.3f} "
              f"({entry['component_change_pct']:+.3f}%) blend {entry['blend_without_residual']:.3f} -> {entry['blend_with_residual']:.3f} ({entry['blend_change_pct_vs_no_residual']:+.3f}%) {entry['seconds']}s", flush=True)

    for held in args.held_out:
        for train_months in plan[held]:
            run(held, train_months)
        if args.controls and held == "2025-12":
            sizes = sorted({int(scope_mask(tables[m], scope).sum()) for m in months if m != held})
            triple = [m for m in months if m != held]
            for size in (min(sizes), int(np.median(sizes))):
                run(held, triple, subsample=size, label=f"three-month set subsampled to {size} rows")
    report["finished_at_utc"] = utc_now()
    write_json(ROOT / args.report, report)


if __name__ == "__main__":
    main()
