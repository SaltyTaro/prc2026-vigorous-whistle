"""Declared evaluation of adsb_v28: the blend weight on LIRF non-population rows WITHOUT an ADS-B match.

No model is fitted. Every candidate is the shipped v23 recipe with one number changed — the weight of
`ens_v22_lirf_q` on the `lirf_nonpop_unmatched` scope — rebuilt from artifacts that already exist:
the cross-fitted v9 frames (scratch/v24), the v23 residual columns (runs/adsb_v25/dev), and the committee
and LIRF learner development tables. Two readings:

  --score    every candidate on all twelve months: RMSE on the touched scope and on all rows of the month
  --select   the pre-registered rule over those twelve months

Nothing is uploaded.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import environment, object_hash, read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET, TIME  # noqa: E402
from run_adsb_v11 import candidate_values, check_parents, rmse, scope_mask  # noqa: E402
from run_adsb_v24 import ALL_MONTHS, adsb_features, extra_columns, load_frame, month_table  # noqa: E402


def paired_day_interval(days, label, candidate, baseline, mask, seed, draws=2000):
    """95% interval of the candidate-minus-baseline RMSE difference on `mask`, resampling whole days."""
    keep = np.flatnonzero(mask)
    if len(keep) == 0:
        return [float("nan"), float("nan")]
    d, y, c, b = days[keep], label[keep], candidate[keep], baseline[keep]
    unique = np.unique(d)
    index = {day: np.flatnonzero(d == day) for day in unique}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(draws):
        picked = rng.choice(unique, size=len(unique), replace=True)
        rows = np.concatenate([index[day] for day in picked])
        out.append(rmse(y[rows], c[rows]) - rmse(y[rows], b[rows]))
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def spec_of(config: dict, weight: float) -> dict:
    extra = []
    for blend in config["extra_blends"]:
        entry = dict(blend)
        if entry["scope"] == config["touched_scope"]:
            entry["weight"] = weight
        extra.append(entry)
    return {"weight": 1.0, "scope": config["residual_scope"], "blend_with": config["blend_with"], "extra_blends": extra}


def score(config: dict, out_dir: Path, cache: Path) -> None:
    matches_root = ROOT / config["matches_dirs"]["v11"]
    out_dir.mkdir(parents=True, exist_ok=True)
    control = config["control_candidate"]
    for month in ALL_MONTHS:
        target = out_dir / f"month_{month}.json"
        if target.exists():
            print(month, "exists", flush=True)
            continue
        table = month_table(load_frame(config, month, "cross_fit"), adsb_features(month, matches_root, cache), month)
        ids = table[ID].to_numpy()
        label = table[TARGET].to_numpy(dtype=float)
        v9 = table["v9"].to_numpy(dtype=float)
        floor = float(table["floor"].iloc[0])
        days = table[TIME].dt.strftime("%Y-%m-%d").to_numpy()
        for name, values in extra_columns(config, month, ids).items():
            table[name] = values
        columns = pd.read_parquet(ROOT / config["residual_columns"][month]).set_index(ID).reindex(ids)
        residual = np.mean([columns[f"residual:{m}"].to_numpy(dtype=float) for m in config["residual_members"]], axis=0)
        if not np.isfinite(residual).all():
            raise ValueError(f"Residual columns do not cover {month}.")
        touched = np.asarray(scope_mask(table, config["touched_scope"]))
        report = {"month": month, "rows": int(len(table)), "rows_touched": int(touched.sum()), "floor_seconds": floor, "candidates": {}}
        baseline = None
        for candidate, weight in config["candidates"].items():
            values, _ = candidate_values(table, v9, residual, spec_of(config, float(weight)), floor)
            if candidate == control:
                baseline = values
            report["candidates"][candidate] = {"weight": float(weight), "rmse_all_rows": rmse(label, values),
                                               "rmse_touched_scope": rmse(label[touched], values[touched]) if touched.any() else None,
                                               "sse_touched_scope": float(np.square(label[touched] - values[touched]).sum()) if touched.any() else 0.0}
        if baseline is None:
            raise ValueError("The control candidate is not declared.")
        for candidate, entry in report["candidates"].items():
            values, _ = candidate_values(table, v9, residual, spec_of(config, float(entry["weight"])), floor)
            entry["ratio_all_rows"] = entry["rmse_all_rows"] / report["candidates"][control]["rmse_all_rows"]
            entry["rows_differing_from_control"] = int((np.abs(values - baseline) > 1e-9).sum())
            entry["paired_day_interval_touched"] = paired_day_interval(days, label, values, baseline, touched, config["seed"])
        report["inputs"] = {"scratch": sha256_file(ROOT / config["cross_fit"]["scratch"] / f"{month}.parquet"),
                            "residual_columns": sha256_file(ROOT / config["residual_columns"][month])}
        report["config_sha256"] = object_hash(config)
        write_json(target, report)
        print(month, "touched", report["rows_touched"],
              {c: round(v["rmse_touched_scope"], 3) if v["rmse_touched_scope"] else None for c, v in report["candidates"].items()}, flush=True)


def select(config: dict, out_dir: Path, report_path: Path) -> None:
    if (out_dir / "selection.json").exists():
        raise ValueError("Preserve the original selection.")
    reports = {m: read_json(out_dir / f"month_{m}.json") for m in ALL_MONTHS}
    for month, report in reports.items():
        if report["config_sha256"] != object_hash(config):
            raise ValueError(f"{month} was scored under another declaration.")
    control = config["control_candidate"]
    recipe = config["recipe_months"]
    candidates = {}
    for name in config["candidates"]:
        sse = sum(reports[m]["candidates"][name]["sse_touched_scope"] for m in ALL_MONTHS)
        rows = sum(reports[m]["rows_touched"] for m in ALL_MONTHS)
        entry = {"weight": reports[ALL_MONTHS[0]]["candidates"][name]["weight"],
                 "pooled_scope_rmse": float(np.sqrt(sse / rows)), "pooled_scope_rows": rows,
                 "months": {m: {"scope_rmse": reports[m]["candidates"][name]["rmse_touched_scope"],
                                "all_rows_rmse": reports[m]["candidates"][name]["rmse_all_rows"],
                                "ratio_all_rows": reports[m]["candidates"][name]["ratio_all_rows"],
                                "paired_day_interval_touched": reports[m]["candidates"][name]["paired_day_interval_touched"]} for m in ALL_MONTHS}}
        candidates[name] = entry
    control_pooled = candidates[control]["pooled_scope_rmse"]
    for name, entry in candidates.items():
        entry["pooled_scope_ratio"] = entry["pooled_scope_rmse"] / control_pooled
        worst_recipe = max(entry["months"][m]["ratio_all_rows"] for m in recipe)
        touched_months = [m for m in ALL_MONTHS if reports[m]["candidates"][name]["rows_differing_from_control"] > 0]
        below_zero = [m for m in touched_months if entry["months"][m]["paired_day_interval_touched"][1] < 0]
        entry["worst_recipe_month_ratio"] = worst_recipe
        entry["months_touched"] = touched_months
        entry["months_with_interval_below_zero"] = below_zero
        entry["eligible"] = bool(name != control
                                 and worst_recipe <= 1.0
                                 and entry["pooled_scope_ratio"] <= 1 - config["minimum_scope_improvement_fraction"]
                                 and len(below_zero) >= config["minimum_months_with_interval_below_zero"])
    eligible = [n for n, e in candidates.items() if e["eligible"]]
    selected = min(eligible, key=lambda n: (candidates[n]["pooled_scope_rmse"], n)) if eligible else None
    report = {"created_at_utc": utc_now(), "configuration": config, "config_sha256": object_hash(config), "audit_used": False,
              "rule": config["selection"], "control_pooled_scope_rmse": control_pooled, "candidates": candidates,
              "selected_candidate": selected,
              "model_code": {p: sha256_file(ROOT / p) for p in ["tools/run_adsb_v28.py", "tools/run_adsb_v24.py", "tools/run_adsb_v11.py"]},
              "environment": environment(ROOT)}
    for path in [out_dir / "selection.json", report_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, report)
    print({"selected": selected, "control_scope": round(control_pooled, 4),
           "candidates": {n: {"weight": e["weight"], "scope_pct": round(100 * (e["pooled_scope_ratio"] - 1), 4),
                              "worst_recipe_pct": round(100 * (e["worst_recipe_month_ratio"] - 1), 4),
                              "intervals_below_zero": f"{len(e['months_with_interval_below_zero'])}/{len(e['months_touched'])}",
                              "eligible": e["eligible"]} for n, e in candidates.items()}})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/adsb_v28_v1.json"))
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/adsb_v28/dev"))
    parser.add_argument("--report", type=Path, default=Path("docs/adsb_v28/development_results.json"))
    parser.add_argument("--cache", type=Path, default=Path("runs/adsb_v24/features"))
    args = parser.parse_args()
    config = read_json(ROOT / args.config)
    if config["audit_used"]:
        raise ValueError("Declaration does not respect its own contract.")
    check_parents(config)
    if args.score:
        score(config, ROOT / args.output, ROOT / args.cache)
    if args.select:
        select(config, ROOT / args.output, ROOT / args.report)


if __name__ == "__main__":
    main()
