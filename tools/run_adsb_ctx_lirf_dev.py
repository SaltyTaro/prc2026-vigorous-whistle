"""Declared development of adsb_ctx round B (configs/adsb_ctx_lirf_v1.json): the LIRF-inclusive context network in v27's LIRF blends.

No model is fitted here. As tools/run_adsb_ctx_dev2.py (its merged configuration and selection rule), except that every candidate keeps
v27's committee mix (ens_ctx, share 0.65) and changes only the LIRF blends' network table and per-scope (v9, tree, net) weights, through
tools/run_adsb_nn2_dev.lirf_blends. Only LIRF non-population rows can change.

  --score   the control (v27) must reproduce its recorded development RMSE to 1e-9 s on every month; every candidate is scored on all rows
  --select  the pre-registered rule (unchanged)
  --smoke   with --score: ens_ctx_lirf replaced by ens_nnl, into a scratch output (checks the control reproduction and code path)
Nothing is uploaded.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import environment, object_hash, read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET, TIME  # noqa: E402
from run_adsb_ctx_dev2 import merged_config  # noqa: E402
from run_adsb_nn_dev import mixed  # noqa: E402
from run_adsb_nn2_dev import check_ensembles, lirf_blends, spec_for, weighted  # noqa: E402
from run_adsb_rnn_dev import check_networks, combine, residual_columns  # noqa: E402
from run_adsb_v11 import candidate_values, check_parents, paired_days, rmse  # noqa: E402
from run_adsb_v24 import adsb_features, extra_columns, load_frame, month_table  # noqa: E402

CODE = ["tools/run_adsb_ctx_lirf_dev.py", "tools/run_adsb_ctx_dev2.py", "tools/run_adsb_rnn_dev.py", "tools/run_adsb_nn2_dev.py", "tools/run_adsb_nn_dev.py",
        "tools/run_adsb_v24.py", "tools/run_adsb_v11.py"]


def score(decl: dict, out_dir: Path, cache: Path, smoke: bool = False) -> None:
    config = merged_config(decl, drop=tuple(decl["new_networks"]) if smoke else ())
    v26 = config["candidates"]["rnet_w90"]
    v25 = config["v25_recipe"]
    control = decl["control"]
    candidates = decl["candidates"]
    if candidates["control"] != {"lirf": control["lirf"], "lirf_network": control["lirf_network"]} or control["lirf"] != v25["lirf"] or control["lirf_network"] != v25["lirf_network"]:
        raise ValueError("The control candidate is not v27's LIRF blends.")
    ensemble_check = {"smoke": "ens_ctx_lirf replaced by ens_nnl"} if smoke else {"committee": check_ensembles(config), "residual_networks": check_networks(config)}
    matches_root = ROOT / config["matches_dirs"]["v11"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for month in decl["recipe_months"]:
        target = out_dir / f"month_{month}.json"
        if target.exists():
            print(month, "exists", flush=True)
            continue
        table = month_table(load_frame(config, month, "cross_fit"), adsb_features(month, matches_root, cache), month)
        ids = table[ID]
        label, v9 = table[TARGET].to_numpy(dtype=float), table["v9"].to_numpy(dtype=float)
        residual = combine(v26, residual_columns(config, month, ids, smoke=False), "rnet_w90")
        floor = float(table["floor"].iloc[0])
        days = table[TIME].dt.strftime("%Y-%m-%d").to_numpy()
        for name, values in extra_columns(config, month, ids.to_numpy()).items():
            table[name] = values
        if smoke:
            for name in decl["new_networks"]:
                table[name] = table["ens_nnl"]
        for name in decl["networks"]:
            if not np.isfinite(table[name].to_numpy(dtype=float)).all():
                raise ValueError(f"{name} does not cover every row of {month}.")
        committee = table[config["committee_table"]].to_numpy(dtype=float)
        table["committee_mix:v27"] = mixed(committee, weighted(table, control["committee_network"]), float(control["committee_share"]))
        values_by = {}
        for name, candidate in candidates.items():
            lirf = lirf_blends(config, table, {**v25, "lirf": candidate["lirf"], "lirf_network": candidate["lirf_network"]}, name)
            values_by[name], _ = candidate_values(table, v9, residual, spec_for(config, "committee_mix:v27", lirf), floor)
        base = values_by["control"]
        recorded = control["recorded_development_rmse"][month]
        if abs(rmse(label, base) - recorded) > 1e-9:
            raise ValueError(f"The control does not reproduce v27's development RMSE on {month}: {rmse(label, base)} vs {recorded}.")
        lirf_rows = table["airport"].eq("LIRF").to_numpy()
        matched = table["adsb_matched"].eq(1).to_numpy()
        airports = table["airport"].to_numpy()
        report = {"month": month, "rows": int(len(table)), "smoke": smoke, "floor_seconds": floor, "control_rmse": rmse(label, base), "candidates": {}}
        for name, values in values_by.items():
            changed = np.abs(values - base) > 1e-9
            if (changed & ~lirf_rows).any():
                raise ValueError(f"{name} changes a non-LIRF row on {month}.")
            entry = {**candidates[name], "rmse": rmse(label, values), "ratio_to_control": rmse(label, values) / report["control_rmse"],
                     "rows_differing_from_control": int(changed.sum()), "max_abs_difference_from_control": float(np.max(np.abs(values - base))),
                     "paired_days_vs_control": paired_days(days, label, values, base, config["seed"]),
                     "lirf": {"control": rmse(label[lirf_rows], base[lirf_rows]), "candidate": rmse(label[lirf_rows], values[lirf_rows])},
                     "lirf_matched": {"control": rmse(label[lirf_rows & matched], base[lirf_rows & matched]),
                                      "candidate": rmse(label[lirf_rows & matched], values[lirf_rows & matched])} if (lirf_rows & matched).any() else None,
                     "lirf_unmatched": {"control": rmse(label[lirf_rows & ~matched], base[lirf_rows & ~matched]),
                                        "candidate": rmse(label[lirf_rows & ~matched], values[lirf_rows & ~matched])} if (lirf_rows & ~matched).any() else None,
                     "airports": {a: {"rows": int((airports == a).sum()), "control": rmse(label[airports == a], base[airports == a]),
                                      "candidate": rmse(label[airports == a], values[airports == a])} for a in sorted(set(airports))}}
            report["candidates"][name] = entry
        report["inputs"] = {"scratch": sha256_file(ROOT / config["cross_fit"]["scratch"] / f"{month}.parquet"),
                            "ensembles": {n: ({"december_leaves": sorted(decl["networks"][n]["learners"])} if month == "2025-12" else
                                              sha256_file(ROOT / config["extra_feature_tables"][n]["development"][0 if month in ("2025-01", "2025-07") else 1]))
                                          for n in decl["networks"] if not (smoke and n in decl["new_networks"])}}
        report["ensemble_check"] = ensemble_check
        report["declaration_sha256"] = object_hash(decl)
        write_json(target, report)
        print(month, {"control": round(report["control_rmse"], 4), "recipe_pct": {n: round(100 * (e["ratio_to_control"] - 1), 4) for n, e in report["candidates"].items()},
                      "lirf": {n: (round(e["lirf"]["control"], 2), round(e["lirf"]["candidate"], 2)) for n, e in report["candidates"].items()}}, flush=True)


def select(decl: dict, out_dir: Path, report_path: Path) -> None:
    if (out_dir / "selection.json").exists():
        raise ValueError("Preserve the original selection.")
    months = decl["recipe_months"]
    reports = {m: read_json(out_dir / f"month_{m}.json") for m in months}
    for month, report in reports.items():
        if report.get("smoke") or report["declaration_sha256"] != object_hash(decl):
            raise ValueError(f"{month} was scored under another declaration.")
    candidates = {}
    for name in decl["candidates"]:
        entry = {**decl["candidates"][name], "months": {m: reports[m]["candidates"][name] for m in months}}
        ratios = [entry["months"][m]["ratio_to_control"] for m in months]
        entry["mean_ratio"] = float(np.mean(ratios))
        touched = [m for m in months if entry["months"][m]["rows_differing_from_control"] > 0]
        below = [m for m in touched if entry["months"][m]["paired_days_vs_control"]["rmse_difference_95pct_interval_seconds"][1] < 0]
        entry["months_touched"], entry["months_with_interval_below_zero"] = touched, below
        entry["eligible"] = bool(name != "control" and touched and all(r <= 1.0 for r in ratios)
                                 and entry["mean_ratio"] <= 1 - decl["minimum_mean_improvement_fraction"] and len(below) == len(touched))
        candidates[name] = entry
    eligible = [n for n, e in candidates.items() if e["eligible"]]
    selected = min(eligible, key=lambda n: (candidates[n]["mean_ratio"], n)) if eligible else None
    report = {"created_at_utc": utc_now(), "declaration": decl, "declaration_sha256": object_hash(decl), "audit_used": False, "rule": decl["selection"],
              "months": {m: {"rows": reports[m]["rows"], "control_rmse": reports[m]["control_rmse"]} for m in months},
              "candidates": candidates, "selected_candidate": selected, "model_code": {p: sha256_file(ROOT / p) for p in CODE}, "environment": environment(ROOT)}
    for path in [out_dir / "selection.json", report_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, report)
    print({"selected": selected})
    for n, e in candidates.items():
        print(f"{n:16s} mean {100 * (e['mean_ratio'] - 1):+.4f}% " + " ".join(f"{m}:{100 * (e['months'][m]['ratio_to_control'] - 1):+.4f}" for m in months)
              + f" intervals<0 {len(e['months_with_interval_below_zero'])}/{len(e['months_touched'])} eligible={e['eligible']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, default=Path("configs/adsb_ctx_lirf_v1.json"))
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/adsb_ctx/dev_lirf"))
    parser.add_argument("--report", type=Path, default=Path("docs/adsb_ctx_lirf/development_results.json"))
    parser.add_argument("--cache", type=Path, default=Path("runs/adsb_v24/features"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    decl = read_json(ROOT / args.declaration)
    if decl["audit_used"]:
        raise ValueError("Declaration does not respect its own contract.")
    check_parents(read_json(ROOT / "configs/adsb_rnn_v1.json"))
    if args.smoke and (args.select or not args.score or (ROOT / args.output).resolve() == (ROOT / "runs/adsb_ctx/dev_lirf").resolve()):
        raise ValueError("--smoke scores only, into a scratch output directory.")
    if args.score:
        score(decl, ROOT / args.output, ROOT / args.cache, smoke=args.smoke)
    if args.select:
        select(decl, ROOT / args.output, ROOT / args.report)


if __name__ == "__main__":
    main()
