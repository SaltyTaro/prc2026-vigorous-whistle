"""Declared development of adsb_v29: the v24 recipe's blend weights re-read now that the committee table carries the network.

No model is fitted here. On each of the four recipe months (2025-01/07/11/12, the only months with out-of-sample committee, LIRF and
network tables) the shipped v24 recipe (adsb_nn candidate nn50) is rebuilt exactly as tools/run_adsb_nn_dev.py rebuilt it --
cross-fitted v9 (scratch/v24), the v23 residual columns (members adj2 and cat), the committee table
(1 - a) x ens_v20_bag_q_all + a x ens_nn, the outer blend with that table on non-LIRF rows, the two LIRF blends, the floor -- with
three declared numbers free per candidate:

  network_share     a, the network's share inside the committee table (v24: 0.5)
  outer_matched     the committee table's weight against v9 + residual on matched non-LIRF rows (v24: 0.7)
  outer_unmatched   the committee table's weight against v9 on unmatched non-LIRF rows, where the residual stage is zero (v24: 0.7)

The control (0.5, 0.7, 0.7) must reproduce v24's recorded development RMSE (runs/adsb_nn/dev/selection.json, nn50) on every month
to 1e-9 s. LIRF rows are identical for every candidate.

  --score    every candidate on the four months (all rows), with paired-day intervals against the control
  --select   the pre-registered rule
  --smoke    with --score: every candidate's numbers replaced by the control's (all collapse onto it), into a scratch output --
             tests the code path and the control reproduction without producing any reading
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
from run_adsb_nn_dev import FOLD_OF, check_network, mixed  # noqa: E402
from run_adsb_v11 import SCOPES, candidate_values, check_parents, paired_days, rmse  # noqa: E402
from run_adsb_v24 import adsb_features, extra_columns, load_frame, month_table  # noqa: E402

# Non-LIRF rows without a matched surface segment: the complement of matched_no_lirf inside all_no_lirf.
SCOPES["unmatched_no_lirf"] = lambda t: t["airport"].ne("LIRF") & t["adsb_matched"].ne(1)
KEYS = ("network_share", "outer_matched", "outer_unmatched")


def spec_for(config: dict, column: str, candidate: dict) -> dict:
    """The residual on its scope, then the committee table on matched and on unmatched non-LIRF rows, then the LIRF blends."""
    return {"weight": 1.0, "scope": config["residual_scope"],
            "blend_with": {"name": column, "scope": "matched_no_lirf", "weight": float(candidate["outer_matched"])},
            "extra_blends": [{"name": column, "scope": "unmatched_no_lirf", "weight": float(candidate["outer_unmatched"])}, *config["extra_blends"]]}


def score(config: dict, out_dir: Path, cache: Path, smoke: bool = False) -> None:
    control = config["control_candidate"]
    candidates = config["candidates"]
    if smoke:
        candidates = {name: dict(candidates[control]) for name in candidates}
    network_check = check_network(config)
    matches_root = ROOT / config["matches_dirs"]["v11"]
    out_dir.mkdir(parents=True, exist_ok=True)
    committee_name, network_name = config["committee_table"], config["network_table"]
    for month in config["development_months"]:
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
        committee, network = table[committee_name].to_numpy(dtype=float), table[network_name].to_numpy(dtype=float)
        non_lirf = table["airport"].ne("LIRF").to_numpy()
        if not np.isfinite(network[non_lirf]).all() or not np.isfinite(committee[non_lirf]).all():
            raise ValueError(f"The committee or network table does not cover every non-LIRF row of {month}.")
        columns = pd.read_parquet(ROOT / config["residual_columns"][month]).set_index(ID).reindex(ids)
        residual = np.mean([columns[f"residual:{m}"].to_numpy(dtype=float) for m in config["residual_members"]], axis=0)
        if not np.isfinite(residual).all():
            raise ValueError(f"Residual columns do not cover {month}.")
        values_by = {}
        for name, candidate in candidates.items():
            column = f"committee_mix:{name}"
            table[column] = mixed(committee, network, float(candidate["network_share"]))
            values_by[name], _ = candidate_values(table, v9, residual, spec_for(config, column, candidate), floor)
        base = values_by[control]
        expected = config["control_rmse"][month]
        if abs(rmse(label, base) - expected) > 1e-9:
            raise ValueError(f"The control does not reproduce v24's development RMSE on {month}: {rmse(label, base)} vs {expected}.")
        matched = table["adsb_matched"].eq(1).to_numpy()
        lirf = ~non_lirf
        airports = table["airport"].to_numpy()
        if smoke:
            report = {"month": month, "smoke": True, "control_rmse": rmse(label, base), "control_reproduces": True,
                      "max_abs_difference_any_candidate_vs_control": float(max(np.max(np.abs(v - base)) for v in values_by.values()))}
            write_json(target, report)
            print(month, report, flush=True)
            continue
        report = {"month": month, "rows": int(len(table)), "fold": FOLD_OF[month], "floor_seconds": floor, "control_rmse": rmse(label, base),
                  "rows_matched_non_lirf": int((matched & non_lirf).sum()), "rows_unmatched_non_lirf": int((~matched & non_lirf).sum()), "candidates": {}}
        for name, values in values_by.items():
            if np.max(np.abs(values[lirf] - base[lirf])) > 0:
                raise ValueError(f"{name} changes a LIRF row on {month}.")
            entry = {**{k: float(candidates[name][k]) for k in KEYS}, "rmse": rmse(label, values), "ratio_to_control": rmse(label, values) / report["control_rmse"],
                     "rows_differing_from_control": int((np.abs(values - base) > 1e-9).sum()),
                     "paired_days_vs_control": paired_days(days, label, values, base, config["seed"]),
                     "matched_non_lirf": {"control": rmse(label[matched & non_lirf], base[matched & non_lirf]), "candidate": rmse(label[matched & non_lirf], values[matched & non_lirf])},
                     "unmatched_non_lirf": {"control": rmse(label[~matched & non_lirf], base[~matched & non_lirf]), "candidate": rmse(label[~matched & non_lirf], values[~matched & non_lirf])},
                     "airports": {}}
            for airport in sorted(set(airports)):
                k = airports == airport
                entry["airports"][airport] = {"rows": int(k.sum()), "control": rmse(label[k], base[k]), "candidate": rmse(label[k], values[k]),
                                              "change_pct": 100 * (rmse(label[k], values[k]) / rmse(label[k], base[k]) - 1)}
            report["candidates"][name] = entry
        report["inputs"] = {"scratch": sha256_file(ROOT / config["cross_fit"]["scratch"] / f"{month}.parquet"), "residual_columns": sha256_file(ROOT / config["residual_columns"][month])}
        report["network_check"] = network_check
        report["config_sha256"] = object_hash(config)
        write_json(target, report)
        print(month, {"control": round(report["control_rmse"], 4)}, {n: round(100 * (e["ratio_to_control"] - 1), 4) for n, e in report["candidates"].items()}, flush=True)


def select(config: dict, out_dir: Path, report_path: Path) -> None:
    if (out_dir / "selection.json").exists():
        raise ValueError("Preserve the original selection.")
    months = config["development_months"]
    reports = {m: read_json(out_dir / f"month_{m}.json") for m in months}
    for month, report in reports.items():
        if report.get("smoke") or report["config_sha256"] != object_hash(config):
            raise ValueError(f"{month} was scored under another declaration.")
    control = config["control_candidate"]
    candidates = {}
    for name in config["candidates"]:
        entry = {**{k: float(config["candidates"][name][k]) for k in KEYS}, "months": {m: reports[m]["candidates"][name] for m in months}}
        ratios = [entry["months"][m]["ratio_to_control"] for m in months]
        entry["mean_ratio"] = float(np.mean(ratios))
        touched = [m for m in months if entry["months"][m]["rows_differing_from_control"] > 0]
        below = [m for m in touched if entry["months"][m]["paired_days_vs_control"]["rmse_difference_95pct_interval_seconds"][1] < 0]
        entry["months_touched"], entry["months_with_interval_below_zero"] = touched, below
        entry["eligible"] = bool(name != control and touched and all(r <= 1.0 for r in ratios)
                                 and entry["mean_ratio"] <= 1 - config["minimum_mean_improvement_fraction"] and len(below) == len(touched))
        candidates[name] = entry
    eligible = [n for n, e in candidates.items() if e["eligible"]]
    selected = min(eligible, key=lambda n: (candidates[n]["mean_ratio"], n)) if eligible else None
    report = {"created_at_utc": utc_now(), "configuration": config, "config_sha256": object_hash(config), "audit_used": False, "rule": config["selection"],
              "months": {m: {k: reports[m][k] for k in ["rows", "fold", "control_rmse", "rows_matched_non_lirf", "rows_unmatched_non_lirf"]} for m in months},
              "candidates": candidates, "selected_candidate": selected,
              "model_code": {p: sha256_file(ROOT / p) for p in ["tools/run_adsb_v29.py", "tools/run_adsb_nn_dev.py", "tools/run_adsb_v24.py", "tools/run_adsb_v11.py"]},
              "environment": environment(ROOT)}
    for path in [out_dir / "selection.json", report_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, report)
    print({"selected": selected})
    for n, e in candidates.items():
        print(f"{n:14s} a={e['network_share']:.2f} wm={e['outer_matched']:.2f} wu={e['outer_unmatched']:.2f} mean {100 * (e['mean_ratio'] - 1):+.4f}% "
              + " ".join(f"{m}:{100 * (e['months'][m]['ratio_to_control'] - 1):+.4f}" for m in months)
              + f" intervals<0 {len(e['months_with_interval_below_zero'])}/{len(e['months_touched'])} eligible={e['eligible']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/adsb_v29/dev"))
    parser.add_argument("--report", type=Path, default=Path("docs/adsb_v29/development_results.json"))
    parser.add_argument("--cache", type=Path, default=Path("runs/adsb_v24/features"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = read_json(ROOT / args.config)
    if config["audit_used"]:
        raise ValueError("Declaration does not respect its own contract.")
    check_parents(config)
    control = config["candidates"][config["control_candidate"]]
    if [control[k] for k in KEYS] != [0.5, 0.7, 0.7]:
        raise ValueError("The control must be the shipped v24 recipe (0.5, 0.7, 0.7).")
    if args.smoke and (args.select or not args.score or (ROOT / args.output).resolve() == (ROOT / "runs/adsb_v29/dev").resolve()):
        raise ValueError("--smoke scores only, into a scratch output directory.")
    if args.score:
        score(config, ROOT / args.output, ROOT / args.cache, smoke=args.smoke)
    if args.select:
        select(config, ROOT / args.output, ROOT / args.report)


if __name__ == "__main__":
    main()
