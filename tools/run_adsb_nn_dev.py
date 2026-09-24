"""Declared development of adsb_nn: the neural committee member mixed into the incumbent committee inside the shipped v23 recipe.

No model is fitted here. For each of the four recipe months (2025-01/07/11/12, the only months with out-of-sample committee
tables) the v23 recipe is rebuilt exactly as tools/run_adsb_v28.py rebuilt it -- cross-fitted v9 (scratch/v24), the v23 residual
columns (runs/adsb_v25/dev, members adj2 and cat), 0.7 x committee on non-LIRF rows, the two LIRF blends, the floor -- except that
the committee table is replaced by (1 - a) x ens_v20_bag_q_all + a x the network ensemble for each declared a. a = 0 is the
control and must reproduce v23's recorded development RMSE on every month (runs/adsb_v28/dev, candidate w25) to 1e-9 s.
The network tables are fold outputs like the committee's: seasonal fold for January/July, forward fold for November, the
January-November refit on December (read from the December leaf cache).

  --score    every candidate on the four months (all rows), with paired-day intervals against the control
  --select   the pre-registered rule
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
from prc2026_december.scoring import leaf_members  # noqa: E402
from run_adsb_v11 import candidate_values, check_parents, paired_days, rmse  # noqa: E402
from run_adsb_v24 import adsb_features, extra_columns, load_frame, month_table  # noqa: E402

FOLD_OF = {"2025-01": "seasonal", "2025-07": "seasonal", "2025-11": "forward", "2025-12": "refit"}


def check_network(config: dict) -> dict:
    """Every leaf of the network ensemble must be a declared seed of the declared variant, fitted by the declared code, on every fold."""
    spec = config["network"]
    table = config["extra_feature_tables"][config["network_table"]]
    leaves = leaf_members(ROOT, table["run_dir"])
    if sorted(leaves) != sorted(spec["learners"]) or len(set(leaves)) != len(leaves):
        raise ValueError("The network ensemble does not hold exactly the declared learners.")
    seen = {}
    for leaf in leaves:
        for fold in ["seasonal", "forward", "refit"]:
            record = read_json(ROOT / leaf / fold / "run.json")
            if record["variant_sha256"] != spec["variant_sha256"] or record["inputs"]["code"]["sha256"] != spec["code_sha256"] or record["fold"] != fold:
                raise ValueError(f"{leaf}/{fold} was not fitted with the declared variant and code.")
            if record["config_sha256"] != spec["config_sha256"]:
                raise ValueError(f"{leaf}/{fold} used another network configuration file.")
            seen.setdefault(leaf, record["seed"])
    if sorted(seen.values()) != sorted(spec["seeds"]):
        raise ValueError("The network seeds differ from the declaration.")
    return {"leaves": leaves, "seeds": seen}


def mixed(committee: np.ndarray, network: np.ndarray, a: float) -> np.ndarray:
    both = np.isfinite(committee) & np.isfinite(network)
    return np.where(both, (1 - a) * committee + a * network, committee)


def spec_for(config: dict, column: str) -> dict:
    return {"weight": 1.0, "scope": config["residual_scope"], "blend_with": {"name": column, **config["recipe_blend"]}, "extra_blends": config["extra_blends"]}


def score(config: dict, out_dir: Path, cache: Path, smoke: bool = False) -> None:
    """`smoke`: the network table is replaced by the committee's own table (every candidate collapses onto the control), which tests the
    code path and the control reproduction without producing any reading of the network."""
    if smoke:
        config = {**config, "extra_feature_tables": {**config["extra_feature_tables"], config["network_table"]: config["extra_feature_tables"][config["committee_table"]]}}
        network_check = {"smoke": "network table replaced by the committee table"}
    else:
        network_check = check_network(config)
    matches_root = ROOT / config["matches_dirs"]["v11"]
    out_dir.mkdir(parents=True, exist_ok=True)
    control = config["control_candidate"]
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
        if not np.isfinite(network[non_lirf]).all():
            raise ValueError(f"The network table does not cover every non-LIRF row of {month}.")
        columns = pd.read_parquet(ROOT / config["residual_columns"][month]).set_index(ID).reindex(ids)
        residual = np.mean([columns[f"residual:{m}"].to_numpy(dtype=float) for m in config["residual_members"]], axis=0)
        if not np.isfinite(residual).all():
            raise ValueError(f"Residual columns do not cover {month}.")
        values_by = {}
        for name, weight in config["candidates"].items():
            column = f"committee_mix:{name}"
            table[column] = mixed(committee, network, float(weight))
            values_by[name], _ = candidate_values(table, v9, residual, spec_for(config, column), floor)
        base = values_by[control]
        expected = config["control_rmse"][month]
        if abs(rmse(label, base) - expected) > 1e-9:
            raise ValueError(f"The control does not reproduce v23's development RMSE on {month}: {rmse(label, base)} vs {expected}.")
        matched = table["adsb_matched"].eq(1).to_numpy()
        airports = table["airport"].to_numpy()
        report = {"month": month, "rows": int(len(table)), "fold": FOLD_OF[month], "floor_seconds": floor, "control_rmse": rmse(label, base),
                  "network_rmse_non_lirf": rmse(label[non_lirf], network[non_lirf]), "committee_rmse_non_lirf": rmse(label[non_lirf], committee[non_lirf]),
                  "network_committee_error_correlation_non_lirf": float(np.corrcoef(label[non_lirf] - committee[non_lirf], label[non_lirf] - network[non_lirf])[0, 1]),
                  "candidates": {}}
        for name, values in values_by.items():
            entry = {"weight": float(config["candidates"][name]), "rmse": rmse(label, values), "ratio_to_control": rmse(label, values) / report["control_rmse"],
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
        report["config_sha256"] = object_hash(config) if not smoke else "smoke"
        if smoke:
            report = {"month": month, "smoke": True, "control_rmse": report["control_rmse"], "control_reproduces": True,
                      "max_abs_difference_any_candidate_vs_control": float(max(np.max(np.abs(v - base)) for v in values_by.values()))}
        write_json(target, report)
        if smoke:
            print(month, report, flush=True)
            continue
        print(month, {"control": round(report["control_rmse"], 4), "network_non_lirf": round(report["network_rmse_non_lirf"], 3), "committee_non_lirf": round(report["committee_rmse_non_lirf"], 3)},
              {n: round(100 * (e["ratio_to_control"] - 1), 4) for n, e in report["candidates"].items()}, flush=True)


def select(config: dict, out_dir: Path, report_path: Path) -> None:
    if (out_dir / "selection.json").exists():
        raise ValueError("Preserve the original selection.")
    months = config["development_months"]
    reports = {m: read_json(out_dir / f"month_{m}.json") for m in months}
    for month, report in reports.items():
        if report["config_sha256"] != object_hash(config):
            raise ValueError(f"{month} was scored under another declaration.")
    control = config["control_candidate"]
    candidates = {}
    for name in config["candidates"]:
        entry = {"weight": float(config["candidates"][name]), "months": {m: reports[m]["candidates"][name] for m in months}}
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
              "months": {m: {k: reports[m][k] for k in ["rows", "fold", "control_rmse", "network_rmse_non_lirf", "committee_rmse_non_lirf", "network_committee_error_correlation_non_lirf"]} for m in months},
              "candidates": candidates, "selected_candidate": selected,
              "model_code": {p: sha256_file(ROOT / p) for p in ["tools/run_adsb_nn_dev.py", "tools/run_adsb_v24.py", "tools/run_adsb_v11.py"]},
              "environment": environment(ROOT)}
    for path in [out_dir / "selection.json", report_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, report)
    print({"selected": selected, "candidates": {n: {"mean_pct": round(100 * (e["mean_ratio"] - 1), 4), "eligible": e["eligible"],
                                                        "months_pct": {m: round(100 * (e["months"][m]["ratio_to_control"] - 1), 4) for m in months},
                                                        "intervals_below_zero": f"{len(e['months_with_interval_below_zero'])}/{len(e['months_touched'])}"} for n, e in candidates.items()}})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/adsb_nn/dev"))
    parser.add_argument("--report", type=Path, default=Path("docs/adsb_nn/development_results.json"))
    parser.add_argument("--cache", type=Path, default=Path("runs/adsb_v24/features"))
    parser.add_argument("--smoke", action="store_true", help="with --score: the network table is the committee's own (no network reading); write to a scratch --output")
    args = parser.parse_args()
    config = read_json(ROOT / args.config)
    if config["audit_used"]:
        raise ValueError("Declaration does not respect its own contract.")
    check_parents(config)
    if args.smoke and (args.select or not args.score or (ROOT / args.output).resolve() == (ROOT / "runs/adsb_nn/dev").resolve()):
        raise ValueError("--smoke scores only, into a scratch output directory.")
    if args.score:
        score(config, ROOT / args.output, ROOT / args.cache, smoke=args.smoke)
    if args.select:
        select(config, ROOT / args.output, ROOT / args.report)


if __name__ == "__main__":
    main()
