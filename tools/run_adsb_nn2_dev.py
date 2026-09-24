"""Declared development of adsb_nn2: further network members for the v24 recipe -- more committee-table seeds and a LIRF-inclusive
network inside the LIRF tables.

No model is fitted here. On each of the four recipe months (2025-01/07/11/12) the shipped v24 recipe (adsb_nn candidate nn50) is
rebuilt exactly as tools/run_adsb_nn_dev.py rebuilt it, with two declared changes per candidate:

  network        {ensemble table: weight} -- the network table inside the committee mix, a weighted mean of declared seed ensembles
                 (weights proportional to seed counts give the plain mean over all seeds); v24: {"ens_nn": 1.0}
  lirf_network   {ensemble table: weight}, the LIRF-side network mean (a LIRF-inclusive network)
  lirf           per LIRF blend scope, the weights {"v9", "tree", "net"} (summing to one) of the running value (v9 on LIRF rows),
                 the LIRF tree table ens_v22_lirf_q and the LIRF-side network; implemented as the inherited blend on that scope with
                 outer weight tree + net and the table (tree x ens_v22_lirf_q + net x network) / (tree + net).
                 v24: lirf_nonpop_matched (0.15, 0.85, 0), lirf_nonpop_unmatched (0.75, 0.25, 0)

The committee share (0.5), the outer weight (0.7, adsb_v29 kept it), the residual columns, the floor and every other number are
v24's. The control must reproduce v24's recorded development RMSE (runs/adsb_nn/dev/selection.json, nn50) on every month to
1e-9 s. Every leaf of every declared ensemble is checked against its declared variant, code and configuration on every fold.

  --score    every candidate on the four months (all rows), with paired-day intervals against the control
  --select   the pre-registered rule
  --smoke    with --score: every new ensemble replaced by the v24 network table (ens_nn) and the LIRF network by the LIRF table
             itself, so every candidate collapses onto the control -- tests the code path and the control reproduction without
             producing any reading; writes to a scratch output
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
from run_adsb_nn_dev import FOLD_OF, mixed  # noqa: E402
from run_adsb_v11 import SCOPES, candidate_values, check_parents, paired_days, rmse  # noqa: E402
from run_adsb_v24 import adsb_features, extra_columns, load_frame, month_table  # noqa: E402


def check_ensembles(config: dict) -> dict:
    """Every leaf of every declared ensemble is a declared seed of its declared variant, fitted by the declared code and configuration."""
    out = {}
    for table, spec in config["ensembles"].items():
        leaves = leaf_members(ROOT, config["extra_feature_tables"][table]["run_dir"])
        if sorted(leaves) != sorted(spec["learners"]) or len(set(leaves)) != len(leaves):
            raise ValueError(f"{table} does not hold exactly its declared learners.")
        seeds = {}
        for leaf in leaves:
            for fold in ["seasonal", "forward", "refit"]:
                record = read_json(ROOT / leaf / fold / "run.json")
                if (record["variant_sha256"] != spec["variant_sha256"] or record["inputs"]["code"]["sha256"] != spec["code_sha256"]
                        or record["config_sha256"] != spec["config_sha256"] or record["fold"] != fold):
                    raise ValueError(f"{leaf}/{fold} was not fitted with {table}'s declared variant, code and configuration.")
                seeds.setdefault(leaf, record["seed"])
        if sorted(seeds.values()) != sorted(spec["seeds"]):
            raise ValueError(f"{table}'s seeds differ from the declaration.")
        out[table] = {"leaves": len(leaves), "seeds": sorted(seeds.values())}
    return out


def weighted(table: pd.DataFrame, weights: dict) -> np.ndarray:
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-12:
        raise ValueError("Ensemble weights must sum to one.")
    return np.sum([w * table[name].to_numpy(dtype=float) for name, w in weights.items()], axis=0)


def lirf_blends(config: dict, table: pd.DataFrame, candidate: dict, name: str) -> list[dict]:
    """The inherited LIRF blends with the candidate's (v9, tree, net) weights: outer weight tree + net, table the normalised mix."""
    blends = []
    for blend in config["extra_blends"]:
        if blend["name"] != config["lirf_table"] or blend["scope"] not in ("lirf_nonpop_matched", "lirf_nonpop_unmatched"):
            raise ValueError("Only the two LIRF tree-table blends are inherited.")
        w = candidate["lirf"][blend["scope"]]
        if abs(w["v9"] + w["tree"] + w["net"] - 1) > 1e-12 or min(w.values()) < 0 or w["tree"] + w["net"] <= 0:
            raise ValueError(f"{name}: LIRF weights on {blend['scope']} must be non-negative and sum to one.")
        column = f"lirf_mix:{name}:{blend['scope']}"
        tree = table[config["lirf_table"]].to_numpy(dtype=float)
        if w["net"] > 0:
            net = weighted(table, candidate["lirf_network"])
            table[column] = (w["tree"] * tree + w["net"] * net) / (w["tree"] + w["net"])
        else:
            table[column] = tree
        blends.append({"name": column, "scope": blend["scope"], "weight": w["tree"] + w["net"]})
    return blends


def spec_for(config: dict, committee_column: str, lirf: list[dict]) -> dict:
    return {"weight": 1.0, "scope": config["residual_scope"], "blend_with": {"name": committee_column, **config["recipe_blend"]}, "extra_blends": lirf}


def score(config: dict, out_dir: Path, cache: Path, smoke: bool = False) -> None:
    control = config["control_candidate"]
    candidates = config["candidates"]
    tables = config["extra_feature_tables"]
    if smoke:
        tables = {**tables, **{name: tables[config["v24_network_table"]] for name in config["ensembles"] if name != config["v24_network_table"]}}
        config = {**config, "extra_feature_tables": tables}
        ensemble_check = {"smoke": "every new ensemble replaced by the v24 network table; LIRF network replaced by the LIRF table"}
    else:
        ensemble_check = check_ensembles(config)
    matches_root = ROOT / config["matches_dirs"]["v11"]
    out_dir.mkdir(parents=True, exist_ok=True)
    committee_name, lirf_name = config["committee_table"], config["lirf_table"]
    inherited = {b["scope"]: b["weight"] for b in config["extra_blends"]}
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
        committee = table[committee_name].to_numpy(dtype=float)
        non_lirf = table["airport"].ne("LIRF").to_numpy()
        for name in config["ensembles"]:
            if not np.isfinite(table[name].to_numpy(dtype=float)).all():
                raise ValueError(f"{name} does not cover every row of {month}.")
        columns = pd.read_parquet(ROOT / config["residual_columns"][month]).set_index(ID).reindex(ids)
        residual = np.mean([columns[f"residual:{m}"].to_numpy(dtype=float) for m in config["residual_members"]], axis=0)
        if not np.isfinite(residual).all():
            raise ValueError(f"Residual columns do not cover {month}.")
        values_by = {}
        for name, candidate in candidates.items():
            network = weighted(table, candidate["network"])
            table[f"committee_mix:{name}"] = mixed(committee, network, float(config["network_share"]))
            if smoke:  # the LIRF network collapses onto the tree table: every LIRF blend equals v24's
                candidate = {**candidate, "lirf": {s: {"v9": 1 - inherited[s], "tree": inherited[s], "net": 0.0} for s in inherited}}
            values_by[name], _ = candidate_values(table, v9, residual, spec_for(config, f"committee_mix:{name}", lirf_blends(config, table, candidate, name)), floor)
            for scope, w in candidate["lirf"].items():  # independent arithmetic on every LIRF blend row
                k = SCOPES[scope](table).to_numpy()
                net = weighted(table, candidate["lirf_network"])[k] if w["net"] > 0 else 0.0
                expected = np.maximum(w["v9"] * v9[k] + w["tree"] * table[lirf_name].to_numpy(dtype=float)[k] + w["net"] * net, floor)
                if np.max(np.abs(values_by[name][k] - expected), initial=0.0) > 1e-6:
                    raise ValueError(f"{name}: the LIRF blend on {scope} does not equal its declared weights on {month}.")
        base = values_by[control]
        expected = config["control_rmse"][month]
        if abs(rmse(label, base) - expected) > 1e-9:
            raise ValueError(f"The control does not reproduce v24's development RMSE on {month}: {rmse(label, base)} vs {expected}.")
        if smoke:
            report = {"month": month, "smoke": True, "control_rmse": rmse(label, base), "control_reproduces": True,
                      "max_abs_difference_any_candidate_vs_control": float(max(np.max(np.abs(v - base)) for v in values_by.values()))}
            write_json(target, report)
            print(month, report, flush=True)
            continue
        matched = table["adsb_matched"].eq(1).to_numpy()
        lirf = ~non_lirf
        lirf_nonpop = lirf & ~table["population"].astype(bool).to_numpy()
        airports = table["airport"].to_numpy()
        report = {"month": month, "rows": int(len(table)), "fold": FOLD_OF[month], "floor_seconds": floor, "control_rmse": rmse(label, base),
                  "tables": {name: {"rmse_non_lirf": rmse(label[non_lirf], table[name].to_numpy(dtype=float)[non_lirf]),
                                    "rmse_lirf_nonpop": rmse(label[lirf_nonpop], table[name].to_numpy(dtype=float)[lirf_nonpop])}
                             for name in [committee_name, lirf_name, *config["ensembles"]]},
                  "candidates": {}}
        for name, values in values_by.items():
            entry = {**candidates[name], "rmse": rmse(label, values), "ratio_to_control": rmse(label, values) / report["control_rmse"],
                     "rows_differing_from_control": int((np.abs(values - base) > 1e-9).sum()),
                     "paired_days_vs_control": paired_days(days, label, values, base, config["seed"]),
                     "non_lirf": {"control": rmse(label[non_lirf], base[non_lirf]), "candidate": rmse(label[non_lirf], values[non_lirf])},
                     "lirf": {"control": rmse(label[lirf], base[lirf]), "candidate": rmse(label[lirf], values[lirf])},
                     "matched_non_lirf": {"control": rmse(label[matched & non_lirf], base[matched & non_lirf]), "candidate": rmse(label[matched & non_lirf], values[matched & non_lirf])},
                     "unmatched_non_lirf": {"control": rmse(label[~matched & non_lirf], base[~matched & non_lirf]), "candidate": rmse(label[~matched & non_lirf], values[~matched & non_lirf])},
                     "airports": {}}
            for airport in sorted(set(airports)):
                k = airports == airport
                entry["airports"][airport] = {"rows": int(k.sum()), "control": rmse(label[k], base[k]), "candidate": rmse(label[k], values[k]),
                                              "change_pct": 100 * (rmse(label[k], values[k]) / rmse(label[k], base[k]) - 1)}
            report["candidates"][name] = entry
        report["inputs"] = {"scratch": sha256_file(ROOT / config["cross_fit"]["scratch"] / f"{month}.parquet"), "residual_columns": sha256_file(ROOT / config["residual_columns"][month])}
        report["ensemble_check"] = ensemble_check
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
        entry = {**config["candidates"][name], "months": {m: reports[m]["candidates"][name] for m in months}}
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
              "months": {m: {k: reports[m][k] for k in ["rows", "fold", "control_rmse", "tables"]} for m in months},
              "candidates": candidates, "selected_candidate": selected,
              "model_code": {p: sha256_file(ROOT / p) for p in ["tools/run_adsb_nn2_dev.py", "tools/run_adsb_nn_dev.py", "tools/run_adsb_v24.py", "tools/run_adsb_v11.py"]},
              "environment": environment(ROOT)}
    for path in [out_dir / "selection.json", report_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, report)
    print({"selected": selected})
    for n, e in candidates.items():
        print(f"{n:22s} mean {100 * (e['mean_ratio'] - 1):+.4f}% " + " ".join(f"{m}:{100 * (e['months'][m]['ratio_to_control'] - 1):+.4f}" for m in months)
              + f" intervals<0 {len(e['months_with_interval_below_zero'])}/{len(e['months_touched'])} eligible={e['eligible']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/adsb_nn2/dev"))
    parser.add_argument("--report", type=Path, default=Path("docs/adsb_nn2/development_results.json"))
    parser.add_argument("--cache", type=Path, default=Path("runs/adsb_v24/features"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = read_json(ROOT / args.config)
    if config["audit_used"]:
        raise ValueError("Declaration does not respect its own contract.")
    check_parents(config)
    control = config["candidates"][config["control_candidate"]]
    v24_lirf = {b["scope"]: {"v9": 1 - b["weight"], "tree": b["weight"], "net": 0.0} for b in config["extra_blends"]}
    if control["network"] != {config["v24_network_table"]: 1.0} or control["lirf"] != v24_lirf:
        raise ValueError("The control must be the shipped v24 recipe.")
    if args.smoke and (args.select or not args.score or (ROOT / args.output).resolve() == (ROOT / "runs/adsb_nn2/dev").resolve()):
        raise ValueError("--smoke scores only, into a scratch output directory.")
    if args.score:
        score(config, ROOT / args.output, ROOT / args.cache, smoke=args.smoke)
    if args.select:
        select(config, ROOT / args.output, ROOT / args.report)


if __name__ == "__main__":
    main()
