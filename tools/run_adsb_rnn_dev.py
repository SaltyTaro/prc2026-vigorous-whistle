"""Declared development of adsb_rnn: a neural-network member of the residual stage, with v25 as the control.

No model is fitted here. Inputs: the adsb_v25 leave-one-month-out residual columns of v25's members (adj2, cat;
runs/adsb_v25/dev/predictions_<M>.parquet) and the residual network's leave-one-month-out predictions (tools/run_adsb_rnn.py,
<learner>/<M>/predictions.parquet), averaged per month over each declared network's seeds; every network run is checked against its
declared variant, code, configuration, seed and held-out month.

A candidate is {"residual": {member: weight}} with non-negative weights summing to one over adj2, cat and the declared network means
(v25: adj2 0.5, cat 0.5), and optionally "recipe_weight", the committee-mix weight of the non-LIRF blend (v25: 0.7). Everything else is
v25 (adsb_nn2's k32_m30_u25): the 32-seed committee mix, the LIRF blends, the floor -- rebuilt with tools/run_adsb_nn2_dev.py's
functions, imported unchanged.

  --score   on each recipe month (2025-01/07/11/12) the control must reproduce k32_m30_u25's recorded development RMSE to 1e-9 s; every
            candidate is scored on all rows with paired-day intervals against the control. On all twelve months every candidate
            residual's scope RMSE (v9 + r on matched_no_lirf) is recorded for the pooled-scope condition.
  --select  the pre-registered rule
  --smoke   with --score: every network mean replaced by the control residual (every candidate collapses onto the control) and the
            network checks skipped, into a scratch output -- tests the code path and the control reproduction before any network fit
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
from run_adsb_nn_dev import mixed  # noqa: E402
from run_adsb_nn2_dev import check_ensembles, lirf_blends, spec_for, weighted  # noqa: E402
from run_adsb_v11 import candidate_values, check_parents, paired_days, rmse, scope_mask  # noqa: E402
from run_adsb_v24 import ALL_MONTHS, adsb_features, extra_columns, load_frame, month_table  # noqa: E402

TREE_MEMBERS = ("adj2", "cat")


def check_networks(config: dict) -> dict:
    """Every declared network learner holds one development run per month, fitted with the declared variant, code, configuration and seed."""
    out = {}
    for name, spec in config["residual_networks"].items():
        seeds = []
        for learner in spec["learners"]:
            for month in ALL_MONTHS:
                record = read_json(ROOT / learner / month / "run.json")
                if (record["variant_sha256"] != spec["variant_sha256"] or record["inputs"]["code"]["sha256"] != spec["code_sha256"]
                        or record["config_sha256"] != spec["config_sha256"] or record["held_out"] != month or record["stage"] != "development"
                        or record["smoke"] or record["audit_used"]):
                    raise ValueError(f"{learner}/{month} was not fitted with {name}'s declared variant, code and configuration.")
                seeds.append(record["seed"])
        if sorted(set(seeds)) != sorted(spec["seeds"]) or len(spec["learners"]) != len(spec["seeds"]):
            raise ValueError(f"{name}'s seeds differ from the declaration.")
        out[name] = {"learners": len(spec["learners"]), "seeds": sorted(set(seeds))}
    return out


def residual_columns(config: dict, month: str, ids: pd.Series, smoke: bool) -> dict[str, np.ndarray]:
    """adj2, cat (v25's leave-one-month-out columns) and every declared network's seed mean for the month, aligned to `ids`."""
    frame = pd.read_parquet(ROOT / config["residual_columns"][month]).set_index(ID).reindex(ids)
    out = {m: frame[f"residual:{m}"].to_numpy(dtype=float) for m in TREE_MEMBERS}
    for name, spec in config["residual_networks"].items():
        if smoke:
            out[name] = np.mean([out[m] for m in TREE_MEMBERS], axis=0)
            continue
        values = [pd.read_parquet(ROOT / learner / month / "predictions.parquet").set_index(ID)["residual"].reindex(ids).to_numpy(dtype=float) for learner in spec["learners"]]
        out[name] = np.mean(values, axis=0)
    for name, values in out.items():
        if not np.isfinite(values).all():
            raise ValueError(f"Residual column {name} does not cover {month}.")
    return out


def combine(candidate: dict, columns: dict[str, np.ndarray], name: str) -> np.ndarray:
    weights = candidate["residual"]
    if abs(sum(weights.values()) - 1) > 1e-12 or min(weights.values()) < 0 or not set(weights) <= set(columns):
        raise ValueError(f"{name}: residual weights must be non-negative, sum to one and name declared members.")
    return np.sum([w * columns[m] for m, w in weights.items()], axis=0)


def score(config: dict, out_dir: Path, cache: Path, smoke: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ensemble_check = {"committee": check_ensembles(config),
                      "residual_networks": "smoke: every network replaced by the control residual" if smoke else check_networks(config)}
    control_name, v25 = config["control_candidate"], config["v25_recipe"]
    candidates = config["candidates"]
    if smoke:  # a declared committee-mix weight is dropped too, so that no candidate differs from the control in any way
        candidates = {name: {k: v for k, v in c.items() if k != "recipe_weight"} for name, c in candidates.items()}
    matches_root = ROOT / config["matches_dirs"]["v11"]
    for month in ALL_MONTHS:
        target = out_dir / f"month_{month}.json"
        if target.exists():
            print(month, "exists", flush=True)
            continue
        table = month_table(load_frame(config, month, "cross_fit"), adsb_features(month, matches_root, cache), month)
        ids = table[ID]
        label, v9 = table[TARGET].to_numpy(dtype=float), table["v9"].to_numpy(dtype=float)
        inside = scope_mask(table, config["residual_scope"])
        columns = residual_columns(config, month, ids, smoke)
        residuals = {name: combine(c, columns, name) for name, c in candidates.items()}
        report = {"month": month, "rows": int(len(table)), "rows_in_scope": int(inside.sum()), "smoke": smoke,
                  "scope": {"v9": rmse(label[inside], v9[inside]), **{f"member:{m}": rmse(label[inside], (v9 + columns[m])[inside]) for m in columns},
                            **{name: rmse(label[inside], (v9 + r)[inside]) for name, r in residuals.items()}},
                  "scope_sse": {name: float(np.square((label - v9 - r)[inside]).sum()) for name, r in residuals.items()}}
        if month in config["recipe_months"]:
            floor = float(table["floor"].iloc[0])
            days = table[TIME].dt.strftime("%Y-%m-%d").to_numpy()
            for name, values in extra_columns(config, month, ids.to_numpy()).items():
                table[name] = values
            for name in config["ensembles"]:
                if not np.isfinite(table[name].to_numpy(dtype=float)).all():
                    raise ValueError(f"{name} does not cover every row of {month}.")
            committee = table[config["committee_table"]].to_numpy(dtype=float)
            table["committee_mix:v25"] = mixed(committee, weighted(table, v25["network"]), float(config["network_share"]))
            spec = spec_for(config, "committee_mix:v25", lirf_blends(config, table, v25, "v25"))
            values_by = {}
            for name, candidate in candidates.items():
                s = {**spec, "blend_with": {**spec["blend_with"], "weight": float(candidate.get("recipe_weight", config["recipe_blend"]["weight"]))}}
                values_by[name], _ = candidate_values(table, v9, residuals[name], s, floor)
            base = values_by[control_name]
            if abs(rmse(label, base) - config["control_rmse"][month]) > 1e-9:
                raise ValueError(f"The control does not reproduce v25's development RMSE on {month}: {rmse(label, base)} vs {config['control_rmse'][month]}.")
            non_lirf = table["airport"].ne("LIRF").to_numpy()
            matched = table["adsb_matched"].eq(1).to_numpy()
            airports = table["airport"].to_numpy()
            report.update({"floor_seconds": floor, "control_rmse": rmse(label, base), "candidates": {}})
            for name, values in values_by.items():
                entry = {**candidates[name], "rmse": rmse(label, values), "ratio_to_control": rmse(label, values) / report["control_rmse"],
                         "rows_differing_from_control": int((np.abs(values - base) > 1e-9).sum()),
                         "max_abs_difference_from_control": float(np.max(np.abs(values - base))),
                         "paired_days_vs_control": paired_days(days, label, values, base, config["seed"]),
                         "matched_non_lirf": {"control": rmse(label[matched & non_lirf], base[matched & non_lirf]), "candidate": rmse(label[matched & non_lirf], values[matched & non_lirf])},
                         "unmatched_non_lirf": {"control": rmse(label[~matched & non_lirf], base[~matched & non_lirf]), "candidate": rmse(label[~matched & non_lirf], values[~matched & non_lirf])},
                         "airports": {}}
                for airport in sorted(set(airports)):
                    k = airports == airport
                    entry["airports"][airport] = {"rows": int(k.sum()), "control": rmse(label[k], base[k]), "candidate": rmse(label[k], values[k]),
                                                  "change_pct": 100 * (rmse(label[k], values[k]) / rmse(label[k], base[k]) - 1)}
                report["candidates"][name] = entry
            if smoke:
                report["max_abs_difference_any_candidate_vs_control"] = float(max(np.max(np.abs(v - base)) for v in values_by.values()))
        report["inputs"] = {"scratch": sha256_file(ROOT / config["cross_fit"]["scratch"] / f"{month}.parquet"), "residual_columns": sha256_file(ROOT / config["residual_columns"][month]),
                            "networks": {} if smoke else {name: {learner: sha256_file(ROOT / learner / month / "predictions.parquet") for learner in spec["learners"]}
                                                           for name, spec in config["residual_networks"].items()}}
        report["ensemble_check"] = ensemble_check
        report["config_sha256"] = object_hash(config)
        write_json(target, report)
        line = {"scope": {n: round(v, 3) for n, v in report["scope"].items()}}
        if "candidates" in report:
            line["recipe_pct"] = {n: round(100 * (e["ratio_to_control"] - 1), 4) for n, e in report["candidates"].items()}
        print(month, line, flush=True)


def select(config: dict, out_dir: Path, report_path: Path) -> None:
    if (out_dir / "selection.json").exists():
        raise ValueError("Preserve the original selection.")
    reports = {m: read_json(out_dir / f"month_{m}.json") for m in ALL_MONTHS}
    for month, report in reports.items():
        if report.get("smoke") or report["config_sha256"] != object_hash(config):
            raise ValueError(f"{month} was scored under another declaration.")
    months, control = config["recipe_months"], config["control_candidate"]
    n_scope = sum(reports[m]["rows_in_scope"] for m in ALL_MONTHS)
    pooled = {name: float(np.sqrt(sum(reports[m]["scope_sse"][name] for m in ALL_MONTHS) / n_scope)) for name in config["candidates"]}
    candidates = {}
    for name in config["candidates"]:
        entry = {**config["candidates"][name], "months": {m: reports[m]["candidates"][name] for m in months},
                 "pooled_scope_rmse": pooled[name], "pooled_scope_ratio_to_control": pooled[name] / pooled[control],
                 "scope_rmse_by_month": {m: reports[m]["scope"][name] for m in ALL_MONTHS}}
        ratios = [entry["months"][m]["ratio_to_control"] for m in months]
        entry["mean_ratio"] = float(np.mean(ratios))
        touched = [m for m in months if entry["months"][m]["rows_differing_from_control"] > 0]
        below = [m for m in touched if entry["months"][m]["paired_days_vs_control"]["rmse_difference_95pct_interval_seconds"][1] < 0]
        entry["months_touched"], entry["months_with_interval_below_zero"] = touched, below
        entry["eligible"] = bool(name != control and touched and all(r <= 1.0 for r in ratios) and entry["mean_ratio"] <= 1 - config["minimum_mean_improvement_fraction"]
                                 and len(below) == len(touched) and entry["pooled_scope_ratio_to_control"] < 1.0)
        candidates[name] = entry
    eligible = [n for n, e in candidates.items() if e["eligible"]]
    selected = min(eligible, key=lambda n: (candidates[n]["mean_ratio"], n)) if eligible else None
    report = {"created_at_utc": utc_now(), "configuration": config, "config_sha256": object_hash(config), "audit_used": False, "rule": config["selection"],
              "months": {m: {k: reports[m].get(k) for k in ["rows", "rows_in_scope", "control_rmse", "scope"]} for m in ALL_MONTHS},
              "candidates": candidates, "selected_candidate": selected,
              "model_code": {p: sha256_file(ROOT / p) for p in ["tools/run_adsb_rnn_dev.py", "tools/run_adsb_nn2_dev.py", "tools/run_adsb_nn_dev.py", "tools/run_adsb_v24.py", "tools/run_adsb_v11.py"]},
              "environment": environment(ROOT)}
    for path in [out_dir / "selection.json", report_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, report)
    print({"selected": selected})
    for n, e in candidates.items():
        print(f"{n:24s} mean {100 * (e['mean_ratio'] - 1):+.4f}% " + " ".join(f"{m}:{100 * (e['months'][m]['ratio_to_control'] - 1):+.4f}" for m in months)
              + f" scope {100 * (e['pooled_scope_ratio_to_control'] - 1):+.3f}% intervals<0 {len(e['months_with_interval_below_zero'])}/{len(e['months_touched'])} eligible={e['eligible']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/adsb_rnn/dev"))
    parser.add_argument("--report", type=Path, default=Path("docs/adsb_rnn/development_results.json"))
    parser.add_argument("--cache", type=Path, default=Path("runs/adsb_v24/features"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = read_json(ROOT / args.config)
    if config["audit_used"]:
        raise ValueError("Declaration does not respect its own contract.")
    check_parents(config)
    nn2 = read_json(ROOT / "configs/adsb_nn2_v1.json")
    if config["v25_recipe"] != nn2["candidates"]["k32_m30_u25"] or config["candidates"][config["control_candidate"]] != {"residual": {"adj2": 0.5, "cat": 0.5}}:
        raise ValueError("The control must be the shipped v25 recipe (adsb_nn2 k32_m30_u25, residual = mean of adj2 and cat).")
    if args.smoke and (args.select or not args.score or (ROOT / args.output).resolve() == (ROOT / "runs/adsb_rnn/dev").resolve()):
        raise ValueError("--smoke scores only, into a scratch output directory.")
    if args.score:
        score(config, ROOT / args.output, ROOT / args.cache, smoke=args.smoke)
    if args.select:
        select(config, ROOT / args.output, ROOT / args.report)


if __name__ == "__main__":
    main()
