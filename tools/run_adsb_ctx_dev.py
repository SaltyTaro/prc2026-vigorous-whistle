"""Declared development of adsb_ctx (configs/adsb_ctx_v1.json): the context network inside the committee mix's network table, against v26.

No model is fitted here. On each recipe month (2025-01/07/11/12) the shipped v26 recipe is rebuilt from its declaration
(configs/adsb_rnn_v1.json candidate rnet_w90: the adsb_nn2 k32_m30_u25 committee mix, LIRF blends and floor, with the residual
0.05 adj2 + 0.05 cat + 0.9 rnet) using tools/run_adsb_rnn_dev.py's and tools/run_adsb_nn2_dev.py's functions, imported unchanged.
A candidate changes only the committee mix: `network` (weights over ensemble tables, v26 = ens_nn 0.5 + ens_new 0.5) and
`network_share` (v26 0.5). The new ensemble ens_ctx is the per-fold seed mean of the declared context learners
(tools/ensemble_adsb_full.py -> runs/adsb_ctx/ens_ctx; December from the learners' January-November refit leaves).

  --score   the control must reproduce v26's recorded development RMSE (rnet_w90) to 1e-9 s on every month; every candidate is scored on
            all rows, with paired-day intervals, matched / unmatched non-LIRF and per-airport RMSE against the control
  --select  the pre-registered rule
  --smoke   with --score: ens_ctx replaced by the control's own network table and every network_share by the control's, so every
            candidate collapses onto the control (no reading of the context network), into a scratch output
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
from run_adsb_nn_dev import mixed  # noqa: E402
from run_adsb_nn2_dev import check_ensembles, lirf_blends, spec_for, weighted  # noqa: E402
from run_adsb_rnn_dev import check_networks, combine, residual_columns  # noqa: E402
from run_adsb_v11 import candidate_values, check_parents, paired_days, rmse  # noqa: E402
from run_adsb_v24 import adsb_features, extra_columns, load_frame, month_table  # noqa: E402

ENS = "ens_ctx"
ENS_DIR = "runs/adsb_ctx/ens_ctx"


def merged_config(decl: dict) -> dict:
    """v26's declaration plus the ens_ctx table and its learner contract."""
    base = read_json(ROOT / "configs/adsb_rnn_v1.json")
    net = decl["network"]
    config = dict(base)
    config["extra_feature_tables"] = {**base["extra_feature_tables"], ENS: {
        "column": "adsb_full", "description": "mean of the declared context-network seeds", "run_dir": ENS_DIR,
        "development": [f"{ENS_DIR}/seasonal/predictions.parquet", f"{ENS_DIR}/forward/predictions.parquet"]}}
    config["ensembles"] = {**base["ensembles"], ENS: {"learners": net["learners"], "seeds": net["seeds"], "variant_sha256": net["variant_sha256"],
                                                      "code_sha256": net["code"]["sha256"], "config_sha256": net["config_sha256"]}}
    return config


def score(decl: dict, out_dir: Path, cache: Path, smoke: bool = False) -> None:
    config = merged_config(decl)
    v26 = config["candidates"]["rnet_w90"]
    v25 = config["v25_recipe"]
    control_name = "control"
    candidates = decl["candidates"]
    if candidates[control_name] != {"network": v25["network"], "network_share": config["network_share"]}:
        raise ValueError("The control must be v26's committee mix (adsb_nn2 k32_m30_u25 network table, share 0.5).")
    if smoke:
        config = {**config, "extra_feature_tables": {k: v for k, v in config["extra_feature_tables"].items() if k != ENS},
                  "ensembles": {k: v for k, v in config["ensembles"].items() if k != ENS}}
        ensemble_check = {"smoke": "ens_ctx replaced by the control's network table; every network_share by the control's"}
    else:
        ensemble_check = {"committee": check_ensembles(config), "residual_networks": check_networks(config)}
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
            table[ENS] = weighted(table, v25["network"])
        for name in config["ensembles"]:
            if not np.isfinite(table[name].to_numpy(dtype=float)).all():
                raise ValueError(f"{name} does not cover every row of {month}.")
        committee = table[config["committee_table"]].to_numpy(dtype=float)
        lirf = None
        values_by = {}
        for name, candidate in candidates.items():
            share = float(config["network_share"] if smoke else candidate["network_share"])
            column = f"committee_mix:{name}"
            table[column] = mixed(committee, weighted(table, candidate["network"]), share)
            lirf = lirf_blends(config, table, v25, "v26")
            spec = spec_for(config, column, lirf)
            values_by[name], _ = candidate_values(table, v9, residual, spec, floor)
        base = values_by[control_name]
        recorded = decl["control"]["recorded_development_rmse"][month]
        if abs(rmse(label, base) - recorded) > 1e-9:
            raise ValueError(f"The control does not reproduce v26's development RMSE on {month}: {rmse(label, base)} vs {recorded}.")
        non_lirf = table["airport"].ne("LIRF").to_numpy()
        matched = table["adsb_matched"].eq(1).to_numpy()
        airports = table["airport"].to_numpy()
        report = {"month": month, "rows": int(len(table)), "smoke": smoke, "floor_seconds": floor, "control_rmse": rmse(label, base), "candidates": {}}
        for name, values in values_by.items():
            entry = {**candidates[name], "rmse": rmse(label, values), "ratio_to_control": rmse(label, values) / report["control_rmse"],
                     "rows_differing_from_control": int((np.abs(values - base) > 1e-9).sum()),
                     "max_abs_difference_from_control": float(np.max(np.abs(values - base))),
                     "paired_days_vs_control": paired_days(days, label, values, base, config["seed"]),
                     "matched_non_lirf": {"control": rmse(label[matched & non_lirf], base[matched & non_lirf]), "candidate": rmse(label[matched & non_lirf], values[matched & non_lirf])},
                     "unmatched_non_lirf": {"control": rmse(label[~matched & non_lirf], base[~matched & non_lirf]), "candidate": rmse(label[~matched & non_lirf], values[~matched & non_lirf])},
                     "ordinary_rows": {"rows": int((label <= 21600).sum()), "control": rmse(label[label <= 21600], base[label <= 21600]), "candidate": rmse(label[label <= 21600], values[label <= 21600])},
                     "airports": {}}
            for airport in sorted(set(airports)):
                k = airports == airport
                entry["airports"][airport] = {"rows": int(k.sum()), "control": rmse(label[k], base[k]), "candidate": rmse(label[k], values[k]),
                                              "change_pct": 100 * (rmse(label[k], values[k]) / rmse(label[k], base[k]) - 1)}
            report["candidates"][name] = entry
        if smoke:
            report["max_abs_difference_any_candidate_vs_control"] = float(max(np.max(np.abs(v - base)) for v in values_by.values()))
        report["inputs"] = {"scratch": sha256_file(ROOT / config["cross_fit"]["scratch"] / f"{month}.parquet"),
                            "ens_ctx": None if smoke else ({"december_leaves": sorted(decl["network"]["learners"])} if month == "2025-12"
                                                           else sha256_file(ROOT / config["extra_feature_tables"][ENS]["development"][0 if month in ("2025-01", "2025-07") else 1]))}
        report["ensemble_check"] = ensemble_check
        report["declaration_sha256"] = object_hash(decl)
        write_json(target, report)
        print(month, {"control": round(report["control_rmse"], 4), **({"max_diff": report["max_abs_difference_any_candidate_vs_control"]} if smoke else
                      {"recipe_pct": {n: round(100 * (e["ratio_to_control"] - 1), 4) for n, e in report["candidates"].items()}})}, flush=True)


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
              "candidates": candidates, "selected_candidate": selected,
              "model_code": {p: sha256_file(ROOT / p) for p in ["tools/run_adsb_ctx_dev.py", "tools/run_adsb_rnn_dev.py", "tools/run_adsb_nn2_dev.py",
                                                               "tools/run_adsb_nn_dev.py", "tools/run_adsb_v24.py", "tools/run_adsb_v11.py"]},
              "environment": environment(ROOT)}
    for path in [out_dir / "selection.json", report_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, report)
    print({"selected": selected})
    for n, e in candidates.items():
        print(f"{n:16s} mean {100 * (e['mean_ratio'] - 1):+.4f}% " + " ".join(f"{m}:{100 * (e['months'][m]['ratio_to_control'] - 1):+.4f}" for m in months)
              + f" intervals<0 {len(e['months_with_interval_below_zero'])}/{len(e['months_touched'])} eligible={e['eligible']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declaration", type=Path, default=Path("configs/adsb_ctx_v1.json"))
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/adsb_ctx/dev"))
    parser.add_argument("--report", type=Path, default=Path("docs/adsb_ctx/development_results.json"))
    parser.add_argument("--cache", type=Path, default=Path("runs/adsb_v24/features"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    decl = read_json(ROOT / args.declaration)
    if decl["audit_used"]:
        raise ValueError("Declaration does not respect its own contract.")
    check_parents(read_json(ROOT / "configs/adsb_rnn_v1.json"))
    if args.smoke and (args.select or not args.score or (ROOT / args.output).resolve() == (ROOT / "runs/adsb_ctx/dev").resolve()):
        raise ValueError("--smoke scores only, into a scratch output directory.")
    if args.score:
        score(decl, ROOT / args.output, ROOT / args.cache, smoke=args.smoke)
    if args.select:
        select(decl, ROOT / args.output, ROOT / args.report)


if __name__ == "__main__":
    main()
