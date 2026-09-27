"""Declared development of the LIRF unmatched-blend round (configs/lirf_unmatched_blend_v1.json): v30's recipe with only the
lirf_nonpop_unmatched (net, tree, v9) weights changed. No model is fitted.

Per recipe month the v30 development recipe is rebuilt exactly as tools/v30_error_map.py does (committee mix c_e24x32_e32h + round B's
lctx_b100_up LIRF blends), after the three recorded reproductions it checks (v28, c_e24x32_e32h with v25's LIRF blends, v27 +
lctx_b100_up, each to 1e-9 s in RMSE). The control must also equal v30's development rows of runs/error_map/dev_rows.parquet to 1e-9 s per
row. Every candidate may change only lirf_nonpop_unmatched rows.

  --score   writes runs/lirf_blend/month_<m>.json
  --select  the pre-registered rule -> runs/lirf_blend/selection.json and docs/lirf_blend/development_results.json
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
from run_adsb_ctx_dev2 import merged_config  # noqa: E402
from run_adsb_nn2_dev import lirf_blends, spec_for, weighted  # noqa: E402
from run_adsb_nn_dev import mixed  # noqa: E402
from run_adsb_rnn_dev import combine, residual_columns  # noqa: E402
import run_adsb_v24  # noqa: E402,F401  (registers the lirf_nonpop_unmatched scope)
from run_adsb_v11 import SCOPES, candidate_values, paired_days, rmse  # noqa: E402
from run_adsb_v24 import adsb_features, extra_columns, load_frame, month_table  # noqa: E402

DECLARATION = "configs/lirf_unmatched_blend_v1.json"
OUT = ROOT / "runs/lirf_blend"
REPORT = ROOT / "docs/lirf_blend/development_results.json"
DEV_ROWS = "runs/error_map/dev_rows.parquet"
CODE = ["tools/run_lirf_blend_dev.py", "tools/v30_error_map.py", "tools/run_adsb_ctx_dev2.py", "tools/run_adsb_rnn_dev.py", "tools/run_adsb_nn2_dev.py",
        "tools/run_adsb_nn_dev.py", "tools/run_adsb_v24.py", "tools/run_adsb_v11.py"]


def score(decl: dict) -> None:
    decl5, decl_b = read_json(ROOT / "configs/adsb_ctx_v5.json"), read_json(ROOT / "configs/adsb_ctx_lirf_v1.json")
    for path, digest in decl["parents"].items():
        if object_hash(read_json(ROOT / path)) != digest:
            raise ValueError(f"Parent declaration {path} changed.")
    combined = {**decl5, "networks": {**decl5["networks"], "ens_ctx_lirf": decl_b["networks"]["ens_ctx_lirf"]}}
    config = merged_config(combined)
    v26, v25 = config["candidates"]["rnet_w90"], config["v25_recipe"]
    control = decl["control"]
    if control["lirf"] != decl_b["candidates"]["lctx_b100_up"]["lirf"] or control["lirf_network"] != decl_b["candidates"]["lctx_b100_up"]["lirf_network"]:
        raise ValueError("The control is not lctx_b100_up.")
    lirf_b = {**v25, "lirf": control["lirf"], "lirf_network": control["lirf_network"]}
    rec5 = {m: e["rmse"] for m, e in read_json(ROOT / "docs/adsb_ctx/development_results_v5.json")["candidates"]["c_e24x32_e32h"]["months"].items()}
    rec_b = {m: e["rmse"] for m, e in read_json(ROOT / "docs/adsb_ctx_lirf/development_results.json")["candidates"]["lctx_b100_up"]["months"].items()}
    cand = decl5["candidates"]["c_e24x32_e32h"]
    dev_rows = pd.read_parquet(ROOT / DEV_ROWS, columns=[ID, "month", "v30"])
    OUT.mkdir(parents=True, exist_ok=True)
    for month in decl["recipe_months"]:
        target = OUT / f"month_{month}.json"
        if target.exists():
            print(month, "exists", flush=True)
            continue
        table = month_table(load_frame(config, month, "cross_fit"), adsb_features(month, ROOT / config["matches_dirs"]["v11"], ROOT / "runs/adsb_v24/features"), month)
        ids = table[ID]
        label, v9 = table[TARGET].to_numpy(dtype=float), table["v9"].to_numpy(dtype=float)
        residual = combine(v26, residual_columns(config, month, ids, smoke=False), "rnet_w90")
        floor = float(table["floor"].iloc[0])
        days = table[TIME].dt.strftime("%Y-%m-%d").to_numpy()
        for name, values in extra_columns(config, month, ids.to_numpy()).items():
            table[name] = values
        committee = table[config["committee_table"]].to_numpy(dtype=float)
        table["mix:v28"] = mixed(committee, weighted(table, decl5["control"]["network_table"]), float(decl5["control"]["network_share"]))
        table["mix:v27"] = mixed(committee, weighted(table, decl_b["control"]["committee_network"]), float(decl_b["control"]["committee_share"]))
        table["mix:v30"] = mixed(committee, weighted(table, cand["network"]), float(cand["network_share"]))
        checks = {"v28": ("mix:v28", v25, "v26", decl5["control"]["recorded_development_rmse"][month]),
                  "v30_v25lirf": ("mix:v30", v25, "v26", rec5[month]), "v27_b": ("mix:v27", lirf_b, "lctx_b100_up", rec_b[month])}
        reproduction = {}
        for key, (column, blend, tag, recorded) in checks.items():
            values, _ = candidate_values(table, v9, residual, spec_for(config, column, lirf_blends(config, table, blend, tag)), floor)
            if abs(rmse(label, values) - recorded) > 1e-9:
                raise ValueError(f"{key} does not reproduce its recorded development RMSE on {month}: {rmse(label, values)} vs {recorded}.")
            reproduction[key] = {"rmse": rmse(label, values), "recorded": recorded}
        base, _ = candidate_values(table, v9, residual, spec_for(config, "mix:v30", lirf_blends(config, table, lirf_b, "control")), floor)
        ref = dev_rows[dev_rows["month"] == month].set_index(ID).reindex(ids)["v30"].to_numpy(dtype=float)
        if np.isnan(ref).any() or np.max(np.abs(base - ref)) > 1e-9:
            raise ValueError(f"The control does not reproduce v30's development rows on {month}.")
        reproduction["v30_rows_max_abs"] = float(np.max(np.abs(base - ref)))
        scope = SCOPES["lirf_nonpop_unmatched"](table).to_numpy()
        airports = table["airport"].to_numpy()
        report = {"month": month, "rows": int(len(table)), "floor_seconds": floor, "control_rmse": rmse(label, base), "reproduction": reproduction,
                  "scope_rows": int(scope.sum()), "candidates": {}}
        for name, spec in {"control": {"lirf_nonpop_unmatched": control["lirf"]["lirf_nonpop_unmatched"]}, **decl["candidates"]}.items():
            lirf = {"lirf_nonpop_matched": control["lirf"]["lirf_nonpop_matched"], "lirf_nonpop_unmatched": spec["lirf_nonpop_unmatched"]}
            values, _ = candidate_values(table, v9, residual, spec_for(config, "mix:v30", lirf_blends(config, table, {**v25, "lirf": lirf, "lirf_network": control["lirf_network"]}, name)), floor)
            changed = np.abs(values - base) > 1e-9
            if (changed & ~scope).any():
                raise ValueError(f"{name} changes a row outside lirf_nonpop_unmatched on {month}.")
            report["candidates"][name] = {**spec, "rmse": rmse(label, values), "ratio_to_control": rmse(label, values) / report["control_rmse"],
                                          "rows_differing_from_control": int(changed.sum()), "max_abs_difference_from_control": float(np.max(np.abs(values - base))),
                                          "paired_days_vs_control": paired_days(days, label, values, base, config["seed"]),
                                          "scope": {"control": rmse(label[scope], base[scope]), "candidate": rmse(label[scope], values[scope])} if scope.any() else None,
                                          "lirf": {"control": rmse(label[airports == "LIRF"], base[airports == "LIRF"]), "candidate": rmse(label[airports == "LIRF"], values[airports == "LIRF"])}}
        report["declaration_sha256"] = object_hash(decl)
        report["inputs"] = {DEV_ROWS: sha256_file(ROOT / DEV_ROWS)}
        write_json(target, report)
        print(month, {"control": round(report["control_rmse"], 4), "scope_rows": report["scope_rows"],
                      "recipe_pct": {n: round(100 * (e["ratio_to_control"] - 1), 4) for n, e in report["candidates"].items()}}, flush=True)


def select(decl: dict) -> None:
    if (OUT / "selection.json").exists():
        raise ValueError("Preserve the original selection.")
    months = decl["recipe_months"]
    reports = {m: read_json(OUT / f"month_{m}.json") for m in months}
    for month, report in reports.items():
        if report["declaration_sha256"] != object_hash(decl):
            raise ValueError(f"{month} was scored under another declaration.")
    candidates = {}
    for name in decl["candidates"]:
        entry = {**decl["candidates"][name], "months": {m: reports[m]["candidates"][name] for m in months}}
        ratios = [entry["months"][m]["ratio_to_control"] for m in months]
        entry["mean_ratio"] = float(np.mean(ratios))
        touched = [m for m in months if entry["months"][m]["rows_differing_from_control"] > 0]
        below = [m for m in touched if entry["months"][m]["paired_days_vs_control"]["rmse_difference_95pct_interval_seconds"][1] < 0]
        entry["months_touched"], entry["months_with_interval_below_zero"] = touched, below
        entry["eligible"] = bool(touched and all(r <= 1.0 for r in ratios) and entry["mean_ratio"] <= 1 - decl["minimum_mean_improvement_fraction"]
                                 and len(below) == len(touched))
        candidates[name] = entry
    eligible = [n for n, e in candidates.items() if e["eligible"]]
    selected = min(eligible, key=lambda n: (candidates[n]["mean_ratio"], n)) if eligible else None
    report = {"created_at_utc": utc_now(), "declaration": decl, "declaration_sha256": object_hash(decl), "audit_used": False, "rule": decl["selection"],
              "months": {m: {"rows": reports[m]["rows"], "control_rmse": reports[m]["control_rmse"], "scope_rows": reports[m]["scope_rows"],
                             "reproduction": reports[m]["reproduction"]} for m in months},
              "candidates": candidates, "selected_candidate": selected, "model_code": {p: sha256_file(ROOT / p) for p in CODE}, "environment": environment(ROOT)}
    for path in [OUT / "selection.json", REPORT]:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, report)
    print({"selected": selected})
    for n, e in candidates.items():
        print(f"{n:10s} mean {100 * (e['mean_ratio'] - 1):+.4f}% " + " ".join(f"{m}:{100 * (e['months'][m]['ratio_to_control'] - 1):+.4f}" for m in months)
              + f" intervals<0 {len(e['months_with_interval_below_zero'])}/{len(e['months_touched'])} eligible={e['eligible']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--select", action="store_true")
    args = parser.parse_args()
    decl = read_json(ROOT / DECLARATION)
    if args.score:
        score(decl)
    if args.select:
        select(decl)


if __name__ == "__main__":
    main()
