"""Declared development of the outer-weight-by-scope round (configs/outer_weight_scope_v1.json): v30's recipe with the outer weight of the
committee mix (0.7 against v9 + residual on every non-LIRF row) changed on LTFM rows and on the other ADS-B-matched non-LIRF rows. No model
is fitted.

Per recipe month the v30 development recipe is rebuilt as tools/run_lirf_blend_dev.py does, after the same three recorded reproductions
(v28, c_e24x32_e32h with v25's LIRF blends, v27 + lctx_b100_up, each to 1e-9 s in RMSE); the control must equal v30's development rows of
runs/error_map/dev_rows.parquet to 1e-9 s per row, and the scope-weight rebuild at 0.7 everywhere must equal the control to 1e-9 s per row.
Every candidate may change only non-LIRF rows.

  --score   writes runs/outer_weight/month_<m>.json
  --select  the pre-registered rule -> runs/outer_weight/selection.json and docs/outer_weight/development_results.json
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

DECLARATION = "configs/outer_weight_scope_v1.json"
OUT = ROOT / "runs/outer_weight"
REPORT = ROOT / "docs/outer_weight/development_results.json"
DEV_ROWS = "runs/error_map/dev_rows.parquet"
CODE = ["tools/run_outer_weight_dev.py", "tools/run_lirf_blend_dev.py", "tools/v30_error_map.py", "tools/run_adsb_ctx_dev2.py", "tools/run_adsb_rnn_dev.py", "tools/run_adsb_nn2_dev.py",
        "tools/run_adsb_nn_dev.py", "tools/run_adsb_v24.py", "tools/run_adsb_v11.py"]


def score(decl: dict) -> None:
    decl5, decl_b = read_json(ROOT / "configs/adsb_ctx_v5.json"), read_json(ROOT / "configs/adsb_ctx_lirf_v1.json")
    for path, digest in decl["parents"].items():
        if object_hash(read_json(ROOT / path)) != digest:
            raise ValueError(f"Parent declaration {path} changed.")
    combined = {**decl5, "networks": {**decl5["networks"], "ens_ctx_lirf": decl_b["networks"]["ens_ctx_lirf"]}}
    config = merged_config(combined)
    v26, v25 = config["candidates"]["rnet_w90"], config["v25_recipe"]
    lirf_b = {**v25, "lirf": decl_b["candidates"]["lctx_b100_up"]["lirf"], "lirf_network": decl_b["candidates"]["lctx_b100_up"]["lirf_network"]}
    if config["recipe_blend"] != {"scope": "all_no_lirf", "weight": decl["control"]["outer_weight"]}:
        raise ValueError(f"The recipe blend is not the control's outer weight: {config['recipe_blend']}.")
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
        base_v, _ = candidate_values(table, v9, residual, {"weight": 1.0, "scope": config["residual_scope"], "blend_with": None, "extra_blends": []}, -1e12)
        mix = table["mix:v30"].to_numpy(dtype=float)
        airports = table["airport"].to_numpy()
        non_lirf = airports != "LIRF"
        ltfm = airports == "LTFM"
        matched_other = table["adsb_matched"].eq(1).to_numpy() & non_lirf & ~ltfm
        unmatched_other = non_lirf & ~ltfm & ~matched_other

        def rebuild(w_ltfm: float, w_matched: float, w_unmatched: float) -> np.ndarray:
            w = np.where(ltfm, w_ltfm, np.where(matched_other, w_matched, w_unmatched))
            return np.where(non_lirf, np.maximum((1 - w) * base_v + w * mix, floor), base)

        same = rebuild(0.7, 0.7, 0.7)
        reproduction["scope_rebuild_at_control_max_abs"] = float(np.max(np.abs(same - base)))
        if reproduction["scope_rebuild_at_control_max_abs"] > 1e-9:
            raise ValueError(f"The scope-weight rebuild at the control weight does not equal the control on {month}.")
        report = {"month": month, "rows": int(len(table)), "floor_seconds": floor, "control_rmse": rmse(label, base), "reproduction": reproduction,
                  "scope_rows": {"ltfm": int(ltfm.sum()), "matched_no_lirf_other": int(matched_other.sum()), "unmatched_no_lirf_other": int(unmatched_other.sum())},
                  "candidates": {}}
        for name, spec in decl["candidates"].items():
            values = rebuild(spec["ltfm"], spec["matched_no_lirf_other"], spec["unmatched_no_lirf_other"])
            changed = np.abs(values - base) > 1e-9
            if (changed & ~non_lirf).any():
                raise ValueError(f"{name} changes a LIRF row on {month}.")
            report["candidates"][name] = {**spec, "rmse": rmse(label, values), "ratio_to_control": rmse(label, values) / report["control_rmse"],
                                          "rows_differing_from_control": int(changed.sum()), "max_abs_difference_from_control": float(np.max(np.abs(values - base))),
                                          "paired_days_vs_control": paired_days(days, label, values, base, config["seed"]),
                                          "scopes": {k: {"control": rmse(label[m], base[m]), "candidate": rmse(label[m], values[m])}
                                                     for k, m in {"ltfm": ltfm, "matched_no_lirf_other": matched_other, "unmatched_no_lirf_other": unmatched_other}.items() if m.any()}}
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
              "december_symmetric_reading": {n: 100 * (e["months"]["2025-12"]["ratio_to_control"] - 1) for n, e in candidates.items()},
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
