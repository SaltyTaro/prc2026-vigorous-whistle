"""Declared LOMO weight round (configs/lomo_weights_v1.json) on runs/lomo_dev/rows.parquet. No model is fitted.

Control v34: outer weights by scope, LIRF v9 shares (unmatched 0), network share 0.75. Candidates change one of them, or add the u300 gate
(configs/disagreement_gate_v1.json) and / or the callsign fix (configs/callsign_fix_v1.json, 50%) on top of the candidate's own values.

  --score --select    -> runs/lomo_weights/selection.json and docs/lomo_dev/lomo_weights_v1_results.json
  --test-fair         code-path test on the fair rows (runs/fair_dev/rows.parquet plus committee/nets rebuilt); prints nothing but pass/fail
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import environment, object_hash, read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID  # noqa: E402
from run_adsb_nn_dev import mixed  # noqa: E402
from run_adsb_v11 import paired_days, rmse  # noqa: E402
from run_callsign_fix import offsets, scope_table  # noqa: E402
from run_disagreement_gate import gate, masks  # noqa: E402
from run_fair_weights import build, candidate_weights  # noqa: E402

DECLARATION = "configs/lomo_weights_v1.json"
ROWS = "runs/lomo_dev/rows.parquet"
OUT = ROOT / "runs/lomo_weights"
REPORT = ROOT / "docs/lomo_dev/lomo_weights_v1_results.json"
SEED = 20260927


def values_for(f: pd.DataFrame, control: dict, spec: dict, gate_spec: dict, cs: pd.DataFrame, off: dict) -> np.ndarray:
    outer, lirf = candidate_weights(control, spec)
    share = spec.get("network_share", control["network_share"])
    g = f.assign(mix=mixed(f["committee"].to_numpy(dtype=float), f["nets"].to_numpy(dtype=float), share))
    values = build(g, outer, lirf)
    airport, matched = g["airport"].to_numpy(), g["matched"].to_numpy()
    floor = g["floor"].to_numpy(dtype=float)
    if spec.get("gate") == "u300":
        m = masks(airport, matched, g["lirf_m"].to_numpy())
        column_m = (lirf["lirf_nonpop_matched"]["net"] * g["lirf_net"].to_numpy() + lirf["lirf_nonpop_matched"]["tree"] * g["lirf_tree"].to_numpy()) \
            / (lirf["lirf_nonpop_matched"]["net"] + lirf["lirf_nonpop_matched"]["tree"])
        values = gate(values, gate_spec, g["base"].to_numpy(dtype=float), g["mix"].to_numpy(dtype=float), g["v9"].to_numpy(dtype=float), column_m, m, floor)
    if "callsign" in spec:
        s = cs.reindex(g[ID])
        inside = s["motion_start"].notna().to_numpy()
        if np.any(inside & matched):
            raise ValueError("A callsign scope row is ADS-B-matched already.")
        taxi = s["takeoff"].to_numpy() - s["motion_start"].to_numpy() + np.array([off.get(a, np.nan) for a in s["airport"].fillna("")])
        w = spec["callsign"]
        values = values.copy()
        values[inside] = np.maximum(w * taxi[inside] + (1 - w) * values[inside], floor[inside])
    return values


def main() -> None:
    global ROWS, OUT, REPORT
    test = "--test-fair" in sys.argv
    decl = read_json(ROOT / DECLARATION)
    if test:
        OUT, REPORT = ROOT / "runs/lomo_weights_test_fair", ROOT / "runs/lomo_weights_test_fair/report.json"
    if (OUT / "selection.json").exists():
        raise ValueError("Preserve the original selection.")
    if test:
        from run_lomo_rows import assemble, configs
        from run_adsb_v24 import extra_columns
        config, net, lirf_net, v33 = configs()
        fair = pd.read_parquet(ROOT / "runs/fair_dev/rows.parquet")
        rows = pd.concat([assemble(fr, fr["v9"].to_numpy(), fr["residual"].to_numpy(), extra_columns(config, mo, fr[ID].to_numpy()), config, net, lirf_net, v33)
                          for mo, fr in ((mo, fair[fair["month"] == mo].reset_index(drop=True)) for mo in ["2025-01", "2025-07", "2025-11", "2025-12"])], ignore_index=True)
        rows_sha = "test"
    else:
        rows = pd.read_parquet(ROOT / ROWS)
        rows_sha = sha256_file(ROOT / ROWS)
    control = decl["control"]
    gate_spec = read_json(ROOT / "configs/disagreement_gate_v1.json")["candidates"]["u300"]
    off = offsets()
    cs = scope_table(read_json(ROOT / "configs/callsign_fix_v1.json")).set_index(ID)
    months = decl["recipe_months"] + decl.get("reported_months", [])
    report = {"created_at_utc": utc_now(), "declaration": decl, "declaration_sha256": object_hash(decl), "rows_sha256": rows_sha, "audit_used": False,
              "months": {}, "candidates": {}}
    per_month = {}
    for month in months:
        f = rows[rows["month"] == month].reset_index(drop=True)
        y, days = f["label"].to_numpy(dtype=float), f["day"].to_numpy()
        mix_check = float(np.nanmax(np.abs(mixed(f["committee"].to_numpy(dtype=float), f["nets"].to_numpy(dtype=float), 0.75) - f["mix"].to_numpy(dtype=float))))
        if not mix_check <= 1e-9:
            raise ValueError(f"The mix column does not equal mixed(committee, nets, 0.75) on {month} ({mix_check}).")
        base_values = values_for(f, control, {}, gate_spec, cs, off)
        v33_check = float(np.max(np.abs(build(f, {"ltfm": 1.0, "matched_other": 0.6, "unmatched_other": 0.7},
                                              {"lirf_nonpop_matched": {"v9": 0.15, "net": 0.40, "tree": 0.45}, "lirf_nonpop_unmatched": {"v9": 0.40, "net": 0.35, "tree": 0.25}}) - f["v33"].to_numpy())))
        if not v33_check <= 1e-9:
            raise ValueError(f"The v33 rebuild does not equal the rows' v33 on {month} ({v33_check}).")
        report["months"][month] = {"rows": int(len(f)), "provenance": sorted(set(f["provenance"])) if "provenance" in f else ["fair"], "control_rmse": rmse(y, base_values)}
        per_month[month] = {}
        for name, spec in decl["candidates"].items():
            values = values_for(f, control, spec, gate_spec, cs, off)
            if not np.isfinite(values).all():
                raise ValueError(f"{name} has non-finite values on {month}.")
            changed = np.abs(values - base_values) > 1e-9
            per_month[month][name] = {"rmse": rmse(y, values), "ratio_to_control": rmse(y, values) / rmse(y, base_values), "rows_differing_from_control": int(changed.sum()),
                                      "paired_days_vs_control": paired_days(days, y, values, base_values, SEED)}
        if not test:
            print(month, {"control": round(report["months"][month]["control_rmse"], 4)}, {n: round(100 * (e["ratio_to_control"] - 1), 4) for n, e in per_month[month].items()}, flush=True)
    recipe = decl["recipe_months"]
    for name in decl["candidates"]:
        entry = {**decl["candidates"][name], "months": {mo: per_month[mo][name] for mo in months}}
        ratios = [entry["months"][mo]["ratio_to_control"] for mo in recipe]
        entry["mean_ratio"] = float(np.mean(ratios))
        touched = [mo for mo in recipe if entry["months"][mo]["rows_differing_from_control"] > 0]
        below = [mo for mo in touched if entry["months"][mo]["paired_days_vs_control"]["rmse_difference_95pct_interval_seconds"][1] < 0]
        entry["months_touched"], entry["months_with_interval_below_zero"] = touched, below
        entry["eligible"] = bool(touched and all(r <= 1.0 for r in ratios) and entry["mean_ratio"] <= 1 - decl["minimum_mean_improvement_fraction"] and len(below) == len(touched))
        report["candidates"][name] = entry
    eligible = [n for n, e in report["candidates"].items() if e["eligible"]]
    report["selected_candidate"] = min(eligible, key=lambda n: (report["candidates"][n]["mean_ratio"], n)) if eligible else None
    report["eligible"] = eligible
    report["model_code"] = {p: sha256_file(ROOT / p) for p in ["tools/run_lomo_round.py", "tools/run_lomo_rows.py", "tools/run_fair_weights.py", "tools/run_disagreement_gate.py", "tools/run_callsign_fix.py"]}
    report["environment"] = environment(ROOT)
    for path in [OUT / "selection.json", REPORT]:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, report)
    if test:
        print("test-fair code path passed")
        return
    print({"selected": report["selected_candidate"], "eligible": eligible})
    for n, e in report["candidates"].items():
        iv = " ".join(f"[{e['months'][mo]['paired_days_vs_control']['rmse_difference_95pct_interval_seconds'][0]:+.3f},{e['months'][mo]['paired_days_vs_control']['rmse_difference_95pct_interval_seconds'][1]:+.3f}]" for mo in recipe)
        print(f"{n:12s} mean {100 * (e['mean_ratio'] - 1):+.4f}% " + " ".join(f"{mo}:{100 * (e['months'][mo]['ratio_to_control'] - 1):+.4f}" for mo in months)
              + f" | {iv} <0 {len(e['months_with_interval_below_zero'])}/{len(e['months_touched'])} eligible={e['eligible']}")


if __name__ == "__main__":
    main()
