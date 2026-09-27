"""Declared disagreement-gated outer weight round (configs/disagreement_gate_v1.json) on the fair development rows
(runs/fair_dev/rows.parquet). No model is fitted.

The v33 rebuild (tools/run_fair_weights.build with fair_weights_v1's control) must equal the rows' v33 column to 1e-9 s per row; the control
is v34 (v33 with the LIRF non-population unmatched v9 share 0). Candidates override the outer weight / LIRF v9 share only on rows where the
v9 side and the fold side disagree by more than the declared threshold, toward the higher side (see the declaration's candidate_semantics).

  --score --select   -> runs/disagreement_gate/selection.json and docs/disagreement/gate_v1_results.json
  --test-shipped     code-path test on runs/fair_dev/rows_shipped.parquet (never a result)
  --ranking-exposure -> also count, label-free, the ranking rows each candidate would touch (runs/disagreement_rank.parquet)
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
from run_adsb_v11 import paired_days, rmse  # noqa: E402
from run_fair_weights import build, candidate_weights  # noqa: E402

DECLARATION = "configs/disagreement_gate_v1.json"
ROWS = "runs/fair_dev/rows.parquet"
OUT = ROOT / "runs/disagreement_gate"
REPORT = ROOT / "docs/disagreement/gate_v1_results.json"
SEED = 20260927


def masks(airport: np.ndarray, matched: np.ndarray, lirf_m: np.ndarray) -> dict[str, np.ndarray]:
    other = (airport != "LIRF") & (airport != "LTFM")
    return {"unm": other & ~matched, "mat": other & matched, "lirf_m": lirf_m & (airport == "LIRF")}


def gate(control: np.ndarray, spec: dict, base: np.ndarray, mix: np.ndarray, v9: np.ndarray, column_m: np.ndarray,
         m: dict[str, np.ndarray], floor: np.ndarray) -> np.ndarray:
    d, d_lirf = mix - base, column_m - v9
    values = control.copy()
    if "unm_up" in spec:
        k = m["unm"] & (d > spec["unm_up"])
        values[k] = np.maximum(mix[k], floor[k])
    if "mat_down" in spec:
        k = m["mat"] & (d < -spec["mat_down"])
        values[k] = np.maximum(base[k], floor[k])
    if "lirf_up" in spec:
        k = m["lirf_m"] & (d_lirf > spec["lirf_up"])
        values[k] = np.maximum(column_m[k], floor[k])
    return values


def exposure(decl: dict) -> dict:
    """Label-free: ranking rows each candidate touches, from the audit's ranking components (runs/disagreement_rank.parquet)."""
    r = pd.read_parquet(ROOT / "runs/disagreement_rank.parquet")
    airport, matched = r["airport"].to_numpy(), r["matched"].to_numpy()
    lirf_m = r["group"].eq("LIRF:matched").to_numpy()
    m = masks(airport, matched, lirf_m)
    d = r["d"].to_numpy()
    out = {}
    for name, spec in decl["candidates"].items():
        e = {}
        for period in ["2026-01", "2026-07"]:
            p = r["period"].to_numpy() == period
            e[period] = {"unm_up": int(np.sum(p & m["unm"] & (d > spec.get("unm_up", np.inf)))),
                         "mat_down": int(np.sum(p & m["mat"] & (d < -spec.get("mat_down", np.inf)))),
                         "lirf_up": int(np.sum(p & m["lirf_m"] & (d > spec.get("lirf_up", np.inf))))}
        out[name] = e
    return out


def main() -> None:
    global ROWS, OUT, REPORT
    if "--test-shipped" in sys.argv:  # code-path test on the shipped-provenance rows; never a result
        ROWS, OUT, REPORT = "runs/fair_dev/rows_shipped.parquet", ROOT / "runs/disagreement_gate_test_shipped", ROOT / "runs/disagreement_gate_test_shipped/report.json"
    decl = read_json(ROOT / DECLARATION)
    if (OUT / "selection.json").exists():
        raise ValueError("Preserve the original selection.")
    base_decl = read_json(ROOT / "configs/fair_weights_v1.json")
    v33_outer, v33_lirf = base_decl["control"]["outer"], base_decl["control"]["lirf"]
    outer, lirf = candidate_weights(base_decl["control"], {"lirf_v9": {"lirf_nonpop_unmatched": 0.0}})
    rows = pd.read_parquet(ROOT / ROWS)
    months = decl["recipe_months"]
    report = {"created_at_utc": utc_now(), "declaration": decl, "declaration_sha256": object_hash(decl), "rows_sha256": sha256_file(ROOT / ROWS),
              "audit_used": False, "months": {}, "candidates": {}}
    per_month = {}
    for month in months:
        f = rows[rows["month"] == month].reset_index(drop=True)
        y, days = f["label"].to_numpy(dtype=float), f["day"].to_numpy()
        v33 = build(f, v33_outer, v33_lirf)
        check = float(np.max(np.abs(v33 - f["v33"].to_numpy())))
        if not check <= 1e-9:  # also catches NaN
            raise ValueError(f"The v33 rebuild does not equal the fair v33 on {month} (max {check}).")
        control = build(f, outer, lirf)
        airport, matched = f["airport"].to_numpy(), f["matched"].to_numpy()
        m = masks(airport, matched, f["lirf_m"].to_numpy())
        base, mix, v9 = (f[c].to_numpy(dtype=float) for c in ["base", "mix", "v9"])
        column_m = (lirf["lirf_nonpop_matched"]["net"] * f["lirf_net"].to_numpy() + lirf["lirf_nonpop_matched"]["tree"] * f["lirf_tree"].to_numpy()) \
            / (lirf["lirf_nonpop_matched"]["net"] + lirf["lirf_nonpop_matched"]["tree"])
        floor = f["floor"].to_numpy(dtype=float)
        # the control's own weights must be what the gate overrides
        w = np.where(airport == "LTFM", outer["ltfm"], np.where(matched, outer["matched_other"], outer["unmatched_other"]))
        recon = np.maximum((1 - w) * base + w * mix, floor)
        other = (airport != "LIRF")
        if not float(np.max(np.abs(recon[other] - control[other]))) <= 1e-9:
            raise ValueError(f"Control non-LIRF rows do not follow the outer-weight formula on {month}.")
        report["months"][month] = {"rows": int(len(f)), "control_rmse": rmse(y, control), "v33_rebuild_max_abs": check,
                                   "v34_vs_v33_fair_pct": 100 * (rmse(y, control) / rmse(y, v33) - 1)}
        per_month[month] = {}
        for name, spec in decl["candidates"].items():
            values = gate(control, spec, base, mix, v9, column_m, m, floor)
            if not np.isfinite(values).all():
                raise ValueError(f"{name} has non-finite values on {month}.")
            changed = np.abs(values - control) > 1e-9
            per_month[month][name] = {"rmse": rmse(y, values), "ratio_to_control": rmse(y, values) / rmse(y, control), "rows_differing_from_control": int(changed.sum()),
                                      "sse_change": float(np.sum((y - values) ** 2) - np.sum((y - control) ** 2)),
                                      "paired_days_vs_control": paired_days(days, y, values, control, SEED)}
        print(month, {"control": round(report["months"][month]["control_rmse"], 4)},
              {n: round(100 * (e["ratio_to_control"] - 1), 4) for n, e in per_month[month].items()}, flush=True)
    for name in decl["candidates"]:
        entry = {**decl["candidates"][name], "months": {mo: per_month[mo][name] for mo in months}}
        ratios = [entry["months"][mo]["ratio_to_control"] for mo in months]
        entry["mean_ratio"] = float(np.mean(ratios))
        touched = [mo for mo in months if entry["months"][mo]["rows_differing_from_control"] > 0]
        below = [mo for mo in touched if entry["months"][mo]["paired_days_vs_control"]["rmse_difference_95pct_interval_seconds"][1] < 0]
        entry["months_touched"], entry["months_with_interval_below_zero"] = touched, below
        entry["eligible"] = bool(touched and all(r <= 1.0 for r in ratios) and entry["mean_ratio"] <= 1 - decl["minimum_mean_improvement_fraction"] and len(below) == len(touched))
        report["candidates"][name] = entry
    eligible = [n for n, e in report["candidates"].items() if e["eligible"]]
    report["selected_candidate"] = min(eligible, key=lambda n: (report["candidates"][n]["mean_ratio"], n)) if eligible else None
    report["eligible"] = eligible
    if "--ranking-exposure" in sys.argv:
        report["ranking_exposure_label_free"] = exposure(decl)
    report["model_code"] = {p: sha256_file(ROOT / p) for p in ["tools/run_disagreement_gate.py", "tools/run_fair_weights.py", "tools/run_fair_dev.py"]}
    report["environment"] = environment(ROOT)
    for path in [OUT / "selection.json", REPORT]:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, report)
    print({"selected": report["selected_candidate"], "eligible": eligible})
    for n, e in report["candidates"].items():
        print(f"{n:16s} mean {100 * (e['mean_ratio'] - 1):+.4f}% " + " ".join(f"{mo}:{100 * (e['months'][mo]['ratio_to_control'] - 1):+.4f}" for mo in months)
              + " | iv " + " ".join(f"[{e['months'][mo]['paired_days_vs_control']['rmse_difference_95pct_interval_seconds'][0]:+.3f},{e['months'][mo]['paired_days_vs_control']['rmse_difference_95pct_interval_seconds'][1]:+.3f}]" for mo in months)
              + f" <0 {len(e['months_with_interval_below_zero'])}/{len(e['months_touched'])} eligible={e['eligible']}")
    if "ranking_exposure_label_free" in report:
        print(report["ranking_exposure_label_free"])


if __name__ == "__main__":
    main()
