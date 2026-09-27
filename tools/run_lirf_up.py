"""Declared LIRF lean-up round (configs/lirf_up_v1.json) on the fair development rows. No model is fitted.

Control: the current stack on fair rows (v34 -> u300 gate -> callsign 50% -> queue trust 0.75 on regular matches), as tools/run_mix_gate.py
builds it. A candidate adds configs/disagreement_gate_v1.json's lirf_up (LIRF matched rows take the column fully where it exceeds v9 by more
than T) in the same gate pass as u300.

  --score --select   -> runs/mix_gate/selection.json and docs/disagreement/mix_gate_v1_results.json
  --test-shipped     code-path test on the shipped-provenance rows (prints only pass / fail)
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
from run_adsb_v24 import extra_columns  # noqa: E402
from run_callsign_fix import offsets, scope_table  # noqa: E402
from run_disagreement_gate import gate, masks  # noqa: E402
from run_fair_weights import build, candidate_weights  # noqa: E402
from run_lomo_rows import assemble, configs  # noqa: E402
from run_queue_trust import offsets_all, segments  # noqa: E402

DECLARATION = "configs/lirf_up_v1.json"
ROWS = "runs/fair_dev/rows.parquet"
OUT = ROOT / "runs/lirf_up"
REPORT = ROOT / "docs/disagreement/lirf_up_v1_results.json"
SEED = 20260927


def gated_mix(f: pd.DataFrame, spec: dict) -> np.ndarray:
    c, n = f["committee"].to_numpy(dtype=float), f["nets"].to_numpy(dtype=float)
    share = np.full(len(f), 0.75)
    dm = n - c
    finite = np.isfinite(c) & np.isfinite(n)
    if "up" in spec:
        share = np.where(finite & (dm > spec["up"]), 1.0, share)
    if "down" in spec:
        share = np.where(finite & (dm < -spec["down"]), 0.25, share)
    return np.where(finite, (1 - share) * c + share * n, c)


def stack(f: pd.DataFrame, mix: np.ndarray, cs: pd.DataFrame, off_cs: dict, seg: pd.DataFrame, off: dict, lirf_up: float | None = None) -> np.ndarray:
    g = f.assign(mix=mix)
    outer, lirf = candidate_weights(read_json(ROOT / "configs/fair_weights_v1.json")["control"], {"lirf_v9": {"lirf_nonpop_unmatched": 0.0}})
    values = build(g, outer, lirf)
    airport, matched = g["airport"].to_numpy(), g["matched"].to_numpy()
    floor = g["floor"].to_numpy(dtype=float)
    column_m = (lirf["lirf_nonpop_matched"]["net"] * g["lirf_net"].to_numpy() + lirf["lirf_nonpop_matched"]["tree"] * g["lirf_tree"].to_numpy()) \
        / (lirf["lirf_nonpop_matched"]["net"] + lirf["lirf_nonpop_matched"]["tree"])
    spec = dict(read_json(ROOT / "configs/disagreement_gate_v1.json")["candidates"]["u300"])
    if lirf_up is not None:
        spec["lirf_up"] = lirf_up
    values = gate(values, spec, g["base"].to_numpy(dtype=float), mix,
                  g["v9"].to_numpy(dtype=float), column_m, masks(airport, matched, g["lirf_m"].to_numpy()), floor)
    s = cs.reindex(g[ID])
    inside = s["motion_start"].notna().to_numpy()
    taxi = s["takeoff"].to_numpy() - s["motion_start"].to_numpy() + np.array([off_cs.get(a, np.nan) for a in s["airport"].fillna("")])
    values = values.copy()
    values[inside] = np.maximum(0.5 * taxi[inside] + 0.5 * values[inside], floor[inside])
    q = seg.reindex(g[ID])
    q_in = q["in_scope"].fillna(False).to_numpy(dtype=bool) & q["source"].eq("regular").to_numpy()
    q_taxi = q["taxi"].to_numpy() + np.array([off.get(a, np.nan) for a in q["airport"].fillna("")])
    values[q_in] = np.maximum(0.75 * q_taxi[q_in] + 0.25 * values[q_in], floor[q_in])
    return values


def main() -> None:
    global ROWS, OUT, REPORT
    test = "--test-shipped" in sys.argv
    if test:
        ROWS, OUT, REPORT = "runs/fair_dev/rows_shipped.parquet", ROOT / "runs/lirf_up_test_shipped", ROOT / "runs/lirf_up_test_shipped/report.json"
    decl = read_json(ROOT / DECLARATION)
    if (OUT / "selection.json").exists():
        raise ValueError("Preserve the original selection.")
    config, net, lirf_net, v33 = configs()
    fair = pd.read_parquet(ROOT / ROWS)
    months = decl["recipe_months"]
    cs = scope_table(read_json(ROOT / "configs/callsign_fix_v1.json")).set_index(ID)
    off_cs, off = offsets(), offsets_all()
    seg = segments(months)
    mix_ref = read_json(ROOT / "docs/disagreement/mix_gate_v1_results.json")
    report = {"created_at_utc": utc_now(), "declaration": decl, "declaration_sha256": object_hash(decl), "rows_sha256": sha256_file(ROOT / ROWS), "audit_used": False,
              "months": {}, "candidates": {}}
    per_month = {}
    for month in months:
        f0 = fair[fair["month"] == month].reset_index(drop=True)
        f = assemble(f0, f0["v9"].to_numpy(), f0["residual"].to_numpy(), extra_columns(config, month, f0[ID].to_numpy()), config, net, lirf_net, v33)
        if not test and float(np.max(np.abs(f["mix"].to_numpy() - f0["mix"].to_numpy()))) > 1e-9:
            raise ValueError(f"The assembled mix does not equal the fair rows' mix on {month}.")
        y, days = f["label"].to_numpy(dtype=float), f["day"].to_numpy()
        control = stack(f, f["mix"].to_numpy(dtype=float), cs, off_cs, seg, off)
        if not test:  # the control must equal the mix-gate round's control
            if abs(rmse(y, control) - mix_ref["months"][month]["control_rmse"]) > 1e-9:
                raise ValueError(f"The control does not reproduce the mix-gate round's control on {month}.")
        report["months"].setdefault(month, {})["control_rmse"] = rmse(y, control)
        per_month[month] = {}
        for name, spec in decl["candidates"].items():
            mix = f["mix"].to_numpy(dtype=float)
            values = stack(f, mix, cs, off_cs, seg, off, spec["lirf_up"])
            if not np.isfinite(values).all():
                raise ValueError(f"{name} has non-finite values on {month}.")
            changed = np.abs(values - control) > 1e-9
            per_month[month][name] = {"rmse": rmse(y, values), "ratio_to_control": rmse(y, values) / rmse(y, control), "rows_differing_from_control": int(changed.sum()),
                                      "paired_days_vs_control": paired_days(days, y, values, control, SEED)}
        if not test:
            print(month, {k: round(v, 4) for k, v in report["months"][month].items()}, {n: round(100 * (e["ratio_to_control"] - 1), 4) for n, e in per_month[month].items()}, flush=True)
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
    report["model_code"] = {p: sha256_file(ROOT / p) for p in ["tools/run_lirf_up.py", "tools/run_queue_trust.py", "tools/run_callsign_fix.py", "tools/run_disagreement_gate.py", "tools/run_lomo_rows.py"]}
    report["environment"] = environment(ROOT)
    for path in [OUT / "selection.json", REPORT]:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, report)
    if test:
        print("test-shipped code path passed")
        return
    print({"selected": report["selected_candidate"], "eligible": eligible})
    for n, e in report["candidates"].items():
        iv = " ".join(f"[{e['months'][mo]['paired_days_vs_control']['rmse_difference_95pct_interval_seconds'][0]:+.3f},{e['months'][mo]['paired_days_vs_control']['rmse_difference_95pct_interval_seconds'][1]:+.3f}]" for mo in months)
        print(f"{n:7s} mean {100 * (e['mean_ratio'] - 1):+.4f}% " + " ".join(f"{mo}:{100 * (e['months'][mo]['ratio_to_control'] - 1):+.4f}" for mo in months)
              + f" | {iv} <0 {len(e['months_with_interval_below_zero'])}/{len(e['months_touched'])} eligible={e['eligible']}")


if __name__ == "__main__":
    main()
