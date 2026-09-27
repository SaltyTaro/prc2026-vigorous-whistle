"""Declared ADS-B callsign-fix round (configs/callsign_fix_v1.json) on the fair development rows. No model is fitted.

Scope: no-NM departures at EDDF / EHAM whose existing key found no ADS-B segment but whose IATA->ICAO mapped callsign yields a gate segment
(tools/adsb_rematch_nonm.py outputs). Observed taxi = takeoff - motion_start + offset[airport], the offset being the median of label - taxi over
the regular gate-matched departures of the non-recipe 2025 months. Control: v34 (fair) exactly as tools/run_disagreement_gate.py builds it.

  --score --select   -> runs/callsign_fix/selection.json and docs/disagreement/callsign_fix_v1_results.json
  --test-shipped     code-path test on runs/fair_dev/rows_shipped.parquet (never a result)
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
from prc2026.features import ID, TARGET  # noqa: E402
from run_adsb_v11 import paired_days, rmse  # noqa: E402
from run_disagreement_gate import gate, masks  # noqa: E402
from run_fair_weights import build, candidate_weights  # noqa: E402

DECLARATION = "configs/callsign_fix_v1.json"
ROWS = "runs/fair_dev/rows.parquet"
OUT = ROOT / "runs/callsign_fix"
REPORT = ROOT / "docs/disagreement/callsign_fix_v1_results.json"
SEED = 20260927
OFFSET_MONTHS = ["2025-02", "2025-03", "2025-04", "2025-05", "2025-06", "2025-08", "2025-09", "2025-10"]


def offsets() -> dict[str, float]:
    out = {}
    frames = []
    for month in OFFSET_MONTHS:
        y, m = int(month[:4]), int(month[5:])
        raw = pd.read_parquet(ROOT / f"data/raw/training_{month}-01_{y}-{m + 1:02d}-01.parquet", columns=[ID, TARGET])
        matches = pd.read_parquet(ROOT / f"data/external/adsblol/matches_v11/matches2_{month}.parquet", columns=[ID, "airport", "gate", "takeoff", "motion_start"])
        frames.append(matches[matches["gate"]].merge(raw, on=ID, how="inner"))
    g = pd.concat(frames, ignore_index=True)
    for airport in ["EDDF", "EHAM"]:
        a = g[(g["airport"] == airport) & g[TARGET].notna() & g["motion_start"].notna()]
        out[airport] = float(np.median(a[TARGET] - (a["takeoff"] - a["motion_start"])))
    return out


def scope_table(decl: dict) -> pd.DataFrame:
    frames = []
    for path, sha in decl["rematch"]["outputs"].items():
        if sha256_file(ROOT / path) != sha:
            raise ValueError(f"{path} changed since the declaration.")
        frames.append(pd.read_parquet(ROOT / path))
    m = pd.concat(frames, ignore_index=True)
    if m[ID].duplicated().any():
        raise ValueError("A departure appears in two rematch outputs.")
    scope = m["key_icao"].isna() & m["iata_mapped_icao"].notna() & m["iata_mapped_gate"].fillna(False).astype(bool) & m["iata_mapped_motion_start"].notna()
    return m.loc[scope, [ID, "airport", "takeoff", "iata_mapped_motion_start"]].rename(columns={"iata_mapped_motion_start": "motion_start"})


def main() -> None:
    global ROWS, OUT, REPORT
    if "--test-shipped" in sys.argv:
        ROWS, OUT, REPORT = "runs/fair_dev/rows_shipped.parquet", ROOT / "runs/callsign_fix_test_shipped", ROOT / "runs/callsign_fix_test_shipped/report.json"
    decl = read_json(ROOT / DECLARATION)
    if (OUT / "selection.json").exists():
        raise ValueError("Preserve the original selection.")
    gate_decl = read_json(ROOT / "configs/disagreement_gate_v1.json")
    base_decl = read_json(ROOT / "configs/fair_weights_v1.json")
    v33_outer, v33_lirf = base_decl["control"]["outer"], base_decl["control"]["lirf"]
    outer, lirf = candidate_weights(base_decl["control"], {"lirf_v9": {"lirf_nonpop_unmatched": 0.0}})
    off = offsets()
    scope = scope_table(decl)
    rows = pd.read_parquet(ROOT / ROWS)
    months = decl["recipe_months"]
    report = {"created_at_utc": utc_now(), "declaration": decl, "declaration_sha256": object_hash(decl), "rows_sha256": sha256_file(ROOT / ROWS),
              "offsets_s": off, "audit_used": False, "months": {}, "candidates": {}}
    per_month = {}
    for month in months:
        f = rows[rows["month"] == month].reset_index(drop=True)
        y, days = f["label"].to_numpy(dtype=float), f["day"].to_numpy()
        v33 = build(f, v33_outer, v33_lirf)
        check = float(np.max(np.abs(v33 - f["v33"].to_numpy())))
        if not check <= 1e-9:
            raise ValueError(f"The v33 rebuild does not equal the fair v33 on {month} (max {check}).")
        control = build(f, outer, lirf)
        airport, matched = f["airport"].to_numpy(), f["matched"].to_numpy()
        m = masks(airport, matched, f["lirf_m"].to_numpy())
        base, mix, v9 = (f[c].to_numpy(dtype=float) for c in ["base", "mix", "v9"])
        column_m = (lirf["lirf_nonpop_matched"]["net"] * f["lirf_net"].to_numpy() + lirf["lirf_nonpop_matched"]["tree"] * f["lirf_tree"].to_numpy()) \
            / (lirf["lirf_nonpop_matched"]["net"] + lirf["lirf_nonpop_matched"]["tree"])
        floor = f["floor"].to_numpy(dtype=float)
        u300 = gate(control, gate_decl["candidates"]["u300"], base, mix, v9, column_m, m, floor)
        s = scope.set_index(ID).reindex(f[ID])
        inside = s["motion_start"].notna().to_numpy()
        if np.any(inside & (airport != s["airport"].fillna("").to_numpy())) or np.any(inside & matched):
            raise ValueError(f"A scope row is ADS-B-matched already or at another airport on {month}.")
        taxi = s["takeoff"].to_numpy() - s["motion_start"].to_numpy() + np.array([off.get(a, np.nan) for a in s["airport"].fillna("")])
        report["months"][month] = {"rows": int(len(f)), "control_rmse": rmse(y, control), "scope_rows": int(inside.sum()),
                                   "scope_rmse_control": rmse(y[inside], control[inside]) if inside.any() else None,
                                   "scope_rmse_observed_taxi": rmse(y[inside], taxi[inside]) if inside.any() else None}
        per_month[month] = {}
        for name, spec in decl["candidates"].items():
            start = u300 if spec.get("gate") == "u300" else control
            values = start.copy()
            values[inside] = np.maximum(spec["weight"] * taxi[inside] + (1 - spec["weight"]) * start[inside], floor[inside])
            if not np.isfinite(values).all():
                raise ValueError(f"{name} has non-finite values on {month}.")
            changed = np.abs(values - control) > 1e-9
            per_month[month][name] = {"rmse": rmse(y, values), "ratio_to_control": rmse(y, values) / rmse(y, control), "rows_differing_from_control": int(changed.sum()),
                                      "scope_rmse": rmse(y[inside], values[inside]) if inside.any() else None,
                                      "sse_change": float(np.sum((y - values) ** 2) - np.sum((y - control) ** 2)),
                                      "paired_days_vs_control": paired_days(days, y, values, control, SEED)}
        print(month, report["months"][month], {n: round(100 * (e["ratio_to_control"] - 1), 4) for n, e in per_month[month].items()}, flush=True)
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
    report["model_code"] = {p: sha256_file(ROOT / p) for p in ["tools/run_callsign_fix.py", "tools/run_disagreement_gate.py", "tools/run_fair_weights.py", "tools/adsb_rematch_nonm.py"]}
    report["environment"] = environment(ROOT)
    for path in [OUT / "selection.json", REPORT]:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, report)
    print({"selected": report["selected_candidate"], "eligible": eligible, "offsets": off})
    for n, e in report["candidates"].items():
        iv = " ".join(f"[{e['months'][mo]['paired_days_vs_control']['rmse_difference_95pct_interval_seconds'][0]:+.3f},{e['months'][mo]['paired_days_vs_control']['rmse_difference_95pct_interval_seconds'][1]:+.3f}]" for mo in months)
        print(f"{n:10s} mean {100 * (e['mean_ratio'] - 1):+.4f}% " + " ".join(f"{mo}:{100 * (e['months'][mo]['ratio_to_control'] - 1):+.4f}" for mo in months)
              + f" | {iv} <0 {len(e['months_with_interval_below_zero'])}/{len(e['months_touched'])} eligible={e['eligible']}")


if __name__ == "__main__":
    main()
