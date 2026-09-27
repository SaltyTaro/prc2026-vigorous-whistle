"""Declared queue-trust round (configs/queue_trust_v1.json) on the fair development rows. No model is fitted.

Control: the u300_cs50 stack (v34 fair + tools/run_disagreement_gate u300 + tools/run_callsign_fix 50%). Scope: gate-covered ADS-B segments
(regular matches and callsign-fix matches) with >= 5 stops, >= 5 km path, <= 75% stationary and taxi = takeoff - motion_start > 3000 s.

  --score --select   -> runs/queue_trust/selection.json and docs/disagreement/queue_trust_v1_results.json
  --test-shipped     code-path test on runs/fair_dev/rows_shipped.parquet (prints only pass / fail)
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
from run_callsign_fix import OFFSET_MONTHS, offsets, scope_table  # noqa: E402
from run_disagreement_gate import gate, masks  # noqa: E402
from run_fair_weights import build, candidate_weights  # noqa: E402

DECLARATION = "configs/queue_trust_v1.json"
ROWS = "runs/fair_dev/rows.parquet"
OUT = ROOT / "runs/queue_trust"
REPORT = ROOT / "docs/disagreement/queue_trust_v1_results.json"
SEED = 20260927
SEGMENT = ["MVT_ID_mvt", "airport", "takeoff", "motion_start", "taxi_stops", "taxi_stopped_s", "taxi_path_m"]


def offsets_all() -> dict[str, float]:
    frames = []
    for month in OFFSET_MONTHS:
        y, m = int(month[:4]), int(month[5:])
        raw = pd.read_parquet(ROOT / f"data/raw/training_{month}-01_{y}-{m + 1:02d}-01.parquet", columns=[ID, TARGET])
        matches = pd.read_parquet(ROOT / f"data/external/adsblol/matches_v11/matches2_{month}.parquet", columns=[ID, "airport", "gate", "takeoff", "motion_start"])
        frames.append(matches[matches["gate"]].merge(raw, on=ID, how="inner"))
    g = pd.concat(frames, ignore_index=True)
    g = g[g[TARGET].notna() & g["motion_start"].notna()]
    return {a: float(np.median(x[TARGET] - (x["takeoff"] - x["motion_start"]))) for a, x in g.groupby("airport")}


def segments(months: list[str]) -> pd.DataFrame:
    """Regular gate matches of the months plus the callsign fix's scope rows (with their segment shape), one row per departure."""
    regular = pd.concat([pd.read_parquet(ROOT / f"data/external/adsblol/matches_v11/matches2_{m}.parquet") for m in months], ignore_index=True)
    regular = regular.loc[regular["gate"], SEGMENT].assign(source="regular")
    decl_cs = read_json(ROOT / "configs/callsign_fix_v1.json")
    cs_ids = set(scope_table(decl_cs)[ID])
    rem = pd.concat([pd.read_parquet(ROOT / p) for p in decl_cs["rematch"]["outputs"]], ignore_index=True)
    rem = rem[rem[ID].isin(cs_ids)].rename(columns={"iata_mapped_motion_start": "motion_start", "iata_mapped_taxi_stops": "taxi_stops",
                                                     "iata_mapped_taxi_stopped_s": "taxi_stopped_s", "iata_mapped_taxi_path_m": "taxi_path_m"})
    seg = pd.concat([regular, rem[SEGMENT].assign(source="callsign")], ignore_index=True)
    if seg[ID].duplicated().any():
        raise ValueError("A departure has both a regular and a callsign segment.")
    seg["taxi"] = seg["takeoff"] - seg["motion_start"]
    seg["in_scope"] = (seg["taxi_stops"] >= 5) & (seg["taxi_path_m"] >= 5000) & (seg["taxi_stopped_s"] / seg["taxi"] <= 0.75) & (seg["taxi"] > 3000)
    return seg.set_index(ID)


def stack_control(f: pd.DataFrame, cs: pd.DataFrame, off_cs: dict) -> tuple[np.ndarray, np.ndarray]:
    """(v34, u300_cs50) on fair rows, exactly as tools/run_callsign_fix.py builds them."""
    base_decl = read_json(ROOT / "configs/fair_weights_v1.json")
    outer, lirf = candidate_weights(base_decl["control"], {"lirf_v9": {"lirf_nonpop_unmatched": 0.0}})
    v34 = build(f, outer, lirf)
    airport, matched = f["airport"].to_numpy(), f["matched"].to_numpy()
    m = masks(airport, matched, f["lirf_m"].to_numpy())
    base, mix, v9 = (f[c].to_numpy(dtype=float) for c in ["base", "mix", "v9"])
    column_m = (lirf["lirf_nonpop_matched"]["net"] * f["lirf_net"].to_numpy() + lirf["lirf_nonpop_matched"]["tree"] * f["lirf_tree"].to_numpy()) \
        / (lirf["lirf_nonpop_matched"]["net"] + lirf["lirf_nonpop_matched"]["tree"])
    floor = f["floor"].to_numpy(dtype=float)
    values = gate(v34, read_json(ROOT / "configs/disagreement_gate_v1.json")["candidates"]["u300"], base, mix, v9, column_m, m, floor)
    s = cs.reindex(f[ID])
    inside = s["motion_start"].notna().to_numpy()
    taxi = s["takeoff"].to_numpy() - s["motion_start"].to_numpy() + np.array([off_cs.get(a, np.nan) for a in s["airport"].fillna("")])
    values = values.copy()
    values[inside] = np.maximum(0.5 * taxi[inside] + 0.5 * values[inside], floor[inside])
    return v34, values


def main() -> None:
    global ROWS, OUT, REPORT
    test = "--test-shipped" in sys.argv
    if test:
        ROWS, OUT, REPORT = "runs/fair_dev/rows_shipped.parquet", ROOT / "runs/queue_trust_test_shipped", ROOT / "runs/queue_trust_test_shipped/report.json"
    decl = read_json(ROOT / DECLARATION)
    if (OUT / "selection.json").exists():
        raise ValueError("Preserve the original selection.")
    off, off_cs = offsets_all(), offsets()
    cs = scope_table(read_json(ROOT / "configs/callsign_fix_v1.json")).set_index(ID)
    months = decl["recipe_months"]
    seg = segments(months)
    rows = pd.read_parquet(ROOT / ROWS)
    recorded = read_json(ROOT / "docs/disagreement/callsign_fix_v1_results.json")["candidates"]["u300_cs50"]
    report = {"created_at_utc": utc_now(), "declaration": decl, "declaration_sha256": object_hash(decl), "rows_sha256": sha256_file(ROOT / ROWS),
              "offsets_s": off, "audit_used": False, "months": {}, "candidates": {}}
    per_month = {}
    for month in months:
        f = rows[rows["month"] == month].reset_index(drop=True)
        y, days = f["label"].to_numpy(dtype=float), f["day"].to_numpy()
        v34, control = stack_control(f, cs, off_cs)
        if not test and abs(rmse(y, control) / rmse(y, v34) - recorded["months"][month]["ratio_to_control"]) > 1e-12:
            raise ValueError(f"The control does not reproduce the recorded u300_cs50 on {month}.")
        s = seg.reindex(f[ID])
        inside = s["in_scope"].fillna(False).to_numpy(dtype=bool)
        taxi = s["taxi"].to_numpy() + np.array([off.get(a, np.nan) for a in s["airport"].fillna("")])
        floor = f["floor"].to_numpy(dtype=float)
        report["months"][month] = {"rows": int(len(f)), "control_rmse": rmse(y, control), "v34_rmse": rmse(y, v34), "scope_rows": int(inside.sum()),
                                   "scope_rows_by_source": s.loc[inside, "source"].value_counts().to_dict() if inside.any() else {},
                                   "scope_rmse_control": rmse(y[inside], control[inside]) if inside.any() else None,
                                   "scope_rmse_observed": rmse(y[inside], taxi[inside]) if inside.any() else None}
        per_month[month] = {}
        for name, spec in decl["candidates"].items():
            values = control.copy()
            values[inside] = np.maximum(spec["weight"] * taxi[inside] + (1 - spec["weight"]) * control[inside], floor[inside])
            if not np.isfinite(values).all():
                raise ValueError(f"{name} has non-finite values on {month}.")
            changed = np.abs(values - control) > 1e-9
            per_month[month][name] = {"rmse": rmse(y, values), "ratio_to_control": rmse(y, values) / rmse(y, control), "ratio_to_v34": rmse(y, values) / rmse(y, v34),
                                      "rows_differing_from_control": int(changed.sum()), "paired_days_vs_control": paired_days(days, y, values, control, SEED)}
        if not test:
            print(month, report["months"][month], {n: round(100 * (e["ratio_to_control"] - 1), 4) for n, e in per_month[month].items()}, flush=True)
    for name in decl["candidates"]:
        entry = {**decl["candidates"][name], "months": {mo: per_month[mo][name] for mo in months}}
        ratios = [entry["months"][mo]["ratio_to_control"] for mo in months]
        entry["mean_ratio"] = float(np.mean(ratios))
        entry["mean_ratio_to_v34"] = float(np.mean([entry["months"][mo]["ratio_to_v34"] for mo in months]))
        touched = [mo for mo in months if entry["months"][mo]["rows_differing_from_control"] > 0]
        below = [mo for mo in touched if entry["months"][mo]["paired_days_vs_control"]["rmse_difference_95pct_interval_seconds"][1] < 0]
        entry["months_touched"], entry["months_with_interval_below_zero"] = touched, below
        entry["eligible"] = bool(touched and all(r <= 1.0 for r in ratios) and entry["mean_ratio"] <= 1 - decl["minimum_mean_improvement_fraction"] and len(below) == len(touched))
        report["candidates"][name] = entry
    eligible = [n for n, e in report["candidates"].items() if e["eligible"]]
    report["selected_candidate"] = min(eligible, key=lambda n: (report["candidates"][n]["mean_ratio"], n)) if eligible else None
    report["eligible"] = eligible
    report["model_code"] = {p: sha256_file(ROOT / p) for p in ["tools/run_queue_trust.py", "tools/run_callsign_fix.py", "tools/run_disagreement_gate.py", "tools/run_fair_weights.py"]}
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
        print(f"{n:6s} mean {100 * (e['mean_ratio'] - 1):+.4f}% (vs v34 {100 * (e['mean_ratio_to_v34'] - 1):+.4f}%) " + " ".join(f"{mo}:{100 * (e['months'][mo]['ratio_to_control'] - 1):+.4f}" for mo in months)
              + f" | {iv} <0 {len(e['months_with_interval_below_zero'])}/{len(e['months_touched'])} eligible={e['eligible']}")


if __name__ == "__main__":
    main()
