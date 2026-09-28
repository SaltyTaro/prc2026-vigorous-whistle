"""Declared snow-day / higher-side round (configs/snow_higher_v1.json) on runs/lomo_dev/rows.parquet. No model is fitted.

Control: the v35 stack (tools/run_mix_gate.stack: v34 weights -> u300 gate -> callsign 50% -> queue trust 0.75 on regular matches).
A candidate changes the committee mix's network share where nets and committee disagree strongly (net_up / net_down), replaces the LIRF
network by the LIRF tree table where the network lies far below it (lirf_net_down), rebuilds the whole stack on those inputs, and then
lifts untracked non-LIRF rows on METAR snow days by k x the airport-day's predicted rise (snow_k).

  --score --select   -> runs/snow_higher/selection.json and docs/disagreement/snow_higher_v1_results.json
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
from run_mix_gate import stack  # noqa: E402
from run_queue_trust import offsets_all, segments  # noqa: E402
from snow_day_scan import weather_days  # noqa: E402

DECLARATION = "configs/snow_higher_v1.json"
ROWS = "runs/lomo_dev/rows.parquet"
OUT = ROOT / "runs/snow_higher"
REPORT = ROOT / "docs/disagreement/snow_higher_v1_results.json"
SEED = 20260928


def candidate_inputs(f: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, np.ndarray]:
    """(rows with the candidate's LIRF network column, the candidate's committee mix)."""
    c, n = f["committee"].to_numpy(dtype=float), f["nets"].to_numpy(dtype=float)
    finite = np.isfinite(c) & np.isfinite(n)
    share = np.full(len(f), 0.75)
    if "net_up" in spec:
        share = np.where(finite & (n - c > spec["net_up"]), 1.0, share)
    if "net_down" in spec:
        share = np.where(finite & (n - c < -spec["net_down"]), 0.5, share)
    mix = np.where(finite, (1 - share) * c + share * n, c)
    g = f
    if "lirf_net_down" in spec:
        net, tree = f["lirf_net"].to_numpy(dtype=float), f["lirf_tree"].to_numpy(dtype=float)
        nonpop = (f["airport"].to_numpy() == "LIRF") & (f["lirf_m"].to_numpy() | f["lirf_u"].to_numpy())
        g = f.assign(lirf_net=np.where(nonpop & np.isfinite(net) & np.isfinite(tree) & (net - tree < -spec["lirf_net_down"]), tree, net))
    return g, mix


def snow_lift(values: np.ndarray, f: pd.DataFrame, snow: set, k: float) -> tuple[np.ndarray, np.ndarray]:
    """values + k * max(day_lift, 0) on untracked non-LIRF rows of snow airport-days (one month's rows); day_lift from `values`."""
    airport, day = f["airport"].to_numpy(), f["day"].to_numpy()
    frame = pd.DataFrame({"airport": airport, "day": day, "v": values})
    means = frame.groupby(["airport", "day"])["v"].mean().rename("mean").reset_index()
    means["snow"] = [(a, d) in snow for a, d in zip(means["airport"], means["day"])]
    normal = means[~means["snow"]].groupby("airport")["mean"].median()
    means["lift"] = np.where(means["snow"], means["mean"] - means["airport"].map(normal), 0.0)
    lift = frame.merge(means[["airport", "day", "lift"]], on=["airport", "day"], how="left")["lift"].fillna(0.0).to_numpy()
    rows = (lift > 0) & ~f["matched"].to_numpy() & (airport != "LIRF")
    out = values.copy()
    out[rows] = np.maximum(values[rows] + k * lift[rows], f["floor"].to_numpy(dtype=float)[rows])
    return out, rows


def main() -> None:
    decl = read_json(ROOT / DECLARATION)
    if (OUT / "selection.json").exists():
        raise ValueError("Preserve the original selection.")
    rows = pd.read_parquet(ROOT / ROWS)
    if sha256_file(ROOT / ROWS) != decl["rows"].split("SHA-256 ")[1].split(")")[0]:
        raise ValueError("The LOMO rows differ from the declared file.")
    months = decl["recipe_months"]
    cs = scope_table(read_json(ROOT / "configs/callsign_fix_v1.json")).set_index(ID)
    off_cs, off = offsets(), offsets_all()
    seg = segments(months)
    wx = weather_days()
    snow = set(zip(wx.loc[wx["snow"], "airport"], wx.loc[wx["snow"], "day"]))
    reference = read_json(ROOT / "docs/disagreement/mix_gate_v1_results.json")["months"]
    report = {"created_at_utc": utc_now(), "declaration": decl, "declaration_sha256": object_hash(decl), "rows_sha256": sha256_file(ROOT / ROWS),
              "audit_used": False, "months": {}, "candidates": {}}
    per_month = {}
    for month in months:
        f = rows[rows["month"] == month].reset_index(drop=True)
        mix_check = float(np.nanmax(np.abs(mixed(f["committee"].to_numpy(dtype=float), f["nets"].to_numpy(dtype=float), 0.75) - f["mix"].to_numpy(dtype=float))))
        if not mix_check <= 1e-9:
            raise ValueError(f"The mix column does not equal mixed(committee, nets, 0.75) on {month}.")
        y, days = f["label"].to_numpy(dtype=float), f["day"].to_numpy()
        control = stack(f, f["mix"].to_numpy(dtype=float), cs, off_cs, seg, off)
        report["months"][month] = {"rows": int(len(f)), "provenance": sorted(set(f["provenance"])), "control_rmse": rmse(y, control),
                                   "snow_airport_days": int(len({(a, d) for a, d in zip(f["airport"], f["day"])} & snow))}
        if month in ("2025-11", "2025-12"):
            gap = abs(report["months"][month]["control_rmse"] - reference[month]["control_rmse"])
            report["months"][month]["control_vs_mix_gate_round"] = gap
            if gap > 1e-6:
                raise ValueError(f"The v35-stack control does not reproduce the mix-gate round's control on {month} ({gap}).")
        per_month[month] = {}
        for name, spec in decl["candidates"].items():
            g, mix = candidate_inputs(f, spec)
            values = stack(g, mix, cs, off_cs, seg, off)
            lifted = np.zeros(len(f), dtype=bool)
            if "snow_k" in spec:
                values, lifted = snow_lift(values, f, snow, spec["snow_k"])
            if not np.isfinite(values).all():
                raise ValueError(f"{name} has non-finite values on {month}.")
            changed = np.abs(values - control) > 1e-9
            per_month[month][name] = {"rmse": rmse(y, values), "ratio_to_control": rmse(y, values) / rmse(y, control), "rows_differing_from_control": int(changed.sum()),
                                      "rows_snow_lifted": int(lifted.sum()), "mean_change_s": float((values - control)[changed].mean()) if changed.any() else 0.0,
                                      "paired_days_vs_control": paired_days(days, y, values, control, SEED)}
        print(month, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in report["months"][month].items()},
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
    report["model_code"] = {p: sha256_file(ROOT / p) for p in ["tools/run_snow_higher.py", "tools/snow_day_scan.py", "tools/run_mix_gate.py", "tools/run_queue_trust.py",
                                                                "tools/run_callsign_fix.py", "tools/run_disagreement_gate.py", "tools/run_fair_weights.py"]}
    report["environment"] = environment(ROOT)
    for path in [OUT / "selection.json", REPORT]:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, report)
    print({"selected": report["selected_candidate"], "eligible": eligible})
    for n, e in report["candidates"].items():
        iv = " ".join(f"[{e['months'][mo]['paired_days_vs_control']['rmse_difference_95pct_interval_seconds'][0]:+.3f},{e['months'][mo]['paired_days_vs_control']['rmse_difference_95pct_interval_seconds'][1]:+.3f}]" for mo in months)
        print(f"{n:8s} mean {100 * (e['mean_ratio'] - 1):+.4f}% " + " ".join(f"{mo}:{100 * (e['months'][mo]['ratio_to_control'] - 1):+.4f}({e['months'][mo]['rows_differing_from_control']})" for mo in months)
              + f" | {iv} <0 {len(e['months_with_interval_below_zero'])}/{len(e['months_touched'])} eligible={e['eligible']}")


if __name__ == "__main__":
    if "--score" not in sys.argv:
        raise SystemExit(__doc__)
    main()
