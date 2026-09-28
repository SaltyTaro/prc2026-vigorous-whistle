"""Declared higher-side bundle round (configs/higher_bundle_v1.json) on runs/lomo_dev/rows.parquet. No model is fitted.

Control: the v35 stack (tools/run_mix_gate.stack). Candidates: mu600 (network share 1.0 where nets - committee > 600, stack rebuilt),
matched lean-up (outer weight 1.0 on matched non-LIRF rows outside the queue scope with 150 < mix - base <= 300), and peer-congestion
lifts (tools/peer_congestion.py) for rows without an NM clock and for untracked rows with one.

  --score --select   -> runs/higher_bundle/selection.json and docs/disagreement/higher_bundle_v1_results.json
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
from run_snow_higher import candidate_inputs  # noqa: E402
from peer_congestion import training_peers  # noqa: E402

DECLARATION = "configs/higher_bundle_v1.json"
ROWS = "runs/lomo_dev/rows.parquet"
OUT = ROOT / "runs/higher_bundle"
REPORT = ROOT / "docs/disagreement/higher_bundle_v1_results.json"
SEED = 20260928
MATCHED_OUTER = 0.6


def layered(f: pd.DataFrame, spec: dict, cs, off_cs, seg, off, peers: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """(values, row masks per layer) for one month's rows."""
    g, mix = candidate_inputs(f, spec)
    values = stack(g, mix, cs, off_cs, seg, off).copy()
    airport, matched = f["airport"].to_numpy(), f["matched"].to_numpy()
    floor = f["floor"].to_numpy(dtype=float)
    masks = {}
    if "matched_lean" in spec:
        lo, hi = spec["matched_lean"]
        d = mix - f["base"].to_numpy(dtype=float)
        q = seg.reindex(f[ID])
        queue = q["in_scope"].fillna(False).to_numpy(dtype=bool) & q["source"].eq("regular").to_numpy()
        rows = matched & (airport != "LIRF") & ~queue & np.isfinite(d) & (d > lo) & (d <= hi)
        values[rows] = np.maximum(values[rows] + (1 - MATCHED_OUTER) * d[rows], floor[rows])
        masks["matched_lean"] = rows
    p = peers.reindex(f[ID])
    ok = (p["peer_n"].fillna(0).to_numpy() >= 5) & (airport != "LIRF")
    peer_c = p["peer_c"].to_numpy(dtype=float)
    no_clock = p["no_clock"].fillna(False).to_numpy(dtype=bool)
    if "noclock_lift" in spec:
        s = spec["noclock_lift"]
        rows = ok & no_clock & (peer_c > s["peer_c_above"])
        values[rows] += s["seconds"]
        masks["noclock_lift"] = rows
    if "clock_untracked_lift" in spec:
        s = spec["clock_untracked_lift"]
        rows = ok & ~no_clock & ~matched & (peer_c > s["peer_c_above"])
        values[rows] += s["seconds"]
        masks["clock_untracked_lift"] = rows
    return values, masks


def main() -> None:
    decl = read_json(ROOT / DECLARATION)
    if (OUT / "selection.json").exists():
        raise ValueError("Preserve the original selection.")
    if sha256_file(ROOT / ROWS) != decl["rows"].split("SHA-256 ")[1].split(")")[0]:
        raise ValueError("The LOMO rows differ from the declared file.")
    rows = pd.read_parquet(ROOT / ROWS)
    months = decl["recipe_months"]
    cs = scope_table(read_json(ROOT / "configs/callsign_fix_v1.json")).set_index(ID)
    off_cs, off = offsets(), offsets_all()
    seg = segments(months)
    peers = training_peers().set_index(ID)
    reference = read_json(ROOT / "docs/disagreement/mix_gate_v1_results.json")["months"]
    report = {"created_at_utc": utc_now(), "declaration": decl, "declaration_sha256": object_hash(decl), "rows_sha256": sha256_file(ROOT / ROWS),
              "audit_used": False, "months": {}, "candidates": {}}
    per_month = {}
    for month in months:
        f = rows[rows["month"] == month].reset_index(drop=True)
        if float(np.nanmax(np.abs(mixed(f["committee"].to_numpy(dtype=float), f["nets"].to_numpy(dtype=float), 0.75) - f["mix"].to_numpy(dtype=float)))) > 1e-9:
            raise ValueError(f"The mix column does not equal mixed(committee, nets, 0.75) on {month}.")
        if f[ID].isin(peers.index).mean() < 0.999:
            raise ValueError(f"Peer congestion does not cover {month}.")
        y, days = f["label"].to_numpy(dtype=float), f["day"].to_numpy()
        control = stack(f, f["mix"].to_numpy(dtype=float), cs, off_cs, seg, off)
        report["months"][month] = {"rows": int(len(f)), "provenance": sorted(set(f["provenance"])), "control_rmse": rmse(y, control)}
        if month in ("2025-11", "2025-12"):
            gap = abs(report["months"][month]["control_rmse"] - reference[month]["control_rmse"])
            report["months"][month]["control_vs_mix_gate_round"] = gap
            if gap > 1e-6:
                raise ValueError(f"The v35-stack control does not reproduce the mix-gate round's control on {month} ({gap}).")
        per_month[month] = {}
        for name, spec in decl["candidates"].items():
            values, masks = layered(f, spec, cs, off_cs, seg, off, peers)
            if not np.isfinite(values).all():
                raise ValueError(f"{name} has non-finite values on {month}.")
            changed = np.abs(values - control) > 1e-9
            per_month[month][name] = {"rmse": rmse(y, values), "ratio_to_control": rmse(y, values) / rmse(y, control), "rows_differing_from_control": int(changed.sum()),
                                      "layer_rows": {k: int(v.sum()) for k, v in masks.items()},
                                      "layer_mean_residual_before_s": {k: float((y - control)[v].mean()) if v.any() else None for k, v in masks.items()},
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
    report["model_code"] = {p: sha256_file(ROOT / p) for p in ["tools/run_higher_bundle.py", "tools/peer_congestion.py", "tools/run_snow_higher.py", "tools/run_mix_gate.py",
                                                                "tools/run_queue_trust.py", "tools/run_callsign_fix.py", "tools/run_disagreement_gate.py", "tools/run_fair_weights.py"]}
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
