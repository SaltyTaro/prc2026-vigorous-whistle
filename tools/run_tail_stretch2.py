"""Declared two-sided tail stretch (configs/tail_stretch_v2.json) on runs/lomo_dev/rows.parquet, against the v35 stack.

Upper side as tools/run_tail_stretch.py (k1 on max(control - 1500, 0)); lower side k2 on max(t_lower - u, 0) with u the upper-stretched
value; both per scope (untracked / tracked non-LIRF outside the callsign / queue layers), fitted on the other three months.

  --score --select   -> runs/tail_stretch_v2/selection.json and docs/disagreement/tail_stretch_v2_results.json
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
from run_adsb_v11 import paired_days, rmse  # noqa: E402
from run_callsign_fix import offsets, scope_table  # noqa: E402
from run_mix_gate import stack  # noqa: E402
from run_queue_trust import offsets_all, segments  # noqa: E402

DECLARATION = "configs/tail_stretch_v2.json"
ROWS = "runs/lomo_dev/rows.parquet"
OUT = ROOT / "runs/tail_stretch_v2"
REPORT = ROOT / "docs/disagreement/tail_stretch_v2_results.json"
SEED = 20260928
SCOPES = ["untracked", "tracked"]


def ls(x: np.ndarray, r: np.ndarray) -> float:
    return float(np.sum(x * r) / np.sum(x * x))


def fit(parts: list[dict], spec: dict) -> dict:
    """{scope: (k1, k2)} from the given months' in-scope rows."""
    out = {}
    for s in SCOPES:
        c = np.concatenate([p["control"][p[s]] for p in parts])
        y = np.concatenate([p["y"][p[s]] for p in parts])
        k1 = ls(np.maximum(c - spec["t_upper"], 0.0), y - c)
        u = c + k1 * np.maximum(c - spec["t_upper"], 0.0)
        k2 = ls(np.maximum(spec["t_lower"] - u, 0.0), y - u) if "t_lower" in spec else 0.0
        out[s] = (k1, k2)
    return out


def apply(p: dict, spec: dict, ks: dict) -> np.ndarray:
    values = p["control"].copy()
    for s in SCOPES:
        m = p[s]
        k1, k2 = ks[s]
        u = values[m] + k1 * np.maximum(values[m] - spec["t_upper"], 0.0)
        if "t_lower" in spec:
            u = u + k2 * np.maximum(spec["t_lower"] - u, 0.0)
        values[m] = np.maximum(u, p["floor"][m])
    return values


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
    v1 = read_json(ROOT / "docs/disagreement/tail_stretch_v1_results.json")["candidates"]["v35_t1500"]["months"]
    parts = {}
    for month in months:
        f = rows[rows["month"] == month].reset_index(drop=True)
        control = stack(f, f["mix"].to_numpy(dtype=float), cs, off_cs, seg, off)
        q = seg.reindex(f[ID])
        anchored = cs.reindex(f[ID])["motion_start"].notna().to_numpy() | (q["in_scope"].fillna(False).to_numpy(dtype=bool) & q["source"].eq("regular").to_numpy())
        non_lirf, matched = f["airport"].to_numpy() != "LIRF", f["matched"].to_numpy()
        parts[month] = {"y": f["label"].to_numpy(dtype=float), "days": f["day"].to_numpy(), "floor": f["floor"].to_numpy(dtype=float), "control": control,
                        "untracked": non_lirf & ~matched & ~anchored, "tracked": non_lirf & matched & ~anchored}
    report = {"created_at_utc": utc_now(), "declaration": decl, "declaration_sha256": object_hash(decl), "rows_sha256": sha256_file(ROOT / ROWS),
              "audit_used": False, "months": {m: {"control_rmse": rmse(p["y"], p["control"])} for m, p in parts.items()}, "candidates": {}}
    for name, spec in decl["candidates"].items():
        entry = {**spec, "months": {}, "k_all_months": {s: list(v) for s, v in fit(list(parts.values()), spec).items()}}
        for month in months:
            p = parts[month]
            ks = fit([parts[m] for m in months if m != month], spec)
            values = apply(p, spec, ks)
            changed = np.abs(values - p["control"]) > 1e-9
            entry["months"][month] = {"k": {s: list(v) for s, v in ks.items()}, "rmse": rmse(p["y"], values), "ratio_to_control": rmse(p["y"], values) / rmse(p["y"], p["control"]),
                                      "rows_differing_from_control": int(changed.sum()), "paired_days_vs_control": paired_days(p["days"], p["y"], values, p["control"], SEED)}
        if name == "upper_only":
            gap = max(abs(entry["months"][m]["ratio_to_control"] - v1[m]["ratio_to_control"]) for m in months)
            if gap > 1e-9:
                raise ValueError(f"upper_only does not reproduce tail_stretch_v1's v35_t1500 ({gap}).")
            entry["reproduces_tail_stretch_v1"] = gap
        ratios = [entry["months"][mo]["ratio_to_control"] for mo in months]
        entry["mean_ratio"] = float(np.mean(ratios))
        touched = [mo for mo in months if entry["months"][mo]["rows_differing_from_control"] > 0]
        below = [mo for mo in touched if entry["months"][mo]["paired_days_vs_control"]["rmse_difference_95pct_interval_seconds"][1] < 0]
        entry["months_touched"], entry["months_with_interval_below_zero"] = touched, below
        entry["eligible"] = bool(touched and all(r <= 1.0 for r in ratios) and entry["mean_ratio"] <= 1 - decl["minimum_mean_improvement_fraction"] and len(below) == len(touched))
        report["candidates"][name] = entry
    eligible = [n for n, e in report["candidates"].items() if e["eligible"] and n != "upper_only"]
    report["selected_candidate"] = min(eligible, key=lambda n: (report["candidates"][n]["mean_ratio"], n)) if eligible else None
    report["model_code"] = {p: sha256_file(ROOT / p) for p in ["tools/run_tail_stretch2.py", "tools/run_mix_gate.py", "tools/run_queue_trust.py", "tools/run_callsign_fix.py"]}
    report["environment"] = environment(ROOT)
    for path in [OUT / "selection.json", REPORT]:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, report)
    print("selected:", report["selected_candidate"])
    for n, e in report["candidates"].items():
        iv = " ".join(f"[{e['months'][mo]['paired_days_vs_control']['rmse_difference_95pct_interval_seconds'][0]:+.3f},{e['months'][mo]['paired_days_vs_control']['rmse_difference_95pct_interval_seconds'][1]:+.3f}]" for mo in months)
        print(f"{n:15s} mean {100 * (e['mean_ratio'] - 1):+.4f}% " + " ".join(f"{mo}:{100 * (e['months'][mo]['ratio_to_control'] - 1):+.4f}" for mo in months)
              + f" | {iv} <0 {len(e['months_with_interval_below_zero'])}/{len(e['months_touched'])} eligible={e['eligible']} k_all={ {s: [round(x, 4) for x in v] for s, v in e['k_all_months'].items()} }")


if __name__ == "__main__":
    if "--score" not in sys.argv:
        raise SystemExit(__doc__)
    main()
