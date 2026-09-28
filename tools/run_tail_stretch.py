"""Declared tail-stretch round (configs/tail_stretch_v1.json) on runs/lomo_dev/rows.parquet.

value + k x max(value - t, 0) on untracked / tracked non-LIRF rows outside the observation-anchored layers (callsign fix, queue trust),
k per scope fitted by least squares on the other three recipe months (cross-month), on two controls: the v35 stack and the v36 stack
(mu600 + peer-congestion lifts). The four-month k (the export's) is reported too.

  --score --select   -> runs/tail_stretch/selection.json and docs/disagreement/tail_stretch_v1_results.json
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
from run_higher_bundle import layered  # noqa: E402
from run_mix_gate import stack  # noqa: E402
from run_queue_trust import offsets_all, segments  # noqa: E402
from peer_congestion import training_peers  # noqa: E402

DECLARATION = "configs/tail_stretch_v1.json"
ROWS = "runs/lomo_dev/rows.parquet"
OUT = ROOT / "runs/tail_stretch"
REPORT = ROOT / "docs/disagreement/tail_stretch_v1_results.json"
SEED = 20260928
SCOPES = ["untracked", "tracked"]


def fit_k(parts: list[dict], variant: str, scope: str, t: float) -> float:
    x = np.concatenate([np.maximum(p[variant][p[scope]] - t, 0.0) for p in parts])
    r = np.concatenate([(p["y"] - p[variant])[p[scope]] for p in parts])
    return float(np.sum(x * r) / np.sum(x * x))


def stretched(p: dict, variant: str, t: float, ks: dict) -> np.ndarray:
    values = p[variant].copy()
    for scope in SCOPES:
        m = p[scope]
        values[m] = np.maximum(values[m] + ks[scope] * np.maximum(values[m] - t, 0.0), p["floor"][m])
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
    peers = training_peers().set_index(ID)
    v36_spec = read_json(ROOT / "configs/higher_bundle_v2.json")["candidates"]["mu_pc_pcc"]
    ref35 = read_json(ROOT / "docs/disagreement/mix_gate_v1_results.json")["months"]
    ref36 = read_json(ROOT / "docs/disagreement/higher_bundle_v2_results.json")["candidates"]["mu_pc_pcc"]["months"]
    parts = {}
    report = {"created_at_utc": utc_now(), "declaration": decl, "declaration_sha256": object_hash(decl), "rows_sha256": sha256_file(ROOT / ROWS),
              "audit_used": False, "months": {}, "candidates": {}}
    for month in months:
        f = rows[rows["month"] == month].reset_index(drop=True)
        y = f["label"].to_numpy(dtype=float)
        v35 = stack(f, f["mix"].to_numpy(dtype=float), cs, off_cs, seg, off)
        v36, _ = layered(f, v36_spec, cs, off_cs, seg, off, peers)
        checks = {"v36_vs_higher_bundle_v2": abs(rmse(y, v36) - ref36[month]["rmse"])}
        if month in ("2025-11", "2025-12"):
            checks["v35_vs_mix_gate_round"] = abs(rmse(y, v35) - ref35[month]["control_rmse"])
        if max(checks.values()) > 1e-6:
            raise ValueError(f"A control does not reproduce its recorded round on {month}: {checks}")
        q = seg.reindex(f[ID])
        anchored = cs.reindex(f[ID])["motion_start"].notna().to_numpy() | (q["in_scope"].fillna(False).to_numpy(dtype=bool) & q["source"].eq("regular").to_numpy())
        non_lirf = f["airport"].to_numpy() != "LIRF"
        matched = f["matched"].to_numpy()
        parts[month] = {"y": y, "days": f["day"].to_numpy(), "floor": f["floor"].to_numpy(dtype=float), "v35": v35, "v36": v36,
                        "untracked": non_lirf & ~matched & ~anchored, "tracked": non_lirf & matched & ~anchored}
        report["months"][month] = {"rows": int(len(f)), "provenance": sorted(set(f["provenance"])), "control_rmse": {"v35": rmse(y, v35), "v36": rmse(y, v36)},
                                   "checks": checks, "anchored_rows": int(anchored.sum())}
    for name, spec in decl["candidates"].items():
        variant, t = spec["control"], float(spec["t"])
        entry = {**spec, "months": {}, "k_all_months": {s: fit_k(list(parts.values()), variant, s, t) for s in SCOPES}}
        for month in months:
            p = parts[month]
            ks = {s: fit_k([parts[m] for m in months if m != month], variant, s, t) for s in SCOPES}
            values = stretched(p, variant, t, ks)
            changed = np.abs(values - p[variant]) > 1e-9
            entry["months"][month] = {"k": ks, "rmse": rmse(p["y"], values), "ratio_to_control": rmse(p["y"], values) / rmse(p["y"], p[variant]),
                                      "rows_differing_from_control": int(changed.sum()), "mean_change_s": float((values - p[variant])[changed].mean()) if changed.any() else 0.0,
                                      "paired_days_vs_control": paired_days(p["days"], p["y"], values, p[variant], SEED)}
        ratios = [entry["months"][mo]["ratio_to_control"] for mo in months]
        entry["mean_ratio"] = float(np.mean(ratios))
        touched = [mo for mo in months if entry["months"][mo]["rows_differing_from_control"] > 0]
        below = [mo for mo in touched if entry["months"][mo]["paired_days_vs_control"]["rmse_difference_95pct_interval_seconds"][1] < 0]
        entry["months_touched"], entry["months_with_interval_below_zero"] = touched, below
        entry["eligible"] = bool(touched and all(r <= 1.0 for r in ratios) and entry["mean_ratio"] <= 1 - decl["minimum_mean_improvement_fraction"] and len(below) == len(touched))
        report["candidates"][name] = entry
    report["selected_by_variant"] = {}
    for variant in ["v35", "v36"]:
        eligible = [n for n, e in report["candidates"].items() if e["control"] == variant and e["eligible"]]
        report["selected_by_variant"][variant] = min(eligible, key=lambda n: (report["candidates"][n]["mean_ratio"], n)) if eligible else None
    report["model_code"] = {p: sha256_file(ROOT / p) for p in ["tools/run_tail_stretch.py", "tools/run_higher_bundle.py", "tools/peer_congestion.py", "tools/run_snow_higher.py",
                                                                "tools/run_mix_gate.py", "tools/run_queue_trust.py", "tools/run_callsign_fix.py", "tools/run_disagreement_gate.py"]}
    report["environment"] = environment(ROOT)
    for path in [OUT / "selection.json", REPORT]:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, report)
    print({m: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in e["control_rmse"].items()} for m, e in report["months"].items()})
    print("selected:", report["selected_by_variant"])
    for n, e in report["candidates"].items():
        iv = " ".join(f"[{e['months'][mo]['paired_days_vs_control']['rmse_difference_95pct_interval_seconds'][0]:+.3f},{e['months'][mo]['paired_days_vs_control']['rmse_difference_95pct_interval_seconds'][1]:+.3f}]" for mo in months)
        print(f"{n:10s} mean {100 * (e['mean_ratio'] - 1):+.4f}% " + " ".join(f"{mo}:{100 * (e['months'][mo]['ratio_to_control'] - 1):+.4f}({e['months'][mo]['rows_differing_from_control']})" for mo in months)
              + f" | {iv} <0 {len(e['months_with_interval_below_zero'])}/{len(e['months_touched'])} eligible={e['eligible']} k_all={ {s: round(v, 3) for s, v in e['k_all_months'].items()} }")


if __name__ == "__main__":
    if "--score" not in sys.argv:
        raise SystemExit(__doc__)
    main()
