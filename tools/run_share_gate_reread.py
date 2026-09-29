"""Re-read of a thinned-regime candidate on the full v45 stack, development months (configs/thinned_regime_round_v1.json "ship").

The declared round scored its candidates at the v34 level. Before an export, the candidate's committee mix goes through the whole
shipped stack on runs/lomo_dev/rows.parquet: v34 weights -> u300 gate -> callsign fix 50% -> queue trust 0.75 (tools/run_mix_gate.stack)
-> two-sided tail stretch with the shipped four-month k (docs/disagreement/tail_stretch_v2_results.json two_sided_900) -> LIRF
disagreement gate (configs/lirf_gate_v1.json u600_m600). Fill pins do not occur in the development months' scope. Descriptive: the
declared rule selected nothing, and this re-read changes no selection.

  --candidate n85_nd1200   -> docs/thinned_regime/reread_<candidate>.json
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
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID  # noqa: E402
from prc2026_final import lirf_gate, share_gate  # noqa: E402
from run_adsb_v11 import paired_days, rmse  # noqa: E402
from run_callsign_fix import offsets, scope_table  # noqa: E402
from run_mix_gate import stack  # noqa: E402
from run_queue_trust import offsets_all, segments  # noqa: E402

DECLARATION = "configs/thinned_regime_round_v1.json"
ROWS = "runs/lomo_dev/rows.parquet"
STRETCH = "docs/disagreement/tail_stretch_v2_results.json"
MONTHS = ["2025-01", "2025-07", "2025-11", "2025-12"]
SEED = 20260929
T_UPPER, T_LOWER = 1500.0, 900.0


def full_stack(f: pd.DataFrame, mix: np.ndarray, cs, off_cs, seg, off, k: dict, gate_spec: dict) -> np.ndarray:
    values = stack(f, mix, cs, off_cs, seg, off)
    q = seg.reindex(f[ID])
    anchored = cs.reindex(f[ID])["motion_start"].notna().to_numpy() | (q["in_scope"].fillna(False).to_numpy(dtype=bool) & q["source"].eq("regular").to_numpy())
    non_lirf, matched, floor = f["airport"].to_numpy() != "LIRF", f["matched"].to_numpy(), f["floor"].to_numpy(dtype=float)
    for scope, m in (("untracked", non_lirf & ~matched & ~anchored), ("tracked", non_lirf & matched & ~anchored)):
        k1, k2 = k[scope]
        u = values[m] + k1 * np.maximum(values[m] - T_UPPER, 0.0)
        u = u + k2 * np.maximum(T_LOWER - u, 0.0)
        values[m] = np.maximum(u, floor[m])
    for flag, scope in (("lirf_u", "unmatched"), ("lirf_m", "matched")):
        m = f[flag].to_numpy() & ~non_lirf
        if not m.any():
            continue
        components = [f.loc[m, c].to_numpy(dtype=float) for c in ["v9", "lirf_net", "lirf_tree"]]
        blend, _ = lirf_gate.gated(*components, scope, None)
        gated, fire = lirf_gate.gated(*components, scope, gate_spec.get(scope))
        idx = np.flatnonzero(m)[fire]
        values[idx] = np.maximum(values[idx] + (gated - blend)[fire], lirf_gate.FLOOR)
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()
    decl = read_json(ROOT / DECLARATION)
    share, gate = share_gate.candidate_spec(decl, args.candidate)
    rows = pd.read_parquet(ROOT / ROWS)
    cs = scope_table(read_json(ROOT / "configs/callsign_fix_v1.json")).set_index(ID)
    off_cs, off = offsets(), offsets_all()
    seg = segments(MONTHS)
    k = read_json(ROOT / STRETCH)["candidates"]["two_sided_900"]["k_all_months"]
    gate_spec = read_json(ROOT / "configs/lirf_gate_v1.json")["candidates"]["u600_m600"]
    report = {"created_at_utc": utc_now(), "declaration": DECLARATION, "declaration_object_hash": object_hash(decl), "candidate": args.candidate,
              "network_share": share, "gate": gate, "rows_sha256": sha256_file(ROOT / ROWS), "stretch_k": k, "audit_used": False,
              "note": "descriptive re-read on the full v45 stack; the declared rule selected nothing (docs/thinned_regime/results_v1.json); November's fold side is on nine months",
              "months": {}}
    for month in MONTHS:
        f = rows[rows["month"] == month].reset_index(drop=True)
        y, days = f["label"].to_numpy(dtype=float), f["day"].to_numpy()
        committee, nets = f["committee"].to_numpy(dtype=float), f["nets"].to_numpy(dtype=float)
        control_mix, _ = share_gate.mix(committee, nets, share_gate.CONTROL_SHARE)
        if not float(np.nanmax(np.abs(control_mix - f["mix"].to_numpy(dtype=float)))) <= 1e-9:
            raise ValueError(f"The control mix does not equal the rows' mix on {month}.")
        candidate_mix, fire = share_gate.mix(committee, nets, share, gate)
        fire &= f["airport"].to_numpy() != "LIRF"
        control = full_stack(f, control_mix, cs, off_cs, seg, off, k, gate_spec)
        values = full_stack(f, candidate_mix, cs, off_cs, seg, off, k, gate_spec)
        if not (np.isfinite(control).all() and np.isfinite(values).all()):
            raise ValueError(f"Non-finite stack values on {month}.")
        changed = np.abs(values - control) > 1e-9
        if (changed & (f["airport"].to_numpy() == "LIRF")).any():
            raise ValueError("A LIRF row changed.")
        report["months"][month] = {"rows": int(len(f)), "provenance": sorted(set(f["provenance"])), "control_rmse": rmse(y, control), "candidate_rmse": rmse(y, values),
                                   "change_pct": 100 * (rmse(y, values) / rmse(y, control) - 1), "rows_changed": int(changed.sum()), "gate_rows": int(fire.sum()),
                                   "mean_change_s": float((values - control)[changed].mean()), "sum_squared_change": float(np.sum((values - control) ** 2)),
                                   "alignment_sum_d_r": float(np.sum((values - control) * (control - y))),
                                   "paired_days_vs_control": paired_days(days, y, values, control, SEED)}
        e = report["months"][month]
        print(month, {"control": round(e["control_rmse"], 4), "change_pct": round(e["change_pct"], 4), "interval": [round(v, 3) for v in e["paired_days_vs_control"]["rmse_difference_95pct_interval_seconds"]],
                      "rows": e["rows_changed"], "gate": e["gate_rows"], "mean_change_s": round(e["mean_change_s"], 2)}, flush=True)
    equal = [m for m in MONTHS if m != "2025-11"]
    report["mean_change_pct_equal_footing"] = float(np.mean([report["months"][m]["change_pct"] for m in equal]))
    report["code"] = {p: sha256_file(ROOT / p) for p in ["tools/run_share_gate_reread.py", "src/prc2026_final/share_gate.py", "src/prc2026_final/lirf_gate.py", "tools/run_mix_gate.py"]}
    write_json(ROOT / f"docs/thinned_regime/reread_{args.candidate}.json", report)
    print({"mean_change_pct_equal_footing": round(report["mean_change_pct_equal_footing"], 4)})


if __name__ == "__main__":
    main()
