"""Declared weight round for the network members v52 ships (configs/member_weights_v1.json); control = v52's model part. No model is fitted.

  table_1  (1 - c) nets + c combo_all       where both are finite (non-LIRF rows)
  table_2  (1 - m) table_1 + m mask_unmatched where both are finite (non-LIRF rows without an ADS-B match)
  mix      v46's on table_2; stack as in tools/run_network_member.py (tools/run_share_gate_reread.full_stack)

Before any candidate is read: the v46 stack must reproduce the v46 re-read, and the control (c, m) = (0.40, 0.40) must reproduce the masked
round's recorded w40 month RMSEs (docs/adsb_ctx3_mask/results_v1.json). Rule and readings as in tools/run_network_member.py.

  --declaration PATH --check-control     the two control reproductions only
  --declaration PATH --score --select    -> the declaration's "selection" and "report" paths
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
from prc2026.artifacts import environment, object_hash, read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID  # noqa: E402
from prc2026_final import network_member, share_gate  # noqa: E402
from run_adsb_v11 import paired_days, rmse  # noqa: E402
from run_callsign_fix import offsets, scope_table  # noqa: E402
from run_network_member import MONTHS, REREAD, ROWS, SEED, STRETCH, member_values, select, slices  # noqa: E402
from run_queue_trust import offsets_all, segments  # noqa: E402
from run_share_gate_reread import full_stack  # noqa: E402

CODE = ["tools/run_member_weights.py", "tools/run_network_member.py", "src/prc2026_final/network_member.py", "src/prc2026_final/share_gate.py",
        "tools/run_share_gate_reread.py"]


def weights(spec: dict) -> tuple[float, float]:
    if set(spec) - {"combo_weight", "mask_weight", "note", "results", "results_candidate"}:
        raise ValueError(f"Unwired keys in {sorted(spec)}.")
    c, m = float(spec["combo_weight"]), float(spec["mask_weight"])
    if not (0.0 <= c <= 1.0 and 0.0 <= m <= 1.0):
        raise ValueError("Member weights lie in [0, 1].")
    return c, m


def table(nets: np.ndarray, combo: np.ndarray, mask: np.ndarray, c: float, m: float) -> tuple[np.ndarray, np.ndarray]:
    """(table_2, rows where either member enters)."""
    t1, in1 = network_member.table_with_member(nets, combo, c)
    t2, in2 = network_member.table_with_member(t1, mask, m)
    return t2, in1 | in2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--declaration", required=True)
    parser.add_argument("--check-control", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--select", action="store_true")
    args = parser.parse_args()
    if not (args.check_control or (args.score and args.select)):
        raise SystemExit("Use --check-control or --score --select.")
    decl = read_json(ROOT / args.declaration)
    if not decl.get("declared_at_utc") or decl.get("audit_used"):
        raise ValueError("The round is not declared, or does not respect its own contract.")
    names = [mem["name"] for mem in decl["members"]]
    if names != ["combo_all", "mask_unmatched"]:
        raise ValueError("The declared members changed.")
    equal = list(decl["equal_footing_months"])
    minimum = float(decl["minimum_mean_improvement_fraction"])
    if args.score:
        selection_path, report_path = ROOT / decl["selection"], ROOT / decl["report"]
        if selection_path.exists() or report_path.exists():
            raise ValueError("Preserve the original selection.")
    recorded = read_json(ROOT / decl["control"]["results"])
    if recorded.get("declaration_sha256") != object_hash(read_json(ROOT / decl["members"][1]["declaration"])):
        raise ValueError("The recorded control results are not for the masked round's declaration.")
    c0, m0 = weights(decl["control"])

    rows = pd.read_parquet(ROOT / ROWS)
    cs = scope_table(read_json(ROOT / "configs/callsign_fix_v1.json")).set_index(ID)
    off_cs, off = offsets(), offsets_all()
    seg = segments(MONTHS)
    k = read_json(ROOT / STRETCH)["candidates"]["two_sided_900"]["k_all_months"]
    gate_spec = read_json(ROOT / "configs/lirf_gate_v1.json")["candidates"]["u600_m600"]
    reread = read_json(ROOT / REREAD)["months"]

    def stack(f, tab):
        mix, fire = share_gate.mix(f["committee"].to_numpy(dtype=float), tab, network_member.V46_SHARE, network_member.V46_GATE)
        return full_stack(f, mix, cs, off_cs, seg, off, k, gate_spec), fire

    frames, sources = {}, {}
    for month in MONTHS:
        f = rows[rows["month"] == month].reset_index(drop=True)
        nets = f["nets"].to_numpy(dtype=float)
        v46, _ = stack(f, nets)
        y = f["label"].to_numpy(dtype=float)
        if not abs(rmse(y, v46) - reread[month]["candidate_rmse"]) <= 1e-9:
            raise ValueError(f"The v46 stack does not reproduce the re-read on {month}.")
        lirf, ids = f["airport"].to_numpy() == "LIRF", f[ID].to_numpy()
        members = []
        for mem in decl["members"]:
            values, found = member_values(mem, month, ids)
            sources.update(found)
            members.append(np.where(lirf, np.nan, values))
        tab, _ = table(nets, members[0], members[1], c0, m0)
        control, _ = stack(f, tab)
        want = recorded["candidates"][decl["control"]["results_candidate"]]["months"][month]["rmse"]
        if not abs(rmse(y, control) - want) <= 1e-9:
            raise ValueError(f"The control does not reproduce the recorded {decl['control']['results_candidate']} on {month}: {rmse(y, control)} against {want}.")
        frames[month] = (f, nets, members, control)
    print("v46 re-read and the control (v52's model part) reproduce on", MONTHS, flush=True)
    if args.check_control:
        return

    report = {"created_at_utc": utc_now(), "declaration": decl, "declaration_sha256": object_hash(decl), "audit_used": False, "rows_sha256": sha256_file(ROOT / ROWS),
              "control": {"combo_weight": c0, "mask_weight": m0, "month_rmse": {}}, "equal_footing_months": equal, "minimum_mean_improvement_fraction": minimum,
              "member_sources": sources, "candidates": {}}
    entries = {name: {**spec, "months": {}} for name, spec in decl["candidates"].items()}
    for month, (f, nets, members, control) in frames.items():
        y, days, lirf = f["label"].to_numpy(dtype=float), f["day"].to_numpy(), f["airport"].to_numpy() == "LIRF"
        report["control"]["month_rmse"][month] = rmse(y, control)
        for name, spec in decl["candidates"].items():
            c, m = weights(spec)
            tab, inside = table(nets, members[0], members[1], c, m)
            values, fire = stack(f, tab)
            if not np.isfinite(values).all():
                raise ValueError(f"{name} has non-finite values on {month}.")
            changed = np.abs(values - control) > 1e-9
            if (changed & lirf).any():
                raise ValueError("A LIRF row changed.")
            if (changed & ~inside).any():
                raise ValueError("A row without a member value changed.")
            d = values - control
            entries[name]["months"][month] = {"rmse": rmse(y, values), "ratio_to_control": rmse(y, values) / rmse(y, control),
                                              "change_pct": 100 * (rmse(y, values) / rmse(y, control) - 1), "rows_changed": int(changed.sum()),
                                              "gate_rows": int((fire & ~lirf).sum()), "sum_squared_change": float(np.sum(d ** 2)),
                                              "alignment": float(-np.sum(d * (control - y)) / np.sum(d ** 2)) if changed.any() else None,
                                              "by_scope_change_pct": {s: 100 * (rmse(y[q], values[q]) / rmse(y[q], control[q]) - 1) for s, q in slices(f).items()},
                                              "paired_days_vs_control": paired_days(days, y, values, control, SEED)}
    eligible, selected = select(entries, equal, minimum)
    report["candidates"], report["eligible"], report["selected_candidate"] = entries, eligible, selected
    report["model_code"] = {p: sha256_file(ROOT / p) for p in CODE}
    report["environment"] = environment(ROOT)
    for path in [selection_path, report_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, report)
    print({"selected": selected, "eligible": eligible})
    print("control", {mm: round(v, 4) for mm, v in report["control"]["month_rmse"].items()})
    for name, e in sorted(entries.items(), key=lambda kv: kv[1]["mean_ratio_equal_footing"]):
        line = " ".join(f"{mm}:{e['months'][mm]['change_pct']:+.4f}% [{e['months'][mm]['paired_days_vs_control']['rmse_difference_95pct_interval_seconds'][0]:+.3f},"
                        f"{e['months'][mm]['paired_days_vs_control']['rmse_difference_95pct_interval_seconds'][1]:+.3f}]" for mm in MONTHS)
        print(f"{name:8s} mean {100 * (e['mean_ratio_equal_footing'] - 1):+.4f}% | {line} | eligible={e['eligible']}")


if __name__ == "__main__":
    main()
