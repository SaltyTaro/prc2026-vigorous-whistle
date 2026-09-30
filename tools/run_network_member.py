"""Declared round for one more member in the committee's network table; control = the v46 development stack. No model is fitted.

The declaration (--declaration PATH, committed before --score) names the member's per-seed prediction tables per development month
(MVT_ID_mvt, adsb_full; January and July on the eleven-month leave-one-month-out footing, November from the forward fold, December from
the January-November refit) and the candidates {name: {"member_weight": w}}.

  table'   (1 - w) nets + w member seed mean where both are finite, nets elsewhere (src/prc2026_final/network_member.py)
  mix      v46's: network share 0.85, share 0.5 where the committee sits more than 1,200 s above table' (src/prc2026_final/share_gate.py)
  stack    tools/run_share_gate_reread.full_stack on runs/lomo_dev/rows.parquet: v34 weights -> u300 gate -> callsign fix 50% -> queue
           trust 0.75 -> two-sided stretch -> LIRF disagreement gate
Before any candidate is read, the control's month RMSEs must equal the v46 re-read (docs/thinned_regime/reread_n85_nd1200.json).

Base member (optional "base_members": [{"declaration", "candidate", "results", "export"}], at most one): a member already shipped enters the
table first at its declared weight, the candidate member after it, and the control is the stack with the base member; its month RMSEs
must equal the base round's recorded RMSEs of that candidate (the base round's "results" file) before any candidate is read.

Rule, unless the declaration states its own months or margin: eligible when no equal-footing month (January, July, December) is worse
on all rows, the mean ratio is at least 0.02% below 1 and every equal-footing month's paired-day 95% interval lies below zero; the
lowest mean ratio among the eligible is selected, otherwise nothing. November is reported only (nine-month fold side).

  --declaration PATH --check-control     the control reproduction only
  --declaration PATH --smoke             code-path test: reads the tables, builds every candidate, prints pass / fail; writes nothing
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
from run_queue_trust import offsets_all, segments  # noqa: E402
from run_share_gate_reread import full_stack  # noqa: E402

ROWS = "runs/lomo_dev/rows.parquet"
REREAD = "docs/thinned_regime/reread_n85_nd1200.json"
STRETCH = "docs/disagreement/tail_stretch_v2_results.json"
MONTHS = ["2025-01", "2025-07", "2025-11", "2025-12"]
EQUAL = ["2025-01", "2025-07", "2025-12"]
MINIMUM = 0.0002
SEED = 20260929
CODE = ["tools/run_network_member.py", "src/prc2026_final/network_member.py", "src/prc2026_final/share_gate.py", "src/prc2026_final/lirf_gate.py",
        "tools/run_share_gate_reread.py", "tools/run_mix_gate.py"]


def member_values(spec: dict, month: str, ids: np.ndarray) -> tuple[np.ndarray, dict]:
    """Seed mean of the member's tables on the rows' ids, and the tables' hashes."""
    paths = spec["tables"].get(month)
    if not paths:
        raise ValueError(f"The declaration names no member table for {month}.")
    column = spec.get("column", "adsb_full")
    tables, sources = [], {}
    for rel in paths:
        t = pd.read_parquet(ROOT / rel, columns=[ID, column])
        if t[ID].duplicated().any():
            raise ValueError(f"{rel} repeats an id.")
        tables.append(t.set_index(ID)[column].reindex(ids).to_numpy(dtype=float))
        sources[rel] = sha256_file(ROOT / rel)
    return network_member.seed_mean(tables), sources


def base_members(decl: dict) -> list[dict]:
    """The declaration's base member (at most one): its member spec, weight, candidate and recorded results."""
    out = []
    for b in decl.get("base_members", []):
        base = read_json(ROOT / b["declaration"])
        out.append({"declaration": b["declaration"], "candidate": b["candidate"], "member": base["member"],
                    "weight": network_member.candidate_weight(base, b["candidate"]), "results": b.get("results"), "export": b.get("export")})
    if len(out) > 1:
        raise ValueError("At most one base member is wired.")
    return out


def with_bases(nets: np.ndarray, lirf: np.ndarray, ids: np.ndarray, month: str, bases: list[dict], sources: dict) -> np.ndarray:
    """The network table with the base members in it (LIRF rows keep the table)."""
    table = nets
    for b in bases:
        values, found = member_values(b["member"], month, ids)
        sources.update(found)
        table, _ = network_member.table_with_member(table, np.where(lirf, np.nan, values), b["weight"])
    return table


def select(entries: dict, equal: list[str], minimum: float) -> tuple[list[str], str | None]:
    """Adds the rule's readings to every candidate entry; (eligible, selected)."""
    for e in entries.values():
        ratios = [e["months"][m]["ratio_to_control"] for m in equal]
        e["mean_ratio_equal_footing"] = float(np.mean(ratios))
        e["months_worse"] = [m for m in equal if e["months"][m]["ratio_to_control"] > 1.0]
        e["intervals_below_zero"] = [m for m in equal if e["months"][m]["paired_days_vs_control"]["rmse_difference_95pct_interval_seconds"][1] < 0]
        e["eligible"] = bool(not e["months_worse"] and e["mean_ratio_equal_footing"] <= 1 - minimum and len(e["intervals_below_zero"]) == len(equal))
    eligible = [n for n, e in entries.items() if e["eligible"]]
    return eligible, (min(eligible, key=lambda n: (entries[n]["mean_ratio_equal_footing"], n)) if eligible else None)


def slices(f: pd.DataFrame) -> dict:
    non_lirf, matched = f["airport"].to_numpy() != "LIRF", f["matched"].to_numpy()
    return {"matched_non_lirf": non_lirf & matched, "unmatched_non_lirf": non_lirf & ~matched}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--declaration", required=True)
    parser.add_argument("--check-control", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--select", action="store_true")
    args = parser.parse_args()
    if not (args.check_control or args.smoke or (args.score and args.select)):
        raise SystemExit("Use --check-control, --smoke or --score --select.")
    decl = read_json(ROOT / args.declaration)
    if not args.check_control:
        if not decl.get("declared_at_utc"):
            raise ValueError("The round is not declared.")
        if decl.get("audit_used"):
            raise ValueError("Configuration does not respect its own contract.")
    equal = list(decl.get("equal_footing_months", EQUAL))
    minimum = float(decl.get("minimum_mean_improvement_fraction", MINIMUM))
    governing = args.score and not args.smoke
    if governing:
        selection_path, report_path = ROOT / decl["selection"], ROOT / decl["report"]
        if selection_path.exists() or report_path.exists():
            raise ValueError("Preserve the original selection.")
        if set(equal) - set(decl["member"]["tables"]):
            raise ValueError("The member has no table for every equal-footing month.")

    rows = pd.read_parquet(ROOT / ROWS)
    cs = scope_table(read_json(ROOT / "configs/callsign_fix_v1.json")).set_index(ID)
    off_cs, off = offsets(), offsets_all()
    seg = segments(MONTHS)
    k = read_json(ROOT / STRETCH)["candidates"]["two_sided_900"]["k_all_months"]
    gate_spec = read_json(ROOT / "configs/lirf_gate_v1.json")["candidates"]["u600_m600"]
    reread = read_json(ROOT / REREAD)["months"]

    frames, controls = {}, {}
    for month in MONTHS:
        f = rows[rows["month"] == month].reset_index(drop=True)
        mix, _ = share_gate.mix(f["committee"].to_numpy(dtype=float), f["nets"].to_numpy(dtype=float), network_member.V46_SHARE, network_member.V46_GATE)
        control = full_stack(f, mix, cs, off_cs, seg, off, k, gate_spec)
        score = rmse(f["label"].to_numpy(dtype=float), control)
        if not abs(score - reread[month]["candidate_rmse"]) <= 1e-9:
            raise ValueError(f"The control does not reproduce the v46 re-read on {month}: {score} against {reread[month]['candidate_rmse']}.")
        frames[month], controls[month] = f, control
    print("control reproduces the v46 re-read on", MONTHS, flush=True)
    bases, base_sources, base_tables, base_control = base_members(decl), {}, {}, {}
    for b in bases:
        recorded = read_json(ROOT / b["results"])
        if recorded.get("declaration_sha256") != object_hash(read_json(ROOT / b["declaration"])):
            raise ValueError("The base member's recorded results are not for its declaration.")
        for month in MONTHS:
            f = frames[month]
            if month not in b["member"]["tables"]:
                raise ValueError(f"The base member has no table for {month}.")
            table = with_bases(f["nets"].to_numpy(dtype=float), f["airport"].to_numpy() == "LIRF", f[ID].to_numpy(), month, [b], base_sources)
            mix, _ = share_gate.mix(f["committee"].to_numpy(dtype=float), table, network_member.V46_SHARE, network_member.V46_GATE)
            control = full_stack(f, mix, cs, off_cs, seg, off, k, gate_spec)
            score, want = rmse(f["label"].to_numpy(dtype=float), control), recorded["candidates"][b["candidate"]]["months"][month]["rmse"]
            if not abs(score - want) <= 1e-9:
                raise ValueError(f"The control with the base member does not reproduce its recorded {b['candidate']} on {month}: {score} against {want}.")
            base_tables[month], controls[month], base_control[month] = table, control, score
        print("control with the base member", b["declaration"], b["candidate"], "reproduces its recorded RMSEs on", MONTHS, flush=True)
    if args.check_control:
        return

    months = [m for m in MONTHS if m in decl["member"]["tables"]]
    report = {"created_at_utc": utc_now(), "declaration": decl, "declaration_sha256": object_hash(decl), "audit_used": False, "rows_sha256": sha256_file(ROOT / ROWS),
              "control": {"name": "v46 development stack" + (" with the base member" if bases else ""), "network_share": network_member.V46_SHARE,
                          "gate": network_member.V46_GATE, "reread": REREAD,
                          "base_members": [{"declaration": b["declaration"], "candidate": b["candidate"], "weight": b["weight"], "results": b["results"],
                                            "month_rmse": base_control, "tables": base_sources} for b in bases]},
              "equal_footing_months": equal, "minimum_mean_improvement_fraction": minimum, "member_sources": {}, "months": {}, "candidates": {}}
    entries = {name: {**decl["candidates"][name], "months": {}} for name in decl["candidates"]}
    for month in months:
        f, control = frames[month], controls[month]
        y, days = f["label"].to_numpy(dtype=float), f["day"].to_numpy()
        committee, nets = f["committee"].to_numpy(dtype=float), (base_tables[month] if bases else f["nets"].to_numpy(dtype=float))
        lirf = f["airport"].to_numpy() == "LIRF"
        member, sources = member_values(decl["member"], month, f[ID].to_numpy())
        member = np.where(lirf, np.nan, member)  # LIRF rows take their own blends; the committee mix does not reach them
        report["member_sources"].update(sources)
        both = np.isfinite(member) & np.isfinite(nets) & np.isfinite(committee)
        if not both.any():
            raise ValueError(f"The member covers no row of {month}.")
        entry = {"rows": int(len(f)), "seeds": len(sources), "control_rmse": rmse(y, control), "member_rows": int(both.sum()), "member_share_of_non_lirf": float(both[~lirf].mean()),
                 "standalone_on_member_rows": {"member": rmse(y[both], member[both]), "network_table": rmse(y[both], nets[both]), "committee": rmse(y[both], committee[both]),
                                               **{f"{s}:{c}": rmse(y[both & m], v[both & m]) for s, m in slices(f).items() if (both & m).any()
                                                  for c, v in (("member", member), ("network_table", nets))}}}
        report["months"][month] = entry
        for name in decl["candidates"]:
            table, inside = network_member.table_with_member(nets, member, network_member.candidate_weight(decl, name))
            mix, fire = share_gate.mix(committee, table, network_member.V46_SHARE, network_member.V46_GATE)
            values = full_stack(f, mix, cs, off_cs, seg, off, k, gate_spec)
            if not np.isfinite(values).all():
                raise ValueError(f"{name} has non-finite values on {month}.")
            changed = np.abs(values - control) > 1e-9
            if (changed & lirf).any():
                raise ValueError("A LIRF row changed.")
            if (changed & ~inside).any():
                raise ValueError("A row without a member value changed.")
            d = values - control
            entries[name]["months"][month] = {"rmse": rmse(y, values), "ratio_to_control": rmse(y, values) / rmse(y, control), "change_pct": 100 * (rmse(y, values) / rmse(y, control) - 1),
                                              "rows_changed": int(changed.sum()), "gate_rows": int((fire & ~lirf).sum()),
                                              "mean_change_s": float(d[changed].mean()) if changed.any() else 0.0, "sum_squared_change": float(np.sum(d ** 2)),
                                              "alignment": float(-np.sum(d * (control - y)) / np.sum(d ** 2)) if changed.any() else None,
                                              "by_scope_change_pct": {s: 100 * (rmse(y[m], values[m]) / rmse(y[m], control[m]) - 1) for s, m in slices(f).items()},
                                              "paired_days_vs_control": paired_days(days, y, values, control, SEED)}
    if args.smoke:
        ok = all(len(e["months"]) == len(months) for e in entries.values())
        print({"smoke": "pass" if ok else "fail", "months": months, "candidates": sorted(entries), "rows_changed": {n: {m: e["months"][m]["rows_changed"] for m in months} for n, e in entries.items()}})
        return
    eligible, selected = select(entries, equal, minimum)
    report["candidates"], report["eligible"], report["selected_candidate"] = entries, eligible, selected
    report["model_code"] = {p: sha256_file(ROOT / p) for p in CODE}
    report["environment"] = environment(ROOT)
    for path in [selection_path, report_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, report)
    print({"selected": selected, "eligible": eligible})
    for month in months:
        print(month, {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in report["months"][month].items() if kk != "standalone_on_member_rows"})
        print("   standalone:", {kk: round(v, 3) for kk, v in report["months"][month]["standalone_on_member_rows"].items()})
    for name, e in entries.items():
        line = " ".join(f"{m}:{e['months'][m]['change_pct']:+.4f}% [{e['months'][m]['paired_days_vs_control']['rmse_difference_95pct_interval_seconds'][0]:+.3f},"
                        f"{e['months'][m]['paired_days_vs_control']['rmse_difference_95pct_interval_seconds'][1]:+.3f}]" for m in months)
        print(f"{name:10s} mean {100 * (e['mean_ratio_equal_footing'] - 1):+.4f}% | {line} | eligible={e['eligible']}")


if __name__ == "__main__":
    main()
