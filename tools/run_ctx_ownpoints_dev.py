"""Stage 1 of the own-points development round (configs/ctx_ownpoints_dev_v1.json): the context-network member with the 15 own-aircraft
columns REPLACES the shipped combination member in the committee's network table. Arithmetic of tools/run_network_member.py (imported).

  control    the v46 development stack with the combination member (configs/adsb_ctx3_combo_dev_v1.json) at w 0.40 in the network
             table = v51's model part; its month RMSEs must equal docs/adsb_ctx3_combo/results_v1.json w40 before any candidate is read
  candidate  the same stack with the combination member removed and the candidate member in its place:
             table' = (1 - w) nets + w member seed mean on non-LIRF rows where both are finite; v46's mix; the whole v46 stack
  members    own_all      the declared member tables (tools/build_combo_member.py --declaration configs/ctx_ownpoints_dev_v1.json)
             own_matched  own_all on rows with adsb_matched = 1, the combination member on the others (built here, --build-matched)
Rule: tools/run_network_member.select with the declaration's equal-footing months and margin (the clean line's unchanged rule).

  --check-control             the control reproductions only (v46 re-read and the combination's w40)
  --identity                  code-path test: the combination member standing in for the candidate at w 0.40 must equal the control
  --build-matched [--leaves]  write runs/ctx_ownpoints_dev/member_matched/s<seed>/<leaf>.parquet (+ member_matched_build.json)
  --score --select            -> the declaration's selection and report paths
No label of a ranking row is read and no official score enters. Nothing is uploaded.
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
import run_network_member as rnm  # noqa: E402
from build_combo_member import matched_flags  # noqa: E402

DECLARATION = "configs/ctx_ownpoints_dev_v1.json"
COMBO = "configs/adsb_ctx3_combo_dev_v1.json"
COMBO_RESULTS = "docs/adsb_ctx3_combo/results_v1.json"
COMBO_W = 0.40
MATCHED_DIR = "runs/ctx_ownpoints_dev/member_matched"
ADSB_CACHE = "runs/adsb_ctx2/inputs/adsb_features_arcw0"
CODE = ["tools/run_ctx_ownpoints_dev.py", "tools/run_network_member.py", "src/prc2026_final/network_member.py", "src/prc2026_final/share_gate.py",
        "tools/run_share_gate_reread.py", "tools/build_combo_member.py"]


def matched_spec(decl: dict) -> dict:
    """The derived own_matched member: same seeds and leaves as the declared member, tables under MATCHED_DIR."""
    tables = {leaf: [p.replace("runs/ctx_ownpoints_dev/member/", MATCHED_DIR + "/") for p in paths] for leaf, paths in decl["member"]["tables"].items()}
    return {"name": "own_matched", "column": "adsb_full", "tables": tables}


def build_matched(decl: dict, leaves: list[str]) -> None:
    combo = read_json(ROOT / COMBO)["member"]["tables"]
    flags, cache = matched_flags(ROOT / ADSB_CACHE)
    spec, built = matched_spec(decl), {}
    for leaf in leaves:
        for own_rel, combo_rel, out_rel in zip(decl["member"]["tables"][leaf], combo[leaf], spec["tables"][leaf]):
            seed = own_rel.split("/s")[-1].split("/")[0]
            if f"/s{seed}/" not in combo_rel:
                raise ValueError(f"{leaf}: the combination table {combo_rel} is not seed {seed}.")
            own, com = pd.read_parquet(ROOT / own_rel), pd.read_parquet(ROOT / combo_rel)
            if not np.array_equal(own[ID].to_numpy(), com[ID].to_numpy()):
                raise ValueError(f"{leaf} seed {seed}: own and combination tables predict different rows.")
            m = flags.reindex(own[ID].to_numpy())
            if m.isna().any():
                raise ValueError(f"{leaf}: rows without a match flag.")
            m = m.to_numpy(dtype=bool)
            values = np.where(m, own["adsb_full"].to_numpy(dtype=float), com["adsb_full"].to_numpy(dtype=float))
            out = ROOT / out_rel
            out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({ID: own[ID].to_numpy(), "adsb_full": values}).to_parquet(out, index=False, compression="zstd")
            built[out_rel] = {"own": [own_rel, sha256_file(ROOT / own_rel)], "combination": [combo_rel, sha256_file(ROOT / combo_rel)],
                              "rows": int(len(values)), "matched_rows": int(m.sum()), "sha256": sha256_file(out)}
        print(leaf, "built", len(decl["member"]["tables"][leaf]), flush=True)
    report = ROOT / "runs/ctx_ownpoints_dev/member_matched_build.json"
    prior = read_json(report) if report.exists() else {"tables": {}}
    prior["tables"].update(built)
    write_json(report, {"created_at_utc": utc_now(), "declaration": DECLARATION, "declaration_sha256": sha256_file(ROOT / DECLARATION),
                        "adsb_cache": {"path": ADSB_CACHE, "sha256": cache}, "tables": prior["tables"], "tool_sha256": sha256_file(ROOT / "tools/run_ctx_ownpoints_dev.py"),
                        "audit_used": False, "note": "own_all on adsb_matched = 1 rows, the combination member elsewhere; no label read"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--declaration", default=DECLARATION)
    parser.add_argument("--check-control", action="store_true")
    parser.add_argument("--identity", action="store_true")
    parser.add_argument("--build-matched", action="store_true")
    parser.add_argument("--leaves", nargs="*")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--select", action="store_true")
    args = parser.parse_args()
    decl = read_json(ROOT / args.declaration)
    if not decl.get("declared_at_utc") or decl.get("audit_used"):
        raise ValueError("The round is not declared or does not respect its own contract.")
    if args.build_matched:
        build_matched(decl, args.leaves or list(decl["member"]["tables"]))
        return
    if not (args.check_control or args.identity or (args.score and args.select)):
        raise SystemExit("Use --check-control, --identity, --build-matched or --score --select.")
    equal = list(decl.get("equal_footing_months", rnm.EQUAL))
    minimum = float(decl.get("minimum_mean_improvement_fraction", rnm.MINIMUM))
    governing = args.score and args.select
    if governing:
        selection_path, report_path = ROOT / decl["selection"], ROOT / decl["report"]
        if selection_path.exists() or report_path.exists():
            raise ValueError("Preserve the original selection.")

    rows = pd.read_parquet(ROOT / rnm.ROWS)
    cs = rnm.scope_table(read_json(ROOT / "configs/callsign_fix_v1.json")).set_index(ID)
    off_cs, off = rnm.offsets(), rnm.offsets_all()
    seg = rnm.segments(rnm.MONTHS)
    k = read_json(ROOT / rnm.STRETCH)["candidates"]["two_sided_900"]["k_all_months"]
    gate_spec = read_json(ROOT / "configs/lirf_gate_v1.json")["candidates"]["u600_m600"]
    reread = read_json(ROOT / rnm.REREAD)["months"]
    combo_decl, combo_rec = read_json(ROOT / COMBO), read_json(ROOT / COMBO_RESULTS)
    if combo_rec.get("declaration_sha256") != object_hash(combo_decl):
        raise ValueError("The combination's recorded results are not for its declaration.")

    def stack(f: pd.DataFrame, table: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mix, fire = share_gate.mix(f["committee"].to_numpy(dtype=float), table, network_member.V46_SHARE, network_member.V46_GATE)
        return rnm.full_stack(f, mix, cs, off_cs, seg, off, k, gate_spec), fire

    frames, controls, inside_combo, sources = {}, {}, {}, {}
    for month in rnm.MONTHS:
        f = rows[rows["month"] == month].reset_index(drop=True)
        y, ids = f["label"].to_numpy(dtype=float), f[ID].to_numpy()
        lirf, nets = f["airport"].to_numpy() == "LIRF", f["nets"].to_numpy(dtype=float)
        v46, _ = stack(f, nets)
        if abs(rnm.rmse(y, v46) - reread[month]["candidate_rmse"]) > 1e-9:
            raise ValueError(f"{month}: the v46 stack does not reproduce its re-read.")
        combo, found = rnm.member_values(combo_decl["member"], month, ids)
        sources.update(found)
        table, inside = network_member.table_with_member(nets, np.where(lirf, np.nan, combo), COMBO_W)
        control, _ = stack(f, table)
        want = combo_rec["candidates"]["w40"]["months"][month]["rmse"]
        if abs(rnm.rmse(y, control) - want) > 1e-9:
            raise ValueError(f"{month}: the control does not reproduce the combination's w40: {rnm.rmse(y, control)} against {want}.")
        frames[month], controls[month], inside_combo[month] = f, control, inside
    print("control (v46 + combination w40) reproduces the recorded RMSEs on", rnm.MONTHS, flush=True)
    if args.check_control:
        return

    specs = {"own_all": decl["member"], "own_matched": matched_spec(decl)}
    if args.identity:
        specs = {name: combo_decl["member"] for name in specs}
    months = [m for m in rnm.MONTHS if all(m in s["tables"] for s in specs.values())]
    if governing and set(equal) - set(months):
        raise ValueError("A member has no table for every equal-footing month.")
    report = {"created_at_utc": utc_now(), "declaration": decl, "declaration_sha256": object_hash(decl), "audit_used": False, "rows_sha256": sha256_file(ROOT / rnm.ROWS),
              "control": {"name": "v46 development stack with the combination member at w 0.40 (v51's model part)", "combination": COMBO, "combination_results": COMBO_RESULTS,
                          "month_rmse": {m: rnm.rmse(frames[m]["label"].to_numpy(dtype=float), controls[m]) for m in rnm.MONTHS}, "combination_tables": sources},
              "equal_footing_months": equal, "minimum_mean_improvement_fraction": minimum, "member_sources": {}, "months": {}, "candidates": {}}
    entries = {name: {**decl["candidates"][name], "months": {}} for name in decl["candidates"]}
    for month in months:
        f, control = frames[month], controls[month]
        y, days, ids = f["label"].to_numpy(dtype=float), f["day"].to_numpy(), f[ID].to_numpy()
        lirf, nets, committee = f["airport"].to_numpy() == "LIRF", f["nets"].to_numpy(dtype=float), f["committee"].to_numpy(dtype=float)
        members, info = {}, {"rows": int(len(f)), "control_rmse": rnm.rmse(y, control)}
        for name, spec in specs.items():
            values, found = rnm.member_values(spec, month, ids)
            report["member_sources"].update(found)
            members[name] = np.where(lirf, np.nan, values)
            both = np.isfinite(members[name]) & np.isfinite(nets)
            info[f"{name}_rows"] = int(both.sum())
            info[f"{name}_standalone"] = {"member": rnm.rmse(y[both], members[name][both]), "network_table": rnm.rmse(y[both], nets[both]),
                                          **{f"{s}:member": rnm.rmse(y[both & m], members[name][both & m]) for s, m in rnm.slices(f).items() if (both & m).any()},
                                          **{f"{s}:network_table": rnm.rmse(y[both & m], nets[both & m]) for s, m in rnm.slices(f).items() if (both & m).any()}}
        report["months"][month] = info
        for name, spec in decl["candidates"].items():
            w = float(spec["member_weight"])
            table, inside = network_member.table_with_member(nets, members[spec["member"]], w)
            values, fire = stack(f, table)
            if not np.isfinite(values).all():
                raise ValueError(f"{name} has non-finite values on {month}.")
            changed = np.abs(values - control) > 1e-9
            if (changed & lirf).any():
                raise ValueError("A LIRF row changed.")
            if (changed & ~(inside | inside_combo[month])).any():
                raise ValueError("A row outside both members changed.")
            if args.identity and abs(w - COMBO_W) < 1e-12 and changed.any():
                raise ValueError(f"identity failed: {name} on {month} differs from the control on {int(changed.sum())} rows.")
            d = values - control
            entries[name]["months"][month] = {"rmse": rnm.rmse(y, values), "ratio_to_control": rnm.rmse(y, values) / rnm.rmse(y, control),
                                              "change_pct": 100 * (rnm.rmse(y, values) / rnm.rmse(y, control) - 1), "rows_changed": int(changed.sum()),
                                              "gate_rows": int((fire & ~lirf).sum()), "sum_squared_change": float(np.sum(d ** 2)),
                                              "alignment": float(-np.sum(d * (control - y)) / np.sum(d ** 2)) if changed.any() else None,
                                              "by_scope_change_pct": {s: 100 * (rnm.rmse(y[m], values[m]) / rnm.rmse(y[m], control[m]) - 1) for s, m in rnm.slices(f).items()},
                                              "paired_days_vs_control": rnm.paired_days(days, y, values, control, rnm.SEED)}
        print(month, "scored", flush=True)
    if args.identity:
        print({"identity": "pass", "months": months, "rows_changed_w40": {n: {m: e["months"][m]["rows_changed"] for m in months} for n, e in entries.items() if abs(float(e["member_weight"]) - COMBO_W) < 1e-12}})
        return
    eligible, selected = rnm.select(entries, equal, minimum)
    report["candidates"], report["eligible"], report["selected_candidate"] = entries, eligible, selected
    report["model_code"] = {p: sha256_file(ROOT / p) for p in CODE}
    report["environment"] = environment(ROOT)
    for path in [selection_path, report_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, report)
    print({"selected": selected, "eligible": eligible})
    for name, e in entries.items():
        line = " ".join(f"{m}:{e['months'][m]['change_pct']:+.4f}% [{e['months'][m]['paired_days_vs_control']['rmse_difference_95pct_interval_seconds'][0]:+.3f},"
                        f"{e['months'][m]['paired_days_vs_control']['rmse_difference_95pct_interval_seconds'][1]:+.3f}]" for m in months)
        print(f"{name:16s} mean {100 * (e['mean_ratio_equal_footing'] - 1):+.4f}% | {line} | eligible={e['eligible']}")


if __name__ == "__main__":
    main()
