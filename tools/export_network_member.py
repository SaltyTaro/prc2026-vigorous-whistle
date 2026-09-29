"""Export v46 with one more member in the committee's network table (a declaration scored by tools/run_network_member.py).

Construction, on the ranking rows:
  1. tools/export_disagreement_gate.rebuild_v34: v31 / v33 / v34 floats, each rounding to its upload; the dry-run recipe detail.
  2. v38's floats (tools/export_stack.layers on v34) must round to the uploaded v38 on every row.
  3. v46's floats: the detail's committee_mix replaced by v46's mix (network share 0.85, share 0.5 where the committee sits more than
     1,200 s above the network table), v34's recipe recomputed, the same layers call. Where they differ from v38's floats they must
     round to the uploaded v46; every other row of v46 must equal v45.
  4. candidate: the network table replaced by table' = (1 - w) table + w member seed mean (the declaration's ranking tables, refit on
     twelve months) where both are finite; v46's mix on table'; the same layers call, so the u300 gate, the callsign and queue layers
     and the stretch act on the candidate's own values.
  5. export = the candidate's float where it differs from v46's float, v46's integer elsewhere. LIRF rows do not depend on the committee
     mix and must not change; a changed row must carry no layer above v38 in v46 other than the mix.
Independent rebuild from the recorded v30 detail's components. Nothing is uploaded.

  --declaration PATH --candidate NAME --dry-run DIR
  --declaration PATH --candidate NAME --version 48 [--user-override TEXT]
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
from prc2026.features import ID, TARGET  # noqa: E402
from prc2026_final import network_member, share_gate  # noqa: E402
from prc2026_submission.contract import export_submission  # noqa: E402
import export_v33  # noqa: E402
from export_disagreement_gate import V34_LIRF, rebuild_v34  # noqa: E402
from export_fair_weights import FLOOR, V33_OUTER, recipe, stats  # noqa: E402
from export_share_gate import STACK, V38, V45, uploaded  # noqa: E402
from export_stack import layers  # noqa: E402
from run_network_member import member_values  # noqa: E402

V46 = ("runs/submissions/vigorous-whistle_v46.parquet", "8abdd87ad5098d7753927f407ffd7e8cd342aa84a2864392e1b64c0dde015af9", 223.9339)
CODE = ["tools/export_network_member.py", "tools/run_network_member.py", "src/prc2026_final/network_member.py", "src/prc2026_final/share_gate.py", "tools/export_share_gate.py",
        "tools/export_stack.py", "tools/export_disagreement_gate.py", "tools/export_fair_weights.py", "tools/export_v33.py"]


def through_stack(detail: pd.DataFrame, table: np.ndarray, v34: np.ndarray, at34: np.ndarray, ids: np.ndarray, share: float, gate: dict | None):
    """(floats, masks, gate mask) of the shipped stack with the committee mix at `share` / `gate` on `table`."""
    mix, fire = share_gate.mix(detail["committee"].to_numpy(dtype=float), table, share, gate)
    d = detail.assign(committee_mix=mix)
    at = recipe(d, V33_OUTER, V34_LIRF)
    values, masks = layers(v34 + (at - at34), d, at, ids, STACK["qw"], STACK["qsource"], STACK["mix_up"], STACK["peer"], STACK["stretch"], STACK["stretch_lower"])
    return values, masks, fire & (detail["airport"].to_numpy() != "LIRF")


def stacks(detail: pd.DataFrame, member: np.ndarray, weight: float, v34: np.ndarray, at34: np.ndarray, ids: np.ndarray):
    """(v38 floats, v46 floats, candidate floats, masks, candidate gate mask, rows where the member enters)."""
    table = share_gate.network_table(detail)
    v38, masks, _ = through_stack(detail, table, v34, at34, ids, share_gate.CONTROL_SHARE, None)
    if not float(np.nanmax(np.abs(share_gate.mix(detail["committee"].to_numpy(dtype=float), table, share_gate.CONTROL_SHARE)[0] - detail["committee_mix"].to_numpy(dtype=float)))) <= 1e-9:
        raise ValueError("The detail's committee_mix is not the v38 mix.")
    v46, _, _ = through_stack(detail, table, v34, at34, ids, network_member.V46_SHARE, network_member.V46_GATE)
    lirf = detail["airport"].to_numpy() == "LIRF"
    new, inside = network_member.table_with_member(table, np.where(lirf, np.nan, member), weight)
    candidate, _, fire = through_stack(detail, new, v34, at34, ids, network_member.V46_SHARE, network_member.V46_GATE)
    return v38, v46, candidate, masks, fire, inside


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--declaration", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--version", type=int, default=48)
    parser.add_argument("--dry-run", type=Path)
    parser.add_argument("--user-override")
    args = parser.parse_args()
    decl = read_json(ROOT / args.declaration)
    weight = network_member.candidate_weight(decl, args.candidate)
    selected = None
    if args.dry_run is None:
        results = read_json(ROOT / decl["report"])
        if results.get("declaration_sha256") != object_hash(decl):
            raise ValueError("The recorded results are not for this declaration.")
        selected = results.get("selected_candidate")
        if selected != args.candidate and not args.user_override:
            raise ValueError(f"The round did not select {args.candidate}; shipping it needs --user-override with the user's words.")
        output = ROOT / f"runs/submissions/vigorous-whistle_v{args.version}.parquet"
        report_path = ROOT / decl.get("export_report_dir", "docs/network_member") / f"export_v{args.version}.json"
    else:
        out_dir = args.dry_run.resolve()
        if (ROOT / "runs/submissions").resolve() in [out_dir, *out_dir.parents]:
            raise ValueError("A dry run never writes under runs/submissions.")
        output, report_path = out_dir / f"vigorous-whistle_v{args.version}.parquet", out_dir / f"export_v{args.version}.json"
    if output.exists():
        raise ValueError(f"{output} exists; never overwrite.")

    rec, ids, dry, lfpg, _, v34, at34 = rebuild_v34()
    v38, v45, v46 = uploaded(*V38, ids), uploaded(V45[0], V45[1], ids), uploaded(V46[0], V46[1], ids)
    member, sources = member_values(decl["member"], "ranking", ids)
    f38, f46, candidate, masks, fire, inside = stacks(dry, member, weight, v34, at34, ids)
    if not (np.isfinite(f38).all() and np.isfinite(f46).all() and np.isfinite(candidate).all()):
        raise ValueError("Non-finite stack values.")
    if not np.array_equal(np.rint(f38).astype(np.int64), v38):
        raise ValueError("The v38 stack does not round to the uploaded v38.")
    shared = np.abs(f46 - f38) > 1e-9
    if not np.array_equal(np.where(shared, np.rint(f46).astype(np.int64), v45), v46):
        raise ValueError("The v46 stack does not reproduce the uploaded v46.")
    airport, month = dry["airport"].to_numpy(), dry["month"].to_numpy()
    changed = np.abs(candidate - f46) > 1e-9
    if (changed & (airport == "LIRF")).any():
        raise ValueError("A LIRF row depends on the committee mix.")
    if (changed & ~inside).any():
        raise ValueError("A row without a member value changed.")
    if not np.array_equal(v45[changed], v38[changed]):
        raise ValueError("A changed row carries a layer above v38 in v45.")
    values = np.where(changed, candidate, v46.astype(float))
    provenance = {"experiment_id": decl["experiment_id"], "candidate": args.candidate, "dry_run": args.dry_run is not None,
                  "selected_by": ("dry run" if args.dry_run is not None else "the declared rule" if selected == args.candidate
                                  else f"the user's override of the declared rule, which selected {selected}: {args.user_override}"),
                  "recipe": f"v46 with the network table at (1 - {weight}) x (0.25 e24 + 0.25 e24b + 0.5 e32) + {weight} x {decl['member']['name']} ({len(sources)} seeds, twelve-month refit) "
                            "where the member has a value, through the shipped stack (v34 weights, network share 0.85 with the networks-far-below gate, u300 gate, callsign fix 50%, "
                            "queue trust 0.75, two-sided stretch 1500 / 900); LIRF rows unchanged; floor -12 s",
                  "incumbent": {"submission": Path(V46[0]).name, "official_rmse_seconds": V46[2], "sha256": V46[1], "note": "the clean model line's best"},
                  "declaration": {args.declaration: {"sha256": sha256_file(ROOT / args.declaration), "object_hash": object_hash(decl)}},
                  "development_results": {decl["report"]: sha256_file(ROOT / decl["report"])} if (ROOT / decl["report"]).exists() else {},
                  "code": {p: sha256_file(ROOT / p) for p in CODE}, "member_tables": sources,
                  "inputs": {p: sha256_file(ROOT / p) for p in [export_v33.DRY, export_v33.REPAIRED, export_v33.V30_DETAIL, V38[0], V45[0], V46[0]]},
                  "external_data": decl["member"].get("external_data", "adsb.lol globe_history (ODbL) surface positions, as in v46"), "audit_used": False, "floor_seconds": FLOOR,
                  "official_scores_used": "none"}
    receipt = export_submission(pd.DataFrame({ID: ids, "prediction": values}), ROOT / "data/raw/submitting.parquet", output, provenance)
    exported = pd.read_parquet(output).set_index(ID)[TARGET].reindex(ids).to_numpy().astype(np.int64)
    change = exported.astype(float) - v46.astype(float)
    if ((change != 0) & ~changed).any():
        raise ValueError("A row outside the candidate's scope changed.")

    # independent rebuild from the recorded v30 detail's own components
    rec34 = recipe(rec, V33_OUTER, V34_LIRF)
    _, f46_r, candidate_r, _, fire_r, _ = stacks(rec, member, weight, np.maximum(rec34 + lfpg, FLOOR), rec34, ids)
    independent = np.where(np.abs(candidate_r - f46_r) > 1e-9, candidate_r, v46.astype(float))
    differing = np.rint(independent).astype(np.int64) != exported
    n = len(ids)
    moved = change != 0
    anchored = masks["callsign"] | masks["queue"]
    report = {"created_at_utc": utc_now(), "version": args.version, "candidate": args.candidate, "member": decl["member"]["name"], "member_weight": weight, "seeds": len(sources),
              "base": 46, "dry_run": args.dry_run is not None, "output": output.as_posix(), "sha256": sha256_file(output), "receipt": receipt, "rows": n,
              "checks": {"v38_stack_rounds_to_uploaded_v38": True, "v46_stack_reproduces_uploaded_v46": True, "lirf_rows_changed": 0, "changed_rows_equal_v38_in_v45": True,
                         "rows_changed_outside_scope": 0},
              "member_rows": int(inside.sum()), "member_share_of_non_lirf": float(inside[airport != "LIRF"].mean()),
              "rows_changed_float": int(changed.sum()), "rows_changed_integer": int(moved.sum()), "gate_rows": int(fire.sum()),
              "gate_rows_by_month": {m: int((fire & (month == m)).sum()) for m in sorted(set(month))},
              "rows_changed_in_anchored_layers": int((moved & anchored).sum()),
              "mean_change_s": float(change[moved].mean()) if moved.any() else 0.0, "mean_abs_change_s": float(np.abs(change[moved]).mean()) if moved.any() else 0.0,
              "max_abs_change_s": float(np.abs(change).max()), "sum_squared_change": float((change ** 2).sum()),
              "by_month": {m: stats(change, month == m) for m in sorted(set(month))}, "by_airport": {a: stats(change, airport == a) for a in sorted(set(airport))},
              "independent_rebuild": {"source": export_v33.V30_DETAIL, "rows_differing": int(differing.sum()), "gate_rows_mask_disagreement": int((fire != fire_r).sum()),
                                      "max_abs_integer_difference": int(np.abs(np.rint(independent).astype(np.int64) - exported).max())},
              "rmse_if_every_changed_label_equals_the_new_value": float(np.sqrt(max(n * V46[2] ** 2 - (change ** 2).sum(), 0.0) / n)),
              "rmse_if_every_changed_label_equals_the_old_value": float(np.sqrt((n * V46[2] ** 2 + (change ** 2).sum()) / n))}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, report)
    print({k: report[k] for k in ["sha256", "member_rows", "rows_changed_integer", "gate_rows", "gate_rows_by_month", "mean_change_s", "mean_abs_change_s", "max_abs_change_s",
                                  "sum_squared_change", "independent_rebuild"]})
    print({a: {k: round(v, 3) if isinstance(v, float) else v for k, v in e.items()} for a, e in report["by_airport"].items()})


if __name__ == "__main__":
    main()
