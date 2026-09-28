"""Export the stacked candidate on v34: u300 gate (configs/disagreement_gate_v1.json) + callsign fix 50% (configs/callsign_fix_v1.json)
+ optionally queue trust (configs/queue_trust_v1.json) at weight W on its scope rows, restricted to regular ADS-B matches with
--queue-source regular (the round's callsign-matched queue row failed in July 2025; disclosed post-hoc restriction).

Construction: v34 floats rebuilt (tools/export_disagreement_gate.rebuild_v34; v31 / v33 / v34 must round to the uploads) -> + u300 gate delta
(dry-run detail) -> callsign rows = max(0.5 x observed taxi + 0.5 x start, -12) -> queue rows = max(W x observed taxi + (1 - W) x that, -12).
Independent rebuild from the recorded v30 detail's components. Nothing is uploaded.
Optional last layer (configs/higher_bundle_v2.json): --peer-lift adds the peer-congestion lifts (tools/peer_congestion.py on
ranking.parquet): +100 s on non-LIRF rows without an NM clock and +50 s on untracked non-LIRF rows with one, where the airport-hour's
NM proxies run more than 600 s above the month's normal (>= 5 peers).

  --queue W --queue-source regular|all --dry-run OUT
  --queue W --queue-source regular --version N --user-override TEXT
  [--mix-up 600] [--peer-lift]
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
from prc2026_submission.contract import export_submission  # noqa: E402
import export_v33  # noqa: E402
from export_disagreement_gate import V34_LIRF, V34_SHA, gate_delta, rebuild_v34  # noqa: E402
from export_fair_weights import FLOOR, V33_OUTER, recipe, stats  # noqa: E402
from run_callsign_fix import offsets, scope_table  # noqa: E402
from run_queue_trust import offsets_all, segments  # noqa: E402
from peer_congestion import ranking_peers  # noqa: E402

DECLS = ["configs/disagreement_gate_v1.json", "configs/callsign_fix_v1.json", "configs/queue_trust_v1.json", "configs/mix_gate_v1.json"]
RESULTS = ["docs/disagreement/gate_v1_results.json", "docs/disagreement/callsign_fix_v1_results.json", "docs/disagreement/queue_trust_v1_results.json",
           "docs/disagreement/mix_gate_v1_results.json"]
V35_FILE, V35_SHA = "runs/submissions/vigorous-whistle_v35.parquet", "75f4a4f4678921d957909ef364c2cae66a81e5b74bbb680ded50d329ce6ee4f5"
STRETCH_DECL, STRETCH_RESULTS = "configs/tail_stretch_v1.json", "docs/disagreement/tail_stretch_v1_results.json"
STRETCH2_DECL, STRETCH2_RESULTS = "configs/tail_stretch_v2.json", "docs/disagreement/tail_stretch_v2_results.json"
PEER_DECL, PEER_RESULTS = "configs/higher_bundle_v2.json", "docs/disagreement/higher_bundle_v2_results.json"


def gated_detail(detail: pd.DataFrame, up: float | None) -> pd.DataFrame:
    """The detail with its committee mix replaced by configs/mix_gate_v1.json's mix' (network share 1.0 where nets - committee > up)."""
    if up is None:
        return detail
    c = detail["committee"].to_numpy(dtype=float)
    n = (0.25 * detail["ens_ctx_e24"] + 0.25 * detail["ens_ctx_e24b"] + 0.5 * detail["ens_ctx_e32"]).to_numpy(dtype=float)
    finite = np.isfinite(c) & np.isfinite(n)
    if not float(np.nanmax(np.abs(np.where(finite, 0.25 * c + 0.75 * n, c) - detail["committee_mix"].to_numpy(dtype=float)))) <= 1e-9:
        raise ValueError("The detail's committee_mix is not mixed(committee, nets, 0.75).")
    share = np.where(finite & (n - c > up), 1.0, 0.75)
    return detail.assign(committee_mix=np.where(finite, (1 - share) * c + share * n, c))


def peer_lifts(values: np.ndarray, detail: pd.DataFrame, ids: np.ndarray) -> tuple[np.ndarray, dict]:
    """configs/higher_bundle_v2.json's noclock / clock-untracked lifts on the ranking rows."""
    spec = read_json(ROOT / PEER_DECL)["candidates"]["mu_pc_pcc"]
    p = ranking_peers().set_index(ID).reindex(ids)
    if p["peer_n"].isna().any():
        raise ValueError("Peer congestion does not cover every ranking row.")
    non_lirf = detail["airport"].to_numpy() != "LIRF"
    untracked = ~detail["scope"].isin(["matched_no_lirf", "lirf_nonpop_matched"]).to_numpy()
    ok = (p["peer_n"].to_numpy() >= 5) & non_lirf
    peer_c, no_clock = p["peer_c"].to_numpy(dtype=float), p["no_clock"].to_numpy(dtype=bool)
    nc = ok & no_clock & (peer_c > spec["noclock_lift"]["peer_c_above"])
    cu = ok & ~no_clock & untracked & (peer_c > spec["clock_untracked_lift"]["peer_c_above"])
    out = values.copy()
    out[nc] += spec["noclock_lift"]["seconds"]
    out[cu] += spec["clock_untracked_lift"]["seconds"]
    return out, {"peer_noclock": nc, "peer_clock_untracked": cu}


def tail_stretch(values: np.ndarray, detail: pd.DataFrame, anchored: np.ndarray, t: float, variant: str, lower: float | None = None) -> tuple[np.ndarray, dict]:
    """configs/tail_stretch_v1.json: value + k x max(value - t, 0) on untracked / tracked non-LIRF rows outside the anchored layers,
    k = the round's four-month fit for the candidate {variant}_t{t}; with `lower`, configs/tail_stretch_v2.json's two_sided_{lower}
    then adds k2 x max(lower - u, 0) to the upper-stretched value u."""
    k = read_json(ROOT / STRETCH_RESULTS)["candidates"][f"{variant}_t{int(t)}"]["k_all_months"]
    k2 = None
    if lower is not None:
        if variant != "v35" or int(t) != 1500:
            raise ValueError("The two-sided stretch is declared only on the v35 stack with the upper threshold 1500.")
        two = read_json(ROOT / STRETCH2_RESULTS)["candidates"][f"two_sided_{int(lower)}"]["k_all_months"]
        if any(abs(two[s][0] - k[s]) > 1e-12 for s in k):
            raise ValueError("tail_stretch_v2's upper k differs from tail_stretch_v1's.")
        k2 = {s: two[s][1] for s in two}
    non_lirf = detail["airport"].to_numpy() != "LIRF"
    tracked = detail["scope"].eq("matched_no_lirf").to_numpy()
    masks = {"stretch_untracked": non_lirf & ~tracked & ~anchored & (values > t), "stretch_tracked": non_lirf & tracked & ~anchored & (values > t)}
    if k2 is not None:
        masks = {"stretch_untracked": non_lirf & ~tracked & ~anchored, "stretch_tracked": non_lirf & tracked & ~anchored}
    out = values.copy()
    for name, scope in [("stretch_untracked", "untracked"), ("stretch_tracked", "tracked")]:
        m = masks[name]
        u = out[m] + k[scope] * np.maximum(out[m] - t, 0.0)
        if k2 is not None:
            u = u + k2[scope] * np.maximum(lower - u, 0.0)
        out[m] = np.maximum(u, FLOOR)
    return out, masks


def layers(v34_start: np.ndarray, detail: pd.DataFrame, at34: np.ndarray, ids: np.ndarray, qw: float | None, qsource: str, mix_up: float | None = None,
           peer: bool = False, stretch: float | None = None, stretch_lower: float | None = None):
    """(values, masks): v34 floats -> [mix gate] -> u300 -> callsign 50% -> queue trust -> [peer lifts]."""
    gd = gated_detail(detail, mix_up)
    at_g = recipe(gd, V33_OUTER, V34_LIRF) if mix_up is not None else at34
    mix_delta = at_g - at34
    gdelta, _ = gate_delta(gd, read_json(ROOT / DECLS[0])["candidates"]["u300"], at_g)
    values = np.maximum(v34_start + mix_delta + gdelta, FLOOR)
    cs = scope_table(read_json(ROOT / DECLS[1])).set_index(ID).reindex(ids)
    off_cs = offsets()
    cs_in = cs["motion_start"].notna().to_numpy()
    cs_taxi = cs["takeoff"].to_numpy() - cs["motion_start"].to_numpy() + np.array([off_cs.get(a, np.nan) for a in cs["airport"].fillna("")])
    values = values.copy()
    values[cs_in] = np.maximum(0.5 * cs_taxi[cs_in] + 0.5 * values[cs_in], FLOOR)
    q_in = np.zeros(len(ids), dtype=bool)
    if qw is not None:
        seg = segments(["2026-01", "2026-07"]).reindex(ids)
        off = offsets_all()
        q_in = seg["in_scope"].fillna(False).to_numpy(dtype=bool)
        if qsource == "regular":
            q_in &= seg["source"].eq("regular").to_numpy()
        q_taxi = seg["taxi"].to_numpy() + np.array([off.get(a, np.nan) for a in seg["airport"].fillna("")])
        values[q_in] = np.maximum(qw * q_taxi[q_in] + (1 - qw) * values[q_in], FLOOR)
    masks = {"mix_gate": mix_delta != 0, "u300": gdelta != 0, "callsign": cs_in, "queue": q_in}
    if peer:
        values, extra = peer_lifts(values, detail, ids)
        masks.update(extra)
    if stretch is not None:
        variant = "v36" if (peer and mix_up == 600) else "v35" if (not peer and mix_up is None) else None
        if variant is None or qw != 0.75 or qsource != "regular":
            raise ValueError("The tail stretch is declared only on the v35 or v36 stack.")
        values, extra = tail_stretch(values, detail, cs_in | q_in, stretch, variant, stretch_lower)
        masks.update(extra)
    return values, masks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queue", type=float)
    parser.add_argument("--queue-source", choices=["regular", "all"], default="regular")
    parser.add_argument("--mix-up", type=float, help="configs/mix_gate_v1.json: network share 1.0 where nets - committee exceeds this")
    parser.add_argument("--peer-lift", action="store_true", help="configs/higher_bundle_v2.json's peer-congestion lifts as the last layer")
    parser.add_argument("--stretch", type=float, help="configs/tail_stretch_v1.json: upper-tail stretch above this threshold (after every other layer)")
    parser.add_argument("--stretch-lower", type=float, help="configs/tail_stretch_v2.json: lower stretch below this threshold (needs --stretch 1500)")
    parser.add_argument("--version", type=int, default=35)
    parser.add_argument("--dry-run", type=Path)
    parser.add_argument("--user-override")
    args = parser.parse_args()
    decls, results = (DECLS + [PEER_DECL], RESULTS + [PEER_RESULTS]) if args.peer_lift else (DECLS, RESULTS)
    if args.stretch is not None:
        decls, results = decls + [STRETCH_DECL], results + [STRETCH_RESULTS]
    if args.stretch_lower is not None:
        decls, results = decls + [STRETCH2_DECL], results + [STRETCH2_RESULTS]
    for d, r in zip(decls, results):
        if read_json(ROOT / r).get("declaration_sha256") != object_hash(read_json(ROOT / d)):
            raise ValueError(f"{r} is not the recorded result of {d}.")
    if args.dry_run is None:
        if not args.user_override:
            raise ValueError("None of the three rounds selected; shipping the stack needs --user-override.")
        output, report_path = ROOT / f"runs/submissions/vigorous-whistle_v{args.version}.parquet", ROOT / f"docs/disagreement/export_v{args.version}.json"
    else:
        out_dir = args.dry_run.resolve()
        if (ROOT / "runs/submissions").resolve() in [out_dir, *out_dir.parents]:
            raise ValueError("A dry run never writes under runs/submissions.")
        output, report_path = out_dir / f"vigorous-whistle_v{args.version}.parquet", out_dir / f"export_v{args.version}.json"
    if output.exists():
        raise ValueError(f"{output} exists; never overwrite.")

    rec, ids, dry, lfpg, uploaded, v34, at34 = rebuild_v34()
    values, m = layers(v34, dry, at34, ids, args.queue, args.queue_source, args.mix_up, args.peer_lift, args.stretch, args.stretch_lower)
    touched = np.logical_or.reduce(list(m.values()))
    if not np.isfinite(values).all():
        raise ValueError("Non-finite export values.")
    provenance = {"experiment_id": "stack_u300_cs50" + (f"_qt{int(round(100 * args.queue))}_{args.queue_source}" if args.queue is not None else "") + (f"_mu{int(args.mix_up)}" if args.mix_up is not None else "") + ("_pc_pcc" if args.peer_lift else "") + (f"_t{int(args.stretch)}" if args.stretch is not None else "") + (f"_l{int(args.stretch_lower)}" if args.stretch_lower is not None else ""),
                  "dry_run": args.dry_run is not None, "selected_by": f"user override of the unchanged rule: {args.user_override}" if args.user_override else "dry run",
                  "recipe": "v34" + (f" + mix gate (network share 1.0 where nets - committee > {args.mix_up} s)" if args.mix_up is not None else "") + " + u300 disagreement gate + callsign fix 50%" + (f" + queue trust {args.queue} ({args.queue_source} matches)" if args.queue is not None else "")
                            + (" + peer-congestion lifts (configs/higher_bundle_v2.json)" if args.peer_lift else "")
                            + (f" + tail stretch above {args.stretch} s (configs/tail_stretch_v1.json, four-month k)" if args.stretch is not None else "")
                            + (f" + lower stretch below {args.stretch_lower} s (configs/tail_stretch_v2.json, four-month k)" if args.stretch_lower is not None else ""),
                  "incumbent": ({"submission": "vigorous-whistle_v35.parquet", "official_rmse_seconds": 226.8989, "sha256": V35_SHA} if args.version >= 36
                                else {"submission": "vigorous-whistle_v34.parquet", "official_rmse_seconds": 228.5933, "sha256": V34_SHA}),
                  "declarations": {d: sha256_file(ROOT / d) for d in decls}, "development_results": {r: sha256_file(ROOT / r) for r in results},
                  "external_data": "adsb.lol globe_history (ODbL) surface positions",
                  "inputs": {p: sha256_file(ROOT / p) for p in [export_v33.DRY, export_v33.REPAIRED, export_v33.V30_DETAIL, *read_json(ROOT / DECLS[1])["rematch"]["outputs"]]},
                  "audit_used": False, "floor_seconds": FLOOR}
    receipt = export_submission(pd.DataFrame({ID: ids, "prediction": values}), ROOT / "data/raw/submitting.parquet", output, provenance)
    exported = pd.read_parquet(output).set_index(ID)[TARGET].reindex(ids).to_numpy().astype(np.int64)
    change = exported.astype(float) - uploaded[34].astype(float)
    if sha256_file(ROOT / V35_FILE) != V35_SHA:
        raise ValueError("The uploaded v35 changed.")
    v35 = pd.read_parquet(ROOT / V35_FILE).set_index(ID)[TARGET].reindex(ids).to_numpy().astype(np.int64)
    change35 = exported.astype(float) - v35.astype(float)

    rec34 = recipe(rec, V33_OUTER, V34_LIRF)
    independent, _ = layers(np.maximum(rec34 + lfpg, FLOOR), rec, rec34, ids, args.queue, args.queue_source, args.mix_up, args.peer_lift, args.stretch, args.stretch_lower)
    differing = np.rint(independent).astype(np.int64) != exported
    airport = rec["airport"].to_numpy()
    report = {"created_at_utc": utc_now(), "version": args.version, "dry_run": args.dry_run is not None, "output": output.as_posix(), "receipt_sha256": receipt.get("sha256"),
              "queue": args.queue, "queue_source": args.queue_source, "mix_up": args.mix_up, "layer_rows": {k: int(v.sum()) for k, v in m.items()},
              "rows_changed_vs_v34": int((change != 0).sum()), "rows_changed_outside_layers": int(((change != 0) & ~touched).sum()),
              "max_abs_change_s": float(np.abs(change).max()), "rms_change_all_rows_s": float(np.sqrt(np.mean(change ** 2))),
              "mean_change_by_layer_s": {k: float(change[v].mean()) if v.any() else 0.0 for k, v in m.items()},
              "by_airport": {a: stats(change, airport == a) for a in sorted(set(airport))},
              "vs_v35": {"rows_changed": int((change35 != 0).sum()), "max_abs_change_s": float(np.abs(change35).max()), "mean_change_s": float(change35[change35 != 0].mean()) if (change35 != 0).any() else 0.0,
                         "sum_squared_change": float(np.sum(change35 ** 2)), "rows_changed_by_layer": {k: int(((change35 != 0) & v).sum()) for k, v in m.items()},
                         "by_airport": {a: stats(change35, airport == a) for a in sorted(set(airport))}},
              "independent_rebuild": {"source": export_v33.V30_DETAIL, "rows_differing": int(differing.sum()), "rows_differing_in_layers": int((differing & touched).sum()),
                                      "max_abs_integer_difference": int(np.abs(np.rint(independent).astype(np.int64) - exported).max())}}
    if report["rows_changed_outside_layers"]:
        raise ValueError("Rows outside the layers changed.")
    write_json(report_path, report)
    print({k: report[k] for k in ["layer_rows", "rows_changed_vs_v34", "max_abs_change_s", "mean_change_by_layer_s", "independent_rebuild"]}, receipt.get("sha256"))
    print("vs v35:", {k: v for k, v in report["vs_v35"].items() if k != "by_airport"})


if __name__ == "__main__":
    main()
