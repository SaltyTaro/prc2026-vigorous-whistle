"""Acceptance of the final-file runner's v31 -> v35 / v36 / v37 / v38 stages on a copy of the ranking file (docs/final_file/runner.md,
"v31–v38 layers"). --layers picks the layer set; its final file (stages/final-<layers>.json) is compared with the shipped export.

Checks, all read-only outside the run root:
- repair: the detector flags exactly LFPG 2026-07, the edit is the 7,510 rows of runs/final_lfpg_eobt/input/modification.json and the
  effect is non-zero only on LFPG July rows;
- rematch: the run's output equals the frozen ranking rematch (runs/adsb_rematch/out_arcw1.parquet) row for row; peers: the run's
  tools/peer_congestion.peer_table equals ranking_peers() exactly;
- layers: the runner's masks equal tools/export_stack.layers' masks on the frozen tables row for row, and the per-layer counts equal the
  shipped export's layer_rows (docs/disagreement/export_v3N.json); masks against the shipped construction are reported too (the
  stretch's value > 1,500 s test can flip on a row within float noise of the threshold);
- floats: export_stack.layers applied to the runner's own v34 on the frozen tables must equal the runner's values exactly (same
  arithmetic, different plumbing); against the shipped construction (rebuild_v34 from the recorded v30 detail) the difference must be
  the dry run's v30 float noise carried through the layer weights (noise identity);
- integers: the final file against the shipped file: rows differing, maximum |delta|, by layer and airport.
  --output JSON.
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
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET  # noqa: E402

REMATCH_FROZEN = "runs/adsb_rematch/out_arcw1.parquet"
V30_DRYRUN_FILE = "runs/final_dryrun/run/export/vigorous-whistle_v900.parquet"
V30_FILE = "runs/submissions/vigorous-whistle_v30.parquet"


def frame_equal(ours: pd.DataFrame, theirs: pd.DataFrame, tol: float) -> dict:
    ours, theirs = ours.set_index(ID).sort_index(), theirs.set_index(ID).sort_index()
    out = {"rows": [int(len(ours)), int(len(theirs))], "same_ids": bool(ours.index.equals(theirs.index)), "same_columns": list(ours.columns) == list(theirs.columns),
           "columns_differing": {}}
    if not (out["same_ids"] and out["same_columns"]):
        return out
    for c in ours.columns:
        a, b = ours[c], theirs[c]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b) and not pd.api.types.is_bool_dtype(a):
            x, y = a.to_numpy(dtype=float), b.to_numpy(dtype=float)
            bad = (np.isnan(x) != np.isnan(y)) | (np.abs(np.nan_to_num(x) - np.nan_to_num(y)) > tol)
        else:
            bad = ~((a == b) | (a.isna() & b.isna())).fillna(False).to_numpy(dtype=bool)
        if bad.any():
            out["columns_differing"][c] = int(bad.sum())
    return out


SHIPPED = {"v35": {"file": "runs/submissions/vigorous-whistle_v35.parquet", "sha256": "75f4a4f4678921d957909ef364c2cae66a81e5b74bbb680ded50d329ce6ee4f5",
                   "record": "docs/disagreement/export_v35.json", "mix_up": None, "peer": False, "stretch": None, "lower": None},
           "v36": {"file": "runs/submissions/vigorous-whistle_v36.parquet", "sha256": "ed4bcd366851696e35fbb3827e37fd7333f091eea8d7f592be5a196e98c727cc",
                   "record": "docs/disagreement/export_v36.json", "mix_up": 600.0, "peer": True, "stretch": None, "lower": None},
           "v37": {"file": "runs/submissions/vigorous-whistle_v37.parquet", "sha256": "edb11dce04a61f18fb15da26f2e88c8eaf8c7a83fc0f745fada350d6f18c1d4f",
                   "record": "docs/disagreement/export_v37.json", "mix_up": None, "peer": False, "stretch": 1500.0, "lower": None},
           "v38": {"file": "runs/submissions/vigorous-whistle_v38.parquet", "sha256": "61264cdb7f0a37a8e5cf943f8e950441df38b826132cfc1097f00ca9d0519fe4",
                   "record": "docs/disagreement/export_v38.json", "mix_up": None, "peer": False, "stretch": 1500.0, "lower": 900.0}}
STRETCH_RESULTS = "docs/disagreement/tail_stretch_v1_results.json"
STRETCH2_RESULTS = "docs/disagreement/tail_stretch_v2_results.json"


def stretch_slope(pre: np.ndarray, masks: dict, t: float, lower: float | None, k1: dict, k2: dict | None) -> tuple[np.ndarray, np.ndarray]:
    """d(stretched)/d(pre) per row and a region code (0 middle, 1 above t, 2 below lower after the upper stretch, 3 both; -1 outside the
    stretch masks), from export_stack.tail_stretch's rule u = pre + k1 max(pre - t, 0), value = u + k2 max(lower - u, 0)."""
    slope, region = np.ones(len(pre)), np.full(len(pre), -1)
    for name, scope in [("stretch_untracked", "untracked"), ("stretch_tracked", "tracked")]:
        m = masks[name]
        up = pre > t
        u = pre + k1[scope] * np.maximum(pre - t, 0.0)
        low = (u < lower) if lower is not None else np.zeros(len(pre), dtype=bool)
        s = np.where(up, 1 + k1[scope], 1.0) * np.where(low, 1 - (k2[scope] if k2 else 0.0), 1.0)
        slope, region = np.where(m, s, slope), np.where(m, up.astype(int) + 2 * low.astype(int), region)
    return slope, region


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("runs/final_dryrun/run"))
    parser.add_argument("--layers", choices=list(SHIPPED), required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = ROOT / args.root
    ship = SHIPPED[args.layers]
    if sha256_file(ROOT / ship["file"]) != ship["sha256"]:
        raise ValueError(f"The shipped {args.layers} file changed.")
    import export_stack
    from export_disagreement_gate import rebuild_v34
    from peer_congestion import ranking_peers
    from prc2026_final.common import Run
    from prc2026_final.runner import STAGES
    run = Run(root)
    for stage in STAGES[:-1] + [f"final-{args.layers}"]:
        run.verify_outputs(stage)
    final_receipt = run.receipt(f"final-{args.layers}")
    if final_receipt["layers"] != args.layers or final_receipt["version"] != args.version:
        raise ValueError("The final receipt is for another layer set or version.")
    report = {"created_at_utc": utc_now(), "root": args.root.as_posix(), "layers": args.layers, "shipped": ship,
              "receipts": {s: sha256_file(run.receipt_path(s)) for s in STAGES[:-1] + [f"final-{args.layers}"]}}

    # repair
    rep = run.receipt("repair")
    modification = read_json(ROOT / "runs/final_lfpg_eobt/input/modification.json")
    stack = pd.read_parquet(root / "stack/detail.parquet")
    ids = stack[ID].to_numpy()
    lfpg_july = (stack["airport"].eq("LFPG") & stack["month"].eq("2026-07")).to_numpy()
    effect = stack["repair_effect"].to_numpy()
    report["repair"] = {"flagged": rep["flagged"], "edited_rows": rep["edited_rows"], "edited_rows_recorded": modification["rows_changed"],
                        "repair_input_sha256": rep["repair_input_sha256"], "recorded_input_sha256": modification["sha256_input"],
                        "effect_nonzero_rows": int((effect != 0).sum()), "effect_nonzero_outside_lfpg_july": int(((effect != 0) & ~lfpg_july).sum()),
                        "closest_unflagged": rep["closest_unflagged"]}
    report["repair"]["pass"] = bool(rep["flagged"] == [{"airport": "LFPG", "month": "2026-07"}] and rep["edited_rows"] == modification["rows_changed"]
                                    and rep["repair_input_sha256"] == modification["sha256_input"] and report["repair"]["effect_nonzero_outside_lfpg_july"] == 0)

    # rematch and peer tables against the frozen ranking tables
    ours = pd.read_parquet(root / "rematch/out.parquet")
    theirs = pd.read_parquet(ROOT / REMATCH_FROZEN)
    report["rematch"] = {"exact": frame_equal(ours, theirs, 0.0), "within_1e-6": frame_equal(ours, theirs, 1e-6)}
    report["rematch"]["pass"] = bool(report["rematch"]["within_1e-6"]["same_ids"] and not report["rematch"]["within_1e-6"]["columns_differing"])
    report["peers"] = {"exact": frame_equal(pd.read_parquet(root / "stack/peers.parquet"), ranking_peers(), 0.0)}
    report["peers"]["pass"] = bool(report["peers"]["exact"]["same_ids"] and not report["peers"]["exact"]["columns_differing"])

    # layers and floats against the frozen construction (export_stack.layers on the frozen ranking tables)
    rec, rids, dry, lfpg, uploaded, v34_shipped, at34 = rebuild_v34()
    if not np.array_equal(rids, ids):
        raise ValueError("The stack detail is not in the recorded v30 detail's order.")
    frozen_on_ours, fm = export_stack.layers(stack["v34"].to_numpy(), dry, at34, ids, 0.75, "regular", ship["mix_up"], ship["peer"], ship["stretch"], ship["lower"])
    shipped, sm = export_stack.layers(v34_shipped, dry, at34, ids, 0.75, "regular", ship["mix_up"], ship["peer"], ship["stretch"], ship["lower"])
    ours_masks = {k: stack[f"{args.layers}_{k}"].to_numpy() for k in fm}
    recorded_rows = read_json(ROOT / ship["record"])["layer_rows"]
    values = stack[args.layers].to_numpy()
    report["layers"] = {k: {"rows": int(m.sum()), "recorded": recorded_rows[k], "rows_differing_from_export_stack_masks": int((m != fm[k]).sum()),
                            "rows_differing_from_the_shipped_masks": int((m != sm[k]).sum())} for k, m in ours_masks.items()}
    report["floats"] = {"export_stack_layers_on_runner_v34_max_abs": float(np.max(np.abs(frozen_on_ours - values))),
                        "runner_v34_minus_shipped_v34": {"rms": float(np.sqrt(np.mean((stack["v34"].to_numpy() - v34_shipped) ** 2))),
                                                         "max_abs": float(np.max(np.abs(stack["v34"].to_numpy() - v34_shipped)))},
                        "runner_minus_shipped": {"rms": float(np.sqrt(np.mean((values - shipped) ** 2))), "max_abs": float(np.max(np.abs(values - shipped))),
                                                 "p99_abs": float(np.quantile(np.abs(values - shipped), 0.99))},
                        "repair_effect_minus_export_lfpg_effect_max_abs": float(np.max(np.abs(0.5 * effect - lfpg)))}
    # the whole float difference is the dry run's v30 noise (runner v30 - recorded v30), carried with weight 1 (the mix gate, u300 and the
    # peer lifts are additive), x0.5 on callsign rows, x0.25 on queue rows and x the stretch's slope on stretched rows; rows at the -12
    # floor carry none, and rows whose stretch region differs between the two constructions (within the noise of a threshold) are excluded
    noise = stack["v30"].to_numpy() - rec["v30"].to_numpy(dtype=float)
    carried = noise * np.where(ours_masks["callsign"], 0.5, 1.0) * np.where(ours_masks["queue"], 0.25, 1.0)
    comparable = np.ones(len(ids), dtype=bool)
    if ship["stretch"] is not None:
        k1 = read_json(ROOT / STRETCH_RESULTS)["candidates"][f"v35_t{int(ship['stretch'])}"]["k_all_months"]
        k2 = None
        if ship["lower"] is not None:
            two = read_json(ROOT / STRETCH2_RESULTS)["candidates"][f"two_sided_{int(ship['lower'])}"]["k_all_months"]
            k2 = {s: two[s][1] for s in two}
        pre_ship, _ = export_stack.layers(v34_shipped, dry, at34, ids, 0.75, "regular")  # the stretch acts on the v35 stack
        slope, region = stretch_slope(stack["v35"].to_numpy(), ours_masks, ship["stretch"], ship["lower"], k1, k2)
        _, region_ship = stretch_slope(pre_ship, sm, ship["stretch"], ship["lower"], k1, k2)
        carried = carried * slope
        comparable = region == region_ship
    unfloored = (values > -12.0) & (shipped > -12.0) & (stack["v34"].to_numpy() > -12.0) & (v34_shipped > -12.0) & comparable
    report["floats"]["noise_identity"] = {"rows": int(unfloored.sum()), "rows_excluded_stretch_boundary": int((~comparable).sum()),
                                          "max_abs": float(np.max(np.abs((values - shipped) - carried)[unfloored])),
                                          "v30_noise_rms": float(np.sqrt(np.mean(noise ** 2))), "v30_noise_max_abs": float(np.max(np.abs(noise)))}
    report["layers_pass"] = bool(all(v["rows"] == v["recorded"] and v["rows_differing_from_export_stack_masks"] == 0 for v in report["layers"].values())
                                 and report["floats"]["export_stack_layers_on_runner_v34_max_abs"] == 0.0)

    # integers
    name = f"final/vigorous-whistle_v{args.version}.parquet"
    final = pd.read_parquet(root / name).set_index(ID)[TARGET].reindex(ids).to_numpy().astype(np.int64)
    upload = pd.read_parquet(ROOT / ship["file"]).set_index(ID)[TARGET].reindex(ids).to_numpy().astype(np.int64)
    if not np.array_equal(np.rint(shipped).astype(np.int64), upload):
        raise ValueError(f"The shipped construction does not round to {ship['file']}.")
    delta = final - upload
    differ = delta != 0
    v30_dry = pd.read_parquet(ROOT / V30_DRYRUN_FILE).set_index(ID)[TARGET].reindex(ids).to_numpy().astype(np.int64)
    v30_up = pd.read_parquet(ROOT / V30_FILE).set_index(ID)[TARGET].reindex(ids).to_numpy().astype(np.int64)
    v30_flip = v30_dry != v30_up
    airport = stack["airport"].to_numpy()
    any_layer = np.logical_or.reduce(list(ours_masks.values()))
    report["integers"] = {"final_file": name, "final_sha256": sha256_file(root / name), "shipped_sha256": ship["sha256"],
                          "identical_file": sha256_file(root / name) == ship["sha256"],
                          "rows_differing": int(differ.sum()), "max_abs_difference_s": int(np.abs(delta).max()),
                          "differences_histogram": {str(k): int(v) for k, v in zip(*np.unique(delta[differ], return_counts=True))},
                          "rows_differing_in_layers": {k: int((differ & m).sum()) for k, m in ours_masks.items()},
                          "rows_differing_lfpg_july": int((differ & lfpg_july).sum()), "rows_differing_outside_layers": int((differ & ~any_layer).sum()),
                          "rows_differing_by_airport": {a: int((differ & (airport == a)).sum()) for a in sorted(set(airport))},
                          "v30_dryrun_flip_rows": int(v30_flip.sum()), "rows_differing_that_also_flip_in_the_v30_dry_run": int((differ & v30_flip).sum())}
    write_json(ROOT / args.output, report)
    print({k: report[k] for k in ["layers", "floats", "layers_pass"]})
    print({k: report[k]["pass"] for k in ["repair", "rematch", "peers"]})
    print(report["integers"])


if __name__ == "__main__":
    main()
