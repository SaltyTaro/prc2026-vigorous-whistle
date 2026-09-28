"""Acceptance of the final-file runner's v31 -> v35 stages on a copy of the ranking file (docs/final_file/runner.md, "v35 layers").

Checks, all read-only outside the run root:
- repair: the detector flags exactly LFPG 2026-07, the edit is the 7,510 rows of runs/final_lfpg_eobt/input/modification.json and the
  effect is non-zero only on LFPG July rows;
- rematch: the run's output equals the frozen ranking rematch (runs/adsb_rematch/out_arcw1.parquet) row for row;
- layers: the runner's masks equal tools/export_stack.layers' masks (the frozen tables) row for row, and the per-layer counts equal
  docs/disagreement/export_v35.json layer_rows (u300 758, callsign 163, queue 161);
- floats: tools/export_stack.layers applied to the runner's own v34 on the frozen tables must equal the runner's v35 exactly (same
  arithmetic, different plumbing); against the shipped construction (rebuild_v34 from the recorded v30 detail) the difference is the
  dry run's v30 float noise, reported;
- integers: the final file against the uploaded v35 (SHA-256 75f4a4f4...): rows differing, maximum |delta|, by layer and airport.
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

V35_FILE, V35_SHA = "runs/submissions/vigorous-whistle_v35.parquet", "75f4a4f4678921d957909ef364c2cae66a81e5b74bbb680ded50d329ce6ee4f5"
V35_RECORD = "docs/disagreement/export_v35.json"
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("runs/final_dryrun/run"))
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = ROOT / args.root
    if sha256_file(ROOT / V35_FILE) != V35_SHA:
        raise ValueError("The uploaded v35 file changed.")
    import export_stack
    from export_disagreement_gate import rebuild_v34
    from prc2026_final.runner import STAGES
    from prc2026_final.common import Run
    run = Run(root)
    for stage in STAGES:
        run.verify_outputs(stage)
    report = {"created_at_utc": utc_now(), "root": args.root.as_posix(), "receipts": {s: sha256_file(run.receipt_path(s)) for s in STAGES}}

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

    # rematch
    ours = pd.read_parquet(root / "rematch/out.parquet")
    theirs = pd.read_parquet(ROOT / REMATCH_FROZEN)
    report["rematch"] = {"exact": frame_equal(ours, theirs, 0.0), "within_1e-6": frame_equal(ours, theirs, 1e-6)}
    report["rematch"]["pass"] = bool(report["rematch"]["within_1e-6"]["same_ids"] and not report["rematch"]["within_1e-6"]["columns_differing"])

    # layers and floats against the frozen construction
    rec, rids, dry, lfpg, uploaded, v34_shipped, at34 = rebuild_v34()
    if not np.array_equal(rids, ids):
        raise ValueError("The stack detail is not in the recorded v30 detail's order.")
    frozen_on_ours, fm = export_stack.layers(stack["v34"].to_numpy(), dry, at34, ids, 0.75, "regular")
    shipped, sm = export_stack.layers(v34_shipped, dry, at34, ids, 0.75, "regular")
    ours_masks = {"u300": stack["layer_gate"].to_numpy(), "callsign": stack["layer_callsign"].to_numpy(), "queue": stack["layer_queue"].to_numpy()}
    recorded_rows = read_json(ROOT / V35_RECORD)["layer_rows"]
    v35 = stack["v35"].to_numpy()
    report["layers"] = {k: {"rows": int(m.sum()), "recorded": recorded_rows[k], "rows_differing_from_export_stack_masks": int((m != fm[k]).sum())} for k, m in ours_masks.items()}
    report["floats"] = {"export_stack_layers_on_runner_v34_max_abs": float(np.max(np.abs(frozen_on_ours - v35))),
                        "runner_v34_minus_shipped_v34": {"rms": float(np.sqrt(np.mean((stack["v34"].to_numpy() - v34_shipped) ** 2))),
                                                         "max_abs": float(np.max(np.abs(stack["v34"].to_numpy() - v34_shipped)))},
                        "runner_v35_minus_shipped_v35": {"rms": float(np.sqrt(np.mean((v35 - shipped) ** 2))), "max_abs": float(np.max(np.abs(v35 - shipped))),
                                                         "p99_abs": float(np.quantile(np.abs(v35 - shipped), 0.99))},
                        "repair_effect_minus_export_lfpg_effect_max_abs": float(np.max(np.abs(0.5 * effect - lfpg)))}
    # the whole float difference is the dry run's v30 noise (runner v30 - recorded v30), carried with weight 1, then x0.5 on callsign rows
    # and x0.25 on queue rows; rows at the -12 floor carry none
    noise = stack["v30"].to_numpy() - rec["v30"].to_numpy(dtype=float)
    carried = noise * np.where(ours_masks["callsign"], 0.5, 1.0) * np.where(ours_masks["queue"], 0.25, 1.0)
    unfloored = (v35 > -12.0) & (shipped > -12.0) & (stack["v34"].to_numpy() > -12.0) & (v34_shipped > -12.0)
    report["floats"]["noise_identity"] = {"rows": int(unfloored.sum()), "max_abs": float(np.max(np.abs((v35 - shipped) - carried)[unfloored])),
                                          "v30_noise_rms": float(np.sqrt(np.mean(noise ** 2))), "v30_noise_max_abs": float(np.max(np.abs(noise)))}
    report["layers_pass"] = bool(all(v["rows"] == v["recorded"] and v["rows_differing_from_export_stack_masks"] == 0 for v in report["layers"].values())
                                 and report["floats"]["export_stack_layers_on_runner_v34_max_abs"] == 0.0)

    # integers
    final = pd.read_parquet(root / f"final/vigorous-whistle_v{args.version}.parquet").set_index(ID)[TARGET].reindex(ids).to_numpy().astype(np.int64)
    upload = pd.read_parquet(ROOT / V35_FILE).set_index(ID)[TARGET].reindex(ids).to_numpy().astype(np.int64)
    if not np.array_equal(np.rint(shipped).astype(np.int64), upload):
        raise ValueError("The shipped construction does not round to the uploaded v35.")
    delta = final - upload
    differ = delta != 0
    v30_dry = pd.read_parquet(ROOT / V30_DRYRUN_FILE).set_index(ID)[TARGET].reindex(ids).to_numpy().astype(np.int64)
    v30_up = pd.read_parquet(ROOT / V30_FILE).set_index(ID)[TARGET].reindex(ids).to_numpy().astype(np.int64)
    v30_flip = v30_dry != v30_up
    airport = stack["airport"].to_numpy()
    any_layer = ours_masks["u300"] | ours_masks["callsign"] | ours_masks["queue"]
    report["integers"] = {"final_file": f"final/vigorous-whistle_v{args.version}.parquet", "final_sha256": sha256_file(root / f"final/vigorous-whistle_v{args.version}.parquet"),
                          "uploaded_v35_sha256": V35_SHA, "identical_file": sha256_file(root / f"final/vigorous-whistle_v{args.version}.parquet") == V35_SHA,
                          "rows_differing": int(differ.sum()), "max_abs_difference_s": int(np.abs(delta).max()),
                          "differences_histogram": {str(k): int(v) for k, v in zip(*np.unique(delta[differ], return_counts=True))},
                          "rows_differing_in_layers": {k: int((differ & m).sum()) for k, m in ours_masks.items()},
                          "rows_differing_lfpg_july": int((differ & lfpg_july).sum()), "rows_differing_outside_layers": int((differ & ~any_layer).sum()),
                          "rows_differing_by_airport": {a: int((differ & (airport == a)).sum()) for a in sorted(set(airport))},
                          "v30_dryrun_flip_rows": int(v30_flip.sum()), "rows_differing_that_also_flip_in_the_v30_dry_run": int((differ & v30_flip).sum())}
    write_json(ROOT / args.output, report)
    print({k: report[k] for k in ["repair", "layers", "floats", "layers_pass"]})
    print(report["rematch"])
    print(report["integers"])


if __name__ == "__main__":
    main()
