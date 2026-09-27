"""Export v30 + w x (LFPG EOBT_1 reset effect) on LFPG July-2026 rows (docs/input_shift/lfpg_eobt_sensitivity.md).

value = v30 (recorded float, runs/adsb_ctx/export_v30/ranking_values.parquet, rounds exactly to the uploaded v30)
        + w * (v30_reset - v30_orig), where v30_reset / v30_orig are the final-file runner's recipe outputs on the ranking file with
        LFPG July-2026 EOBT_1 := IOBT where EOBT_1 > IOBT (runs/final_lfpg_eobt/run) and unmodified (runs/final_dryrun/run),
        floored at -12 s. Both runs share code, models and machine, so their difference carries no float noise.
Refuses unless the effect is exactly zero outside LFPG July 2026 and the float v30 reproduces the uploaded integers. The user
approved uploading the 50% blend on 27 September 2026 ("yeah upload it"). Writes runs/submissions/vigorous-whistle_v<N>.parquet
and its receipt; never uploads.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import sha256_file, utc_now, write_json  # noqa: E402
from prc2026_submission.contract import export_submission  # noqa: E402

ID, TARGET = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
V30_DETAIL = "runs/adsb_ctx/export_v30/ranking_values.parquet"
V30_FILE = "runs/submissions/vigorous-whistle_v30.parquet"
ORIG = "runs/final_dryrun/run"
RESET = "runs/final_lfpg_eobt/run"
FLOOR = -12.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--weight", type=float, default=0.5)
    args = parser.parse_args()
    v30 = pd.read_parquet(ROOT / V30_DETAIL, columns=[ID, "v30", "airport", "month"])
    uploaded = pd.read_parquet(ROOT / V30_FILE)
    ids = uploaded[ID].to_numpy()
    v30 = v30.set_index(ID).reindex(ids)
    if v30["v30"].isna().any() or not np.array_equal(np.rint(v30["v30"].to_numpy()).astype(np.int64), uploaded[TARGET].to_numpy().astype(np.int64)):
        raise ValueError("The recorded float v30 does not reproduce the uploaded v30.")
    orig = pd.read_parquet(ROOT / ORIG / "recipe/detail.parquet", columns=[ID, "v30"]).set_index(ID).reindex(ids)["v30"].to_numpy(dtype=float)
    reset = pd.read_parquet(ROOT / RESET / "recipe/detail.parquet", columns=[ID, "v30"]).set_index(ID).reindex(ids)["v30"].to_numpy(dtype=float)
    effect = reset - orig
    scope = (v30["airport"].eq("LFPG") & v30["month"].eq("2026-07")).to_numpy()
    if not np.isfinite(effect).all() or np.any(effect[~scope] != 0.0):
        raise ValueError("The reset effect is not confined to LFPG July 2026.")
    value = np.maximum(v30["v30"].to_numpy() + args.weight * effect, FLOOR)
    modification = json.loads((ROOT / "runs/final_lfpg_eobt/input/modification.json").read_text(encoding="utf-8"))
    provenance = {"audit_used": False, "experiment_id": "lfpg_eobt_blend_v1", "selected_candidate": f"lfpg_eobt_reset_w{int(round(args.weight * 100)):03d}",
                  "base": {"file": V30_FILE, "sha256": sha256_file(ROOT / V30_FILE), "float_detail": V30_DETAIL, "float_detail_sha256": sha256_file(ROOT / V30_DETAIL)},
                  "effect_runs": {"original": ORIG, "original_recipe_receipt_sha256": sha256_file(ROOT / ORIG / "stages/recipe.json"),
                                  "reset": RESET, "reset_recipe_receipt_sha256": sha256_file(ROOT / RESET / "stages/recipe.json"),
                                  "reset_input_sha256": modification["sha256_input"], "edited_rows": modification["rows_changed"]},
                  "weight": args.weight, "floor_seconds": FLOOR, "scope": "LFPG departures in 2026-07 (effect exactly zero elsewhere)",
                  "evidence": "docs/input_shift/lfpg_eobt_sensitivity.md", "approval": "user, 27 Sep 2026: 'yeah upload it' (50% blend)"}
    output = ROOT / "runs/submissions" / f"vigorous-whistle_v{args.version}.parquet"
    receipt = export_submission(pd.DataFrame({ID: ids, "prediction": value}), ROOT / "data/raw/submitting.parquet", output, provenance)
    written = pd.read_parquet(output)
    delta = written[TARGET].to_numpy().astype(np.int64) - uploaded[TARGET].to_numpy().astype(np.int64)
    summary = {"created_at_utc": utc_now(), "file": output.relative_to(ROOT).as_posix(), "sha256": sha256_file(output), "rows": int(len(written)),
               "same_order_as_v30": bool(np.array_equal(written[ID].to_numpy(), ids)), "rows_in_scope": int(scope.sum()),
               "rows_changed_vs_v30": int((delta != 0).sum()), "rows_changed_outside_scope": int((delta[~scope] != 0).sum()),
               "delta_vs_v30_in_scope": {"mean": float(delta[scope].mean()), "rms": float(np.sqrt(np.mean(delta[scope] ** 2))), "min": int(delta[scope].min()), "max": int(delta[scope].max())},
               "minimum_prediction": int(written[TARGET].min()), "receipt": receipt}
    if summary["rows_changed_outside_scope"]:
        raise ValueError("Rows outside LFPG July changed.")
    write_json(ROOT / "docs/input_shift" / f"export_v{args.version}.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "receipt"}, indent=1))


if __name__ == "__main__":
    main()
