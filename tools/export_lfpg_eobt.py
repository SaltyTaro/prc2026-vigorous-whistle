"""LFPG July-2026 EOBT_1 repair candidate (27 September 2026): v30 plus a weight w of the measured effect of resetting the changed EOBT_1.

In July 2026 LFPG's EOBT_1 changed meaning (docs/input_shift/: EOBT_1 > IOBT on 35% of departures against 6-7% in every 2025 month and
January 2026, values at IOBT+300/+600 s, within a median 120 s of AOBT_3). The sensitivity run (docs/input_shift/lfpg_eobt_sensitivity.md)
rebuilt v30 with EOBT_1 := IOBT on those 7,510 departures through the final-file runner; against the unmodified dry run (same code,
models and machine) only LFPG July rows move. This export adds w x (repaired - dry run) to the shipped v30 values (the recorded v30 detail),
floors at -12 s and writes the submission contract. It refuses unless every row outside LFPG July 2026 has a zero effect and the
recorded v30 reproduces the uploaded v30 integers. The output goes to --output (not runs/submissions unless --version is given).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET  # noqa: E402
from prc2026_submission.contract import export_submission  # noqa: E402

V30_DETAIL = "runs/adsb_ctx/export_v30/ranking_values.parquet"
V30_FILE = "runs/submissions/vigorous-whistle_v30.parquet"
V30_SHA = "3a19fa27f5f1d4a5bed4dfa8e128bdbd58054687bb4a43ee2281f61f6b7765dc"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight", type=float, required=True)
    parser.add_argument("--repaired", type=Path, default=Path("runs/final_lfpg_eobt/run/recipe/detail.parquet"))
    parser.add_argument("--dryrun", type=Path, default=Path("runs/final_dryrun/run/recipe/detail.parquet"))
    parser.add_argument("--version", type=int, default=None, help="write runs/submissions/vigorous-whistle_v<N>.parquet (only after approval)")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.weight <= 1:
        raise ValueError("Weight must be in (0, 1].")
    if sha256_file(ROOT / V30_FILE) != V30_SHA:
        raise ValueError("The uploaded v30 file changed.")
    detail = pd.read_parquet(ROOT / V30_DETAIL, columns=[ID, "airport", "month", "v30"])
    ids = detail[ID].to_numpy()
    uploaded = pd.read_parquet(ROOT / V30_FILE).set_index(ID)[TARGET].reindex(ids).to_numpy()
    v30 = detail["v30"].to_numpy(dtype=float)
    if not np.array_equal(np.rint(v30).astype(np.int64), uploaded.astype(np.int64)):
        raise ValueError("The recorded v30 detail does not reproduce the uploaded v30 integers.")
    repaired = pd.read_parquet(ROOT / args.repaired, columns=[ID, "v30"]).set_index(ID)["v30"].reindex(ids).to_numpy(dtype=float)
    dry = pd.read_parquet(ROOT / args.dryrun, columns=[ID, "v30"]).set_index(ID)["v30"].reindex(ids).to_numpy(dtype=float)
    effect = repaired - dry
    lfpg_july = detail["airport"].eq("LFPG").to_numpy() & detail["month"].eq("2026-07").to_numpy()
    if not np.isfinite(effect).all() or float(np.max(np.abs(effect[~lfpg_july]))) != 0.0:
        raise ValueError("The repair effect is not confined to LFPG July 2026.")
    values = np.maximum(v30 + args.weight * effect, -12.0)
    output = ROOT / (f"runs/submissions/vigorous-whistle_v{args.version}.parquet" if args.version else args.output)
    if output.exists():
        raise ValueError(f"{output} exists; never overwrite.")
    output.parent.mkdir(parents=True, exist_ok=True)
    provenance = {"recipe": f"v30 + {args.weight} x (EOBT_1-repaired rebuild - unmodified rebuild) on LFPG July 2026 rows",
                  "incumbent": {"submission": "vigorous-whistle_v30.parquet", "official_rmse_seconds": 229.6009, "sha256": V30_SHA},
                  "repair": {"input_modification": read_json(ROOT / "runs/final_lfpg_eobt/input/modification.json"),
                             "repaired_detail_sha256": sha256_file(ROOT / args.repaired), "dryrun_detail_sha256": sha256_file(ROOT / args.dryrun)},
                  "weight": args.weight, "selected_by": "user decision on a 2026-only input regime the 2025 development folds cannot score",
                  "evidence": ["docs/input_shift/README.md", "docs/input_shift/lfpg_eobt_sensitivity.md"], "audit_used": False, "floor_seconds": -12.0}
    receipt = export_submission(pd.DataFrame({ID: ids, "prediction": values}), ROOT / "data/raw/submitting.parquet", output, provenance)
    exported = pd.read_parquet(output).set_index(ID)[TARGET].reindex(ids).to_numpy()
    delta = exported.astype(float) - uploaded.astype(float)
    changed = delta != 0
    report = {"created_at_utc": utc_now(), "output": output.relative_to(ROOT).as_posix(), "receipt": receipt, "weight": args.weight,
              "rows_changed_vs_v30": int(changed.sum()), "rows_changed_outside_lfpg_july": int((changed & ~lfpg_july).sum()),
              "lfpg_july_rows": int(lfpg_july.sum()), "mean_change_lfpg_july_s": float(delta[lfpg_july].mean()),
              "rms_change_lfpg_july_s": float(np.sqrt(np.mean(delta[lfpg_july] ** 2))), "max_abs_change_s": float(np.abs(delta).max()),
              "effect_sum_sq_weighted": float(np.sum((args.weight * effect) ** 2))}
    if report["rows_changed_outside_lfpg_july"]:
        raise ValueError("Rows outside LFPG July changed.")
    write_json(ROOT / args.report, report)
    print({k: v for k, v in report.items() if k != "receipt"}, receipt.get("sha256"))


if __name__ == "__main__":
    main()
