"""v32 (27 September 2026): v31 with the LIRF non-population UNMATCHED blend changed from lctx_b100_up (0.40 v9 + 0.35 ens_ctx_lirf +
0.25 ens_v22_lirf_q) to u50_00_50 (0.50 v9 + 0.50 ens_ctx_lirf), floor -12, on the 17,718 ranking rows of that scope only.

The LIRF round (configs/lirf_unmatched_blend_v1.json, docs/lirf_blend/) selected nothing under the unchanged rule: u50_00_50 improved
every recipe month (mean -0.023%) with day intervals below zero on 3 of 4 months (December [-0.087, +0.030] s). The user chose to ship it
("ship it", 27 Sep 2026) as a recorded post-hoc relaxation. v31 = the recorded v30 float detail + 0.5 x the LFPG July EOBT_1 repair effect
(tools/export_lfpg_eobt.py); this tool rebuilds v31's floats, checks they round to the uploaded v31 integers, checks the lctx_b100_up
formula reproduces v30 on the scope rows exactly, then swaps the blend there. Every other row must equal v31.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET  # noqa: E402
from prc2026_submission.contract import export_submission  # noqa: E402

V31_FILE, V31_SHA = "runs/submissions/vigorous-whistle_v31.parquet", "ea1c17568e49b83cb1c8667212c2097ab8a8dec774b4db57e01b9b79bb531be2"
SCOPE = "lirf_nonpop_unmatched"


def main():
    if sha256_file(ROOT / V31_FILE) != V31_SHA:
        raise ValueError("The uploaded v31 file changed.")
    d = pd.read_parquet(ROOT / "runs/adsb_ctx/export_v30/ranking_values.parquet")
    ids = d[ID].to_numpy()
    repaired = pd.read_parquet(ROOT / "runs/final_lfpg_eobt/run/recipe/detail.parquet", columns=[ID, "v30"]).set_index(ID)["v30"].reindex(ids).to_numpy(dtype=float)
    dry = pd.read_parquet(ROOT / "runs/final_dryrun/run/recipe/detail.parquet", columns=[ID, "v30"]).set_index(ID)["v30"].reindex(ids).to_numpy(dtype=float)
    v31 = np.maximum(d["v30"].to_numpy(dtype=float) + 0.5 * (repaired - dry), -12.0)
    uploaded31 = pd.read_parquet(ROOT / V31_FILE).set_index(ID)[TARGET].reindex(ids).to_numpy()
    if not np.array_equal(np.rint(v31).astype(np.int64), uploaded31.astype(np.int64)):
        raise ValueError("Rebuilt v31 floats do not round to the uploaded v31.")
    scope = d["scope"].eq(SCOPE).to_numpy()
    v9, net, tree = (d[c].to_numpy(dtype=float) for c in ["v9", "ens_ctx_lirf", "ens_v22_lirf_q"])
    old = np.maximum(0.40 * v9 + 0.35 * net + 0.25 * tree, -12.0)
    if float(np.max(np.abs(old[scope] - d["v30"].to_numpy(dtype=float)[scope]))) > 1e-9 or float(np.max(np.abs(v31[scope] - old[scope]))) > 1e-9:
        raise ValueError("lctx_b100_up does not reproduce v30/v31 on the unmatched LIRF scope.")
    v32 = v31.copy()
    v32[scope] = np.maximum(0.50 * v9[scope] + 0.50 * net[scope], -12.0)
    output = ROOT / "runs/submissions/vigorous-whistle_v32.parquet"
    if output.exists():
        raise ValueError("v32 exists; never overwrite.")
    provenance = {"experiment_id": "lirf_unmatched_blend_v1", "selected_candidate": "u50_00_50 (user override)",
                  "recipe": "v31 with the LIRF non-population unmatched blend u50_00_50 (0.50 v9 + 0.50 ens_ctx_lirf) instead of lctx_b100_up",
                  "incumbent": {"submission": "vigorous-whistle_v31.parquet", "official_rmse_seconds": 229.5832, "sha256": V31_SHA},
                  "declaration": "configs/lirf_unmatched_blend_v1.json", "declaration_sha256": sha256_file(ROOT / "configs/lirf_unmatched_blend_v1.json"),
                  "development_results": "docs/lirf_blend/development_results.json",
                  "selected_by": "user override of the unchanged rule ('ship it', 27 Sep 2026): every month better, day intervals below zero on 3/4",
                  "audit_used": False, "floor_seconds": -12.0}
    receipt = export_submission(pd.DataFrame({ID: ids, "prediction": v32}), ROOT / "data/raw/submitting.parquet", output, provenance)
    exported = pd.read_parquet(output).set_index(ID)[TARGET].reindex(ids).to_numpy()
    changed = exported != uploaded31
    report = {"created_at_utc": utc_now(), "output": output.relative_to(ROOT).as_posix(), "receipt": receipt, "scope_rows": int(scope.sum()),
              "rows_changed_vs_v31": int(changed.sum()), "rows_changed_outside_scope": int((changed & ~scope).sum()),
              "scope_mean_change_s": float((exported - uploaded31)[scope].mean()), "scope_rms_change_s": float(np.sqrt(np.mean(((exported - uploaded31)[scope]) ** 2))),
              "max_abs_change_s": float(np.abs(exported - uploaded31).max()),
              "scope_rows_by_month": d.loc[scope, "month"].value_counts().to_dict()}
    if report["rows_changed_outside_scope"]:
        raise ValueError("Rows outside the scope changed.")
    write_json(ROOT / "docs/lirf_blend/export_v32.json", report)
    print({k: v for k, v in report.items() if k != "receipt"}, receipt.get("sha256"))


if __name__ == "__main__":
    main()
