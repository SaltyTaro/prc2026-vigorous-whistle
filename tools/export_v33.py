"""v33 (27 September 2026): v31 with the outer-weight round's selection ltfm100_m60 (configs/outer_weight_scope_v1.json).

v30's recipe gives the committee mix weight 0.7 against v9 + the residual stage on every non-LIRF row. The selection sets it to 1.0 on LTFM
rows and 0.6 on the other ADS-B-matched non-LIRF rows; unmatched non-LTFM rows stay 0.7 and every LIRF row is unchanged. Per the declared
ship clause the recipe is rebuilt on the final-file runner's dry-run detail (runs/final_dryrun/run/recipe/detail.parquet), the rebuild at 0.7
must equal the dry run's v30, and the delta (selected weights minus 0.7) is added to v31's floats (recorded v30 detail + 0.5 x the LFPG
July EOBT_1 effect), floored at -12 s. The rebuilt v31 floats must round to the uploaded v31 integers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET  # noqa: E402
from prc2026_submission.contract import export_submission  # noqa: E402

DECLARATION = "configs/outer_weight_scope_v1.json"
V31_FILE, V31_SHA = "runs/submissions/vigorous-whistle_v31.parquet", "ea1c17568e49b83cb1c8667212c2097ab8a8dec774b4db57e01b9b79bb531be2"
DRY = "runs/final_dryrun/run/recipe/detail.parquet"
REPAIRED = "runs/final_lfpg_eobt/run/recipe/detail.parquet"
V30_DETAIL = "runs/adsb_ctx/export_v30/ranking_values.parquet"


def recipe(detail: pd.DataFrame, w_ltfm: float, w_matched: float, w_unmatched: float) -> np.ndarray:
    """v30's non-LIRF recipe with scope weights; LIRF rows keep the detail's v30."""
    airport = detail["airport"].to_numpy()
    matched = detail["scope"].eq("matched_no_lirf").to_numpy()
    base = detail["v9"].to_numpy(dtype=float) + np.where(matched, detail["residual"].to_numpy(dtype=float), 0.0)
    mix = detail["committee_mix"].to_numpy(dtype=float)
    w = np.where(airport == "LTFM", w_ltfm, np.where(matched, w_matched, w_unmatched))
    return np.where(airport != "LIRF", np.maximum((1 - w) * base + w * mix, -12.0), detail["v30"].to_numpy(dtype=float))


def main():
    decl = read_json(ROOT / DECLARATION)
    selection = read_json(ROOT / "runs/outer_weight/selection.json")
    if selection["declaration_sha256"] != object_hash(decl) or selection["selected_candidate"] != "ltfm100_m60":
        raise ValueError("The round did not select ltfm100_m60 under this declaration.")
    spec = decl["candidates"]["ltfm100_m60"]
    if sha256_file(ROOT / V31_FILE) != V31_SHA:
        raise ValueError("The uploaded v31 file changed.")
    rec = pd.read_parquet(ROOT / V30_DETAIL)
    ids = rec[ID].to_numpy()
    dry = pd.read_parquet(ROOT / DRY).set_index(ID).reindex(ids).reset_index()
    repaired = pd.read_parquet(ROOT / REPAIRED, columns=[ID, "v30"]).set_index(ID)["v30"].reindex(ids).to_numpy(dtype=float)
    if dry["v30"].isna().any() or np.isnan(repaired).any() or not (dry["airport"].to_numpy() == rec["airport"].to_numpy()).all():
        raise ValueError("The dry-run / repaired details do not align with the v30 detail.")
    v31 = np.maximum(rec["v30"].to_numpy(dtype=float) + 0.5 * (repaired - dry["v30"].to_numpy(dtype=float)), -12.0)
    uploaded31 = pd.read_parquet(ROOT / V31_FILE).set_index(ID)[TARGET].reindex(ids).to_numpy()
    if not np.array_equal(np.rint(v31).astype(np.int64), uploaded31.astype(np.int64)):
        raise ValueError("Rebuilt v31 floats do not round to the uploaded v31.")
    control = recipe(dry, 0.7, 0.7, 0.7)
    control_check = float(np.max(np.abs(control - dry["v30"].to_numpy(dtype=float))))
    if control_check > 1e-9:
        raise ValueError(f"The rebuild at 0.7 does not equal the dry run's v30 (max {control_check}).")
    delta = recipe(dry, spec["ltfm"], spec["matched_no_lirf_other"], spec["unmatched_no_lirf_other"]) - control
    airport = rec["airport"].to_numpy()
    matched = rec["scope"].eq("matched_no_lirf").to_numpy()
    in_scope = (airport == "LTFM") | ((airport != "LIRF") & matched)
    if np.any(np.abs(delta[~in_scope]) > 0):
        raise ValueError("The delta touches rows outside LTFM and the matched non-LIRF scope.")
    v33 = np.maximum(v31 + delta, -12.0)
    output = ROOT / "runs/submissions/vigorous-whistle_v33.parquet"
    if output.exists():
        raise ValueError("v33 exists; never overwrite.")
    provenance = {"experiment_id": decl["experiment_id"], "selected_candidate": "ltfm100_m60",
                  "recipe": "v31 with the outer weight of the committee mix set to 1.0 on LTFM rows and 0.6 on other ADS-B-matched non-LIRF rows (0.7 elsewhere); delta from the dry-run detail added to v31's floats",
                  "incumbent": {"submission": "vigorous-whistle_v31.parquet", "official_rmse_seconds": 229.5832, "sha256": V31_SHA},
                  "declaration": DECLARATION, "declaration_sha256": sha256_file(ROOT / DECLARATION),
                  "selection": "runs/outer_weight/selection.json", "selection_sha256": sha256_file(ROOT / "runs/outer_weight/selection.json"),
                  "development_results": "docs/outer_weight/development_results.json",
                  "selected_by": "the unchanged rule: every recipe month better, mean -0.264%, paired-day intervals below zero on 4/4 months",
                  "inputs": {DRY: sha256_file(ROOT / DRY), REPAIRED: sha256_file(ROOT / REPAIRED), V30_DETAIL: sha256_file(ROOT / V30_DETAIL)},
                  "audit_used": False, "floor_seconds": -12.0}
    receipt = export_submission(pd.DataFrame({ID: ids, "prediction": v33}), ROOT / "data/raw/submitting.parquet", output, provenance)
    exported = pd.read_parquet(output).set_index(ID)[TARGET].reindex(ids).to_numpy()
    change = exported.astype(float) - uploaded31.astype(float)
    changed = change != 0
    groups = {"LTFM": airport == "LTFM", "matched_non_lirf_other": (airport != "LIRF") & (airport != "LTFM") & matched}
    report = {"created_at_utc": utc_now(), "output": output.relative_to(ROOT).as_posix(), "receipt": receipt,
              "control_rebuild_max_abs": control_check, "rows_changed_vs_v31": int(changed.sum()),
              "rows_changed_outside_scope": int((changed & ~in_scope).sum()), "max_abs_change_s": float(np.abs(change).max()),
              "groups": {k: {"rows": int(m.sum()), "changed": int((changed & m).sum()), "mean_change_s": float(change[m].mean()),
                             "rms_change_s": float(np.sqrt(np.mean(change[m] ** 2)))} for k, m in groups.items()},
              "by_airport_mean_change_s": {a: float(change[airport == a].mean()) for a in sorted(set(airport))}}
    if report["rows_changed_outside_scope"]:
        raise ValueError("Rows outside the scope changed.")
    write_json(ROOT / "docs/outer_weight/export_v33.json", report)
    print({k: v for k, v in report.items() if k != "receipt"}, receipt.get("sha256"))


if __name__ == "__main__":
    main()
