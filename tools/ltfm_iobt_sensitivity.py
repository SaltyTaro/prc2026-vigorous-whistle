"""LTFM IOBT sensitivity (docs/input_shift/ltfm_iobt_sensitivity.md, configs/ltfm_iobt_sensitivity_v1.json): v30's chain on the ranking
file with the 2025 early-IOBT convention restored on the 16,618 July-2026 THY LTFM departures with IOBT == SCHED
(runs/final_ltfm_iobt/run) against the unmodified dry run (runs/final_dryrun/run).

Same code, models and machine, so float noise cancels and every difference comes from the edited IOBT / EOBT_1 / LOBT values (and any
context feature that reads other flights' clocks). Per stage, per-row delta = restored - original on: edited rows (IOBT + EOBT_1 +
LOBT, or IOBT only), unedited LTFM July rows, and all other rows (must be exactly zero). The declared per-row weight q
(tools/ltfm_iobt_convention.row_weights) gives the soft restoration q * delta; its stats, the ten largest |q * delta| rows and the
per-row table runs/final_ltfm_iobt/delta.parquet are written too. Label-free; read-only apart from --output and the delta table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import sha256_file, utc_now, write_json  # noqa: E402
from lfpg_eobt_sensitivity import STAGES, stats  # noqa: E402
from ltfm_iobt_convention import ltfm, row_weights  # noqa: E402

ID = "MVT_ID_mvt"
BASE = Path("runs/final_dryrun/run")
MOD = Path("runs/final_ltfm_iobt/run")
INPUT = ROOT / "runs/final_ltfm_iobt/input/final.parquet"
DELTA = ROOT / "runs/final_ltfm_iobt/delta.parquet"
FLOOR = -12.0
BUCKETS = ["edited_iobt_eobt1_lobt", "edited_iobt_only", "ltfm_jul_unedited", "other_rows"]


def groups() -> pd.DataFrame:
    cols = [ID, "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "FLIGHT_mvt", "IOBT_flt", "EOBT_1_flt", "LOBT_flt", "AOBT_3_flt"]
    raw = pq.read_table(ROOT / "data/raw/ranking.parquet", columns=cols).to_pandas()
    new = pq.read_table(INPUT, columns=cols).to_pandas()
    dep = raw["PHASE_mvt"].eq("DEP").to_numpy()
    raw, new = raw[dep].reset_index(drop=True), new[dep].reset_index(drop=True)
    if not raw[ID].equals(new[ID]):
        raise ValueError("Row order differs between the raw and the edited file.")
    diff = {c: ~((raw[c] == new[c]).fillna(False) | (raw[c].isna() & new[c].isna())).to_numpy() for c in ["IOBT_flt", "EOBT_1_flt", "LOBT_flt"]}
    ltfm_jul = (raw["ADEP_mvt"].eq("LTFM") & raw["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m").eq("2026-07")).to_numpy()
    edited = diff["IOBT_flt"]
    bucket = np.where(~ltfm_jul, "other_rows", np.where(~edited, "ltfm_jul_unedited", np.where(diff["EOBT_1_flt"], "edited_iobt_eobt1_lobt", "edited_iobt_only")))
    q, p0 = row_weights(ltfm())
    out = pd.DataFrame({ID: raw[ID].to_numpy(), "bucket": bucket, "ltfm_jul": ltfm_jul, "edited": edited, "flight": raw["FLIGHT_mvt"].to_numpy(),
                        "sched_utc": raw["SCHED_TIME_UTC_mvt"].to_numpy(),
                        "eobt1_minus_sched_s": (raw["EOBT_1_flt"] - raw["SCHED_TIME_UTC_mvt"]).dt.total_seconds().to_numpy(),
                        "aobt3_minus_iobt_s": (raw["AOBT_3_flt"] - raw["IOBT_flt"]).dt.total_seconds().to_numpy(),
                        "mvt_minus_sched_s": (raw["MVT_TIME_UTC_mvt"] - raw["SCHED_TIME_UTC_mvt"]).dt.total_seconds().to_numpy()})
    out["q"] = np.where(edited, out["flight"].map(q).fillna(p0).astype(float), 0.0)
    out["q_seen_number"] = edited & out["flight"].isin(q.index).to_numpy()
    return out


def feature_changes(ids: np.ndarray, focus: np.ndarray) -> dict:
    out = {}
    for name in ["timeline.parquet"]:
        a_path, b_path = ROOT / BASE / "features" / name, ROOT / MOD / "features" / name
        names = [c for c in pq.ParquetFile(a_path).schema_arrow.names if c != ID]
        for start in range(0, len(names), 40):
            cols = names[start:start + 40]
            a = pd.read_parquet(a_path, columns=[ID, *cols]).set_index(ID).reindex(ids)
            b = pd.read_parquet(b_path, columns=[ID, *cols]).set_index(ID).reindex(ids)
            for c in cols:
                x, y = a[c], b[c]
                if pd.api.types.is_numeric_dtype(x):
                    xv, yv = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
                    diff = ~((xv == yv) | (np.isnan(xv) & np.isnan(yv)))
                    shift = yv[diff & focus] - xv[diff & focus]
                    med = float(np.nanmedian(shift)) if np.isfinite(shift).any() else None
                else:
                    diff = (x.astype("string").fillna("<NA>") != y.astype("string").fillna("<NA>")).to_numpy()
                    med = None
                if diff.any():
                    out[c] = {"ltfm_jul_rows": int((diff & focus).sum()), "other_rows": int((diff & ~focus).sum()), "median_change_ltfm_jul": med}
            del a, b
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("docs/input_shift/ltfm_iobt_sensitivity.json"))
    args = parser.parse_args()
    g = groups()
    ids = g[ID].to_numpy()
    masks = {k: (g["bucket"] == k).to_numpy() for k in BUCKETS}
    report = {"created_at_utc": utc_now(), "base": BASE.as_posix(), "modified": MOD.as_posix(), "declaration": "configs/ltfm_iobt_sensitivity_v1.json",
              "declaration_sha256": sha256_file(ROOT / "configs/ltfm_iobt_sensitivity_v1.json"),
              "modification": json.loads((ROOT / "runs/final_ltfm_iobt/input/modification.json").read_text(encoding="utf-8")),
              "rows": {k: int(v.sum()) for k, v in masks.items()}, "stages": {}}
    outside_max = 0.0
    v30 = mix = None
    for path, columns in STAGES.items():
        a = pd.read_parquet(ROOT / BASE / path).set_index(ID).reindex(ids)
        b = pd.read_parquet(ROOT / MOD / path).set_index(ID).reindex(ids)
        for c in columns:
            if c not in a or c not in b:
                continue
            delta = b[c].to_numpy(dtype=float) - a[c].to_numpy(dtype=float)
            both_nan = np.isnan(a[c].to_numpy(dtype=float)) & np.isnan(b[c].to_numpy(dtype=float))
            delta = np.where(both_nan, 0.0, delta)
            outside = delta[~g["ltfm_jul"].to_numpy()]
            outside_max = max(outside_max, float(np.nanmax(np.abs(outside))) if len(outside) else 0.0, float(np.isnan(outside).sum()))
            report["stages"][f"{path}:{c}"] = {"by_bucket": {k: stats(delta[m]) for k, m in masks.items()}, "ltfm_jul_all": stats(delta[g["ltfm_jul"].to_numpy()])}
            if path == "recipe/detail.parquet" and c == "v30":
                v30 = (a[c].to_numpy(dtype=float), delta)
            if path == "recipe/detail.parquet" and c == "committee_mix":  # v33 on LTFM rows = max(committee mix, floor) (outer weight 1.0)
                mix = (a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float))
        del a, b
    report["max_abs_delta_outside_ltfm_july_all_stages"] = outside_max
    if outside_max != 0.0:
        raise ValueError(f"The restoration moved a row outside LTFM July 2026 (max |delta| {outside_max}).")
    orig, delta = v30
    g["v30_orig"], g["delta_v30"] = orig, delta
    g["q_delta"] = g["q"] * g["delta_v30"]
    g["v33_orig"] = np.maximum(mix[0], FLOOR)
    g["delta_v33"] = np.maximum(mix[1], FLOOR) - g["v33_orig"]
    g["q_delta_v33"] = g["q"] * g["delta_v33"]
    e = g["edited"].to_numpy()
    report["soft_restoration_q_delta"] = {
        "edited_rows": stats(g.loc[e, "q_delta"].to_numpy()), "q": stats(g.loc[e, "q"].to_numpy()),
        "v33_ltfm_mix_only": {"note": "v33 sets the outer weight to 1.0 on LTFM, so its LTFM value is max(committee mix, -12); delta_v33 is that value's change",
                              "delta_edited": stats(g.loc[e, "delta_v33"].to_numpy()), "q_delta_edited": stats(g.loc[e, "q_delta_v33"].to_numpy()),
                              "delta_unedited_ltfm_july": stats(g.loc[g["bucket"].eq("ltfm_jul_unedited").to_numpy(), "delta_v33"].to_numpy()),
                              "sum_sq_q_delta": float((g.loc[e, "q_delta_v33"] ** 2).sum()), "sum_sq_delta": float((g.loc[e, "delta_v33"] ** 2).sum())},
        "seen_number_rows": stats(g.loc[e & g["q_seen_number"].to_numpy(), "q_delta"].to_numpy()),
        "unseen_number_rows_q_p0": stats(g.loc[e & ~g["q_seen_number"].to_numpy(), "q_delta"].to_numpy()),
        "sum_sq_q_delta": float((g.loc[e, "q_delta"] ** 2).sum()), "sum_sq_delta": float((g.loc[e, "delta_v30"] ** 2).sum()),
        "delta_by_original_aobt3_minus_iobt": {str(k): stats(s.to_numpy()) for k, s in g.loc[e].groupby(pd.cut(g.loc[e, "aobt3_minus_iobt_s"], [-1e9, -600, -300, 0, 300, 600, 900, 1800, 1e9]), observed=True)["delta_v30"]},
        "top10_abs_q_delta": g.loc[e].assign(abs_qd=g.loc[e, "q_delta"].abs()).nlargest(10, "abs_qd")[
            [ID, "flight", "sched_utc", "q", "delta_v30", "q_delta", "v30_orig", "eobt1_minus_sched_s", "aobt3_minus_iobt_s", "mvt_minus_sched_s"]].astype({"sched_utc": str}).to_dict("records")}
    ea = pd.read_parquet(ROOT / BASE / "export/vigorous-whistle_v900.parquet").set_index(ID).reindex(ids)
    eb = pd.read_parquet(ROOT / MOD / "export/vigorous-whistle_v902.parquet").set_index(ID).reindex(ids)
    di = eb["TAXITIME_SEC_mvt"].to_numpy(dtype=float) - ea["TAXITIME_SEC_mvt"].to_numpy(dtype=float)
    report["export_integers"] = {k: stats(di[m]) for k, m in masks.items()}
    report["feature_columns_changed"] = feature_changes(ids, g["ltfm_jul"].to_numpy())
    report["inputs"] = {"base_recipe_receipt": sha256_file(ROOT / BASE / "stages/recipe.json"), "modified_recipe_receipt": sha256_file(ROOT / MOD / "stages/recipe.json")}
    g.loc[g["ltfm_jul"], [ID, "bucket", "flight", "q", "q_seen_number", "v30_orig", "delta_v30", "q_delta", "v33_orig", "delta_v33", "q_delta_v33",
                          "eobt1_minus_sched_s", "aobt3_minus_iobt_s", "mvt_minus_sched_s"]].to_parquet(DELTA, index=False)
    report["delta_table"] = {"path": DELTA.relative_to(ROOT).as_posix(), "sha256": sha256_file(DELTA)}
    out = args.output if args.output.is_absolute() else ROOT / args.output
    write_json(out, json.loads(json.dumps(report, default=float).replace("NaN", "null")))
    v = report["stages"]["recipe/detail.parquet:v30"]
    print(json.dumps({"v30_by_bucket": {k: {m: round(x, 2) for m, x in s.items()} for k, s in v["by_bucket"].items()},
                      "q_delta": {k: v for k, v in report["soft_restoration_q_delta"].items() if k != "top10_abs_q_delta"}}, indent=1, default=float))


if __name__ == "__main__":
    main()
