"""LFPG EOBT_1 sensitivity (docs/input_shift/lfpg_eobt_sensitivity.md): v30's chain on the ranking file with LFPG July-2026 EOBT_1
reset to IOBT where EOBT_1 > IOBT (runs/final_lfpg_eobt/run) against the unmodified dry run (runs/final_dryrun/run).

Both runs use the same code, models and machine, so float noise cancels and every difference comes from the 7,510 edited EOBT_1
values (and whatever context features read other flights' EOBT_1). Per stage, per-row delta = modified - original, summarised on:
the edited rows by EOBT_1 - IOBT bucket (+300 s, +600 s, other), the unedited LFPG July rows, and all other rows (spill-over);
LFPG July rows also by ADS-B status and carrier. Also which feature columns change, and on how many rows. Label-free; read-only.
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
from prc2026.artifacts import sha256_file, utc_now, write_json  # noqa: E402

ID = "MVT_ID_mvt"
BASE = Path("runs/final_dryrun/run")
MOD = Path("runs/final_lfpg_eobt/run")
STAGES = {
    "v9/v9.parquet": ["v7", "v9_frozen", "v7_twelve", "v9"],
    "leaves/tables.parquet": ["ens_v20_bag_q_all", "ens_v22_lirf_q"],
    "networks/ensembles.parquet": ["ens_ctx_e24", "ens_ctx_e24b", "ens_ctx_e32", "ens_ctx_lirf", "ens_ctx", "ens_new", "ens_nn", "ens_nnl"],
    "residual/residual.parquet": ["residual:adj2", "residual:cat", "network", "residual"],
    "recipe/detail.parquet": ["committee_mix", "v30"],
}


def stats(delta: np.ndarray) -> dict:
    d = delta[np.isfinite(delta)]
    if not len(d):
        return {"n": 0}
    return {"n": int(len(d)), "mean": float(d.mean()), "rms": float(np.sqrt(np.mean(d ** 2))), "p1": float(np.quantile(d, 0.01)),
            "p99": float(np.quantile(d, 0.99)), "max_abs": float(np.abs(d).max()), "share_nonzero": float((d != 0).mean())}


def groups() -> pd.DataFrame:
    raw = pq.read_table(ROOT / "data/raw/ranking.parquet", columns=[ID, "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "EOBT_1_flt", "IOBT_flt", "FLIGHT_mvt"]).to_pandas()
    dep = raw.loc[raw["PHASE_mvt"].eq("DEP")].reset_index(drop=True)
    gap = (dep["EOBT_1_flt"] - dep["IOBT_flt"]).dt.total_seconds()
    lfpg_jul = dep["ADEP_mvt"].eq("LFPG") & dep["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m").eq("2026-07")
    edited = lfpg_jul & (gap > 0)
    bucket = np.where(~lfpg_jul, "other_rows", np.where(~edited, "lfpg_jul_unedited", np.where(gap == 300, "edited_+300", np.where(gap == 600, "edited_+600", "edited_other"))))
    carrier = dep["FLIGHT_mvt"].astype("string").str.upper().str.extract(r"^([A-Z]+)", expand=False).str[:3].fillna("?")
    status = pd.read_parquet(ROOT / "runs/error_map/ranking_rows.parquet", columns=[ID, "status"]).set_index(ID)["status"]
    out = pd.DataFrame({ID: dep[ID].to_numpy(), "bucket": bucket, "lfpg_jul": lfpg_jul.to_numpy(), "edited": edited.to_numpy(),
                        "carrier": carrier.to_numpy(), "airport": dep["ADEP_mvt"].to_numpy(), "month": dep["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m").to_numpy()})
    out["status"] = status.reindex(out[ID]).to_numpy()
    return out


def feature_changes(ids: np.ndarray, lfpg_jul: np.ndarray) -> dict:
    a_path, b_path = ROOT / BASE / "features/timeline.parquet", ROOT / MOD / "features/timeline.parquet"
    names = [c for c in pq.ParquetFile(a_path).schema_arrow.names if c != ID]
    out = {}
    for start in range(0, len(names), 40):
        cols = names[start:start + 40]
        a = pd.read_parquet(a_path, columns=[ID, *cols]).set_index(ID).reindex(ids)
        b = pd.read_parquet(b_path, columns=[ID, *cols]).set_index(ID).reindex(ids)
        for c in cols:
            x, y = a[c], b[c]
            if pd.api.types.is_numeric_dtype(x):
                xv, yv = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
                diff = ~((xv == yv) | (np.isnan(xv) & np.isnan(yv)))
            else:
                diff = (x.astype("string").fillna("<NA>") != y.astype("string").fillna("<NA>")).to_numpy()
            if diff.any():
                out[c] = {"lfpg_jul_rows": int((diff & lfpg_jul).sum()), "other_rows": int((diff & ~lfpg_jul).sum())}
        del a, b
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("docs/input_shift/lfpg_eobt_sensitivity.json"))
    args = parser.parse_args()
    g = groups()
    ids = g[ID].to_numpy()
    report = {"created_at_utc": utc_now(), "base": BASE.as_posix(), "modified": MOD.as_posix(),
              "modification": json.loads((ROOT / "runs/final_lfpg_eobt/input/modification.json").read_text(encoding="utf-8")),
              "rows": {k: int(v) for k, v in g["bucket"].value_counts().items()}, "stages": {}}
    for path, columns in STAGES.items():
        a = pd.read_parquet(ROOT / BASE / path).set_index(ID).reindex(ids)
        b = pd.read_parquet(ROOT / MOD / path).set_index(ID).reindex(ids)
        for c in columns:
            if c not in a or c not in b:
                continue
            delta = b[c].to_numpy(dtype=float) - a[c].to_numpy(dtype=float)
            rec = {"by_bucket": {k: stats(delta[(g["bucket"] == k).to_numpy()]) for k in ["edited_+300", "edited_+600", "edited_other", "lfpg_jul_unedited", "other_rows"]},
                   "lfpg_jul_all": stats(delta[g["lfpg_jul"].to_numpy()])}
            if c in ("v30", "committee_mix", "ens_v20_bag_q_all", "v9"):
                lj = g["lfpg_jul"].to_numpy()
                rec["lfpg_jul_by_status"] = {s: stats(delta[lj & (g["status"] == s).to_numpy()]) for s in sorted(g.loc[lj, "status"].dropna().unique())}
                top = g.loc[lj, "carrier"].value_counts().head(6).index
                rec["lfpg_jul_by_carrier"] = {k: stats(delta[lj & (g["carrier"] == k).to_numpy()]) for k in top}
                rec["lfpg_jul_edited_by_status"] = {s: stats(delta[g["edited"].to_numpy() & (g["status"] == s).to_numpy()]) for s in sorted(g.loc[lj, "status"].dropna().unique())}
                other = ~lj
                rec["other_rows_by_airport_month"] = {f"{ap}:{mo}": stats(delta[other & (g["airport"] == ap).to_numpy() & (g["month"] == mo).to_numpy()])
                                                      for ap, mo in g.loc[other, ["airport", "month"]].drop_duplicates().itertuples(index=False)}
            report["stages"][f"{path}:{c}"] = rec
        del a, b
    ea = pd.read_parquet(ROOT / BASE / "export/vigorous-whistle_v900.parquet").set_index(ID).reindex(ids)
    eb = pd.read_parquet(ROOT / MOD / "export/vigorous-whistle_v901.parquet").set_index(ID).reindex(ids)
    di = eb["TAXITIME_SEC_mvt"].to_numpy(dtype=float) - ea["TAXITIME_SEC_mvt"].to_numpy(dtype=float)
    report["export_integers"] = {k: stats(di[(g["bucket"] == k).to_numpy()]) for k in ["edited_+300", "edited_+600", "edited_other", "lfpg_jul_unedited", "other_rows"]}
    report["feature_columns_changed"] = feature_changes(ids, g["lfpg_jul"].to_numpy())
    report["inputs"] = {"base_recipe_receipt": sha256_file(ROOT / BASE / "stages/recipe.json"), "modified_recipe_receipt": sha256_file(ROOT / MOD / "stages/recipe.json")}
    out = args.output if args.output.is_absolute() else ROOT / args.output
    write_json(out, json.loads(json.dumps(report, default=float)))
    v = report["stages"]["recipe/detail.parquet:v30"]
    print(json.dumps({"v30_by_bucket": {k: {m: round(x, 2) for m, x in s.items()} for k, s in v["by_bucket"].items()}}, indent=1))


if __name__ == "__main__":
    main()
