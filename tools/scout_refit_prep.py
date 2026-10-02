"""Build the rows tables for the refit experiment: the current model part (a column of runs/dev_model_part/rows.parquet) in place of
the v46 stack, with the implied off-block day, the published totals and 'complete' recomputed.

  rows_<col>.parquet            every month from the current model part (candidate: layers fitted on it)
  rows_<col>_ctl_<month>.parquet  the scored month from the current model part, the other months the v46 stack (control: the shipped
                                parameters, fitted on the v46-era months, applied to the current model part)
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path
sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "src"))
from screen_daily_totals import published

OUT = ROOT / "runs/scout/refit"
MONTHS = ["2025-01", "2025-07", "2025-11", "2025-12"]


def takeoff(ids):
    raw = pd.concat([pd.read_parquet(p, columns=["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt"]) for m in MONTHS for p in sorted((ROOT / "data/raw").glob(f"training_{m}-*.parquet"))])
    raw = raw[raw["PHASE_mvt"].eq("DEP")].drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")["MVT_TIME_UTC_mvt"]
    return pd.to_datetime(raw.reindex(ids).to_numpy(), utc=True)


def with_stack(t, stack, tk, pub):
    t = t.copy()
    t["stack"] = np.asarray(stack, dtype=float)
    t["bday"] = (tk - pd.to_timedelta(t["stack"].to_numpy(), unit="s")).strftime("%Y-%m-%d")
    t["pos_hat"] = np.maximum(t["a"] - t["stack"], 0.0)
    t["r"] = t["label"] - t["stack"]
    t["se"] = t["r"] ** 2
    t = t.drop(columns=["pub", "n_pub", "complete"]).join(pub, on=["airport", "bday"])
    t["complete"] = (t["pub"].notna() & (t.groupby(["airport", "bday"])["a"].transform("size") >= 0.9 * t["n_pub"])).to_numpy()
    return t


def main(col):
    OUT.mkdir(parents=True, exist_ok=True)
    base = pd.read_parquet(ROOT / "runs/daily_totals/rows.parquet")
    dm = pd.read_parquet(ROOT / "runs/dev_model_part/rows.parquet").drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt")
    v = dm[col].reindex(base["MVT_ID_mvt"]).to_numpy(dtype=float)
    missing = np.isnan(v)
    print("rows", len(base), "without the column", int(missing.sum()))
    v = np.where(missing, base["stack"].to_numpy(dtype=float), v)
    if "v46" in dm:
        d = dm["v46"].reindex(base["MVT_ID_mvt"]).to_numpy(dtype=float)
        ok = ~np.isnan(d)
        print("check: v46 column against rows.parquet's stack, max |diff| =", float(np.nanmax(np.abs(d[ok] - base["stack"].to_numpy()[ok]))))
    tk, pub = takeoff(base["MVT_ID_mvt"].to_numpy()), published()
    new = with_stack(base, v, tk, pub)
    new.to_parquet(OUT / f"rows_{col}.parquet", index=False)
    for m in MONTHS:
        mix = pd.concat([base[base["month"] != m], new[new["month"] == m]], ignore_index=True)
        mix.to_parquet(OUT / f"rows_{col}_ctl_{m}.parquet", index=False)
    for m in MONTHS:
        g = new[new["month"] == m]
        print(m, "RMSE v46", round(float(np.sqrt(((g["label"] - base.loc[base["month"] == m, "stack"].to_numpy()) ** 2).mean())), 4), col, round(float(np.sqrt(g["se"].mean())), 4))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "v55m")
