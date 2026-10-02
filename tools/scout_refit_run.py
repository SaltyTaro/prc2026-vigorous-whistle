"""Refit experiment: the published-data layers (v48 -> extreme-day -> v57) on the current model part, with parameters fitted
  control:   on the v46-era months (the shipped parameters), applied to the current model part's scored month
  candidate: on the current model part's own months
Month-blocked in both. Usage: refit_run.py v55m [v48|layers|score]"""
import shutil, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path
sys.path.insert(0, str(ROOT / "tools")); sys.path.insert(0, str(ROOT / "src"))
MONTHS = ["2025-01", "2025-07", "2025-11", "2025-12"]
REF = ROOT / "runs/scout/refit"


def v48_job(args):
    month, tag, rows = args
    import run_daily_totals as rd
    from prc2026.artifacts import read_json
    settings, tau_scale, prior = rd.settings_from(read_json(ROOT / "configs/daily_totals_v1.json"))
    rd.run_month(month, tag, tau_scale, settings, rows=rows, prior=prior)


def v48(col):
    jobs = [(m, f"scout_refit_ctl_{col}", f"runs/scout/refit/rows_{col}_ctl_{m}.parquet") for m in MONTHS]
    jobs += [(m, f"scout_refit_new_{col}", f"runs/scout/refit/rows_{col}.parquet") for m in MONTHS]
    with ProcessPoolExecutor(2) as pool:
        list(pool.map(v48_job, jobs))
    for m in MONTHS:   # the control's v48 directory for scored month m: the shipped run's other months + this month's control run
        d = REF / f"v48_ctl_{col}_{m}"
        d.mkdir(parents=True, exist_ok=True)
        for k in MONTHS:
            src = ROOT / (f"runs/daily_totals/scout_refit_ctl_{col}/{k}.parquet" if k == m else f"runs/daily_totals/daily_totals_v1/{k}.parquet")
            shutil.copyfile(src, d / f"{k}.parquet")


def layers(col):
    import scout_extreme_shift as se
    import scout_var_alloc as sva
    se.TABLES[f"new_{col}"] = (REF / f"rows_{col}.parquet", ROOT / f"runs/daily_totals/scout_refit_new_{col}")
    t, ks = sva.layered(f"new_{col}")
    t[["MVT_ID_mvt", "month", "airport", "bday", "label", "stack", "v48", "ext", "new"]].to_parquet(REF / f"cand_{col}.parquet", index=False)
    print("candidate k", ks, flush=True)
    parts = []
    for m in MONTHS:
        se.TABLES[f"ctl_{col}_{m}"] = (REF / f"rows_{col}_ctl_{m}.parquet", REF / f"v48_ctl_{col}_{m}")
        t, ks = sva.layered(f"ctl_{col}_{m}")
        parts.append(t.loc[t["month"] == m, ["MVT_ID_mvt", "month", "airport", "bday", "label", "stack", "v48", "ext", "new"]])
        print("control", m, "k", ks[m], flush=True)
    pd.concat(parts, ignore_index=True).to_parquet(REF / f"ctl_{col}.parquet", index=False)


def score(col):
    from run_adsb_v11 import paired_days
    c = pd.read_parquet(REF / f"ctl_{col}.parquet").set_index("MVT_ID_mvt")
    n = pd.read_parquet(REF / f"cand_{col}.parquet").set_index("MVT_ID_mvt").reindex(c.index)
    out = {}
    for m in MONTHS:
        k = (c["month"] == m).to_numpy()
        y, a, b, s = c["label"].to_numpy()[k], c["new"].to_numpy()[k], n["new"].to_numpy()[k], c["stack"].to_numpy()[k]
        r = lambda v: float(np.sqrt(np.mean((y - v) ** 2)))
        iv = paired_days(c["bday"].to_numpy()[k], y, b, a, 20260929)["rmse_difference_95pct_interval_seconds"]
        out[m] = {"model_part": r(s), "control": r(a), "candidate": r(b), "change_pct": 100 * (r(b) / r(a) - 1), "interval": iv, "rows_changed": int((np.abs(b - a) > 1e-9).sum())}
        print(m, {k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in out[m].items()})
    print("equal-footing mean change %:", round(float(np.mean([out[m]["change_pct"] for m in ["2025-01", "2025-07", "2025-12"]])), 4))
    return out


if __name__ == "__main__":
    col, step = sys.argv[1], sys.argv[2]
    {"v48": v48, "layers": layers, "score": score}[step](col)
