import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools"))
from screen_published_means import AIRPORTS, CAP, published, label_means
from run_published_means import fit_mapping, shift_for, method
import run_intruder_bound as R
pub = published()
table = label_means().merge(pub, on=["airport", "month"], how="inner", validate="one_to_one")
MONTHS = ["2025-01", "2025-07", "2025-11", "2025-12"]
dev = pd.read_parquet("runs/opdi_stack_lszh/dev_v68_proxy.parquet")
raw = pd.concat([pd.read_parquet(p, columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADEP_flt", "AIRCRAFT_TYPE_mvt"]) for p in sorted(Path("data/raw").glob("training_2025-*.parquet"))])
raw = raw[raw.PHASE_mvt == "DEP"].drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt").reindex(dev["MVT_ID_mvt"])
pop = (raw.ADEP_mvt.eq("LIRF") & raw.ADEP_flt.isna() & raw.AIRCRAFT_TYPE_mvt.isna()).to_numpy()
guard = dev["MVT_ID_mvt"].isin(R.GUARD).to_numpy()
lvl = {}
for m in MONTHS:
    f = dev[dev.month == m]
    p, y, a = f["value"].to_numpy(float), f["label"].to_numpy(float), f["airport"].to_numpy()
    inside = (p > 0) & (p <= CAP)
    lvl[m] = {}
    for ap in AIRPORTS:
        k = (a == ap) & inside; o = k & (y > 0) & (y <= CAP)
        q = pub[(pub.airport == ap) & (pub.month == m)].iloc[0]; mp = fit_mapping(table, ap, m)
        lvl[m][ap] = {"estimate": float(p[k].mean() - (q.P + mp["function"](q.u))), "true": float(p[o].mean() - y[o].mean()), "sd_map": mp["sd_map"], "u": float(q.u), "u_max": mp["u_max"]}
rng = np.random.default_rng(20261004)
for max_sigma in (8.0, 5.0):
    print("max_sigma", max_sigma)
    pcts = []
    for m in MONTHS:
        f = dev[dev.month == m].reset_index(drop=True)
        p = f["value"].to_numpy(float); a = f["airport"].to_numpy(); vals = p.copy()
        fm = (dev.month == m).to_numpy()
        keep = pop[fm] | guard[fm]
        cells = {}
        for ap in AIRPORTS:
            e = lvl[m][ap]; bias, noise = method(lvl, ap, m)
            sh, rec = shift_for(e["estimate"], bias, max(e["sd_map"], noise), e["u"], e["u_max"], max_sigma)
            k = (a == ap) & (p > 0) & (p <= CAP) & ~keep
            vals[k] = np.maximum(p[k] - sh, -12.0)
            if rec["eligible"]: cells[ap] = (round(sh, 1), round(e["true"], 1))
        pct, lo, hi = R.interval(f, "value", vals, rng)
        pcts.append(pct)
        print(" ", m, round(pct, 4), round(lo, 3), round(hi, 3), cells)
    print("  eq mean", round(np.mean([pcts[0], pcts[1], pcts[3]]), 4))
# ranking stake on v68
ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
rk = pd.read_parquet("data/raw/ranking.parquet", columns=[ID, "PHASE_mvt", "ADEP_mvt", "ADEP_flt", "AIRCRAFT_TYPE_mvt", "MVT_TIME_UTC_mvt"])
rk = rk[rk.PHASE_mvt.eq("DEP")].assign(month=lambda x: x.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m"))
v = pd.read_parquet("runs/submissions/vigorous-whistle_v68.parquet").merge(rk, on=ID)
for max_sigma in (8.0, 5.0):
    stake, cells = 0.0, {}
    for (ap, m), g in v.groupby(["ADEP_mvt", "month"]):
        q = pub[(pub.airport == ap) & (pub.month == m)].iloc[0]; mp = fit_mapping(table, ap, None)
        pp = g[T].to_numpy(float); ins = (pp > 0) & (pp <= CAP)
        bias, noise = method(lvl, ap, None)
        sh, rec = shift_for(float(pp[ins].mean() - (q.P + mp["function"](q.u))), bias, max(mp["sd_map"], noise), float(q.u), mp["u_max"], max_sigma)
        if rec["eligible"]:
            popk = (g.ADEP_mvt.eq("LIRF") & g.ADEP_flt.isna() & g.AIRCRAFT_TYPE_mvt.isna()).to_numpy() | g[ID].isin(R.GUARD).to_numpy()
            n = int((ins & ~popk).sum()); stake += n * (2 * sh * rec["effective_level_s"] - sh ** 2)
            cells[f"{ap} {m}"] = (round(sh, 1), round(rec["effective_level_s"], 1), round(rec["sigma_s"], 1), n)
    N = len(v); print("ranking max_sigma", max_sigma, cells, "stake SSE", round(stake), "RMSE if holds", round(np.sqrt((N * 215.9449 ** 2 - stake) / N), 4))
