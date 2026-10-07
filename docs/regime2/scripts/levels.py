import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools"))
from screen_published_means import AIRPORTS, CAP, published, label_means
from run_published_means import fit_mapping
pub = published()
table = label_means().merge(pub, on=["airport", "month"], how="inner", validate="one_to_one")
ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
# development: v68 proxy estimate vs true
dev = pd.read_parquet("runs/opdi_stack_lszh/dev_v68_proxy.parquet")
rows = []
for (a, m), g in dev.groupby(["airport", "month"]):
    p = g["value"].to_numpy(float); y = g["label"].to_numpy(float)
    inside = (p > 0) & (p <= CAP); ordn = inside & (y > 0) & (y <= CAP)
    q = pub[(pub.airport == a) & (pub.month == m)].iloc[0]
    mp = fit_mapping(table, a, m)
    est = p[inside].mean() - (q.P + mp["function"](q.u)); true = p[ordn].mean() - y[ordn].mean()
    rows.append((a, m, round(est, 1), round(true, 1), round(est - true, 1), round(mp["sd_map"], 1)))
d = pd.DataFrame(rows, columns=["airport", "month", "est", "true", "est_minus_true", "sd_map"])
bias = d.groupby("airport").est_minus_true.agg(["mean", "std"]).round(1)
print(d.pivot(index="airport", columns="month", values="true").round(1).to_string())
print(bias.to_string())
# ranking: v68 and v46
raw = pd.read_parquet("data/raw/ranking.parquet", columns=[ID, "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"])
raw = raw[raw.PHASE_mvt.eq("DEP")].assign(month=lambda x: x.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m"))
out = []
for ver in ["v46", "v66", "v68"]:
    v = pd.read_parquet(f"runs/submissions/vigorous-whistle_{ver}.parquet").merge(raw[[ID, "ADEP_mvt", "month"]], on=ID)
    for (a, m), g in v.groupby(["ADEP_mvt", "month"]):
        p = g[T].to_numpy(float); p = p[(p > 0) & (p <= CAP)]
        q = pub[(pub.airport == a) & (pub.month == m)].iloc[0]; mp = fit_mapping(table, a, None)
        out.append((ver, a, m, round(p.mean() - (q.P + mp["function"](q.u)) - bias.loc[a, "mean"], 1), round(q.u, 3), round(mp["u_max"], 3), len(p)))
o = pd.DataFrame(out, columns=["ver", "airport", "month", "level_minus_bias", "u", "u_max", "rows"])
print(o.pivot_table(index=["airport", "month"], columns="ver", values="level_minus_bias").to_string())
print(o[o.ver == "v68"][["airport", "month", "u", "u_max", "rows"]].to_string(index=False))
