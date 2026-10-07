"""Fork M: pool table. Classes per departure: gate (traced from the stand), pinned (a shipped hard layer moved it: rematch, intruder,
own-motion v1/v2; ranking only), dark (segment starts moving), unm_seen (no segment, own aircraft seen at the airport after its
inbound: p_n >= 1), unm_unseen. Development squared error from the v68 dev proxy (LIRF population and guard rows reported apart);
ranking projection = ranking rows x development mean squared error of the same (airport, class)."""
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
GUARD = [213142471, 213142469]


def cls(d):
    k = d["k"].to_numpy().astype(object)
    seen = d["own_n"].fillna(0).to_numpy() >= 1
    k = np.where((k == "unm") & seen, "unm_seen", np.where(k == "unm", "unm_unseen", k))
    return k


r = pd.read_parquet(ROOT / "runs/hunt/census_rank.parquet")
v = pd.read_parquet(ROOT / "runs/hunt/census_dev.parquet")
r["c"] = cls(r); v["c"] = cls(v)
r["c"] = np.where(r["L_intruder"] | r["L_own1"] | r["L_own2"] | r["L_rematch"], "pinned_layer", r["c"])
v["c"] = np.where(v["cls"].eq("LIRF_pop") | v["MVT_ID_mvt"].isin(GUARD), "LIRF_pop", v["c"])
tot = v["se"].sum()
g = v.groupby(["airport", "c"]).agg(n=("se", "size"), sse=("se", "sum"))
g["mse"] = g["sse"] / g["n"]; g["rmse"] = np.sqrt(g["mse"]); g["dev_share"] = g["sse"] / tot
rk = r.groupby(["ADEP_mvt", "c"]).size().rename("rank_n")
rk.index.names = ["airport", "c"]
t = g.join(rk, how="outer")
t["rank_proj_sse"] = t["rank_n"] * t["mse"]
T = t["rank_proj_sse"].sum()
t["rank_proj_share"] = t["rank_proj_sse"] / T
t = t.sort_values("rank_proj_sse", ascending=False)
pd.set_option("display.width", 200)
print(t.round(4).to_string())
print("ranking projected total SSE %.3e (rows with a dev class)" % T)
s = t.groupby(level="c")[["n", "sse", "rank_n", "rank_proj_sse"]].sum()
s["dev_share"] = s["sse"] / tot; s["rank_share"] = s["rank_proj_sse"] / T
print(s.sort_values("rank_proj_sse", ascending=False).round(4).to_string())
t.reset_index().to_csv(ROOT / "runs/hunt/pools.csv", index=False)
