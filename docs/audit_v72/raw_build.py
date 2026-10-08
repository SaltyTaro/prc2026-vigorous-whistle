"""Raw ADS-B inputs for the end-of-chain audit (exploration): features_v19own (26 columns) + matches2 numeric columns, times as seconds
before take-off, for the development rows (runs/stack_second/dev_table.parquet) and the ranking rows (rank_table.parquet)."""
import sys, numpy as np, pandas as pd
ROOT=""; ID="MVT_ID_mvt"
def raw(months):
    f=pd.concat([pd.read_parquet(ROOT+f"data/external/adsblol/features_v19own/features_v19_{m}.parquet") for m in months]).drop_duplicates(ID)
    g=pd.concat([pd.read_parquet(ROOT+f"data/external/adsblol/matches/matches2_{m}.parquet") for m in months]).drop_duplicates(ID)
    tk=pd.to_datetime(g["takeoff"],utc=True)
    for c in ["segment_start","motion_start","arrival_last_ground","last_ground"]:
        g["m_"+c]=(tk-pd.to_datetime(g[c],utc=True)).dt.total_seconds()
    keep=["first_gs","stop_seconds","first_lat","first_lon","taxi_path_m","taxi_stops","taxi_stopped_s","slow_start_s","gap_before_s","points",
          "at_known_stand","first_dist_stand_m","anchor_dist_stand_m"]
    g=g[[ID]+keep+[c for c in g.columns if c.startswith("m_")]].rename(columns={c:"m_"+c for c in keep})
    for c in g.columns:
        if c!=ID: g[c]=pd.to_numeric(g[c],errors="coerce").astype("float32")
    for c in f.columns:
        if c!=ID: f[c]=pd.to_numeric(f[c],errors="coerce").astype("float32")
    return f.merge(g,on=ID,how="outer")
for nm,months in [("dev",["2025-01","2025-07","2025-11","2025-12"]),("rank",["2026-01","2026-07"])]:
    ids=pd.read_parquet(ROOT+f"runs/stack_second/{nm}_table.parquet",columns=[ID])
    r=ids.merge(raw(months),on=ID,how="left")
    r.to_parquet(ROOT+f"runs/audit_v72/raw_{nm}.parquet",index=False)
    print(nm,r.shape,"coverage v19",round(r["adsb_appear_s"].notna().mean(),3) if "adsb_appear_s" in r else None,"matches2",round(r["m_points"].notna().mean(),3))
