"""Raw ADS-B inputs for the end-of-chain audit (exploration): features_v19own (26 columns) + matches2 numeric columns, times as seconds
before take-off, for the development rows (runs/stack_second/dev_table.parquet) and the ranking rows (rank_table.parquet).

Run as a script it writes runs/audit_v72/raw_{dev,rank}.parquet exactly as the governing run of stack_third_v1 did (the computation is
unchanged; only the repository root is now found from this file instead of a fixed D:/ path). build() is the same computation for any
movement file's months, folders and row table (final-file chain, docs/final_file/lane_L.md).

Known slip, kept on purpose (v73 depends on it): matches2's times are epoch-second floats, which pd.to_datetime reads as nanoseconds, so the
four m_ time columns (segment_start, motion_start, arrival_last_ground, last_ground) hold differences in nanoseconds-as-seconds
(docs/audit_v72/README.md, aud9)."""
import sys, numpy as np, pandas as pd
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2].as_posix()+"/"; ID="MVT_ID_mvt"
FEATURES_DIR=ROOT+"data/external/adsblol/features_v19own/"
MATCHES_DIR=ROOT+"data/external/adsblol/matches/"
def raw(months, features_dir=None, matches_dir=None):
    features_dir=features_dir or FEATURES_DIR; matches_dir=matches_dir or MATCHES_DIR
    f=pd.concat([pd.read_parquet(features_dir+f"features_v19_{m}.parquet") for m in months]).drop_duplicates(ID)
    g=pd.concat([pd.read_parquet(matches_dir+f"matches2_{m}.parquet") for m in months]).drop_duplicates(ID)
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
def build(months, table, out, features_dir=None, matches_dir=None):
    """table: a parquet whose MVT_ID_mvt column fixes the row order (stack_second's {dev,rank}_table); out: the raw table written."""
    ids=pd.read_parquet(table,columns=[ID])
    r=ids.merge(raw(months,features_dir,matches_dir),on=ID,how="left")
    r.to_parquet(out,index=False)
    return r
if __name__=="__main__":
    for nm,months in [("dev",["2025-01","2025-07","2025-11","2025-12"]),("rank",["2026-01","2026-07"])]:
        r=build(months,ROOT+f"runs/stack_second/{nm}_table.parquet",ROOT+f"runs/audit_v72/raw_{nm}.parquet")
        print(nm,r.shape,"coverage v19",round(r["adsb_appear_s"].notna().mean(),3) if "adsb_appear_s" in r else None,"matches2",round(r["m_points"].notna().mean(),3))
