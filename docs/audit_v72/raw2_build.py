"""Corrected matches2 timing columns (epoch seconds; raw_build.py read them as nanoseconds, so its m_ time columns are all zero and
carry no information in v73's third pass). Writes runs/audit_v72/raw2_{dev,rank}.parquet with mt_* = take-off minus segment start,
motion start, last ground, arrival last ground (seconds), aligned with runs/stack_third/{dev,rank}_t3.parquet. raw_*.parquet stay as
they are (their hashes are in runs/stack_third/tables.json)."""
import pandas as pd, numpy as np
ROOT=""; ID="MVT_ID_mvt"
for nm,months in [("dev",["2025-01","2025-07","2025-11","2025-12"]),("rank",["2026-01","2026-07"])]:
    g=pd.concat([pd.read_parquet(ROOT+f"data/external/adsblol/matches/matches2_{m}.parquet",columns=[ID,"takeoff","segment_start","motion_start","last_ground","arrival_last_ground"]) for m in months]).drop_duplicates(ID)
    out=pd.DataFrame({ID:g[ID]})
    for c in ["segment_start","motion_start","last_ground","arrival_last_ground"]:
        out["mt_"+c]=(g["takeoff"]-g[c]).astype("float32")
    ids=pd.read_parquet(ROOT+f"runs/stack_third/{nm}_t3.parquet",columns=[ID])
    r=ids.merge(out,on=ID,how="left"); assert (r[ID].to_numpy()==ids[ID].to_numpy()).all()
    r.to_parquet(ROOT+f"runs/audit_v72/raw2_{nm}.parquet",index=False)
    print(nm,r.describe().loc[["count","mean","50%"]].round(1).to_string())
