"""Weather inputs (IEM surface observations, the frozen weather bundle's wx_ columns) for the development rows of runs/stack_third and
the ranking rows: runs/audit_v72/wx_{dev,rank}.parquet (exploration)."""
import glob, pyarrow.parquet as pq, pandas as pd
ROOT=""; ID="MVT_ID_mvt"
W=ROOT+"data/weather/827d828796d9cb9c4581/features/"
names=[c for c in pq.ParquetFile(W+"ranking.parquet").schema_arrow.names if c.startswith("wx_")]
def load(paths):
    out=[]
    for p in paths:
        n=pq.ParquetFile(p).schema_arrow.names
        cols=[ID]+[c for c in names if c in n]
        out.append(pd.read_parquet(p,columns=cols))
    return pd.concat(out).drop_duplicates(ID)
dev=load([W+"training_2025-01-01_2025-02-01.parquet",W+"training_2025-07-01_2025-08-01.parquet",W+"training_2025-11-01_2025-12-01.parquet",
          ROOT+"data/december/d198dea10625b0a75977/features/training_2025-12-01_2026-01-01.parquet"])
rank=load([W+"ranking.parquet"])
for nm,src in [("dev",dev),("rank",rank)]:
    ids=pd.read_parquet(ROOT+f"runs/stack_third/{nm}_t3.parquet",columns=[ID,"month"])
    r=ids[[ID]].merge(src,on=ID,how="left")
    for c in r.columns:
        if c!=ID: r[c]=pd.to_numeric(r[c],errors="coerce").astype("float32")
    r.to_parquet(ROOT+f"runs/audit_v72/wx_{nm}.parquet",index=False)
    print(nm,r.shape,"coverage by month",ids.assign(ok=r["wx_temperature_c"].notna().to_numpy()).groupby("month").ok.mean().round(3).to_dict())
