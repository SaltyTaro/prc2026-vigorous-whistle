"""Leaf predictions for the tree_pos read-out: mean over the three x19 leaves of ctl (runs/tree_own/fits, dev folds; runs/tree_pos/fits,
refit12) and pos (runs/tree_pos/fits) -> runs/audit_v72/leaves_{dev,rank}.parquet aligned with runs/stack_third/{dev,rank}_t3.parquet;
also prints the standalone RMSE change of the leaf mean by airport-month on development (labels from dev_t3)."""
import numpy as np, pandas as pd
ROOT=""; ID="MVT_ID_mvt"
LEAVES=["adsb_v20__lgb_x19_s24","adsb_v20__lgb_x19_s25","adsb_v20__lgb_x19_s26"]
DEV={"2025-01":"lomo_2025-01","2025-07":"lomo_2025-07","2025-11":"forward","2025-12":"jan_nov"}
def read(root,leaf,var,fold):
    return pd.read_parquet(ROOT+f"runs/{root}/fits/{leaf}/{var}/{fold}/predictions.parquet").drop_duplicates(ID).set_index(ID)["adsb_full"]
T=pd.read_parquet(ROOT+"runs/stack_third/dev_t3.parquet",columns=[ID,"month","airport","label"])
out=pd.DataFrame({ID:T[ID]}); out["lf_ctl"]=np.nan; out["lf_pos"]=np.nan
for mo,fold in DEV.items():
    s=(T.month==mo).to_numpy(); ids=T.loc[s,ID]
    out.loc[s,"lf_ctl"]=np.mean([read("tree_own",l,"ctl",fold).reindex(ids).to_numpy() for l in LEAVES],axis=0)
    out.loc[s,"lf_pos"]=np.mean([read("tree_pos",l,"pos",fold).reindex(ids).to_numpy() for l in LEAVES],axis=0)
out.to_parquet(ROOT+"runs/audit_v72/leaves_dev.parquet",index=False)
K=pd.read_parquet(ROOT+"runs/stack_third/rank_t3.parquet",columns=[ID])
rk=pd.DataFrame({ID:K[ID]})
rk["lf_ctl"]=np.mean([read("tree_pos",l,"ctl","refit12").reindex(K[ID]).to_numpy() for l in LEAVES],axis=0)
rk["lf_pos"]=np.mean([read("tree_pos",l,"pos","refit12").reindex(K[ID]).to_numpy() for l in LEAVES],axis=0)
rk.to_parquet(ROOT+"runs/audit_v72/leaves_rank.parquet",index=False)
print("coverage dev",out[["lf_ctl","lf_pos"]].notna().mean().round(3).to_dict(),"rank",rk[["lf_ctl","lf_pos"]].notna().mean().round(3).to_dict())
y=T.label.to_numpy(float); ok=out.lf_ctl.notna()&out.lf_pos.notna()&(y<20000)
d=pd.DataFrame({"ap":T.airport,"mo":T.month,"e0":(out.lf_ctl-y)**2,"e1":(out.lf_pos-y)**2})[ok]
g=d.groupby(["ap","mo"]).sum(); print(((np.sqrt(g.e1)/np.sqrt(g.e0)-1)*100).unstack().round(2).to_string())
print("rank mean |pos-ctl|",round(float((rk.lf_pos-rk.lf_ctl).abs().mean()),2),"dev",round(float((out.lf_pos-out.lf_ctl).abs().mean()),2))
