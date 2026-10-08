"""Table for the W25 follow-up (aud13): runs/season_pass/dev_sp.parquet (raw, chain, clocks; ship = v73 dev) + the stacker / neighbour /
queue columns of runs/stack_second/dev_table.parquet + season_pass_v1's out-of-month W25 predictions (p4 = w25 bag, ship4 = v74 dev on
November / December at EGLL / LIRF / LEBL) -> runs/audit_v72/dev_w25.parquet (exploration)."""
import numpy as np, pandas as pd
ROOT=""; ID="MVT_ID_mvt"
A=pd.read_parquet(ROOT+"runs/season_pass/dev_sp.parquet")
S=pd.read_parquet(ROOT+"runs/stack_second/dev_table.parquet")
keep=[c for c in S.columns if c.startswith("s_")]+["nx","nxk","nn","x_nmerr","gp","gn","gp2","gp4","gn4","bpos","blen"]
assert (S[ID].to_numpy()==A[ID].to_numpy()).all()
A=pd.concat([A,S[keep].reset_index(drop=True)],axis=1)
P=pd.read_parquet(ROOT+"runs/season_pass/preds_dev.parquet").set_index(ID).reindex(A[ID])
p4=np.full(len(A),np.nan)
for a in ["EGLL","LIRF","LEBL"]:
    v=P[f"sp_{a}"].to_numpy(); m=~np.isnan(v); p4[m]=v[m]
v73=A.ship.to_numpy(float); v74=v73.copy(); m=~np.isnan(p4); v74[m]=np.maximum(v73[m]+np.clip(0.5*p4[m],-120,120),-12)
A["p4"]=p4; A["d4"]=v74-v73; A["ship"]=v74
A.to_parquet(ROOT+"runs/audit_v72/dev_w25.parquet",index=False); print(A.shape, "moved rows", int(m.sum()))
