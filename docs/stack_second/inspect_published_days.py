import numpy as np, pandas as pd
P=""; R=P+"runs/stack_second/"; ID="MVT_ID_mvt"
AP7=["EDDF","EDDM","EGLL","EHAM","LEBL","LEMD","LSZH"]
def apply(v,p):
    n=v.copy(); m=~np.isnan(p); n[m]=np.maximum(v[m]+np.clip(0.5*p[m],-120,120),-12); return n
T=pd.read_parquet(R+"dev_table.parquet",columns=[ID,"month","airport","label","v71","s_c_a","s_p_pub"])
K=pd.read_parquet(R+"rank_table.parquet",columns=[ID,"month","airport","v71","s_c_a","s_p_pub"])
T["v72"]=apply(T.v71.to_numpy(float),pd.read_parquet(R+"preds_dev.parquet").set_index(ID)["sp_hicov7"].reindex(T[ID]).to_numpy())
K["v72"]=apply(K.v71.to_numpy(float),pd.read_parquet(R+"preds_rank.parquet").set_index(ID)["sp_hicov7"].reindex(K[ID]).to_numpy())
# v72 ranking must equal the dry run after rounding
dry=pd.read_parquet(R+"rank/dry_v72/vigorous-whistle_v72.parquet").set_index(ID)["TAXITIME_SEC_mvt"].reindex(K[ID]).to_numpy()
print("dry run reproduced:", bool((np.rint(K.v72.to_numpy())==dry).all()))
rd=pd.read_parquet(P+"runs/ctx_ownpoints_weight/chain/own_all_w100/rows_all.parquet",columns=[ID,"bday","pos_true","pub"]).set_index(ID).reindex(T[ID])
rk=pd.read_parquet(P+"runs/ctx_ownpoints_weight/rank/dry_v66/steps/Y/rows.parquet",columns=[ID,"bday"]).set_index(ID).reindex(K[ID])
T["bday"]=rd.bday.to_numpy(); T["pos_true"]=rd.pos_true.to_numpy(); K["bday"]=rk.bday.to_numpy()
print("pub agrees with rows_all:", float(np.nanmax(np.abs(rd.pub.to_numpy()-T.s_p_pub.to_numpy()))))
def days(X,lab):
    X=X[X.airport.isin(AP7)].copy()
    a=X.s_c_a.to_numpy(float)
    X["p71"]=np.maximum(a-X.v71,0); X["p72"]=np.maximum(a-X.v72,0)
    if lab: X["se71"]=(X.label-X.v71)**2; X["se72"]=(X.label-X.v72)**2
    agg={"pub":("s_p_pub","first"),"p71":("p71","sum"),"p72":("p72","sum"),"n":("p71","size")}
    if lab: agg.update(pt=("pos_true","sum"),se71=("se71","sum"),se72=("se72","sum"))
    D=X.groupby(["airport","bday"]).agg(**agg).dropna(subset=["pub"])
    D["disc71"]=D.pub-D.p71; D["dlt"]=D.p72-D.p71; D["disc72"]=D.pub-D.p72
    return D
for nm,X,lab in [("dev",T,True),("rank",K,False)]:
    D=days(X,lab)
    mv=D.dlt.abs()>1
    tow=(np.sign(D.dlt)==np.sign(D.disc71))[mv]
    print(f"\n{nm}: airport-days {len(D)}, moved {mv.sum()}, toward published {tow.mean():.3f}, "
          f"sum|disc| v71 {D.disc71.abs().sum()/1e6:.3f}e6 -> v72 {D.disc72.abs().sum()/1e6:.3f}e6 ({(D.disc72.abs().sum()/D.disc71.abs().sum()-1)*100:+.2f}%), "
          f"mean implied change per row {D.dlt.sum()/D.n.sum():+.3f} s, mean disc71 per row {D.disc71.sum()/D.n.sum():+.2f} s")
    by=D.groupby("airport").apply(lambda g: pd.Series({"tow":(np.sign(g.dlt)==np.sign(g.disc71))[g.dlt.abs()>1].mean(),
        "dabs%":(g.disc72.abs().sum()/g.disc71.abs().sum()-1)*100,"dlt_pr":g.dlt.sum()/g.n.sum(),"disc_pr":g.disc71.sum()/g.n.sum(),
        "corr":np.corrcoef(g.dlt,g.disc71)[0,1]}))
    print(by.round(3).to_string())
    if lab:
        D["dse"]=D.se72-D.se71
        print("dev: days moved toward pub: dSSE", round(D.dse[mv&(np.sign(D.dlt)==np.sign(D.disc71))].sum()/1e6,2),"e6; away:",round(D.dse[mv&(np.sign(D.dlt)!=np.sign(D.disc71))].sum()/1e6,2),"e6")
        print("dev: true total vs pub, mean |pub-true| per row", round((D.pub-D.pt).abs().sum()/D.n.sum(),2), "| v71 |disc| per row", round(D.disc71.abs().sum()/D.n.sum(),2))
