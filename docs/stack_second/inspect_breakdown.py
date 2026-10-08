import numpy as np, pandas as pd
R=""
ID="MVT_ID_mvt"
T=pd.read_parquet(R+"dev_table.parquet",columns=[ID,"month","airport","day","label","v71","kind","nn","c_layers"])
K=pd.read_parquet(R+"rank_table.parquet",columns=[ID,"month","airport","v71","kind","nn"])
P=pd.read_parquet(R+"preds_dev.parquet").set_index(ID)["sp_hicov7"].reindex(T[ID]).to_numpy()
Q=pd.read_parquet(R+"preds_rank.parquet").set_index(ID)["sp_hicov7"].reindex(K[ID]).to_numpy()
def apply(v,p):
    n=v.copy(); m=~np.isnan(p); n[m]=np.maximum(v[m]+np.clip(0.5*p[m],-120,120),-12); return n
T["d"]=apply(T.v71.to_numpy(float),P)-T.v71; K["d"]=apply(K.v71.to_numpy(float),Q)-K.v71
r=(T.label-T.v71).to_numpy(); d=T.d.to_numpy()
T["gain"]=r**2-(r-d)**2; T["S"]=d**2
T["cap"]=np.abs(d)>=119.5; K["cap"]=np.abs(K.d)>=119.5
mv=T.d!=0; mk=K.d!=0
print("dev moved",mv.sum(),"rank moved",mk.sum())
g=T[mv].groupby(["airport","kind"]).agg(n=("d","size"),mean=("d","mean"),rms=("d",lambda x:np.sqrt((x**2).mean())),cap=("cap","mean"),gain=("gain","sum"),S=("S","sum"))
g["g/S"]=g.gain/g.S
h=K[mk].groupby(["airport","kind"]).agg(nk=("d","size"),meank=("d","mean"),rmsk=("d",lambda x:np.sqrt((x**2).mean())),capk=("cap","mean"))
x=g.join(h); x["ratio"]=x.rmsk/x.rms
pd.set_option("display.width",250); pd.set_option("display.max_rows",200)
print(x.round(3))
print("by kind"); 
gk=T[mv].groupby("kind").agg(n=("d","size"),rms=("d",lambda x:np.sqrt((x**2).mean())),gain=("gain","sum"),S=("S","sum")); gk["g/S"]=gk.gain/gk.S
hk=K[mk].groupby("kind").agg(nk=("d","size"),rmsk=("d",lambda x:np.sqrt((x**2).mean())),Sk=("d",lambda x:(x**2).sum())); print(gk.join(hk).round(3))
print("by month dev g/S", (T[mv].groupby("month").gain.sum()/T[mv].groupby("month").S.sum()).round(3).to_dict())
# concentration
o=np.argsort(-np.abs(d[mv.to_numpy()])); gg=T.gain.to_numpy()[mv.to_numpy()][o]; SS=T.S.to_numpy()[mv.to_numpy()][o]
for q in [0.001,0.01,0.05]:
    k=int(len(o)*q); print(f"top {q:.1%} |d| rows: share of S {SS[:k].sum()/SS.sum():.3f}, share of gain {gg[:k].sum()/gg.sum():.3f}, g/S {gg[:k].sum()/SS[:k].sum():.2f}")
print("cap rows dev", T.cap.sum(), "gain", T.gain[T.cap].sum()/1e6, "S", T.S[T.cap].sum()/1e6, "| rank cap rows", K.cap.sum(), "S", (K.d[K.cap]**2).sum()/1e6)
# sign and size distribution compare
for nm,dd in [("dev",d[mv.to_numpy()]),("rank",K.d.to_numpy()[mk.to_numpy()])]:
    print(nm, "quantiles", np.round(np.quantile(dd,[0.001,0.01,0.1,0.5,0.9,0.99,0.999]),1), "mean",round(dd.mean(),2), "pos share", round((dd>0).mean(),3))
# airport-day mean change
T["date"]=T.day; 
ad=T[mv].groupby(["airport","date"]).d.mean()
K2=K.copy()
print("dev airport-day mean change: sd", round(ad.std(),2), "abs max", round(ad.abs().max(),2))
print(x[["n","gain","S","g/S","nk","ratio"]].assign(gain=lambda z:(z.gain/1e6).round(2),S=lambda z:(z.S/1e6).round(2)).round(2).to_string())
ga=T[mv].groupby(["airport","month"]).agg(gain=("gain","sum"),S=("S","sum")); ga["g/S"]=ga.gain/ga.S
print((ga.assign(gain=ga.gain/1e6,S=ga.S/1e6)).round(2).unstack("month").to_string())
# ranking S per airport-month
K["S"]=K.d**2
print((K[mk].groupby(["airport","month"]).S.sum()/1e6).round(2).unstack().to_string())
