"""Screen: do EUROCONTROL's daily ATC pre-departure delay and ATFM slot-adherence series explain
the v33 stack's airport-day residual (2025 development months)? Descriptive only; nothing declared."""
import json, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
D='data/external/eurocontrol_aiu/'
r=pd.read_parquet('runs/lomo_dev/rows.parquet')
r=r[(r.label>0)&(r.label<=7200)].copy(); r['res']=r.label-r.v33
r['date']=pd.to_datetime(r.day)
atc=pd.read_parquet(D+'ATC_Pre-Departure_Delay.parquet'); slot=pd.read_parquet(D+'ATFM_Slot_Adherence.parquet')
allp=pd.read_parquet(D+'All_Pre-Departure_Delay.parquet')
for d in (atc,slot,allp): d['date']=pd.to_datetime(d.FLT_DATE); d['airport']=d.APT_ICAO
f=atc[['date','airport','FLT_DEP_1','FLT_DEP_IFR_2','DLY_ATC_PRE_2','FLT_DEP_3','DLY_ATC_PRE_3']].merge(
  slot[['date','airport','FLT_DEP_REG_1','FLT_DEP_OUT_EARLY_1','FLT_DEP_IN_1','FLT_DEP_OUT_LATE_1']],on=['date','airport'],how='outer').merge(
  allp[['date','airport','DLY_ALL_PRE_2']],on=['date','airport'],how='left')
f['atc_pf']=f.DLY_ATC_PRE_2/f.FLT_DEP_IFR_2; f['atc_share']=f.FLT_DEP_3/f.FLT_DEP_1
f['atc_mean_aff']=f.DLY_ATC_PRE_3/f.FLT_DEP_3; f['reg_share']=f.FLT_DEP_REG_1/f.FLT_DEP_1
f['late_share']=f.FLT_DEP_OUT_LATE_1/f.FLT_DEP_REG_1.clip(lower=1); f['early_share']=f.FLT_DEP_OUT_EARLY_1/f.FLT_DEP_REG_1.clip(lower=1)
f['all_pf']=f.DLY_ALL_PRE_2/f.FLT_DEP_IFR_2
f['atc_over_all']=f.DLY_ATC_PRE_2/f.DLY_ALL_PRE_2.clip(lower=1)
feats=['atc_pf','atc_share','atc_mean_aff','reg_share','late_share','early_share','atc_over_all']
g=r.groupby(['airport','date','month']).agg(n=('res','size'),m=('res','mean'),sse=('res',lambda x:(x**2).sum())).reset_index()
g=g.merge(f[['date','airport']+feats+['all_pf']],on=['date','airport'],how='left')
tot=g.sse.sum(); out={'total_sse_rows':int(len(r))}
res={}
for use_all in (False,True):
  fs=feats+(['all_pf'] if use_all else [])
  gain=0.0; per={}
  for m in sorted(g.month.unique()):
    tr=g[g.month!=m]; te=g[g.month==m]
    sse_m=te.sse.sum(); red=0
    for a,t in te.groupby('airport'):
        x=tr[tr.airport==a].dropna(subset=fs); xt=t.dropna(subset=fs)
        if len(x)<30 or xt.empty: continue
        mu,sd=x[fs].mean(),x[fs].std().replace(0,1)
        mdl=Ridge(alpha=50).fit((x[fs]-mu)/sd,x.m,sample_weight=x.n)
        p=mdl.predict((xt[fs]-mu)/sd)
        # change in SSE of the airport-day rows if every row is shifted by -p (shrunk 0.5)
        for w in (0.5,):
            red+=float((xt.n*(2*w*p*xt.m*1 - (w*p)**2)).sum())  # = sum over rows of res^2 - (res-wp)^2
    per[m]=red/sse_m
  res['with_all_pf' if use_all else 'atc_slot']=per
out['gain_share_of_month_sse(w=.5)']=res
print(json.dumps(out,indent=1)); json.dump(out,open('runs/published_screen/daily_extra.json','w'),indent=1)
# plain correlation of day mean residual with each feature, by airport (pooled sign check)
c=g.dropna(subset=feats).groupby('airport').apply(lambda d:pd.Series({k:np.corrcoef(d[k],d.m)[0,1] for k in feats}))
print(c.round(3).to_string())
