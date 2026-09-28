import numpy as np, pandas as pd
ID='MVT_ID_mvt'
def gate(month):
    m=pd.read_parquet(f'data/external/adsblol/matches_v11/matches2_{month}.parquet',columns=[ID,'takeoff','gate','motion_start'])
    m=m[m.gate&m.motion_start.notna()]
    return m.assign(obs=m.takeoff-m.motion_start)[[ID,'obs']]
fair=pd.read_parquet('runs/fair_dev/rows.parquet',columns=[ID,'month','day','airport','matched','label','v9','v33'])
dev=pd.concat([fair[fair.month==mo].merge(gate(mo),on=ID,how='left') for mo in ['2025-01','2025-07','2025-11','2025-12']])
dev['e9']=dev.obs-dev.v9; dev['ep']=dev.obs-dev.v33
gd=dev[dev.obs.notna()].groupby(['airport','month','day']).agg(n_g=('e9','size'),ob9=('e9','median'),obp=('ep','median')).reset_index()
cen=gd.groupby('airport')[['ob9','obp']].median()
gd['d9']=gd.ob9-gd.airport.map(cen.ob9); gd['dp']=gd.obp-gd.airport.map(cen.obp)
un=dev[~dev.matched].assign(r=lambda d:d.label-d.v33)
u=un.groupby(['airport','month','day']).agg(n_u=('r','size'),u_res=('r','mean'),u_sse=('r',lambda s: float((s**2).sum()))).reset_index()
j=gd.merge(u,on=['airport','month','day']); j=j[(j.n_g>=20)&(j.n_u>=10)]
print('dev airport-days',len(j),'airports',sorted(j.airport.unique()))
def slope(x,y,w):
    x=x-np.average(x,weights=w); y=y-np.average(y,weights=w); return float((w*x*y).sum()/(w*x*x).sum())
for mo in ['2025-01','2025-07','2025-11','2025-12','all']:
    s=j if mo=='all' else j[j.month==mo]
    print(mo, 'n',len(s),'slope u_res~d9',round(slope(s.d9,s.u_res,s.n_u),3),'corr',round(float(np.corrcoef(s.d9,s.u_res)[0,1]),3),
          '| slope u_res~dp',round(slope(s.dp,s.u_res,s.n_u),3),'corr',round(float(np.corrcoef(s.dp,s.u_res)[0,1]),3))
print('dev d9 quantiles',gd[gd.n_g>=20].d9.quantile([.01,.05,.5,.95,.99]).round(0).to_dict())
print('top dev days by d9'); print(j.sort_values('d9',ascending=False).head(12)[['airport','day','n_g','d9','dp','n_u','u_res']].round(0).to_string())
# ranking
r=pd.read_parquet('runs/adsb_ctx/export_v30/ranking_values.parquet',columns=[ID,'airport','month','scope','v9'])
v35=pd.read_parquet('runs/submissions/vigorous-whistle_v35.parquet').rename(columns={'TAXITIME_SEC_mvt':'pred'})
tk=pd.read_parquet('data/raw/ranking.parquet',columns=[ID,'MVT_TIME_UTC_mvt'])
r=r.merge(v35,on=ID).merge(tk,on=ID); r['day']=pd.to_datetime(r.MVT_TIME_UTC_mvt).dt.strftime('%Y-%m-%d')
rk=pd.concat([r[r.month==mo].merge(gate(mo),on=ID,how='left') for mo in ['2026-01','2026-07']])
rk['e9']=rk.obs-rk.v9; rk['ep']=rk.obs-rk.pred
rg=rk[rk.obs.notna()].groupby(['airport','month','day']).agg(n_g=('e9','size'),ob9=('e9','median'),obp=('ep','median')).reset_index()
rg['d9']=rg.ob9-rg.airport.map(cen.ob9); rg['dp']=rg.obp-rg.airport.map(cen.obp)
print('scope values',rk.scope.value_counts().to_dict())
ru=rk[rk.obs.isna()].groupby(['airport','month','day']).agg(n_u=('pred','size'),u_lift=('pred',lambda s: 0.0)).reset_index()
tmp=rk[rk.obs.isna()].assign(l=lambda d:d.pred-d.v9).groupby(['airport','month','day']).l.mean().rename('v35_minus_v9').reset_index()
R=rg.merge(ru,on=['airport','month','day']).merge(tmp,on=['airport','month','day']); R=R[R.n_g>=20]
for mo in ['2026-01','2026-07']:
    print(mo,'ranking d9 quantiles',R[R.month==mo].d9.quantile([.01,.05,.5,.95,.99]).round(0).to_dict())
print('share |d9|>dev99', 'dev',float((gd[gd.n_g>=20].d9>gd[gd.n_g>=20].d9.quantile(.99)).mean()),'rank',float((R.d9>gd[gd.n_g>=20].d9.quantile(.99)).mean()))
print('top ranking days by d9'); print(R.sort_values('d9',ascending=False).head(20)[['airport','day','n_g','d9','dp','n_u','v35_minus_v9']].round(0).to_string())
j.to_parquet(f'{__import__("os").path.dirname(__file__)}/daybias_dev.parquet'); R.to_parquet(f'{__import__("os").path.dirname(__file__)}/daybias_rank.parquet')
