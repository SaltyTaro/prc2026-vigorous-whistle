import pandas as pd, numpy as np, glob
ID='MVT_ID_mvt'
cols=[ID,'PHASE_mvt','ADEP_mvt','MVT_TIME_UTC_mvt','AOBT_3_flt']
def peer_table(raw):
    raw=raw[raw.PHASE_mvt.eq('DEP')].drop_duplicates(ID).copy()
    raw['t']=pd.to_datetime(raw.MVT_TIME_UTC_mvt); raw['proxy']=(raw.t-pd.to_datetime(raw.AOBT_3_flt)).dt.total_seconds()
    raw['hour']=raw.t.dt.floor('h'); raw['airport']=raw.ADEP_mvt; raw['month']=raw.t.dt.strftime('%Y-%m')
    ok=raw.proxy.between(0,7200)
    p=raw[ok].groupby(['airport','month','hour']).proxy.agg(peer_mean='mean',peer_n='size').reset_index()
    p['peer_c']=p.peer_mean-p.groupby(['airport','month']).peer_mean.transform('median')
    raw=raw.merge(p,on=['airport','month','hour'],how='left')
    return raw
tr=peer_table(pd.concat([pd.read_parquet(p,columns=cols) for p in sorted(glob.glob('data/raw/training_*.parquet'))])); tr['noclock']=tr.proxy.isna()
rk=peer_table(pd.read_parquet('data/raw/ranking.parquet',columns=cols))
scope=pd.read_parquet('runs/adsb_ctx/export_v30/ranking_values.parquet',columns=[ID,'scope'])
rk=rk.merge(scope,on=ID)
rk['tracked']=rk.scope.isin(['matched_no_lirf','lirf_nonpop_matched'])
rk['noclock']=rk.proxy.isna()
out=[]
for name,d in [('train',tr),('rank',rk)]:
    d=d[(d.peer_n>=5)&d.airport.ne('LIRF')]
    for mo,g in d.groupby('month'):
        out.append({'set':name,'month':mo,'rows':len(g),'gt300':int((g.peer_c>300).sum()),'gt600':int((g.peer_c>600).sum()),
                    'noclock_gt300':int(((g.peer_c>300)&g.noclock).sum()),'noclock_gt600':int(((g.peer_c>600)&g.noclock).sum())})
t=pd.DataFrame(out); t['share_gt300_%']=(100*t.gt300/t.rows).round(2); t['share_gt600_%']=(100*t.gt600/t.rows).round(2)
print(t.to_string())
r=rk[(rk.peer_n>=5)&rk.airport.ne('LIRF')&(rk.peer_c>600)]
print(r.groupby(['month','airport']).size().unstack(fill_value=0).to_string())
print(r.assign(day=r.hour.dt.strftime('%m-%d')).groupby(['airport','day']).size().sort_values(ascending=False).head(15).to_string())
