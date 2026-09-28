import pandas as pd, numpy as np, glob
ID='MVT_ID_mvt'
cols=[ID,'PHASE_mvt','ADEP_mvt','MVT_TIME_UTC_mvt','AOBT_3_flt','TAXITIME_SEC_mvt']
raw=pd.concat([pd.read_parquet(p,columns=cols) for p in sorted(glob.glob('data/raw/training_*.parquet'))])
raw=raw[raw.PHASE_mvt.eq('DEP')].drop_duplicates(ID).copy()
raw['t']=pd.to_datetime(raw.MVT_TIME_UTC_mvt); raw['proxy']=(raw.t-pd.to_datetime(raw.AOBT_3_flt)).dt.total_seconds()
raw['hour']=raw.t.dt.floor('h'); raw['airport']=raw.ADEP_mvt; raw['month']=raw.t.dt.strftime('%Y-%m')
ok=raw.proxy.between(0,7200)
p=raw[ok].groupby(['airport','month','hour']).proxy.agg(peer_mean='mean',peer_n='size').reset_index()
p['peer_c']=p.peer_mean-p.groupby(['airport','month']).peer_mean.transform('median')
raw=raw.merge(p,on=['airport','month','hour'],how='left')
cv=pd.concat([pd.read_parquet(f'scratch/v24/{m}.parquet',columns=[ID,'v9']) for m in [f'2025-{i:02d}' for i in range(1,13)]])
raw=raw.merge(cv,on=ID)
raw['r']=raw.TAXITIME_SEC_mvt-raw.v9; raw['noclock']=raw.proxy.isna()
x=raw[(raw.peer_n>=5)&raw.airport.ne('LIRF')]
x=x.assign(b=pd.cut(x.peer_c,[-1e9,0,300,600,1200,1e9]))
recipe=['2025-01','2025-07','2025-11','2025-12']
for nm,s in [('no-clock',x[x.noclock]),('clock',x[~x.noclock])]:
    for part,ss in [('other 8 months',s[~s.month.isin(recipe)]),('recipe 4 months',s[s.month.isin(recipe)])]:
        print(nm,part); print(ss.groupby('b',observed=True).agg(n=('r','size'),mean_res_v9=('r','mean'),months=('month','nunique')).round(1).to_string())
    t=s[s.peer_c>600].groupby('month').agg(n=('r','size'),mean_res=('r','mean')).round(0)
    print(nm,'peer_c>600 per month:',t.to_dict('index'))
