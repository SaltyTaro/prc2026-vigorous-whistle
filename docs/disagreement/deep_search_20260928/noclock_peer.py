import pandas as pd, numpy as np, glob, sys
ID='MVT_ID_mvt'
f=pd.read_parquet(sys.argv[1]); f['r']=f.label-f.control
cols=[ID,'PHASE_mvt','ADEP_mvt','MVT_TIME_UTC_mvt','AOBT_3_flt','FLIGHT_ID_mvt']
raw=pd.concat([pd.read_parquet(p,columns=cols) for p in sorted(glob.glob('data/raw/training_*.parquet'))])
raw=raw[raw.PHASE_mvt.eq('DEP')].drop_duplicates(ID)
raw['t']=pd.to_datetime(raw.MVT_TIME_UTC_mvt); raw['proxy']=(raw.t-pd.to_datetime(raw.AOBT_3_flt)).dt.total_seconds()
raw['hour']=raw.t.dt.floor('h'); raw['airport']=raw.ADEP_mvt
ok=raw.proxy.between(0,7200)
peer=raw[ok].groupby(['airport','hour']).proxy.agg(peer_mean='mean',peer_n='size').reset_index()
# airport-month typical peer mean for centring
peer['month']=peer.hour.dt.strftime('%Y-%m')
peer['peer_c']=peer.peer_mean-peer.groupby(['airport','month']).peer_mean.transform('median')
x=f.merge(raw[[ID,'hour','proxy','FLIGHT_ID_mvt']],on=ID).merge(peer[['airport','hour','peer_mean','peer_n','peer_c']],on=['airport','hour'],how='left')
x['noclock']=x.proxy.isna()
x=x[x.airport.ne('LIRF')]
for name,s in [('no-clock untracked',x[x.noclock&~x.matched]),('no-clock tracked',x[x.noclock&x.matched]),('clock untracked',x[~x.noclock&~x.matched])]:
    s=s[s.peer_n>=5]
    w=s.peer_c; y=s.r
    print(f'{name:20s} n {len(s):6d}  mean_res {y.mean():7.1f}  slope res~peer_c {float(((w-w.mean())*(y-y.mean())).sum()/((w-w.mean())**2).sum()):6.3f}  corr {np.corrcoef(w,y)[0,1]:.3f}')
    s=s.assign(b=pd.cut(s.peer_c,[-1e9,-120,-60,0,60,120,300,600,1e9]))
    print(s.groupby('b',observed=True).agg(n=('r','size'),mean_res=('r','mean'),month_n=('month','nunique')).round(1).to_string())
    for mo,h in s[s.peer_c>300].groupby('month'): print('    ',mo,'peer_c>300 n',len(h),'mean_res',round(h.r.mean(),1))
