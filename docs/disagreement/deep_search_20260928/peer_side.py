import pandas as pd, numpy as np, glob, sys
ID='MVT_ID_mvt'
f=pd.read_parquet(sys.argv[1])
cols=[ID,'PHASE_mvt','ADEP_mvt','MVT_TIME_UTC_mvt','AOBT_3_flt']
raw=pd.concat([pd.read_parquet(p,columns=cols) for p in sorted(glob.glob('data/raw/training_*.parquet'))])
raw=raw[raw.PHASE_mvt.eq('DEP')].drop_duplicates(ID).copy()
raw['t']=pd.to_datetime(raw.MVT_TIME_UTC_mvt); raw['proxy']=(raw.t-pd.to_datetime(raw.AOBT_3_flt)).dt.total_seconds()
raw['hour']=raw.t.dt.floor('h'); raw['airport']=raw.ADEP_mvt
ok=raw.proxy.between(0,7200)
p=raw[ok].groupby(['airport','hour']).proxy.agg(peer_mean='mean',peer_med='median',peer_n='size').reset_index()
x=f.merge(raw[[ID,'hour','proxy']],on=ID).merge(p,on=['airport','hour'],how='left')
x=x[x.proxy.isna()&(x.peer_n>=5)&x.airport.ne('LIRF')]
for col in ['peer_mean','peer_med']:
    x['D']=x[col]-x.control; x['y']=x.label-x.control
    x['b']=pd.cut(x.D,[-1e9,-300,0,300,600,1200,1e9])
    t=x.groupby(['b','month'],observed=True).apply(lambda h: pd.Series({'n':len(h),'w':float((h.y*h.D).sum()/(h.D**2).sum())})).unstack('month')
    print(col, 'no-clock rows',len(x)); print(t.round(2).to_string())
    for sc,h in [('tracked',x[x.matched&(x.D>300)]),('untracked',x[~x.matched&(x.D>300)])]:
        print('  ',sc,'D>300 n',len(h),'w',round(float((h.y*h.D).sum()/(h.D**2).sum()),3))
