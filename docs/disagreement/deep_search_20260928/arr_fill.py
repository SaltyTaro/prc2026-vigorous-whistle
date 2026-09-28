import pandas as pd, numpy as np, glob
cols=['MVT_ID_mvt','PHASE_mvt','ADEP_mvt','ADES_mvt','MVT_TIME_UTC_mvt','BLOCK_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','TAXITIME_SEC_mvt','FLIGHT_ID_mvt']
tr=pd.concat([pd.read_parquet(p,columns=cols) for p in sorted(glob.glob('data/raw/training_*.parquet'))]); tr['set']='train'
rk=pd.read_parquet('data/raw/ranking.parquet',columns=cols); rk['set']='rank'
x=pd.concat([tr,rk]).drop_duplicates(['MVT_ID_mvt','PHASE_mvt'])
t=lambda c: pd.to_datetime(x[c])
x['month']=t('MVT_TIME_UTC_mvt').dt.strftime('%Y-%m')
x['airport']=np.where(x.PHASE_mvt.eq('DEP'),x.ADEP_mvt,x.ADES_mvt)
x['bms']=(t('BLOCK_TIME_UTC_mvt')-t('SCHED_TIME_UTC_mvt')).dt.total_seconds()
x['fill']=x.bms.abs()<=5
x['has_block']=x.BLOCK_TIME_UTC_mvt.notna()
x['nonm']=x.FLIGHT_ID_mvt.isna()
a=x[x.PHASE_mvt.eq('ARR')]
d=x[x.PHASE_mvt.eq('DEP')]
A=a.groupby(['airport','month']).agg(arr_n=('fill','size'),arr_block=('has_block','mean'),arr_fill=('fill','mean'),arr_taxi_med=('TAXITIME_SEC_mvt','median'),
      arr_taxi_p90=('TAXITIME_SEC_mvt',lambda s:s.quantile(.9)), arr_nonm=('nonm','mean'))
D=d.groupby(['airport','month']).agg(dep_n=('fill','size'),dep_block=('has_block','mean'),dep_fill=('fill','mean'),dep_nonm=('nonm','mean'))
M=A.join(D)
pd.set_option('display.width',250)
print((M.assign(arr_fill=M.arr_fill*100,dep_fill=M.dep_fill*100,arr_nonm=M.arr_nonm*100,dep_nonm=M.dep_nonm*100)).round(2).to_string())
