import pandas as pd, numpy as np, glob
ID='MVT_ID_mvt'
raw=pd.concat([pd.read_parquet(p) for p in sorted(glob.glob('data/raw/training_*.parquet'))])
raw=raw[raw.PHASE_mvt.eq('DEP')].drop_duplicates(ID)
t=lambda d,c: pd.to_datetime(d[c])
raw['sched_int']=(t(raw,'MVT_TIME_UTC_mvt')-t(raw,'SCHED_TIME_UTC_mvt')).dt.total_seconds()
raw['label']=raw.TAXITIME_SEC_mvt; raw['nm']=raw.FLIGHT_ID_mvt.notna(); raw['month']=t(raw,'MVT_TIME_UTC_mvt').dt.strftime('%Y-%m')
raw['bms']=(t(raw,'BLOCK_TIME_UTC_mvt')-t(raw,'SCHED_TIME_UTC_mvt')).dt.total_seconds()
raw['airport']=raw.ADEP_mvt
cv=pd.concat([pd.read_parquet(f'scratch/v24/{m}.parquet',columns=[ID,'v9']) for m in [f'2025-{i:02d}' for i in range(1,13)]])
raw=raw.merge(cv,on=ID,how='left')
n=raw[~raw.nm]
print('no-NM departures', len(n), 'of', len(raw))
bins=[-1e9,0,900,1800,3600,7200,14400,43200,1e9]
n=n.assign(b=pd.cut(n.sched_int,bins))
def stats(g):
    return pd.Series({'n':len(g),'label_med':g.label.median(),'v9_med':g.v9.median(),'sched_med':g.sched_int.median(),
        'share_block_within_15min_of_sched':float((g.bms.abs()<=900).mean()),'share_label_gt_half_sched':float((g.label>0.5*g.sched_int).mean()) if g.sched_int.median()>0 else np.nan,
        'rmse_v9':float(np.sqrt(((g.label-g.v9)**2).mean())),'mean_res_v9':float((g.label-g.v9).mean())})
print(n.groupby('b',observed=True).apply(stats).round(2).to_string())
print('\nby airport, no-NM rows with sched_int > 3600 s:')
big=n[n.sched_int>3600]
print(big.groupby('airport').apply(stats).round(2).to_string())
print('\nNM rows sched_int>3600 for comparison:'); nm=raw[raw.nm&(raw.sched_int>3600)]; print(stats(nm).round(2).to_dict())
n.to_parquet(__import__('os').path.dirname(__file__)+'/nonm_train.parquet')
