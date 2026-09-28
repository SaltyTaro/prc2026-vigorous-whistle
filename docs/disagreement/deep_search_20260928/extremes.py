import pandas as pd, numpy as np, glob
ID='MVT_ID_mvt'
f=pd.read_parquet(r'SP/control_rows.parquet'.replace('SP',__import__('os').path.dirname(__file__)))
f['r']=f.label-f.control
raw=pd.concat([pd.read_parquet(p) for p in sorted(glob.glob('data/raw/training_*.parquet'))])
raw=raw[raw.PHASE_mvt.eq('DEP')].drop_duplicates(ID)
x=f[f.r.abs()>3600].merge(raw,on=ID,how='left')
t=lambda c: pd.to_datetime(x[c])
x['sched_int']=(t('MVT_TIME_UTC_mvt')-t('SCHED_TIME_UTC_mvt')).dt.total_seconds()
x['proxy']=(t('MVT_TIME_UTC_mvt')-t('AOBT_3_flt')).dt.total_seconds()
x['eobt_int']=(t('MVT_TIME_UTC_mvt')-t('EOBT_1_flt')).dt.total_seconds()
x['block_minus_sched']=(t('BLOCK_TIME_UTC_mvt')-t('SCHED_TIME_UTC_mvt')).dt.total_seconds()
x['nm']=x.FLIGHT_ID_mvt.notna()
pd.set_option('display.width',250)
cols=['month','airport','label','control','v9','r','matched','nm','sched_int','proxy','eobt_int','block_minus_sched','AIRCRAFT_TYPE_mvt','FLIGHT_TYPE_flt','AIRCRAFT_OPERATOR_flt','STAND_mvt']
print(x.sort_values('r')[cols].round(0).to_string())
# classes
x['cls']=np.select([x.label>80000, (x.block_minus_sched.abs()<=5), x.proxy<0, ~x.nm],['one_day','sched_fill','proxy_neg','no_nm'],'other')
print(x.groupby(['month','cls']).agg(n=('r','size'),sse=('r',lambda s:(s**2).sum())).assign(sse=lambda d:d.sse/1e6).round(0).to_string())
x.to_parquet(__import__('os').path.dirname(__file__)+'/extremes.parquet')
