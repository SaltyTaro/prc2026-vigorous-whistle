import pandas as pd, numpy as np, glob, warnings; warnings.filterwarnings('ignore')
ID='MVT_ID_mvt'
cols=[ID,'PHASE_mvt','ADEP_mvt','ADES_mvt','MVT_TIME_UTC_mvt','TAXITIME_SEC_mvt']
tr=pd.concat([pd.read_parquet(p,columns=cols) for p in sorted(glob.glob('data/raw/training_*.parquet'))])
rk=pd.read_parquet('data/raw/ranking.parquet',columns=cols)
x=pd.concat([tr,rk]).drop_duplicates([ID,'PHASE_mvt'])
x['month']=pd.to_datetime(x.MVT_TIME_UTC_mvt).dt.strftime('%Y-%m')
x['airport']=np.where(x.PHASE_mvt.eq('DEP'),x.ADEP_mvt,x.ADES_mvt)
a=x[x.PHASE_mvt.eq('ARR')].groupby(['airport','month']).TAXITIME_SEC_mvt.agg(arr_mean=lambda s: s.clip(upper=3600).mean(), arr_med='median')
d=x[x.PHASE_mvt.eq('DEP')]
v35=pd.read_parquet('runs/submissions/vigorous-whistle_v35.parquet').rename(columns={'TAXITIME_SEC_mvt':'v35'})
d=d.merge(v35,on=ID,how='left')
cv=pd.concat([pd.read_parquet(f'scratch/v24/{m}.parquet',columns=[ID,'v9']) for m in [f'2025-{i:02d}' for i in range(1,13)]])
d=d.merge(cv,on=ID,how='left')
d['y']=np.where(d.month.str.startswith('2025'), d.TAXITIME_SEC_mvt, d.v35)   # label in 2025, v35 prediction in 2026
D=d.groupby(['airport','month']).agg(dep=('y',lambda s: s.clip(upper=3600).mean()), dep_med=('y','median'), v9cf=('v9',lambda s: s.clip(upper=3600).mean())).join(a)
out=[]
for ap,g in D.groupby(level=0):
    g=g.droplevel(0); tr_=g.loc[[m for m in g.index if m.startswith('2025')]]
    b=np.polyfit(tr_.arr_mean, tr_.dep, 1)
    for m in ['2026-01','2026-07']:
        same=m.replace('2026','2025')
        out.append({'airport':ap,'month':m,'arr_mean_2026':g.loc[m,'arr_mean'],'arr_mean_2025_same':g.loc[same,'arr_mean'],
                    'dep_label_2025_same':g.loc[same,'dep'],'v35_mean_2026':g.loc[m,'dep'],'slope_dep_on_arr':b[0],
                    'fit_dep_2026_from_arr':np.polyval(b,g.loc[m,'arr_mean']),'gap_fit_minus_v35':np.polyval(b,g.loc[m,'arr_mean'])-g.loc[m,'dep'],
                    'r_2025':np.corrcoef(tr_.arr_mean,tr_.dep)[0,1],'cv_v9_resid_2025_same':g.loc[same,'dep']-g.loc[same,'v9cf']})
pd.set_option('display.width',250)
print(pd.DataFrame(out).round(1).to_string())
