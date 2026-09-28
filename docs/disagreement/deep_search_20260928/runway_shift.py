import pandas as pd, numpy as np, glob
cols=['MVT_ID_mvt','PHASE_mvt','ADEP_mvt','ADES_mvt','MVT_TIME_UTC_mvt','RUNWAY_mvt','STAND_mvt','TAXITIME_SEC_mvt']
tr=pd.concat([pd.read_parquet(p,columns=cols) for p in sorted(glob.glob('data/raw/training_*.parquet'))])
rk=pd.read_parquet('data/raw/ranking.parquet',columns=cols)
x=pd.concat([tr,rk]).drop_duplicates(['MVT_ID_mvt','PHASE_mvt'])
x['month']=pd.to_datetime(x.MVT_TIME_UTC_mvt).dt.strftime('%Y-%m')
x['airport']=np.where(x.PHASE_mvt.eq('DEP'),x.ADEP_mvt,x.ADES_mvt)
# runway share per airport-month-phase; flag ranking months where a runway's share is outside 2025 [min,max] by > 5 points
out=[]
for (ap,ph),g in x.groupby(['airport','PHASE_mvt']):
    s=g.groupby(['month','RUNWAY_mvt']).size().unstack(fill_value=0); s=s.div(s.sum(1),axis=0)*100
    tr_=s.loc[[m for m in s.index if m.startswith('2025')]]
    for m in ['2026-01','2026-07']:
        if m not in s.index: continue
        for rw in s.columns:
            v=s.loc[m,rw]; lo,hi=tr_[rw].min(),tr_[rw].max()
            if v>hi+5 or v<lo-5: out.append((ap,ph,m,rw,round(v,1),round(lo,1),round(hi,1)))
print(pd.DataFrame(out,columns=['airport','phase','month','runway','share_2026','min_2025','max_2025']).to_string())
# arrival taxi-in by runway at LIRF: July 2025 vs July 2026
l=x[(x.airport=='LIRF')&x.PHASE_mvt.eq('ARR')&x.month.isin(['2025-07','2025-08','2026-07'])]
print(l.groupby(['month','RUNWAY_mvt']).TAXITIME_SEC_mvt.agg(['size','median',lambda s:s.quantile(.9)]).round(0).to_string())
d=x[(x.airport=='LIRF')&x.PHASE_mvt.eq('DEP')&x.month.isin(['2025-06','2025-07','2025-08','2025-09','2026-07'])]
print(d.groupby(['month','RUNWAY_mvt']).size().unstack(fill_value=0).to_string())
