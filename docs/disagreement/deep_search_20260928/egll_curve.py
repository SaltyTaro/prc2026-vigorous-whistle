import numpy as np, pandas as pd
ID='MVT_ID_mvt'
cols=[ID,'takeoff','gate','motion_start','taxi_path_m','points','first_dist_stand_m','slow_start_s']
def gate(month):
    m=pd.read_parquet(f'data/external/adsblol/matches_v11/matches2_{month}.parquet',columns=cols)
    m=m[m.gate&m.motion_start.notna()]
    return m.assign(obs=m.takeoff-m.motion_start)
fair=pd.read_parquet('runs/fair_dev/rows.parquet',columns=[ID,'month','airport','label','v9','v33'])
dev=pd.concat([fair[fair.month==mo].merge(gate(mo),on=ID) for mo in ['2025-01','2025-07','2025-11','2025-12']])
r=pd.read_parquet('runs/adsb_ctx/export_v30/ranking_values.parquet',columns=[ID,'airport','month','v9'])
v35=pd.read_parquet('runs/submissions/vigorous-whistle_v35.parquet').rename(columns={'TAXITIME_SEC_mvt':'pred'})
r=r.merge(v35,on=ID)
rk=pd.concat([r[r.month==mo].merge(gate(mo),on=ID) for mo in ['2026-01','2026-07']])
bins=[0,300,450,600,750,900,1200,1800,1e9]
out=[]
for ap in sorted(rk.airport.unique()):
    d=dev[dev.airport==ap]; k=rk[rk.airport==ap]
    if len(d)<200: continue
    d=d.assign(b=pd.cut(d.obs,bins)); 
    rel=d.groupby('b',observed=True).agg(n=('label','size'),lab_obs=('label',lambda s: 0)).n
    dl=(d.label-d.obs).groupby(d.b,observed=True).median(); dp=(d.v33-d.obs).groupby(d.b,observed=True).median()
    for mo in ['2026-01','2026-07']:
        kk=k[k.month==mo]
        if len(kk)<200: continue
        kk=kk.assign(b=pd.cut(kk.obs,bins))
        kp=(kk.pred-kk.obs).groupby(kk.b,observed=True).median(); kn=kk.groupby('b',observed=True).size()
        gap=(kp-dl).reindex(kn.index)
        w=kn.reindex(gap.index).fillna(0)
        out.append({'airport':ap,'month':mo,'rows':len(kk),'dev_rows':len(d),'weighted_gap_pred_vs_dev_label_rel':float(np.nansum(gap*w)/w[gap.notna()].sum()),
                    'dev_pred_vs_label_rel':float(np.nansum((dp-dl).reindex(kn.index)*w)/w[gap.notna()].sum())})
        if ap=='EGLL' or abs(out[-1]['weighted_gap_pred_vs_dev_label_rel'])>40:
            print(ap,mo,pd.DataFrame({'dev_n':d.groupby('b',observed=True).size(),'dev_label-obs':dl,'dev_pred-obs':dp,'rank_n':kn,'rank_pred-obs':kp}).round(0).to_string())
print(pd.DataFrame(out).round(1).to_string())
print('EGLL dev per month rows', dev[dev.airport=='EGLL'].month.value_counts().to_dict())
e=rk[(rk.airport=='EGLL')]
print('EGLL rank gate rows by month', e.month.value_counts().to_dict())
for mo,d in [('dev',dev[dev.airport=='EGLL']),('2026-01',e[e.month=='2026-01']),('2026-07',e[e.month=='2026-07'])]:
    print(mo, d[['obs','taxi_path_m','points','first_dist_stand_m','slow_start_s']].median().round(0).to_dict())
