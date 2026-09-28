import sys; from pathlib import Path
sys.path.insert(0,'src'); sys.path.insert(0,'tools')
import pandas as pd, numpy as np
from run_queue_trust import offsets_all
ID='MVT_ID_mvt'
off=offsets_all(); print('offsets',{k:round(v,1) for k,v in off.items()})
f=pd.read_parquet(sys.argv[1])
cols=[ID,'takeoff','gate','motion_start','taxi_path_m','taxi_stops','taxi_stopped_s','points']
def gate(month):
    m=pd.read_parquet(f'data/external/adsblol/matches_v11/matches2_{month}.parquet',columns=cols)
    return m[m.gate&m.motion_start.notna()]
g=pd.concat([f[f.month==mo].merge(gate(mo),on=ID) for mo in ['2025-01','2025-07','2025-11','2025-12']])
g['obs']=g.takeoff-g.motion_start+g.airport.map(off)
g['gap']=g.obs-g.control
g['good']=(g.taxi_stops>=3)&(g.taxi_path_m>=2000)&(g.taxi_stopped_s<=0.75*(g.takeoff-g.motion_start))
print('gate rows',len(g))
bins=[-1e9,-600,-300,-150,150,300,600,1200,1e9]
for q,s in [('all',g),('good',g[g.good]),('bad',g[~g.good])]:
    s=s.assign(b=pd.cut(s.gap,bins))
    def st(h):
        d=h.obs-h.control; y=h.label-h.control
        return pd.Series({'n':len(h),'ls_w_obs':float((y*d).sum()/(d*d).sum()),'rmse_ctrl':float(np.sqrt(((h.label-h.control)**2).mean())),'rmse_obs':float(np.sqrt(((h.label-h.obs)**2).mean())),
                          'months':h.month.nunique()})
    print(q); print(s.groupby('b',observed=True).apply(st).round(3).to_string())
    for mo,h in s[s.gap>600].groupby('month'):
        d=h.obs-h.control; y=h.label-h.control; print('   ',q,mo,'gap>600 n',len(h),'w',round(float((y*d).sum()/(d*d).sum()),3))
