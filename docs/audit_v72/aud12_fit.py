"""Adjacency test, remaining airports (aud12) (exploration, single seed): is the late passes' November / December gain a time-adjacency effect? For an airport set,
fit the raw-input pass (aud6's inputs) on ONE development month and predict another; compare adjacent pairs (Nov -> Dec, Dec -> Nov)
with distant ones (Jan -> Dec, Jul -> Dec, Jan -> Nov, Jul -> Nov, Dec -> Jan, Nov -> Jan, Dec -> Jul, Nov -> Jul). v73 base,
0.5 x capped 120 s, fixed rows held; change of the target month's complete RMSE with the paired-day interval."""
import sys, json, numpy as np, pandas as pd, lightgbm as lgb
d=sys.argv[1]; out=sys.argv[2]; ID='MVT_ID_mvt'
A=pd.read_parquet(d+'/dev_t3.parquet'); P3=pd.read_parquet(d+'/preds_dev.parquet').set_index(ID)['t3_ro_3'].reindex(A[ID]).to_numpy()
v72=A.ship.to_numpy(float).copy(); v73=v72.copy(); m=~np.isnan(P3); v73[m]=np.maximum(v72[m]+np.clip(0.5*P3[m],-120,120),-12)
A['p3']=P3; A['d3']=v73-v72; A['ship']=v73
def interval(s,new,rng,reps=2000):
    e0=((s.label-s.ship)**2).to_numpy(); e1=(s.label.to_numpy()-new)**2
    ud,inv=np.unique(s.day.to_numpy(),return_inverse=True); s0,s1,n=np.bincount(inv,e0),np.bincount(inv,e1),np.bincount(inv)
    o=[]
    for _ in range(reps):
        k=np.bincount(rng.integers(0,len(ud),len(ud)),minlength=len(ud)); o.append(100*(np.sqrt((k*s1).sum()/(k*n).sum())/np.sqrt((k*s0).sum()/(k*n).sum())-1))
    lo,hi=np.percentile(o,[2.5,97.5]); return 100*(np.sqrt(e1.mean())/np.sqrt(e0.mean())-1),lo,hi
RAW=[c for c in A.columns if c.startswith(('adsb_','sfc_','p_','m_'))]
CL=['c_sched','c_aobt','c_lobt','c_eobt','c_iobt','c_arvt','c_hour','c_dow','c_layers']; BASE=['kind','ap']
F=RAW+['p2','d2','p3','d3','ship']+CL+BASE
SETS={'eddm':['EDDM'],'lebl':['LEBL'],'lemd':['LEMD'],'lfpg':['LFPG'],'ltfm':['LTFM'],'lirf':['LIRF'],'r3':['LFPG','LTFM','LIRF'],'o4r3':['EDDM','EGLL','LEBL','LEMD','LFPG','LTFM','LIRF'],'all10':['EDDF','EDDM','EGLL','EHAM','LEBL','LEMD','LFPG','LIRF','LSZH','LTFM']}
PAIRS=[('2025-11','2025-12'),('2025-12','2025-11'),('2025-12','2025-01')]
PR=dict(objective='regression',learning_rate=0.03,num_leaves=31,min_data_in_leaf=800,lambda_l2=50,feature_fraction=0.7,bagging_fraction=0.7,bagging_freq=1,verbose=-1,seed=7,num_threads=16,max_bin=127)
y=np.clip((A.label-A.ship).to_numpy(),-1800,1800); nf=(A.kind!=3).to_numpy(); ap=A.airport.to_numpy(); res={}
for sn,aps in SETS.items():
    inset=nf&np.isin(ap,aps)
    for tr,te in PAIRS:
        f=inset&(A.month==tr).to_numpy(); t=inset&(A.month==te).to_numpy()
        mdl=lgb.train(PR,lgb.Dataset(A.loc[f,F].astype(np.float32),y[f],categorical_feature=['kind','ap']),400)
        s=(A.month==te).to_numpy(); new=A.ship.to_numpy(float).copy()
        new[t]=np.maximum(new[t]+np.clip(0.5*mdl.predict(A.loc[t,F].astype(np.float32)),-120,120),-12)
        pct,lo,hi=interval(A[s],new[s],np.random.default_rng(20261004))
        dS=float(((A.label[t]-new[t])**2-(A.label[t]-A.ship[t])**2).sum())
        res[f'{sn} {tr}->{te}']={'pct':pct,'lo':lo,'hi':hi,'dsse':dS}
        print(f'{sn:5s} {tr[5:]}->{te[5:]} {pct:+.4f} [{lo:+.3f},{hi:+.3f}] dSSE {dS/1e6:+.2f}e6',flush=True)
        json.dump(res,open(out,'w'),indent=1)
