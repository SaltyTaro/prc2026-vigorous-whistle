"""'What's left' audit on v71's development residuals (exploration; nothing declared). Month-blocked LightGBM per feature group,
correction 0.5 x prediction capped +-120 s, floor -12, fixed rows held; complete-month RMSE change vs v71 with paired-day intervals."""
import sys, json, time, numpy as np, pandas as pd, lightgbm as lgb
A=pd.read_parquet(sys.argv[1]); out=sys.argv[2]
MONTHS=['2025-01','2025-07','2025-11','2025-12']; EQ=['2025-01','2025-07','2025-12']
def interval(s,new,rng,reps=2000):
    e0=((s.label-s.ship)**2).to_numpy(); e1=(s.label.to_numpy()-new)**2
    ud,inv=np.unique(s.day.to_numpy(),return_inverse=True); s0,s1,n=np.bincount(inv,e0),np.bincount(inv,e1),np.bincount(inv)
    o=[]
    for _ in range(reps):
        k=np.bincount(rng.integers(0,len(ud),len(ud)),minlength=len(ud)); o.append(100*(np.sqrt((k*s1).sum()/(k*n).sum())/np.sqrt((k*s0).sum()/(k*n).sum())-1))
    lo,hi=np.percentile(o,[2.5,97.5]); return 100*(np.sqrt(e1.mean())/np.sqrt(e0.mean())-1),lo,hi
S=[c for c in A.columns if c.startswith('s_')]
NB=['nx','nxk','nn','x_nmerr']; QU=['gp','gn','gp2','gp4','gn4','bpos','blen']; CL=['c_sched','c_aobt','c_lobt','c_eobt','c_iobt','c_arvt','c_hour','c_dow','c_ship_m_value']
BASE=['kind','ap']
G={'all':S+NB+QU+CL+BASE,'stacker':S+BASE,'no_stacker':NB+QU+CL+BASE,'nbr':NB+BASE,'clocks':CL+BASE,'queue':QU+BASE,'base_only':BASE}
P=dict(objective='regression',learning_rate=0.03,num_leaves=31,min_data_in_leaf=800,lambda_l2=50,feature_fraction=0.7,bagging_fraction=0.7,bagging_freq=1,verbose=-1,seed=7,num_threads=16,max_bin=127)
y=np.clip((A.label-A.ship).to_numpy(),-1800,1800); nf=~A.fixed.to_numpy()
res={}
for g,F in G.items():
    t0=time.time(); new=A.ship.to_numpy(float).copy(); imp=np.zeros(len(F))
    for m in MONTHS:
        f=nf&(A.month!=m).to_numpy(); t=nf&(A.month==m).to_numpy()
        mdl=lgb.train(P,lgb.Dataset(A.loc[f,F].astype(np.float32),y[f],categorical_feature=['kind','ap']),400)
        p=mdl.predict(A.loc[t,F].astype(np.float32)); new[t]=np.maximum(new[t]+np.clip(0.5*p,-120,120),-12)
        imp+=mdl.feature_importance('gain')
    rng=np.random.default_rng(20261004); r={}
    for m in MONTHS:
        s=A[A.month==m]; r[m]=interval(s,new[(A.month==m).to_numpy()],rng)
    eq=float(np.mean([r[m][0] for m in EQ]))
    res[g]={'months':{m:[float(v) for v in r[m]] for m in MONTHS},'eq':eq,'secs':time.time()-t0,
            'top':sorted(zip(F,(imp/imp.sum()).round(4).tolist()),key=lambda z:-z[1])[:25]}
    # per-airport SSE change
    A['_n']=new; res[g]['airport_month_dsse_M']=((A.label-A._n)**2-(A.label-A.ship)**2).groupby([A.airport,A.month]).sum().div(1e6).round(2).unstack().to_dict()
    print(g,' '.join(f'{m[5:]} {r[m][0]:+.4f}[{r[m][1]:+.3f},{r[m][2]:+.3f}]' for m in MONTHS),f'eq {eq:+.4f}',round(time.time()-t0),'s',flush=True)
    json.dump(res,open(out,'w'),indent=1)
