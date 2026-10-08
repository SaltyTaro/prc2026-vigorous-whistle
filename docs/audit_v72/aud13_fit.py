"""W25 follow-up on v74's development values (exploration, single seed): per airport, fit on ONE W25 month and read the other
(Nov->Dec, Dec->Nov), plus the cross-season Dec->Jan 2025; input sets: 'stk' = stacker inputs + neighbour + queue + clocks + chain columns,
'all' = stk + raw ADS-B, 'raw_reg' = raw (as the season pass) regularised. 0.5 x capped 120 s, fixed rows held."""
import sys, json, numpy as np, pandas as pd, lightgbm as lgb
d=sys.argv[1]; out=sys.argv[2]; ID='MVT_ID_mvt'
A=pd.read_parquet(d+'/dev_w25.parquet')
def interval(s,new,rng,reps=2000):
    e0=((s.label-s.ship)**2).to_numpy(); e1=(s.label.to_numpy()-new)**2
    ud,inv=np.unique(s.day.to_numpy(),return_inverse=True); s0,s1,n=np.bincount(inv,e0),np.bincount(inv,e1),np.bincount(inv)
    o=[]
    for _ in range(reps):
        k=np.bincount(rng.integers(0,len(ud),len(ud)),minlength=len(ud)); o.append(100*(np.sqrt((k*s1).sum()/(k*n).sum())/np.sqrt((k*s0).sum()/(k*n).sum())-1))
    lo,hi=np.percentile(o,[2.5,97.5]); return 100*(np.sqrt(e1.mean())/np.sqrt(e0.mean())-1),lo,hi
RAW=[c for c in A.columns if c.startswith(('adsb_','sfc_','p_','m_'))]; STK=[c for c in A.columns if c.startswith('s_')]
NBQ=['nx','nxk','nn','x_nmerr','gp','gn','gp2','gp4','gn4','bpos','blen']
CL=['c_sched','c_aobt','c_lobt','c_eobt','c_iobt','c_arvt','c_hour','c_dow','c_layers']; BASE=['kind','ap']; CH=['p2','d2','p3','d3','p4','d4','ship']
FS={'stk':STK+NBQ+CL+CH+BASE,'all':STK+NBQ+RAW+CL+CH+BASE,'raw_reg':RAW+CL+CH+BASE}
REG=dict(num_leaves=15,min_data_in_leaf=2000,lambda_l2=200); NR={'stk':400,'all':400,'raw_reg':200}
PR=dict(objective='regression',learning_rate=0.03,num_leaves=31,min_data_in_leaf=800,lambda_l2=50,feature_fraction=0.7,bagging_fraction=0.7,bagging_freq=1,verbose=-1,seed=7,num_threads=16,max_bin=127)
AIR=sys.argv[3].split(',') if len(sys.argv)>3 else ['EDDF','EDDM','EGLL','EHAM','LEBL','LEMD','LFPG','LIRF','LSZH','LTFM']
SETS=sys.argv[4].split(',') if len(sys.argv)>4 else ['stk','all','raw_reg']
PAIRS=[('2025-11','2025-12'),('2025-12','2025-11'),('2025-12','2025-01')]
y=np.clip((A.label-A.ship).to_numpy(),-1800,1800); nf=(A.kind!=3).to_numpy(); ap=A.airport.to_numpy(); res={}
for a in AIR:
    for fs in SETS:
        F=FS[fs]; inset=nf&(ap==a); line=[]
        for tr,te in PAIRS:
            f=inset&(A.month==tr).to_numpy(); t=inset&(A.month==te).to_numpy()
            mdl=lgb.train({**PR,**(REG if fs=='raw_reg' else {})},lgb.Dataset(A.loc[f,F].astype(np.float32),y[f],categorical_feature=['kind','ap']),NR[fs])
            s=(A.month==te).to_numpy(); new=A.ship.to_numpy(float).copy()
            new[t]=np.maximum(new[t]+np.clip(0.5*mdl.predict(A.loc[t,F].astype(np.float32)),-120,120),-12)
            pct,lo,hi=interval(A[s],new[s],np.random.default_rng(20261004))
            dS=float(((A.label[t]-new[t])**2-(A.label[t]-A.ship[t])**2).sum())
            res[f'{a} {fs} {tr}->{te}']={'pct':pct,'lo':lo,'hi':hi,'dsse':dS}; line.append(f'{tr[5:]}->{te[5:]} {pct:+.4f}[{lo:+.3f},{hi:+.3f}] {dS/1e6:+.2f}e6')
        print(f'{a} {fs:7s} '+' | '.join(line),flush=True)
        json.dump(res,open(out,'w'),indent=1)
