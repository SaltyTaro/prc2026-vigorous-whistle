"""Weather inputs in a late pass on v73's development values (exploration; nothing declared). v73 dev = v72 dev (runs/stack_third/dev_t3.parquet
'ship') + t3_ro_3's out-of-month bag (0.5 x, cap 120, floor -12) at EDDF / EHAM / LSZH. Single-seed month-blocked LightGBM per variant on
clip(label - v73, +-1800), 0.5 x capped 120 s, fixed rows held; complete-month change vs v73 with paired-day intervals."""
import sys, json, time, numpy as np, pandas as pd, lightgbm as lgb
d=sys.argv[1]; out=sys.argv[2]; ID='MVT_ID_mvt'
A=pd.read_parquet(d+'/dev_t3.parquet'); WX=pd.read_parquet(d+'/wx_dev.parquet'); assert (WX[ID].to_numpy()==A[ID].to_numpy()).all(); A=pd.concat([A,WX.drop(columns=[ID])],axis=1); P3=pd.read_parquet(d+'/preds_dev.parquet').set_index(ID)['t3_ro_3'].reindex(A[ID]).to_numpy()
v72=A.ship.to_numpy(float).copy(); v73=v72.copy(); m=~np.isnan(P3); v73[m]=np.maximum(v72[m]+np.clip(0.5*P3[m],-120,120),-12)
A['p3']=P3; A['d3']=v73-v72; A['ship']=v73; A['fixed']=(A.kind==3)
MONTHS=['2025-01','2025-07','2025-11','2025-12']; EQ=['2025-01','2025-07','2025-12']
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
H3=['EDDF','EHAM','LSZH']; O4=['EDDM','EGLL','LEBL','LEMD']; H7=H3+O4
WXC=[c for c in WX.columns if c!=ID]; R3=['LFPG','LTFM','LIRF']; AP=H7+R3
FW=WXC+['p2','d2','p3','d3','ship']+CL+BASE
V={'all10_wx':(FW,AP,AP),'r3_wx':(FW,R3,R3),'ltfm_wx':(FW,['LTFM'],['LTFM']),'lfpg_wx':(FW,['LFPG'],['LFPG']),'lirf_wx':(FW,['LIRF'],['LIRF']),
   'h3_wx_raw':(F+WXC,H3,H3),'h7_wx':(FW,H7,H7)}
PR=dict(objective='regression',learning_rate=0.03,num_leaves=31,min_data_in_leaf=800,lambda_l2=50,feature_fraction=0.7,bagging_fraction=0.7,bagging_freq=1,verbose=-1,seed=7,num_threads=16,max_bin=127)
y=np.clip((A.label-A.ship).to_numpy(),-1800,1800); nf=~A.fixed.to_numpy(); ap=A.airport.to_numpy()
res={}
for g,(FF,tr,apl) in V.items():
    t0=time.time(); new=A.ship.to_numpy(float).copy()
    TR=nf&np.isin(ap,tr); AL=nf&np.isin(ap,apl)
    for mo in MONTHS:
        f=TR&(A.month!=mo).to_numpy(); t=AL&(A.month==mo).to_numpy()
        mdl=lgb.train(PR,lgb.Dataset(A.loc[f,FF].astype(np.float32),y[f],categorical_feature=['kind','ap']),400)
        new[t]=np.maximum(new[t]+np.clip(0.5*mdl.predict(A.loc[t,FF].astype(np.float32)),-120,120),-12)
    rng=np.random.default_rng(20261004); r={mo:interval(A[A.month==mo],new[(A.month==mo).to_numpy()],rng) for mo in MONTHS}
    eq=float(np.mean([r[mo][0] for mo in EQ]))
    dS=((A.label-new)**2-(A.label-A.ship)**2)
    res[g]={'months':{mo:[float(v) for v in r[mo]] for mo in MONTHS},'eq':eq,'airport_month_dsse_M':dS.groupby([A.airport,A.month]).sum().div(1e6).round(2).unstack().to_dict()}
    print(g,' '.join(f'{mo[5:]} {r[mo][0]:+.4f}[{r[mo][1]:+.3f},{r[mo][2]:+.3f}]' for mo in MONTHS),f'eq {eq:+.4f}',round(time.time()-t0),'s',flush=True)
    json.dump(res,open(out,'w'),indent=1)
