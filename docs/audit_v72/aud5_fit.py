"""Raw ADS-B inputs follow-up of the 'what's left' audit on v72's development residuals (exploration; nothing declared). v72 dev = v71 dev + stack_second sp_hicov7
(0.5 x out-of-month five-seed bag, cap 120, floor -12). Month-blocked single-seed LightGBM per variant, correction 0.5 x prediction capped
+-120 s, floor -12, fixed rows held; complete-month RMSE change vs v72 with paired-day intervals; per airport-month SSE change."""
import sys, json, time, numpy as np, pandas as pd, lightgbm as lgb
d=sys.argv[1]; out=sys.argv[2]; ID='MVT_ID_mvt'
A=pd.read_parquet(d+'/dev_table.parquet'); RAWT=pd.read_parquet(d+'/raw_dev.parquet'); assert (RAWT[ID].to_numpy()==A[ID].to_numpy()).all(); A=pd.concat([A,RAWT.drop(columns=[ID])],axis=1); P=pd.read_parquet(d+'/preds_dev.parquet').set_index(ID)['sp_hicov7'].reindex(A[ID]).to_numpy()
v71=A.v71.to_numpy(float); v72=v71.copy(); m=~np.isnan(P); v72[m]=np.maximum(v71[m]+np.clip(0.5*P[m],-120,120),-12)
A['ship']=v72; A['p2']=P; A['d2']=v72-v71
MONTHS=['2025-01','2025-07','2025-11','2025-12']; EQ=['2025-01','2025-07','2025-12']
def interval(s,new,rng,reps=2000):
    e0=((s.label-s.ship)**2).to_numpy(); e1=(s.label.to_numpy()-new)**2
    ud,inv=np.unique(s.day.to_numpy(),return_inverse=True); s0,s1,n=np.bincount(inv,e0),np.bincount(inv,e1),np.bincount(inv)
    o=[]
    for _ in range(reps):
        k=np.bincount(rng.integers(0,len(ud),len(ud)),minlength=len(ud)); o.append(100*(np.sqrt((k*s1).sum()/(k*n).sum())/np.sqrt((k*s0).sum()/(k*n).sum())-1))
    lo,hi=np.percentile(o,[2.5,97.5]); return 100*(np.sqrt(e1.mean())/np.sqrt(e0.mean())-1),lo,hi
S=[c for c in A.columns if c.startswith('s_')]
NB=['nx','nxk','nn','x_nmerr']; QU=['gp','gn','gp2','gp4','gn4','bpos','blen']; CL=['c_sched','c_aobt','c_lobt','c_eobt','c_iobt','c_arvt','c_hour','c_dow','c_layers']
BASE=['kind','ap']; ALL=S+NB+QU+CL+BASE; P2=['p2','d2']
H7=['EDDF','EDDM','EGLL','EHAM','LEBL','LEMD','LSZH']; R3=['LFPG','LTFM','LIRF']; AP=H7+R3
V0={'all10':(ALL,AP,AP),'all10_p2':(ALL+P2,AP,AP),'h7_in':(ALL,H7,H7),'h7_in_p2':(ALL+P2,H7,H7),'h7_stk_p2':(S+P2+BASE,H7,H7),
   'r3_in':(ALL,R3,R3),'all_to_r3':(ALL,AP,R3),'qn_all_to_r3':(NB+QU+BASE,AP,R3),'qn_lfpg_in':(NB+QU+BASE,['LFPG'],['LFPG'])}
REG=dict(num_leaves=15,min_data_in_leaf=4000,lambda_l2=200)
RAW=[c for c in RAWT.columns if c!=ID]
V={'h7_raw':(ALL+RAW,H7,H7,{},400),'h7_raw_p2':(ALL+P2+RAW,H7,H7,{},400),'h7_rawonly':(RAW+P2+['ship']+CL+BASE,H7,H7,{},400),
   'h7_raw_reg':(ALL+P2+RAW,H7,H7,REG,200),'all10_raw':(ALL+RAW,AP,AP,{},400),'r3_raw_reg':(ALL+RAW,R3,R3,REG,200),
   'lfpg_raw_reg':(ALL+RAW,['LFPG'],['LFPG'],REG,200),'lirf_raw_reg':(ALL+RAW,['LIRF'],['LIRF'],REG,200)}
if len(sys.argv)>3: V={k:V[k] for k in sys.argv[3].split(',')}
PR=dict(objective='regression',learning_rate=0.03,num_leaves=31,min_data_in_leaf=800,lambda_l2=50,feature_fraction=0.7,bagging_fraction=0.7,bagging_freq=1,verbose=-1,seed=7,num_threads=16,max_bin=127)
y=np.clip((A.label-A.ship).to_numpy(),-1800,1800); nf=~A.fixed.to_numpy(); ap=A.airport.to_numpy()
res={}
for g,(F,tr,apl,ov,NR) in V.items():
    t0=time.time(); new=A.ship.to_numpy(float).copy(); imp=np.zeros(len(F))
    TR=nf&np.isin(ap,tr); AL=nf&np.isin(ap,apl)
    for mo in MONTHS:
        f=TR&(A.month!=mo).to_numpy(); t=AL&(A.month==mo).to_numpy()
        mdl=lgb.train({**PR,**ov},lgb.Dataset(A.loc[f,F].astype(np.float32),y[f],categorical_feature=['kind','ap']),NR)
        p=mdl.predict(A.loc[t,F].astype(np.float32)); new[t]=np.maximum(new[t]+np.clip(0.5*p,-120,120),-12)
        imp+=mdl.feature_importance('gain')
    rng=np.random.default_rng(20261004); r={}
    for mo in MONTHS:
        s=A[A.month==mo]; r[mo]=interval(s,new[(A.month==mo).to_numpy()],rng)
    eq=float(np.mean([r[mo][0] for mo in EQ]))
    dS=((A.label-new)**2-(A.label-A.ship)**2); sq=(new-A.ship)**2
    res[g]={'train':tr,'apply':apl,'months':{mo:[float(v) for v in r[mo]] for mo in MONTHS},'eq':eq,'secs':time.time()-t0,
            'top':sorted(zip(F,(imp/imp.sum()).round(4).tolist()),key=lambda z:-z[1])[:20],
            'airport_month_dsse_M':dS.groupby([A.airport,A.month]).sum().div(1e6).round(2).unstack().to_dict(),
            'airport_month_S_M':sq.groupby([A.airport,A.month]).sum().div(1e6).round(2).unstack().to_dict()}
    print(g,' '.join(f'{mo[5:]} {r[mo][0]:+.4f}[{r[mo][1]:+.3f},{r[mo][2]:+.3f}]' for mo in MONTHS),f'eq {eq:+.4f}',round(time.time()-t0),'s',flush=True)
    json.dump(res,open(out,'w'),indent=1)
