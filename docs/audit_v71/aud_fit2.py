import sys, json, time, numpy as np, pandas as pd, lightgbm as lgb
exec(open('aud_fit.py').read().split("res={}")[0].split("A=pd.read_parquet")[0])
A=pd.read_parquet('audit_table.parquet')
exec("\n".join(l for l in open('aud_fit.py').read().split("res={}")[0].splitlines() if not l.startswith('A=pd.read_parquet') and not l.startswith('import')))
SETS={'noLIRF':['EDDF','EDDM','EGLL','EHAM','LEBL','LEMD','LFPG','LSZH','LTFM'],'hicov7':['EDDF','EDDM','EGLL','EHAM','LEBL','LEMD','LSZH']}
out={}
for sname,aps in SETS.items():
    inset=A.airport.isin(aps).to_numpy()
    for g in ['all','stacker']:
        F=G[g]
        for train_scope in ['allrows','inset']:
            new=A.ship.to_numpy(float).copy()
            for m in MONTHS:
                f=nf&(A.month!=m).to_numpy(); t=nf&(A.month==m).to_numpy()&inset
                if train_scope=='inset': f=f&inset
                mdl=lgb.train(P,lgb.Dataset(A.loc[f,F].astype(np.float32),y[f],categorical_feature=['kind','ap']),400)
                new[t]=np.maximum(new[t]+np.clip(0.5*mdl.predict(A.loc[t,F].astype(np.float32)),-120,120),-12)
            rng=np.random.default_rng(20261004); r={}
            for m in MONTHS:
                s=A[A.month==m]; r[m]=interval(s,new[(A.month==m).to_numpy()],rng)
            eq=float(np.mean([r[m][0] for m in EQ])); ok=all(r[m][0]<=0.005 for m in MONTHS) and all(r[m][2]<0 for m in EQ) and eq<=-0.02
            S2=float(((new-A.ship.to_numpy())**2).sum()/4)
            print(f'{sname:7s} {g:8s} train={train_scope:7s}',' '.join(f'{m[5:]} {r[m][0]:+.4f}[{r[m][1]:+.3f},{r[m][2]:+.3f}]' for m in MONTHS),f'eq {eq:+.4f} S/mo {S2/1e6:.2f}M','PASS' if ok else '',flush=True)
            out[f'{sname}_{g}_{train_scope}']={'months':{m:list(map(float,r[m])) for m in MONTHS},'eq':eq,'S_per_month':S2}
json.dump(out,open('audit2_results.json','w'),indent=1)
