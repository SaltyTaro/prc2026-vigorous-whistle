import pandas as pd, numpy as np, sys
f=pd.read_parquet(sys.argv[1])
f['scope']=np.where(f.airport=='LIRF','LIRF',np.where(f.matched,'tracked','untracked'))
months=['2025-01','2025-07','2025-11','2025-12']
def rm(y,p): return float(np.sqrt(np.mean((y-p)**2)))
def fit_tail(g,t):
    x=np.maximum(g.control-t,0); y=g.label-g.control
    return float((x*y).sum()/(x*x).sum())
def fit_lin(g):
    # y - p = a + (b-1)(p - pbar) per scope, LS
    X=np.c_[np.ones(len(g)), g.control]; y=g.label.to_numpy()
    return np.linalg.lstsq(X,y,rcond=None)[0]
print('in-sample slopes per month and scope (label ~ a + b*pred):')
for mo in months:
    print(mo, {s: np.round(fit_lin(f[(f.month==mo)&(f.scope==s)]),3).tolist() for s in ['untracked','tracked','LIRF']})
for t in [1200,1500,1800,2400]:
    out=[]
    for mo in months:
        tr=f[f.month!=mo]; te=f[f.month==mo].copy()
        newp=te.control.to_numpy().copy()
        ks={}
        for s in ['untracked','tracked']:
            k=fit_tail(tr[tr.scope==s],t); ks[s]=round(k,3)
            m=(te.scope==s).to_numpy(); newp[m]=te.control.to_numpy()[m]+k*np.maximum(te.control.to_numpy()[m]-t,0)
        y=te.label.to_numpy(); out.append((mo, 100*(rm(y,newp)/rm(y,te.control.to_numpy())-1), ks, int((newp!=te.control.to_numpy()).sum())))
    print(f'tail t={t}:', ' '.join(f'{mo}:{d:+.3f}%({n}) k={ks}' for mo,d,ks,n in out), ' mean', round(np.mean([o[1] for o in out]),4))
# linear recalibration LOMO per scope (non-LIRF)
out=[]
for mo in months:
    tr=f[f.month!=mo]; te=f[f.month==mo]
    newp=te.control.to_numpy().copy()
    for s in ['untracked','tracked']:
        a,b=fit_lin(tr[tr.scope==s]); m=(te.scope==s).to_numpy(); newp[m]=a+b*te.control.to_numpy()[m]
    y=te.label.to_numpy(); out.append((mo,100*(rm(y,newp)/rm(y,te.control.to_numpy())-1)))
print('linear:', out, round(np.mean([o[1] for o in out]),4))
