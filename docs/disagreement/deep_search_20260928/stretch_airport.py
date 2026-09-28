import pandas as pd, numpy as np, sys
f=pd.read_parquet(sys.argv[1]); f=f[f.airport!='LIRF'].copy()
f['scope']=np.where(f.matched,'tracked','untracked')
months=['2025-01','2025-07','2025-11','2025-12']; t=1500
def k_of(g): x=np.maximum(g.control-t,0); r=g.label-g.control; return float((x*r).sum()/(x*x).sum()) if (x*x).sum()>0 else 0.0
def rm(y,p): return float(np.sqrt(np.mean((y-p)**2)))
full=pd.read_parquet(sys.argv[1])
print('per-airport four-month k (untracked / tracked) and per-held-month k spread:')
for ap,g in f.groupby('airport'):
    ks={s:[round(k_of(g[(g.scope==s)&(g.month!=mo)]),3) for mo in months] for s in ['untracked','tracked']}
    print(f'  {ap}', {s:(round(k_of(g[g.scope==s]),3), v) for s,v in ks.items()}, 'rows>t', int((g.control>t).sum()))
for mode in ['pooled','airport','shrunk']:
    out=[]
    for mo in months:
        tr=f[f.month!=mo]; te=full[full.month==mo].copy(); newp=te.control.to_numpy().copy()
        for s in ['untracked','tracked']:
            kp=k_of(tr[tr.scope==s])
            for ap in tr.airport.unique():
                g=tr[(tr.scope==s)&(tr.airport==ap)]
                ka=k_of(g) if len(g) else kp
                n=int((g.control>t).sum())
                k={'pooled':kp,'airport':ka,'shrunk':(n*ka+2000*kp)/(n+2000)}[mode]
                m=((te.airport==ap)&(te.matched==(s=='tracked'))).to_numpy()
                newp[m]=te.control.to_numpy()[m]+k*np.maximum(te.control.to_numpy()[m]-t,0)
        y=te.label.to_numpy(); out.append(100*(rm(y,newp)/rm(y,te.control.to_numpy())-1))
    print(mode, [round(o,4) for o in out], 'mean', round(np.mean(out),4))
