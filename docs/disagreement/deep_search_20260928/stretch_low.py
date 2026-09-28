import pandas as pd, numpy as np, sys
full=pd.read_parquet(sys.argv[1])
months=['2025-01','2025-07','2025-11','2025-12']
full['scope']=np.where(full.airport=='LIRF','LIRF',np.where(full.matched,'tracked','untracked'))
K={'untracked':0.126,'tracked':0.085}
def rm(y,p): return float(np.sqrt(np.mean((y-p)**2)))
# v37-style upper stretch with cross-month k (recomputed per held month for honesty)
def k_up(g,t=1500): x=np.maximum(g.control-t,0); r=g.label-g.control; return float((x*r).sum()/(x*x).sum())
for t2 in [600,900,1200]:
    out=[]
    for mo in months:
        tr=full[full.month!=mo]; te=full[full.month==mo]
        base=te.control.to_numpy().copy(); up=base.copy()
        for s in ['untracked','tracked']:
            m=(te.scope==s).to_numpy(); k=k_up(tr[tr.scope==s]); up[m]=base[m]+k*np.maximum(base[m]-1500,0)
        new=up.copy()
        for s in ['untracked','tracked']:
            g=tr[tr.scope==s]; ku=k_up(g)
            p=g.control+ku*np.maximum(g.control-1500,0); x=np.maximum(t2-p,0); r=g.label-p
            k2=float((x*r).sum()/(x*x).sum())
            m=(te.scope==s).to_numpy(); new[m]=np.maximum(up[m]+k2*np.maximum(t2-up[m],0), -12)
        y=te.label.to_numpy(); out.append((round(100*(rm(y,up)/rm(y,base)-1),4), round(100*(rm(y,new)/rm(y,up)-1),4), round(k2,3)))
    print(f'low t2={t2}: (upper vs control, low-end add-on vs upper, k2 tracked)', out, 'mean add-on', round(np.mean([o[1] for o in out]),4))
