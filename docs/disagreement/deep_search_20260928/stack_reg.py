import pandas as pd, numpy as np, sys
f=pd.read_parquet(sys.argv[1])
months=['2025-01','2025-07','2025-11','2025-12']
def rm(y,p): return float(np.sqrt(np.mean((y-p)**2)))
f['scope']=np.where(f.airport=='LIRF',np.where(f.lirf_m|f.lirf_u,'lirf_nonpop','lirf_pop'),np.where(f.matched,'tracked','untracked'))
# (b) LIRF non-pop stretch at t1500, cross-month
def k_up(g,t=1500): x=np.maximum(g.control-t,0); r=g.label-g.control; return float((x*r).sum()/(x*x).sum())
out=[]
for mo in months:
    tr=f[(f.month!=mo)&(f.scope=='lirf_nonpop')]; te=f[f.month==mo]; k=k_up(tr)
    m=(te.scope=='lirf_nonpop').to_numpy(); newp=te.control.to_numpy().copy(); newp[m]=newp[m]+k*np.maximum(newp[m]-1500,0)
    y=te.label.to_numpy(); out.append((round(100*(rm(y,newp)/rm(y,te.control.to_numpy())-1),4), round(k,3), int((te.scope=='lirf_nonpop').sum())))
print('(b) LIRF non-pop stretch t1500 per held month (ratio%, k, rows):', out)
# (a) stacking regression per scope y = a + b1*base + b2*mix on non-LIRF rows (untracked: base = v9), cross-month; compare with control
out=[]
for mo in months:
    tr=f[f.month!=mo]; te=f[f.month==mo]; newp=te.control.to_numpy().copy()
    for s in ['untracked','tracked']:
        g=tr[tr.scope==s]; X=np.c_[np.ones(len(g)),g.base,g.mix]; b=np.linalg.lstsq(X,g.label.to_numpy(),rcond=None)[0]
        m=(te.scope==s).to_numpy(); newp[m]=np.maximum(b[0]+b[1]*te.base.to_numpy()[m]+b[2]*te.mix.to_numpy()[m],-12)
        if mo=='2025-12': print('   ',s,'coef (a, b_base, b_mix) fitted without Dec:',np.round(b,3))
    y=te.label.to_numpy(); out.append(round(100*(rm(y,newp)/rm(y,te.control.to_numpy())-1),4))
print('(a) free stacking regression vs v35 stack:', out, 'mean', round(np.mean(out),4))
