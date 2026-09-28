# v35-stack control values on the LOMO rows (Jan/Jul LOMO, Nov/Dec fair) -> scratch parquet for label-free signal screens
import sys; from pathlib import Path
ROOT=Path('.').resolve(); sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'tools'))
import numpy as np, pandas as pd
from prc2026.artifacts import read_json
from prc2026.features import ID
from run_callsign_fix import offsets, scope_table
from run_mix_gate import stack
from run_queue_trust import offsets_all, segments
rows=pd.read_parquet('runs/lomo_dev/rows.parquet')
months=['2025-01','2025-07','2025-11','2025-12']
cs=scope_table(read_json(ROOT/'configs/callsign_fix_v1.json')).set_index(ID); off_cs,off=offsets(),offsets_all(); seg=segments(months)
out=[]
for m in months:
    f=rows[rows.month==m].reset_index(drop=True)
    f['control']=stack(f,f['mix'].to_numpy(float),cs,off_cs,seg,off); out.append(f)
pd.concat(out).to_parquet(sys.argv[1]); print('ok')
