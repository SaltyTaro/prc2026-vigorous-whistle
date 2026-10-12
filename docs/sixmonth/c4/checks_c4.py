"""Final candidate 4 checks (label-free) against candidate 3 -> runs/final_blind/final_candidate4/checks_c4.json."""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, "src")
from prc2026_lirf_mixture.features import DEPARTURE_COLUMNS, population_rows  # noqa: E402

ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
C3, C4 = "runs/final_blind/final_candidate3/chain_out_cap.parquet", "runs/final_blind/final_candidate4/chain_out_cap.parquet"
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
a, b = pd.read_parquet(C3), pd.read_parquet(C4)
tpl = pd.read_parquet("data/raw/final_submitting.parquet")
mv = pd.read_parquet("data/raw/final_ranking.parquet", columns=DEPARTURE_COLUMNS)
dep = mv[mv["PHASE_mvt"].eq("DEP")].drop_duplicates(ID).set_index(ID)
pop = set(population_rows(mv)[ID])
out = {"c3_sha256": sha(C3), "c4_sha256": sha(C4), "rows": len(b), "template_order": bool(b[ID].equals(tpl[ID])),
       "dtypes": {k: str(v) for k, v in b.dtypes.items()}, "min": int(b[T].min()), "max": int(b[T].max()), "negatives": int((b[T] < 0).sum()),
       "nan": int(b[T].isna().sum())}
d = a.merge(b, on=ID, suffixes=("_c3", "_c4"))
d["airport"] = dep["ADEP_mvt"].reindex(d[ID]).to_numpy()
d["month"] = pd.to_datetime(dep["MVT_TIME_UTC_mvt"].reindex(d[ID]).to_numpy()).strftime("%Y-%m")
d["dl"] = (d[f"{T}_c4"] - d[f"{T}_c3"]).astype(float)
ch = d[d["dl"] != 0]
out["rows_changed"] = int(len(ch))
out["squared_change"] = float((ch["dl"] ** 2).sum())
out["max_abs_change"] = float(ch["dl"].abs().max()) if len(ch) else 0.0
out["mean_change"] = float(ch["dl"].mean()) if len(ch) else 0.0
out["by_airport_month_rows"] = ch.groupby(["airport", "month"]).size().unstack(fill_value=0).to_dict()
out["by_airport_squared"] = ch.groupby("airport")["dl"].apply(lambda x: float((x ** 2).sum())).to_dict()
out["by_month_squared"] = ch.groupby("month")["dl"].apply(lambda x: float((x ** 2).sum())).to_dict()
guards = [213142471.0, 213142469.0]
out["guard_rows"] = d[d[ID].isin(guards)][[ID, f"{T}_c3", f"{T}_c4"]].to_dict("records")
out["population_rows_changed"] = int(ch[ID].isin(pop).sum())
cap_rows = {213145579.0: 111666, 213152353.0: 88392}
out["cap_rows"] = {str(int(k)): int(b.set_index(ID)[T][k]) for k in cap_rows}
out["cap_rows_ok"] = all(int(b.set_index(ID)[T][k]) == v for k, v in cap_rows.items())
json.dump(out, open("runs/final_blind/final_candidate4/checks_c4.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in out.items() if k != "by_airport_month_rows"}, indent=1, default=str))
print(pd.DataFrame(out["by_airport_month_rows"]).T.to_string() if out["rows_changed"] else "")
