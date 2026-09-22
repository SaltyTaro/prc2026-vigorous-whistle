"""Ground-point coverage per airport per day across the whole local adsb.lol archive. Read-only, label-free."""
import glob
import re
from pathlib import Path

import pandas as pd

ROOT = Path
rows = []
for path in sorted(glob.glob(str(ROOT / "data/external/adsblol/surface/surface_*.parquet"))):
    day = re.search(r"surface_(\d{8})", path).group(1)
    d = pd.read_parquet(path, columns=["airport", "ground", "icao"])
    g = d[d["ground"]]
    for airport in sorted(d["airport"].unique()):
        rows.append({"day": day, "month": day[:6], "airport": airport,
                     "points": int((d["airport"] == airport).sum()),
                     "ground": int((g["airport"] == airport).sum()),
                     "icao_ground": int(g.loc[g["airport"] == airport, "icao"].nunique())})
t = pd.DataFrame(rows)
t.to_parquet(ROOT / "runs/adsb_coverage/ground_points_by_day.parquet", index=False)
piv = t.pivot_table(index="month", columns="airport", values="ground", aggfunc="sum").fillna(0).astype(int)
print("=== ground points per month per airport (thousands) ===")
print((piv / 1000).round(1).to_string())
print()
days = t.pivot_table(index="month", columns="airport", values="ground", aggfunc=lambda s: int((s > 0).sum())).fillna(0).astype(int)
print("=== days in the month with ANY ground point ===")
print(days.to_string())
