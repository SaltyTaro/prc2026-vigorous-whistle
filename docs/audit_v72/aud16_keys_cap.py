"""aud16 (exploration, nothing declared): aud14's flight-number memory (beta 0.25, lam 30) with the late passes' +-120 s cap, all ten
airports and without EGLL; v74's development values."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
src = (ROOT / "docs/audit_v72/aud14_keys.py").read_text(encoding="utf-8").split("rows = []")[0]
ns = {"__file__": str(ROOT / "docs/audit_v72/aud14_keys.py")}
exec(compile(src, "aud14_head", "exec"), ns)
t, correction, KEYS, R = ns["t"], ns["correction"], ns["KEYS"], ns["R"]
for cap in [None, 120.0, 60.0]:
    for excl in [[], ["EGLL"]]:
        out = []
        for src_m, tgt_m in [("2025-11", "2025-12"), ("2025-12", "2025-11"), ("2025-12", "2025-01")]:
            tg, c = correction(src_m, tgt_m, KEYS["fn"], 0.25, 30.0)
            sub = t[tg].assign(value=t["ship"].to_numpy()[tg]).reset_index(drop=True)
            c = np.where(sub["airport"].isin(excl).to_numpy(), 0.0, c)
            if cap:
                c = np.clip(c, -cap, cap)
            new = np.maximum(sub["value"].to_numpy() + c, -12.0)
            pct, lo, hi = R.interval(sub, "value", new, np.random.default_rng(20261004))
            out.append(f"{src_m[5:]}>{tgt_m[5:]} {pct:+.4f} [{lo:+.4f},{hi:+.4f}] maxc {np.abs(c).max():.0f}")
        print("cap", cap, "excl", excl, " | ".join(out), flush=True)
