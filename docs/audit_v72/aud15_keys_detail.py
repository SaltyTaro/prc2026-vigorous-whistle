"""aud15 (exploration, nothing declared): detail of aud14's flight-number memory (beta 0.25, lam 30) on v74's development values.
Per airport and kind, the concentration of the gain (top rows), a variant without EGLL, sequential fn -> op_hr, and the RMS of the change."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docs/audit_v72"))
sys.argv = ["x"]
import importlib.util  # noqa: E402

src = (ROOT / "docs/audit_v72/aud14_keys.py").read_text(encoding="utf-8").split("rows = []")[0]
ns: dict = {"__file__": str(ROOT / "docs/audit_v72/aud14_keys.py")}
exec(compile(src, "aud14_head", "exec"), ns)
t, correction, KEYS, fixed = ns["t"], ns["correction"], ns["KEYS"], ns["fixed"]
R = ns["R"]


def seq(src_m, tgt_m, keys, beta, lam):
    tg = (t["month"] == tgt_m).to_numpy()
    total = np.zeros(tg.sum())
    saved = t["resc"].copy()
    sm = (t["month"] == src_m).to_numpy()
    for k in keys:
        _, c = correction(src_m, tgt_m, KEYS[k], beta, lam)
        total += c
        # residual of the fitting month after this key's own (in-month) correction, for the next key
        _, cs = correction(src_m, src_m, KEYS[k], beta, lam)
        t.loc[sm, "resc"] = (t.loc[sm, "resc"].to_numpy() - cs).clip(-1800, 1800)
    t["resc"] = saved
    return tg, total


for label, keys, excl in [("fn", ["fn"], []), ("fn_noEGLL", ["fn"], ["EGLL"]), ("fn+op_hr", ["fn", "op_hr"], []),
                          ("fn_noEGLL_LEBL_LEMD", ["fn"], ["EGLL", "LEBL", "LEMD"])]:
    for src_m, tgt_m in [("2025-11", "2025-12"), ("2025-12", "2025-11"), ("2025-12", "2025-01")]:
        tg, c = seq(src_m, tgt_m, keys, 0.25, 30.0)
        sub = t[tg].assign(value=t["ship"].to_numpy()[tg]).reset_index(drop=True)
        c = np.where(sub["airport"].isin(excl).to_numpy(), 0.0, c)
        new = np.maximum(sub["value"].to_numpy() + c, -12.0)
        d = (sub["label"] - new) ** 2 - (sub["label"] - sub["value"]) ** 2
        pct, lo, hi = R.interval(sub, "value", new, np.random.default_rng(20261004))
        top = np.sort(d.to_numpy())[:20].sum()
        ch = new - sub["value"].to_numpy()
        print(f"{label:22s} {src_m}->{tgt_m} rmse% {pct:+.4f} [{lo:+.4f}, {hi:+.4f}] dSSE {d.sum() / 1e6:+.2f}e6 top20 {top / 1e6:+.2f}e6 "
              f"S {np.sum(ch ** 2) / 1e6:.2f}e6 rows {(ch != 0).sum()} rms {np.sqrt(np.mean(ch[ch != 0] ** 2)):.1f} g/S {-d.sum() / max(np.sum(ch ** 2), 1):.2f}", flush=True)
        if label == "fn" and tgt_m != "2025-01":
            print("   by airport x kind dSSE e6:", (d.groupby([sub["airport"], sub["kind"]]).sum() / 1e6).round(2).loc[lambda s: s.abs() >= 0.1].to_dict())
            lab = sub["label"].to_numpy()
            print("   dSSE by label band e6:", (d.groupby(pd.cut(lab, [-1e9, 300, 900, 1800, 3600, 20000, 1e9])).sum() / 1e6).round(2).to_dict())
