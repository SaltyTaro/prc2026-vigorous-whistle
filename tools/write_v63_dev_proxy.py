"""Write the v63 development chain proxy (runs/stack_resid/dev_v63_proxy.parquet) for later declared layers.

v63's chain on the four development months, recomputed with tools/run_stack_resid_v3.py's functions (frozen; imported, not changed):
ctl_v55m `new` -> no-NM rematch -> stack_resid_v3 delta (LIRF keeps v2) -> intruder bound -> identity extension -> own-motion bound v1,
every layer month-blocked as in its governing run. Rows and keys are those of the v61 / v62 proxy (runs/dig/census61/dev_v61.parquet).
Columns: MVT_ID_mvt, month, airport, label, cls, v62 (the v62 proxy), v63. The month RMSEs must equal docs/stack_resid/results_v3.json
(candidate pregap_replace_w50_nolirf) and its control exactly. No official score is used.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402
import run_stack_resid as S  # noqa: E402
import run_stack_resid_v3 as V3  # noqa: E402

OUT = ROOT / "runs/stack_resid/dev_v63_proxy.parquet"


def main() -> None:
    if OUT.exists():
        raise ValueError(f"{OUT} exists; never overwrite.")
    if sha256_file(ROOT / V3.DEV61[0]) != V3.DEV61[1]:
        raise ValueError("The v61 development proxy differs from the declared file.")
    dev, X46, X65, Pn, res = V3.dev_inputs()
    pr = V3.dev_predictions(dev, X46, X65, Pn, res)
    new, air = dev["new"].to_numpy(dtype=float), dev["airport"].to_numpy()
    d61 = pd.read_parquet(ROOT / V3.DEV61[0]).set_index(S.ID).reindex(dev[S.ID]).reset_index()
    s1 = d61["s1"].to_numpy(dtype=float)
    d = d61[[S.ID, "month", "airport", "label", "cls"]].copy()
    d["s2_v2"] = s1 + (pr["base2"] - new)
    d["s2_v3"] = s1 + (np.where(air == "LIRF", pr["base2"], pr["base3"]) - new)
    ch = V3.Chain()
    v61_ctl, v62_ctl = ch.run(d, "s2_v2")
    if np.abs(v61_ctl - d61["v61"].to_numpy()).max() > 1e-6:
        raise ValueError("The control chain does not reproduce the v61 proxy.")
    _, v63 = ch.run(d, "s2_v3")
    r3 = read_json(ROOT / "docs/stack_resid/results_v3.json")["candidates"]["pregap_replace_w50_nolirf"]["months"]
    y, month = d["label"].to_numpy(dtype=float), d["month"].to_numpy()
    got = {m: (float(np.sqrt(np.mean((y[month == m] - v62_ctl[month == m]) ** 2))), float(np.sqrt(np.mean((y[month == m] - v63[month == m]) ** 2))))
           for m in S.MONTHS}
    dev_max = max(max(abs(got[m][0] - r3[m]["control_rmse"]), abs(got[m][1] - r3[m]["rmse"])) for m in S.MONTHS)
    if dev_max > 1e-9:
        raise ValueError(f"The proxy does not reproduce results_v3.json: {got}")
    out = d[[S.ID, "month", "airport", "label", "cls"]].assign(v62=v62_ctl, v63=v63)
    out.to_parquet(OUT, index=False)
    rec = {"written_at_utc": utc_now(), "file": OUT.relative_to(ROOT).as_posix(), "sha256": sha256_file(OUT), "rows": int(len(out)),
           "month_rmse_s": {m: {"v62": got[m][0], "v63": got[m][1]} for m in S.MONTHS}, "reproduces_results_v3_max_abs_s": dev_max,
           "code": {c: sha256_file(ROOT / c) for c in ["tools/write_v63_dev_proxy.py", "tools/run_stack_resid_v3.py"]}}
    write_json(OUT.with_suffix(".receipt.json"), rec)
    print(rec)


if __name__ == "__main__":
    main()
