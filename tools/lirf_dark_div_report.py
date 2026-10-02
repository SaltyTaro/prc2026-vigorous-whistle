"""Scorer of the LIRF dark diversity design (configs/lirf_dark_div_design_v1.json); written before any fit of the design.

Dark sets = tools/run_lirf_dark.build_sets' thinned_july and natural:2025-01 (same tables, same checks). Member files:
  thinned_july  <transfer>/<variant>_s<seed>/2025-07_thinned.parquet   (tools/ctx3_lirf_predict.py --fold seasonal transfer directories)
  2025-01       <nets>/<variant>_s<seed>/lomo_2025-01/predictions.parquet
Control M = v54's member, the 16-seed lm25_32 mean (runs/adsb_ctx3_lirf_mask, seeds 801-816; checked equal to run_lirf_dark's member).
Candidate mix = 0.5 M + 0.5 x the 16-seed lm25dl32 mean (runs/lirf_dark_div, seeds 1001-1016). Reported only: lm25dl32 alone, the
8 -> 16 seed doubling reading of lm25_32 (801-808 against the mean of 801-808 and 809-816), standalone route RMSEs.
Value = v54's dark arithmetic (tools/run_lirf_dark.values, w 0.70, s 0.75, gate u600). Rule: the declaration's. --smoke reads whatever
candidate seeds exist and writes runs/lirf_dark_div/smoke.json (never a result).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID  # noqa: E402
import run_lirf_dark as rld  # noqa: E402
import run_lirf_member as rlm  # noqa: E402

DECL = "configs/lirf_dark_div_design_v1.json"
OLD = {"nets": ROOT / "runs/adsb_ctx3_lirf_mask/nets", "transfer": ROOT / "runs/adsb_ctx3_lirf_mask/transfer"}
NEW = {"nets": ROOT / "runs/lirf_dark_div/nets", "transfer": ROOT / "runs/lirf_dark_div/transfer"}
W, S, T = 0.70, 0.75, 600.0
V54_SEEDS, HALF_A, HALF_B = list(range(801, 817)), list(range(801, 809)), list(range(809, 817))
BAR = 0.00574


def path(where: dict, variant: str, seed: int, key: str) -> Path:
    if key == "thinned_july":
        return where["transfer"] / f"{variant}_s{seed}" / "2025-07_thinned.parquet"
    return where["nets"] / f"{variant}_s{seed}" / "lomo_2025-01" / "predictions.parquet"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    decl = read_json(ROOT / DECL)
    if not decl.get("declared_at_utc"):
        raise ValueError("The declaration is not recorded.")
    sets, checks, inputs = rld.build_sets()
    keys = list(rld.DARK)
    ids = {}
    rows = pd.read_parquet(ROOT / rlm.ROWS, columns=[ID, "month", "airport", "lirf_m", "lirf_u", "label"])
    rows = rows[rows["airport"].eq("LIRF")]
    ids["natural:2025-01"] = rows[rows["month"].eq("2025-01") & rows["lirf_u"].astype(bool)][ID].to_numpy()
    measure = pd.read_parquet(ROOT / rlm.MEASURE, columns=[ID, "class_thinned"])
    nets = pd.read_parquet(ROOT / rlm.NETS, columns=[ID])
    july = rows[rows["month"].eq("2025-07") & (rows["lirf_m"] | rows["lirf_u"]).astype(bool)]
    t = july.merge(measure, on=ID, how="inner", validate="one_to_one").merge(nets, on=ID, how="inner", validate="one_to_one")
    ids["thinned_july"] = t[t["class_thinned"].eq("unmatched")][ID].to_numpy()
    for k in keys:
        if len(ids[k]) != len(sets[k]["labels"]):
            raise ValueError(f"{k}: ID recovery does not match the set.")

    def mean(where: dict, variant: str, seeds: list[int], key: str, smoke: bool = False) -> tuple[np.ndarray, int]:
        files = [path(where, variant, s, key) for s in seeds]
        have = [f for f in files if f.exists()]
        if len(have) != len(files) and not smoke:
            raise ValueError(f"{variant} {key}: {len(files) - len(have)} member files missing.")
        if not have:
            raise ValueError(f"{variant} {key}: no member file.")
        return np.mean([rlm.read(f, ids[key]) for f in have], axis=0), len(have)

    control = {}
    for k in keys:
        m16, _ = mean(OLD, "lm25_32", V54_SEEDS, k)
        diff = float(np.max(np.abs(m16 - sets[k]["member"])))
        checks[f"v54_member_{k}_max_abs_s"] = diff
        if diff > 1e-6:
            raise ValueError(f"{k}: the 16-seed lm25_32 mean differs from run_lirf_dark's member ({diff}).")
        control[k] = m16

    def reading(k: str, m_base: np.ndarray, m_new: np.ndarray) -> dict:
        s = sets[k]
        base, _ = rld.values({**s, "member": m_base}, W, S, T)
        new, fire = rld.values({**s, "member": m_new}, W, S, T)
        return rld.compare(s, base, new, fire)

    def standalone(k: str, m: np.ndarray) -> float:
        y = sets[k]["labels"]
        keep = y < rld.CAP
        return float(np.sqrt(((m - y)[keep] ** 2).mean()))

    variant = decl["fits"]["variants"][0]
    results, used = {}, {}
    for k in keys:
        dl, n = mean(NEW, variant, decl["fits"]["seeds"], k, a.smoke)
        used[k] = n
        half_a = mean(OLD, "lm25_32", HALF_A, k)[0]
        half_b = mean(OLD, "lm25_32", HALF_B, k)[0]
        results[k] = {"mix": reading(k, control[k], 0.5 * control[k] + 0.5 * dl),
                      "dl_alone": reading(k, control[k], dl),
                      "doubling_reading": reading(k, half_a, 0.5 * half_a + 0.5 * half_b),
                      "standalone_rmse": {"lm25_32_16": standalone(k, control[k]), "lm25dl32_16": standalone(k, dl),
                                          "mix": standalone(k, 0.5 * control[k] + 0.5 * dl)},
                      "member_correlation": float(np.corrcoef(control[k] - sets[k]["labels"], dl - sets[k]["labels"])[0, 1])}
    jan, tj = results["natural:2025-01"]["mix"]["primary"], results["thinned_july"]["mix"]["primary"]
    go = (jan["relative_change"] <= -BAR and tj["relative_change"] <= 0
          and all(not (results[k]["mix"]["primary"]["interval_95"][0] > 0) for k in keys))
    for k in keys:
        r = results[k]
        print(k, {x: (round(100 * r[x]["primary"]["relative_change"], 3), [round(100 * v, 3) for v in r[x]["primary"]["interval_95"]])
                  for x in ("mix", "dl_alone", "doubling_reading")}, {x: round(v, 2) for x, v in r["standalone_rmse"].items()},
              "corr", round(r["member_correlation"], 3), flush=True)
    report = {"created_at_utc": utc_now(), "smoke": a.smoke, "declaration": {DECL: {"sha256": sha256_file(ROOT / DECL), "object_hash": object_hash(decl)}},
              "code": {q: sha256_file(ROOT / q) for q in ["tools/lirf_dark_div_report.py", "tools/run_lirf_dark.py", "tools/run_lirf_member.py", "src/prc2026_final/lirf_gate.py"]},
              "inputs": inputs, "checks": checks, "seeds_used": used, "arithmetic": {"w": W, "s": S, "T": T}, "bar": BAR,
              "results": results, "go": None if a.smoke else bool(go), "rule": decl["rule"]}
    out = ROOT / ("runs/lirf_dark_div/smoke.json" if a.smoke else decl["report"])
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, report)
    print("bar", round(100 * BAR, 3), "GO:", None if a.smoke else bool(go), "->", out)


if __name__ == "__main__":
    main()
