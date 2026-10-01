"""Scorer of the stronger-masking LIRF design (configs/lirf_mask2_design_v1.json); written before any fit of the design finished.

Dark sets = tools/run_lirf_dark.build_sets' thinned_july and natural:2025-01 (same tables, same checks). Member files:
  thinned_july  <runs>/<variant>_s<seed>/2025-07_thinned.parquet   (tools/ctx3_lirf_predict.py --fold seasonal transfer directories)
  2025-01       <nets>/<variant>_s<seed>/lomo_2025-01/predictions.parquet
for the control lm25_32 (runs/adsb_ctx3_lirf_mask, seeds 801-808; the null 809-816) and each candidate (runs/lirf_mask2, seeds 901-908).
Value = v54's dark arithmetic (tools/run_lirf_dark.values, w 0.70, s 0.75, gate u600). Rule: the declaration's. --smoke reads whatever
candidate seeds exist and writes runs/lirf_mask2/smoke.json (never a result).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, write_json  # noqa: E402
import run_lirf_dark as rld  # noqa: E402
import run_lirf_member as rlm  # noqa: E402

DECL = "configs/lirf_mask2_design_v1.json"
OLD = {"nets": ROOT / "runs/adsb_ctx3_lirf_mask/nets", "transfer": ROOT / "runs/adsb_ctx3_lirf_mask/transfer"}
NEW = {"nets": ROOT / "runs/lirf_mask2/nets", "transfer": ROOT / "runs/lirf_mask2/transfer"}
W, S, T = 0.70, 0.75, 600.0
CONTROL_SEEDS, NULL_SEEDS, V54_SEEDS = list(range(801, 809)), list(range(809, 817)), list(range(801, 817))


def path(where: dict, variant: str, seed: int, key: str) -> Path:
    if key == "thinned_july":
        return where["transfer"] / f"{variant}_s{seed}" / "2025-07_thinned.parquet"
    return where["nets"] / f"{variant}_s{seed}" / "lomo_2025-01" / "predictions.parquet"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    decl = read_json(ROOT / DECL)
    if not decl.get("declared_at_utc") or decl["declared_at_utc"].startswith("<"):
        raise ValueError("The declaration is not recorded.")
    sets, checks, inputs = rld.build_sets()
    keys = list(rld.DARK)
    ids = {}
    # build_sets keeps row order of its tables; recover each dark set's IDs the same way to read member files
    import pandas as pd  # noqa: E402
    from prc2026.features import ID  # noqa: E402
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

    # check: the 16-seed lm25_32 mean reproduces the member of run_lirf_dark (v54's footing) on both sets
    for k in keys:
        m16, _ = mean(OLD, "lm25_32", V54_SEEDS, k)
        diff = float(np.max(np.abs(m16 - sets[k]["member"])))
        checks[f"v54_member_{k}_max_abs_s"] = diff
        if diff > 1e-6:
            raise ValueError(f"{k}: the 16-seed lm25_32 mean differs from run_lirf_dark's member ({diff}).")

    def reading(k: str, m_base: np.ndarray, m_new: np.ndarray) -> dict:
        s = sets[k]
        base, _ = rld.values({**s, "member": m_base}, W, S, T)
        new, fire = rld.values({**s, "member": m_new}, W, S, T)
        return rld.compare(s, base, new, fire)

    def standalone(k: str, m: np.ndarray) -> float:
        y = sets[k]["labels"]
        keep = y < rld.CAP
        return float(np.sqrt(((m - y)[keep] ** 2).mean()))

    results, used = {}, {}
    controls = {k: mean(OLD, "lm25_32", CONTROL_SEEDS, k)[0] for k in keys}
    v54m = {k: sets[k]["member"] for k in keys}
    null = {k: reading(k, controls[k], mean(OLD, "lm25_32", NULL_SEEDS, k)[0]) for k in keys}
    for variant in decl["fits"]["variants"]:
        res = {}
        for k in keys:
            m, n = mean(NEW, variant, decl["fits"]["seeds"], k, a.smoke)
            used[f"{variant}:{k}"] = n
            res[k] = {"vs_control_8_seeds": reading(k, controls[k], m), "vs_v54_member_16_seeds": reading(k, v54m[k], m),
                      "standalone_rmse": standalone(k, m), "control_standalone_rmse": standalone(k, controls[k])}
        results[variant] = res
    null_jan = abs(null["natural:2025-01"]["primary"]["relative_change"])
    bar = max(0.0025, 2 * null_jan)
    go, score = [], {}
    for variant, r in results.items():
        jan = r["natural:2025-01"]["vs_control_8_seeds"]["primary"]["relative_change"]
        tj = r["thinned_july"]["vs_control_8_seeds"]["primary"]["relative_change"]
        score[variant] = jan
        if jan <= -bar and tj <= 0:
            go.append(variant)
        print(f"{variant}: Jan {100 * jan:+.3f}% thinned July {100 * tj:+.3f}%", "GO" if variant in go else "-",
              {k: (round(100 * v["vs_v54_member_16_seeds"]["primary"]["relative_change"], 3), round(v["standalone_rmse"], 2), round(v["control_standalone_rmse"], 2))
               for k, v in r.items()}, flush=True)
    selected = min(go, key=lambda v: score[v]) if go else None
    report = {"created_at_utc": utc_now(), "smoke": a.smoke, "declaration": {DECL: {"sha256": sha256_file(ROOT / DECL), "object_hash": object_hash(decl)}},
              "code": {q: sha256_file(ROOT / q) for q in ["tools/lirf_mask2_report.py", "tools/run_lirf_dark.py", "tools/run_lirf_member.py", "src/prc2026_final/lirf_gate.py"]},
              "inputs": inputs, "checks": checks, "seeds_used": used, "arithmetic": {"w": W, "s": S, "T": T}, "seed_split_null": null,
              "bar": bar, "results": results, "go": go, "selected_variant": None if a.smoke else selected, "rule": decl["rule"]}
    out = ROOT / ("runs/lirf_mask2/smoke.json" if a.smoke else decl["report"])
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, report)
    print("null Jan", round(100 * null["natural:2025-01"]["primary"]["relative_change"], 3), "thinned", round(100 * null["thinned_july"]["primary"]["relative_change"], 3),
          "bar", round(100 * bar, 3), "GO:", go, "selected:", None if a.smoke else selected, "->", out)


if __name__ == "__main__":
    main()
