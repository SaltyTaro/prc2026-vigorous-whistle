"""adsb_nn2 design readings (configs/adsb_nn2_design_v1.json protocol): October only.

On the monitored non-LIRF October rows, against the committee October reference (tools/nn_design_report.py's readings and recipe
proxy, imported unchanged):
  - every nmres and nnl design run alone, and the recorded nnb design seeds (runs/adsb_nn/design/base_s1..s8) for reference;
  - seed means of each variant;
  - the declared equal-compute comparison: mean(base 1..k + nmres 1..k) against mean(base 1..2k), k = 1..min(4, #nmres).
For nnl, also a descriptive reading on October LIRF non-population rows against the cross-fitted v9 (scratch/v24/2025-10), split by
ADS-B match: RMSE of v9, of the network, and of (1 - b) x v9 + b x network for b = 0.1 .. 0.9. No scoring month is read.
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
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET  # noqa: E402
from nn_design_report import RecipeProxy, reading, rmse  # noqa: E402


def load_runs(directory: Path, variant: str) -> dict[int, pd.Series]:
    out = {}
    for path in sorted(directory.glob(f"{variant}_s*/design/run.json")):
        record = read_json(path)
        out[int(record["seed"])] = pd.read_parquet(path.parent / "stop_predictions.parquet").set_index(ID)["adsb_full"]
    return dict(sorted(out.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/adsb_nn2/design"))
    parser.add_argument("--base-runs", type=Path, default=Path("runs/adsb_nn/design"))
    parser.add_argument("--committee", type=Path, default=Path("runs/adsb_nn/design/committee_october.parquet"))
    parser.add_argument("--report", type=Path, default=Path("docs/adsb_nn2/design_report.json"))
    args = parser.parse_args()
    full_ref = pd.read_parquet(ROOT / args.committee)
    ref = full_ref.loc[full_ref["airport"].ne("LIRF")].reset_index(drop=True)
    label, committee = ref["label"].to_numpy(dtype=float), ref["committee"].to_numpy(dtype=float)
    matched, days = ref["adsb_matched"].eq(1).to_numpy(), ref["day"].to_numpy()
    proxy = RecipeProxy(full_ref)
    runs = {"base": load_runs(ROOT / args.base_runs, "base"), "nmres": load_runs(ROOT / args.runs, "nmres"), "nnl": load_runs(ROOT / args.runs, "nnl")}

    def entry(series: pd.Series) -> dict:
        values = series.reindex(ref[ID]).to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("A design run does not cover the monitored October rows.")
        out = reading(label, committee, values, matched, days)
        out["recipe_proxy"] = proxy.reading(series)
        return out

    def mean(series: list[pd.Series]) -> pd.Series:
        return pd.concat(series, axis=1).mean(axis=1)

    report = {"created_at_utc": utc_now(), "rows": int(len(ref)), "committee_rmse": rmse(label, committee), "committee_sha256": sha256_file(ROOT / args.committee),
              "recipe_proxy_control_rmse_all": rmse(proxy.label, proxy.control), "single": {}, "means": {}, "equal_compute": {}, "nnl_lirf": {},
              "audit_used": False, "note": "October only; no scoring month read."}
    for variant, by_seed in runs.items():
        for seed, series in by_seed.items():
            report["single"][f"{variant}_s{seed}"] = entry(series)
        for k in range(2, len(by_seed) + 1):
            report["means"][f"{variant}_mean{k}"] = entry(mean(list(by_seed.values())[:k]))
    base, nmres = list(runs["base"].values()), list(runs["nmres"].values())
    for k in range(1, min(4, len(nmres)) + 1):
        mixed = entry(mean(base[:k] + nmres[:k]))
        pure = entry(mean(base[:2 * k])) if k > 1 else report["means"]["base_mean2"]
        report["equal_compute"][f"k{k}"] = {"mixed_base_nmres": mixed, "base_2k": pure,
                                            "mixed_minus_base_best_gain_pp": mixed["best_gain_pct"] - pure["best_gain_pct"],
                                            "mixed_minus_base_recipe_pp": mixed["recipe_proxy"]["best_pct_all"] - pure["recipe_proxy"]["best_pct_all"]}
    # nnl on October LIRF non-population rows, against the cross-fitted v9 (descriptive).
    table = proxy.table
    lirf = table["airport"].eq("LIRF").to_numpy() & ~table["population"].astype(bool).to_numpy()
    lab, v9 = table[TARGET].to_numpy(dtype=float), table["v9"].to_numpy(dtype=float)
    lmatched = table["adsb_matched"].eq(1).to_numpy()
    candidates = {**{f"nnl_s{s}": v for s, v in runs["nnl"].items()}, **({"nnl_mean": mean(list(runs["nnl"].values()))} if len(runs["nnl"]) > 1 else {}),
                  "base_mean8 (never saw LIRF)": mean(base[:8])}
    for name, series in candidates.items():
        net = series.reindex(table[ID]).to_numpy(dtype=float)
        block = {}
        for part, k in [("all", lirf), ("matched", lirf & lmatched), ("unmatched", lirf & ~lmatched)]:
            if not k.any():
                continue
            block[part] = {"rows": int(k.sum()), "v9": rmse(lab[k], v9[k]), "network": rmse(lab[k], net[k]),
                           "blend": {str(b): rmse(lab[k], (1 - b) * v9[k] + b * net[k]) for b in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}}
        report["nnl_lirf"][name] = block
    write_json(ROOT / args.report, report)

    print(f"committee October RMSE (non-LIRF, {len(ref)} rows): {report['committee_rmse']:.3f}; recipe proxy control {report['recipe_proxy_control_rmse_all']:.3f}")
    print(f"{'run':30s} {'net':>8s} {'best a':>6s} {'gain%':>8s} {'err corr':>8s} {'recipe%':>8s}")
    for name, e in [*report["single"].items(), *report["means"].items()]:
        print(f"{name:30s} {e['network_rmse']:8.3f} {e['best_weight']:6.2f} {e['best_gain_pct']:8.4f} {e['error_correlation']:8.4f} {e['recipe_proxy']['best_pct_all']:8.4f}")
    print("equal compute: mixed(base 1..k + nmres 1..k) vs base 1..2k")
    for k, e in report["equal_compute"].items():
        print(f"  {k}: mixed net {e['mixed_base_nmres']['network_rmse']:.3f} gain {e['mixed_base_nmres']['best_gain_pct']:.4f}% recipe {e['mixed_base_nmres']['recipe_proxy']['best_pct_all']:.4f}% | "
              f"base net {e['base_2k']['network_rmse']:.3f} gain {e['base_2k']['best_gain_pct']:.4f}% recipe {e['base_2k']['recipe_proxy']['best_pct_all']:.4f}% | "
              f"diff {e['mixed_minus_base_best_gain_pp']:+.4f} pp / recipe {e['mixed_minus_base_recipe_pp']:+.4f} pp")
    print("LIRF non-population October rows (vs cross-fitted v9):")
    for name, block in report["nnl_lirf"].items():
        print(" ", name, {part: {"rows": b["rows"], "v9": round(b["v9"], 2), "net": round(b["network"], 2),
                               "best_b": min(b["blend"], key=b["blend"].get), "best": round(min(b["blend"].values()), 2)} for part, b in block.items()})


if __name__ == "__main__":
    main()
