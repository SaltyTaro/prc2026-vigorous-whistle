"""adsb_nn3 design readings (configs/adsb_nn3_design_v1.json protocol): October only, on the two declared LIRF blend scopes.

For seed means of nnl (adsb_nn2 design seeds 1-2 + this round's 3-4) and nnlr (this round's 1-6), and the equal-compute mix
nnl 1-2 + nnlr 1-2, on October 2025 rows of lirf_nonpop_matched and lirf_nonpop_unmatched (tools/run_adsb_v24 definitions) against the
cross-fitted v9 (scratch/v24/2025-10): RMSE of v9, of the network, and of (1 - b) x v9 + b x network for b = 0.1 .. 0.9 -- on the union
of the scopes (the primary reading), per scope and on long rows (v9 >= 7,200 s). The non-LIRF committee reading of each run and mean is
descriptive (tools/nn_design_report.reading against the committee October reference). No scoring month is read.
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
from run_adsb_v11 import SCOPES  # noqa: E402
import run_adsb_v24  # noqa: E402,F401  (registers lirf_nonpop_unmatched)

BLENDS = [round(0.1 * i, 1) for i in range(1, 10)]


def load(directory: Path, variant: str) -> dict[int, pd.Series]:
    out = {}
    for path in sorted(directory.glob(f"{variant}_s*/design/run.json")):
        out[int(read_json(path)["seed"])] = pd.read_parquet(path.parent / "stop_predictions.parquet").set_index(ID)["adsb_full"]
    return dict(sorted(out.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/adsb_nn3/design"))
    parser.add_argument("--nn2-runs", type=Path, default=Path("runs/adsb_nn2/design"))
    parser.add_argument("--committee", type=Path, default=Path("runs/adsb_nn/design/committee_october.parquet"))
    parser.add_argument("--report", type=Path, default=Path("docs/adsb_nn3/design_report.json"))
    args = parser.parse_args()
    full_ref = pd.read_parquet(ROOT / args.committee)
    ref = full_ref.loc[full_ref["airport"].ne("LIRF")].reset_index(drop=True)
    proxy = RecipeProxy(full_ref)
    table = proxy.table
    label, v9 = table[TARGET].to_numpy(dtype=float), table["v9"].to_numpy(dtype=float)
    scopes = {s: SCOPES[s](table).to_numpy() for s in ["lirf_nonpop_matched", "lirf_nonpop_unmatched"]}
    union = scopes["lirf_nonpop_matched"] | scopes["lirf_nonpop_unmatched"]
    parts = {"union": union, **scopes, "long (v9 >= 7200 s)": union & (v9 >= 7200)}
    nnl = {**load(ROOT / args.nn2_runs, "nnl"), **load(ROOT / args.runs, "nnl")}
    nnlr = load(ROOT / args.runs, "nnlr")

    def mean(series: list[pd.Series]) -> pd.Series:
        return pd.concat(series, axis=1).mean(axis=1)

    def lirf_reading(series: pd.Series) -> dict:
        net = series.reindex(table[ID]).to_numpy(dtype=float)
        out = {}
        for name, k in parts.items():
            if not k.any():
                continue
            blends = {str(b): rmse(label[k], (1 - b) * v9[k] + b * net[k]) for b in BLENDS}
            best = min(blends, key=blends.get)
            out[name] = {"rows": int(k.sum()), "v9": rmse(label[k], v9[k]), "network": rmse(label[k], net[k]), "blend": blends,
                         "best_b": float(best), "best_rmse": blends[best]}
        return out

    def committee_reading(series: pd.Series) -> dict:
        values = series.reindex(ref[ID]).to_numpy(dtype=float)
        e = reading(ref["label"].to_numpy(dtype=float), ref["committee"].to_numpy(dtype=float), values, ref["adsb_matched"].eq(1).to_numpy(), ref["day"].to_numpy())
        return {k: e[k] for k in ["network_rmse", "best_weight", "best_gain_pct", "error_correlation"]}

    entries = {**{f"nnl_s{s}": v for s, v in nnl.items()}, **{f"nnlr_s{s}": v for s, v in nnlr.items()}}
    nl, nr = list(nnl.values()), list(nnlr.values())
    for k in range(2, len(nl) + 1):
        entries[f"nnl_mean{k}"] = mean(nl[:k])
    for k in range(2, len(nr) + 1):
        entries[f"nnlr_mean{k}"] = mean(nr[:k])
    entries["mix_nnl12_nnlr12"] = mean(nl[:2] + nr[:2])
    if len(nl) >= 4 and len(nr) >= 4:
        entries["mix_nnl4_nnlr4"] = mean(nl[:4] + nr[:4])
    report = {"created_at_utc": utc_now(), "committee_sha256": sha256_file(ROOT / args.committee), "rows": {n: int(k.sum()) for n, k in parts.items()},
              "readings": {name: {"lirf": lirf_reading(s), "non_lirf_committee": committee_reading(s)} for name, s in entries.items()},
              "audit_used": False, "note": "October only; no scoring month read."}
    primary = {name: r["lirf"]["union"]["best_rmse"] for name, r in report["readings"].items()}
    decision = {"nnlr_mean4": primary.get("nnlr_mean4"), "nnl_mean4": primary.get("nnl_mean4"), "mix_nnl12_nnlr12": primary.get("mix_nnl12_nnlr12")}
    decision["go"] = bool(decision["nnl_mean4"] is not None and ((decision["nnlr_mean4"] or np.inf) < decision["nnl_mean4"] or (decision["mix_nnl12_nnlr12"] or np.inf) < decision["nnl_mean4"]))
    report["decision"] = decision
    write_json(ROOT / args.report, report)
    print("rows", report["rows"])
    print(f"{'run':20s} {'union v9':>9s} {'net':>8s} {'best b':>6s} {'best':>8s} | {'matched best':>12s} {'unmatched best':>14s} {'long v9':>8s} {'long net':>9s} {'long best':>9s} | committee gain%")
    for name, r in report["readings"].items():
        u, m, un, lg = (r["lirf"].get(p, {}) for p in ["union", "lirf_nonpop_matched", "lirf_nonpop_unmatched", "long (v9 >= 7200 s)"])
        print(f"{name:20s} {u['v9']:9.2f} {u['network']:8.2f} {u['best_b']:6.1f} {u['best_rmse']:8.2f} | {m.get('best_rmse', float('nan')):12.2f} {un.get('best_rmse', float('nan')):14.2f} "
              f"{lg.get('v9', float('nan')):8.1f} {lg.get('network', float('nan')):9.1f} {lg.get('best_rmse', float('nan')):9.1f} | {r['non_lirf_committee']['best_gain_pct']:.4f}")
    print("decision", decision)


if __name__ == "__main__":
    main()
