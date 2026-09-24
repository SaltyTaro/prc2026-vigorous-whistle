"""Design readings for the neural committee member (configs/adsb_nn_design_v1.json protocol): October only.

For every design run found under --runs (<variant>_s<seed>/design/{run.json, stop_predictions.parquet}) and for every variant's
seed mean, on the monitored October rows (non-LIRF): the network's RMSE, the committee's RMSE, the RMSE of (1 - a) * committee +
a * network for a = 0.05 .. 0.60, the best weight and its gain, the gain split by ADS-B match, the correlation of the two error
vectors, and a paired-day bootstrap interval of the blend-minus-committee RMSE at the best weight (descriptive). No scoring month
is read. Output: a JSON report and a printed table.
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

WEIGHTS = [round(0.05 * i, 2) for i in range(1, 13)]
OCTOBER = "2025-10"


class RecipeProxy:
    """The v23 recipe rebuilt on October from October-out-of-sample parts: cross-fitted v9 (scratch/v24, fold B held October out),
    the v25 leave-one-month-out residual columns for October (members adj2 and cat), and 0.7 x committee on non-LIRF rows. The LIRF
    learner table has no October prediction, so LIRF rows stay at v9 for every candidate (the network never touches them)."""

    def __init__(self, ref: pd.DataFrame):
        from run_adsb_v11 import candidate_values
        from run_adsb_v24 import adsb_features, load_frame, month_table
        config = read_json(ROOT / "configs/adsb_v28_v2.json")
        table = month_table(load_frame(config, OCTOBER, "cross_fit"), adsb_features(OCTOBER, ROOT / config["matches_dirs"]["v11"], ROOT / "runs/adsb_v24/features"), OCTOBER)
        columns = pd.read_parquet(ROOT / config["residual_columns"][OCTOBER]).set_index(ID).reindex(table[ID])
        self.residual = np.mean([columns[f"residual:{m}"].to_numpy(dtype=float) for m in config["residual_members"]], axis=0)
        self.table, self.config, self.candidate_values = table, config, candidate_values
        self.label = table[TARGET].to_numpy(dtype=float)
        self.non_lirf = table["airport"].ne("LIRF").to_numpy()
        self.committee = ref.set_index(ID)["committee"].reindex(table[ID]).to_numpy(dtype=float)
        self.ids = table[ID]
        self.control = self.values(self.committee)
        self.inputs = {"scratch": sha256_file(ROOT / config["cross_fit"]["scratch"] / f"{OCTOBER}.parquet"), "residual_columns": sha256_file(ROOT / config["residual_columns"][OCTOBER])}

    def values(self, committee_table: np.ndarray) -> np.ndarray:
        self.table["_committee"] = committee_table
        spec = {"weight": 1.0, "scope": self.config["residual_scope"], "blend_with": {"name": "_committee", **{k: self.config["blend_with"][k] for k in ("scope", "weight")}}, "extra_blends": []}
        values, _ = self.candidate_values(self.table, self.table["v9"].to_numpy(dtype=float), self.residual, spec, float(self.table["floor"].iloc[0]))
        return values

    def reading(self, network_by_id: pd.Series) -> dict:
        network = network_by_id.reindex(self.ids).to_numpy(dtype=float)
        both = np.isfinite(self.committee) & np.isfinite(network)
        out = {"rows": int(len(self.label)), "control_rmse_all": rmse(self.label, self.control), "ratio_all": {}, "ratio_non_lirf": {}}
        for a in WEIGHTS:
            v = self.values(np.where(both, (1 - a) * self.committee + a * network, self.committee))
            out["ratio_all"][str(a)] = rmse(self.label, v) / out["control_rmse_all"]
            out["ratio_non_lirf"][str(a)] = rmse(self.label[self.non_lirf], v[self.non_lirf]) / rmse(self.label[self.non_lirf], self.control[self.non_lirf])
        best = min(WEIGHTS, key=lambda a: out["ratio_all"][str(a)])
        out["best_weight"], out["best_pct_all"] = best, 100 * (out["ratio_all"][str(best)] - 1)
        return out


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def day_interval(days: np.ndarray, label: np.ndarray, candidate: np.ndarray, reference: np.ndarray, seed: int = 20260923) -> list[float]:
    frame = pd.DataFrame({"day": days, "c": (candidate - label) ** 2, "r": (reference - label) ** 2})
    grouped = frame.groupby("day").agg(c=("c", "sum"), r=("r", "sum"), n=("c", "size"))
    idx = np.random.default_rng(seed).integers(0, len(grouped), size=(2000, len(grouped)))
    n = grouped["n"].to_numpy()[idx].sum(axis=1)
    delta = np.sqrt(grouped["c"].to_numpy()[idx].sum(axis=1) / n) - np.sqrt(grouped["r"].to_numpy()[idx].sum(axis=1) / n)
    return list(map(float, np.quantile(delta, [0.025, 0.975])))


def reading(label, committee, network, matched, days) -> dict:
    base = rmse(label, committee)
    blends = {a: rmse(label, (1 - a) * committee + a * network) for a in WEIGHTS}
    best = min(blends, key=lambda a: blends[a])
    blend = (1 - best) * committee + best * network
    out = {"network_rmse": rmse(label, network), "committee_rmse": base, "blend_rmse": blends,
           "best_weight": best, "best_gain_pct": 100 * (blends[best] / base - 1),
           "gain_pct_at": {str(a): 100 * (blends[a] / base - 1) for a in (0.1, 0.2, 0.3)},
           "error_correlation": float(np.corrcoef(label - committee, label - network)[0, 1]),
           "prediction_correlation": float(np.corrcoef(committee, network)[0, 1]),
           "best_day_interval_seconds": day_interval(days, label, blend, committee)}
    for name, k in [("matched", matched), ("unmatched", ~matched)]:
        out[f"{name}_rows"] = int(k.sum())
        out[f"{name}_committee_rmse"] = rmse(label[k], committee[k])
        out[f"{name}_gain_pct_at_best"] = 100 * (rmse(label[k], blend[k]) / rmse(label[k], committee[k]) - 1)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/adsb_nn/design"))
    parser.add_argument("--committee", type=Path, default=Path("runs/adsb_nn/design/committee_october.parquet"))
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    ref = pd.read_parquet(ROOT / args.committee)
    keep = ref["airport"].ne("LIRF").to_numpy()
    ref = ref.loc[keep].reset_index(drop=True)
    label, committee = ref["label"].to_numpy(dtype=float), ref["committee"].to_numpy(dtype=float)
    matched, days = ref["adsb_matched"].eq(1).to_numpy(), ref["day"].to_numpy()
    proxy = RecipeProxy(pd.read_parquet(ROOT / args.committee))
    runs, by_variant = {}, {}
    for path in sorted((ROOT / args.runs).glob("*/design/run.json")):
        record = read_json(path)
        full = pd.read_parquet(path.parent / "stop_predictions.parquet").set_index(ID)["adsb_full"]
        values = full.reindex(ref[ID]).to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{path.parent} does not cover the monitored October rows.")
        name = path.parent.parent.name
        entry = reading(label, committee, values, matched, days)
        entry["recipe_proxy"] = proxy.reading(full)
        entry.update({"variant": record["variant"], "seed": record["seed"], "epochs": record["epochs"], "fit_seconds": record["fit_seconds"],
                      "parameters": record["parameters"], "curve": [round(e["monitor_rmse"], 3) for e in record["history"]],
                      "best_monitor_epoch": record["best_monitor_epoch"], "variant_sha256": record["variant_sha256"],
                      "predictions_sha256": sha256_file(path.parent / "stop_predictions.parquet")})
        runs[name] = entry
        by_variant.setdefault(record["variant_sha256"], {"variant": record["variant"], "values": [], "full": []})
        by_variant[record["variant_sha256"]]["values"].append(values)
        by_variant[record["variant_sha256"]]["full"].append(full)
    means = {}
    for group in by_variant.values():
        for n in range(2, len(group["values"]) + 1):  # the seed-scaling curve: the mean of the first n seeds found (sorted run names)
            entry = reading(label, committee, np.mean(group["values"][:n], axis=0), matched, days)
            entry["recipe_proxy"] = proxy.reading(pd.concat(group["full"][:n], axis=1).mean(axis=1))
            means[f"{group['variant']}_mean{n}"] = entry
    report = {"created_at_utc": utc_now(), "rows": int(len(ref)), "committee_rmse": rmse(label, committee), "committee_sha256": sha256_file(ROOT / args.committee),
              "recipe_proxy_rows": int(len(proxy.label)), "recipe_proxy_control_rmse_all": rmse(proxy.label, proxy.control), "recipe_proxy_inputs": proxy.inputs,
              "runs": runs, "seed_means": means, "audit_used": False, "note": "October only (the committee's early-stopping month); no scoring month read."}
    write_json(ROOT / args.report, report)
    print(f"committee October RMSE (non-LIRF, {len(ref)} rows): {report['committee_rmse']:.3f}; recipe proxy control (all {len(proxy.label)} rows): {report['recipe_proxy_control_rmse_all']:.3f}")
    print(f"{'run':24s} {'net':>8s} {'best a':>6s} {'gain%':>8s} {'@0.2%':>8s} {'matched%':>9s} {'unmatch%':>9s} {'err corr':>8s} {'recipe a':>8s} {'recipe%':>8s} {'r@0.3%':>8s}  last epochs")
    for name, e in [*runs.items(), *means.items()]:
        tail = e.get("curve", [])[-3:]
        r = e["recipe_proxy"]
        print(f"{name:24s} {e['network_rmse']:8.3f} {e['best_weight']:6.2f} {e['best_gain_pct']:8.4f} {e['gain_pct_at']['0.2']:8.4f} "
              f"{e['matched_gain_pct_at_best']:9.4f} {e['unmatched_gain_pct_at_best']:9.4f} {e['error_correlation']:8.4f} "
              f"{r['best_weight']:8.2f} {r['best_pct_all']:8.4f} {100 * (r['ratio_all']['0.3'] - 1):8.4f}  {tail}")


if __name__ == "__main__":
    main()
