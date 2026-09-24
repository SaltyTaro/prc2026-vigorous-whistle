"""adsb_rnn design readings (configs/adsb_rnn_design_v1.json protocol): October 2025 only.

Every residual network design run (runs/adsb_rnn/design/<variant>_s<seed>/2025-10) and every variant's seed mean (first k seeds,
sorted) is read against v25's residual stage on October's in-scope rows (matched_no_lirf), where the control residual is the mean of
the adsb_v25 leave-one-month-out columns adj2 and cat for October (runs/adsb_v25/dev/predictions_2025-10.parquet):

  scope      RMSE of v9 + r for the network alone, the equal thirds mean(adj2, cat, net) and (1 - w) x control + w x net, w = 0.1 .. 0.9;
             the correlation of the network's and the control's residual errors
  recipe     v25's non-LIRF recipe rebuilt on October from October-out-of-sample parts: max(0.3 x (v9 + r) + 0.7 x (0.5 x committee +
             0.5 x the 32-seed network pool), floor), with the committee rebuilt on October (runs/adsb_nn/design/committee_october.parquet)
             and the pool the mean of the 32 nnb seeds' seasonal-fold October predictions (seeds 101-132 = ens_nn + ens_new, equal
             weights); LIRF rows stay at v9 for every candidate. Ratio to the control on all rows (primary) and on non-LIRF rows.
             Descriptive: the outer weight 0.2 .. 0.5 for the control and the best candidate.
No recipe or ranking month is read. Output: a JSON report and a printed table.
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
from prc2026.features import ID, TARGET, TIME  # noqa: E402
from nn_design_report import day_interval  # noqa: E402
from run_adsb_v11 import candidate_values, scope_mask  # noqa: E402
from run_adsb_v24 import adsb_features, load_frame, month_table  # noqa: E402

OCTOBER = "2025-10"
WEIGHTS = [round(0.1 * i, 1) for i in range(1, 10)]
OUTER = [0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
POOL_SEEDS = list(range(101, 133))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=Path("configs/adsb_rnn_design_v1.json"))
    parser.add_argument("--runs", type=Path, default=Path("runs/adsb_rnn/design"))
    parser.add_argument("--report", type=Path, default=Path("docs/adsb_rnn/design_report.json"))
    args = parser.parse_args()
    design = read_json(ROOT / args.design)
    network_config = read_json(ROOT / design["network_config"]["path"])
    if sha256_file(ROOT / design["network_config"]["path"]) != design["network_config"]["sha256"]:
        raise ValueError("The network configuration changed since the design declaration.")
    stage = network_config["residual_stage"]
    table = month_table(load_frame(stage, OCTOBER, "cross_fit"), adsb_features(OCTOBER, ROOT / stage["matches_dirs"]["v11"], ROOT / "runs/adsb_v24/features"), OCTOBER)
    ids = table[ID]
    label, v9 = table[TARGET].to_numpy(dtype=float), table["v9"].to_numpy(dtype=float)
    floor = float(table["floor"].iloc[0])
    inside = scope_mask(table, "matched_no_lirf")
    non_lirf = table["airport"].ne("LIRF").to_numpy()
    days = table[TIME].dt.strftime("%Y-%m-%d").to_numpy()
    columns = pd.read_parquet(ROOT / design["inputs"]["residual_columns"]).set_index(ID).reindex(ids)
    members = {m: columns[f"residual:{m}"].to_numpy(dtype=float) for m in ("adj2", "cat")}
    control_r = np.mean(list(members.values()), axis=0)
    committee = pd.read_parquet(ROOT / design["inputs"]["committee_october"]).set_index(ID)["committee"].reindex(ids).to_numpy(dtype=float)
    pool = np.mean([pd.read_parquet(ROOT / f"runs/adsb_nn/nnb_s{s}/seasonal/stop_predictions.parquet").set_index(ID)["adsb_full"].reindex(ids).to_numpy(dtype=float)
                    for s in POOL_SEEDS], axis=0)
    both = np.isfinite(committee) & np.isfinite(pool)
    table["_mix"] = np.where(both, 0.5 * committee + 0.5 * pool, committee)
    if not np.isfinite(table.loc[non_lirf, "_mix"]).all() or not np.isfinite(control_r).all():
        raise ValueError("October parts do not cover every row.")

    def recipe(residual: np.ndarray, outer: float = 0.3) -> np.ndarray:
        spec = {"weight": 1.0, "scope": "matched_no_lirf", "blend_with": {"name": "_mix", "scope": "all_no_lirf", "weight": 1 - outer}, "extra_blends": []}
        values, _ = candidate_values(table, v9, residual, spec, floor)
        return values

    control = recipe(control_r)
    base_all, base_nl = rmse(label, control), rmse(label[non_lirf], control[non_lirf])
    y_s, v9_s, c_s = label[inside], v9[inside], control_r[inside]

    def reading(net: np.ndarray) -> dict:
        n_s = net[inside]
        out = {"scope": {"v9": rmse(y_s, v9_s), "adj2": rmse(y_s, v9_s + members["adj2"][inside]), "cat": rmse(y_s, v9_s + members["cat"][inside]),
                         "control": rmse(y_s, v9_s + c_s), "network": rmse(y_s, v9_s + n_s),
                         "thirds": rmse(y_s, v9_s + (members["adj2"][inside] + members["cat"][inside] + n_s) / 3),
                         "blend": {str(w): rmse(y_s, v9_s + (1 - w) * c_s + w * n_s) for w in WEIGHTS},
                         "error_correlation": float(np.corrcoef(y_s - v9_s - c_s, y_s - v9_s - n_s)[0, 1])},
               "recipe": {"thirds": {}, "blend": {}}}
        thirds = (members["adj2"] + members["cat"] + net) / 3
        v = recipe(thirds)
        out["recipe"]["thirds"] = {"ratio_all": rmse(label, v) / base_all, "ratio_non_lirf": rmse(label[non_lirf], v[non_lirf]) / base_nl}
        for w in WEIGHTS:
            v = recipe((1 - w) * control_r + w * net)
            out["recipe"]["blend"][str(w)] = {"ratio_all": rmse(label, v) / base_all, "ratio_non_lirf": rmse(label[non_lirf], v[non_lirf]) / base_nl}
        best = min(WEIGHTS, key=lambda w: out["recipe"]["blend"][str(w)]["ratio_all"])
        best_r = (1 - best) * control_r + best * net
        out["recipe"]["best_w"] = best
        out["recipe"]["best_pct_all"] = 100 * (out["recipe"]["blend"][str(best)]["ratio_all"] - 1)
        out["recipe"]["best_day_interval_seconds"] = day_interval(days, label, recipe(best_r), control)
        out["recipe"]["outer_descriptive"] = {str(o): {"control": rmse(label, recipe(control_r, o)) / base_all, "best_w": rmse(label, recipe(best_r, o)) / base_all} for o in OUTER}
        return out

    runs = {}
    for path in sorted((ROOT / args.runs).glob(f"*/{OCTOBER}/run.json")):
        record = read_json(path)
        if record["stage"] != "design" or record["smoke"] or record["config_sha256"] != design["network_config"]["sha256"]:
            raise ValueError(f"{path} is not a design run of the declared network configuration.")
        frame = pd.read_parquet(path.parent / "predictions.parquet").set_index(ID).reindex(ids)
        if frame["residual"].isna().any() or not np.array_equal(frame["in_scope"].to_numpy(dtype=bool), inside):
            raise ValueError(f"{path.parent} does not cover October's scope.")
        runs[path.parent.parent.name] = {"variant": record["variant"], "seed": record["seed"], "values": frame["residual"].to_numpy(dtype=float),
                                         "curve": [round(e["monitor_rmse"], 3) for e in record["history"]], "fit_seconds": record["fit_seconds"],
                                         "best_monitor_epoch": record["best_monitor_epoch"], "sha256": sha256_file(path.parent / "predictions.parquet")}
    entries = {}
    for name, run in runs.items():
        entries[name] = {**reading(run["values"]), **{k: run[k] for k in ["variant", "seed", "curve", "fit_seconds", "best_monitor_epoch", "sha256"]}}
    by_variant = {}
    for name, run in sorted(runs.items(), key=lambda kv: (kv[1]["variant"], kv[1]["seed"])):
        by_variant.setdefault(run["variant"], []).append(run["values"])
    for variant, values in by_variant.items():
        for k in range(2, len(values) + 1):
            entries[f"{variant}_mean{k}"] = {**reading(np.mean(values[:k], axis=0)), "variant": variant, "seeds": k}
    if len(by_variant) == 2 and all(len(v) >= 2 for v in by_variant.values()):
        a, b = sorted(by_variant)
        entries[f"mix_{a}2_{b}2"] = {**reading(np.mean(by_variant[a][:2] + by_variant[b][:2], axis=0)), "seeds": 4}
    decision = {"rule": design["protocol"]["decision"]}
    four = {v: entries.get(f"{v}_mean4") for v in by_variant}
    decision["mean4_best_pct_all"] = {v: (e["recipe"]["best_pct_all"] if e else None) for v, e in four.items()}
    decision["mean4_best_w"] = {v: (e["recipe"]["best_w"] if e else None) for v, e in four.items()}
    ready = {v: p for v, p in decision["mean4_best_pct_all"].items() if p is not None}
    decision["better_variant"] = min(ready, key=lambda v: ready[v]) if ready else None
    decision["go"] = bool(ready and min(ready.values()) <= -100 * design["protocol"]["go_threshold_fraction"])
    report = {"created_at_utc": utc_now(), "design": args.design.as_posix(), "design_sha256": sha256_file(ROOT / args.design),
              "rows": int(len(table)), "rows_in_scope": int(inside.sum()), "rows_non_lirf": int(non_lirf.sum()),
              "control": {"recipe_rmse_all": base_all, "recipe_rmse_non_lirf": base_nl, "scope_rmse": rmse(y_s, v9_s + c_s)},
              "inputs": {"residual_columns": sha256_file(ROOT / design["inputs"]["residual_columns"]), "committee_october": sha256_file(ROOT / design["inputs"]["committee_october"]),
                         "scratch": sha256_file(ROOT / stage["cross_fit"]["scratch"] / f"{OCTOBER}.parquet")},
              "readings": entries, "decision": decision, "audit_used": False, "note": "October only; no recipe or ranking month read."}
    write_json(ROOT / args.report, report)
    print(f"control: scope {report['control']['scope_rmse']:.3f}  recipe all {base_all:.3f}  non-LIRF {base_nl:.3f}  (rows {len(table)}, scope {int(inside.sum())})")
    print(f"{'run':16s} {'net':>8s} {'thirds':>8s} {'best w':>6s} {'blend@w':>8s} {'errcorr':>7s} | {'rec thirds%':>11s} {'rec best%':>9s} {'w*':>4s} {'nonLIRF%':>8s} interval")
    for name, e in entries.items():
        s, r = e["scope"], e["recipe"]
        bw = min(WEIGHTS, key=lambda w: s["blend"][str(w)])
        print(f"{name:16s} {s['network']:8.3f} {s['thirds']:8.3f} {bw:6.1f} {s['blend'][str(bw)]:8.3f} {s['error_correlation']:7.4f} | "
              f"{100 * (r['thirds']['ratio_all'] - 1):11.4f} {r['best_pct_all']:9.4f} {r['best_w']:4.1f} {100 * (r['blend'][str(r['best_w'])]['ratio_non_lirf'] - 1):8.4f} "
              f"[{r['best_day_interval_seconds'][0]:.3f}, {r['best_day_interval_seconds'][1]:.3f}]  {e.get('curve', [])[-3:]}")
    print("decision", decision)


if __name__ == "__main__":
    main()
