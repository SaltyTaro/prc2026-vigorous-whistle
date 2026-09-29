"""Design readings of the arrival second task (configs/adsb_arr_design_v1.json): October only, no scoring month read.

R1  October RMSE on the monitored non-LIRF rows: every run; each variant's seed mean against the plain network's seed mean at the
    same seeds (runs/adsb_nn/design/base_s*); the per-seed paired differences and their spread (the noise scale).
R2  Recipe proxy (tools/nn_design_report.RecipeProxy, all October rows): table(b) = (1 - s) x committee + s x [(1 - b) x context
    pool + b x member], against b = 0, with the paired-day 95% interval; s = 0.75 governs, s = 0.85 is reported beside it. The
    same grid with member = the plain network's seed mean is what four plain networks add without the second task.
R3  The decisive pair: mt_l05 against mt_l05_notarget.
Slices: R1 by airport and by ADS-B match; the monitored departures with a category outside the plain network's vocabulary.
Nesting: the --no-arrivals run against base_s1 (a code check).
Rule (declared): GO if for some variant and b the governing R2 is at or below -0.10% with its interval below zero, the variant's R2
is below base's at that b, and the variant's standalone seed mean is below base's. Output: a JSON report and a printed summary.
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
from prc2026.features import ID, TIME  # noqa: E402
from nn_design_report import RecipeProxy, day_interval, rmse  # noqa: E402
from ctx_design_report import october  # noqa: E402

B_VALUES = [0.125, 0.25, 0.5]
SHARES = [0.75, 0.85]
GOVERNING = 0.75
BAR_PCT = -0.10


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/adsb_arr_design_v1.json"))
    parser.add_argument("--runs", type=Path, default=Path("runs/adsb_arr/design"))
    parser.add_argument("--base-runs", type=Path, default=Path("runs/adsb_nn/design"))
    parser.add_argument("--pool-runs", type=Path, default=Path("runs/adsb_ctx"))
    parser.add_argument("--committee", type=Path, default=Path("runs/adsb_nn/design/committee_october.parquet"))
    parser.add_argument("--nesting", default="nest_s1")
    parser.add_argument("--report", type=Path, default=Path("docs/adsb_arr/design_report.json"))
    args = parser.parse_args()
    config = read_json(ROOT / args.config)
    seeds = list(config["seeds"])
    proxy = RecipeProxy(pd.read_parquet(ROOT / args.committee))
    ids, label, non_lirf, table = proxy.ids, proxy.label, proxy.non_lirf, proxy.table
    days = table[TIME].dt.strftime("%Y-%m-%d").to_numpy()
    matched = table["adsb_matched"].eq(1).to_numpy()
    airport = table["airport"].to_numpy()
    committee = proxy.committee
    k = non_lirf

    def read(path: Path) -> np.ndarray:
        values = pd.read_parquet(path).set_index(ID)["adsb_full"].reindex(ids).to_numpy(dtype=float)
        if not np.isfinite(values[k]).all():
            raise ValueError(f"{path} does not cover the monitored October rows.")
        return values

    ctx = [ROOT / args.pool_runs / f"ctx_s{s}/seasonal/stop_predictions.parquet" for s in range(101, 117)]
    e24 = [ROOT / args.pool_runs / f"ctx_e24_s{s}/seasonal/stop_predictions.parquet" for s in range(117, 133)]
    pool = 0.5 * np.mean([october(p, ids) for p in ctx], axis=0) + 0.5 * np.mean([october(p, ids) for p in e24], axis=0)
    base = {s: read(ROOT / args.base_runs / f"base_s{s}/design/stop_predictions.parquet") for s in seeds}

    runs, records = {}, {}
    for name in config["variants"]:
        for s in seeds:
            directory = ROOT / args.runs / f"{name}_s{s}/design"
            if not (directory / "run.json").exists():
                continue
            record = read_json(directory / "run.json")
            if record["smoke"] or record["variant"] != name or record["seed"] != s or record["config_sha256"] != sha256_file(ROOT / args.config):
                raise ValueError(f"{directory} is not a run of this declaration.")
            if sha256_file(directory / "stop_predictions.parquet") != record["predictions_sha256"]:
                raise ValueError(f"{directory}: predictions changed after the run.")
            runs.setdefault(name, {})[s] = read(directory / "stop_predictions.parquet")
            records[f"{name}_s{s}"] = record
    slices = None
    for name, by_seed in runs.items():
        for s in by_seed:
            flags = pd.read_parquet(ROOT / args.runs / f"{name}_s{s}/design/stop_slices.parquet").set_index(ID)["unseen_category"].reindex(ids).fillna(False).to_numpy(dtype=bool)
            if slices is not None and not np.array_equal(slices, flags):
                raise ValueError("The unseen-category slice differs between runs.")
            slices = flags

    def net(values: np.ndarray, mask: np.ndarray = k) -> float:
        return rmse(label[mask], values[mask])

    def proxy_reading(member: np.ndarray, share: float) -> dict:
        control = proxy.values((1 - share) * committee + share * pool)
        out = {"control_rmse": rmse(label, control), "by_b": {}}
        for b in B_VALUES:
            mixed = np.where(np.isfinite(member), (1 - b) * pool + b * member, pool)
            values = proxy.values((1 - share) * committee + share * mixed)
            out["by_b"][str(b)] = {"pct": 100 * (rmse(label, values) / out["control_rmse"] - 1), "interval_s": day_interval(days, label, values, control),
                                   "matched_pct": 100 * (rmse(label[k & matched], values[k & matched]) / rmse(label[k & matched], control[k & matched]) - 1),
                                   "unmatched_pct": 100 * (rmse(label[k & ~matched], values[k & ~matched]) / rmse(label[k & ~matched], control[k & ~matched]) - 1)}
        return out

    def member_reading(by_seed: dict, reference: dict | None) -> dict:
        used = sorted(by_seed)
        mean = np.mean([by_seed[s] for s in used], axis=0)
        out = {"seeds": used, "runs_rmse": {str(s): net(by_seed[s]) for s in used}, "seed_mean_rmse": net(mean),
               "matched_rmse": net(mean, k & matched), "unmatched_rmse": net(mean, k & ~matched),
               "airports_rmse": {a: net(mean, airport == a) for a in sorted(set(airport[k]))},
               "pool_correlation_of_errors": float(np.corrcoef(label[k] - pool[k], label[k] - mean[k])[0, 1]),
               "proxy": {str(share): proxy_reading(mean, share) for share in SHARES}}
        if slices is not None and slices.any():
            out["unseen_category_rows"] = int(slices.sum())
            out["unseen_category_rmse"] = net(mean, slices)
        if reference is not None:
            ref_mean = np.mean([reference[s] for s in used], axis=0)
            paired = [net(by_seed[s]) - net(reference[s]) for s in used]
            out["base_seed_mean_rmse_same_seeds"] = net(ref_mean)
            out["seed_mean_minus_base_pct"] = 100 * (net(mean) / net(ref_mean) - 1)
            out["paired_seed_differences_s"] = paired
            out["paired_mean_s"], out["paired_sd_s"] = float(np.mean(paired)), float(np.std(paired, ddof=1)) if len(paired) > 1 else None
            out["airports_minus_base_pct"] = {a: 100 * (net(mean, airport == a) / net(ref_mean, airport == a) - 1) for a in sorted(set(airport[k]))}
            out["matched_minus_base_pct"] = 100 * (net(mean, k & matched) / net(ref_mean, k & matched) - 1)
            out["unmatched_minus_base_pct"] = 100 * (net(mean, k & ~matched) / net(ref_mean, k & ~matched) - 1)
            if slices is not None and slices.any():
                out["unseen_category_minus_base_pct"] = 100 * (net(mean, slices) / net(ref_mean, slices) - 1)
        return out

    members = {"base": member_reading(base, None), **{name: member_reading(by_seed, base) for name, by_seed in runs.items()}}
    governing = str(GOVERNING)
    decisions = {}
    for name in runs:
        complete = sorted(runs[name]) == seeds
        for b in B_VALUES:
            entry = members[name]["proxy"][governing]["by_b"][str(b)]
            reference = members["base"]["proxy"][governing]["by_b"][str(b)]
            decisions[f"{name}@{b}"] = {"complete": complete, "pct": entry["pct"], "interval_s": entry["interval_s"], "base_pct": reference["pct"],
                                        "bar": bool(entry["pct"] <= BAR_PCT and entry["interval_s"][1] < 0), "below_base_member": bool(entry["pct"] < reference["pct"]),
                                        "standalone_below_base": bool(members[name]["seed_mean_minus_base_pct"] < 0)}
            decisions[f"{name}@{b}"]["go"] = bool(complete and all(decisions[f"{name}@{b}"][c] for c in ("bar", "below_base_member", "standalone_below_base")))
    pair = None
    if "mt_l05" in runs and "mt_l05_notarget" in runs:
        shared = sorted(set(runs["mt_l05"]) & set(runs["mt_l05_notarget"]))
        differences = [net(runs["mt_l05"][s]) - net(runs["mt_l05_notarget"][s]) for s in shared]
        a, b_ = np.mean([runs["mt_l05"][s] for s in shared], axis=0), np.mean([runs["mt_l05_notarget"][s] for s in shared], axis=0)
        pair = {"seeds": shared, "paired_differences_s": differences, "mean_s": float(np.mean(differences)), "sd_s": float(np.std(differences, ddof=1)) if len(shared) > 1 else None,
                "seed_mean_target_minus_notarget_pct": 100 * (net(a) / net(b_) - 1), "interval_s": day_interval(days[k], label[k], a[k], b_[k]),
                "unseen_category_pct": 100 * (net(a, slices) / net(b_, slices) - 1) if slices is not None and slices.any() else None,
                "proxy_pct_difference_by_b": {str(b): members["mt_l05"]["proxy"][governing]["by_b"][str(b)]["pct"] - members["mt_l05_notarget"]["proxy"][governing]["by_b"][str(b)]["pct"] for b in B_VALUES}}
    nesting = None
    directory = ROOT / args.runs / args.nesting / "design"
    if (directory / "run.json").exists():
        record = read_json(directory / "run.json")
        values, reference = read(directory / "stop_predictions.parquet"), base[record["seed"]]
        base_record = read_json(ROOT / args.base_runs / f"base_s{record['seed']}/design/run.json")
        nesting = {"seed": record["seed"], "rmse": net(values), "base_rmse": net(reference), "max_abs_difference_s": float(np.max(np.abs(values[k] - reference[k]))),
                   "rms_difference_s": float(np.sqrt(np.mean((values[k] - reference[k]) ** 2))), "threads": record["inputs"]["threads"], "base_threads": base_record["inputs"]["threads"],
                   "curve": [round(e["monitor_rmse"], 4) for e in record["history"]], "base_curve": [round(e["monitor_rmse"], 4) for e in base_record["history"]],
                   "vocabulary_equal": record["preprocessing"]["categories_kept"] == base_record["preprocessing"]["categories_kept"]}
    arrival_task = {n: {key: r["arrival"].get(key) for key in ["rows", "fit_month_rows", "target_month_rows", "lambda", "target_month_rmse", "target_month_rmse_of_fit_mean", "added_categories"]}
                    | {"fit_seconds": r["fit_seconds"], "curve_last": [round(e["monitor_rmse"], 3) for e in r["history"][-3:]], "arrival_loss_last": r["history"][-1].get("arrival_loss_scaled")}
                    for n, r in records.items()}
    report = {"created_at_utc": utc_now(), "config_sha256": sha256_file(ROOT / args.config), "rows_non_lirf": int(k.sum()), "rows_proxy": int(len(label)),
              "pool_rmse_non_lirf": net(pool), "committee_rmse_non_lirf": net(committee), "b_values": B_VALUES, "shares": SHARES, "governing_share": GOVERNING, "bar_pct": BAR_PCT,
              "members": members, "decisions": decisions, "go": bool(any(d["go"] for d in decisions.values())), "R3_target_against_notarget": pair, "nesting": nesting,
              "runs": arrival_task, "inputs": {"committee_sha256": sha256_file(ROOT / args.committee), "recipe_proxy_inputs": proxy.inputs,
                                               "tool_sha256": sha256_file(ROOT / "tools/arr_design_report.py"),
                                               "predictions_sha256": {n: r["predictions_sha256"] for n, r in records.items()}},
              "audit_used": False, "note": "October only (the committee's early-stopping month); no scoring month, ranking row or official score read."}
    (ROOT / args.report).parent.mkdir(parents=True, exist_ok=True)
    write_json(ROOT / args.report, report)
    print(f"October non-LIRF RMSE: context pool {report['pool_rmse_non_lirf']:.3f}, committee {report['committee_rmse_non_lirf']:.3f}; "
          f"control proxy {members['base']['proxy'][governing]['control_rmse']:.3f} (share {GOVERNING})")
    if nesting:
        print("nesting:", {key: nesting[key] for key in ["rmse", "base_rmse", "max_abs_difference_s", "rms_difference_s", "vocabulary_equal"]})
    print(f"{'member':18s} {'seeds':>5s} {'mean':>8s} {'vs base%':>9s} {'paired s':>9s} {'sd':>6s} | proxy% s=.75 b=.125 .25 .5 | s=.85 b=.125 .25 .5 | runs")
    for name, m in members.items():
        cells = " ".join(f"{m['proxy'][str(share)]['by_b'][str(b)]['pct']:+.4f}" for share in SHARES for b in B_VALUES)
        print(f"{name:18s} {len(m['seeds']):5d} {m['seed_mean_rmse']:8.3f} {m.get('seed_mean_minus_base_pct', 0.0):+9.4f} {m.get('paired_mean_s', 0.0):+9.3f} "
              f"{(m.get('paired_sd_s') or 0.0):6.3f} | {cells} | {[round(v, 2) for v in m['runs_rmse'].values()]}")
    for name, d in decisions.items():
        print(f"  {name:22s} pct {d['pct']:+.4f} interval [{d['interval_s'][0]:+.3f}, {d['interval_s'][1]:+.3f}] base {d['base_pct']:+.4f} "
              f"bar {d['bar']} below-base {d['below_base_member']} standalone {d['standalone_below_base']} -> {'GO' if d['go'] else 'no'}")
    if pair:
        print("R3 target against notarget:", {key: (round(v, 4) if isinstance(v, float) else v) for key, v in pair.items() if key != "proxy_pct_difference_by_b"},
              {b: round(v, 4) for b, v in pair["proxy_pct_difference_by_b"].items()})
    print("arrival task:", {n: (round(r["target_month_rmse"], 1), round(r["target_month_rmse_of_fit_mean"], 1), r["fit_seconds"]) for n, r in arrival_task.items()})
    print("GO" if report["go"] else "NO-GO")


if __name__ == "__main__":
    main()
