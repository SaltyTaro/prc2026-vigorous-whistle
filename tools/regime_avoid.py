"""coverage_regime_v1 step 3 (configs/coverage_regime_v1.json): is the residual stage's thinned-regime penalty avoidable?

--prepare  thin each of the eleven other 2025 months toward the evaluated month's target regime with the declared attempt's settings
           (tools/regime_thin_match.py, residual stand table only; LIRF is outside the residual scope and is not thinned here) and
           build their residual feature tables (run_adsb_v24.adsb_features on the thinned tables). Label-free.
--fit      fit adj2 (v25 settings) on the eleven months as `mix` (natural rows plus a thinned copy of every row of the thinned
           airports) and as `thinned` (those rows replaced by their thinned version), and score both, with step 2's natural-trained
           adj2, on the thinned and on the natural evaluated month: scope RMSE of v9 + adj2, and the recipe with a pure-adj2 residual
           (0.3 x (v9 + adj2) + 0.7 x committee mix on non-LIRF rows, LIRF blends as v26) built from step 2's saved blend tables.
Outputs: runs/coverage_regime/avoid/<month>/ and docs/coverage_regime/avoidability_<month>.json. Nothing is uploaded.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET  # noqa: E402
from run_adsb_nn_dev import mixed  # noqa: E402
from run_adsb_nn2_dev import lirf_blends, spec_for, weighted  # noqa: E402
from run_adsb_v11 import candidate_values, rmse, scope_mask  # noqa: E402
from run_adsb_v24 import adsb_features, load_frame, month_table  # noqa: E402

CONFIG = ROOT / "configs/coverage_regime_v1.json"
RECIPE = ROOT / "configs/adsb_rnn_v1.json"
V25 = ROOT / "configs/adsb_v25_v1.json"
RUNS = ROOT / "runs/coverage_regime"
REGIME = ROOT / "data/external/adsblol/regime"
DOCS = ROOT / "docs/coverage_regime"
TARGETS = {"2025-12": "2026-01", "2025-07": "2026-07"}
MONTHS = [f"2025-{i:02}" for i in range(1, 13)]


def regime(month: str) -> dict:
    declared = next(v for k, v in read_json(CONFIG)["regimes"].items() if k.startswith(month))
    record = read_json(RUNS / declared["attempt"] / f"attempt_{month}.json")
    airports = [a for a in declared["airports_thinned"] if a != "LIRF"]
    return {"attempt": declared["attempt"], "record": record, "airports": airports}


def pair(value, airports) -> str:
    if isinstance(value, dict):
        return ",".join(f"{a}={value.get(a, 0)}" for a in airports)
    return str(value)


def prepare(month: str, workers: int) -> None:
    spec = regime(month)
    rec, airports = spec["record"], spec["airports"]
    for m in [x for x in MONTHS if x != month]:
        attempt = f"s3_{month}_{m}"
        if not (RUNS / attempt / f"attempt_{m}.json").exists():
            cmd = [sys.executable, "-X", "utf8", "-B", str(ROOT / "tools/regime_thin_match.py"), "--attempt", attempt, "--source", m, "--target", TARGETS[month],
                   "--airports", *airports, "--alpha", pair(rec["alpha"], airports), "--session-drop", pair(rec["session_drop"], airports),
                   "--multiplier", pair(rec.get("multiplier") or {}, airports) if rec.get("multiplier") else "", "--density", rec["density"], "--prior", str(rec["prior"]),
                   "--seed", str(rec["seed"]), "--session-gap", str(rec["session_gap_s"]), "--workers", str(workers), "--stand-tables", "matches_v11"]
            if rec.get("calibrate"):
                cmd.append("--calibrate")
            if not rec.get("session_rescale", True):
                cmd.append("--no-session-rescale")
            if not rec.get("multiplier"):
                i = cmd.index("--multiplier")
                del cmd[i:i + 2]
            t0 = time.time()
            subprocess.run(cmd, check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
            print(month, m, "thinned", round(time.time() - t0, 1), flush=True)
        features = adsb_features(m, REGIME / attempt / "matches_v11", RUNS / attempt / "features" / "thinned" / "residual")
        print(month, m, "features", len(features), flush=True)


def fit(month: str, threads: int) -> None:
    from run_adsb_v25 import fit_variant, predict_variant
    spec = regime(month)
    airports = spec["airports"]
    v25, recipe = read_json(V25), read_json(RECIPE)
    natural_tables, thinned_tables, thin_stats = [], [], {}
    for m in [x for x in MONTHS if x != month]:
        attempt = f"s3_{month}_{m}"
        frame = load_frame(v25, m, "cross_fit")
        nat = month_table(frame, adsb_features(m, ROOT / v25["matches_dirs"]["v11"], ROOT / "runs/adsb_v24/features"), m)
        thin = month_table(frame, adsb_features(m, REGIME / attempt / "matches_v11", RUNS / attempt / "features" / "thinned" / "residual"), m)
        natural_tables.append(nat)
        thinned_tables.append(thin[thin["airport"].isin(airports)])
        s = read_json(RUNS / attempt / f"attempt_{m}.json")["stats"]
        thin_stats[m] = {a: {k: {kk: round(vv, 2) for kk, vv in s[k][a].items() if kk in ("match_pct", "gate_pct", "points_median") and vv is not None}
                             for k in ("natural_source", "thinned_source")} for a in airports if a in s["thinned_source"]}
    nat_all = pd.concat(natural_tables, ignore_index=True)
    thin_part = pd.concat(thinned_tables, ignore_index=True)
    trainings = {"mix": pd.concat([nat_all, thin_part], ignore_index=True),
                 "thinned": pd.concat([nat_all[~nat_all["airport"].isin(airports)], thin_part], ignore_index=True)}
    del natural_tables, thinned_tables
    declared = read_json(CONFIG)["regimes"]
    attempt = next(v for k, v in declared.items() if k.startswith(month))["attempt"]
    frame = load_frame(recipe, month, "cross_fit")
    evaluation = {kind: month_table(frame, pd.read_parquet(RUNS / attempt / "features" / kind / "residual" / f"matches_v11_{month}.parquet"), month) for kind in ("natural", "thinned")}
    saved = pd.read_parquet(RUNS / "measure" / month / "predictions.parquet")
    ids = evaluation["thinned"][ID].to_numpy()
    saved = saved.set_index(ID).reindex(ids)
    adj2 = {"natural_trained": {kind: saved[f"{kind}:residual:adj2"].to_numpy(dtype=float) for kind in ("natural", "thinned")}}
    report = {"created_at_utc": utc_now(), "month": month, "target_regime": TARGETS[month], "airports_thinned_in_training": airports,
              "training_thinning": thin_stats, "fits": {}, "declaration_sha256": sha256_file(CONFIG)}
    for name, train in trainings.items():
        t0 = time.time()
        model = fit_variant(train, v25, v25["residual_variants"]["adj2"], month, threads)
        report["fits"][name] = {"rows": model["rows"], "rounds": model["rounds"], "inner_best_iterations": model["inner_best_iterations"], "seconds": round(time.time() - t0, 1)}
        adj2[f"{name}_trained"] = {kind: predict_variant(model, evaluation[kind].set_index(ID).reindex(ids).reset_index(), v25) for kind in ("natural", "thinned")}
        print(month, name, report["fits"][name], flush=True)
    del trainings
    label = saved[TARGET].to_numpy(dtype=float)
    airports_col = saved["airport"].to_numpy()
    v25_recipe = recipe["v25_recipe"]
    floor = float(frame["floor"].iloc[0])
    scores = {}
    for kind in ("natural", "thinned"):
        table = evaluation[kind].set_index(ID).reindex(ids).reset_index()
        inside = scope_mask(table, recipe["residual_scope"])
        shifted = inside & np.isin(table["airport"].to_numpy(), airports)
        v9 = table["v9"].to_numpy(dtype=float)
        blend_table = table.copy()
        for column in saved.columns:
            if column.startswith(f"{kind}:") and ":residual:" not in column:
                blend_table[column.split(":", 1)[1]] = saved[column].to_numpy(dtype=float)
        committee = blend_table[recipe["committee_table"]].to_numpy(dtype=float)
        blend_table["committee_mix:v25"] = mixed(committee, weighted(blend_table, v25_recipe["network"]), float(recipe["network_share"]))
        spec = spec_for(recipe, "committee_mix:v25", lirf_blends(recipe, blend_table, v25_recipe, "v25"))
        for trained, values in adj2.items():
            r = values[kind]
            recipe_values, _ = candidate_values(blend_table, v9, r, spec, floor)
            scores[f"{kind}:{trained}"] = {"scope_rmse": rmse(label[inside], (v9 + r)[inside]), "scope_rows": int(inside.sum()),
                                           "shifted_scope_rmse": rmse(label[shifted], (v9 + r)[shifted]), "shifted_scope_rows": int(shifted.sum()),
                                           "pure_adj2_recipe_rmse": rmse(label, recipe_values),
                                           "pure_adj2_recipe_shifted_airports_rmse": rmse(label[np.isin(airports_col, airports)], recipe_values[np.isin(airports_col, airports)])}
    report["scores"] = scores
    base = scores["thinned:natural_trained"]
    report["avoidable"] = {name: {"shifted_scope_rmse_change_pct": 100 * (scores[f"thinned:{name}_trained"]["shifted_scope_rmse"] / base["shifted_scope_rmse"] - 1),
                                  "scope_rmse_change_pct": 100 * (scores[f"thinned:{name}_trained"]["scope_rmse"] / base["scope_rmse"] - 1),
                                  "pure_adj2_recipe_change_pct": 100 * (scores[f"thinned:{name}_trained"]["pure_adj2_recipe_rmse"] / base["pure_adj2_recipe_rmse"] - 1),
                                  "natural_month_scope_rmse_change_pct": 100 * (scores[f"natural:{name}_trained"]["scope_rmse"] / scores["natural:natural_trained"]["scope_rmse"] - 1)}
                           for name in ("mix", "thinned")}
    nat_scope = scores["natural:natural_trained"]
    report["residual_penalty_on_shifted_scope"] = {"note": "natural-trained adj2: thinned-month shifted-scope RMSE vs natural-month shifted-scope RMSE; the rows differ (thinning removes matches), so this is descriptive",
                                                   "natural_month": nat_scope["shifted_scope_rmse"], "thinned_month": base["shifted_scope_rmse"]}
    report["code_sha256"] = {p: sha256_file(ROOT / p) for p in ["tools/regime_avoid.py", "tools/regime_thin_match.py", "src/prc2026_regime/thinning.py"]}
    DOCS.mkdir(parents=True, exist_ok=True)
    write_json(DOCS / f"avoidability_{month}.json", report)
    print(json.dumps({"avoidable": report["avoidable"], "scores": {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in scores.items()}}, indent=1), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", required=True, choices=list(TARGETS))
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--threads", type=int, default=12)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.month, min(args.workers, 6))
    if args.fit:
        fit(args.month, min(args.threads, 12))


if __name__ == "__main__":
    main()
