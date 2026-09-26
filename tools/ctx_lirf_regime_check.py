"""Coverage-regime ship check of a candidate selected by configs/adsb_ctx_lirf_v1.json (round B, LIRF blends), against v27
(configs/adsb_ctx_lirf_regime_check_v1.json).

As tools/ctx_regime_check2.py (same months, thinning attempts, token rebuild, member folds, recipe assembly and gate), except that the
control and the candidate share v27's committee mix (ens_ctx, share 0.65) and differ only in the two LIRF non-population blends:
1. context tokens of the month rebuilt from the raw file with the natural and the thinned match tables; the natural rebuild must
   reproduce the stored token file (content hash);
2. ens_ctx natural and thinned from the round-1 check's saved re-predictions (hash-bound); every other context ensemble the candidate's
   LIRF network uses (ens_ctx_lirf) re-predicted here from its 16 learners (December: January-November refit models; July: seasonal);
   ens_nnl natural and thinned from the coverage campaign's saved tables;
3. the recipe from the coverage campaign's saved tables and residual members: v27 control and the candidate, natural and thinned. The
   natural control must reproduce v27's recorded development RMSE and the natural candidate its round-B development result (0.01 s).
Output: runs/adsb_ctx/regime_check_lirf/<month>.parquet and docs/adsb_ctx_lirf/regime_check_<month>.json. Nothing is uploaded.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET, TIME  # noqa: E402
from build_ctx_tokens import content_hash  # noqa: E402
from ctx_regime_check import NATURAL_MATCHES, TOKENS, build_store, ctx_predictions, stored_ctx  # noqa: E402
from run_adsb_nn_dev import mixed  # noqa: E402
from run_adsb_nn2_dev import lirf_blends, spec_for, weighted  # noqa: E402
from run_adsb_v11 import candidate_values, paired_days, rmse  # noqa: E402
from run_adsb_v24 import load_frame, month_table  # noqa: E402
from regime_measure import RUNS as REGIME_RUNS, features  # noqa: E402

DECLARATION = ROOT / "configs/adsb_ctx_lirf_regime_check_v1.json"
RECIPE = ROOT / "configs/adsb_rnn_v1.json"
ROUND_B = ROOT / "configs/adsb_ctx_lirf_v1.json"
REGIME = ROOT / "configs/coverage_regime_v1.json"
ROUND1 = ROOT / "runs/adsb_ctx/regime_check"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", required=True, choices=["2025-07", "2025-12"])
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    started = time.time()
    import torch
    torch.set_num_threads(args.threads)
    decl, recipe, round_b, regime_decl = read_json(DECLARATION), read_json(RECIPE), read_json(ROUND_B), read_json(REGIME)
    if sha256_file(REGIME) != decl["regimes"]["source_sha256"]:
        raise ValueError("The coverage-regime declaration changed.")
    if sha256_file(ROUND_B) != decl["round_b"]["sha256"]:
        raise ValueError("The round-B declaration changed.")
    selection = read_json(ROOT / decl["round_b"]["selection"])
    candidate = decl["candidate"]
    if selection["selected_candidate"] != candidate["name"] or round_b["candidates"][candidate["name"]] != {"lirf": candidate["lirf"], "lirf_network": candidate["lirf_network"]}:
        raise ValueError("The declared candidate is not round B's selection.")
    control = round_b["control"]
    committee_spec = {"network": control["committee_network"], "network_share": control["committee_share"]}
    regime = next(v for k, v in regime_decl["regimes"].items() if k.startswith(args.month))
    attempt = regime["attempt"]
    if sha256_file(REGIME_RUNS / attempt / f"attempt_{args.month}.json") != regime["attempt_record_sha256"]:
        raise ValueError("The thinning attempt differs from the declared one.")
    fold = "refit" if args.month == "2025-12" else "seasonal"
    report = {"created_at_utc": utc_now(), "month": args.month, "attempt": attempt, "declaration_sha256": sha256_file(DECLARATION), "member_fold": fold,
              "committee_mix": committee_spec, "control_lirf": {"lirf": control["lirf"], "lirf_network": control["lirf_network"]}, "candidate": candidate}

    # 1. tokens, natural (must reproduce the stored file) and thinned
    frames = {kind: features(attempt, args.month, kind) for kind in ("natural", "thinned")}
    ids = frames["natural"][0][ID].to_numpy()
    if not np.array_equal(ids, frames["thinned"][0][ID].to_numpy()):
        raise ValueError("Natural and thinned frames differ in rows.")
    stores, tmp = {}, Path(tempfile.mkdtemp(prefix="ctx_lirf_regime_"))
    thinned_matches = ROOT / "data/external/adsblol/regime" / attempt / "matches" / f"matches2_{args.month}.parquet"
    report["tokens"] = {}
    for kind, path in (("natural", NATURAL_MATCHES / f"matches2_{args.month}.parquet"), ("thinned", thinned_matches)):
        stores[kind], matched = build_store(args.month, path, tmp / kind)
        report["tokens"][kind] = {"matches": path.relative_to(ROOT).as_posix(), "matches_sha256": sha256_file(path), "matched_departure_tokens": matched,
                                  "content_sha256": content_hash(tmp / kind / f"{args.month}.npz")}
    report["tokens"]["natural_reproduces_stored"] = report["tokens"]["natural"]["content_sha256"] == content_hash(TOKENS / f"{args.month}.npz")
    if not report["tokens"]["natural_reproduces_stored"]:
        raise ValueError("The natural token rebuild differs from the stored token file.")
    print({"tokens": report["tokens"], "seconds": round(time.time() - started, 1)}, flush=True)

    # 2. ensembles: ens_ctx from the round-1 check, the candidate's new context ensembles re-predicted
    round1_report = read_json(ROOT / "docs/adsb_ctx" / f"regime_check_{args.month}.json")
    round1_path = ROUND1 / f"{args.month}.parquet"
    if sha256_file(round1_path) != round1_report["outputs"]["predictions_sha256"]:
        raise ValueError("The round-1 check's saved predictions changed.")
    round1 = pd.read_parquet(round1_path)
    if not np.array_equal(round1[ID].to_numpy(), ids):
        raise ValueError("The round-1 check's rows differ.")
    ens = {"ens_ctx": {kind: round1[f"ens_ctx:{kind}"].to_numpy(dtype=float) for kind in ("natural", "thinned")}}
    report["ensembles"] = {"ens_ctx": {"source": round1_path.relative_to(ROOT).as_posix(), "sha256": round1_report["outputs"]["predictions_sha256"]}}
    table_names = list(recipe["extra_feature_tables"])
    for name in sorted(set(candidate["lirf_network"]) | set(committee_spec["network"])):
        if name in ens or name in table_names:
            continue
        leaves = sorted(round_b["networks"][name]["learners"])
        preds = {"natural": [], "thinned": []}
        entry = {"learners": leaves, "networks": {}}
        for leaf in leaves:
            for kind in ("natural", "thinned"):
                preds[kind].append(ctx_predictions(ROOT / leaf, fold, frames[kind][0], stores[kind]))
            diff = np.abs(preds["natural"][-1] - stored_ctx(leaf, args.month, ids))
            entry["networks"][leaf] = {"natural_vs_stored_max_abs": float(np.max(diff)), "natural_vs_stored_rms": float(np.sqrt(np.mean(diff ** 2))),
                                       "natural_vs_stored_rows_over_1s": int((diff > 1.0).sum()),
                                       "rows_changed_by_thinning": int((np.abs(preds["thinned"][-1] - preds["natural"][-1]) > 1e-9).sum())}
            print(name, leaf, entry["networks"][leaf], flush=True)
        ens[name] = {kind: np.mean(preds[kind], axis=0) for kind in preds}
        diff = np.abs(ens[name]["natural"] - np.mean([stored_ctx(leaf, args.month, ids) for leaf in leaves], axis=0))
        entry["ens_natural_vs_stored"] = {"max_abs": float(np.max(diff)), "rms": float(np.sqrt(np.mean(diff ** 2))), "rows_over_1s": int((diff > 1.0).sum()),
                                          "rows_over_0p01s": int((diff > 0.01).sum())}
        print({name: entry["ens_natural_vs_stored"]}, flush=True)
        report["ensembles"][name] = entry

    # 3. the recipe, from the coverage campaign's saved tables and residual members
    saved = pd.read_parquet(REGIME_RUNS / "measure" / args.month / "predictions.parquet")
    if not np.array_equal(saved[ID].to_numpy(), ids):
        raise ValueError("The saved coverage-regime predictions are not in the frame's row order.")
    frame_v9 = load_frame(recipe, args.month, "cross_fit")
    month_tables = {}
    for kind in ("natural", "thinned"):
        table = month_table(frame_v9, frames[kind][1], args.month)
        month_tables[kind] = table if np.array_equal(table[ID].to_numpy(), ids) else table.set_index(ID).reindex(ids).reset_index()
    v26 = recipe["candidates"]["rnet_w90"]
    v25 = recipe["v25_recipe"]
    label = month_tables["natural"][TARGET].to_numpy(dtype=float)
    airports = month_tables["natural"]["airport"].to_numpy()
    days = month_tables["natural"][TIME].dt.strftime("%Y-%m-%d").to_numpy()
    floor = float(month_tables["natural"]["floor"].iloc[0])

    def recipe_values(kind: str, lirf: dict, who: str) -> np.ndarray:
        table = month_tables[kind].copy()
        for name in table_names:
            table[name] = saved[f"{kind}:{name}"].to_numpy(dtype=float)
        for name in ens:
            table[name] = ens[name][kind]
        committee = table[recipe["committee_table"]].to_numpy(dtype=float)
        table["committee_mix"] = mixed(committee, weighted(table, committee_spec["network"]), float(committee_spec["network_share"]))
        spec = spec_for(recipe, "committee_mix", lirf_blends(recipe, table, {**v25, "lirf": lirf["lirf"], "lirf_network": lirf["lirf_network"]}, who))
        spec = {**spec, "blend_with": {**spec["blend_with"], "weight": float(v26.get("recipe_weight", recipe["recipe_blend"]["weight"]))}}
        r = np.sum([w * saved[f"{kind}:residual:{m}"].to_numpy(dtype=float) for m, w in v26["residual"].items()], axis=0)
        values, _ = candidate_values(table, table["v9"].to_numpy(dtype=float), r, spec, floor)
        return values

    values = {(kind, who): recipe_values(kind, spec, who) for kind in ("natural", "thinned")
              for who, spec in (("control", report["control_lirf"]), ("candidate", candidate))}
    lirf_rows = airports == "LIRF"
    for kind in ("natural", "thinned"):
        if (np.abs(values[kind, "candidate"] - values[kind, "control"])[~lirf_rows] > 1e-9).any():
            raise ValueError(f"The candidate changes a non-LIRF row ({kind}).")
    recorded_control = control["recorded_development_rmse"][args.month]
    recorded_candidate = read_json(ROOT / decl["round_b"]["development_dir"] / f"month_{args.month}.json")["candidates"][candidate["name"]]["rmse"]
    rep = {"control_recorded": recorded_control, "control_rebuilt": rmse(label, values["natural", "control"]),
           "candidate_recorded": recorded_candidate, "candidate_rebuilt": rmse(label, values["natural", "candidate"]), "tolerance_s": 0.01}
    rep["control_diff_s"] = rep["control_rebuilt"] - recorded_control
    rep["candidate_diff_s"] = rep["candidate_rebuilt"] - recorded_candidate
    rep["control_ok"] = abs(rep["control_diff_s"]) <= 0.01
    rep["candidate_ok"] = abs(rep["candidate_diff_s"]) <= 0.01
    report["reproduction"] = rep
    if not (rep["control_ok"] and rep["candidate_ok"]):
        raise ValueError(f"The natural rebuild does not reproduce the development results: {rep}")
    matched = {kind: month_tables[kind]["adsb_matched"].eq(1).to_numpy() for kind in ("natural", "thinned")}
    report["regimes"] = {}
    for kind in ("natural", "thinned"):
        c, k = values[kind, "control"], values[kind, "candidate"]
        entry = {"control_rmse": rmse(label, c), "candidate_rmse": rmse(label, k), "change_pct": 100 * (rmse(label, k) / rmse(label, c) - 1),
                 "lirf_change_pct": 100 * (rmse(label[lirf_rows], k[lirf_rows]) / rmse(label[lirf_rows], c[lirf_rows]) - 1),
                 "rows_changed": int((np.abs(k - c) > 1e-9).sum()),
                 "paired_days": paired_days(days, label, k, c, recipe["seed"])}
        for part, mask in (("lirf_matched", lirf_rows & matched[kind]), ("lirf_unmatched", lirf_rows & ~matched[kind])):
            entry[part] = {"rows": int(mask.sum()), "control": rmse(label[mask], c[mask]), "candidate": rmse(label[mask], k[mask])} if mask.any() else None
        report["regimes"][kind] = entry
    nat, thin = report["regimes"]["natural"], report["regimes"]["thinned"]
    report["retained"] = {"whole_month": thin["change_pct"] / nat["change_pct"] if nat["change_pct"] else None,
                          "lirf": thin["lirf_change_pct"] / nat["lirf_change_pct"] if nat["lirf_change_pct"] else None}
    interval = thin["paired_days"]["rmse_difference_95pct_interval_seconds"]
    report["gate_month"] = {"thinned_candidate_below_control": thin["candidate_rmse"] < thin["control_rmse"], "thinned_interval_below_zero": bool(interval[1] < 0)}
    report["gate_month"]["passes"] = all(report["gate_month"].values())
    out = ROOT / "runs/adsb_ctx/regime_check_lirf"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ID: ids, **{f"{name}:{kind}": ens[name][kind] for name in ens for kind in ("natural", "thinned")},
                  **{f"{kind}:{who}": v for (kind, who), v in values.items()}}).to_parquet(out / f"{args.month}.parquet", index=False, compression="zstd")
    report["outputs"] = {"predictions_sha256": sha256_file(out / f"{args.month}.parquet")}
    report["code_sha256"] = {p: sha256_file(ROOT / p) for p in ["tools/ctx_lirf_regime_check.py", "tools/ctx_regime_check.py", "tools/regime_measure.py",
                                                                 "src/prc2026_ctx/tokens.py", "src/prc2026_ctx/train.py"]}
    shutil.rmtree(tmp, ignore_errors=True)
    report["seconds"] = round(time.time() - started, 1)
    write_json(ROOT / "docs/adsb_ctx_lirf" / f"regime_check_{args.month}.json", report)
    print({"reproduction": report["reproduction"], "natural": round(nat["change_pct"], 4), "thinned": round(thin["change_pct"], 4), "retained": report["retained"],
           "thinned_interval": interval, "gate_month": report["gate_month"]}, flush=True)
    print({part: (nat[part], thin[part]) for part in ("lirf_matched", "lirf_unmatched")}, flush=True)


if __name__ == "__main__":
    main()
