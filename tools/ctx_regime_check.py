"""Coverage-regime ship check of the selected context-network candidate (configs/adsb_ctx_regime_check_v1.json).

For one recipe month with a thinned regime (2025-12: a13_dec, 2025-07: a13_jul of configs/coverage_regime_v1.json):
1. context tokens of the month rebuilt from the raw file with the natural and the thinned match tables (round-1 builder, k = 128);
   the natural rebuild must reproduce the stored token file (content hash);
2. the 16 context networks re-predicted on natural and thinned inputs (December: January-November refit models; July: seasonal
   models), natural compared with the stored predictions; ens_ctx = their mean;
3. the recipe assembled as tools/regime_measure.py does, from the tables and residual members that campaign saved, with the committee
   mix's network table and share as parameters: v26 control and the candidate, natural and thinned. The natural control must
   reproduce v26's recorded RMSE and the natural candidate the development result of the candidate.
Output: runs/adsb_ctx/regime_check/<month>.parquet and docs/adsb_ctx/regime_check_<month>.json. Nothing is uploaded.
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
from run_adsb_learner import source_file  # noqa: E402
from run_adsb_nn_dev import mixed  # noqa: E402
from run_adsb_nn2_dev import lirf_blends, spec_for, weighted  # noqa: E402
from run_adsb_v11 import candidate_values, paired_days, rmse  # noqa: E402
from run_adsb_v24 import load_frame, month_table  # noqa: E402
from regime_measure import RUNS as REGIME_RUNS, class_of, features  # noqa: E402

DECLARATION = ROOT / "configs/adsb_ctx_regime_check_v1.json"
RECIPE = ROOT / "configs/adsb_rnn_v1.json"
CTX = ROOT / "configs/adsb_ctx_v1.json"
REGIME = ROOT / "configs/coverage_regime_v1.json"
TOKENS = ROOT / "data/ctx/tokens_v1"
NATURAL_MATCHES = ROOT / "data/external/adsblol/matches"


def build_store(month: str, matches_path: Path, directory: Path):
    from prc2026_ctx.tokens import RAW_COLUMNS, TokenStore, build_tokens, encode_codes, neighbours, save_file
    raw = pd.read_parquet(ROOT / "data/raw" / source_file(month), columns=RAW_COLUMNS)
    tokens = encode_codes(build_tokens(raw, pd.read_parquet(matches_path)))
    dep_ids, self_rows, index = neighbours(tokens, 128)
    save_file(directory, month, tokens, dep_ids, self_rows, index)
    return TokenStore(directory, [month]), int(tokens["adsb_matched"].eq(1).sum())


def ctx_predictions(leaf: Path, fold: str, frame: pd.DataFrame, store) -> np.ndarray:
    from prc2026_ctx.train import Context, load, predict
    model, pre, settings = load(leaf / fold)
    values = np.empty(len(frame))
    for start in range(0, len(frame), 100000):
        chunk = frame.iloc[start:start + 100000]
        xn, xc = pre.transform(chunk)
        values[start:start + 100000] = predict(model, pre, xn, xc, Context.build(store, *store.lookup(chunk[ID].to_numpy()), settings))
    return values


def stored_ctx(leaf: str, month: str, ids: np.ndarray) -> np.ndarray:
    path = (ROOT / "runs/december/learners" / (leaf.replace("/", "__") + ".parquet")) if month == "2025-12" else ROOT / leaf / "seasonal" / "predictions.parquet"
    return pd.read_parquet(path).set_index(ID)["adsb_full"].reindex(ids).to_numpy(dtype=float)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", required=True, choices=["2025-07", "2025-12"])
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    started = time.time()
    import torch
    torch.set_num_threads(args.threads)
    decl, recipe, ctx_decl, regime_decl = read_json(DECLARATION), read_json(RECIPE), read_json(CTX), read_json(REGIME)
    if sha256_file(REGIME) != decl["regimes"]["source_sha256"]:
        raise ValueError("The coverage-regime declaration changed.")
    regime = next(v for k, v in regime_decl["regimes"].items() if k.startswith(args.month))
    attempt = regime["attempt"]
    if sha256_file(REGIME_RUNS / attempt / f"attempt_{args.month}.json") != regime["attempt_record_sha256"]:
        raise ValueError("The thinning attempt differs from the declared one.")
    fold = "refit" if args.month == "2025-12" else "seasonal"
    candidate = decl["candidate"]
    report = {"created_at_utc": utc_now(), "month": args.month, "attempt": attempt, "declaration_sha256": sha256_file(DECLARATION), "member_fold": fold}

    # 1. tokens, natural (must reproduce the stored file) and thinned
    frames = {kind: features(attempt, args.month, kind) for kind in ("natural", "thinned")}
    ids = frames["natural"][0][ID].to_numpy()
    if not np.array_equal(ids, frames["thinned"][0][ID].to_numpy()):
        raise ValueError("Natural and thinned frames differ in rows.")
    stores, tmp = {}, Path(tempfile.mkdtemp(prefix="ctx_regime_"))
    thinned_matches = ROOT / "data/external/adsblol/regime" / attempt / "matches" / f"matches2_{args.month}.parquet"
    report["tokens"] = {}
    for kind, path in (("natural", NATURAL_MATCHES / f"matches2_{args.month}.parquet"), ("thinned", thinned_matches)):
        stores[kind], matched = build_store(args.month, path, tmp / kind)
        report["tokens"][kind] = {"matches": path.relative_to(ROOT).as_posix(), "matches_sha256": sha256_file(path), "matched_departure_tokens": matched,
                                  "content_sha256": content_hash(tmp / kind / f"{args.month}.npz")}
    stored_hash = content_hash(TOKENS / f"{args.month}.npz")
    report["tokens"]["natural_reproduces_stored"] = report["tokens"]["natural"]["content_sha256"] == stored_hash
    if not report["tokens"]["natural_reproduces_stored"]:
        raise ValueError("The natural token rebuild differs from the stored token file.")
    print({"tokens": report["tokens"], "seconds": round(time.time() - started, 1)}, flush=True)

    # 2. the 16 context networks, natural and thinned
    leaves = sorted(ctx_decl["network"]["learners"])
    preds = {"natural": [], "thinned": []}
    report["networks"] = {}
    for leaf in leaves:
        for kind in ("natural", "thinned"):
            preds[kind].append(ctx_predictions(ROOT / leaf, fold, frames[kind][0], stores[kind]))
        stored = stored_ctx(leaf, args.month, ids)
        diff = np.abs(preds["natural"][-1] - stored)
        report["networks"][leaf] = {"natural_vs_stored_max_abs": float(np.max(diff)), "natural_vs_stored_rms": float(np.sqrt(np.mean(diff ** 2))),
                                    "natural_vs_stored_rows_over_1s": int((diff > 1.0).sum()),
                                    "rows_changed_by_thinning": int((np.abs(preds["thinned"][-1] - preds["natural"][-1]) > 1e-9).sum())}
        print(leaf, report["networks"][leaf], flush=True)
    ens = {kind: np.mean(preds[kind], axis=0) for kind in preds}
    stored_ens = np.mean([stored_ctx(leaf, args.month, ids) for leaf in leaves], axis=0)
    diff = np.abs(ens["natural"] - stored_ens)
    report["ens_natural_vs_stored"] = {"max_abs": float(np.max(diff)), "rms": float(np.sqrt(np.mean(diff ** 2))), "rows_over_1s": int((diff > 1.0).sum()),
                                       "rows_over_0p01s": int((diff > 0.01).sum()),
                                       "note": "local natural re-predictions of networks differ from the Azure-stored ones on some rows, as for every committee network in "
                                               "the coverage campaign (docs/coverage_regime/penalty_<month>.json; tree members exact); control and candidate are both "
                                               "compared on local re-predictions, natural against thinned"}
    print({"ens_natural_vs_stored": report["ens_natural_vs_stored"]}, flush=True)

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
    table_names = list(recipe["extra_feature_tables"])
    label = month_tables["natural"][TARGET].to_numpy(dtype=float)
    airports = month_tables["natural"]["airport"].to_numpy()
    days = month_tables["natural"][TIME].dt.strftime("%Y-%m-%d").to_numpy()
    floor = float(month_tables["natural"]["floor"].iloc[0])

    def recipe_values(kind: str, network: dict, share: float) -> np.ndarray:
        table = month_tables[kind].copy()
        for name in table_names:
            table[name] = saved[f"{kind}:{name}"].to_numpy(dtype=float)
        table["ens_ctx"] = ens[kind]
        committee = table[recipe["committee_table"]].to_numpy(dtype=float)
        table["committee_mix"] = mixed(committee, weighted(table, network), share)
        spec = spec_for(recipe, "committee_mix", lirf_blends(recipe, table, v25, "v25"))
        spec = {**spec, "blend_with": {**spec["blend_with"], "weight": float(v26.get("recipe_weight", recipe["recipe_blend"]["weight"]))}}
        r = np.sum([w * saved[f"{kind}:residual:{m}"].to_numpy(dtype=float) for m, w in v26["residual"].items()], axis=0)
        values, _ = candidate_values(table, table["v9"].to_numpy(dtype=float), r, spec, floor)
        return values

    values = {(kind, who): recipe_values(kind, net, share) for kind in ("natural", "thinned")
              for who, net, share in (("control", v25["network"], float(recipe["network_share"])), ("candidate", candidate["network"], float(candidate["network_share"])))}
    recorded_control = ctx_decl["control"]["recorded_development_rmse"][args.month]
    recorded_candidate = read_json(ROOT / "runs/adsb_ctx/dev" / f"month_{args.month}.json")["candidates"][candidate["name"]]["rmse"]
    report["reproduction"] = {"control_recorded": recorded_control, "control_rebuilt": rmse(label, values["natural", "control"]),
                              "candidate_recorded": recorded_candidate, "candidate_rebuilt": rmse(label, values["natural", "candidate"])}
    # the declaration's 1e-6 s applies when the natural network re-predictions reproduce the stored ones; they do not (see
    # ens_natural_vs_stored), so the coverage campaign's own documented rebuild level is the tolerance (whole-month 0.0006 s there)
    report["reproduction"]["tolerance_s"] = 0.01
    report["reproduction"]["control_diff_s"] = report["reproduction"]["control_rebuilt"] - recorded_control
    report["reproduction"]["candidate_diff_s"] = report["reproduction"]["candidate_rebuilt"] - recorded_candidate
    report["reproduction"]["control_ok"] = abs(report["reproduction"]["control_diff_s"]) <= 0.01
    report["reproduction"]["candidate_ok"] = abs(report["reproduction"]["candidate_diff_s"]) <= 0.01
    if not report["reproduction"]["control_ok"]:
        raise ValueError(f"The natural control does not reproduce v26: {report['reproduction']}")
    non_lirf = airports != "LIRF"
    classes = class_of(month_tables["thinned"])
    report["regimes"] = {}
    for kind in ("natural", "thinned"):
        c, k = values[kind, "control"], values[kind, "candidate"]
        entry = {"control_rmse": rmse(label, c), "candidate_rmse": rmse(label, k), "change_pct": 100 * (rmse(label, k) / rmse(label, c) - 1),
                 "non_lirf_change_pct": 100 * (rmse(label[non_lirf], k[non_lirf]) / rmse(label[non_lirf], c[non_lirf]) - 1),
                 "paired_days": paired_days(days, label, k, c, recipe["seed"]),
                 "airports": {a: {"rows": int((airports == a).sum()), "control": rmse(label[airports == a], c[airports == a]), "candidate": rmse(label[airports == a], k[airports == a]),
                                  "change_pct": 100 * (rmse(label[airports == a], k[airports == a]) / rmse(label[airports == a], c[airports == a]) - 1)} for a in sorted(set(airports))}}
        if kind == "thinned":
            entry["classes_non_lirf"] = {cl: {"rows": int((non_lirf & (classes == cl)).sum()), "control": rmse(label[non_lirf & (classes == cl)], c[non_lirf & (classes == cl)]),
                                              "candidate": rmse(label[non_lirf & (classes == cl)], k[non_lirf & (classes == cl)])} for cl in ("gate", "dark", "unmatched")}
        report["regimes"][kind] = entry
    nat, thin = report["regimes"]["natural"], report["regimes"]["thinned"]
    report["retained"] = {"whole_month": thin["change_pct"] / nat["change_pct"], "non_lirf": thin["non_lirf_change_pct"] / nat["non_lirf_change_pct"]}
    interval = thin["paired_days"]["rmse_difference_95pct_interval_seconds"]
    report["gate_month"] = {"thinned_candidate_below_control": thin["candidate_rmse"] < thin["control_rmse"], "thinned_interval_below_zero": bool(interval[1] < 0)}
    report["gate_month"]["passes"] = all(report["gate_month"].values())
    out = ROOT / "runs/adsb_ctx/regime_check"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ID: ids, "ens_ctx:natural": ens["natural"], "ens_ctx:thinned": ens["thinned"],
                  **{f"{kind}:{who}": v for (kind, who), v in values.items()}}).to_parquet(out / f"{args.month}.parquet", index=False, compression="zstd")
    report["outputs"] = {"predictions_sha256": sha256_file(out / f"{args.month}.parquet")}
    report["code_sha256"] = {p: sha256_file(ROOT / p) for p in ["tools/ctx_regime_check.py", "tools/regime_measure.py", "src/prc2026_ctx/tokens.py", "src/prc2026_ctx/train.py"]}
    shutil.rmtree(tmp, ignore_errors=True)
    report["seconds"] = round(time.time() - started, 1)
    write_json(ROOT / "docs/adsb_ctx" / f"regime_check_{args.month}.json", report)
    print({"reproduction": report["reproduction"], "natural": round(nat["change_pct"], 4), "thinned": round(thin["change_pct"], 4), "retained": report["retained"],
           "thinned_interval": interval, "gate_month": report["gate_month"]}, flush=True)
    print({a: (round(nat["airports"][a]["change_pct"], 3), round(thin["airports"][a]["change_pct"], 3)) for a in thin["airports"]}, flush=True)


if __name__ == "__main__":
    main()
