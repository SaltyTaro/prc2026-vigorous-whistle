"""Compare a final_predict.py run on a copy of the ranking file with the recorded v30 chain, stage by stage (the stage-3 dry run).

Acceptance (docs/final_file/runner.md):
- features equal the frozen timeline bundle's ranking rows (exact; the 4 cyclic NM-clock float64 columns within 2e-16);
- matches equal the frozen match tables row for row; the learner ADS-B table and v19 features equal the recorded ranking tables
  except where the unstable neighbour sort moves nb_* values (reported); tokens' content hash equals the frozen ranking split's;
- v7, v7 twelve-month, v9, committee and ens_v22_lirf_q within 1e-9 of the recorded values (v9 and the tables via the v30 detail);
- each CPU network seed within 0.01 s max abs of its saved refit12 predictions; each GPU (e32) seed within 0.5 s RMS / 2 s p99;
- residual members and the residual reported (rows within 1e-6 s and the maximum), the recipe columns against the v30 detail;
- the exported integers against the uploaded v30: differing rows and max |delta| reported (never overwritten, never uploaded).
Writes --output (JSON). Read-only on everything outside the run root.
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
from prc2026.validation import identity_hash  # noqa: E402

V30_DETAIL = "runs/adsb_ctx/export_v30/ranking_values.parquet"
V30_FILE = "runs/submissions/vigorous-whistle_v30.parquet"
V7_FROZEN = "runs/azure_arcw0_surface_refit/20260914T185207842456Z_refit_full_timeline_v6_100/ranking_predictions.parquet"
V7_TWELVE = "runs/adsb_v24/v7/refit/ranking_predictions.parquet"
V23_VALUES = "runs/adsb_v25/export_v23/ranking_values.parquet"
ADSB_TABLE = "runs/adsb_ctx2/inputs/adsb_features_arcw0/adsb_ranking.parquet"  # the table every context seed used (sha 1dcf5c7f...)


def diff(a: np.ndarray, b: np.ndarray, tol: float) -> dict:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    both = ~nan_a & ~nan_b
    delta = np.abs(a[both] - b[both])
    return {"rows": int(len(a)), "nan_pattern_equal": bool(np.array_equal(nan_a, nan_b)), "max_abs": float(delta.max()) if len(delta) else 0.0,
            "rms": float(np.sqrt(np.mean(delta ** 2))) if len(delta) else 0.0, "p99": float(np.quantile(delta, 0.99)) if len(delta) else 0.0,
            "rows_over_tolerance": int((delta > tol).sum()), "tolerance": tol,
            "passes": bool(np.array_equal(nan_a, nan_b) and (delta <= tol).all())}


def frame_diff(ours: pd.DataFrame, theirs: pd.DataFrame, ids: np.ndarray, tol: float = 0.0) -> dict:
    a = ours.set_index(ID).reindex(ids)
    b = theirs.set_index(ID).reindex(ids)
    out = {}
    for c in [c for c in a.columns if c in b.columns]:
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            d = diff(x.to_numpy(dtype=float), y.to_numpy(dtype=float), tol)
            if not d["passes"]:
                out[c] = {k: d[k] for k in ["nan_pattern_equal", "max_abs", "rows_over_tolerance"]}
        else:
            same = (x.astype("string").fillna("<NA>") == y.astype("string").fillna("<NA>"))
            if not bool(same.all()):
                out[c] = {"rows_differing": int((~same).sum())}
    return {"columns_compared": int(len([c for c in a.columns if c in b.columns])), "columns_differing": out, "passes": not out}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", type=int, default=None)
    parser.add_argument("--allow-subset", action="store_true", help="compare a run on a slice of the ranking file on its own departures (smoke tests; "
                        "context-dependent features legitimately differ near the slice's edges)")
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else ROOT / args.root
    meta = read_json(root / "input.json")
    detail = pd.read_parquet(ROOT / V30_DETAIL)
    ids = detail[ID].to_numpy()
    if args.allow_subset:
        run_ids = pd.read_parquet(root / "features/timeline.parquet", columns=[ID])[ID].to_numpy()
        if not set(run_ids) <= set(ids):
            raise ValueError("The run's departures are not ranking departures.")
        ids = run_ids
    stages = sorted(p.stem for p in (root / "stages").glob("*.json"))
    report = {"created_at_utc": utc_now(), "root": root.as_posix(), "input": {k: meta[k] for k in ["raw", "raw_sha256", "departures", "months"]},
              "stages_present": stages, "results": {}}
    res = report["results"]
    if not args.allow_subset and meta["departure_ids_sha256"] != identity_hash(ids):
        raise ValueError("The run's departures are not the ranking departures; the dry-run comparison needs a copy of the ranking file.")
    if "features" in stages:
        from prc2026_final.features import compare_frozen
        res["features"] = compare_frozen(ROOT, pd.read_parquet(root / "features/timeline.parquet"))
        res["features"].pop("differing_columns_detail", None)
    if "adsb" in stages:
        from prc2026_final.adsb import MATCH_DIRS
        m = {}
        for label, directory in MATCH_DIRS.items():
            for month in meta["months"]:
                ours = pd.read_parquet(root / "adsb" / label / f"matches2_{month}.parquet").sort_values(ID).reset_index(drop=True)
                theirs = pd.read_parquet(ROOT / directory / f"matches2_{month}.parquet")
                theirs = theirs.loc[theirs[ID].isin(ids)].sort_values(ID).reset_index(drop=True)
                same_ids = bool(np.array_equal(ours[ID].to_numpy(), theirs[ID].to_numpy()))
                m[f"{label}:{month}"] = {"rows": [int(len(ours)), int(len(theirs))], "same_ids": same_ids,
                                         **(frame_diff(ours, theirs, ours[ID].to_numpy(), 1e-9) if same_ids else {"passes": False})}
        res["matches"] = m
        res["learner_adsb_vs_context_seeds_table"] = frame_diff(pd.read_parquet(root / "adsb/learner_adsb.parquet"), pd.read_parquet(ROOT / ADSB_TABLE), ids, 0.0)
        v19 = pd.concat([pd.read_parquet(ROOT / f"data/external/adsblol/features_v19/features_v19_{mo}.parquet") for mo in meta["months"]], ignore_index=True)
        res["v19"] = frame_diff(pd.read_parquet(root / "adsb/v19.parquet"), v19, ids, 1e-9)
    if "tokens" in stages:
        from prc2026_final.tokens import frozen_ranking_content
        got = read_json(root / "stages/tokens.json")["content_sha256"]
        res["tokens"] = {"content_sha256": got, "frozen_ranking_content_sha256": frozen_ranking_content(), "passes": got == frozen_ranking_content()}
    if "v9" in stages:
        ours = pd.read_parquet(root / "v9/v9.parquet").set_index(ID).reindex(ids)
        v7 = pd.read_parquet(ROOT / V7_FROZEN).set_index(ID).reindex(ids)["prediction"].to_numpy(dtype=float)
        v7t = pd.read_parquet(ROOT / V7_TWELVE).set_index(ID).reindex(ids)["prediction"].to_numpy(dtype=float)
        v23 = pd.read_parquet(ROOT / V23_VALUES).set_index(ID).reindex(ids)
        res["v9"] = {"v7": diff(ours["v7"], v7, 1e-9), "v7_twelve": diff(ours["v7_twelve"], v7t, 1e-9), "v9_frozen": diff(ours["v9_frozen"], v23["v9_frozen"], 1e-9),
                     "v9": diff(ours["v9"], detail.set_index(ID).reindex(ids)["v9"], 1e-9)}
    if "leaves" in stages:
        cfg = read_json(ROOT / "configs/adsb_v23_v1.json")["leaf_learners"]
        leaf = {}
        for name, spec in cfg.items():
            ours = pd.read_parquet(root / "leaves/leaf" / (name.replace("/", "__") + ".parquet")).set_index(ID).reindex(ids)["adsb_full"]
            theirs = pd.read_parquet(ROOT / spec["refit_output"] / "ranking_predictions.parquet").set_index(ID).reindex(ids)["adsb_full"]
            leaf[name] = diff(ours, theirs, 1e-6)
        tables = pd.read_parquet(root / "leaves/tables.parquet").set_index(ID).reindex(ids)
        d = detail.set_index(ID).reindex(ids)
        res["leaves"] = {"leaves_passing": int(sum(v["passes"] for v in leaf.values())), "leaves": leaf,
                         "committee": diff(tables["ens_v20_bag_q_all"], d["committee"], 1e-9), "ens_v22_lirf_q": diff(tables["ens_v22_lirf_q"], d["ens_v22_lirf_q"], 1e-9)}
    if "networks" in stages:
        from prc2026_final.networks import ensemble_learners
        learners = ensemble_learners()
        ens = pd.read_parquet(root / "networks/ensembles.parquet").set_index(ID).reindex(ids)
        d = detail.set_index(ID).reindex(ids)
        per = {}
        for ensemble in [c for c in ens.columns]:
            for learner in learners[ensemble]:
                path = root / "networks" / ensemble / (learner.replace("/", "__") + ".parquet")
                ours = pd.read_parquet(path).set_index(ID).reindex(ids)["adsb_full"]
                theirs = pd.read_parquet(ROOT / learner / "refit12/ranking_predictions.parquet").set_index(ID).reindex(ids)["adsb_full"]
                gpu = read_json(ROOT / learner / "refit12/run.json")["inputs"].get("device") == "cuda"
                r = diff(ours, theirs, 0.01 if not gpu else np.inf)
                r["trained_on"] = "gpu" if gpu else "cpu"
                if gpu:
                    r["passes"] = bool(r["nan_pattern_equal"] and r["rms"] <= 0.5 and r["p99"] <= 2.0)
                    r["tolerance"], r["rows_over_tolerance"] = {"rms": 0.5, "p99": 2.0}, None
                per[learner] = r
            res.setdefault("ensembles", {})[ensemble] = diff(ens[ensemble], d[ensemble], 0.05) if ensemble in d.columns else None
        res["networks"] = {"learners_passing": int(sum(v["passes"] for v in per.values())), "learners": len(per), "per_learner": per}
    if "residual" in stages:
        ours = pd.read_parquet(root / "residual/residual.parquet").set_index(ID).reindex(ids)
        v23 = pd.read_parquet(ROOT / V23_VALUES).set_index(ID).reindex(ids)
        rnet = np.mean([pd.read_parquet(ROOT / l / "refit12/ranking_predictions.parquet").set_index(ID).reindex(ids)["residual"].to_numpy(dtype=float)
                        for l in read_json(ROOT / "configs/adsb_rnn_v1.json")["residual_networks"]["rnet"]["learners"]], axis=0)
        res["residual"] = {"in_scope_equal": bool(np.array_equal(ours["in_scope"].to_numpy(dtype=bool), detail.set_index(ID).reindex(ids)["scope"].eq("matched_no_lirf").to_numpy())),
                           "adj2": diff(ours["residual:adj2"], v23["residual:adj2"], 1e-6), "cat": diff(ours["residual:cat"], v23["residual:cat"], 1e-6),
                           "rnet_mean": diff(ours["network"], rnet, 0.01), "residual": diff(ours["residual"], detail.set_index(ID).reindex(ids)["residual"], 0.01),
                           "note": "adj2/cat: 1e-6 except rows whose neighbour columns moved (unstable sort); the rnet seeds are Linux-saved float32 networks "
                                   "reloaded here (~1e-4 s, the CPU-network tolerance 0.01 s applies)"}
    if "recipe" in stages:
        ours = pd.read_parquet(root / "recipe/detail.parquet")
        d = detail.set_index(ID).reindex(ids)
        o = ours.set_index(ID).reindex(ids)
        res["recipe"] = {"scope_equal": bool((o["scope"] == d["scope"]).all()), "committee_mix": diff(o["committee_mix"], d["committee_mix"], 0.5),
                         **{c: diff(o[c], d[c], 0.01) for c in o.columns if c.startswith("lirf_mix:") and c in d.columns},
                         "v30": diff(o["v30"], d["v30"], 0.5),
                         "note": "committee_mix and v30 carry the e32 GPU->CPU reload difference (0.75 x 0.5 of the e32 table), hence 0.5 s; lirf_mix uses CPU nets only"}
    if "export" in stages and args.version is not None:
        ours = pd.read_parquet(root / "export" / f"vigorous-whistle_v{args.version}.parquet")
        uploaded = pd.read_parquet(ROOT / V30_FILE)
        same_order = bool(np.array_equal(ours[ID].to_numpy(), uploaded[ID].to_numpy()))
        same_order = same_order or args.allow_subset
        delta = ours[TARGET].to_numpy().astype(float) - uploaded.set_index(ID).reindex(ours[ID])[TARGET].to_numpy().astype(float)
        res["export_vs_uploaded_v30"] = {"same_template_order": same_order, "rows_differing": int((delta != 0).sum()), "max_abs_delta_seconds": float(np.abs(delta).max()),
                                         "uploaded_sha256": sha256_file(ROOT / V30_FILE), "ours_sha256": sha256_file(root / "export" / f"vigorous-whistle_v{args.version}.parquet")}
    out = args.output if args.output.is_absolute() else ROOT / args.output
    write_json(out, report)
    print({k: (v.get("passes") if isinstance(v, dict) and "passes" in v else "see report") for k, v in res.items()})


if __name__ == "__main__":
    main()
