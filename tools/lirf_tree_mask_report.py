"""Report of the masked LIRF tree design (configs/lirf_tree_mask_design_v1.json); written before any design prediction was read.

1. Sets: tools/run_lirf_dark.build_sets() (thinned_july, natural:2025-01 selectable for GO; natural:2025-07 descriptive). The row
   identities are rebuilt with the same filters and checked against the sets' labels.
2. Saved ensemble: ens_v22_lirf_q's own leaf list (runs/adsb_v22/ens_v22_lirf_q/run.json, multiplicities kept, nan-mean). Natural
   values = each leaf's seasonal predictions.parquet; thinned values = each leaf re-predicted on the a13 thinned July frame (LIRF rows)
   with prc2026_december.scoring.learner_prediction (cached in runs/lirf_tree_mask/thinned/). Check before any candidate: the saved
   natural ensemble equals runs/adsb_v22/ens_v22_lirf_q/seasonal/predictions.parquet and the saved thinned ensemble equals the measure
   table's 'thinned:ens_v22_lirf_q' (both within 0.01 s RMS on the set rows).
3. Candidate / control ensembles: the leaf list with the declared learners replaced by their runs; tree' = set tree + (ensemble - saved).
4. Values: v54's dark arithmetic (run_lirf_dark.values at w 0.70, s 0.75, gate u600); metric run_lirf_dark.compare; GO rule as declared.
  --threads N (local predictions; default 8)
  --check-saved  step 2's checks with the saved learners only (no design run is read), then exit
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
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID  # noqa: E402
from prc2026_december.scoring import VocabularyCache, learner_prediction  # noqa: E402
import run_lirf_dark as rld  # noqa: E402
import run_lirf_member as rlm  # noqa: E402

DECL = "configs/lirf_tree_mask_design_v1.json"
ENS = "runs/adsb_v22/ens_v22_lirf_q"
DESIGN = "runs/lirf_tree_mask/design"
THINNED = ROOT / "runs/lirf_tree_mask/thinned"
W, SHARE, T = 0.70, 0.75, 600.0


def set_ids() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """(ids, labels) per set, built with run_lirf_dark.build_sets' own filters and order."""
    rows = pd.read_parquet(ROOT / rlm.ROWS, columns=[ID, "month", "airport", "label", "lirf_m", "lirf_u"])
    rows = rows[rows["airport"].eq("LIRF")]
    out = {}
    for month in rlm.NATURAL:
        d = rows[rows["month"].eq(month) & rows["lirf_u"].astype(bool)]
        out[f"natural:{month}"] = (d[ID].to_numpy(), d["label"].to_numpy(float))
    measure = pd.read_parquet(ROOT / rlm.MEASURE, columns=[ID, "class_thinned"])
    nets = pd.read_parquet(ROOT / rlm.NETS, columns=[ID])
    july = rows[rows["month"].eq("2025-07") & (rows["lirf_m"] | rows["lirf_u"]).astype(bool)]
    t = july.merge(measure, on=ID, how="inner", validate="one_to_one").merge(nets, on=ID, how="inner", validate="one_to_one")
    t = t[t["class_thinned"].eq("unmatched")]
    out["thinned_july"] = (t[ID].to_numpy(), t["label"].to_numpy(float))
    m = rows[rows["month"].eq("2025-07") & rows["lirf_m"].astype(bool)]
    out["matched:2025-07"] = (m[ID].to_numpy(), m["label"].to_numpy(float))
    return out


def natural(leaf: str, ids: np.ndarray) -> np.ndarray:
    return pd.read_parquet(ROOT / leaf / "seasonal" / "predictions.parquet").set_index(ID)["adsb_full"].reindex(ids).to_numpy(dtype=float)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--check-saved", action="store_true")
    a = p.parse_args()
    decl = read_json(ROOT / DECL)
    if not decl.get("declared_at_utc"):
        raise ValueError("The declaration is a draft.")
    sets, net_checks, inputs = rld.build_sets()
    ids = set_ids()
    for k in ("natural:2025-01", "natural:2025-07", "thinned_july"):
        if not np.allclose(ids[k][1], sets[k]["labels"]):
            raise ValueError(f"{k}: rebuilt identities do not match the set.")
    leaves = read_json(ROOT / ENS / "run.json")["learners"]
    original = {k: v["original"] for k, v in decl["learners"].items()}
    by_path = {v: k for k, v in original.items()}
    if set(leaves) != set(original.values()):
        raise ValueError("The ensemble's leaves are not the declared learners.")
    run_dirs = {} if a.check_saved else {r: f"{DESIGN}/{r}" for r in decl["runs"]}
    missing = [r for r, d in run_dirs.items() if not (ROOT / d / "seasonal" / "run.json").exists()]
    if missing:
        raise ValueError(f"Runs missing: {missing}")

    # natural values on every set's ids (seasonal fold predictions)
    nat_ids = np.unique(np.concatenate([ids[k][0] for k in ("natural:2025-01", "natural:2025-07", "matched:2025-07")]))
    nat = {name: pd.Series(natural(path, nat_ids), index=nat_ids) for name, path in {**original, **run_dirs}.items()}
    # thinned values (LIRF rows of the a13 frame), cached
    import regime_measure as rm
    tj = ids["thinned_july"][0]
    THINNED.mkdir(parents=True, exist_ok=True)
    thin = {}
    frame = None
    vocab = None
    for name, path in {**original, **run_dirs}.items():
        cache = THINNED / f"{name}.parquet"
        if not cache.exists():
            if frame is None:
                full, _ = rm.features("a13_jul", "2025-07", "thinned")
                frame = full[full[ID].isin(tj)].reset_index(drop=True)
                vocab = VocabularyCache(ROOT)
            values, meta = learner_prediction(ROOT, Path(path), frame, vocab, a.threads, fold="seasonal")
            pd.DataFrame({ID: frame[ID].to_numpy(), "adsb_full": values}).to_parquet(cache, index=False)
            print("thinned", name, meta["finite_rows"], flush=True)
        thin[name] = pd.read_parquet(cache).set_index(ID)["adsb_full"]

    def ensemble(source: dict, replace: dict, idx: np.ndarray) -> np.ndarray:
        cols = [source[replace.get(by_path[leaf], by_path[leaf])].reindex(idx).to_numpy(dtype=float) for leaf in leaves]
        with np.errstate(invalid="ignore"):
            return np.nanmean(np.column_stack(cols), axis=1)

    # checks before any candidate
    checks = {}
    stored = pd.read_parquet(ROOT / ENS / "seasonal" / "predictions.parquet").set_index(ID)["adsb_full"]
    for k in ("natural:2025-01", "natural:2025-07"):
        d = ensemble(nat, {}, ids[k][0]) - stored.reindex(ids[k][0]).to_numpy(dtype=float)
        checks[k] = {"saved_natural_vs_stored_ens_rms_s": float(np.sqrt(np.nanmean(d ** 2)))}
    measure = pd.read_parquet(ROOT / rlm.MEASURE, columns=[ID, "thinned:ens_v22_lirf_q"]).set_index(ID)["thinned:ens_v22_lirf_q"]
    d = ensemble(thin, {}, tj) - measure.reindex(tj).to_numpy(dtype=float)
    checks["thinned_july"] = {"saved_thinned_vs_measure_rms_s": float(np.sqrt(np.nanmean(d ** 2))), "set_tree_vs_measure_max_abs_s": float(np.max(np.abs(sets["thinned_july"]["tree"] - measure.reindex(tj).to_numpy(dtype=float))))}
    bad = [k for k, v in checks.items() if any(x > 0.01 for x in v.values())]
    print("checks:", checks, flush=True)
    if bad:
        raise ValueError(f"Saved ensemble not reproduced on {bad}.")
    if a.check_saved:
        print("saved-learner checks pass; no design run read.")
        return

    def reading(replace: dict) -> dict:
        res = {}
        for k, src in (("thinned_july", thin), ("natural:2025-01", nat), ("natural:2025-07", nat)):
            s = sets[k]
            idx = ids[k][0]
            delta = ensemble(src, replace, idx) - ensemble(src, {}, idx)
            if not np.isfinite(delta).all():
                raise ValueError(f"{k}: non-finite tree delta.")
            s2 = {**s, "tree": s["tree"] + delta}
            base, _ = rld.values(s, W, SHARE, T)
            new, fire = rld.values(s2, W, SHARE, T)
            out = rld.compare(s, base, new, fire)
            keep = s["labels"] < rld.CAP
            out["tree_rmse_primary"] = {"control": float(np.sqrt(np.mean((s["tree"] - s["labels"])[keep] ** 2))),
                                        "candidate": float(np.sqrt(np.mean((s2["tree"] - s["labels"])[keep] ** 2)))}
            out["tree_delta"] = {"mean_s": float(delta.mean()), "rms_s": float(np.sqrt(np.mean(delta ** 2)))}
            if k != "natural:2025-07":
                best = None
                for share in np.round(np.arange(0.40, 1.0001, 0.025), 3):
                    v, _ = rld.values(s2, W, float(share), T)
                    sse = float(((v - s["labels"])[keep] ** 2).sum())
                    if best is None or sse < best[0]:
                        best = (sse, float(share))
                e0 = float(((base - s["labels"])[keep] ** 2).sum())
                out["least_squares_network_share"] = {"share": best[1], "relative_change": (best[0] - e0) / e0}
            res[k] = out
        mi, ml = ids["matched:2025-07"]
        keep = ml < rld.CAP
        res["matched:2025-07_standalone"] = {"rows": int(keep.sum()),
                                             "saved_rmse": float(np.sqrt(np.mean((ensemble(nat, {}, mi) - ml)[keep] ** 2))),
                                             "candidate_rmse": float(np.sqrt(np.mean((ensemble(nat, replace, mi) - ml)[keep] ** 2)))}
        return res

    control = reading(decl["control_reading"]["replace"])
    ctrl_tj = control["thinned_july"]["primary"]["relative_change"]
    results, go = {}, {}
    for name, cand in decl["candidates"].items():
        r = reading(cand["replace"])
        results[name] = r
        tjp, jan = r["thinned_july"]["primary"], r["natural:2025-01"]["primary"]
        go[name] = bool(tjp["relative_change"] <= -0.005 and tjp["interval_95"][1] < 0 and jan["relative_change"] <= 0 and tjp["relative_change"] < -2 * abs(ctrl_tj))
        print(f"{name:12s}", "GO" if go[name] else "- ", {k: (round(100 * v["primary"]["relative_change"], 3), [round(100 * x, 3) for x in v["primary"]["interval_95"]],
                                                            round(v["tree_rmse_primary"]["control"], 2), round(v["tree_rmse_primary"]["candidate"], 2))
                                                        for k, v in r.items() if "primary" in v},
              "matched Jul standalone", {kk: round(vv, 2) for kk, vv in r["matched:2025-07_standalone"].items() if kk != "rows"}, flush=True)
    print("control", {k: (round(100 * v["primary"]["relative_change"], 3), [round(100 * x, 3) for x in v["primary"]["interval_95"]]) for k, v in control.items() if "primary" in v})
    runs = {r: {kk: read_json(ROOT / d / "seasonal" / "run.json").get(kk) for kk in ("best_iteration", "masked_copies", "seconds", "score_rmse_lirf_unmatched_by_month")}
            for r, d in run_dirs.items()}
    report = {"created_at_utc": utc_now(), "declaration": {DECL: {"sha256": sha256_file(ROOT / DECL), "object_hash": object_hash(decl)}},
              "code": {q: sha256_file(ROOT / q) for q in ["tools/lirf_tree_mask_report.py", "tools/run_lirf_tree_mask.py", "tools/run_lirf_dark.py", "src/prc2026_final/lirf_gate.py"]},
              "inputs": inputs, "lirf_net_checks": net_checks, "reproduction_checks": checks, "arithmetic": {"member_weight": W, "network_share": SHARE, "gate_T": T},
              "runs": runs, "run_sha256": {r: {f: sha256_file(ROOT / d / "seasonal" / f) for f in ("run.json", "predictions.parquet")} for r, d in run_dirs.items()},
              "thinned_cache_sha256": {p.name: sha256_file(p) for p in sorted(THINNED.glob("*.parquet"))},
              "control_reading": control, "results": results, "go": go, "any_go": any(go.values()), "go_rule": decl["go_rule"]}
    path = ROOT / decl["report"]
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, report)
    print("GO:", {k: v for k, v in go.items()}, "->", path)


if __name__ == "__main__":
    main()
