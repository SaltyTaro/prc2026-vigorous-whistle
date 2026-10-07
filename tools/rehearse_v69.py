"""Rehearsal of the v69 chain (docs/final_file/v69_chain.md): rerun every step that built the uploaded v68 and v69 into
runs/final_rehearsal_v69/<full_chain | from_shipped_step5>/ and require their bytes exactly.

  A. v66's ten steps: tools/export_ctx_ownpoints.chain, imported unchanged, with the globals tools/export_ownpoints_weight_override.export sets
     (CHAIN = runs/ctx_ownpoints_weight/chain, DECLARATION = configs/ctx_ownpoints_weight_v1.json), candidate own_all_w100 -> OUT/v66/
     (must give v66 b4ea727e...; every intermediate compared with the shipped dry run). Steps 1-5 of this run feed B and C.
     --step5 shipped skips A and reads the shipped step-1..5 outputs (runs/ctx_ownpoints_weight/rank/dry_v66/steps) instead.
  B. v68: step 6 replaced by tools/run_opdi_stack_decl.step_stack_opdi(..., ['LSZH']) (configs/opdi_stack_lszh_v1.json; seeds 20261003, 11, 12,
     weight 0.5), then run_stack_resid_v3.chain_from_stack (steps 7-9) and run_own_motion_v2.dry_run (step 10) -> OUT/v68/
     (must give v68 d41d8705...).
  C. v69: step 6 as built by tools/run_stack_weight.py --rank-fit / --rank-dry (configs/stack_weight_v1.json, selected w65_bag10): the
     same three learners (v2 at LIRF, OPDI at LSZH OPDI rows, v3 elsewhere) fitted one model per seed for the ten seeds 20261003, 11-19, their
     mean at weight 0.65, then steps 7-10 -> OUT/v69/ (must give v69 192c4a30...).
Nothing is written under runs/submissions; nothing is uploaded; no official score enters.

  python tools/rehearse_v69.py [--step5 rehearse|shipped]
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID  # noqa: E402

BASE = ROOT / "runs/final_rehearsal_v69"
OUT = BASE / "full_chain"   # set in main: full_chain (--step5 rehearse) or from_shipped_step5 (--step5 shipped)
REPORT = ROOT / "docs/final_file/v69_rehearsal.json"
CANDIDATE = "own_all_w100"
EXPECTED = {
    66: ("runs/submissions/vigorous-whistle_v66.parquet", "b4ea727e1644b6add104f44832898bfecfd3d5607f7d6a9ce0471efe581322e4"),
    68: ("runs/submissions/vigorous-whistle_v68.parquet", "d41d8705e75a71e865fe21a8d6ad2f62044a5ed07f125fd208067914b2b10eab"),
    69: ("runs/submissions/vigorous-whistle_v69.parquet", "192c4a30c76ae1967a4d7594eeca4704554bdc7535f337f5022ef77eaea17665"),
}
SHIPPED = {
    66: ROOT / "runs/ctx_ownpoints_weight/rank/dry_v66",
    68: ROOT / "runs/opdi_stack_lszh/rank/dry_v68",
    69: ROOT / "runs/stack_weight/rank/dry_v69",
}
SHIPPED_PREDS = ROOT / "runs/stack_weight/rank/preds.parquet"
SEEDS3 = [20261003, 11, 12]
INPUTS = [
    "data/raw/ranking.parquet", "data/raw/submitting.parquet",
    "runs/lirf_combo/export_dry/vigorous-whistle_v55.parquet",
    "runs/ctx_ownpoints_weight/chain/own_all_w100/ctl.parquet", "runs/ctx_ownpoints_weight/chain/own_all_w100/rows_all.parquet",
    "runs/ctx_ownpoints_weight/chain/own_all_w100/dmp.parquet",
    "runs/stack_resid/v3_features/dev.parquet", "runs/stack_resid/v3_features/rank.parquet",
    "runs/opdi_stack/features_dev.parquet", "runs/opdi_stack/features_rank.parquet",
    "runs/rematch_nonm/key_map.parquet", "runs/intruder_identity/ids.parquet",
    "runs/dig/position/ids_ranking.parquet", "runs/dig/position/park_ranking.parquet",
    "data/external/adsblol/matches_v11/stands.parquet",
    "data/external/eurocontrol_aiu/All_Pre-Departure_Delay.parquet",
    "configs/ctx_ownpoints_weight_v1.json", "configs/ctx_ownpoints_dev_v1.json", "configs/ctx_ownpoints_network_v1.json",
    "configs/adsb_ctx3_combo_dev_v1.json", "configs/adsb_ctx3_mask_dev_v1.json",
    "configs/rematch_nonm_v1.json", "configs/daily_totals_v1.json", "configs/scout_extreme_shift_v1.json", "configs/scout_var_alloc_v1.json",
    "configs/stack_resid_v2.json", "configs/stack_resid_v3.json", "configs/opdi_stack_v1.json", "configs/opdi_stack_lszh_v1.json",
    "configs/stack_weight_v1.json", "configs/intruder_bound_v1.json", "configs/intruder_identity_v1.json",
    "configs/own_motion_v1.json", "configs/own_motion_v2.json",
    "docs/ctx_ownpoints_weight/override_v66.json", "docs/opdi_stack_lszh/results_v1.json", "docs/stack_weight/results_v1.json",
]
CODE = ["tools/rehearse_v69.py", "tools/export_ownpoints_weight_override.py", "tools/run_ownpoints_weight.py", "tools/export_ctx_ownpoints.py",
        "tools/run_ctx_ownpoints_chain.py", "tools/run_network_member.py", "tools/export_network_member.py", "tools/export_disagreement_gate.py",
        "tools/run_rematch_nonm.py", "tools/export_daily_totals.py", "tools/scout_extreme_shift.py", "tools/scout_var_alloc.py",
        "tools/run_stack_resid.py", "tools/run_stack_resid_v3.py", "tools/run_opdi_stack.py", "tools/run_opdi_stack_decl.py",
        "tools/run_opdi_stack_lszh.py", "tools/run_stack_weight.py", "tools/run_intruder_bound.py", "tools/run_intruder_identity.py",
        "tools/run_own_motion.py", "tools/run_own_motion_v2.py", "src/prc2026_submission/contract.py"]


class PeakRSS:
    """Samples this process's resident memory every 2 s (psutil when available)."""

    def __init__(self) -> None:
        self.peak, self._stop = 0, threading.Event()
        try:
            import psutil
            self._proc = psutil.Process()
        except Exception:  # noqa: BLE001
            self._proc = None

    def __enter__(self):
        if self._proc is not None:
            threading.Thread(target=self._run, daemon=True).start()
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.peak = max(self.peak, self._proc.memory_info().rss)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(2)

    def __exit__(self, *exc) -> None:
        self._stop.set()


def versions() -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ["numpy", "pandas", "pyarrow", "lightgbm", "sklearn"]:
        try:
            out[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            out[mod] = None
    return out


def step_hashes(base: Path) -> dict:
    return {p.relative_to(base).as_posix(): sha256_file(p) for p in sorted((base / "steps").rglob("vigorous-whistle_v*.parquet"))}


def compare_steps(mine: dict, version: int) -> dict:
    ref_dir = SHIPPED[version]
    if not ref_dir.exists():
        return {"shipped_dir_missing": ref_dir.relative_to(ROOT).as_posix()}
    ref = step_hashes(ref_dir)
    keys = sorted(set(ref) | set(mine))
    out = {k: {"shipped": ref.get(k), "rehearsal": mine.get(k), "equal": ref.get(k) == mine.get(k)} for k in keys if k in ref}
    return {"steps": out, "all_equal": all(v["equal"] for v in out.values())}


def final_check(path: Path, version: int) -> dict:
    sha = sha256_file(path)
    up, want = EXPECTED[version]
    rec = {"file": path.relative_to(ROOT).as_posix(), "sha256": sha, "expected": want, "byte_identical": sha == want}
    if (ROOT / up).exists():
        rec["uploaded_file_sha256"] = sha256_file(ROOT / up)
    return rec


def stage_a(t: dict) -> Path:
    import export_ctx_ownpoints as E
    import run_ownpoints_weight as W
    E.CHAIN, E.DECLARATION = W.OUT / "chain", W.DECLARATION
    t0 = time.time()
    result = E.chain(CANDIDATE, OUT / "v66", 66, False)
    t["v66"] = {"wall_seconds": round(time.time() - t0, 1), "final": final_check(result, 66),
                "vs_shipped_dry_run": compare_steps(step_hashes(OUT / "v66"), 66)}
    print("v66", json.dumps(t["v66"]["final"]), "steps equal", t["v66"]["vs_shipped_dry_run"].get("all_equal"), flush=True)
    return OUT / "v66" / "steps"


def stage_b(sh: Path, t: dict) -> None:
    import run_opdi_stack_decl as D
    import run_own_motion_v2 as O2
    import run_stack_resid as S
    import run_stack_resid_v3 as V3
    import export_ctx_ownpoints as E
    import run_ownpoints_weight as W
    E.CHAIN, E.DECLARATION = W.OUT / "chain", W.DECLARATION
    D.SH = sh
    S.SEEDS, S.WEIGHT = list(SEEDS3), 0.5
    t0 = time.time()
    out = OUT / "v68"
    st, _, _, _, use = D.step_stack_opdi(sh / "rematch/base/vigorous-whistle_v55m_rematch.parquet", sh / "Y/vigorous-whistle_v66.parquet",
                                         sh / "Y/rows.parquet", sh / "model_part/member_delta.parquet", out / "steps/stack", 68, ["LSZH"])
    v63like = V3.chain_from_stack(st, 68, out / "steps")
    O2.dry_run(v63like, sha256_file(v63like), 68, out / "steps/om2")
    final = out / "vigorous-whistle_v68.parquet"
    shutil.copyfile(out / "steps/om2/vigorous-whistle_v68.parquet", final)
    t["v68"] = {"wall_seconds": round(time.time() - t0, 1), "opdi_rows_used": int(use.sum()), "final": final_check(final, 68),
                "vs_shipped_dry_run": compare_steps(step_hashes(out), 68)}
    print("v68", json.dumps(t["v68"]["final"]), "steps equal", t["v68"]["vs_shipped_dry_run"].get("all_equal"), flush=True)


def stage_c(sh: Path, t: dict) -> None:
    import run_opdi_stack_decl as D
    import run_own_motion_v2 as O2
    import run_stack_resid as S
    import run_stack_resid_v3 as V3
    import run_stack_weight as SW
    from prc2026_submission.contract import export_submission
    res = read_json(ROOT / "docs/stack_weight/results_v1.json")
    sel = res["selected"]
    cand = res["candidates"][sel]
    w, seeds = float(cand["weight"]), list(cand["seeds"])
    k = SW.k_of(seeds)
    D.SH = sh
    t0 = time.time()
    out = OUT / "v69"
    out.mkdir(parents=True, exist_ok=True)
    # tools/run_stack_weight.rank_fit's loop: one model per seed, guarded ranking predictions of the three learners
    dev, res_t, fx, mats, rk, fxr, h2, h3, ho, Por = SW.rank_inputs()
    dom = dev["dom"].to_numpy()
    P = pd.DataFrame({ID: rk[ID].to_numpy(), "airport": rk["airport"].to_numpy(), "new": rk["new"].to_numpy(dtype=float), "fx": fxr,
                      "h2": h2, "h3": h3, "ho": ho, "o_has": Por["o_has"].to_numpy()})
    trees = {}
    saved = list(S.SEEDS)
    try:
        for key, X, Xr in mats:
            Xa, Xra = X.to_numpy(np.float32), Xr.to_numpy(np.float32)
            for i, seed in enumerate(SW.SEEDS10[:k]):
                S.SEEDS = [seed]
                models = S.fit_bag(Xa, res_t, ~fx, dom)
                P[f"{key}_{i}"] = models[0].predict(Xra, num_iteration=models[0].best_iteration)
                trees[f"{key} {seed}"] = int(models[0].best_iteration)
                print("rehearsal rank", key, seed, "trees", trees[f"{key} {seed}"], flush=True)
    finally:
        S.SEEDS = saved
    P.to_parquet(out / "preds.parquet", index=False)
    preds_cmp = {"file": (out / "preds.parquet").relative_to(ROOT).as_posix(), "sha256": sha256_file(out / "preds.parquet")}
    if SHIPPED_PREDS.exists():
        ref = pd.read_parquet(SHIPPED_PREDS)
        cols = [c for c in ref.columns if c in P.columns]
        preds_cmp.update({"shipped_sha256": sha256_file(SHIPPED_PREDS), "columns_compared": len(cols),
                          "values_equal": bool(ref[ID].equals(P[ID]) and all(np.array_equal(ref[c].to_numpy(), P[c].to_numpy()) for c in cols)),
                          "max_abs_pred_diff": float(max(np.abs(ref[c].to_numpy(dtype=float) - P[c].to_numpy(dtype=float)).max()
                                                         for c in cols if c.startswith(("p2_", "p3_", "po_"))))})
    values, use, _ = SW.rank_value(P, w, k)
    st = out / "steps/stack/vigorous-whistle_v69.parquet"
    export_submission(pd.DataFrame({ID: P[ID].to_numpy(), "prediction": values}), ROOT / "data/raw/submitting.parquet", st,
                      {"dry_run": True, "step": f"stack_weight_v1 {sel}: stacking value at weight {cand['weight']} with {len(seeds)} seeds",
                       "official_scores_used": "none", "audit_used": False})
    v63like = V3.chain_from_stack(st, 69, out / "steps")
    O2.dry_run(v63like, sha256_file(v63like), 69, out / "steps/om2")
    final = out / "vigorous-whistle_v69.parquet"
    shutil.copyfile(out / "steps/om2/vigorous-whistle_v69.parquet", final)
    t["v69"] = {"wall_seconds": round(time.time() - t0, 1), "candidate": sel, "weight": w, "seeds": seeds, "trees": trees,
                "opdi_rows_used": int(use.sum()), "preds": preds_cmp, "final": final_check(final, 69),
                "vs_shipped_dry_run": compare_steps(step_hashes(out), 69)}
    print("v69", json.dumps(t["v69"]["final"]), "preds", json.dumps({k2: v for k2, v in preds_cmp.items() if k2 != "file"}),
          "steps equal", t["v69"]["vs_shipped_dry_run"].get("all_equal"), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step5", choices=["rehearse", "shipped"], default="rehearse")
    a = ap.parse_args()
    global OUT
    OUT = BASE / ("full_chain" if a.step5 == "rehearse" else "from_shipped_step5")
    if OUT.exists():
        raise ValueError(f"{OUT} exists; never overwrite a rehearsal.")
    OUT.mkdir(parents=True)
    inputs_before = {p: (sha256_file(ROOT / p) if (ROOT / p).exists() else None) for p in INPUTS}
    start, t0, t = utc_now(), time.time(), {}
    with PeakRSS() as mem:
        sh = stage_a(t) if a.step5 == "rehearse" else SHIPPED[66] / "steps"
        stage_b(sh, t)
        stage_c(sh, t)
    wall = time.time() - t0
    inputs_after = {p: (sha256_file(ROOT / p) if (ROOT / p).exists() else None) for p in INPUTS}
    report = {
        "created_at_utc": utc_now(), "started_at_utc": start, "wall_seconds": round(wall, 1),
        "peak_rss_gib_driver": round(mem.peak / 2 ** 30, 2) if mem.peak else None, "step5": a.step5,
        "byte_identical": {str(v): t[f"v{v}"]["final"]["byte_identical"] for v in (66, 68, 69) if f"v{v}" in t},
        "stages": t,
        "inputs": inputs_before, "inputs_unchanged": all(inputs_before[p] == inputs_after[p] for p in INPUTS),
        "code": {p: sha256_file(ROOT / p) for p in CODE},
        "environment": versions(),
        "official_scores_used": "none", "uploaded": False,
    }
    write_json(OUT / "rehearsal.json", report)
    print(json.dumps({"byte_identical": report["byte_identical"], "wall_seconds": report["wall_seconds"],
                      "inputs_unchanged": report["inputs_unchanged"]}, indent=1), flush=True)
    if not all(report["byte_identical"].values()):
        sys.exit(2)


if __name__ == "__main__":
    main()
