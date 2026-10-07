"""Screen val2026, step 1: per-row component tables of v69's chain for LFPG departures (2025 development rows and 2026 ranking rows).

Nothing is fitted, declared, built or uploaded; no leaderboard score is read. 2025 labels enter only the development table.

Components (both sides, LFPG rows only):
  v9, residual       the non-ADS-B path (runs/lomo_dev/rows.parquet on 2025; the final-file runner's recipe detail on 2026)
  committee          the tree committee (same sources)
  nets_old           v30's network table 0.25 e24 + 0.25 e24b + 0.5 e32 (share_gate.network_table; on 2025 the lomo 'nets' column)
  own                the own_all context-network member, seed mean of 16 (configs/ctx_ownpoints_dev_v1.json member tables)
  msk                the masked member m25_32, seed mean of 16 (configs/adsb_ctx3_mask_dev_v1.json member tables)
  v55m               the model part (2025: runs/pubrefit/chain/ship/dmp.parquet; 2026: stack/detail v55m + the own_all member's delta d52,
                     checked against the shipped model-part file M')
  C                  the chain after the published-data steps (2025: ship/cand.parquet 'new'; 2026: dry_v66 steps/Y file)
  new_ctl, p3, h3, fx  the stacker's base and its ten-seed v3 prediction (2025: runs/stack_weight/dev/preds_<m>.parquet on the ctl chain;
                     2026: runs/stack_weight/rank/preds.parquet)
  v69                2025: runs/stack_weight/dev/w65_bag10/proxy.parquet 'value'; 2026: the uploaded v69 file; v70 too on 2026
  v9_r, residual_r, committee_r, nets_old_r, v30, v30_r, repair  2026 only: the EOBT_1-repaired run (runs/final_lfpg_eobt/run) and
                     stack/detail's repair effect (v30_r - v30; v31 adds 0.5 x repair, LFPG July 2026 only)
  cls                runs/hunt census class: gate / dark / unm_seen (own_n >= 1) / unm_unseen

  .venv\\Scripts\\python.exe -X utf8 -B tools/val2026_components.py   -> runs/val2026/components_{dev,rank}.parquet + components.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026_final import share_gate  # noqa: E402

ID, TARGET = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
OUT = ROOT / "runs" / "val2026"
DEV_MONTHS = ["2025-01", "2025-07", "2025-11", "2025-12"]
OWN_DECL = "configs/ctx_ownpoints_dev_v1.json"
MASK_DECL = "configs/adsb_ctx3_mask_dev_v1.json"
LOMO = "runs/lomo_dev/rows.parquet"
SHIP = "runs/pubrefit/chain/ship"
DEV_PREDS = "runs/stack_weight/dev/preds_{m}.parquet"
DEV_PROXY = ("runs/stack_weight/dev/w65_bag10/proxy.parquet", "c088e409de153d81b006bcd26b5b2a28e0e306043a5172c2158d0ecc6e6f5cb8")
CENSUS_DEV, CENSUS_RANK = "runs/hunt/census_dev.parquet", "runs/hunt/census_rank.parquet"
RUNNER, REPAIRED = "runs/final_dryrun/run", "runs/final_lfpg_eobt/run"
STEPS = "runs/ctx_ownpoints_weight/rank/dry_v66/steps"
RANK_PREDS = ("runs/stack_weight/rank/preds.parquet", "8083ffcfdf861e278ac8693fb2320872cd95378f3f9ac861decf23c9a6163772")
V69 = ("runs/submissions/vigorous-whistle_v69.parquet", "192c4a30c76ae1967a4d7594eeca4704554bdc7535f337f5022ef77eaea17665")
V70 = ("runs/submissions/vigorous-whistle_v70.parquet", "f1d17747e0e384ccec520a760d42c8e55a3e88a1476daafacdf7fe1f9603099f")
M_PRIME = (f"{STEPS}/model_part/vigorous-whistle_v66.parquet", "73eef8daa3c48500db13497491bee2926498c06334f29743d403969e0ede470c")
C_FILE = (f"{STEPS}/Y/vigorous-whistle_v66.parquet", "12a50ea0011164f6ecdf2df1f227f930a9130b7f669eb523bb311d9e8771bad5")


def sha(rel: str) -> str:
    h = hashlib.sha256()
    with open(ROOT / rel, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def checked(pair: tuple[str, str], inputs: dict) -> str:
    got = sha(pair[0])
    if got != pair[1]:
        raise ValueError(f"{pair[0]}: SHA-256 {got} differs from {pair[1]}.")
    inputs[pair[0]] = got
    return pair[0]


def seed_mean(spec: dict, key: str, ids: np.ndarray, inputs: dict) -> np.ndarray:
    """network_member.seed_mean over the member's tables (a row missing in any seed is missing)."""
    tabs = []
    for rel in spec["tables"][key]:
        t = pd.read_parquet(ROOT / rel, columns=[ID, spec.get("column", "adsb_full")])
        tabs.append(t.set_index(ID).iloc[:, 0].reindex(ids).to_numpy(dtype=float))
        inputs[rel] = sha(rel)
    s = np.vstack(tabs)
    return np.where(np.isfinite(s).all(axis=0), s.mean(axis=0), np.nan)


def census_class(path: str, ids: np.ndarray) -> np.ndarray:
    c = pd.read_parquet(ROOT / path, columns=[ID, "k", "own_n"]).drop_duplicates(ID).set_index(ID).reindex(ids)
    k = c["k"].to_numpy(dtype=object)
    own = c["own_n"].fillna(0).to_numpy(dtype=float)
    return np.where(k == "unm", np.where(own >= 1, "unm_seen", "unm_unseen"), k).astype(str)


def p3_bag(P: pd.DataFrame) -> np.ndarray:
    """run_stack_weight.bag('p3', 10) with fixed and held rows zeroed (rank_value / dev_value arithmetic)."""
    q = np.mean([P[f"p3_{i}"].to_numpy(dtype=np.float64) for i in range(10)], axis=0)
    q[P["fx"].to_numpy(dtype=bool) | P["h3"].to_numpy(dtype=bool)] = 0.0
    return q


def dev_table(inputs: dict, checks: dict) -> pd.DataFrame:
    own_spec = json.loads((ROOT / OWN_DECL).read_text(encoding="utf-8"))["member"]
    msk_spec = json.loads((ROOT / MASK_DECL).read_text(encoding="utf-8"))["member"]
    inputs[OWN_DECL], inputs[MASK_DECL] = sha(OWN_DECL), sha(MASK_DECL)
    lo = pd.read_parquet(ROOT / LOMO, columns=[ID, "month", "day", "airport", "label", "matched", "v9", "residual", "committee", "nets"])
    inputs[LOMO] = sha(LOMO)
    lo = lo[lo["airport"].eq("LFPG") & lo["month"].isin(DEV_MONTHS)].reset_index(drop=True).rename(columns={"nets": "nets_old"})
    ids = lo[ID].to_numpy()
    lo["own"], lo["msk"] = np.nan, np.nan
    for m in DEV_MONTHS:
        sel = lo["month"].eq(m).to_numpy()
        lo.loc[sel, "own"] = seed_mean(own_spec, m, ids[sel], inputs)
        lo.loc[sel, "msk"] = seed_mean(msk_spec, m, ids[sel], inputs)
    dmp = pd.read_parquet(ROOT / f"{SHIP}/dmp.parquet", columns=[ID, "v52m", "v55m"]).set_index(ID).reindex(ids)
    cand = pd.read_parquet(ROOT / f"{SHIP}/cand.parquet", columns=[ID, "new"]).set_index(ID)["new"].reindex(ids)
    ctl = pd.read_parquet(ROOT / f"{SHIP}/ctl.parquet", columns=[ID, "new"]).set_index(ID)["new"].reindex(ids)
    for f in ["dmp", "cand", "ctl"]:
        inputs[f"{SHIP}/{f}.parquet"] = sha(f"{SHIP}/{f}.parquet")
    lo["v52m"], lo["v55m"], lo["C"], lo["new_ctl"] = dmp["v52m"].to_numpy(), dmp["v55m"].to_numpy(), cand.to_numpy(), ctl.to_numpy()
    P = pd.concat([pd.read_parquet(ROOT / DEV_PREDS.format(m=m)) for m in DEV_MONTHS], ignore_index=True)
    for m in DEV_MONTHS:
        inputs[DEV_PREDS.format(m=m)] = sha(DEV_PREDS.format(m=m))
    P = P.drop_duplicates(ID).set_index(ID).reindex(ids).reset_index()
    if P["p3_0"].isna().any():
        raise ValueError("The stored stacking predictions do not cover the LFPG development rows.")
    lo["p3"], lo["fx"], lo["h3"] = p3_bag(P), P["fx"].to_numpy(dtype=bool), P["h3"].to_numpy(dtype=bool)
    o_has = (P["o_has"].fillna(0.0).to_numpy() > 0) & ~P["ho"].to_numpy(dtype=bool)
    checks["dev_lfpg_rows_on_opdi_route"] = 0  # the OPDI learner is LSZH only (run_stack_weight.AIRPORTS)
    checks["dev_lfpg_rows_with_opdi_columns"] = int(o_has.sum())
    proxy = pd.read_parquet(ROOT / checked(DEV_PROXY, inputs), columns=[ID, "value", "cls"]).set_index(ID).reindex(ids)
    lo["v69"] = proxy["value"].to_numpy(dtype=float)
    # stack step on the dev chain (run_stack_weight.dev_value, LFPG = v3 route): s2 = C + (max(ctl + 0.65 p3, -12) - ctl)
    st = np.where(lo["fx"], lo["new_ctl"], np.maximum(lo["new_ctl"] + 0.65 * lo["p3"], -12.0))
    lo["s6"] = lo["C"] + (st - lo["new_ctl"])
    lo["hard"] = np.abs(lo["v69"] - lo["s6"]).to_numpy() > 1e-6
    checks["dev_rows_moved_after_the_stacker_(intruder/own_motion)"] = int(lo["hard"].sum())
    lo["cls"] = census_class(CENSUS_DEV, ids)
    inputs[CENSUS_DEV] = sha(CENSUS_DEV)
    return lo


def rank_table(inputs: dict, checks: dict) -> pd.DataFrame:
    own_spec = json.loads((ROOT / OWN_DECL).read_text(encoding="utf-8"))["member"]
    msk_spec = json.loads((ROOT / MASK_DECL).read_text(encoding="utf-8"))["member"]
    cols = [ID, "airport", "month", "scope", "v9", "residual", "committee", "ens_ctx_e24", "ens_ctx_e24b", "ens_ctx_e32", "committee_mix", "v30"]
    rec = pd.read_parquet(ROOT / f"{RUNNER}/recipe/detail.parquet", columns=cols)
    rep = pd.read_parquet(ROOT / f"{REPAIRED}/recipe/detail.parquet", columns=cols)
    for r in [f"{RUNNER}/recipe/detail.parquet", f"{REPAIRED}/recipe/detail.parquet", f"{RUNNER}/stack/detail.parquet"]:
        inputs[r] = sha(r)
    rec = rec[rec["airport"].eq("LFPG")].reset_index(drop=True)
    ids = rec[ID].to_numpy()
    rep = rep.set_index(ID).reindex(ids)
    t = rec[[ID, "month", "scope", "v9", "residual", "committee"]].copy()
    t["matched"] = rec["scope"].eq("matched_no_lirf").to_numpy()
    t["nets_old"] = share_gate.network_table(rec)
    t["v30"] = rec["v30"].to_numpy(dtype=float)
    for c in ["v9", "residual", "committee"]:
        t[f"{c}_r"] = rep[c].to_numpy(dtype=float)
    t["nets_old_r"] = share_gate.network_table(rep)
    t["v30_r"] = rep["v30"].to_numpy(dtype=float)
    st = pd.read_parquet(ROOT / f"{RUNNER}/stack/detail.parquet", columns=[ID, "repair_effect", "v31", "v46", "v52m", "v55m"]).set_index(ID).reindex(ids)
    t["repair"] = st["repair_effect"].to_numpy(dtype=float)
    checks["rank_repair_effect_vs_v30r_minus_v30_max_abs"] = float(np.abs(t["repair"] - (t["v30_r"] - t["v30"])).max())
    checks["rank_repair_nonzero_rows_by_month"] = {m: int(((t["repair"] != 0) & t["month"].eq(m)).sum()) for m in sorted(t["month"].unique())}
    t["own"] = seed_mean(own_spec, "ranking", ids, inputs)
    t["msk"] = seed_mean(msk_spec, "ranking", ids, inputs)
    d = pd.read_parquet(ROOT / f"{STEPS}/model_part/member_delta.parquet").set_index(ID).reindex(ids)
    inputs[f"{STEPS}/model_part/member_delta.parquet"] = sha(f"{STEPS}/model_part/member_delta.parquet")
    t["v55m"] = st["v55m"].to_numpy(dtype=float) + d["d52"].to_numpy(dtype=float)
    mp = pd.read_parquet(ROOT / checked(M_PRIME, inputs)).set_index(ID)[TARGET].reindex(ids).to_numpy(dtype=float)
    checks["rank_v55m_float_rounds_to_model_part_file_rows_differ"] = int((np.rint(t["v55m"].to_numpy()) != mp).sum())
    t["C"] = pd.read_parquet(ROOT / checked(C_FILE, inputs)).set_index(ID)[TARGET].reindex(ids).to_numpy(dtype=float)
    P = pd.read_parquet(ROOT / checked(RANK_PREDS, inputs)).set_index(ID).reindex(ids).reset_index()
    if not np.allclose(P["new"].to_numpy(dtype=float), t["C"].to_numpy()):
        raise ValueError("The stacker's base is not the step-5 file.")
    t["new_ctl"] = t["C"]
    t["p3"], t["fx"], t["h3"] = p3_bag(P), P["fx"].to_numpy(dtype=bool), P["h3"].to_numpy(dtype=bool)
    t["s6"] = np.where(t["fx"], t["C"], np.maximum(t["C"] + 0.65 * t["p3"], -12.0))
    t["v69"] = pd.read_parquet(ROOT / checked(V69, inputs)).set_index(ID)[TARGET].reindex(ids).to_numpy(dtype=float)
    t["v70"] = pd.read_parquet(ROOT / checked(V70, inputs)).set_index(ID)[TARGET].reindex(ids).to_numpy(dtype=float)
    t["hard"] = np.rint(t["s6"].to_numpy()) != t["v69"].to_numpy()
    checks["rank_rows_moved_after_the_stacker_(intruder/own_motion)"] = int(t["hard"].sum())
    t["cls"] = census_class(CENSUS_RANK, ids)
    inputs[CENSUS_RANK] = sha(CENSUS_RANK)
    return t


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inputs, checks = {}, {}
    dev = dev_table(inputs, checks)
    rank = rank_table(inputs, checks)
    for name, t in [("dev", dev), ("rank", rank)]:
        p = OUT / f"components_{name}.parquet"
        t.to_parquet(p, index=False)
        checks[f"{name}_rows"] = int(len(t))
        checks[f"{name}_finite_share"] = {c: float(np.isfinite(t[c].to_numpy(dtype=float)).mean()) for c in
                                          ["v9", "residual", "committee", "nets_old", "own", "msk", "v55m", "C", "p3", "v69"]}
        checks[f"{name}_rows_by_month_class"] = {f"{m} {c}": int(n) for (m, c), n in t.groupby(["month", "cls"]).size().items()}
        checks[f"{name}_msk_finite_by_class"] = {c: float(np.isfinite(g["msk"].to_numpy()).mean()) for c, g in t.groupby("cls")}
    out = {"outputs": {f"runs/val2026/components_{n}.parquet": sha(f"runs/val2026/components_{n}.parquet") for n in ["dev", "rank"]},
           "inputs_sha256": inputs, "checks": checks, "code_sha256": {"tools/val2026_components.py": sha("tools/val2026_components.py")},
           "official_scores_used": "none"}
    (OUT / "components.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
