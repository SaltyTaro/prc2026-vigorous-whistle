"""Tier B, chain side (lane C; configs/febjun_tierb_v1.json, configs/febjun_tierb_readings_chain_v1.json, docs/final_file/tierb_chain.md):
the final file's chain after the model part -- the own-points member (step 1), steps 2-10 of tools/final_early_chain.py and steps 11-18 of
tools/final_late_chain.py -- run on a February + June 2025 movement file, so that each step's output against its input is that layer's
February / June 2025 reading (the chain with the layer against the chain with the layer passed through at that position).

  input        F = every movement (DEP and ARR) of the twelve 2025 training files whose MVT_TIME_UTC_mvt falls in February or June 2025 (the
               catalog's month assignment, as the tree LOMO held rows), in file order; departures carry no block time and no label (as
               data/raw/ranking.parquet); T = its departures' IDs with an empty prediction column (as data/raw/submitting.parquet)
  own-member   the own-points member (own_all: dl32own on rows with adsb_matched = 1, m25dl32own on the others; seeds 601-604; folds
               lomo_2025-02 / lomo_2025-06 of runs/febjun_nets/nets) per seed -> own/s<seed>/<month>.parquet; --check rebuilds the January
               2025 development member tables (16 seeds, runs/ctx_ownpoints_dev/nets) and requires them byte-for-value equal to the stored ones
  root         a runner root for F from lane M's tables (runs/febjun_tierb/model/chain_<month>.parquet): recipe/detail, networks/ensembles,
               leaves/tables, stack/detail, members/members, input.json; the model part M = v55m (nearest integer, template order). F's
               departures without a lane-M row (no out-of-sample v9: missing NM clock, LIRF fill population, month-boundary rows) enter at
               their airport-month's median v55m in every column (the rows12 convention for its 'extra' rows); --standin builds the same
               from runs/daily_totals/rows12.parquet 'stack' (code-path test only, never read)
  c1           step 1: M' = M with the own_all member at its shipped weight (1.00) in place of the combination member (0.40), then the
               masked member at 0.40, on non-LIRF rows (tools/final_early_chain.py step 1's arithmetic, on lane M's stack functions); also
               member_delta.parquet (d51, d52) for the stacker's v51m-v54m columns
  early        steps 2-10 on M' with tools/final_early_chain.Chain (new-file path, --published-missing auto, as the final file). Every fit
               on 2025 labels that the shipped ranking path takes over all twelve 2025 months (the rematch offsets, the intruder bound's
               airport set and offsets, the identity extension's, own-motion v1's and v2's offsets) is month-blocked: run once with February
               2025 and once with June 2025 removed from that fit, February rows taken from the first run and June rows from the second
               (the layers' own development / twelve-month form, 'every other 2025 month'). Steps 3-6 and the late layers fit on the four
               development months only.
  late         steps 11-18 with tools/final_late_chain.py (init / step / fit), as the final file
  read         the readings of configs/febjun_tierb_readings_chain_v1.json; refuses unless that declaration is committed and unchanged

Labels: no February / June 2025 label is read before `read`. Nothing is uploaded; nothing is written under runs/submissions.
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import json
import shutil
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
from prc2026.features import ID, TARGET, TIME  # noqa: E402

BASE = ROOT / "runs/febjun_tierb/chain"
INPUT = BASE / "input"
F_PATH, T_PATH = INPUT / "febjun2025.parquet", INPUT / "febjun2025_template.parquet"
MONTHS = ["2025-02", "2025-06"]
# January 2025's arrivals ride along as context: the final file holds January 2026 before its February rows (inbound arrivals, stand
# in-blocks and identities of the turn before a 1 February departure), but nothing before June 2026 -- so June 2025 gets no May context
CONTEXT_ARR_MONTHS = ["2025-01"]
ALL12 = {f"2025-{m:02d}" for m in range(1, 13)}
MODEL_DIR = ROOT / "runs/febjun_tierb/model"
NETS = ROOT / "runs/febjun_nets/nets"
OWN_DIR = BASE / "own"
ADSB_CACHE = ROOT / "runs/adsb_ctx2/inputs/adsb_features_arcw0"
OWN_CONFIG = "configs/ctx_ownpoints_network_v1.json"
OWN_ROUTE = {"matched": "dl32own", "unmatched": "m25dl32own"}
OWN_SEEDS = [601, 602, 603, 604]
OWN_DECL, OWN_CAND = "configs/ctx_ownpoints_weight_v1.json", "own_all_w100"
DECLARATION = "configs/febjun_tierb_readings_chain_v1.json"
VERSION = 9210   # work-directory name of the early chain's files (not a board version)
READ_SEED = 20261004


def rel(p: Path | str) -> str:
    p = Path(p)
    p = p if p.is_absolute() else ROOT / p
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def departures() -> pd.DataFrame:
    """F's departures in template order with their MVT month."""
    raw = pd.read_parquet(F_PATH, columns=[ID, "PHASE_mvt", "ADEP_mvt", "ADEP_flt", "AIRCRAFT_TYPE_mvt", TIME])
    dep = raw[raw["PHASE_mvt"].eq("DEP")].reset_index(drop=True)
    tmpl = pd.read_parquet(T_PATH, columns=[ID])
    if not np.array_equal(dep[ID].to_numpy(), tmpl[ID].to_numpy()):
        raise ValueError("The template order differs from F's departures.")
    dep["month"] = dep[TIME].dt.strftime("%Y-%m")
    dep["pop"] = dep["ADEP_mvt"].eq("LIRF") & dep["ADEP_flt"].isna() & dep["AIRCRAFT_TYPE_mvt"].isna()
    return dep


# ---------------------------------------------------------------- input

def build_input() -> None:
    files = sorted(glob.glob(str(ROOT / "data/raw/training_2025-*.parquet")))
    rank_schema = pd.read_parquet(ROOT / "data/raw/ranking.parquet").iloc[:0]
    parts = []
    for f in files:
        t = pd.read_parquet(f)
        if list(t.columns) != list(rank_schema.columns):
            raise ValueError(f"{f}: columns differ from the ranking file.")
        mo = t[TIME].dt.strftime("%Y-%m")
        parts.append(t[mo.isin(MONTHS) | (mo.isin(CONTEXT_ARR_MONTHS) & t["PHASE_mvt"].eq("ARR"))])
    f_ = pd.concat(parts, ignore_index=True)
    if f_[ID].duplicated().any():
        raise ValueError("Duplicate movement IDs across the training files.")
    dep = f_["PHASE_mvt"].eq("DEP")
    f_[TARGET] = f_[TARGET].astype("float64")
    f_.loc[dep, TARGET] = np.nan
    f_.loc[dep, "BLOCK_TIME_UTC_mvt"] = pd.NaT
    f_ = f_.astype(rank_schema.dtypes.to_dict())
    if not (f_.dtypes == rank_schema.dtypes).all():
        raise ValueError("Column types differ from the ranking file.")
    INPUT.mkdir(parents=True, exist_ok=True)
    if F_PATH.exists() or T_PATH.exists():
        raise ValueError("The input files exist; never overwrite.")
    f_.to_parquet(F_PATH, index=False)
    import pyarrow as pa
    import pyarrow.parquet as pq
    tmpl = pd.DataFrame({ID: f_.loc[dep, ID].to_numpy(dtype="float64")})
    pq.write_table(pa.table({ID: pa.array(tmpl[ID].to_numpy(), type=pa.float64()), TARGET: pa.nulls(len(tmpl), type=pa.int32())}), T_PATH)  # as submitting.parquet
    month = f_[TIME].dt.strftime("%Y-%m")
    rec = {"created_at_utc": utc_now(), "months": MONTHS, "context_arrival_months": CONTEXT_ARR_MONTHS,
           "rule": "MVT_TIME_UTC_mvt month in the months, all phases, plus the arrivals of the context months; departures without block time / label",
           "sources": {Path(f).relative_to(ROOT).as_posix(): sha256_file(Path(f)) for f in files},
           "movement_file": {"path": rel(F_PATH), "sha256": sha256_file(F_PATH), "rows": int(len(f_))},
           "template": {"path": rel(T_PATH), "sha256": sha256_file(T_PATH), "rows": int(len(tmpl))},
           "departures_by_month": f_[dep].groupby(month[dep]).size().to_dict(), "arrivals_by_month": f_[~dep].groupby(month[~dep]).size().to_dict()}
    write_json(INPUT / "input.json", rec)
    print({k: rec[k] for k in ["movement_file", "template", "departures_by_month", "arrivals_by_month"]})


# ---------------------------------------------------------------- own-points member

def route_seed(nets: Path, seed: int, fold: str, file: str, flags: pd.Series, cache_hashes: dict, config_sha: str) -> pd.DataFrame:
    """build_combo_member's per-seed routing for the own_all member (dl32own matched / m25dl32own unmatched)."""
    from build_combo_member import check_cache
    frames = {}
    for side, variant in OWN_ROUTE.items():
        d = nets / f"{variant}_s{seed}" / fold
        r = read_json(d / "run.json")
        if (r["config_sha256"] != config_sha or r["variant"] != variant or r["seed"] != seed or r["fold"] != fold
                or r.get("smoke_limit_rows") is not None or r["inputs"]["device"] != "cuda"):
            raise ValueError(f"{d} is not a declared fit.")
        check_cache(r, cache_hashes, d)
        frames[side] = pd.read_parquet(d / file)
    ids = frames["matched"][ID].to_numpy()
    if not np.array_equal(ids, frames["unmatched"][ID].to_numpy()):
        raise ValueError(f"seed {seed} {fold}: the networks predict different rows.")
    matched = flags.reindex(ids)
    if matched.isna().any():
        raise ValueError(f"seed {seed} {fold}: rows without a match flag.")
    m = matched.to_numpy(dtype=bool)
    values = np.where(m, frames["matched"]["adsb_full"].to_numpy(dtype=float), frames["unmatched"]["adsb_full"].to_numpy(dtype=float))
    return pd.DataFrame({ID: ids, "adsb_full": values})


def cmd_own_member(check: bool) -> None:
    from build_combo_member import matched_flags
    flags, cache_hashes = matched_flags(ADSB_CACHE)
    config_sha = sha256_file(ROOT / OWN_CONFIG)
    rec = {"created_at_utc": utc_now(), "route": OWN_ROUTE, "network_config": OWN_CONFIG, "network_config_sha256": config_sha,
           "adsb_cache": rel(ADSB_CACHE), "tool_sha256": sha256_file(Path(__file__))}
    if check:
        decl = read_json(ROOT / "configs/ctx_ownpoints_dev_v1.json")
        out = {}
        for seed, stored in zip(decl["fits"]["seeds"], decl["member"]["tables"]["2025-01"]):
            built = route_seed(ROOT / "runs/ctx_ownpoints_dev/nets", seed, "lomo_2025-01", "predictions.parquet", flags, cache_hashes, config_sha)
            ref = pd.read_parquet(ROOT / stored)
            same_ids = np.array_equal(built[ID].to_numpy(), ref[ID].to_numpy())
            a, b = built["adsb_full"].to_numpy(dtype=float), ref["adsb_full"].to_numpy(dtype=float)
            out[stored] = {"ids_equal": same_ids, "nan_pattern_equal": bool((np.isnan(a) == np.isnan(b)).all()),
                           "max_abs": float(np.nanmax(np.abs(a - b))) if same_ids else None}
            if not (same_ids and out[stored]["nan_pattern_equal"] and out[stored]["max_abs"] == 0.0):
                raise ValueError(f"own-member check fails for {stored}: {out[stored]}")
        rec["check_january_2025_16_seeds"] = out
        write_json(BASE / "checks" / "own_member.json", rec)
        print("own member check: 16 seeds of 2025-01 reproduce the stored development member tables exactly")
        return
    built = {}
    for month in MONTHS:
        for seed in OWN_SEEDS:
            t = route_seed(NETS, seed, f"lomo_{month}", "predictions.parquet", flags, cache_hashes, config_sha)
            p = OWN_DIR / f"s{seed}" / f"{month}.parquet"
            p.parent.mkdir(parents=True, exist_ok=True)
            t.to_parquet(p, index=False, compression="zstd")
            built[rel(p)] = {"rows": int(len(t)), "finite": int(np.isfinite(t["adsb_full"]).sum()), "sha256": sha256_file(p)}
    rec["tables"] = built
    write_json(OWN_DIR / "member_build.json", rec)
    print(json.dumps(built, indent=1))


def own_member_values(ids: np.ndarray) -> np.ndarray:
    from prc2026_final import network_member
    tabs = []
    for seed in OWN_SEEDS:
        t = pd.concat([pd.read_parquet(OWN_DIR / f"s{seed}" / f"{m}.parquet") for m in MONTHS], ignore_index=True)
        tabs.append(t.set_index(ID)["adsb_full"].reindex(ids).to_numpy(dtype=float))
    return network_member.seed_mean(tabs)


# ---------------------------------------------------------------- runner root from lane M's tables

STACK_COLS = ["v30", "v31", "v33", "v34", "v35", "v38", "v40", "v45", "v46", "v51m", "v52m", "v53m", "v54m", "v55m"]
COMP_COLS = ["v9", "residual", "committee", "lirf_tree", "ens_ctx_e24", "ens_ctx_e24b", "ens_ctx_e32", "ens_ctx_lirf", "mix"]


def residual_matched() -> pd.Series:
    """The residual stage's ADS-B match flag (run_adsb_v24.adsb_features' 'adsb_matched' on matches_v11, all twelve 2025 months): lane M's
    'matched' column for its rows; the scope of F's departures without a lane-M row."""
    parts = [pd.read_parquet(p, columns=[ID, "adsb_matched"]) for p in sorted((ROOT / "runs/adsb_v24/features").glob("matches_v11_2025-*.parquet"))]
    t = pd.concat(parts, ignore_index=True).drop_duplicates(ID)
    return t.set_index(ID)["adsb_matched"].eq(1)


def model_rows(standin: bool) -> pd.DataFrame:
    if not standin:
        rows = pd.concat([pd.read_parquet(MODEL_DIR / f"chain_{m}.parquet") for m in MONTHS], ignore_index=True)
        return rows.drop(columns=[c for c in ["label"] if c in rows.columns])   # lane M's tables carry the label; the chain never reads it
    # code-path stand-in: rows12 'stack' in every component and layer column (never read)
    r12 = pd.read_parquet(ROOT / "runs/daily_totals/rows12.parquet", columns=[ID, "month", "airport", "stack"])
    r12 = r12[r12["month"].isin(MONTHS)].reset_index(drop=True)
    flags = residual_matched()
    from build_combo_member import matched_flags
    learner, _ = matched_flags(ADSB_CACHE)
    f = r12[[ID, "month", "airport"]].copy()
    f["matched"] = flags.reindex(f[ID]).fillna(False).to_numpy(dtype=bool)
    lirf = f["airport"].eq("LIRF").to_numpy()
    f["lirf_m"], f["lirf_u"] = lirf & f["matched"].to_numpy(), lirf & ~f["matched"].to_numpy()
    for c in COMP_COLS + STACK_COLS + ["nets", "base", "lirf_net"]:
        f[c] = r12["stack"].to_numpy(dtype=float)
    f["residual"] = 0.0
    f["floor"] = -12.0
    f["adsb_matched_learner"] = learner.reindex(f[ID]).fillna(False).to_numpy(dtype=bool)
    for c in ["combo_all", "mask_unmatched", "lirf_mask"]:
        f[c] = np.nan
    return f


def cmd_root(standin: bool) -> None:
    from export_daily_totals import SCOPES
    from prc2026_submission.contract import export_submission
    root = BASE / ("root_standin" if standin else "root")
    if (root / "input.json").exists():
        raise ValueError(f"{root} exists; never overwrite.")
    dep = departures()
    rows = model_rows(standin).drop_duplicates(ID)
    have = dep[ID].isin(rows[ID]).to_numpy()
    extra_ids = set(rows[ID]) - set(dep[ID])
    if extra_ids:
        raise ValueError(f"{len(extra_ids)} lane-M rows are not departures of F.")
    m = rows.set_index(ID).reindex(dep[ID])
    v55 = m["v55m"].to_numpy(dtype=float)
    # placeholder: the airport-month median of lane M's v55m over its rows (rows12's convention for its extra rows)
    med = pd.DataFrame({"airport": dep["ADEP_mvt"].to_numpy(), "month": dep["month"].to_numpy(), "v": v55})[have].groupby(["airport", "month"])["v"].median()
    ph = pd.MultiIndex.from_arrays([dep["ADEP_mvt"], dep["month"]]).map(med.to_dict()).to_numpy(dtype=float)
    if np.isnan(ph[~have]).any():
        raise ValueError("An airport-month without lane-M rows.")
    resid_flag = residual_matched().reindex(dep[ID]).fillna(False).to_numpy(dtype=bool)
    matched = np.where(have, m["matched"].to_numpy(dtype=bool) if "matched" in m else resid_flag, resid_flag)
    lirf = dep["ADEP_mvt"].eq("LIRF").to_numpy()
    lirf_m = np.where(have, m["lirf_m"].fillna(False).to_numpy(dtype=bool), lirf & matched & ~dep["pop"].to_numpy())
    lirf_u = np.where(have, m["lirf_u"].fillna(False).to_numpy(dtype=bool), lirf & ~matched & ~dep["pop"].to_numpy())
    scope = np.where(~lirf, np.where(matched, "matched_no_lirf", "all_no_lirf"),
                     np.where(lirf_m, "lirf_nonpop_matched", np.where(lirf_u, "lirf_nonpop_unmatched", "fixed")))
    if set(scope) - set(SCOPES):
        raise ValueError("Unknown scope.")

    def col(c: str) -> np.ndarray:
        v = m[c].to_numpy(dtype=float) if c in m else np.full(len(dep), np.nan)
        return np.where(have, v, ph if c != "residual" else 0.0)

    for d in ["recipe", "networks", "leaves", "stack", "members", "stages", "final"]:
        (root / d).mkdir(parents=True, exist_ok=True)
    ids = dep[ID].to_numpy()
    detail = pd.DataFrame({ID: ids, "airport": dep["ADEP_mvt"].to_numpy(), "month": dep["month"].to_numpy(), "scope": scope, "v9": col("v9"),
                           "residual": col("residual"), "committee": col("committee"), "ens_v22_lirf_q": col("lirf_tree"),
                           "ens_ctx_e24": col("ens_ctx_e24"), "ens_ctx_e24b": col("ens_ctx_e24b"), "ens_ctx_e32": col("ens_ctx_e32"),
                           "ens_ctx_lirf": col("ens_ctx_lirf"), "committee_mix": col("mix"), "v30": col("v30")})
    detail.to_parquet(root / "recipe" / "detail.parquet", index=False)
    pd.DataFrame({ID: ids, **{c: col(c) for c in ["ens_ctx_e24", "ens_ctx_e24b", "ens_ctx_e32", "ens_ctx_lirf"]}}).to_parquet(root / "networks" / "ensembles.parquet", index=False)
    pd.DataFrame({ID: ids, "ens_v20_bag_q_all": col("committee"), "ens_v22_lirf_q": col("lirf_tree")}).to_parquet(root / "leaves" / "tables.parquet", index=False)
    st = pd.DataFrame({ID: ids, "airport": dep["ADEP_mvt"].to_numpy(), "month": dep["month"].to_numpy(), "scope": scope, **{c: col(c) for c in STACK_COLS}})
    st.to_parquet(root / "stack" / "detail.parquet", index=False)
    mem = pd.DataFrame({ID: ids, "adsb_matched": np.where(have, m["adsb_matched_learner"].fillna(False).to_numpy(dtype=bool), False) if "adsb_matched_learner" in m else False,
                        **{c: np.where(have, m[c].to_numpy(dtype=float), np.nan) if c in m else np.nan for c in ["combo_all", "mask_unmatched", "lirf_mask"]}})
    mem.to_parquet(root / "members" / "members.parquet", index=False)
    mp = root / "final" / "vigorous-whistle_v9201.parquet"
    export_submission(pd.DataFrame({ID: ids, "prediction": col("v55m")}), T_PATH, mp,
                      {"dry_run": True, "step": "Tier B model part v55m (lane M's chain tables; placeholder rows at the airport-month median)",
                       "official_scores_used": "none", "audit_used": False, "standin": standin})
    info = {"created_at_utc": utc_now(), "standin": standin, "months": MONTHS, "departures": int(len(dep)), "lane_m_rows": int(have.sum()),
            "placeholder_rows": int((~have).sum()), "placeholder_by_month_airport": pd.DataFrame({"m": dep["month"], "a": dep["ADEP_mvt"]})[~have].value_counts().sort_index().to_dict(),
            "placeholder_population_rows": int((~have & dep["pop"].to_numpy()).sum()), "scope_counts": pd.Series(scope).value_counts().to_dict(),
            "raw": rel(F_PATH), "raw_sha256": sha256_file(F_PATH), "template": rel(T_PATH), "template_sha256": sha256_file(T_PATH),
            "model_part": {"file": rel(mp), "sha256": sha256_file(mp)},
            "lane_m_tables": {rel(MODEL_DIR / f"chain_{mo}.parquet"): sha256_file(MODEL_DIR / f"chain_{mo}.parquet") for mo in MONTHS} if not standin else None,
            "files": {rel(p): sha256_file(p) for p in sorted(root.rglob("*.parquet"))}}
    info["placeholder_by_month_airport"] = {f"{k[0]} {k[1]}": int(v) for k, v in info["placeholder_by_month_airport"].items()}
    write_json(root / "input.json", info)
    print({k: v for k, v in info.items() if k not in ("files", "placeholder_by_month_airport")})


# ---------------------------------------------------------------- step 1 (C1): the own-points member

def cmd_c1(standin: bool) -> None:
    from prc2026_submission.contract import export_submission
    from prc2026_final import network_member, share_gate
    root = BASE / ("root_standin" if standin else "root")
    work = BASE / ("work_standin" if standin else "work")
    out = work / "steps" / "model_part"
    path = out / "vigorous-whistle_v66.parquet"
    if path.exists():
        raise ValueError(f"{path} exists; never overwrite.")
    dep = departures()
    ids = dep[ID].to_numpy()
    mp = root / "final" / "vigorous-whistle_v9201.parquet"
    shipped = pd.read_parquet(mp).set_index(ID)[TARGET].reindex(ids).to_numpy(dtype=float)
    d51, d52 = np.zeros(len(ids)), np.zeros(len(ids))
    rec: dict = {"created_at_utc": utc_now(), "standin": standin}
    if not standin:
        import febjun_tierb_model as TM
        # read as tools/final_early_chain.py reads it: candidate_weight refuses the candidate's extra 'member' key
        cand = read_json(ROOT / OWN_DECL)["candidates"][OWN_CAND]
        if cand.get("member") != "own_all":
            raise ValueError(f"{OWN_CAND}: unexpected member {cand.get('member')!r}")
        w = float(cand["member_weight"])
        wm = network_member.candidate_weight(read_json(ROOT / "configs/adsb_ctx3_mask_dev_v1.json"), "w40")
        k = read_json(ROOT / TM.STRETCH)["candidates"]["two_sided_900"]["k_all_months"]
        gate_spec = read_json(ROOT / TM.GATE_DECL)["candidates"]["u600_m600"]
        share, spec = share_gate.candidate_spec(read_json(ROOT / TM.SHARE_DECL), "n85_nd1200")
        cs, off_cs, seg, off = TM.queue_inputs(MONTHS, {m: TM.OUT / "rematch" / f"rematch_{m}.parquet" for m in MONTHS})
        pos = pd.Series(np.arange(len(ids)), index=ids)
        checks = {}
        for month in MONTHS:
            f = pd.read_parquet(MODEL_DIR / f"chain_{month}.parquet")
            f = f.drop(columns=[c for c in ["label"] if c in f.columns])
            lirf = f["airport"].to_numpy() == "LIRF"
            nets, committee = f["nets"].to_numpy(dtype=float), f["committee"].to_numpy(dtype=float)
            own = np.where(lirf, np.nan, own_member_values(f[ID].to_numpy()))
            combo, mask = np.where(lirf, np.nan, f["combo_all"].to_numpy(dtype=float)), np.where(lirf, np.nan, f["mask_unmatched"].to_numpy(dtype=float))

            def stack_of(member: np.ndarray, weight: float, with_mask: bool) -> np.ndarray:
                t, _ = network_member.table_with_member(nets, member, weight)
                if with_mask:
                    t, _ = network_member.table_with_member(t, mask, wm)
                mix_m, _ = share_gate.mix(committee, t, share, spec)
                return TM.full_stack(f, mix_m, cs, off_cs, seg, off, k, gate_spec)

            k51, k52 = stack_of(combo, 0.40, False), stack_of(combo, 0.40, True)
            c51, c52 = stack_of(own, w, False), stack_of(own, w, True)
            checks[month] = {"k51_vs_lane_m_v51m_max_abs": float(np.max(np.abs(k51 - f["v51m"].to_numpy(dtype=float)))),
                             "k52_vs_lane_m_v52m_max_abs": float(np.max(np.abs(k52 - f["v52m"].to_numpy(dtype=float)))),
                             "lirf_member_effect_max_abs": float(np.abs((c52 - k52)[lirf]).max(initial=0.0)),
                             "own_member_finite_rows": int(np.isfinite(own).sum()), "rows": int(len(f))}
            if checks[month]["k51_vs_lane_m_v51m_max_abs"] > 1e-9 or checks[month]["k52_vs_lane_m_v52m_max_abs"] > 1e-9 or checks[month]["lirf_member_effect_max_abs"] > 1e-9:
                raise ValueError(f"step-1 checks fail: {checks[month]}")
            p = pos.reindex(f[ID]).to_numpy()
            d51[p], d52[p] = c51 - k51, c52 - k52
        rec.update({"member_weight": w, "mask_weight": wm, "checks": checks, "own_seeds": OWN_SEEDS})
    changed = np.abs(d52) > 1e-9
    values = np.where(changed, shipped + d52, shipped)  # on non-LIRF rows v55m = v52m, so c52 = v55m + (c52 - k52)
    out.mkdir(parents=True, exist_ok=True)
    export_submission(pd.DataFrame({ID: ids, "prediction": values}), T_PATH, path,
                      {"dry_run": True, "step": "Tier B step 1: model part with the own_all member (1.00) in place of the combination member",
                       "official_scores_used": "none", "audit_used": False, "standin": standin})
    pd.DataFrame({ID: ids, "d51": d51, "d52": d52}).to_parquet(out / "member_delta.parquet", index=False)
    exported = pd.read_parquet(path).set_index(ID)[TARGET].reindex(ids).to_numpy(dtype=float)
    rec.update({"file": rel(path), "sha256": sha256_file(path), "rows_float_changed": int(changed.sum()),
                "rows_changed_vs_model_part": int((exported != shipped).sum()), "member_delta_sha256": sha256_file(out / "member_delta.parquet"),
                "model_part": {"file": rel(mp), "sha256": sha256_file(mp)}})
    write_json(out / "step1.json", rec)
    print({k: v for k, v in rec.items() if k != "checks"}, rec.get("checks"))


# ---------------------------------------------------------------- month-blocked 2025-label fits

class Blocked:
    """Within the context, the shipped functions that fit 2025-label offsets for the ranking side drop month H from a fit over all twelve
    2025 months (calls on fewer months -- the tools' own month-blocked development tables -- are left as they are). Every call is logged."""

    def __init__(self, held: str, log: list):
        self.held, self.log = held, log
        self.saved = []

    def __enter__(self):
        import run_intruder_bound as R
        import run_intruder_identity as I
        import run_own_motion as OM
        for mod in (R, I, OM):
            orig = mod.fit
            self.saved.append((mod, orig))

            def wrapped(fitting, *a, _orig=orig, _name=mod.__name__, **kw):
                months = set(fitting["month"].astype(str).unique())
                drop = months >= ALL12
                if drop:
                    fitting = fitting[fitting["month"].astype(str) != self.held]
                self.log.append({"module": _name, "months": len(months), "held_removed": bool(drop), "held": self.held, "rows": int(len(fitting))})
                return _orig(fitting, *a, **kw)
            mod.fit = wrapped
        return self

    def __exit__(self, *exc):
        for mod, orig in self.saved:
            mod.fit = orig
        return False


def splice(files: dict[str, Path], target: Path, template: Path) -> Path:
    """February rows from the February-blocked run, June rows from the June-blocked run, in the template's order."""
    dep = departures()
    a = {m: pd.read_parquet(p) for m, p in files.items()}
    for m, t in a.items():
        if not np.array_equal(t[ID].to_numpy(), dep[ID].to_numpy()):
            raise ValueError(f"{files[m]}: row order differs from the template.")
    out = a[MONTHS[0]].copy()
    for m in MONTHS:
        sel = dep["month"].eq(m).to_numpy()
        out.loc[sel, TARGET] = a[m].loc[sel, TARGET].to_numpy()
    if not dep["month"].isin(MONTHS).all():
        raise ValueError("A departure outside the two months.")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ValueError(f"{target} exists; never overwrite.")
    out.to_parquet(target, index=False)
    return target


# ---------------------------------------------------------------- OPDI features of F

OPDI_SPANS = [("2025-01-31", "2025-03-02"), ("2025-05-31", "2025-07-01")]


def opdi_build(O, kind: str, files: list[Path], dest: Path) -> dict:
    """tools/run_opdi_stack.build's body with the event files given explicitly: its 'rank' branch selects the 2026 windows by name; F
    needs the February / June 2025 windows (fetched with tools/opdi_stack_fetch.py --need). Same matching, same columns."""
    dep = O.departures(kind)
    of = O.opdi_flights(files)
    m = dep.merge(of, on=["airport", "cs"], how="inner")
    m["dt"] = m["ref"] - m["t"]
    m = m[(m["dt"] >= O.LO) & (m["dt"] <= O.HI)].copy()
    m["adt"] = m["dt"].abs()
    m = m.sort_values([ID, "adt"]).drop_duplicates(ID)
    t = m["t"]
    out = pd.DataFrame({ID: m[ID].to_numpy()})
    out["o_ref_rel"] = (m["dt"]).to_numpy()
    for c, src in [("o_stand_exit", "stand_exit"), ("o_apron_exit", "apron_exit"), ("o_twy_first", "twy_first"), ("o_first_seen", "first_seen"),
                   ("o_rwy_entry", "rwy_entry")]:
        out[c] = (t - m[src]).to_numpy()
    out["o_n_twy"] = m["n_twy"].fillna(0).to_numpy()
    out["o_deice"] = m["deice"].fillna(0).to_numpy()
    out["o_deice_dur"] = m["deice_dur"].to_numpy()
    out["o_has"] = 1.0
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest, index=False)
    rec = {"written_at_utc": utc_now(), "kind": kind, "rows": int(len(out)), "departures": int(len(dep)), "sha256": sha256_file(dest),
           "event_files": {f.name: sha256_file(f) for f in files}, "code_sha256": sha256_file(ROOT / "tools/run_opdi_stack.py"),
           "keymap_sha256": sha256_file(O.KEYMAP), "builder": "tools/febjun_tierb_chain.opdi_build"}
    write_json(dest.with_suffix(".receipt.json"), rec)
    print(kind, "departures", len(dep), "matched", len(out), flush=True)
    return rec


def opdi_files_for_f(O) -> list[Path]:
    out = []
    for f in sorted(O.EVENTS.glob("events_*.parquet")):
        a, b = pd.Timestamp(f.name[7:15]), pd.Timestamp(f.name[16:24])
        if any(a < pd.Timestamp(hi) and b > pd.Timestamp(lo) for lo, hi in OPDI_SPANS):
            out.append(f)
    return out


def cmd_opdi_check() -> None:
    """The replica on the ranking file with the shipped 'rank' selection must reproduce runs/opdi_stack/features_rank.parquet."""
    import run_opdi_stack as O
    files = [f for f in sorted(O.EVENTS.glob("events_*.parquet")) if f.name[7:11] == "2026" or f.name.startswith("events_20251231")]
    shipped = read_json(ROOT / "runs/opdi_stack/features_rank.receipt.json")
    files = [f for f in files if f.name in shipped["event_files"]]
    dest = BASE / "checks" / "opdi_features_rank_replica.parquet"
    rec = opdi_build(O, "rank", files, dest)
    a, b = pd.read_parquet(dest), pd.read_parquet(ROOT / "runs/opdi_stack/features_rank.parquet")
    same = a.equals(b)
    write_json(BASE / "checks" / "opdi_build.json", {"created_at_utc": utc_now(), "replica": rec, "shipped": shipped, "frames_equal": bool(same),
                                                      "event_files_equal": sorted(rec["event_files"]) == sorted(shipped["event_files"])})
    print("OPDI replica equals the shipped ranking features:", same)
    if not same:
        raise ValueError("The OPDI build replica does not reproduce the shipped ranking features.")


# ---------------------------------------------------------------- early chain (steps 2-10)

def chain_args(root: Path, work: Path):
    import final_early_chain as FE
    return FE.parser().parse_args(["--raw", rel(F_PATH), "--template", rel(T_PATH), "--runner", rel(root), "--model-part",
                                   rel(root / "final" / "vigorous-whistle_v9201.parquet"), "--out", rel(work), "--version", str(VERSION),
                                   "--published-missing", "auto"])


def blocked_7_to_10(st: Path, work: Path, rec: dict) -> Path:
    """Steps 7-8 (intruder bound + identity extension), 9 (own-motion v1), 10 (own-motion v2), each month-blocked; the accepted intruder
    layer's own output (the identity extension's input) is kept as steps/intruder/ for its separate reading."""
    import run_intruder_bound as R
    import run_intruder_identity as I
    import run_own_motion as OM
    import run_own_motion_v2 as O2
    blk = work / "blocked"
    v = VERSION
    stage_in = st
    for name, fn in [("identity", lambda base, out: I.dry_run(base, sha256_file(base), v, out)),
                     ("own", lambda base, out: OM.dry_run(base, sha256_file(base), v, out)),
                     ("om2", lambda base, out: O2.dry_run(base, sha256_file(base), v, out))]:
        t = time.time()
        outs, afters = {}, {}
        for held in MONTHS:
            log: list = []
            captured = {}
            orig_values = R.values

            def capture(cand, pred, model, w, _o=orig_values, _c=captured):
                res = _o(cand, pred, model, w)
                if len(cand) and set(cand["month"].astype(str).unique()) == {"ranking"}:
                    _c["accepted"] = res
                return res
            R.values = capture
            try:
                with Blocked(held, log):
                    fn(stage_in, blk / f"{name}_{held}")
            finally:
                R.values = orig_values
            outs[held] = blk / f"{name}_{held}" / f"vigorous-whistle_v{v}.parquet"
            rec.setdefault("blocked_fit_calls", {})[f"{name}_{held}"] = log
            if not any(e["held_removed"] for e in log):
                raise ValueError(f"{name} {held}: no twelve-month fit dropped the held month.")
            if name == "identity":
                base = pd.read_parquet(stage_in)
                ca = captured["accepted"]
                after = base.copy()
                moved = pd.Series(np.rint(ca["value"].to_numpy()), index=ca[ID].to_numpy())
                idx = after[ID].map(moved)
                after[TARGET] = np.where(idx.notna(), idx, after[TARGET]).astype(base[TARGET].dtype)
                p = blk / f"intruder_{held}" / f"vigorous-whistle_v{v}.parquet"
                p.parent.mkdir(parents=True, exist_ok=True)
                after.to_parquet(p, index=False)
                afters[held] = p
        if name == "identity":
            splice(afters, work / "steps" / "intruder" / f"vigorous-whistle_v{v}.parquet", T_PATH)
        stage_in = splice(outs, work / "steps" / name / f"vigorous-whistle_v{v}.parquet", T_PATH)
        rec.setdefault("seconds", {})[name] = round(time.time() - t, 1)
    return stage_in


def cmd_early(standin: bool) -> None:
    import final_early_chain as FE
    root = BASE / ("root_standin" if standin else "root")
    work = BASE / ("work_standin" if standin else "work")
    blk = work / "blocked"
    rec_path = work / "chain_tierb.json"
    if rec_path.exists():
        raise ValueError(f"{rec_path} exists: a finished chain.")
    t0 = time.time()
    c = FE.Chain(chain_args(root, work))
    rec = {"created_at_utc": utc_now(), "standin": standin, "chain": c.record, "seconds": {}, "blocked_fit_calls": {}}
    # step 0's key map and the no-NM matches of F (label-free) are already under work/steps/rematch (tools/run_rematch_nonm.py --matches)
    for f in ["key_map.parquet", "matches_rank.parquet"]:
        if not (c.rematch / f).exists():
            raise ValueError(f"step 0 output missing: {c.rematch / f}")
    import run_opdi_stack as O
    O.RANKING, O.KEYMAP = rel(F_PATH), c.rematch / "key_map.parquet"
    rank_opdi = work / "cache" / "opdi" / "features_rank.parquet"
    if not rank_opdi.exists():   # configure() would build it from the 2026 windows (none cover F): the February / June 2025 windows instead
        rec["opdi_rank"] = opdi_build(O, "rank", opdi_files_for_f(O), rank_opdi)
    rec["configured"] = c.configure()
    m1 = work / "steps" / "model_part" / "vigorous-whistle_v66.parquet"
    if not m1.exists():
        raise ValueError("Run c1 first.")
    # step 2: the no-NM rematch with month-blocked offsets
    import run_rematch_nonm as RN
    import final_chain_overrides as FCO
    outs = {}
    for held in MONTHS:
        d = blk / f"rematch_{held}"
        d.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(c.rematch / "matches_rank.parquet", d / "matches_rank.parquet")
        RN.OUT, RN.BASE, RN.RANKING = d, (rel(m1), sha256_file(m1)), ROOT / rel(F_PATH)
        orig = RN.offsets
        RN.offsets = lambda m25, lab, ex, _o=orig, _h=held: _o(m25[m25["month"] != _h] if ex is None else m25, lab, ex)
        try:
            RN.step_ranking_base()
        finally:
            RN.offsets = orig
        outs[held] = d / "base" / "vigorous-whistle_v55m_rematch.parquet"
        rec.setdefault("rematch_offsets", {})[held] = read_json(d / "base" / "base_receipt.json")["offsets"]
    b = splice(outs, c.rematch / "base" / "vigorous-whistle_v55m_rematch.parquet", T_PATH)
    rec["seconds"]["2"] = round(time.time() - t0, 1)
    t = time.time()
    y, _ = c.steps_3_to_5(b)
    rec["seconds"]["3-5"] = round(time.time() - t, 1)
    t = time.time()
    st, info = c.step6()
    rec["stack"] = info
    rec["seconds"]["6"] = round(time.time() - t, 1)
    v = VERSION
    stage_in = blocked_7_to_10(st, work, rec)
    final = work / f"vigorous-whistle_v{v}.parquet"
    shutil.copyfile(stage_in, final)
    rec["steps"] = {rel(p): sha256_file(p) for p in sorted((work / "steps").rglob("vigorous-whistle_v*.parquet"))}
    rec["final"] = {"file": rel(final), "sha256": sha256_file(final)}
    rec["seconds"]["total"] = round(time.time() - t0, 1)
    rec["code"] = {p: sha256_file(ROOT / p) for p in ["tools/febjun_tierb_chain.py", "tools/final_early_chain.py", "tools/run_rematch_nonm.py",
                                                      "tools/run_stack_weight.py", "tools/run_stack_resid.py", "tools/run_stack_resid_v3.py",
                                                      "tools/run_intruder_bound.py", "tools/run_intruder_identity.py", "tools/run_own_motion.py",
                                                      "tools/run_own_motion_v2.py"]}
    write_json(rec_path, rec)
    print("early chain final", rec["final"], rec["seconds"], flush=True)


def cmd_repro12() -> None:
    """Code-path reproduction of the month-blocked steps 7-10 on F against the stored twelve-month readings of the same layers on
    runs/daily_totals/rows12.parquet 'stack' (already read before Tier B and recorded in configs/febjun_check_v1.json): base = rows12 'stack'
    (nearest integer) on F; steps 7-10 as in `early`; each layer's February / June change against the stored one. Differences come from the
    integer base, F's movements alone (no arrival of the previous month as context) and the relearned key map; they are reported."""
    import final_early_chain as FE
    import run_intruder_bound as R
    from prc2026_submission.contract import export_submission
    work = BASE / "work_repro12"
    if (work / "repro12.json").exists():
        raise ValueError("repro12 exists.")
    (work / "steps" / "rematch").mkdir(parents=True, exist_ok=True)
    for f in ["key_map.parquet", "matches_rank.parquet"]:
        shutil.copyfile(BASE / "work" / "steps" / "rematch" / f, work / "steps" / "rematch" / f)
    c = FE.Chain(chain_args(BASE / "root_standin", work))
    import run_opdi_stack as O
    O.RANKING, O.KEYMAP = rel(F_PATH), c.rematch / "key_map.parquet"
    rec = {"created_at_utc": utc_now(), "configured": c.configure()}
    dep = departures()
    r12 = pd.read_parquet(ROOT / "runs/daily_totals/rows12.parquet", columns=[ID, "month", "stack", "label"]).set_index(ID).reindex(dep[ID])
    if r12["stack"].isna().any():
        raise ValueError("rows12 does not cover F.")
    st = work / "steps" / "stack" / f"vigorous-whistle_v{VERSION}.parquet"
    if not st.exists():
        export_submission(pd.DataFrame({ID: dep[ID].to_numpy(), "prediction": r12["stack"].to_numpy(dtype=float)}), T_PATH, st,
                          {"dry_run": True, "step": "repro12 base: rows12 stack", "official_scores_used": "none", "audit_used": False})
    blocked_7_to_10(st, work, rec)
    v = VERSION
    files = {"base": st, "C7": work / "steps/intruder" / f"vigorous-whistle_v{v}.parquet", "C8": work / "steps/identity" / f"vigorous-whistle_v{v}.parquet",
             "C9": work / "steps/own" / f"vigorous-whistle_v{v}.parquet", "C10": work / "steps/om2" / f"vigorous-whistle_v{v}.parquet"}
    vals = {k: pd.read_parquet(p).set_index(ID)[TARGET].reindex(dep[ID]).to_numpy(dtype=float) for k, p in files.items()}
    stored = read_json(ROOT / "configs/febjun_check_v1.json")["governed_layers_and_readings"]["already_known_at_declaration"]["readings_pct_and_interval_s"]
    keys = {"C7": "intruder_bound_v1 g3600_w100", "C8": "intruder_identity_v1", "C9": "own_motion_v1 om_b2700_w100", "C10": "own_motion_v2 om5_b2700_w50"}
    order = ["base", "C7", "C8", "C9", "C10"]
    day = dep[TIME].dt.strftime("%Y-%m-%d").to_numpy()
    y = r12["label"].to_numpy(dtype=float)
    out = {}
    for i in range(1, len(order)):
        a, b = vals[order[i - 1]], vals[order[i]]
        rng = np.random.default_rng(READ_SEED)
        e = {}
        for m in MONTHS:
            sel = dep["month"].eq(m).to_numpy()
            pct, lo, hi = R.interval(pd.DataFrame({"label": y[sel], "base": a[sel], "day": day[sel]}), "base", b[sel], rng)
            sv = stored[keys[order[i]]][m]
            e[m] = {"path_pct": pct, "path_lo": lo, "path_hi": hi, "stored_pct": sv[0], "stored_lo": sv[1], "stored_hi": sv[2], "rows_moved": int((a[sel] != b[sel]).sum())}
        out[order[i]] = e
        print(order[i], {m: f"path {x['path_pct']:+.4f}% [{x['path_lo']:+.3f},{x['path_hi']:+.3f}] n{x['rows_moved']} | stored {x['stored_pct']:+.4f}% [{x['stored_lo']:+.3f},{x['stored_hi']:+.3f}]"
                         for m, x in e.items()}, flush=True)
    rec["readings_vs_stored"] = out
    rec["files"] = {k: {"file": rel(p), "sha256": sha256_file(p)} for k, p in files.items()}
    write_json(work / "repro12.json", rec)
    write_json(BASE / "checks" / "repro12.json", {k: v for k, v in rec.items() if k != "configured"})


# ---------------------------------------------------------------- late chain (steps 11-18)

LATE_SEQ = [("step", "oooi"), ("step", "nbr_state"), ("step", "nbr_dark"), ("step", "ss_build"), ("fit", "ss"), ("step", "ss_score"),
            ("step", "raw"), ("step", "st_build"), ("fit", "st"), ("step", "st_score"), ("step", "sp_build"), ("fit", "sp"),
            ("step", "sp_score"), ("step", "sp2_build"), ("fit", "sp2"), ("step", "sp2_score"), ("step", "km")]


def cmd_late(standin: bool, start: str | None) -> None:
    root = BASE / ("root_standin" if standin else "root")
    work = BASE / ("work_standin" if standin else "work")
    late = work / "late"
    py = [sys.executable, "-X", "utf8", "-B", "tools/final_late_chain.py"]
    if not (late / "late_chain.json").exists():
        early = read_json(work / "chain_tierb.json")
        subprocess.run(py + ["init", "--work", rel(late), "--raw", rel(F_PATH), "--base", early["final"]["file"], "--recipe", rel(root / "recipe" / "detail.parquet"),
                             "--months", ",".join(MONTHS), "--matches-dir", "data/external/adsblol/matches", "--features-own-dir", "data/external/adsblol/features_v19own",
                             "--messages", "runs/febjun_check/oooi/vm/messages_unique.parquet", "--messages-merge", "files",
                             "--stacker-rank", rel(work / "steps" / "stack" / "stacker_rank.parquet")], cwd=ROOT, check=True)
    go = start is None
    for kind, name in LATE_SEQ:
        go = go or name == start
        if not go:
            continue
        t = time.time()
        args = ["step", "--work", rel(late), "--name", name] if kind == "step" else ["fit", "--work", rel(late), "--stage", name]
        print(">", " ".join(args), flush=True)
        subprocess.run(py + args, cwd=ROOT, check=True)
        print("<", name, round(time.time() - t, 1), "s", flush=True)


# ---------------------------------------------------------------- readings

def committed_unchanged(path: str) -> bool:
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", path], cwd=ROOT, capture_output=True).returncode == 0
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", path], cwd=ROOT).returncode == 0
    return tracked and clean


def layer_files(work: Path) -> list[tuple[str, Path, Path]]:
    """(layer, input file, output file) in chain order."""
    v = VERSION
    s = work / "steps"
    late = read_json(work / "late" / "late_chain.json")["outputs"]
    lf = {k: ROOT / o["file"] for k, o in late.items()}
    seq = [("C1 own-points member", BASE / "root" / "final" / "vigorous-whistle_v9201.parquet", s / "model_part" / "vigorous-whistle_v66.parquet"),
           ("C2 no-NM rematch", s / "model_part" / "vigorous-whistle_v66.parquet", s / "rematch" / "base" / "vigorous-whistle_v55m_rematch.parquet"),
           ("C3 v48 reconciliation", s / "rematch" / "base" / "vigorous-whistle_v55m_rematch.parquet", s / "R" / "vigorous-whistle_v66.parquet"),
           ("C4 extreme-day", s / "R" / "vigorous-whistle_v66.parquet", s / "X" / "vigorous-whistle_v66.parquet"),
           ("C5 variance re-allocation", s / "X" / "vigorous-whistle_v66.parquet", s / "Y" / "vigorous-whistle_v66.parquet"),
           ("C6 guarded stacker", s / "Y" / "vigorous-whistle_v66.parquet", s / "stack" / f"vigorous-whistle_v{v}.parquet"),
           ("C7 intruder bound", s / "stack" / f"vigorous-whistle_v{v}.parquet", s / "intruder" / f"vigorous-whistle_v{v}.parquet"),
           ("C8 identity extension", s / "intruder" / f"vigorous-whistle_v{v}.parquet", s / "identity" / f"vigorous-whistle_v{v}.parquet"),
           ("C9 own-motion v1", s / "identity" / f"vigorous-whistle_v{v}.parquet", s / "own" / f"vigorous-whistle_v{v}.parquet"),
           ("C10 own-motion v2", s / "own" / f"vigorous-whistle_v{v}.parquet", s / "om2" / f"vigorous-whistle_v{v}.parquet"),
           ("C11 OUT pin", s / "om2" / f"vigorous-whistle_v{v}.parquet", lf["oooi"]),
           ("C12 nbr_state", lf["oooi"], lf["nbr_state"]),
           ("C13 nbr_dark", lf["nbr_state"], lf["nbr_dark"]),
           ("C14 second pass", lf["nbr_dark"], lf["ss"]),
           ("C15 third pass", lf["ss"], lf["st"])]
    return seq


def cmd_read() -> None:
    import run_intruder_bound as R
    if not committed_unchanged(DECLARATION):
        raise ValueError(f"{DECLARATION} must be committed and unchanged before any reading.")
    decl = read_json(ROOT / DECLARATION)
    work = BASE / "work"
    late = read_json(work / "late" / "late_chain.json")["outputs"]
    for k in ["sp", "sp2", "km"]:
        if late[k]["sha256"] != late[{"sp": "st", "sp2": "sp", "km": "sp2"}[k]]["sha256"]:
            raise ValueError(f"{k} is not a pass-through on F.")
    dep = departures()
    lab = pd.concat([pd.read_parquet(p, columns=[ID, "PHASE_mvt", TARGET]) for p in sorted(glob.glob(str(ROOT / "data/raw/training_2025-*.parquet")))])
    lab = lab[lab["PHASE_mvt"].eq("DEP")].drop_duplicates(ID).set_index(ID)[TARGET].astype(float)
    y = lab.reindex(dep[ID]).to_numpy()
    if np.isnan(y).any():
        raise ValueError("A departure of F without a label.")
    lane_m = set(pd.concat([pd.read_parquet(MODEL_DIR / f"chain_{m}.parquet", columns=[ID]) for m in MONTHS])[ID])
    in_m = dep[ID].isin(lane_m).to_numpy()
    day = dep[TIME].dt.strftime("%Y-%m-%d").to_numpy()
    out = {"created_at_utc": utc_now(), "declaration": DECLARATION, "declaration_sha256": sha256_file(ROOT / DECLARATION), "rule": decl["rule"],
           "rows": {m: {"all": int(dep["month"].eq(m).sum()), "lane_m_read": int((dep["month"].eq(m).to_numpy() & in_m).sum())} for m in MONTHS}, "layers": {}}
    for name, fin, fout in layer_files(work):
        a = pd.read_parquet(fin).set_index(ID)[TARGET].reindex(dep[ID]).to_numpy(dtype=float)
        b = pd.read_parquet(fout).set_index(ID)[TARGET].reindex(dep[ID]).to_numpy(dtype=float)
        entry = {"input": {"file": rel(fin), "sha256": sha256_file(fin)}, "output": {"file": rel(fout), "sha256": sha256_file(fout)}, "months": {}}
        for pop, mask_all in [("governing_lane_m_rows", in_m), ("all_departures", np.ones(len(dep), dtype=bool)), ("placeholder_rows", ~in_m)]:
            rng = np.random.default_rng(READ_SEED)
            for m in MONTHS:
                sel = dep["month"].eq(m).to_numpy() & mask_all
                s = pd.DataFrame({"label": y[sel], "base": a[sel], "day": day[sel]})
                moved = int((a[sel] != b[sel]).sum())
                if not sel.any():
                    entry["months"].setdefault(m, {})[pop] = {"rows": 0}
                    continue
                pct, lo, hi = R.interval(s, "base", b[sel], rng)
                entry["months"].setdefault(m, {})[pop] = {"rows": int(sel.sum()), "change_pct": pct, "lo_s": lo, "hi_s": hi, "rows_moved": moved,
                                                         "rmse_in": float(np.sqrt(np.mean((y[sel] - a[sel]) ** 2))),
                                                         "rmse_out": float(np.sqrt(np.mean((y[sel] - b[sel]) ** 2))),
                                                         "squared_change": float(((a[sel] - b[sel]) ** 2).sum())}
        g = entry["months"]
        gov = "governing_lane_m_rows"
        for m in MONTHS:
            if g[m][gov]["rows_moved"] == 0:
                g[m][gov]["unreadable"] = "moves no row read"
        if name.startswith("C11"):
            g["2025-02"][gov]["unreadable"] = "no data-link archive asset before 14 June 2025 (Tier A)"
        entry["switch_off"] = {m: bool(g[m][gov]["change_pct"] > 0 and g[m][gov]["lo_s"] > 0 and not g[m][gov].get("unreadable")) for m in MONTHS}
        out["layers"][name] = entry
        print(name, " | ".join(f"{m} {g[m][gov]['change_pct']:+.4f}% [{g[m][gov]['lo_s']:+.3f}, {g[m][gov]['hi_s']:+.3f}] n{g[m][gov]['rows_moved']}"
                                  + (" UNREADABLE" if g[m][gov].get("unreadable") else "")
                                  + f" (all departures {g[m]['all_departures']['change_pct']:+.4f}%)" for m in MONTHS),
              "SWITCH OFF" if any(entry["switch_off"].values()) else "", flush=True)
    out["switched_off"] = {name: [m for m, v in e["switch_off"].items() if v] for name, e in out["layers"].items() if any(e["switch_off"].values())}
    write_json(ROOT / "docs/final_file/tierb_chain_results.json", out)
    print("switched off:", out["switched_off"] or "none")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["input", "own-member", "opdi-check", "root", "c1", "early", "repro12", "late", "read"])
    p.add_argument("--check", action="store_true", help="own-member: the January 2025 reproduction")
    p.add_argument("--standin", action="store_true", help="root / c1 / early / late: the rows12 code-path stand-in (never read)")
    p.add_argument("--start", default=None, help="late: resume at this step name")
    a = p.parse_args()
    if a.action == "input":
        build_input()
    elif a.action == "own-member":
        cmd_own_member(a.check)
    elif a.action == "opdi-check":
        cmd_opdi_check()
    elif a.action == "root":
        cmd_root(a.standin)
    elif a.action == "c1":
        cmd_c1(a.standin)
    elif a.action == "early":
        cmd_early(a.standin)
    elif a.action == "repro12":
        cmd_repro12()
    elif a.action == "late":
        cmd_late(a.standin, a.start)
    else:
        cmd_read()


if __name__ == "__main__":
    main()
