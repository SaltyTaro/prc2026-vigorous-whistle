"""Development round of the LIRF own-points member (configs/lirf_own_dev_v1.json): the masked LIRF member M (lm25_32, 16 seeds 801-816)
replaced on own-route rows (own ground motion seen: p_first_mv not null) by M' = (1 - s) M + s Mown, Mown = the 16-seed mean of
lm25_32own (seeds 801-816, the same folds), inside the shipped LIRF arithmetic (tools/build_dev_model_part.py):
  v53m  unmatched non-population rows (lirf_u), non-dark months: v34's unmatched blend with the u600 gate on 0.45 net + 0.55 M
  v54m  dark months (January 2025): no own-route row (asserted), unchanged
  v55m  matched non-population rows (lirf_m): v34's matched blend with the m600 gate on 0.60 net + 0.40 M
  --stage1        route-level readings (labels below 20,000 s, paired-day intervals) and whole-month shares; the stage-1 rule
                  -> runs/lirf_own/dev/selection.json, docs/lirf_own/results_v1.json
  --model-part C  the v66 development model part (runs/ctx_ownpoints_weight/chain/own_all_w100/dmp.parquet) with LIRF route rows of
                  v53m-v55m recomputed with M' -> runs/lirf_own/chain/<C>/dmp.parquet (--identity: M' = M must reproduce it exactly)
  --chain C       no-NM rematch -> v48 -> extreme-day -> variance re-allocation with the shipped parameters (tools/run_ctx_ownpoints_chain
                  step_chain's procedure; daily-totals runs under runs/lirf_own/chain/<C>/dt_<kind>)
  --score C       stack_resid_v3 refitted on the rebuilt components (LIRF: refitted v2), intruder bound, identity extension, own-motion
                  v1, own_motion_v2; control = the pinned identity rebuild of the v66 chain (the stored v66 development proxy
                  runs/ctx_ownpoints_weight/chain/own_all_w100/dev_v66_proxy.parquet reported beside it); v57's clauses as declared
                  -> docs/lirf_own/results_chain_v1.json
  --scorer-identity  the stage-2 scorer on v66's own chain files must reproduce the stored v66 development proxy exactly
No official score enters. Nothing is uploaded.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import object_hash, read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID  # noqa: E402
from prc2026_final import lirf_gate as lg  # noqa: E402
import run_lirf_combo as rlc  # noqa: E402
import run_lirf_member as rlm  # noqa: E402
from prc2026_scout import var_alloc as _VA  # noqa: E402

# The variance re-allocation's LightGBM fits leave LightGBM's row-/column-wise choice to a timing test, so a rerun of the shipped chain
# under load differs from the stored proxies by up to 0.0075% of a month (identity run, 6 October). This round pins the choice
# (deterministic, column-wise; settings otherwise unchanged) in every run of this tool, identity and candidate alike.
_VA.PARAMS = {**_VA.PARAMS, "deterministic": True, "force_col_wise": True}

DECL = "configs/lirf_own_dev_v1.json"
SHIPPED_MEMBER = "configs/adsb_ctx3_lirf_mask_dev_v1.json"
ROWS = "runs/lomo_dev/rows.parquet"
OWN = ROOT / "data/external/adsblol/features_v19own"
# The incumbent (coordinator, 6 October 08:20 UTC): v66 = v65's chain with the own-points member at weight 1.00 (docs/ctx_ownpoints_weight)
INC_DIR = ROOT / "runs/ctx_ownpoints_weight/chain/own_all_w100"
INC_PROXY = INC_DIR / "dev_v66_proxy.parquet"
INC, NEXT = "v66", "v67"
OUT = ROOT / "runs/lirf_own"
MONTHS = ["2025-01", "2025-07", "2025-11", "2025-12"]
FLOOR, CAP, DRAWS, SEED = -12.0, 20000.0, 10000, 20261006


def member_mean(variant_root: Path, variant: str, seeds: list[int], month: str, ids: np.ndarray) -> np.ndarray:
    fold, file = rlm.NATURAL[month]
    return np.mean([rlm.read(variant_root / f"{variant}_s{s}" / fold / file, ids) for s in seeds], axis=0)


def route_of(month: str, ids: np.ndarray) -> np.ndarray:
    own = pd.read_parquet(OWN / f"features_v19_{month}.parquet", columns=[ID, "p_first_mv"]).set_index(ID)["p_first_mv"]
    return own.reindex(ids).notna().to_numpy()


def lirf_sets(decl: dict | None) -> dict:
    """Per month: LIRF non-population rows with v9, net, tree, labels, days, matched flag, route flag, M and Mown (16-seed means).
    decl None (identity): Mown = M, no candidate file is read."""
    mask = read_json(ROOT / SHIPPED_MEMBER)["fits"]
    m_root = ROOT / mask["outputs"].split("<variant>")[0]
    if decl is not None:
        fits = decl["fits"]
        o_root = ROOT / fits["outputs"].split("<variant>")[0]
    rows = pd.read_parquet(ROOT / ROWS, columns=[ID, "month", "day", "airport", "label", "lirf_m", "lirf_u", "v9", "lirf_net", "lirf_tree", "v33"])
    month_sse = {m: float(((g["v33"] - g["label"]) ** 2).sum()) for m, g in rows.groupby("month")}
    rows = rows[rows["airport"].eq("LIRF") & (rows["lirf_m"] | rows["lirf_u"])]
    out = {}
    for month in MONTHS:
        d = rows[rows["month"].eq(month)]
        ids = d[ID].to_numpy()
        route = route_of(month, ids)
        s = {"ids": ids, "v9": d["v9"].to_numpy(float), "net": d["lirf_net"].to_numpy(float), "tree": d["lirf_tree"].to_numpy(float),
             "y": d["label"].to_numpy(float), "days": d["day"].astype(str).to_numpy(), "m": d["lirf_m"].to_numpy(bool), "u": d["lirf_u"].to_numpy(bool),
             "route": route, "month_sse_v33": month_sse[month]}
        s["M"] = member_mean(m_root, mask["variants"][0], mask["seeds"], month, ids)
        if route.any() and decl is not None:
            s["Mown"] = member_mean(o_root, fits["variants"][0], fits["seeds"], month, ids)
        else:
            s["Mown"] = s["M"].copy()  # identity, or no own-route row: the candidate equals the control by construction
        out[month] = s
    return out


def member_prime(s: dict, share: float) -> np.ndarray:
    return np.where(s["route"], (1 - share) * s["M"] + share * s["Mown"], s["M"])


def route_values(s: dict, member: np.ndarray, dark: bool) -> np.ndarray:
    """The LIRF route's value with member M (shipped arithmetic of v53m / v55m; dark months keep v54m, asserted route-free)."""
    v = np.full(len(s["y"]), np.nan)
    m, u = s["m"], s["u"]
    if m.any():
        v[m] = rlc.values(s["v9"][m], s["net"][m], s["tree"][m], member[m], 0.40)[0]
    if u.any():
        if dark:
            if s["route"][u].any():
                raise ValueError("A dark month has own-route rows.")
            v[u] = np.nan
        else:
            v[u] = lg.gated(s["v9"][u], 0.45 * s["net"][u] + 0.55 * member[u], s["tree"][u], "unmatched", rlm.GATE)[0]
    return v


def dark_month(s: dict) -> bool:
    return s["m"].sum() / max(s["m"].sum() + s["u"].sum(), 1) < 0.05


def stage1(decl: dict) -> None:
    sets = lirf_sets(decl)
    results, standalone = {}, {}
    for month, s in sets.items():
        keep = s["y"] < CAP
        r = s["route"] & keep
        standalone[month] = {"route_rows": int(s["route"].sum()),
                             "M_route_rmse": float(np.sqrt(np.mean((s["M"][r] - s["y"][r]) ** 2))) if r.any() else None,
                             "Mown_route_rmse": float(np.sqrt(np.mean((s["Mown"][r] - s["y"][r]) ** 2))) if r.any() else None}
    for name, cand in decl["candidates"].items():
        share = float(cand["own_share"])
        res = {}
        for month, s in sets.items():
            dark = dark_month(s)
            base = route_values(s, s["M"], dark)
            vals = route_values(s, member_prime(s, share), dark)
            live = np.isfinite(base)
            y, days = s["y"][live], s["days"][live]
            base, vals = base[live], vals[live]
            out = {"rows": int(live.sum()), "route_rows": int((s["route"] & live).sum()), "dark": dark,
                   "rows_changed": int((np.abs(vals - base) > 1e-9).sum()),
                   "route_sse_change_share_of_month_v33": float((((vals - y) ** 2).sum() - ((base - y) ** 2).sum()) / s["month_sse_v33"])}
            for metric, keep in (("primary", y < CAP), ("unfiltered", np.ones(len(y), dtype=bool))):
                e0, e1 = (base - y) ** 2, (vals - y) ** 2
                total = float(e0[keep].sum())
                ud = np.unique(days)
                by_day = np.array([float(e1[keep & (days == d)].sum() - e0[keep & (days == d)].sum()) for d in ud])
                lo, hi = lg.paired_day_interval(by_day, total, DRAWS, SEED) if total > 0 and by_day.any() else (0.0, 0.0)
                out[metric] = {"rows_kept": int(keep.sum()), "rmse_control": float(np.sqrt(e0[keep].mean())), "rmse": float(np.sqrt(e1[keep].mean())),
                               "relative_change": float((e1[keep].sum() - total) / total), "interval_95": [lo, hi],
                               "days_better": int((by_day < 0).sum()), "days_worse": int((by_day > 0).sum()), "days": int(len(by_day))}
            res[month] = out
        results[name] = res
    rule = decl["stage_1"]["rule_values"]
    eligible, score = [], {}
    for name in decl["candidates"]:
        r = results[name]
        dec, jul, nov = (r[m]["primary"] for m in ("2025-12", "2025-07", "2025-11"))
        ok = dec["relative_change"] <= rule["december_max_relative_change"] and dec["interval_95"][1] < 0
        ok &= jul["relative_change"] <= 0 and nov["relative_change"] <= 0
        ok &= all(not (v["primary"]["interval_95"][0] > 0) for v in r.values())
        score[name] = float(np.mean([dec["relative_change"], jul["relative_change"], nov["relative_change"]]))
        if ok:
            eligible.append(name)
        print(name, "eligible" if ok else "-", {m: (round(100 * v["primary"]["relative_change"], 3), [round(100 * x, 3) for x in v["primary"]["interval_95"]],
                                                    v["route_rows"], round(100 * v["route_sse_change_share_of_month_v33"], 4)) for m, v in r.items()}, flush=True)
    selected = min(eligible, key=lambda n: score[n]) if eligible else None
    report = {"created_at_utc": utc_now(), "declaration": {DECL: {"sha256": sha256_file(ROOT / DECL), "object_hash": object_hash(decl)}},
              "code": {q: sha256_file(ROOT / q) for q in ["tools/run_lirf_own_dev.py", "tools/run_lirf_combo.py", "tools/run_lirf_member.py", "src/prc2026_final/lirf_gate.py"]},
              "inputs": {ROWS: sha256_file(ROOT / ROWS)}, "cap_s": CAP, "bootstrap": {"draws": DRAWS, "seed": SEED},
              "standalone_primary_route_rows": standalone, "results": results, "scores": score, "eligible": eligible, "selected_candidate": selected,
              "rule": decl["stage_1"]["rule"]}
    (OUT / "dev").mkdir(parents=True, exist_ok=True)
    write_json(OUT / "dev/selection.json", {"created_at_utc": utc_now(), "selected_candidate": selected, "eligible": eligible, "scores": score})
    dest = ROOT / "docs/lirf_own/results_v1.json"
    if dest.exists():
        raise ValueError("Preserve the original stage-1 result.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_json(dest, report)
    print("standalone:", standalone)
    print("selected:", selected, "eligible:", eligible)


def step_model_part(decl: dict, candidate: str, identity: bool) -> None:
    """dmp' = the incumbent (v66) development model part with LIRF route rows of v53m / v54m / v55m recomputed with M' (builder's delta arithmetic)."""
    sets = lirf_sets(None if identity else decl)
    share = 0.0 if identity else float(decl["candidates"][candidate]["own_share"])
    dmp = pd.read_parquet(INC_DIR / "dmp.parquet")
    if sha256_file(INC_DIR / "dmp.parquet") != read_json(INC_DIR / "dmp.json")["sha256"]:
        raise ValueError("The incumbent development model part changed.")
    shipped = pd.read_parquet(ROOT / "runs/dev_model_part/rows.parquet", columns=[ID, "v52m", "v53m", "v54m", "v55m"]).set_index(ID)
    out = dmp.copy()
    idx = pd.Series(np.arange(len(dmp)), index=dmp[ID].to_numpy())
    checks = {}
    for month, s in sets.items():
        dark = dark_month(s)
        pos = idx.reindex(s["ids"]).to_numpy()
        sh = shipped.reindex(s["ids"])
        # the incumbent model part's LIRF rows equal the shipped development model part's
        for c in ["v52m", "v53m", "v54m", "v55m"]:
            if np.abs(dmp[c].to_numpy()[pos] - sh[c].to_numpy()).max() > 1e-9:
                raise ValueError(f"{month}: the incumbent's {c} differs from the shipped model part on LIRF rows.")
        m, u = s["m"], s["u"]
        v52 = dmp["v52m"].to_numpy()[pos]
        v53, v54, v55 = (dmp[c].to_numpy()[pos].copy() for c in ("v53m", "v54m", "v55m"))
        mp = member_prime(s, share)
        if u.any() and not dark:
            ctrl_u, _ = lg.gated(s["v9"][u], s["net"][u], s["tree"][u], "unmatched", rlm.GATE)
            cand_u, _ = lg.gated(s["v9"][u], 0.45 * s["net"][u] + 0.55 * mp[u], s["tree"][u], "unmatched", rlm.GATE)
            v53[u] = np.maximum(v52[u] + (cand_u - ctrl_u), FLOOR)
            v54[u] = v53[u]
            v55[u] = v54[u]
        elif u.any() and s["route"][u].any():
            raise ValueError("A dark month has own-route rows.")
        if m.any():
            ctrl_m, _ = rlc.values(s["v9"][m], s["net"][m], s["tree"][m], None, 0.0)
            cand_m, _ = rlc.values(s["v9"][m], s["net"][m], s["tree"][m], mp[m], 0.40)
            v55[m] = np.maximum(v54[m] + (cand_m - ctrl_m), FLOOR)
        diff = {c: float(np.abs(new - dmp[c].to_numpy()[pos]).max()) for c, new in (("v53m", v53), ("v54m", v54), ("v55m", v55))}
        if identity and max(diff.values()) > 1e-9:
            raise ValueError(f"identity failed on {month}: {diff}")
        for c, new in (("v53m", v53), ("v54m", v54), ("v55m", v55)):
            col = out[c].to_numpy().copy()
            col[pos] = new
            out[c] = col
        y = dmp["label"].to_numpy() if "label" in dmp else None
        checks[month] = {"route_rows": int(s["route"].sum()), "rows_changed_v55m": int((np.abs(v55 - dmp["v55m"].to_numpy()[pos]) > 1e-9).sum()), "max_abs_diff": diff}
        print(month, checks[month], flush=True)
    if identity:
        print("identity: M' = M reproduces the incumbent development model part exactly on every LIRF route row")
        return
    d = OUT / "chain" / candidate
    d.mkdir(parents=True, exist_ok=True)
    out.to_parquet(d / "dmp.parquet", index=False)
    write_json(d / "dmp.json", {"created_at_utc": utc_now(), "candidate": candidate, "own_share": share, "sha256": sha256_file(d / "dmp.parquet"),
                                "parent": {"path": str((INC_DIR / "dmp.parquet").relative_to(ROOT)), "sha256": sha256_file(INC_DIR / "dmp.parquet")},
                                "checks": checks, "code": sha256_file(ROOT / "tools/run_lirf_own_dev.py")})


def step_chain(candidate: str) -> None:
    """tools/run_ctx_ownpoints_chain.step_chain's procedure on runs/lirf_own/chain/<candidate>/dmp.parquet; daily totals in the candidate's dir."""
    import run_daily_totals as rd
    import run_rematch_nonm as RM
    import scout_extreme_shift as se
    import scout_var_alloc as sva
    from scout_refit_prep import takeoff, with_stack
    from screen_daily_totals import published
    d = OUT / "chain" / candidate
    dmp = pd.read_parquet(d / "dmp.parquet").drop_duplicates(ID).set_index(ID)
    m25 = pd.read_parquet(RM.OUT / "matches_2025.parquet")
    lab = RM.labels_2025()
    base = pd.read_parquet(ROOT / "runs/daily_totals/rows.parquet")
    ids = base[ID].to_numpy()
    v = dmp["v55m"].reindex(ids).to_numpy(dtype=float)
    v = np.where(np.isnan(v), base["stack"].to_numpy(dtype=float), v)
    offs = read_json(RM.OUT / "dev/offsets.json")
    cand, changed = v.copy(), np.zeros(len(v), dtype=bool)
    for m in RM.MONTHS:
        idx = np.flatnonzero(base["month"].eq(m).to_numpy())
        if RM.offsets(m25, lab, m) != offs[m]:
            raise ValueError("The rematch offsets differ from the shipped development run.")
        c, ch = RM.apply(v[idx], ids[idx], m25, offs[m])
        cand[idx], changed[idx] = c, ch
    tk, pub = takeoff(ids), published()
    with_stack(base, v, tk, pub).to_parquet(d / "rows_all.parquet", index=False)
    settings, tau_scale, prior = rd.settings_from(read_json(ROOT / "configs/daily_totals_v1.json"))
    rel_dir = Path("..") / "lirf_own" / "chain" / candidate
    for kind, col in [("ctl", v), ("cand", cand)]:
        new = with_stack(base, col, tk, pub)
        parts = []
        for m in RM.MONTHS:
            mix = pd.concat([base[base["month"] != m], new[new["month"] == m]], ignore_index=True)
            rows_path = d / f"rows_{kind}_{m}.parquet"
            mix.to_parquet(rows_path, index=False)
            tag = (rel_dir / f"dt_{kind}").as_posix()  # -> runs/lirf_own/chain/<candidate>/dt_<kind>/<month>.parquet
            if not (d / f"dt_{kind}" / f"{m}.parquet").exists():
                rd.run_month(m, tag, tau_scale, settings, rows=str(rows_path.relative_to(ROOT)).replace("\\", "/"), prior=prior)
            vd = d / f"v48_{kind}_{m}"
            vd.mkdir(parents=True, exist_ok=True)
            for kk in RM.MONTHS:
                src = (d / f"dt_{kind}" / f"{kk}.parquet") if kk == m else ROOT / f"runs/daily_totals/daily_totals_v1/{kk}.parquet"
                shutil.copyfile(src, vd / f"{kk}.parquet")
            se.TABLES[f"lirfown_{candidate}_{kind}_{m}"] = (rows_path, vd)
            t, ks = sva.layered(f"lirfown_{candidate}_{kind}_{m}")
            parts.append(t.loc[t["month"] == m, [ID, "month", "airport", "bday", "label", "stack", "v48", "ext", "new"]].assign(k=ks[m]))
            print(candidate, kind, m, "k", ks[m], flush=True)
        pd.concat(parts, ignore_index=True).to_parquet(d / f"{kind}.parquet", index=False)
    write_json(d / "chain.json", {"created_at_utc": utc_now(), "rematch_rows": int(changed.sum()),
                                  "files": {p.name: sha256_file(p) for p in sorted(d.glob("*.parquet"))}, "code": sha256_file(ROOT / "tools/run_lirf_own_dev.py")})


def step_score(decl: dict | None, candidate: str, mode: str) -> None:
    """tools/run_ctx_ownpoints_chain.step_score's arithmetic. mode: "candidate" (this candidate's rebuilt files; control = the pinned
    identity rebuild, the stored incumbent proxy reported beside it), "identity" (a pinned M' = M rebuild, compared with the stored proxy)
    or "scorer" (the incumbent's own chain files: must reproduce the stored incumbent proxy exactly)."""
    import run_ctx_ownpoints_chain as CH
    import run_stack_resid as S
    import run_stack_resid_v3 as V3
    import run_own_motion_v2 as O2
    import run_intruder_bound as R
    d = OUT / "chain" / candidate
    d.mkdir(parents=True, exist_ok=True)
    src = INC_DIR if mode == "scorer" else d
    if decl is not None:
        proxy = decl["stage_2"]["stored_incumbent_proxy"]
    else:
        proxy = {"path": INC_PROXY.relative_to(ROOT).as_posix(), "sha256": sha256_file(INC_PROXY)}
    if sha256_file(ROOT / proxy["path"]) != proxy["sha256"]:
        raise ValueError("The stored incumbent development proxy differs from the declared file.")
    files = (src / "ctl.parquet", src / "rows_all.parquet", src / "dmp.parquet", src / "cand.parquet")
    dev = CH.dev_table_from(*files[:3])
    X46, res = S.dev_matrix(dev)
    X46 = X46.reset_index(drop=True)
    core = V3.own_points("dev", dev[ID].to_numpy())
    Pn = V3.new_columns(core, dev["new"].to_numpy(dtype=float))
    X65 = pd.concat([X46, Pn], axis=1)
    pr = V3.dev_predictions(dev, X46, X65, Pn, res)
    new, air = dev["new"].to_numpy(dtype=float), dev["airport"].to_numpy()
    s1 = pd.read_parquet(files[3]).set_index(ID)["new"].reindex(dev[ID]).to_numpy(dtype=float)
    keys, _ = O2.O.dev_table()
    keys = keys.set_index(ID).reindex(dev[ID]).reset_index()
    if not np.allclose(keys["label"].to_numpy(), dev["label"].to_numpy()):
        raise ValueError("Row alignment with the v61 proxy failed.")
    t = keys[[ID, "month", "airport", "label", "cls", "day"]].copy()
    t["s2"] = s1 + (np.where(air == "LIRF", pr["base2"], pr["base3"]) - new)
    _, cand63 = V3.Chain().run(t, "s2")
    dev12, t12, _ = O2.v62_tables()
    trig, _ = O2.triggers_2025(O2.IDS_DIR, use_cache=True)
    t["c63"] = cand63
    cand = CH.own_motion_v2(t, "c63", t12, trig)
    stored_proxy = pd.read_parquet(ROOT / proxy["path"]).set_index(ID).reindex(t[ID])
    t[INC] = stored_proxy[INC].to_numpy()
    if mode in ("identity", "scorer"):
        diff = cand - t[INC].to_numpy()
        stats = {"mode": mode, "files": {p.name: sha256_file(p) for p in files}, f"max_abs_diff_vs_{INC}_proxy_s": float(np.abs(diff).max()),
                 "rms_diff_s": float(np.sqrt(np.mean(diff ** 2))), "rows_differ": int((np.abs(diff) > 1e-6).sum()), "rows": int(len(t)), "months": {}}
        for m in MONTHS:
            k = t["month"].to_numpy() == m
            yy = t["label"].to_numpy()[k]
            r0, r1 = float(np.sqrt(np.mean((yy - t[INC].to_numpy()[k]) ** 2))), float(np.sqrt(np.mean((yy - cand[k]) ** 2)))
            stats["months"][m] = {"rows_differ": int((np.abs(diff[k]) > 1e-6).sum()), "rms_diff_s": float(np.sqrt(np.mean(diff[k] ** 2))),
                                  f"rmse_{INC}_proxy": r0, "rmse_rebuilt": r1, "pct": 100 * (r1 / r0 - 1)}
        print(f"{mode} against the stored {INC} development proxy:", stats, flush=True)
        t.assign(rebuilt=cand)[[ID, "month", "airport", "label", INC, "rebuilt"]].to_parquet(d / "identity_rebuilt.parquet", index=False)
        write_json(d / "identity_score.json", {"created_at_utc": utc_now(), **stats})
        if mode == "scorer" and stats[f"max_abs_diff_vs_{INC}_proxy_s"] > 1e-6:
            raise ValueError(f"The stage-2 scorer does not reproduce the stored {INC} proxy from its own files.")
        return
    rule = decl["stage_2"]["rule_values"]
    ident = decl["stage_2"]["identity_file"]
    if sha256_file(ROOT / ident["path"]) != ident["sha256"]:
        raise ValueError("The pinned identity rebuild differs from the declared file.")
    ctl = pd.read_parquet(ROOT / ident["path"]).set_index(ID).reindex(t[ID])
    t["ctl"] = ctl["rebuilt"].to_numpy()
    rng = np.random.default_rng(20261006)
    t["cand"] = cand
    months, stored = {}, {}
    for m in R.DEV_MONTHS:
        s = t[t["month"] == m]
        dch = s["cand"].to_numpy() - s["ctl"].to_numpy()
        changed = int((np.abs(dch) > 1e-9).sum())
        if changed:
            pct, lo, hi = R.interval(s, "ctl", s["cand"].to_numpy(), rng)
        else:
            pct, lo, hi = 0.0, 0.0, 0.0
        yy = s["label"].to_numpy()
        months[m] = {"control_rmse": float(np.sqrt(np.mean((yy - s["ctl"].to_numpy()) ** 2))), "rmse": float(np.sqrt(np.mean((yy - s["cand"].to_numpy()) ** 2))),
                     "pct": pct, "lo": lo, "hi": hi, "rows_changed": changed, "lirf_rows_changed": int(((np.abs(dch) > 1e-9) & s["airport"].eq("LIRF").to_numpy()).sum()),
                     "squared_change": float(np.sum(dch ** 2))}
        r_inc = float(np.sqrt(np.mean((yy - s[INC].to_numpy()) ** 2)))
        stored[m] = {f"{INC}_proxy_rmse": r_inc, f"pct_vs_{INC}_proxy": 100 * (months[m]["rmse"] / r_inc - 1)}
        print(m, {kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in months[m].items()}, stored[m], flush=True)
    eq = float(np.mean([months[m]["pct"] for m in rule["equal_footing"]]))
    live = [m for m in rule["equal_footing"] if months[m]["rows_changed"] > 0]
    checks = {"1_no_month_worse": all(months[m]["pct"] <= rule["max_month_worsening_pct"] for m in R.DEV_MONTHS),
              "2_equal_footing_mean_pct": eq, "2_mean_ok": eq <= -rule["min_mean_improvement_pct"],
              "3_months_with_changed_rows": live,
              "3_every_equal_interval_below_zero_where_rows_change": bool(live) and all(months[m]["hi"] < 0 for m in live)}
    checks["accepted"] = bool(checks["1_no_month_worse"] and checks["2_mean_ok"] and checks["3_every_equal_interval_below_zero_where_rows_change"])
    t[[ID, "month", "airport", "label", "cls", INC, "ctl", "cand"]].rename(columns={"cand": NEXT}).to_parquet(d / f"dev_{NEXT}_proxy.parquet", index=False)
    report = {"created_at_utc": utc_now(), "declaration": DECL, "declaration_sha256": sha256_file(ROOT / DECL), "candidate": candidate,
              "rule": decl["stage_2"]["rule"], "control": ident, f"stored_{INC}_proxy": {**proxy, "months": stored}, "inputs": {p.name: sha256_file(p) for p in files},
              "months": months, "checks": checks, "stacker_trees": pr["trees"],
              "proxy": {"path": (d / f"dev_{NEXT}_proxy.parquet").relative_to(ROOT).as_posix(), "sha256": sha256_file(d / f"dev_{NEXT}_proxy.parquet")},
              "code": {c: sha256_file(ROOT / c) for c in ["tools/run_lirf_own_dev.py", "tools/run_ctx_ownpoints_chain.py", "tools/run_stack_resid_v3.py", "tools/run_own_motion_v2.py"]}}
    dest = ROOT / "docs/lirf_own/results_chain_v1.json"
    if dest.exists():
        raise ValueError("Preserve the original stage-2 result.")
    write_json(dest, report)
    print(checks)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage1", action="store_true")
    ap.add_argument("--candidate", default=None)
    ap.add_argument("--model-part", action="store_true")
    ap.add_argument("--chain", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--identity", action="store_true", help="M' = M through the whole pipeline (candidate directory 'identity')")
    ap.add_argument("--tag", default="identity_pinned_v66", help="chain directory of an identity run")
    ap.add_argument("--scorer-identity", action="store_true")
    a = ap.parse_args()
    decl = read_json(ROOT / DECL) if (ROOT / DECL).exists() else None
    if not (a.identity or a.scorer_identity) and (decl is None or not decl.get("declared_at_utc")):
        raise ValueError("The declaration is not recorded.")
    if decl is not None and read_json(ROOT / decl["shipped_member"]["declaration"]) != read_json(ROOT / SHIPPED_MEMBER):
        raise ValueError("The shipped member's declaration differs.")
    if a.stage1:
        stage1(decl)
        return
    if a.scorer_identity:
        step_score(None, "scorer_identity_v66", "scorer")
        return
    cand = a.tag if a.identity else a.candidate
    if not a.identity:
        sel = read_json(OUT / "dev/selection.json")
        if sel.get("selected_candidate") != cand:
            raise ValueError(f"Stage 2 runs the stage-1 selection only ({sel.get('selected_candidate')}).")
    if a.model_part:
        if a.identity:
            step_model_part(None, cand, True)
            # the identity directory receives the incumbent's model part unchanged
            d = OUT / "chain" / cand
            d.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(INC_DIR / "dmp.parquet", d / "dmp.parquet")
        else:
            step_model_part(decl, cand, False)
    if a.chain:
        step_chain(cand)
    if a.score:
        step_score(None if a.identity else decl, cand, "identity" if a.identity else "candidate")


if __name__ == "__main__":
    main()
