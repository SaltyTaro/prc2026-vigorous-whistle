"""Governing run of configs/stack_queue_v1.json (docs/stack_queue/README.md): seven label-free runway take-off-sequence columns added to
each learner of the guarded stacking residual (step 6 of docs/final_file/v69_chain.md), candidate q65_bag10 read in v70's development chain
against v70's development proxy (runs/oooi_lhr/dev_v70_proxy.parquet).

Reuses tools/run_stack_weight.py's functions unchanged (stack_weight_v1's governing run); the only switch is QUEUE (off / on), which adds the
queue columns to the three learners' feature matrices (v2 46 + 7 = 53, v3 65 + 7 = 72, OPDI 79 + 7 = 86). Guards, LightGBM settings,
month-blocking and the ten seeds 20261003, 11-19 are run_stack_weight's.

  --dev-month M --queue off|on   month-blocked out-of-month predictions (ten seeds x v2 / v3 / OPDI) on development month M
                                 -> runs/stack_queue/dev/<queue>/preds_<M>.parquet
  --dev-score                    identity first (queue off, w 0.65, ten seeds, steps 7-10, step 11 must equal the control's value column: max
                                 |diff| 0 or the 4.5e-13 s Linux bound; else stop, nothing read), then q65_bag10 (queue on) through the same
                                 chain; clauses 1-3 -> docs/stack_queue/results_v1.json
  --rank-fit --queue off|on      the three learners on all four development months, ten seeds, applied to v66's shipped step-5 ranking table
                                 (run_stack_weight.rank_inputs) -> runs/stack_queue/rank/<queue>/preds.parquet
  --rank-identity                queue off: step 6 at w 0.65 with ten seeds, chain_from_stack, own_motion_v2, step 11 must reproduce the uploaded
                                 v70 (f1d17747...) byte for byte; else no candidate ranking file
  --rank-check                   clause 4 (transfer guard) for the queue-on ten-seed bags; acceptance -> results
  --rank-dry                     only if accepted: q65_bag10 through steps 6-11 -> runs/stack_queue/rank/dry_v71/ + report against v70

Queue columns (function queue_columns; declaration queue_columns): per airport and runway, departures sorted by take-off within the row's
month file (development: the month's data/raw/training_*.parquet; ranking: data/raw/ranking.parquet as one file).
No official score enters. Nothing is uploaded; nothing is written under runs/submissions.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET  # noqa: E402
import run_stack_weight as SW  # noqa: E402

DECL = "configs/stack_queue_v1.json"
OUT = ROOT / "runs/stack_queue"
DOCS = ROOT / "docs/stack_queue"
RESULTS = DOCS / "results_v1.json"
CONTROL = ("runs/oooi_lhr/dev_v70_proxy.parquet", "51aaa6a62f64897290199f6c6c9e7d3d43245a3862f36b2bdbc35be8012eabbe")
V69_PROXY = ("runs/stack_weight/dev/w65_bag10/proxy.parquet", "c088e409de153d81b006bcd26b5b2a28e0e306043a5172c2158d0ecc6e6f5cb8")
V69 = "192c4a30c76ae1967a4d7594eeca4704554bdc7535f337f5022ef77eaea17665"
V70 = "f1d17747e0e384ccec520a760d42c8e55a3e88a1476daafacdf7fe1f9603099f"
STACK_WEIGHT_RANK_PREDS = ("runs/stack_weight/rank/preds.parquet", "8083ffcfdf861e278ac8693fb2320872cd95378f3f9ac861decf23c9a6163772")
LINUX_BOUND = 4.5e-13
WEIGHT, K = 0.65, 10
CANDIDATE = "q65_bag10"
QCOLS = ["gp", "gn", "gp2", "gp4", "gn4", "bpos", "blen"]
DEV_FILES = {m: f"data/raw/training_{m}-01_*.parquet" for m in SW.MONTHS}
RANKING = "data/raw/ranking.parquet"
CODE = ["tools/run_stack_queue.py"] + SW.CODE + ["tools/run_oooi_pin.py", "tools/run_oooi_lhr.py"]


# ---------------------------------------------------------------- queue columns (declaration: queue_columns)

def queue_columns(path: Path) -> pd.DataFrame:
    """The coordinator's exploration code (runs/quick_qburst_build.py, build) on ONE month file: departures per airport and runway sorted
    by take-off; gaps to the previous / next / second / fourth previous / fourth next take-off, position and length of the burst (a run of
    take-offs with consecutive gaps <= 150 s)."""
    raw = pd.read_parquet(path, columns=[ID, "PHASE_mvt", "ADEP_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt"])
    raw = raw[raw.PHASE_mvt == "DEP"].copy()
    raw["t"] = (raw.MVT_TIME_UTC_mvt - pd.Timestamp("2025-01-01", tz="UTC")).dt.total_seconds()
    raw = raw.sort_values(["ADEP_mvt", "RUNWAY_mvt", "t"])
    g = raw.groupby(["ADEP_mvt", "RUNWAY_mvt"]).t
    raw["gp"] = raw.t - g.shift(1)
    raw["gn"] = g.shift(-1) - raw.t
    raw["gp2"] = raw.t - g.shift(2)
    raw["gp4"] = raw.t - g.shift(4)
    raw["gn4"] = g.shift(-4) - raw.t
    b = (raw.gp > 150) | raw.gp.isna()
    raw["bid"] = b.cumsum()
    raw["bpos"] = raw.groupby("bid").cumcount()
    raw["blen"] = raw.groupby("bid").t.transform("size")
    if raw[ID].duplicated().any():
        raise ValueError(f"{path}: duplicate departure IDs.")
    return raw[[ID] + QCOLS].reset_index(drop=True)


def queue_dev(dev: pd.DataFrame) -> pd.DataFrame:
    """Queue columns of the development rows, each from its own month's training file."""
    parts = []
    for m, pat in DEV_FILES.items():
        files = sorted(ROOT.glob(pat))
        if len(files) != 1:
            raise ValueError(f"{pat}: {len(files)} files.")
        parts.append(queue_columns(files[0]).assign(qmonth=m))
    q = pd.concat(parts, ignore_index=True)
    if q[ID].duplicated().any():
        raise ValueError("A departure appears in two month files.")
    q = q.set_index(ID).reindex(dev[ID].to_numpy())
    if q["qmonth"].isna().any() or not np.array_equal(q["qmonth"].to_numpy(), dev["month"].to_numpy()):
        raise ValueError("A development row is not a departure of its own month's file.")
    return q[QCOLS].astype(np.float32).reset_index(drop=True)


def queue_rank(rk: pd.DataFrame) -> pd.DataFrame:
    q = queue_columns(ROOT / RANKING).set_index(ID).reindex(rk[ID].to_numpy())
    if q["bpos"].isna().any():
        raise ValueError("A ranking row is not a departure of data/raw/ranking.parquet.")
    return q[QCOLS].astype(np.float32).reset_index(drop=True)


def with_queue(X: pd.DataFrame, Q: pd.DataFrame) -> pd.DataFrame:
    if len(X) != len(Q):
        raise ValueError("Row counts differ.")
    return pd.concat([X.reset_index(drop=True), Q], axis=1)


# ---------------------------------------------------------------- development

def dev_month(m: str, queue: str) -> None:
    """run_stack_weight.dev_month with the queue switch (guards on the unchanged X46 / Pn / Pg, as declared)."""
    import run_stack_resid as S
    import run_stack_resid_v3 as V3
    path = OUT / "dev" / queue / f"preds_{m}.parquet"
    if path.exists():
        raise ValueError(f"{path} exists.")
    dev, X46, X65, X79, Pn, Pg, Po, res = SW.dev_inputs()
    mats = {"p2": X46, "p3": X65, "po": X79}
    if queue == "on":
        Q = queue_dev(dev)
        mats = {k: with_queue(X, Q) for k, X in mats.items()}
    fx = S.fixed_rows(dev)
    month, air, dom = dev["month"].to_numpy(), dev["airport"].to_numpy(), dev["dom"].to_numpy()
    fit, ev = (month != m) & ~fx, month == m
    h2 = S.outside_support(X46[fit], air[fit], X46[ev].reset_index(drop=True), air[ev]) & ~fx[ev]
    hn3 = V3.guard_new(Pn[fit].reset_index(drop=True), air[fit], Pn[ev].reset_index(drop=True), air[ev]) & ~fx[ev]
    hno = V3.guard_new(Pg[fit].reset_index(drop=True), air[fit], Pg[ev].reset_index(drop=True), air[ev]) & ~fx[ev]
    out = pd.DataFrame({ID: dev[ID].to_numpy()[ev], "fx": fx[ev], "h2": h2, "h3": h2 | hn3, "ho": h2 | hno,
                        "o_has": Po["o_has"].to_numpy()[ev]})
    trees, ncol = {}, {}
    for key in ["p2", "p3", "po"]:
        Xa = mats[key].to_numpy(np.float32)
        ncol[key] = int(Xa.shape[1])
        for i, seed in enumerate(SW.SEEDS10):
            S.SEEDS = [seed]
            models = S.fit_bag(Xa, res, fit, dom)
            out[f"{key}_{i}"] = models[0].predict(Xa[ev], num_iteration=models[0].best_iteration)
            trees[f"{key} {seed}"] = int(models[0].best_iteration)
            print(m, queue, key, ncol[key], seed, "trees", trees[f"{key} {seed}"], flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    write_json(path.with_suffix(".json"), {"created_at_utc": utc_now(), "month": m, "queue": queue, "columns": ncol, "rows": int(ev.sum()),
                                           "trees": trees, "sha256": sha256_file(path), "code": {p: sha256_file(ROOT / p) for p in CODE}})


def load_preds(queue: str, dev: pd.DataFrame) -> pd.DataFrame:
    import run_stack_resid as S
    P = pd.concat([pd.read_parquet(OUT / "dev" / queue / f"preds_{m}.parquet") for m in SW.MONTHS], ignore_index=True)
    P = P.set_index(ID).reindex(dev[ID]).reset_index()
    if P["p2_0"].isna().any() or not np.array_equal(P["fx"].to_numpy(dtype=bool), S.fixed_rows(dev)):
        raise ValueError("The stored predictions do not cover the development rows.")
    return P


def step11_dev_pins() -> pd.DataFrame:
    """Step 11's development pins exactly as tools/run_oooi_lhr.v70_dev_proxy (run_oooi_pin's development part): on the v69 proxy's rows,
    month-blocked offsets over July / November / December 2025 (labels and take-off - OUT only; independent of the chain's value)."""
    import run_oooi_pin as P
    import run_stack_resid as S
    if sha256_file(ROOT / V69_PROXY[0]) != V69_PROXY[1] or P.PROXY.resolve() != (ROOT / V69_PROXY[0]).resolve():
        raise ValueError("The v69 development proxy differs.")
    pin_res = read_json(ROOT / "docs/oooi_pin/results_v1.json")
    if sha256_file(P.RUN / "joined_dev.parquet") != pin_res["joined_sha256"]["dev"]:
        raise ValueError("runs/oooi_pin/joined_dev.parquet changed since oooi_pin_v1's governing run.")
    pin_dev = P.scope(pd.read_parquet(P.RUN / "joined_dev.parquet"))
    dev = pd.read_parquet(P.PROXY, columns=[ID, "month", "airport", "label", "cls", "day", "value"])
    dev = dev.merge(P.takeoff_days(P.DEV_MONTHS), on=ID, how="left", validate="one_to_one")
    dev["day"] = dev["tday"]
    dev = dev.merge(pin_dev[[ID, "source", "raw"]], on=ID, how="left", validate="one_to_one")
    fixed = S.fixed_rows(dev)
    dev["in_scope"] = dev["raw"].notna().to_numpy() & ~fixed & dev["airport"].eq("LFPG").to_numpy()
    dev["pin"] = np.nan
    for m in P.PIN_DEV_MONTHS:
        off = P.offsets(dev[dev["in_scope"] & dev["month"].ne(m) & dev["month"].isin(P.PIN_DEV_MONTHS)])
        sel = dev["in_scope"] & dev["month"].eq(m)
        dev.loc[sel, "pin"] = dev.loc[sel, "raw"] + dev.loc[sel, "source"].map(off)
    return dev[[ID, "day", "in_scope", "pin"]]


def step11_dev(ids: np.ndarray, value: np.ndarray, pins: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """oooi_pin_v1 w50 on the in-scope LFPG rows: value = max(0.5 x pin + 0.5 x chain, -12)."""
    import run_oooi_pin as P
    p = pins.set_index(ID).reindex(ids)
    if p["day"].isna().any():
        raise ValueError("The pin table does not cover the rows.")
    sc = p["in_scope"].to_numpy(dtype=bool)
    v = value.astype(float).copy()
    v[sc] = P.cand_value(0.5, p["pin"].to_numpy()[sc], v[sc])
    return v, p["day"].to_numpy()


def dev_score() -> None:
    import run_ctx_ownpoints_chain as CH
    import run_intruder_bound as R
    import run_opdi_stack as O
    import run_own_motion_v2 as O2
    import run_stack_resid as S
    import run_stack_resid_v3 as V3
    if RESULTS.exists():
        raise ValueError("Preserve the reading.")
    if sha256_file(ROOT / CONTROL[0]) != CONTROL[1]:
        raise ValueError("The control differs from the declared file.")
    dev = CH.dev_table_from(O.SHIP / "ctl.parquet", O.SHIP / "rows_all.parquet", O.SHIP / "dmp.parquet")
    s1 = pd.read_parquet(O.SHIP / "cand.parquet").set_index(ID)["new"].reindex(dev[ID]).to_numpy(dtype=float)
    keys, _ = O2.O.dev_table()
    keys = keys.set_index(ID).reindex(dev[ID]).reset_index()
    if not np.allclose(keys["label"].to_numpy(), dev["label"].to_numpy()):
        raise ValueError("Row alignment with the v61 proxy failed.")
    ctl = pd.read_parquet(ROOT / CONTROL[0]).set_index(ID).reindex(dev[ID]).reset_index()
    if ctl["value"].isna().any() or not np.allclose(ctl["label"].to_numpy(dtype=float), dev["label"].to_numpy(dtype=float)):
        raise ValueError("The control does not cover the development rows.")
    v69p = pd.read_parquet(ROOT / V69_PROXY[0]).set_index(ID).reindex(dev[ID]).reset_index()
    pins = step11_dev_pins()
    chain = V3.Chain()
    _, t12, _ = O2.v62_tables()
    trig, _ = O2.triggers_2025(O2.IDS_DIR, use_cache=True)
    new, air, month = dev["new"].to_numpy(dtype=float), dev["airport"].to_numpy(), dev["month"].to_numpy()
    res = {"created_at_utc": utc_now(), "declaration": DECL, "declaration_sha256": sha256_file(ROOT / DECL), "control": CONTROL[0],
           "control_sha256": sha256_file(ROOT / CONTROL[0]), "official_scores_used": "none", "audit_used": False,
           "code": {p: sha256_file(ROOT / p) for p in CODE}, "queue_columns": QCOLS, "weight": WEIGHT, "seeds": SW.SEEDS10}

    def through_chain(queue: str):
        P = load_preds(queue, dev)
        value, use, p = SW.dev_value(dev, P, WEIGHT, K)
        t = keys[[ID, "month", "airport", "label", "cls", "day"]].copy()
        t["s2"] = s1 + (value - new)
        _, cand63 = chain.run(t, "s2")
        t["c63"] = cand63
        c66 = CH.own_motion_v2(t, "c63", t12, trig)
        c70, tday = step11_dev(t[ID].to_numpy(), c66, pins)
        d = OUT / "dev" / (f"{queue}_w65_bag10" if queue == "off" else CANDIDATE)
        d.mkdir(parents=True, exist_ok=True)
        out = t[[ID, "month", "airport", "label", "cls"]].copy()
        out["day"] = tday
        out["value"] = c70
        out["v69like"] = c66
        out["opdi_used"] = use
        out.to_parquet(d / "proxy.parquet", index=False)
        return c66, c70, use, p, d / "proxy.parquet"

    # identity first: queue off must reproduce the control's value column
    c66_off, c70_off, _, _, path_off = through_chain("off")
    ident = float(np.abs(c70_off - ctl["value"].to_numpy(dtype=float)).max())
    ident69 = float(np.abs(c66_off - v69p["value"].to_numpy(dtype=float)).max())
    res["identity"] = {"queue_off_vs_control_max_abs_s": ident, "queue_off_v69like_vs_v69_proxy_max_abs_s": ident69,
                       "bound_s": LINUX_BOUND, "platform": sys.platform, "proxy_sha256": sha256_file(path_off),
                       "holds": bool(ident == 0.0 or ident <= LINUX_BOUND)}
    print("identity: queue off vs v70 dev proxy max |diff|", ident, "| v69-like vs v69 proxy", ident69, flush=True)
    if not res["identity"]["holds"]:
        DOCS.mkdir(parents=True, exist_ok=True)
        write_json(OUT / "dev" / "identity_failed.json", res)
        raise ValueError("identity: the queue-off path does not reproduce v70's development proxy; nothing is read.")

    # the candidate
    c66, c70, use, p, path_on = through_chain("on")
    tt = ctl[[ID, "month", "label", "day"]].assign(ctl=ctl["value"].to_numpy(dtype=float), cand=c70)
    rng = np.random.default_rng(20261004)
    months = {}
    for m in SW.MONTHS:
        s = tt[tt["month"] == m]
        pct, lo, hi = R.interval(s, "ctl", s["cand"].to_numpy(), rng)
        ch = s["cand"].to_numpy() - s["ctl"].to_numpy()
        months[m] = {"control_rmse": float(np.sqrt(np.mean((s["label"] - s["ctl"]) ** 2))), "rmse": float(np.sqrt(np.mean((s["label"] - s["cand"]) ** 2))),
                     "pct": pct, "lo": lo, "hi": hi, "rows_changed": int((np.abs(ch) > 1e-9).sum()), "squared_change": float((ch ** 2).sum()),
                     "rms_change_s": float(np.sqrt(np.mean(ch ** 2))), "max_abs_change_s": float(np.abs(ch).max())}
    eq = float(np.mean([months[m]["pct"] for m in SW.EQUAL]))
    checks = {"1_no_month_worse_than_0.005pct": all(months[m]["pct"] <= 0.005 for m in SW.MONTHS), "2_equal_footing_mean_pct": eq,
              "2_mean_ok": eq <= -0.02, "3_every_equal_interval_below_zero": all(months[m]["hi"] < 0 for m in SW.EQUAL)}
    diff = c70 - ctl["value"].to_numpy(dtype=float)
    by = pd.DataFrame({"airport": air, "month": month, "c": (ctl["value"].to_numpy() - ctl["label"].to_numpy()) ** 2,
                       "n": (c70 - ctl["label"].to_numpy()) ** 2}).groupby(["airport", "month"]).sum()
    fxm = S.fixed_rows(dev)
    res["candidate"] = {"name": CANDIDATE, "weight": WEIGHT, "seeds": SW.SEEDS10, "months": months, "checks_1_to_3": checks,
                        "clauses_1_to_3": bool(checks["1_no_month_worse_than_0.005pct"] and checks["2_mean_ok"] and checks["3_every_equal_interval_below_zero"]),
                        "airport_month_change_pct": {f"{a} {m}": float(100 * (np.sqrt(r["n"]) / np.sqrt(r["c"]) - 1)) for (a, m), r in by.iterrows()},
                        "fixed_rows_moved": int((np.abs(diff)[fxm] > 1e-9).sum()), "proxy_sha256": sha256_file(path_on)}
    print(CANDIDATE, " ".join(f"{m[5:]} {months[m]['pct']:+.4f}% [{months[m]['lo']:+.4f},{months[m]['hi']:+.4f}]" for m in SW.MONTHS),
          f"| eq {eq:+.4f}%", "clauses 1-3", res["candidate"]["clauses_1_to_3"], flush=True)
    # development side of clause 4 (out-of-month guarded predictions, rows where each bag applies)
    lirf = air == "LIRF"
    res["dev_oof_prediction_by_airport"] = {
        "v3": S.by_airport_rms(dev, np.where(~lirf & ~fxm, p["p3"], 0.0), ~(~lirf & ~fxm)),
        "v2_LIRF": S.by_airport_rms(dev, np.where(lirf & ~fxm, p["p2"], 0.0), ~(lirf & ~fxm)),
        "opdi_LSZH_used": S.by_airport_rms(dev, np.where(use, p["po"], 0.0), ~use)}
    write_json(RESULTS, res)


# ---------------------------------------------------------------- ranking

def rank_fit(queue: str) -> None:
    import run_stack_resid as S
    path = OUT / "rank" / queue / "preds.parquet"
    if path.exists():
        raise ValueError(f"{path} exists.")
    dev, res, fx, mats, rk, fxr, h2, h3, ho, Por = SW.rank_inputs()
    qinfo = {}
    if queue == "on":
        Qd, Qr = queue_dev(dev), queue_rank(rk)
        mats = [(k, with_queue(X, Qd), with_queue(Xr, Qr)) for k, X, Xr in mats]
        # rows whose gaps cross the January-July boundary of the ranking file (declared: the ranking file as one file)
        mon = pd.read_parquet(ROOT / RANKING, columns=[ID, "MVT_TIME_UTC_mvt"]).set_index(ID).reindex(rk[ID])["MVT_TIME_UTC_mvt"].dt.month.to_numpy()
        cross = np.zeros(len(rk), bool)
        for c in ["gp", "gp2", "gp4"]:
            cross |= (mon == 7) & (Qr[c].to_numpy() > 40 * 86400)
        for c in ["gn", "gn4"]:
            cross |= (mon == 1) & (Qr[c].to_numpy() > 40 * 86400)
        qinfo = {"ranking_rows_with_a_gap_across_january_july": int(cross.sum())}
        print("queue on: ranking rows with a gap across January-July", int(cross.sum()), flush=True)
    dom = dev["dom"].to_numpy()
    out = pd.DataFrame({ID: rk[ID].to_numpy(), "airport": rk["airport"].to_numpy(), "new": rk["new"].to_numpy(dtype=float), "fx": fxr,
                        "h2": h2, "h3": h3, "ho": ho, "o_has": Por["o_has"].to_numpy()})
    trees, ncol = {}, {}
    for key, X, Xr in mats:
        Xa, Xra = X.to_numpy(np.float32), Xr.to_numpy(np.float32)
        ncol[key] = int(Xa.shape[1])
        if list(X.columns) != list(Xr.columns):
            raise ValueError("Development and ranking columns differ.")
        for i, seed in enumerate(SW.SEEDS10):
            S.SEEDS = [seed]
            models = S.fit_bag(Xa, res, ~fx, dom)
            out[f"{key}_{i}"] = models[0].predict(Xra, num_iteration=models[0].best_iteration)
            trees[f"{key} {seed}"] = int(models[0].best_iteration)
            print("rank", queue, key, ncol[key], seed, "trees", trees[f"{key} {seed}"], flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    write_json(path.with_suffix(".json"), {"created_at_utc": utc_now(), "queue": queue, "columns": ncol, "rows": int(len(out)), "trees": trees,
                                           **qinfo, "sha256": sha256_file(path), "code": {p: sha256_file(ROOT / p) for p in CODE}})


def step11_rank(base: Path, out_path: Path) -> dict:
    """Step 11 exactly as tools/run_oooi_pin.py's ranking part (selected w50): offsets from July / November / December 2025 on the v69
    development proxy, LFPG in-scope rows of the base file, np.rint into the file's dtype."""
    import run_intruder_bound as R
    import run_oooi_pin as P
    import run_stack_resid as S
    if sha256_file(P.PROXY) != P.PROXY_SHA:
        raise ValueError("The v69 development proxy differs.")
    pin_res = read_json(ROOT / "docs/oooi_pin/results_v1.json")
    for k, f in [("dev", "joined_dev.parquet"), ("rank", "joined_rank.parquet")]:
        if sha256_file(P.RUN / f) != pin_res["joined_sha256"][k]:
            raise ValueError(f"runs/oooi_pin/{f} changed since oooi_pin_v1's governing run.")
    pin_dev = P.scope(pd.read_parquet(P.RUN / "joined_dev.parquet"))
    pin_rk = P.scope(pd.read_parquet(P.RUN / "joined_rank.parquet"))
    dev = pd.read_parquet(P.PROXY, columns=[ID, "month", "airport", "label", "cls", "day", "value"])
    dev = dev.merge(P.takeoff_days(P.DEV_MONTHS), on=ID, how="left", validate="one_to_one")
    dev["day"] = dev["tday"]
    dev = dev.merge(pin_dev[[ID, "source", "raw"]], on=ID, how="left", validate="one_to_one")
    fixed = S.fixed_rows(dev)
    dev["in_scope"] = dev["raw"].notna().to_numpy() & ~fixed & dev["airport"].eq("LFPG").to_numpy()
    off_all = P.offsets(dev[dev["in_scope"] & dev["month"].isin(P.PIN_DEV_MONTHS)])
    v69 = pd.read_parquet(base)
    rk = pd.read_parquet(P.RANKING, columns=[ID, "PHASE_mvt", "ADEP_mvt", "ADEP_flt", "AIRCRAFT_TYPE_mvt"])
    rk = rk[rk["PHASE_mvt"].eq("DEP")]
    if list(v69.columns) != [ID, "TAXITIME_SEC_mvt"] or len(v69) != len(rk):
        raise ValueError("The base is not a full submission.")
    pop = rk["ADEP_mvt"].eq("LIRF") & rk["ADEP_flt"].isna() & rk["AIRCRAFT_TYPE_mvt"].isna()
    rfixed_ids = set(rk.loc[pop, ID]) | set(R.GUARD) | set(rk.loc[rk["ADEP_mvt"].eq("LIRF"), ID])
    rkt = v69.merge(rk[[ID, "ADEP_mvt"]], on=ID, how="left", validate="one_to_one")
    rkt = rkt.merge(pin_rk[[ID, "source", "raw", "month_dep"]], on=ID, how="left", validate="one_to_one")
    rkt["in_scope"] = rkt["raw"].notna() & rkt["ADEP_mvt"].eq("LFPG") & ~rkt[ID].isin(rfixed_ids)
    rkt["pin"] = np.where(rkt["in_scope"], rkt["raw"] + rkt["source"].map(off_all), np.nan)
    rkt["chain"] = rkt["TAXITIME_SEC_mvt"].astype(float)
    rs = rkt["in_scope"].to_numpy()
    rv = rkt["chain"].to_numpy().copy()
    rv[rs] = P.cand_value(0.5, rkt.loc[rs, "pin"].to_numpy(), rkt.loc[rs, "chain"].to_numpy())
    new_int = np.rint(rv).astype(v69["TAXITIME_SEC_mvt"].dtype)
    out_df = v69.copy()
    out_df["TAXITIME_SEC_mvt"] = pd.Series(new_int, index=v69.index).astype(v69["TAXITIME_SEC_mvt"].dtype)
    if not out_df[ID].equals(v69[ID]):
        raise ValueError("Row order changed.")
    changed = out_df["TAXITIME_SEC_mvt"].ne(v69["TAXITIME_SEC_mvt"])
    fixed_r = v69[ID].isin(rfixed_ids)
    if (changed & fixed_r).any() or (changed & ~rkt["in_scope"]).any():
        raise ValueError("A row outside the scope, a LIRF, population or guard row moved.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        raise ValueError(f"{out_path} exists.")
    out_df.to_parquet(out_path, index=False)
    if not pd.read_parquet(out_path).equals(out_df):
        raise ValueError("The written file does not read back identically.")
    rec = {"base_sha256": sha256_file(base), "sha256": sha256_file(out_path), "rows_in_scope": int(rs.sum()), "rows_changed": int(changed.sum()),
           "offsets_jul_nov_dec": {k: float(v) for k, v in off_all.items()}}
    write_json(out_path.with_suffix(".receipt.json"), rec)
    return rec


def steps_6_to_11(P: pd.DataFrame, out: Path, version: int) -> tuple[Path, Path, np.ndarray]:
    """Step 6 at weight 0.65 with ten seeds (run_stack_weight.rank_value), steps 7-10 (chain_from_stack, own_motion_v2), step 11."""
    import run_own_motion_v2 as O2
    import run_stack_resid_v3 as V3
    from prc2026_submission.contract import export_submission
    values, use, _ = SW.rank_value(P, WEIGHT, K)
    st = out / f"steps/stack/vigorous-whistle_v{version}.parquet"
    export_submission(pd.DataFrame({ID: P[ID].to_numpy(), "prediction": values}), ROOT / "data/raw/submitting.parquet", st,
                      {"dry_run": True, "step": f"stack_queue_v1: stacking value at weight {WEIGHT} with {K} seeds", "official_scores_used": "none",
                       "audit_used": False})
    v63like = V3.chain_from_stack(st, version, out / "steps")
    O2.dry_run(v63like, sha256_file(v63like), version, out / "steps/om2")
    v69like = out / f"steps/om2/vigorous-whistle_v{version}.parquet"
    step11_rank(v69like, out / f"steps/pin/vigorous-whistle_v{version}.parquet")
    return v69like, out / f"steps/pin/vigorous-whistle_v{version}.parquet", use


def rank_identity() -> None:
    out = OUT / "rank" / "identity_v70"
    if out.exists():
        raise ValueError(f"{out} exists.")
    P = pd.read_parquet(OUT / "rank" / "off" / "preds.parquet")
    rec = {"created_at_utc": utc_now()}
    ref = ROOT / STACK_WEIGHT_RANK_PREDS[0]
    if ref.exists() and sha256_file(ref) == STACK_WEIGHT_RANK_PREDS[1]:
        R0 = pd.read_parquet(ref)
        cols = [f"{k}_{i}" for k in SW.LEARNERS for i in range(10)]
        rec["queue_off_preds_vs_stack_weight_preds_max_abs"] = (float(np.abs(P[cols].to_numpy() - R0[cols].to_numpy()).max())
                                                                if P[ID].equals(R0[ID]) else "row order differs")
        print("queue-off ranking predictions vs stack_weight's (8083ffcf):", rec["queue_off_preds_vs_stack_weight_preds_max_abs"], flush=True)
    v69like, final, use = steps_6_to_11(P, out, 70)
    rec.update({"v69like_sha256": sha256_file(v69like), "reproduces_v69": sha256_file(v69like) == V69, "final_sha256": sha256_file(final),
                "reproduces_v70": sha256_file(final) == V70, "opdi_rows_used": int(use.sum())})
    write_json(out / "identity.json", rec)
    print(rec, flush=True)
    if not rec["reproduces_v70"]:
        raise ValueError("identity: the ranking path with the queue columns off does not reproduce v70; no candidate ranking file.")


def rank_check() -> None:
    import run_stack_resid as S
    res = read_json(RESULTS)
    ident = read_json(OUT / "rank" / "identity_v70" / "identity.json")
    res["rank_identity"] = ident
    if not ident["reproduces_v70"]:
        write_json(RESULTS, res)
        raise ValueError("The ranking identity failed.")
    P = pd.read_parquet(OUT / "rank" / "on" / "preds.parquet")
    fxr, lirf = P["fx"].to_numpy(dtype=bool), P["airport"].eq("LIRF").to_numpy()
    _, use, p = SW.rank_value(P, WEIGHT, K)
    rk = {"v3": S.by_airport_rms(P, np.where(~lirf & ~fxr, p["p3"], 0.0), ~(~lirf & ~fxr)),
          "v2_LIRF": S.by_airport_rms(P, np.where(lirf & ~fxr, p["p2"], 0.0), ~(lirf & ~fxr)),
          "opdi_LSZH_used": S.by_airport_rms(P, np.where(use, p["po"], 0.0), ~use)}
    dv = res["dev_oof_prediction_by_airport"]
    ratios = {f"{part} {a}": rk[part][a]["rms_s"] / dv[part][a]["rms_s"] for part in rk for a in rk[part] if a in dv[part]}
    ok = all(0.5 <= r <= 2.0 for r in ratios.values())
    res["clause_4_transfer_guard"] = {"ranking_prediction_by_airport": rk, "ratios": ratios, "ok": bool(ok)}
    print("clause 4", {k: round(r, 3) for k, r in ratios.items()}, "ok", ok, flush=True)
    acc = bool(res["candidate"]["clauses_1_to_3"] and ok)
    res["candidate"]["accepted"] = acc
    res["selected"] = CANDIDATE if acc else None
    write_json(RESULTS, res)
    print("selected", res["selected"], flush=True)


def rank_dry() -> None:
    import run_intruder_bound as R
    res = read_json(RESULTS)
    if res.get("selected") != CANDIDATE:
        raise ValueError("The candidate is not accepted; no ranking file.")
    out = OUT / "rank" / "dry_v71"
    if out.exists():
        raise ValueError(f"{out} exists; never overwrite a dry run.")
    P = pd.read_parquet(OUT / "rank" / "on" / "preds.parquet")
    _, pinned, use = steps_6_to_11(P, out, 71)
    final = out / "vigorous-whistle_v71.parquet"
    shutil.copyfile(pinned, final)
    ref = OUT / "rank" / "identity_v70" / "steps/pin/vigorous-whistle_v70.parquet"
    if sha256_file(ref) != V70:
        raise ValueError("The v70 reference differs.")
    v = pd.read_parquet(final)
    v70 = pd.read_parquet(ref)
    if not v[ID].equals(v70[ID]):
        raise ValueError("Row order differs from v70.")
    raw = pd.read_parquet(ROOT / RANKING, columns=[ID, "PHASE_mvt", "ADEP_mvt", "ADEP_flt", "AIRCRAFT_TYPE_mvt", "MVT_TIME_UTC_mvt"])
    raw = raw[raw["PHASE_mvt"].eq("DEP")].set_index(ID).reindex(v[ID])
    pop = (raw["ADEP_mvt"].eq("LIRF") & raw["ADEP_flt"].isna() & raw["AIRCRAFT_TYPE_mvt"].isna()).to_numpy()
    guard = v[ID].isin(R.GUARD).to_numpy()
    diff = v[TARGET].to_numpy(dtype=float) - v70[TARGET].to_numpy(dtype=float)
    ch = diff != 0
    by = pd.DataFrame({"airport": raw["ADEP_mvt"].to_numpy(), "month": raw["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m").to_numpy(), "d": diff})
    rep = {"created_at_utc": utc_now(), "candidate": CANDIDATE, "file": final.relative_to(ROOT).as_posix(), "sha256": sha256_file(final),
           "bytes": final.stat().st_size, "opdi_rows_used_ranking": int(use.sum()),
           "vs_v70": {"rows_changed": int(ch.sum()), "squared_change": float((diff ** 2).sum()), "rms_change_s": float(np.sqrt((diff ** 2).mean())),
                      "max_up_s": float(diff.max()), "max_down_s": float(diff.min()), "max_abs_s": float(np.abs(diff).max()),
                      "mean_change_s": float(diff.mean())},
           "lirf_rows_changed": int((ch & raw["ADEP_mvt"].eq("LIRF").to_numpy()).sum()), "lirf_population_rows_changed": int((ch & pop).sum()),
           "guard_rows_changed": int((ch & guard).sum()), "guard_rows_v71": {str(int(i)): int(x) for i, x in zip(v.loc[guard, ID], v.loc[guard, TARGET])},
           "guard_rows_v70": {str(int(i)): int(x) for i, x in zip(v70.loc[guard, ID], v70.loc[guard, TARGET])},
           "min_value": int(v[TARGET].min()), "dtype": str(v[TARGET].dtype),
           "by_airport": {a: {"rows_changed": int((g["d"] != 0).sum()), "squared_change": float((g["d"] ** 2).sum()),
                              "max_abs_s": float(g["d"].abs().max())} for a, g in by.groupby("airport")},
           "by_airport_month": {f"{a} {m}": {"rows_changed": int((g["d"] != 0).sum()), "squared_change": float((g["d"] ** 2).sum()),
                                             "rms_s": float(np.sqrt((g["d"] ** 2).mean())), "max_abs_s": float(g["d"].abs().max())}
                                for (a, m), g in by.groupby(["airport", "month"])},
           "steps": {p.relative_to(out).as_posix(): sha256_file(p) for p in sorted((out / "steps").rglob("vigorous-whistle_v*.parquet"))},
           "official_scores_used": "none"}
    if rep["lirf_population_rows_changed"] or rep["guard_rows_changed"]:
        raise ValueError("A LIRF population or guard row changed against v70.")
    write_json(out / "vigorous-whistle_v71.report.json", rep)
    res["ranking_dry_run"] = {k: rep[k] for k in ["candidate", "file", "sha256", "vs_v70", "by_airport", "lirf_rows_changed",
                                                  "lirf_population_rows_changed", "guard_rows_changed"]}
    write_json(RESULTS, res)
    print({k: rep[k] for k in ["sha256", "vs_v70", "lirf_rows_changed", "lirf_population_rows_changed", "guard_rows_changed"]}, flush=True)


def warm(part: str) -> None:
    """VM preparation only (no fit, no score, nothing written under runs/stack_queue): open every input of the fits (part 'fit') or of the
    development chain (part 'chain') once, in one process, so that the lazily fetched files arrive before parallel jobs read them."""
    if part == "fit":
        dev = SW.dev_inputs()[0]
        queue_dev(dev)
        rk = SW.rank_inputs()[4]
        queue_rank(rk)
        print("warm fit: development", len(dev), "ranking", len(rk), flush=True)
    else:
        import run_ctx_ownpoints_chain as CH
        import run_opdi_stack as O
        import run_own_motion_v2 as O2
        import run_stack_resid_v3 as V3
        if sha256_file(ROOT / CONTROL[0]) != CONTROL[1]:
            raise ValueError("The control differs from the declared file.")
        CH.dev_table_from(O.SHIP / "ctl.parquet", O.SHIP / "rows_all.parquet", O.SHIP / "dmp.parquet")
        pd.read_parquet(O.SHIP / "cand.parquet")
        O2.O.dev_table()
        step11_dev_pins()
        V3.Chain()
        O2.v62_tables()
        O2.triggers_2025(O2.IDS_DIR, use_cache=True)
        print("warm chain: done", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--warm", choices=["fit", "chain"])
    ap.add_argument("--dev-month")
    ap.add_argument("--queue", choices=["off", "on"])
    ap.add_argument("--dev-score", action="store_true")
    ap.add_argument("--rank-fit", action="store_true")
    ap.add_argument("--rank-identity", action="store_true")
    ap.add_argument("--rank-check", action="store_true")
    ap.add_argument("--rank-dry", action="store_true")
    ap.add_argument("--queue-preview", help="print the queue columns' summary for one raw file (no fit)")
    a = ap.parse_args()
    if (a.dev_month or a.rank_fit) and not a.queue:
        ap.error("--queue off|on is required with --dev-month / --rank-fit")
    if a.queue_preview:
        q = queue_columns(ROOT / a.queue_preview)
        print(len(q), q[QCOLS].describe().T.to_string())
    if a.warm:
        warm(a.warm)
    if a.dev_month:
        dev_month(a.dev_month, a.queue)
    if a.dev_score:
        dev_score()
    if a.rank_fit:
        rank_fit(a.queue)
    if a.rank_identity:
        rank_identity()
    if a.rank_check:
        rank_check()
    if a.rank_dry:
        rank_dry()


if __name__ == "__main__":
    main()
