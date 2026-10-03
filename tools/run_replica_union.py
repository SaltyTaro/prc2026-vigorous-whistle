"""Replica union (configs/replica_union_v1.json): adsb.lol's two daily replicas (-prod-0 + -staging-0) as one surface day file.

Departures that the shipped prod-0 matches do not trace from the gate (unmatched, or dark matched) but that the shipped base matcher
traces from their known stand on the union day file move halfway to the union ADS-B taxi plus a per-airport offset (median of
label - taxi over the other months' union gate-at-stand rows). Eight airports (LIRF excluded), applied last on the chain.

--matches   union day files (prod-0 + data/external/adsblol/surface_staging) -> match_departures with the learned stands for every 2025 and
            ranking day that has a staging replica; other days take matches2 rows -> runs/replica_union/union_matches_{2025,rank}.parquet
--score     development (the v62 proxy by default, or --control) and twelve-month tables, month-blocked, with the declared rule
            -> docs/replica_union/results_v1.json (results_v1_<tag>.json under another control)
--dry-run   label-free ranking file on a base file (read only) -> --out (default runs/replica_union/dry_v<version>/)
--build     rerun the dry run into runs/replica_union/build_v<version>, require byte identity with the dry run, write
            runs/submissions/vigorous-whistle_v<version>.parquet (never overwrite) and docs/replica_union/export_v<version>.json
Nothing is uploaded.
"""
from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, PHASE, TARGET, TIME  # noqa: E402
from prc2026_adsb.features import DEPARTURE_COLUMNS, match_departures  # noqa: E402

CONFIG = ROOT / "configs/replica_union_v1.json"
SURFACE = ROOT / "data/external/adsblol/surface"
STAGING = ROOT / "data/external/adsblol/surface_staging"
MATCHES = ROOT / "data/external/adsblol/matches"
REMATCH = ROOT / "runs/rematch_nonm"
OUT = ROOT / "runs/replica_union"
DOCS = ROOT / "docs/replica_union"
RANKING = ROOT / "data/raw/ranking.parquet"
TRAINING = sorted(glob.glob(str(ROOT / "data/raw/training_2025-*.parquet")))
KEY = ["icao", "airport", "time", "lat", "lon"]
KEEP = [ID, "airport", "day", "takeoff", "gate", "at_known_stand", "motion_start", "segment_start", "points", "icao"]
GUARD = [213142471, 213142469]
V62 = ("runs/submissions/vigorous-whistle_v62.parquet", "cba143f84debdca0d9aff345ff4669ff6a3f0f7f5c4e02d08bcd3e0d02c32d9d")
DEV_MONTHS = ["2025-01", "2025-07", "2025-11", "2025-12"]


# ---------------------------------------------------------------- union day files and matches (label-free)

def union_day(day: str) -> tuple[pd.DataFrame, bool]:
    prod = pd.read_parquet(SURFACE / f"surface_{day}.parquet")
    sf = STAGING / f"surface_{day}_staging-0.parquet"
    has = sf.exists()
    u = pd.concat([prod, pd.read_parquet(sf)], ignore_index=True) if has else prod
    u = u.drop_duplicates(KEY).sort_values(["airport", "icao", "time"], kind="mergesort").reset_index(drop=True)
    u["callsign"] = u["callsign"].fillna("").str.strip().str.upper()
    return u, has


def _match_job(job: tuple) -> pd.DataFrame:
    """One day's union matches; cached under runs/replica_union/days/ (the staging day files are written once and never changed)."""
    day, dep = job
    cache = OUT / "days" / f"union_{day}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    stands = pd.read_parquet(MATCHES / "stands.parquet")
    surface, _ = union_day(day)
    m = match_departures(dep, surface, stands)
    if not len(m):
        m = pd.DataFrame(columns=KEEP)
    else:
        m["day"] = day
        m = m[KEEP]
    cache.parent.mkdir(parents=True, exist_ok=True)
    m.to_parquet(cache, index=False)
    return m


def _prod_matches(months: list[str]) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(MATCHES / f"matches2_{m}.parquet") for m in months], ignore_index=True)


def step_matches(names: list[str], cache_only: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stag_days = sorted(p.name[8:16] for p in STAGING.glob("surface_*_staging-0.parquet"))
    receipt = {"created_at_utc": utc_now(), "code_sha256": sha256_file(Path(__file__)), "config_sha256": sha256_file(CONFIG),
               "stands_sha256": sha256_file(MATCHES / "stands.parquet"), "staging_files": {}, "tables": {}}
    for name, paths, months in [("2025", TRAINING, [f"2025-{k:02d}" for k in range(1, 13)]), ("rank", [str(RANKING)], ["2026-01", "2026-07"])]:
        if name not in names:
            continue
        target = OUT / f"union_matches_{name}.parquet"
        if target.exists():
            print("keep", target.name)
            continue
        raw = pd.concat([pd.read_parquet(p, columns=DEPARTURE_COLUMNS) for p in paths], ignore_index=True)
        dep = raw[raw[PHASE].eq("DEP")].drop_duplicates(ID).reset_index(drop=True)
        dep["dayk"] = dep[TIME].dt.strftime("%Y%m%d")
        dep = dep[dep[TIME].dt.strftime("%Y-%m").isin(months)]
        days = [d for d in stag_days if d in set(dep["dayk"])]
        jobs = [(d, dep[dep["dayk"].eq(d)].drop(columns=["dayk"]).reset_index(drop=True)) for d in days]
        with mp.Pool(5) as pool:
            parts = pool.map(_match_job, jobs, chunksize=1)
        if cache_only:
            print(name, "cached days", len(days), flush=True)
            continue
        uni = pd.concat([p for p in parts if len(p)], ignore_index=True)
        uni["source"] = "union"
        prod = _prod_matches(months)
        prod = prod[~prod["day"].astype(str).isin(days)].copy()
        prod["day"] = prod["day"].astype(str)
        prod = prod[KEEP].assign(source="prod0_only_day")
        t = pd.concat([uni, prod], ignore_index=True)
        t["gate"] = t["gate"].astype(bool)
        t["at_known_stand"] = t["at_known_stand"].astype(bool)
        t.to_parquet(target, index=False)
        receipt["tables"][name] = {"file": str(target.relative_to(ROOT).as_posix()), "sha256": sha256_file(target), "rows": int(len(t)),
                                   "union_days": len(days), "prod0_only_days": sorted(set(dep["dayk"]) - set(days)),
                                   "gate_at_stand_rows": int((t["gate"] & t["at_known_stand"]).sum())}
        print(name, receipt["tables"][name]["rows"], "union days", len(days), flush=True)
    if cache_only:
        return
    old = read_json(OUT / "union_matches_receipt.json") if (OUT / "union_matches_receipt.json").exists() else {"tables": {}}
    receipt["tables"] = {**old.get("tables", {}), **receipt["tables"]}
    for p in sorted(STAGING.glob("surface_*_staging-0.parquet")):
        receipt["staging_files"][p.name] = sha256_file(p)
    write_json(OUT / "union_matches_receipt.json", receipt)


# ---------------------------------------------------------------- per-row state and the layer

def keyed_ids(name: str) -> set:
    t = pd.read_parquet(REMATCH / f"matches_{name}.parquet", columns=[ID, "iata_gate", "learned_gate"])
    return set(t.loc[t["iata_gate"].fillna(False).astype(bool) | t["learned_gate"].fillna(False).astype(bool), ID])


def row_state(name: str, months: list[str]) -> pd.DataFrame:
    """MVT_ID_mvt -> union gate-at-stand flag and taxi T, prod-0 gate / matched flags, keyed flag (label-free)."""
    u = pd.read_parquet(OUT / f"union_matches_{name}.parquet")
    u = u[u["gate"] & u["at_known_stand"]][[ID, "airport", "day", "takeoff", "motion_start"]].copy()
    u["T"] = u["takeoff"] - u["motion_start"]
    prod = _prod_matches(months)
    st = u[[ID, "day", "T"]].drop_duplicates(ID)
    st["prod_gate"] = st[ID].isin(set(prod.loc[prod["gate"].astype(bool), ID]))
    st["prod_matched"] = st[ID].isin(set(prod[ID]))
    st["keyed"] = st[ID].isin(keyed_ids(name))
    return st.rename(columns={"day": "uday"})


def labels_2025() -> pd.Series:
    raw = pd.concat([pd.read_parquet(p, columns=[ID, PHASE, TARGET]) for p in TRAINING], ignore_index=True)
    return raw[raw[PHASE].eq("DEP")].drop_duplicates(ID).set_index(ID)[TARGET].astype(float)


def offset_rows(t12: pd.DataFrame, st: pd.DataFrame) -> pd.DataFrame:
    """Fitting rows for the offsets: every union gate-at-stand 2025 departure (LIRF_pop and guard rows excluded), label - T."""
    f = t12[["MVT_ID_mvt", "month", "airport", "label", "cls"]].merge(st[[ID, "T"]], on=ID)
    f = f[~f["cls"].eq("LIRF_pop") & ~f[ID].isin(GUARD)].copy()
    f["e"] = f["label"] - f["T"]
    return f


def fit(fitting: pd.DataFrame, airports: list[str]) -> dict:
    return {a: float(np.median(fitting.loc[fitting["airport"].eq(a), "e"])) for a in airports if fitting["airport"].eq(a).any()}


def candidates(tab: pd.DataFrame, pred: str, st: pd.DataFrame, spec: dict) -> pd.DataFrame:
    c = tab.merge(st, on=ID)
    keep = (c["airport"].isin(spec["airports"]) & ~c["prod_gate"] & ~c["keyed"] & ~c["cls"].eq("LIRF_pop") & ~c[ID].isin(GUARD))
    if not spec["include_upgraded"]:
        keep &= ~c["prod_matched"]
    return c[keep].copy()


def values(cand: pd.DataFrame, pred: str, off: dict, w: float) -> pd.DataFrame:
    c = cand.copy()
    o = c["airport"].map(off).to_numpy(dtype=float)
    c["value"] = c[pred] + w * (c["T"] + o - c[pred])
    return c


def tables() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    import run_own_motion_v2 as O2
    dev, t12, rmse = O2.v62_tables()
    return dev, t12, rmse


def score(control: str | None, control_col: str | None, control_sha: str | None, tag: str | None) -> None:
    import run_intruder_bound as R
    cfg = read_json(CONFIG)
    rule = cfg["rule"]
    dev, t12, v62_rmse = tables()
    pred = "v62"
    if control:
        path = ROOT / control
        if control_sha and sha256_file(path) != control_sha:
            raise ValueError("The control file's SHA-256 differs from the expected one.")
        ctl = pd.read_parquet(path, columns=["MVT_ID_mvt", control_col]).rename(columns={control_col: "ctl"})
        dev = dev.merge(ctl, on="MVT_ID_mvt", how="left")
        if dev["ctl"].isna().any():
            raise ValueError("The control does not cover every development row.")
        pred = "ctl"
    st = row_state("2025", [f"2025-{k:02d}" for k in range(1, 13)])
    fitrows = offset_rows(t12, st)
    rec = read_json(OUT / "union_matches_receipt.json")
    specs = [cfg["candidate"]] + cfg["sensitivity_reported_not_selectable"]
    report = {"experiment_id": cfg["experiment_id"], "config_sha256": sha256_file(CONFIG), "code_sha256": sha256_file(Path(__file__)),
              "computed_at_utc": utc_now(), "union_matches": rec["tables"], "official_scores_used": "none",
              "control": {"kind": "v62 development proxy" if not control else control, "column": pred if not control else control_col,
                          "sha256": sha256_file(ROOT / control) if control else None,
                          "month_rmse_s": {m: float(np.sqrt(((g["label"] - g[pred]) ** 2).mean())) for m, g in dev.groupby("month")}},
              "v62_proxy_month_rmse_s": v62_rmse, "offset_fitting_rows_by_airport": fitrows["airport"].value_counts().to_dict(), "candidates": {}}
    for spec in specs:
        rng = np.random.default_rng(20261003)
        out = {"development": {}, "twelve": {}}
        for name, tab, p, months in [("development", dev, pred, DEV_MONTHS), ("twelve", t12, "after3", sorted(t12["month"].unique()))]:
            cand = candidates(tab, p, st, spec)
            for m in months:
                off = fit(fitrows[fitrows["month"] != m], spec["airports"])
                c = values(cand[cand["month"] == m], p, off, spec["w"])
                s = tab[tab["month"] == m]
                new = s[p].to_numpy(dtype=float).copy()
                pos = pd.Series(np.arange(len(s)), index=s["MVT_ID_mvt"].to_numpy())
                new[pos.loc[c["MVT_ID_mvt"]].to_numpy()] = c["value"].to_numpy()
                pct, lo, hi = R.interval(s, p, new, rng)
                sq = float(((c["value"] - c[p]) ** 2).sum())
                gain = float((((c["label"] - c[p]) ** 2) - (c["label"] - c["value"]) ** 2).sum())
                top = c.assign(g=((c["label"] - c[p]) ** 2) - (c["label"] - c["value"]) ** 2)
                out[name][m] = {"pct": pct, "lo": lo, "hi": hi, "rows": int(len(c)), "upgraded_rows": int(c["prod_matched"].sum()),
                                "by_airport_rows": c["airport"].value_counts().to_dict(), "squared_change": sq, "gain": gain,
                                "gain_over_squared_change": gain / sq if sq else None,
                                "top_day_rows": top.groupby("uday").size().sort_values(ascending=False).head(3).to_dict(),
                                "top_day_gain_share": float(top.groupby("uday")["g"].sum().max() / gain) if gain > 0 else None,
                                "mean_label_minus_T_minus_o": float((c["label"] - c["T"] - c["airport"].map(off)).mean()) if len(c) else None,
                                "offsets": off}
        dv, tw = out["development"], out["twelve"]
        eq = float(np.mean([dv[m]["pct"] for m in rule["equal_footing"]]))
        checks = {"1_max_development_month_pct": max(dv[m]["pct"] for m in DEV_MONTHS),
                  "1_no_month_worse_than_0.005pct": all(dv[m]["pct"] <= rule["max_month_worsening_pct"] for m in DEV_MONTHS),
                  "2_equal_footing_mean_pct": eq, "2_mean_ok": eq <= -rule["min_mean_improvement_pct"],
                  "3_equal_intervals_hi": {m: dv[m]["hi"] for m in rule["equal_footing"]},
                  "3_every_equal_interval_below_zero": all(dv[m]["hi"] < 0 for m in rule["equal_footing"]),
                  "4_twelve_months_better": int(sum(v["pct"] < 0 for v in tw.values())),
                  "4_twelve_months_worse": int(sum(v["pct"] > 0 for v in tw.values())),
                  "4_every_month_better_or_unchanged": all(v["pct"] <= 0 for v in tw.values()),
                  "twelve_intervals_below_zero": int(sum(v["hi"] < 0 for v in tw.values())),
                  "twelve_mean_pct": float(np.mean([v["pct"] for v in tw.values()]))}
        checks["accepted"] = bool(checks["1_no_month_worse_than_0.005pct"] and checks["2_mean_ok"] and checks["3_every_equal_interval_below_zero"]
                                  and checks["4_every_month_better_or_unchanged"])
        out["checks"] = checks
        out["offsets_all_twelve"] = fit(fitrows, spec["airports"])
        report["candidates"][spec["name"]] = out
        print(spec["name"], "dev:", " ".join(f"{m[5:]} {dv[m]['pct']:+.4f}% [{dv[m]['lo']:+.3f},{dv[m]['hi']:+.3f}] n{dv[m]['rows']}" for m in DEV_MONTHS))
        print("   twelve:", " ".join(f"{m[5:]} {v['pct']:+.3f}%" for m, v in tw.items()))
        print("  ", checks, flush=True)
    name = cfg["candidate"]["name"]
    report["selected"] = name if report["candidates"][name]["checks"]["accepted"] else None
    DOCS.mkdir(parents=True, exist_ok=True)
    dest = DOCS / ("results_v1.json" if not tag else f"results_v1_{tag}.json")
    write_json(dest, report)
    print("selected:", report["selected"], "->", dest.relative_to(ROOT))


# ---------------------------------------------------------------- ranking

def dry_run(base: Path, base_sha: str, version: int, out_dir: Path) -> None:
    cfg = read_json(CONFIG)
    if sha256_file(base) != base_sha:
        raise ValueError("The base file's SHA-256 differs from the expected one.")
    spec = cfg["candidate"]
    _, t12, _ = tables()
    st25 = row_state("2025", [f"2025-{k:02d}" for k in range(1, 13)])
    off = fit(offset_rows(t12, st25), spec["airports"])
    rk = pd.read_parquet(RANKING, columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADEP_flt", "AIRCRAFT_TYPE_mvt"])
    rk = rk[rk["PHASE_mvt"].eq("DEP")]
    pop = rk["ADEP_mvt"].eq("LIRF") & rk["ADEP_flt"].isna() & rk["AIRCRAFT_TYPE_mvt"].isna()
    b = pd.read_parquet(base)
    if list(b.columns) != ["MVT_ID_mvt", "TAXITIME_SEC_mvt"] or len(b) != len(rk):
        raise ValueError("The base file is not a full submission.")
    tab = b.merge(rk[["MVT_ID_mvt", "ADEP_mvt"]].assign(cls=np.where(pop, "LIRF_pop", "x")).rename(columns={"ADEP_mvt": "airport"}), on="MVT_ID_mvt")
    tab["base"] = tab["TAXITIME_SEC_mvt"].astype(float)
    st = row_state("rank", ["2026-01", "2026-07"])
    c = values(candidates(tab, "base", st, spec), "base", off, spec["w"])
    out = b.copy()
    new = pd.Series(np.rint(c["value"].to_numpy()).astype(out["TAXITIME_SEC_mvt"].dtype), index=c["MVT_ID_mvt"].to_numpy())
    idx = out["MVT_ID_mvt"].map(new)
    out["TAXITIME_SEC_mvt"] = np.where(idx.notna(), idx, out["TAXITIME_SEC_mvt"]).astype(b["TAXITIME_SEC_mvt"].dtype)
    if not out["MVT_ID_mvt"].equals(b["MVT_ID_mvt"]):
        raise ValueError("Row order changed.")
    changed = out["TAXITIME_SEC_mvt"].ne(b["TAXITIME_SEC_mvt"])
    fixed = b["MVT_ID_mvt"].isin(set(rk.loc[pop, "MVT_ID_mvt"]) | set(GUARD) | set(rk.loc[rk["ADEP_mvt"].eq("LIRF"), "MVT_ID_mvt"]))
    if (changed & fixed).any():
        raise ValueError("A LIRF, population or guard row moved.")
    if not set(b.loc[changed, "MVT_ID_mvt"]) <= set(c["MVT_ID_mvt"]):
        raise ValueError("A row outside the candidates moved.")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"vigorous-whistle_v{version}.parquet"
    out.to_parquet(path, index=False)
    diff = out["TAXITIME_SEC_mvt"].astype(float) - b["TAXITIME_SEC_mvt"].astype(float)
    moved = c.assign(new=np.rint(c["value"]), move=np.rint(c["value"]) - c["base"])
    receipt = {"experiment_id": cfg["experiment_id"], "config_sha256": sha256_file(CONFIG), "code_sha256": sha256_file(Path(__file__)),
               "written_at_utc": utc_now(), "kind": "dry run, not a submission",
               "union_matches_sha256": {n: sha256_file(OUT / f"union_matches_{n}.parquet") for n in ["2025", "rank"]},
               "base": str(base.relative_to(ROOT).as_posix()) if base.is_relative_to(ROOT) else str(base), "base_sha256": base_sha,
               "file": str(path.relative_to(ROOT).as_posix()) if path.is_relative_to(ROOT) else str(path), "sha256": sha256_file(path),
               "bytes": path.stat().st_size, "rows_changed": int(changed.sum()), "candidate_rows": int(len(c)), "upgraded_rows": int(c["prod_matched"].sum()),
               "squared_change": float((diff ** 2).sum()), "max_move_s": float(diff.max()), "min_move_s": float(diff.min()),
               "mean_move_s": float(diff[changed].mean()) if changed.any() else 0.0,
               "lirf_population_guard_rows_changed": int((changed & fixed).sum()), "other_rows_identical_to_base": bool(out.loc[~changed].equals(b.loc[~changed])),
               "offsets": off, "by_airport": moved.groupby("airport")["move"].agg(["size", "mean", "max", "min"]).round(1).to_dict(orient="index"),
               "by_day_top": moved.groupby("uday")["move"].agg(["size", "mean"]).sort_values("size", ascending=False).head(8).round(1).to_dict(orient="index"),
               "rows": moved[["MVT_ID_mvt", "airport", "uday", "base", "T", "prod_matched", "new"]].round(1).to_dict(orient="records")}
    write_json(out_dir / f"vigorous-whistle_v{version}.receipt.json", receipt)
    print({k: receipt[k] for k in ["sha256", "bytes", "rows_changed", "candidate_rows", "upgraded_rows", "squared_change", "max_move_s", "min_move_s",
                                   "lirf_population_guard_rows_changed", "other_rows_identical_to_base"]}, receipt["by_airport"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matches", nargs="*", default=None, help="tables to build: 2025 and / or rank (default both)")
    ap.add_argument("--cache-only", action="store_true", help="--matches: fill the per-day cache for the staging days present, assemble nothing")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--control", default=None)
    ap.add_argument("--control-col", default=None)
    ap.add_argument("--control-sha256", default=None)
    ap.add_argument("--control-tag", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--base", default=V62[0])
    ap.add_argument("--base-sha256", default=V62[1])
    ap.add_argument("--version", type=int, default=None)
    ap.add_argument("--out", default=None, help="the dry run's directory (default runs/replica_union/dry_v<N>)")
    ap.add_argument("--dry-dir", default=None, help="--build: the dry run to match (default runs/replica_union/dry_v<N>)")
    ap.add_argument("--build", action="store_true", help="write runs/submissions/vigorous-whistle_v<version>.parquet; it must equal the dry run "
                    "byte for byte and must not exist yet")
    ap.add_argument("--user-decision", default=None)
    a = ap.parse_args()
    if a.matches is not None:
        step_matches(a.matches or ["2025", "rank"], a.cache_only)
    if a.score:
        score(a.control, a.control_col, a.control_sha256, a.control_tag)
    if a.dry_run:
        if a.version is None:
            raise ValueError("--dry-run needs --version.")
        out = OUT / f"dry_v{a.version}" if not a.out else ROOT / a.out
        if (ROOT / "runs/submissions").resolve() in [out.resolve(), *out.resolve().parents]:
            raise ValueError("A dry run never writes under runs/submissions.")
        dry_run(ROOT / a.base, a.base_sha256, a.version, out)
    if a.build:
        if a.version is None:
            raise ValueError("--build needs --version.")
        dry = (ROOT / a.dry_dir if a.dry_dir else OUT / f"dry_v{a.version}") / f"vigorous-whistle_v{a.version}.parquet"
        target = ROOT / f"runs/submissions/vigorous-whistle_v{a.version}.parquet"
        if target.exists():
            raise ValueError(f"{target.name} exists; never overwrite a submission file.")
        if not dry.exists() or not a.user_decision:
            raise ValueError("Run the dry run first and record the user's decision.")
        tmp = OUT / f"build_v{a.version}"
        if tmp.exists() and any(tmp.iterdir()):
            raise ValueError(f"{tmp} is not empty; never overwrite a build directory.")
        dry_run(ROOT / a.base, a.base_sha256, a.version, tmp)
        built = tmp / f"vigorous-whistle_v{a.version}.parquet"
        if sha256_file(built) != sha256_file(dry):
            raise ValueError("The rebuilt file differs from the dry run.")
        target.write_bytes(built.read_bytes())
        from prc2026_submission.contract import validate_submission
        sub = validate_submission(target, ROOT / "data/raw/submitting.parquet")
        cfg = read_json(CONFIG)
        sub.update({"version": a.version, "rounding": "numpy.rint, nearest integer with ties to even; no clipping",
                    "provenance": {"audit_used": False, "dry_run": False, "experiment_id": cfg["experiment_id"], "selected_candidate": cfg["candidate"]["name"],
                                   "selected_by": "the declared rule (docs/replica_union/results_v1.json)", "official_scores_used": "none",
                                   "user_decision": a.user_decision, "declaration": {"configs/replica_union_v1.json": sha256_file(CONFIG)},
                                   "code": {"tools/run_replica_union.py": sha256_file(Path(__file__))},
                                   "base": {"path": a.base, "sha256": a.base_sha256},
                                   "recipe": "the base chain, then the replica-union layer last (rows newly gate-traced at their known stand on the adsb.lol prod-0 + staging-0 union; 0.5 toward union taxi + per-airport offset fitted on the twelve 2025 months)"}})
        write_json(target.with_suffix(".receipt.json"), sub)
        receipt = read_json(tmp / f"vigorous-whistle_v{a.version}.receipt.json")
        receipt.update({"kind": "submission build", "file": str(target.relative_to(ROOT).as_posix()), "sha256": sha256_file(target),
                        "dry_run_sha256": sha256_file(dry)})
        DOCS.mkdir(parents=True, exist_ok=True)
        write_json(DOCS / f"export_v{a.version}.json", receipt)
        print("built", target.name, receipt["sha256"])


if __name__ == "__main__":
    main()
