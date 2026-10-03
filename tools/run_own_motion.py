"""Own-motion bound (configs/own_motion_v1.json): the aircraft's own ground motion after its inbound blocked in at the departure's stand.

Rows the shipped ADS-B matcher does not trace from the gate (dark or unmatched) at EDDF / EGLL / LFPG: the departure's aircraft is
identified by ICAO address (runs/intruder_identity/ids.parquet); its latest identified arrival at the airport blocked in at the same
stand 10 min to 18 h before take-off; its first adsb.lol ground point at >= 3 kt after that in-block + 120 s lies B seconds before
take-off (60 < B <= 2,700). The off-block is not earlier than that motion: label >= B. Where the chain's prediction p < B the value is
p + w (B + o(airport) - p), o = the fitting months' mean of label - B shrunk to the pooled mean. Applied last, on v61's chain.

--score     development (v61 proxy) and twelve-month tables (after the intruder bound and identity extension), month-blocked, with the
            declared rule -> docs/own_motion/results_v1.json
--dry-run   label-free ranking file on a base file (default the uploaded v61, read only) -> runs/own_motion/dry_v<version>/
--build     at the user's word: rerun the dry run into runs/own_motion/build_v<version>, require byte identity with the dry run, write
            runs/submissions/vigorous-whistle_v<version>.parquet (never overwrite) and docs/own_motion/export_v<version>.json
Nothing is uploaded.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026_adsb.features import DEPARTURE_COLUMNS, departure_features  # noqa: E402
import run_intruder_bound as R  # noqa: E402
import run_intruder_identity as I  # noqa: E402

CONFIG = ROOT / "configs/own_motion_v1.json"
DEV = ROOT / "runs/dig/census61/dev_v61.parquet"
DEV_SHA = "9739d9c5bfa6258affaf8f9a6db3c001a46c1e1e92308bebcdcd6920c51448de"
IDS = ROOT / "runs/intruder_identity/ids.parquet"
SURFACE = ROOT / "data/external/adsblol/surface"
MATCHES = ROOT / "data/external/adsblol/matches"
RANKING = "data/raw/ranking.parquet"   # tools/final_chain_overrides.py --raw
AIRPORTS = ["EDDF", "EGLL", "LFPG"]
INB_MIN, INB_MAX, AFTER_IN, GS_MIN, B_MIN = 600.0, 64800.0, 120.0, 3.0, 60.0
SHRINK = 20.0
V61 = ("runs/submissions/vigorous-whistle_v61.parquet", "fe137f092fa68a3f838ed0cb5b3cc5d7e995707790accecef4fb3649dd6af4e8")


def triggers(paths: list[Path]) -> pd.DataFrame:
    """Label-free: MVT_ID_mvt, airport, B for every departure meeting the trigger (prediction not yet involved)."""
    ids = pd.read_parquet(IDS, columns=["MVT_ID_mvt", "PHASE_mvt", "airport", "icao"])
    ids = ids[ids["icao"].notna() & ids["airport"].isin(AIRPORTS)]
    cols = sorted(set(DEPARTURE_COLUMNS + ["MVT_ID_mvt", "PHASE_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "STAND_mvt"]))
    raw = pd.concat([pd.read_parquet(p, columns=cols) for p in paths], ignore_index=True)
    dep_all = raw[raw["PHASE_mvt"].eq("DEP") & raw["ADEP_mvt"].isin(AIRPORTS)]
    # matcher state (gate-traced or not), per month of take-off
    gates = []
    for month, g in dep_all.groupby(dep_all["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m")):
        mfile = MATCHES / f"matches2_{month}.parquet"
        matches = pd.read_parquet(mfile) if mfile.exists() else pd.DataFrame()
        gates.append(departure_features(g[DEPARTURE_COLUMNS], matches, pd.read_parquet(MATCHES / "stands.parquet"))[["MVT_ID_mvt", "adsb_gate"]])
    gate = pd.concat(gates)
    dep = dep_all[["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "STAND_mvt"]].rename(columns={"ADEP_mvt": "airport"})
    dep = dep.merge(ids[ids["PHASE_mvt"].eq("DEP")][["MVT_ID_mvt", "icao"]], on="MVT_ID_mvt").merge(gate, on="MVT_ID_mvt", how="left")
    dep = dep[dep["adsb_gate"].ne(1) & dep["STAND_mvt"].notna()].copy()
    arr = raw[raw["PHASE_mvt"].eq("ARR") & raw["ADES_mvt"].isin(AIRPORTS)][["MVT_ID_mvt", "BLOCK_TIME_UTC_mvt", "STAND_mvt"]]
    arr = arr.merge(ids[ids["PHASE_mvt"].eq("ARR")][["MVT_ID_mvt", "airport", "icao"]], on="MVT_ID_mvt")
    arr = arr.rename(columns={"BLOCK_TIME_UTC_mvt": "in_blk", "STAND_mvt": "in_stand"}).dropna(subset=["in_blk"])[["airport", "icao", "in_blk", "in_stand"]]
    arr["in_blk"] = arr["in_blk"].astype("datetime64[us, UTC]")
    dep["MVT_TIME_UTC_mvt"] = dep["MVT_TIME_UTC_mvt"].astype("datetime64[us, UTC]")
    d = pd.merge_asof(dep.sort_values("MVT_TIME_UTC_mvt"), arr.sort_values("in_blk"), left_on="MVT_TIME_UTC_mvt", right_on="in_blk",
                      by=["icao", "airport"], direction="backward")
    rel = (d["MVT_TIME_UTC_mvt"] - d["in_blk"]).dt.total_seconds()
    d = d[(rel >= INB_MIN) & (rel <= INB_MAX) & d["in_stand"].eq(d["STAND_mvt"])].copy()
    d["lo"] = (d["in_blk"] + pd.Timedelta(seconds=AFTER_IN)).astype("datetime64[us, UTC]")
    d["day"] = d["MVT_TIME_UTC_mvt"].dt.strftime("%Y%m%d")
    out = []
    for day, g in d.groupby("day"):
        prev = (pd.Timestamp(day) - pd.Timedelta(days=1)).strftime("%Y%m%d")
        pts = []
        for dd in (prev, day):
            f = SURFACE / f"surface_{dd}.parquet"
            if f.exists():
                pts.append(pd.read_parquet(f, columns=["icao", "airport", "time", "ground", "gs"]))
        if not pts:
            continue
        p = pd.concat(pts)
        p = p[p["ground"].eq(True) & (p["gs"] >= GS_MIN) & p["icao"].isin(set(g["icao"])) & p["airport"].isin(AIRPORTS)].copy()
        p["tm"] = pd.to_datetime(p["time"], unit="s", utc=True).astype("datetime64[us, UTC]")
        m = pd.merge_asof(g.sort_values("lo"), p.sort_values("tm")[["icao", "airport", "tm"]], left_on="lo", right_on="tm",
                          by=["icao", "airport"], direction="forward")
        out.append(m)
    t = pd.concat(out, ignore_index=True)
    t["B"] = (t["MVT_TIME_UTC_mvt"] - t["tm"]).dt.total_seconds()
    t = t[t["B"] > B_MIN]
    return t[["MVT_ID_mvt", "airport", "B", "adsb_gate"]].reset_index(drop=True)


def candidates(tab: pd.DataFrame, pred: str, trig: pd.DataFrame, b_max: float) -> pd.DataFrame:
    c = tab.merge(trig[["MVT_ID_mvt", "B"]], on="MVT_ID_mvt")
    keep = (c["airport"].isin(AIRPORTS) & (c["B"] <= b_max) & (c[pred] < c["B"]) & ~c["cls"].eq("LIRF_pop") & ~c["MVT_ID_mvt"].isin(R.GUARD))
    return c[keep].copy()


def fit(fitting: pd.DataFrame) -> dict:
    over = fitting["label"] - fitting["B"]
    pooled = float(over.mean())
    g = over.groupby(fitting["airport"]).agg(["sum", "size"])
    o = {a: float((r["sum"] + SHRINK * pooled) / (r["size"] + SHRINK)) for a, r in g.iterrows()}
    return {"pooled": pooled, "offsets": o, "rows": {a: int(r["size"]) for a, r in g.iterrows()},
            "validity": {a: float((h["label"] >= h["B"] - 60).mean()) for a, h in fitting.groupby("airport")}}


def values(cand: pd.DataFrame, pred: str, model: dict, w: float) -> pd.DataFrame:
    c = cand.copy()
    o = c["airport"].map(model["offsets"]).fillna(model["pooled"]).to_numpy(dtype=float)
    c["value"] = c[pred] + w * (c["B"] + o - c[pred])
    return c


def twelve_after() -> pd.DataFrame:
    """rows12 `stack` -> accepted intruder bound -> identity extension, both month-blocked as in their governing runs; column `after2`."""
    acc, icfg = read_json(R.CONFIG)["candidate"], read_json(I.CONFIG)["candidate"]
    files = I.training_files()
    rdep, rarr = R.movements(files)
    dep, arr = I.movements(files)
    twelve, p12 = R.read_table("twelve", rdep)
    c12acc = R.candidates(twelve, p12, rdep, rarr, acc["G_s"])
    months = sorted(twelve["month"].unique())
    after12 = pd.concat([I.accepted_after(twelve, p12, rdep, rarr, c12acc, m, acc)[0] for m in months])
    cid12 = I.candidates(after12, "after", dep, arr, icfg["G_s"])
    parts = []
    for m in months:
        s = after12[after12["month"] == m].copy()
        imodel = I.fit(cid12[cid12["month"] != m], R.fit(c12acc[c12acc["month"] != m])["airports"])
        ci = I.values(I.candidates(s, "after", dep, arr, icfg["G_s"]), "after", imodel)
        s["after2"] = s["after"].astype(float)
        s = s.set_index("MVT_ID_mvt")
        s.loc[ci["MVT_ID_mvt"].to_numpy(), "after2"] = ci["value"].to_numpy()
        parts.append(s.reset_index())
    return pd.concat(parts, ignore_index=True)


def dev_table() -> pd.DataFrame:
    if sha256_file(DEV) != DEV_SHA:
        raise ValueError("The v61 development proxy differs from the declared file.")
    d = pd.read_parquet(DEV, columns=["MVT_ID_mvt", "month", "airport", "label", "cls", "v61"])
    want = read_json(CONFIG)["control"]["development_month_rmse_s"]
    got = {m: float(np.sqrt(((g["label"] - g["v61"]) ** 2).mean())) for m, g in d.groupby("month")}
    if any(abs(got[m] - want[m]) > 1e-9 for m in want):
        raise ValueError(f"The control does not reproduce census61's month RMSEs: {got}")
    tko = pd.concat([pd.read_parquet(f, columns=["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt"]) for f in I.training_files()])
    tko = tko[tko["PHASE_mvt"].eq("DEP")]
    d = d.merge(tko[["MVT_ID_mvt", "MVT_TIME_UTC_mvt"]], on="MVT_ID_mvt")
    d["day"] = d["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d")
    return d.drop(columns=["MVT_TIME_UTC_mvt"]), got


def score() -> None:
    cfg = read_json(CONFIG)
    rule = cfg["rule"]
    trig = triggers(I.training_files())
    t12 = twelve_after()
    dev, dev_rmse = dev_table()
    specs = [cfg["candidate"]] + cfg["sensitivity_reported_not_selectable"]
    report = {"experiment_id": cfg["experiment_id"], "config_sha256": sha256_file(CONFIG), "computed_at_utc": utc_now(),
              "inputs": {"dev": sha256_file(DEV), "twelve": sha256_file(R.TWELVE), "ids": sha256_file(IDS)},
              "control_month_rmse_s": dev_rmse, "trigger_rows_2025": int(len(trig)),
              "trigger_rows_by_airport": trig["airport"].value_counts().to_dict(), "candidates": {}}
    for spec in specs:
        rng = np.random.default_rng(20261003)
        c12 = candidates(t12, "after2", trig, spec["B_max_s"])
        cdv = candidates(dev, "v61", trig, spec["B_max_s"])
        out = {"development": {}, "twelve": {}}
        for name, tab, pred, cand, months in [("development", dev, "v61", cdv, R.DEV_MONTHS),
                                              ("twelve", t12, "after2", c12, sorted(t12["month"].unique()))]:
            for m in months:
                model = fit(c12[c12["month"] != m])
                c = values(cand[cand["month"] == m], pred, model, spec["w"])
                s = tab[tab["month"] == m]
                new = s[pred].to_numpy(dtype=float).copy()
                pos = pd.Series(np.arange(len(s)), index=s["MVT_ID_mvt"].to_numpy())
                new[pos.loc[c["MVT_ID_mvt"]].to_numpy()] = c["value"].to_numpy()
                pct, lo, hi = R.interval(s, pred, new, rng)
                out[name][m] = {"pct": pct, "lo": lo, "hi": hi, "rows": int(len(c)), "by_airport_rows": c["airport"].value_counts().to_dict(),
                                "valid_share": float((c["label"] >= c["B"] - 60).mean()) if len(c) else None,
                                "squared_change": float(((c["value"] - c[pred]) ** 2).sum()),
                                "gain_over_squared_change": float((((c["label"] - c[pred]) ** 2) - (c["label"] - c["value"]) ** 2).sum()
                                                                  / max(((c["value"] - c[pred]) ** 2).sum(), 1e-9)),
                                "offsets": model["offsets"]}
        dv, tw = out["development"], out["twelve"]
        eq = float(np.mean([dv[m]["pct"] for m in rule["equal_footing"]]))
        checks = {"1_max_development_month_pct": max(dv[m]["pct"] for m in R.DEV_MONTHS),
                  "1_no_month_worse_than_0.005pct": all(dv[m]["pct"] <= rule["max_month_worsening_pct"] for m in R.DEV_MONTHS),
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
        report["candidates"][spec["name"]] = out
        print(spec["name"], "dev:", " ".join(f"{m[5:]} {dv[m]['pct']:+.4f}% [{dv[m]['lo']:+.3f},{dv[m]['hi']:+.3f}] n{dv[m]['rows']}" for m in R.DEV_MONTHS))
        print("   twelve:", " ".join(f"{m[5:]} {v['pct']:+.3f}%" for m, v in tw.items()))
        print("  ", checks)
    name = cfg["candidate"]["name"]
    report["selected"] = name if report["candidates"][name]["checks"]["accepted"] else None
    (ROOT / "docs/own_motion").mkdir(parents=True, exist_ok=True)
    write_json(ROOT / "docs/own_motion/results_v1.json", report)
    print("selected:", report["selected"])


def dry_run(base: Path, base_sha: str, version: int, out_dir: Path) -> None:
    cfg = read_json(CONFIG)
    if sha256_file(base) != base_sha:
        raise ValueError("The base file's SHA-256 differs from the expected one.")
    spec = cfg["candidate"]
    trig12 = triggers(I.training_files())
    t12 = twelve_after()
    model = fit(candidates(t12, "after2", trig12, spec["B_max_s"]))
    rk_path = ROOT / RANKING
    rk = pd.read_parquet(rk_path, columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADEP_flt", "AIRCRAFT_TYPE_mvt"])
    rk = rk[rk["PHASE_mvt"].eq("DEP")]
    pop = rk["ADEP_mvt"].eq("LIRF") & rk["ADEP_flt"].isna() & rk["AIRCRAFT_TYPE_mvt"].isna()
    b = pd.read_parquet(base)
    if list(b.columns) != ["MVT_ID_mvt", "TAXITIME_SEC_mvt"] or len(b) != len(rk):
        raise ValueError("The base file is not a full submission.")
    tab = b.merge(rk[["MVT_ID_mvt", "ADEP_mvt"]].assign(cls=np.where(pop, "LIRF_pop", "x")).rename(columns={"ADEP_mvt": "airport"}), on="MVT_ID_mvt")
    tab["base"] = tab["TAXITIME_SEC_mvt"].astype(float)
    trig = triggers([rk_path])
    c = values(candidates(tab, "base", trig, spec["B_max_s"]), "base", model, spec["w"])
    out = b.copy()
    new = pd.Series(np.rint(c["value"].to_numpy()).astype(out["TAXITIME_SEC_mvt"].dtype), index=c["MVT_ID_mvt"].to_numpy())
    idx = out["MVT_ID_mvt"].map(new)
    out["TAXITIME_SEC_mvt"] = np.where(idx.notna(), idx, out["TAXITIME_SEC_mvt"]).astype(b["TAXITIME_SEC_mvt"].dtype)
    if not out["MVT_ID_mvt"].equals(b["MVT_ID_mvt"]):
        raise ValueError("Row order changed.")
    changed = out["TAXITIME_SEC_mvt"].ne(b["TAXITIME_SEC_mvt"])
    fixed = b["MVT_ID_mvt"].isin(set(rk.loc[pop, "MVT_ID_mvt"]) | set(R.GUARD) | set(rk.loc[rk["ADEP_mvt"].eq("LIRF"), "MVT_ID_mvt"]))
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
               "written_at_utc": utc_now(), "kind": "dry run, not a submission", "ids_sha256": sha256_file(IDS),
               "base": str(base.relative_to(ROOT)) if base.is_relative_to(ROOT) else str(base), "base_sha256": base_sha,
               "file": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), "sha256": sha256_file(path),
               "bytes": path.stat().st_size, "rows_changed": int(changed.sum()), "squared_change": float((diff ** 2).sum()),
               "max_move_s": float(diff.max()), "min_move_s": float(diff.min()), "mean_move_s": float(diff[changed].mean()) if changed.any() else 0.0,
               "lirf_population_guard_rows_changed": int((changed & fixed).sum()), "other_rows_identical_to_base": bool(out.loc[~changed].equals(b.loc[~changed])),
               "model": model, "ranking_trigger_rows": int(len(trig)),
               "by_airport": moved.groupby("airport")["move"].agg(["size", "mean", "max"]).round(1).to_dict(orient="index"),
               "rows": moved[["MVT_ID_mvt", "airport", "base", "B", "new"]].round(1).to_dict(orient="records")}
    write_json(out_dir / f"vigorous-whistle_v{version}.receipt.json", receipt)
    print({k: receipt[k] for k in ["sha256", "bytes", "rows_changed", "squared_change", "max_move_s", "min_move_s", "lirf_population_guard_rows_changed",
                                   "other_rows_identical_to_base"]}, receipt["by_airport"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--base", default=V61[0])
    ap.add_argument("--base-sha256", default=V61[1])
    ap.add_argument("--version", type=int, default=62)
    ap.add_argument("--out", default=None, help="the dry run's directory (default runs/own_motion/dry_v<N>)")
    ap.add_argument("--ids", default=None, help="final file: the identity table (default runs/intruder_identity/ids.parquet)")
    ap.add_argument("--matches-dir", default=None, help="final file: directory holding matches2_<YYYY-MM>.parquet for the new file's months")
    ap.add_argument("--build", action="store_true", help="at the user's word: write runs/submissions/vigorous-whistle_v<version>.parquet; "
                    "it must equal the dry run byte for byte and must not exist yet")
    ap.add_argument("--user-decision", default=None)
    import final_chain_overrides
    final_chain_overrides.add_arguments(ap)
    a = ap.parse_args()
    final_chain_overrides.apply(a)
    global IDS, MATCHES
    if a.ids:
        IDS = ROOT / final_chain_overrides._rel(a.ids)
    if a.matches_dir:
        MATCHES = ROOT / final_chain_overrides._rel(a.matches_dir)
    if a.score:
        score()
    if a.build:
        dry = ROOT / f"runs/own_motion/dry_v{a.version}/vigorous-whistle_v{a.version}.parquet"
        target = ROOT / f"runs/submissions/vigorous-whistle_v{a.version}.parquet"
        if target.exists():
            raise ValueError(f"{target.name} exists; never overwrite a submission file.")
        if not dry.exists() or not a.user_decision:
            raise ValueError("Run the dry run first and record the user's decision.")
        tmp = ROOT / f"runs/own_motion/build_v{a.version}"
        if tmp.exists() and any(tmp.iterdir()):
            raise ValueError(f"{tmp} is not empty; never overwrite a build directory.")
        dry_run(ROOT / a.base, a.base_sha256, a.version, tmp)
        built = tmp / f"vigorous-whistle_v{a.version}.parquet"
        if sha256_file(built) != sha256_file(dry):
            raise ValueError("The rebuilt file differs from the dry run.")
        target.write_bytes(built.read_bytes())
        from prc2026_submission.contract import validate_submission
        sub = validate_submission(target, ROOT / (final_chain_overrides._rel(a.template) if a.template else "data/raw/submitting.parquet"))
        cfg = read_json(CONFIG)
        sub.update({"version": a.version, "rounding": "numpy.rint, nearest integer with ties to even; no clipping",
                    "provenance": {"audit_used": False, "dry_run": False, "experiment_id": cfg["experiment_id"], "selected_candidate": cfg["candidate"]["name"],
                                   "selected_by": "the declared rule (docs/own_motion/results_v1.json)", "official_scores_used": "none",
                                   "user_decision": a.user_decision, "declaration": {"configs/own_motion_v1.json": sha256_file(CONFIG)},
                                   "code": {"tools/run_own_motion.py": sha256_file(Path(__file__))}, "ids_sha256": sha256_file(IDS),
                                   "base": {"path": a.base, "sha256": a.base_sha256},
                                   "recipe": "v61's chain, then the own-motion bound last (EDDF / EGLL / LFPG; value take-off - the aircraft's first ground motion after its inbound at the same stand + offset fitted on the twelve 2025 months)"}})
        write_json(target.with_suffix(".receipt.json"), sub)
        receipt = read_json(tmp / f"vigorous-whistle_v{a.version}.receipt.json")
        receipt.update({"kind": "submission build at the user's word", "file": str(target.relative_to(ROOT)), "sha256": sha256_file(target),
                        "dry_run_sha256": sha256_file(dry)})
        write_json(ROOT / f"docs/own_motion/export_v{a.version}.json", receipt)
        print("built", target.name, receipt["sha256"])
    if a.dry_run:
        out = ROOT / f"runs/own_motion/dry_v{a.version}" if not a.out else ROOT / final_chain_overrides._rel(a.out)
        if (ROOT / "runs/submissions").resolve() in [out.resolve(), *out.resolve().parents]:
            raise ValueError("A dry run never writes under runs/submissions.")
        dry_run(ROOT / a.base, a.base_sha256, a.version, out)


if __name__ == "__main__":
    main()
