"""Identity extension of the intruder stand bound (configs/intruder_identity_v1.json).

The accepted layer (tools/run_intruder_bound.py) uses only intruders of a different aircraft type. Here arrivals and departures at
EDDF / EGLL / LFPG are identified by their ICAO 24-bit address from the adsb.lol day files (nearest point of the same callsign at the
same airport within 600 s of the landing / take-off time); a same-type intruder Y != X that blocked in at our stand after our own inbound
X and before our predicted off-block proves we left earlier: label > B = take-off - Y's in-block. Applied after the accepted layer.

--build-ids  identity table -> runs/intruder_identity/ids.parquet (2025 months and the ranking file)
--score      development and twelve-month tables, month-blocked, with the declared rule -> docs/intruder_identity/results_v1.json
--dry-run    ranking: base (v60's pre-intruder file) -> accepted layer (must equal v60) -> this extension -> runs/intruder_identity/dry_v<N>/
Nothing is written under runs/submissions and nothing is uploaded.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402
import run_intruder_bound as R  # noqa: E402

CONFIG = ROOT / "configs/intruder_identity_v1.json"
IDS = ROOT / "runs/intruder_identity/ids.parquet"
SURFACE = ROOT / "data/external/adsblol/surface"
KEYMAP = ROOT / "runs/rematch_nonm/key_map.parquet"
ID_AIRPORTS = ["EDDF", "EGLL", "LFPG", "LTFM"]
AIRPORTS = ["EDDF", "EGLL", "LFPG"]
TOL = 600
SHRINK = 20.0
V60 = ("runs/submissions/vigorous-whistle_v60.parquet", "30021455005a712c241fd57354c7a57c65ea8e81cd7a440e7175d0f27598b95f")


def training_files() -> list[Path]:
    return [Path(f) for f in sorted(glob.glob(str(ROOT / "data/raw/training_2025-*.parquet")))]


def build_ids() -> None:
    cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "CALLSIGN_flt", "FLIGHT_mvt"]
    mv = pd.concat([pd.read_parquet(f, columns=cols) for f in training_files() + [ROOT / "data/raw/ranking.parquet"]])
    mv["airport"] = np.where(mv["PHASE_mvt"].eq("ARR"), mv["ADES_mvt"], mv["ADEP_mvt"])
    mv = mv[mv["airport"].isin(ID_AIRPORTS)]
    km = pd.read_parquet(KEYMAP)[["ADEP_mvt", "FLIGHT_mvt", "key"]].rename(columns={"ADEP_mvt": "airport"})
    mv = mv.merge(km, on=["airport", "FLIGHT_mvt"], how="left")
    mv["cs"] = mv["CALLSIGN_flt"].fillna(mv["key"]).str.upper().str.strip()
    mv = mv[mv["cs"].notna()].copy()
    mv["t"] = (mv["MVT_TIME_UTC_mvt"] - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
    mv["day"] = mv["MVT_TIME_UTC_mvt"].dt.strftime("%Y%m%d")
    out, used = [], {}
    for day, g in mv.groupby("day"):
        p = SURFACE / f"surface_{day}.parquet"
        if not p.exists():
            continue
        used[p.name] = sha256_file(p)
        x = pd.read_parquet(p, columns=["icao", "registration", "type", "airport", "time", "callsign"])
        x = x[x["airport"].isin(ID_AIRPORTS) & x["callsign"].notna()].copy()
        x["cs"] = x["callsign"].str.upper().str.strip()
        m = pd.merge_asof(g.sort_values("t")[["MVT_ID_mvt", "PHASE_mvt", "airport", "cs", "t"]], x.sort_values("time")[["airport", "cs", "time", "icao", "registration", "type"]],
                          left_on="t", right_on="time", by=["airport", "cs"], direction="nearest", tolerance=TOL)
        out.append(m[["MVT_ID_mvt", "PHASE_mvt", "airport", "icao", "registration", "type"]].assign(dt=m["time"] - m["t"]))
    IDS.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(out).to_parquet(IDS, index=False)
    write_json(IDS.with_suffix(".receipt.json"), {"written_at_utc": utc_now(), "sha256": sha256_file(IDS), "tolerance_s": TOL,
                                                  "keymap_sha256": sha256_file(KEYMAP), "surface_files": used})
    print("ids", sha256_file(IDS))


def movements(paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = pd.read_parquet(IDS)
    dep_id = ids[ids["PHASE_mvt"].eq("DEP")].set_index("MVT_ID_mvt")["icao"]
    arr_id = ids[ids["PHASE_mvt"].eq("ARR")].set_index("MVT_ID_mvt")["icao"]
    cols = ["MVT_ID_mvt", "PHASE_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt"]
    raw = pd.concat([pd.read_parquet(p, columns=cols) for p in paths])
    dep = raw[raw["PHASE_mvt"].eq("DEP")][["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt"]].copy()
    dep["X"] = dep["MVT_ID_mvt"].map(dep_id)
    arr = raw[raw["PHASE_mvt"].eq("ARR")].rename(columns={"ADES_mvt": "airport", "BLOCK_TIME_UTC_mvt": "in_blk", "AIRCRAFT_TYPE_mvt": "arr_type"})
    arr = arr[arr["airport"].isin(AIRPORTS)][["MVT_ID_mvt", "airport", "STAND_mvt", "in_blk", "arr_type"]].dropna(subset=["in_blk", "STAND_mvt"]).copy()
    arr["Y"] = arr["MVT_ID_mvt"].map(arr_id)
    arr["in_blk"] = arr["in_blk"].astype("datetime64[us, UTC]")
    arr = arr.sort_values(["airport", "STAND_mvt", "in_blk"])
    gb = arr.groupby(["airport", "STAND_mvt"])
    arr["prev_type"] = gb["arr_type"].shift(1)
    arr["prev_Y"] = gb["Y"].shift(1)
    return dep, arr.drop(columns=["MVT_ID_mvt"]).sort_values("in_blk")


def candidates(tab: pd.DataFrame, pred: str, dep: pd.DataFrame, arr: pd.DataFrame, gap_max: float) -> pd.DataFrame:
    d = tab[tab["airport"].isin(AIRPORTS)].merge(dep, on="MVT_ID_mvt").dropna(subset=["STAND_mvt"])
    d["pred_off"] = (d["MVT_TIME_UTC_mvt"] - pd.to_timedelta(d[pred].round(), unit="s")).astype("datetime64[us, UTC]")
    m = pd.merge_asof(d.sort_values("pred_off"), arr, left_on="pred_off", right_on="in_blk", by=["airport", "STAND_mvt"], direction="backward")
    m["gap"] = (m["pred_off"] - m["in_blk"]).dt.total_seconds()
    m["B"] = (m["MVT_TIME_UTC_mvt"] - m["in_blk"]).dt.total_seconds()
    accepted = m["arr_type"].notna() & m["arr_type"].ne(m["AIRCRAFT_TYPE_mvt"]) & m["prev_type"].eq(m["AIRCRAFT_TYPE_mvt"])
    keep = (m["X"].notna() & m["Y"].notna() & m["Y"].ne(m["X"]) & m["prev_Y"].eq(m["X"]) & ~accepted & (m["gap"] > 0) & (m["gap"] <= gap_max)
            & ~m["cls"].eq("LIRF_pop") & ~m["MVT_ID_mvt"].isin(R.GUARD))
    m = m[keep].copy()
    m["band"] = pd.cut(m["gap"], R.BANDS, labels=False)
    return m


def fit(fitting: pd.DataFrame, accepted_airports: list[str]) -> dict:
    airports = [a for a in AIRPORTS if a in accepted_airports]
    f = fitting[fitting["airport"].isin(airports)].assign(over=lambda x: x["label"] - x["B"])
    g0 = float(f["over"].mean())
    am = f.groupby("airport")["over"].agg(["sum", "size"])
    am = ((am["sum"] + SHRINK * g0) / (am["size"] + SHRINK)).to_dict()
    gb = f.groupby(["airport", "band"])["over"].agg(["sum", "size"]).reset_index()
    gb["o"] = (gb["sum"] + SHRINK * gb["airport"].map(am)) / (gb["size"] + SHRINK)
    return {"airports": airports, "pooled": g0, "airport_mean": am, "offsets": {f"{a}|{int(b)}": float(o) for a, b, o in gb[["airport", "band", "o"]].itertuples(index=False)}}


def values(cand: pd.DataFrame, pred: str, model: dict) -> pd.DataFrame:
    c = cand[cand["airport"].isin(model["airports"])].copy()
    off = np.array([model["offsets"].get(f"{a}|{int(b)}", model["airport_mean"].get(a, model["pooled"])) for a, b in zip(c["airport"], c["band"])])
    c["value"] = np.maximum(c["B"] + off, c[pred])
    return c


def accepted_after(tab: pd.DataFrame, pred: str, rdep, rarr, c12acc: pd.DataFrame, month: str, spec: dict) -> tuple[pd.DataFrame, dict]:
    """The scored month's table with column `after` = the accepted intruder layer applied month-blocked."""
    model = R.fit(c12acc[c12acc["month"] != month])
    s = tab[tab["month"] == month].copy()
    c = R.values(R.candidates(s, pred, rdep, rarr, spec["G_s"]), pred, model, spec["w"])
    s["after"] = s[pred].astype(float)
    s = s.set_index("MVT_ID_mvt")
    s.loc[c["MVT_ID_mvt"].to_numpy(), "after"] = c["value"].to_numpy()
    return s.reset_index(), model


def score() -> None:
    cfg, acc = read_json(CONFIG), read_json(R.CONFIG)["candidate"]
    files = training_files()
    rdep, rarr = R.movements(files)
    dep, arr = movements(files)
    twelve, p12 = R.read_table("twelve", rdep)
    dev, pdv = R.read_table("dev", rdep)
    c12acc = R.candidates(twelve, p12, rdep, rarr, acc["G_s"])
    months12 = sorted(twelve["month"].unique())
    after12 = pd.concat([accepted_after(twelve, p12, rdep, rarr, c12acc, m, acc)[0] for m in months12])
    cid12 = candidates(after12, "after", dep, arr, cfg["candidate"]["G_s"])
    after_dev = pd.concat([accepted_after(dev, pdv, rdep, rarr, c12acc, m, acc)[0] for m in R.DEV_MONTHS])
    cid_dev = candidates(after_dev, "after", dep, arr, cfg["candidate"]["G_s"])
    rng = np.random.default_rng(20261003)
    out = {}
    for name, tab, cid, months in [("development", after_dev, cid_dev, R.DEV_MONTHS), ("twelve", after12, cid12, months12)]:
        out[name] = {}
        for m in months:
            model = fit(cid12[cid12["month"] != m], R.fit(c12acc[c12acc["month"] != m])["airports"])
            c = values(cid[cid["month"] == m], "after", model)
            s = tab[tab["month"] == m]
            new = s["after"].to_numpy(dtype=float).copy()
            pos = pd.Series(np.arange(len(s)), index=s["MVT_ID_mvt"].to_numpy())
            new[pos.loc[c["MVT_ID_mvt"]].to_numpy()] = c["value"].to_numpy()
            pct, lo, hi = R.interval(s, "after", new, rng)
            out[name][m] = {"pct": pct, "lo": lo, "hi": hi, "rows": int(len(c)), "airports": model["airports"],
                            "valid_share": float((c["label"] >= c["B"] - 60).mean()) if len(c) else None,
                            "squared_change": float(((c["value"] - c["after"]) ** 2).sum())}
    dv, tw = out["development"], out["twelve"]
    eq = float(np.mean([dv[m]["pct"] for m in R.EQUAL]))
    checks = {"equal_footing_mean_pct": eq, "mean_ok": eq <= -0.05, "no_development_month_worse": all(dv[m]["pct"] <= 0 for m in R.DEV_MONTHS),
              "equal_intervals_below_zero": int(sum(dv[m]["hi"] < 0 for m in R.EQUAL)),
              "twelve_months_better": int(sum(v["pct"] < 0 for v in tw.values())), "twelve_intervals_below_zero": int(sum(v["hi"] < 0 for v in tw.values()))}
    checks["accepted"] = bool(checks["mean_ok"] and checks["no_development_month_worse"] and checks["equal_intervals_below_zero"] >= 2 and checks["twelve_months_better"] >= 10)
    report = {"experiment_id": cfg["experiment_id"], "config_sha256": sha256_file(CONFIG), "computed_at_utc": utc_now(), "ids_sha256": sha256_file(IDS),
              "inputs": {"dev": sha256_file(R.DEV), "twelve": sha256_file(R.TWELVE)}, "results": out, "checks": checks,
              "selected": cfg["candidate"]["name"] if checks["accepted"] else None}
    (ROOT / "docs/intruder_identity").mkdir(parents=True, exist_ok=True)
    write_json(ROOT / "docs/intruder_identity/results_v1.json", report)
    print("dev:", " ".join(f"{m[5:]} {v['pct']:+.4f}% [{v['lo']:+.3f},{v['hi']:+.3f}] n{v['rows']}" for m, v in dv.items()))
    print("twelve:", " ".join(f"{m[5:]} {v['pct']:+.3f}%" for m, v in tw.items()))
    print(checks, "selected:", report["selected"])


def dry_run(base: Path, base_sha: str, version: int, out_dir: Path) -> None:
    cfg, acc = read_json(CONFIG), read_json(R.CONFIG)["candidate"]
    if sha256_file(base) != base_sha:
        raise ValueError("The base file's SHA-256 differs from the expected one.")
    files = training_files()
    rdep, rarr = R.movements(files)
    dep, arr = movements(files)
    twelve, p12 = R.read_table("twelve", rdep)
    c12acc = R.candidates(twelve, p12, rdep, rarr, acc["G_s"])
    acc_model = R.fit(c12acc)
    after12 = pd.concat([accepted_after(twelve, p12, rdep, rarr, c12acc, m, acc)[0] for m in sorted(twelve["month"].unique())])
    model = fit(candidates(after12, "after", dep, arr, cfg["candidate"]["G_s"]), acc_model["airports"])
    rk_path = ROOT / "data/raw/ranking.parquet"
    rkdep, rkarr = R.movements([rk_path])
    idep, iarr = movements([rk_path])
    rk = pd.read_parquet(rk_path, columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADEP_flt", "AIRCRAFT_TYPE_mvt"])
    rk = rk[rk["PHASE_mvt"].eq("DEP")]
    pop = rk["ADEP_mvt"].eq("LIRF") & rk["ADEP_flt"].isna() & rk["AIRCRAFT_TYPE_mvt"].isna()
    b = pd.read_parquet(base)
    tab = b.merge(rk[["MVT_ID_mvt", "ADEP_mvt"]].assign(cls=np.where(pop, "LIRF_pop", "x")).rename(columns={"ADEP_mvt": "airport"}), on="MVT_ID_mvt")
    tab["month"] = "ranking"
    tab["base"] = tab["TAXITIME_SEC_mvt"].astype(float)
    ca = R.values(R.candidates(tab, "base", rkdep, rkarr, acc["G_s"]), "base", acc_model, acc["w"])
    after = b.copy()
    moved = pd.Series(np.rint(ca["value"].to_numpy()), index=ca["MVT_ID_mvt"].to_numpy())
    idx = after["MVT_ID_mvt"].map(moved)
    after["TAXITIME_SEC_mvt"] = np.where(idx.notna(), idx, after["TAXITIME_SEC_mvt"]).astype(b["TAXITIME_SEC_mvt"].dtype)
    v60 = pd.read_parquet(ROOT / V60[0])
    if sha256_file(ROOT / V60[0]) != V60[1] or not after["TAXITIME_SEC_mvt"].equals(v60["TAXITIME_SEC_mvt"]) or not after["MVT_ID_mvt"].equals(v60["MVT_ID_mvt"]):
        raise ValueError("The base plus the accepted layer does not reproduce v60.")
    tab2 = tab.drop(columns=["base"]).assign(after=after["TAXITIME_SEC_mvt"].astype(float).to_numpy())
    ci = values(candidates(tab2, "after", idep, iarr, cfg["candidate"]["G_s"]), "after", model)
    out = after.copy()
    new = pd.Series(np.rint(ci["value"].to_numpy()), index=ci["MVT_ID_mvt"].to_numpy())
    idx = out["MVT_ID_mvt"].map(new)
    out["TAXITIME_SEC_mvt"] = np.where(idx.notna(), idx, out["TAXITIME_SEC_mvt"]).astype(b["TAXITIME_SEC_mvt"].dtype)
    if out.loc[out["MVT_ID_mvt"].isin(R.GUARD), "TAXITIME_SEC_mvt"].tolist() != v60.loc[v60["MVT_ID_mvt"].isin(R.GUARD), "TAXITIME_SEC_mvt"].tolist():
        raise ValueError("A guard row moved.")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"vigorous-whistle_v{version}.parquet"
    out.to_parquet(path, index=False)
    diff = out["TAXITIME_SEC_mvt"].astype(float) - v60["TAXITIME_SEC_mvt"].astype(float)
    rows = ci.assign(new=np.rint(ci["value"]), move=np.rint(ci["value"]) - ci["after"])
    receipt = {"experiment_id": cfg["experiment_id"], "config_sha256": sha256_file(CONFIG), "written_at_utc": utc_now(), "kind": "dry run, not a submission",
               "base": str(base.relative_to(ROOT)), "base_sha256": base_sha, "reproduces_v60_before_extension": True, "ids_sha256": sha256_file(IDS),
               "file": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "rows_changed_vs_v60": int((diff != 0).sum()),
               "squared_change_vs_v60": float((diff ** 2).sum()), "max_move_s": float(diff.max()), "model": model,
               "by_airport": rows.groupby("airport")["move"].agg(["size", "mean", "max"]).round(1).to_dict(orient="index"),
               "rows": rows[["MVT_ID_mvt", "airport", "after", "B", "gap", "new"]].round(1).to_dict(orient="records")}
    write_json(out_dir / f"vigorous-whistle_v{version}.receipt.json", receipt)
    print({k: receipt[k] for k in ["sha256", "rows_changed_vs_v60", "squared_change_vs_v60", "max_move_s"]}, receipt["by_airport"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-ids", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--base", default="runs/stack_resid/dry_v60/vigorous-whistle_v60.parquet")
    ap.add_argument("--base-sha256", default="cff778e06fa7e402ee4170c64b6404e9c6bf394e11570ca79e128d18a3f45b1f")
    ap.add_argument("--version", type=int, default=61)
    ap.add_argument("--build", action="store_true", help="at the user's word: write runs/submissions/vigorous-whistle_v<version>.parquet; "
                    "it must equal the dry run byte for byte and must not exist yet")
    ap.add_argument("--user-decision", default=None)
    a = ap.parse_args()
    if a.build_ids:
        build_ids()
    if a.score:
        score()
    if a.build:
        dry = ROOT / f"runs/intruder_identity/dry_v{a.version}/vigorous-whistle_v{a.version}.parquet"
        target = ROOT / f"runs/submissions/vigorous-whistle_v{a.version}.parquet"
        if target.exists():
            raise ValueError(f"{target.name} exists; never overwrite a submission file.")
        if not dry.exists() or not a.user_decision:
            raise ValueError("Run the dry run first and record the user's decision.")
        tmp = ROOT / f"runs/intruder_identity/build_v{a.version}"
        dry_run(ROOT / a.base, a.base_sha256, a.version, tmp)
        built = tmp / f"vigorous-whistle_v{a.version}.parquet"
        if sha256_file(built) != sha256_file(dry):
            raise ValueError("The rebuilt file differs from the dry run.")
        target.write_bytes(built.read_bytes())
        from prc2026_submission.contract import validate_submission
        sub = validate_submission(target, ROOT / "data/raw/submitting.parquet")
        sub.update({"version": a.version, "rounding": "numpy.rint, nearest integer with ties to even; no clipping",
                    "provenance": {"audit_used": False, "dry_run": False, "experiment_id": "intruder_identity_v1", "selected_candidate": "identity extension",
                                   "selected_by": "the declared rule (accepted)", "official_scores_used": "none", "user_decision": a.user_decision,
                                   "declaration": {"configs/intruder_identity_v1.json": sha256_file(CONFIG)},
                                   "code": {"tools/run_intruder_identity.py": sha256_file(Path(__file__)), "tools/run_intruder_bound.py": sha256_file(ROOT / "tools/run_intruder_bound.py")},
                                   "base": {"path": a.base, "sha256": a.base_sha256}, "ids_sha256": sha256_file(IDS),
                                   "recipe": "v60's chain before the intruder bound -> the accepted different-type intruder bound (reproduces v60) -> the same-type identity extension"}})
        write_json(target.with_suffix(".receipt.json"), sub)
        receipt = read_json(tmp / f"vigorous-whistle_v{a.version}.receipt.json")
        receipt.update({"kind": "submission build at the user's word", "file": str(target.relative_to(ROOT)), "sha256": sha256_file(target)})
        write_json(ROOT / f"docs/intruder_identity/export_v{a.version}.json", receipt)
        print("built", target.name, sub["sha256"])
    if a.dry_run:
        out = ROOT / f"runs/intruder_identity/dry_v{a.version}"
        dry_run(ROOT / a.base, a.base_sha256, a.version, out)


if __name__ == "__main__":
    main()
