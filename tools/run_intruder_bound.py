"""Intruder stand bound (configs/intruder_bound_v1.json): a departure leaves its stand before a different aircraft blocks in there.

Candidate: departure at airport A from stand S with prediction p; predicted off-block = take-off - p; the last arrival at (A, S) whose
in-block precedes it has a different aircraft type, the arrival before that has the departure's type (its inbound), and the gap
between the predicted off-block and that last in-block is in (0, G]. Then label > B = take-off - the intruder's in-block. Airports
enter by their fitting months' validity; the value is p + w (B + o - p), o = the fitting months' mean of label - B by airport and gap
band, shrunk to the airport mean. Applied last, on v57's chain.

--score     development (shipped chain proxy) and twelve-month tables, month-blocked, with the declared rule -> docs/intruder_bound/
--dry-run   label-free ranking file on a base file (default the uploaded v57, read only) -> runs/intruder_bound/dry_v<version>/
Nothing is written under runs/submissions and nothing is uploaded.
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
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402

CONFIG = ROOT / "configs/intruder_bound_v1.json"
DEV = ROOT / "runs/scout/refit/ctl_v55m.parquet"
DEV_CLS = ROOT / "runs/daily_totals/rows.parquet"
TWELVE = ROOT / "runs/daily_totals/rows12.parquet"
BANDS = [0, 300, 600, 900, 1800, 3600]
GUARD = [213142471, 213142469]
EQUAL = ["2025-01", "2025-07", "2025-12"]
DEV_MONTHS = ["2025-01", "2025-07", "2025-11", "2025-12"]
SHRINK, MIN_ROWS, MIN_VALID, SLACK = 20.0, 30, 0.90, 60.0
FLOOR = -12.0


def movements(paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(departures: MVT_ID, take-off, stand, type), (arrivals by airport and stand with the previous arrival's type)."""
    cols = ["MVT_ID_mvt", "PHASE_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt"]
    raw = pd.concat([pd.read_parquet(p, columns=cols) for p in paths])
    dep = raw[raw["PHASE_mvt"].eq("DEP")][["MVT_ID_mvt", "MVT_TIME_UTC_mvt", "STAND_mvt", "AIRCRAFT_TYPE_mvt"]]
    arr = raw[raw["PHASE_mvt"].eq("ARR")].rename(columns={"ADES_mvt": "airport", "BLOCK_TIME_UTC_mvt": "in_blk", "AIRCRAFT_TYPE_mvt": "arr_type"})
    arr = arr[["airport", "STAND_mvt", "in_blk", "arr_type"]].dropna()
    arr["in_blk"] = arr["in_blk"].astype("datetime64[us, UTC]")
    arr = arr.sort_values(["airport", "STAND_mvt", "in_blk"])
    arr["prev_type"] = arr.groupby(["airport", "STAND_mvt"])["arr_type"].shift(1)
    return dep, arr.sort_values("in_blk")


def candidates(tab: pd.DataFrame, pred: str, dep: pd.DataFrame, arr: pd.DataFrame, gap_max: float) -> pd.DataFrame:
    """Candidate rows of `tab` (MVT_ID_mvt, month, airport, cls, label?, pred) with B, gap and gap band."""
    d = tab.merge(dep, on="MVT_ID_mvt").dropna(subset=["STAND_mvt"])
    d["pred_off"] = (d["MVT_TIME_UTC_mvt"] - pd.to_timedelta(d[pred].round(), unit="s")).astype("datetime64[us, UTC]")
    m = pd.merge_asof(d.sort_values("pred_off"), arr, left_on="pred_off", right_on="in_blk", by=["airport", "STAND_mvt"], direction="backward")
    m["gap"] = (m["pred_off"] - m["in_blk"]).dt.total_seconds()
    m["B"] = (m["MVT_TIME_UTC_mvt"] - m["in_blk"]).dt.total_seconds()
    keep = (m["arr_type"].notna() & m["arr_type"].ne(m["AIRCRAFT_TYPE_mvt"]) & m["prev_type"].eq(m["AIRCRAFT_TYPE_mvt"])
            & (m["gap"] > 0) & (m["gap"] <= gap_max) & ~m["cls"].eq("LIRF_pop") & ~m["MVT_ID_mvt"].isin(GUARD))
    m = m[keep].copy()
    m["band"] = pd.cut(m["gap"], BANDS, labels=False)
    return m.drop(columns=["pred_off"])


def fit(fitting: pd.DataFrame) -> dict:
    """Airport set and offsets from the fitting months' candidates (with labels)."""
    val = fitting.groupby("airport").apply(lambda h: pd.Series({"n": len(h), "valid": float((h["label"] >= h["B"] - SLACK).mean())}))
    airports = sorted(val[(val["n"] >= MIN_ROWS) & (val["valid"] >= MIN_VALID)].index)
    f = fitting[fitting["airport"].isin(airports)].assign(over=lambda x: x["label"] - x["B"])
    am = f.groupby("airport")["over"].mean()
    g = f.groupby(["airport", "band"])["over"].agg(["sum", "size"]).reset_index()
    g["o"] = (g["sum"] + SHRINK * g["airport"].map(am)) / (g["size"] + SHRINK)
    return {"airports": airports, "validity": val.round(4).to_dict(orient="index"), "airport_mean": am.to_dict(),
            "offsets": {f"{a}|{int(b)}": float(o) for a, b, o in g[["airport", "band", "o"]].itertuples(index=False)}}


def values(cand: pd.DataFrame, pred: str, model: dict, w: float) -> pd.DataFrame:
    c = cand[cand["airport"].isin(model["airports"])].copy()
    off = np.array([model["offsets"].get(f"{a}|{int(b)}", model["airport_mean"][a]) for a, b in zip(c["airport"], c["band"])])
    c["value"] = np.maximum(c[pred] + w * (c["B"] + off - c[pred]), FLOOR)
    return c


def interval(s: pd.DataFrame, pred: str, new: np.ndarray, rng: np.random.Generator, reps: int = 2000) -> tuple[float, float, float]:
    e0, e1 = ((s["label"] - s[pred]) ** 2).to_numpy(), (s["label"].to_numpy() - new) ** 2
    days = s["day"].to_numpy()
    ud, inv = np.unique(days, return_inverse=True)
    s0, s1, n = np.bincount(inv, e0), np.bincount(inv, e1), np.bincount(inv)
    out = []
    for _ in range(reps):
        k = np.bincount(rng.integers(0, len(ud), len(ud)), minlength=len(ud))
        out.append(100 * (np.sqrt((k * s1).sum() / (k * n).sum()) / np.sqrt((k * s0).sum() / (k * n).sum()) - 1))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return 100 * (np.sqrt(e1.mean()) / np.sqrt(e0.mean()) - 1), float(lo), float(hi)


def read_table(table: str, dep: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if table == "dev":
        t = pd.read_parquet(DEV, columns=["MVT_ID_mvt", "month", "airport", "label", "new"]).merge(
            pd.read_parquet(DEV_CLS, columns=["MVT_ID_mvt", "cls"]), on="MVT_ID_mvt")
        pred = "new"
    else:
        t = pd.read_parquet(TWELVE, columns=["MVT_ID_mvt", "month", "airport", "label", "stack", "cls"])
        pred = "stack"
    t = t.merge(dep[["MVT_ID_mvt", "MVT_TIME_UTC_mvt"]], on="MVT_ID_mvt")
    t["day"] = t["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m-%d")
    return t.drop(columns=["MVT_TIME_UTC_mvt"]), pred


def score() -> None:
    cfg = read_json(CONFIG)
    files = sorted(glob.glob(str(ROOT / "data/raw/training_2025-*.parquet")))
    dep, arr = movements([Path(f) for f in files])
    twelve, p12 = read_table("twelve", dep)
    dev, pdv = read_table("dev", dep)
    specs = [cfg["candidate"]] + cfg["sensitivity_reported_not_selectable"]
    report = {"experiment_id": cfg["experiment_id"], "config_sha256": sha256_file(CONFIG), "computed_at_utc": utc_now(),
              "inputs": {"dev": sha256_file(DEV), "dev_cls": sha256_file(DEV_CLS), "twelve": sha256_file(TWELVE)}, "candidates": {}}
    for spec in specs:
        rng = np.random.default_rng(20261003)
        c12 = candidates(twelve, p12, dep, arr, spec["G_s"])
        cdv = candidates(dev, pdv, dep, arr, spec["G_s"])
        out = {"development": {}, "twelve": {}}
        for name, tab, pred, cand, months in [("development", dev, pdv, cdv, DEV_MONTHS), ("twelve", twelve, p12, c12, sorted(twelve["month"].unique()))]:
            for m in months:
                model = fit(c12[c12["month"] != m])
                c = values(cand[cand["month"] == m], pred, model, spec["w"])
                s = tab[tab["month"] == m]
                new = s[pred].to_numpy(dtype=float).copy()
                pos = pd.Series(np.arange(len(s)), index=s["MVT_ID_mvt"].to_numpy())
                new[pos.loc[c["MVT_ID_mvt"]].to_numpy()] = c["value"].to_numpy()
                pct, lo, hi = interval(s, pred, new, rng)
                out[name][m] = {"pct": pct, "lo": lo, "hi": hi, "rows": int(len(c)), "airports": model["airports"],
                                "by_airport_rows": c["airport"].value_counts().to_dict(),
                                "squared_change": float(((c["value"] - c[pred]) ** 2).sum())}
        dv, tw = out["development"], out["twelve"]
        eq_mean = float(np.mean([dv[m]["pct"] for m in EQUAL]))
        checks = {"equal_footing_mean_pct": eq_mean, "mean_ok": eq_mean <= -0.05,
                  "equal_intervals_below_zero": all(dv[m]["hi"] < 0 for m in EQUAL),
                  "no_development_month_worse": all(dv[m]["pct"] <= 0 for m in DEV_MONTHS),
                  "twelve_months_better": int(sum(v["pct"] < 0 for v in tw.values())),
                  "twelve_intervals_below_zero": int(sum(v["hi"] < 0 for v in tw.values()))}
        checks["twelve_ok"] = checks["twelve_months_better"] >= 11
        checks["accepted"] = bool(checks["mean_ok"] and checks["equal_intervals_below_zero"] and checks["no_development_month_worse"] and checks["twelve_ok"])
        out["checks"] = checks
        report["candidates"][spec["name"]] = out
        print(spec["name"], "dev:", " ".join(f"{m[5:]} {dv[m]['pct']:+.4f}% [{dv[m]['lo']:+.3f},{dv[m]['hi']:+.3f}] n{dv[m]['rows']}" for m in DEV_MONTHS))
        print("   twelve:", " ".join(f"{m[5:]} {v['pct']:+.3f}%" for m, v in tw.items()), "|", checks)
    name = cfg["candidate"]["name"]
    report["selected"] = name if report["candidates"][name]["checks"]["accepted"] else None
    (ROOT / "docs/intruder_bound").mkdir(parents=True, exist_ok=True)
    write_json(ROOT / "docs/intruder_bound/results_v1.json", report)
    print("selected:", report["selected"])


def dry_run(base: Path, base_sha: str, version: int, out_dir: Path) -> None:
    cfg = read_json(CONFIG)
    if sha256_file(base) != base_sha:
        raise ValueError("The base file's SHA-256 differs from the expected one.")
    spec = cfg["candidate"]
    files = sorted(glob.glob(str(ROOT / "data/raw/training_2025-*.parquet")))
    dep, arr = movements([Path(f) for f in files])
    twelve, p12 = read_table("twelve", dep)
    model = fit(candidates(twelve, p12, dep, arr, spec["G_s"]))
    rk_path = ROOT / "data/raw/ranking.parquet"
    rdep, rarr = movements([rk_path])
    rk = pd.read_parquet(rk_path, columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADEP_flt", "AIRCRAFT_TYPE_mvt"])
    rk = rk[rk["PHASE_mvt"].eq("DEP")]
    pop = rk["ADEP_mvt"].eq("LIRF") & rk["ADEP_flt"].isna() & rk["AIRCRAFT_TYPE_mvt"].isna()
    b = pd.read_parquet(base)
    if list(b.columns) != ["MVT_ID_mvt", "TAXITIME_SEC_mvt"] or len(b) != 344841:
        raise ValueError("The base file is not a full submission.")
    tab = b.merge(rk[["MVT_ID_mvt", "ADEP_mvt"]].assign(cls=np.where(pop, "LIRF_pop", "x")).rename(columns={"ADEP_mvt": "airport"}), on="MVT_ID_mvt")
    tab["month"] = "ranking"
    tab["base"] = tab["TAXITIME_SEC_mvt"].astype(float)
    c = values(candidates(tab, "base", rdep, rarr, spec["G_s"]), "base", model, spec["w"])
    out = b.copy()
    new = pd.Series(np.rint(c["value"].to_numpy()).astype(out["TAXITIME_SEC_mvt"].dtype), index=c["MVT_ID_mvt"].to_numpy())
    idx = out["MVT_ID_mvt"].map(new)
    out["TAXITIME_SEC_mvt"] = np.where(idx.notna(), idx, out["TAXITIME_SEC_mvt"]).astype(b["TAXITIME_SEC_mvt"].dtype)
    if not out["MVT_ID_mvt"].equals(b["MVT_ID_mvt"]):
        raise ValueError("Row order changed.")
    if out.loc[out["MVT_ID_mvt"].isin(GUARD), "TAXITIME_SEC_mvt"].tolist() != b.loc[b["MVT_ID_mvt"].isin(GUARD), "TAXITIME_SEC_mvt"].tolist():
        raise ValueError("A guard row moved.")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"vigorous-whistle_v{version}.parquet"
    out.to_parquet(path, index=False)
    diff = out["TAXITIME_SEC_mvt"].astype(float) - b["TAXITIME_SEC_mvt"].astype(float)
    moved = c.assign(new=np.rint(c["value"]), move=np.rint(c["value"]) - c["base"])
    receipt = {"experiment_id": cfg["experiment_id"], "config_sha256": sha256_file(CONFIG), "written_at_utc": utc_now(), "kind": "dry run, not a submission",
               "base": str(base.relative_to(ROOT)), "base_sha256": base_sha, "file": str(path.relative_to(ROOT)), "sha256": sha256_file(path),
               "rows_changed": int((diff != 0).sum()), "squared_change": float((diff ** 2).sum()), "max_move_s": float(diff.max()),
               "min_move_s": float(diff.min()), "airports": model["airports"], "fitting_validity": model["validity"],
               "by_airport": moved.groupby("airport")["move"].agg(["size", "mean", "max"]).round(1).to_dict(orient="index"),
               "rows": moved[["MVT_ID_mvt", "airport", "base", "B", "gap", "new"]].round(1).to_dict(orient="records")}
    write_json(out_dir / f"vigorous-whistle_v{version}.receipt.json", receipt)
    print({k: receipt[k] for k in ["sha256", "rows_changed", "squared_change", "max_move_s", "min_move_s", "airports"]})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--base", default="runs/submissions/vigorous-whistle_v57.parquet")
    ap.add_argument("--base-sha256", default="31d909ff2e5d1937e65404c396893b9e5524d082c07422be9390c75aad4daed8")
    ap.add_argument("--version", type=int, default=58)
    ap.add_argument("--out", default=None)
    ap.add_argument("--build", action="store_true", help="at the user's word: write runs/submissions/vigorous-whistle_v<version>.parquet; "
                    "it must equal the dry run byte for byte and must not exist yet")
    a = ap.parse_args()
    if a.score:
        score()
    if a.build:
        dry = ROOT / f"runs/intruder_bound/dry_v{a.version}/vigorous-whistle_v{a.version}.parquet"
        target = ROOT / f"runs/submissions/vigorous-whistle_v{a.version}.parquet"
        if target.exists():
            raise ValueError(f"{target.name} exists; never overwrite a submission file.")
        if not dry.exists():
            raise ValueError("Run the dry run first.")
        tmp = ROOT / f"runs/intruder_bound/build_v{a.version}"
        dry_run(ROOT / a.base, a.base_sha256, a.version, tmp)
        built = tmp / f"vigorous-whistle_v{a.version}.parquet"
        if sha256_file(built) != sha256_file(dry):
            raise ValueError("The rebuilt file differs from the dry run.")
        target.write_bytes(built.read_bytes())
        from prc2026_submission.contract import validate_submission
        sub = validate_submission(target, ROOT / "data/raw/submitting.parquet")
        sub.update({"version": a.version, "rounding": "numpy.rint, nearest integer with ties to even; no clipping",
                    "provenance": {"audit_used": False, "dry_run": False, "experiment_id": "intruder_bound_v1", "selected_candidate": "g3600_w100",
                                   "selected_by": "the declared rule (accepted)", "official_scores_used": "none",
                                   "user_decision": "Build + upload v58 now (3 October 2026, AskUserQuestion)",
                                   "declaration": {"configs/intruder_bound_v1.json": sha256_file(CONFIG)},
                                   "code": {"tools/run_intruder_bound.py": sha256_file(Path(__file__))},
                                   "base": {"path": a.base, "sha256": a.base_sha256},
                                   "recipe": "v57's chain, then the intruder stand bound last (EDDF / EGLL / LFPG / LTFM; value take-off - intruder in-block + offset fitted on the twelve 2025 months)"}})
        write_json(target.with_suffix(".receipt.json"), sub)
        receipt = read_json(tmp / f"vigorous-whistle_v{a.version}.receipt.json")
        receipt.update({"kind": "submission build at the user's word", "file": str(target.relative_to(ROOT)), "sha256": sha256_file(target),
                        "dry_run_sha256": sha256_file(dry)})
        write_json(ROOT / f"docs/intruder_bound/export_v{a.version}.json", receipt)
        print("built", target.name, receipt["sha256"])
    if a.dry_run:
        out = Path(a.out) if a.out else ROOT / f"runs/intruder_bound/dry_v{a.version}"
        if "runs/submissions" in str(out).replace("\\", "/"):
            raise ValueError("The dry run never writes under runs/submissions.")
        dry_run(ROOT / a.base, a.base_sha256, a.version, out)


if __name__ == "__main__":
    main()
