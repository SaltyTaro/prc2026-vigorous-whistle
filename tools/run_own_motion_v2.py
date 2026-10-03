"""Own-motion bound at the five other ADS-B airports (configs/own_motion_v2.json), applied after own_motion_v1.

Rows the shipped ADS-B matcher does not trace from the gate (dark or unmatched) at EHAM / EDDM / LSZH / LEMD / LEBL: the departure's
aircraft is identified by ICAO address (tunnel 11's identity tables runs/dig/position/ids_<tag>.parquet); its latest identified arrival
at the airport blocked in at the same stand 10 min to 18 h before take-off; its first adsb.lol ground point at >= 3 kt after that
in-block + 120 s lies B seconds before take-off (60 < B <= 2,700 s), and the aircraft does not come back to its parking fix afterwards.
The off-block is not earlier than that motion: label >= B. Where the chain's prediction p < B the value is p + w (B + o(airport) - p),
w = 0.5, o = the fitting months' mean of label - B shrunk to the pooled mean. tools/run_own_motion.py (v1) is imported unchanged.

--score       development (the v62 proxy by default, or --control) and twelve-month tables, month-blocked, with the declared rule
              -> docs/own_motion/results_v2.json (results_v2_<control-tag>.json for another control) + a README section is written by hand
--dry-run     label-free ranking file on a base file (read only) -> --out (default runs/own_motion/dry_v<version>/)
--build       rerun the dry run into runs/own_motion/build_v<version>, require byte identity with the dry run (--dry-dir), write
              runs/submissions/vigorous-whistle_v<version>.parquet (never overwrite) and docs/own_motion/export_v<version>.json
--build-ids   rebuild an identity table for a movement file (final file; tunnel 11's rule) into --ids-dir
Nothing is uploaded.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026_adsb.features import DEPARTURE_COLUMNS, departure_features  # noqa: E402
import run_own_motion as O  # noqa: E402

R, I = O.R, O.I
CONFIG = ROOT / "configs/own_motion_v2.json"
IDS_DIR = ROOT / "runs/dig/position"
SURFACE = ROOT / "data/external/adsblol/surface"
MATCHES = ROOT / "data/external/adsblol/matches"
KEYMAP = ROOT / "runs/rematch_nonm/key_map.parquet"
RANKING = "data/raw/ranking.parquet"   # tools/final_chain_overrides.py --raw
CACHE = ROOT / "runs/own_motion/v2_triggers_2025.parquet"
AIRPORTS = ["EHAM", "EDDM", "LSZH", "LEMD", "LEBL"]
ID_AIRPORTS = ["EDDM", "EHAM", "LSZH", "LEMD", "LEBL", "EDDF", "EGLL", "LFPG"]
INB_MIN, INB_MAX, AFTER_IN, GS_MIN, B_MIN = 600.0, 64800.0, 120.0, 3.0, 60.0
STILL_KT, RET_M, RET_S = 1.5, 60.0, 300.0
V62 = ("runs/submissions/vigorous-whistle_v62.parquet", "cba143f84debdca0d9aff345ff4669ff6a3f0f7f5c4e02d08bcd3e0d02c32d9d")


def _dist(lat1, lon1, lat2, lon2):
    return np.hypot((lat1 - lat2) * 111320.0, (lon1 - lon2) * 111320.0 * np.cos(np.radians(lat2)))


def _ids(tag: str, ids_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(ids_dir / f"ids_{tag}.parquet")[["MVT_ID_mvt", "PHASE_mvt", "airport", "icao"]]


def _gate(dep: pd.DataFrame) -> pd.DataFrame:
    out = []
    stands = pd.read_parquet(MATCHES / "stands.parquet")
    for month, g in dep.groupby(dep["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m")):
        mfile = MATCHES / f"matches2_{month}.parquet"
        matches = pd.read_parquet(mfile) if mfile.exists() else pd.DataFrame()
        out.append(departure_features(g[DEPARTURE_COLUMNS], matches, stands)[["MVT_ID_mvt", "adsb_gate"]])
    return pd.concat(out)


def month_triggers(job: tuple) -> pd.DataFrame:
    """Label-free trigger rows of one movement file: MVT_ID_mvt, airport, B, ret (same stand, inbound window and not gate-traced applied;
    the band and the prediction are applied later). job = (file, previous file or None, id tag, previous id tag or None, ids dir)."""
    path, prev_path, tag, prev_tag, ids_dir = job
    cols = sorted(set(DEPARTURE_COLUMNS + ["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "STAND_mvt"]))
    raw = pd.read_parquet(path, columns=cols)
    dep_all = raw[raw["PHASE_mvt"].eq("DEP") & raw["ADEP_mvt"].isin(AIRPORTS)]
    gate = _gate(dep_all)
    notgate = set(gate.loc[gate["adsb_gate"].ne(1), "MVT_ID_mvt"]) | (set(dep_all["MVT_ID_mvt"]) - set(gate["MVT_ID_mvt"]))
    x = raw.merge(_ids(tag, Path(ids_dir)), on=["MVT_ID_mvt", "PHASE_mvt"])
    parts = [x]
    if prev_path is not None:
        p = pd.read_parquet(prev_path, columns=cols)
        parts.append(p.merge(_ids(prev_tag, Path(ids_dir)), on=["MVT_ID_mvt", "PHASE_mvt"]))
    x = pd.concat(parts, ignore_index=True)
    x = x[x["icao"].notna() & x["airport"].isin(AIRPORTS)]
    dep = x[x["PHASE_mvt"].eq("DEP") & x["MVT_ID_mvt"].isin(notgate)][["MVT_ID_mvt", "airport", "icao", "MVT_TIME_UTC_mvt", "STAND_mvt"]].copy()
    arr = x[x["PHASE_mvt"].eq("ARR")].rename(columns={"BLOCK_TIME_UTC_mvt": "inb", "STAND_mvt": "inb_stand"})[["icao", "airport", "inb", "inb_stand"]]
    arr = arr.dropna(subset=["inb"]).copy()
    dep["MVT_TIME_UTC_mvt"] = dep["MVT_TIME_UTC_mvt"].astype("datetime64[us, UTC]")
    arr["inb"] = arr["inb"].astype("datetime64[us, UTC]")
    dep = pd.merge_asof(dep.sort_values("MVT_TIME_UTC_mvt"), arr.sort_values("inb"), left_on="MVT_TIME_UTC_mvt", right_on="inb",
                        by=["icao", "airport"], direction="backward")
    dep["inb_rel"] = (dep["MVT_TIME_UTC_mvt"] - dep["inb"]).dt.total_seconds()
    dep = dep[dep["inb_rel"].between(INB_MIN, INB_MAX) & dep["inb_stand"].eq(dep["STAND_mvt"])].copy()
    dep["day"] = dep["MVT_TIME_UTC_mvt"].dt.strftime("%Y%m%d")
    rows = []
    for day, g in dep.groupby("day"):
        prev = (pd.Timestamp(day) - pd.Timedelta(days=1)).strftime("%Y%m%d")
        pts = []
        for dd in (prev, day):
            f = SURFACE / f"surface_{dd}.parquet"
            if f.exists():
                pts.append(pd.read_parquet(f, columns=["icao", "airport", "time", "ground", "gs", "lat", "lon"], filters=[("icao", "in", sorted(set(g["icao"])))]))
        if not pts:
            continue
        p = pd.concat(pts)
        p = p[p["ground"].eq(True)].copy()
        p["ts"] = pd.to_datetime(p["time"], unit="s", utc=True)
        p = p.sort_values("ts", kind="stable")
        pg = {k: v for k, v in p.groupby(["icao", "airport"])}
        for r in g.itertuples(index=False):
            q = pg.get((r.icao, r.airport))
            if q is None:
                continue
            q = q[(q["ts"] > r.inb) & (q["ts"] <= r.MVT_TIME_UTC_mvt)]
            if not len(q):
                continue
            rel = (q["ts"] - r.MVT_TIME_UTC_mvt).dt.total_seconds().to_numpy()
            gs = q["gs"].to_numpy()
            after = rel > -r.inb_rel + AFTER_IN
            mv = after & (gs >= GS_MIN)
            if not mv.any():
                continue
            i = int(np.argmax(mv))
            still = gs <= STILL_KT
            idx = np.arange(len(q))
            pre = still & after & (idx < i)
            ret = False
            if pre.any():
                lat, lon = q["lat"].to_numpy(), q["lon"].to_numpy()
                plat, plon = np.median(lat[pre]), np.median(lon[pre])
                ret = bool(((idx > i) & still & (rel > rel[i] + RET_S) & (_dist(lat, lon, plat, plon) < RET_M)).any())
            rows.append((r.MVT_ID_mvt, r.airport, float(-rel[i]), ret))
    t = pd.DataFrame(rows, columns=["MVT_ID_mvt", "airport", "B", "ret"])
    return t[t["B"] > B_MIN].reset_index(drop=True)


def training_jobs(ids_dir: Path) -> list[tuple]:
    files = I.training_files()
    tags = [f.name.split("training_")[1][:7] for f in files]
    return [(str(f), str(files[k - 1]) if k else None, tags[k], tags[k - 1] if k else None, str(ids_dir)) for k, f in enumerate(files)]


def triggers_2025(ids_dir: Path, use_cache: bool) -> tuple[pd.DataFrame, dict]:
    key = {"code_sha256": sha256_file(Path(__file__)), "config_sha256": sha256_file(CONFIG),
           "ids_sha256": {p.name: sha256_file(p) for p in sorted(ids_dir.glob("ids_2025-*.parquet"))}}
    meta = CACHE.with_suffix(".json")
    if use_cache and CACHE.exists() and meta.exists() and read_json(meta) == key:
        return pd.read_parquet(CACHE), {"cache": "used", **key}
    with ProcessPoolExecutor(max_workers=2) as ex:
        t = pd.concat(list(ex.map(month_triggers, training_jobs(ids_dir))), ignore_index=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    t.to_parquet(CACHE, index=False)
    write_json(meta, key)
    return t, {"cache": "recomputed", **key}


def candidates(tab: pd.DataFrame, pred: str, trig: pd.DataFrame, spec: dict) -> pd.DataFrame:
    c = tab.merge(trig[["MVT_ID_mvt", "B", "ret"]], on="MVT_ID_mvt")
    keep = (c["airport"].isin(spec["airports"]) & ~c["ret"] & (c["B"] > B_MIN) & (c["B"] <= spec["B_max_s"]) & (c[pred] < c["B"])
            & ~c["cls"].eq("LIRF_pop") & ~c["MVT_ID_mvt"].isin(R.GUARD))
    return c[keep].copy()


def v62_tables() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """The v62 development proxy (`v62`) and the twelve-month table after own_motion_v1 (`after3`), exactly as tools/run_own_motion.py
    --score builds the candidate om_b2700_w100."""
    cfg1 = read_json(O.CONFIG)
    spec = cfg1["candidate"]
    trig1 = O.triggers(I.training_files())
    t12 = O.twelve_after()
    dev, _ = O.dev_table()
    c12 = O.candidates(t12, "after2", trig1, spec["B_max_s"])
    cdv = O.candidates(dev, "v61", trig1, spec["B_max_s"])

    def apply(tab, pred, cand, months, col):
        parts = []
        for m in months:
            c = O.values(cand[cand["month"] == m], pred, O.fit(c12[c12["month"] != m]), spec["w"])
            s = tab[tab["month"] == m].copy()
            s[col] = s[pred].astype(float)
            s = s.set_index("MVT_ID_mvt")
            s.loc[c["MVT_ID_mvt"].to_numpy(), col] = c["value"].to_numpy()
            parts.append(s.reset_index())
        return pd.concat(parts, ignore_index=True)

    dev = apply(dev, "v61", cdv, R.DEV_MONTHS, "v62")
    t12 = apply(t12, "after2", c12, sorted(t12["month"].unique()), "after3")
    r1 = read_json(ROOT / "docs/own_motion/results_v1.json")
    want = {m: r1["control_month_rmse_s"][m] * (1 + r1["candidates"][spec["name"]]["development"][m]["pct"] / 100) for m in R.DEV_MONTHS}
    got = {m: float(np.sqrt(((g["label"] - g["v62"]) ** 2).mean())) for m, g in dev.groupby("month")}
    if any(abs(got[m] / want[m] - 1) > 1e-9 for m in want):
        raise ValueError(f"The v62 proxy does not reproduce own_motion_v1's readings: {got} vs {want}")
    return dev, t12, got


def score(control: str | None, control_col: str | None, control_sha: str | None, tag: str | None) -> None:
    cfg = read_json(CONFIG)
    rule = cfg["rule"]
    dev, t12, v62_rmse = v62_tables()
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
    trig, trig_meta = triggers_2025(IDS_DIR, use_cache=False)
    specs = [cfg["candidate"]] + cfg["sensitivity_reported_not_selectable"]
    report = {"experiment_id": cfg["experiment_id"], "config_sha256": sha256_file(CONFIG), "code_sha256": sha256_file(Path(__file__)),
              "computed_at_utc": utc_now(), "control": {"kind": "v62 development proxy" if not control else control, "column": pred if not control else control_col,
                                                        "sha256": sha256_file(ROOT / control) if control else None,
                                                        "month_rmse_s": {m: float(np.sqrt(((g["label"] - g[pred]) ** 2).mean())) for m, g in dev.groupby("month")}},
              "v62_proxy_month_rmse_s": v62_rmse, "triggers": trig_meta, "trigger_rows_2025": int(len(trig)),
              "trigger_rows_by_airport": trig["airport"].value_counts().to_dict(), "candidates": {}}
    for spec in specs:
        rng = np.random.default_rng(20261003)
        c12 = candidates(t12, "after3", trig, spec)
        cdv = candidates(dev, pred, trig, spec)
        out = {"development": {}, "twelve": {}}
        for name, tab, p, cand, months in [("development", dev, pred, cdv, R.DEV_MONTHS), ("twelve", t12, "after3", c12, sorted(t12["month"].unique()))]:
            for m in months:
                model = O.fit(c12[c12["month"] != m])
                c = O.values(cand[cand["month"] == m], p, model, spec["w"])
                s = tab[tab["month"] == m]
                new = s[p].to_numpy(dtype=float).copy()
                pos = pd.Series(np.arange(len(s)), index=s["MVT_ID_mvt"].to_numpy())
                new[pos.loc[c["MVT_ID_mvt"]].to_numpy()] = c["value"].to_numpy()
                pct, lo, hi = R.interval(s, p, new, rng)
                sq = float(((c["value"] - c[p]) ** 2).sum())
                gain = float((((c["label"] - c[p]) ** 2) - (c["label"] - c["value"]) ** 2).sum())
                out[name][m] = {"pct": pct, "lo": lo, "hi": hi, "rows": int(len(c)), "by_airport_rows": c["airport"].value_counts().to_dict(),
                                "valid_share": float((c["label"] >= c["B"] - 60).mean()) if len(c) else None, "squared_change": sq,
                                "gain_over_squared_change": gain / sq if sq else None, "offsets": model["offsets"]}
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
        out["offsets_all_twelve"] = O.fit(c12)
        report["candidates"][spec["name"]] = out
        print(spec["name"], "dev:", " ".join(f"{m[5:]} {dv[m]['pct']:+.4f}% [{dv[m]['lo']:+.3f},{dv[m]['hi']:+.3f}] n{dv[m]['rows']}" for m in R.DEV_MONTHS))
        print("   twelve:", " ".join(f"{m[5:]} {v['pct']:+.3f}%" for m, v in tw.items()))
        print("  ", checks)
    name = cfg["candidate"]["name"]
    report["selected"] = name if report["candidates"][name]["checks"]["accepted"] else None
    dest = ROOT / ("docs/own_motion/results_v2.json" if not tag else f"docs/own_motion/results_v2_{tag}.json")
    write_json(dest, report)
    print("selected:", report["selected"], "->", dest.relative_to(ROOT))


def dry_run(base: Path, base_sha: str, version: int, out_dir: Path) -> None:
    cfg = read_json(CONFIG)
    if sha256_file(base) != base_sha:
        raise ValueError("The base file's SHA-256 differs from the expected one.")
    spec = cfg["candidate"]
    _, t12, _ = v62_tables()
    trig12, trig_meta = triggers_2025(IDS_DIR, use_cache=True)
    model = O.fit(candidates(t12, "after3", trig12, spec))
    rk_path = ROOT / RANKING
    rk = pd.read_parquet(rk_path, columns=["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADEP_flt", "AIRCRAFT_TYPE_mvt"])
    rk = rk[rk["PHASE_mvt"].eq("DEP")]
    pop = rk["ADEP_mvt"].eq("LIRF") & rk["ADEP_flt"].isna() & rk["AIRCRAFT_TYPE_mvt"].isna()
    b = pd.read_parquet(base)
    if list(b.columns) != ["MVT_ID_mvt", "TAXITIME_SEC_mvt"] or len(b) != len(rk):
        raise ValueError("The base file is not a full submission.")
    tab = b.merge(rk[["MVT_ID_mvt", "ADEP_mvt"]].assign(cls=np.where(pop, "LIRF_pop", "x")).rename(columns={"ADEP_mvt": "airport"}), on="MVT_ID_mvt")
    tab["base"] = tab["TAXITIME_SEC_mvt"].astype(float)
    trig = month_triggers((str(rk_path), None, "ranking", None, str(IDS_DIR)))
    c = O.values(candidates(tab, "base", trig, spec), "base", model, spec["w"])
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
    if not set(rk.loc[rk["MVT_ID_mvt"].isin(set(b.loc[changed, "MVT_ID_mvt"])), "ADEP_mvt"]) <= set(AIRPORTS):
        raise ValueError("A row outside the five airports moved.")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"vigorous-whistle_v{version}.parquet"
    out.to_parquet(path, index=False)
    diff = out["TAXITIME_SEC_mvt"].astype(float) - b["TAXITIME_SEC_mvt"].astype(float)
    moved = c.assign(new=np.rint(c["value"]), move=np.rint(c["value"]) - c["base"])
    receipt = {"experiment_id": cfg["experiment_id"], "config_sha256": sha256_file(CONFIG), "code_sha256": sha256_file(Path(__file__)),
               "written_at_utc": utc_now(), "kind": "dry run, not a submission", "triggers_2025": trig_meta,
               "ids_ranking_sha256": sha256_file(IDS_DIR / "ids_ranking.parquet"),
               "base": str(base.relative_to(ROOT)) if base.is_relative_to(ROOT) else str(base), "base_sha256": base_sha,
               "file": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path), "sha256": sha256_file(path),
               "bytes": path.stat().st_size, "rows_changed": int(changed.sum()), "squared_change": float((diff ** 2).sum()),
               "max_move_s": float(diff.max()), "min_move_s": float(diff.min()), "mean_move_s": float(diff[changed].mean()) if changed.any() else 0.0,
               "lirf_population_guard_rows_changed": int((changed & fixed).sum()), "other_rows_identical_to_base": bool(out.loc[~changed].equals(b.loc[~changed])),
               "model": model, "ranking_trigger_rows": int(len(trig)), "ranking_trigger_rows_no_return_band": int(((~trig["ret"]) & (trig["B"] <= spec["B_max_s"])).sum()),
               "by_airport": moved.groupby("airport")["move"].agg(["size", "mean", "max"]).round(1).to_dict(orient="index"),
               "rows": moved[["MVT_ID_mvt", "airport", "base", "B", "new"]].round(1).to_dict(orient="records")}
    write_json(out_dir / f"vigorous-whistle_v{version}.receipt.json", receipt)
    print({k: receipt[k] for k in ["sha256", "bytes", "rows_changed", "squared_change", "max_move_s", "min_move_s", "lirf_population_guard_rows_changed",
                                   "other_rows_identical_to_base"]}, receipt["by_airport"])


def build_ids(tag: str, src: Path, ids_dir: Path) -> None:
    """Tunnel 11's identity rule (runs/dig/position/build_pos.py, identity part): nearest adsb.lol point of the same callsign (CALLSIGN_flt,
    else the rematch key on FLIGHT_mvt) at the same airport within 600 s of the movement time; day files of that day and the next."""
    cols = ["MVT_ID_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "CALLSIGN_flt", "FLIGHT_mvt"]
    mv = pd.read_parquet(src, columns=cols)
    mv["airport"] = np.where(mv["PHASE_mvt"].eq("ARR"), mv["ADES_mvt"], mv["ADEP_mvt"])
    mv = mv[mv["airport"].isin(ID_AIRPORTS)].copy()
    km = pd.read_parquet(KEYMAP).drop_duplicates("FLIGHT_mvt")[["FLIGHT_mvt", "key"]]
    mv = mv.merge(km, on="FLIGHT_mvt", how="left")
    mv["cs"] = mv["CALLSIGN_flt"].fillna(mv["key"]).str.upper().str.strip()
    epoch = pd.Timestamp(0, tz="UTC")
    mv["t"] = (mv["MVT_TIME_UTC_mvt"] - epoch).dt.total_seconds()
    mv["blk"] = (mv["BLOCK_TIME_UTC_mvt"] - epoch).dt.total_seconds()
    mv.loc[mv["PHASE_mvt"].eq("DEP"), "blk"] = np.nan
    mv["day"] = mv["MVT_TIME_UTC_mvt"].dt.strftime("%Y%m%d")
    out, cache = [], {}
    for day in sorted(mv["day"].unique()):
        nxt = (pd.Timestamp(day) + pd.Timedelta(days=1)).strftime("%Y%m%d")
        for d in (day, nxt):
            if d not in cache:
                f = SURFACE / f"surface_{d}.parquet"
                cache[d] = pd.read_parquet(f, columns=["icao", "airport", "time", "callsign"]) if f.exists() else None
                if cache[d] is not None:
                    cache[d] = cache[d][cache[d]["airport"].isin(ID_AIRPORTS)]
        for d in list(cache):
            if d not in (day, nxt):
                del cache[d]
        parts = [cache[d] for d in (day, nxt) if cache[d] is not None]
        if not parts:
            continue
        pts = pd.concat(parts)
        g = mv[(mv["day"] == day) & mv["cs"].notna()]
        named = pts[pts["callsign"].notna()][["airport", "callsign", "time", "icao"]].assign(cs=lambda x: x["callsign"].str.upper().str.strip()).drop(columns="callsign")
        mm = pd.merge_asof(g.sort_values("t")[["MVT_ID_mvt", "PHASE_mvt", "airport", "cs", "t", "blk"]], named.sort_values("time"),
                           left_on="t", right_on="time", by=["airport", "cs"], direction="nearest", tolerance=600)
        out.append(mm[["MVT_ID_mvt", "PHASE_mvt", "airport", "icao", "t", "blk"]])
    ids_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(out).to_parquet(ids_dir / f"ids_{tag}.parquet", index=False)
    print("ids", tag, sha256_file(ids_dir / f"ids_{tag}.parquet"))


def main() -> None:
    global IDS_DIR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--control", default=None, help="a development chain proxy (parquet with MVT_ID_mvt and --control-col); default the v62 proxy")
    ap.add_argument("--control-col", default=None)
    ap.add_argument("--control-sha256", default=None)
    ap.add_argument("--control-tag", default=None, help="results file suffix for another control (results_v2_<tag>.json)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--base", default=V62[0])
    ap.add_argument("--base-sha256", default=V62[1])
    ap.add_argument("--version", type=int, default=63)
    ap.add_argument("--out", default=None, help="the dry run's directory (default runs/own_motion/dry_v<N>)")
    ap.add_argument("--dry-dir", default=None, help="--build: the dry run to match (default runs/own_motion/dry_v<N>)")
    ap.add_argument("--ids-dir", default=None, help="identity tables directory (default runs/dig/position)")
    ap.add_argument("--build-ids", default=None, help="tag of an identity table to rebuild from the movement file (--raw for a new file)")
    ap.add_argument("--build", action="store_true", help="write runs/submissions/vigorous-whistle_v<version>.parquet; it must equal the dry run "
                    "byte for byte and must not exist yet")
    ap.add_argument("--user-decision", default=None)
    import final_chain_overrides
    final_chain_overrides.add_arguments(ap)
    a = ap.parse_args()
    final_chain_overrides.apply(a)
    if a.ids_dir:
        IDS_DIR = ROOT / final_chain_overrides._rel(a.ids_dir)
    if a.build_ids:
        build_ids(a.build_ids, ROOT / RANKING, IDS_DIR)
    if a.score:
        if a.control and not (a.control_col and a.control_tag):
            raise ValueError("--control needs --control-col and --control-tag.")
        score(a.control, a.control_col, a.control_sha256, a.control_tag)
    if a.build:
        dry = (ROOT / final_chain_overrides._rel(a.dry_dir) if a.dry_dir else ROOT / f"runs/own_motion/dry_v{a.version}") / f"vigorous-whistle_v{a.version}.parquet"
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
                                   "selected_by": "the declared rule (docs/own_motion/results_v2.json)", "official_scores_used": "none",
                                   "user_decision": a.user_decision, "declaration": {"configs/own_motion_v2.json": sha256_file(CONFIG)},
                                   "code": {"tools/run_own_motion_v2.py": sha256_file(Path(__file__)), "tools/run_own_motion.py": sha256_file(Path(O.__file__))},
                                   "base": {"path": a.base, "sha256": a.base_sha256},
                                   "recipe": "the base chain (ending with own_motion_v1), then the own-motion bound at EHAM / EDDM / LSZH / LEMD / LEBL last (w 0.5; value take-off - the aircraft's first ground motion after its inbound at the same stand + offset fitted on the twelve 2025 months; no return to the parking fix)"}})
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
