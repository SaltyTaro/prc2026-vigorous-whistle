"""Lever 9 (docs/ctx_opdi/README.md): OPDI flight-event columns for the context network, every month the network folds need.

For each month M (2025-01 .. 2025-12, 2026-01, 2026-07) the departures of M (data/raw/training_<M>.parquet; the ranking file for 2026) are
matched to OPDI departures exactly as lever 7 (tools/run_opdi_stack.py: airport + callsign, CALLSIGN_flt else the rematch key map, OPDI
take-off reference within [-900, +300] s of the supplied take-off, nearest; only OPDI events up to that reference). Ten columns, seconds
before the supplied take-off unless named otherwise (no chain-relative column: the network never sees the chain's prediction):
  opdi_ref_rel, opdi_stand_exit, opdi_apron_exit, opdi_twy_first, opdi_first_seen, opdi_rwy_entry, opdi_n_twy, opdi_deice,
  opdi_deice_dur, opdi_has (1 when matched, else 0; the others NaN when absent).
Coverage-shift mask (declared in configs/ctx_opdi_design_v1.json before any fit, label-free; amendment 1 before any fit): per airport, the share
of departures with opdi_has = 1 and with an opdi_stand_exit in the two ranking months over the same shares in all of 2025; an airport whose matched
ratio lies outside [0.5, 2.0], whose stand-exit ratio does so while that share is material (>= 2% in either period), or whose 2025 matched share is
below 2%, has all ten columns blanked (opdi_has 0, the rest NaN) in every month.
Output: data/external/opdi/features_v19ownopdi/features_v19_<M>.parquet = the shipped own-points table of M
(data/external/adsblol/features_v19own, hash-checked against docs/dig/ctx_ownpoints_features_manifest.json) with the ten columns appended
(left join on MVT_ID_mvt), manifest.json (hashes, coverage, mask). Departure labels and block times are never read.

  python tools/ctx_opdi_features.py
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
import run_opdi_stack as S  # noqa: E402  (lever 7: matching constants and opdi_flights)

ID = "MVT_ID_mvt"
EVENT_DIRS = [ROOT / "data/external/opdi/stack_v1", ROOT / "data/external/opdi/ctx_v1"]
OWN = ROOT / "data/external/adsblol/features_v19own"
OWN_MANIFEST = ROOT / "docs/dig/ctx_ownpoints_features_manifest.json"
OUT = ROOT / "data/external/opdi/features_v19ownopdi"
KEYMAP = ROOT / "runs/rematch_nonm/key_map.parquet"
MONTHS = [f"2025-{m:02d}" for m in range(1, 13)] + ["2026-01", "2026-07"]
RANKING = ["2026-01", "2026-07"]
COLS = ["opdi_ref_rel", "opdi_stand_exit", "opdi_apron_exit", "opdi_twy_first", "opdi_first_seen", "opdi_rwy_entry", "opdi_n_twy",
        "opdi_deice", "opdi_deice_dur", "opdi_has"]
RATIO = (0.5, 2.0)
MIN_2025_MATCHED = 0.02
MATERIAL_STAND_EXIT = 0.02  # amendment 1 (configs/ctx_opdi_design_v1.json): the stand-exit ratio applies only when that share is material


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def departures(month: str, km: pd.DataFrame) -> pd.DataFrame:
    cols = [ID, "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "CALLSIGN_flt", "FLIGHT_mvt"]
    if month.startswith("2025"):
        y, m = map(int, month.split("-"))
        nxt = f"{y + (m == 12)}-{1 if m == 12 else m + 1:02d}"
        mv = pd.read_parquet(ROOT / f"data/raw/training_{month}-01_{nxt}-01.parquet", columns=cols)
    else:
        mv = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=cols)
        mv = mv[mv["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m").eq(month)]
    dep = mv[mv["PHASE_mvt"].eq("DEP")].drop_duplicates(ID).rename(columns={"ADEP_mvt": "airport"}).copy()
    dep = dep.merge(km, on=["airport", "FLIGHT_mvt"], how="left")
    dep["cs"] = dep["CALLSIGN_flt"].fillna(dep["key"]).astype("string").str.upper().str.strip()
    dep["t"] = (dep["MVT_TIME_UTC_mvt"] - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
    return dep[[ID, "airport", "cs", "t"]]


def event_files(month: str) -> list[Path]:
    start = pd.Timestamp(f"{month}-01") - pd.Timedelta(days=1)
    end = start + pd.Timedelta(days=1) + pd.offsets.MonthBegin(1) + pd.Timedelta(days=1)
    out = {}
    for d in EVENT_DIRS:
        for f in sorted(d.glob("events_*.parquet")):
            a, b = pd.Timestamp(f.stem[7:15]), pd.Timestamp(f.stem[16:24])
            if a < end and b > start:
                out.setdefault(f.name, f)  # lever 7's extract wins when both exist (ctx_v1 never re-downloads those)
    return list(out.values())


def match(month: str, km: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    dep = departures(month, km)
    files = event_files(month)
    of = S.opdi_flights(files)
    m = dep.merge(of, on=["airport", "cs"], how="inner")
    m["dt"] = m["ref"] - m["t"]
    m = m[(m["dt"] >= S.LO) & (m["dt"] <= S.HI)].copy()
    m["adt"] = m["dt"].abs()
    m = m.sort_values([ID, "adt"]).drop_duplicates(ID)
    t = m["t"]
    out = pd.DataFrame({ID: m[ID].to_numpy(), "airport": m["airport"].to_numpy()})
    out["opdi_ref_rel"] = m["dt"].to_numpy()
    for c, src in [("opdi_stand_exit", "stand_exit"), ("opdi_apron_exit", "apron_exit"), ("opdi_twy_first", "twy_first"),
                   ("opdi_first_seen", "first_seen"), ("opdi_rwy_entry", "rwy_entry")]:
        out[c] = (t - m[src]).to_numpy()
    out["opdi_n_twy"] = m["n_twy"].fillna(0).to_numpy()
    out["opdi_deice"] = m["deice"].fillna(0).to_numpy()
    out["opdi_deice_dur"] = m["deice_dur"].to_numpy()
    out["opdi_has"] = 1.0
    return dep, out, files


def main() -> None:
    own_manifest = json.loads(OWN_MANIFEST.read_text())["files"]
    km = pd.read_parquet(KEYMAP)[["ADEP_mvt", "FLIGHT_mvt", "key"]].rename(columns={"ADEP_mvt": "airport"})
    raw, cov = {}, {}
    for month in MONTHS:
        dep, out, files = match(month, km)
        raw[month] = (dep, out, files)
        a = dep[[ID, "airport"]].merge(out.drop(columns=["airport"]), on=ID, how="left")
        for ap, g in a.groupby("airport"):
            cov.setdefault(ap, {})[month] = {"departures": int(len(g)), "matched": float(g["opdi_has"].notna().mean()),
                                             "stand_exit": float(g["opdi_stand_exit"].notna().mean())}
        print(month, "departures", len(dep), "matched", len(out), "event files", len(files), flush=True)
    mask, ratios = [], {}
    for ap, by in sorted(cov.items()):
        def share(months, key):
            n = sum(by[m]["departures"] for m in months if m in by)
            return sum(by[m][key] * by[m]["departures"] for m in months if m in by) / max(n, 1)
        y25 = [m for m in MONTHS if m.startswith("2025")]
        r = {}
        for key in ("matched", "stand_exit"):
            base, rk = share(y25, key), share(RANKING, key)
            r[key] = {"2025": base, "ranking": rk, "ratio": (rk / base) if base > 0 else None}
        def off(v):
            return v["ratio"] is None or not RATIO[0] <= v["ratio"] <= RATIO[1]
        material = max(r["stand_exit"]["2025"], r["stand_exit"]["ranking"]) >= MATERIAL_STAND_EXIT
        bad = r["matched"]["2025"] < MIN_2025_MATCHED or off(r["matched"]) or (material and off(r["stand_exit"]))
        ratios[ap] = r
        if bad:
            mask.append(ap)
    print("masked airports:", mask, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"created_at_utc": pd.Timestamp.now("UTC").isoformat(), "columns": COLS, "mask_rule": {"ratio": RATIO, "min_2025_matched": MIN_2025_MATCHED, "material_stand_exit": MATERIAL_STAND_EXIT},
                "masked_airports": mask, "coverage_ratios": ratios, "coverage_by_airport_month": cov, "code_sha256": sha(Path(__file__)),
                "lever7_code_sha256": sha(ROOT / "tools/run_opdi_stack.py"), "keymap_sha256": sha(KEYMAP), "files": {}}
    for month in MONTHS:
        dep, out, files = raw[month]
        name = f"features_v19_{month}.parquet"
        own_path = OWN / name
        if sha(own_path) != own_manifest[name]["sha256"]:
            raise ValueError(f"{own_path} differs from the shipped own-points manifest.")
        base = pd.read_parquet(own_path)
        o = out[~out["airport"].isin(mask)].drop(columns=["airport"])
        t = base.merge(o, on=ID, how="left")
        t["opdi_has"] = t["opdi_has"].fillna(0.0)
        for c in COLS:
            t[c] = t[c].astype("float64")
        if len(t) != len(base) or not t[ID].equals(base[ID]):
            raise ValueError(f"{month}: the join changed the own-points table's rows.")
        dest = OUT / name
        t.to_parquet(dest, index=False)
        manifest["files"][name] = {"rows": int(len(t)), "sha256": sha(dest), "own_sha256": own_manifest[name]["sha256"],
                                   "opdi_rows": int(t["opdi_has"].sum()), "event_files": {f.name: sha(f) for f in files}}
        print(name, len(t), "opdi rows", int(t["opdi_has"].sum()), flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    print("done", flush=True)


if __name__ == "__main__":
    main()
