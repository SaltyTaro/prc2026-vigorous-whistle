"""OOOI extension screen, step 3b (runs on the VM next to scan/matched.parquet): explicit parsers for the data-link
formats that the token scan flagged as carrying an OUT time, applied to the label-free message-departure matches.

Formats (parsers fixed after the token scan, before any pin was scored):
  MVA_AD   any label / airline, MVT-style 'AD0645/0659 EA0758 EIDW' -> OUT = first AD time, OFF = second
           (Aer Lingus label 80 'MVA' at EGLL)
  UA_OUT13 United label 13 '/13 OUT EVENT / EGLL KIAH 13 125324/TIME 1253' -> OUT = HHMMSS
  QP_ORIG  any airline OOOI OUT report 'EGLLKJFK0829 ...' (origin first) -> OUT = characters 9-12
  QQ_FR    OOOI OFF report holding OFF then OUT 'EGBBLIRF08080752' -> OFF = 9-12, OUT = 13-16 (when 16 digits)
  SK_OUTRP SAS label 8A '/OUTRP/EGLL/ESSA/12/0858/' -> OUT = HHMM
  BA_MVT16 British Airways label 16 'MVT01085606EGLLLIPE...' -> OUT = HHMMSS (lags the record by ~2 min)
  KL_BIS   KLM label 10 'BIS..' third line 'HHMMHHMMHHMMHHMM' -> OUT, OFF, ON, IN
  EZY_8D   easyJet label 8D delay report, airports other than LFPG -> OUT = scheduled time + sum of delay fields 0/2/4
  AF_2N    Air France label 2N at airports other than LFPG -> OUT, OFF
Each clock is anchored to the instant nearest the departure's take-off. Label-free guard, uniform, fixed before
scoring: take-off - OUT in [180, 3600] s and, where an OFF time exists, |take-off - OFF| <= 180 s.
Block times and labels are kept only for July / November / December 2025 (scored months) and blanked otherwise.

Usage: python oooi_ext_formats.py --matched scan/matched.parquet --out scan
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

SCORED = ["2025-07", "2025-11", "2025-12"]
RE_AD = re.compile(r"(?<![A-Z0-9])AD(\d{4})/(\d{4})")
RE_UA13 = re.compile(r"/13 OUT EVENT\s*/\s*([A-Z]{4}) ([A-Z]{4}) \d{2} (\d{6})")
RE_SK = re.compile(r"/OUTRP/([A-Z]{4})/([A-Z]{4})/\d{2}/(\d{4})/")
RE_BA16 = re.compile(r"^MVT\d{2}(\d{6})([A-Z]{4})([A-Z]{4})")
RE_BIS = re.compile(r"^(\d{4})(\d{4})(\d{4})(\d{4})\s*$", re.M)
RE_2N = re.compile(r"^(\w{2})(\w{4})/(\d{2})(\d{2})(\d{4})([A-Z]{4})([A-Z]{4})\s+(\d{4})(\d{4})")


def hhmm_ok(s: str | None) -> bool:
    return s is not None and len(s) >= 4 and s[:4].isdigit() and int(s[:2]) < 24 and int(s[2:4]) < 60


def parse(r) -> list[tuple[str, str, str | None]]:
    """[(format, OUT clock 'HHMM' or 'HHMMSS', OFF clock or None)] for one matched message."""
    t, out = r.text, []
    if r.label == "2N":
        m = RE_2N.match(t.replace("\r\n", "  ").replace("\n", " "))
        if m and r.ADEP_mvt != "LFPG":
            out.append(("AF_2N", m.group(8), m.group(9)))
    if r.label == "8D" and r.ADEP_mvt != "LFPG":
        f = t.replace("\r", "").split("\n")[0].split(",")
        if len(f) >= 10:
            out.append(("EZY_8D", "SCHED+" + ",".join([f[0], f[2], f[4]]), None))
    m = RE_AD.search(t)
    if m:
        out.append(("MVA_AD", m.group(1), m.group(2)))
    if r.label == "13":
        m = RE_UA13.search(t)
        if m:
            out.append(("UA_OUT13", m.group(3), None))
    if r.label == "QP" and t[:4] == r.ADEP_mvt:
        out.append(("QP_ORIG", t[8:12], None))
    if r.label == "QQ" and t[:4] == r.ADEP_mvt and t[8:16].isdigit():
        out.append(("QQ_FR", t[12:16], t[8:12]))
    if r.label == "8A":
        m = RE_SK.search(t)
        if m:
            out.append(("SK_OUTRP", m.group(3), None))
    if r.label == "16":
        m = RE_BA16.match(t)
        if m:
            out.append(("BA_MVT16", m.group(1), None))
    if r.label == "10" and t.startswith("BIS"):
        m = RE_BIS.search(t)
        if m:
            out.append(("KL_BIS", m.group(1), m.group(2)))
    return [(f, a, b) for f, a, b in out if a.startswith("SCHED+") or hhmm_ok(a)]


def anchor(clock: str, tko: pd.Timestamp) -> pd.Timestamp:
    sec = int(clock[4:6]) if len(clock) >= 6 else 0
    t = tko.normalize() + pd.Timedelta(hours=int(clock[:2]), minutes=int(clock[2:4]), seconds=sec)
    c = [t + pd.Timedelta(days=k) for k in (-1, 0, 1)]
    return min(c, key=lambda x: abs((x - tko).total_seconds()))


def dur(v: str) -> int:
    return int(v[:2]) * 60 + int(v[2:]) if len(v) == 4 and v.isdigit() else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matched", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    j = pd.read_parquet(a.matched)
    rows = []
    for r in j.itertuples(index=False):
        for fmt, c1, c2 in parse(r):
            if c1.startswith("SCHED+"):
                if pd.isna(r.SCHED_TIME_UTC_mvt):
                    continue
                t1 = r.SCHED_TIME_UTC_mvt + pd.Timedelta(minutes=sum(dur(x) for x in c1[6:].split(",")))
            else:
                t1 = anchor(c1, r.tko)
            t2 = anchor(c2, r.tko) if c2 and hhmm_ok(c2) else pd.NaT
            rows.append({"format": fmt, "MVT_ID_mvt": r.MVT_ID_mvt, "al2": r.al2, "label": r.label, "ADEP_mvt": r.ADEP_mvt,
                         "src": r.src, "dep_month": r.dep_month, "tko": r.tko, "t1": t1, "t2": t2, "ts": r.ts,
                         "AOBT_3_flt": r.AOBT_3_flt, "block": r.BLOCK_TIME_UTC_mvt, "label_s": r.TAXITIME_SEC_mvt})
    f = pd.DataFrame(rows)
    raw = (f["tko"] - f["t1"]).dt.total_seconds()
    d2 = (f["tko"] - f["t2"]).dt.total_seconds().abs()
    f["guard"] = raw.between(180, 3600) & (f["t2"].isna() | d2.le(180))
    # one message per (format, departure): the earliest
    f = f.sort_values("ts").drop_duplicates(["format", "MVT_ID_mvt"]).reset_index(drop=True)
    scored = f["src"].eq("train") & f["dep_month"].isin(SCORED)
    f.loc[~scored, ["block", "label_s"]] = np.nan
    f.to_parquet(Path(a.out) / "formats.parquet", index=False)
    q = {}
    for fmt, g in f.groupby("format"):
        s = g[g["src"].eq("train") & g["dep_month"].isin(SCORED)]
        d = (s["block"] - s["t1"]).dt.total_seconds().dropna()
        dg = (s.loc[s["guard"], "block"] - s.loc[s["guard"], "t1"]).dt.total_seconds().dropna()
        med = float(d.median()) if len(d) else None
        q[fmt] = {"rows_by_src_month": {f"{x}|{y}": int(v) for (x, y), v in g.groupby(["src", "dep_month"]).size().items()},
                  "airports": {k: int(v) for k, v in g["ADEP_mvt"].value_counts().items()},
                  "airlines": {k: int(v) for k, v in g["al2"].value_counts().head(12).items()},
                  "scored_rows": int(len(d)), "scored_guarded_rows": int(len(dg)),
                  "median_blk_minus_out_s": med}
        if len(d):
            c = (d - med).abs()
            q[fmt].update({"same_minute_abs_le30": float((d.abs() <= 30).mean()),
                           "centred_le30": float((c <= 30).mean()), "centred_le60": float((c <= 60).mean()),
                           "centred_le120": float((c <= 120).mean()), "centred_gt300": float((c > 300).mean()),
                           "centred_gt1800": float((c > 1800).mean())})
        if len(dg):
            mg = float(dg.median())
            cg = (dg - mg).abs()
            q[fmt].update({"guarded_centred_le30": float((cg <= 30).mean()), "guarded_centred_le120": float((cg <= 120).mean()),
                           "guarded_centred_gt300": float((cg > 300).mean())})
        t2 = (s["tko"] - s["t2"]).dt.total_seconds().dropna()
        if len(t2):
            q[fmt]["off_vs_takeoff_le60"] = float((t2.abs() <= 60).mean())
    (Path(a.out) / "format_quality.json").write_text(json.dumps(q, indent=1))
    for k, v in q.items():
        print(k, {x: (round(y, 3) if isinstance(y, float) else y) for x, y in v.items() if x not in ("airlines",)})


if __name__ == "__main__":
    main()
