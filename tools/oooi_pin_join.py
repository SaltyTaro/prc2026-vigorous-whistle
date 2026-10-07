"""OOOI pin screen, step 2 (local, light): parse the kept data-link messages and join them to departures.

Inputs : runs/oooi_pin/messages_unique.parquet (from tools/oooi_pin_extract.py + oooi_pin_pack.py on arcw0)
         data/raw/training_2025-{07,11,12}*.parquet, data/raw/ranking.parquet (only the needed columns)
Outputs: runs/oooi_pin/parsed.parquet, runs/oooi_pin/joined_dev.parquet, runs/oooi_pin/joined_rank.parquet,
         runs/oooi_pin/join_stats.json

Sources parsed (origin LFPG / LTFM / LIRF only):
  2N   Air France in-flight report  'AF098G/15151004LFPGKIAH  09170941'  -> t1 = 0917, t2 = 0941
  8D   easyJet delay report         '0028,93,...,0028,1919,LFPG,EGAA'     -> t1 = 1919 (actual time), t2 none
  QP / QQ OOOI                      'LFPGxxxxHHMM' / 'LFPGxxxxOFFFOUTT'  -> t1 = OUT, t2 = OFF
  free text 'OUT hhmm' / 'OFF hhmm' with an origin code (Delta 30, United 5Z, label 80, ...)
Every HHMM is anchored to the latest UTC instant with that clock time at or before the message time (+5 min).
The join uses only label-free fields: date / message time, destination, airline prefix + callsign suffix, and the
supplied take-off time (which must precede the message).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "oooi_pin"
AIRPORTS = {"LFPG", "LTFM", "LIRF"}
# ACARS two-character airline designator -> ICAO callsign prefix
AL = {"AF": "AFR", "U2": "EZY", "EC": "EJU", "DS": "EZS", "DL": "DAL", "UA": "UAL", "AA": "AAL", "TK": "THY",
      "FR": "RYR", "AZ": "ITY", "KL": "KLM", "BA": "BAW", "LH": "DLH", "IB": "IBE", "VY": "VLG", "TO": "TVF",
      "HV": "TRA", "EI": "EIN", "SN": "BEL", "LX": "SWR", "OS": "AUA", "AC": "ACA", "NW": "DAL", "W6": "WZZ",
      "JU": "ASL", "TS": "TSC", "SK": "SAS", "AY": "FIN"}

RE_2N = re.compile(r"^(\w{2})(\w{4})/(\d{2})(\d{2})(\d{4})([A-Z]{4})([A-Z]{4})\s+(\d{4})(\d{4})")
RE_ROUTE = re.compile(r"\b(LFPG|LTFM|LIRF)[ /]?([A-Z]{4})\b")
RE_OUT = re.compile(r"(?<![A-Z])OUT(?: TIME)?[ :/]?(\d{4})Z?\b")
RE_OFF = re.compile(r"(?<![A-Z])OFF(?: TIME)?[ :/]?(\d{4})Z?\b")


def anchor(hhmm: str | None, ref: pd.Timestamp, slack_min: int = 5) -> pd.Timestamp | None:
    if hhmm is None or not re.fullmatch(r"\d{4}", hhmm):
        return None
    h, m = int(hhmm[:2]), int(hhmm[2:])
    if h > 23 or m > 59:
        return None
    lim = ref + pd.Timedelta(minutes=slack_min)
    t = lim.normalize() + pd.Timedelta(hours=h, minutes=m)
    if t > lim:
        t -= pd.Timedelta(days=1)
    return t


def parse(u: pd.DataFrame) -> pd.DataFrame:
    out = []
    for r in u.itertuples(index=False):
        text = r.text if isinstance(r.text, str) else ""
        fl = r.flight.strip() if isinstance(r.flight, str) else ""
        rec = None
        if r.label == "2N":
            m = RE_2N.match(text.replace("\r\n", "  ").replace("\n", " "))
            if m and m.group(6) in AIRPORTS:
                rec = dict(source="2N", al=m.group(1), sfx=m.group(2), org=m.group(6), dst=m.group(7),
                           t1s=m.group(8), t2s=m.group(9), day1=m.group(3), day2=m.group(4))
        elif r.label == "8D":
            f = text.split("\r\n")[0].split("\n")[0].split(",")
            if len(f) >= 10 and f[8] in AIRPORTS:
                # field 7 is a clock time ~45 min after the message (an ETA, not OUT); field 6 is the total
                # departure delay HHMM, so OUT is estimated after the join as SCHED_TIME + delay
                rec = dict(source="8D", al=fl[:2], sfx=fl[2:], org=f[8], dst=f[9][:4], t1s=None, t2s=None,
                           eta_s=f[7], delay_total=f[6])
        elif r.label in ("QP", "QQ"):
            if text[:4] in AIRPORTS:
                t1s = text[8:12] if r.label == "QP" else text[12:16]
                t2s = None if r.label == "QP" else text[8:12]
                rec = dict(source=r.label, al=fl[:2], sfx=fl[2:], org=text[:4], dst=text[4:8], t1s=t1s, t2s=t2s)
        else:
            mr = RE_ROUTE.search(text)
            mo, mf = RE_OUT.search(text), RE_OFF.search(text)
            if mr and (mo or mf) and r.label not in ("B4", "H1", "34", "35", "8D"):
                rec = dict(source="text_" + str(r.label), al=fl[:2], sfx=fl[2:], org=mr.group(1), dst=mr.group(2),
                           t1s=mo.group(1) if mo else None, t2s=mf.group(1) if mf else None)
        if rec is None:
            continue
        ts = r.ts_utc
        t2 = anchor(rec.get("t2s"), ts)
        t1 = anchor(rec.get("t1s"), t2 if t2 is not None else ts)
        rec.update(ts=ts, t1=t1, t2=t2, label=r.label, flight=fl, reg=r.reg, station=r.station, kind=r.kind,
                   file=r.file, text=text, n_copies=r.n_copies)
        out.append(rec)
    p = pd.DataFrame(out)
    p["cs_prefix"] = p["al"].map(AL)
    p["sfx_norm"] = p["sfx"].str.lstrip("0")
    # one record per report (the same report is repeated with new message times); keep the earliest copy
    p["text_key"] = p["text"].str.replace("\r", "", regex=False).str.split("\n").str[0].str.strip()
    p["day_key"] = p["ts"].dt.strftime("%Y-%m-%d")
    p = p.sort_values("ts").drop_duplicates(["source", "flight", "org", "dst", "text_key", "day_key"])
    p["month"] = p["ts"].dt.strftime("%Y-%m")
    return p.reset_index(drop=True)


def load_deps(files: list[Path]) -> pd.DataFrame:
    cols = ["MVT_ID_mvt", "FLIGHT_mvt", "CALLSIGN_flt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt",
            "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "AOBT_3_flt", "AIRCRAFT_TYPE_mvt",
            "STAND_mvt"]
    parts = []
    for f in files:
        d = pd.read_parquet(f, columns=cols)
        d = d[(d["PHASE_mvt"] == "DEP") & d["ADEP_mvt"].isin(AIRPORTS)]
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    d["cs"] = d["CALLSIGN_flt"].fillna(d["FLIGHT_mvt"]).astype(str)
    d["cs_prefix"] = d["cs"].str[:3]
    d["cs_sfx_norm"] = d["cs"].str[3:].str.lstrip("0")
    d["fl_prefix"] = d["FLIGHT_mvt"].astype(str).str[:3]
    d["fl_sfx_norm"] = d["FLIGHT_mvt"].astype(str).str[3:].str.lstrip("0")
    return d


def join(p: pd.DataFrame, d: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Key join: origin, destination, callsign prefix, normalised suffix; take-off in [t1 - 1 h, msg time + 5 min]."""
    p = p[p["cs_prefix"].notna()].copy()
    p["msg_i"] = np.arange(len(p))
    m = p.merge(d, left_on=["org", "dst", "cs_prefix"], right_on=["ADEP_mvt", "ADES_mvt", "cs_prefix"], how="inner")
    # 8D has no OUT/OFF clock: its window is [msg - 8 h, msg + 5 min]
    ref = m["t1"].fillna(m["t2"]).fillna(m["ts"] - pd.Timedelta(hours=7))
    tko = m["MVT_TIME_UTC_mvt"]
    win = (tko >= ref - pd.Timedelta(hours=1)) & (tko <= m["ts"] + pd.Timedelta(minutes=5)) & \
          (tko <= ref + pd.Timedelta(hours=8))
    m = m[win.fillna(False)].copy()
    m["sfx_ok"] = (m["sfx_norm"] == m["cs_sfx_norm"]) | (
        (m["sfx_norm"] == m["fl_sfx_norm"]) & (m["fl_prefix"] == m["cs_prefix"]))
    # keyed match: suffix agrees; fallback: unique candidate on (route, airline, window)
    n_ok = m.groupby("msg_i")["sfx_ok"].transform("sum")
    n_all = m.groupby("msg_i")["msg_i"].transform("size")
    m["match_kind"] = np.where(m["sfx_ok"] & (n_ok == 1), "suffix_unique",
                               np.where(m["sfx_ok"] & (n_ok > 1), "suffix_ambiguous",
                                        np.where((n_ok == 0) & (n_all == 1), "route_window_unique", "drop")))
    keep = m[m["match_kind"].isin(["suffix_unique", "route_window_unique"])].copy()
    # one message per departure (several sources may hit the same departure: keep 2N > QP/QQ > text > 8D order)
    prio = {"2N": 0, "QP": 1, "QQ": 1, "8D": 3}
    keep["prio"] = keep["source"].map(prio).fillna(2)
    keep = keep.sort_values(["MVT_ID_mvt", "prio", "ts"])
    stats = {
        "messages_with_target_origin": int(len(p)),
        "messages_with_any_candidate": int(m["msg_i"].nunique()),
        "by_kind_messages": {k: int(v) for k, v in m.drop_duplicates("msg_i")["match_kind"].value_counts().items()},
        "departures_hit_by_several_messages": int(keep["MVT_ID_mvt"].duplicated().sum()),
    }
    keep = keep.drop_duplicates("MVT_ID_mvt")
    # easyJet 8D: OUT estimate = scheduled time + the sum of the reported delay durations (fields 0, 2, 4, HHMM;
    # field 6 is usually their total but sometimes the STD clock, so it is not used)
    is8 = keep["source"].eq("8D")
    f = keep.loc[is8, "text_key"].str.split(",")

    def dur(v: pd.Series) -> pd.Series:
        v = v.astype(str)
        ok = v.str.fullmatch(r"\d{4}")
        out = pd.Series(0.0, index=v.index)
        out[ok] = v[ok].str[:2].astype(int) * 60 + v[ok].str[2:].astype(int)
        return out

    mins = dur(f.str[0]) + dur(f.str[2]) + dur(f.str[4])
    keep.loc[is8, "delay_sum_min"] = mins
    keep.loc[is8, "t1"] = keep.loc[is8, "SCHED_TIME_UTC_mvt"] + pd.to_timedelta(mins, unit="min")
    stats["departures_matched"] = int(len(keep))
    return keep, stats


def main() -> None:
    u = pd.read_parquet(RUN / "messages_unique.parquet")
    p = parse(u)
    p.to_parquet(RUN / "parsed.parquet", index=False)
    stats = {"parsed_counts": {f"{k[0]}|{k[1]}|{k[2]}": int(v)
                               for k, v in p.groupby(["source", "org", "month"]).size().items()}}
    raw = ROOT / "data" / "raw"
    dev = load_deps([raw / "training_2025-07-01_2025-08-01.parquet", raw / "training_2025-11-01_2025-12-01.parquet",
                     raw / "training_2025-12-01_2026-01-01.parquet"])
    rank = load_deps([raw / "ranking.parquet"])
    jd, sd = join(p, dev)
    jr, sr = join(p, rank)
    for j in (jd, jr):
        j["month_dep"] = j["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m")
    jd.to_parquet(RUN / "joined_dev.parquet", index=False)
    jr.to_parquet(RUN / "joined_rank.parquet", index=False)
    stats["dev_join"] = sd
    stats["rank_join"] = sr
    stats["dev_matched_by_month_source"] = {f"{k[0]}|{k[1]}|{k[2]}": int(v) for k, v in
                                            jd.groupby(["month_dep", "ADEP_mvt", "source"]).size().items()}
    stats["rank_matched_by_month_source"] = {f"{k[0]}|{k[1]}|{k[2]}": int(v) for k, v in
                                             jr.groupby(["month_dep", "ADEP_mvt", "source"]).size().items()}
    stats["dev_match_kind"] = {k: int(v) for k, v in jd["match_kind"].value_counts().items()}
    stats["rank_match_kind"] = {k: int(v) for k, v in jr["match_kind"].value_counts().items()}
    (RUN / "join_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
