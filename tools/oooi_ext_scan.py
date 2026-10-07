"""OOOI extension screen, step 2-3 (runs on the VM where messages_unique.parquet and data/raw live; label-free join,
label-based token statistics on July / November / December 2025 only).

1. census of every kept message by (label, airline prefix, text skeleton), month and airport;
2. label-free join of messages that carry an ACARS flight id to departures of the ten airports (training July-December
   2025 and the ranking file): callsign suffix (leading zeros stripped) + UTC day (+-1) + origin / destination when the
   text names them + take-off in [message - 14 h, message + 3 h]; the ACARS two-character airline prefix must map to the
   departure's ICAO callsign prefix (mapping learned from unique route-confirmed matches); unique candidate only;
3. token scan: every HHMM-like token of the text (4-digit runs, 6-16-digit runs at even offsets, HH:MM / HH.MM)
   is anchored to the clock instant nearest the departure's take-off; per (label, airline, token id) the share of rows
   where it equals the recorded off-block or the take-off. Labels are read for July, November and December 2025 only;
   August-October 2025 are counted, never scored.

Usage: python oooi_ext_scan.py --msgs DIR/messages_unique.parquet --raw DATA_RAW --out OUTDIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ICAO = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
IATA = {"FRA": "EDDF", "MUC": "EDDM", "LHR": "EGLL", "AMS": "EHAM", "BCN": "LEBL", "MAD": "LEMD", "CDG": "LFPG",
        "FCO": "LIRF", "ZRH": "LSZH", "IST": "LTFM"}
TEN = "|".join(ICAO)
# origin + destination, e.g. 'LFPGKIAH', 'EGLL KJFK', 'EHAM/LEMD', 'LEMD,EGLL', 'LIRF-EIDW'
ROUTE_RE = re.compile(r"(?<![A-Z])(" + TEN + r")[ /,.\-]{0,3}([A-Z]{4})(?![A-Z])")
ROUTE_REV_RE = re.compile(r"(?<![A-Z])([A-Z]{4})[ /,.\-]{0,3}(" + TEN + r")(?![A-Z])")
ANY_TEN_RE = re.compile(r"(?<![A-Z])(" + TEN + r")(?![A-Z])")
IATA_RE = re.compile(r"(?<![A-Z])(" + "|".join(IATA) + r")(?![A-Z])")
MVT_RE = re.compile(r"(?<![A-Z0-9])(AD|AA|EA)(\d{4})")
RUN_RE = re.compile(r"\d+")
SEP_RE = re.compile(r"(?<!\d)([01]\d|2[0-3])[:.]([0-5]\d)(?:[:.]\d{2})?(?!\d)")
DEV_MONTHS = ["2025-07", "2025-11", "2025-12"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def skeleton(text: str, n: int = 24) -> str:
    first = text.replace("\r", "").split("\n")[0]
    s = ANY_TEN_RE.sub("@@@@", first)
    s = re.sub(r"[0-9]", "9", s)
    s = re.sub(r"[A-Z]", "A", s)
    return s[:n]


def route_of(text: str) -> tuple[str | None, str | None, str]:
    """Origin among the ten and destination when the text names a route; else (None, dest, kind) for MVT texts."""
    m = ROUTE_RE.search(text)
    if m and m.group(2) not in ("AAAA",):
        return m.group(1), m.group(2), "route"
    m = ROUTE_REV_RE.search(text)
    if m:
        # 'XXXX EHAM' is ambiguous (origin XXXX to a target airport, or the reverse); keep the target as destination
        return None, m.group(2), "route_rev"
    mv = MVT_RE.findall(text)
    if mv:
        ea = re.search(r"(?<![A-Z0-9])EA\d{4}\s*([A-Z]{4})", text)
        return None, (ea.group(1) if ea else None), "mvt"
    return None, None, "none"


def tokens(text: str, limit: int = 400) -> list[tuple[str, str]]:
    """(token id, HHMM) for every HHMM-like group in the first `limit` characters."""
    t = text[:limit]
    out = []
    for i, m in enumerate(RUN_RE.finditer(t)):
        s = m.group(0)
        L = len(s)
        if L < 4 or L > 16:
            continue
        for off in range(0, L - 3, 2):
            hh, mm = int(s[off:off + 2]), int(s[off + 2:off + 4])
            if hh < 24 and mm < 60:
                out.append((f"r{i}L{L}o{off}", s[off:off + 4]))
    for i, m in enumerate(SEP_RE.finditer(t)):
        out.append((f"s{i}", m.group(1) + m.group(2)))
    return out


def departures(raw: Path) -> tuple[pd.DataFrame, dict]:
    files = [raw / f"training_2025-{m:02d}-01_{'2026-01' if m == 12 else f'2025-{m + 1:02d}'}-01.parquet" for m in range(7, 13)]
    files.append(raw / "ranking.parquet")
    cols = ["MVT_ID_mvt", "FLIGHT_mvt", "CALLSIGN_flt", "ADEP_mvt", "ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt",
            "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "AOBT_3_flt", "ADEP_flt", "AIRCRAFT_TYPE_mvt"]
    parts = []
    for f in files:
        d = pd.read_parquet(f, columns=cols)
        d = d[d["PHASE_mvt"].eq("DEP") & d["ADEP_mvt"].isin(ICAO)].copy()
        d["src"] = "rank" if f.name == "ranking.parquet" else "train"
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)
    d["cs"] = d["CALLSIGN_flt"].fillna(d["FLIGHT_mvt"]).astype(str)
    d["pfx3"] = d["cs"].str[:3]
    d["sfx"] = d["cs"].str[3:].str.lstrip("0")
    d["fl_pfx3"] = d["FLIGHT_mvt"].astype(str).str[:3]
    d["fl_sfx"] = d["FLIGHT_mvt"].astype(str).str[3:].str.lstrip("0")
    d["tko"] = pd.to_datetime(d["MVT_TIME_UTC_mvt"], utc=True)
    d["month"] = d["tko"].dt.strftime("%Y-%m")
    d["day"] = d["tko"].dt.floor("D")
    return d, {f.name: sha(f) for f in files}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--msgs", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    u = pd.read_parquet(a.msgs, columns=["file", "kind", "ts", "station", "label", "flight", "reg", "air_status", "why",
                                          "text", "n_copies"])
    u["text"] = u["text"].fillna("").astype(str)
    u["flight"] = u["flight"].fillna("").astype(str).str.strip()
    u["ts"] = pd.to_datetime(u["ts"], unit="s", utc=True)
    u["month"] = u["ts"].dt.strftime("%Y-%m")
    u["al2"] = u["flight"].str[:2]
    u["skel"] = [skeleton(t) for t in u["text"]]
    rt = [route_of(t) for t in u["text"]]
    u["orig"] = [r[0] for r in rt]
    u["dest"] = [r[1] for r in rt]
    u["rkind"] = [r[2] for r in rt]
    first_ten = [ANY_TEN_RE.search(t) for t in u["text"]]
    u["airport"] = u["orig"].fillna(pd.Series([m.group(1) if m else None for m in first_ten], index=u.index))
    iata_hit = [IATA_RE.search(t) for t in u["text"]]
    u["airport"] = u["airport"].fillna(pd.Series([IATA[m.group(1)] if m else None for m in iata_hit], index=u.index))

    # 1. census
    cen = u.groupby(["label", "al2", "skel", "rkind", "month", "airport"], dropna=False).size().rename("n").reset_index()
    cen.to_parquet(out / "census.parquet", index=False)
    top = cen.groupby(["label", "al2", "skel", "rkind"], dropna=False)["n"].sum().sort_values(ascending=False)
    top.head(400).reset_index().to_csv(out / "census_top400.csv", index=False)
    print("messages", len(u), "census rows", len(cen), flush=True)

    # 2. label-free join
    d, raw_hashes = departures(Path(a.raw))
    m = u[(u["flight"].str.len() >= 3) & u["rkind"].isin(["route", "route_rev", "mvt"])].copy()
    m["sfx"] = m["flight"].str[2:].str.lstrip("0")
    m["msg_i"] = np.arange(len(m))
    m["day0"] = m["ts"].dt.floor("D")
    mm = pd.concat([m.assign(day=m["day0"] + pd.Timedelta(days=k)) for k in (-1, 0, 1)], ignore_index=True)
    dk = d[["MVT_ID_mvt", "ADEP_mvt", "ADES_mvt", "pfx3", "sfx", "fl_pfx3", "fl_sfx", "tko", "day", "src", "month"]]
    j = mm.merge(dk.rename(columns={"month": "dep_month"}), on=["sfx", "day"], how="inner")
    keep = (j["tko"] >= j["ts"] - pd.Timedelta(hours=14)) & (j["tko"] <= j["ts"] + pd.Timedelta(hours=3))
    keep &= j["orig"].isna() | j["orig"].eq(j["ADEP_mvt"])
    keep &= j["dest"].isna() | j["dest"].eq(j["ADES_mvt"])
    j = j[keep].copy()
    print("raw candidate pairs", len(j), flush=True)
    # learn the airline prefix mapping from unique, route-confirmed candidates
    conf = j[j["rkind"].eq("route")]
    conf = conf[conf.groupby("msg_i")["msg_i"].transform("size").eq(1)]
    pairs = conf.groupby(["al2", "pfx3"]).size().rename("n").reset_index()
    tot = pairs.groupby("al2")["n"].transform("sum")
    pairs["share"] = pairs["n"] / tot
    mapping = pairs[(pairs["n"] >= 20) & (pairs["share"] >= 0.5)].sort_values("n", ascending=False).drop_duplicates("al2")
    amap = dict(zip(mapping["al2"], mapping["pfx3"]))
    pairs.sort_values("n", ascending=False).to_csv(out / "airline_prefix_pairs.csv", index=False)
    j = j[j["al2"].map(amap).eq(j["pfx3"])].copy()
    j["n_cand"] = j.groupby("msg_i")["msg_i"].transform("size")
    amb = j[j["n_cand"] > 1]["msg_i"].nunique()
    j = j[j["n_cand"].eq(1)].copy()
    print("matched messages", len(j), "ambiguous dropped", amb, "airlines mapped", len(amap), flush=True)
    det = d.set_index("MVT_ID_mvt")[["BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "AOBT_3_flt",
                                     "ADEP_flt", "AIRCRAFT_TYPE_mvt", "cs"]]
    j = j.join(det, on="MVT_ID_mvt")
    keep_cols = ["msg_i", "file", "kind", "ts", "station", "label", "flight", "reg", "air_status", "al2", "skel", "rkind",
                 "orig", "dest", "text", "n_copies", "MVT_ID_mvt", "ADEP_mvt", "ADES_mvt", "pfx3", "cs", "tko", "src",
                 "dep_month", "BLOCK_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "AOBT_3_flt", "ADEP_flt",
                 "AIRCRAFT_TYPE_mvt"]
    j = j[keep_cols].reset_index(drop=True)
    j["text"] = j["text"].str[:600]
    j.to_parquet(out / "matched.parquet", compression="zstd", index=False)

    # 3. token scan on July / November / December 2025 labels only
    s = j[j["src"].eq("train") & j["dep_month"].isin(DEV_MONTHS)].copy()
    rows = []
    for r in s.itertuples(index=False):
        base = r.tko.normalize()
        for tid, hhmm in tokens(r.text):
            t = base + pd.Timedelta(hours=int(hhmm[:2]), minutes=int(hhmm[2:]))
            c = np.array([(t + pd.Timedelta(days=k) - r.tko).total_seconds() for k in (-1, 0, 1)])
            t = t + pd.Timedelta(days=int(np.argmin(np.abs(c))) - 1)
            rows.append((r.label, r.al2, r.ADEP_mvt, tid, r.MVT_ID_mvt, (r.BLOCK_TIME_UTC_mvt - t).total_seconds(),
                         (r.tko - t).total_seconds()))
    tk = pd.DataFrame(rows, columns=["label", "al2", "airport", "tid", "MVT_ID_mvt", "d_blk", "d_tko"])
    tk.to_parquet(out / "tokens_dev.parquet", index=False)

    def agg(g: pd.DataFrame) -> pd.Series:
        b, t = g["d_blk"].abs(), g["d_tko"].abs()
        return pd.Series({"n": len(g), "deps": g["MVT_ID_mvt"].nunique(),
                          "blk_le10": (b <= 10).mean(), "blk_le30": (b <= 30).mean(), "blk_le60": (b <= 60).mean(),
                          "blk_le120": (b <= 120).mean(), "blk_le300": (b <= 300).mean(), "blk_med": g["d_blk"].median(),
                          "tko_le30": (t <= 30).mean(), "tko_le60": (t <= 60).mean(), "tko_le120": (t <= 120).mean()})

    st = tk.groupby(["label", "al2", "tid"]).apply(agg, include_groups=False).reset_index()
    st = st[st["n"] >= 30].sort_values("blk_le30", ascending=False)
    st.to_csv(out / "token_stats_label_airline.csv", index=False)
    st2 = tk.groupby(["label", "al2", "airport", "tid"]).apply(agg, include_groups=False).reset_index()
    st2 = st2[st2["n"] >= 30].sort_values("blk_le30", ascending=False)
    st2.to_csv(out / "token_stats_label_airline_airport.csv", index=False)
    summ = {"messages": int(len(u)), "matched_messages": int(len(j)), "ambiguous_dropped": int(amb),
            "airline_map": amap, "raw_sha256": raw_hashes,
            "matched_by_src_month": {f"{k[0]}|{k[1]}": int(v) for k, v in j.groupby(["src", "dep_month"]).size().items()}}
    (out / "scan_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: v for k, v in summ.items() if k != "raw_sha256"}, indent=1))
    print(st.head(60).to_string())


if __name__ == "__main__":
    main()
