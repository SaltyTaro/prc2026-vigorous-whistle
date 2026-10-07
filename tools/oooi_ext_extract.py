"""OOOI extension screen, step 1 (runs on an Azure VM): stream-filter the adsb.lol Aircraft Data Links archive for every
data-link format that may carry an OUT / OFF time at the ten competition airports.

CC0 1.0 archive: https://github.com/adsblol/aircraft-data-links-2025 and -2026 (GitHub releases, zstd JSONL).
Each asset is downloaded into memory, hashed (SHA-256 of the compressed bytes), decompressed with the zstd CLI and
filtered; only kept messages are written to disk.

Kept messages:
  * every OOOI label QP / QQ / QR / QS and every label 2N / 8D message (any airline),
  * every message whose text holds one of the ten ICAO codes (EDDF EDDM EGLL EHAM LEBL LEMD LFPG LIRF LSZH LTFM) or,
    as a separate word, one of their IATA codes (FRA MUC LHR AMS BCN MAD CDG FCO ZRH IST) within 16 characters of a
    run of at least four digits (a time group),
  * every MVT-style text with an AD / AA / EA group (e.g. 'AD1012/1019 EA1310 LEMD').

Usage (on the VM):
  python3 oooi_ext_extract.py list --out DIR --months 2025-07,2025-08,...
  python3 oooi_ext_extract.py run  --out DIR --kinds acars,hfdl,vdl2 --workers 32 [--months ...]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

REPOS = ["adsblol/aircraft-data-links-2025", "adsblol/aircraft-data-links-2026"]
MONTHS = ["2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12", "2026-01", "2026-07"]
NAME_RE = re.compile(r"^adsblol-adl-(acars|vdl2|hfdl)_(\d{8})-(\d{6})_(\d{8})-(\d{6})\.jsonl(\.zst)?$")
ICAO = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
IATA = ["FRA", "MUC", "LHR", "AMS", "BCN", "MAD", "CDG", "FCO", "ZRH", "IST"]
KEY_RE = re.compile(("|".join(ICAO + IATA) + r'|"Q[PQRS]"|"2N"|"8D"|AD\d{4}|AA\d{4}|EA\d{4}').encode())
ICAO_RE = re.compile("|".join(ICAO))
IATA_RE = re.compile(r"(?<![A-Z])(?:" + "|".join(IATA) + r")(?![A-Z])")
DIGITS_RE = re.compile(r"\d{4,}")
MVT_RE = re.compile(r"(?<![A-Z0-9])(?:AD|AA|EA)\d{4}")
LABELS = {"QP", "QQ", "QR", "QS", "2N", "8D"}
UA = {"User-Agent": "prc2026-oooi-ext-screen"}


def http_get(url: str, tries: int = 6, timeout: int = 120) -> bytes:
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(60, 2 ** k))
    raise RuntimeError(f"GET failed {url}: {last}")


def month_window(month: str) -> tuple[str, str]:
    y, m = map(int, month.split("-"))
    start = dt.date(y, m, 1)
    end = dt.date(y + (m == 12), (m % 12) + 1, 1)
    return (start - dt.timedelta(days=1)).strftime("%Y%m%d"), (end - dt.timedelta(days=1)).strftime("%Y%m%d")


def cmd_list(out: str, months: list[str]) -> None:
    os.makedirs(out, exist_ok=True)
    rows, listing_hashes = [], {}
    for repo in REPOS:
        page = 1
        while True:
            url = f"https://api.github.com/repos/{repo}/releases?per_page=100&page={page}"
            raw = http_get(url)
            listing_hashes[url] = hashlib.sha256(raw).hexdigest()
            rel = json.loads(raw)
            if not rel:
                break
            for r in rel:
                for a in r.get("assets", []):
                    rows.append({"repo": repo, "tag": r.get("tag_name"), "name": a["name"], "size": a["size"],
                                 "url": a["browser_download_url"], "digest": a.get("digest")})
            page += 1
    windows = [month_window(m) for m in months]
    sel = []
    for r in rows:
        mm = NAME_RE.match(r["name"])
        if not mm:
            continue
        # a file belongs to the first listed month whose window holds its start date; month order is chronological
        hit = [months[i] for i, (a, b) in enumerate(windows) if a <= mm.group(2) <= b]
        if hit:
            r["kind"], r["month"] = mm.group(1), hit[-1] if mm.group(2)[:6] == hit[-1].replace("-", "") else hit[0]
            sel.append(r)
    seen, uniq = set(), []
    for r in sel:
        if r["name"] not in seen:
            seen.add(r["name"])
            uniq.append(r)
    json.dump(uniq, open(os.path.join(out, "assets_selected.json"), "w"))
    json.dump({"retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "listing_sha256": listing_hashes,
               "n_assets_total": len(rows), "n_selected": len(uniq), "months": months},
              open(os.path.join(out, "listing_receipt.json"), "w"), indent=1)
    by = {}
    for r in uniq:
        k = (r["month"], r["kind"])
        by.setdefault(k, [0, 0])
        by[k][0] += 1
        by[k][1] += r["size"]
    for k in sorted(by):
        print(k, by[k][0], round(by[k][1] / 1e6, 1), "MB")
    print("total assets", len(rows), "selected", len(uniq))


def find_acars(obj):
    if isinstance(obj, dict):
        if "label" in obj and ("msg_text" in obj or "text" in obj or "flight" in obj or "reg" in obj or "tail" in obj):
            return obj
        for v in obj.values():
            r = find_acars(v)
            if r is not None:
                return r
    return None


def msg_time(obj):
    if isinstance(obj, dict):
        if "timestamp" in obj and isinstance(obj["timestamp"], (int, float)):
            return float(obj["timestamp"])
        for key in ("vdl2", "hfdl"):
            if key in obj and isinstance(obj[key], dict):
                t = obj[key].get("t")
                if isinstance(t, dict) and "sec" in t:
                    return t["sec"] + t.get("usec", 0) / 1e6
    return None


def station_of(obj):
    if isinstance(obj, dict):
        for k in ("station_id", "station"):
            if k in obj and isinstance(obj[k], str):
                return obj[k]
        for key in ("vdl2", "hfdl"):
            if key in obj and isinstance(obj[key], dict):
                s = obj[key].get("station")
                if isinstance(s, str):
                    return s
    return None


def src_info(obj):
    try:
        src = obj["vdl2"]["avlc"]["src"]
        return src.get("addr"), src.get("status")
    except Exception:  # noqa: BLE001
        pass
    try:
        return str(obj["hfdl"]["lpdu"]["src"].get("id")), None
    except Exception:  # noqa: BLE001
        return None, None


def near_digits(text: str, rx: re.Pattern, span: int = 16) -> bool:
    for m in rx.finditer(text):
        lo, hi = max(0, m.start() - span), min(len(text), m.end() + span)
        if DIGITS_RE.search(text[lo:hi]):
            return True
    return False


def keep(label: str, text: str) -> str | None:
    if label in LABELS:
        return "label"
    if not text:
        return None
    if MVT_RE.search(text):
        return "mvt"
    if near_digits(text, ICAO_RE):
        return "icao"
    if near_digits(text, IATA_RE):
        return "iata"
    return None


def process(asset: dict) -> dict:
    t0 = time.time()
    raw = http_get(asset["url"])
    sha = hashlib.sha256(raw).hexdigest()
    if asset["name"].endswith(".zst"):
        data = subprocess.run(["zstd", "-dcq"], input=raw, capture_output=True, check=True).stdout
    else:
        data = raw
    n_lines = data.count(b"\n")
    starts = set()
    for m in KEY_RE.finditer(data):
        starts.add(data.rfind(b"\n", 0, m.start()) + 1)
    kept, bad = [], 0
    for s in sorted(starts):
        e = data.find(b"\n", s)
        line = data[s:(e if e >= 0 else len(data))]
        try:
            obj = json.loads(line)
        except Exception:  # noqa: BLE001
            bad += 1
            continue
        a = find_acars(obj)
        if a is None:
            continue
        label = str(a.get("label") or "")
        text = a.get("msg_text", a.get("text")) or ""
        if not isinstance(text, str):
            text = str(text)
        why = keep(label, text)
        if why is None:
            continue
        addr, status = src_info(obj)
        kept.append({"file": asset["name"], "kind": asset["kind"], "ts": msg_time(obj), "station": station_of(obj),
                     "label": label, "flight": a.get("flight"), "reg": a.get("reg", a.get("tail")),
                     "msgno": a.get("msg_num", a.get("msgno")), "blk": a.get("blk_id", a.get("block_id")),
                     "mode": a.get("mode"), "addr": addr, "air_status": status, "why": why, "text": text})
    return {"name": asset["name"], "url": asset["url"], "size": len(raw), "sha256": sha, "digest_api": asset.get("digest"),
            "lines": n_lines, "kept": kept, "bad_json": bad, "secs": round(time.time() - t0, 2)}


def cmd_run(out: str, kinds: list[str], workers: int, months: list[str] | None) -> None:
    sel = json.load(open(os.path.join(out, "assets_selected.json")))
    sel = [a for a in sel if a["kind"] in kinds and (months is None or a["month"] in months)]
    done_path = os.path.join(out, "receipt_files.jsonl")
    done = set()
    if os.path.exists(done_path):
        done = {json.loads(x)["name"] for x in open(done_path)}
    todo = [a for a in sel if a["name"] not in done]
    print(f"selected {len(sel)}, already done {len(sel) - len(todo)}, to do {len(todo)}", flush=True)
    msg_f = {m: open(os.path.join(out, f"messages_{m}.jsonl"), "a") for m in sorted({a["month"] for a in sel})}
    rec_f = open(done_path, "a")
    t0, nbytes, nk = time.time(), 0, 0
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process, a): a for a in todo}
        for i, fu in enumerate(cf.as_completed(futs), 1):
            a = futs[fu]
            try:
                r = fu.result()
            except Exception as e:  # noqa: BLE001
                print("FAILED", a["name"], e, flush=True)
                continue
            for k in r.pop("kept"):
                msg_f[a["month"]].write(json.dumps(k) + "\n")
                nk += 1
            r["month"], r["kind"] = a["month"], a["kind"]
            rec_f.write(json.dumps(r) + "\n")
            nbytes += r["size"]
            if i % 500 == 0:
                for f in msg_f.values():
                    f.flush()
                rec_f.flush()
                el = time.time() - t0
                print(f"{i}/{len(todo)} files, {nbytes / 1e9:.2f} GB, {nk} kept, {el:.0f}s, {nbytes / 1e6 / max(el, 1):.1f} MB/s", flush=True)
    for f in msg_f.values():
        f.close()
    rec_f.close()
    print("done", len(todo), "files", round(nbytes / 1e9, 2), "GB", nk, "kept", round(time.time() - t0), "s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "run"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--kinds", default="acars,hfdl,vdl2")
    ap.add_argument("--months", default=None)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    months = a.months.split(",") if a.months else None
    if a.cmd == "list":
        cmd_list(a.out, months or MONTHS)
    else:
        cmd_run(a.out, a.kinds.split(","), a.workers, months)


if __name__ == "__main__":
    sys.exit(main())
