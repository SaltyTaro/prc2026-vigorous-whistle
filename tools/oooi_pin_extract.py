"""OOOI pin screen, step 1 (runs on arcw0): stream-filter the adsb.lol Aircraft Data Links archive.

CC0 1.0 archive: https://github.com/adsblol/aircraft-data-links-2025 and -2026 (GitHub releases, zstd JSONL).
For every asset of the selected months the file is downloaded into memory, hashed (SHA-256 of the compressed
bytes), decompressed and filtered; nothing but the kept messages is written to disk.

Kept messages:
  * every label 2N message (any airline),
  * every OOOI label QP / QQ / QR / QS message,
  * every message whose text contains LFPG, LTFM or LIRF together with at least two 4-digit groups.

Usage (on arcw0):
  python oooi_pin_extract.py list   --out DIR                 # asset listing via the public GitHub API
  python oooi_pin_extract.py run    --out DIR --kinds acars,hfdl,vdl2 --workers 24
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import os
import re
import sys
import time
import urllib.request

REPOS = ["adsblol/aircraft-data-links-2025", "adsblol/aircraft-data-links-2026"]
MONTHS = ["2025-07", "2025-11", "2025-12", "2026-01", "2026-07"]
NAME_RE = re.compile(r"^adsblol-adl-(acars|vdl2|hfdl)_(\d{8})-(\d{6})_(\d{8})-(\d{6})\.jsonl(\.zst)?$")
KEY_RE = re.compile(rb'LFPG|LTFM|LIRF|"2N"|"Q[PQRS]"')
TEXT_AIRPORT_RE = re.compile(r"LFPG|LTFM|LIRF")
FOUR_DIGITS_RE = re.compile(r"(?<!\d)\d{4}(?!\d)")
OOOI = {"QP", "QQ", "QR", "QS"}
UA = {"User-Agent": "prc2026-oooi-pin-screen"}


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
    # include files that start on the previous day (they can hold messages of the 1st)
    return (start - dt.timedelta(days=1)).strftime("%Y%m%d"), (end - dt.timedelta(days=1)).strftime("%Y%m%d")


def cmd_list(out: str) -> None:
    os.makedirs(out, exist_ok=True)
    rows = []
    listing_hashes = {}
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
                    rows.append({
                        "repo": repo, "tag": r.get("tag_name"), "name": a["name"], "size": a["size"],
                        "url": a["browser_download_url"], "digest": a.get("digest"),
                        "n_assets_in_release": len(r.get("assets", [])),
                    })
            page += 1
    windows = [month_window(m) for m in MONTHS]
    sel = []
    for r in rows:
        mm = NAME_RE.match(r["name"])
        if not mm:
            continue
        d0 = mm.group(2)
        hit = [MONTHS[i] for i, (a, b) in enumerate(windows) if a <= d0 <= b]
        if hit:
            r["kind"] = mm.group(1)
            r["month"] = hit[0]
            sel.append(r)
    with open(os.path.join(out, "assets_all.json"), "w") as f:
        json.dump(rows, f)
    with open(os.path.join(out, "assets_selected.json"), "w") as f:
        json.dump(sel, f)
    with open(os.path.join(out, "listing_receipt.json"), "w") as f:
        json.dump({"retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "listing_sha256": listing_hashes,
                   "n_assets_total": len(rows), "n_selected": len(sel),
                   "max_assets_per_release": max(r["n_assets_in_release"] for r in rows)}, f, indent=1)
    by = {}
    for r in sel:
        k = (r["month"], r["kind"])
        by.setdefault(k, [0, 0])
        by[k][0] += 1
        by[k][1] += r["size"]
    for k in sorted(by):
        print(k, by[k][0], round(by[k][1] / 1e6, 1), "MB")
    print("total assets", len(rows), "selected", len(sel))


def find_acars(obj):
    """Return the innermost dict that carries an ACARS 'label'."""
    if isinstance(obj, dict):
        if "label" in obj and ("msg_text" in obj or "text" in obj or "flight" in obj or "reg" in obj or "tail" in obj):
            return obj
        for v in obj.values():
            r = find_acars(v)
            if r is not None:
                return r
    return None


def msg_time(obj) -> float | None:
    if isinstance(obj, dict):
        if "timestamp" in obj and isinstance(obj["timestamp"], (int, float)):
            return float(obj["timestamp"])
        for key in ("vdl2", "hfdl"):
            if key in obj and isinstance(obj[key], dict):
                t = obj[key].get("t")
                if isinstance(t, dict) and "sec" in t:
                    return t["sec"] + t.get("usec", 0) / 1e6
    return None


def station_of(obj) -> str | None:
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


def src_info(obj) -> tuple[str | None, str | None]:
    try:
        src = obj["vdl2"]["avlc"]["src"]
        return src.get("addr"), src.get("status")
    except Exception:  # noqa: BLE001
        pass
    try:
        src = obj["hfdl"]["lpdu"]["src"]
        return str(src.get("id")), None
    except Exception:  # noqa: BLE001
        return None, None


def keep(label: str, text: str) -> bool:
    if label == "2N" or label in OOOI:
        return True
    if text and TEXT_AIRPORT_RE.search(text) and len(FOUR_DIGITS_RE.findall(text)) >= 2:
        return True
    return False


def process(asset: dict) -> dict:
    import subprocess

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
        s = data.rfind(b"\n", 0, m.start()) + 1
        starts.add(s)
    kept = []
    bad = 0
    for s in sorted(starts):
        e = data.find(b"\n", s)
        if e < 0:
            e = len(data)
        line = data[s:e]
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
        if not keep(label, text):
            continue
        addr, status = src_info(obj)
        kept.append({
            "file": asset["name"], "kind": asset["kind"], "ts": msg_time(obj), "station": station_of(obj),
            "label": label, "flight": a.get("flight"), "reg": a.get("reg", a.get("tail")),
            "msgno": a.get("msg_num", a.get("msgno")), "blk": a.get("blk_id", a.get("block_id")),
            "mode": a.get("mode"), "addr": addr, "air_status": status, "text": text,
        })
    return {"name": asset["name"], "url": asset["url"], "size": len(raw), "sha256": sha,
            "digest_api": asset.get("digest"), "lines": n_lines, "kept": kept, "bad_json": bad,
            "secs": round(time.time() - t0, 2)}


def cmd_run(out: str, kinds: list[str], workers: int, months: list[str] | None) -> None:
    sel = json.load(open(os.path.join(out, "assets_selected.json")))
    sel = [a for a in sel if a["kind"] in kinds and (months is None or a["month"] in months)]
    done_path = os.path.join(out, "receipt_files.jsonl")
    done = set()
    if os.path.exists(done_path):
        for line in open(done_path):
            done.add(json.loads(line)["name"])
    todo = [a for a in sel if a["name"] not in done]
    print(f"selected {len(sel)}, already done {len(sel) - len(todo)}, to do {len(todo)}", flush=True)
    msg_f = open(os.path.join(out, "messages.jsonl"), "a")
    rec_f = open(done_path, "a")
    t0 = time.time()
    nbytes = 0
    nk = 0
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
                msg_f.write(json.dumps(k) + "\n")
                nk += 1
            r["month"] = a["month"]
            r["kind"] = a["kind"]
            rec_f.write(json.dumps(r) + "\n")
            nbytes += r["size"]
            if i % 200 == 0:
                msg_f.flush()
                rec_f.flush()
                el = time.time() - t0
                print(f"{i}/{len(todo)} files, {nbytes / 1e9:.2f} GB, {nk} kept, {el:.0f}s, "
                      f"{nbytes / 1e6 / max(el, 1):.1f} MB/s", flush=True)
    msg_f.close()
    rec_f.close()
    print("done", len(todo), "files", round(nbytes / 1e9, 2), "GB", nk, "kept", round(time.time() - t0), "s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "run"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--kinds", default="acars,hfdl,vdl2")
    ap.add_argument("--months", default=None)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    if a.cmd == "list":
        cmd_list(a.out)
    else:
        cmd_run(a.out, a.kinds.split(","), a.workers, a.months.split(",") if a.months else None)


if __name__ == "__main__":
    sys.exit(main())
