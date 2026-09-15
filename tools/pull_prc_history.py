"""Cache the public sources listed in the previous-editions research register.

Only public HTTPS GETs are made. Downloaded participant code is text evidence;
this tool never imports or executes it and never reads competition credentials.
Existing responses are immutable. --offline verifies hashes and rebuilds extracts.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import re
import urllib.request

from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/competition/history_20260914"
CACHE = ROOT / "docs/competition/sources/history_20260914"
LIMIT = 25 * 1024 * 1024


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fetch(spec: dict, previous: dict, offline: bool) -> dict:
    key, url, kind = spec["id"], spec["url"], spec["kind"]
    if not re.fullmatch(r"[a-z0-9_-]+", key) or not url.startswith("https://"):
        raise ValueError("Expected a safe source ID and public HTTPS URL")
    suffix = {"pdf": ".pdf", "html": ".html", "json": ".json", "text": ".txt"}[kind]
    path = CACHE / (key + suffix)
    record = dict(previous.get(key, {}))
    if record and record["url"] != url:
        raise ValueError(f"Source URL changed for {key}; use a new source ID")
    if path.exists():
        payload = path.read_bytes()
        if record and digest(payload) != record["sha256"]:
            raise ValueError(f"Cached source hash mismatch: {key}")
        if not record:
            raise ValueError(f"Unregistered cached source: {key}")
    else:
        if offline:
            raise FileNotFoundError(f"Missing cached source: {key}")
        request = urllib.request.Request(
            url, headers={"User-Agent": "PRC-vigorous-whistle-public-research/1.0"}
        )
        with urllib.request.urlopen(request, timeout=35) as response:
            payload = response.read(LIMIT + 1)
            if len(payload) > LIMIT:
                raise ValueError(f"Source exceeds {LIMIT} byte limit: {key}")
            record.update(
                id=key,
                url=url,
                retrieved_at_utc=datetime.now(timezone.utc).isoformat(),
                final_url=response.url,
                http_status=response.status,
                content_type=response.headers.get("Content-Type"),
                etag=response.headers.get("ETag"),
            )
        path.write_bytes(payload)
    record.update(
        kind=kind,
        path=path.relative_to(ROOT).as_posix(),
        bytes=len(payload),
        sha256=digest(payload),
    )
    extracted = None
    if kind == "pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(payload))
        extracted = "\n\n".join(
            f"=== PAGE {i + 1} ===\n{page.extract_text() or ''}"
            for i, page in enumerate(reader.pages)
        )
        record["pages"] = len(reader.pages)
        record["extractor"] = "pypdf " + importlib.metadata.version("pypdf")
    elif kind == "html":
        soup = BeautifulSoup(payload, "html.parser")
        main = soup.find("main") or soup.find("article") or soup.body or soup
        for item in main.find_all(["script", "style", "nav"]):
            item.decompose()
        extracted = main.get_text("\n", strip=True)
        record["links"] = [
            {"text": a.get_text(" ", strip=True), "href": a["href"]}
            for a in main.find_all("a", href=True)
        ]
    elif kind == "json":
        json.loads(payload)
    if extracted is not None:
        target = CACHE / (key + ".extracted.txt")
        target.write_text(f"Source: {url}\n\n{extracted}\n", encoding="utf-8")
        record["extract_path"] = target.relative_to(ROOT).as_posix()
        record["extract_sha256"] = digest(target.read_bytes())
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--only", nargs="+")
    args = parser.parse_args()
    specs = json.loads((REPORT / "requests.json").read_text(encoding="utf-8"))
    ids = [spec["id"] for spec in specs]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate source IDs")
    if args.only:
        if set(args.only) - set(ids):
            raise ValueError("Unknown requested source IDs")
        specs = [s for s in specs if s["id"] in args.only]
    CACHE.mkdir(parents=True, exist_ok=True)
    manifest = REPORT / "source_manifest.json"
    old = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else []
    records = {row["id"]: row for row in old}
    failures = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks = [(s, pool.submit(fetch, s, records, args.offline)) for s in specs]
        for spec, task in tasks:
            try:
                record = task.result()
                records[spec["id"]] = record
                manifest.write_text(
                    json.dumps(sorted(records.values(), key=lambda r: r["id"]),
                               indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
                )
                print(f"{spec['id']}: {record['bytes']} bytes verified", flush=True)
            except Exception as exc:
                failure = {"id": spec["id"], "url": spec["url"],
                           "error": f"{type(exc).__name__}: {exc}"}
                failures.append(failure)
                print(json.dumps(failure), flush=True)
    (REPORT / "last_pull.json").write_text(
        json.dumps({"at_utc": datetime.now(timezone.utc).isoformat(),
                    "offline": args.offline, "requested": len(specs),
                    "failures": failures}, indent=2) + "\n", encoding="utf-8"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
