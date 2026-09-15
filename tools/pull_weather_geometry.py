"""Restore the exact licensed runway snapshot from its frozen public-source receipt."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    receipt = json.loads((root/"docs/competition/weather_20260914/runway_sources.json").read_text(encoding="utf-8"))
    for item in receipt["objects"]:
        target = root/item["path"]
        if not target.resolve().is_relative_to((root/"data/external/ourairports").resolve()):
            raise ValueError("Runway input escaped its declared cache.")
        if target.is_file():
            body = target.read_bytes()
        else:
            if args.offline:
                raise ValueError("The frozen runway snapshot is missing from the local cache.")
            url = urllib.parse.urlparse(item["url"])
            if url.scheme != "https" or url.hostname not in {"ourairports.com", "davidmegginson.github.io"}:
                raise ValueError("Only the declared public runway sources may be read.")
            request = urllib.request.Request(item["url"], headers={"User-Agent": "PRC2026-reproducible-research/1.0"})
            with urllib.request.urlopen(request, timeout=45) as response:
                body = response.read(20_000_001)
            if len(body) > 20_000_000:
                raise ValueError("Unexpected public source size.")
        if len(body) != item["size_bytes"] or hashlib.sha256(body).hexdigest() != item["sha256"]:
            raise ValueError("The public source changed. Restore the archived snapshot or declare a fresh experiment; do not overwrite this receipt.")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(body)
        print(json.dumps({"path": item["path"], "sha256": item["sha256"], "verified": True}))


if __name__ == "__main__":
    main()
