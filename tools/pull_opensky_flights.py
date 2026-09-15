"""Pull OpenSky flight records (firstSeen/lastSeen per flight) for competition airports and dates.

Uses the OpenSky REST API with an OAuth2 client (OPENSKY_CLIENT_ID and
OPENSKY_CLIENT_SECRET in the workspace .env; never printed). Requests use
two-day windows (30 credits each under the published schedule) and are saved
raw with hashes under data/external/opensky/flights/<airport>/. Existing
windows are never re-downloaded. Nothing else is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import urllib.error
import urllib.parse
import urllib.request

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
TOKEN_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
API = "https://opensky-network.org/api"
OUTPUT = ROOT / "data/external/opensky/flights"
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]


def credentials() -> tuple[str, str]:
    values = dotenv_values(ROOT / ".env", interpolate=False)
    client_id, secret = values.get("OPENSKY_CLIENT_ID"), values.get("OPENSKY_CLIENT_SECRET")
    if not client_id or not secret:
        raise SystemExit("OPENSKY_CLIENT_ID and OPENSKY_CLIENT_SECRET must be set in .env.")
    return client_id, secret


class Token:
    def __init__(self):
        self.value, self.expires = None, 0.0

    def get(self) -> str:
        if self.value is None or time.time() > self.expires - 60:
            client_id, secret = credentials()
            body = urllib.parse.urlencode({"grant_type": "client_credentials", "client_id": client_id, "client_secret": secret}).encode()
            request = urllib.request.Request(TOKEN_URL, data=body, headers={"Content-Type": "application/x-www-form-urlencoded"})
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read())
            self.value, self.expires = payload["access_token"], time.time() + float(payload.get("expires_in", 1800))
        return self.value


def fetch(token: Token, endpoint: str, airport: str, begin: int, end: int) -> tuple[int, bytes, dict]:
    query = urllib.parse.urlencode({"airport": airport, "begin": begin, "end": end})
    for attempt in range(6):
        request = urllib.request.Request(f"{API}/flights/{endpoint}?{query}",
                                         headers={"Authorization": f"Bearer {token.get()}", "User-Agent": "prc2026-vigorous-whistle/0.1"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.status, response.read(), dict(response.headers)
        except urllib.error.HTTPError as error:
            headers = dict(error.headers)
            if error.code == 404:
                return 404, b"[]", headers
            if error.code == 401:
                token.value = None
                continue
            if error.code == 429:
                wait = float(headers.get("X-Rate-Limit-Retry-After-Seconds", 60))
                print(f"rate limited; waiting {wait:.0f} s", flush=True)
                time.sleep(min(wait, 3600) + 1)
                continue
            if error.code >= 500:
                time.sleep(10 * (attempt + 1))
                continue
            raise
    raise RuntimeError("Repeated request failures.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--airports", nargs="+", default=AIRPORTS)
    parser.add_argument("--start", required=True, help="First UTC date, inclusive (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="Last UTC date, exclusive (YYYY-MM-DD)")
    parser.add_argument("--endpoint", choices=["departure", "arrival"], default="departure")
    parser.add_argument("--max-credits", type=int, default=3600, help="Stop before spending more than this many credits (30 per two-day window)")
    parser.add_argument("--window-days", type=int, default=2)
    args = parser.parse_args()
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    if not 1 <= args.window_days <= 2 or end <= start:
        raise SystemExit("Windows must be one or two days and the range non-empty.")
    token = Token()
    spent, downloaded, skipped = 0, 0, 0
    for airport in args.airports:
        directory = OUTPUT / airport
        directory.mkdir(parents=True, exist_ok=True)
        cursor = start
        while cursor < end:
            stop = min(cursor + timedelta(days=args.window_days), end)
            name = f"{args.endpoint}_{cursor:%Y%m%d}_{stop:%Y%m%d}"
            path = directory / (name + ".json")
            if path.exists() and (directory / (name + ".meta.json")).exists():
                skipped += 1
                cursor = stop
                continue
            if spent + 30 > args.max_credits:
                print(f"credit budget reached ({spent}); stopping before {airport} {name}", flush=True)
                return
            status, payload, headers = fetch(token, args.endpoint, airport, int(cursor.timestamp()), int(stop.timestamp()))
            records = json.loads(payload) if status == 200 else []
            path.write_bytes(payload if status == 200 else b"[]")
            meta = {"airport": airport, "endpoint": args.endpoint, "begin_utc": cursor.isoformat(), "end_utc": stop.isoformat(),
                    "begin": int(cursor.timestamp()), "end": int(stop.timestamp()), "http_status": status, "records": len(records),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                    "rate_limit_remaining": headers.get("X-Rate-Limit-Remaining"), "source": f"{API}/flights/{args.endpoint}",
                    "license_note": "OpenSky Network data; account terms of use apply; retrieved for the PRC Data Challenge 2026 evaluation only."}
            (directory / (name + ".meta.json")).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            spent += 30
            downloaded += 1
            print(f"{airport} {name}: status {status}, {len(records)} records, remaining credits {meta['rate_limit_remaining']}", flush=True)
            time.sleep(1.0)
            cursor = stop
    print({"downloaded": downloaded, "skipped_existing": skipped, "credits_spent_estimate": spent})


if __name__ == "__main__":
    main()
