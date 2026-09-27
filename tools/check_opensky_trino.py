"""One-step re-check of the OpenSky Trino historical-database grant (requested 16 September 2026; PERMISSION_DENIED on 18 September).

Prints a browser login URL (OAuth2, the user's OpenSky account). Once the user has logged in, it runs `SELECT current_user`. If the
account is on the allow-list, it also counts on-ground state vectors in the LFPG, LTFM and LIRF boxes over one hour. Reads only;
writes nothing but docs/opensky/trino_check_<utc>.json. It never uses the REST-API client token from .env: Trino rejects that token
regardless of the grant.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import trino
from trino.auth import OAuth2Authentication, RedirectHandler
from trino.auth import _OAuth2TokenBearer

ROOT = Path(__file__).resolve().parents[1]
BOXES = {"LFPG": (48.97, 49.05, 2.50, 2.62), "LTFM": (41.22, 41.32, 28.68, 28.80), "LIRF": (41.76, 41.85, 12.20, 12.30)}
HOUR = int(dt.datetime(2026, 7, 15, 6, tzinfo=dt.timezone.utc).timestamp())


class PrintHandler(RedirectHandler):
    def __call__(self, url: str) -> None:
        print(f"\nOPEN THIS URL AND LOG IN WITH THE OPENSKY ACCOUNT:\n{url}\n", flush=True)


def main():
    _OAuth2TokenBearer.MAX_OAUTH_ATTEMPTS = 120
    conn = trino.dbapi.connect(host="trino.opensky-network.org", port=443, http_scheme="https", user="saltytaro",
                               catalog="minio", schema="osky", auth=OAuth2Authentication(redirect_auth_url_handler=PrintHandler()))
    record = {"checked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    cur = conn.cursor()
    try:
        cur.execute("SELECT current_user")
        record["current_user"] = cur.fetchall()[0][0]
        counts = {}
        for airport, (lat0, lat1, lon0, lon1) in BOXES.items():
            cur.execute(f"SELECT count(*), count(DISTINCT icao24) FROM state_vectors_data4 WHERE hour = {HOUR} AND onground = true "
                        f"AND lat BETWEEN {lat0} AND {lat1} AND lon BETWEEN {lon0} AND {lon1}")
            rows, aircraft = cur.fetchall()[0]
            counts[airport] = {"onground_rows": int(rows), "aircraft": int(aircraft)}
            print(airport, counts[airport], flush=True)
        record["granted"], record["onground_2026_07_15_06utc"] = True, counts
    except Exception as exc:  # noqa: BLE001 - the denial message is the result
        record["granted"], record["error"] = False, f"{type(exc).__name__}: {str(exc)[:500]}"
    print(json.dumps(record, indent=1), flush=True)
    out = ROOT / "docs/opensky" / f"trino_check_{record['checked_at_utc'][:16].replace(':', '').replace('-', '')}.json"
    out.write_text(json.dumps(record, indent=1), encoding="utf-8")
    return 0 if record["granted"] else 1


if __name__ == "__main__":
    sys.exit(main())
