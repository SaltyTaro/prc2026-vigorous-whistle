"""Upload a validated, hashed version once and check its official processing."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
import urllib.parse
import urllib.request

from botocore.exceptions import BotoCoreError, ClientError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prc2026.artifacts import read_json, sha256_file, utc_now, write_json
from prc2026_submission.contract import TEAM, filename_version, validate_submission
from prc2026_submission.remote import BUCKET, ENDPOINT, check_upload_window, list_team_objects
from prc_data import make_client

ROOT = Path(__file__).resolve().parents[1]
LEADERBOARD = "https://datacomp.opensky-network.org/api/competitions/bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard"


def upload(path: Path, expected_sha256: str, calendar_day: bool = False):
    template = ROOT / "data/raw/submitting.parquet"
    validation = validate_submission(path, template)
    receipt = read_json(path.with_suffix(".receipt.json"))
    if validation["sha256"] != expected_sha256 or receipt["sha256"] != expected_sha256:
        raise ValueError("Upload hash differs from the reviewed export.")
    if validation["template_sha256"] != receipt["template_sha256"]:
        raise ValueError("Original template changed after export.")
    if receipt["provenance"]["audit_used"]:
        raise ValueError("Phase-2 export unexpectedly used the audit period.")
    client = make_client()
    inputs = read_json(ROOT / "docs/competition/download_manifest.json")["objects"]
    for name in ["ranking.parquet", "submitting.parquet"]:
        source = next(item for item in inputs if item["key"] == name)
        current = client.head_object(Bucket="prc-2026-datasets", Key=name)
        if current["ETag"].strip('"') != source["etag"] or current["ContentLength"] != source["size_bytes"]:
            raise ValueError("Organizer ranking/template data changed; refresh before submitting.")
    objects = list_team_objects(client)
    limits = check_upload_window(objects, path.name, path.stat().st_size, calendar_day=calendar_day)
    log_path = path.with_suffix(".upload.json")
    provenance = receipt["provenance"]
    # Post-processing exports carry no refit of their own; identify them by declaration and candidate.
    run_id = provenance.get("refit_run_id") or f"{provenance['experiment_id']}:{provenance['selected_candidate']}"
    log = {"created_at_utc": utc_now(), "status": "upload_started", "endpoint": ENDPOINT,
           "bucket": BUCKET, "filename": path.name, "sha256": expected_sha256,
           "validation": validation, "limits": limits, "refit_run_id": run_id}
    write_json(log_path, log)
    payload = path.read_bytes()
    try:
        response = client.put_object(
            Bucket=BUCKET, Key=path.name, Body=payload, IfNoneMatch="*",
            ContentType="application/vnd.apache.parquet",
            ContentMD5=base64.b64encode(hashlib.md5(payload).digest()).decode("ascii"),
            Metadata={"sha256": expected_sha256, "refit-run-id": log["refit_run_id"]},
        )
    except (ClientError, BotoCoreError):
        # A response can be lost after a successful write. Check that same key,
        # without issuing another upload or allocating a new version.
        try:
            head = client.head_object(Bucket=BUCKET, Key=path.name)
        except (ClientError, BotoCoreError):
            log["status"] = "upload_not_confirmed"
            write_json(log_path, log)
            raise RuntimeError("Upload was not confirmed. Inspect this version before any retry.") from None
        if head.get("Metadata", {}).get("sha256") != expected_sha256 or head["ContentLength"] != len(payload):
            log["status"] = "remote_object_does_not_match_export"
            write_json(log_path, log)
            raise RuntimeError("Remote version does not match this export; no overwrite attempted.") from None
        response = head
    head = client.head_object(Bucket=BUCKET, Key=path.name)
    if head["ContentLength"] != len(payload) or head.get("Metadata", {}).get("sha256") != expected_sha256:
        raise RuntimeError("Post-upload object verification failed.")
    log.update(status="uploaded_processing_not_yet_confirmed", uploaded_at_utc=utc_now(),
               etag=head["ETag"].strip('"'), response_http_status=response.get("ResponseMetadata", {}).get("HTTPStatusCode"))
    write_json(log_path, log)
    print(json.dumps({"status": log["status"], "bucket": BUCKET, "filename": path.name,
                      "rows": validation["rows"], "sha256": expected_sha256}, indent=2))


def status(name: str, report_dir: Path | None = None):
    filename_version(name)
    objects = list_team_objects(make_client())
    matches, cursor, cursors = [], None, set()
    for _ in range(50):
        query = {"limit": 200, "teamName": TEAM}
        if cursor:
            query["cursor"] = cursor
        request = urllib.request.Request(LEADERBOARD + "?" + urllib.parse.urlencode(query),
                                         headers={"User-Agent": "prc2026-vigorous-whistle/0.1"})
        with urllib.request.urlopen(request, timeout=30) as response:
            page = json.load(response)
        rows = page.get("items", page.get("content", []))
        matches.extend(row for row in rows if row.get("teamName") == TEAM and row.get("filename") == name)
        cursor = page.get("nextCursor") or page.get("next")
        if not cursor:
            break
        if cursor in cursors:
            raise ValueError("Repeated leaderboard cursor.")
        cursors.add(cursor)
    else:
        raise ValueError("Leaderboard pagination did not terminate.")
    report = {
        "checked_at_utc": utc_now(), "filename": name, "bucket": BUCKET,
        "objects": [{"key": item["Key"], "size_bytes": item["Size"],
                     "last_modified_utc": item["LastModified"].isoformat()} for item in objects],
        "official_results": matches, "scored": bool(matches),
    }
    write_json(ROOT / "runs/submissions" / name.replace(".parquet", ".status.json"), report)
    if matches:
        if any(row["usedPairs"] != 344841 for row in matches):
            raise ValueError("Official scored-pair count differs from the full template.")
        # Keep historical phase reports intact when checking newer versions.
        destination = (report_dir / "official_result.json" if report_dir is not None
                       else ROOT / "docs/submissions" / name.replace(".parquet", ".status.json"))
        write_json(destination, report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sending = sub.add_parser("upload")
    sending.add_argument("--file", type=Path, required=True)
    sending.add_argument("--expected-sha256", required=True)
    sending.add_argument("--calendar-day", action="store_true", help="Count the five-per-day allowance by UTC calendar day instead of a rolling 24 hours (user-directed)")
    checking = sub.add_parser("status")
    checking.add_argument("--filename", required=True)
    checking.add_argument("--report-dir", type=Path,
                          help="Optional phase report directory; otherwise write a version-specific receipt.")
    args = parser.parse_args()
    try:
        if args.command == "upload":
            upload(ROOT / args.file, args.expected_sha256, args.calendar_day)
        else:
            status(args.filename, ROOT / args.report_dir if args.report_dir else None)
    except ClientError as exc:
        raise SystemExit(f"OpenSky request failed: {exc.response.get('Error', {}).get('Code', 'unknown')}") from None
    except BotoCoreError as exc:
        raise SystemExit(f"OpenSky connection failed: {type(exc).__name__}") from None
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
