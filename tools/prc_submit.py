"""Upload a validated, hashed version once and check its official processing."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import urllib.parse
import urllib.request

from botocore.exceptions import BotoCoreError, ClientError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prc2026.artifacts import read_json, sha256_file, utc_now, write_json
from prc2026_submission.contract import FINAL_NAME, TEAM, filename_version, validate_final_submission, validate_submission
from prc2026_submission.remote import BUCKET, ENDPOINT, check_final_upload, check_upload_window, list_team_objects
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
    # Stacked overrides (tools/export_stack.py) name no single candidate; their experiment id alone identifies them.
    run_id = provenance.get("refit_run_id") or ":".join(str(x) for x in [provenance["experiment_id"], provenance.get("selected_candidate")] if x)
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


def upload_final(source: Path, expected_sha256: str, check_only: bool = False, replace_sha256: str | None = None):
    """The final phase's file: a byte copy of the reviewed final-chain export, uploaded as FINAL_NAME.

    The organisers keep the last upload. A replacement must name the SHA-256 of
    the file now in the bucket; the earlier local copy and its upload log are kept under names carrying that hash."""
    template = ROOT / "data/raw/final_submitting.parquet"
    if sha256_file(source) != expected_sha256:
        raise ValueError("Source hash differs from the reviewed final export.")
    if replace_sha256 == expected_sha256:
        raise ValueError("The replacement is the file already uploaded; nothing to do.")
    target = ROOT / "runs/submissions" / FINAL_NAME
    upload_log = target.with_name(FINAL_NAME.replace(".parquet", ".upload.json"))
    if not check_only and target.exists() and sha256_file(target) != expected_sha256:
        if replace_sha256 is None or sha256_file(target) != replace_sha256:
            raise ValueError(f"{target.relative_to(ROOT)} already holds a different file; name it with --replace-sha256 to replace it.")
    client = make_client()
    replaced = None
    if replace_sha256 is not None:
        try:
            head = client.head_object(Bucket=BUCKET, Key=FINAL_NAME)
        except ClientError:
            raise ValueError(f"--replace-sha256 was given but {FINAL_NAME} is not in the team bucket; nothing to replace.") from None
        if head.get("Metadata", {}).get("sha256") != replace_sha256:
            raise ValueError("The final file in the bucket is not the one named by --replace-sha256; nothing is replaced.")
        replaced = {"sha256": replace_sha256, "etag": head["ETag"].strip('"'), "size_bytes": head["ContentLength"],
                    "last_modified_utc": head["LastModified"].isoformat()}
    inputs = read_json(ROOT / "docs/competition/download_manifest.json")["objects"]
    organiser = {}
    for name in ["final_ranking.parquet", "final_submitting.parquet"]:
        recorded = next(item for item in inputs if item["key"] == name)
        current = client.head_object(Bucket="prc-2026-datasets", Key=name)
        if current["ETag"].strip('"') != recorded["etag"] or current["ContentLength"] != recorded["size_bytes"]:
            raise ValueError(f"Organiser {name} changed since our download; refresh and rebuild before the final upload.")
        organiser[name] = {"etag": recorded["etag"], "size_bytes": recorded["size_bytes"]}
    # Every check runs on a temporary copy under the final name; runs/submissions changes only for a real upload.
    staged = Path(tempfile.mkdtemp(prefix="prc_final_check_")) / FINAL_NAME
    staged.write_bytes(source.read_bytes())
    validation = validate_final_submission(staged, template)
    if validation["sha256"] != expected_sha256:
        raise ValueError("The staged final file differs from the reviewed export.")
    limits = check_final_upload(list_team_objects(client), staged.stat().st_size, replacing=replaced is not None)
    log_path = upload_log.with_name(FINAL_NAME.replace(".parquet", ".check.json")) if check_only else upload_log
    if not check_only:
        if target.exists() and (earlier := sha256_file(target)) != expected_sha256:
            # The earlier upload (checked above to be the file named by --replace-sha256) and its log are kept, renamed.
            stem = f"{FINAL_NAME.removesuffix('.parquet')}.replaced_{utc_now().replace(':', '').replace('-', '')[:15]}_{earlier[:12]}"
            target.rename(target.with_name(stem + ".parquet"))
            if upload_log.exists():
                upload_log.rename(upload_log.with_name(stem + ".upload.json"))
        if not target.exists():
            target.write_bytes(source.read_bytes())
        if sha256_file(target) != expected_sha256:
            raise ValueError("The staged final file differs from the reviewed export.")
    log = {"created_at_utc": utc_now(), "status": "checked_not_uploaded" if check_only else "upload_started",
           "endpoint": ENDPOINT, "bucket": BUCKET, "filename": FINAL_NAME, "sha256": expected_sha256,
           "source": source.relative_to(ROOT).as_posix() if source.is_relative_to(ROOT) else str(source),
           "organiser_inputs": organiser, "validation": validation, "limits": limits, "replaces": replaced}
    write_json(log_path, log)
    if check_only:
        print(json.dumps({"status": log["status"], "filename": FINAL_NAME, "rows": validation["rows"], "sha256": expected_sha256,
                          "replaces": replace_sha256}, indent=2))
        return
    payload = target.read_bytes()
    # A first upload must not overwrite anything; a replacement overwrites only the file checked above.
    condition = {} if replaced else {"IfNoneMatch": "*"}
    try:
        client.put_object(
            Bucket=BUCKET, Key=FINAL_NAME, Body=payload, **condition,
            ContentType="application/vnd.apache.parquet",
            ContentMD5=base64.b64encode(hashlib.md5(payload).digest()).decode("ascii"),
            Metadata={"sha256": expected_sha256},
        )
    except (ClientError, BotoCoreError):
        # As for ranked versions: a lost response after a successful write is checked on the same key, never re-sent.
        try:
            client.head_object(Bucket=BUCKET, Key=FINAL_NAME)
        except (ClientError, BotoCoreError):
            log["status"] = "upload_not_confirmed"
            write_json(log_path, log)
            raise RuntimeError("Final upload was not confirmed. Inspect the bucket before any retry.") from None
    head = client.head_object(Bucket=BUCKET, Key=FINAL_NAME)
    readback = hashlib.sha256(client.get_object(Bucket=BUCKET, Key=FINAL_NAME)["Body"].read()).hexdigest()
    if head["ContentLength"] != len(payload) or head.get("Metadata", {}).get("sha256") != expected_sha256 or readback != expected_sha256:
        log["status"] = "remote_object_does_not_match_export"
        write_json(log_path, log)
        raise RuntimeError("Post-upload verification of the final file failed.")
    log.update(status="uploaded_and_read_back", uploaded_at_utc=utc_now(), etag=head["ETag"].strip('"'),
               last_modified_utc=head["LastModified"].isoformat(), readback_sha256=readback)
    write_json(log_path, log)
    print(json.dumps({"status": log["status"], "bucket": BUCKET, "filename": FINAL_NAME,
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
    final = sub.add_parser("upload-final", help=f"the final phase's file, uploaded as {FINAL_NAME} (the last upload counts)")
    final.add_argument("--source", type=Path, required=True, help="the reviewed final-chain export (validated against final_submitting.parquet)")
    final.add_argument("--expected-sha256", required=True)
    final.add_argument("--check-only", action="store_true", help="stage and run every check, upload nothing")
    final.add_argument("--replace-sha256", help="overwrite the final file already in the bucket, which must carry this SHA-256")
    checking = sub.add_parser("status")
    checking.add_argument("--filename", required=True)
    checking.add_argument("--report-dir", type=Path,
                          help="Optional phase report directory; otherwise write a version-specific receipt.")
    args = parser.parse_args()
    try:
        if args.command == "upload":
            upload(ROOT / args.file, args.expected_sha256, args.calendar_day)
        elif args.command == "upload-final":
            upload_final((ROOT / args.source).resolve(), args.expected_sha256, args.check_only, args.replace_sha256)
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
