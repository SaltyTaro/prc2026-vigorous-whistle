"""Read-only discovery and download of the team's PRC competition data.

Credentials are read from the workspace .env, never printed or persisted.
No upload, deletion, or bucket mutation is implemented in this tool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = "https://s3.opensky-network.org"
TEAM_BUCKET = "prc-2026-vigorous-whistle"
INVENTORY = ROOT / "docs/competition/data_inventory.json"
DATA_DIR = ROOT / "data/raw"
DATA_FILENAME = re.compile(
    r"(?:training_\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}|ranking|submitting|final_ranking|final_submitting)\.parquet"
)


def make_client():
    # Read only the explicitly supplied PRC credentials, not unrelated AWS keys.
    values = dotenv_values(ROOT / ".env", interpolate=False)
    access = values.get("PRC_ACCESS_KEY")
    secret = values.get("PRC_SECRET_KEY")
    if not access or not secret:
        raise SystemExit("PRC_ACCESS_KEY and PRC_SECRET_KEY must be set in .env.")
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        region_name="us-east-1",
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            connect_timeout=15,
            read_timeout=60,
            retries={"max_attempts": 2, "mode": "standard"},
        ),
    )


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def inventory(client, requested_buckets: list[str]):
    response = client.list_buckets()
    available = sorted(item["Name"] for item in response.get("Buckets", []))
    # Listing buckets does not authorize inspecting other teams' submissions.
    selected = requested_buckets or [
        name for name in available
        if name in {TEAM_BUCKET, "prc-2026-datasets"} or "competition-data" in name
    ]
    document = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": ENDPOINT,
        "team_bucket": TEAM_BUCKET,
        "visible_buckets": available,
        "buckets": [],
    }
    for bucket in selected:
        entry = {"name": bucket, "objects": []}
        try:
            for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket):
                for item in page.get("Contents", []):
                    entry["objects"].append({
                        "key": item["Key"],
                        "size_bytes": item["Size"],
                        "etag": item["ETag"].strip('"'),
                        "last_modified_utc": item["LastModified"].isoformat(),
                    })
            entry["total_bytes"] = sum(item["size_bytes"] for item in entry["objects"])
        except ClientError as exc:
            entry["error_code"] = exc.response.get("Error", {}).get("Code", "unknown")
        document["buckets"].append(entry)
    write_json(INVENTORY, document)
    print(json.dumps(document, indent=2))


def download(client, bucket: str, names: list[str]):
    manifest = json.loads(INVENTORY.read_text(encoding="utf-8"))
    entries = next((item for item in manifest["buckets"] if item["name"] == bucket), None)
    if entries is None:
        raise SystemExit("Run inventory for this bucket first.")
    objects = [item for item in entries["objects"] if DATA_FILENAME.fullmatch(item["key"])]
    if names:
        unknown = set(names) - {item["key"] for item in objects}
        if unknown:
            raise SystemExit(f"Files not present in the data inventory: {sorted(unknown)}")
        objects = [item for item in objects if item["key"] in names]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    receipt_path = ROOT / "docs/competition/download_manifest.json"
    previous = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists() else {"objects": []}
    records = {(row["bucket"], row["key"]): row for row in previous["objects"]}
    for item in objects:
        target = DATA_DIR / item["key"]
        old_record = records.get((bucket, item["key"]), {})
        digest = None
        if (target.exists() and target.stat().st_size == item["size_bytes"]
                and old_record.get("etag") == item["etag"]):
            with target.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest is None or digest != old_record.get("sha256"):
            temporary = target.with_suffix(target.suffix + ".part")
            client.download_file(bucket, item["key"], str(temporary))
            if temporary.stat().st_size != item["size_bytes"]:
                raise RuntimeError(f"Unexpected download size: {item['key']}")
            temporary.replace(target)
            with target.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
        records[(bucket, item["key"])] = {**item, "bucket": bucket, "sha256": digest}
        print(f"Verified {item['key']}: {item['size_bytes']:,} bytes", flush=True)
    write_json(receipt_path, {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": ENDPOINT,
        "objects": [records[key] for key in sorted(records)],
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("inventory")
    listing.add_argument("--bucket", action="append", default=[])
    fetching = commands.add_parser("download")
    fetching.add_argument("--bucket", required=True)
    fetching.add_argument("--file", action="append", default=[])
    args = parser.parse_args()
    try:
        client = make_client()
        if args.command == "inventory":
            inventory(client, args.bucket)
        else:
            download(client, args.bucket, args.file)
    except ClientError as exc:
        # Do not print signed requests or credentials in exceptions.
        code = exc.response.get("Error", {}).get("Code", "unknown")
        raise SystemExit(f"OpenSky S3 request failed: {code}") from None
    except BotoCoreError as exc:
        raise SystemExit(f"OpenSky S3 connection failed: {type(exc).__name__}") from None


if __name__ == "__main__":
    main()
