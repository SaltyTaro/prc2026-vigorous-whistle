"""Preflight limits for the single authorized OpenSky submission bucket."""

from datetime import datetime, timedelta, timezone

from .contract import FILENAME, filename_version

ENDPOINT = "https://s3.opensky-network.org"
BUCKET = "prc-2026-vigorous-whistle"


def check_upload_window(objects: list, name: str, size: int, now: datetime | None = None) -> dict:
    version = filename_version(name)
    now = now or datetime.now(timezone.utc)
    if size <= 0:
        raise ValueError("Empty upload.")
    if any(item["Key"] == name for item in objects):
        raise ValueError("That submission version already exists; uploads never overwrite.")
    submissions = [item for item in objects if FILENAME.fullmatch(item["Key"])]
    highest = max((filename_version(item["Key"]) for item in submissions), default=0)
    if version <= highest:
        raise ValueError("Submission version must increase beyond existing versions.")
    recent = [item for item in submissions if item["LastModified"] >= now - timedelta(hours=24)]
    if len(recent) >= 5:
        raise ValueError("Five submissions are present in the last 24 hours; wait for capacity.")
    occupied = sum(item["Size"] for item in objects)
    if occupied + size > 1_000_000_000:
        raise ValueError("Upload would exceed the published 1 GB bucket allowance.")
    return {"existing_submission_versions": len(submissions), "highest_existing_version": highest,
            "submissions_in_previous_24h": len(recent), "bucket_bytes_before": occupied,
            "bucket_bytes_after_upload": occupied + size,
            "daily_count_policy": "conservative rolling 24 hours while reset timezone is unspecified"}


def list_team_objects(client) -> list:
    if client.meta.endpoint_url.rstrip("/") != ENDPOINT:
        raise ValueError("Remote client must use the organizer's OpenSky endpoint.")
    objects = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=BUCKET):
        objects.extend(page.get("Contents", []))
    return objects
