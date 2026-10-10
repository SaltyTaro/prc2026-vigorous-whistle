"""Preflight limits for the single authorized OpenSky submission bucket."""

from datetime import datetime, timedelta, timezone

from .contract import FILENAME, FINAL_NAME, filename_version

ENDPOINT = "https://s3.opensky-network.org"
BUCKET = "prc-2026-vigorous-whistle"


def check_upload_window(objects: list, name: str, size: int, now: datetime | None = None, calendar_day: bool = False) -> dict:
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
    today = [item for item in submissions if item["LastModified"].astimezone(timezone.utc).date() == now.astimezone(timezone.utc).date()]
    if calendar_day:
        if len(today) >= 5:
            raise ValueError("Five submissions are present in the current UTC calendar day; wait for capacity.")
    elif len(recent) >= 5:
        raise ValueError("Five submissions are present in the last 24 hours; wait for capacity.")
    occupied = sum(item["Size"] for item in objects)
    if occupied + size > 1_000_000_000:
        raise ValueError("Upload would exceed the published 1 GB bucket allowance.")
    return {"existing_submission_versions": len(submissions), "highest_existing_version": highest,
            "submissions_in_previous_24h": len(recent), "submissions_in_utc_calendar_day": len(today), "bucket_bytes_before": occupied,
            "bucket_bytes_after_upload": occupied + size,
            "daily_count_policy": "UTC calendar day (user-directed override)" if calendar_day else "conservative rolling 24 hours while reset timezone is unspecified"}


def check_final_upload(objects: list, size: int) -> dict:
    """The final file is uploaded once: refuse an existing final object and the 1 GB bucket allowance."""
    if size <= 0:
        raise ValueError("Empty upload.")
    if any(item["Key"] == FINAL_NAME for item in objects):
        raise ValueError(f"{FINAL_NAME} already exists in the team bucket; the final file is uploaded once and never overwritten.")
    occupied = sum(item["Size"] for item in objects)
    if occupied + size > 1_000_000_000:
        raise ValueError("Upload would exceed the published 1 GB bucket allowance.")
    submissions = [item for item in objects if FILENAME.fullmatch(item["Key"])]
    return {"final_object_present": False, "existing_submission_versions": len(submissions),
            "bucket_bytes_before": occupied, "bucket_bytes_after_upload": occupied + size}


def list_team_objects(client) -> list:
    if client.meta.endpoint_url.rstrip("/") != ENDPOINT:
        raise ValueError("Remote client must use the organizer's OpenSky endpoint.")
    objects = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=BUCKET):
        objects.extend(page.get("Contents", []))
    return objects
