"""Archive public standings after the authorized v7 submission."""

from pathlib import Path
import json
import math
import sys
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026.artifacts import sha256_file, utc_now, write_json

URL = "https://datacomp.opensky-network.org/api/competitions/bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard"
OUTPUT = ROOT / "docs/surface_refit/leaderboard_snapshot.json"
RAW = ROOT / "docs/surface_refit/leaderboard_pages"


def main():
    if OUTPUT.exists() or RAW.exists():
        raise ValueError("Preserve the original post-submission snapshot.")
    RAW.mkdir(parents=True)
    start = utc_now()
    rows, pages, cursor, seen = [], [], None, set()
    for number in range(1, 51):
        query = {"limit": 200}
        if cursor:
            query["cursor"] = cursor
        address = URL + "?" + urllib.parse.urlencode(query)
        for attempt, delay in enumerate([0, 2, 4]):
            if delay:
                time.sleep(delay)
            try:
                request = urllib.request.Request(
                    address, headers={"User-Agent": "prc2026-vigorous-whistle/0.1"}
                )
                with urllib.request.urlopen(request, timeout=30) as response:
                    payload = response.read()
                break
            except Exception:
                if attempt == 2:
                    raise
        path = RAW / f"page_{number}.json"
        path.write_bytes(payload)
        page = json.loads(payload)
        rows.extend(page.get("items", page.get("content", [])))
        pages.append({"path": path.relative_to(ROOT).as_posix(),
                      "sha256": sha256_file(path), "size_bytes": len(payload)})
        cursor = page.get("nextCursor") or page.get("next")
        if not cursor:
            break
        if cursor in seen:
            raise ValueError("Repeated public leaderboard cursor.")
        seen.add(cursor)
    else:
        raise ValueError("Unexpected leaderboard pagination.")

    unique = {row["submissionId"]: row for row in rows}
    best = {}
    for row in unique.values():
        score = float(row["score"])
        if not math.isfinite(score):
            continue
        if row["teamId"] not in best or score < float(best[row["teamId"]]["score"]):
            best[row["teamId"]] = row
    ordered = sorted(best.values(), key=lambda row: (float(row["score"]), row["teamName"]))
    ours = next(row for row in ordered if row["teamName"] == "vigorous-whistle")
    if ours["filename"] != "vigorous-whistle_v7.parquet" or ours["usedPairs"] != 344841:
        raise ValueError("Public snapshot does not show the complete v7 result as the team best.")
    rank = 1 + sum(float(row["score"]) < float(ours["score"]) for row in ordered)
    fields = ["teamName", "teamId", "filename", "score", "usedPairs", "processedAt", "submissionId"]
    trim = lambda row: {key: row[key] for key in fields}
    report = {
        "created_at_utc": utc_now(), "retrieval_start_utc": start, "source": URL,
        "pages": pages, "successful_submissions": len(unique), "scored_teams": len(ordered),
        "team_rank": rank, "team_best": trim(ours),
        "top_three": [trim(row) for row in ordered[:3]],
        "rmse_reduction_fraction_to_third": 1 - float(ordered[2]["score"]) / float(ours["score"]),
        "best_by_team": [trim(row) for row in ordered],
        "interpretation": "Dated public standings after v7; this read-only tool accesses no hidden labels.",
        "tool_sha256": sha256_file(Path(__file__)),
    }
    write_json(OUTPUT, report)
    print(json.dumps({key: report[key] for key in [
        "created_at_utc", "team_rank", "scored_teams", "team_best", "top_three",
        "rmse_reduction_fraction_to_third"
    ]}, indent=2))


if __name__ == "__main__":
    main()
