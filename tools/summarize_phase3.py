"""Assemble a portable Phase-3 receipt from verified runs and public results."""

import json
import math
from pathlib import Path
import sys
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json

root = Path(__file__).resolve().parents[1]
docs = root / "docs/phase3"
selection = read_json(docs / "development_results.json")
official = read_json(docs / "official_result.json")
export = read_json(root / "runs/submissions/vigorous-whistle_v2.receipt.json")
upload = read_json(root / "runs/submissions/vigorous-whistle_v2.upload.json")
if not official["scored"] or not all(r["usedPairs"] == export["rows"] for r in official["official_results"]):
    raise ValueError("Complete official scoring has not been confirmed.")
if export["sha256"] != upload["sha256"] or export["provenance"]["bundle_id"] != selection["bundle_id"]:
    raise ValueError("Export/upload/model provenance mismatch.")
refit = root / "runs/azure_arcw0_phase3" / export["provenance"]["refit_run_id"]
if sha256_file(refit / "run.json") != export["provenance"]["refit_run_sha256"]:
    raise ValueError("Refit receipt differs from the exported version.")
pages = sorted((root / "runs/phase3/leaderboard_pages").glob("page_*.json"),
               key=lambda p: int(p.stem.split("_")[-1]))
if not pages:
    raise ValueError("No public leaderboard pages were fetched.")
submissions = {}
for index, path in enumerate(pages):
    page = read_json(path)
    items = page.get("items", page.get("content", []))
    if not items:
        raise ValueError("Empty or invalid leaderboard response.")
    has_next = bool(page.get("nextCursor") or page.get("next"))
    if has_next != (index < len(pages)-1):
        raise ValueError("Leaderboard pagination is incomplete.")
    for item in items:
        if not math.isfinite(float(item["score"])):
            raise ValueError("Leaderboard score is nonfinite.")
        submissions[item["submissionId"]] = item
best = {}
for item in submissions.values():
    team = item["teamId"]
    if team not in best or item["score"] < best[team]["score"]:
        best[team] = item
standing = sorted(best.values(), key=lambda x: (x["score"], x["teamName"]))
ours = next(x for x in standing if x["teamName"] == "vigorous-whistle")
score = official["official_results"][0]["score"]
if ours["filename"] != export["filename"] or ours["score"] != score:
    raise ValueError("The full leaderboard does not confirm v2 as the team's best version.")
rank = 1 + sum(x["score"] < score for x in standing)
snapshot = {
    "created_at_utc": utc_now(), "source": "https://datacomp.opensky-network.org/api/competitions/bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard",
    "retrieval_start_utc": datetime.fromtimestamp(min(p.stat().st_mtime for p in pages), timezone.utc).isoformat(),
    "retrieval_end_utc": datetime.fromtimestamp(max(p.stat().st_mtime for p in pages), timezone.utc).isoformat(),
    "successful_submissions": len(submissions), "scored_teams": len(best),
    "team_rank": rank, "team_best": ours, "top_three": standing[:3],
    "rmse_reduction_fraction_to_third": 1-standing[2]["score"]/score,
    "pages": [{"path": p.relative_to(root).as_posix(), "sha256": sha256_file(p), "size_bytes": p.stat().st_size} for p in pages],
}
write_json(docs / "leaderboard_snapshot.json", snapshot)
previous = read_json(root / "docs/phase2/official_result.json")["official_results"][0]["score"]
report = {"created_at_utc": utc_now(), "stage": "phase3_complete", "export": export,
          "upload_event": upload, "refit": read_json(refit / "run.json"), "official": official,
          "leaderboard_snapshot": snapshot, "previous_official_rmse_seconds": previous,
          "official_rmse_improvement_seconds": previous-score,
          "official_rmse_improvement_fraction": 1-score/previous,
          "local_verification": read_json(docs / "local_verification.json"),
          "tests": {"windows": "61 passed, 1 pre-existing pandas warning",
                    "azure": "61 passed, 1 pre-existing pandas warning"}}
write_json(docs / "submission_v2.json", report)
print(json.dumps({"official_rmse": score, "improvement_percent": 100*(1-score/previous),
                  "rank": rank, "teams": len(best), "submissions": len(submissions),
                  "third_place_rmse": standing[2]["score"],
                  "cutoff_reduction_percent": 100*(1-standing[2]["score"]/score)}, indent=2))
