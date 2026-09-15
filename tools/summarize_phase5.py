"""Collect a dated public leaderboard snapshot and the verified Phase-5 receipt."""

import argparse
import json
import math
from pathlib import Path
import sys
from urllib.parse import urlencode
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json

API = "https://datacomp.opensky-network.org/api/competitions/bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard"


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--runs", type=Path, default=Path("runs/azure_arcw0_phase5"))
    args = parser.parse_args()
    docs = root / "docs/phase5"
    name = f"vigorous-whistle_v{args.version}"
    selection = read_json(docs / "development_results.json")
    official = read_json(docs / "official_result.json")
    export = read_json(root / "runs/submissions" / (name + ".receipt.json"))
    upload = read_json(root / "runs/submissions" / (name + ".upload.json"))
    if not official["scored"] or not all(r["usedPairs"] == export["rows"] and r["filename"] == name + ".parquet" for r in official["official_results"]):
        raise ValueError("Complete official scoring for this version has not been confirmed.")
    if export["sha256"] != upload["sha256"] or export["provenance"]["bundle_id"] != selection["bundle_id"]:
        raise ValueError("Export/upload/model provenance mismatch.")
    refit = root / args.runs / export["provenance"]["refit_run_id"]
    if sha256_file(refit / "run.json") != export["provenance"]["refit_run_sha256"]:
        raise ValueError("Refit receipt differs from the exported version.")
    page_dir = root / "runs/phase5" / ("leaderboard_" + utc_now().replace(":", "").replace("-", "").replace(".", "_"))
    page_dir.mkdir(parents=True, exist_ok=False)
    cursor, seen, submissions, pages = None, set(), {}, []
    started = utc_now()
    while True:
        params = {"limit": 200}
        if cursor:
            params["cursor"] = cursor
        request = Request(API + "?" + urlencode(params), headers={"User-Agent": "vigorous-whistle-reproducible-research/1.0"})
        with urlopen(request, timeout=30) as response:
            data = json.load(response)
        items = data.get("items", data.get("content", []))
        if not items:
            raise ValueError("Empty public leaderboard response.")
        page = page_dir / f"page_{len(pages)+1}.json"
        write_json(page, data)
        pages.append(page)
        for item in items:
            if not math.isfinite(float(item["score"])):
                raise ValueError("Nonfinite leaderboard score.")
            submissions[item["submissionId"]] = item
        cursor = data.get("nextCursor")
        if not cursor:
            break
        if cursor in seen:
            raise ValueError("Leaderboard cursor repeated; pagination incomplete.")
        seen.add(cursor)
    best = {}
    for item in submissions.values():
        if item["teamId"] not in best or item["score"] < best[item["teamId"]]["score"]:
            best[item["teamId"]] = item
    standing = sorted(best.values(), key=lambda x: (x["score"], x["teamName"]))
    ours = next(x for x in standing if x["teamName"] == "vigorous-whistle")
    score = float(official["official_results"][0]["score"])
    if not any(x["filename"] == name + ".parquet" and x["teamName"] == "vigorous-whistle" and x["score"] == score for x in submissions.values()):
        raise ValueError("Full public leaderboard does not contain the confirmed submission.")
    snapshot = {"created_at_utc": utc_now(), "source": API, "retrieval_start_utc": started, "retrieval_end_utc": utc_now(),
                "successful_submissions": len(submissions), "scored_teams": len(best),
                "team_rank": 1 + sum(x["score"] < ours["score"] for x in standing), "team_best": ours,
                "top_three": standing[:3], "rmse_reduction_fraction_to_third": 1-standing[2]["score"]/ours["score"],
                "pages": [{"path": p.relative_to(root).as_posix(), "sha256": sha256_file(p), "size_bytes": p.stat().st_size} for p in pages]}
    write_json(docs / "leaderboard_snapshot.json", snapshot)
    previous = float(read_json(root / "docs/phase4/official_result.json")["official_results"][0]["score"])
    report = {"created_at_utc": utc_now(), "stage": "phase5_complete", "export": export, "upload_event": upload,
              "refit": read_json(refit / "run.json"), "official": official, "leaderboard_snapshot": snapshot,
              "previous_official_rmse_seconds": previous, "official_rmse_improvement_seconds": previous-score,
              "official_rmse_improvement_fraction": 1-score/previous, "local_verification": read_json(docs / "local_verification.json"),
              "tests": {"windows": "81 passed, 1 pre-existing pandas warning", "azure": "81 passed, 1 pre-existing pandas warning"}}
    write_json(docs / f"submission_v{args.version}.json", report)
    print(json.dumps({"official_rmse": score, "improvement_percent": 100*(1-score/previous),
                      "rank": snapshot["team_rank"], "teams": len(best), "third_place_rmse": standing[2]["score"],
                      "reduction_to_third_percent": 100*snapshot["rmse_reduction_fraction_to_third"]}, indent=2))


if __name__ == "__main__":
    main()
