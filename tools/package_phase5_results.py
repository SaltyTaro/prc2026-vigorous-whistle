"""Archive completed Phase-5 models and selection evidence without credentials."""

import argparse
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prc2026.artifacts import read_json, sha256_file, utc_now, verify_receipts, write_json


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/phase5"))
    parser.add_argument("--followup-runs", type=Path, default=Path("runs/phase5_followup"))
    parser.add_argument("--output", type=Path, default=Path("runs/phase5-models.tar.gz"))
    args = parser.parse_args()
    destination = root / args.output
    if destination.exists() or destination.with_suffix(".receipt.json").exists():
        raise ValueError("Use a fresh archive name; existing evidence is immutable.")
    files, run_count = set(), 0
    for directory in [root / args.runs, root / args.followup_runs]:
        if not directory.resolve().is_relative_to(root.resolve()):
            raise ValueError("Only this workspace's phase results can be archived.")
        for path in directory.glob("*/run.json"):
            report = read_json(path)
            if report["audit_used"] or report["stage"] not in ["development", "refit"]:
                raise ValueError("Unexpected model stage or audit use.")
            verify_receipts(path.parent, report["artifacts"])
            files.add(path)
            files.update(path.parent / item["path"] for item in report["artifacts"])
            run_count += 1
    if not run_count:
        raise ValueError("No completed models to archive.")
    selection_path = root / args.runs / "selection.json"
    selection = read_json(selection_path)
    files.update([selection_path, root / args.runs / "seed_plan.json"])
    for candidate in selection["candidates"].values():
        for fold in candidate["folds"].values():
            artifact = fold["prediction_artifact"]
            verify_receipts(root / args.runs, [artifact])
            files.add(root / args.runs / artifact["path"])
    for name in ["initial_results.json", "development_results.json", "airport_errors.csv", "verification.json"]:
        path = root / "docs/phase5" / name
        if path.exists():
            files.add(path)
    for name in ["phase5-console.log", "phase5-remaining-console.log", "phase5-refit-console.log", "phase5-verification-console.log"]:
        path = root / "runs" / name
        if path.exists():
            files.add(path)
    paths = sorted(files)
    for path in paths:
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Unexpected or escaping artifact path.")
    with tarfile.open(destination, "w:gz", compresslevel=3) as archive:
        for path in paths:
            archive.add(path, arcname=path.relative_to(root).as_posix(), recursive=False)
    receipt = {"created_at_utc": utc_now(), "sha256": sha256_file(destination), "size_bytes": destination.stat().st_size,
               "completed_runs": run_count, "files": [p.relative_to(root).as_posix() for p in paths]}
    write_json(destination.with_suffix(".receipt.json"), receipt)
    print({"archive": destination.name, "size_bytes": receipt["size_bytes"], "sha256": receipt["sha256"], "completed_runs": run_count})


if __name__ == "__main__":
    main()
