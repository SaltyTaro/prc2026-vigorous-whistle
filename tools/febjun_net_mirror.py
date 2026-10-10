"""Check Tier B's mirrored network fits (runs/febjun_nets/nets) against the nodes' SHA-256 lists (runs/febjun_nets/node_hashes/<node>.txt,
written by tools/febjun_net_nodes.sh mirror) and the job lists (runs/febjun_nets/jobs/node<i>.txt), then write runs/febjun_nets/mirror.json:
per declared job its fold, variant, seed, held month, held rows, fitted months and the predictions' SHA-256. No RMSE is read or written (the
held-month RMSE of these folds is None by construction, configs/febjun_tierb_v1.json). Exit code 1 if a file differs or a job is missing.
Usage: tools/febjun_net_mirror.py [--allow-missing]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "runs/febjun_nets"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()
    node_hashes, problems = {}, []
    for listing in sorted((R / "node_hashes").glob("*.txt")):
        for line in listing.read_text(encoding="utf-8").splitlines():
            digest, path = line.split(maxsplit=1)
            path = path.strip().lstrip("*")
            if path in node_hashes and node_hashes[path][0] != digest:
                problems.append(f"{path}: two nodes disagree")
            node_hashes[path] = (digest, listing.stem)
    checked = 0
    for path, (digest, node) in sorted(node_hashes.items()):
        local = ROOT / path
        if not local.exists():
            problems.append(f"{path}: not mirrored (on {node})")
        elif sha(local) != digest:
            problems.append(f"{path}: SHA-256 differs from {node}")
        else:
            checked += 1
    runs, missing = {}, []
    for jobs in sorted((R / "jobs").glob("node*.txt")):
        for line in jobs.read_text(encoding="utf-8").splitlines():
            config, variant, seed, fold, output, extra = line.split()
            run_dir = ROOT / output / fold
            if not (run_dir / "run.json").exists():
                missing.append(f"{output}/{fold}")
                continue
            rec = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            if rec.get("held_rmse") is not None:
                problems.append(f"{output}/{fold}: a held-month RMSE was written")
            key = f"{output}/{fold}"
            node = node_hashes.get(f"{output}/{fold}/predictions.parquet", (None, None))[1]
            runs[key] = {"config": config, "variant": variant, "seed": int(seed), "fold": fold, "held_month": rec.get("held_month"),
                         "held_rows": rec.get("held_rows"), "held_ids_sha256": rec.get("held_ids_sha256"), "fitted_months": rec.get("fitted_months"),
                         "extra_features": None if extra == "-" else extra, "node": node,
                         "predictions_sha256": sha(run_dir / "predictions.parquet"),
                         "files": sorted(p.name for p in run_dir.iterdir() if p.is_file())}
            if rec.get("held_month") != fold[len("lomo_"):] or (rec.get("fitted_months") and fold[len("lomo_"):] in rec["fitted_months"]):
                problems.append(f"{key}: held month / fitted months inconsistent")
    report = {"declaration": "configs/febjun_tierb_v1.json", "files_hash_checked": checked, "runs": len(runs), "missing": missing,
              "problems": problems, "runs_detail": runs}
    (R / "mirror.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "runs_detail"}, indent=1))
    if problems or (missing and not args.allow_missing):
        sys.exit(1)


if __name__ == "__main__":
    main()
