"""OOOI pin screen, step 4 (local): assemble docs/oooi_pin/results.json and copy the download receipts.

Reads runs/oooi_pin/{eval.json, join_stats.json, listing_receipt.json, receipt_files.csv.gz, parsed.parquet,
messages_unique.parquet, messages_all.parquet} and writes docs/oooi_pin/results.json,
docs/oooi_pin/receipts/{listing_receipt.json, archive_files.csv.gz} and
data/external/adsblol_links/target_origin_messages.csv.gz (the parsed LFPG / LTFM / LIRF-origin messages).
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "oooi_pin"
DOC = ROOT / "docs" / "oooi_pin"
EXT = ROOT / "data" / "external" / "adsblol_links"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    (DOC / "receipts").mkdir(parents=True, exist_ok=True)
    EXT.mkdir(parents=True, exist_ok=True)
    ev = json.loads((RUN / "eval.json").read_text())
    js = json.loads((RUN / "join_stats.json").read_text())
    lr = json.loads((RUN / "listing_receipt.json").read_text())
    shutil.copyfile(RUN / "listing_receipt.json", DOC / "receipts" / "listing_receipt.json")
    shutil.copyfile(RUN / "receipt_files.csv.gz", DOC / "receipts" / "archive_files.csv.gz")
    rec = pd.read_csv(RUN / "receipt_files.csv.gz")
    p = pd.read_parquet(RUN / "parsed.parquet")
    cols = ["file", "kind", "ts", "station", "label", "flight", "reg", "source", "org", "dst", "t1s", "t2s", "t1",
            "t2", "day1", "day2", "eta_s", "delay_total", "text_key"]
    p[[c for c in cols if c in p.columns]].to_csv(EXT / "target_origin_messages.csv.gz", index=False,
                                                    compression="gzip")
    by_month_kind = rec.groupby(["month", "kind"]).agg(files=("name", "size"), bytes=("size", "sum"),
                                                       lines=("lines", "sum"))
    res = {
        "screen": "OOOI / data-link pin of the recorded off-block (LFPG / LTFM / LIRF), 7 October 2026",
        "archive": {
            "source": "adsb.lol Aircraft Data Links archive, CC0 1.0 (GitHub releases)",
            "repositories": ["https://github.com/adsblol/aircraft-data-links-2025",
                             "https://github.com/adsblol/aircraft-data-links-2026"],
            "listing": lr,
            "files_read": int(len(rec)), "bytes_read": int(rec["size"].sum()), "lines_read": int(rec["lines"].sum()),
            "github_digest_mismatches": int((~rec["digest_ok"]).sum()),
            "by_month_kind": {f"{a}|{b}": {k: int(v) for k, v in r.items()} for (a, b), r in
                              by_month_kind.iterrows()},
            "per_file_receipt": {"path": "docs/oooi_pin/receipts/archive_files.csv.gz",
                                 "sha256": sha(DOC / "receipts" / "archive_files.csv.gz"),
                                 "columns": "month, kind, name, url, size, sha256 (of the compressed bytes), "
                                            "digest_api (GitHub's sha256 digest), digest_ok, lines, bad_json"},
            "compute": "arcw0 (Standard_D16ds_v7), 10:00-10:15 UTC 7 October 2026, stream-filtered in memory; "
                       "deallocated 10:15 UTC",
        },
        "filtered_tables": {
            "runs/oooi_pin/messages_all.parquet": sha(RUN / "messages_all.parquet"),
            "runs/oooi_pin/messages_unique.parquet": sha(RUN / "messages_unique.parquet"),
            "runs/oooi_pin/parsed.parquet": sha(RUN / "parsed.parquet"),
            "data/external/adsblol_links/target_origin_messages.csv.gz": sha(EXT / "target_origin_messages.csv.gz"),
        },
        "join": js,
        "evaluation": ev,
    }
    (DOC / "results.json").write_text(json.dumps(res, indent=1, default=float))
    print("written", DOC / "results.json")


if __name__ == "__main__":
    main()
