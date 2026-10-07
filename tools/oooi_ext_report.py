"""OOOI extension screen, step 5 (local): assemble docs/oooi_ext/results.json, the census tables and the receipts.

Reads runs/oooi_ext/{listing_receipt.json, receipt_files.csv.gz, remote_sha256.txt, scan/*, screen.json}.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "oooi_ext"
DOC = ROOT / "docs" / "oooi_ext"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    (DOC / "receipts").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(RUN / "listing_receipt.json", DOC / "receipts" / "listing_receipt.json")
    shutil.copyfile(RUN / "receipt_files.csv.gz", DOC / "receipts" / "archive_files.csv.gz")
    rec = pd.read_csv(RUN / "receipt_files.csv.gz")
    cen = pd.read_parquet(RUN / "scan" / "census.parquet")
    # census: messages naming a route out of one of the ten airports, by (label, airline) and month
    dep = cen[cen["rkind"].eq("route")]
    top = dep.groupby(["label", "al2"])["n"].sum().sort_values(ascending=False).head(40)
    piv = dep.set_index(["label", "al2"]).loc[top.index].reset_index().pivot_table(
        index=["label", "al2"], columns="month", values="n", aggfunc="sum", fill_value=0)
    piv["total"] = piv.sum(axis=1)
    piv = piv.sort_values("total", ascending=False)
    piv.to_csv(DOC / "census_top40_departure_routes.csv")
    by_airport = dep.pivot_table(index="airport", columns="month", values="n", aggfunc="sum", fill_value=0)
    by_airport.to_csv(DOC / "census_departure_routes_by_airport.csv")
    allkept = cen.groupby("month")["n"].sum()
    q = json.loads((RUN / "scan" / "format_quality.json").read_text())
    scr = json.loads((RUN / "screen.json").read_text())
    summ = json.loads((RUN / "scan" / "scan_summary.json").read_text())
    res = {
        "screen": "OOOI extension: data-link formats carrying an OUT time at the ten airports, 7 October 2026",
        "archive": {"source": "adsb.lol Aircraft Data Links archive, CC0 1.0",
                    "listing": json.loads((RUN / "listing_receipt.json").read_text()),
                    "files_read": int(len(rec)), "bytes_read": int(rec["size"].sum()), "lines_read": int(rec["lines"].sum()),
                    "github_digest_mismatches": int((~rec["digest_ok"]).sum()),
                    "per_file_receipt": {"path": "docs/oooi_ext/receipts/archive_files.csv.gz",
                                         "sha256": sha(DOC / "receipts" / "archive_files.csv.gz")},
                    "copy_back_sha256": (RUN / "remote_sha256.txt").read_text().splitlines(),
                    "vm": "arcw0 only (arcw1-arcw3 not started); started 11:3x UTC, deallocated 11:54 UTC 7 October 2026"},
        "census": {"unique_messages_by_month": {k: int(v) for k, v in allkept.items()},
                   "departure_route_messages_by_airport_month": json.loads(by_airport.to_json(orient="index")),
                   "top40_departure_route_label_airline": json.loads(piv.reset_index().to_json(orient="records"))},
        "join": {k: v for k, v in summ.items() if k != "airline_map"} | {"airlines_mapped": len(summ["airline_map"])},
        "format_quality_jul_nov_dec_2025": q,
        "screen_w50_vs_v69_proxy": scr,
        "files": {p: sha(RUN / p) for p in ["messages_unique.parquet", "scan/matched.parquet", "scan/formats.parquet",
                                             "scan/tokens_dev.parquet", "screen.json"]},
    }
    (DOC / "results.json").write_text(json.dumps(res, indent=1, default=float))
    print(piv.head(25).to_string())
    print(by_airport.to_string())


if __name__ == "__main__":
    main()
