"""OOOI pin screen, step 1b (runs on arcw0): pack the kept messages and the per-file receipt into parquet.

Usage: <venv python> oooi_pin_pack.py DIR
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd


def main(d: str) -> None:
    rows = [json.loads(x) for x in open(os.path.join(d, "messages.jsonl"))]
    m = pd.DataFrame(rows)
    m["ts_utc"] = pd.to_datetime(m["ts"], unit="s", utc=True)
    n0 = len(m)
    # the same message is often heard by several feeders: keep every copy but flag duplicates
    m["dup_key"] = m["label"].astype(str) + "|" + m["flight"].astype(str) + "|" + m["text"].astype(str)
    m["n_copies"] = m.groupby("dup_key")["dup_key"].transform("size")
    m = m.sort_values(["dup_key", "ts"]).reset_index(drop=True)
    m["first_copy"] = ~m.duplicated("dup_key")
    m.to_parquet(os.path.join(d, "messages_all.parquet"), compression="zstd", index=False)
    u = m[m["first_copy"]].drop(columns=["dup_key"])
    u.to_parquet(os.path.join(d, "messages_unique.parquet"), compression="zstd", index=False)
    rec = pd.DataFrame([json.loads(x) for x in open(os.path.join(d, "receipt_files.jsonl"))])
    rec["digest_ok"] = rec["digest_api"].isna() | (rec["digest_api"] == "sha256:" + rec["sha256"])
    rec[["month", "kind", "name", "url", "size", "sha256", "digest_api", "digest_ok", "lines", "bad_json"]].to_csv(
        os.path.join(d, "receipt_files.csv.gz"), index=False, compression="gzip")
    print("messages", n0, "unique", len(u))
    print(u.groupby(["kind"]).size())
    print(u["label"].value_counts().head(20))
    print("files", len(rec), "bytes", rec["size"].sum(), "digest mismatches", int((~rec["digest_ok"]).sum()),
          "bad json lines", int(rec["bad_json"].sum()), "lines", int(rec["lines"].sum()))
    print(rec.groupby(["month", "kind"]).agg(files=("name", "size"), bytes=("size", "sum"), lines=("lines", "sum")))


if __name__ == "__main__":
    main(sys.argv[1])
