"""OOOI extension screen, step 1b (runs on the VM): pack kept messages and the per-file receipt into parquet.

Usage: <python with pandas/pyarrow> oooi_ext_pack.py DIR
Reads DIR/messages_<month>.jsonl one month at a time; writes DIR/messages_unique.parquet (first copy of each distinct
label|flight|text within a month, with the copy count), DIR/receipt_files.csv.gz and prints a summary.
"""
from __future__ import annotations

import json
import os
import sys

import pandas as pd


def main(d: str) -> None:
    parts, n_all = [], 0
    for f in sorted(os.listdir(d)):
        if not (f.startswith("messages_") and f.endswith(".jsonl")):
            continue
        rows = [json.loads(x) for x in open(os.path.join(d, f))]
        m = pd.DataFrame(rows)
        del rows
        n_all += len(m)
        key = m["label"].astype(str) + "|" + m["flight"].astype(str) + "|" + m["text"].astype(str)
        m["n_copies"] = key.map(key.value_counts()).to_numpy()
        m = m.assign(_k=key).sort_values(["_k", "ts"])
        m = m[~m["_k"].duplicated()].drop(columns=["_k"])
        m["file_month"] = f[len("messages_"):-len(".jsonl")]
        parts.append(m)
        print(f, n_all, "cum; unique this month", len(m), flush=True)
    u = pd.concat(parts, ignore_index=True)
    u["ts_utc"] = pd.to_datetime(u["ts"], unit="s", utc=True)
    u.to_parquet(os.path.join(d, "messages_unique.parquet"), compression="zstd", index=False)
    rec = pd.DataFrame([json.loads(x) for x in open(os.path.join(d, "receipt_files.jsonl"))])
    rec["digest_ok"] = rec["digest_api"].isna() | (rec["digest_api"] == "sha256:" + rec["sha256"])
    rec[["month", "kind", "name", "url", "size", "sha256", "digest_api", "digest_ok", "lines", "bad_json"]].to_csv(
        os.path.join(d, "receipt_files.csv.gz"), index=False, compression="gzip")
    print("messages", n_all, "unique (within month)", len(u))
    print(u["why"].value_counts())
    print(u["label"].value_counts().head(30))
    print("files", len(rec), "bytes", int(rec["size"].sum()), "digest mismatches", int((~rec["digest_ok"]).sum()),
          "bad json lines", int(rec["bad_json"].sum()), "lines", int(rec["lines"].sum()))
    print(rec.groupby(["month", "kind"]).agg(files=("name", "size"), bytes=("size", "sum"), lines=("lines", "sum")))


if __name__ == "__main__":
    main(sys.argv[1])
