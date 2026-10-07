"""Rehearsal of v70 (docs/final_file/v69_chain.md, step 11): v69's chain rebuilt by tools/rehearse_v69.py, then step 11, the
data-link OUT pin at LFPG (configs/oooi_pin_v1.json, selected w50), rebuilt by tools/run_oooi_pin.main imported unchanged with
only its paths redirected into runs/final_rehearsal_v70/<run>/v70/. The final file must equal the uploaded v70 byte for byte
(f1d17747...).

  --start full   (run full_chain): tools/rehearse_v69.py's stages A (v66, steps 1-10), B (v68) and C (v69) into OUT, then D
                 from the v69 this run rebuilt.
  --start v69    (run from_v69_rehearsal): D only, from the verified v69 output of the v69 rehearsal (--v69, default
                 runs/final_rehearsal_v69/full_chain/v69/vigorous-whistle_v69.parquet). Allowed only when that file is 192c4a30...
                 and every input and code file recorded by that rehearsal still has its recorded SHA-256 on this machine.
  D. step 11. The filtered data-link message tables (runs/oooi_pin/messages_unique.parquet and runs/oooi_pin/fresh/
     messages_unique.parquet, SHA-256 as recorded in docs/oooi_pin/results_v1.json) are copied into OUT/v70/oooi_pin[/fresh];
     run_oooi_pin.main() then re-parses and re-joins them (tools/oooi_pin_join.py), refits the offsets on July / November /
     December 2025 against the v69 development proxy, re-reads clauses 1-5 (August-October 2025 against rows12), re-selects
     and writes the dry run from the rehearsal's v69. Checked against the shipped intermediates (parsed and joined tables, join
     statistics), docs/oooi_pin/results_v1.json, the shipped dry-run receipt, and the published CSV
     data/external/adsblol_links/target_origin_messages.csv.gz (rewritten from the rehearsal's parsed table with the columns and
     call of tools/oooi_pin_report.py; compared decompressed, since gzip stores a time stamp).
  E. extraction binding (no download). The per-file receipts docs/oooi_pin/receipts/{archive_files,fresh_archive_files}.csv.gz
     carry every archive file's URL and SHA-256 (checked against GitHub's digest at download); every message of the two tables
     must come from a receipted file. The extraction itself is documented in EXTRACTION and is not rerun here.

  python tools/rehearse_v70.py --start full|v69 [--v69 PATH]
  python tools/rehearse_v70.py --summary        # docs/final_file/v70_rehearsal.json from runs/final_rehearsal_v70/*/rehearsal.json

Nothing is written under runs/submissions; nothing is uploaded; no official score enters.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402
import rehearse_v69 as R69  # noqa: E402

ID = "MVT_ID_mvt"
BASE = ROOT / "runs/final_rehearsal_v70"
OUT = BASE / "full_chain"   # set in main
REPORT = ROOT / "docs/final_file/v70_rehearsal.json"
V69_REHEARSAL = ROOT / "runs/final_rehearsal_v69/full_chain"
EXPECTED70 = ("runs/submissions/vigorous-whistle_v70.parquet", "f1d17747e0e384ccec520a760d42c8e55a3e88a1476daafacdf7fe1f9603099f")
SHIPPED_DRY70 = ROOT / "runs/oooi_pin/rank/dry_v70"
SHIPPED_RESULTS = ROOT / "docs/oooi_pin/results_v1.json"
PUBLISHED_CSV = ROOT / "data/external/adsblol_links/target_origin_messages.csv.gz"
CSV_COLS = ["file", "kind", "ts", "station", "label", "flight", "reg", "source", "org", "dst", "t1s", "t2s", "t1",
            "t2", "day1", "day2", "eta_s", "delay_total", "text_key"]   # tools/oooi_pin_report.py
MESSAGES = {"dev": "runs/oooi_pin/messages_unique.parquet", "fresh": "runs/oooi_pin/fresh/messages_unique.parquet"}
RECEIPTS = {
    "dev": ("docs/oooi_pin/receipts/listing_receipt.json", "docs/oooi_pin/receipts/archive_files.csv.gz"),
    "fresh": ("docs/oooi_pin/receipts/fresh_listing_receipt.json", "docs/oooi_pin/receipts/fresh_archive_files.csv.gz"),
}
SHIPPED_INTERMEDIATES = {
    "parsed": "parsed.parquet", "joined_dev": "joined_dev.parquet", "joined_rank": "joined_rank.parquet",
    "join_stats": "join_stats.json",
}
STEP11_INPUTS = [
    MESSAGES["dev"], MESSAGES["fresh"],
    "runs/stack_weight/dev/w65_bag10/proxy.parquet", "runs/daily_totals/rows12.parquet", "runs/hunt/census_rank.parquet",
    "data/raw/training_2025-01-01_2025-02-01.parquet", "data/raw/training_2025-07-01_2025-08-01.parquet",
    "data/raw/training_2025-08-01_2025-09-01.parquet", "data/raw/training_2025-09-01_2025-10-01.parquet",
    "data/raw/training_2025-10-01_2025-11-01.parquet", "data/raw/training_2025-11-01_2025-12-01.parquet",
    "data/raw/training_2025-12-01_2026-01-01.parquet", "data/raw/ranking.parquet",
    "configs/oooi_pin_v1.json", "docs/oooi_pin/results_v1.json",
    RECEIPTS["dev"][0], RECEIPTS["dev"][1], RECEIPTS["fresh"][0], RECEIPTS["fresh"][1],
    "data/external/adsblol_links/target_origin_messages.csv.gz",
    "runs/oooi_pin/rank/dry_v70/vigorous-whistle_v70.parquet", "runs/oooi_pin/rank/dry_v70/vigorous-whistle_v70.receipt.json",
] + [f"runs/oooi_pin/{f}" for f in SHIPPED_INTERMEDIATES.values()] + [f"runs/oooi_pin/fresh/{f}" for f in SHIPPED_INTERMEDIATES.values()]
STEP11_CODE = ["tools/rehearse_v70.py", "tools/run_oooi_pin.py", "tools/oooi_pin_join.py", "tools/oooi_pin_extract.py",
               "tools/oooi_pin_pack.py", "tools/run_intruder_bound.py", "tools/run_stack_resid.py"]
ARCHIVE_REPOS = ("https://github.com/adsblol/aircraft-data-links-2025/", "https://github.com/adsblol/aircraft-data-links-2026/")
EXTRACTION = {
    "source": "adsb.lol Aircraft Data Links archive, CC0 1.0: GitHub releases of adsblol/aircraft-data-links-2025 and "
              "adsblol/aircraft-data-links-2026 (zstd-compressed JSONL of ACARS, VDL2 and HFDL messages)",
    "where": "Azure arcw0 (Linux), 7 October 2026, with the repository's .venv and the zstd command-line tool; the raw archive is "
             "never stored: each file is downloaded into memory, hashed (SHA-256 of the compressed bytes), decompressed and filtered",
    "filter": "tools/oooi_pin_extract.keep: every label 2N and QP / QQ / QR / QS message, plus every message whose text names "
              "LFPG, LTFM or LIRF and has at least two 4-digit groups; tools/oooi_pin_pack.py keeps the first copy of each "
              "(label, flight, text) as messages_unique.parquet",
    "runs": {
        "dev": {
            "months": ["2025-07", "2025-11", "2025-12", "2026-01", "2026-07"],
            "commands": ["python tools/oooi_pin_extract.py list --out runs/oooi_pin",
                         "python tools/oooi_pin_extract.py run --out runs/oooi_pin --kinds acars,hfdl",
                         "python tools/oooi_pin_extract.py run --out runs/oooi_pin --kinds vdl2",
                         "python tools/oooi_pin_pack.py runs/oooi_pin"],
            "files_by_pass": {"acars,hfdl": 16400, "vdl2": 9099},
            "output": MESSAGES["dev"],
            "receipts": {"listing": RECEIPTS["dev"][0], "files": RECEIPTS["dev"][1]},
        },
        "fresh": {
            "months": ["2025-08", "2025-09", "2025-10"],
            "commands": ["python tools/oooi_pin_extract.py list --out runs/oooi_pin/fresh --months 2025-08,2025-09,2025-10",
                         "python tools/oooi_pin_extract.py run --out runs/oooi_pin/fresh --kinds acars,hfdl",
                         "python tools/oooi_pin_extract.py run --out runs/oooi_pin/fresh --kinds vdl2",
                         "python tools/oooi_pin_pack.py runs/oooi_pin/fresh"],
            "files_by_pass": {"acars,hfdl": 11526, "vdl2": 6469},
            "output": MESSAGES["fresh"],
            "receipts": {"listing": RECEIPTS["fresh"][0], "files": RECEIPTS["fresh"][1]},
        },
    },
    "month_rule": "a file belongs to a month when its start date (from the file name) lies between the day before the month's "
                  "first day and the month's last day (tools/oooi_pin_extract.month_window)",
    "worker_count": "--workers (default 24) changes only the completion order of messages.jsonl, not the set of kept messages",
    "re_extraction_note": "A new extraction lists the live GitHub releases; the receipts fix the file set used here (name, URL, size, "
                          "SHA-256, line count per file). Its messages_unique.parquet may order copies with equal time stamps "
                          "differently; the byte-for-byte check of v70 starts from the stored message tables, whose SHA-256 is "
                          "recorded in docs/oooi_pin/results_v1.json (inputs_sha256.messages_dev / messages_fresh).",
}


def sha(p: Path) -> str | None:
    return sha256_file(p) if p.exists() else None


def clean(x):
    """JSON-safe copy (non-finite floats as strings)."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    return x


def flatten(x, prefix: str = ""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from flatten(v, f"{prefix}[{i}]")
    else:
        yield prefix, x


def same(x, y) -> bool:
    if isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y):
        return True
    return x == y and isinstance(x, bool) == isinstance(y, bool)


def compare_json(shipped, mine, skip: list[str]) -> dict:
    """Leaf-by-leaf comparison of two JSON trees, exact (no tolerance); skipped keys are listed."""
    a, b = dict(flatten(shipped)), dict(flatten(mine))
    keys = [k for k in sorted(set(a) | set(b)) if not any(k == s or k.startswith(s + ".") or k.startswith(s + "[") for s in skip)]
    mism, maxd = [], 0.0
    for k in keys:
        x, y = a.get(k, "<missing>"), b.get(k, "<missing>")
        if not same(x, y):
            if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
                maxd = max(maxd, abs(float(x) - float(y)))
            mism.append({"key": k, "shipped": x, "rehearsal": y})
    return {"leaves_compared": len(keys), "leaves_equal": len(keys) - len(mism), "all_equal": not mism,
            "max_abs_numeric_diff": maxd, "mismatches_first_25": mism[:25], "skipped": skip}


def compare_file(shipped: Path, mine: Path) -> dict:
    rec = {"shipped": shipped.relative_to(ROOT).as_posix(), "shipped_sha256": sha(shipped),
           "rehearsal": mine.relative_to(ROOT).as_posix(), "rehearsal_sha256": sha(mine)}
    rec["bytes_equal"] = rec["shipped_sha256"] is not None and rec["shipped_sha256"] == rec["rehearsal_sha256"]
    if mine.suffix == ".json" and rec["shipped_sha256"] is not None:
        # JSON written by Path.write_text on Windows carries CRLF; on Linux LF
        rec["bytes_equal_modulo_crlf"] = shipped.read_bytes().replace(b"\r\n", b"\n") == mine.read_bytes().replace(b"\r\n", b"\n")
    if rec["shipped_sha256"] is None:
        rec["values_equal"] = None
    elif rec["bytes_equal"]:
        rec["values_equal"] = True
    elif mine.suffix == ".parquet":
        rec["values_equal"] = bool(pd.read_parquet(shipped).equals(pd.read_parquet(mine)))
    else:
        rec["values_equal"] = read_json(shipped) == read_json(mine)
    return rec


def extraction_binding() -> dict:
    """No download: the receipts bind every archive file (URL, size, SHA-256); every stored message comes from a receipted file."""
    out = {}
    for key, (listing, files) in RECEIPTS.items():
        rec = pd.read_csv(ROOT / files)
        lr = read_json(ROOT / listing)
        digest = rec["digest_api"].astype("string")
        recomputed_ok = digest.isna() | (digest == "sha256:" + rec["sha256"].astype("string"))
        msgs = pd.read_parquet(ROOT / MESSAGES[key], columns=["file"])
        out[key] = {
            "listing_receipt": listing, "listing_receipt_sha256": sha(ROOT / listing), "listing_retrieved_utc": lr.get("retrieved_utc"),
            "listing_pages_hashed": len(lr.get("listing_sha256", {})), "n_assets_total": lr.get("n_assets_total"),
            "n_selected": lr.get("n_selected"),
            "files_receipt": files, "files_receipt_sha256": sha(ROOT / files), "files": int(len(rec)),
            "files_equal_n_selected": int(len(rec)) == lr.get("n_selected"),
            "unique_names": int(rec["name"].nunique()), "unique_urls": int(rec["url"].nunique()),
            "urls_on_archive_repos": bool(rec["url"].astype(str).str.startswith(ARCHIVE_REPOS).all()),
            "sha256_well_formed": bool(rec["sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all()),
            "github_digest_present": int(digest.notna().sum()), "github_digest_equal_sha256": bool(recomputed_ok.all()),
            "digest_ok_column_all_true": bool(rec["digest_ok"].astype(bool).all()),
            "months": sorted(rec["month"].astype(str).unique().tolist()),
            "files_by_month_kind": {f"{m}|{k}": int(v) for (m, k), v in rec.groupby(["month", "kind"]).size().items()},
            "bytes": int(rec["size"].sum()), "lines": int(rec["lines"].sum()),
            "messages_table": MESSAGES[key], "messages_table_sha256": sha(ROOT / MESSAGES[key]), "messages": int(len(msgs)),
            "message_source_files": int(msgs["file"].nunique()),
            "every_message_from_a_receipted_file": bool(msgs["file"].isin(set(rec["name"])).all()),
        }
    out["all_ok"] = all(out[k]["files_equal_n_selected"] and out[k]["urls_on_archive_repos"] and out[k]["sha256_well_formed"]
                        and out[k]["github_digest_equal_sha256"] and out[k]["every_message_from_a_receipted_file"]
                        and out[k]["unique_urls"] == out[k]["files"] for k in RECEIPTS)
    return out


def v69_rehearsal_valid(v69: Path) -> dict:
    """--start v69 only: the v69 rehearsal's own output and every input / code file it recorded are unchanged here."""
    rp = V69_REHEARSAL / "rehearsal.json"
    rep = read_json(rp)
    bad = [p for p, h in {**rep["inputs"], **rep["code"]}.items() if sha(ROOT / p) != h]
    rec = {"report": rp.relative_to(ROOT).as_posix(), "report_sha256": sha(rp), "report_created_at_utc": rep.get("created_at_utc"),
           "report_byte_identical": rep.get("byte_identical"), "files_checked": len(rep["inputs"]) + len(rep["code"]),
           "files_changed": bad, "v69": v69.relative_to(ROOT).as_posix(), "v69_sha256": sha(v69)}
    rec["valid"] = (not bad and rec["v69_sha256"] == R69.EXPECTED[69][1] and all(rep.get("byte_identical", {}).values()))
    return rec


def stage_d(v69: Path, t: dict) -> None:
    import run_oooi_pin as P
    d = OUT / "v70"
    run, fresh = d / "oooi_pin", d / "oooi_pin" / "fresh"
    fresh.mkdir(parents=True)
    declared = read_json(SHIPPED_RESULTS)["inputs_sha256"]
    copies = {}
    for key, dst, want in [("dev", run / "messages_unique.parquet", declared["messages_dev"]),
                           ("fresh", fresh / "messages_unique.parquet", declared["messages_fresh"])]:
        shutil.copyfile(ROOT / MESSAGES[key], dst)
        copies[key] = {"from": MESSAGES[key], "sha256": sha(dst), "declared_sha256": want, "equal": sha(dst) == want}
        if sha(dst) != want:
            raise ValueError(f"{MESSAGES[key]} differs from the governing run's recorded SHA-256.")
    if sha(v69) != R69.EXPECTED[69][1]:
        raise ValueError(f"{v69} is not v69 (192c4a30...).")
    P.RUN, P.FRESH, P.V69 = run, fresh, v69
    P.OUT_DIR, P.RESULTS = d / "dry", d / "results_v1.json"
    t0 = time.time()
    P.main()
    wall = round(time.time() - t0, 1)
    final = d / "vigorous-whistle_v70.parquet"
    shutil.copyfile(d / "dry/vigorous-whistle_v70.parquet", final)
    up, want = EXPECTED70
    fin = {"file": final.relative_to(ROOT).as_posix(), "sha256": sha(final), "expected": want, "byte_identical": sha(final) == want,
           "shipped_dry_run_sha256": sha(SHIPPED_DRY70 / "vigorous-whistle_v70.parquet")}
    if (ROOT / up).exists():
        fin["uploaded_file_sha256"] = sha(ROOT / up)
    # independent check of the step: only in-scope LFPG rows move against v69, row order and dtypes kept
    a, b = pd.read_parquet(v69), pd.read_parquet(final)
    rk = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=[ID, "PHASE_mvt", "ADEP_mvt"])
    rk = rk[rk["PHASE_mvt"].eq("DEP")]
    ch = a["TAXITIME_SEC_mvt"].ne(b["TAXITIME_SEC_mvt"])
    ap_ = a.loc[ch, [ID]].merge(rk[[ID, "ADEP_mvt"]], on=ID, how="left")["ADEP_mvt"]
    diff = (b["TAXITIME_SEC_mvt"].astype(float) - a["TAXITIME_SEC_mvt"].astype(float))[ch]
    fin["vs_v69"] = {"same_ids_and_order": bool(a[ID].equals(b[ID])), "same_dtypes": bool((a.dtypes == b.dtypes).all()),
                     "rows_changed": int(ch.sum()), "rows_changed_by_airport": {k: int(v) for k, v in ap_.value_counts().items()},
                     "squared_change": float((diff ** 2).sum()), "max_up_s": float(diff.max()), "max_down_s": float(diff.min()),
                     "guard_rows_changed": int(ch[a[ID].isin([213142471, 213142469])].sum())}
    inter = {}
    for sub, shipped_dir, mine_dir in [("dev", ROOT / "runs/oooi_pin", run), ("fresh", ROOT / "runs/oooi_pin/fresh", fresh)]:
        for k, f in SHIPPED_INTERMEDIATES.items():
            inter[f"{sub}/{k}"] = compare_file(shipped_dir / f, mine_dir / f)
    inter["dry_run"] = compare_file(SHIPPED_DRY70 / "vigorous-whistle_v70.parquet", d / "dry/vigorous-whistle_v70.parquet")
    res = read_json(d / "results_v1.json")
    results_cmp = compare_json(read_json(SHIPPED_RESULTS), res,
                               ["computed_at_utc", "code_sha256", "dry_run.written_at_utc", "dry_run.base", "dry_run.file"])
    receipt_cmp = compare_json(read_json(SHIPPED_DRY70 / "vigorous-whistle_v70.receipt.json"),
                               read_json(d / "dry/vigorous-whistle_v70.receipt.json"), ["written_at_utc", "base", "file"])
    # the published CSV, rewritten from the rehearsal's parsed table as tools/oooi_pin_report.py writes it; that call uses pandas'
    # default line terminator os.linesep, CRLF on the Windows workstation that wrote the published file, so CRLF is passed here
    p = pd.read_parquet(run / "parsed.parquet")
    mine_csv = d / "target_origin_messages.csv.gz"
    p[[c for c in CSV_COLS if c in p.columns]].to_csv(mine_csv, index=False, compression="gzip", lineterminator="\r\n")
    pub, mine = gzip.decompress(PUBLISHED_CSV.read_bytes()), gzip.decompress(mine_csv.read_bytes())
    csv_cmp = {"published": PUBLISHED_CSV.relative_to(ROOT).as_posix(), "published_sha256": sha(PUBLISHED_CSV),
               "published_decompressed_sha256": hashlib.sha256(pub).hexdigest(),
               "rehearsal_decompressed_sha256": hashlib.sha256(mine).hexdigest(), "decompressed_equal": pub == mine,
               "line_terminator": "CRLF (pandas' os.linesep default where the published file was written)", "rows": int(len(p))}
    t["v70"] = {"wall_seconds": wall, "v69_input": {"file": v69.relative_to(ROOT).as_posix(), "sha256": sha(v69)},
                "message_tables": copies, "selected": res.get("selected"), "offsets_jul_nov_dec": res.get("offsets_jul_nov_dec"),
                "rows_in_scope": res.get("dry_run", {}).get("rows_in_scope"), "final": fin, "intermediates": inter,
                "intermediates_values_equal_shipped": all(v["values_equal"] for v in inter.values()),
                "intermediates_bytes_equal_shipped": all(v["bytes_equal"] for v in inter.values()),
                "results_v1_vs_shipped": results_cmp, "dry_run_receipt_vs_shipped": receipt_cmp, "published_csv": csv_cmp,
                "rehearsal_results": (d / "results_v1.json").relative_to(ROOT).as_posix()}
    print("v70", json.dumps(fin), "intermediates equal", t["v70"]["intermediates_values_equal_shipped"],
          "results equal", results_cmp["all_equal"], "receipt equal", receipt_cmp["all_equal"], "csv equal",
          csv_cmp["decompressed_equal"], flush=True)


def summarize(r: dict, path: Path) -> dict:
    st = {}
    for v, s in r["stages"].items():
        if v == "v70":
            st[v] = s
            continue
        s2 = {k: s[k] for k in s if k not in ("vs_shipped_dry_run", "preds", "trees")}
        cmp_ = s.get("vs_shipped_dry_run", {})
        s2["intermediates"] = {k: x["equal"] for k, x in cmp_.get("steps", {}).items()}
        s2["intermediates_equal_shipped_dry_run"] = cmp_.get("all_equal")
        if "preds" in s:
            s2["per_seed_ranking_predictions"] = {k: x for k, x in s["preds"].items() if k != "file"}
        st[v] = s2
    keep = ["start", "started_at_utc", "created_at_utc", "wall_seconds", "peak_rss_gib_driver", "byte_identical", "v69_rehearsal_check",
            "extraction_binding", "inputs", "inputs_unchanged", "code", "environment"]
    out = {k: r[k] for k in keep if k in r}
    out["report"] = path.relative_to(ROOT).as_posix()
    out["report_sha256"] = sha(path)
    out["stages"] = st
    return out


def write_summary() -> None:
    runs = {name: summarize(read_json(BASE / name / "rehearsal.json"), BASE / name / "rehearsal.json")
            for name in ["full_chain", "from_v69_rehearsal"] if (BASE / name / "rehearsal.json").exists()}
    doc = {"what": "rehearsal of docs/final_file/v69_chain.md steps 1-11 by tools/rehearse_v70.py on Azure arcw4 (Linux); the uploaded "
                   "v70 bytes must be reproduced exactly (v66, v68 and v69 on the way, as in v69_rehearsal.json)",
           "expected": {f"v{v}": R69.EXPECTED[v][1] for v in (66, 68, 69)} | {"v70": EXPECTED70[1]},
           "step11": {"declaration": "configs/oooi_pin_v1.json", "selected": "w50", "tool": "tools/run_oooi_pin.py (main, paths redirected)",
                      "extraction": EXTRACTION},
           "runs": runs, "official_scores_used": "none", "uploaded": False}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(clean(doc), indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print("wrote", REPORT.relative_to(ROOT).as_posix(), {k: v.get("byte_identical") for k, v in runs.items()})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", choices=["full", "v69"], default="full")
    ap.add_argument("--v69", default="runs/final_rehearsal_v69/full_chain/v69/vigorous-whistle_v69.parquet")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        write_summary()
        return
    global OUT
    OUT = BASE / ("full_chain" if a.start == "full" else "from_v69_rehearsal")
    if OUT.exists():
        raise ValueError(f"{OUT} exists; never overwrite a rehearsal.")
    v69_check = v69_rehearsal_valid(ROOT / a.v69) if a.start == "v69" else None
    if v69_check is not None and not v69_check["valid"]:
        raise ValueError(f"The v69 rehearsal output or its recorded inputs changed: {json.dumps(v69_check)[:2000]}")
    OUT.mkdir(parents=True)
    R69.OUT = OUT
    inputs = list(dict.fromkeys(R69.INPUTS + STEP11_INPUTS))
    code = list(dict.fromkeys(R69.CODE + STEP11_CODE))
    inputs_before = {p: sha(ROOT / p) for p in inputs}
    binding = extraction_binding()
    print("extraction binding", json.dumps({k: binding[k] if k == "all_ok" else {x: binding[k][x] for x in (
        "files", "github_digest_equal_sha256", "messages", "every_message_from_a_receipted_file")} for k in binding}), flush=True)
    start, t0, t = utc_now(), time.time(), {}
    with R69.PeakRSS() as mem:
        if a.start == "full":
            sh = R69.stage_a(t)
            R69.stage_b(sh, t)
            R69.stage_c(sh, t)
            v69 = OUT / "v69" / "vigorous-whistle_v69.parquet"
        else:
            v69 = ROOT / a.v69
        stage_d(v69, t)
    wall = time.time() - t0
    inputs_after = {p: sha(ROOT / p) for p in inputs}
    byte = {str(v): t[f"v{v}"]["final"]["byte_identical"] for v in (66, 68, 69, 70) if f"v{v}" in t}
    report = {
        "created_at_utc": utc_now(), "started_at_utc": start, "wall_seconds": round(wall, 1), "start": a.start,
        "peak_rss_gib_driver": round(mem.peak / 2 ** 30, 2) if mem.peak else None,
        "byte_identical": byte, "v69_rehearsal_check": v69_check, "extraction_binding": binding, "stages": t,
        "inputs": inputs_before, "inputs_unchanged": inputs_before == inputs_after,
        "code": {p: sha(ROOT / p) for p in code}, "environment": R69.versions(),
        "official_scores_used": "none", "uploaded": False,
    }
    write_json(OUT / "rehearsal.json", clean(report))
    print(json.dumps({"byte_identical": byte, "wall_seconds": report["wall_seconds"], "inputs_unchanged": report["inputs_unchanged"],
                      "extraction_binding_ok": binding["all_ok"]}, indent=1), flush=True)
    if not all(byte.values()):
        sys.exit(2)


if __name__ == "__main__":
    main()
