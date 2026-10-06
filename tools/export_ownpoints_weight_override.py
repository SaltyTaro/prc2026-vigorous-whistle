"""v66 at the user's override of the own-points weight round (docs/ctx_ownpoints_weight/override_v66.json): v65's chain with own_all at
weight 1.00. Every step is tools/export_ctx_ownpoints.chain, imported unchanged; the stacker refit reads the candidate's rebuilt development
tables from runs/ctx_ownpoints_weight/chain/own_all_w100/ (written by the round's stage 2).

  --identity --version N   own_all_w40 with the v65 round's declaration and tables through the same path must give v65 byte for byte
                           (tools/run_ownpoints_weight.export_identity)
  --dry-run --version N    -> runs/ctx_ownpoints_weight/rank/dry_vN/
  --build --version N --user-decision TEXT   rerun into runs/ctx_ownpoints_weight/rank/build_vN, byte identity with the dry run, write
                           runs/submissions/vigorous-whistle_vN.parquet (never overwrite) + docs/ctx_ownpoints_weight/export_vN.json
No official score enters. Nothing is uploaded.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET  # noqa: E402
import run_ownpoints_weight as W  # noqa: E402

OVERRIDE = "docs/ctx_ownpoints_weight/override_v66.json"
CANDIDATE = "own_all_w100"
CODE = ["tools/export_ownpoints_weight_override.py", *W.CODE]


def details(result: Path, ref: Path) -> dict:
    a, v = pd.read_parquet(ref), pd.read_parquet(result)
    if not v[ID].equals(a[ID]):
        raise ValueError("Row order differs.")
    raw = pd.read_parquet(ROOT / "data/raw/ranking.parquet", columns=[ID, "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt"])
    raw = raw[raw["PHASE_mvt"].eq("DEP")].set_index(ID).reindex(v[ID])
    d = pd.Series(v[TARGET].to_numpy(dtype=float) - a[TARGET].to_numpy(dtype=float), index=v[ID].to_numpy())
    air = raw["ADEP_mvt"].to_numpy()
    month = pd.to_datetime(raw["MVT_TIME_UTC_mvt"], utc=True).dt.strftime("%Y-%m").to_numpy()
    by = {}
    for k in sorted(set(air)):
        g = d.to_numpy()[air == k]
        by[k] = {"rows": int(len(g)), "rows_changed": int((g != 0).sum()), "rms_s": float(np.sqrt(np.mean(g ** 2))), "mean_s": float(g.mean()),
                 "squared_change": float((g ** 2).sum())}
    order = np.argsort(-np.abs(d.to_numpy()))[:10]
    return {"by_airport": by, "by_month": {m: {"rms_s": float(np.sqrt(np.mean(d.to_numpy()[month == m] ** 2))), "mean_s": float(d.to_numpy()[month == m].mean())}
                                           for m in sorted(set(month))},
            "rms_s": float(np.sqrt(np.mean(d.to_numpy() ** 2))), "mean_s": float(d.mean()),
            "largest_moves": [{"MVT_ID": float(d.index[i]), "airport": str(air[i]), "change_s": float(d.iloc[i]), "v65": float(a[TARGET].iloc[i]),
                               "new": float(v[TARGET].iloc[i])} for i in order]}


def export(version: int, build: bool, decision: str | None) -> None:
    import export_ctx_ownpoints as E
    from prc2026_submission.contract import validate_submission
    ov = read_json(ROOT / OVERRIDE)
    if ov["choice"]["candidate"] != CANDIDATE or ov["official_scores_used"].split()[0] != "none":
        raise ValueError("The override record does not name this candidate.")
    ident = W.OUT / "rank" / f"identity_v{version}" / "identity.json"
    if not ident.exists() or not read_json(ident)["byte_identical"]:
        raise ValueError("Run --identity first.")
    E.CHAIN, E.DECLARATION = W.OUT / "chain", W.DECLARATION
    ref = ROOT / W.V65_FILE[0]
    if sha256_file(ref) != W.V65_FILE[1]:
        raise ValueError("v65 changed.")
    dry = W.OUT / "rank" / f"dry_v{version}" / f"vigorous-whistle_v{version}.parquet"
    if not build:
        out = dry.parent
        if out.exists():
            raise ValueError(f"{out} exists.")
        result = E.chain(CANDIDATE, out, version, False)
        rep = E.compare(result, ref, out)
        rep.update({"file": E.rel(result), "sha256": sha256_file(result), "candidate": CANDIDATE, **details(result, ref)})
        write_json(out / f"vigorous-whistle_v{version}.report.json", rep)
        print({k: v for k, v in rep.items() if k not in ("by_airport", "largest_moves", "by_month")})
        return
    target = ROOT / f"runs/submissions/vigorous-whistle_v{version}.parquet"
    if target.exists() or not dry.exists() or not decision:
        raise ValueError("Never overwrite; run the dry run first; record the user's decision.")
    out = W.OUT / "rank" / f"build_v{version}"
    if out.exists():
        raise ValueError(f"{out} exists; never overwrite a build directory.")
    built = E.chain(CANDIDATE, out, version, False)
    if sha256_file(built) != sha256_file(dry):
        raise ValueError("The rebuilt file differs from the dry run.")
    target.write_bytes(built.read_bytes())
    sub = validate_submission(target, ROOT / "data/raw/submitting.parquet")
    decl = W.declaration()
    sub.update({"version": version, "provenance": {"experiment_id": decl["experiment_id"], "selected_candidate": CANDIDATE,
                                                   "selected_by": "the user's override of the declared rule (clause 1 failed in November)",
                                                   "override": {OVERRIDE: sha256_file(ROOT / OVERRIDE)}, "user_decision": decision,
                                                   "official_scores_used": "none", "audit_used": False,
                                                   "declaration": {W.DECLARATION: sha256_file(ROOT / W.DECLARATION)},
                                                   "code": {p: sha256_file(ROOT / p) for p in CODE}}})
    write_json(target.with_suffix(".receipt.json"), sub)
    rep = E.compare(target, ref, out)
    rep.update({"kind": "submission build at the user's override", "file": E.rel(target), "sha256": sha256_file(target), "dry_run_sha256": sha256_file(dry),
                "candidate": CANDIDATE, "created_at_utc": utc_now(), **details(target, ref)})
    write_json(W.DOCS / f"export_v{version}.json", rep)
    print("built", target.name, rep["sha256"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--identity", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--version", type=int, required=True)
    ap.add_argument("--user-decision", default=None)
    a = ap.parse_args()
    if a.identity:
        W.export_identity(a.version)
    if a.dry_run:
        export(a.version, False, None)
    if a.build:
        export(a.version, True, a.user_decision)


if __name__ == "__main__":
    main()
