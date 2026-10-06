"""v67 at the user's override of lever 5 (docs/ctx_own2/override_v67.json): v66's chain (docs/final_file/v66_chain.md) with only step 1
changed: the network member mix_own (0.5 own_all + 0.5 ref_own; ranking tables from both members' twelve-month refits) in place of own_all
at weight 1.00; steps 2-10 re-run through tools/export_ctx_ownpoints.chain, imported unchanged, with the stacker (step 6) refitted on the
candidate's rebuilt development tables runs/ctx_own2/chain/mix_own_w100/ (the stage-2 chain of configs/ctx_own2_dev_v1.json).

  --identity --version N          the unchanged path (own_all_w100, v66's chain directory and declaration) must give v66 byte for byte
  --dry-run --version N [--pinned]  -> runs/ctx_own2/rank/dry_vN/ (unpinned, as v66 was built) or dry_vN_pinned/ (the variance
                                  re-allocation's LightGBM pinned deterministic / column-wise through tools/lirf_own_pinned.py)
  --build --version N --user-decision TEXT   rerun (unpinned) into runs/ctx_own2/rank/build_vN, byte identity with the unpinned dry run,
                                  write runs/submissions/vigorous-whistle_vN.parquet (never overwrite) + docs/ctx_own2/export_vN.json
No official score enters. Nothing is uploaded.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET  # noqa: E402

OVERRIDE = "docs/ctx_own2/override_v67.json"
DECLARATION = "configs/ctx_own2_dev_v1.json"
CANDIDATE = "mix_own_w100"
OUT = ROOT / "runs/ctx_own2"
DOCS = ROOT / "docs/ctx_own2"
V66 = ("runs/submissions/vigorous-whistle_v66.parquet", "b4ea727e1644b6add104f44832898bfecfd3d5607f7d6a9ce0471efe581322e4")
GUARD = [213142471, 213142469]
CANDS = {"own_all_w100": {"member": "own_all", "member_weight": 1.0}, "mix_own_w100": {"member": "mix_own", "member_weight": 1.0}}
CODE = ["tools/export_ctx_own2_override.py", "tools/export_ctx_ownpoints.py", "tools/run_ctx_ownpoints_chain.py", "tools/export_ownpoints_weight_override.py",
        "tools/run_network_member.py", "tools/export_network_member.py", "tools/export_disagreement_gate.py", "tools/run_rematch_nonm.py",
        "tools/export_daily_totals.py", "tools/scout_extreme_shift.py", "tools/scout_var_alloc.py", "tools/run_stack_resid.py",
        "tools/run_stack_resid_v3.py", "tools/run_intruder_bound.py", "tools/run_intruder_identity.py", "tools/run_own_motion.py",
        "tools/run_own_motion_v2.py", "tools/lirf_own_pinned.py"]


def specs() -> dict:
    decl = read_json(ROOT / DECLARATION)
    return {"own_all": read_json(ROOT / "configs/ctx_ownpoints_dev_v1.json")["member"],
            "mix_own": {"name": "mix_own", "column": "adsb_full", "tables": decl["members_derived"]["mix_own"]["tables"]}}


def setup(pinned: bool):
    import export_ctx_ownpoints as E
    import run_ctx_ownpoints_chain as CH
    sp = specs()
    CH.member_spec = lambda _decl, name: sp[name]
    path = OUT / "export_candidates.json"
    write_json(path, {"note": "runtime candidate list for tools/export_ctx_ownpoints.chain (lever 5, v67)", "candidates": CANDS})
    E.DECLARATION, E.CHAIN = path.relative_to(ROOT).as_posix(), OUT / "chain"
    if pinned:
        def pinned_run(args: list[str]) -> None:
            print("> (pinned)", " ".join(args), flush=True)
            subprocess.run([sys.executable, "-X", "utf8", "-B", "tools/lirf_own_pinned.py", *args], cwd=ROOT, check=True)
        E.run = pinned_run
    return E


def guard_check(result: Path) -> dict:
    v66, v = pd.read_parquet(ROOT / V66[0]), pd.read_parquet(result)
    if not v[ID].equals(v66[ID]):
        raise ValueError("Row order differs from v66.")
    raw = pd.read_parquet(ROOT / "data/raw/ranking.parquet")
    dep = raw[raw["PHASE_mvt"].eq("DEP")].drop_duplicates(ID).set_index(ID).reindex(v[ID])
    pop = (dep["ADEP_mvt"].eq("LIRF") & dep["AOBT_3_flt"].isna() & dep["LOBT_flt"].isna() & dep["AIRCRAFT_TYPE_mvt"].isna()).to_numpy()
    guard = v[ID].isin(GUARD).to_numpy()
    changed = v[TARGET].to_numpy() != v66[TARGET].to_numpy()
    out = {"lirf_population_rows": int(pop.sum()), "lirf_population_rows_changed": int((changed & pop).sum()),
           "guard_rows_changed": int((changed & guard).sum()), "lirf_rows_changed": int((changed & dep["ADEP_mvt"].eq("LIRF").to_numpy()).sum())}
    if out["lirf_population_rows_changed"] or out["guard_rows_changed"]:
        raise ValueError(f"A LIRF population row or a guard row changed: {out}")
    return out


def identity(version: int) -> None:
    import export_ctx_ownpoints as E
    import run_ownpoints_weight as W
    out = OUT / "rank" / f"identity_v{version}"
    if out.exists():
        raise ValueError(f"{out} exists.")
    E.CHAIN, E.DECLARATION = W.OUT / "chain", W.DECLARATION  # exactly v66's export (tools/export_ownpoints_weight_override.export)
    result = E.chain("own_all_w100", out, version, False)
    same = sha256_file(result) == V66[1] == sha256_file(ROOT / V66[0])
    print("identity: v66's unchanged path reproduces v66:", same, sha256_file(result), flush=True)
    write_json(out / "identity.json", {"created_at_utc": utc_now(), "result": sha256_file(result), "v66": V66, "byte_identical": same,
                                       "chain": "runs/ctx_ownpoints_weight/chain, configs/ctx_ownpoints_weight_v1.json, own_all_w100, unpinned"})
    if not same:
        raise ValueError("The unchanged path does not reproduce v66.")


def dry_run(version: int, pinned: bool) -> None:
    from export_ownpoints_weight_override import details
    if not read_json(OUT / "rank" / f"identity_v{version}" / "identity.json")["byte_identical"]:
        raise ValueError("Run --identity first.")
    ov = read_json(ROOT / OVERRIDE)
    if ov["choice"]["candidate"] != CANDIDATE or not ov["official_scores_used"].startswith("none"):
        raise ValueError("The override record does not name this candidate.")
    out = OUT / "rank" / (f"dry_v{version}_pinned" if pinned else f"dry_v{version}")
    if out.exists():
        raise ValueError(f"{out} exists.")
    E = setup(pinned)
    result = E.chain(CANDIDATE, out, version, False)
    ref = ROOT / V66[0]
    rep = E.compare(result, ref, out)
    rep.update({"file": E.rel(result), "sha256": sha256_file(result), "candidate": CANDIDATE, "pinned_var_alloc": pinned,
                **guard_check(result), **details(result, ref)})
    write_json(out / f"vigorous-whistle_v{version}.report.json", rep)
    other = OUT / "rank" / (f"dry_v{version}" if pinned else f"dry_v{version}_pinned") / f"vigorous-whistle_v{version}.parquet"
    if other.exists():
        print("pinned and unpinned dry runs byte-identical:", sha256_file(other) == sha256_file(result), flush=True)
    print({k: v for k, v in rep.items() if k not in ("by_airport", "largest_moves", "by_month")})


def build(version: int, decision: str) -> None:
    from export_ownpoints_weight_override import details
    from prc2026_submission.contract import validate_submission
    target = ROOT / f"runs/submissions/vigorous-whistle_v{version}.parquet"
    dry = OUT / "rank" / f"dry_v{version}" / f"vigorous-whistle_v{version}.parquet"
    if target.exists() or not dry.exists() or not decision:
        raise ValueError("Never overwrite; run the dry run first; record the user's decision.")
    out = OUT / "rank" / f"build_v{version}"
    if out.exists():
        raise ValueError(f"{out} exists; never overwrite a build directory.")
    E = setup(False)
    built = E.chain(CANDIDATE, out, version, False)
    if sha256_file(built) != sha256_file(dry):
        raise ValueError("The rebuilt file differs from the dry run.")
    target.write_bytes(built.read_bytes())
    sub = validate_submission(target, ROOT / "data/raw/submitting.parquet")
    pinned = OUT / "rank" / f"dry_v{version}_pinned" / f"vigorous-whistle_v{version}.parquet"
    sub.update({"version": version, "provenance": {"experiment_id": read_json(ROOT / DECLARATION)["experiment_id"], "selected_candidate": CANDIDATE,
                                                   "selected_by": "the user's override of the declared rule (stage 2 clause 3 failed)",
                                                   "override": {OVERRIDE: sha256_file(ROOT / OVERRIDE)}, "user_decision": decision,
                                                   "official_scores_used": "none", "audit_used": False,
                                                   "declaration": {DECLARATION: sha256_file(ROOT / DECLARATION)},
                                                   "var_alloc": "unpinned (as v66); the pinned dry run is "
                                                                + ("byte-identical" if pinned.exists() and sha256_file(pinned) == sha256_file(dry) else "different or absent"),
                                                   "code": {p: sha256_file(ROOT / p) for p in CODE}}})
    write_json(target.with_suffix(".receipt.json"), sub)
    ref = ROOT / V66[0]
    rep = E.compare(target, ref, out)
    rep.update({"kind": "submission build at the user's override", "file": E.rel(target), "sha256": sha256_file(target), "dry_run_sha256": sha256_file(dry),
                "pinned_dry_run_sha256": sha256_file(pinned) if pinned.exists() else None, "candidate": CANDIDATE, "created_at_utc": utc_now(),
                **guard_check(target), **details(target, ref)})
    write_json(DOCS / f"export_v{version}.json", rep)
    print("built", target.name, rep["sha256"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--identity", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--pinned", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--version", type=int, required=True)
    ap.add_argument("--user-decision", default=None)
    a = ap.parse_args()
    if a.identity:
        identity(a.version)
    if a.dry_run:
        dry_run(a.version, a.pinned)
    if a.build:
        build(a.version, a.user_decision)


if __name__ == "__main__":
    main()
