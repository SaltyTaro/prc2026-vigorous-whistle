"""v66 step 1's member on any movement file (docs/final_file/lane_E.md): the own_all member of configs/ctx_ownpoints_dev_v1.json
(dl32own on ADS-B-matched rows, m25dl32own on the others; seeds 601-616; twelve-month refits, GPU-trained) predicted on the CPU from a
final-file runner root, the way the runner predicts the v51m-v55m members (src/prc2026_final/networks.predict_learner, ctx3 path).

  features --root R --keymap K [--surface DIR] [--own-dir D]   (D defaults to R/own; every command takes it)
      the 15 own-aircraft turnaround columns of the run's movement file: tools/build_ownpoints_features.py's load / identify / features,
      unchanged, on the file alone (one block, as that tool built the ranking block), with the key map K (the no-NM rematch step's
      relearned key_map.parquet; runs/rematch_nonm/key_map.parquet for the ranking file)
      -> D/own_points.parquet: MVT_ID_mvt + 15 columns for every departure (p_has_inb 0 and the rest NaN without an inbound)
  predict --root R [--threads 12]
      the 32 networks on the runner's learner frame (timeline + 27 ADS-B + 11 v19 columns) plus the 15 own columns
      -> D/learners/<variant>_s<seed>.parquet (MVT_ID_mvt, adsb_full) and D/members/s<seed>.parquet: the routed per-seed member
         (dl32own where the runner's adsb_matched = 1, m25dl32own elsewhere), the form of runs/ctx_ownpoints_dev/member/s<seed>/ranking.parquet
  export-v19own --root R --out DIR
      features_v19_<month>.parquet per month of the file: the runner's v19 table (R/adsb/v19.parquet, 11 columns) with the 15 own columns
      appended, the layout of data/external/adsblol/features_v19own (input of tools/final_late_chain.py --features-own-dir)
  compare --root R --output OUT.json   (the ranking file only) own columns, v19 columns, per-learner and per-seed values against the
      shipped tables (data/external/adsblol/features_v19own, refit12/ranking_predictions.parquet, runs/ctx_ownpoints_dev/member)
Resumable: a learner whose file exists is not predicted again. No label is read. Nothing is uploaded.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID  # noqa: E402

DECLARATION = "configs/ctx_ownpoints_dev_v1.json"
SHIPPED_OWN = ROOT / "data/external/adsblol/features_v19own"
SHIPPED_V19 = ROOT / "data/external/adsblol/features_v19"


def spec() -> dict:
    decl = read_json(ROOT / DECLARATION)
    fits, member = decl["fits"], decl["member"]
    base = fits["outputs"].split("/nets/")[0] + "/nets"
    learners = {v: [f"{base}/{v}_s{s}" for s in fits["seeds"]] for v in fits["variants"]}
    return {"seeds": list(fits["seeds"]), "route": member["route"], "column": member.get("column", "adsb_full"), "learners": learners}


def run_context(root: Path):
    from prc2026_final.common import Run
    from prc2026_final.runner import Context
    return Context(Run(root))


def features(root: Path, own_dir: Path, keymap: Path, surface: Path | None) -> Path:
    import build_ownpoints_features as B
    if surface is not None:
        B.SURF = surface
    ctx = run_context(root)
    km = pd.read_parquet(keymap)[["ADEP_mvt", "FLIGHT_mvt", "key"]].rename(columns={"ADEP_mvt": "airport"})
    started = time.time()
    mv = B.identify(B.load([ctx.raw_path], km))
    f = B.features(mv)[[ID, *B.OWN]]
    out = pd.DataFrame({ID: ctx.ids}).merge(f, on=ID, how="left", validate="one_to_one")
    out["p_has_inb"] = out["p_has_inb"].fillna(0.0)
    out[B.OWN] = out[B.OWN].astype("float64")
    own_dir.mkdir(parents=True, exist_ok=True)
    dest = own_dir / "own_points.parquet"
    out.to_parquet(dest, index=False, compression="zstd")
    dep = mv[mv["PHASE_mvt"] == "DEP"]
    write_json(dest.with_suffix(".receipt.json"), {
        "created_at_utc": utc_now(), "tool": "tools/final_own_member.py features", "tool_sha256": sha256_file(Path(__file__)),
        "builder": "tools/build_ownpoints_features.py", "builder_sha256": sha256_file(ROOT / "tools/build_ownpoints_features.py"),
        "raw": ctx.raw_path.as_posix(), "raw_sha256": sha256_file(ctx.raw_path), "keymap": keymap.as_posix(), "keymap_sha256": sha256_file(keymap),
        "surface_dir": Path(B.SURF).as_posix(), "rows": int(len(out)), "sha256": sha256_file(dest), "inbound_rows": int(out["p_has_inb"].sum()),
        "identified_share_by_airport": dep.groupby("airport")["icao"].apply(lambda s: round(float(s.notna().mean()), 4)).to_dict(),
        "seconds": round(time.time() - started, 1), "label_free": True})
    print("own points", len(out), "inbound", int(out["p_has_inb"].sum()), round(time.time() - started, 1), "s", flush=True)
    return dest


def predict(root: Path, own_dir: Path, threads: int) -> dict:
    from prc2026_ctx.tokens import TokenStore
    from prc2026_final import networks, tokens
    from prc2026_final.runner import learner_frame
    sp = spec()
    ctx = run_context(root)
    frame = learner_frame(ctx)
    own = pd.read_parquet(own_dir / "own_points.parquet")
    if not np.array_equal(own[ID].to_numpy(), ctx.ids):
        raise ValueError("own_points.parquet is not the file's departures in order.")
    clash = [c for c in own.columns if c != ID and c in frame.columns]
    if clash:
        raise ValueError(f"The learner frame already holds {clash}.")
    frame = frame.merge(own, on=ID, how="left", validate="one_to_one")
    host = TokenStore(root / "tokens", [tokens.NAME])
    cache, info, values = {}, {}, {}
    for variant, learners in sp["learners"].items():
        for learner in learners:
            target = own_dir / "learners" / (Path(learner).name + ".parquet")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                t = pd.read_parquet(target)
                if not np.array_equal(t[ID].to_numpy(), ctx.ids):
                    raise ValueError(f"{target} is not in the file's order.")
                values[learner] = t["adsb_full"].to_numpy(dtype=float)
                continue
            t0 = time.time()
            values[learner], info[learner] = networks.predict_learner(learner, frame, host, threads, cache)
            pd.DataFrame({ID: ctx.ids, "adsb_full": values[learner]}).to_parquet(target, index=False, compression="zstd")
            print(learner, round(time.time() - t0), "s", flush=True)
    matched = pd.read_parquet(root / "adsb" / "learner_adsb.parquet", columns=[ID, "adsb_matched"]).set_index(ID)["adsb_matched"].reindex(ctx.ids)
    if matched.isna().any():
        raise ValueError("The ADS-B table does not cover every departure.")
    flag = matched.to_numpy() == 1
    route, files = sp["route"], {}
    for k, seed in enumerate(sp["seeds"]):
        mv, uv = route["matched"], route["unmatched"]
        m = values[sp["learners"][mv][k]] if mv else np.full(len(flag), np.nan)
        u = values[sp["learners"][uv][k]] if uv else np.full(len(flag), np.nan)
        dest = own_dir / "members" / f"s{seed}.parquet"
        dest.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({ID: ctx.ids, sp["column"]: np.where(flag, m, u)}).to_parquet(dest, index=False, compression="zstd")
        files[dest.relative_to(own_dir).as_posix()] = sha256_file(dest)
    record = {"created_at_utc": utc_now(), "declaration": DECLARATION, "declaration_sha256": sha256_file(ROOT / DECLARATION), "route": route,
              "seeds": sp["seeds"], "matched_rows": int(flag.sum()), "rows": int(len(flag)), "member_tables": files, "device": "cpu", "threads": threads,
              "models_sha256": {lr: sha256_file(ROOT / lr / "refit12" / "model.pt") for lrs in sp["learners"].values() for lr in lrs},
              "own_points_sha256": sha256_file(own_dir / "own_points.parquet"), "learners_predicted_this_session": info,
              "runner_root": _rel(root), "tool_sha256": sha256_file(Path(__file__))}
    write_json(own_dir / "members.json", record)
    return record


def export_v19own(root: Path, own_dir: Path, out: Path) -> dict:
    import build_ownpoints_features as B
    ctx = run_context(root)
    v19 = pd.read_parquet(root / "adsb" / "v19.parquet")
    if not np.array_equal(v19[ID].to_numpy(), ctx.ids):
        # The runner joins v19 by ID; on the final file its rows come in another order (the ranking file's happened to match the template).
        # Same departures, each once: put them in the file's order (a no-op on the ranking file).
        if len(v19) != len(ctx.ids) or v19[ID].duplicated().any() or not np.array_equal(np.sort(v19[ID].to_numpy()), np.sort(ctx.ids)):
            raise ValueError("R/adsb/v19.parquet does not hold exactly the file's departures.")
        v19 = v19.set_index(ID).loc[ctx.ids].reset_index()
    own = pd.read_parquet(own_dir / "own_points.parquet")
    t = v19.merge(own, on=ID, how="left", validate="one_to_one")
    t["p_has_inb"] = t["p_has_inb"].fillna(0.0)
    t[B.OWN] = t[B.OWN].astype("float64")
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for month in ctx.months:
        target = out / f"features_v19_{month}.parquet"
        if target.exists():
            raise ValueError(f"{target} exists; never overwrite.")
        t[ctx.row_months == month].reset_index(drop=True).to_parquet(target, index=False, compression="zstd")
        files[target.name] = {"rows": int((ctx.row_months == month).sum()), "sha256": sha256_file(target)}
    write_json(out / "manifest.json", {"created_at_utc": utc_now(), "tool": "tools/final_own_member.py export-v19own", "tool_sha256": sha256_file(Path(__file__)),
                                       "v19_sha256": sha256_file(root / "adsb" / "v19.parquet"), "own_points_sha256": sha256_file(own_dir / "own_points.parquet"),
                                       "own_columns": B.OWN, "files": files})
    print(files, flush=True)
    return files


def member_spec(own_dir: Path) -> dict:
    """The declaration's member spec with the run's routed per-seed tables as its ranking tables (tools/run_network_member.member_values)."""
    sp = spec()
    rec = read_json(own_dir / "members.json")
    tables = [(own_dir / "members" / f"s{seed}.parquet") for seed in sp["seeds"]]
    for t in tables:
        rel = t.relative_to(own_dir).as_posix()
        if rec["member_tables"].get(rel) != sha256_file(t):
            raise ValueError(f"{t} differs from its members.json record.")
    return {**read_json(ROOT / DECLARATION)["member"], "tables": {"ranking": [_rel(t) for t in tables]}}


def _rel(p: Path) -> str:
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def compare(root: Path, own_dir: Path, output: Path) -> dict:
    """Ranking file only: the run's own columns / v19 columns / CPU predictions against the shipped tables."""
    import build_ownpoints_features as B
    sp = spec()
    ctx = run_context(root)
    ids = ctx.ids
    rep: dict = {"created_at_utc": utc_now(), "root": root.as_posix()}
    shipped = pd.concat([pd.read_parquet(SHIPPED_OWN / f"features_v19_{m}.parquet") for m in ctx.months], ignore_index=True).set_index(ID).reindex(ids)
    own = pd.read_parquet(own_dir / "own_points.parquet").set_index(ID).reindex(ids)
    rep["own_columns"] = {}
    for c in B.OWN:
        a, b = own[c].to_numpy(dtype=float), shipped[c].to_numpy(dtype=float)
        rep["own_columns"][c] = {"nan_pattern_equal": bool((np.isnan(a) == np.isnan(b)).all()),
                                 "max_abs": float(np.nanmax(np.abs(a - b))) if np.isfinite(a - b).any() else 0.0}
    v19 = pd.read_parquet(root / "adsb" / "v19.parquet").set_index(ID).reindex(ids)
    rep["v19_columns_max_abs"] = {c: float(np.nanmax(np.abs(v19[c].to_numpy(dtype=float) - shipped[c].to_numpy(dtype=float))))
                                  if np.isfinite(v19[c].to_numpy(dtype=float) - shipped[c].to_numpy(dtype=float)).any() else 0.0
                                  for c in v19.columns if c in shipped.columns}
    rep["learners"] = {}
    for lrs in sp["learners"].values():
        for lr in lrs:
            cpu = pd.read_parquet(own_dir / "learners" / (Path(lr).name + ".parquet")).set_index(ID)["adsb_full"].reindex(ids).to_numpy(dtype=float)
            gpu = pd.read_parquet(ROOT / lr / "refit12" / "ranking_predictions.parquet").set_index(ID)["adsb_full"].reindex(ids).to_numpy(dtype=float)
            d = cpu - gpu
            rep["learners"][Path(lr).name] = {"rms_s": float(np.sqrt(np.nanmean(d ** 2))), "max_abs_s": float(np.nanmax(np.abs(d))),
                                              "p99_abs_s": float(np.nanpercentile(np.abs(d), 99))}
    decl_member = read_json(ROOT / DECLARATION)["member"]
    seeds_cpu, seeds_gpu = [], []
    for seed, path in zip(sp["seeds"], decl_member["tables"]["ranking"]):
        cpu = pd.read_parquet(own_dir / "members" / f"s{seed}.parquet").set_index(ID)["adsb_full"].reindex(ids).to_numpy(dtype=float)
        gpu = pd.read_parquet(ROOT / path).set_index(ID)["adsb_full"].reindex(ids).to_numpy(dtype=float)
        seeds_cpu.append(cpu)
        seeds_gpu.append(gpu)
    mc, mg = np.mean(seeds_cpu, axis=0), np.mean(seeds_gpu, axis=0)
    d = mc - mg
    rep["seed_mean"] = {"nan_pattern_equal": bool((np.isnan(mc) == np.isnan(mg)).all()), "rms_s": float(np.sqrt(np.nanmean(d ** 2))),
                        "p99_abs_s": float(np.nanpercentile(np.abs(d), 99)), "max_abs_s": float(np.nanmax(np.abs(d))),
                        "rows_abs_gt_1s": int((np.abs(d) > 1).sum())}
    write_json(output, rep)
    print({k: v for k, v in rep.items() if k in ("seed_mean", "v19_columns_max_abs")}, flush=True)
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["features", "predict", "export-v19own", "compare"])
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--keymap", type=Path, default=ROOT / "runs/rematch_nonm/key_map.parquet")
    ap.add_argument("--surface", type=Path, default=None)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None, help="export-v19own: the directory for features_v19_<month>.parquet")
    ap.add_argument("--own-dir", type=Path, default=None, help="where the own columns / predictions / member tables go (default R/own)")
    a = ap.parse_args()
    root = a.root if a.root.is_absolute() else ROOT / a.root
    own_dir = (a.own_dir if a.own_dir.is_absolute() else ROOT / a.own_dir) if a.own_dir else root / "own"
    if a.command == "features":
        features(root, own_dir, a.keymap if a.keymap.is_absolute() else ROOT / a.keymap, a.surface)
    elif a.command == "predict":
        predict(root, own_dir, a.threads)
    elif a.command == "export-v19own":
        if not a.out:
            raise ValueError("export-v19own needs --out.")
        export_v19own(root, own_dir, a.out if a.out.is_absolute() else ROOT / a.out)
    else:
        compare(root, own_dir, a.output or own_dir / "compare.json")


if __name__ == "__main__":
    main()
