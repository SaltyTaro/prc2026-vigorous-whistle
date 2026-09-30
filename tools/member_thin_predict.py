"""December 2025 predictions of the refit models of v52's network table, natural and thinned (configs/member_weights_thin_v1.json).

For every learner of the declaration (base networks e24 / e24b / e32, members dl32 / m25dl32 / m25_32), the refit fold's saved model
predicts every December departure on the workstation CPU twice:
  natural  the December build + the tables the refit runs read (hash-checked against each run's record) + tokens data/ctx/tokens_v1
  thinned  the same rows with coverage-regime attempt a13's thinned committee / v19 tables and tokens data/ctx/tokens_thin_a13/2025-12
The natural prediction is compared with the run's own december_predictions.parquet (the declared check). No label is read.
Output runs/member_thin/<family>/<learner name>/{natural,thinned}.parquet and runs/member_thin/predictions.json.
Usage: tools/member_thin_predict.py [--families e24 e24b ...] [--threads 12]
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
from prc2026.validation import identity_hash  # noqa: E402
from prc2026_timeline.bundle import load_bundle  # noqa: E402
from prc2026_final.networks import kind_of  # noqa: E402
from run_adsb_learner import BUNDLE_NAMES, source_file  # noqa: E402
from ctx3_pos_predict import DECEMBER_BUILD, NATURAL_CACHE, NATURAL_V19, REGIME, table  # noqa: E402

DECLARATION = "configs/member_weights_thin_v1.json"
MONTH = "2025-12"
TOKENS = {"natural": "data/ctx/tokens_v1", "thinned": "data/ctx/tokens_thin_a13/2025-12"}
TOLERANCE = {"ctx": 0.01, "ctx3": 0.25}
BATCH = 50000


def families(decl: dict) -> dict[str, list[str]]:
    m = decl["models"]
    return {**{k: m["nets"][k] for k in ("e24", "e24b", "e32")}, **{k: m["members"][k] for k in ("dl32", "m25dl32", "m25_32")}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("runs/member_thin"))
    args = parser.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    from prc2026_ctx.tokens import TokenStore
    device = torch.device("cpu")
    decl = read_json(ROOT / DECLARATION)
    fams = families(decl)
    chosen = args.families or list(fams)

    base = ROOT / read_json(ROOT / "data/timeline/latest.json")["bundle"]
    manifest, _, _, _ = load_bundle(ROOT, base)
    meta = read_json(ROOT / DECEMBER_BUILD / "manifest.json")
    if meta["timeline_bundle_id"] != manifest["bundle_id"]:
        raise ValueError("December build does not belong to this timeline bundle.")
    rows = pd.read_parquet(ROOT / DECEMBER_BUILD / "features" / source_file(MONTH), columns=[ID, *BUNDLE_NAMES])
    if identity_hash(rows[ID]) != meta["ids_sha256"]:
        raise ValueError("December identities changed.")
    frames, hashes = {}, {}
    for regime, (cdir, vdir) in {"natural": (ROOT / NATURAL_CACHE, ROOT / NATURAL_V19),
                                 "thinned": (ROOT / REGIME[MONTH] / "thinned" / "committee", ROOT / REGIME[MONTH] / "thinned" / "v19")}.items():
        t, h = table(MONTH, cdir, vdir)
        frame = rows.merge(t, on=ID, how="left", validate="one_to_one")
        if frame["adsb_matched"].isna().any():
            raise ValueError(f"{regime}: ADS-B table does not cover every December departure.")
        frames[regime], hashes[regime] = frame, h
    del rows
    hosts = {r: TokenStore(ROOT / TOKENS[r], [MONTH]) for r in TOKENS}
    stores = {}

    out = ROOT / args.output
    receipt_path = out / "predictions.json"
    receipt = read_json(receipt_path) if receipt_path.exists() else {"learners": {}}
    receipt.update({"updated_at_utc": utc_now(), "declaration": DECLARATION, "declaration_sha256": sha256_file(ROOT / DECLARATION),
                    "inputs": hashes, "december_build_id": meta["build_id"],
                    "tokens": {"natural": sha256_file(ROOT / TOKENS["natural"] / "manifest.json"),
                               "thinned": sha256_file(ROOT / TOKENS["thinned"] / f"manifest_{MONTH}.json")},
                    "code": {p: sha256_file(ROOT / p) for p in ["tools/member_thin_predict.py", "tools/ctx3_pos_predict.py", "src/prc2026_final/networks.py"]}})
    for family in chosen:
        for learner in fams[family]:
            name = Path(learner).name
            info = receipt["learners"].get(name, {})
            if info.get("files") and len(info["files"]) == 2 and "check" in info:
                continue
            started = time.time()
            run = ROOT / learner / "refit"
            record = read_json(run / "run.json")
            if sha256_file(run / "model.pt") != record["model_sha256"]:
                raise ValueError(f"{learner}: model differs from its record.")
            recorded = record["inputs"]
            if recorded.get("december_build_id") != meta["build_id"] or \
                    recorded["adsb_cache_sha256"][source_file(MONTH)] != hashes["natural"][(ROOT / NATURAL_CACHE / f"adsb_{source_file(MONTH).replace('.parquet', '')}.parquet").as_posix()]:
                raise ValueError(f"{learner}: its December inputs are not the natural tables used here.")
            kind = kind_of(record)
            if kind == "ctx":
                from prc2026_ctx.train import Context, load, predict
                model, pre, settings = load(run)
            elif kind == "ctx3":
                from prc2026_ctx3.train import Context, Store, load, predict
                model, pre, settings = load(run, device)
            else:
                raise ValueError(f"{learner}: unexpected kind {kind}.")
            target = out / family / name
            target.mkdir(parents=True, exist_ok=True)
            info = {"family": family, "kind": kind, "model_sha256": record["model_sha256"], "files": {}}
            values_by = {}
            for regime, frame in frames.items():
                host = hosts[regime]
                values = np.empty(len(frame))
                for start in range(0, len(frame), BATCH):
                    chunk = frame.iloc[start:start + BATCH]
                    xn, xc = pre.transform(chunk)
                    self_rows, nb = host.lookup(chunk[ID].to_numpy())
                    if kind == "ctx":
                        values[start:start + BATCH] = predict(model, pre, xn, xc, Context.build(host, self_rows, nb, settings))
                    else:
                        store = stores.setdefault(regime, Store.build(host, device))
                        values[start:start + BATCH] = predict(model, pre, torch.from_numpy(xn), torch.from_numpy(xc), Context.build(store, self_rows, nb, settings))
                if not np.isfinite(values).all():
                    raise ValueError(f"{learner} {regime}: non-finite predictions.")
                path = target / f"{regime}.parquet"
                pd.DataFrame({ID: frame[ID].to_numpy(), "adsb_full": values}).to_parquet(path, index=False, compression="zstd")
                info["files"][regime] = sha256_file(path)
                values_by[regime] = values
            saved = pd.read_parquet(run / "december_predictions.parquet").set_index(ID)["adsb_full"].reindex(frames["natural"][ID]).to_numpy(dtype=float)
            d = values_by["natural"] - saved
            rms = float(np.sqrt(np.mean(d ** 2)))
            info["check"] = {"rms_s": rms, "max_abs_s": float(np.max(np.abs(d))), "tolerance_rms_s": TOLERANCE[kind], "passes": bool(rms <= TOLERANCE[kind])}
            info["seconds"] = round(time.time() - started, 1)
            receipt["learners"][name] = info
            write_json(receipt_path, receipt)
            print(family, name, kind, info["check"], info["seconds"], "s", flush=True)


if __name__ == "__main__":
    main()
