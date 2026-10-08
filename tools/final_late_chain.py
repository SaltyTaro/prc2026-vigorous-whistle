"""Final-file chain, steps 11-18 (v70 -> v76's layers) on ANY movement file (docs/final_file/lane_L.md).

Each step runs the layer's own governing-run code (tools/run_oooi_pin.py, run_nbr_state.py, run_nbr_dark.py, run_stack_second.py,
docs/audit_v72/raw_build.py, run_stack_third.py, run_season_pass.py, run_season_pass2.py, run_key_memory.py) with its module constants
pointed at a work directory: the movement file, the base file (the previous step's output), the runner's recipe detail, the ADS-B folders
and the stacker inputs of the new file. The development side of every layer is unchanged (stored development tables and development
predictions are copied after a hash check against the governing run's receipts, never rebuilt from other data). The declared rule of
each layer, including its per-airport transfer guard, is applied to the new file as it was to the ranking file. A step whose layer selects
nothing, or a W25 layer on a file without January 2026 departures (the declared rules act on 2026-01 rows only), passes its input through
unchanged.

  init      --work DIR --raw FILE --base FILE --recipe FILE [--months M,M] [--matches-dir DIR] [--features-own-dir DIR]
            [--messages FILE] [--stacker-rank FILE]                     writes DIR/late_chain.json
  stacker   --work DIR                     (ranking.parquet only) writes the stacker-inputs interface file from run_stack_weight.rank_inputs()
  step      --work DIR --name NAME         NAME in oooi, nbr_state, nbr_dark, ss_build, ss_score, raw, st_build, st_score, sp_build,
                                           sp_score, sp2_build, sp2_score, km   (workstation, light)
  fit       --work DIR --stage ss|st|sp|sp2 (Azure VM; ranking bags only, development predictions copied from the governing run)
  check     --work DIR                     compares every output with the shipped files (rehearsal on data/raw/ranking.parquet)

The work directory must lie under the repository (the tools write repository-relative paths into their receipts). Nothing is uploaded and
nothing is written under runs/submissions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
ID, TARGET = "MVT_ID_mvt", "TAXITIME_SEC_mvt"

# the governing runs' stored development artefacts (hash-checked before use) and the shipped outputs (rehearsal check)
STORED = {
    "ss": {"dir": "runs/stack_second", "dev_table": "dev_table", "preds_dev": "preds_dev.parquet"},
    "st": {"dir": "runs/stack_third", "preds_dev": "preds_dev.parquet"},
    "sp": {"dir": "runs/season_pass", "preds_dev": "preds_dev.parquet"},
    "sp2": {"dir": "runs/season_pass2", "preds_dev": "preds_dev.parquet"},
}
SHIPPED = {  # step -> SHA-256 of its output on data/raw/ranking.parquet from v69
    "oooi": "f1d17747e0e384ccec520a760d42c8e55a3e88a1476daafacdf7fe1f9603099f",       # v70
    "nbr_state": "7a0b1431eef1146d7afd40c728bea7af6b90a411e257ca277564866e410b84ed",  # nbr_state_v1 dry run (v70 + w75)
    "nbr_dark": "2dfb3b9f53284c7ae3a02882d19cfd61b19116b1b00504176551bcae8b82e6ba",   # v71
    "ss": "4280c399aa148847c4be4588dfc3d98d4adfd1a5e64981035b4e0fa3ee802f80",         # v72
    "st": "649e99725ee3f700041b07a231dcd2fa656fdbeef3aa8db5b5b9c9fc341af3ba",         # v73
    "sp": "85e57d05322009d886d7423ba818ec6435f023bb8fd054463dd8adca6983f761",         # v74
    "sp2": "a227c51ca90589bf6966f49ae94e2113394ee8f508bee0794d20871a800a6036",        # v75 dry run (never uploaded)
    "km": "a2ff395db4fdf22be86e1001f5dd97c8f27adc7b836fd7b4902690a576f9725c",         # v76
}
STEP_ORDER = ["oooi", "nbr_state", "nbr_dark", "ss", "st", "sp", "sp2", "km"]
SHIPPED_SELECTION = {"oooi": "w50", "nbr_state": "w75", "nbr_dark": "True", "ss": "sp_hicov7", "st": "t3_ro_3", "sp": "w25_e3",
                     "sp2": "w25b_5", "km": "km_9"}   # what each declared rule selected on the ranking file


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(p: Path) -> str:
    p = Path(p)
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def load(work: Path) -> dict:
    return json.loads((work / "late_chain.json").read_text(encoding="utf-8"))


def save(work: Path, cfg: dict) -> None:
    (work / "late_chain.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")


def path(cfg: dict, key: str) -> Path:
    return ROOT / cfg[key]


def output_of(cfg: dict, step: str) -> Path:
    return ROOT / cfg["outputs"][step]["file"]


def base_for(cfg: dict, step: str) -> Path:
    i = STEP_ORDER.index(step)
    return path(cfg, "base") if i == 0 else output_of(cfg, STEP_ORDER[i - 1])


def finish(cfg: dict, work: Path, step: str, produced: Path | None, note: str) -> None:
    """Record the step's output: the layer's dry run, or (nothing selected / not applicable) a byte copy of its input."""
    base = base_for(cfg, step)
    if produced is None or not Path(produced).exists():
        out = work / "outputs" / f"{step}_identity.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(base, out)
        produced, note = out, note or "identity"
    sel = note.rsplit(" ", 1)[-1] if (" selected " in note or " accepted " in note) else None
    flag = None if sel is None or sel == SHIPPED_SELECTION[step] else f"selection differs from the shipped one ({SHIPPED_SELECTION[step]})"
    cfg["outputs"][step] = {"file": rel(produced), "sha256": sha(produced), "base_sha256": sha(base), "note": note, "selection_flag": flag}
    if flag:
        print("WARNING", step, flag, flush=True)
    save(work, cfg)
    print(step, "->", cfg["outputs"][step], flush=True)


def has_january_2026(cfg: dict) -> bool:
    return "2026-01" in cfg["months"]


def stored_copy(stage: str, name: str, dest: Path, tables_key: str | None = None) -> Path:
    """Copy a governing-run artefact into the work directory; development tables are checked against the governing tables.json."""
    src_dir = ROOT / STORED[stage]["dir"]
    src = src_dir / name
    if tables_key is not None:
        tabs = json.loads((src_dir / "tables.json").read_text(encoding="utf-8"))
        if sha(src) != tabs["sha256"][tables_key]:
            raise ValueError(f"{src} differs from its governing tables.json")
    elif stage in ("ss", "st", "sp", "sp2") and name == "preds_dev.parquet":
        vm = (src_dir / "vm_preds.sha256").read_text(encoding="utf-8")
        if sha(src) not in vm:
            raise ValueError(f"{src} differs from its Azure receipt")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dest)
    return dest


# ---------------------------------------------------------------- module set-up

def setup_nbr(cfg: dict, base70: Path):
    import run_nbr_state as N
    N.V70 = (base70, sha(base70))
    N.RECIPE = path(cfg, "recipe")
    N.RANKING = path(cfg, "raw")
    return N


def step_oooi(cfg: dict, work: Path) -> None:
    import pandas as pd
    import oooi_pin_join as J
    import run_oooi_pin as O
    base = base_for(cfg, "oooi")
    run, fresh = work / "oooi", work / "oooi" / "fresh"
    fresh.mkdir(parents=True, exist_ok=True)
    msgs = pd.read_parquet(ROOT / "runs/oooi_pin/messages_unique.parquet")
    if cfg.get("messages"):
        new = pd.read_parquet(path(cfg, "messages"))
        if set(new["ts_utc"].dt.strftime("%Y-%m")) & set(msgs["ts_utc"].dt.strftime("%Y-%m")):
            raise ValueError("the new months' data-link messages overlap the stored ones")
        msgs = pd.concat([msgs, new[msgs.columns]], ignore_index=True)
        msgs.to_parquet(run / "messages_unique.parquet", index=False)
    else:
        shutil.copyfile(ROOT / "runs/oooi_pin/messages_unique.parquet", run / "messages_unique.parquet")
    shutil.copyfile(ROOT / "runs/oooi_pin/fresh/messages_unique.parquet", fresh / "messages_unique.parquet")
    O.RUN, O.FRESH = run, fresh
    O.V69, O.V69_SHA = base, sha(base)
    O.RANKING = J.RANKING = path(cfg, "raw")
    O.OUT_DIR, O.RESULTS = run / "rank" / "dry_v70", run / "results_v1.json"
    O.main()
    res = json.loads(O.RESULTS.read_text(encoding="utf-8"))
    finish(cfg, work, "oooi", (O.OUT_DIR / "vigorous-whistle_v70.parquet") if res.get("selected") else None,
           f"oooi_pin_v1 selected {res.get('selected')}")


def step_nbr_state(cfg: dict, work: Path) -> None:
    N = setup_nbr(cfg, base_for(cfg, "nbr_state"))
    N.OUT, N.RUNS = work / "nbr_state_docs", work / "nbr_state"
    N.DRY = N.RUNS / "rank/dry_v71/vigorous-whistle_v71.parquet"
    N.main()
    res = json.loads((N.OUT / "results_v1.json").read_text(encoding="utf-8"))
    finish(cfg, work, "nbr_state", N.DRY if res.get("selected") else None, f"nbr_state_v1 selected {res.get('selected')}")


def step_nbr_dark(cfg: dict, work: Path) -> None:
    N = setup_nbr(cfg, output_of(cfg, "oooi"))
    import run_nbr_dark as D
    base = base_for(cfg, "nbr_dark")
    D.V71 = (base, sha(base))
    D.RANK_MONTHS, D.MATCHES = cfg["months"], path(cfg, "matches_dir")
    D.DARK_RESULTS = work / "nbr_state_docs" / "dark_results_v1.json"
    D.DRY = work / "nbr_state/rank/dry_v72/vigorous-whistle_v72.parquet"
    D.main()
    res = json.loads(D.DARK_RESULTS.read_text(encoding="utf-8"))
    finish(cfg, work, "nbr_dark", D.DRY if res.get("accepted") else None, f"nbr_dark_v1 accepted {res.get('accepted')}")


def setup_ss(cfg: dict, work: Path):
    import run_stack_second as SS
    setup_nbr(cfg, output_of(cfg, "oooi"))
    v71 = output_of(cfg, "nbr_dark")
    SS.OUT, SS.DOCS = work / "stack_second", work / "stack_second_docs"
    SS.V70, SS.V71 = (output_of(cfg, "oooi"), sha(output_of(cfg, "oooi"))), (v71, sha(v71))
    SS.RANKING, SS.RANK_MONTHS, SS.MATCHES = path(cfg, "raw"), cfg["months"], path(cfg, "matches_dir")
    return SS


def step_ss_build(cfg: dict, work: Path) -> None:
    SS = setup_ss(cfg, work)
    SS.DEV_TABLE_FROM = stored_copy("ss", "dev_table.parquet", work / "stored" / "ss_dev_table.parquet", "dev_table")
    SS.STACKER_RANK = path(cfg, "stacker_rank") if cfg.get("stacker_rank") else None
    SS.build()
    stored_copy("ss", "preds_dev.parquet", SS.OUT / "preds_dev_stored.parquet")


def step_ss_score(cfg: dict, work: Path) -> None:
    SS = setup_ss(cfg, work)
    SS.score()
    res = json.loads((SS.DOCS / "results_v1.json").read_text(encoding="utf-8"))
    finish(cfg, work, "ss", (SS.OUT / "rank/dry_v72/vigorous-whistle_v72.parquet") if res.get("selected") else None,
           f"stack_second_v1 selected {res.get('selected')}")


def step_raw(cfg: dict, work: Path) -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location("raw_build", ROOT / "docs/audit_v72/raw_build.py")
    rb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rb)
    out = work / "raw"
    out.mkdir(parents=True, exist_ok=True)
    tabs = json.loads((ROOT / "runs/stack_third/tables.json").read_text(encoding="utf-8"))
    if sha(ROOT / "runs/audit_v72/raw_dev.parquet") != tabs["sha256"]["raw_dev"]:
        raise ValueError("stored raw_dev differs from stack_third's tables.json")
    shutil.copyfile(ROOT / "runs/audit_v72/raw_dev.parquet", out / "raw_dev.parquet")
    feat = (path(cfg, "features_own_dir").as_posix() + "/") if cfg.get("features_own_dir") else None
    rb.build(cfg["months"], (work / "stack_second/rank_table.parquet").as_posix(), (out / "raw_rank.parquet").as_posix(),
             feat, path(cfg, "matches_dir").as_posix() + "/")
    print("raw_rank", sha(out / "raw_rank.parquet"), flush=True)


def setup_st(cfg: dict, work: Path):
    import run_stack_third as ST
    v72 = output_of(cfg, "ss")
    ST.SS, ST.OUT, ST.DOCS, ST.RAW_DIR = work / "stack_second", work / "stack_third", work / "stack_third_docs", work / "raw"
    ST.V72 = (v72, sha(v72))
    return ST


def step_st_build(cfg: dict, work: Path) -> None:
    ST = setup_st(cfg, work)
    ST.build()
    stored_copy("st", "preds_dev.parquet", ST.OUT / "preds_dev_stored.parquet")


def step_st_score(cfg: dict, work: Path) -> None:
    ST = setup_st(cfg, work)
    ST.score()
    res = json.loads((ST.DOCS / "results_v1.json").read_text(encoding="utf-8"))
    finish(cfg, work, "st", (ST.OUT / "rank/dry_v73/vigorous-whistle_v73.parquet") if res.get("selected") else None,
           f"stack_third_v1 selected {res.get('selected')}")


def setup_sp(cfg: dict, work: Path):
    import run_season_pass as SP
    v73 = output_of(cfg, "st")
    SP.ST, SP.OUT, SP.DOCS = work / "stack_third", work / "season_pass", work / "season_pass_docs"
    SP.V73 = (v73, sha(v73))
    return SP


def step_sp_build(cfg: dict, work: Path) -> None:
    if not has_january_2026(cfg):
        print("season_pass_v1: no 2026-01 rows; nothing to build", flush=True)
        return
    SP = setup_sp(cfg, work)
    SP.build()
    stored_copy("sp", "preds_dev.parquet", SP.OUT / "preds_dev_stored.parquet")


def step_sp_score(cfg: dict, work: Path) -> None:
    if not has_january_2026(cfg):
        finish(cfg, work, "sp", None, "season_pass_v1 acts on 2026-01 rows only; the file has none")
        return
    SP = setup_sp(cfg, work)
    SP.score()
    res = json.loads((SP.DOCS / "results_v1.json").read_text(encoding="utf-8"))
    finish(cfg, work, "sp", (SP.OUT / "rank/dry_v74/vigorous-whistle_v74.parquet") if res.get("selected") else None,
           f"season_pass_v1 selected {res.get('selected')}")


def setup_sp2(cfg: dict, work: Path):
    import run_season_pass2 as SP2
    v74 = output_of(cfg, "sp")
    SP2.SP, SP2.SS, SP2.OUT, SP2.DOCS = work / "season_pass", work / "stack_second", work / "season_pass2", work / "season_pass2_docs"
    SP2.V74 = (v74, sha(v74))
    return SP2


def step_sp2_build(cfg: dict, work: Path) -> None:
    if not has_january_2026(cfg):
        print("season_pass_v2: no 2026-01 rows; nothing to build", flush=True)
        return
    SP2 = setup_sp2(cfg, work)
    SP2.build()
    stored_copy("sp2", "preds_dev.parquet", SP2.OUT / "preds_dev_stored.parquet")


def step_sp2_score(cfg: dict, work: Path) -> None:
    if not has_january_2026(cfg):
        finish(cfg, work, "sp2", None, "season_pass_v2 acts on 2026-01 rows only; the file has none")
        return
    SP2 = setup_sp2(cfg, work)
    SP2.score()
    res = json.loads((SP2.DOCS / "results_v1.json").read_text(encoding="utf-8"))
    finish(cfg, work, "sp2", (SP2.OUT / "rank/dry_v75/vigorous-whistle_v75.parquet") if res.get("selected") else None,
           f"season_pass_v2 selected {res.get('selected')}")


def step_km(cfg: dict, work: Path) -> None:
    if not has_january_2026(cfg):
        finish(cfg, work, "km", None, "key_memory_v1 acts on 2026-01 rows only; the file has none")
        return
    import run_key_memory as KM
    v74 = output_of(cfg, "sp")
    KM.SP2, KM.OUT, KM.DOCS = work / "season_pass2", work / "key_memory", work / "key_memory_docs"
    KM.V74 = (v74, sha(v74))
    KM.SP2_RES = work / "season_pass2_docs" / "results_v1.json"
    KM.RANKING = path(cfg, "raw")
    out = KM.run()
    dry = out.get("dry_run", {}).get("file")
    finish(cfg, work, "km", (ROOT / dry) if out.get("selected") and dry else None, f"key_memory_v1 selected {out.get('selected')}")


STEPS = {"oooi": step_oooi, "nbr_state": step_nbr_state, "nbr_dark": step_nbr_dark, "ss_build": step_ss_build, "ss_score": step_ss_score,
         "raw": step_raw, "st_build": step_st_build, "st_score": step_st_score, "sp_build": step_sp_build, "sp_score": step_sp_score,
         "sp2_build": step_sp2_build, "sp2_score": step_sp2_score, "km": step_km}


# ---------------------------------------------------------------- fit (Azure)

def fit(work: Path, stage: str) -> None:
    """Ranking bags of one pass on its work table directory; the development predictions are the governing run's (copied at build)."""
    import importlib
    mod, sub = {"ss": ("run_stack_second", "stack_second"), "st": ("run_stack_third", "stack_third"), "sp": ("run_season_pass", "season_pass"),
                "sp2": ("run_season_pass2", "season_pass2")}[stage]
    if stage in ("sp", "sp2") and not has_january_2026(load(work)):
        print(stage, "acts on 2026-01 rows only; the file has none: no fit", flush=True)
        return
    M = importlib.import_module(mod)
    d = work / sub
    M.DEV_PREDS_FROM = d / "preds_dev_stored.parquet"
    M.fit(d)
    print(stage, "preds_rank", sha(d / "preds_rank.parquet"), "preds_dev", sha(d / "preds_dev.parquet"), flush=True)


# ---------------------------------------------------------------- rehearsal check

def check(work: Path) -> dict:
    cfg = load(work)
    out = {"outputs": {}, "tables": {}}
    for step in STEP_ORDER:
        o = cfg["outputs"].get(step)
        out["outputs"][step] = {"sha256": o and o["sha256"], "shipped": SHIPPED[step], "equal": bool(o and o["sha256"] == SHIPPED[step])}
    pairs = [("stack_second/rank_table.parquet", "runs/stack_second/tables.json", "rank_table"),
             ("stack_third/rank_t3.parquet", "runs/stack_third/tables.json", "rank_t3"),
             ("season_pass/rank_sp.parquet", "runs/season_pass/tables.json", "rank_sp"),
             ("season_pass2/rank_sp2.parquet", "runs/season_pass2/tables.json", "rank_sp2"),
             ("raw/raw_rank.parquet", "runs/stack_third/tables.json", "raw_rank")]
    for f, receipt, key in pairs:
        p = work / f
        want = json.loads((ROOT / receipt).read_text(encoding="utf-8"))["sha256"][key]
        out["tables"][f] = {"sha256": sha(p) if p.exists() else None, "shipped": want, "equal": p.exists() and sha(p) == want}
    for sub, stage in [("stack_second", "runs/stack_second"), ("stack_third", "runs/stack_third"), ("season_pass", "runs/season_pass"),
                       ("season_pass2", "runs/season_pass2")]:
        p = work / sub / "preds_rank.parquet"
        want = sha(ROOT / stage / "preds_rank.parquet")
        out["tables"][f"{sub}/preds_rank.parquet"] = {"sha256": sha(p) if p.exists() else None, "shipped": want, "equal": p.exists() and sha(p) == want}
    out["all_equal"] = all(v["equal"] for v in out["outputs"].values()) and all(v["equal"] for v in out["tables"].values())
    (work / "check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["init", "stacker", "step", "fit", "check"])
    ap.add_argument("--work", required=True)
    ap.add_argument("--raw")
    ap.add_argument("--base")
    ap.add_argument("--recipe")
    ap.add_argument("--months", help="comma-separated months of the movement file (default: read from it)")
    ap.add_argument("--matches-dir", default="data/external/adsblol/matches")
    ap.add_argument("--features-own-dir", default=None, help="folder of features_v19_<month>.parquet (default the stored one)")
    ap.add_argument("--messages", default=None, help="messages_unique.parquet of the new months (tools/oooi_pin_extract.py + oooi_pin_pack.py)")
    ap.add_argument("--stacker-rank", default=None, help="MVT_ID_mvt + the 79 s_* stacker columns of the file's rows (Lane E)")
    ap.add_argument("--name")
    ap.add_argument("--stage")
    a = ap.parse_args()
    work = (ROOT / a.work) if not Path(a.work).is_absolute() else Path(a.work)
    if a.command == "init":
        import pandas as pd
        work.mkdir(parents=True, exist_ok=True)
        raw = ROOT / a.raw
        months = a.months.split(",") if a.months else sorted(
            pd.read_parquet(raw, columns=["PHASE_mvt", "MVT_TIME_UTC_mvt"]).query("PHASE_mvt == 'DEP'")["MVT_TIME_UTC_mvt"].dt.strftime("%Y-%m").unique())
        cfg = {"raw": rel(raw), "raw_sha256": sha(raw), "base": rel(ROOT / a.base), "base_sha256": sha(ROOT / a.base),
               "recipe": rel(ROOT / a.recipe), "recipe_sha256": sha(ROOT / a.recipe), "months": months,
               "matches_dir": rel(ROOT / a.matches_dir), "features_own_dir": rel(ROOT / a.features_own_dir) if a.features_own_dir else None,
               "messages": rel(ROOT / a.messages) if a.messages else None, "stacker_rank": rel(ROOT / a.stacker_rank) if a.stacker_rank else None,
               "outputs": {}}
        save(work, cfg)
        print(json.dumps(cfg, indent=1))
    elif a.command == "stacker":
        import pandas as pd
        import run_stack_weight as SW
        cfg = load(work)
        if cfg["raw"] != "data/raw/ranking.parquet":
            raise ValueError("run_stack_weight.rank_inputs() reads the shipped ranking chain; a new file's stacker inputs come from the early chain")
        s_cols = [c for c in pd.read_parquet(ROOT / "runs/stack_second/dev_table.parquet", columns=None).columns if c.startswith("s_")]
        _, _, _, mats, rk, *_ = SW.rank_inputs()
        X = [m for m in mats if m[0] == "po"][0][2].reset_index(drop=True)
        X.columns = s_cols
        X.insert(0, ID, rk[ID].to_numpy())
        f = work / "stacker_rank.parquet"
        X.to_parquet(f, index=False)
        cfg["stacker_rank"], cfg["stacker_rank_sha256"] = rel(f), sha(f)
        save(work, cfg)
        print("stacker inputs", X.shape, sha(f))
    elif a.command == "step":
        STEPS[a.name](load(work), work)
    elif a.command == "fit":
        fit(work, a.stage)
    elif a.command == "check":
        check(work)


if __name__ == "__main__":
    main()
