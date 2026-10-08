"""The early final-file chain on any movement file (docs/final_file/lane_E.md): v66's steps 1-10 with v69's step 6, i.e. the file v69 is
on the ranking file. Every step calls the shipped functions; this tool only points their ranking-bound paths at the run's own inputs.

  python tools/final_early_chain.py --raw F --template T --runner R --model-part M --out O --version N
         [--step1-source runner|shipped] [--members runner|shipped] [--published-table P --workbook W --workbook-sha256 S | --no-published]
         [--stack-preds FILE] [--surface DIR]

Inputs
  F, T          the movement file and its submission template (data/raw/ranking.parquet / submitting.parquet for the ranking file)
  R             a completed final-file runner root for F (tools/final_predict.py run --layers v55m; members stage done) and, for
                --members runner, tools/final_own_member.py features + predict on R
  M             the v55m model-part file the chain starts from: R's final v55m file for a new file; the shipped
                runs/lirf_combo/export_dry/vigorous-whistle_v55.parquet for the ranking rehearsal
  --step1-source  where step 1 takes v34 / the recipe detail / the run tables from: runner = R (recipe/detail.parquet, stack/detail.parquet,
                rematch/out.parquet, adsb/residual matches, stack/peers.parquet); shipped = tools/export_disagreement_gate.rebuild_v34
                (the uploaded v31 / v33 / v34; ranking file only)
  --members     runner = the combination / masked members of R/members/members.parquet and the own_all member of R/own/members
                (CPU); shipped = the declarations' shipped ranking tables (GPU; ranking file only)
  P / W         a published-totals extract covering F's period (and its workbook); --no-published skips steps 3-6 as
                docs/final_file/v61_chain.md prescribes for a period EUROCONTROL has not published
  --stack-preds a per-seed prediction file of step 6 (the form of runs/stack_weight/rank/preds.parquet) to use instead of fitting; its
                rows and guard columns must equal the recomputed ones (resume / rehearsal)

Steps (work directory O, never under runs/submissions)
  0   key map + no-NM matches: tools/run_rematch_nonm.py --matches --out O/steps/rematch (key map relearned from 2025 + F)
  1   model part M' (tools/export_ctx_ownpoints.step_model_part's arithmetic; member own_all at 1.00): O/steps/model_part/
      (steps 1-5 keep the shipped v66 layout and file names, vigorous-whistle_v66.parquet, which step 6's shipped reader expects)
  2   no-NM rematch base B: tools/run_rematch_nonm.py --ranking-base --base M'
  3-5 v48 reconciliation, extreme-day layer, variance re-allocation (tools/export_daily_totals.py, scout_extreme_shift.py, scout_var_alloc.py)
  6   v69's stacker (tools/run_stack_weight.rank_inputs with run_opdi_stack_decl.SH = O/steps; the three learners, ten seeds, weight 0.65)
  7-9 tools/run_stack_resid_v3.chain_from_stack: intruder bound + identity extension, own-motion bound v1
  10  tools/run_own_motion_v2.dry_run -> O/vigorous-whistle_vN.parquet
For a new file the identity tables (runs/intruder_identity/ids.parquet's 2025 rows + F; runs/dig/position ids_2025-* + ids_ranking of F),
the own-point cache (runs/stack_resid/v3_features: dev copied, rank rebuilt), the OPDI features (runs/opdi_stack: dev copied, rank rebuilt
from data/external/opdi/stack_v1) and the match directory (R/adsb/learners + data/external/adsblol/matches/stands.parquet) are built
under O/cache; nothing shipped is overwritten. No label is read beyond the 2025 development tables the shipped steps fit on. No official
score enters. Nothing is uploaded.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from prc2026.artifacts import read_json, sha256_file, utc_now, write_json  # noqa: E402
from prc2026.features import ID, TARGET  # noqa: E402
from prc2026_submission.contract import export_submission  # noqa: E402

PY = [sys.executable, "-X", "utf8", "-B"]
SHIPPED_RANKING, SHIPPED_TEMPLATE = "data/raw/ranking.parquet", "data/raw/submitting.parquet"
SHIPPED_M55 = "runs/lirf_combo/export_dry/vigorous-whistle_v55.parquet"
COMBO, MASK = "configs/adsb_ctx3_combo_dev_v1.json", "configs/adsb_ctx3_mask_dev_v1.json"
OWN_DECLARATION, OWN_WEIGHT = "configs/ctx_ownpoints_weight_v1.json", "own_all_w100"
STACK_RESULTS = "docs/stack_weight/results_v1.json"
EXPORT_MODULES = ["run_stack_resid", "run_stack_resid_v3", "run_opdi_stack_decl", "run_stack_weight", "run_intruder_bound", "run_intruder_identity",
                  "run_own_motion", "run_own_motion_v2", "export_ctx_ownpoints"]


def rel(p: Path | str) -> str:
    p = Path(p)
    p = p if p.is_absolute() else ROOT / p
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def run(args: list[str]) -> None:
    print(">", " ".join(args), flush=True)
    subprocess.run(PY + args, cwd=ROOT, check=True)


class Chain:
    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.raw, self.template = rel(a.raw), rel(a.template)
        self.new_file = self.raw != SHIPPED_RANKING
        self.runner = ROOT / rel(a.runner)
        self.model_part = rel(a.model_part)
        self.out = ROOT / rel(a.out)
        self.steps = self.out / "steps"
        self.cache = self.out / "cache"
        self.own_dir = ROOT / rel(a.own_dir) if a.own_dir else self.runner / "own"
        self.version = a.version
        # steps 1-5 keep the shipped v66 layout and names (tools/run_stack_weight.rank_inputs reads <steps>/rematch/base/..., <steps>/Y/
        # vigorous-whistle_v66.parquet, <steps>/Y/rows.parquet, <steps>/model_part/member_delta.parquet); steps 6-10 are named by --version
        self.v15 = 66
        self.rematch = self.steps / "rematch"
        self.record: dict = {"created_at_utc": utc_now(), "raw": self.raw, "raw_sha256": sha256_file(ROOT / self.raw), "template": self.template,
                             "template_sha256": sha256_file(ROOT / self.template), "runner": rel(self.runner), "model_part": self.model_part,
                             "model_part_sha256": sha256_file(ROOT / self.model_part), "step1_source": a.step1_source, "members": a.members,
                             "published": None if a.no_published else (rel(a.published_table) if a.published_table else "shipped"),
                             "steps": {}, "seconds": {}}
        if (ROOT / "runs/submissions").resolve() in [self.out.resolve(), *self.out.resolve().parents]:
            raise ValueError("The chain never writes under runs/submissions.")
        if self.new_file and (a.step1_source == "shipped" or a.members != "runner"):
            raise ValueError("Shipped step-1 inputs and shipped member tables exist for the ranking file only.")
        repair = read_json(self.runner / "stages" / "repair.json") if (self.runner / "stages" / "repair.json").exists() else None
        self.repaired = None
        if repair is not None and repair.get("repair_root"):
            self.repaired = Path(repair["repair_root"])
            if not self.repaired.is_absolute():
                self.repaired = ROOT / self.repaired
        elif not self.new_file:
            self.repaired = ROOT / "runs/final_lfpg_eobt/run"   # the shipped chain's repaired ranking run (tools/run_stack_resid.REPAIRED)
        self.record["repaired"] = rel(self.repaired) if self.repaired is not None else None

    # ------------------------------------------------------------ overrides

    def template_wrapper(self) -> None:
        """Every exporting module's export_submission takes the file's template (as tools/final_chain_overrides.py --template does)."""
        if self.template == SHIPPED_TEMPLATE:
            return
        template = ROOT / self.template
        for name in EXPORT_MODULES + ["__main__", __name__]:
            mod = sys.modules.get(name)
            fn = getattr(mod, "export_submission", None) if mod is not None else None
            if callable(fn) and not getattr(fn, "_final_template", False):
                def wrapped(predictions, template_path, output, provenance, _fn=fn):
                    return _fn(predictions, template, output, provenance)
                wrapped._final_template = True
                mod.export_submission = wrapped

    def configure(self) -> dict:
        """Point the in-process steps (6-10) at the run's inputs. With the ranking file and the shipped runner every value equals the shipped
        default, except the caches, which are rebuilt under O/cache when --rebuild-caches is given (rehearsal of the new-file path)."""
        import export_ctx_ownpoints  # noqa: F401  (imported before the template wrapper below)
        import run_intruder_bound as R
        import run_intruder_identity as I
        import run_opdi_stack as O
        import run_opdi_stack_decl as D
        import run_own_motion as OM
        import run_own_motion_v2 as O2
        import run_stack_resid as S
        import run_stack_resid_v3 as V3
        import run_stack_weight  # noqa: F401
        import screen_daily_totals
        done = {}
        D.SH = self.steps
        S.RUNNER = self.runner
        S.REPAIRED = self.repaired
        S.MODEL_PART = (self.model_part, None)
        for mod in (S, R, I, OM, O2):
            mod.RANKING = self.raw
        O.RANKING = self.raw
        if self.a.published_table:
            screen_daily_totals.TABLE = ROOT / rel(self.a.published_table)
        keymap = self.rematch / "key_map.parquet"
        rebuild = self.new_file or self.a.rebuild_caches
        if rebuild:
            V3.KEYMAP = O.KEYMAP = O2.KEYMAP = I.KEYMAP = keymap
            # own-point cache: the shipped development table, the movement file's rows rebuilt
            V3.FEAT = self.cache / "v3_features"
            V3.FEAT.mkdir(parents=True, exist_ok=True)
            for f in ["dev.parquet", "dev.receipt.json"]:
                shutil.copyfile(ROOT / "runs/stack_resid/v3_features" / f, V3.FEAT / f)
            # OPDI features: the shipped development table, the movement file's rows rebuilt from the saved event extracts
            O.OUT = self.cache / "opdi"
            O.OUT.mkdir(parents=True, exist_ok=True)
            for f in ["features_dev.parquet", "features_dev.receipt.json"]:
                shutil.copyfile(ROOT / "runs/opdi_stack" / f, O.OUT / f)
            if not (O.OUT / "features_rank.parquet").exists():
                O.build("rank")
            # identity tables: the shipped table's 2025 rows + the movement file (run_intruder_identity.build_ids, new-file mode)
            I.IDS = OM.IDS = self.cache / "intruder_identity" / "ids.parquet"
            if not I.IDS.exists():
                I.build_ids()
            # tunnel-11 identity tables: the shipped 2025 months + the movement file as tag "ranking" (run_own_motion_v2.build_ids)
            O2.IDS_DIR = self.cache / "position"
            O2.IDS_DIR.mkdir(parents=True, exist_ok=True)
            for p in sorted((ROOT / "runs/dig/position").glob("ids_2025-*.parquet")):
                shutil.copyfile(p, O2.IDS_DIR / p.name)
            if not (O2.IDS_DIR / "ids_ranking.parquet").exists():
                O2.build_ids("ranking", ROOT / self.raw, O2.IDS_DIR)
            # match directory: the shipped 2025 months (own-motion v1 / v2 re-read their 2025 development proxies from it), the runner's
            # learner matches for the file's own months, and the frozen stands table
            mdir = self.cache / "matches"
            mdir.mkdir(parents=True, exist_ok=True)
            for p in sorted((ROOT / "data/external/adsblol/matches").glob("matches2_2025-*.parquet")):
                shutil.copyfile(p, mdir / p.name)
            for p in sorted((self.runner / "adsb" / "learners").glob("matches2_*.parquet")):
                if p.name.startswith("matches2_2025-"):
                    raise ValueError(f"The runner's months overlap the 2025 development months ({p.name}).")
                shutil.copyfile(p, mdir / p.name)
            shutil.copyfile(ROOT / "data/external/adsblol/matches/stands.parquet", mdir / "stands.parquet")
            OM.MATCHES = O2.MATCHES = mdir
            done.update({"keymap": rel(keymap), "own_point_cache": rel(V3.FEAT), "opdi_features": rel(O.OUT), "ids": rel(I.IDS),
                         "position_ids": rel(O2.IDS_DIR), "matches": rel(mdir)})
            done["cache_sha256"] = {rel(p): sha256_file(p) for p in sorted(self.cache.rglob("*.parquet"))}
        I.V60 = None
        self.template_wrapper()
        return done

    # ------------------------------------------------------------ step 1

    def runner_tables(self):
        """src/prc2026_final/stack.compute's three export_stack tables from the runner root (callsign scope, queue segments, peers)."""
        from prc2026_final import stack as FS
        months = read_json(self.runner / "input.json")["months"]
        rematch_path = self.runner / "rematch" / "out.parquet"
        rematch_out = pd.read_parquet(rematch_path)
        callsign = FS.scope_table({"rematch": {"outputs": {rematch_path.as_posix(): sha256_file(rematch_path)}}})
        regular = pd.concat([pd.read_parquet(self.runner / "adsb" / "residual" / f"matches2_{m}.parquet") for m in months], ignore_index=True)
        segments = FS.queue_segments(regular, rematch_out, set(callsign[ID]))
        peer = pd.read_parquet(self.runner / "stack" / "peers.parquet")
        return FS.run_tables(callsign, segments, peer)

    def step1(self) -> dict:
        import run_network_member as rnm
        import run_ctx_ownpoints_chain as CH
        from export_disagreement_gate import V34_LIRF, rebuild_v34
        from export_fair_weights import V33_OUTER, recipe
        from export_network_member import stacks
        from prc2026_final import network_member
        combo_decl, mask_decl = read_json(ROOT / COMBO), read_json(ROOT / MASK)
        own_decl = read_json(ROOT / OWN_DECLARATION)
        cand = own_decl["candidates"][OWN_WEIGHT]
        w = float(cand["member_weight"])
        if self.a.step1_source == "shipped":
            _, ids, dry, _, _, v34, at34 = rebuild_v34()
            tables = contextlib.nullcontext()
        else:
            dry = pd.read_parquet(self.runner / "recipe" / "detail.parquet")
            ids = dry[ID].to_numpy()
            st = pd.read_parquet(self.runner / "stack" / "detail.parquet").set_index(ID).reindex(ids)
            v34 = st["v34"].to_numpy(dtype=float)
            at34 = recipe(dry, V33_OUTER, V34_LIRF)
            tables = self.runner_tables()
        if self.a.members in ("shipped", "own"):
            combo = rnm.member_values(combo_decl["member"], "ranking", ids)[0]
            mask = rnm.member_values(mask_decl["member"], "ranking", ids)[0]
            if self.a.members == "shipped":
                member, sources = rnm.member_values(CH.member_spec(own_decl, cand["member"]), "ranking", ids)
            else:   # the shipped combination / masked tables with the own_all member from final_own_member.py (isolates its CPU noise)
                import final_own_member
                member, sources = rnm.member_values(final_own_member.member_spec(self.own_dir), "ranking", ids)
        else:
            import final_own_member
            mem = pd.read_parquet(self.runner / "members" / "members.parquet").set_index(ID).reindex(ids)
            combo, mask = mem["combo_all"].to_numpy(dtype=float), mem["mask_unmatched"].to_numpy(dtype=float)
            member, sources = rnm.member_values(final_own_member.member_spec(self.own_dir), "ranking", ids)
            sources["members.parquet"] = sha256_file(self.runner / "members" / "members.parquet")
        wm = network_member.candidate_weight(mask_decl, "w40")
        with tables:
            k51 = stacks(dry, combo, 0.40, v34, at34, ids)[3]
            k52 = stacks(dry, mask, wm, v34, at34, ids, ((combo, 0.40),))[3]
            c51 = stacks(dry, member, w, v34, at34, ids)[3]
            c52 = stacks(dry, mask, wm, v34, at34, ids, ((member, w),))[3]
        part = pd.read_parquet(ROOT / self.model_part).set_index(ID)[TARGET].reindex(ids)
        if part.isna().any():
            raise ValueError("The model-part file does not cover the file's departures.")
        shipped = part.to_numpy().astype(np.int64)
        lirf = dry["airport"].to_numpy() == "LIRF"
        if np.abs((c52 - k52)[lirf]).max(initial=0.0) > 1e-9 or np.abs((c51 - k51)[lirf]).max(initial=0.0) > 1e-9:
            raise ValueError("A LIRF row depends on the member.")
        changed = (np.abs(c52 - k52) > 1e-9) & ~lirf
        values = np.where(changed, c52, shipped.astype(float))
        out = self.steps / "model_part"
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"vigorous-whistle_v{self.v15}.parquet"
        if path.exists():
            raise ValueError(f"{path} exists; never overwrite.")
        export_submission(pd.DataFrame({ID: ids, "prediction": values}), ROOT / self.template, path,
                          {"dry_run": True, "step": "model part with the member replaced (tools/final_early_chain.py)", "member_tables": sources,
                           "member_weight": w, "step1_source": self.a.step1_source, "members": self.a.members, "official_scores_used": "none", "audit_used": False})
        exported = pd.read_parquet(path).set_index(ID)[TARGET].reindex(ids).to_numpy().astype(np.int64)
        pd.DataFrame({ID: ids, "d51": c51 - k51, "d52": c52 - k52}).to_parquet(out / "member_delta.parquet", index=False)
        info = {"file": rel(path), "sha256": sha256_file(path), "rows_changed_vs_model_part": int((exported != shipped).sum()),
                "rows_float_changed": int(changed.sum()), "squared_change": float(((exported - shipped).astype(float) ** 2).sum()),
                "member_delta_sha256": sha256_file(out / "member_delta.parquet"), "member_tables": sources}
        print("model part", {k: v for k, v in info.items() if k != "member_tables"}, flush=True)
        return info

    # ------------------------------------------------------------ steps 0, 2-5 (the shipped tools with their final-file flags)

    def flags(self, detail: bool = False, published: bool = False) -> list[str]:
        f = []
        if self.raw != SHIPPED_RANKING:
            f += ["--raw", self.raw]
        if self.template != SHIPPED_TEMPLATE:
            f += ["--template", self.template]
        if detail and rel(self.runner) != "runs/final_dryrun/run":
            f += ["--detail", rel(self.runner / "recipe" / "detail.parquet")]
        if published and self.a.published_table:
            f += ["--published-table", rel(self.a.published_table)]
        return f

    def step0(self) -> None:
        run(["tools/run_rematch_nonm.py", "--matches", "--out", rel(self.rematch), *self.flags()])

    def step2(self, m: Path) -> Path:
        run(["tools/run_rematch_nonm.py", "--ranking-base", "--out", rel(self.rematch), "--base", rel(m), "--base-sha256", sha256_file(m), *self.flags()])
        return self.rematch / "base" / "vigorous-whistle_v55m_rematch.parquet"

    def steps_3_to_5(self, b: Path) -> tuple[Path, Path]:
        v, s = str(self.v15), self.steps
        wb = ["--workbook", rel(self.a.workbook), "--workbook-sha256", self.a.workbook_sha256] if self.a.workbook else []
        run(["tools/export_daily_totals.py", "--base", rel(b), "--version", v, "--dry-run", rel(s / "R"), "--no-incumbent-report", *wb,
             *self.flags(detail=True, published=True)])
        run(["tools/scout_extreme_shift.py", "--base", rel(b), "--recon", rel(s / "R"), "--version", v, "--dry-run", rel(s / "X"),
             *self.flags(detail=True, published=True)])
        x = s / "X" / f"vigorous-whistle_v{v}.parquet"
        run(["tools/scout_var_alloc.py", "--base", rel(b), "--recon", rel(s / "R"), "--version", v, "--previous", rel(x), "--previous-sha256", sha256_file(x),
             "--dry-run", rel(s / "Y"), *self.flags(detail=True, published=True)])
        return s / "Y" / f"vigorous-whistle_v{v}.parquet", s / "Y" / "rows.parquet"

    # ------------------------------------------------------------ step 6 (v69's form)

    def step6(self) -> tuple[Path, dict]:
        import export_ctx_ownpoints as E
        import run_ownpoints_weight as W
        import run_stack_resid as S
        import run_stack_weight as SW
        E.CHAIN, E.DECLARATION = W.OUT / "chain", W.DECLARATION
        res = read_json(ROOT / STACK_RESULTS)
        sel = res["selected"]
        cand = res["candidates"][sel]
        w, k = float(cand["weight"]), SW.k_of(cand["seeds"])
        dev, res_t, fx, mats, rk, fxr, h2, h3, ho, Por = SW.rank_inputs()
        P = pd.DataFrame({ID: rk[ID].to_numpy(), "airport": rk["airport"].to_numpy(), "new": rk["new"].to_numpy(dtype=float), "fx": fxr,
                          "h2": h2, "h3": h3, "ho": ho, "o_has": Por["o_has"].to_numpy()})
        info: dict = {"candidate": sel, "weight": w, "seeds": list(cand["seeds"]), "matrices_sha256": {}}
        for key, X, Xr in mats:
            info["matrices_sha256"][key] = {"dev": _frame_hash(X), "rank": _frame_hash(Xr)}
        if self.a.stack_preds:
            ref = pd.read_parquet(ROOT / rel(self.a.stack_preds))
            same = ref[ID].equals(P[ID]) and all(np.array_equal(ref[c].to_numpy(), P[c].to_numpy(), equal_nan=True) for c in ["fx", "h2", "h3", "ho", "o_has"]) \
                and np.array_equal(ref["new"].to_numpy(dtype=float), P["new"].to_numpy(dtype=float))
            if not same:
                raise ValueError("--stack-preds: rows, base or guard columns differ from the recomputed ones.")
            for c in ref.columns:
                if c.startswith(("p2_", "p3_", "po_")):
                    P[c] = ref[c].to_numpy()
            info["preds"] = {"reused": rel(self.a.stack_preds), "sha256": sha256_file(ROOT / rel(self.a.stack_preds))}
        else:
            dom = dev["dom"].to_numpy()
            saved, trees = list(S.SEEDS), {}
            try:
                for key, X, Xr in mats:
                    Xa, Xra = X.to_numpy(np.float32), Xr.to_numpy(np.float32)
                    for i, seed in enumerate(SW.SEEDS10[:k]):
                        S.SEEDS = [seed]
                        models = S.fit_bag(Xa, res_t, ~fx, dom)
                        P[f"{key}_{i}"] = models[0].predict(Xra, num_iteration=models[0].best_iteration)
                        trees[f"{key} {seed}"] = int(models[0].best_iteration)
                        print("stack", key, seed, "trees", trees[f"{key} {seed}"], flush=True)
            finally:
                S.SEEDS = saved
            info["trees"] = trees
        pp = self.steps / "stack" / "preds.parquet"
        pp.parent.mkdir(parents=True, exist_ok=True)
        P.to_parquet(pp, index=False)
        # the late chain's stacker inputs (tools/final_late_chain.py --stacker-rank): MVT_ID_mvt + the 'po' learner's 79 ranking columns,
        # named like runs/stack_second/dev_table.parquet's s_* columns (on the ranking file final_late_chain.py stacker writes 95a5eedc...)
        import pyarrow.parquet as pq
        s_cols = [c for c in pq.ParquetFile(ROOT / "runs/stack_second/dev_table.parquet").schema_arrow.names if c.startswith("s_")]
        X79r = [m for m in mats if m[0] == "po"][0][2].reset_index(drop=True)
        if len(s_cols) != X79r.shape[1]:
            raise ValueError("The stacker's 'po' columns do not match the development table's s_* columns.")
        sr = X79r.copy()
        sr.columns = s_cols
        sr.insert(0, ID, rk[ID].to_numpy())
        srp = self.steps / "stack" / "stacker_rank.parquet"
        sr.to_parquet(srp, index=False)
        info["stacker_rank"] = {"file": rel(srp), "sha256": sha256_file(srp)}
        info["preds_file"] = {"file": rel(pp), "sha256": sha256_file(pp)}
        values, use, _ = SW.rank_value(P, w, k)
        st = self.steps / "stack" / f"vigorous-whistle_v{self.version}.parquet"
        export_submission(pd.DataFrame({ID: P[ID].to_numpy(), "prediction": values}), ROOT / self.template, st,
                          {"dry_run": True, "step": f"stack_weight_v1 {sel}: stacking value at weight {cand['weight']} with {len(cand['seeds'])} seeds (tools/final_early_chain.py)",
                           "official_scores_used": "none", "audit_used": False})
        info["opdi_rows_used"] = int(use.sum())
        return st, info

    # ------------------------------------------------------------ steps 7-10

    def steps_7_to_10(self, st: Path) -> Path:
        import run_own_motion_v2 as O2
        import run_stack_resid_v3 as V3
        v63like = V3.chain_from_stack(st, self.version, self.steps)
        O2.dry_run(v63like, sha256_file(v63like), self.version, self.steps / "om2")
        final = self.out / f"vigorous-whistle_v{self.version}.parquet"
        shutil.copyfile(self.steps / "om2" / f"vigorous-whistle_v{self.version}.parquet", final)
        return final

    # ------------------------------------------------------------ the chain

    def stop(self, step: int) -> bool:
        """--stop-after N: write the partial record (chain_partial.json) and stop after step N."""
        if self.a.stop_after != step:
            return False
        self.record["steps"] = {p.relative_to(self.out).as_posix(): sha256_file(p) for p in sorted(self.steps.rglob("vigorous-whistle_v*.parquet"))}
        self.record["stopped_after"] = step
        write_json(self.out / "chain_partial.json", self.record)
        print("stopped after step", step, json.dumps(self.record["steps"]), flush=True)
        return True

    def run(self) -> Path:
        t0 = time.time()
        self.out.mkdir(parents=True, exist_ok=True)
        if (self.out / "chain.json").exists():
            raise ValueError(f"{self.out} holds a finished chain; use a fresh directory.")
        self.step0()
        self.record["seconds"]["0"] = round(time.time() - t0, 1)
        self.record["configured"] = self.configure()
        t = time.time()
        m = self.step1()
        self.record["model_part_step"] = m
        self.record["seconds"]["1"] = round(time.time() - t, 1)
        if self.stop(1):
            return ROOT / m["file"]
        t = time.time()
        b = self.step2(ROOT / m["file"])
        self.record["seconds"]["2"] = round(time.time() - t, 1)
        if self.stop(2):
            return b
        if self.a.no_published:
            st = b
            self.record["skipped"] = "steps 3-6 (no published daily totals for the period; docs/final_file/v61_chain.md)"
        else:
            t = time.time()
            c, _ = self.steps_3_to_5(b)
            self.record["seconds"]["3-5"] = round(time.time() - t, 1)
            if self.stop(5):
                return c
            t = time.time()
            st, info = self.step6()
            self.record["stack"] = info
            self.record["seconds"]["6"] = round(time.time() - t, 1)
        t = time.time()
        final = self.steps_7_to_10(st)
        self.record["seconds"]["7-10"] = round(time.time() - t, 1)
        self.record["steps"] = {p.relative_to(self.out).as_posix(): sha256_file(p) for p in sorted(self.steps.rglob("vigorous-whistle_v*.parquet"))}
        self.record["final"] = {"file": rel(final), "sha256": sha256_file(final)}
        self.record["seconds"]["total"] = round(time.time() - t0, 1)
        self.record["code"] = {p: sha256_file(ROOT / p) for p in ["tools/final_early_chain.py", "tools/final_own_member.py", "tools/run_opdi_stack.py",
                                                                  "tools/run_stack_weight.py", "tools/run_stack_resid.py", "tools/run_stack_resid_v3.py",
                                                                  "tools/run_intruder_identity.py", "tools/run_own_motion.py", "tools/run_own_motion_v2.py"]}
        write_json(self.out / "chain.json", self.record)
        print("final", self.record["final"], flush=True)
        return final


def _frame_hash(frame: pd.DataFrame) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(json.dumps(list(map(str, frame.columns))).encode())
    h.update(np.ascontiguousarray(frame.to_numpy(np.float32)).tobytes())
    return h.hexdigest()


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=SHIPPED_RANKING)
    ap.add_argument("--template", default=SHIPPED_TEMPLATE)
    ap.add_argument("--runner", default="runs/final_dryrun/run")
    ap.add_argument("--model-part", default=SHIPPED_M55)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", type=int, required=True)
    ap.add_argument("--step1-source", choices=["runner", "shipped"], default="runner")
    ap.add_argument("--members", choices=["runner", "shipped", "own"], default="runner",
                    help="runner: R's members + final_own_member.py's own_all; shipped: the declarations' GPU tables; own: shipped combination / masked + the CPU own_all (ranking file)")
    ap.add_argument("--published-table", default=None)
    ap.add_argument("--workbook", default=None)
    ap.add_argument("--workbook-sha256", default=None)
    ap.add_argument("--no-published", action="store_true")
    ap.add_argument("--stack-preds", default=None)
    ap.add_argument("--own-dir", default=None, help="--members runner: tools/final_own_member.py's output directory (default R/own)")
    ap.add_argument("--stop-after", type=int, choices=[1, 2, 5], default=None, help="stop after step 1, 2 or 5 (checks)")
    ap.add_argument("--rebuild-caches", action="store_true", help="ranking file: rebuild the identity tables, own-point and OPDI caches under O/cache as for a new file")
    return ap


def main() -> None:
    a = parser().parse_args()
    if a.no_published and a.published_table:
        raise ValueError("--no-published and --published-table exclude each other.")
    Chain(a).run()


if __name__ == "__main__":
    main()
