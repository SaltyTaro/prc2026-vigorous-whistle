"""Resumable stage orchestration for tools/final_predict.py (each stage writes <root>/stages/<stage>.json last)."""
from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from prc2026.artifacts import read_json, sha256_file, write_json
from prc2026.features import ID, TIME
from prc2026.validation import identity_hash
from prc2026_submission.contract import export_submission

from . import adsb, features, leaves, networks, recipe, residual, tokens, v9
from .common import ROOT, Run, departures_of, inspect, load_movements, save_frame

CHAIN = ["features", "adsb", "tokens", "v9", "leaves", "networks", "residual", "recipe"]  # v30 on the file (also run on the repaired input)
STAGES = CHAIN + ["repair", "rematch", "stack", "final"]  # default order: v30 -> v35 / v36 and the chosen layer set's file
EXTRA_STAGES = ["export"]  # v30's own integer file (component check against the uploaded v30); run only when named


@dataclass
class Options:
    threads: int = 12
    workers: int = 8
    weather_inventory: list[Path] = field(default_factory=list)
    allow_missing_weather: bool = False
    version: int | None = None
    detail_ensembles: bool = False
    vocab_cache: Path | None = None
    repair_root: Path | None = None
    layers: str = "v40"  # the final file's layer set: v40 (the official best, 224.4588; default), v39, v38, v35, v36 or v37


class Context:
    """The run's input (masked movements) loaded once per process."""

    def __init__(self, run: Run):
        self.run = run
        meta = run.input()
        self.meta = meta
        self.raw_path = Path(meta["raw"])
        self.template = Path(meta["template"])
        if sha256_file(self.raw_path) != meta["raw_sha256"] or sha256_file(self.template) != meta["template_sha256"]:
            raise ValueError("The movement file or template changed since `inspect`.")
        self.movements = load_movements(self.raw_path)
        self.departures = departures_of(self.movements)
        self.months = meta["months"]
        self.ids = self.departures[ID].to_numpy()
        self.row_months = self.departures[TIME].dt.strftime("%Y-%m").to_numpy()


def stage_inspect(run: Run, raw: Path, template: Path) -> dict:
    record = inspect(raw, template)
    if (run.root / "input.json").exists():
        old = run.input()
        if old["raw_sha256"] != record["raw_sha256"] or old["template_sha256"] != record["template_sha256"]:
            raise ValueError("This run root belongs to another movement file or template; use a fresh root.")
    write_json(run.root / "input.json", record)
    return record


def stage_features(ctx: Context, extra_inventories: list[Path], allow_missing_weather: bool) -> dict:
    started = time.time()
    inventories = [read_json(p) for p in extra_inventories]
    table, record = features.build(ROOT, ctx.movements, ctx.months, inventories, allow_missing_weather)
    if not np.array_equal(table[ID].to_numpy(), ctx.ids):
        raise ValueError("Feature rows are not the file's departures in order.")
    out = save_frame(table, ctx.run.path("features", "timeline.parquet"))
    record["extra_weather_inventories"] = [{"path": p.as_posix(), "sha256": sha256_file(p)} for p in extra_inventories]
    return ctx.run.finish("features", record, [out], started)


def stage_adsb(ctx: Context, workers: int) -> dict:
    started = time.time()
    record = adsb.build(ctx.run, ctx.departures, ctx.months, workers)
    outputs = record.pop("outputs")
    if record["surface_days_missing"]:
        record["warning"] = "surface days missing: those departures are unmatched (run `final_predict.py external` for the extraction commands)"
    return ctx.run.finish("adsb", record, outputs, started)


def learner_matches(ctx: Context, label: str) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(ctx.run.root / "adsb" / label / f"matches2_{m}.parquet") for m in ctx.months], ignore_index=True)


def stage_tokens(ctx: Context) -> dict:
    started = time.time()
    record = tokens.build(ctx.run.root / "tokens", ctx.movements, learner_matches(ctx, "learners"))
    return ctx.run.finish("tokens", record, [ctx.run.root / "tokens" / f"{tokens.NAME}.npz"], started)


def timeline(ctx: Context) -> pd.DataFrame:
    return pd.read_parquet(ctx.run.root / "features" / "timeline.parquet")


def stage_v9(ctx: Context, threads: int) -> dict:
    started = time.time()
    frame, record = v9.build(timeline(ctx), ctx.movements, threads)
    out = save_frame(frame, ctx.run.path("v9", "v9.parquet"))
    return ctx.run.finish("v9", record, [out], started)


def learner_frame(ctx: Context) -> pd.DataFrame:
    frame = timeline(ctx).merge(pd.read_parquet(ctx.run.root / "adsb" / "learner_adsb.parquet"), on=ID, how="left", validate="one_to_one")
    frame = frame.merge(pd.read_parquet(ctx.run.root / "adsb" / "v19.parquet"), on=ID, how="left", validate="one_to_one")
    if frame["adsb_matched"].isna().any() or not np.array_equal(frame[ID].to_numpy(), ctx.ids):
        raise ValueError("The learner frame does not cover the departures in order.")
    return frame


def stage_leaves(ctx: Context, threads: int, vocab_cache: Path | None = None) -> dict:
    started = time.time()
    specs = leaves.leaf_specs()
    cache_path = ctx.run.path("leaves", "vocabularies.json")
    seeded = leaves.seed_vocabularies(cache_path, vocab_cache) if vocab_cache else None
    vocab = leaves.vocabularies(cache_path, specs)
    frame = learner_frame(ctx)
    predictions, info = {}, {}
    for leaf, spec in specs.items():
        target = ctx.run.path("leaves", "leaf", leaf.replace("/", "__") + ".parquet")
        if target.exists():
            predictions[leaf] = pd.read_parquet(target)["adsb_full"].to_numpy(dtype=float)
        else:
            t0 = time.time()
            predictions[leaf] = leaves.predict_leaf(spec, frame, vocab[leaves.vocab_key(spec["inputs"])], threads)
            save_frame(pd.DataFrame({ID: ctx.ids, "adsb_full": predictions[leaf]}), target)
            print(f"leaf {leaf} {round(time.time() - t0)} s", flush=True)
        info[leaf] = {"kind": spec["kind"], "target": spec["inputs"].get("target"), "model_sha256": spec["model_sha256"], "finite_rows": int(np.isfinite(predictions[leaf]).sum())}
    del frame
    gc.collect()
    tables = pd.DataFrame({ID: ctx.ids})
    members = leaves.members()
    for name, member_list in members.items():
        tables[name] = leaves.table_mean(member_list, predictions)
    out = save_frame(tables, ctx.run.path("leaves", "tables.parquet"))
    record = {"leaves": info, "tables": {n: {"members": len(m), "distinct": len(set(m)), "all_nan_rows": int(tables[n].isna().sum())} for n, m in members.items()},
              "vocabulary_cache": {"seed": seeded, "sha256": sha256_file(cache_path)}}
    return ctx.run.finish("leaves", record, [out], started)


def stage_networks(ctx: Context, threads: int, ensembles: list[str]) -> dict:
    started = time.time()
    from prc2026_ctx.tokens import TokenStore
    frame = learner_frame(ctx)
    host = TokenStore(ctx.run.root / "tokens", [tokens.NAME])
    learners = networks.ensemble_learners()
    cache, info = {}, {}
    table = pd.DataFrame({ID: ctx.ids})
    for ensemble in ensembles:
        columns = []
        for learner in learners[ensemble]:
            target = ctx.run.path("networks", ensemble, learner.replace("/", "__") + ".parquet")
            if target.exists():
                values = pd.read_parquet(target)["adsb_full"].to_numpy(dtype=float)
            else:
                t0 = time.time()
                values, info[learner] = networks.predict_learner(learner, frame, host, threads, cache)
                save_frame(pd.DataFrame({ID: ctx.ids, "adsb_full": values}), target)
                print(f"{ensemble} {learner} {round(time.time() - t0)} s", flush=True)
            columns.append(values)
        table[ensemble] = np.mean(columns, axis=0)
    out = save_frame(table, ctx.run.path("networks", "ensembles.parquet"))
    return ctx.run.finish("networks", {"ensembles": {e: len(learners[e]) for e in ensembles}, "learners_predicted_this_session": info}, [out], started)


def stage_residual(ctx: Context, threads: int) -> dict:
    started = time.time()
    v9_frame = pd.read_parquet(ctx.run.root / "v9" / "v9.parquet")
    table = residual.residual_table(ctx.departures, learner_matches(ctx, "residual"), v9_frame)
    table_out = save_frame(table, ctx.run.path("residual", "table.parquet"))
    frame, record = residual.build(table, ctx.row_months, ctx.run.root / "residual" / "adj2", threads)
    out = save_frame(frame, ctx.run.path("residual", "residual.parquet"))
    record["table_sha256"] = sha256_file(table_out)
    return ctx.run.finish("residual", record, [table_out, out], started)


def stage_recipe(ctx: Context) -> dict:
    started = time.time()
    table = pd.read_parquet(ctx.run.root / "residual" / "table.parquet")
    v9_frame = pd.read_parquet(ctx.run.root / "v9" / "v9.parquet")
    parts = [pd.read_parquet(ctx.run.root / "residual" / "residual.parquet", columns=[ID, "residual"]),
             pd.read_parquet(ctx.run.root / "leaves" / "tables.parquet"), pd.read_parquet(ctx.run.root / "networks" / "ensembles.parquet")]
    for part in parts:
        table = table.merge(part, on=ID, how="left", validate="one_to_one")
    if not np.array_equal(table[ID].to_numpy(), ctx.ids) or not np.array_equal(table["v9"].to_numpy(), v9_frame["v9"].to_numpy()):
        raise ValueError("Recipe inputs are misaligned.")
    table["month"] = ctx.row_months
    floor = float(v9_frame["floor"].iloc[0])
    detail, record = recipe.compute(table, floor)
    out = save_frame(detail, ctx.run.path("recipe", "detail.parquet"))
    record["ids_sha256"] = identity_hash(detail[ID])
    return ctx.run.finish("recipe", record, [out], started)


def stage_export(ctx: Context, version: int) -> dict:
    started = time.time()
    detail = pd.read_parquet(ctx.run.root / "recipe" / "detail.parquet")
    output = ctx.run.path("export", f"vigorous-whistle_v{version}.parquet")
    provenance = {"runner": "tools/final_predict.py", "input": {k: ctx.meta[k] for k in ["raw", "raw_sha256", "template", "template_sha256", "months", "departures"]},
                  "stages": {s: sha256_file(ctx.run.receipt_path(s)) for s in CHAIN}, "recipe": ctx.run.receipt("recipe")["candidate"],
                  "note": "written into the run root only; never copied to runs/submissions or uploaded by this tool"}
    receipt = recipe.export(detail, ctx.template, output, provenance)
    return ctx.run.finish("export", {"receipt": receipt}, [output, output.with_suffix(".receipt.json")], started)


def stage_repair(ctx: Context, options: Options) -> dict:
    """v31's input: the label-free EOBT_1 detector; for flagged airport-months, the chain on the edited input in a sub-root."""
    from . import stack
    started = time.time()
    raw = pd.read_parquet(ctx.raw_path)
    flagged, table = stack.detect(raw)
    shares_out = save_frame(table, ctx.run.path("repair", "shares.parquet"))
    record = {"rule": "flag an airport-month when its share of departures with EOBT_1 > IOBT exceeds the airport's 2025 maximum by more than "
                      "the 2025 span (maximum - minimum over the twelve 2025 months); edit EOBT_1 := IOBT where EOBT_1 > IOBT on its departures",
              "flagged": [{"airport": a, "month": m} for a, m in flagged],
              "closest_unflagged": table.loc[~table["flagged"]].nlargest(3, "margin")[["airport", "month", "share", "threshold", "margin"]].to_dict(orient="records"),
              "repair_root": None, "edited_rows": 0}
    outputs = [shares_out]
    if flagged:
        edited, mask = stack.reset_input(raw, flagged)
        del raw
        ids_out = save_frame(pd.DataFrame({ID: edited.loc[mask, ID].to_numpy()}), ctx.run.path("repair", "edited_ids.parquet"))
        outputs.append(ids_out)
        record["edited_rows"] = int(mask.sum())
        if options.repair_root is not None:
            sub = Run(options.repair_root)
            for stage in CHAIN:
                if not sub.done(stage):
                    raise ValueError(f"--repair-root {sub.root} has no {stage} receipt.")
                sub.verify_outputs(stage)
            meta = sub.input()
            if meta["template_sha256"] != ctx.meta["template_sha256"] or not stack.same_content(pd.read_parquet(meta["raw"]), edited):
                raise ValueError(f"--repair-root {sub.root} was not run on this file's edited input (or on another template).")
            record["reused"] = True
        else:
            path = ctx.run.path("repair", "input", "final.parquet")
            if path.exists():
                if not stack.same_content(pd.read_parquet(path), edited):
                    raise ValueError(f"{path} exists with other content; delete it (and {ctx.run.root / 'repair' / 'run'}) to rebuild.")
            else:
                save_frame(edited, path)
            del edited
            gc.collect()
            sub = Run(ctx.run.root / "repair" / "run")
            stage_inspect(sub, path, ctx.template)
            sub_ctx = Context(sub)
            cache = ctx.run.root / "leaves" / "vocabularies.json"
            sub_options = replace(options, detail_ensembles=False, vocab_cache=cache if cache.exists() else options.vocab_cache, repair_root=None)
            for stage in CHAIN:
                if sub.done(stage):
                    sub.verify_outputs(stage)
                    continue
                print("repair:", stage, flush=True)
                run_stage(sub_ctx, stage, sub_options)
            del sub_ctx
            gc.collect()
            record["reused"] = False
        record["repair_root"] = sub.root.resolve().as_posix()
        record["repair_input_sha256"] = sub.input()["raw_sha256"]
        record["repair_recipe_receipt_sha256"] = sha256_file(sub.receipt_path("recipe"))
        record["repair_detail_sha256"] = sha256_file(sub.root / "recipe" / "detail.parquet")
    return ctx.run.finish("repair", record, outputs, started)


def stage_rematch(ctx: Context) -> dict:
    """The callsign fix's input: tools/adsb_rematch_nonm.py on the no-NM EDDF / EHAM departures."""
    from . import stack
    started = time.time()
    deps = stack.rematch_departures(ctx.departures)
    deps_out = save_frame(deps, ctx.run.path("rematch", "deps.parquet"))
    out = ctx.run.path("rematch", "out.parquet")
    stack.rematch(deps_out, out)
    days = sorted(deps[TIME].dt.strftime("%Y%m%d").unique().tolist())
    missing = [d for d in days if not (adsb.SURFACE / f"surface_{d}.parquet").exists()]
    record = {"departures": int(len(deps)), "rows_out": int(len(pd.read_parquet(out))), "surface_days_missing": missing}
    if missing:
        record["warning"] = "surface days missing: those departures get no rematch (run `final_predict.py external`)"
    return ctx.run.finish("rematch", record, [deps_out, out], started)


def stage_stack(ctx: Context) -> dict:
    """v31 -> v33 -> v34 -> v35 on the recipe detail (tools/export_stack.py's construction, the run's own tables)."""
    from . import stack
    started = time.time()
    for stage in ["recipe", "repair", "rematch"]:
        ctx.run.verify_outputs(stage)
    detail = pd.read_parquet(ctx.run.root / "recipe" / "detail.parquet")
    if not np.array_equal(detail[ID].to_numpy(), ctx.ids):
        raise ValueError("The recipe detail is not in the file's departure order.")
    rep = ctx.run.receipt("repair")
    repaired = None
    if rep["repair_root"] is not None:
        path = Path(rep["repair_root"]) / "recipe" / "detail.parquet"
        if sha256_file(path) != rep["repair_detail_sha256"]:
            raise ValueError(f"{path} changed after the repair receipt was written.")
        repaired = pd.read_parquet(path, columns=[ID, "v30"]).set_index(ID)["v30"].reindex(ctx.ids).to_numpy(dtype=float)
    regular = pd.concat([pd.read_parquet(ctx.run.root / "adsb" / "residual" / f"matches2_{m}.parquet") for m in ctx.months], ignore_index=True)
    peer = stack.peers(ctx.raw_path)
    peer_out = save_frame(peer, ctx.run.path("stack", "peers.parquet"))
    fill = stack.pins(ctx.raw_path)
    fill_out = save_frame(fill, ctx.run.path("stack", "fill_pins.parquet"))
    frame, record = stack.compute(detail, repaired, ctx.run.root / "rematch" / "out.parquet", regular, peer, fill)
    flagged = {(f["airport"], f["month"]) for f in rep["flagged"]}
    in_flagged = np.array([(a, m) in flagged for a, m in zip(frame["airport"], frame["month"])], dtype=bool)
    record["rows"]["repair_effect_nonzero_outside_flagged"] = int(((frame["repair_effect"].to_numpy() != 0) & ~in_flagged).sum())
    out = save_frame(frame, ctx.run.path("stack", "detail.parquet"))
    return ctx.run.finish("stack", record, [out, peer_out, fill_out], started)


RECIPES = {"v35": "v35: v30 + EOBT_1 repair 0.5 + v33/v34 weights + u300 + callsign fix 0.5 + queue trust 0.75 (regular matches)",
           "v39": "v39: v38, then the fill pins (configs/fill_pin_v1.json: LIRF long-interval population and one-day-clock departures at their schedule interval)",
           "v40": "v40: v38, then configs/fill_pin_v2.json evidenced_nine (LIRF long-interval population departures with intervals in [43,200, 58,000] or above 79,200 s at their schedule interval)",
           "v36": "v36: v35's layers after the mu600 mix gate, then the peer-congestion lifts (configs/higher_bundle_v2.json mu_pc_pcc)",
           "v37": "v37: v35, then the tail stretch above 1,500 s (configs/tail_stretch_v1.json v35_t1500, four-month k)",
           "v38": "v38: v35, then the two-sided stretch (above 1,500 s, below 900 s; configs/tail_stretch_v2.json two_sided_900, four-month k)"}


def receipt_key(stage: str, options: Options) -> str:
    """The final stage keeps one receipt per layer set (stages/final-v35.json, stages/final-v36.json), so both files can live in one root."""
    return f"final-{options.layers}" if stage == "final" else stage


def stage_final(ctx: Context, version: int, layers: str) -> dict:
    """The chosen layer set's integer file in the template's order (export_submission), inside the run root."""
    from .stack import LAYER_SETS
    if layers not in LAYER_SETS:
        raise ValueError(f"--layers must be one of {LAYER_SETS}")
    started = time.time()
    ctx.run.verify_outputs("stack")
    frame = pd.read_parquet(ctx.run.root / "stack" / "detail.parquet", columns=[ID, layers])
    output = ctx.run.path("final", f"vigorous-whistle_v{version}.parquet")
    provenance = {"runner": "tools/final_predict.py", "layers": layers, "recipe": RECIPES[layers],
                  "input": {k: ctx.meta[k] for k in ["raw", "raw_sha256", "template", "template_sha256", "months", "departures"]},
                  "stages": {s: sha256_file(ctx.run.receipt_path(s)) for s in STAGES[:-1]},
                  "note": "written into the run root only; never copied to runs/submissions or uploaded by this tool"}
    receipt = export_submission(pd.DataFrame({ID: frame[ID].to_numpy(), "prediction": frame[layers].to_numpy(dtype=float)}), ctx.template, output, provenance)
    return ctx.run.finish(f"final-{layers}", {"layers": layers, "version": version, "receipt": receipt}, [output, output.with_suffix(".receipt.json")], started)


def run_stage(ctx: Context, stage: str, options: Options) -> dict:
    if stage == "features":
        return stage_features(ctx, [p.resolve() for p in options.weather_inventory], options.allow_missing_weather)
    if stage == "adsb":
        return stage_adsb(ctx, options.workers)
    if stage == "tokens":
        return stage_tokens(ctx)
    if stage == "v9":
        return stage_v9(ctx, options.threads)
    if stage == "leaves":
        return stage_leaves(ctx, options.threads, options.vocab_cache.resolve() if options.vocab_cache else None)
    if stage == "networks":
        from .networks import DETAIL_ONLY, RECIPE_ENSEMBLES
        return stage_networks(ctx, options.threads, RECIPE_ENSEMBLES + (DETAIL_ONLY if options.detail_ensembles else []))
    if stage == "residual":
        return stage_residual(ctx, options.threads)
    if stage == "recipe":
        return stage_recipe(ctx)
    if stage == "repair":
        return stage_repair(ctx, options)
    if stage == "rematch":
        return stage_rematch(ctx)
    if stage == "stack":
        return stage_stack(ctx)
    if options.version is None:
        raise SystemExit(f"the {stage} stage needs --version N (the file is written inside the run root)")
    if stage == "export":
        return stage_export(ctx, options.version)
    if stage == "final":
        return stage_final(ctx, options.version, options.layers)
    raise SystemExit(f"Unknown stage {stage}")
