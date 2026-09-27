"""Resumable stage orchestration for tools/final_predict.py (each stage writes <root>/stages/<stage>.json last)."""
from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

from prc2026.artifacts import read_json, sha256_file, write_json
from prc2026.features import ID, TIME
from prc2026.validation import identity_hash

from . import adsb, features, leaves, networks, recipe, residual, tokens, v9
from .common import ROOT, Run, departures_of, inspect, load_movements, save_frame

STAGES = ["features", "adsb", "tokens", "v9", "leaves", "networks", "residual", "recipe", "export"]


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


def stage_leaves(ctx: Context, threads: int) -> dict:
    started = time.time()
    specs = leaves.leaf_specs()
    vocab = leaves.vocabularies(ctx.run.path("leaves", "vocabularies.json"), specs)
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
    record = {"leaves": info, "tables": {n: {"members": len(m), "distinct": len(set(m)), "all_nan_rows": int(tables[n].isna().sum())} for n, m in members.items()}}
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
                  "stages": {s: sha256_file(ctx.run.receipt_path(s)) for s in STAGES[:-1]}, "recipe": ctx.run.receipt("recipe")["candidate"],
                  "note": "written into the run root only; never copied to runs/submissions or uploaded by this tool"}
    receipt = recipe.export(detail, ctx.template, output, provenance)
    return ctx.run.finish("export", {"receipt": receipt}, [output, output.with_suffix(".receipt.json")], started)
