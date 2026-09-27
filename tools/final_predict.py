"""v30's recipe on any movement file with the ranking file's schema, from the saved models only (docs/final_file/runner.md).

Subcommands (all take --root, a fresh directory per input file; every stage is resumable and writes <root>/stages/<stage>.json last):
  inspect      --raw F --template T   schema, months, arrivals, departures vs the template -> <root>/input.json
  external                            what new dates would need: missing adsb.lol surface days (extraction commands for an Azure
                                      worker) and weather station-months no inventory covers; prints only, fetches nothing
  extract-adsb [--execute]            adsb.lol surface extraction for the file's missing days (tools/adsb_extract_day.py, 2-4 GB download
                                      per day from GitHub; an Azure worker is the recommended host, then copy the surface_<day>.parquet
                                      files back); without --execute it only prints the commands
  pull-weather [--execute]            the IEM pull for the uncovered station-months (tools/pull_prc_weather.fetch, one request at a
                                      time) into data/external/iem_final/<root name>/ plus an inventory JSON for --weather-inventory;
                                      without --execute it only prints the requests
  refit-adj2   --month M              the declared residual adj2 refit for a target month without a saved booster (~5 min, 16 threads)
  run          [--stages ...]         features, adsb, tokens, v9, leaves, networks, residual, recipe, export (default: all pending)
Nothing is uploaded and nothing is written to runs/submissions; the export stage writes <root>/export/vigorous-whistle_v<N>.parquet.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["inspect", "external", "extract-adsb", "pull-weather", "refit-adj2", "run"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--stages", nargs="*", default=None)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--workers", type=int, default=8, help="day-parallel ADS-B matching / v19 processes")
    parser.add_argument("--weather-inventory", type=Path, nargs="*", default=[], help="extra IEM inventories (pull-weather output) for new dates")
    parser.add_argument("--allow-missing-weather", action="store_true")
    parser.add_argument("--version", type=int, default=None, help="export file name vigorous-whistle_v<N>.parquet inside the root (export stage)")
    parser.add_argument("--detail-ensembles", action="store_true", help="also predict ens_ctx/ens_new/ens_nn/ens_nnl (v30-detail columns the recipe does not use)")
    parser.add_argument("--month", default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    from prc2026_final.common import Run
    run = Run(ROOT / args.root if not args.root.is_absolute() else args.root)
    if args.command == "inspect":
        from prc2026_final.runner import stage_inspect
        if not args.raw or not args.template:
            raise SystemExit("inspect needs --raw and --template")
        record = stage_inspect(run, args.raw.resolve(), args.template.resolve())
        print(json.dumps({k: record[k] for k in ["movements", "departures", "arrivals", "arrivals_with_in_block", "months", "departure_months",
                                                  "departures_in_template_order", "departures_with_label_or_block_in_file", "extra_columns_ignored"]}, indent=1))
        return
    if args.command in ("external", "extract-adsb", "pull-weather"):
        external(run, args)
        return
    if args.command == "refit-adj2":
        from prc2026_final.residual import refit_adj2
        if not args.month:
            raise SystemExit("refit-adj2 needs --month YYYY-MM")
        print(refit_adj2(args.month, run.root / "residual" / "adj2", args.threads))
        return
    from prc2026_final import runner
    ctx = runner.Context(run)
    stages = args.stages or runner.STAGES
    for stage in stages:
        if stage not in runner.STAGES:
            raise SystemExit(f"Unknown stage {stage}")
        if run.done(stage):
            run.verify_outputs(stage)
            print(stage, "done (receipt present; outputs verified)", flush=True)
            continue
        started = time.time()
        if stage == "features":
            record = runner.stage_features(ctx, [p.resolve() for p in args.weather_inventory], args.allow_missing_weather)
        elif stage == "adsb":
            record = runner.stage_adsb(ctx, args.workers)
        elif stage == "tokens":
            record = runner.stage_tokens(ctx)
        elif stage == "v9":
            record = runner.stage_v9(ctx, args.threads)
        elif stage == "leaves":
            record = runner.stage_leaves(ctx, args.threads)
        elif stage == "networks":
            from prc2026_final.networks import DETAIL_ONLY, RECIPE_ENSEMBLES
            record = runner.stage_networks(ctx, args.threads, RECIPE_ENSEMBLES + (DETAIL_ONLY if args.detail_ensembles else []))
        elif stage == "residual":
            record = runner.stage_residual(ctx, args.threads)
        elif stage == "recipe":
            record = runner.stage_recipe(ctx)
        else:
            if args.version is None:
                raise SystemExit("the export stage needs --version N (the file is written inside the run root)")
            record = runner.stage_export(ctx, args.version)
        print(stage, "finished in", round(time.time() - started, 1), "s", flush=True)


def external(run, args) -> None:
    import pandas as pd
    from prc2026.artifacts import read_json
    from prc2026_final import adsb, features
    from prc2026_final.common import departures_of, load_movements
    meta = run.input()
    departures = departures_of(load_movements(Path(meta["raw"])))
    missing_days = adsb.surface_days_missing(departures)
    manifest, _, _, _ = features.timeline_bundle(ROOT)
    frozen = features.verify_external(ROOT, features.weather_config(ROOT, manifest))["weather"]
    inventories = [frozen, *[read_json(p) for p in args.weather_inventory]]
    uncovered = features.weather_coverage(inventories, departures)
    plan = {"surface_days_missing": missing_days,
            "adsb_extraction_commands (run on an Azure worker, then copy data/external/adsblol/surface/surface_<day>.parquet back)":
                [f".venv/bin/python -X utf8 tools/adsb_extract_day.py --day {d[:4]}.{d[4:6]}.{d[6:]}" for d in missing_days],
            "weather_station_months_uncovered": uncovered}
    if args.command == "external":
        print(json.dumps(plan, indent=1))
        return
    if args.command == "extract-adsb":
        import subprocess
        commands = [[sys.executable, "-X", "utf8", "tools/adsb_extract_day.py", "--day", f"{d[:4]}.{d[4:6]}.{d[6:]}", "--workers", str(args.workers)] for d in missing_days]
        print(json.dumps({"would_run" if not args.execute else "running": [" ".join(c[1:]) for c in commands]}, indent=1))
        if args.execute:
            for command in commands:
                subprocess.run(command, cwd=ROOT, check=True)
        return
    from pull_prc_weather import fetch
    directory = ROOT / "data/external/iem_final" / run.root.name
    requests = [(station, month) for station, months in uncovered.items() for month in months]
    print(json.dumps({"would_fetch" if not args.execute else "fetching": [{"station": s, "month": m, "into": (directory / f"{s}_{m}.csv").relative_to(ROOT).as_posix()} for s, m in requests]}, indent=1))
    if not args.execute:
        return
    results = []
    for station, month in requests:
        start = pd.Timestamp(f"{month}-01")
        end = (start + pd.offsets.MonthBegin(1))
        results.append(fetch(station, month, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), directory, False))
    inventory = {"provider": "Iowa Environmental Mesonet", "purpose": f"final-file weather for {meta['raw']}", "objects": sorted(results, key=lambda r: (r["station"], r["period"])), "failures": []}
    out = directory / "inventory.json"
    out.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    print({"inventory": out.relative_to(ROOT).as_posix(), "objects": len(results)})


if __name__ == "__main__":
    main()
