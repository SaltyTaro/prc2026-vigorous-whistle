"""Optional input overrides for running the post-runner steps of the v61 chain on a new movement file (docs/final_file/v61_chain.md).

Every flag defaults to None; with no flag given no module attribute is touched, so each tool keeps its shipped defaults byte for byte.

  --raw FILE              the movement file (schema of data/raw/ranking.parquet) instead of data/raw/ranking.parquet
  --detail FILE           the final-file runner's <root>/recipe/detail.parquet (scope per departure) instead of runs/final_dryrun/run's
  --published-table FILE  the parquet extract of EUROCONTROL's All_Pre-Departure_Delay workbook (tools/screen_daily_totals.extract into
                          another path) covering the new period, instead of data/external/eurocontrol_aiu/All_Pre-Departure_Delay.parquet
  --template FILE         the new file's submission template instead of data/raw/submitting.parquet (tools that export through
                          prc2026_submission.contract.export_submission)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RANKING_MODULES = ["export_daily_totals", "scout_extreme_shift", "scout_var_alloc", "run_stack_resid", "run_intruder_bound", "run_intruder_identity",
                   "run_rematch_nonm"]


def add_arguments(parser) -> None:
    parser.add_argument("--raw", default=None, help="movement file instead of data/raw/ranking.parquet (docs/final_file/v61_chain.md)")
    parser.add_argument("--detail", default=None, help="the runner's <root>/recipe/detail.parquet instead of runs/final_dryrun/run's")
    parser.add_argument("--published-table", default=None, help="the published-totals parquet extract covering the new period")
    parser.add_argument("--template", default=None, help="the new file's submission template instead of data/raw/submitting.parquet")


def _rel(path: str) -> str:
    p = Path(path)
    if not p.is_absolute():
        return p.as_posix()
    try:
        return p.relative_to(ROOT).as_posix()   # lexical: junctions under the project root (runs/ on E:) keep their project path
    except ValueError:
        return p.as_posix()


def apply(args) -> dict:
    """Set the overrides on every loaded module that carries the attribute; returns what was overridden (for provenance)."""
    done = {}
    if getattr(args, "raw", None):
        import export_daily_totals  # noqa: F401  (the scout tools import its ranking_table lazily: load it so the override reaches it)
        rel = _rel(args.raw)
        for name in [*RANKING_MODULES, "__main__"]:   # a tool run as a script is __main__, not its own module name
            mod = sys.modules.get(name)
            if mod is not None and hasattr(mod, "RANKING"):
                mod.RANKING = (ROOT / rel) if isinstance(mod.RANKING, Path) else rel
        done["raw"] = rel
    if getattr(args, "detail", None):
        import export_v33
        export_v33.DRY = _rel(args.detail)
        done["detail"] = export_v33.DRY
    if getattr(args, "published_table", None):
        import screen_daily_totals
        screen_daily_totals.TABLE = ROOT / _rel(args.published_table)
        done["published_table"] = _rel(args.published_table)
    if getattr(args, "template", None):
        template = ROOT / _rel(args.template)
        for name in list(sys.modules):
            mod = sys.modules[name]
            fn = getattr(mod, "export_submission", None)
            if name in [*RANKING_MODULES, "__main__"] and callable(fn):
                def wrapped(predictions, template_path, output, provenance, _fn=fn):
                    return _fn(predictions, template, output, provenance)
                mod.export_submission = wrapped
        done["template"] = _rel(args.template)
    return done
