"""Run tools/screen_trace_features.py (the 29 September raw-trace feature builder, unchanged) with a different surface folder and output
folder. Usage: trace_build.py SURFACE_DIR OUT_DIR MONTH [MONTH ...]. The builder's source is executed as is apart from those two paths."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
src_path = ROOT / "tools/screen_trace_features.py"
src = src_path.read_text(encoding="utf-8")
print("builder sha256", hashlib.sha256(src_path.read_bytes()).hexdigest(), flush=True)
surface, out, months = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), sys.argv[3:]
old_surf = 'ROOT / f"data/external/adsblol/surface/surface_{day}.parquet"'
old_out = 'OUT = ROOT / "runs/clean_line_screens/trace"'
assert src.count(old_surf) == 1 and src.count(old_out) == 1
src = src.replace(old_surf, f'Path({str(surface)!r}) / f"surface_{{day}}.parquet"').replace(old_out, f"OUT = Path({str(out)!r})")
out.mkdir(parents=True, exist_ok=True)
sys.argv = [str(src_path)] + months
exec(compile(src, str(src_path), "exec"), {"__name__": "__main__", "__file__": str(src_path)})
