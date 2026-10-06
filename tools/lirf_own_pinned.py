"""Run a shipped chain script with the variance re-allocation's LightGBM fits pinned (deterministic, column-wise), as every run of the
LIRF own-points round does (tools/run_lirf_own_dev.py): LightGBM otherwise picks row- or column-wise histograms by a timing test, so
reruns under load differ. Only src/prc2026_scout/var_alloc.PARAMS gains {"deterministic": True, "force_col_wise": True}; the script
runs unchanged. Usage: tools/lirf_own_pinned.py tools/<script>.py [args ...]
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026_scout import var_alloc  # noqa: E402

var_alloc.PARAMS = {**var_alloc.PARAMS, "deterministic": True, "force_col_wise": True}

if __name__ == "__main__":
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    sys.path.insert(0, str((ROOT / script).parent))
    runpy.run_path(str(ROOT / script), run_name="__main__")
