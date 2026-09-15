"""Run the declared fitting-only LIRF/LFPG reporting audit."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026_source_geometry.source_audit import main

if __name__ == "__main__":
    main(ROOT)
