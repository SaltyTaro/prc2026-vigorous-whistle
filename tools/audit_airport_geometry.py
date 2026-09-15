"""Run exact parking/graph coverage, optionally including all historical snapshots."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026_source_geometry.geometry_audit import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complete", action="store_true")
    arguments = parser.parse_args()
    main(ROOT, complete=arguments.complete)
