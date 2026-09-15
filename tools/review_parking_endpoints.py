"""Check documented parking-way endpoints against explicit mapped parking nodes."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prc2026_source_geometry.endpoint_review import main

if __name__ == "__main__":
    main(ROOT)
