"""Prepare the declared label-free Rome geometry feature extension."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prc2026_rome_geometry.bundle import main

if __name__ == "__main__":
    main()
