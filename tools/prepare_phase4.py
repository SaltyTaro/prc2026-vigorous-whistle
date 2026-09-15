"""Materialize an immutable Phase-4 feature bundle."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prc2026_phase4.bundle import main

if __name__ == "__main__":
    main()
