"""Run the declared Phase-4 development suite or its frozen refit."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prc2026_phase4.training import main

if __name__ == "__main__":
    main()
