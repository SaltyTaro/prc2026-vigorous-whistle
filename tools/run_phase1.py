"""Repository entry point; works without an editable package installation."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prc2026.phase1 import main

if __name__ == "__main__":
    main()
