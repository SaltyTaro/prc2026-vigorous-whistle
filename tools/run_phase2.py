"""Entry point for PRC CPU baselines and a frozen-recipe ranking refit."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prc2026_models.experiments import main

if __name__ == "__main__":
    main()
