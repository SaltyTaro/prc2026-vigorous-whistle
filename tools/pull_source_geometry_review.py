"""Reuse the bounded public reader under the separate mapping-review declaration."""
from pathlib import Path
import pull_source_geometry

if __name__ == "__main__":
    pull_source_geometry.CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs/source_geometry_mapping_review_v1.json"
    pull_source_geometry.main()
