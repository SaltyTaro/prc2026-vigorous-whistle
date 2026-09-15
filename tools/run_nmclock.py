"""Train the declared NM timestamp comparisons on the authorized CPU worker."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from prc2026_nmclock.training import main

if __name__ == '__main__':
    main()
