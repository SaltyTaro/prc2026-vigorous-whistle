"""December 2025, opened on 19 September 2026 (docs/december/authorization_20260919.json).

The frozen feature packages refuse December by construction (five `raise` guards). Nothing frozen is
edited: `patched.py` loads those modules from their unchanged source with exactly one guard line each
replaced by `pass`, records the source hashes, and `features.py` runs the six feature layers
(phase 3, 4, 5R, weather, NM clock, timeline) for December's departures on top of the Phase-1
bundle's December rows. `--verify` runs the same orchestration for November and requires exact
equality with the frozen timeline bundle, which is the evidence that the orchestration matches.
"""
