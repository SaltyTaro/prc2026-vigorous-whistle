"""Final-file runner: v30's recipe on any movement file with the ranking file's schema, from the saved models only.

Insurance against a separate final test file (docs/final_file/inventory.md). Every stage imports the frozen functions that
first computed its column for the ranking file and writes its outputs under a run root with a JSON receipt, so a run is
resumable and can be split across sessions. Nothing frozen is edited: where a frozen function hard-codes the ranking file,
its months or a "must reproduce vN" guard, the stage re-implements only the plumbing (the December package is the template).
No refit happens except the residual stage's per-target-month adj2 booster, which exists only for 2026-01 and 2026-07 and
is refitted by an explicit, separately invoked subcommand for any other month. Nothing is downloaded or uploaded here.
"""
