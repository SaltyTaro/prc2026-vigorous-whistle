# Fill pin: v39 = v38 with twelve LIRF long-interval rows at their schedule interval (28 September 2026)

Declaration `configs/fill_pin_v1.json` (04:40 UTC), rule `src/prc2026_final/fill_pin.py`, export `tools/export_fill_pin.py`,
record `export_v39.json`, runner layer set `v39` (`tools/final_predict.py --layers v39`). No model was fitted, no development score
exists for these rows, no label was read. The choice between the declared candidates was the user's ("all twelve").

## What the rows are

LIRF departures whose takeoff is more than 12 hours after their scheduled departure. For the eleven **population** rows (no NM flight
record, no aircraft type) the 2025 labels take only two forms: a **schedule fill** (the airport wrote the scheduled time as the block
time, so the label equals the interval) or a **one-day date error** (label = 86,400 s + an ordinary taxi). 2025 holds 18 fills and 11
one-day rows with no separating feature, so the frozen v7 learned a partial hedge: interval + 2.8..9.1 ks. The only 2025 precedent for that clock pattern
(AEZ146R, 2025-07-15) was a fill.

## Why the fill reading

v8 (15 September) hedged the same eleven rows harder, toward the one-day value, and scored 8.611 s worse than v7 (+1.659e9 SSE).
Under "all fills" the hedge alone costs +1.248e9, the rest being the routing and floor rules. Had even one of the eleven been a
one-day row, the hedge would have removed a ~30 ks error on it (−0.45e9 on average, range −0.88e9..−0.014e9) and v8 would have
scored *better* than v7. The v10 result (+0.18e9 with the same routing rules) corroborates that the non-hedge rules cost no more than
about 0.4e9. The 2025 prior alone would argue for *more* hedge, which v8 refuted.

This reading uses an official score to conclude something about specific ranking rows. Earlier records
(`docs/postprocess/README.md`, `docs/audit_20260922/README.md` section 4) declined to do so under the project's own no-inference
stance; v8 was a genuine candidate rather than a probe, and the decision was put to the user with both readings and taken by them.

## Stake

| candidate | rows | if every row is a fill | if one population row is a one-day error |
|---|---|---|---|
| all_twelve (chosen) | 12 | −4.35e8 SSE = 2.45% of v38's, **226.8053 → 224.0097 (−2.80 s)** | about −1.09 s net |
| dominant_three | 3 | about −1.1e8 SSE (−0.7 s) | still better (both mechanisms lie below the current prediction) |

Break-even for all_twelve is roughly two one-day rows among the eleven.

## The export

The dry run and the real export produce the same
parquet content. The independent scan in `export_v39.json` re-derives the scope from the raw file and checks that no row outside it
moved. Audit: `docs/export_audit/v39_audit.json`.

## Final-file readiness

The same rule is a layer set of the final-file runner (`v39` = `v38` + `fill_pins` of the file's own departures; `stack/fill_pins.parquet`
records the scope), so a hidden final file gets the pin on its own long-interval LIRF rows. Tests: `tests/test_fill_pin.py`
(scope rule, application, the ranking's twelve rows) and the stack arithmetic test's v39 branch. The dry run on the ranking file
(`docs/final_file/dryrun_v39_compare.json`) reproduces the twelve pinned rows exactly; its other differences to the export are the
runner's known float noise (1,157 rows, ≤ 7 s, identical to v938 vs v38).

Export SHA-256: `06a9acbcd4ec91d3887070855e2730835a17f2f031c2194602275c15d3fc107b` (dry run and real export identical). Not uploaded.
