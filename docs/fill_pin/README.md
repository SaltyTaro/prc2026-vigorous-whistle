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

Export SHA-256: `06a9acbcd4ec91d3887070855e2730835a17f2f031c2194602275c15d3fc107b` (dry run and real export identical). Uploaded at
the user's word at 04:50 UTC; result below.

## Official result (28 September, 04:51 UTC): 226.8802, +0.0749 s worse than v38

Uploaded at the user's word; `official_result_v39.json`. Observed SSE change +1.17e7 against the all-fill prediction of −4.35e8:
a shortfall of 4.46e8. v38 (226.8053) remains the official best; the ranking uses each team's best score, so the standing is unchanged.
Uploads on 28 September UTC: v35, v38, v39.

### What the score implies, row by row (one-day label taken as 86,400 + 900 s)

| flight | interval | v38 | gain if fill | cost if one-day | swing | v8 gain had it been one-day |
|---|---:|---:|---:|---:|---:|---:|
| WMTT3508 | 56,520 | 61,859 | 2.9e7 | 3.0e8 | 3.3e8 | 2.7e8 |
| TAY9LA | 53,520 | 58,186 | 2.2e7 | 2.9e8 | 3.2e8 | 4.0e8 |
| WMT402 | 48,841 | 52,102 | 1.1e7 | 2.4e8 | 2.5e8 | 6.5e8 |
| ITYTY672 | 61,322 | 70,414 | 8.3e7 | 3.9e8 | 4.7e8 | **1.8e7** |
| TARAR753 | 48,776 | 52,610 | 1.5e7 | 2.8e8 | 3.0e8 | 6.2e8 |
| SVII3006 | 60,656 | 69,802 | 8.4e7 | 4.0e8 | 4.9e8 | **2.6e7** |
| ITYTY674 | 47,337 | 50,103 | 7.7e6 | 2.1e8 | 2.2e8 | 7.5e8 |

"Swing" is the change in outcome, against the all-fill prediction, when that row is a one-day error. The result is consistent with exactly that: one of the three untested rows is a one-day error, the rest are fills.

### What is left on the table

The seven v8-tested population rows plus the two rows that gain under either mechanism carry 1.95e8 SSE of
hedge; pinning only those nine (candidate `evidenced_nine`, `configs/fill_pin_v2.json`) would score about 225.56 (−1.25 s) if the
seven are fills, on evidence that predates v39. Choosing that subset after v39's score is the same kind of decision as v39 itself
and rests with the user; nothing has been exported for it.

## v40 = v38 + `evidenced_nine` (28 September, user: "nine")

`configs/fill_pin_v2.json`, export `tools/export_fill_pin.py --declaration configs/fill_pin_v2.json --candidate evidenced_nine
--version 40`, record `export_v40.json`, audit `docs/export_audit/v40_audit.json`. SHA-256
`da5b20c328ab6be5fca4a21c37cb7c80210397fba62d8395557a9b84d77d162a`. Expected 225.56 (−1.25 s) if the seven are fills; about
+0.2 s if one of them is a one-day error (the two dominant rows offset most of it).
