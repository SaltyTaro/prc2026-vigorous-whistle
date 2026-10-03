# Intruder stand bound — ACCEPTED by its declared rule (3 October 2026)

Declaration [`configs/intruder_bound_v1.json`](../../configs/intruder_bound_v1.json) (8d04cfa, after the disclosed exploration of the
dig campaign, `docs/dig/README.md`); governing run `tools/run_intruder_bound.py --score` → [`results_v1.json`](results_v1.json).
No official score and no leaderboard information is used.

## Mechanism

A departure leaves its stand before another aircraft can block in there. If our own aircraft (an arrival of our type) blocked in at our
stand and then an arrival of a different type blocked in at the same stand before the off-block our prediction implies, we must have
left before that second arrival: label > B = take-off − the intruder's in-block. The truth sits some minutes beyond the bound (the
intruder taxis in after we push: median 483 s in 2025), so the value is B + o, o fitted per airport and gap band on the other months.

The bound holds for 96.7–100% of such rows at **EDDF, EGLL, LFPG and LTFM** in every 2025 month and fails elsewhere (EDDM, EHAM, LEBL,
LEMD, LSZH, LIRF: stand codes shared by several positions, or towing). This is why the pooled clamp closed on 28 September (+17% dev
SSE) failed; the airport set here comes from the fitting months' validity, and the inbound rule removes the previous occupant.
Intruders are arrivals from other airports (100%), with ordinary taxi-in (median 786 s); none coincides with the true off-block.

## Governing run (candidate g3600_w100: gap up to 3,600 s, full move)

| month | change of complete-month RMSE | paired-day 95% | rows moved |
|---|---:|---|---:|
| January 2025 | −0.1694% | [−0.659, −0.045] | 24 |
| July 2025 | −1.1591% | [−2.084, −0.386] | 63 |
| November 2025 | −0.2291% | [−0.569, −0.055] | 22 |
| December 2025 | −0.2036% | [−0.480, −0.024] | 14 |

Control: the shipped chain on the shipped model part (`runs/scout/refit/ctl_v55m.parquet`). Equal-footing mean **−0.511%**, every
interval below zero, no month worse. Twelve-month check (older predictor): **12 of 12 months better, 11 intervals below zero**
(January −0.18, February −1.09, July −1.14, November −0.57%, the others −0.09 to −0.33%). **Rule: accepted.** The disclosed
sensitivity g1800_w50 also passes (−0.152%; twelve months 12/12 with 12 intervals).

July's gain is spread over 63 rows (the top three carry 28%); in the other months the top three carry 55–82%. Development gain per
unit of squared change: 1.68 / 1.16 / 0.94 / 0.67 (January / July / November / December).

## Ranking dry run (label-free; nothing under runs/submissions)

`tools/run_intruder_bound.py --dry-run --version 58` on the uploaded v57 (read only):
`runs/intruder_bound/dry_v58/vigorous-whistle_v58.parquet`, SHA-256 `eabd3bd6cfccfb70de3ecdc12ad32872d20e5316619b9a87fd44f121be46c247`.
52 rows move, all upward (EDDF 9, EGLL 12, LFPG 17, LTFM 14; January 33, July 19, spread over many days), squared change 7.10e7, largest
move +3,050 s; guard rows and the LIRF population untouched. At the 2025 gain ratios (0.67–1.68 × the squared change) the expected effect
is about −0.3 to −0.8 s. Building it under runs/submissions and uploading are the user's word.
