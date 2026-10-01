# LIRF dark-regime re-read (1 October 2026, after v53): `w70_s750` selected by the declared rule

Declaration [`configs/lirf_dark_reread_v1.json`](../../configs/lirf_dark_reread_v1.json) (18:24:29 UTC, d9b0d4c, before any candidate
score), scorer `tools/run_lirf_dark.py` (same commit), report [`results_v1.json`](results_v1.json). No fit: the tables and member files of
the LIRF masked-member round ([`docs/adsb_ctx3_lirf_mask`](../adsb_ctx3_lirf_mask/README.md)) are re-read.

## Question

v53 put the masked LIRF network M into the LIRF network at weight 0.55 (`network' = 0.45 ens_ctx_lirf + 0.55 M`) and kept v34's unmatched
blend around it (network 0.583 / tree 0.417, v9 0) with v45's gate u600. Both numbers were set before M existed. Where LIRF is dark for the
whole month (July 2026 in the ranking, 0.2% of LIRF non-population rows matched; January 2026 is 80% matched and stays at v53), do a higher
member weight w and a higher network share s do better?

- Dark sets (selectable): **thinned July** (July 2025 thinned to July 2026's coverage, 14,771 route rows below 20,000 s) and **January
  2025** (LIRF naturally dark, 11,254 rows). Covered months July / November / December 2025 are read descriptively only; the candidate never
  applies there.
- Grid: w ∈ {0.55, 0.70, 0.85, 1.00} × s ∈ {0.583, 0.667, 0.75}, gate u600 recomputed on network'. Control = v53's arithmetic.
- Rule: thinned July ≤ −0.25% with the paired-day interval below zero, January 2025 ≤ 0 with its interval not entirely above zero; lowest
  sum of the two. **Disclosed:** the w grid above 0.55 and the dark-month scope were chosen from the masked-member round's monotone trend on
  these same two sets (reuse, not fresh evidence). No network share other than 0.583 had been read with M.

## Control check

The control reproduces the masked-member round's w55 reading on all five sets exactly (route RMSE difference 0.0 s; thinned July 415.62 s,
January 2025 271.36 s), and every natural month's `lirf_net` equals the shipped network's 16-seed fold mean.

## Result (primary: route rows with labels below 20,000 s; change against v53's arithmetic)

| candidate | thinned July | January 2025 | eligible |
|---|---|---|---|
| w55_s667 | −0.836% [−1.58, −0.37] | −0.109% [−0.38, +0.15] | yes |
| w55_s750 | −1.421% [−2.86, −0.52] | −0.064% [−0.60, +0.45] | yes |
| w70_s583 | −0.128% [−0.28, +0.02] | −0.061% | no |
| w70_s667 | −0.976% [−1.70, −0.48] | −0.164% [−0.53, +0.15] | yes |
| **w70_s750** | **−1.570% [−2.98, −0.66], 23/31 days** | **−0.110% [−0.71, +0.44], 13/31 days** | **selected** |
| w85_s583 | −0.088% | +0.018% | no |
| w85_s667 | −0.906% [−1.64, −0.34] | −0.052% | yes |
| w85_s750 | −1.467% | +0.039% | no |
| w100_s583 | +0.009% | +0.197% | no |
| w100_s667 | −0.765% | +0.182% | no |
| w100_s750 | −1.278% | +0.332% | no |

w70_s750 moves the route RMSE 415.62 → 412.35 s on thinned July and 271.36 → 271.21 s on January 2025; the gate fires on 92 thinned-July
rows (control 91).

- **The gain is the network share, not the member weight.** At s 0.583 every higher w is flat (−0.13% at best); at s 0.75 every w
  gains 1.3–1.6% on thinned July. Replacing the shipped network entirely (w 1.0) is worse on January at every share.
- January 2025 (naturally dark) is nearly flat in s: its least-squares optimum on the grid is w 0.75 / s 0.675 at only −0.17%; thinned July's
  is w 0.70 / s 0.90 at −2.0%. The selected candidate rests on thinned July, the simulation of the summer regime that July 2026 is in.
- Descriptive, covered months (the candidate never applies there): w70_s750 would read July 2025 −0.62%, November −0.22%, December −0.30%;
  the direction of a higher network share holds outside the dark sets too.
- Gate, descriptive at the control: T 1200 reads thinned July +0.22%, no gate +2.41% (January +0.30% / +0.20%); u600 still pays with M.

## Expectation (not a prediction)

Member step on v53's model part (`tools/export_lirf_dark.py`, dry run `runs/lirf_dark/export_dry`): 15,024 of July 2026's 15,515 route rows
change, mean −2.3 s, RMS 28 s, largest 352 s; squared change S = 1.25e7; gate rows 72 → 72; no other row changes. ΔSSE = S(1 − 2α):
−0.08 s at alignment 1, about −0.23 s at thinned July's implied alignment (1.89: S 1.44e7, ΔSSE −4.0e7 on its route), about −0.02 s
at January 2025's (0.64: S 3.2e6, ΔSSE −0.9e6), +0.08 s if the moves are noise.

## Dry run of a v54 (not built under runs/submissions, not uploaded)

v48's reconciliation applied last (`tools/export_daily_totals.py --base runs/lirf_dark/export_dry/vigorous-whistle_v54.parquet --dry-run
runs/daily_totals/dry_v54 --version 54`): SHA-256 `d9dcefe7857ef45d1298757fb8068553cd00690eab6f7e073fac0e8bcef2bed1` (member step
`b0fbbd19…`). Against the uploaded v53: 15,128 rows differ, **all LIRF July 2026** (mean −1.6 s, largest 376 s), squared change 1.31e7:
−0.09 s at alignment 1, about −0.24 s at thinned July's implied alignment, +0.09 s if noise. Building v54 under `runs/submissions`, its audit
and any upload are the user's decision.

## v54 (built at the user's word, "build v54"; NOT uploaded)

`tools/export_daily_totals.py --base runs/lirf_dark/export_dry/vigorous-whistle_v54.parquet --version 54` →
`runs/submissions/vigorous-whistle_v54.parquet`, SHA-256 `d9dcefe7857ef45d1298757fb8068553cd00690eab6f7e073fac0e8bcef2bed1`,
byte-identical to the dry run. Record `docs/daily_totals/export_v54.json`. Audit (`docs/export_audit/v54_audit.json`): clean; against v53
only LIRF rows differ (RMS 6.2 s over the file, none above 600 s), every other airport identical; the remaining flags (January 2026
level shifts, EHAM 2–9 January disruption days) are v53's unchanged. Upload only at the user's word:
`tools/prc_submit.py upload --file runs/submissions/vigorous-whistle_v54.parquet --expected-sha256 d9dcefe7… --calendar-day`.

## v54 official: 221.8686 (18:38 UTC, 1 October) — −0.0240 s against v53; the official best

Uploaded at the user's word ("upload v54") at about 18:38 UTC (`tools/prc_submit.py upload --calendar-day`), processed 18:38:47 UTC,
344,841 rows ([`official_result_v54.json`](official_result_v54.json)). The squared change against v53 was 1.31e7 and the score moved by
ΔSSE ≈ −3.7e6, an alignment of about 0.64: below alignment 1 and far below thinned July's 1.89, and equal to January 2025's reading
(0.64), the dark set that barely preferred the change. The figure describes the outcome; nothing is rescaled from it. Never re-upload v1–v54. New layers build on v54's steps: v52's members, the LIRF masked member, the
dark-month LIRF re-read, then the reconciliation last.
