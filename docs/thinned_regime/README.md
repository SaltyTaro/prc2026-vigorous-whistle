# Network share on the ranking-regime sets (`configs/thinned_regime_round_v1.json`, 29 September 2026): nothing selected; v46 (the user's override) officially 223.9339

Model work on the clean line (v45 = 224.2398). No official score enters this round.

## Question

The committee mix holds the context networks at share 0.75 against the tree committee. Three earlier readings on natural coverage
selected no other share (`docs/share_scope/`, `docs/lomo_dev/`, `docs/disagreement/`). The scored months have less adsb.lol coverage
than 2025 at some airports, and unmatched rows are where the networks beat the trees, so this round reads the share once on the coverage
campaign's thinned months (`configs/coverage_regime_v1.json`):

| set | regime it stands for | what changes against natural coverage |
|---|---|---|
| July 2025 thinned (`a13_jul`) | July 2026 | EDDF 91.7% → 64.9% matched, LIRF 93.9% → 0.6%, LSZH 91.1% → 90.3% |
| December 2025 thinned (`a13_dec`) | January 2026 | EDDF 94.8% → 79.4% matched |

The question was prompted by the leaderboard calibration's reading that the network layers were under-applied
. The declaration discloses that, and the earlier natural readings. Sets, candidates, weights and rule
use 2025 labels only.

## Method

`tools/run_thinned_regime_round.py` assembles v34's recipe per row from the saved thinned re-predictions of every component (committee,
LIRF trees, residual members, context-network ensembles, LIRF context network) and scores candidates with
`tools/run_fair_weights.build`. Control: v34 (outer weights LTFM 1.0 / matched 0.6 / unmatched 0.7, network share 0.75). The natural
equal-footing months (January and July on the eleven-month footing, December fair, `runs/lomo_dev/rows.parquet`) are no-harm checks.

Assembly check before any candidate was scored: the natural assembly reproduces the development v33 (July 261.47475 against 261.47481 s,
December 168.16078 against 168.16150 s; 8 and 12 rows differ by more than 1 s, from the context networks' local re-predictions).

Rule, declared at 06:17 UTC and committed (`e47f395`) before scoring: on **both** thinned months the RMSE is below the control's with the
paired-day 95% interval below zero, the mean thinned ratio is at least 0.02% below 1, and no natural equal-footing month is worse.

## Results (`results_v1.json`)

Change in month RMSE against the control; interval of the RMSE difference in seconds.

| candidate | thinned July | thinned December | natural January | natural July | natural December | eligible |
|---|---|---|---|---|---|---|
| n85 (share 0.85) | −0.016% [−0.18, +0.10] | −0.139% [−0.35, −0.14] | −0.011% | −0.045% | −0.134% | no |
| n100 (share 1.0) | +0.062% | −0.176% [−0.53, −0.10] | +0.022% | −0.007% | −0.163% | no |
| u85 (unmatched rows 0.85) | −0.029% [−0.20, +0.06] | −0.085% [−0.24, −0.04] | −0.004% | −0.051% | −0.084% | no |
| u100 (unmatched rows 1.0) | +0.016% | −0.080% [−0.34, +0.09] | +0.030% | −0.039% | −0.076% | no |
| nd1200 (share 0.5 where the networks sit > 1,200 s below the committee) | −0.138% [−0.66, −0.15] | +0.014% | −0.025% | −0.099% | +0.017% | no |
| n85_nd1200 (both) | −0.221% [−0.91, −0.34] | −0.133% [−0.46, +0.005] | −0.048% [−0.39, −0.04] | −0.195% [−0.79, −0.28] | −0.125% [−0.44, +0.02] | no |

Control RMSE: thinned July 265.345, thinned December 170.192, natural January 323.413, July 259.486, December 168.101.

**Nothing is selected.**

- The thinned months read like the natural ones. The regime changes coverage at three airports only, so the share's reading barely
  moves: the higher share is a December effect (−0.13% to −0.18%) and is within noise in July.
- The gate is a July effect and loses December, on thinned coverage as on natural coverage.
- `n85_nd1200` is better on all five sets (thinned mean −0.177%, natural mean −0.123%) with three of five intervals below zero. It
  fails the rule on December: the thinned interval's upper end is +0.005 s.

## Standing

The share question is closed on four readings. `n85_nd1200` is the only candidate that improves every set; shipping it would be an
override of the declared rule and is the user's decision. At full transfer its development mean is worth about 0.3 s on the ranking;
the precedents for rule overrides are mixed (v26 gained, v32 lost, v38 gained a tenth of its development reading).

## v46 (built 29 September on the user's override)

The user's decision ("ok build it") overrides the declared rule for `n85_nd1200`: network share 0.85, and share 0.5 where the committee
sits more than 1,200 s above the networks.

**Re-read on the full v45 stack** (`tools/run_share_gate_reread.py`, [`reread_n85_nd1200.json`](reread_n85_nd1200.json)): the candidate's
mix through v34's weights, the u300 gate, the callsign and queue layers, the two-sided stretch and the LIRF gate, on
`runs/lomo_dev/rows.parquet`.

| month | control RMSE | change | paired-day 95% interval (s) | alignment of the move |
|---|---|---|---|---|
| January (eleven-month footing) | 322.996 | −0.043% | [−0.36, −0.03] | 1.49 |
| July (eleven-month footing) | 258.154 | −0.169% | [−0.76, −0.17] | 1.89 |
| November (nine-month fold side) | 184.936 | −0.017% | [−0.19, +0.14] | 0.62 |
| December (fair) | 167.621 | −0.064% | [−0.36, +0.14] | 0.87 |

Mean over the equal-footing months −0.092%, smaller than the v34-level reading (−0.123%): the stretch and the u300 gate take part of
the same ground. "Alignment" is −Σ d·r / Σ d² of the move d against the control's residual r: 1 means the truth lies at the new
value on average, 0.5 is break-even.

**Export** (`tools/export_share_gate.py --candidate n85_nd1200 --version 46 --user-override …`, [`export_v46.json`](export_v46.json)):
`runs/submissions/vigorous-whistle_v46.parquet`, SHA-256 `8abdd87ad5098d7753927f407ffd7e8cd342aa84a2864392e1b64c0dde015af9`,
byte-identical to the dry run.

- Base v45 (the clean line). The control stack rebuilt from the dry-run detail rounds to the uploaded v38 on every row; v46 takes the
  candidate's float where it differs from the control and v45's integer elsewhere.
- 263,907 rows change, all outside LIRF: mean −0.23 s, mean absolute 4.1 s, largest 665 s. The gate fires on 55 rows (January 40,
  July 15), which rise by 341 s on average. LIRF rows, fill pins and the LIRF gate are untouched. Sum of squared moves 2.51e7.
- Independent rebuild from the recorded v30 detail: 181 rows differ by 1 s (float noise), the gate rows are the same.
- Audit `docs/export_audit/v46_audit.json`: clean (no missing or extra rows, floor −12 s, three rows move by more than 600 s).
- Tests: `tests/test_share_gate.py` (4), with `tests/test_lirf_gate.py` (2): all pass.

**Expectation, with its assumptions:** against v45's 224.2398, a move of this size is worth −0.16 s at alignment 1, −0.32 s at 1.5
(January and July read 1.5–1.9 in development, December 0.9), and costs +0.16 s if it is noise. No official score enters the file. Upload is the user's
decision; the final-file runner gets a `v46` layer set only if the layer ships and gains.

## v46 official: 223.9339 (06:37 UTC, 29 September) — −0.3059 s against v45; the clean line's best

Uploaded at the user's word ("upload it") at 06:36 UTC, processed 06:37:01 UTC, 344,841 rows
([`official_result_v46.json`](official_result_v46.json)). Against v45 (224.2398) the squared error falls by 4.73e7 for a sum of squared
moves of 2.51e7: the move's alignment on the ranking is 1.44, inside the range the development months read for January and July
(1.5–1.9) and above December's (0.9). This is recorded as an outcome only; no score enters any weight.

- v46 is the best file of the clean model
  line: v45 224.2398, v40 224.4588, v38 226.8053.
- Never re-upload v1–v46.
- The final-file runner has the `v46` layer set and defaults to it (`src/prc2026_final/stack.py`, `share_gate.py`). Rebuilt on the
  dry-run root it reproduces the uploaded v46 within the chain's float noise: 1,196 integers differ by at most 7 s, the 55 gate rows
  are the same, and every earlier layer's column is unchanged ([`../final_file/dryrun_v46_compare.json`](../final_file/dryrun_v46_compare.json)).
- The declared rule selected nothing, and the override gained. One outcome does not change the rule: v32 was an override that lost.
