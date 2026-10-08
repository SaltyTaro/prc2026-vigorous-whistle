# "What's left" audit on v72's development values (8 October 2026, exploration; nothing declared)

v72 development values = v71's (the control of `configs/stack_second_v1.json`) + sp_hicov7's out-of-month five-seed bag (0.5 x, cap 120 s,
floor -12) at the seven high-coverage airports. Month-blocked single-seed LightGBM per variant on clip(label - v72, +-1800), correction
0.5 x prediction capped +-120 s, fixed rows held; complete-month RMSE change against v72 with paired-day intervals (2,000 replicates).
Scripts [`aud3_fit.py`](aud3_fit.py) (base settings) and [`aud4_fit.py`](aud4_fit.py) (regularised: 15 leaves, min_data 4,000, L2 200,
200 rounds; Huber alpha 300); logs [`aud3.log`](aud3.log), [`aud4.log`](aud4.log); per airport-month tables in `audit3_results.json`,
`audit4_results.json`. Run on arcw0 (seconds per variant), inputs hash-checked (dev_table 344f9393..., preds_dev 6a92ed72...).

**Third pass at the seven airports: none.** In-set variants read eq +0.006% (all inputs), -0.002% (with the second pass's prediction as
an input), -0.001% (stacker inputs + that prediction); July worse in all (+0.04 to +0.05%), EGLL worse in three months. The second pass
took what the development months contain there.

**LFPG / LTFM / LIRF: below the bar in every form.** Best readings: queue + neighbour columns trained on all airports, applied at the
three, eq -0.0209% with January [-0.007, +0.004] and July [-0.024, +0.005] across zero (LFPG better in all four months, -0.9 / -1.9 /
-2.2 / -2.4e6 s²; LIRF and LTFM mixed); applied at LFPG alone it is about -0.013% (arithmetic from the same table); Huber version at LFPG
-0.0085%; regularised all-input pass at the three -0.0082% (January worse); LTFM alone -0.0015%; LIRF alone -0.0022% (January worse).
Full-input passes overfit LIRF (July +12e6 s²) as on v71. **Closed: the residual-pass family is exhausted on development.**

## Follow-up on v73's development values (8 October, after v73; exploration, single seed; `aud6_fit.py`, `aud7_fit.py`, `aud8_fit.py`)

v73 dev = v72 dev + t3_ro_3's out-of-month bag at EDDF / EHAM / LSZH. Inputs: raw ADS-B (as t3) plus the second / third passes' predictions
and changes; `aud7` adds the IEM weather columns (`wx_build.py`, 61 columns, full coverage); `aud8` fits and applies on matched rows only
(dark / gate-traced). Runs on arcw0-2 (arcw0's first run stopped after four variants; the rest was rerun, `aud6b`).

| variant | January | July | November | December | eq |
|---|---:|---:|---:|---:|---:|
| EDDM+EGLL+LEBL+LEMD in-set (raw) | +0.0087 | +0.0401 | -0.0492 | -0.0658 | -0.0057 |
| the same, trained on the seven | +0.0079 | +0.0376 | -0.0318 | -0.0689 | -0.0078 |
| EDDM / EGLL / LEBL / LEMD alone | -0.0007 / +0.0044 / +0.0052 / +0.0031 | +0.0015 / +0.0376 / +0.0095 / +0.0016 | | | -0.0016 / -0.0056 / +0.0028 / +0.0014 |
| fourth pass at EDDF / EHAM / LSZH | +0.0028 | +0.0025 | -0.0118 | -0.0066 | -0.0004 |
| weather, all ten / LFPG+LTFM+LIRF / LTFM / LFPG / LIRF | +0.060 / +0.072 / +0.013 / +0.047 / +0.033 | +0.053 / +0.034 / +0.008 / +0.020 / +0.056 | | | +0.037 / +0.044 / +0.022 / +0.029 / +0.023 |
| weather + raw at the three / weather at the seven | +0.0035 / +0.0046 | +0.0045 / +0.0231 | | | +0.0030 / +0.0083 |
| matched rows only: EGLL | +0.0005 | +0.0107 | -0.0829 | -0.0600 | -0.0163 |
| matched rows only: EDDM / LEBL / LEMD | -0.0005 / +0.0051 / 0.0000 | +0.0015 / +0.0087 / -0.0012 | | | -0.0016 / +0.0030 / -0.0006 |
| matched rows only: LIRF / LFPG | 0 (no rows) | +0.1146 / +0.0021 | -0.0213 / -0.0046 | -0.0404 / +0.0103 | +0.0247 / +0.0042 |

**Closed.** No fourth pass at the three; the raw pass does not extend to the other airports (EGLL gains November / December but loses July
2025 even on matched rows, its July 2025 coverage collapse); weather in a late pass overfits month-specific weather everywhere. The late-pass
family is exhausted on development. Noted, not run: the networks and committee learners do not see the first ADS-B point's position (lat /
lon), the third pass's top inputs; a member refit with it would need a GPU round and a full chain rebuild.

Also closed the same morning (no file): a Rome dark-month level shift. On the thinned-July simulation the shipped dark arithmetic (v54) is
unbiased on average (all rows -0.6 s, ordinary rows +1.2 s; typical rows +32 s against an under-predicted tail), so the +27 s that the
published monthly mean implies for July 2026 is not a simulated dark-mode bias.
