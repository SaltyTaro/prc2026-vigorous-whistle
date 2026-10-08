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
