# Weights of the two network members (30 September 2026): nothing selected

The user: "do Weight round for the two members next". Both GPU rounds read their largest declared weight (0.40) as below the optimum in
seven of eight month readings (alignment of the change 1.5 / 1.2 / 0.8 / 1.1 for the combination, 1.1 / 1.5 / 2.0 / 1.6 for the masked
member). This round re-weights the two members v52 ships on their existing tables; no model is fitted.

- Declaration `configs/member_weights_v1.json` (09:12 UTC, commit 288a301), before any score of its candidates: the combination weight
  c and the masked-member weight m each 0.40 / 0.55 / 0.70 (eight candidates); control = both at 0.40, v52's model part; unchanged
  rule. No official score enters.
- Scorer `tools/run_member_weights.py` (commit 3fca794). Before any candidate was read, the v46 stack reproduced the v46 re-read and
  the control reproduced the masked round's recorded w40 RMSEs on all four months (322.5887 / 257.4131 / 184.6605 / 167.1763).

## Result (`docs/member_weights/results_v1.json`): the rule selects nothing

| candidate | January | July | December | mean (rule months) | November (reported) |
|---|---:|---:|---:|---:|---:|
| `c40_m70` | −0.010% [−0.074, +0.005] | −0.004% [−0.065, +0.046] | −0.025% [−0.088, +0.004] | −0.013% | −0.011% |
| `c70_m70` | −0.013% | +0.000% | −0.019% | −0.010% | −0.000% |
| `c55_m70` | −0.013% | −0.000% | −0.016% | −0.010% | +0.001% |
| `c55_m55` | −0.012% | −0.007% | −0.011% | −0.010% | −0.006% |
| `c70_m55` | −0.015% | −0.002% | −0.008% | −0.008% | +0.009% |
| `c40_m55` | −0.006% | −0.007% | −0.011% | −0.008% | −0.017% |
| `c55_m40` | −0.008% | −0.003% | +0.012% | +0.001% | +0.018% |
| `c70_m40` | −0.011% | +0.003% | +0.021% | +0.004% | +0.036% |

No candidate reaches the 0.02% mean bar, and no candidate has all three intervals below zero. The two members overlap on the unmatched
rows (both are masked networks there), so the under-application each round read on its own does not add up: once both are in the
table at 0.40, larger weights move the stack by a hundredth of a percent. v52's weights stay. Nothing is exported.

## Round v2: combination weight 0.25 (1 October 2026, UTC 30 September 21:56): nothing selected

**Why.** The thinned-coverage check (`docs/member_thin/`) read December at the network-table level only. There, combination 0.25 with
masked 0.70 was −0.171% on natural inputs: the combination member slightly hurts December's matched rows at that level. 0.25 was not
in this round's grid, so the user asked for the stack-level re-read.

**Declaration.** [`configs/member_weights_v2.json`](../../configs/member_weights_v2.json) (cc19769), before any stack-level score of its
candidates. It keeps the scorer, the control and the rule and discloses that the grid is adaptive. The candidates are combination 0.25
with masked 0.25, 0.40, 0.55 and 0.70. The v46 re-read and the control reproduce on all four months.
Results: [`results_v2.json`](results_v2.json), `runs/member_weights/selection_v2.json`.

| candidate | January | July | December | mean (rule months) | November (reported) |
|---|---:|---:|---:|---:|---:|
| `c25_m70` | −0.004% [−0.054, +0.043] | +0.001% [−0.060, +0.065] | −0.009% [−0.063, +0.032] | −0.004% | −0.010% |
| `c25_m55` | +0.002% | +0.004% | +0.008% | +0.005% | −0.012% |
| `c25_m40` | +0.012% [+0.011, +0.081] | +0.011% | +0.018% [+0.004, +0.060] | +0.014% | +0.006% |
| `c25_m25` | +0.018% | +0.024% | +0.045% | +0.029% | +0.032% |

**Nothing is selected.**
- At the stack level, 0.25 is worse than v52's 0.40: at masked 0.40 it is +0.014%, with January's and December's intervals above zero.
- The best pair (−0.004%) is weaker than round 1's c40_m70 (−0.013%).
- The network-table preference does not survive the mix with the committee. v52's weights (0.40 / 0.40) stay.
- The member-weight lever is closed: 12 stack-level candidates over two rounds, none within reach of the bar.
