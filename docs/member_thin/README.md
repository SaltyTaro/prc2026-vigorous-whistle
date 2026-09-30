# v52's member weights on thinned coverage (30 September – 1 October 2026): the rule fires, but not because of coverage

Declaration [`configs/member_weights_thin_v1.json`](../../configs/member_weights_thin_v1.json) (16:02 UTC 30 September, 96b1f72,
before any thinned prediction). Tools: `tools/member_thin_predict.py`, `tools/member_thin_report.py` (6f65643). Report:
[`report.json`](report.json). Workstation CPU only; nothing exported or uploaded; December 2025 labels only.

## Question and scope

v52's member weights (combination 0.40, masked 0.40) were chosen on 2025's natural adsb.lol coverage. Do they still hold under the
scored months' coverage? This reads the question at the network-table level, on December 2025 as it is and thinned to January 2026's
coverage (coverage-regime attempt a13: EDDF only).

- **July could not be read.** The base networks' July leave-one-month-out models kept predictions but not their model files.
- **Models:** the refit (January–November) models of the base networks (e24, e24b, e32; 48) and of the members (dl32, m25dl32,
  m25_32; 48). Each was predicted on the workstation CPU with natural and with thinned inputs.
- **Checks passed before any label was read:**
  - every natural prediction reproduces the run's own December predictions (CPU-trained: RMS at most 0.0001 s; GPU-trained: at most
    0.074 s);
  - the natural base table reproduces the development table's December column (RMS 0.009 s).
- **Grid:** combination and masked weight in {0.25, 0.40, 0.55, 0.70}; control 0.40 / 0.40.

## Result

Network-table RMSE, December non-LIRF rows (153,609; EDDF 17,324). Change against the control in brackets.

| weights | natural | thinned | thinned EDDF | thinned interval (s) |
|---|---:|---:|---:|---|
| c25_m70 | 146.541 (−0.171%) | 149.019 (−0.172%) | −0.358% | [−0.383, −0.135] |
| c25_m55 | 146.588 (−0.139%) | 149.068 (−0.139%) | −0.282% | [−0.302, −0.117] |
| c40_m70 | 146.661 (−0.089%) | 149.139 (−0.091%) | −0.147% | [−0.242, −0.039] |
| c25_m40 | 146.665 (−0.087%) | 149.150 (−0.084%) | −0.179% | [−0.204, −0.050] |
| c40_m55 | 146.710 (−0.056%) | 149.191 (−0.057%) | −0.083% | [−0.140, −0.034] |
| **c40_m40 (v52)** | 146.792 | 149.275 | | control |
| c70_m25 (worst) | 147.428 (+0.433%) | 149.910 (+0.425%) | +0.612% | [+0.408, +0.890] |

- **By the declared rule five candidates "shift"** (at least 0.05% better on thinned inputs, interval below zero). The lowest thinned
  RMSE, **c25_m70**, is carried forward to a declared follow-up; nothing is exported.
- **The coverage regime changes nothing.** Every candidate's change on thinned inputs equals its change on natural inputs to within
  0.008 points (c25_m70: −0.172% against −0.171%). What the rule picked up is a preference of the December network table that exists
  on natural inputs as well. The declared rule tested the thinned gain alone. It did not ask for the gain to exceed the natural one, so
  it cannot tell the two apart.
- **Where the preference comes from.**
  - Matched rows: at the table level the combination member makes December slightly worse. The table alone reads 112.62 s; with the
    member at 0.40 it reads 113.01 s, and at 0.25 it reads 112.79 s.
  - Unmatched rows: more weight on the masked member helps (204.43 → 204.11 s at 0.70).
- **The table level is not where the weights were chosen.**
  - The combination round selected 0.40 over 0.25 at the stack level, where the table mixes with the committee at share 0.85 and runs
    through the layers: January −0.066%, July −0.071%, December −0.119% at 0.40.
  - The later stack-level member-weight round (grid 0.40 / 0.55 / 0.70, no 0.25) found its best, c40_m70, at only −0.013% (null).
  - Both levels agree on the direction: the masked weight a little higher, the combination weight not higher.

## What follows

The declared follow-up is a stack-level reading. Since the effect is not a coverage effect, re-predicting the committee on thinned
inputs adds nothing. The decisive reading is the usual January / July / November / December stack-level one
(`tools/run_member_weights.py`) with 0.25 added to the combination grid.

- It needs no fit: the member tables for the four months exist, and it takes minutes of CPU.
- It must be declared as adaptive: its grid comes from this reading.
- Expected gain is small, of the order of the member-weight round's −0.013%.
- Whether to run it is the user's decision.
