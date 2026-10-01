# Learner-family design (1 October 2026): no GO, closed

Declaration `configs/learner_family_design_v1.json` (db188be), before any fit. October design fold only; no scoring month, ranking row or
official score read. 16 fits (seeds 1-4 of `ref_lgb`, `dart`, `linear_tree`, `huber_tail`) on arcw0-arcw7 (`tools/launch_learner_family.sh`,
`tools/run_learner_family_design.py`; report `tools/learner_family_report.py`, `design_report.json`). Workers deallocated after the mirror.

October non-LIRF RMSE: committee 159.04, network pool 154.02, control table proxy 176.26. Variant four-seed mean in the committee slot at
c = 0.5, share 0.75 (R2 = change against the table without a new member; positive = worse):

| variant | standalone RMSE | vs ref_lgb | R2 at c=0.5 | gain vs ref_lgb | interval vs ref_lgb (s) | decision |
|---|---:|---:|---:|---:|---|---|
| ref_lgb (control) | 167.16 | | +0.141% | | | |
| dart | 171.15 | +2.38% | +0.233% | −0.092 pts | [+0.08, +0.25] | no |
| linear_tree | 203.90 | +21.98% | +0.275% | −0.134 pts | [−0.14, +0.73] | no |
| huber_tail | 168.13 | +0.58% | +0.165% | −0.024 pts | [−0.01, +0.11] | no |

Every variant is worse than the same-family control; even the control in the committee slot is worse than the committee (+0.14%).
Different inductive biases (DART, linear trees, Huber tail) add no diversity that the trees and networks do not already hold.
**Learner-family lever closed.** Nothing exported or uploaded.
