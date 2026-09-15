# Two fixed combinations of the new feature routes

This separate declaration at **18:01:34 UTC on 14 September 2026** adds no native
training. It combines two predeclared parent families after all twelve parent
fits and both native verifications finish. The [configuration](../../configs/nextlever_joint_v1.json)
records that seven individual timeline component scores and the fitting-only
arrival audit had been inspected; the complete timeline selection and every
arrival-model score were still unavailable.

For clock-present departures, use `combined_mean_v6_100`: 25% timeline LightGBM,
25% timeline CatBoost, and 50% original v6 airport models. For missing-clock
departures, use `arrival_mean_v6_100`: 50% arrival-context LightGBM and 50%
arrival-context CatBoost. The **two candidates** use 50% or 100% of this combined
prediction against complete v6. These weights apply equally to all airports and
both original folds; no further interpolation is included.

This declaration is separate from each parent's feature comparison. Neither
parent declaration, selected recipe nor prediction is changed. A parent need
not qualify individually, because the two experiments affect complementary
groups. The joint recipe itself must lower complete RMSE by at least 0.5% on
**both** original folds. If both qualify, choose the lower mean relative RMSE.
December remains reserved and no submission is automatic.

`tools/nextlever_joint.py` checks every parent run's artifact hashes, both complete
verification reports, identical v6 references, exact protected scoring IDs and
the declared family weights. It reconstructs both source recipes, checks their
complete saved metrics, then forms the two disjoint-route predictions. It saves
all four complete prediction populations. Verification recomputes all metrics
and requires byte-equivalent numerical predictions. The two route/weight tests
pass locally; the arcw0 batch will include them in its full test run.

Results will be saved in `docs/nextlever_joint/` and `runs/nextlever_joint/`.
Native models remain in their original timeline and arrival-context runs. A
qualifying result would still need a reproducible refit before export.
