# Two fixed combinations of the new feature routes

Both fixed recipes are complete and **neither qualifies**. The 50% combination
scores **325.63629171756054 seasonal / 222.62731781120414 forward**; the 100%
combination scores **328.9348500679548 / 223.5125444295747**. Both worsen the
seasonal fold. No refit or export was made from this declaration.

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
pass on both platforms as part of the recorded test suites.

The [complete results](development_results.json), [Azure verification](verification.json)
and [Windows verification](local_verification.json) are final. Both platforms
reconstruct all 1,013,502 saved prediction rows with zero numerical difference.
The local mirror is `runs/azure_arcw0_nextlever_joint/`; native models remain
in their original timeline and arrival-context runs.

Selection SHA-256:
`8eb8161938ce6cc9379d5680437505a623ec9085e22900b264b0fad099e0ea55`.
The [development mirror](development_mirror.json) restores all ten files from
an 11,457,970-byte archive, SHA-256
`3e309dec5f97a82bccf77f2695568f5f7f3615e346e96c17ae575d02fd2b299c`.
