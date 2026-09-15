# Completed ladder campaign after v6

**V6 remains the official best: 277.4508 seconds RMSE on all 344,841 rows.**
All three follow-up batches are complete. They contain **16 new native fits and
64 declared recipes**; no candidate meets the fixed requirement of at least
0.5% lower complete RMSE on each original development fold. There is no new
refit, v7 export or submission.

The [public leaderboard snapshot](leaderboard_after_campaign.json) at
**16:38:49 UTC on 14 September 2026** places v6 **20th of 134 teams**. Third
place is 252.9192 seconds, requiring a further 8.84% reduction from v6 to that
dated score. V6 improved the previous official best, v5, by 14.08%.

| Completed batch | New native fits | Declared recipes | Leading mean-ratio development RMSE, seasonal / forward | Improvement over v6, seasonal / forward |
| --- | ---: | ---: | ---: | ---: |
| [Longer CatBoost and missing-clock experts](README.md) | 8 | 36 | 324.609866 / 222.966982 | 0.0945% / 0.1613% |
| [Covariate-scaled residuals](../ladder_scale/README.md) | 8 | 20 | 322.405169 / 222.687699 | 0.7731% / 0.2863% |
| [Fixed cross-campaign mixtures](../ladder_blend/README.md) | 0 | 8 | 322.588400 / 222.439633 | 0.7167% / 0.3974% |

The recipe closest to meeting both margins is
`missing_ground_v6_75__clock300_missing300_v6_50__left25`. Its seasonal RMSE
is **323.0088876**, a **0.5873%** improvement, and forward RMSE is **222.2824682**,
a **0.4678%** improvement. The reference v6 development scores are
324.9170675 / 223.3271391. The forward gain falls below the declared margin,
so the selection remains null. This descriptive comparison does not change
the selection rule. No further weight interpolation was evaluated.

These scores cover every row of the original seasonal and forward sets,
including signed and extreme labels. They reuse development labels and are
distinct from official leaderboard scores. December remains reserved for the
final audit. No hidden-label probes were made.

All **153 distinct tests pass on Windows and Azure**. Each batch passes both
platform verifiers, including complete scores and native reconstruction with
zero prediction difference. Independent scale reconstruction also has zero
difference. The first two batches' complete RMSE values differ between platforms
by at most 1.14e-13 seconds; the final mixture verification fields agree exactly.
The [aggregate receipt](campaign_summary.json) binds every selection, comparison,
archive and test report.

Only the existing arcw0 worker was used. The sixteen fits total **3,519.03 wall
seconds**, **52,694.39 CPU seconds**, and **11.23 GiB** maximum sampled RSS.
All campaign processes have finished; arcw0 remains running. Models and receipts
are mirrored locally with checked archive hashes. The
[final preservation check](final_submission_integrity.json) confirms unchanged
v1-v6 exports, live submission objects and organizer input ETags.

The completed comparisons give little support for more capacity or fine blend
adjustments on these branches. A future experiment should test new information
about activity during the taxi interval and stand/runway context, with a fresh
feature-availability review and declaration. No such experiment was started by
these three batches.
