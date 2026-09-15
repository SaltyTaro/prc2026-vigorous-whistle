# Supplied arrival taxi-in context

This separate experiment was declared at **17:48:33 UTC on 14 September 2026**,
after the [fitting-only signal audit](../next_lever/arrival_context_audit.json)
and before any new arrival-context model score. It tests **four native fits
and six fixed recipes** on the two original complete development folds.
Training is queued behind the timeline experiment on the existing arcw0.
December remains reserved; this batch does not submit a file.

The [declaration](../../configs/arrivalctx_v1.json) adds fourteen features to
the original 232-feature ground/weather contract. Two missing-clock experts
use exactly the earlier ladder CatBoost and regularized LightGBM settings,
including their conditional LIRF schedule offset and October stopping rule.
The candidate families are the new LightGBM, the new CatBoost, and their equal
mean; each receives 50% or 100% weight against v6 on missing-clock rows.
Every clock-present row retains v6 exactly. All six recipes must complete;
promotion requires at least 0.5% lower complete RMSE than v6 on both folds.
There are no new mixtures with the timeline or earlier ladder candidates.

The ranking data explicitly retain arrival taxi-in values and arrival block
timestamps. The [availability declaration](../next_lever/arrival_context_audit_declaration.json)
uses only supplied arrivals completed strictly before each departure query,
where completion is the later of landing and arrival block time. Every
occurrence of the focal flight ID is excluded. No departure label or departure
airport block time enters the context projection. Actual UTC December movement
context is excluded. The task is retrospective; publication times are not
available and no online availability claim is made.

The features summarize airport taxi-in counts, means and anomaly fractions
over one and three hours, plus normal taxi-in means and counts at the same
runway over three hours and at the same stand over seven days. A separate
normal mean uses arrival durations from zero through three hours; raw means
and anomaly flags remain available. No departure label or prediction is clipped.
Missing runway/stand identifiers never form a pooled group.

The exploratory audit used 111,177 common fitting departures, including all
12,179 missing-clock rows in that intersection. Within-airport correlations
between the raw offset residual and stand/runway taxi-in means suggest a
signal, but they do not establish an improvement over v6. Stand mean context
is available for 342,148 of 344,841 ranking departures; exact runway mean context
is available for 90,911. Missingness is retained and all rows are scored.

Source is isolated in `src/prc2026_arrivalctx/`. The fresh bundle is
`data/arrivalctx/ad97a0d16cb9cec64782`, full ID
`ad97a0d16cb9cec6478255c8e133e4b7996fc5f541f95fdf4d8fee53dacd33ab`.
It references the unchanged 261-feature NM parent and stores fourteen new
columns in an ID-aligned table, giving 275 available shared features and 246
model predictors. All **2,264,211** inherited rows remain. Windows preparation
took **12.50 seconds** and **1.78 GiB** sampled peak RSS; all native competition
fits run only on arcw0. This representation avoids another full copy of the
parent feature files on the VM.

The [local feature verifier](local_feature_verification.json) checks every
parent feature-file hash, all feature identities, 896 independent direct-scan
values, and exact reproduction of 6,384,252 values from the earlier audit table.
Every difference is zero. All **176 tests pass locally**, including both native
learner fit/provenance/reload paths and strict arrival availability boundaries.
Azure repeats the tests and feature verification before fitting.

Feature-code SHA-256:
`2811bc3bbd5fdefeb0b641f41c3af9f954a4661cf7dc88fccd24b211b24c3623`.
Model-code SHA-256:
`af910d80897bd0f07b5e5a312471ecbfdd3803b825f67d5b0736f96f730a1a13`.
Keep feature code frozen once this bundle exists and model code frozen once
completed runs exist. Earlier packages, declarations and predictions are unchanged.

`tools/batch_arrivalctx.py` waits for the earlier timeline job to finish and
verify, then runs the complete declaration, selection, native verification and
summary. New runs use `runs/arrivalctx/`; the Windows mirror uses
`runs/azure_arcw0_arrivalctx/`. The full verifier checks every new prediction
on the missing-clock scoring rows and exact preservation of the clock-present
v6 predictions. A successful development result still requires a reproducible
refit and verification before any export or official submission.
