# PRC Data Challenge 2026 — vigorous-whistle

Predict taxi-out time at ten European airports. The objective is a top-three
finish. Cloud compute is **Azure only**. The user allocated eight VMs and
authorized starting **only arcw0**; this project uses that existing worker.

Research, phases 1-5, the initial Phase-5R campaign, weather, NM off-block clocks
and the next-lever campaign are complete. The best official submission,
`vigorous-whistle_v7.parquet`, scores **275.0570 seconds RMSE** on all
**344,841 departures**, improving v6 by **2.3938 seconds / 0.8628%**.
The team is **19th of 135** in the 14 September 20:04:02 UTC snapshot, up from
21st in the preceding v6 snapshot. Third place is 252.9192; reaching that dated
score needs another **8.05%**
RMSE reduction. December remains reserved for the final audit.
All 14 supplied Parquet files remain in `data/raw/` (330.22 MB).

- [Post-v7 reporting and geometry audit](docs/source_geometry/README.md):
  fitting-only reporting checks support no additional clock correction. LIRF
  has supported historical parking-to-runway geometry for 98.32-100% of the
  checked departures; a separate model comparison is justified. LFPG's older
  parking-line representations fail the endpoint controls, so exact name
  matches are not treated as known parking points. No model or submission was
  made by this audit; v7 and December's reserved audit are preserved.
- [Next-lever search](docs/next_lever/README.md): **32 new development fits and
  22 fixed recipes** are complete. Applying frozen timeline and surface features
  to the airport models qualifies, scoring **322.8193 seasonal / 221.1382 forward**,
  improvements of **0.6456% / 0.9802%** over v6 development. All ten airports
  improve on both folds. Both platform verifiers and all 196 current tests pass.
  The [selected refit](docs/surface_refit/README.md) is complete at the frozen
  October counts, and both platforms reproduce every ranking prediction exactly.
  **V7 was uploaded once and officially scored 275.0570 seconds.** The
  [official submission receipt](docs/surface_refit/submission_v7_official.json)
  binds the export, remote readback, complete official result and dated standings.
  December's reserved audit remains unused.
- [Supplied arrival taxi-in context](docs/arrivalctx/README.md): all four fits
  and six fixed recipes are complete; none qualifies and every seasonal score
  worsens. Both platform verifiers reproduce every new missing-clock prediction
  exactly. The two [separate joint recipes](docs/nextlever_joint/README.md) also
  fail the seasonal margin. These branches made no refit or export.
- [Flown timeline and off-block traffic](docs/timeline/README.md): eight fits
  and ten recipes are complete. The leading recipe improves development by
  0.4568% / 0.6439%, narrowly missing the seasonal margin. Its 74 new features
  and two combined globals are reused by the separately declared airport
  extension. Both platform feature/native verifiers pass. The original partial
  fit and its provenance-only import correction remain documented.
- [Completed ladder results](docs/ladder/CAMPAIGN_RESULT.md): all three batches
  are complete, with **16 new native fits and 64 declared recipes**. None meets
  the fixed 0.5% improvement requirement on both complete folds. The closest
  recipe improves seasonal/forward RMSE by **0.5873% / 0.4678%**. That ladder
  retained v6 and made no refit or export. All 153 distinct tests and all three
  campaign verifiers pass on Windows and Azure, with exact native predictions.
- [Ladder target-scale follow-up](docs/ladder_scale/README.md): eight LightGBM
  fits and twenty recipes are complete. The leading blend scores 322.405169
  seasonal / 222.687699 forward, improving v6 by 0.7731% / 0.2863%; none qualifies.
  The [final cross-campaign check](docs/ladder_blend/README.md) evaluates eight
  fixed mixtures of the completed branches and also retains v6.
- [Ladder campaign after v6](docs/ladder/README.md): eight fits and 36 declared
  recipes are complete. Five blends improve both folds, but none clears the
  0.5% margin. The leading blend scores 324.609866 seasonal / 222.966982 forward,
  improvements of 0.09455% / 0.16127%. The batch retains v6 without a refit or upload.
- [NM off-block clock experiment](docs/nmclock/README.md): the user-approved
  campaign completed eight runs and twenty candidate comparisons. Selected
  CatBoost/airport residual models score **324.9171 seasonal / 223.3271 forward**,
  improving v5 development by **11.02% / 7.20%**. Both platform verifiers pass
  with exact native predictions, and 117 tests pass per platform. The fixed
  January-November refit is complete and **v6 officially scores 277.4508 on all
  344,841 rows**. It was uploaded once at **14:49:49 UTC on 14 September** and
  processed at 14:50:24 UTC. The [submission receipt](docs/nmclock/submission_v6.json)
  binds the exact export, uploaded bytes, official score and dated standing.
  December remains reserved and v1-v6 are preserved.
- [Phase-5R new-feature results](docs/phase5r/README.md): ten model fits and
  twenty declared recipes tested demand/runway service, cross-fitted taxi
  baselines, target scaling, and missing-record experts. None improved both
  complete folds, so that batch retained v4. The new features improve a matched
  LightGBM baseline modestly; 91 tests pass on both platforms.
- [Surface-weather experiments](docs/weather/README.md): six fits and sixteen
  recipes selected CatBoost with weather plus the frozen airport models.
  Development RMSE is 365.1711 seasonal / 240.6522 forward; official v5 is
  322.9266. There are 102 passing tests per platform, both full verifiers pass,
  and the final native pipeline reproduces ranking samples exactly. Only arcw0
  was used; original inputs, prior models and submissions are preserved.
- [Where to seek substantial gains](docs/competition/history_20260914/README.md):
  14 September review of earlier winners, six full-text reports/papers, selected
  source code, and our remaining error concentration. It proposes a new campaign
  on clock/source handling and ground congestion before the reserved audit;
  the first implementation results are in Phase 5R above.
- [Phase-5 tuning and ensembles](docs/phase5/README.md): 20 development
  evaluations, 28 declared candidate recipes, a selected CatBoost/LightGBM
  ensemble, 81 passing tests on both platforms, and official v4 scoring.
  All model and blend verification checks pass; December remains reserved.
- [Phase-4 models and results](docs/phase4/README.md): reporting-clock features,
  a conditional schedule offset, and a fixed global/airport blend. Fourteen
  development runs, 71 passing tests on both platforms, and a verified official
  v3 score. December stays reserved for the final audit.
- [Phase-3 features and results](docs/phase3/README.md): 16 controlled feature
  experiments, selected timing/traffic features, 61 passing tests on both
  platforms, official v2 score, and Phase 4 priorities. December stays reserved.
- [Phase-2 results and reproduction](docs/phase2/README.md): LightGBM/CatBoost
  comparison, selected refit, official score, 48 passing tests on both
  platforms, and priorities for phase 3.
- [Phase-1 results and reproduction](docs/phase1/README.md): frozen validation,
  25-feature contract, 26 passing tests on both platforms, and measured
  runtime/memory. December remains reserved for the final audit.
- [Azure pilot instructions](infra/azure/README.md): allocated pool
  `arcw0`–`arcw7`, with **only arcw0 authorized to start for now**.

- [Competition brief](docs/competition/BRIEF.md): rules, deadline, prizes,
  verified dataset sizes, team access, current leaders, and next steps.
- [Phase-by-phase project plan](docs/competition/PROJECT_PLAN.md): milestones
  through 10 October, validation priorities, and use of the Azure VMs.
- [Data audit](docs/competition/DATA_AUDIT.md) and
  [observed schema](docs/competition/DATA_SCHEMA.md): actual file contents,
  field availability, label anomalies, and submission alignment.
- [Research references](docs/competition/RESEARCH.md): public approaches,
  previous challenge papers, and evidence limits.
- [Open questions](docs/competition/OPEN_QUESTIONS.md): ambiguities in the
  published information and announcements that require organizer access.
- [Azure compute plan](docs/competition/AZURE_COMPUTE.md): CPU/RAM priorities
  and how to use up to eight VMs for independent experiments.
- [Source index](docs/competition/SOURCES.md): original URLs and local archives.

## Refreshing the evidence

For a new environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-research.txt
```

The current `.venv` uses available system packages plus a local PyArrow install.
Exact versions used in the audit are recorded in
`docs/competition/data_profile.json`.

```powershell
# Public documentation, leaderboard, approved team directory, and references.
.\.venv\Scripts\python.exe -X utf8 -B tools\pull_prc_public_info.py --refresh

# Authorized reads using the PRC keys in .env.
.\.venv\Scripts\python.exe -X utf8 -B tools\prc_data.py inventory
.\.venv\Scripts\python.exe -X utf8 -B tools\prc_data.py download --bucket prc-2026-datasets

# Local checks; no network or model training.
.\.venv\Scripts\python.exe -X utf8 -B tools\profile_prc_data.py
```

Omit `--refresh` to reuse cached public responses; use `--offline` to rebuild
public extracts from the cache. `--section leaderboard --refresh` updates only
the standings. Refreshes update machine-readable snapshots; the dated narrative
brief must be reviewed before treating it as a new snapshot.

Credentials, raw data, and downloaded source archives are ignored by Git.
Their manifests and derived audit reports are retained. The existing
`infra/sky/` files are legacy AWS infrastructure and must not be used for PRC.
