# PRC Data Challenge 2026 — vigorous-whistle

Predict taxi-out time at ten European airports. The objective is a top-three
finish. Cloud compute is **Azure only, with at most eight VMs**, per the user's
instruction on 13 September 2026.

Research, phases 1-5, the initial Phase-5R campaign, weather, and NM off-block
clock campaigns are complete. The best official submission, `vigorous-whistle_v6.parquet`,
scores **277.4508 seconds RMSE** on all **344,841 departures**, improving v5 by
**14.08%**. The team is **20th of 134** in the 14 September 14:52:58 UTC snapshot.
Third place is 252.9192; reaching that dated score needs another **8.84%**
RMSE reduction. December remains reserved for the final audit.
All 14 supplied Parquet files remain in `data/raw/` (330.22 MB).

- [Ladder target-scale follow-up](docs/ladder_scale/README.md): separately
  declared eight LightGBM fits and twenty fixed recipes, running on arcw0 after
  the first ladder's verified decision. It scales residuals using supplied clock
  covariates while retaining squared error in seconds and the same protected folds.
  All 149 tests pass on Windows and Azure.
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
