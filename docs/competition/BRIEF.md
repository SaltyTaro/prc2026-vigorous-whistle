# PRC Data Challenge 2026 — competition brief

Researched on **13 September 2026** for team **vigorous-whistle**. The campaign
objective is a **top-three finish**. This brief separates published rules from
facts checked in the current data and from proposed modeling work.

This is the initial research snapshot. Subsequent
[Phase-5 work](../phase5/README.md) produced the current best submission, v4:
326.6895 seconds RMSE on all 344,841 pairs, improving v3's 331.9269 by 1.58%.
At 20:40 UTC on 13 September the team ranked 68th of 131. Use that report and
its dated leaderboard snapshot for current project status.

The official pages, API responses, public repository descriptions, and paper
landing pages are archived locally. All 14 competition files have been
downloaded and profiled. Cloud compute is **Azure only, up to eight VMs**, as
specified by the user. No cloud machines have been provisioned and no competition
submission has been made during this research.

## Competition at a glance

| Item | Confirmed information |
| --- | --- |
| Organizer | EUROCONTROL Performance Review Commission, supported by OpenSky Network |
| Task | Estimate departure taxi-out time in seconds |
| Target | `TAXITIME_SEC_mvt = MVT_TIME_UTC_mvt - BLOCK_TIME_UTC_mvt` for `PHASE_mvt = DEP` |
| Metric | Root Mean Square Error (RMSE); lower is better; each team's best submission counts |
| Training period | January–December 2025 |
| Evaluation period | January and July 2026 |
| Reporting airports | **10**, verified in the supplied files; some website prose says 11 |
| Opens | 1 September 2026 |
| Published closing time | **11 October 2026, 23:59:59 CET** |
| Prize pool | **€5,000 combined for the first three teams**; individual prize shares are not specified |
| Submission allowance | **Five per day**; total team-bucket capacity **1 GB** |
| Team | `vigorous-whistle`; approved; registered 13 September 2026 |
| Data bucket | `prc-2026-datasets` |
| Submission bucket | `prc-2026-vigorous-whistle` |
| Required filename | `vigorous-whistle_vN.parquet`, with an increasing integer version |

Primary sources: [overview](https://prc-data-challenge-2026.netlify.app/),
[data](https://prc-data-challenge-2026.netlify.app/data.html),
[ranking](https://prc-data-challenge-2026.netlify.app/ranking.html),
[eligibility](https://prc-data-challenge-2026.netlify.app/eligibility.html), and
[the team's public page](https://prc-data-challenge-2026.netlify.app/teams/vigorous-whistle.html).
The official [EUROCONTROL-hosted site](https://ansperformance.eu/study/data-challenge/dc2026/)
also carries the challenge.

## What is being estimated

Taxi-out covers the interval from the aircraft leaving its stand to takeoff,
including movement and waiting. In the ranking data, takeoff time is already
supplied. The organizers describe the purpose as **post-operations analysis**:
identifying constrained operations and excess fuel/CO2 relative to unconstrained
conditions. This context matters when judging which observed traffic features
the task supports.

Each row is an airport movement. Departures and arrivals are both present;
only departures require predictions. Movement records, suffixed `_mvt`, have
already been left-joined to Network Manager flight records, suffixed `_flt`.
An arrival's relevant reporting airport is `ADES_mvt`, while a departure's is
`ADEP_mvt`. A flight observed at two challenge airports may therefore contribute
two movement records with different movement IDs.

The supplied information includes airport, runway, stand, aircraft type,
schedule, takeoff/landing timestamps, flight identifiers, market segment,
anonymized operator, wake category, and several Network Manager timestamps.
There are **30 columns** in training and ranking data. The full published field
dictionary is in [the archived data page](pages/data.md); actual Arrow types and
null counts are in [DATA_SCHEMA.md](DATA_SCHEMA.md).

## Verified data inventory

| Dataset | Files | Rows | Departure rows | Compressed size, decimal MB |
| --- | ---: | ---: | ---: | ---: |
| Training: all months of 2025 | 12 | **4,167,797** | **2,085,047** | 284.98 |
| Ranking: January + July 2026 | 1 | **689,534** | **344,841** | 43.56 |
| Submission template | 1 | **344,841** | 344,841 | 1.68 |
| Total | **14** | — | — | **330.22** |

Training has 2,082,750 arrival rows. Ranking has 344,693 arrival rows, all with
their supplied taxi-in targets still populated. The ranking departures have
both `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` entirely blank. Their movement
IDs match the submission template exactly, including order.

The current ranking and template objects were last modified on **4 September
2026**. Their byte sizes differ from the smaller sizes still listed on the
website. Use the downloaded files and [download manifest](download_manifest.json)
as the concrete input version. It records file sizes, source ETags, modification
times, and locally computed SHA-256 hashes. The [bucket inventory](data_inventory.json)
contains the original read-only object listing.

| ICAO | Airport | Training departures | Ranking departures |
| --- | --- | ---: | ---: |
| EDDF | Frankfurt Main | 230,141 | 36,317 |
| EDDM | Munich | 167,334 | 25,966 |
| EGLL | London Heathrow | 239,546 | 39,840 |
| EHAM | Amsterdam Schiphol | 247,951 | 38,182 |
| LEBL | Barcelona–El Prat | 179,705 | 30,080 |
| LEMD | Madrid–Barajas | 212,242 | 36,954 |
| LFPG | Paris Charles de Gaulle | 239,552 | 39,872 |
| LIRF | Rome Fiumicino | 160,704 | 26,899 |
| LSZH | Zürich | 134,907 | 23,150 |
| LTFM | İstanbul | 272,965 | 47,581 |

The organizers removed military, head-of-state, and sensitive movements. They
explicitly warn that real-world movement and flight information can disagree
and that they have not reconciled those inconsistencies.

## Scoring and submission contract

For (n) scored departures, RMSE in seconds is

\[
\operatorname{RMSE}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(\hat y_i-y_i)^2}.
\]

Use `submitting.parquet` as the template and fill `TAXITIME_SEC_mvt`, keeping
the matching `MVT_ID_mvt` values. Upload the result into the assigned team bucket
as, for example, `vigorous-whistle_v1.parquet`. Preserve the template order as
the safest implementation of the row-matching requirement.

The official instructions say mismatched IDs, missing rows, or extra rows
invalidate a submission. The observed template types are `MVT_ID_mvt: double`
and `TAXITIME_SEC_mvt: int32`; the public rules do not separately state whether
floating-point predictions are accepted. A conservative first exporter can
preserve the original types and round predictions to integer seconds, with
finite-value and range checks. This is an implementation choice, not an
additional published rule.

The acceptance email supplied by the user says a result file should appear in
the team bucket after successful processing. The bucket was accessible and empty
when inspected. The public API also reports successful submissions and their
scores. A successful upload alone is not evidence that a submission was scored.

The five-per-day rule does not specify a reset timezone, rolling-window behavior,
or whether failed attempts count. Keep a dated submission ledger and resolve
those details before operating at the limit. No public/private leaderboard split,
separate hidden-final metric, tie-break procedure, or final selection mechanism
beyond "best RMSE" is stated in the pages retrieved.

## Field to beat

Snapshot from responses retrieved **13 September 2026 at 12:53 UTC**, covering all six API pages:
**1,174 successful submissions**, **130 teams with scores**, and **378 approved
teams** in the separate registration API. Every successful submission in this
snapshot reports `usedPairs = 344841`.

| Rank | Team | Best RMSE, seconds | Minutes |
| ---: | --- | ---: | ---: |

The current third-place reference is **261.9189 seconds**. It is a moving
benchmark, not a predicted final qualifying score. There is no score yet for
vigorous-whistle in this snapshot.

The API returns submissions, so taking the first three rows would incorrectly
count repeated submissions from one team. The archived standings group by team
ID and take each team's minimum score. See [the full JSON](leaderboard_snapshot.json),
[team standings CSV](leaderboard_teams.csv), and
[live viewer](https://prc-challenge-2026.vercel.app/).

## Participation, awards, and publication

The [eligibility page](https://prc-data-challenge-2026.netlify.app/eligibility.html)
lists these award conditions:

- All external datasets used must be openly accessible/usable, openly licensed,
  and documented.
- Produced source code must be public on GitHub under **GNU GPLv3**. The
  challenge organization will fork it for organizational purposes.
- Documentation must be sufficient to understand and reproduce the results.
- The solution must be original. Reusing another implementation requires the
  relevant rights and significant modifications; changing only input/output
  handling is expressly insufficient.
- Country/institution sanctions and eligibility restrictions apply.

The overview strongly encourages top teams to publish an open-access JOAS
paper. A separate compulsory paper deadline or paper-based scoring component is
not specified in the award conditions retrieved.

The detailed eligibility page's wording
about non-EUROCONTROL countries is broader than the overview's "open to all"
wording. Record the acceptance, and obtain a definitive organizer answer on
cash-prize eligibility if needed; neither automatic disqualification nor a prize
guarantee follows from these conflicting sentences alone.

The organizers reserve the right to change rules or stop the challenge. Their
ranking page explicitly says they monitor attempts to learn from or exploit the
ranking process. Do not use leaderboard probes to recover hidden targets.

## Main findings for a top-three campaign

1. **Validation and unusual labels deserve early attention.** All 2,085,047
   departure targets are present, but 369 are negative and 202 exceed three
   hours. The maximum is 131,167 seconds. Rome accounts for 188 of the 202
   observations over three hours. Evaluate the complete validation set and
   inspect each airport's squared-error contribution before deciding how to
   handle these regimes.
2. **Use all supplied context deliberately.** Arrivals, runway/stand combinations,
   schedules, aircraft/operator categories, and traffic intensity are natural
   feature families. Network Manager records are available for most departures;
   any special restrictions from organizer announcements still need to be
   checked. Their timestamps are noisy proxies, not exact target values.
3. **Match both season and year shift.** January and July 2025 holdouts can mimic
   evaluation-season composition. Add a forward-time check because those
   seasonal holdouts alone do not reproduce a 2025-to-2026 shift. Keep fitting,
   iteration selection, model selection, and final audit roles explicit.
4. **Start with CPU tabular models.** Establish airport/group means, LightGBM,
   and CatBoost on fixed splits, then test residual models, airport specialists,
   traffic features, and ensembles. These are proposed experiments; no local
   model result exists yet.
5. **Use Azure capacity for independent experiments.** Begin with one measured
   CPU run, then distribute model/split/feature experiments across up to eight
   VMs. The [compute plan](AZURE_COMPUTE.md) prioritizes 32–64 GiB RAM and 8–16
   vCPUs for the pilot, subject to the actual boxes available.

Detailed findings are in [DATA_AUDIT.md](DATA_AUDIT.md). Public methods and prior
challenge papers are catalogued in [RESEARCH.md](RESEARCH.md).

## Remaining information gaps

The public pages
leave the CET/CEST interpretation, prize shares, submission-day reset, and some
final-review details unspecified. Public repositories also disagree about using
particular supplied Network Manager timestamps; none of those repository claims
is an organizer ruling. These points are tracked in
[OPEN_QUESTIONS.md](OPEN_QUESTIONS.md), with no administrator contact sent.
