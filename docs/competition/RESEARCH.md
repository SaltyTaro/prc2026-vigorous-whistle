# Research references and public approaches

Collected on **13 September 2026**. This is a review of public documentation;
the implementations below have not been independently reproduced in this
workspace. No participant code has been executed or incorporated into a model.

**14 September follow-up:** [Where to seek substantial gains](history_20260914/README.md)
reviews the earlier winners in detail, archives six full-text reports/papers and
selected files from seven repository snapshots, and connects their methods to
our Phase-5 error concentration. It supersedes the initial review's limited
previous-edition coverage; the dated 2026 leaderboard table below is unchanged.

The official website describes the task and fields but does not link a 2026
starter model on its five substantive pages. A public GitHub repository search
returned 14 relevant repositories, including the official website and leaderboard
viewer. The search result and README snapshots are archived; this is not a
guarantee that every public or private implementation has been found.

## Current public implementations

The scores below come from our complete **12:53 UTC official API snapshot**,
not from unverified README claims. They are best scores per team on 344,841
pairs. A repository can describe a different version from its team's best
submission, so the table does not establish that each published recipe exactly
reproduces the listed score.

| Repository / team | Publicly described approach | Official snapshot | Evidence limits |
| --- | --- | --- | --- |

Additional repositories include [victoralcadi](https://github.com/victoralcadi/prc-data-challenge-2026),
[ahmetabdullahgultekin](https://github.com/ahmetabdullahgultekin/prc-taxiout-2026),
[fenil264](https://github.com/fenil264/prc-2026-taxi-out), and preparation repositories
from [Enzoferrariz](https://github.com/Enzoferrariz/PRC-Data-Challenge-2026) and
[Necklacesaura](https://github.com/Necklacesaura/PRC-Data-Challenge-2026).
Repository metadata, declared licenses, and local README paths are recorded in
[public_repositories.json](public_repositories.json).

We have not identified and verified a public
solution repository for any of those three. Do not attribute the above public
methods to the leaders or infer that a particular algorithm guarantees top three.

## Findings worth testing independently

Several public approaches converge on boosted trees, airport/stand/runway
structure, calendar and schedule deltas, arrival/departure traffic context,
separate handling of missing flight records, and residual correction of observed
Network Manager timestamps. Our own data audit confirms substantial NM coverage
and airport-dependent discrepancies between its timestamps and airport-reported
block times. Specific field-use interpretations remain subject to the official
rules and announcements; see [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md).

LIRF's unusually long labels deserve explicit analysis before large model
searches. They dominate the count of labels over three hours in our audit.
That supports a dedicated validation slice, not copying another team's hand-tuned
rules. Any specialist must show gains on complete, fixed validation periods.

Public reports often show large differences between January/July local RMSE,
December RMSE, and official 2026 RMSE. Validation period, label inclusion,
training-data version, feature availability, and model-selection reuse must
accompany every score comparison. A filtered "clean" score cannot replace the
official all-departure objective.

For a first campaign, compare a group-mean baseline with LightGBM and CatBoost,
then perform bounded feature-family ablations and residual/specialist experiments.
Use fixed seasonal and forward-time checks before allocating many Azure workers.
This plan is based on the task structure and the reviewed reports, not a result
already measured for vigorous-whistle.

## Relevant earlier PRC work

The official 2026 overview links these papers. Their tasks and score units differ
from taxi-out prediction; use them for methods and reporting practice.

| Paper | Relevance |
| --- | --- |
| [Open Machine Learning Models for Actual Takeoff Weight Prediction](https://journals.open.tudelft.nl/joas/article/view/7963), JOAS, 2025 | Operational feature engineering and CatBoost/LightGBM/XGBoost/neural-network ensembles on open aviation data |
| [A Physics-Guided Gradient Boosting Framework for Open-Source Aviation Fuel Estimation](https://journals.open.tudelft.nl/joas/article/view/8764), JOAS, 2026 | Aviation-domain features, weather/infrastructure enrichment, feature selection, and gradient boosting |
| [Aircraft Fuel Burn Estimation: The EUROCONTROL PRC 2025 Data Challenge](https://journals.open.tudelft.nl/joas/article/view/8750) | Previous competition's dataset and evaluation documentation |
| [Aircraft Takeoff Weight Estimation: The EUROCONTROL PRC 2024 Data Challenge](https://journals.open.tudelft.nl/joas/article/view/8252) | Previous data-release context and open-science approach |

The [2024](https://github.com/prc-data-challenge-2024) and
[2025](https://github.com/prc-data-challenge-2025) GitHub organizations contain
published team solutions. The initial review archived their repository metadata
and four paper landing pages. The [14 September review](history_20260914/README.md)
adds full PDFs, the 2025 winning report, a further August 2026 paper, and selected
source files at exact commits. Participant models have not been reproduced.

## Candidate external data

External data is an experiment to justify with validation, not a prerequisite
for the first baseline. These source pages were retrieved; raw external datasets
have not been downloaded for modeling.

| Source | Potential signal | Verified source information / remaining checks |
| --- | --- | --- |
| [Iowa Environmental Mesonet airport weather archive](https://mesonet.agron.iastate.edu/request/download.phtml) | METAR wind, visibility, precipitation, freezing/deicing conditions; runway-relative wind | Page documents an airport weather archive and warns of limited quality control. Check coverage for each airport/month, station mapping, timestamp alignment, and redistribution terms before adoption. |
| [OurAirports downloads](https://ourairports.com/data/) | Airport/runway coordinates, runway headings and geometry | The data page explicitly releases its data to the **public domain**. Validate identifiers, completeness, and the date/version used. It does not supply complete stand/taxiway geometry. |
| [OpenStreetMap](https://www.openstreetmap.org/copyright) | Stands, taxiways, graph distances between stands and runways | Database is available under **ODbL** with attribution and applicable share-alike requirements. Check airport coverage and record provenance of derived data. |
| [EUROCONTROL taxi-out dashboard](https://ansperformance.eu/efficiency/taxiout/) | Domain context on additional taxi-out time and airport variation | The retrieved page embeds a Power BI dashboard. The static page is not a downloaded flight-level dataset or a validated feature source. |

Public teams also describe ATFM and OpenSky/OPDI enrichments. Their value,
2025/2026 coverage, access conditions, and applicable licenses have not been
independently checked here. Evaluate those only after establishing a strong
baseline on the supplied data.

Originality, GPLv3 publication, and documented openly usable external data are
award conditions. Public availability of a repository or model checkpoint alone
does not establish compatibility with those conditions.
