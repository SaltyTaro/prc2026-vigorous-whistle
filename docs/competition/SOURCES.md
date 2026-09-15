# Source index

Collected on **13 September 2026**. Individual retrieval times, local archive
paths, byte sizes, and SHA-256 hashes are in [source_manifest.json](source_manifest.json).
Raw website/API/repository responses are under `sources/` and are ignored by Git;
the pull script recreates them. Clean readable extracts are under `pages/`.

The **14 September previous-editions review** has a separate
[source manifest](history_20260914/source_manifest.json),
[request register](history_20260914/requests.json), and
[research report](history_20260914/README.md). Its 65 source objects include six
full-text PDFs and selected source files from seven exact repository snapshots.
It preserves this initial collection and its dated leaderboard evidence.

## Official competition material

| Topic | Original source | Local readable copy |
| --- | --- | --- |
| Overview, dates, prize pool | [Netlify overview](https://prc-data-challenge-2026.netlify.app/) | [index.md](pages/index.md) |
| Purpose and retrospective use | [Rationale](https://prc-data-challenge-2026.netlify.app/rationale.html) | [rationale.md](pages/rationale.md) |
| Files and field dictionary | [Data](https://prc-data-challenge-2026.netlify.app/data.html) | [data.md](pages/data.md) |
| Scoring and submissions | [Ranking](https://prc-data-challenge-2026.netlify.app/ranking.html) | [ranking.md](pages/ranking.md) |
| Participation and award conditions | [Eligibility](https://prc-data-challenge-2026.netlify.app/eligibility.html) | [eligibility.md](pages/eligibility.md) |
| Team registration directory | [Teams](https://prc-data-challenge-2026.netlify.app/teams/index.html) | [team directory](pages/teams/index.md) |
| Our team | [vigorous-whistle](https://prc-data-challenge-2026.netlify.app/teams/vigorous-whistle.html) | [team page](pages/teams/vigorous-whistle.md) |
| Site search index | [search.json](https://prc-data-challenge-2026.netlify.app/search.json) | `sources/site/search.json` |
| Source repository | [euctrl-pru website repository](https://github.com/euctrl-pru/prc_data_challenge_website_2026) | `sources/official_source/` and repository-tree JSON |
| EUROCONTROL-hosted overview | [Overview mirror](https://ansperformance.eu/study/data-challenge/dc2026/) | [mirror overview](pages/references/eurocontrol_2026_mirror.md) |
| EUROCONTROL-hosted rules | [Ranking mirror](https://ansperformance.eu/study/data-challenge/dc2026/ranking.html) | [mirror ranking](pages/references/eurocontrol_2026_ranking.md) |
| EUROCONTROL-hosted data page | [Data mirror](https://ansperformance.eu/study/data-challenge/dc2026/data.html) | [mirror data](pages/references/eurocontrol_2026_data.md) |
| EUROCONTROL-hosted eligibility | [Eligibility mirror](https://ansperformance.eu/study/data-challenge/dc2026/eligibility.html) | [mirror eligibility](pages/references/eurocontrol_2026_eligibility.md) |
| Storage instructions linked by 2026 data page | [2024 MinIO instructions](https://ansperformance.eu/study/data-challenge/dc2024/data.html#using-minio-client) | [archived page](pages/references/minio_instructions_2024.md) |

The site's search index and repository tree identify the five substantive
non-team pages: overview, rationale, data, eligibility, and ranking. All five
were retrieved. Individual other-team profile pages were not separately crawled;
the complete approved team directory was retrieved through the public API.

## Official APIs and live standings

Competition ID: `bb3693e1-26bc-4a9e-8619-4fe78b4eab0c`.

| Resource | URL | Local result |
| --- | --- | --- |
| Successful submissions | [Leaderboard API](https://datacomp.opensky-network.org/api/competitions/bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/leaderboard) | [full snapshot](leaderboard_snapshot.json), [best score per team](leaderboard_teams.csv) |
| Approved teams | [Teams API](https://datacomp.opensky-network.org/api/competitions/bb3693e1-26bc-4a9e-8619-4fe78b4eab0c/teams) | [teams snapshot](teams_snapshot.json) |
| API specification | [OpenAPI](https://datacomp.opensky-network.org/api/openapi?format=json), [Swagger UI](https://datacomp.opensky-network.org/api/swagger-ui/) | `sources/api/openapi.json`, `sources/api/swagger.html` |
| Website-linked community viewer | [Live leaderboard](https://prc-challenge-2026.vercel.app/) | `sources/references/leaderboard_app.html` |

Leaderboard pagination uses `limit=200` and each returned `nextCursor` in the
next `cursor` parameter. All six pages were fetched. Team pagination uses
`size=100`, one-based `page`, and `participants=false`; all four pages were
fetched. Participants' personal details were not requested from the API.

## Authorized storage inspection

Storage endpoint: `https://s3.opensky-network.org`.

- [data_inventory.json](data_inventory.json): accessible dataset and team buckets,
  object names, source ETags, sizes, and modification times.
- [download_manifest.json](download_manifest.json): all 14 downloaded files and
  their local SHA-256 hashes.
- [data_profile.json](data_profile.json): actual schema, counts, distributions,
  nulls, ID alignment, and timestamp checks.

Authentication used only the PRC access and secret keys supplied in the local
`.env`. Key values are absent from the reports. Both the dataset bucket and the
team's empty submission bucket were read; no remote objects were changed.

## Research and external-data references

[RESEARCH.md](RESEARCH.md) links the individual repositories and explains their
evidence limits. [public_repositories.json](public_repositories.json) records the
14 matching public repository results, declared licenses, and archived README
locations. The original GitHub search and previous-edition organization listings
are also in `sources/references/`.

| Reference | Local readable copy |
| --- | --- |
| [JOAS takeoff-weight models, article 7963](https://journals.open.tudelft.nl/joas/article/view/7963) | [paper page](pages/references/paper_7963.md) |
| [JOAS physics-guided fuel estimation, article 8764](https://journals.open.tudelft.nl/joas/article/view/8764) | [paper page](pages/references/paper_8764.md) |
| [PRC 2025 data paper, article 8750](https://journals.open.tudelft.nl/joas/article/view/8750) | [paper page](pages/references/paper_8750.md) |
| [PRC 2024 data paper, article 8252](https://journals.open.tudelft.nl/joas/article/view/8252) | [paper page](pages/references/paper_8252.md) |
| [IEM airport weather](https://mesonet.agron.iastate.edu/request/download.phtml) | [source page](pages/references/external_iem_metar.md) |
| [OurAirports data](https://ourairports.com/data/) | [source page](pages/references/external_ourairports.md) |
| [OpenStreetMap license](https://www.openstreetmap.org/copyright) | [license page](pages/references/external_osm_license.md) |
| [EUROCONTROL taxi-out dashboard](https://ansperformance.eu/efficiency/taxiout/) | [static page](pages/references/taxiout_methodology.md) |

On 14 September UTC, the separate [surface-weather coverage study](weather_20260914/README.md)
acquired 30 IEM CSV exports covering the ten airports in January-November 2025
and January/July 2026. [Its inventory](weather_20260914/inventory.json) records
every exact request URL, retrieval time, input hash, row count, and the archived
[public-domain usage statement](https://mesonet.agron.iastate.edu/disclaimer.php).
[Coverage](weather_20260914/coverage.json) and [offline verification](weather_20260914/verification.json)
bind those observations to the competition population without reading departure
labels. The separately declared [weather experiment](../weather/README.md) uses
these inputs; they are absent from the initial Phase-5R models and official v4.
[OurAirports source receipts](weather_20260914/runway_sources.json) additionally
bind the runway-coordinate snapshot and archived public-domain usage page used
for geodesic runway-relative wind features. The frozen source manifests are
verified without rewriting them by the new weather tools.

## Information supplied by the user or not retrieved

The acceptance-email text supplied in the conversation is the source for the
team's bucket assignment, filename example, SSO login instructions, and contact
email `challenge@opensky-network.org`. Public API status and authorized storage
access independently confirmed that the team is approved and the bucket exists.
For browser access, the email instructs: **Other Authentication Methods → Login
with SSO → OpenSky credentials in Keycloak**.

Authenticated channel posts, private emails
beyond the supplied acceptance text were not retrieved. Full article PDFs and
selected older participant source files were added in the separate 14 September
review linked above; the initial collection covered README snapshots. Unknowns are listed in
[OPEN_QUESTIONS.md](OPEN_QUESTIONS.md).

The Netlify `sitemap.xml` endpoint returned HTTP 404. Page coverage was instead
checked using the navigation links, Quarto search index, and official repository
tree. This missing sitemap did not prevent retrieving the substantive pages.
