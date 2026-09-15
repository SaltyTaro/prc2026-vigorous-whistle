# Surface-weather input coverage

The Iowa Environmental Mesonet (IEM) download and coverage audit completed on
14 September 2026 UTC. This is preparation for a separately declared weather
experiment. **No weather model was trained or scored in the initial Phase-5R
batch.** No departure labels were read by this coverage audit, and December's
audit remains unused.

The cache has **30 source CSVs, 190,466 observations and 18,435,163 bytes**:
January-November 2025 and January/July 2026 for each of the ten competition
airports. All expected days have at least one observation at every station.
The fields include temperature/dew point, wind, visibility, pressure, cloud
layers and reported weather codes. Row coverage does not imply every field is
present: for example, sea-level pressure is absent in these downloads, and
wind direction is missing in some reports. The next feature contract must
handle each variable's missingness and units explicitly.

For a conservative coverage check, use the last observation **at least 15
minutes before takeoff**, allowing a maximum observation age of **90 minutes**.
The provider does not supply publication timestamps in these exports, so this
delay is an explicit convention, not proof of exact live availability.

| Population | Departures | With a matching report | Coverage |
| --- | ---: | ---: | ---: |
| January-November training | 1,919,370 | 1,919,184 | 99.9903% |
| Ranking | 344,841 | 344,822 | 99.9945% |

Every departure is counted, including those without a matching report. No
complete validation set may be reduced to the weather-covered subset.

IEM's [official usage statement](https://mesonet.agron.iastate.edu/disclaimer.php)
places its website materials in the **public domain**, with attribution
appreciated. A saved copy and its hash accompany [inventory.json](inventory.json).
The full source URLs, retrieval times, CSV hashes, per-variable availability,
and daily coverage are recorded there. [coverage.json](coverage.json) binds
the matching calculation to the original competition input hashes.
[verification.json](verification.json) independently rechecks all 30 cached
objects, metadata, license and coverage receipt without new network requests.

The first small probe used concurrent requests and encountered HTTP 429
responses. It is preserved as [probe_initial.json](probe_initial.json). The
reader now uses one request at a time, a delay, and bounded Retry-After handling.
The completed [probe.json](probe.json) covers all ten stations; the full download
then completed with no failed objects. Do not restore concurrent request load.

```powershell
.\.venv\Scripts\python.exe -X utf8 -B tools/pull_prc_weather.py --offline
.\.venv\Scripts\python.exe -X utf8 -B tools/audit_prc_weather.py
```

Raw observations and individual download receipts are cached under the ignored
`data/external/iem/` directory. They do not enter the frozen initial Phase-5R
feature bundle, which still declares no external data.

The subsequent [weather-model campaign](../../weather/README.md) was declared
on 14 September at 01:59 UTC. It preserves this original inventory and coverage
receipt. The feature audit found that `p01i` is zero in every downloaded report;
models omit it and use precipitation weather codes instead. The additional
[OurAirports source receipt](runway_sources.json) records public-domain runway
coordinates and usage evidence for runway-relative wind features.

The commands above recreate the inventory and its creation timestamp. Do not
run them over a frozen experiment's inputs without intentionally creating a
new experiment identity. `tools/verify_weather.py` verifies the saved inputs
offline without rewriting their inventories.
