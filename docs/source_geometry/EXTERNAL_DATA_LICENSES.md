# External source attribution

The geometry study uses **copyright OpenStreetMap contributors** data, made
available under the [Open Database License 1.0](https://opendatacommons.org/licenses/odbl/1-0/).
See the [OpenStreetMap copyright page](https://www.openstreetmap.org/copyright).
The exact copyright page and license text are archived with hashes in the
[source inventory](source_inventory.json).

The raw map extracts under `data/external/osm/source_geometry_v1/` and the
derived `*_locations.json` and `parking_endpoint_controls.json` files are
identified as ODbL map data. The separate map-data archive retains those paths,
source receipts, this notice and the license/source documents. A derived
parking endpoint is a documented interpretation of OSM geometry, not a new
surveyed location. Retain attribution, access to the license and applicable
database share-alike terms when distributing these databases.

The parking-tag guide at
<https://wiki.openstreetmap.org/wiki/Tag:aeroway%3Dparking_position> is
documentation from OpenStreetMap Wiki contributors. Its archived footer links
to [CC BY-SA 2.0](https://creativecommons.org/licenses/by-sa/2.0/). The generic
map-license wording in the original download receipt does not replace the
wiki content license. The guide explains that an explicitly mapped node marks
the nose-wheel position, and that a parking way is usually drawn from the
taxiway towards the parking position. The endpoint audit tests that convention
against mapped parking nodes before using it.

Project implementation code remains under the repository's GPLv3 license.
That license does not relicense the separately attributed external map data
or documentation. Figures carry OpenStreetMap attribution. The source/evidence
archive is an extension of the frozen competition project; its map source
and derived database records are delivered in a separate archive.

No external airport dataset was copied from another competition team. Public
Overpass responses were downloaded sequentially, with bounded requests and
preserved failure records. No additional cloud compute was used.
