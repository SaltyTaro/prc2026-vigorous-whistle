# Rome geometry experiment against v7

Declared at **01:38:03 UTC on 15 September 2026**, after the completed
[source and geometry audit](../source_geometry/README.md). This separate
campaign tests historical LIRF geometry at the unchanged v7 airport-model
settings. V7 remains the official best, **275.0570 seconds RMSE**. December
remains reserved. Only the existing arcw0 is authorized for model training.

The [feature declaration](../../configs/features_rome_geometry_v1.json) assigns
January 2025 maps to January-June movements, July 2025 maps to July-November,
and the matching January/July 2026 maps to ranking departures. It adds 15 route
features and 15 route-sharing traffic features. No departure label or airport
block time enters shared feature generation. OSM data and derived routes retain
OpenStreetMap contributor attribution and ODbL 1.0; see the
[source license notice](../source_geometry/EXTERNAL_DATA_LICENSES.md).

The [model declaration](../../configs/rome_geometry_v1.json) fixes six native
fits: a reproduction control, geometry, and geometry plus queue context on
each original development fold. Settings match v7's 31-leaf LIRF LightGBM,
including its signed NM residual target and October stopping rule. All original
clock-present LIRF fitting rows remain included, even with missing geometry.

Four complete candidates use 50% or 100% replacement weight for each new
variant within v7's existing airport half. Only LIRF scoring rows with a clock
and supported connected route can change. Every other value stays exactly at
v7. Selection must finish all six fits, reproduce both controls, and lower
complete RMSE by at least 0.5% on **both** original folds. No individual scoring
row, airport-specific tuning, clipping, or extra worker is introduced.

Routes are undirected geometric minima to any mapped access on the recorded
runway. Shared-route traffic uses strictly earlier other-flight pushes and
takeoffs; the focal movement and all peers with its nonmissing flight ID are
excluded. Active occupancy follows the existing retrospective NM-clock
convention and is not a measured queue order or actual taxi clearance.

The extension stores only new LIRF columns and binds the complete unchanged
parent feature bundle by hash. It avoids duplicating the 276 inherited model
predictors on arcw0's limited disk. Current progress, immutable source identity
and tests are recorded in [the declaration receipt](declaration.json).
The initial local test fixture used an invalid pandas timezone conversion;
its failure is retained in `tests_local_initial.xml`, and all **26 corrected
boundary tests pass** in [the local report](tests_local.xml). No model was fitted
before these checks. A declaration-only spelling correction to the actual
movement timestamp is preserved in
[the correction receipt](preparation_corrections/movement_time_column.json).

A qualifying model may receive a separately verified January-November refit
with median October-selected trees. There is no automatic competition upload.
Final development results and both platform verification reports will be
recorded here when the declared batch is complete.
