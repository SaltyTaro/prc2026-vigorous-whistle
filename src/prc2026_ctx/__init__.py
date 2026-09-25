"""Context network: the committee network's tabular branch plus attention over the individual movements around each departure.

`tokens` builds, per supplied source file, a table of every movement (departures and arrivals, label-free) and each
departure's context neighbours; `model` defines the network; `train` fits and predicts it. The frozen `prc2026_nn` package
is imported, never edited.
"""
