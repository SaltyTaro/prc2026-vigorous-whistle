"""Neural committee member for the ADS-B stack: a tabular MLP on the committee learners' 314-feature contract.

The package is independent of every earlier model package. It holds the fitting-row preprocessing (quantile bin edges
for piecewise-linear numeric encoding, missing-value flags, categorical vocabularies with a rare bucket, target scaling),
the network (per-feature piecewise-linear embeddings, categorical embeddings, a ReLU MLP, optionally a TabM-mini
multiplicative ensemble), the training loop (AdamW, warm-up plus cosine schedule, fixed epoch budget, optional weight
EMA) and save/load. Every statistic is estimated on fitting rows only; nothing here reads a label that is not passed in.
"""
