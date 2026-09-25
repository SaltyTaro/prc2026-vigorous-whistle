"""Context network, second design round: device-aware (GPU) training, a configurable neighbour window and optional
self-attention among the neighbour tokens. The first round's package (prc2026_ctx) is imported unchanged and stays frozen
while its development round runs; its token contract (tokens.build_tokens) and relative token features
(model.token_features) are reused as they are.
"""
