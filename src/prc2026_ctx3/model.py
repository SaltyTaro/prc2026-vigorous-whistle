"""Round-3 context network: prc2026_ctx2.model.ContextNet2 unchanged, plus an optional linear auxiliary head on the backbone's last
hidden layer (the input of the main head). The main prediction is exactly ContextNet2's forward pass; `forward_aux` returns it together
with the auxiliary outputs. With `n_aux = 0` the module, its parameters and their initialisation are ContextNet2's.
"""
from __future__ import annotations

import torch
from torch import nn

from prc2026_ctx2.model import ContextNet2


class ContextNet3(ContextNet2):
    def __init__(self, *args, n_aux: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_aux = int(n_aux)
        if self.n_aux:
            # created after every ContextNet2 parameter, so those keep ContextNet2's initialisation for the same seed
            self.aux_head = nn.Linear(self.head.in_features, self.n_aux)

    def forward_aux(self, x_num: torch.Tensor, x_cat: torch.Tensor, tok: torch.Tensor, tok_nan: torch.Tensor, tok_wake: torch.Tensor,
                    valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(N, 1) main predictions (ContextNet2's forward) and (N, n_aux) auxiliary outputs from the same hidden layer."""
        if not self.n_aux:
            raise ValueError("This network has no auxiliary head.")
        captured = {}
        handle = self.head.register_forward_pre_hook(lambda module, inputs: captured.__setitem__("hidden", inputs[0]))
        try:
            out = self(x_num, x_cat, tok, tok_nan, tok_wake, valid)
        finally:
            handle.remove()
        return out, self.aux_head(captured["hidden"])
