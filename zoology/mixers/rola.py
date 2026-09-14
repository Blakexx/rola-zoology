"""RoLA sequence mixer for zoology: a thin adapter over `fla.layers.RoLA`, as `gla.py` adapts fla's GLA.

It adds zoology's mixer convention and nothing else: the model builds ``mixer(d_model=..., layer_idx=..., **kwargs)``
from a serializable kwargs dict and calls ``mixer(x)`` expecting a tensor. The routing levels are fla's
`LevelRoutingConfig` dicts (``width``, ``read``/``write`` in ``{'dense', 'sparse'}``, ``tied``, ``alpha``), ``decay`` a
dict naming one of rola's decay sources; everything is validated and refused by fla and rola, never here.
"""
from __future__ import annotations

from typing import Optional

import torch.nn as nn
from fla.layers.rola import RoLA


class RoLAMixer(nn.Module):
    """zoology's mixer over fla's RoLA layer: ``d_model`` is the layer's ``hidden_size``, ``n_heads`` its
    ``num_heads``, ``d_v`` its ``head_v_dim``."""

    def __init__(self, d_model: int, layer_idx: Optional[int] = None, *, n_heads: int, levels: list, d_v: int = 64,
                 decay: Optional[dict] = None, router_bias: bool = False, gain_bias_init: Optional[float] = None):
        super().__init__()
        self.layer = RoLA(hidden_size=d_model, num_heads=n_heads, head_v_dim=d_v, levels=levels, decay=decay,
                          router_bias=router_bias, gain_bias_init=gain_bias_init, layer_idx=layer_idx)

    def forward(self, x):
        return self.layer(x)[0]

    def state_size(self, sequence_length: Optional[int] = None, **kwargs) -> int:
        return self.layer.state_size(sequence_length)
