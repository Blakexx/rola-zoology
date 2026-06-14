
import math
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..config import ModuleConfig


class Hybrid(nn.Module):
    def __init__(
        self,
        d_model: int,
        configs: List[ModuleConfig],
        layer_idx: int=None,
        **kwargs
    ):
        super().__init__()
        
        self.d_model = d_model

        self.mixer = ModuleConfig(
            **configs[layer_idx % len(configs)]
        ).instantiate(d_model=d_model, layer_idx=layer_idx)

    def forward(self, u, *args, **kwargs):
        """
        Args:
            u: (b, l, d) tensor
        Returns:
            y: (b, l, d) tensor
        """
        out = self.mixer(u, *args, **kwargs)
        # Canonical baselines from fla.layers (LinearAttention, GatedDeltaNet) follow the HF layer
        # contract and return a tuple (hidden_states, attn, cache); zoology's block expects a bare
        # tensor. Unwrap so HF-style sub-mixers compose with the rest of the block.
        return out[0] if isinstance(out, tuple) else out

    def state_size(self, **kwargs):
        # Canonical baselines (fla.layers.LinearAttention/GatedDeltaNet) intentionally expose no
        # state_size — their realized state is read from the built projections by the companion
        # verifier, not from this method. Guard so the harness's state-logging doesn't crash on them.
        return self.mixer.state_size(**kwargs) if hasattr(self.mixer, "state_size") else 0
    