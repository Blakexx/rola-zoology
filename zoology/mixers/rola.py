"""RoLA sequence mixer for Zoology — a first-class zoology mixer backed by the `rola` package.

Like the other zoology mixers (gla.py -> fla, based.py -> opt_einsum), this imports its backing
library (the installed `rola` package) and adapts it to zoology's sequence-mixer API
(instantiated with `d_model=...`, `layer_idx=...`; layer_idx unused). Single source of truth:
the model lives in `rola`; this is the thin zoology adapter.
"""
from __future__ import annotations
from typing import Optional

from rola import RoLA as _RoLA                              # the RoLA orchestrator
from rola import RecurrentGLA as _RecurrentGLA
from rola import RecurrentLinearAttention as _RecurrentLA
from rola import RecurrentGatedDelta as _RecurrentGDN


class RoLAMixer(_RoLA):
    """Zoology-compatible RoLA mixer: accepts `layer_idx` (required by Zoology's
    TransformerBlock) and passes everything else through to the RoLA base class."""
    def __init__(
        self,
        d_model: int,
        d_qk: int,
        d_v: int,
        num_chunks: int,
        n_heads: int = 4,
        tie_routers: bool = False,
        writer: str = "softmax_linear",
        reader: str = "softmax_linear",
        layer_idx: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            d_model=d_model,
            d_qk=d_qk,
            d_v=d_v,
            num_chunks=num_chunks,
            n_heads=n_heads,
            tie_routers=tie_routers,
            writer=writer,
            reader=reader,
            **kwargs,
        )
        self.layer_idx = layer_idx


class RecurrentGLA(_RecurrentGLA):
    """Zoology wrapper around our V+1-normalized RecurrentGLA single-state baseline.

    Different from Zoology's own GatedLinearAttention because we apply the V+1
    denominator trick: output is `(Σ k v^T q) / (Σ k^T q)`, mathematically faithful
    to softmax attention's normalization, where Zoology's drops the denominator.
    """
    def __init__(self, d_model: int, d_qk: int, d_v: int, n_heads: int = 4,
                 layer_idx: Optional[int] = None, **kwargs):
        super().__init__(d_model=d_model, d_qk=d_qk, d_v=d_v, n_heads=n_heads, **kwargs)
        self.layer_idx = layer_idx


class RecurrentLinearAttention(_RecurrentLA):
    """Zoology wrapper for our vanilla LA (with V+1 normalization)."""
    def __init__(self, d_model: int, d_qk: int, d_v: int, n_heads: int = 4,
                 layer_idx: Optional[int] = None, **kwargs):
        super().__init__(d_model=d_model, d_qk=d_qk, d_v=d_v, n_heads=n_heads, **kwargs)
        self.layer_idx = layer_idx


class RecurrentGatedDelta(_RecurrentGDN):
    """Zoology wrapper for our GDN (delta rule keeps state bounded; no V+1)."""
    def __init__(self, d_model: int, d_qk: int, d_v: int, n_heads: int = 4,
                 layer_idx: Optional[int] = None, **kwargs):
        super().__init__(d_model=d_model, d_qk=d_qk, d_v=d_v, n_heads=n_heads, **kwargs)
        self.layer_idx = layer_idx
