"""Chunked Linear Attention (CLA) mixer for Zoology.

Wraps the ChunkedLinearAttention class from /mnt/c/Users/Blake/Documents/VSCode/CLA/cla_bench.py
so we have a single source of truth. Zoology's TransformerBlock instantiates sequence_mixers
with `d_model=...` and `layer_idx=...`; we accept both (layer_idx unused).
"""
from __future__ import annotations
import sys, os
from typing import Optional

# Add the CLA repo to sys.path so we can import ChunkedLinearAttention.
_CLA_REPO = "/mnt/c/Users/Blake/Documents/VSCode/CLA"
if _CLA_REPO not in sys.path:
    sys.path.insert(0, _CLA_REPO)

from rola import RoLA as _CLA  # noqa: E402  (RoLA orchestrator; legacy local alias _CLA)
from rola import RecurrentGLA as _RecurrentGLA  # noqa: E402
from rola import RecurrentLinearAttention as _RecurrentLA  # noqa: E402
from rola import RecurrentGatedDelta as _RecurrentGDN  # noqa: E402


class ChunkedLinearAttention(_CLA):
    """Zoology-compatible wrapper around our ChunkedLinearAttention.

    Accepts `layer_idx` (required by Zoology's TransformerBlock) and passes
    everything else to the base class.
    """
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
