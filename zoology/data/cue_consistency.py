"""CUE-CONSISTENCY MQAR — the data instrument for the capacity theory's ADDRESS-AGREEMENT boundary.

Built 2026-08-02 (Phase 2). This is the generator the paper's second named boundary needs and that
`rola-bench` did not have (Phase 1 findings §7 item 1, audit §4.1 item 2).

THE THEORY IT INSTRUMENTS. Write `addr_w` / `addr_r` for the addresses the write and read sides of a
routed memory assign. A retrieval contract sending a query symbol `u` to the stored symbol `pi(u)` is
served when the two agree,

    addr_r(u) = addr_w(pi(u))   for every u                       (paper Eq. (7); theory notes T3)

and otherwise "the read marginalizes densely over the digits it cannot fix, at an active-leaf count
proportional to the matching instances -- the bill attention pays for a multi-match query."

Standard MQAR cannot move that axis: its key IS its cue, so agreement holds by construction at every
capacity. This task makes the cue structure a controlled axis while holding sequence geometry, item
count and vocabulary fixed, so that a change in accuracy is attributable to the CONTRACT and not to
length, item load or token statistics.

SYMBOLS, defined at first use.
  `D`  = `key_digits`         -- a key is a tuple of D sub-symbols ("digits"), one token each.
  `P`  = `digit_vocab`        -- the alphabet size of one digit slot.
  `q`  = `cue_digits`         -- how many of the D digits the QUERY presents (0 <= q <= D).
                                 The remaining `D - q` slots carry the reserved WILDCARD token, so
                                 the query's token geometry is identical at every q.
  `kv` = `num_kv_pairs`       -- stored items per episode.
  `m`  = `matches_per_cue`    -- stored items sharing the query's VISIBLE digits (m must divide kv).
  `G`  = kv / m               -- groups, and therefore queries, per episode.
  `L`  = `input_seq_len`.

THE THREE KNOBS, and the mechanism each one moves.

(a) CUE INFORMATIVENESS -- `cue_digits` q.
    The write addresses with all D digits; the query fixes q of them. Mutual information between the
    cue and the target's write address is `q * log P` nats out of `H(addr) = D * log P`, so the
    normalized quantity is exactly `q / D` and the conditional entropy `H(addr | cue)` is exactly
    `(D - q) * log P` -- literally "the open digits the read must marginalize over". q = D is
    agreement (Eq. (7) satisfiable by a global map); q = 0 is no cue at all.

(b) CUE CONSISTENCY -- `slot_consistency`, a three-rung ladder of how stably a token identifies the
    ADDRESS DIGIT it contributes:
      "consistent"       -- digit slot `s` draws from its own disjoint token pool, so token identity
                            determines slot. A global, compositional (radix) address map exists and
                            the router pays `sum_l b_l`, with systematic generalization to unseen
                            digit combinations (theory notes §2).
      "shared_pool"      -- every slot draws from ONE pool, so the same token is digit 0 in one item
                            and digit 1 in another. The address is still a global function, but of
                            (within-item offset, token) jointly rather than of the token alone.
      "episode_shuffled" -- shared pool, PLUS a per-episode permutation of the write side's slot
                            order that the query cannot observe (the query always presents canonical
                            order). No context-free map satisfies Eq. (7): the required
                            correspondence is redrawn every episode, so a static layer-1 router must
                            average over `D!` orders and only a context-mixed (depth-supplied,
                            sequence-dependent) router can recover it.
    (b) is designed to leave (a) untouched: the marginal law of every digit is uniform on `P` in all
    three rungs, so cue<->address mutual information is a function of `q` alone. That orthogonality
    is gated, not asserted.

(c) MULTI-MATCH -- `matches_per_cue` m, with `match_resolution` fixing what the contract asks for
    when m > 1:
      "recency"   -- the m matching items carry DISTINCT values and the target is the one written
                     LAST. Well-posed, and the same contract attention serves with a recency-biased
                     query. A decay-free normalized blend returns the mass-weighted mixture of m
                     distinct values with no signal to break the tie -- this is the harmful half of
                     the aliasing ledger (theory notes §3, "engineered aliasing").
      "consensus" -- the m matching items carry the SAME value. The marginalization is exactly as
                     wide, and the normalized readout is EXACT: `(R m v) / (R m) = v` (paper §5.1's
                     first error-free case). This is the harmless half, and the pair
                     (recency, consensus) at equal m is the cleanest available separation of
                     "the read spread" from "the contract mismatch".
    m is meaningful only below full cue: at q = D the cue determines the item, so m = 1 is forced.

STRUCTURE OF ONE EPISODE (D = 2, q = 1, m = 2, kv = 4, `.` = filler):

    context (kv items, D+1 tokens each)          queries (G items, D+1 slots each)
    d0 d1 v | d0 d1 v | d0 d1 v | d0 d1 v   . .  c0 W ? . . . c0 W ?  . .
     A  X a    B  Y b    A  Z c    B  W d         A          B
    labels                                              ^c            ^d

  Group A = {(A,X)->a, (A,Z)->c}: both match the cue `A`; under "recency" the target is `c`, the one
  written later. `W` is the WILDCARD token, which occurs at hidden query slots and NOWHERE else --
  filler never draws it -- so the open-digit positions are unambiguous.

DELIBERATE DEPARTURES FROM `multiquery_ar.py`, each for a reason:
  * filler is drawn from `[2, vocab)` rather than `[0, vocab)`, i.e. it can never be PAD or WILDCARD.
    Random wildcards in filler positions would make the "open digit" marker noisy, which is the one
    signal this task's independent variable is carried by.
  * randomness comes from `np.random.default_rng(seed)` rather than the global `np.random.seed`, so a
    segment's draw cannot be perturbed by an unrelated global consumer. Determinism per seed is
    gated.
  * weighted sampling without replacement for query placement uses the Gumbel-top-k identity rather
    than a per-row `np.random.choice`; it is the same distribution, vectorized (the per-row loop is
    the reason the existing generators are slow at 100k examples).
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from zoology.config import DataSegmentConfig
from zoology.data.utils import DataSegment

#: Reserved token ids. PAD is the build-time placeholder (zoology's convention); WILDCARD marks a
#: query slot whose digit the cue does NOT fix, and is excluded from filler so it never occurs by
#: accident.
PAD_TOKEN = 0
WILDCARD_TOKEN = 1
FIRST_FREE_TOKEN = 2

SLOT_CONSISTENCY = ("consistent", "shared_pool", "episode_shuffled")
MATCH_RESOLUTION = ("recency", "consensus")


class CueConsistencyConfig(DataSegmentConfig):
    name: str = "cue_consistency"
    num_kv_pairs: int = 8
    key_digits: int = 2
    cue_digits: int = 2
    digit_vocab: int = 64
    slot_consistency: Literal["consistent", "shared_pool", "episode_shuffled"] = "consistent"
    matches_per_cue: int = 1
    match_resolution: Literal["recency", "consensus"] = "recency"
    power_a: float = 0.01
    random_non_queries: bool = True
    include_slices: bool = True

    def build(self, seed: int) -> DataSegment:
        return cue_consistency(**self.model_dump(), seed=seed)


@dataclass
class CueEpisodes:
    """The ground truth behind the tokens, exposed so the unit gates measure the ACTUAL arrays that
    become the sequence rather than a re-derivation of them ([[matching-the-naive-antipattern]])."""

    digits: np.ndarray          # [n, kv, D] canonical-order address digits, per item, as DIGIT INDEX
    values: np.ndarray          # [n, kv] value token per item
    context_order: np.ndarray   # [n, kv] item index occupying each context slot
    write_perm: np.ndarray      # [n, D] slot permutation applied to the WRITE side
    cue: np.ndarray             # [n, G, q] the digit indices the query presents
    target_item: np.ndarray     # [n, G] flat item index the contract names for each query
    group_of_item: np.ndarray   # [n, kv] group id of each item (flat item index -> group)
    digit_tokens: np.ndarray    # [D, P] the token id of digit index p at slot s
    meta: dict


def _distinct_per_row(rng, n_rows, universe, k, chunk=8192):
    """`[n_rows, k]` integers in `[0, universe)`, distinct WITHIN a row, uniform.

    Chunked so the transient key matrix stays bounded (a full `n x universe` random matrix is
    gigabytes at 100k examples over a 4k value pool)."""
    if k > universe:
        raise ValueError(f"cannot draw {k} distinct values from a universe of {universe}")
    out = np.empty((n_rows, k), dtype=np.int64)
    for lo in range(0, n_rows, chunk):
        hi = min(lo + chunk, n_rows)
        out[lo:hi] = np.argsort(rng.random((hi - lo, universe)), axis=1)[:, :k]
    return out


def _weighted_distinct(rng, n_rows, weights, k):
    """Weighted sampling WITHOUT replacement, `k` per row, over `len(weights)` positions.

    The Gumbel-top-k identity: taking the k largest of `log w_i + Gumbel_i` reproduces successive
    weighted sampling without replacement, i.e. exactly `np.random.choice(p=w, replace=False)`."""
    g = -np.log(-np.log(rng.random((n_rows, len(weights)))))
    return np.argsort(-(np.log(weights)[None, :] + g), axis=1)[:, :k]


def _validate(*, vocab_size, input_seq_len, num_kv_pairs, key_digits, cue_digits, digit_vocab,
              slot_consistency, matches_per_cue, match_resolution):
    D, q, kv, m, P = key_digits, cue_digits, num_kv_pairs, matches_per_cue, digit_vocab
    if D < 1:
        raise ValueError(f"key_digits={D} must be >= 1.")
    if not 0 <= q <= D:
        raise ValueError(f"cue_digits={q} must satisfy 0 <= cue_digits <= key_digits={D}.")
    if slot_consistency not in SLOT_CONSISTENCY:
        raise ValueError(f"slot_consistency={slot_consistency!r} not in {SLOT_CONSISTENCY}.")
    if match_resolution not in MATCH_RESOLUTION:
        raise ValueError(f"match_resolution={match_resolution!r} not in {MATCH_RESOLUTION}.")
    if m < 1 or kv % m:
        raise ValueError(
            f"matches_per_cue={m} must be >= 1 and divide num_kv_pairs={kv}: the episode is built as "
            f"{kv // m if m else 0} groups of exactly m items sharing a cue, and a remainder would "
            "make the realized match count differ between groups -- silently averaging two values of "
            "the independent variable.")
    if q == D and m != 1:
        raise ValueError(
            f"cue_digits={q} == key_digits fixes the whole address, so a cue names exactly ONE item; "
            f"matches_per_cue={m} is unrealizable there (it would need duplicate keys). Full-cue "
            "cells are m = 1 by construction -- that is what agreement MEANS.")
    if q == 0 and m != kv:
        raise ValueError(
            f"cue_digits=0 presents no address information, so EVERY stored item matches: "
            f"matches_per_cue must equal num_kv_pairs={kv}, got {m}.")
    G = kv // m
    if q > 0 and P < G:
        raise ValueError(
            f"digit_vocab={P} < groups={G}: cue tuples are made distinct across groups by giving "
            "cue slot 0 a distinct digit per group, which needs at least G digit values.")
    if q < D and P < m:
        raise ValueError(
            f"digit_vocab={P} < matches_per_cue={m}: the m members of a group are separated by "
            "distinct digits in the first HIDDEN slot, which needs at least m digit values.")
    pools = D if slot_consistency == "consistent" else 1
    key_region = vocab_size // 2 - FIRST_FREE_TOKEN
    if pools * P > key_region:
        raise ValueError(
            f"the key half of the vocabulary holds {key_region} tokens, but "
            f"slot_consistency={slot_consistency!r} needs {pools} disjoint pool(s) of "
            f"digit_vocab={P}. Raise vocab_size or lower digit_vocab.")
    if vocab_size <= input_seq_len:
        raise ValueError(f"vocab_size={vocab_size} must exceed input_seq_len={input_seq_len}.")
    stride = D + 1
    context = kv * stride
    space = (input_seq_len - context) // stride
    if space < G:
        raise ValueError(
            f"input_seq_len={input_seq_len} does not fit the episode: {kv} items x {stride} tokens "
            f"= {context} of context leaves room for {max(space, 0)} query slot(s) of width {stride}, "
            f"and the contract needs {G}. Structural -- drop the rung rather than letting it assert "
            "deep in the build.")
    return D, q, kv, m, P, G, stride, context, space


def build_episodes(
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    seed: int,
    *,
    num_kv_pairs: int = 8,
    key_digits: int = 2,
    cue_digits: int = 2,
    digit_vocab: int = 64,
    slot_consistency: str = "consistent",
    matches_per_cue: int = 1,
    match_resolution: str = "recency",
    power_a: float = 0.01,
) -> CueEpisodes:
    """The episode ground truth, before tokenization. Separated from `cue_consistency` so the unit
    gates can measure cue<->address mutual information, realized match counts and the contract's
    target directly off the arrays the renderer consumes."""
    D, q, kv, m, P, G, stride, context, space = _validate(
        vocab_size=vocab_size, input_seq_len=input_seq_len, num_kv_pairs=num_kv_pairs,
        key_digits=key_digits, cue_digits=cue_digits, digit_vocab=digit_vocab,
        slot_consistency=slot_consistency, matches_per_cue=matches_per_cue,
        match_resolution=match_resolution)
    rng = np.random.default_rng(seed)
    n = num_examples

    # --- the digit token table: [D, P]. "consistent" gives each slot its own disjoint pool, so
    # token identity determines the slot; the shared rungs give every slot the SAME pool, so it does
    # not. Both keep the per-slot alphabet at exactly P, which is what holds knob (a) fixed.
    if slot_consistency == "consistent":
        digit_tokens = (FIRST_FREE_TOKEN + np.arange(D * P).reshape(D, P)).astype(np.int64)
    else:
        digit_tokens = np.tile(FIRST_FREE_TOKEN + np.arange(P), (D, 1)).astype(np.int64)

    # --- cue digits: [n, G, q]. Slot 0 distinct ACROSS groups (that alone makes the cue tuples
    # distinct, hence the realized match count exactly m); the remaining cue slots i.i.d. uniform, so
    # every digit stays marginally uniform on P and the MI arithmetic above holds.
    cue = np.empty((n, G, q), dtype=np.int64)
    if q:
        cue[:, :, 0] = _distinct_per_row(rng, n, P, G)
        if q > 1:
            cue[:, :, 1:] = rng.integers(0, P, size=(n, G, q - 1))

    # --- hidden digits: [n, G, m, D-q]. First hidden slot distinct ACROSS the m group members, so
    # the members are distinct items; the rest i.i.d. uniform.
    hidden = np.empty((n, G, m, D - q), dtype=np.int64)
    if D - q:
        first = _distinct_per_row(rng, n * G, P, m).reshape(n, G, m)
        hidden[:, :, :, 0] = first
        if D - q > 1:
            hidden[:, :, :, 1:] = rng.integers(0, P, size=(n, G, m, D - q - 1))

    digits = np.concatenate(
        [np.repeat(cue[:, :, None, :], m, axis=2), hidden], axis=3).reshape(n, kv, D)
    group_of_item = np.repeat(np.arange(G), m)[None, :].repeat(n, axis=0)

    # --- values. "recency": every item gets its own value, so the m matches of a group are m
    # DISTINCT values and only the write order picks the target. "consensus": one value per group,
    # so the marginalization is exactly as wide and exactly harmless.
    value_pool = np.arange(vocab_size // 2, vocab_size)
    if match_resolution == "consensus":
        idx = _distinct_per_row(rng, n, len(value_pool), G)
        values = np.repeat(value_pool[idx], m, axis=1)
    else:
        idx = _distinct_per_row(rng, n, len(value_pool), kv)
        values = value_pool[idx]

    # --- context order: a per-episode permutation, so group members interleave and "written last" is
    # not a positional artefact of the grouping.
    context_order = _distinct_per_row(rng, n, kv, kv)          # a full permutation of the kv items
    slot_of_item = np.argsort(context_order, axis=1)           # item -> its context slot

    # --- the contract's target per group.
    if match_resolution == "consensus":
        target_item = np.arange(G)[None, :].repeat(n, axis=0) * m      # any member; values agree
    else:
        pos = slot_of_item.reshape(n, G, m)
        target_item = np.arange(G)[None, :] * m + np.argmax(pos, axis=2)

    # --- write-side slot permutation (knob (b)'s third rung).
    if slot_consistency == "episode_shuffled":
        write_perm = _distinct_per_row(rng, n, D, D)
    else:
        write_perm = np.arange(D)[None, :].repeat(n, axis=0)

    meta = dict(key_digits=D, cue_digits=q, num_kv_pairs=kv, matches_per_cue=m, groups=G,
                digit_vocab=P, slot_consistency=slot_consistency, match_resolution=match_resolution,
                stride=stride, context_size=context, query_space=space, power_a=power_a,
                input_seq_len=input_seq_len, vocab_size=vocab_size)
    return CueEpisodes(digits=digits, values=values, context_order=context_order,
                       write_perm=write_perm, cue=cue, target_item=target_item,
                       group_of_item=group_of_item, digit_tokens=digit_tokens, meta=meta)


def cue_consistency(
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    seed: int,
    num_kv_pairs: int = 8,
    key_digits: int = 2,
    cue_digits: int = 2,
    digit_vocab: int = 64,
    slot_consistency: str = "consistent",
    matches_per_cue: int = 1,
    match_resolution: str = "recency",
    power_a: float = 0.01,
    random_non_queries: bool = True,
    include_slices: bool = True,
    **kwargs,
) -> DataSegment:
    """Render `build_episodes` into (inputs, labels). See the module docstring for the contract."""
    ep = build_episodes(
        vocab_size=vocab_size, num_examples=num_examples, input_seq_len=input_seq_len, seed=seed,
        num_kv_pairs=num_kv_pairs, key_digits=key_digits, cue_digits=cue_digits,
        digit_vocab=digit_vocab, slot_consistency=slot_consistency,
        matches_per_cue=matches_per_cue, match_resolution=match_resolution, power_a=power_a)
    M = ep.meta
    n, D, q, kv, G = num_examples, M["key_digits"], M["cue_digits"], M["num_kv_pairs"], M["groups"]
    stride, context, space = M["stride"], M["context_size"], M["query_space"]
    rng = np.random.default_rng(seed + 1_000_003)     # rendering noise, disjoint from the episode draw

    rows = np.arange(n)[:, None]
    # --- context: items in their episode order, digits emitted under the write-side slot permutation
    ordered_digits = ep.digits[rows, ep.context_order]                     # [n, kv, D]
    ordered_digits = np.take_along_axis(
        ordered_digits, ep.write_perm[:, None, :].repeat(kv, axis=1), axis=2)
    slot_index = np.arange(D)[None, None, :].repeat(n, 0).repeat(kv, 1)
    ordered_tokens = ep.digit_tokens[slot_index, ordered_digits]           # [n, kv, D]
    ordered_values = ep.values[rows, ep.context_order]                     # [n, kv]

    ctx = np.zeros((n, context), dtype=np.int64)
    for s in range(D):
        ctx[:, s::stride] = ordered_tokens[:, :, s]
    ctx[:, D::stride] = ordered_values

    # --- queries: canonical slot order always, WILDCARD on every digit the cue does not fix.
    query_tokens = np.full((n, G, D), WILDCARD_TOKEN, dtype=np.int64)
    if q:
        cue_slots = np.arange(q)[None, None, :].repeat(n, 0).repeat(G, 1)
        query_tokens[:, :, :q] = ep.digit_tokens[cue_slots, ep.cue]
    target_values = ep.values[rows, ep.target_item]                        # [n, G]

    # power-law gap distribution, verbatim from `multiquery_ar` (same `power_a` semantics)
    p = power_a * np.arange(1, space + 1) ** (power_a - 1)
    gaps = _weighted_distinct(rng, n, p / p.sum(), G)                      # [n, G]

    qregion = np.zeros((n, input_seq_len - context + 1), dtype=np.int64)
    for s in range(D):
        np.put_along_axis(qregion, gaps * stride + s, values=query_tokens[:, :, s], axis=1)
    examples = np.concatenate([ctx, qregion], axis=1)

    labels = np.full((n, input_seq_len + 1), -100, dtype=np.int64)
    np.put_along_axis(labels, gaps * stride + context + D, values=target_values, axis=1)

    if random_non_queries:
        # filler NEVER draws PAD or WILDCARD: a stray wildcard would blur the one marker that carries
        # this task's independent variable. It is drawn from THIS generator's stream, not torch's
        # global one — `multiquery_ar` uses `torch.randint` here, which makes its filler depend on
        # whatever else touched the global torch RNG first (caught by the determinism gate).
        holes = examples == PAD_TOKEN
        examples[holes] = rng.integers(FIRST_FREE_TOKEN, vocab_size, size=int(holes.sum()))
    inputs = torch.tensor(examples[:, :-1])
    labels = torch.tensor(labels[:, 1:])

    slices = {}
    if include_slices:
        slices = {"num_kv_pairs": kv, "input_seq_len": input_seq_len, "cue_digits": q,
                  "matches_per_cue": M["matches_per_cue"], "key_digits": D}
    return DataSegment(inputs, labels, slices=slices)
