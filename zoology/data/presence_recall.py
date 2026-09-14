"""PRESENCE RECALL — the task that tests ADDRESSING alone: "have I already seen pi(x)?".

Built 2026-08-03, on the `graph_recall` spine (its conflict-group primitive, its chi solver, its
coverage stamp and its determinism discipline are IMPORTED here, not re-implemented). This is v1's
sole theory-confirmation instrument; the multi-family graph grid is retained as the sequel's.

SYMBOLS, defined at first use.
  `pi`   -- the RANDOM-ONCE-FROZEN map from a key token to the key token it wants to read, with
            `pi(x) != x` for every x. Drawn per CELL, never per segment.
  `K`    -- `demand`: the size of the key pool, which is also this task's chi (see THE DEMAND).
  `s`    -- `keys_per_sequence`: how many key events one sequence emits (each key at most once).
  `L`    -- `input_seq_len`, in tokens. `N` -- addressable states per head of the model under test
            (not a knob here; the grid crosses it against chi).
  `chi`  -- the DEMAND: the chromatic number of the realized union co-occurrence graph.

THE TASK. A sequence of key tokens `x_0, x_1, ...`; the LABEL at every key position `t` is the single
bit "does `pi(x_t)` appear STRICTLY BEFORE t?". Four properties, and each one is why this task and
not `multiquery_ar`:

  (1) THE DEMAND IS PRESERVED. A false positive is an error, so every drawn key must be separately
      distinguishable in the state at once: the conflict group of a sequence is its whole drawn key
      set, and the union over sequences of those cliques is the COMPLETE graph on the pool. chi = K,
      solved by `graph_chi` and gated against the closed form exactly as `graph_recall.complete(k)`.
  (2) THE VALUE CONFOUND IS ELIMINATED. The answer is one bit, so the cell tests N and nothing else.
      MQAR entangles addressing with log|V|-bit value retrieval, and a capacity curve measured there
      cannot say which of the two moved.
  (3) READS AND WRITES ARE ASYMMETRIC BY CONSTRUCTION. Every position both WRITES its own address
      (x_t is now present) and READS another (`pi(x_t)`), so `addr_read(x) = addr_write(pi(x))` and
      address agreement becomes a LEARNED property instead of a free coincidence. In MQAR the read
      cue IS the written content, which is precisely the regime where a shared routing solve gets
      agreement for nothing. The consequence for the wirings is stated in `presence_grid.yaml`: the
      shared-support arm must emit the support UNION {f(x), f(pi(x))}, so its identity-cue advantage
      is gone.
  (4) SUPERVISION IS DENSE. Every key position carries a question, and "before" is the causal mask's
      own semantics, so nothing about the layout has to teach the model what time means.

WHY THE DRAWN SUBSET VARIES, which is the whole reason `s < K`. If every sequence drew the entire
pool, the j-th key event would be positive with probability ~ j / (s - 1): a POSITIONAL PRIOR worth
about 75% accuracy to a model that never addresses anything. The generator therefore PLANTS the class
of every position and draws the key to match it (see `_emit`), which makes the positive rate flat in
position by construction; `presence_position_bias` is stamped so the property is measured on the
arrays the renderer consumes and not assumed. The planted class is a Bernoulli(`positive_rate`) draw
and the key is taken uniformly from the candidate set that realizes it -- CONSTRUCTION, never
rejection sampling. Where a planted class has no candidate the other is taken and the REALIZED label
is stamped (`presence_forced_rate`); with the default `s = K // 2` that is a fraction of a percent,
and it is a number in the stamp rather than a hope.

THE FIRST EVENT OF EVERY SEQUENCE IS NOT LABELLED, and that is a correctness requirement rather than
a trim. Its prefix is empty, so its answer is the constant ABSENT: scoring it would hand every model
a free `1 / s` of accuracy for no addressing at all AND would pull the realized class balance off
0.5, which is the line the whole metric is read against. The `s - 1` remaining events are labelled,
their planted balance is ~50/50, and the stamp carries the realized number
(`presence_positive_rate`, measured over the labelled positions).

THE MAP. `pi` is a uniformly random single K-cycle over the pool: draw a permutation `p` and set
`pi(p[i]) = p[(i + 1) % K]`. That is a derangement BY CONSTRUCTION -- `pi(x) != x` everywhere with no
rejection loop -- and it is frozen per CELL from `pi_seed`, never from the segment `seed`, so a
cell's train and eval segments share one map. Keying it off the segment seed would make a cell
unlearnable by construction rather than by capacity, and the failure would read as a capacity result.

TOKEN LAYOUT. `s` key events at random INCREASING positions with a minimum gap of `min_separation`
tokens; every other position is filler drawn from the non-key, non-label vocabulary.

    presence(K=8, s=4, L=16, min_separation=4), `.` = filler, `^` = a labelled position

      pos   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
      tok   .  . k3  .  .  .  . k7  .  . k1  .  .  . k3'  .
      lab         ^0             ^1        ^0           ^1

    (the label at a key position is 1 iff that key's pi-target sits somewhere to its LEFT.)

`min_separation` defaults to 4 tokens, which exceeds the hybrid backbone's short-conv kernel of 3: a
read whose target sits in the immediately preceding tokens is served by the convolution and measures
nothing about the state. Every pair of key events is at least that far apart, so the statement holds
for every pair and not merely for consecutive ones.

THE LABEL IS A RESERVED TOKEN PAIR, not a new head. zoology's discrete `ce` path is a full-vocabulary
softmax with `ignore_index=-100`, and `compute_metrics` is token accuracy over the labelled
positions; a boolean target is expressed exactly by reserving two ids (`PRESENT = vocab-1`,
`ABSENT = vocab-2`), which are excluded from the key pool AND from the filler and never appear in the
INPUT, so there is nothing to copy. This is the least-invasive correct form: it needs no change to
`train.py`, to the model or to the metric. THE CHANCE LINE IS 0.5 -- the best label-free predictor of
a ~50/50 balance -- and an untrained model sits BELOW it, since it must first learn to answer inside
the reserved pair at all. `rola_bench.mqar.graph_analysis --task presence` prints the line explicitly
and defines the collapse threshold above it.

THE DEMAND IS MEASURED, NEVER ASSUMED, on the same terms as `graph_recall`: every segment solves its
own realized union graph (`zoology.data.graph_chi.solve_chi`, with a verified clique and a verified
coloring) and stamps `graph_nodes / graph_edges / graph_demand_lower / graph_demand_upper /
graph_demand_exact / graph_predicted_N` into `DataSegment.slices`. Because the drawn subset varies,
union completeness is a property of the SAMPLE rather than of the knob -- a segment with too few
examples realizes a strict subgraph of K_K -- so `verify_construction` gates it and a coverage
shortfall is refused instead of shipped with an x-coordinate the model was never shown.

DETERMINISM. Every draw comes from an `np.random.default_rng(...)` stream local to the call -- never
`np.random.seed` (global numpy) and never an unseeded `torch.randint` (global torch), the two live
bugs `multiquery_ar` carried until 2026-08-02. `gen_version` is a plain pydantic field, so it
participates in `model_dump()` and therefore in the on-disk cache key.
"""
import hashlib
import json
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch

from zoology.config import DataSegmentConfig
from zoology.data.graph_chi import verify_construction
from zoology.data.graph_recall import _distinct_per_row, demand_bounds, realized_graph
from zoology.data.utils import DataSegment

#: Bump when the DRAW SEQUENCE or the token layout changes. Participates in the on-disk cache key.
PRESENCE_RECALL_GEN_VERSION = 1

#: Reserved token ids. PAD is the build-time placeholder and is fully overwritten before the segment
#: is returned; the two LABEL ids are targets only and never enter the input.
PAD_TOKEN = 0
FIRST_FREE_TOKEN = 1

#: The two sides of the contract, stamped for symmetry with `graph_recall`. Presence-recall asks
#: about EVERY key position on both sides -- the contract is exhaustive by construction, so the
#: split changes no draw; it is carried so a result row says which segment it came from.
SPLITS = ("train", "eval")

STAMP_FIELDS = ("graph_construction", "graph_nodes", "graph_edges", "graph_contract_edges",
                "graph_contract_coverage", "graph_contract_full", "graph_split",
                "graph_demand_lower", "graph_demand_upper", "graph_demand_exact",
                "graph_demand_method", "graph_predicted_N", "graph_log2_demand",
                "graph_demand_nominal", "presence_keys_per_sequence", "presence_positive_rate",
                "presence_position_bias", "presence_forced_rate", "presence_chance",
                "presence_min_separation", "presence_pi_hash", "presence_pi_derangement")

#: The accuracy a label-free predictor reaches on a balanced binary task. Stamped so no plot has to
#: re-derive it and no threshold can be quoted below it.
CHANCE = 0.5


def label_tokens(vocab_size: int) -> tuple:
    """`(PRESENT, ABSENT)` -- the reserved target ids for this vocabulary. See THE LABEL, above."""
    return vocab_size - 1, vocab_size - 2


class PresenceRecallConfig(DataSegmentConfig):
    name: str = "presence_recall"
    demand: int = 16                  # K: the key pool, which IS this task's chi
    keys_per_sequence: int = 0        # s; 0 -> the default K // 2 (see WHY THE SUBSET VARIES)
    positive_rate: float = 0.5        # the PLANTED class balance
    pi_seed: int = 12345              # pi is FROZEN per cell -- never keyed off the segment seed
    min_separation: int = 4           # minimum token gap between key events
    split: Literal["train", "eval"] = "train"   # set by the segment builder, not by a spec
    include_slices: bool = True
    gen_version: int = PRESENCE_RECALL_GEN_VERSION

    def build(self, seed: int) -> DataSegment:
        return presence_recall(**self.model_dump(), seed=seed)


@dataclass
class PresenceEpisodes:
    """The episode ground truth, before tokenization.

    Exposed so the gates measure the ACTUAL arrays the renderer consumes rather than a
    re-derivation of them ([[matching-the-naive-antipattern]])."""

    key_token: np.ndarray      # [n, s] key token emitted at each event, in emission order
    event_pos: np.ndarray      # [n, s] token position of each event (strictly increasing per row)
    label: np.ndarray          # [n, s] bool: was pi(key) already present when this event fired?
    planned: np.ndarray        # [n, s] bool: the class that was PLANTED (label where it was met)
    groups: np.ndarray         # [n, 1, s] the conflict group: the whole drawn key set
    key_pool: np.ndarray       # [K] every key token the pool holds
    pi: np.ndarray             # [K] pi as POOL INDICES: pi(key_pool[i]) == key_pool[pi[i]]
    meta: dict = field(default_factory=dict)


def frozen_pi(pool_size: int, pi_seed: int) -> np.ndarray:
    """A uniformly random single `pool_size`-cycle, as an index map. `pi[i] != i` by construction.

    A derangement without a rejection loop: draw a permutation `p` and walk it cyclically. Frozen per
    CELL (`pi_seed`), so every segment of a cell -- train and eval alike -- shares one map."""
    if pool_size < 2:
        raise ValueError(f"demand={pool_size} must be >= 2: pi(x) != x has no spelling on one key.")
    p = np.random.default_rng(pi_seed).permutation(pool_size)
    pi = np.empty(pool_size, dtype=np.int64)
    pi[p] = np.roll(p, -1)
    return pi


def pi_hash(pi: np.ndarray) -> str:
    """A stable short digest of the frozen map, stamped so two cells can be compared for identity."""
    return hashlib.sha1(np.asarray(pi, dtype=np.int64).tobytes()).hexdigest()[:12]


def _validate(*, vocab_size, input_seq_len, demand, keys_per_sequence, positive_rate,
              min_separation, split):
    if split not in SPLITS:
        raise ValueError(f"split={split!r} not in {SPLITS}.")
    if demand < 2:
        raise ValueError(f"demand={demand} must be >= 2: pi(x) != x has no spelling on one key.")
    if not 0.0 < positive_rate < 1.0:
        raise ValueError(
            f"positive_rate={positive_rate} must satisfy 0 < r < 1: the sharp 0.5 chance line is the "
            "point of the task, and a degenerate balance would make accuracy readable without "
            "addressing anything.")
    if min_separation < 1:
        raise ValueError(
            f"min_separation={min_separation} must be >= 1: a read whose target sits in the "
            "immediately preceding tokens is served by the backbone's short convolution and measures "
            "nothing about the state.")
    s = keys_per_sequence or max(2, demand // 2)
    if s < 2:
        raise ValueError(f"keys_per_sequence={s} must be >= 2: a one-event sequence has no prefix.")
    if s > demand:
        raise ValueError(
            f"keys_per_sequence={s} exceeds demand={demand}: a key is emitted at most once per "
            "sequence (a repeat is monotone in the presence relation and carries no new question), "
            "so a sequence cannot hold more events than the pool holds keys.")
    if s == demand:
        raise ValueError(
            f"keys_per_sequence={s} == demand={demand}: with the whole pool drawn, the negative "
            "class runs out at the tail of every sequence (after m emissions only ~(K-m)^2/K "
            "unemitted keys still have pi(x) absent), so the planted balance would degenerate into "
            "a positional prior -- exactly the confound the subset draw exists to remove. Use "
            "keys_per_sequence < demand (the default is demand // 2).")
    span = input_seq_len - (s - 1) * (min_separation - 1)
    if span < s:
        raise ValueError(
            f"input_seq_len={input_seq_len} cannot hold {s} key events at min_separation="
            f"{min_separation} (it needs at least {s + (s - 1) * (min_separation - 1)} tokens). "
            "Structural: raise L, lower keys_per_sequence, or lower min_separation, rather than "
            "letting the placement assert deep in the draw.")
    # two reserved LABEL ids at the top, the key pool at the bottom, and filler must still have
    # somewhere to come from
    if demand + 2 + FIRST_FREE_TOKEN >= vocab_size:
        raise ValueError(
            f"vocab_size={vocab_size} cannot hold a key pool of {demand}, the two reserved label "
            "ids and any filler. Raise vocab_size.")
    if vocab_size <= input_seq_len:
        raise ValueError(f"vocab_size={vocab_size} must exceed input_seq_len={input_seq_len}.")
    return s


def _emit(rng, n, pool_size, s, pi, positive_rate):
    """The PLANTED-CLASS emission: `(key_idx, label, planned)`, each `[n, s]`.

    Left to right, vectorized over rows. At step t the planted class `planned[:, t]` selects the
    candidate set -- unemitted keys whose `pi(x)` is already in the prefix (positive) or is not
    (negative) -- and the key is drawn UNIFORMLY from it. This is what makes the positive rate flat
    in position: the class is chosen first and the key second, so nothing about how full the prefix
    is leaks into the label. The classic alternative (draw keys, read off the labels) has a positive
    rate that rises with t and is worth ~75% accuracy to a model that never addresses anything.

    A planted class with no candidate falls back to the other one and the REALIZED label is
    returned; the caller stamps the rate. Both classes cannot be empty while any key is unemitted,
    since every unemitted key is in exactly one of them.
    """
    rows = np.arange(n)
    emitted = np.zeros((n, pool_size), dtype=bool)
    planned = rng.random((n, s)) < positive_rate
    key_idx = np.empty((n, s), dtype=np.int64)
    label = np.empty((n, s), dtype=bool)
    for t in range(s):
        emitted_pi = emitted[:, pi]                    # emitted_pi[r, x] == emitted[r, pi(x)]
        free = ~emitted
        cand_pos = free & emitted_pi
        cand_neg = free & ~emitted_pi
        has_pos, has_neg = cand_pos.any(1), cand_neg.any(1)
        want = planned[:, t]
        take_pos = (want & has_pos) | (~want & ~has_neg)
        cand = np.where(take_pos[:, None], cand_pos, cand_neg)
        if not cand.any(1).all():                      # unreachable: see the docstring
            raise AssertionError("presence emission found a row with no candidate key at step "
                                 f"{t}; every unemitted key belongs to exactly one class.")
        pick = np.argmax(np.where(cand, rng.random((n, pool_size)), -1.0), axis=1)
        key_idx[:, t] = pick
        label[:, t] = take_pos
        emitted[rows, pick] = True
    return key_idx, label, planned


def _place(rng, n, s, input_seq_len, min_separation):
    """`[n, s]` strictly increasing token positions with every gap >= `min_separation`.

    The standard order-statistics transform: draw `s` distinct positions from the SHRUNKEN span
    `L - (s-1)*(min_separation-1)`, sort them, and push the i-th one right by `i*(min_separation-1)`.
    That is a uniform draw over the min-gap configurations, and the largest position it can produce
    is exactly `L - 1`."""
    span = input_seq_len - (s - 1) * (min_separation - 1)
    base = np.sort(_distinct_per_row(rng, n, span, s), axis=1)
    return base + np.arange(s, dtype=np.int64)[None, :] * (min_separation - 1)


def build_episodes(
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    seed: int,
    *,
    demand: int = 16,
    keys_per_sequence: int = 0,
    positive_rate: float = 0.5,
    pi_seed: int = 12345,
    min_separation: int = 4,
    split: str = "train",
) -> PresenceEpisodes:
    """Episode ground truth for one segment. Separated from `presence_recall` so the gates measure
    the realized graph, the planted balance and the label rule directly."""
    s = _validate(vocab_size=vocab_size, input_seq_len=input_seq_len, demand=demand,
                  keys_per_sequence=keys_per_sequence, positive_rate=positive_rate,
                  min_separation=min_separation, split=split)
    pi = frozen_pi(demand, pi_seed)
    rng = np.random.default_rng(seed)
    key_pool = FIRST_FREE_TOKEN + np.arange(demand, dtype=np.int64)
    key_idx, label, planned = _emit(rng, num_examples, demand, s, pi, positive_rate)
    event_pos = _place(rng, num_examples, s, input_seq_len, min_separation)
    key_token = key_pool[key_idx]
    meta = dict(construction="presence", demand=demand, keys_per_sequence=s,
                positive_rate=positive_rate, pi_seed=pi_seed, min_separation=min_separation,
                input_seq_len=input_seq_len, vocab_size=vocab_size, split=split,
                pi=[int(x) for x in pi], pi_hash=pi_hash(pi))
    return PresenceEpisodes(key_token=key_token, event_pos=event_pos, label=label, planned=planned,
                            groups=key_token[:, None, :], key_pool=key_pool, pi=pi, meta=meta)


def brute_force_labels(key_token: np.ndarray, key_pool: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """The label rule, spelled out one position at a time: `pi(x_t)` somewhere strictly before `t`.

    The definition of the task, written independently of the emission machinery that produced the
    sequence, so the gates can check the fast path against it ([[matching-the-naive-antipattern]])."""
    lut = {int(tok): i for i, tok in enumerate(key_pool)}
    out = np.zeros(key_token.shape, dtype=bool)
    for r, row in enumerate(key_token):
        seen = set()
        for t, tok in enumerate(row):
            out[r, t] = int(key_pool[pi[lut[int(tok)]]]) in seen
            seen.add(int(tok))
    return out


def stamp(ep: PresenceEpisodes) -> dict:
    """The coverage + balance stamp for one segment's episodes.

    Two independent things are measured here and neither is inferred from a knob:
      * THE DEMAND -- the realized union graph and its solved chi (`graph_recall.demand_bounds`, i.e.
        `graph_chi.solve_chi` with a verified clique and a verified coloring). The drawn subset
        varies, so union completeness is a property of the SAMPLE; an under-sampled segment realizes
        a strict subgraph of K_K and says so in `graph_edges`.
      * THE REALIZED BALANCE -- the global positive rate, the worst per-POSITION deviation from it
        (`presence_position_bias`, the number that says the positional prior really is gone) and the
        rate at which a planted class had to fall back (`presence_forced_rate`).
    """
    nodes, adj = realized_graph(ep.groups)
    edges = int(adj.sum() // 2)
    d = demand_bounds(ep.groups)
    # measured over the LABELLED positions only: event 0 has an empty prefix, is a constant ABSENT,
    # and is not scored (see the module docstring), so including it would report a balance the metric
    # never sees.
    scored = ep.label[:, 1:]
    per_pos = scored.mean(axis=0)
    return {"graph_construction": "presence", "graph_nodes": int(len(nodes)), "graph_edges": edges,
            # EVERY key position carries a question, so the contract is exhaustive by construction
            # and every co-live pair is demanded apart. The fields are stamped in `graph_recall`'s
            # vocabulary so one analysis path reads both instruments.
            "graph_contract_edges": edges, "graph_contract_coverage": 1.0,
            "graph_contract_full": True, "graph_split": ep.meta["split"],
            "graph_demand_lower": d["lower"], "graph_demand_upper": d["upper"],
            "graph_demand_exact": d["exact"], "graph_demand_method": d["method"],
            "graph_predicted_N": (str(d["lower"]) if d["exact"]
                                  else f"{d['lower']}-{d['upper']}"),
            "graph_log2_demand": round(float(np.log2(max(d["lower"], 1))), 4),
            "graph_demand_nominal": ep.meta["demand"],
            "presence_keys_per_sequence": ep.meta["keys_per_sequence"],
            "presence_positive_rate": round(float(scored.mean()), 6),
            # the load-bearing balance number: the worst per-position deviation from 0.5. A layout
            # whose label rate rose with position would be readable without addressing anything, and
            # this is where that would show up.
            "presence_position_bias": round(float(np.abs(per_pos - 0.5).max()), 6),
            "presence_forced_rate": round(float((ep.label != ep.planned)[:, 1:].mean()), 6),
            "presence_chance": CHANCE, "presence_min_separation": ep.meta["min_separation"],
            "presence_pi_hash": ep.meta["pi_hash"],
            "presence_pi_derangement": bool((ep.pi != np.arange(len(ep.pi))).all())}


def presence_recall(
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    seed: int,
    demand: int = 16,
    keys_per_sequence: int = 0,
    positive_rate: float = 0.5,
    pi_seed: int = 12345,
    min_separation: int = 4,
    split: str = "train",
    include_slices: bool = True,
    **kwargs,
) -> DataSegment:
    """Render `build_episodes` into `(inputs, labels)` and stamp the realized graph and balance."""
    ep = build_episodes(vocab_size=vocab_size, num_examples=num_examples,
                        input_seq_len=input_seq_len, seed=seed, demand=demand,
                        keys_per_sequence=keys_per_sequence, positive_rate=positive_rate,
                        pi_seed=pi_seed, min_separation=min_separation, split=split)
    n = num_examples
    rng = np.random.default_rng(seed + 1_000_003)   # rendering noise, disjoint from the episode draw
    present, absent = label_tokens(vocab_size)

    inputs = np.full((n, input_seq_len), PAD_TOKEN, dtype=np.int64)
    labels = np.full((n, input_seq_len), -100, dtype=np.int64)
    np.put_along_axis(inputs, ep.event_pos, ep.key_token, axis=1)
    # ... and the label on every event BUT THE FIRST, whose prefix is empty and whose answer is
    # therefore the constant ABSENT (see the module docstring: scoring it would be a free 1/s of
    # accuracy and would move the balance off the 0.5 the metric is read against).
    np.put_along_axis(labels, ep.event_pos[:, 1:],
                      np.where(ep.label[:, 1:], present, absent), axis=1)

    # Filler EXCLUDES the key pool (a filler that coincided with a live key would be an uninstructed
    # event, silently rewriting the presence set) and the two reserved label ids (they are targets,
    # and an input token that IS the answer is a copy channel).
    holes = inputs == PAD_TOKEN
    n_holes = int(holes.sum())
    if n_holes:
        allowed = np.setdiff1d(np.arange(FIRST_FREE_TOKEN, vocab_size - 2), ep.key_pool)
        inputs[holes] = allowed[rng.integers(0, len(allowed), size=n_holes)]

    marks = stamp(ep)
    # THE COVERAGE GATE. The union graph of a presence cell is the COMPLETE graph on the pool, so the
    # closed-form family is `complete` and the solved chi must be K. Because the drawn subset varies,
    # a segment with too few examples genuinely realizes less than that -- and it is refused here,
    # on the dataset path, rather than discovered later as an unexplained curve.
    verify_construction("complete", dict(demand=demand), dict(
        lower=marks["graph_demand_lower"], upper=marks["graph_demand_upper"],
        exact=marks["graph_demand_exact"], method=marks["graph_demand_method"]))

    slices = {}
    if include_slices:
        slices = {"input_seq_len": input_seq_len,
                  "num_kv_pairs": int(ep.key_token.shape[1]), **marks}
    # Informational only. The AUTHORITATIVE copy is the one in `slices`, which the on-disk segment
    # cache stores and replays; this line is absent on a cache hit, so nothing downstream may key
    # off it.
    print("PRESENCE_RECALL_STAMP_JSON " + json.dumps(marks), flush=True)
    return DataSegment(torch.tensor(inputs), torch.tensor(labels), slices=slices)
