"""GRAPH RECALL — associative recall over a sequence distribution with a CHOSEN co-occurrence graph.

Built 2026-08-02. This is the data instrument for the claim that recall capacity is governed not by
the raw item count of a sequence but by the CHROMATIC NUMBER of the co-occurrence graph the task
realizes: nodes are key items, an edge joins two keys that some sequence forces to be separately
retrievable at the same time, and the quantity that must be paid for is the number of colors that
graph needs. `multiquery_ar` can only move the item count, and its graph is a clique by construction
at every setting, so it cannot separate the two.

SYMBOLS, defined at first use.
  `chi`  -- the DEMAND: the chromatic number of the realized union graph. The swept quantity.
  `|V|`  -- realized node count, i.e. distinct key tokens the dataset ever writes.
  `k`    -- `demand`, keys a single sequence forces apart (`complete`, `clustered`, `reuse`).
  `m`    -- `keys_per_class`, keys per cluster (`clustered`) or per birth slot (`overlap`).
  `w`    -- `demand` for `overlap`: the maximum number of simultaneously LIVE keys.
  `B`    -- `births`, key lifetimes started per sequence (`overlap`).
  `Lg`   -- `long_lived`, birth slots that are written once at the start and queried at the very end.
  `L`    -- `input_seq_len`. `N` -- addressable states per head of the model under test (not a knob
            here; the grid crosses it against `chi`).

THE FOUR CONSTRUCTIONS, and the graph each one realizes.

  `complete(k)`        -- the key pool IS `k` keys and every sequence writes and queries all of them.
                          The union graph is the complete graph K_k, so chi = k and |V| = k. This is
                          the standard MQAR regime with its graph made explicit. NOTE the deliberate
                          reading of "fresh keys": what is fresh per sequence is the key->VALUE
                          BINDING and the write/query order, NOT the key identities. Drawing fresh
                          key identities from a large pool (what `multiquery_ar` does) makes the
                          UNION graph complete on the whole pool, i.e. chi ~ |pool| and uncontrolled
                          -- which is exactly the coverage hazard this module verifies against.
  `clustered(k, m)`    -- `k` clusters of `m` keys each; a sequence draws a random transversal, one
                          key per cluster. The union graph is the complete k-partite graph
                          K_{m,...,m}: chi = k independent of m, while |V| = k*m. Run against
                          `complete(k)` this DECOUPLES the demand from the node count and from the
                          key-vocabulary size.
  `overlap(w, B, ...)` -- keys are born and expire; the live set is a sliding window, so the graph is
                          (a blow-up of) an interval graph and chi = w = the maximum number of keys
                          alive at one moment, however many keys the sequence writes in total. With
                          `long_lived = Lg > 0` the profile is MIXED: `Lg` slots are written once at
                          the start and queried at the very end while the rest churn through a window
                          of width `w - Lg`. Max simultaneous liveness is still exactly w. That cell
                          is the one that separates a WRITE-MASS clock from a TIME clock: a time
                          decay erodes the long-lived items (nothing re-touches them), a write-mass
                          decay does not (their slots receive no further mass).
  `reuse(pool, k, a)`  -- a fixed key pool reused across sequences with Zipf-like sharing
                          (`P(rank r) ~ r^-a`), the natural-distribution proxy. Its chi is NOT known
                          by construction; it is reported as an interval (see COVERAGE below) and the
                          cell is stamped with the bounds, never with a nominal number.

TOKEN LAYOUT. One uniform stride-2 EVENT grid of `L / 2` slots, two tokens per slot:

    WRITE(key, value) -> [key, value]        QUERY(key) -> [key, filler], label at the KEY position

    events (complete(3), L = 16, `.` = filler, `^` = a labelled position)

      slot   0     1     2     3     4     5     6     7
      tok  k1 v1 k2 v2 k3 v3 k2  .  .  .  k1  .  .  .  k3  .
      lab        .        .     ^v2        ^v1           ^v3

The value is never emitted in the query region, exactly as in `multiquery_ar` (there the same effect
is produced by building length L+1 arrays and slicing `[:-1]`/`[1:]`; constructing the aligned arrays
directly is the same contract with the shift written out). ONE renderer serves all four
constructions, including the interleaved `overlap` schedule that MQAR's context-block-then-query-
block layout cannot express.

THE CONTRACT -- which pairs the queries actually demand apart.

A point lookup demands separation NODE-wise: a query for `u` demands the item it answers from apart
from everything live beside it. So an edge `(u, v)` of the co-occurrence graph is ENFORCED when
either endpoint is read in a sequence realizing both, and the demand is solved on the ENFORCED graph,
never the nominal one. `graph_contract_coverage` stamps the ratio.

THE DEFAULT CONTRACT IS TOTAL. Every covered item of every sequence is read, on both sides: the
per-sequence read set is a permutation of the live items, so coverage is 1 by construction and the
question "could the model have predicted at write time which items would be asked about?" does not
arise -- all of them are. Predictability is harmless precisely because the coverage is total, which
is why the default is a permutation rather than a sample. EVAL is total in the same sense and for a
sharper reason: every aliased pair is counted in every eval sequence, so the measured collapse
threshold is sharp against chi instead of being smeared by sampling luck.

`queries_per_sequence > 0` selects the SAMPLED variant (that many reads per training sequence, drawn
uniformly without replacement). It exists as a control -- see the contract axis below -- and its
selection is uniform over items and over write positions, which is gated, because a sampled contract
that correlated with anything visible at write time WOULD be a leak.

THE CUE. By default the cue is the ITEM'S OWN KEY (`cue_permutation=False`): the query re-presents
the key that was written and the answer is that key's value. With `cue_permutation=True` a FIXED
random permutation `pi` over classes is drawn once per cell (from `cue_permutation_seed`, never from
the segment seed, so train and eval share it) and a query presenting cue `u` must answer with the
value of item `pi(u)`. The storage demand is unchanged -- every value must still be separately
retrievable -- so the cell isolates the ADDRESSING map from the storage question. It is defined for
`complete` and `clustered` only; `overlap` would let `pi(u)` name an item that has already expired
(unanswerable, not harder) and `reuse` holds a different subset of its pool in every sequence, so a
frozen map over classes is not well defined there. Both are refused by name.

TEMPORAL SEPARATION IS ASSERTED, not assumed (`_assert_separation`, `min_separation`, default 2 event
slots = 4 tokens). A read adjacent to its own write is served straight out of the residual stream and
measures nothing about storage. The assertion has two parts: every read trails the write it answers
by at least `min_separation` slots, and at most ONE item per episode is read with no other item's
WRITE in between. The exception is structural, not a concession -- some item is written last in any
causal layout and nothing can follow it, so all-but-one is the strongest true form of the statement.

THE CONTRACT AXIS. `contract_fraction < 1` covers only the first `f * n_classes` classes, dataset-wide
and on BOTH sides. The distribution is untouched and the demanded-apart relation shrinks, so the
solver's chi drops and the prediction is that the measured threshold follows the INDUCED chi rather
than the full graph's -- the contract-side twin of the `complete` / `clustered` decoupling pair. Note
what the node-wise rule implies and the solver reports: covering half of `complete(k)` leaves every
edge that touches a covered item, i.e. K_{k/2} joined to an independent set, whose chi is `k/2 + 1`
and not `k/2`. MEASURED, and the reason the grid carries a control: a per-sequence RESAMPLED contract
does NOT reduce the union demand (every pair is demanded apart by some sequence, coverage returns to
1) while a fixed covered subset does.

A MIXTURE contract (a query whose target is a fixed mixture of stored values -- the nonnegative-rank
regime) is NOT implemented here: it needs mixture-valued evaluation plumbing that neither zoology's
`compute_metrics` nor the label format has. Recorded as a design note in the workflow findings, not
half-built.

COVERAGE VERIFICATION, and why the nominal knob is not trusted. A model experiences the UNION of the
sampled sequences' co-occurrence patches, which is a SUBGRAPH of the construction's target graph --
`clustered` only realizes a cross-cluster edge when some sequence happens to draw both endpoints, so
a small `num_examples` can silently realize a smaller graph than the knob names. Every segment
therefore MEASURES its own realized graph and stamps it into `DataSegment.slices`:
`graph_nodes`, `graph_edges`, `graph_demand_lower`, `graph_demand_upper`, `graph_demand_exact`,
`graph_demand_nominal`, `graph_construction`. Cells trust `graph_demand_*`, never the knob.

The stamp is a property of the SEGMENT, not of the knob set: `reuse`'s realized demand grows with
`num_examples` (more sequences realize more of the Zipf tail's co-occurrences), so a 20k-example
train segment and a 1k-example test segment of one cell legitimately carry different demands. Both
are stamped and both are true; averaging them or quoting the train one for the test slice would not
be.

The demand is SOLVED, by `zoology.data.graph_chi.solve_chi`, which returns `[lower, upper]` plus the
argument that produced it and the certificates that make it checkable (a verified clique for the
lower bound, a verified proper coloring for the upper). It recognizes each construction's structure
and proves the answer where a proof exists -- `complete` as disjoint cliques, `clustered` as complete
multipartite, `overlap` as a chordal graph (on its false-twin quotient, since blowing a vertex up
into `keys_per_class` twins destroys chordality while preserving perfection and therefore chi) --
and falls back to DSATUR-ordered exact branch-and-bound under a node budget, then to the certified
interval, for `reuse`, whose realized graph has no closed form. MEASURED: the law-tier `reuse` cell
(512 nodes, 64k edges) closes exactly, in 1.5 s, inside the data build.

Then the closed-form families are GATED: `graph_chi.verify_construction` demands the solved chi equal
the construction's formula and raises `DemandMismatch` otherwise. A cell whose realized graph is not
the graph its construction claims is refused rather than shipped with the wrong x-coordinate.

CONFLICT GROUPS are the one primitive behind all of this. An episode emits sets of keys that some
moment of it forces apart: one group (all the episode's keys) for complete/clustered/reuse, one group
per live window for overlap. The union graph is the union of those cliques. Four constructions, one
verification path, and the clique lower bound is exact by construction rather than by search.

DETERMINISM. Every draw comes from an `np.random.default_rng(seed)` stream local to the call -- never
`np.random.seed` (global numpy) and never an unseeded `torch.randint` (global torch). Both of those
were live bugs in `multiquery_ar` until 2026-08-02; see its "DETERMINISM" docstring section and
`rola_bench/test_mqar_determinism.py`. `gen_version` is a plain pydantic field, so it participates in
`model_dump()` and therefore in the on-disk cache key (`zoology/data/utils.py`) -- bumping it
invalidates stale caches without deleting anyone's files.
"""
import json
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch

from zoology.config import DataSegmentConfig
from zoology.data.graph_chi import verify_construction
from zoology.data.utils import DataSegment

#: Bump when the DRAW SEQUENCE or the token layout changes. Participates in the on-disk cache key.
GRAPH_RECALL_GEN_VERSION = 1

#: Reserved token ids. PAD is the build-time placeholder (zoology's convention) and is fully
#: overwritten before the segment is returned.
PAD_TOKEN = 0
FIRST_FREE_TOKEN = 1

CONSTRUCTIONS = ("complete", "clustered", "overlap", "reuse")
#: The two sides of the contract. `train` draws random point-lookup targets; `eval` is EXHAUSTIVE
#: over the contract's items. Set by the segment builder, never by a spec -- it is a property of
#: which segment this is. See "THE CONTRACT" in the module docstring.
SPLITS = ("train", "eval")

#: Stamped fields, in the order they are printed. `graph_demand_lower/upper` are the coordinate a
#: cell is read at; `graph_demand_nominal` is the knob, kept only so the two can be compared.
STAMP_FIELDS = ("graph_construction", "graph_nodes", "graph_edges", "graph_contract_edges",
                "graph_contract_coverage", "graph_contract_full", "graph_contract_fraction",
                "graph_cue_permuted", "graph_min_separation", "graph_split", "graph_demand_lower",
                "graph_demand_upper", "graph_demand_exact", "graph_demand_method",
                "graph_predicted_N", "graph_log2_demand", "graph_demand_nominal")


class GraphRecallConfig(DataSegmentConfig):
    name: str = "graph_recall"
    construction: Literal["complete", "clustered", "overlap", "reuse"] = "complete"
    demand: int = 16                 # k (complete/clustered/reuse) or w (overlap)
    keys_per_class: int = 1          # m: keys per cluster (clustered) / per birth slot (overlap)
    births: int = 0                  # B (overlap); 0 -> the default 2 * demand
    long_lived: int = 0              # Lg (overlap); > 0 is the MIXED-lifetime cell
    pool_size: int = 0               # reuse: key-pool size; 0 -> 8 * demand
    zipf_a: float = 1.0              # reuse: P(rank r) ~ r^-zipf_a
    power_a: float = 0.01            # query-gap power law, `multiquery_ar` semantics
    contract_fraction: float = 1.0   # the fraction of CLASSES the contract ever asks for
    queries_per_sequence: int = 0    # 0 = TOTAL coverage (the default); >0 = the sampled variant
    cue_permutation: bool = False    # False = identity cue; True = answer with pi(u)'s value
    cue_permutation_seed: int = 12345  # pi is FROZEN per cell -- never keyed off the segment seed
    min_separation: int = 2          # event slots a query must trail its write by
    split: Literal["train", "eval"] = "train"   # set by the segment builder, not by a spec
    include_slices: bool = True
    gen_version: int = GRAPH_RECALL_GEN_VERSION

    def build(self, seed: int) -> DataSegment:
        return graph_recall(**self.model_dump(), seed=seed)


@dataclass
class GraphEpisodes:
    """The episode ground truth, before tokenization.

    Exposed so the gates measure the ACTUAL arrays the renderer consumes rather than a
    re-derivation of them ([[matching-the-naive-antipattern]]): a second implementation of the
    schedule would only ever prove itself self-consistent.
    """

    write_key: np.ndarray       # [n, W]  key token, in write order
    write_value: np.ndarray     # [n, W]  its value token
    write_slot: np.ndarray      # [n, W]  event slot of the write
    query_key: np.ndarray       # [n, Q]  key token queried
    query_value: np.ndarray     # [n, Q]  the value the contract names
    query_slot: np.ndarray      # [n, Q]  event slot of the query
    query_target_key: np.ndarray  # [n, Q] the key of the item the answer comes FROM (pi(cue); the
                                #         cue itself under the default identity contract)
    groups: np.ndarray          # [n, G, gmax] conflict groups, key tokens, -1 padded
    key_pool: np.ndarray        # [|pool|] every key token the construction may ever use
    meta: dict = field(default_factory=dict)


# --- sampling primitives (shared with cue_consistency / multiquery_ar, same identities) -----------
def _distinct_per_row(rng, n_rows, universe, k, chunk=8192):
    """`[n_rows, k]` integers in `[0, universe)`, distinct WITHIN a row, uniform.

    The argsort-of-random-keys identity for a per-row `np.random.choice(replace=False)`, chunked so
    the transient key matrix stays bounded. Byte-identical in form to
    `zoology.data.cue_consistency._distinct_per_row`."""
    if k > universe:
        raise ValueError(f"cannot draw {k} distinct values from a universe of {universe}")
    out = np.empty((n_rows, k), dtype=np.int64)
    for lo in range(0, n_rows, chunk):
        hi = min(lo + chunk, n_rows)
        out[lo:hi] = np.argsort(rng.random((hi - lo, universe)), axis=1)[:, :k]
    return out


def _weighted_distinct(rng, n_rows, weights, k):
    """Weighted sampling WITHOUT replacement, `k` per row, over `len(weights)` positions.

    The Gumbel-top-k identity, which reproduces `np.random.choice(p=weights, replace=False)`
    exactly, vectorized over rows."""
    g = -np.log(-np.log(rng.random((n_rows, len(weights)))))
    return np.argsort(-(np.log(weights)[None, :] + g), axis=1)[:, :k]


# =================================================================== the realized graph
def enforced_mask(groups: np.ndarray, query_key: np.ndarray) -> np.ndarray:
    """`[n, G, gmax]` bool: was this group member QUERIED in this episode?

    A point lookup demands separation NODE-wise -- a query for `u` demands `u` apart from every item
    live beside it -- so an edge `(u, v)` is enforced when EITHER endpoint is queried in a sequence
    that realizes both. That is what `realized_graph(..., enforced=...)` consumes."""
    n = groups.shape[0]
    width = int(max(int(groups.max()), int(query_key.max()))) + 2
    lut = np.zeros((n, width), dtype=bool)
    np.put_along_axis(lut, query_key, True, axis=1)
    flat = groups.reshape(n, -1)
    got = np.take_along_axis(lut, np.clip(flat, 0, None), axis=1) & (flat >= 0)
    return got.reshape(groups.shape)


def realized_graph(groups: np.ndarray, enforced: np.ndarray | None = None) -> tuple:
    """`(nodes, adjacency)` for the union of the cliques induced by `groups` ([n, G, gmax], -1 pad).

    With `enforced` (same shape, boolean) an edge is added only when at least ONE endpoint is
    enforced in that group -- the contract graph, i.e. the pairs some query actually demanded apart.
    Without it, every co-live pair is an edge -- the TARGET graph the distribution realizes. The two
    differ only under a partial contract, and their ratio is the stamped contract coverage.

    `nodes` are the distinct key tokens in construction order (ascending token id, which for every
    construction here is also the pool order -- birth order for `overlap`, which is what makes the
    insertion-order greedy coloring optimal there). `adjacency` is a dense boolean `[|V|, |V|]`
    matrix: the pools are 16-1024 keys wide, so a dense matrix is both smaller and faster than an
    edge list of the tens of millions of (mostly duplicate) pairs the groups induce, and it makes the
    coloring below a pure numpy walk.
    """
    flat = groups.reshape(-1, groups.shape[-1])
    enf = None if enforced is None else enforced.reshape(-1, groups.shape[-1])
    nodes = np.unique(flat[flat >= 0])
    index = {int(t): i for i, t in enumerate(nodes)}
    v = len(nodes)
    adj = np.zeros((v, v), dtype=bool)
    lut = np.full(int(nodes.max()) + 2 if v else 2, -1, dtype=np.int64)
    for tok, i in index.items():
        lut[tok] = i
    # Chunked so the transient pair arrays stay bounded on a 100k-example segment.
    for lo in range(0, len(flat), 4096):
        blk = flat[lo:lo + 4096]
        eblk = None if enf is None else enf[lo:lo + 4096]
        idx = np.where(blk >= 0, lut[np.clip(blk, 0, None)], -1)
        for a in range(blk.shape[1]):
            for b in range(a + 1, blk.shape[1]):
                ia, ib = idx[:, a], idx[:, b]
                ok = (ia >= 0) & (ib >= 0) & (ia != ib)
                if eblk is not None:
                    ok = ok & (eblk[:, a] | eblk[:, b])
                adj[ia[ok], ib[ok]] = True
                adj[ib[ok], ia[ok]] = True
    return nodes, adj


def demand_bounds(groups: np.ndarray, enforced: np.ndarray | None = None) -> dict:
    """Measure the realized union graph's demand. See the module docstring, COVERAGE VERIFICATION.

    Returns `{nodes, edges, lower, upper, exact, method}` where `lower <= chi <= upper`, `exact` says
    the interval collapsed and `method` names the argument that produced it. The solving is
    `zoology.data.graph_chi.solve_chi` -- a separate module because it is a general graph question
    with its own certificates and its own proof obligations, and because the analysis side
    (`rola_bench.mqar.graph_analysis`) asks it the same question about the same graphs.

    Computing chi is NP-hard in general, so the interval is the honest object and exactness is a
    derived FACT about a particular realized graph -- never a claim the construction is allowed to
    make about itself. A conflict group is a clique by construction, so `max |group|` is a free
    lower bound and is passed to the solver as a floor.
    """
    from zoology.data.graph_chi import solve_chi

    nodes, adj = realized_graph(groups, enforced)
    if len(nodes) == 0:
        return dict(nodes=0, edges=0, lower=0, upper=0, exact=True, method="empty")
    res = solve_chi(adj)
    lower = int(res["lower"])
    if enforced is None:            # a conflict group is a clique, so its size is a free bound
        lower = max(lower, int((groups >= 0).sum(-1).max()))
    return dict(nodes=int(len(nodes)), edges=int(adj.sum() // 2), lower=lower,
                upper=int(res["upper"]), exact=bool(lower == int(res["upper"])),
                method=res["method"])


# =================================================================== the constructions
def _validate(*, vocab_size, input_seq_len, construction, demand, keys_per_class, births,
              long_lived, pool_size, zipf_a, contract_fraction=1.0, queries_per_sequence=0,
              split="train", cue_permutation=False, min_separation=2):
    if construction not in CONSTRUCTIONS:
        raise ValueError(f"construction={construction!r} not in {CONSTRUCTIONS}.")
    if split not in SPLITS:
        raise ValueError(f"split={split!r} not in {SPLITS}.")
    if min_separation < 1:
        raise ValueError(
            f"min_separation={min_separation} must be >= 1: a query adjacent to its own write is "
            "solvable straight out of the residual stream and measures nothing about storage.")
    if cue_permutation and construction not in ("complete", "clustered"):
        raise ValueError(
            f"cue_permutation is defined for `complete` and `clustered` only, not {construction!r}: "
            "pi maps a cue to ANOTHER item, and in `overlap` that item may have expired by the time "
            "the cue is presented (its value is not in the state at all, so the cell would be "
            "unanswerable rather than harder), while in `reuse` a sequence holds a different subset "
            "of the pool every time so a frozen pi over classes is not even well defined.")
    if not 0 < contract_fraction <= 1:
        raise ValueError(f"contract_fraction={contract_fraction} must satisfy 0 < f <= 1.")
    if construction == "reuse" and contract_fraction < 1:
        raise ValueError(
            "contract_fraction has no meaning for `reuse`: its classes are Zipf RANKS and a sequence "
            "draws a different subset of them every time, so 'the covered classes' is not a fixed set "
            "of queried items and the number of queries per episode would vary. The contract axis is "
            "defined on the constructions with fixed classes (complete / clustered / overlap).")
    if input_seq_len % 2:
        raise ValueError(f"input_seq_len={input_seq_len} must be even (the event grid is stride 2).")
    if demand < 1:
        raise ValueError(f"demand={demand} must be >= 1.")
    if keys_per_class < 1:
        raise ValueError(f"keys_per_class={keys_per_class} must be >= 1.")
    slots = input_seq_len // 2
    if construction == "overlap":
        b = births or 2 * demand
        if long_lived >= demand:
            raise ValueError(
                f"long_lived={long_lived} must be < demand={demand}: the churn window is "
                f"demand - long_lived and a width-0 window has no keys to churn, which would make "
                "the realized demand long_lived, not demand.")
        if b < demand:
            raise ValueError(
                f"births={b} < demand={demand}: fewer lifetimes are started than the window is wide, "
                f"so at most {b} keys are ever simultaneously live and the realized demand would be "
                f"{b}. Raise births or lower demand rather than shipping a cell whose nominal knob "
                "the coverage stamp will contradict.")
        events, n_keys = 2 * b, b * keys_per_class
    elif construction == "clustered":
        b, events, n_keys = 0, 2 * demand, demand * keys_per_class
    elif construction == "reuse":
        b, events = 0, 2 * demand
        n_keys = pool_size or 8 * demand
        if n_keys < demand:
            raise ValueError(f"pool_size={n_keys} < demand={demand}: a sequence cannot draw "
                             f"{demand} distinct keys from a pool of {n_keys}.")
        if zipf_a < 0:
            raise ValueError(f"zipf_a={zipf_a} must be >= 0.")
    else:
        b, events, n_keys = 0, 2 * demand, demand
    if events > slots:
        raise ValueError(
            f"the episode needs {events} event slots of 2 tokens each = {2 * events} tokens, but "
            f"input_seq_len={input_seq_len} provides {slots}. Structural: raise L or lower the "
            "construction's size, rather than letting the build assert deep inside the schedule.")
    key_region = vocab_size // 2 - FIRST_FREE_TOKEN
    if n_keys > key_region:
        raise ValueError(
            f"the construction needs a key pool of {n_keys}, but the key half of the vocabulary "
            f"holds {key_region} tokens. Raise vocab_size.")
    if vocab_size <= input_seq_len:
        raise ValueError(f"vocab_size={vocab_size} must exceed input_seq_len={input_seq_len}.")
    return slots, events, n_keys, b


def _queried_classes(rng, n, n_classes, contract_fraction, queries_per_sequence, split):
    """`[n, Q]` CLASS indices this episode asks for. See "THE CONTRACT" in the module docstring.

    A class is one addressable slot of the construction: a key for `complete`, a cluster for
    `clustered`, a birth slot for `overlap`. The contract COVERS the first
    `ceil(contract_fraction * n_classes)` of them, dataset-wide and fixed; within that cover,

      * `split='eval'`  -- EXHAUSTIVE. Every covered item of every eval sequence is queried, so every
        aliased pair is counted every time and the measured collapse threshold is sharp against chi.
        Sampling eval targets instead would smear sub-chi degradation by sampling luck.
      * `split='train'` -- `queries_per_sequence` targets drawn UNIFORMLY WITHOUT REPLACEMENT from the
        covered items (0 = all of them). The load-bearing property is WRITE-TIME UNPREDICTABILITY:
        the probability an item is queried must not correlate with anything visible when it is
        written -- not its position, not its key identity, not its cluster. A correlated target
        distribution lets the model alias the low-probability pairs, and the effective demand drops
        below the solved chi, which would look like the instrument failing rather than the contract
        leaking. Uniform-without-replacement over the covered set has that property exactly, and it
        is gated empirically (flatness over items AND over write positions).
    """
    covered = max(1, int(round(contract_fraction * n_classes)))
    if split == "eval" or not queries_per_sequence or queries_per_sequence >= covered:
        return np.broadcast_to(np.arange(covered), (n, covered)).copy()
    return _distinct_per_row(rng, n, covered, queries_per_sequence)


def _block_episode(rng, n, keys_row, value_pool, slots, power_a, contract, pi, min_separation):
    """The shared layout for `complete` / `clustered` / `reuse`: a write block, then power-law-placed
    queries. `keys_row` is `[n, k]`, the keys THIS sequence writes, column `j` being class `j`.

    Query placement reuses `multiquery_ar`'s power-law gap distribution verbatim (same `power_a`
    semantics), so the gap statistics of a graph-recall sequence and an MQAR sequence of the same
    (L, k) are the same distribution and any difference between the two tasks is the GRAPH.
    """
    k = keys_row.shape[1]
    order = _distinct_per_row(rng, n, k, k)                       # write order: a permutation
    rows = np.arange(n)[:, None]
    write_key = keys_row[rows, order]
    write_value = value_pool[_distinct_per_row(rng, n, len(value_pool), k)]
    write_slot = np.broadcast_to(np.arange(k), (n, k)).copy()
    where = np.argsort(order, axis=1)                             # class -> its position in the writes

    cls = _queried_classes(rng, n, k, *contract)
    target = cls if pi is None else pi[cls]                       # the item the answer comes from
    query_key = keys_row[rows, cls]                               # the CUE the query presents
    query_target_key = keys_row[rows, target]
    query_value = write_value[rows, np.take_along_axis(where, target, axis=1)]
    # TEMPORAL SEPARATION: the query region starts `min_separation` slots after the last write, so
    # even the LAST-written item's query is that many event slots (2x that many tokens) away from it.
    # Without the offset a query landing on the first free slot would sit immediately after its own
    # write and be answerable out of the residual stream.
    space = slots - k - (min_separation - 1)
    p = power_a * np.arange(1, space + 1) ** (power_a - 1)
    gaps = _weighted_distinct(rng, n, p / p.sum(), cls.shape[1])
    query_slot = k + (min_separation - 1) + gaps
    groups = keys_row[:, None, :]                                 # ONE group: every key of the episode
    return (write_key, write_value, write_slot, query_key, query_value, query_slot, groups,
            query_target_key)


def _assert_separation(write_key, write_slot, target_key, query_slot, min_separation):
    """Raise unless every query trails the write of the item it ANSWERS by `min_separation` slots.

    Also checks the stronger statement where it can hold: at most ONE item per episode may be read
    with no other item's WRITE in between. Some item is written last in any causal layout and nothing
    can follow it, so demanding an intervening write for every item is unsatisfiable; all-but-one is
    the strongest true form, and it is what rules out a layout whose reads all sit in the tail.
    """
    n = len(write_key)
    width = int(max(int(write_key.max()), int(target_key.max()))) + 2
    lut = np.full((n, width), -1, dtype=np.int64)
    np.put_along_axis(lut, write_key, write_slot, axis=1)
    src = np.take_along_axis(lut, target_key, axis=1)          # write slot of the answering item
    gap = query_slot - src
    if gap.min() < min_separation:
        raise AssertionError(
            f"a query sits {int(gap.min())} event slot(s) after the write it answers, below "
            f"min_separation={min_separation}. A read adjacent to its own write is served out of the "
            "residual stream and measures nothing about storage.")
    # an item has an intervening write iff some write is later than its own, i.e. iff it is not the
    # episode's final write
    unseparated = (src == write_slot.max(axis=1, keepdims=True)).sum(1)
    if unseparated.max() > 1:
        raise AssertionError(
            f"{int(unseparated.max())} reads in an episode have NO intervening write; at most one "
            "(the last-written item's) can, by causality.")


def _overlap_schedule(demand, births, long_lived):
    """The event schedule for `overlap`, as index arrays over the `2 * births` events.

    Returns `(write_of_event, query_of_event, groups_idx)` where the first two map an event index to
    a birth-slot index (or -1), and `groups_idx` is `[G, demand]` (-1 padded) listing the slot
    indices simultaneously live at each moment.

    The schedule: `Lg` long-lived slots are written first and queried last; the remaining
    `Bc = births - Lg` churn slots stream through a window of width `wc = demand - Lg` -- write slot
    `i`, and once `i >= wc` also query slot `i - wc`, so exactly `wc` churn slots are live from then
    on. Maximum simultaneous liveness is `Lg + wc = demand`, exactly, at every setting.
    """
    lg, wc = long_lived, demand - long_lived
    bc = births - lg
    writes, queries = [], []
    for i in range(lg):                                # the long-lived block, written up front
        writes.append(i)
        queries.append(-1)
    for i in range(bc):                                # the churn stream
        writes.append(lg + i)
        queries.append(lg + i - wc if i >= wc else -1)
    for j in range(max(bc - wc, 0), bc):               # flush the window still open at the end
        writes.append(-1)
        queries.append(lg + j)
    for i in range(lg):                                # ... and the long-lived queries, at the very end
        writes.append(-1)
        queries.append(i)
    groups = []
    for i in range(bc):
        live = list(range(lg)) + [lg + j for j in range(max(0, i - wc + 1), i + 1)]
        groups.append(live + [-1] * (demand - len(live)))
    if not groups:                                     # wc == demand and bc == 0 cannot happen
        groups = [list(range(lg)) + [-1] * (demand - lg)]
    return (np.array(writes, dtype=np.int64), np.array(queries, dtype=np.int64),
            np.array(groups, dtype=np.int64))


def build_episodes(
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    seed: int,
    *,
    construction: str = "complete",
    demand: int = 16,
    keys_per_class: int = 1,
    births: int = 0,
    long_lived: int = 0,
    pool_size: int = 0,
    zipf_a: float = 1.0,
    power_a: float = 0.01,
    contract_fraction: float = 1.0,
    queries_per_sequence: int = 0,
    split: str = "train",
    cue_permutation: bool = False,
    cue_permutation_seed: int = 12345,
    min_separation: int = 2,
) -> GraphEpisodes:
    """Episode ground truth for one segment. Separated from `graph_recall` so the gates can measure
    the realized graph, the liveness schedule and the write/query pairing directly."""
    slots, events, n_keys, b = _validate(
        vocab_size=vocab_size, input_seq_len=input_seq_len, construction=construction,
        demand=demand, keys_per_class=keys_per_class, births=births, long_lived=long_lived,
        pool_size=pool_size, zipf_a=zipf_a, contract_fraction=contract_fraction,
        queries_per_sequence=queries_per_sequence, split=split, cue_permutation=cue_permutation,
        min_separation=min_separation)
    contract = (contract_fraction, queries_per_sequence, split)
    # pi is FROZEN per cell -- drawn from `cue_permutation_seed`, never from the segment `seed`, so
    # the train segment and every eval segment of one cell share the same correspondence. Keying it
    # off `seed` would give each segment its own map and the cell would be unlearnable by
    # construction rather than by capacity.
    pi = (np.random.default_rng(cue_permutation_seed).permutation(demand)
          if cue_permutation else None)
    rng = np.random.default_rng(seed)
    n, k = num_examples, demand
    rows = np.arange(n)[:, None]
    key_pool = FIRST_FREE_TOKEN + np.arange(n_keys, dtype=np.int64)
    value_pool = np.arange(vocab_size // 2, vocab_size, dtype=np.int64)

    if construction == "complete":
        keys_row = np.broadcast_to(key_pool, (n, k)).copy()
        parts = _block_episode(rng, n, keys_row, value_pool, slots, power_a, contract, pi,
                               min_separation)
    elif construction == "clustered":
        # one key per cluster: cluster c owns pool[c*m : (c+1)*m]
        pick = rng.integers(0, keys_per_class, size=(n, k))
        keys_row = key_pool[np.arange(k)[None, :] * keys_per_class + pick]
        parts = _block_episode(rng, n, keys_row, value_pool, slots, power_a, contract, pi,
                               min_separation)
    elif construction == "reuse":
        w = (np.arange(1, n_keys + 1, dtype=np.float64)) ** (-zipf_a)
        keys_row = key_pool[_weighted_distinct(rng, n, w / w.sum(), k)]
        parts = _block_episode(rng, n, keys_row, value_pool, slots, power_a, contract, pi,
                               min_separation)
    else:
        parts = None

    if parts is not None:
        (write_key, write_value, write_slot, query_key, query_value, query_slot, groups,
         query_target_key) = parts
    else:
        # --- overlap: an interleaved lifetime schedule -------------------------------------------
        w_of_e, q_of_e, grp_idx = _overlap_schedule(k, b, long_lived)
        # one key per birth slot, drawn from that slot's own disjoint sub-pool (the blow-up: it
        # widens |V| by `keys_per_class` and leaves chi untouched, since blowing up a vertex of a
        # perfect graph cannot raise its chromatic number)
        pick = rng.integers(0, keys_per_class, size=(n, b))
        slot_key = key_pool[np.arange(b)[None, :] * keys_per_class + pick]        # [n, B]
        slot_value = value_pool[_distinct_per_row(rng, n, len(value_pool), b)]    # [n, B]
        # the 2B events are placed on a random INCREASING injection into the slot grid, so the
        # schedule's order (and therefore the liveness structure) is preserved while the gaps vary
        place = np.sort(_distinct_per_row(rng, n, slots, len(w_of_e)), axis=1)    # [n, 2B]
        wmask, qmask = w_of_e >= 0, q_of_e >= 0
        write_key = slot_key[rows, w_of_e[wmask][None, :]]
        write_value = slot_value[rows, w_of_e[wmask][None, :]]
        write_slot = place[:, wmask]
        # THE CONTRACT AXIS on the lifetime schedule: the queried BIRTH SLOTS are a subset, and the
        # query events of the others simply do not occur (their positions become filler). The
        # liveness structure -- and therefore the TARGET graph -- is untouched; what changes is which
        # pairs the contract demands apart.
        q_slots = q_of_e[qmask]                                    # slot index per query column
        full_key = slot_key[rows, q_slots[None, :]]
        full_value = slot_value[rows, q_slots[None, :]]
        full_slot = place[:, qmask]
        cls = _queried_classes(rng, n, b, *contract)
        if cls.shape[1] == b:
            query_key, query_value, query_slot = full_key, full_value, full_slot
        else:
            col_of_slot = np.argsort(q_slots)                      # slot -> its query column
            take = col_of_slot[cls]
            query_key = np.take_along_axis(full_key, take, axis=1)
            query_value = np.take_along_axis(full_value, take, axis=1)
            query_slot = np.take_along_axis(full_slot, take, axis=1)
        groups = np.where(grp_idx[None, :, :] >= 0,
                          slot_key[:, np.clip(grp_idx, 0, None)], -1)
        query_target_key = query_key      # identity cue: `overlap` admits no pi (see `_validate`)

    # THE SEPARATION ASSERTION, on the arrays the renderer consumes. Two statements, both checked:
    # (1) every query trails the write of the item it ANSWERS by at least `min_separation` event
    # slots, and (2) every item but one has at least one other item's WRITE between the two. The
    # exception is structural rather than a concession -- some item is written last in any causal
    # layout, and no write can follow it.
    _assert_separation(write_key, write_slot, query_target_key, query_slot, min_separation)

    meta = dict(construction=construction, demand=demand, keys_per_class=keys_per_class,
                births=b, long_lived=long_lived, pool_size=n_keys if construction == "reuse" else 0,
                zipf_a=zipf_a, power_a=power_a, input_seq_len=input_seq_len, vocab_size=vocab_size,
                event_slots=slots, events=events, key_pool_size=n_keys,
                contract_fraction=contract_fraction, cue_permutation=cue_permutation,
                min_separation=min_separation,
                queries_per_sequence=queries_per_sequence, split=split,
                covered_classes=max(1, int(round(contract_fraction * (b or demand)))))
    return GraphEpisodes(write_key=write_key, write_value=write_value, write_slot=write_slot,
                         query_key=query_key, query_value=query_value, query_slot=query_slot,
                         query_target_key=query_target_key, groups=groups, key_pool=key_pool,
                         meta=meta)


def stamp(ep: GraphEpisodes) -> dict:
    """The coverage stamp for one segment's episodes -- the measured realized graph.

    `graph_demand_nominal` is the demand the CONSTRUCTION claims, and is `0` for `reuse`, which
    claims none: its `demand` knob is keys-per-sequence, and the realized chi of a Zipf-reuse pool is
    an emergent property an order of magnitude above it (measured: k = 16 over a 512-key pool at
    zipf_a = 1.5 realizes chi in [124, 126]). Stamping the knob there would invite a cell to be read
    at a demand it does not have -- the exact substitution this field exists to make impossible.
    """
    enforced = enforced_mask(ep.groups, ep.query_key)
    target_nodes, target_adj = realized_graph(ep.groups)
    target_edges = int(target_adj.sum() // 2)
    d = demand_bounds(ep.groups, enforced)          # the CONTRACT graph: what queries demand apart
    coverage = 1.0 if target_edges == 0 else d["edges"] / target_edges
    return {"graph_construction": ep.meta["construction"], "graph_nodes": len(target_nodes),
            "graph_edges": target_edges,
            # THE CONTRACT COVERAGE: the fraction of the distribution's co-live pairs that the
            # realized queries actually demand apart. The demand below is solved on the CONTRACT
            # graph, never on the nominal one -- an under-covered contract would let the model alias
            # the unasked pairs and the measured threshold would undercut the solved chi.
            "graph_contract_edges": d["edges"], "graph_contract_coverage": round(coverage, 6),
            "graph_contract_full": bool(abs(coverage - 1.0) < 1e-9),
            "graph_contract_fraction": ep.meta["contract_fraction"],
            "graph_cue_permuted": ep.meta["cue_permutation"],
            "graph_min_separation": ep.meta["min_separation"],
            "graph_split": ep.meta["split"],
            "graph_demand_lower": d["lower"],
            "graph_demand_upper": d["upper"], "graph_demand_exact": d["exact"],
            "graph_demand_method": d["method"],
            # `graph_predicted_N` is the STATE COUNT the theory predicts this cell needs: the solved
            # demand when it is exact, and the interval's midpoint-free spelling "lo-hi" when it is
            # not. It is the quantity the grid's measured collapse threshold is compared against
            # (`rola_bench.mqar.graph_analysis`), which is why it is stamped as its own field rather
            # than left to be re-derived from the bounds by whoever plots them.
            "graph_predicted_N": (str(d["lower"]) if d["exact"]
                                  else f"{d['lower']}-{d['upper']}"),
            # log2 of the solved demand: the ATTENTION reference's capacity axis is predicted to be
            # linear in it (an attention head packs ~exp(c*d_qk) addresses, so its collapse threshold
            # sits at d_qk* ~ log(chi)/c). Stamped beside chi so the two-axis fit reads one field.
            "graph_log2_demand": round(float(np.log2(max(d["lower"], 1))), 4),
            "graph_demand_nominal": 0 if ep.meta["construction"] == "reuse" else ep.meta["demand"]}


def graph_recall(
    vocab_size: int,
    num_examples: int,
    input_seq_len: int,
    seed: int,
    construction: str = "complete",
    demand: int = 16,
    keys_per_class: int = 1,
    births: int = 0,
    long_lived: int = 0,
    pool_size: int = 0,
    zipf_a: float = 1.0,
    power_a: float = 0.01,
    contract_fraction: float = 1.0,
    queries_per_sequence: int = 0,
    cue_permutation: bool = False,
    cue_permutation_seed: int = 12345,
    min_separation: int = 2,
    split: str = "train",
    include_slices: bool = True,
    **kwargs,
) -> DataSegment:
    """Render `build_episodes` into `(inputs, labels)` and stamp the realized graph into `slices`."""
    ep = build_episodes(
        vocab_size=vocab_size, num_examples=num_examples, input_seq_len=input_seq_len, seed=seed,
        construction=construction, demand=demand, keys_per_class=keys_per_class, births=births,
        long_lived=long_lived, pool_size=pool_size, zipf_a=zipf_a, power_a=power_a,
        contract_fraction=contract_fraction, queries_per_sequence=queries_per_sequence,
        cue_permutation=cue_permutation, cue_permutation_seed=cue_permutation_seed,
        min_separation=min_separation, split=split)
    n = num_examples
    rng = np.random.default_rng(seed + 1_000_003)     # rendering noise, disjoint from the episode draw

    inputs = np.full((n, input_seq_len), PAD_TOKEN, dtype=np.int64)
    labels = np.full((n, input_seq_len), -100, dtype=np.int64)
    np.put_along_axis(inputs, 2 * ep.write_slot, ep.write_key, axis=1)
    np.put_along_axis(inputs, 2 * ep.write_slot + 1, ep.write_value, axis=1)
    np.put_along_axis(inputs, 2 * ep.query_slot, ep.query_key, axis=1)
    # The label sits on the QUERY KEY position: the model reads the key and must emit the value next.
    np.put_along_axis(labels, 2 * ep.query_slot, ep.query_value, axis=1)

    # Filler EXCLUDES the key pool. `multiquery_ar` fills from the whole vocabulary, so a filler
    # token can coincide with a live key and act as an uninstructed query; the pool here is a few
    # dozen tokens out of `vocab_size`, so excluding it costs nothing and removes the ambiguity.
    holes = inputs == PAD_TOKEN
    n_holes = int(holes.sum())
    if n_holes:
        allowed = np.setdiff1d(np.arange(FIRST_FREE_TOKEN, vocab_size), ep.key_pool)
        inputs[holes] = allowed[rng.integers(0, len(allowed), size=n_holes)]

    marks = stamp(ep)
    # THE COVERAGE GATE. For a closed-form family the solved demand must BE the construction's
    # formula; a mismatch is a coverage shortfall or a construction bug, and either way the segment
    # would carry an x-coordinate that is not the graph the model is about to be shown. Refused here,
    # on the dataset path, rather than discovered as an unexplained curve. `build_episodes` +
    # `demand_bounds` remain ungated, so analysis and the gates can measure any graph they like.
    if marks["graph_contract_full"]:
        verify_construction(construction, dict(demand=demand), dict(
            lower=marks["graph_demand_lower"], upper=marks["graph_demand_upper"],
            exact=marks["graph_demand_exact"], method=marks["graph_demand_method"]))
    else:
        # LOUDLY NOTED, per the contract rule: under a partial contract the stamped demand is the
        # chi of the INDUCED (demanded-apart) subgraph, which is a different and smaller number than
        # the construction's closed form. It is the correct prediction for this cell, and it is not
        # the construction's nominal one.
        print(f"GRAPH_RECALL_PARTIAL_CONTRACT contract_fraction="
              f"{marks['graph_contract_fraction']} coverage={marks['graph_contract_coverage']}: "
              f"predicted_N={marks['graph_predicted_N']} is chi of the DEMANDED-APART subgraph, not "
              f"of the full {construction} graph (nominal {demand}).", flush=True)

    slices = {}
    if include_slices:
        slices = {"input_seq_len": input_seq_len, "num_kv_pairs": int(ep.write_key.shape[1]),
                  **marks}
    # Informational only. The AUTHORITATIVE copy is the one in `slices`, which the on-disk segment
    # cache stores and replays; this line is absent on a cache hit, so nothing downstream may key
    # off it (see the module docstring, COVERAGE VERIFICATION).
    print("GRAPH_RECALL_STAMP_JSON " + json.dumps(marks), flush=True)
    return DataSegment(torch.tensor(inputs), torch.tensor(labels), slices=slices)
