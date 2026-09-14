"""THE CHI SOLVER — the chromatic number of a realized co-occurrence graph, with certificates.

Built 2026-08-02, alongside `zoology.data.graph_recall`. The graph-recall instrument sweeps a
DEMAND, and the demand is a chromatic number; a benchmark that sweeps a quantity it cannot compute
is sweeping a knob, not a quantity. This module computes it, says HOW it computed it, and hands back
the certificates that make the answer checkable.

WHAT IT RETURNS. `solve_chi(adj)` -> `{lower, upper, exact, method, clique, coloring}` where
`lower <= chi(G) <= upper`, `exact` says the interval collapsed, `method` names the argument that
produced it, and the two certificates are VERIFIED before they are returned:
  * `clique`   -- a vertex set that is pairwise adjacent, hence a lower bound of its own size. It is
                  re-checked against the adjacency matrix (`_check_clique`), so a bug in the search
                  cannot inflate a lower bound.
  * `coloring` -- a color per vertex with no monochromatic edge, hence an upper bound of its color
                  count, likewise re-checked (`_check_coloring`).
An unverifiable certificate raises rather than downgrades: a wrong bound on this axis is worse than
no bound, because it silently moves the x-coordinate of every cell that reports it.

WHY IT IS EXACT WHERE IT IS. Chromatic number is NP-hard in general, so the solver is a ladder and
each rung states its own argument:
  * `disjoint_cliques`      -- every connected component is complete: chi = the largest component.
                               (`graph_recall.complete(k)` realizes exactly this.)
  * `complete_multipartite` -- the COMPLEMENT is a disjoint union of cliques, i.e. non-adjacency is
                               an equivalence relation: chi = the number of parts, and one color per
                               part is the coloring. (`clustered(k, m)` realizes exactly this, with
                               chi = k for every m -- the decoupling the instrument is built for.)
  * `chordal`               -- a perfect elimination ordering exists (found by maximum cardinality
                               search). Chordal graphs are perfect, so chi = omega, and greedy on
                               the reverse PEO achieves it: both bounds come out of the same walk.
                               (`overlap(w, ...)` realizes an interval graph blown up by
                               `keys_per_class`, and interval graphs and their blow-ups are chordal.)
  * `exact_branch_and_bound`-- for everything else, k-colorability is decided by DSATUR-ordered
                               backtracking for k = lower, lower+1, ... under a node budget. At the
                               law-tier sizes the `reuse` family reaches (100-500 vertices) this
                               settles the small dense cases and times out on the large ones.
  * `certified_bounds`      -- the budget ran out: report the interval, both certificates attached.
Nothing here guesses. A family whose structure the solver recognizes gets a proof; a family whose
structure it does not gets an interval, and the cell is stamped with the interval.

VERIFY, DON'T TRUST THE KNOB. `verify_construction(construction, nominal, result)` is the coverage
gate: for the three closed-form families it demands the SOLVED chi equal the construction's formula
and raises otherwise, naming both numbers. It catches two different bugs with one check -- a
construction that does not build the graph it documents, and a sample that does not realize the
graph the construction would build. (For `graph_recall`'s three closed-form families the realized chi
happens to be sample-INVARIANT by design, because their key pools are structured -- per-cluster and
per-birth-slot disjoint sub-pools -- so under-sampling shows up in the node and edge counts and not
in chi. The gate is still the right shape: it is what would catch an unstructured pool, a truncated
group set, or a schedule bug, and none of those are hypothetical failure modes for a generator whose
whole claim is the graph it presents.)

NO NEW DEPENDENCY. `networkx` is present in this environment, but it carries no exact chromatic
solver (its `coloring` module is greedy heuristics), so the only piece it would replace is the
chordality test. `zoology.data` is imported by every data build on every box and `networkx` is not in
zoology's `install_requires`, so the three algorithms actually needed are implemented here in numpy
rather than making a packaging change to borrow one of them.
"""
import numpy as np

#: Backtracking nodes the exact k-colorability search may expand before it gives up and the answer
#: degrades to a certified interval. Sized so a law-tier `reuse` graph (a few hundred vertices)
#: settles or times out in seconds, not minutes -- this runs inside a data build.
DEFAULT_NODE_BUDGET = 200_000

#: Constructions whose chi is known in closed form, and the formula, in terms of the knob dict.
CLOSED_FORM = {
    "complete": lambda kn: kn["demand"],
    "clustered": lambda kn: kn["demand"],
    "overlap": lambda kn: kn["demand"],
}


# --- certificate checks (a bound is only as good as its witness) ----------------------------------
def _check_clique(adj, verts):
    for i, a in enumerate(verts):
        for b in verts[i + 1:]:
            if not adj[a, b]:
                raise AssertionError(f"clique certificate is not a clique: {a} and {b} are not "
                                     "adjacent. A bad lower bound would silently move the demand "
                                     "coordinate of every cell that reports it.")
    return len(verts)


def _check_coloring(adj, color):
    if (color < 0).any():
        raise AssertionError("coloring certificate leaves a vertex uncolored")
    src, dst = np.nonzero(np.triu(adj, 1))
    bad = np.nonzero(color[src] == color[dst])[0]
    if bad.size:
        i = int(bad[0])
        raise AssertionError(f"coloring certificate is not proper: {int(src[i])} and {int(dst[i])} "
                             "are adjacent and share a color")
    return int(color.max()) + 1 if len(color) else 0


# --- structure recognizers ------------------------------------------------------------------------
def _components(adj):
    """Connected components as a label array, by repeated boolean frontier expansion."""
    n = adj.shape[0]
    label = np.full(n, -1, dtype=np.int64)
    cur = 0
    for s in range(n):
        if label[s] >= 0:
            continue
        seen = np.zeros(n, dtype=bool)
        seen[s] = True
        frontier = seen.copy()
        while frontier.any():
            nxt = adj[frontier].any(0) & ~seen
            seen |= nxt
            frontier = nxt
        label[seen] = cur
        cur += 1
    return label, cur


def _as_disjoint_cliques(adj):
    """`(chi, coloring)` if every component is complete, else None."""
    label, k = _components(adj)
    color = np.full(len(label), -1, dtype=np.int64)
    biggest = 0
    for c in range(k):
        members = np.nonzero(label == c)[0]
        sub = adj[np.ix_(members, members)]
        if not (sub | np.eye(len(members), dtype=bool)).all():
            return None
        color[members] = np.arange(len(members))
        biggest = max(biggest, len(members))
    return biggest, color


def _as_complete_multipartite(adj):
    """`(chi, coloring)` if the COMPLEMENT is a disjoint union of cliques, else None.

    Equivalently: non-adjacency is transitive, so the vertex set splits into independent parts with
    every cross-part pair adjacent. One color per part, and no fewer will do (the parts pairwise
    conflict), so this is exact."""
    n = adj.shape[0]
    comp = ~adj & ~np.eye(n, dtype=bool)
    label, k = _components(comp)
    for c in range(k):
        members = np.nonzero(label == c)[0]
        sub = comp[np.ix_(members, members)]
        if not (sub | np.eye(len(members), dtype=bool)).all():
            return None                     # a complement component that is not complete
    return k, label


def _mcs_peo(adj):
    """Maximum cardinality search order, and the perfect-elimination check on it.

    Returns `(order, ok)`: `order` is the MCS elimination order and `ok` says it is a PEO, which is
    the definition of chordality. On a chordal graph, chi = omega and greedy on the REVERSE of this
    order attains it."""
    n = adj.shape[0]
    weight = np.zeros(n, dtype=np.int64)
    chosen = np.zeros(n, dtype=bool)
    order = np.empty(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        v = int(np.argmax(np.where(chosen, -1, weight)))
        order[i] = v
        chosen[v] = True
        weight[adj[v] & ~chosen] += 1
    pos = np.empty(n, dtype=np.int64)
    pos[order] = np.arange(n)
    for i in range(n):
        v = int(order[i])
        later = adj[v] & (pos > i)
        if not later.any():
            continue
        u = int(np.argmin(np.where(later, pos, n + 1)))
        rest = later.copy()
        rest[u] = False
        if (rest & ~adj[u]).any():
            return order, False
    return order, True


def _false_twin_quotient(adj):
    """`(representatives, class_of_vertex)` for the FALSE-TWIN partition (equal neighbourhoods).

    Two vertices with identical adjacency rows are non-adjacent (the row says so at each other's
    index) and interchangeable, i.e. one is a blow-up copy of the other. Collapsing them changes
    neither chi nor omega, because the classes are independent sets."""
    rows = np.ascontiguousarray(adj).view(np.uint8).reshape(adj.shape[0], -1)
    _, first, inverse = np.unique(rows, axis=0, return_index=True, return_inverse=True)
    return first.astype(np.int64), inverse.reshape(-1).astype(np.int64)


def _color_along(adj, order):
    """First-fit greedy over `order`, returning the coloring (not just its size)."""
    n = adj.shape[0]
    color = np.full(n, -1, dtype=np.int64)
    for v in order:
        taken = color[adj[v] & (color >= 0)]
        c = 0
        if taken.size:
            seen = np.zeros(taken.max() + 2, dtype=bool)
            seen[taken] = True
            c = int(np.argmin(seen))
        color[v] = c
    return color


def _degeneracy_order(adj):
    deg = adj.sum(1).astype(np.int64)
    alive = np.ones(adj.shape[0], dtype=bool)
    out = []
    for _ in range(adj.shape[0]):
        v = int(np.argmin(np.where(alive, deg, np.iinfo(np.int64).max)))
        out.append(v)
        alive[v] = False
        deg[adj[v] & alive] -= 1
    return np.array(out[::-1], dtype=np.int64)


def _greedy_clique(adj, seeds):
    """Greedy clique growth from each seed; returns the largest vertex list found."""
    best = []
    for s in seeds:
        verts = [int(s)]
        cand = adj[s].copy()
        while cand.any():
            deg = (adj & cand[None, :]).sum(1)
            v = int(np.argmax(np.where(cand, deg, -1)))
            verts.append(v)
            cand &= adj[v]
            cand[v] = False
        if len(verts) > len(best):
            best = verts
    return best


def _k_colorable(adj, k, budget):
    """DSATUR-ordered backtracking: `(True, coloring)`, `(False, None)`, or `(None, None)` on budget.

    Branches on the vertex with the highest saturation (most distinct colors already among its
    neighbors), which is the standard ordering for this decision problem, and prunes the symmetric
    duplicates by never opening color `c+1` before color `c` has been used."""
    n = adj.shape[0]
    color = np.full(n, -1, dtype=np.int64)
    spent = [0]

    def rec(depth, used):
        if depth == n:
            return True
        spent[0] += 1
        if spent[0] > budget:
            raise TimeoutError
        sat, best, best_v = -1, None, -1
        for v in range(n):
            if color[v] >= 0:
                continue
            nb = color[adj[v] & (color >= 0)]
            s = len(np.unique(nb)) if nb.size else 0
            if s > sat:
                sat, best_v, best = s, v, set(int(x) for x in nb)
        for c in range(min(used + 1, k)):
            if c in best:
                continue
            color[best_v] = c
            if rec(depth + 1, max(used, c + 1)):
                return True
            color[best_v] = -1
        return False

    try:
        ok = rec(0, 0)
    except TimeoutError:
        return None, None
    except RecursionError:
        return None, None
    return (True, color.copy()) if ok else (False, None)


def solve_chi(adj: np.ndarray, node_budget: int = DEFAULT_NODE_BUDGET) -> dict:
    """chi of the graph given by a dense boolean adjacency matrix. See the module docstring."""
    n = adj.shape[0]
    if n == 0:
        return dict(lower=0, upper=0, exact=True, method="empty", clique=[], coloring=[])
    if not adj.any():
        return dict(lower=1, upper=1, exact=True, method="edgeless", clique=[0],
                    coloring=[0] * n)

    for name, fn in (("disjoint_cliques", _as_disjoint_cliques),
                     ("complete_multipartite", _as_complete_multipartite)):
        got = fn(adj)
        if got is not None:
            chi, color = got
            clique = _greedy_clique(adj, np.argsort(-adj.sum(1))[:8])
            assert _check_coloring(adj, color) == chi
            _check_clique(adj, clique)
            return dict(lower=chi, upper=chi, exact=True, method=name,
                        clique=sorted(int(v) for v in clique[:chi]), coloring=color.tolist())

    # The CHORDAL rung, taken on the FALSE-TWIN QUOTIENT rather than on the graph itself. A
    # `graph_recall.overlap` cell with `keys_per_class > 1` is an interval graph with each vertex
    # blown up into an independent set of twins, and a blow-up of an interval graph is NOT chordal
    # (blowing up the middle of a path creates an induced 4-cycle) even though it is still perfect --
    # substituting an independent set into a vertex of a perfect graph is perfect (Lovász). Since the
    # blown-up classes are independent, chi and omega are both invariant under the quotient, so
    # collapsing the twins first recovers the exact argument on the graph the construction actually
    # built. With no twins the quotient is the identity and this is the plain chordal rung.
    rep, cls_of = _false_twin_quotient(adj)
    q = adj[np.ix_(rep, rep)]
    order, chordal = _mcs_peo(q)
    if chordal:
        qcolor = _color_along(q, order[::-1])
        chi = _check_coloring(q, qcolor)
        color = qcolor[cls_of]                      # every twin inherits its class's color
        assert _check_coloring(adj, color) == chi
        # omega, read straight off the PEO: a vertex plus its later neighbours is a clique
        pos = np.empty(len(rep), dtype=np.int64)
        pos[order] = np.arange(len(rep))
        best = []
        for i in range(len(rep)):
            v = int(order[i])
            verts = [v] + [int(u) for u in np.nonzero(q[v] & (pos > i))[0]]
            if len(verts) > len(best):
                best = verts
        lifted = [int(rep[v]) for v in best]
        _check_clique(adj, lifted)
        if len(lifted) == chi:
            return dict(lower=chi, upper=chi, exact=True,
                        method="chordal" if len(rep) == n else "chordal_quotient",
                        clique=sorted(lifted), coloring=color.tolist())

    # --- general case: certified bounds, then try to close the gap exactly
    clique = _greedy_clique(adj, np.argsort(-adj.sum(1))[:16])
    lower = _check_clique(adj, clique)
    best_color, upper = None, None
    for o in (np.arange(n), np.argsort(-adj.sum(1)), _degeneracy_order(adj)):
        c = _color_along(adj, o)
        u = _check_coloring(adj, c)
        if upper is None or u < upper:
            best_color, upper = c, u
    method = "certified_bounds"
    k = lower
    while k < upper:
        ok, color = _k_colorable(adj, k, node_budget)
        if ok is None:
            break                                     # budget spent: keep the interval
        if ok:
            best_color, upper, method = color, _check_coloring(adj, color), "exact_branch_and_bound"
            break
        lower, method = k + 1, "exact_branch_and_bound"
        k += 1
    return dict(lower=int(lower), upper=int(upper), exact=bool(lower == upper), method=method,
                clique=sorted(int(v) for v in clique), coloring=best_color.tolist())


class DemandMismatch(ValueError):
    """The solved demand of a realized graph is not the demand its construction claims.

    Loud on purpose. It means one of two things, and both are the kind of bug that would otherwise be
    discovered as an unexplained curve: the construction does not build the graph it documents, or
    the sample does not realize the graph the construction builds (coverage shortfall)."""


def verify_construction(construction: str, knobs: dict, result: dict) -> None:
    """Gate a solved demand against the construction's closed form. No-op where there is none."""
    formula = CLOSED_FORM.get(construction)
    if formula is None:
        return
    want = int(formula(knobs))
    if not result["exact"] or result["lower"] != want:
        got = (f"exactly {result['lower']}" if result["exact"]
               else f"the interval [{result['lower']}, {result['upper']}]")
        raise DemandMismatch(
            f"construction {construction!r} claims chi = {want}, but the REALIZED graph solves to "
            f"{got} (method={result['method']}). Either the sample does not realize the target graph "
            "(coverage shortfall -- raise num_examples) or the construction does not build the graph "
            "it documents. The cell is refused rather than shipped with an x-coordinate that is not "
            "the graph the model was shown.")
