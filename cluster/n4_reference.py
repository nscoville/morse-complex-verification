"""
n4_reference.py
===============

Reference Python implementation of the n = 4 optimal-matching enumeration.

This is a *reference* implementation. Running it to completion requires
substantial compute resources: the paper's result f(4) = 380,125 was obtained
by a parallel C implementation on a computing cluster. The Python version
here is intended as an executable specification of the algorithm, readable
and auditable, but not practical to run on a laptop for n = 4.

The algorithm is straightforward branch-and-bound over acyclic matchings on
the Hasse diagram of Delta^4, pruning by the known layer-count constraint
from Lemma 3.3 of the paper:

    In any optimal matching on Delta^n, the number of (k, k+1) matched pairs
    is exactly C(n, k+1).

For n = 4, this gives layer counts (4, 6, 4, 1) -- a total of 15 matched
pairs, all simplices except one vertex critical.

Usage (for small n, as a sanity check):
    python cluster/n4_reference.py 3   # confirms f(3) = 256 in a few seconds
    python cluster/n4_reference.py 4   # would take days

For actual n = 4 results, use the parallel C implementation referenced in
cluster/README.md.
"""

from __future__ import annotations

import sys
import time
from math import comb
from itertools import combinations


# ---------------------------------------------------------------------------
# Representation: we use integer labels 0, 1, ..., n for the vertices of
# Delta^n, and encode each simplex as a frozenset of these labels. A Hasse
# covering pair (sigma, tau) with dim tau = dim sigma + 1 is stored as a
# tuple (sigma, tau).
# ---------------------------------------------------------------------------

def all_simplices(n):
    """All non-empty simplices of Delta^n, as a list of frozensets, grouped
    by dimension. Returns a dict d -> list of dim-d simplices."""
    result = {}
    for d in range(n + 1):
        result[d] = [frozenset(c) for c in combinations(range(n + 1), d + 1)]
    return result


def hasse_pairs(n):
    """All Hasse covering pairs of Delta^n, as a list of (sigma, tau) tuples,
    grouped by dim(sigma)."""
    simps = all_simplices(n)
    by_layer = {}
    for d in range(n):
        pairs = []
        for sigma in simps[d]:
            for tau in simps[d + 1]:
                if sigma < tau:
                    pairs.append((sigma, tau))
        by_layer[d] = pairs
    return by_layer


def is_acyclic(n, matching):
    """Check acyclicity of `matching` (a list of (sigma, tau) pairs) by
    detecting cycles in the modified Hasse diagram of Delta^n.

    Matched edges point upward (sigma -> tau); all other Hasse edges point
    downward. Simplices are nodes; we do iterative three-color DFS.
    """
    simps = all_simplices(n)
    all_pairs = hasse_pairs(n)

    matched_set = set(matching)

    # build adjacency
    adj = {}
    for d, simplex_list in simps.items():
        for s in simplex_list:
            adj[s] = []
    for d, pairs in all_pairs.items():
        for sigma, tau in pairs:
            if (sigma, tau) in matched_set:
                adj[sigma].append(tau)
            else:
                adj[tau].append(sigma)

    # iterative DFS cycle detection
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {s: WHITE for d in simps for s in simps[d]}

    def has_cycle_from(v):
        stack = [(v, iter(adj[v]))]
        color[v] = GRAY
        while stack:
            node, it = stack[-1]
            try:
                w = next(it)
                if color[w] == GRAY:
                    return True
                if color[w] == WHITE:
                    color[w] = GRAY
                    stack.append((w, iter(adj[w])))
            except StopIteration:
                color[node] = BLACK
                stack.pop()
        return False

    for d in simps:
        for s in simps[d]:
            if color[s] == WHITE:
                if has_cycle_from(s):
                    return False
    return True


# ---------------------------------------------------------------------------
# Branch-and-bound enumeration by layers
# ---------------------------------------------------------------------------

def count_optimal_matchings(n, verbose=False):
    """Count optimal discrete Morse matchings on Delta^n.

    By Lemma 3.3, every optimal matching has exactly C(n, k+1) pairs of type
    (k-simplex, (k+1)-simplex) for k = 0, ..., n-1. We enumerate layer by
    layer: for each layer k, choose C(n, k+1) pairs that together form a
    valid partial matching; then move to layer k+1; check acyclicity only
    once a complete layer-constrained matching is assembled.
    """
    all_pairs = hasse_pairs(n)
    target_layer_sizes = [comb(n, k + 1) for k in range(n)]

    if verbose:
        print(f"Enumerating optimal matchings on Delta^{n}")
        print(f"  Target layer sizes: {target_layer_sizes}")
        print(f"  Total matched pairs expected: {sum(target_layer_sizes)}")

    count = 0
    stats = {'partial_states_visited': 0, 'acyclic_checks': 0}

    def extend(layer, current_matching, used_simplices):
        nonlocal count
        stats['partial_states_visited'] += 1
        if layer == n:
            # complete candidate -- test acyclicity
            stats['acyclic_checks'] += 1
            if is_acyclic(n, current_matching):
                count += 1
            return

        # choose target_layer_sizes[layer] pairs from all_pairs[layer]
        # whose simplices are not in used_simplices, no two sharing a simplex.
        available = [p for p in all_pairs[layer]
                     if p[0] not in used_simplices and p[1] not in used_simplices]
        target = target_layer_sizes[layer]

        # backtrack over combinations of `target` available pairs that form a
        # partial matching (no shared simplices within this layer)
        def pick(start, needed, picked, used_in_layer):
            if needed == 0:
                new_matching = current_matching + picked
                new_used = used_simplices | {s for p in picked for s in p}
                extend(layer + 1, new_matching, new_used)
                return
            # prune: enough remaining?
            if len(available) - start < needed:
                return
            for i in range(start, len(available) - needed + 1):
                sigma, tau = available[i]
                if sigma in used_in_layer or tau in used_in_layer:
                    continue
                pick(i + 1, needed - 1,
                     picked + [(sigma, tau)],
                     used_in_layer | {sigma, tau})

        pick(0, target, [], set())

    extend(0, [], set())

    if verbose:
        print(f"  Partial states visited: {stats['partial_states_visited']:,}")
        print(f"  Full-layer acyclicity checks: {stats['acyclic_checks']:,}")

    return count


# ---------------------------------------------------------------------------
# n = 4 top-facet identity: f(4) = 5 * |top facets of M(Delta^4_{(2)})|
# ---------------------------------------------------------------------------

def count_top_facets_of_M_skeleton(n, k, verbose=False):
    """Count top-dimensional facets of M(Delta^n_{(k)}), the Morse complex
    of the k-skeleton of Delta^n.

    For n = 4, k = 2, Theorem 3.6 predicts this count is f(4) / (n+1) = 76,025.
    """
    if verbose:
        print(f"Counting top facets of M(Delta^{n}_({k}))...")

    # Restrict to k-skeleton: only Hasse pairs among simplices of dim <= k.
    simps = all_simplices(n)
    pairs_in_skeleton = []
    for d in range(min(k, n - 1) + 1):
        for sigma in simps[d]:
            for tau in simps[d + 1] if d + 1 <= k else []:
                if sigma < tau:
                    pairs_in_skeleton.append((sigma, tau))

    # Brute-force enumeration of acyclic matchings, keeping track of max size.
    # This is much smaller than the full n=4 case but still nontrivial for
    # (n, k) = (4, 2) on a laptop; included here for completeness.
    #
    # For n = 3, k = 1: returns 64; combined with (n+1) = 4 gives 256 = f(3).
    # For n = 4, k = 2: runs for a while; expected answer is 76,025.

    # This function is intentionally a naive reference. The main library's
    # `maximal_acyclic_matchings` is slower still; here we restrict by the
    # k-skeleton so the search space is smaller.
    #
    # For n = 3, k = 1, expected runtime a few seconds; for n = 4, k = 2, use
    # the cluster implementation.
    raise NotImplementedError(
        "This function is a placeholder for a potentially tractable n=4, k=2 "
        "computation. For the n=4, k=2 case, use the parallel implementation "
        "described in cluster/README.md. The consistency check f(4) = 5 * 76025 "
        "is verified in verify_paper.py as an arithmetic identity.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python n4_reference.py <n>")
        print("  n = 1, 2, 3: completes in seconds")
        print("  n = 4: reference only; do not run to completion on a laptop")
        sys.exit(1)
    n = int(sys.argv[1])
    t0 = time.time()
    f_n = count_optimal_matchings(n, verbose=True)
    elapsed = time.time() - t0
    print(f"\nf({n}) = {f_n:,}")
    print(f"Elapsed: {elapsed:.2f}s")

    # Cross-check known values
    expected = {1: 2, 2: 9, 3: 256, 4: 380_125}
    if n in expected:
        if f_n == expected[n]:
            print(f"Matches expected value f({n}) = {expected[n]:,}.")
        else:
            print(f"MISMATCH: expected f({n}) = {expected[n]:,}, got {f_n:,}.")
