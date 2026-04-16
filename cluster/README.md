# Cluster computation for n = 4

The claim `f(4) = 380,125` in the paper was obtained by exhaustive enumeration
of all maximal acyclic matchings on the Hasse diagram of Δ⁴. This directory
documents that computation.

## What was actually run

A parallel C implementation was executed on a computing cluster. The
algorithm is the same as in `n4_reference.py` (layer-by-layer branch and
bound with the layer-count constraint from Lemma 3.3), parallelized by
distributing different layer-0 configurations across compute nodes.

The key numerical results from that run:

| Quantity | Value |
|----------|-------|
| `f(4)` (top facets of `M(Δ⁴)`) | 380,125 |
| `f`-vector of `M(Δ⁴)` | (75, 2485, 47955, 598425, 5071367, 29844505, 122685075, 350017175, 680808105, 876110235, 712961065, 343320335, 88467825, 10315975, 380125) |
| `χ(M(Δ⁴))` | 212,457 |
| `χ̃(M(Δ⁴))` | 212,456 |
| Top facets of `M(Δ⁴_(2))` | 76,025 |

These values are reproduced exactly in the paper and embedded as
consistency checks in `../verify_paper.py`. The identity

    f(4) = 5 · 76,025 = 380,125

is an independent verification of Theorem 3.6 for `n = 4` (since
the theorem predicts `f(n) = (n+1) · |top facets of M(Δⁿ_(n-2))|`).

## Reference Python implementation

`n4_reference.py` is an executable specification of the enumeration in pure
Python. It is readable, auditable, and tractable for small `n`:

```bash
$ python n4_reference.py 1    # completes instantly
f(1) = 2

$ python n4_reference.py 2    # completes instantly
f(2) = 9

$ python n4_reference.py 3    # completes in under a second
f(3) = 256

$ python n4_reference.py 4    # DO NOT run to completion on a laptop
```

For `n = 4`, the Python implementation is far too slow to complete in any
reasonable time on a single machine. Running the program will produce partial
progress but will not finish without cluster-scale parallelism.

## Reproducing the cluster result

If you wish to independently reproduce `f(4) = 380,125`, the recommended path
is:

1. Port `n4_reference.py` to C or Rust for raw speed.
2. Parallelize by distributing different choices of the 4 layer-0 pairs
   across worker processes (there are `C(25, 4) = 12,650` combinations of
   4 Hasse-edge pairs on the vertex→edge layer of Δ⁴, though only valid
   partial matchings need to be pursued).
3. Aggregate counts across all workers.

The paper used the top-facet bijection of Proposition 3.1 (`f(n)` equals
the number of top facets of `M(∂Δⁿ)` as well) as an independent cross-check:
the computation was performed on both `Δ⁴` and `∂Δ⁴`, and both yielded the
same value `380,125`.

## Homology of M(Δ⁴)

The homology of `M(Δ⁴)` has *not* been computed: the largest boundary matrix
has dimensions approximately `8.76 × 10⁸` by `7.13 × 10⁸`, which is beyond
current computational reach. Only the `f`-vector was obtained on the cluster;
the partial structural conclusion (that `M(Δ⁴)` is not a wedge of spheres in
any single odd dimension) follows from `χ̃(M(Δ⁴)) = 212,456 > 0`, which is
verified as an arithmetic identity in `verify_paper.py`.
