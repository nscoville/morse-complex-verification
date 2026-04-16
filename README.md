# Morse-complex-verification

Computational verification for the numerical and structural claims in:

**"The complex of discrete Morse matchings of the $n$-simplex: homotopy types and structural results**
Nicholas A. Scoville

Every $f$-vector, Euler characteristic, homology group, structural bijection,
and numerical identity stated in the paper is independently recomputed here
from first principles and checked against the paper's claims.

## What is verified

| Paper section | Claims verified |
|---|---|
| §2 (Background) | Homotopy types of M(Δ²) ≃ ⋁⁴ S¹ and M(∂Δ²) via direct computation |
| §3 (Counting) | f(1), f(2), f(3), top-facet bijection for n = 2, 3, layer count (Lem 3.3) on all 256 optimal matchings on Δ³, spanning tree property (Lem 3.4) on all 64 optimal matchings on Δ³_(1), restriction theorem (Thm 3.6) for n = 3 |
| §4 (Inclusions) | H_k(M(Δ³), M(∂Δ³)) for all k, LES exactness, rank of inclusion-induced map on H_4 |
| §5 (n = 3 data) | Every entry of Table 1 (f-vectors, Euler characteristics) and Table 2 (integral homology) for all four complexes M(Δ³), M(∂Δ³), M_P(Δ³), M_P(∂Δ³) |
| §5.2 (n = 4 data) | Consistency identities: χ = 212,457, χ̃ = 212,456, f(4) = 5·76,025, CJ upper bound |
| §6 (GM) | Homotopy types GM(Δ³) ≃ GM(∂Δ³) ≃ ⋁³⁹ S⁴, shared 1-skeleton of M and GM, the surjection Z⁹⁹ → Z³⁹ on H_4, and the striking Prop 6.4 result that the link L_i has H̃_3 = H̃_4 = Z² for n = 3 |

A total of **51 independent checks**, all of which pass.

## Running it

Requires Python 3.10+ with `numpy` and `sympy`.

```bash
pip install numpy sympy
python verify_paper.py
```

Expected runtime: 2–3 minutes on a modern laptop. The output reports each
check with `[ OK ]` or `[FAIL]`, grouped by paper section, with a summary at
the end.

For a faster run (~90s) that skips one slow induced-map-rank computation:

```bash
python verify_paper.py --quick
```

## Contents

| File | Purpose |
|---|---|
| `morse_complex.py` | Core library: simplicial complexes, Hasse diagrams, enumeration of (acyclic) matchings, construction of M(K), M_P(K), GM(K) |
| `homology.py` | Integer boundary matrices, Betti numbers, relative homology, and rank of the induced map on homology from a subcomplex inclusion |
| `verify_paper.py` | Top-level driver; runs every claim as an independent computation |
| `tests/test_small_cases.py` | Library sanity checks (does not depend on the paper) |
| `cluster/n4_reference.py` | Python reference implementation of the n = 4 optimal-matching enumeration |
| `cluster/README.md` | Documentation of the cluster computation for f(4) = 380,125 |

## Methodology

- **Simplicial complexes** are represented as frozensets of frozensets, with
  vertex labels of arbitrary hashable type.
- **Acyclic matchings** are enumerated by backtracking, with acyclicity
  checked via iterative three-color DFS on the modified Hasse diagram.
- **Homology** is computed by building integer boundary matrices and taking
  ranks. For small matrices, `sympy` gives exact rank over ℤ. For larger
  matrices, rank over 𝔽_p is computed for two independent large primes
  (10⁹+7 and 998244353) and the maximum is taken. All homology groups in
  the paper are torsion-free, so Q-rank = Betti number.
- **Rank of the induced map** H_k(L) → H_k(K) for a subcomplex inclusion
  L ⊆ K is computed by combining a modular null-space basis for Z_k(L)
  with the boundary image B_k(K) inside C_k(K), then taking rank over 𝔽_p.

## Caveat on n = 4

The full f-vector of M(Δ⁴) was computed on a computing cluster in C; the
Python reference implementation in `cluster/` documents the algorithm but
will not complete for n = 4 on a single machine. The homology of M(Δ⁴) has
not been computed: the largest boundary matrix has roughly 8.76 × 10⁸ rows
and 7.13 × 10⁸ columns.

## Acknowledgment

This verification code was developed with substantial assistance from
Claude Opus 4.6 (Anthropic) as a dialog partner during the paper's
development and as an aid in writing and validating the computational
scripts.

## License

MIT License. See the paper for the mathematical content; this repository
distributes only the verification code.
