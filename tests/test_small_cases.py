"""
Unit tests for the morse_complex and homology libraries.

These test against known small cases that don't depend on any of the paper's
claims, so they're independent sanity checks for the library itself.

Run with:
    python -m pytest tests/
or
    python tests/test_small_cases.py
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

from morse_complex import (
    simplex, boundary_simplex, skeleton, cone, closure,
    f_vector, euler_characteristic, dim,
    hasse_pairs, lower, upper, is_matching, is_acyclic,
    all_matchings, all_acyclic_matchings, optimal_matchings,
    morse_complex, morse_complex_pure, gm_complex,
)
from homology import (
    boundary_matrix, betti_numbers, reduced_betti_numbers,
    relative_betti_numbers, induced_map_rank,
)


# ============================================================================
# Simplicial complex basics
# ============================================================================

class TestSimplicialComplexBasics(unittest.TestCase):
    def test_simplex_dims(self):
        self.assertEqual(dim(simplex(0)), 0)
        self.assertEqual(dim(simplex(1)), 1)
        self.assertEqual(dim(simplex(2)), 2)
        self.assertEqual(dim(simplex(3)), 3)

    def test_simplex_f_vectors(self):
        # f_k(Δⁿ) = C(n+1, k+1)
        from math import comb
        for n in range(5):
            f = f_vector(simplex(n))
            expected = tuple(comb(n + 1, k + 1) for k in range(n + 1))
            self.assertEqual(f, expected)

    def test_boundary_simplex_f_vectors(self):
        # ∂Δⁿ: all proper non-empty subsets of {0,...,n}
        from math import comb
        for n in range(1, 5):
            f = f_vector(boundary_simplex(n))
            # dims 0 through n-1
            expected = tuple(comb(n + 1, k + 1) for k in range(n))
            self.assertEqual(f, expected)

    def test_euler_char_ball(self):
        # Δⁿ is contractible => χ = 1
        for n in range(5):
            self.assertEqual(euler_characteristic(simplex(n)), 1)

    def test_euler_char_sphere(self):
        # ∂Δⁿ ≃ S^{n-1} => χ = 1 + (-1)^{n-1}
        for n in range(1, 5):
            self.assertEqual(euler_characteristic(boundary_simplex(n)),
                             1 + (-1) ** (n - 1))

    def test_closure_is_idempotent(self):
        K = simplex(3)
        self.assertEqual(closure(K), K)

    def test_closure_of_single_simplex(self):
        # Closure of {0, 1, 2} is Δ²
        K = closure([frozenset([0, 1, 2])])
        self.assertEqual(K, simplex(2))

    def test_skeleton(self):
        K = simplex(3)
        sk0 = skeleton(K, 0)
        sk1 = skeleton(K, 1)
        sk2 = skeleton(K, 2)  # = ∂Δ³
        self.assertEqual(len(sk0), 4)  # 4 vertices
        self.assertEqual(len(sk1), 10)  # 4 + 6
        self.assertEqual(sk2, boundary_simplex(3))

    def test_cone(self):
        # Cone on ∂Δⁿ = disk = contractible triangulation of Δⁿ with one more vertex
        K = boundary_simplex(2)  # triangle boundary = S¹
        C = cone(K, apex_label='v')
        self.assertEqual(euler_characteristic(C), 1)  # contractible
        # Cone has one more vertex
        self.assertEqual(sum(1 for s in C if len(s) == 1),
                         sum(1 for s in K if len(s) == 1) + 1)


# ============================================================================
# Hasse diagram and matchings
# ============================================================================

class TestHasseAndMatchings(unittest.TestCase):
    def test_hasse_pairs_count(self):
        # Δ²: 3 (v-e) + 3 (e-f) = 6 ... wait: 3 vertices, 3 edges, 1 face.
        # v-e edges: each vertex in 2 edges = 3 * 2 = 6... actually each edge
        # has 2 vertices, each face has 3 edges. # pairs: 3*2 (v-e) + 1*3 (e-f) = 9
        self.assertEqual(len(hasse_pairs(simplex(2))), 9)
        # Δ³: vertices-edges: 4*3=12; edges-faces: 6*2=12; faces-3simplex: 4*1=4; total 28
        self.assertEqual(len(hasse_pairs(simplex(3))), 28)

    def test_lower_upper(self):
        K = simplex(2)
        for p in hasse_pairs(K):
            self.assertLess(len(lower(p)), len(upper(p)))

    def test_is_matching(self):
        K = simplex(2)
        pairs = list(hasse_pairs(K))
        self.assertTrue(is_matching([pairs[0]]))
        self.assertTrue(is_matching([]))
        # duplicate a simplex across two pairs -> not a matching
        # Find two pairs sharing a simplex:
        for p1 in pairs:
            for p2 in pairs:
                if p1 != p2 and p1 & p2:
                    self.assertFalse(is_matching([p1, p2]))
                    return
        self.fail("Δ² should have two pairs sharing a simplex")

    def test_empty_matching_is_acyclic(self):
        K = simplex(3)
        self.assertTrue(is_acyclic(K, []))

    def test_optimal_matching_sizes(self):
        # Δⁿ is collapsible, so optimal matching size = (#simplices - 1) / 2
        for n in range(1, 4):
            K = simplex(n)
            total = sum(f_vector(K))
            opt = list(optimal_matchings(K))
            for m in opt:
                self.assertEqual(len(m), (total - 1) // 2)


# ============================================================================
# M(K), M_pure(K), GM(K) — small known cases
# ============================================================================

class TestMorseComplexes(unittest.TestCase):
    def test_MD1(self):
        # M(Δ¹) = 2 disjoint points (the two primitive GVFs of an edge)
        M = morse_complex(simplex(1))
        self.assertEqual(f_vector(M), (2,))
        self.assertEqual(euler_characteristic(M), 2)

    def test_MD2_is_4S1(self):
        # M(Δ²) ≃ ⋁⁴ S¹
        M = morse_complex(simplex(2))
        self.assertEqual(reduced_betti_numbers(M), [0, 4, 0])

    def test_MbD2_is_4S1(self):
        # M(∂Δ²) is the 1-skeleton of a triangular prism, ≃ ⋁⁴ S¹
        M = morse_complex(boundary_simplex(2))
        self.assertEqual(f_vector(M), (6, 9))
        self.assertEqual(reduced_betti_numbers(M), [0, 4])

    def test_M_pure_subset_of_M(self):
        for n in range(1, 4):
            M = morse_complex(simplex(n))
            Mp = morse_complex_pure(simplex(n))
            for s in Mp:
                self.assertIn(s, M)

    def test_M_subset_of_GM(self):
        for n in range(1, 4):
            M = morse_complex(simplex(n))
            GM = gm_complex(simplex(n))
            for s in M:
                self.assertIn(s, GM)

    def test_M_and_GM_same_1_skeleton(self):
        for n in range(1, 4):
            M = morse_complex(simplex(n))
            GM = gm_complex(simplex(n))
            M_1 = frozenset(s for s in M if len(s) <= 2)
            GM_1 = frozenset(s for s in GM if len(s) <= 2)
            self.assertEqual(M_1, GM_1)


# ============================================================================
# Homology basics
# ============================================================================

class TestHomology(unittest.TestCase):
    def test_betti_ball(self):
        # Δⁿ contractible => b_0=1, others 0
        for n in range(4):
            b = betti_numbers(simplex(n))
            self.assertEqual(b[0], 1)
            for k in range(1, len(b)):
                self.assertEqual(b[k], 0)

    def test_betti_sphere(self):
        # ∂Δⁿ ≃ S^{n-1} => b_{n-1} = 1, b_0 = 1 for n ≥ 2; special case n=1 (∂Δ¹ = S⁰ = 2 pts)
        for n in range(1, 5):
            b = betti_numbers(boundary_simplex(n))
            if n == 1:
                # S⁰: b_0 = 2, nothing else
                self.assertEqual(b[0], 2)
            else:
                # connected sphere: b_0 = 1, b_{n-1} = 1
                self.assertEqual(b[0], 1)
                self.assertEqual(b[n - 1], 1)
                for k in range(1, len(b)):
                    if k != n - 1:
                        self.assertEqual(b[k], 0, f"b_{k} of ∂Δ^{n}")

    def test_relative_homology_ball_vs_sphere(self):
        # H_k(Δⁿ, ∂Δⁿ) = Z if k=n, else 0 (Δⁿ/∂Δⁿ ≃ Sⁿ)
        for n in range(1, 4):
            rel = relative_betti_numbers(simplex(n), boundary_simplex(n))
            self.assertEqual(rel[n], 1, f"H_{n}(Δ^{n}, ∂Δ^{n})")
            for k in range(len(rel)):
                if k != n:
                    self.assertEqual(rel[k], 0, f"H_{k}(Δ^{n}, ∂Δ^{n}) should be 0")

    def test_induced_map_rank_identity(self):
        # Identity inclusion induces identity map, so rank = b_k
        K = boundary_simplex(3)
        for k in range(3):
            r = induced_map_rank(K, K, k)
            self.assertEqual(r, betti_numbers(K)[k])


# ============================================================================
# Runner
# ============================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
