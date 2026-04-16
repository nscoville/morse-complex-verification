"""
homology.py
===========

Integer homology computation for simplicial complexes.

Provides:
- Boundary matrices (absolute and relative)
- Betti numbers b_k = rank H_k (via rank of boundary maps)
- Reduced Betti numbers
- Relative Betti numbers b_k(K, L)
- Rank of the induced map i_*: H_k(L) -> H_k(K) for L ⊆ K

For rank over Z, we compute rank over Q via exact sympy arithmetic for small
matrices, and via rank over F_p (with cross-checking across two large primes)
for larger matrices. Since our homology is torsion-free in all cases in the
paper, rank over Q = Betti number.
"""

from __future__ import annotations

import numpy as np
import sympy


# -------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------

def _ordered_simplex(s, vertex_order: dict) -> tuple:
    """Return the vertices of simplex s as a tuple ordered by `vertex_order`."""
    return tuple(sorted(s, key=lambda v: vertex_order[v]))


def _build_vertex_order(K) -> dict:
    """Stable ordering of all vertices in K (works for any hashable type)."""
    vertices = set()
    for s in K:
        for v in s:
            vertices.add(v)
    return {v: i for i, v in enumerate(sorted(vertices, key=repr))}


# -------------------------------------------------------------------
# Boundary matrices
# -------------------------------------------------------------------

def boundary_matrix(K, k: int) -> np.ndarray:
    """Integer boundary matrix d_k: C_k(K) -> C_{k-1}(K).

    Returns numpy int64 array; rows = (k-1)-simplices, columns = k-simplices.
    """
    if k <= 0:
        f0 = sum(1 for s in K if len(s) == 1)
        return np.zeros((0, f0), dtype=np.int64)

    vo = _build_vertex_order(K)

    k_simps = sorted(
        [_ordered_simplex(s, vo) for s in K if len(s) == k + 1],
        key=lambda t: tuple(vo[v] for v in t)
    )
    km1_simps = sorted(
        [_ordered_simplex(s, vo) for s in K if len(s) == k],
        key=lambda t: tuple(vo[v] for v in t)
    )
    km1_idx = {s: i for i, s in enumerate(km1_simps)}

    M = np.zeros((len(km1_simps), len(k_simps)), dtype=np.int64)
    for j, s in enumerate(k_simps):
        for i in range(len(s)):
            face = s[:i] + s[i + 1:]
            M[km1_idx[face], j] += (-1) ** i
    return M


def relative_boundary_matrix(K, L, k: int) -> np.ndarray:
    """Boundary matrix of the relative chain complex C_*(K, L) in degree k.

    C_k(K, L) = free abelian group on k-simplices of K not in L. The relative
    boundary of sigma is d sigma with all (k-1)-faces that lie in L set to zero.
    """
    if k <= 0:
        f0 = sum(1 for s in K if len(s) == 1 and s not in L)
        return np.zeros((0, f0), dtype=np.int64)

    vo = _build_vertex_order(K)

    k_simps = sorted(
        [_ordered_simplex(s, vo) for s in K if len(s) == k + 1 and s not in L],
        key=lambda t: tuple(vo[v] for v in t)
    )
    km1_simps = sorted(
        [_ordered_simplex(s, vo) for s in K if len(s) == k and s not in L],
        key=lambda t: tuple(vo[v] for v in t)
    )
    km1_idx = {s: i for i, s in enumerate(km1_simps)}

    L_tuples = set()
    for s in L:
        L_tuples.add(_ordered_simplex(s, vo))

    M = np.zeros((len(km1_simps), len(k_simps)), dtype=np.int64)
    for j, s in enumerate(k_simps):
        for i in range(len(s)):
            face = s[:i] + s[i + 1:]
            if face in L_tuples:
                continue
            M[km1_idx[face], j] += (-1) ** i
    return M


# -------------------------------------------------------------------
# Rank computation
# -------------------------------------------------------------------

def matrix_rank_Z(M: np.ndarray) -> int:
    """Rank of M over Z (= rank over Q).

    Small matrices: exact rank via sympy. Larger: rank over F_p for two
    large primes p, taking the maximum.
    """
    if M.size == 0:
        return 0
    if M.shape[0] * M.shape[1] < 20_000:
        return sympy.Matrix(M.tolist()).rank()
    primes = [1_000_000_007, 998_244_353]
    ranks = [_modular_rank(M, p) for p in primes]
    r = max(ranks)
    if ranks[0] != ranks[1]:
        import sys
        print(f"warning: modular ranks differ ({ranks[0]} vs {ranks[1]}); using max={r}",
              file=sys.stderr)
    return r


def _modular_rank(M: np.ndarray, p: int) -> int:
    """Rank of M over F_p via Gaussian elimination in int64 arithmetic."""
    A = (M.astype(np.int64) % p).copy()
    rows, cols = A.shape
    r = 0
    c = 0
    while r < rows and c < cols:
        pivot = -1
        for i in range(r, rows):
            if A[i, c] != 0:
                pivot = i
                break
        if pivot == -1:
            c += 1
            continue
        if pivot != r:
            A[[r, pivot]] = A[[pivot, r]]
        inv = pow(int(A[r, c]), p - 2, p)
        A[r] = (A[r] * inv) % p
        for i in range(rows):
            if i != r and A[i, c] != 0:
                factor = int(A[i, c])
                A[i] = (A[i] - factor * A[r]) % p
        r += 1
        c += 1
    return r


# -------------------------------------------------------------------
# Betti numbers
# -------------------------------------------------------------------

def betti_numbers(K, max_dim: int | None = None) -> list[int]:
    """Betti numbers b_0, b_1, ..., b_{max_dim} of K.

    Computed via b_k = f_k - rank d_k - rank d_{k+1}.
    """
    if max_dim is None:
        max_dim = max((len(s) - 1 for s in K), default=-1)
    if max_dim < 0:
        return []

    f = [0] * (max_dim + 2)
    for s in K:
        d = len(s) - 1
        if d < len(f):
            f[d] += 1

    ranks = [0] * (max_dim + 2)
    for k in range(1, max_dim + 2):
        if f[k] == 0 or f[k - 1] == 0:
            ranks[k] = 0
            continue
        if k > max_dim:
            ranks[k] = 0
            continue
        M = boundary_matrix(K, k)
        ranks[k] = matrix_rank_Z(M)

    bettis = []
    for k in range(max_dim + 1):
        b = f[k] - ranks[k] - (ranks[k + 1] if k + 1 < len(ranks) else 0)
        bettis.append(b)
    return bettis


def reduced_betti_numbers(K, max_dim: int | None = None) -> list[int]:
    """Reduced Betti numbers: b_0 decremented by 1 if K is non-empty."""
    b = betti_numbers(K, max_dim=max_dim)
    if b and b[0] > 0:
        b[0] -= 1
    return b


def relative_betti_numbers(K, L, max_dim: int | None = None) -> list[int]:
    """Relative Betti numbers b_k(K, L) = rank H_k(K, L)."""
    if max_dim is None:
        max_dim = max((len(s) - 1 for s in K), default=-1)
    if max_dim < 0:
        return []

    f = [0] * (max_dim + 2)
    for s in K:
        if s in L:
            continue
        d = len(s) - 1
        if d < len(f):
            f[d] += 1

    ranks = [0] * (max_dim + 2)
    for k in range(1, max_dim + 2):
        if f[k] == 0 or (k - 1 < len(f) and f[k - 1] == 0):
            ranks[k] = 0
            continue
        if k > max_dim:
            ranks[k] = 0
            continue
        M = relative_boundary_matrix(K, L, k)
        ranks[k] = matrix_rank_Z(M)

    bettis = []
    for k in range(max_dim + 1):
        b = f[k] - ranks[k] - (ranks[k + 1] if k + 1 < len(ranks) else 0)
        bettis.append(b)
    return bettis


# -------------------------------------------------------------------
# Induced map on homology
# -------------------------------------------------------------------

def induced_map_rank(K_sub, K_full, k: int) -> int:
    """Rank of the map H_k(K_sub) -> H_k(K_full) induced by inclusion.

    Assumes K_sub subset of K_full. Uses the formula
        rank(H_k(K_sub) -> H_k(K_full))
            = dim(Z_k(K_sub) + B_k(K_full)) - dim(B_k(K_full))
    computed entirely over F_p for a large prime p (so Q-rank).

    The key observation: we don't need an explicit basis for Z_k(K_sub);
    we only need to compute rank quantities. Work directly with the
    kernel-matrix pattern:
        - Stack the boundary matrices d_k^sub and d_{k+1}^full so their
          interaction in the full chain complex gives the needed ranks.
    """
    vo = _build_vertex_order(K_full)
    k_full = sorted(
        [_ordered_simplex(s, vo) for s in K_full if len(s) == k + 1],
        key=lambda t: tuple(vo[v] for v in t)
    )
    k_sub = sorted(
        [_ordered_simplex(s, vo) for s in K_sub if len(s) == k + 1],
        key=lambda t: tuple(vo[v] for v in t)
    )
    if not k_sub:
        return 0

    # Indices in k_full for each k_sub simplex
    full_idx = {s: i for i, s in enumerate(k_full)}
    sub_positions = [full_idx[s] for s in k_sub]

    # d_k on K_sub: rows = (k-1)-simps of K_sub, cols = k_sub
    M_sub = boundary_matrix(K_sub, k)
    # d_{k+1} on K_full: rows = k_full, cols = (k+1)-simps of K_full
    M_full_next = boundary_matrix(K_full, k + 1)

    # rank(Z_k(K_sub)) = #cols(M_sub) - rank(M_sub) = n_sub - rank d_k^sub
    n_sub = len(k_sub)
    rank_d_sub = matrix_rank_Z(M_sub) if M_sub.size else 0
    dim_Z_sub = n_sub - rank_d_sub

    if dim_Z_sub == 0:
        return 0

    # We need rank(Z_k(K_sub) + B_k(K_full)) as subspaces of C_k(K_full) over Q.
    # Strategy: compute rank over F_p using modular null space for Z_k(K_sub).
    p = 1_000_000_007

    def nullspace_mod(M, p):
        """Return a basis for null space of M over F_p as columns (k x n)."""
        if M.size == 0:
            return np.eye(M.shape[1], dtype=np.int64) if M.shape[1] > 0 else np.zeros((0, 0), dtype=np.int64)
        A = (M.astype(np.int64) % p).copy()
        rows, cols = A.shape
        # row reduce
        pivot_cols = []
        r = 0
        for c in range(cols):
            pivot = -1
            for i in range(r, rows):
                if A[i, c] != 0:
                    pivot = i
                    break
            if pivot == -1:
                continue
            if pivot != r:
                A[[r, pivot]] = A[[pivot, r]]
            inv = pow(int(A[r, c]), p - 2, p)
            A[r] = (A[r] * inv) % p
            for i in range(rows):
                if i != r and A[i, c] != 0:
                    factor = int(A[i, c])
                    A[i] = (A[i] - factor * A[r]) % p
            pivot_cols.append(c)
            r += 1
        free_cols = [c for c in range(cols) if c not in set(pivot_cols)]
        # build null space: one column per free variable
        basis = np.zeros((cols, len(free_cols)), dtype=np.int64)
        pivot_row = {pc: i for i, pc in enumerate(pivot_cols)}
        for j, fc in enumerate(free_cols):
            basis[fc, j] = 1
            for pc in pivot_cols:
                val = int(A[pivot_row[pc], fc])
                basis[pc, j] = (-val) % p
        return basis

    ns_sub = nullspace_mod(M_sub, p)  # shape (n_sub, dim_Z_sub)

    # Lift null space columns to C_k(K_full) basis
    lifted = np.zeros((len(k_full), ns_sub.shape[1]), dtype=np.int64)
    for i_sub, full_pos in enumerate(sub_positions):
        lifted[full_pos] = ns_sub[i_sub] % p

    # Build combined matrix [lifted | M_full_next] and compute rank mod p
    if M_full_next.size:
        combined = np.zeros((len(k_full), lifted.shape[1] + M_full_next.shape[1]), dtype=np.int64)
        combined[:, :lifted.shape[1]] = lifted
        combined[:, lifted.shape[1]:] = M_full_next % p
    else:
        combined = lifted

    rank_combined = _modular_rank(combined, p)
    rank_B_full = _modular_rank(M_full_next, p) if M_full_next.size else 0
    return rank_combined - rank_B_full
