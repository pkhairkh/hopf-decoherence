#!/usr/bin/env python3
"""
Compute dim H̃¹_b(B⁺(u_q(sl_2)), C) at ℓ = 3.

Cross-check for compute_h1b_bplus_sl3.py.  Expected: dim H̃¹_b = 1.

Mastnak–Witherspoon bialgebra cohomology H̃¹_b(B) at degree 1 (truncated
normalized complex) is the kernel of

    ∂_b : Hom(B̄, B̄) → Hom(B̄⊗B̄, B̄) ⊕ Hom(B̄, B̄⊗B̄)

    ∂_b h = (∂^h h, -∂^c h)

    (∂^h h)(a, b) = a·h(b) - h(a·b) + h(a)·b
    (∂^c h)(c)    = c₁⊗h(c₂) - Δ(h(c)) + h(c₁)⊗c₂

with a, b, c ∈ B̄ = ker ε.  Since (Tot B⁰_+)¹ = 0, B̃¹_b = 0 and
H̃¹_b = Z̃¹_b = ker(∂_b).

B⁺(u_q(sl_2)) at ℓ=3:
  Generators K, E with K³=1, E³=0, K E = q² E K.
  PBW basis {K^a E^b : 0 ≤ a, b ≤ 2}, dim = 9.
  ε(K^a E^b) = 1 if b=0 else 0.  dim B̄ = 8.
  Weights: wt(K^a E^b) = 2b mod 3.
  Coproduct: Δ(K) = K⊗K,  Δ(E) = E⊗K + 1⊗E.

We compute the kernel of ∂_b by weight-block decomposition (3 blocks for
sl_2; one per shift s ∈ Z/3), then dim H̃¹_b = Σ_s (dim input_s - rank_s).
"""
import cmath
import math
import time
import sys
import numpy as np
from scipy import sparse
from scipy.linalg import eigvalsh


# ============================================================
# Algebra setup
# ============================================================

ELL = 3
Q = cmath.exp(2j * math.pi / ELL)
Q_INV = Q ** (-1)
assert abs(Q + Q_INV - (-1)) < 1e-12, f"q + q^-1 = {Q + Q_INV}"
DIM = ELL * ELL  # 9


def idx(a, b):
    return a * ELL + b


def from_idx(i):
    return i // ELL, i % ELL


def weight(i):
    """E-weight of K^a E^b is 2b mod ℓ."""
    _, b = from_idx(i)
    return (2 * b) % ELL


# ============================================================
# Multiplication (PBW basis, dim 9)
# ============================================================

def multiply_monomials(a1, b1, a2, b2):
    """(K^a1 E^b1) * (K^a2 E^b2) = q^{-2 a2 b1} K^{a1+a2} E^{b1+b2}."""
    phase = Q ** (-2 * a2 * b1)
    a = (a1 + a2) % ELL
    b = b1 + b2
    if b >= ELL:
        return {}
    return {(a, b): phase}


def build_mult(dim):
    """ms[i][j] = [(l, v), ...]  for basis[i]*basis[j] = Σ v · basis[l]."""
    ms = [[[] for _ in range(dim)] for _ in range(dim)]
    for i in range(dim):
        a1, b1 = from_idx(i)
        for j in range(dim):
            a2, b2 = from_idx(j)
            for (a, b), v in multiply_monomials(a1, b1, a2, b2).items():
                ms[i][j].append((idx(a, b), v))
    return ms


# ============================================================
# Comultiplication (PBW basis, dim 9)
# ============================================================

def tensor_mult(d1, d2):
    """Multiply two elements of B⊗B (component-wise)."""
    new = {}
    for (l1, r1), v1 in d1.items():
        for (l2, r2), v2 in d2.items():
            left = multiply_monomials(*l1, *l2)
            right = multiply_monomials(*r1, *r2)
            for lk, lv in left.items():
                for rk, rv in right.items():
                    key = (lk, rk)
                    new[key] = new.get(key, 0) + v1 * v2 * lv * rv
    return {k: v for k, v in new.items() if abs(v) > 1e-13}


def delta_monomial(a, b):
    """Δ(K^a E^b) as a dict {(l_tuple, r_tuple): coeff}."""
    result = {((0, 0), (0, 0)): 1.0}
    for _ in range(a):
        result = tensor_mult(result, {((1, 0), (1, 0)): 1.0})  # · Δ(K) = K ⊗ K
    for _ in range(b):
        result = tensor_mult(result, {  # · Δ(E) = E ⊗ K + 1 ⊗ E
            ((0, 1), (1, 0)): 1.0,
            ((0, 0), (0, 1)): 1.0,
        })
    return result


def build_delta(dim):
    """Delta[c] = [(j, k, v), ...] for Δ(basis[c]) = Σ v · basis[j]⊗basis[k]."""
    Delta = [None] * dim
    for c in range(dim):
        a, b = from_idx(c)
        Delta[c] = [(idx(*lt), idx(*rt), v)
                    for (lt, rt), v in delta_monomial(a, b).items()]
    return Delta


def verify_coassoc(Delta, dim):
    """Verify (Δ⊗1)Δ = (1⊗Δ)Δ on all basis elements."""
    max_err = 0.0
    for c in range(dim):
        left, right = {}, {}
        for (j, k, v) in Delta[c]:
            for (j2, k2, v2) in Delta[j]:
                key = (j2, k2, k)
                left[key] = left.get(key, 0) + v * v2
            for (j2, k2, v2) in Delta[k]:
                key = (j, j2, k2)
                right[key] = right.get(key, 0) + v * v2
        for key in set(left) | set(right):
            max_err = max(max_err, abs(left.get(key, 0) - right.get(key, 0)))
    return max_err


def verify_counital(Delta, dim, epsilon):
    """Verify (ε⊗1)Δ = id and (1⊗ε)Δ = id."""
    max_err = 0.0
    for c in range(dim):
        for target in range(dim):
            s_l = sum(v * epsilon[j] for (j, k, v) in Delta[c] if k == target)
            s_r = sum(v * epsilon[k] for (j, k, v) in Delta[c] if j == target)
            expected = 1.0 if target == c else 0.0
            max_err = max(max_err, abs(s_l - expected), abs(s_r - expected))
    return max_err


# ============================================================
# B̄ basis and modified structure constants
# ============================================================
#
# B̄ = ker ε has dimension dim-1.  As a basis we use
#   b̄_i := basis[i] - ε[i]·basis[0]      for i = 1, ..., dim-1
# (so b̄_i = basis[i] when ε[i]=0, and b̄_i = basis[i]-1 when ε[i]=1).
#
# In this basis the structure constants pick up simple corrections
# (see W3-1c-h1b-computation.md for the derivation):
#   mult_bar[l, a, b] = mult[l, a, b] - ε[b]·δ_{l,a} - ε[a]·δ_{l,b}
#   delta_bar[c, j, k] = delta[c, j, k]              (just drop unit factors)
#
# These are restricted to indices in B̄ = {1, ..., dim-1}.

def build_epsilon(dim):
    eps = np.zeros(dim, dtype=complex)
    for a in range(ELL):
        eps[idx(a, 0)] = 1.0
    return eps


def build_mult_bar(ms, epsilon, dim):
    """Returns dict {(l, a, b): val} for a, b, l ∈ B̄."""
    mult_bar = {}
    for a in range(1, dim):
        for b in range(1, dim):
            terms = {l: v for (l, v) in ms[a][b]}
            for l in range(1, dim):
                v = terms.get(l, 0.0)
                if l == a:
                    v -= epsilon[b]
                if l == b:
                    v -= epsilon[a]
                if abs(v) > 1e-13:
                    mult_bar[(l, a, b)] = v
    return mult_bar


def build_delta_bar(Delta, dim):
    """Returns dict {(c, j, k): val} for c, j, k ∈ B̄."""
    delta_bar = {}
    for c in range(1, dim):
        for (j, k, v) in Delta[c]:
            if j == 0 or k == 0:
                continue
            delta_bar[(c, j, k)] = v
    return delta_bar


# ============================================================
# ∂_b per shift
# ============================================================

def compute_shift_block(s, n_weights, B_bar, weights,
                        mult_bar, delta_bar,
                        ms_by_b, ms_by_a, inv_mult_bar,
                        inv_delta_left, inv_delta_right, delta_by_c):
    """Build ∂_b_s as a sparse matrix and compute its rank.

    Returns (n_cols, n_rows_h, n_rows_c, rank, nullity).
    """
    # Input columns (j, k) with (wt(j) - wt(k)) mod n_weights == s.
    cols_list = [(j, k) for j in B_bar for k in B_bar
                 if (weights[j] - weights[k]) % n_weights == s]
    col_map = {pair: i for i, pair in enumerate(cols_list)}
    n_cols = len(cols_list)

    # ∂^h output rows (a, b, t) with (wt(t) - wt(a) - wt(b)) mod n_weights == s.
    h_rows_list = [(a, b, t) for a in B_bar for b in B_bar for t in B_bar
                   if (weights[t] - weights[a] - weights[b]) % n_weights == s]
    h_row_map = {triple: i for i, triple in enumerate(h_rows_list)}
    n_rows_h = len(h_rows_list)

    # ∂^c output rows (c, α, β) with (wt(α) + wt(β) - wt(c)) mod n_weights == s.
    c_rows_list = [(c, al, be) for c in B_bar for al in B_bar for be in B_bar
                   if (weights[al] + weights[be] - weights[c]) % n_weights == s]
    c_row_map = {triple: i for i, triple in enumerate(c_rows_list)}
    n_rows_c = len(c_rows_list)

    if n_cols == 0:
        return (0, n_rows_h, n_rows_c, 0, 0)

    rows_l, cols_l, vals_l = [], [], []

    for col_idx, (j, k) in enumerate(cols_list):
        # ∂^h term 1: row (a, k, t), val = mult_bar[t, a, j]
        # ms_by_b[j] = list of (a, t, v) with mult_bar[t, a, j] = v.
        for (a, t, v) in ms_by_b[j]:
            row = h_row_map.get((a, k, t))
            if row is not None:
                rows_l.append(row); cols_l.append(col_idx); vals_l.append(v)

        # ∂^h term 2: row (a, b, j), val = -mult_bar[k, a, b]
        # inv_mult_bar[k] = list of (a, b, v) with mult_bar[k, a, b] = v.
        for (a, b, v) in inv_mult_bar[k]:
            row = h_row_map.get((a, b, j))
            if row is not None:
                rows_l.append(row); cols_l.append(col_idx); vals_l.append(-v)

        # ∂^h term 3: row (k, b, t), val = mult_bar[t, j, b]
        # ms_by_a[j] = list of (b, t, v) with mult_bar[t, j, b] = v.
        for (b, t, v) in ms_by_a[j]:
            row = h_row_map.get((k, b, t))
            if row is not None:
                rows_l.append(row); cols_l.append(col_idx); vals_l.append(v)

        # -∂^c term 1: row (c, α, j), val = -delta_bar[c, α, k]
        # inv_delta_left[(c, k)] = list of (α, v) with delta_bar[c, α, k] = v.
        for c in B_bar:
            for (al, v) in inv_delta_left.get((c, k), []):
                row = c_row_map.get((c, al, j))
                if row is not None:
                    rows_l.append(n_rows_h + row); cols_l.append(col_idx); vals_l.append(-v)

        # -∂^c term 2: only c = k.  row (k, α, β), val = +delta_bar[j, α, β].
        # delta_by_c[j] = list of (α, β, v) with delta_bar[j, α, β] = v.
        for (al, be, v) in delta_by_c[j]:
            row = c_row_map.get((k, al, be))
            if row is not None:
                rows_l.append(n_rows_h + row); cols_l.append(col_idx); vals_l.append(v)

        # -∂^c term 3: row (c, j, β), val = -delta_bar[c, k, β]
        # inv_delta_right[(c, k)] = list of (β, v) with delta_bar[c, k, β] = v.
        for c in B_bar:
            for (be, v) in inv_delta_right.get((c, k), []):
                row = c_row_map.get((c, j, be))
                if row is not None:
                    rows_l.append(n_rows_h + row); cols_l.append(col_idx); vals_l.append(-v)

    n_rows_total = n_rows_h + n_rows_c
    A = sparse.csr_matrix(
        (np.array(vals_l, dtype=complex),
         (np.array(rows_l, dtype=np.int64), np.array(cols_l, dtype=np.int64))),
        shape=(n_rows_total, n_cols), dtype=complex)

    G = (A.conj().T @ A).toarray()
    if G.size == 0:
        rank = 0
    else:
        eigs = np.sort(np.abs(eigvalsh(G)))[::-1]
        if len(eigs) == 0 or eigs[0] == 0:
            rank = 0
        else:
            tol = max(G.shape) * eigs[0] * 1e-10
            rank = int(np.sum(eigs > tol))
    nullity = n_cols - rank
    return (n_cols, n_rows_h, n_rows_c, rank, nullity)


def compute(dim, verbose=True):
    if verbose:
        print(f"\n=== H̃¹_b(B⁺(u_q(sl_2)), C) at ℓ = {ELL} ===")
        print(f"dim B = {dim}, dim B̄ = {dim-1}")
        print(f"q = {Q}")

    t0 = time.time()
    ms = build_mult(dim)
    if verbose:
        print(f"Mult table built: {time.time()-t0:.2f}s")

    t0 = time.time()
    Delta = build_delta(dim)
    if verbose:
        print(f"Delta table built: {time.time()-t0:.2f}s")

    coassoc_err = verify_coassoc(Delta, dim)
    epsilon = build_epsilon(dim)
    counital_err = verify_counital(Delta, dim, epsilon)
    if verbose:
        print(f"Coassoc max error: {coassoc_err:.2e}")
        print(f"Counital max error: {counital_err:.2e}")
    assert coassoc_err < 1e-10, "Coassociativity failed!"
    assert counital_err < 1e-10, "Counitality failed!"

    B_bar = list(range(1, dim))
    mult_bar = build_mult_bar(ms, epsilon, dim)
    delta_bar = build_delta_bar(Delta, dim)
    if verbose:
        print(f"mult_bar nnz: {len(mult_bar)}")
        print(f"delta_bar nnz: {len(delta_bar)}")

    # Precompute inverse tables.
    inv_mult_bar = {l: [] for l in B_bar}
    ms_by_b = {b: [] for b in B_bar}
    ms_by_a = {a: [] for a in B_bar}
    for (l, a, b), v in mult_bar.items():
        inv_mult_bar[l].append((a, b, v))
        ms_by_b[b].append((a, l, v))   # mult_bar[l, a, b] = v  ⇒  for fixed b, gives (a, l, v)
        ms_by_a[a].append((b, l, v))   # mult_bar[l, a, b] = v  ⇒  for fixed a, gives (b, l, v)

    inv_delta_left = {}
    inv_delta_right = {}
    delta_by_c = {c: [] for c in B_bar}
    for (c, j, k), v in delta_bar.items():
        inv_delta_left.setdefault((c, k), []).append((j, v))
        inv_delta_right.setdefault((c, j), []).append((k, v))
        delta_by_c[c].append((j, k, v))

    weights = np.array([weight(i) for i in range(dim)])
    n_weights = ELL

    total_rank = 0
    total_input = 0
    per_shift = []
    t0 = time.time()
    for s in range(n_weights):
        ts = time.time()
        n_cols, n_rows_h, n_rows_c, rank, nullity = compute_shift_block(
            s, n_weights, B_bar, weights,
            mult_bar, delta_bar,
            ms_by_b, ms_by_a, inv_mult_bar,
            inv_delta_left, inv_delta_right, delta_by_c)
        total_rank += rank
        total_input += n_cols
        per_shift.append((s, n_cols, n_rows_h, n_rows_c, rank, nullity))
        if verbose:
            print(f"  shift {s}: cols={n_cols}, rows_h={n_rows_h}, rows_c={n_rows_c}, "
                  f"rank={rank}, nullity={nullity}  [{time.time()-ts:.2f}s]")

    dim_ker = total_input - total_rank
    if verbose:
        print(f"\nTotal input dim (dim Hom(B̄,B̄)): {total_input}")
        print(f"Total rank of ∂_b:                {total_rank}")
        print(f"dim H̃¹_b(B⁺(u_q(sl_2)), C)  = dim ker(∂_b) = {dim_ker}")
        print(f"Expected (conjecture at A₁, ℓ=3): 1")
        print(f"MATCH: {dim_ker == 1}")
        print(f"Total time: {time.time()-t0:.1f}s")
    return dim_ker


if __name__ == "__main__":
    result = compute(DIM, verbose=True)
    sys.exit(0 if result == 1 else 1)
