#!/usr/bin/env python3
"""
Bar-complex computation of HH^2(u_q(sl_2), C) using weight-space decomposition.

The algebra u_q(sl_2) decomposes by K-weight: u = ⊕_{a=0}^{ℓ-1} u_a.
The Hochschild cochains C^n(u, C) decompose by total weight c ∈ Z/ℓ:
  C^n = ⊕_c C^n_c,  where C^n_c = ⊕_{a_1+...+a_n ≡ c} (u_{a_1} ⊗ ... ⊗ u_{a_n})*
The differential preserves total weight, so we compute each weight block independently.

This makes ℓ = 5 feasible: each weight block of C^2 has dim ~15625/5 = 3125,
and C^3 has dim ~2M/5 = 400K.
"""
import cmath
import math
import time

import numpy as np
from scipy import sparse


def make_constants(ell):
    Q = cmath.exp(2j * math.pi / ell)
    return ell, Q, Q ** (-1), Q - Q ** (-1), ell ** 3


def idx(a, b, c, ell):
    return a * ell * ell + b * ell + c


def from_idx(i, ell):
    return i // (ell * ell), (i // ell) % ell, i % ell


def weight(i, ell):
    """E-F weight of basis element K^a E^b F^c is 2b - 2c (mod ell).
    This is the weight under the adjoint K-action, and it's the grading
    preserved by both the multiplication and the counit:
    - ε(K^a E^b F^c) = 1 if b=c=0 (weight 0), else 0. ✓
    - Product preserves total E-F weight (q-commutation doesn't change it). ✓
    """
    a, b, c = from_idx(i, ell)
    return (2 * b - 2 * c) % ell


def multiply_by_F_left(elem, ell, Q, Q_inv, D):
    """Multiply a normal-form element by F on the LEFT.

    Uses: F K^a = q^{2a} K^a F, and F E = E F - delta with delta = (K - K^{-1})/(q - q^{-1}).
    By induction: F E^b = E^b F - sum_{k=0}^{b-1} E^k delta E^{b-1-k}.
    Since E^k K = q^{-2k} K E^k and E^k K^{-1} = q^{2k} K^{-1} E^k:
      E^k delta E^{b-1-k} = [(q^{-2k} K - q^{2k} K^{-1}) / (q - q^{-1})] E^{b-1}
    So F * K^a E^b F^c = q^{2a} K^a E^b F^{c+1}
                       - q^{2a} sum_{k=0}^{b-1} [(q^{-2k} K^{a+1} - q^{2k} K^{a-1}) / (q - q^{-1})] E^{b-1} F^c
    Note: K^{-1} = K^{ell-1}, so K^{a-1} = K^{(a + ell - 1) % ell}.
    """
    new = {}
    for (a, b, c), v in elem.items():
        coeff_a = v * Q ** (2 * a)
        # Term 1: q^{2a} K^a E^b F^{c+1} (vanishes if c+1 == ell)
        if c + 1 < ell:
            key = (a, b, c + 1)
            new[key] = new.get(key, 0) + coeff_a
        # Term 2: correction from commuting F past E^b
        for k in range(b):
            # K^{a+1} with coefficient -q^{2a} q^{-2k} / D
            key1 = ((a + 1) % ell, b - 1, c)
            new[key1] = new.get(key1, 0) + (-coeff_a * Q ** (-2 * k) / D)
            # K^{a-1} = K^{(a + ell - 1) % ell} with coefficient +q^{2a} q^{2k} / D
            key2 = ((a + ell - 1) % ell, b - 1, c)
            new[key2] = new.get(key2, 0) + (coeff_a * Q ** (2 * k) / D)
    return {k: v for k, v in new.items() if abs(v) > 1e-13}


def multiply_by_E_left(elem, ell, Q, Q_inv, D):
    new = {}
    for (a, b, c), v in elem.items():
        if b + 1 < ell:
            coeff = v * Q ** (-2 * a)
            key = (a, b + 1, c)
            new[key] = new.get(key, 0) + coeff
    return new


def multiply_monomials(a, b, c, ap, bp, cp, ell, Q, Q_inv, D):
    factor = Q ** (2 * ap * (c - b))
    a1 = (a + ap) % ell
    middle = {(0, bp, 0): 1}
    for _ in range(c):
        middle = multiply_by_F_left(middle, ell, Q, Q_inv, D)
    for _ in range(b):
        middle = multiply_by_E_left(middle, ell, Q, Q_inv, D)
    after_F = {}
    for (a2, b2, c2), v in middle.items():
        c3 = c2 + cp
        if c3 < ell:
            after_F[(a2, b2, c3)] = after_F.get((a2, b2, c3), 0) + v
    final = {}
    for (a2, b2, c2), v in after_F.items():
        a3 = (a1 + a2) % ell
        final[(a3, b2, c2)] = final.get((a3, b2, c2), 0) + v * factor
    return {k: v for k, v in final.items() if abs(v) > 1e-13}


def build_mult(ell, Q, Q_inv, D, dim):
    mult = np.zeros((dim, dim, dim), dtype=complex)
    for i in range(dim):
        a, b, c = from_idx(i, ell)
        for j in range(dim):
            ap, bp, cp = from_idx(j, ell)
            prod = multiply_monomials(a, b, c, ap, bp, cp, ell, Q, Q_inv, D)
            for (a2, b2, c2), v in prod.items():
                mult[idx(a2, b2, c2, ell), i, j] = v
    return mult


def compute_hh2_weight_decomposed(ell):
    """Compute dim HH^2(u_q(sl_2), C) at the given ell using weight decomposition."""
    Q, Q_inv, D = cmath.exp(2j * math.pi / ell), cmath.exp(-2j * math.pi / ell), None
    D = Q - Q_inv
    dim = ell ** 3
    print(f"\n=== ℓ = {ell}, dim u_q(sl_2) = {dim} ===")
    print(f"  C^1 = {dim}, C^2 = {dim**2}, C^3 = {dim**3}")
    t0 = time.time()
    mult = build_mult(ell, Q, Q_inv, D, dim)
    print(f"  Multiplication table: {time.time()-t0:.1f}s, "
          f"nnz = {int(np.sum(np.abs(mult) > 1e-13))}")

    # Counit: ε(K^a E^b F^c) = 1 if b=c=0, else 0
    epsilon = np.zeros(dim, dtype=complex)
    for a in range(ell):
        epsilon[idx(a, 0, 0, ell)] = 1.0

    # Weight of basis element i
    wts = np.array([weight(i, ell) for i in range(dim)])

    # ================================================================
    # Build d^1 : C^1 -> C^2, decomposed by total weight.
    # C^1_c = (u_c)^*, dim = |{i : wts[i] == c}|
    # C^2_c = ⊕_{a+b≡c} (u_a ⊗ u_b)^*, indexed by (i,j) with wts[i]+wts[j] ≡ c
    # (d^1 f)(i, j) = ε(i) f(j) - f(i*j) + f(i) ε(j)
    # Total weight of (i,j) = wts[i] + wts[j] (mod ell)
    # ================================================================
    print(f"  Building d^1 by weight blocks...")
    t0 = time.time()

    # Index mapping: for each weight c, list the basis elements in that weight
    wt_indices = {c: [i for i in range(dim) if wts[i] == c] for c in range(ell)}
    wt_sizes = {c: len(wt_indices[c]) for c in range(ell)}
    print(f"  Weight sizes: {wt_sizes}")

    # For C^2, index by (i, j) pairs grouped by total weight
    # C^2_c: pairs (i, j) with wts[i] + wts[j] ≡ c (mod ell)
    c2_indices = {c: [] for c in range(ell)}
    for i in range(dim):
        for j in range(dim):
            c = (wts[i] + wts[j]) % ell
            c2_indices[c].append(i * dim + j)
    c2_sizes = {c: len(c2_indices[c]) for c in range(ell)}

    # For C^3, index by (i, j, k) triples grouped by total weight
    # We won't precompute all triples (too many); instead build d^2 per weight block

    total_rank_d1 = 0
    total_dim_ker_d2 = 0  # will accumulate after computing d^2

    # Store d^1 blocks for later use
    d1_blocks = {}

    for c in range(ell):
        rows_c2 = c2_indices[c]
        n_rows = len(rows_c2)
        n_cols = wt_sizes[c]  # C^1_c has this dimension
        if n_cols == 0 or n_rows == 0:
            d1_blocks[c] = None
            continue
        d1_c = np.zeros((n_rows, n_cols), dtype=complex)
        for col, i_f in enumerate(wt_indices[c]):  # f = indicator on basis[i_f]
            for row, ij in enumerate(rows_c2):
                i, j = ij // dim, ij % dim
                term1 = epsilon[i] * (1.0 if j == i_f else 0.0)
                term2 = mult[i_f, i, j]
                term3 = (1.0 if i == i_f else 0.0) * epsilon[j]
                d1_c[row, col] = term1 - term2 + term3
        d1_blocks[c] = d1_c
        # Rank
        s = np.linalg.svd(d1_c, compute_uv=False)
        tol = max(d1_c.shape) * (s[0] if len(s) > 0 else 0) * 1e-14
        rank_c = int(np.sum(s > tol)) if len(s) > 0 else 0
        total_rank_d1 += rank_c
    print(f"  d^1 built in {time.time()-t0:.1f}s, total rank = {total_rank_d1}")

    # ================================================================
    # Build d^2 : C^2 -> C^3, decomposed by total weight.
    # C^2_c: pairs (i,j) with wts[i]+wts[j] ≡ c
    # C^3_c: triples (i,j,k) with wts[i]+wts[j]+wts[k] ≡ c
    # (d^2 g)(i, j, k) = ε(i) g(j,k) - g(i*j, k) + g(i, j*k) - g(i,j) ε(k)
    # ================================================================
    print(f"  Building d^2 by weight blocks...")
    t0 = time.time()

    total_dim_ker_d2 = 0
    total_rank_d2 = 0

    for c in range(ell):
        # C^3_c: enumerate triples (i, j, k) with total weight c
        # This can be huge for ell=5: dim^3/ell ~ 400K
        rows_c3 = []
        for i in range(dim):
            for j in range(dim):
                ij_wt = (wts[i] + wts[j]) % ell
                k_wt = (c - ij_wt) % ell
                for k in wt_indices[k_wt]:
                    rows_c3.append((i, j, k))
        n_rows = len(rows_c3)
        n_cols = len(c2_indices[c])
        if n_rows == 0 or n_cols == 0:
            continue

        # Build sparse d^2 block
        from scipy.sparse import csr_matrix
        r_idx, c_idx, vals = [], [], []
        for row, (i, j, k) in enumerate(rows_c3):
            # term1: ε(i) * g(j, k)  -> col = index of (j,k) in c2_indices[c]
            ei = epsilon[i]
            if abs(ei) > 1e-13:
                jk_col = j * dim + k
                if jk_col in c2_indices[c]:
                    col = c2_indices[c].index(jk_col)
                    r_idx.append(row); c_idx.append(col); vals.append(ei)
            # term2: -g(i*j, k)
            for l in range(dim):
                v = mult[l, i, j]
                if abs(v) > 1e-13:
                    lk_col = l * dim + k
                    if lk_col in c2_indices[c]:
                        col = c2_indices[c].index(lk_col)
                        r_idx.append(row); c_idx.append(col); vals.append(-v)
            # term3: +g(i, j*k)
            for l in range(dim):
                v = mult[l, j, k]
                if abs(v) > 1e-13:
                    il_col = i * dim + l
                    if il_col in c2_indices[c]:
                        col = c2_indices[c].index(il_col)
                        r_idx.append(row); c_idx.append(col); vals.append(v)
            # term4: -g(i, j) * ε(k)
            ek = epsilon[k]
            if abs(ek) > 1e-13:
                ij_col = i * dim + j
                if ij_col in c2_indices[c]:
                    col = c2_indices[c].index(ij_col)
                    r_idx.append(row); c_idx.append(col); vals.append(-ek)

        d2_c = csr_matrix((vals, (r_idx, c_idx)), shape=(n_rows, n_cols), dtype=complex)
        # Rank via Gram matrix
        gram = (d2_c.conj().T @ d2_c).toarray()
        eigvals = np.sort(np.abs(np.linalg.eigvalsh(gram)))[::-1]
        if len(eigvals) == 0 or eigvals[0] == 0:
            rank_c = 0
        else:
            tol = max(d2_c.shape) * eigvals[0] * 1e-10
            rank_c = int(np.sum(eigvals > tol))
        dim_ker_c = n_cols - rank_c
        total_rank_d2 += rank_c
        total_dim_ker_d2 += dim_ker_c
        print(f"    weight {c}: C^2 dim = {n_cols}, C^3 dim = {n_rows}, "
              f"rank(d^2) = {rank_c}, dim ker(d^2) = {dim_ker_c}")

    print(f"  d^2 built in {time.time()-t0:.1f}s")
    print(f"  total rank(d^2) = {total_rank_d2}")
    print(f"  total dim ker(d^2) = {total_dim_ker_d2}")
    print(f"  total rank(d^1) = {total_rank_d1}")
    hh2 = total_dim_ker_d2 - total_rank_d1
    print(f"  dim HH^2 = {hh2}")
    expected = 1 + 2 * 1  # C(2,2) + 2|Phi^+(A_1)| = 3
    print(f"  Predicted = {expected}")
    print(f"  MATCH: {hh2 == expected}")
    return hh2


if __name__ == "__main__":
    for ell in [3, 5]:
        compute_hh2_weight_decomposed(ell)
