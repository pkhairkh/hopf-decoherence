#!/usr/bin/env python3
"""
Rigorous verification of HH^2(B+(u_q(sl_2)), C) at multiple ell values.

B+(sl_2) = C[K]/(K^ell - 1) ⋊ C[E]/(E^ell), with KE = q^2 EK.
Basis: {K^a E^b : 0 <= a,b < ell}, dimension ell^2.

Relations:
  K^ell = 1, E^ell = 0, KE = q^2 EK  (i.e., EK = q^{-2} KE)

This script computes HH^2 via the full bar complex (no weight decomposition)
and verifies:
  1. Multiplication table is correct (spot-check relations)
  2. Associativity on random triples
  3. d^2 o d^1 = 0 (chain complex property)
  4. Counit is an algebra map
  5. Spectral gap is clean
  6. dim HH^2 = ?
"""
import cmath, math
import numpy as np
from scipy import sparse
import random


def compute_hh2(ell, verbose=True):
    q = cmath.exp(2j * math.pi / ell)
    q_inv = q ** (-1)
    dim = ell * ell

    def idx(a, b): return a * ell + b
    def from_idx(i): return i // ell, i % ell

    # Build multiplication table: mult[k, i, j] = coeff of basis[k] in (basis[i] * basis[j])
    # (K^a1 E^b1)(K^a2 E^b2) = q^{-2*a2*b1} K^{a1+a2} E^{b1+b2}  (if b1+b2 < ell, else 0)
    mult = np.zeros((dim, dim, dim), dtype=complex)
    for i in range(dim):
        a1, b1 = from_idx(i)
        for j in range(dim):
            a2, b2 = from_idx(j)
            b_sum = b1 + b2
            if b_sum < ell:
                phase = q ** (-2 * a2 * b1)
                a_sum = (a1 + a2) % ell
                mult[idx(a_sum, b_sum), i, j] = phase

    if verbose:
        print(f"\n=== B+(sl_2) at ell = {ell}, dim = {dim} ===")

    # --- Sanity checks ---
    e0 = idx(0, 0)
    K = idx(1, 0); E = idx(0, 1)
    K1 = idx(1, 0); K2 = idx(2, 0) if ell >= 3 else None
    E1 = idx(0, 1); E2 = idx(0, 2) if ell >= 3 else None

    checks = []

    # 1 * x = x * 1 = x
    ok = all(abs(mult[i, e0, i] - 1) < 1e-12 and abs(mult[i, i, e0] - 1) < 1e-12 for i in range(dim))
    checks.append(("1*x = x*1 = x", ok))

    # K^ell = 1
    # K * K^{ell-1} = K^ell = 1 = e0
    Km1 = idx(ell - 1, 0)
    ok = abs(mult[e0, K, Km1] - 1.0) < 1e-12
    checks.append((f"K^{ell} = 1", ok))

    # E^ell = 0
    Em1 = idx(0, ell - 1)
    ok = all(abs(mult[i, Em1, E]) < 1e-12 for i in range(dim))
    checks.append((f"E^{ell} = 0", ok))

    # KE = q^2 EK  =>  mult[K,E] has K^1 E^1 with phase 1, mult[E,K] has K^1 E^1 with phase q^{-2}
    KE_idx = idx(1, 1)
    ok = (abs(mult[KE_idx, K, E] - 1.0) < 1e-12 and
          abs(mult[KE_idx, E, K] - q_inv**2) < 1e-12)
    checks.append(("KE = q^2 EK", ok))

    if verbose:
        for name, ok in checks:
            print(f"  {name}: {'OK' if ok else 'FAIL'}")

    # --- Associativity on 1000 random triples ---
    random.seed(42)
    fails = 0
    for _ in range(1000):
        i = random.randrange(dim); j = random.randrange(dim); k = random.randrange(dim)
        # (i*j)*k
        left = np.zeros(dim, dtype=complex)
        for l in range(dim):
            v = mult[l, i, j]
            if abs(v) > 1e-15:
                left += v * mult[:, l, k]
        # i*(j*k)
        right = np.zeros(dim, dtype=complex)
        for l in range(dim):
            v = mult[l, j, k]
            if abs(v) > 1e-15:
                right += v * mult[:, i, l]
        if np.max(np.abs(left - right)) > 1e-9:
            fails += 1
    if verbose:
        print(f"  Associativity (1000 triples): {fails} failures")
    assert fails == 0, "Associativity failed!"

    # --- Counit ---
    epsilon = np.zeros(dim, dtype=complex)
    for a in range(ell):
        epsilon[idx(a, 0)] = 1.0  # eps(K^a) = 1, eps(K^a E^b) = 0 for b > 0

    # Verify counit is algebra map: eps(x*y) = eps(x)*eps(y)
    ok = True
    for i in range(dim):
        for j in range(dim):
            exy = np.sum(epsilon * mult[:, i, j])
            ex = epsilon[i]; ey = epsilon[j]
            if abs(exy - ex * ey) > 1e-10:
                ok = False; break
        if not ok: break
    if verbose:
        print(f"  Counit is algebra map: {'OK' if ok else 'FAIL'}")
    assert ok, "Counit is not an algebra map!"

    # --- Build d^1: C^1 -> C^2 ---
    # (d^1 f)(i, j) = eps(i) f(j) - f(i*j) + f(i) eps(j)
    # For f = indicator on basis[k]:
    #   (d^1 e_k)(i, j) = eps(i) * [j==k] - mult[k, i, j] + [i==k] * eps(j)
    dim_c1 = dim
    dim_c2 = dim * dim
    d1 = np.zeros((dim_c2, dim_c1), dtype=complex)
    for k in range(dim):
        for i in range(dim):
            for j in range(dim):
                row = i * dim + j
                d1[row, k] = (epsilon[i] * (1.0 if j == k else 0.0)
                              - mult[k, i, j]
                              + (1.0 if i == k else 0.0) * epsilon[j])

    # --- Build d^2: C^2 -> C^3 (sparse) ---
    # (d^2 g)(i, j, k) = eps(i) g(j,k) - g(i*j, k) + g(i, j*k) - g(i,j) eps(k)
    dim_c3 = dim ** 3
    rows_l, cols_l, vals_l = [], [], []
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                row = i * dim * dim + j * dim + k
                # term1: eps(i) * g(j, k)
                ei = epsilon[i]
                if abs(ei) > 1e-15:
                    rows_l.append(row); cols_l.append(j * dim + k); vals_l.append(ei)
                # term2: -g(i*j, k) = -sum_l mult[l,i,j] * g(l, k)
                for l in range(dim):
                    v = mult[l, i, j]
                    if abs(v) > 1e-15:
                        rows_l.append(row); cols_l.append(l * dim + k); vals_l.append(-v)
                # term3: +g(i, j*k) = +sum_l mult[l,j,k] * g(i, l)
                for l in range(dim):
                    v = mult[l, j, k]
                    if abs(v) > 1e-15:
                        rows_l.append(row); cols_l.append(i * dim + l); vals_l.append(v)
                # term4: -g(i, j) * eps(k)
                ek = epsilon[k]
                if abs(ek) > 1e-15:
                    rows_l.append(row); cols_l.append(i * dim + j); vals_l.append(-ek)
    d2 = sparse.csr_matrix((vals_l, (rows_l, cols_l)), shape=(dim_c3, dim_c2), dtype=complex)

    # --- Verify d^2 o d^1 = 0 ---
    prod = d2 @ d1
    if hasattr(prod, 'toarray'):
        prod = prod.toarray()
    max_d2d1 = np.max(np.abs(prod))
    if verbose:
        print(f"  ||d^2 o d^1|| = {max_d2d1:.2e}")
    assert max_d2d1 < 1e-10, "d^2 o d^1 != 0!"

    # --- Ranks ---
    s1 = np.linalg.svd(d1, compute_uv=False)
    tol1 = max(d1.shape) * s1[0] * 1e-14
    rank1 = int(np.sum(s1 > tol1))

    gram = (d2.conj().T @ d2).toarray()
    eigvals = np.sort(np.abs(np.linalg.eigvalsh(gram)))[::-1]
    if len(eigvals) == 0 or eigvals[0] == 0:
        rank2 = 0
    else:
        tol2 = max(d2.shape) * eigvals[0] * 1e-10
        rank2 = int(np.sum(eigvals > tol2))

    dim_ker_d2 = dim_c2 - rank2
    hh2 = dim_ker_d2 - rank1

    if verbose:
        print(f"  rank(d^1) = {rank1}")
        print(f"  rank(d^2) = {rank2}")
        print(f"  dim ker(d^2) = {dim_ker_d2}")
        print(f"  dim HH^2 = {hh2}")
        # Spectral gap
        if rank2 < len(eigvals):
            gap_ratio = eigvals[rank2 - 1] / max(eigvals[rank2], 1e-300) if eigvals[rank2] > 0 else float('inf')
            print(f"  Spectral gap: {eigvals[rank2-1]:.3e} -> {eigvals[rank2]:.3e} (ratio {gap_ratio:.1e})")

    return hh2


if __name__ == "__main__":
    results = {}
    for ell in [3, 5, 7]:
        hh2 = compute_hh2(ell, verbose=True)
        results[ell] = hh2

    print("\n" + "=" * 60)
    print("SUMMARY: dim HH^2(B+(sl_2), C)")
    print("=" * 60)
    for ell, hh2 in results.items():
        predicted = 2 * 1 - 1  # 2|Phi+| - 1 = 1
        print(f"  ell={ell}: dim = {hh2}, predicted (2|Phi+|-1) = {predicted}, match = {hh2 == predicted}")
