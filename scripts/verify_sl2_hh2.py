#!/usr/bin/env python3
"""
Direct Hochschild bar-complex computation of HH^2(u_q(sl_2), C) at ell = 3.

Basis of u_q(sl_2): { K^a E^b F^c : 0 <= a, b, c < ell }.

Relations (q = e^{2*pi*i/ell}, ell = 3):
    K^ell  = 1
    E^ell  = F^ell = 0
    K E K^{-1} = q^2 E      ->   K E = q^2 E K,  E K = q^{-2} K E
    K F K^{-1} = q^{-2} F   ->   K F = q^{-2} F K,  F K = q^2 K F
    [E, F] = (K - K^{-1}) / (q - q^{-1})

Hochschild differentials (trivial coefficients in C):
    (d^1 f)(a, b)   = eps(a) f(b) - f(a*b) + f(a) eps(b)
    (d^2 g)(a, b, c)= eps(a) g(b,c) - g(a*b, c) + g(a, b*c) - g(a, b) eps(c)

Then dim HH^2 = dim ker(d^2) - dim im(d^1).

This script verifies the A_1 case of the paper's main formula:
    dim HH^2(u_q(sl_2), C) = C(2,2) + 2*1 = 3.
"""
import cmath
import math
import time

import numpy as np
from scipy import sparse


ELL = 3
Q = cmath.exp(2j * math.pi / ELL)
Q_INV = Q ** (-1)
D = Q - Q_INV                 # = i*sqrt(3) for ell = 3
DIM = ELL ** 3                # 27


def idx(a: int, b: int, c: int) -> int:
    """Linear index of basis element K^a E^b F^c."""
    return a * ELL * ELL + b * ELL + c


def from_idx(i: int):
    return i // (ELL * ELL), (i // ELL) % ELL, i % ELL


# --------------------------------------------------------------------------
# Algebra multiplication.
# --------------------------------------------------------------------------
def multiply_by_F_left(elem):
    """Multiply a normal-form element (dict) by F on the LEFT.

    Uses: F K^a = q^{2a} K^a F, and F E = E F - delta with delta = (K - K^{-1})/(q - q^{-1}).
    By induction: F E^b = E^b F - sum_{k=0}^{b-1} E^k delta E^{b-1-k}.
    Since E^k K = q^{-2k} K E^k and E^k K^{-1} = q^{2k} K^{-1} E^k:
      E^k delta E^{b-1-k} = [(q^{-2k} K - q^{2k} K^{-1}) / (q - q^{-1})] E^{b-1}
    So F * K^a E^b F^c = q^{2a} K^a E^b F^{c+1}
                       - q^{2a} sum_{k=0}^{b-1} [(q^{-2k} K^{a+1} - q^{2k} K^{a-1}) / (q - q^{-1})] E^{b-1} F^c
    Note: K^{-1} = K^{ELL-1}, so K^{a-1} = K^{(a + ELL - 1) % ELL}.
    """
    new = {}
    for (a, b, c), v in elem.items():
        coeff_a = v * Q ** (2 * a)
        # Term 1: q^{2a} K^a E^b F^{c+1} (vanishes if c+1 == ell)
        if c + 1 < ELL:
            key = (a, b, c + 1)
            new[key] = new.get(key, 0) + coeff_a
        # Term 2: correction from commuting F past E^b
        for k in range(b):
            # K^{a+1} with coefficient -q^{2a} q^{-2k} / D
            key1 = ((a + 1) % ELL, b - 1, c)
            new[key1] = new.get(key1, 0) + (-coeff_a * Q ** (-2 * k) / D)
            # K^{a-1} = K^{(a + ELL - 1) % ELL} with coefficient +q^{2a} q^{2k} / D
            key2 = ((a + ELL - 1) % ELL, b - 1, c)
            new[key2] = new.get(key2, 0) + (coeff_a * Q ** (2 * k) / D)
    return {k: v for k, v in new.items() if abs(v) > 1e-13}


def multiply_by_E_left(elem):
    """Multiply a normal-form element by E on the LEFT.

    Uses E K^a = q^{-2a} K^a E.  Hence
        E * K^a E^b F^c = q^{-2a} K^a E^{b+1} F^c   (vanishes if b+1 == ell).
    """
    new = {}
    for (a, b, c), v in elem.items():
        if b + 1 < ELL:
            coeff = v * Q ** (-2 * a)
            key = (a, b + 1, c)
            new[key] = new.get(key, 0) + coeff
    return new


def multiply_by_K_left(elem):
    """Multiply a normal-form element by K on the LEFT (K^3 = 1)."""
    new = {}
    for (a, b, c), v in elem.items():
        a1 = (a + 1) % ELL
        new[(a1, b, c)] = new.get((a1, b, c), 0) + v
    return new


def multiply_monomials(a, b, c, ap, bp, cp):
    """Return (K^a E^b F^c) * (K^{a'} E^{b'} F^{c'}) as a dict {(a,b,c): coeff}."""
    # Step 1: pull K^{a'} to the left past E^b F^c.
    #   F^c K^{a'} = q^{2 a' c} K^{a'} F^c   (each F moves past K^{a'} with q^{2 a'})
    #   E^b K^{a'} = q^{-2 a' b} K^{a'} E^b  (each E moves past K^{a'} with q^{-2 a'})
    # So the total scalar is q^{2 a'(c - b)}.
    factor = Q ** (2 * ap * (c - b))
    a1 = (a + ap) % ELL

    # Step 2: reduce the "middle" part F^c E^{b'} to normal form by applying
    # F-on-the-left c times to E^{b'}.
    middle = {(0, bp, 0): 1}
    for _ in range(c):
        middle = multiply_by_F_left(middle)

    # Step 3: now we have factor * K^{a1} * E^b * (middle) * F^{c'}.
    # Multiply middle by E^b on the LEFT (b applications of multiply_by_E_left).
    for _ in range(b):
        middle = multiply_by_E_left(middle)

    # Step 4: multiply by F^{c'} on the RIGHT (easy: just adds to F-exponent).
    after_F = {}
    for (a2, b2, c2), v in middle.items():
        c3 = c2 + cp
        if c3 < ELL:
            after_F[(a2, b2, c3)] = after_F.get((a2, b2, c3), 0) + v

    # Step 5: multiply by K^{a1} on the LEFT (a1 applications of multiply_by_K_left).
    final = {}
    for (a2, b2, c2), v in after_F.items():
        a3 = (a1 + a2) % ELL
        final[(a3, b2, c2)] = final.get((a3, b2, c2), 0) + v * factor
    return {k: v for k, v in final.items() if abs(v) > 1e-13}


def build_multiplication_table():
    """mult[k, i, j] = coefficient of basis[k] in (basis[i] * basis[j])."""
    mult = np.zeros((DIM, DIM, DIM), dtype=complex)
    for i in range(DIM):
        a, b, c = from_idx(i)
        for j in range(DIM):
            ap, bp, cp = from_idx(j)
            prod = multiply_monomials(a, b, c, ap, bp, cp)
            for (a2, b2, c2), v in prod.items():
                mult[idx(a2, b2, c2), i, j] = v
    return mult


# --------------------------------------------------------------------------
# Sanity checks.
# --------------------------------------------------------------------------
def sanity_checks(mult):
    """Verify a few algebraic identities."""
    print("Sanity checks:")

    # 1) Identity: 1 * x = x * 1 = x.
    e0 = idx(0, 0, 0)
    for i in range(DIM):
        assert abs(mult[i, e0, i] - 1.0) < 1e-12, f"1 * x failed at {i}"
        assert abs(mult[i, i, e0] - 1.0) < 1e-12, f"x * 1 failed at {i}"
    print("  1 * x = x * 1 = x  OK")

    # 2) K^3 = 1: K * K * K = 1.
    K = idx(1, 0, 0)
    K2 = idx(2, 0, 0)
    # K * K = K^2
    assert abs(mult[K2, K, K] - 1.0) < 1e-12, "K * K != K^2"
    # K * K^2 = 1
    assert abs(mult[e0, K, K2] - 1.0) < 1e-12, "K * K^2 != 1"
    print("  K^3 = 1  OK")

    # 3) E^3 = 0: E * E * E should be the zero vector.
    E = idx(0, 1, 0)
    # First compute E * E.
    EE = mult[:, E, E]   # vector
    # Then (E*E) * E.
    EEE = np.tensordot(EE, mult[:, :, E], axes=([0], [0]))  # E*E * E... hmm
    # Cleaner: just check that multiplying any basis element containing E^2 by E gives 0.
    E2_idx = idx(0, 2, 0)
    for i in range(DIM):
        assert abs(mult[i, E2_idx, E]) < 1e-12, f"E^2 * E != 0 at {i}"
    print("  E^3 = 0  OK")

    # 4) F^3 = 0.
    F = idx(0, 0, 1)
    F2_idx = idx(0, 0, 2)
    for i in range(DIM):
        assert abs(mult[i, F2_idx, F]) < 1e-12, f"F^2 * F != 0 at {i}"
    print("  F^3 = 0  OK")

    # 5) K E = q^2 E K.
    # K * E = q^2 * E * K.
    # In normal form: K * E = K^1 E^1 F^0, which is basis idx(1, 1, 0).
    # E * K = q^{-2} K E, i.e., E K = q^{-2} * (K^1 E^1 F^0).
    # So mult[idx(1,1,0), K, E] should be 1, and mult[idx(1,1,0), E, K] should be q^{-2}.
    KE_idx = idx(1, 1, 0)
    assert abs(mult[KE_idx, K, E] - 1.0) < 1e-12, f"K*E != K E: {mult[KE_idx, K, E]}"
    assert abs(mult[KE_idx, E, K] - Q_INV**2) < 1e-12, f"E*K != q^-2 K E: {mult[KE_idx, E, K]}"
    print(f"  K E = q^2 E K  OK  (q^2 = {Q**2:.4f}, q^-2 = {Q_INV**2:.4f})")

    # 6) [E, F] = (K - K^{-1})/(q - q^{-1}).
    # E F - F E should equal (K - K^2)/(q - q^{-1}).
    EF_minus_FE = mult[:, E, F] - mult[:, F, E]
    expected = np.zeros(DIM, dtype=complex)
    expected[K]  = 1.0 / D
    expected[K2] = -1.0 / D
    diff = np.max(np.abs(EF_minus_FE - expected))
    assert diff < 1e-12, f"[E,F] check failed: max diff = {diff}"
    print(f"  [E, F] = (K - K^-1)/(q - q^-1)  OK  (1/d = {1.0/D:.4f})")

    print()


# --------------------------------------------------------------------------
# Differentials.
# --------------------------------------------------------------------------
def build_d1(mult, epsilon):
    """d^1: C^1 -> C^2. Shape (DIM^2, DIM)."""
    N2 = DIM * DIM
    d1 = np.zeros((N2, DIM), dtype=complex)
    for i in range(DIM):  # f = indicator on basis[i]
        for a in range(DIM):
            for b in range(DIM):
                row = a * DIM + b
                term1 = epsilon[a] * (1.0 if b == i else 0.0)
                term2 = mult[i, a, b]
                term3 = (1.0 if a == i else 0.0) * epsilon[b]
                d1[row, i] = term1 - term2 + term3
    return d1


def build_d2_sparse(mult, epsilon):
    """d^2: C^2 -> C^3. Shape (DIM^3, DIM^2). Returned as CSR sparse."""
    N2 = DIM * DIM
    N3 = DIM ** 3
    rows, cols, vals = [], [], []
    for a in range(DIM):
        for b in range(DIM):
            # Precompute products a*b and b*c once per (a, b, c) pair — but c varies, so b*c depends on c.
            for c in range(DIM):
                row = a * DIM * DIM + b * DIM + c
                # Term 1: eps(a) * g(b, c)
                ea = epsilon[a]
                if abs(ea) > 1e-13:
                    rows.append(row); cols.append(b * DIM + c); vals.append(ea)
                # Term 2: - g(a*b, c)   -- (a*b) = sum_k mult[k, a, b] basis[k]
                for k in range(DIM):
                    v = mult[k, a, b]
                    if abs(v) > 1e-13:
                        rows.append(row); cols.append(k * DIM + c); vals.append(-v)
                # Term 3: + g(a, b*c)   -- (b*c) = sum_k mult[k, b, c] basis[k]
                for k in range(DIM):
                    v = mult[k, b, c]
                    if abs(v) > 1e-13:
                        rows.append(row); cols.append(a * DIM + k); vals.append(v)
                # Term 4: - g(a, b) * eps(c)
                ec = epsilon[c]
                if abs(ec) > 1e-13:
                    rows.append(row); cols.append(a * DIM + b); vals.append(-ec)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(N3, N2), dtype=complex)


def matrix_rank_complex(mat, tol=1e-9):
    """Compute rank of a complex matrix via SVD (real singular values)."""
    s = np.linalg.svd(mat, compute_uv=False)
    if len(s) == 0:
        return 0, np.array([])
    s_max = s[0]
    return int(np.sum(s > tol * s_max)), s


def sparse_rank_via_gram(mat_csr, tol=1e-9):
    """Compute rank of a (tall) sparse matrix via the Gram matrix.

    rank(A) = rank(A^* A).  A^* A is small (N_cols x N_cols) and Hermitian,
    so we use eigh on its dense form.
    """
    n = mat_csr.shape[1]
    gram = (mat_csr.conj().T @ mat_csr).toarray()
    eigvals = np.linalg.eigvalsh(gram)  # real, ascending
    eigvals = np.sort(np.abs(eigvals))[::-1]
    if eigvals[0] == 0:
        return 0, eigvals
    return int(np.sum(eigvals > tol * eigvals[0])), eigvals


# --------------------------------------------------------------------------
# Main.
# --------------------------------------------------------------------------
def main():
    print(f"=== HH^2(u_q(sl_2), C) at ell = {ELL} ===")
    print(f"q = {Q},  q - q^-1 = {D}")
    print(f"dim u_q(sl_2) = {DIM}")
    print(f"dim C^1 = {DIM},  dim C^2 = {DIM**2},  dim C^3 = {DIM**3}")
    print()

    t0 = time.time()
    mult = build_multiplication_table()
    print(f"Multiplication table built in {time.time()-t0:.1f}s, "
          f"nonzero entries: {int(np.sum(np.abs(mult) > 1e-13))}")
    sanity_checks(mult)

    epsilon = np.zeros(DIM, dtype=complex)
    # Counit: eps(K^a E^b F^c) = 1 if b = c = 0, else 0  (eps(K) = eps(E) = eps(F)... actually
    # eps(K) = 1, eps(E) = eps(F) = 0, extended as algebra map.  So eps(K^a E^b F^c) = 1^a * 0^b * 0^c,
    # which is 1 if b = c = 0 and 0 otherwise.)
    for a in range(ELL):
        epsilon[idx(a, 0, 0)] = 1.0

    print("Building d^1 ...")
    t0 = time.time()
    d1 = build_d1(mult, epsilon)
    rank1, s1 = matrix_rank_complex(d1)
    print(f"  rank(d^1) = {rank1}  (singular values: {s1[:rank1+2]})  "
          f"[{time.time()-t0:.1f}s]")

    print("Building d^2 (sparse) ...")
    t0 = time.time()
    d2 = build_d2_sparse(mult, epsilon)
    nnz = d2.nnz
    print(f"  d^2 has {nnz} nonzeros  [{time.time()-t0:.1f}s]")

    print("Computing rank(d^2) via Gram matrix ...")
    t0 = time.time()
    rank2, eigvals = sparse_rank_via_gram(d2)
    print(f"  rank(d^2) = {rank2}  "
          f"(top 10 |eigvals|: {eigvals[:10]})  "
          f"[{time.time()-t0:.1f}s]")
    print(f"  bottom 10 |eigvals|: {eigvals[-10:]}")

    dim_ker_d2 = DIM * DIM - rank2
    hh2 = dim_ker_d2 - rank1
    print()
    print(f"  dim ker(d^2)        = {dim_ker_d2}")
    print(f"  dim im(d^1)         = {rank1}")
    print(f"  dim HH^2            = {hh2}")
    print()
    expected = 3  # C(2,2) + 2*|Phi^+(A_1)| = 1 + 2 = 3
    print(f"  Paper's prediction  = {expected}")
    print(f"  MATCH: {hh2 == expected}")
    return hh2


if __name__ == "__main__":
    main()
