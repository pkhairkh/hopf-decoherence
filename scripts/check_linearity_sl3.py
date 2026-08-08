#!/usr/bin/env python3
"""
Verify that the sl_3 diagonal cocycles are LINEAR functions of the PBW exponents.
If so, the sl_4 cocycles are likely also linear, and we can use the linear
ansatz to compute the sl_4 kernel without building the full multiplication table.
"""
import os
import sys
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compute_h1b_bplus_sl3 as sl3


def main():
    print("Building sl_3 tables...")
    ms = sl3.build_mult(sl3.DIM)
    sl3.ms_global = ms
    Delta = sl3.build_delta(sl3.DIM)
    epsilon = sl3.build_epsilon(sl3.DIM)
    mult_bar = sl3.build_mult_bar(ms, epsilon, sl3.DIM)
    delta_bar = sl3.build_delta_bar(Delta, sl3.DIM)

    B_bar = list(range(1, sl3.DIM))
    n = len(B_bar)
    idx_map = {b: i for i, b in enumerate(B_bar)}

    # Build constraint matrix
    rows = []
    cols = []
    vals = []
    n_h = 0
    for (l, a, b) in mult_bar.keys():
        rows.append(n_h); cols.append(idx_map[a]); vals.append(1.0)
        rows.append(n_h); cols.append(idx_map[b]); vals.append(1.0)
        rows.append(n_h); cols.append(idx_map[l]); vals.append(-1.0)
        n_h += 1
    n_c = 0
    for (c, j, k) in delta_bar.keys():
        rows.append(n_h + n_c); cols.append(idx_map[j]); vals.append(1.0)
        rows.append(n_h + n_c); cols.append(idx_map[k]); vals.append(1.0)
        rows.append(n_h + n_c); cols.append(idx_map[c]); vals.append(-1.0)
        n_c += 1
    n_rows = n_h + n_c
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(n_rows, n), dtype=complex)
    G = (A.conj().T @ A).tocsr()
    G = (G + G.conj().T) * 0.5

    # Get the 2 kernel vectors
    eigvals, eigvecs = eigsh(G, k=2, sigma=0, which='LM',
                              return_eigenvectors=True, tol=1e-14, maxiter=50000)
    print(f"Smallest 2 |eigvals|: {np.abs(eigvals)}")

    # For each kernel vector, check if it's linear in the exponents.
    # The PBW exponents for basis[i] are (a, b, c, e, d) from from_idx(i).
    # A linear function: c_i = α_1 * a + α_2 * b + α_3 * c + α_4 * e + α_5 * d
    # We can check linearity by solving a least-squares problem.
    print("\nChecking linearity of each cocycle...")
    for k in range(2):
        v = eigvecs[:, k]
        # Normalize
        v = v / np.max(np.abs(v))
        # Build the design matrix: rows are basis elements, columns are exponents (a, b, c, e, d) + constant
        X = np.zeros((n, 6), dtype=complex)
        for i, b_idx in enumerate(B_bar):
            a, b, c, e, d = sl3.from_idx(b_idx)
            X[i, 0] = a
            X[i, 1] = b
            X[i, 2] = c
            X[i, 3] = e
            X[i, 4] = d
            X[i, 5] = 1.0  # constant term
        # Solve least-squares: v = X @ coeffs
        coeffs, residuals, rank, sv = np.linalg.lstsq(X, v, rcond=None)
        residual = np.linalg.norm(v - X @ coeffs)
        rel_residual = residual / np.linalg.norm(v)
        print(f"\nCocycle {k+1}:")
        print(f"  Coefficients (a, b, c, e, d, const): {coeffs}")
        print(f"  Relative residual (linearity check): {rel_residual:.3e}")
        if rel_residual < 1e-6:
            print(f"  → Cocycle is LINEAR in PBW exponents ✓")
        else:
            print(f"  → Cocycle is NOT linear; check quadratic terms...")

    # Also check the sl_2 case for comparison.
    print("\n" + "="*60)
    print("Checking sl_2 cocycle linearity...")
    import compute_h1b_bplus_sl2 as sl2
    ms2 = sl2.build_mult(sl2.DIM)
    Delta2 = sl2.build_delta(sl2.DIM)
    eps2 = sl2.build_epsilon(sl2.DIM)
    mult_bar2 = sl2.build_mult_bar(ms2, eps2, sl2.DIM)
    delta_bar2 = sl2.build_delta_bar(Delta2, sl2.DIM)

    B_bar2 = list(range(1, sl2.DIM))
    n2 = len(B_bar2)
    idx_map2 = {b: i for i, b in enumerate(B_bar2)}
    rows2, cols2, vals2 = [], [], []
    n_h2 = 0
    for (l, a, b) in mult_bar2.keys():
        rows2.append(n_h2); cols2.append(idx_map2[a]); vals2.append(1.0)
        rows2.append(n_h2); cols2.append(idx_map2[b]); vals2.append(1.0)
        rows2.append(n_h2); cols2.append(idx_map2[l]); vals2.append(-1.0)
        n_h2 += 1
    n_c2 = 0
    for (c, j, k) in delta_bar2.keys():
        rows2.append(n_h2 + n_c2); cols2.append(idx_map2[j]); vals2.append(1.0)
        rows2.append(n_h2 + n_c2); cols2.append(idx_map2[k]); vals2.append(1.0)
        rows2.append(n_h2 + n_c2); cols2.append(idx_map2[c]); vals2.append(-1.0)
        n_c2 += 1
    A2 = sparse.csr_matrix((vals2, (rows2, cols2)), shape=(n_h2+n_c2, n2), dtype=complex)
    G2 = (A2.conj().T @ A2).tocsr()
    G2 = (G2 + G2.conj().T) * 0.5
    eigvals2, eigvecs2 = eigsh(G2, k=1, sigma=0, which='LM',
                                return_eigenvectors=True, tol=1e-14, maxiter=50000)
    v = eigvecs2[:, 0]
    v = v / np.max(np.abs(v))
    X2 = np.zeros((n2, 3), dtype=complex)
    for i, b_idx in enumerate(B_bar2):
        a, b = sl2.from_idx(b_idx)
        X2[i, 0] = a
        X2[i, 1] = b
        X2[i, 2] = 1.0
    coeffs2, _, _, _ = np.linalg.lstsq(X2, v, rcond=None)
    residual2 = np.linalg.norm(v - X2 @ coeffs2)
    rel_residual2 = residual2 / np.linalg.norm(v)
    print(f"sl_2 cocycle coefficients (a, b, const): {coeffs2}")
    print(f"  Relative residual: {rel_residual2:.3e}")
    if rel_residual2 < 1e-6:
        print(f"  → sl_2 cocycle is LINEAR ✓")


if __name__ == "__main__":
    main()
