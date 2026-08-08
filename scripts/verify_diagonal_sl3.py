#!/usr/bin/env python3
"""
Verify that the diagonal-cocycle sub-block captures the full kernel of ∂_b
for sl_3 at ℓ=3.

For a diagonal h: B̄ → B̄ (h(basis[i]) = c_i * basis[i]), the conditions
∂^h h = 0 and ∂^c h = 0 become LINEAR constraints on the c_i's:

    c_a + c_b - c_l = 0    for each (l, a, b) with mult_bar[l, a, b] ≠ 0
    c_j + c_k - c_c = 0    for each (c, j, k) with delta_bar[c, j, k] ≠ 0

The kernel of this linear system is the "diagonal kernel". If the sl_3
cocycles are diagonal (which we observed by extraction), this should equal
the full kernel of ∂_b.

Expected: dim diagonal kernel = 2 (matching dim H̃¹_b(B⁺(sl_3)) = 2).
"""
import os
import sys
import time
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
    n = len(B_bar)  # 242
    idx_map = {b: i for i, b in enumerate(B_bar)}

    print(f"dim B̄ = {n}")
    print(f"nnz(mult_bar) = {len(mult_bar)}")
    print(f"nnz(delta_bar) = {len(delta_bar)}")

    # Build constraint matrix A: each row is a constraint c_a + c_b - c_l = 0
    # (or c_j + c_k - c_c = 0). Variables are c_i for i ∈ B̄.
    rows = []
    cols = []
    vals = []

    # ∂^h constraints: c_a + c_b - c_l = 0 for each (l, a, b) with mult_bar ≠ 0.
    n_h = 0
    for (l, a, b) in mult_bar.keys():
        rows.append(n_h); cols.append(idx_map[a]); vals.append(1.0)
        rows.append(n_h); cols.append(idx_map[b]); vals.append(1.0)
        rows.append(n_h); cols.append(idx_map[l]); vals.append(-1.0)
        n_h += 1

    # ∂^c constraints: c_j + c_k - c_c = 0 for each (c, j, k) with delta_bar ≠ 0.
    n_c = 0
    for (c, j, k) in delta_bar.keys():
        rows.append(n_h + n_c); cols.append(idx_map[j]); vals.append(1.0)
        rows.append(n_h + n_c); cols.append(idx_map[k]); vals.append(1.0)
        rows.append(n_h + n_c); cols.append(idx_map[c]); vals.append(-1.0)
        n_c += 1

    n_rows = n_h + n_c
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(n_rows, n), dtype=complex)
    print(f"Constraint matrix A: shape = {A.shape}, nnz = {A.nnz}")

    # Compute kernel via Gram matrix eigsh.
    G = (A.conj().T @ A).tocsr()
    G = (G + G.conj().T) * 0.5
    print(f"Gram matrix G: shape = {G.shape}, nnz = {G.nnz}")

    # Find smallest eigenvalues.
    k = min(20, n - 1)
    t0 = time.time()
    eigvals = eigsh(G, k=k, sigma=0, which='LM', return_eigenvectors=False,
                    tol=1e-12, maxiter=50000)
    eigvals = np.sort(np.abs(eigvals))
    print(f"  eigsh took {time.time()-t0:.1f}s")
    print(f"  Smallest {k} |eigvals|: {eigvals}")

    # Largest eigenvalue for tolerance
    eigvals_large = eigsh(G, k=1, which='LM', return_eigenvectors=False,
                          tol=1e-6, maxiter=10000)
    largest = float(np.abs(eigvals_large[0]))
    tol = max(n, 1) * largest * 1e-9
    n_zero = int(np.sum(eigvals < tol))
    print(f"  Largest |eigval|: {largest:.3e}")
    print(f"  Tolerance: {tol:.3e}")
    print(f"  # zero eigenvalues (candidate diagonal cocycles): {n_zero}")

    # If all k smallest are zero, bump up k.
    if n_zero == k:
        print(f"  All {k} smallest are zero — bumping up k...")
        k2 = min(k * 3, n - 1)
        eigvals2 = eigsh(G, k=k2, sigma=0, which='LM', return_eigenvectors=False,
                         tol=1e-12, maxiter=50000)
        eigvals2 = np.sort(np.abs(eigvals2))
        n_zero = int(np.sum(eigvals2 < tol))
        print(f"  Smallest {k2} |eigvals|: {eigvals2}")
        print(f"  # zero eigenvalues: {n_zero}")

    print(f"\n=== RESULT: dim diagonal kernel for sl_3 = {n_zero} ===")
    print(f"Expected (full kernel dim): 2")
    print(f"MATCH: {n_zero == 2}")


if __name__ == "__main__":
    main()
