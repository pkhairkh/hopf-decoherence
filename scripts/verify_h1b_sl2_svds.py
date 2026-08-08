#!/usr/bin/env python3
"""
Cross-check: apply the same eigsh-based rank computation to sl_2.
Expected: dim H̃¹_b(B⁺(u_q(sl_2))) = 1 (conjecture at A_1).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

import compute_h1b_bplus_sl2 as sl2


def main():
    print('Building sl_2 tables...')
    ms = sl2.build_mult(sl2.DIM)
    Delta = sl2.build_delta(sl2.DIM)
    epsilon = sl2.build_epsilon(sl2.DIM)
    mult_bar = sl2.build_mult_bar(ms, epsilon, sl2.DIM)
    delta_bar = sl2.build_delta_bar(Delta, sl2.DIM)

    B_bar = list(range(1, sl2.DIM))
    weights = np.array([sl2.weight(i) for i in range(sl2.DIM)])
    n_weights = sl2.ELL

    inv_mult_bar = {l: [] for l in B_bar}
    ms_by_b = {b: [] for b in B_bar}
    ms_by_a = {a: [] for a in B_bar}
    for (l, a, b), v in mult_bar.items():
        inv_mult_bar[l].append((a, b, v))
        ms_by_b[b].append((a, l, v))
        ms_by_a[a].append((b, l, v))

    inv_delta_left = {}
    inv_delta_right = {}
    delta_by_c = {c: [] for c in B_bar}
    for (c, j, k), v in delta_bar.items():
        inv_delta_left.setdefault((c, k), []).append((j, v))
        inv_delta_right.setdefault((c, j), []).append((k, v))
        delta_by_c[c].append((j, k, v))

    def build_shift_matrix(s):
        cols_list = [(j, k) for j in B_bar for k in B_bar
                     if (weights[j] - weights[k]) % n_weights == s]
        n_cols = len(cols_list)
        h_rows_list = [(a, b, t) for a in B_bar for b in B_bar for t in B_bar
                       if (weights[t] - weights[a] - weights[b]) % n_weights == s]
        n_rows_h = len(h_rows_list)
        h_row_map = {triple: i for i, triple in enumerate(h_rows_list)}
        c_rows_list = [(c, al, be) for c in B_bar for al in B_bar for be in B_bar
                       if (weights[al] + weights[be] - weights[c]) % n_weights == s]
        n_rows_c = len(c_rows_list)
        c_row_map = {triple: i for i, triple in enumerate(c_rows_list)}
        return cols_list, h_row_map, c_row_map, n_cols, n_rows_h, n_rows_c

    def build_A(s):
        cols_list, h_row_map, c_row_map, n_cols, n_rows_h, n_rows_c = build_shift_matrix(s)
        if n_cols == 0:
            return None, 0
        rows_l, cols_l, vals_l = [], [], []
        for col_idx, (j, k) in enumerate(cols_list):
            for (a, t, v) in ms_by_b[j]:
                row = h_row_map.get((a, k, t))
                if row is not None:
                    rows_l.append(row); cols_l.append(col_idx); vals_l.append(v)
            for (a, b, v) in inv_mult_bar[k]:
                row = h_row_map.get((a, b, j))
                if row is not None:
                    rows_l.append(row); cols_l.append(col_idx); vals_l.append(-v)
            for (b, t, v) in ms_by_a[j]:
                row = h_row_map.get((k, b, t))
                if row is not None:
                    rows_l.append(row); cols_l.append(col_idx); vals_l.append(v)
            for c in B_bar:
                for (al, v) in inv_delta_left.get((c, k), []):
                    row = c_row_map.get((c, al, j))
                    if row is not None:
                        rows_l.append(n_rows_h + row); cols_l.append(col_idx); vals_l.append(-v)
            for (al, be, v) in delta_by_c[j]:
                row = c_row_map.get((k, al, be))
                if row is not None:
                    rows_l.append(n_rows_h + row); cols_l.append(col_idx); vals_l.append(v)
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
        return A, n_cols

    total_nullity = 0
    for s in range(n_weights):
        print(f'\n=== Shift {s} ===')
        A, n_cols = build_A(s)
        if A is None:
            print(f'  No columns; nullity = 0.')
            continue
        print(f'  A shape: {A.shape}, nnz={A.nnz}')

        G_sparse = (A.conj().T @ A).tocsr()
        k = min(8, n_cols - 1) if n_cols > 1 else 1
        eigvals_small = eigsh(G_sparse, k=k, sigma=0, which='LM',
                              return_eigenvectors=False, tol=1e-12, maxiter=10000)
        eigvals_small = np.sort(np.abs(eigvals_small))
        eigvals_large = eigsh(G_sparse, k=1, which='LM', return_eigenvectors=False, tol=1e-6)
        largest = float(eigvals_large[0])
        tol = n_cols * largest * 1e-10
        n_zero = int(np.sum(eigvals_small < tol))
        nullity = n_zero
        print(f'  Smallest {k} eigvals: {eigvals_small}')
        print(f'  Largest eigval: {largest:.6e}, tol={tol:.3e}')
        print(f'  nullity = {nullity}')
        total_nullity += nullity

    print(f'\n=== TOTAL nullity (dim H̃¹_b(sl_2)): {total_nullity} ===')
    print(f'Conjecture predicts: 1')
    return total_nullity


if __name__ == "__main__":
    main()
