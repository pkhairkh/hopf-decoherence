#!/usr/bin/env python3
"""
Extract the 2 cocycles (kernel of ∂_b at shift (0,0) for sl_3) and verify
they actually satisfy ∂_b h = 0.

Also extract the single sl_2 cocycle for comparison.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

import compute_h1b_bplus_sl3 as sl3


def main():
    print('Building sl_3 tables...')
    ms = sl3.build_mult(sl3.DIM)
    sl3.ms_global = ms
    Delta = sl3.build_delta(sl3.DIM)
    epsilon = sl3.build_epsilon(sl3.DIM)
    mult_bar = sl3.build_mult_bar(ms, epsilon, sl3.DIM)
    delta_bar = sl3.build_delta_bar(Delta, sl3.DIM)

    B_bar = list(range(1, sl3.DIM))
    weights = [sl3.weight(i) for i in range(sl3.DIM)]

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
    inv_delta_left_by_k = {k: [] for k in B_bar}
    inv_delta_right_by_k = {k: [] for k in B_bar}
    for (c, j, k), v in delta_bar.items():
        inv_delta_left.setdefault((c, k), []).append((j, v))
        inv_delta_right.setdefault((c, j), []).append((k, v))
        delta_by_c[c].append((j, k, v))
        inv_delta_left_by_k[k].append((c, j, v))
        inv_delta_right_by_k[j].append((c, k, v))

    def to_arr3(lst):
        if not lst:
            return (np.array([], dtype=np.int32), np.array([], dtype=np.int32),
                    np.array([], dtype=complex))
        return (np.array([x[0] for x in lst], dtype=np.int32),
                np.array([x[1] for x in lst], dtype=np.int32),
                np.array([x[2] for x in lst], dtype=complex))

    ms_by_b_arr = {b: to_arr3(ms_by_b[b]) for b in B_bar}
    ms_by_a_arr = {a: to_arr3(ms_by_a[a]) for a in B_bar}
    inv_mult_bar_arr = {k: to_arr3(inv_mult_bar[k]) for k in B_bar}
    inv_delta_left_by_k_arr = {k: to_arr3(inv_delta_left_by_k[k]) for k in B_bar}
    inv_delta_right_by_k_arr = {k: to_arr3(inv_delta_right_by_k[k]) for k in B_bar}
    delta_by_c_arr = {c: to_arr3(delta_by_c[c]) for c in B_bar}

    def build_shift_matrix(s):
        cols_list = [(j, k) for j in B_bar for k in B_bar
                     if ((weights[j][0] - weights[k][0]) % sl3.ELL,
                         (weights[j][1] - weights[k][1]) % sl3.ELL) == s]
        n_cols = len(cols_list)
        h_rows_list = [(a, b, t) for a in B_bar for b in B_bar for t in B_bar
                       if ((weights[t][0] - weights[a][0] - weights[b][0]) % sl3.ELL,
                           (weights[t][1] - weights[a][1] - weights[b][1]) % sl3.ELL) == s]
        n_rows_h = len(h_rows_list)
        h_row_map = np.full((sl3.DIM, sl3.DIM, sl3.DIM), -1, dtype=np.int32)
        for i, (a, b, t) in enumerate(h_rows_list):
            h_row_map[a, b, t] = i
        c_rows_list = [(c, al, be) for c in B_bar for al in B_bar for be in B_bar
                       if ((weights[al][0] + weights[be][0] - weights[c][0]) % sl3.ELL,
                           (weights[al][1] + weights[be][1] - weights[c][1]) % sl3.ELL) == s]
        n_rows_c = len(c_rows_list)
        c_row_map = np.full((sl3.DIM, sl3.DIM, sl3.DIM), -1, dtype=np.int32)
        for i, (c, al, be) in enumerate(c_rows_list):
            c_row_map[c, al, be] = i
        return cols_list, h_row_map, c_row_map, n_cols, n_rows_h, n_rows_c

    def build_A(s):
        cols_list, h_row_map, c_row_map, n_cols, n_rows_h, n_rows_c = build_shift_matrix(s)
        if n_cols == 0:
            return None, None, 0
        rows_chunks = []
        cols_chunks = []
        vals_chunks = []
        for col_idx, (j, k) in enumerate(cols_list):
            a_arr, t_arr, v_arr = ms_by_b_arr[j]
            if a_arr.size > 0:
                row_arr = h_row_map[a_arr, k, t_arr]
                valid = row_arr >= 0
                if np.any(valid):
                    rows_chunks.append(row_arr[valid].astype(np.int64))
                    cols_chunks.append(np.full(int(np.sum(valid)), col_idx, dtype=np.int64))
                    vals_chunks.append(v_arr[valid])
            a_arr, b_arr, v_arr = inv_mult_bar_arr[k]
            if a_arr.size > 0:
                row_arr = h_row_map[a_arr, b_arr, j]
                valid = row_arr >= 0
                if np.any(valid):
                    rows_chunks.append(row_arr[valid].astype(np.int64))
                    cols_chunks.append(np.full(int(np.sum(valid)), col_idx, dtype=np.int64))
                    vals_chunks.append(-v_arr[valid])
            b_arr, t_arr, v_arr = ms_by_a_arr[j]
            if b_arr.size > 0:
                row_arr = h_row_map[k, b_arr, t_arr]
                valid = row_arr >= 0
                if np.any(valid):
                    rows_chunks.append(row_arr[valid].astype(np.int64))
                    cols_chunks.append(np.full(int(np.sum(valid)), col_idx, dtype=np.int64))
                    vals_chunks.append(v_arr[valid])
            c_arr, al_arr, v_arr = inv_delta_left_by_k_arr[k]
            if c_arr.size > 0:
                row_arr = c_row_map[c_arr, al_arr, j]
                valid = row_arr >= 0
                if np.any(valid):
                    rows_chunks.append((n_rows_h + row_arr[valid]).astype(np.int64))
                    cols_chunks.append(np.full(int(np.sum(valid)), col_idx, dtype=np.int64))
                    vals_chunks.append(-v_arr[valid])
            al_arr, be_arr, v_arr = delta_by_c_arr[j]
            if al_arr.size > 0:
                row_arr = c_row_map[k, al_arr, be_arr]
                valid = row_arr >= 0
                if np.any(valid):
                    rows_chunks.append((n_rows_h + row_arr[valid]).astype(np.int64))
                    cols_chunks.append(np.full(int(np.sum(valid)), col_idx, dtype=np.int64))
                    vals_chunks.append(v_arr[valid])
            c_arr, be_arr, v_arr = inv_delta_right_by_k_arr[k]
            if c_arr.size > 0:
                row_arr = c_row_map[c_arr, j, be_arr]
                valid = row_arr >= 0
                if np.any(valid):
                    rows_chunks.append((n_rows_h + row_arr[valid]).astype(np.int64))
                    cols_chunks.append(np.full(int(np.sum(valid)), col_idx, dtype=np.int64))
                    vals_chunks.append(-v_arr[valid])
        rows_arr = np.concatenate(rows_chunks)
        cols_arr = np.concatenate(cols_chunks)
        vals_arr = np.concatenate(vals_chunks)
        n_rows_total = n_rows_h + n_rows_c
        A = sparse.csr_matrix((vals_arr, (rows_arr, cols_arr)),
                              shape=(n_rows_total, n_cols), dtype=complex)
        return A, cols_list, n_cols

    s = (0, 0)
    print(f'\n=== Shift {s} ===')
    A, cols_list, n_cols = build_A(s)
    print(f'  A shape: {A.shape}, nnz={A.nnz}')

    G_sparse = (A.conj().T @ A).tocsr()
    # Get 6 smallest eigenvalues and eigenvectors
    k = 6
    eigvals, eigvecs = eigsh(G_sparse, k=k, sigma=0, which='LM',
                              return_eigenvectors=True, tol=1e-14, maxiter=20000)
    order = np.argsort(np.abs(eigvals))
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    print(f'  Smallest {k} eigvals: {eigvals}')

    # Extract the 2 cocycles (eigenvectors with smallest |eigval|)
    print('\nExtracting cocycles...')
    for i in range(2):
        v = eigvecs[:, i]
        # Normalize
        v = v / np.linalg.norm(v)
        print(f'\nCocycle {i+1} (eigval = {eigvals[i]:.3e}, ||v|| = {np.linalg.norm(v):.4f}):')
        # Verify ∂_b h = 0
        residual = A @ v
        print(f'  ||∂_b h|| / ||h|| = {np.linalg.norm(residual):.3e}')
        # Show significant entries
        abs_v = np.abs(v)
        threshold = 1e-3 * np.max(abs_v)
        significant = np.where(abs_v > threshold)[0]
        print(f'  Significant entries (|v| > {threshold:.3e}): {len(significant)}')
        # Show top 30 significant entries (sorted by magnitude)
        top = sorted(significant, key=lambda i: -abs_v[i])[:30]
        for col_idx in top:
            j, k = cols_list[col_idx]
            a, b, c, e, d = sl3.from_idx(j)
            a2, b2, c2, e2, d2 = sl3.from_idx(k)
            from_basis = f'K1^{a}K2^{b}E1^{c}E12^{e}E2^{d}'
            to_basis = f'K1^{a2}K2^{b2}E1^{c2}E12^{e2}E2^{d2}'
            print(f'    h[{to_basis:>30}] -> coeff of [{from_basis:>30}]: {v[col_idx]:+.4f}')


if __name__ == "__main__":
    main()
