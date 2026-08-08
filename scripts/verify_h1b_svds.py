#!/usr/bin/env python3
"""
Verification script: use sparse SVD to find smallest singular values of ∂_b
for each weight shift, computing the rank efficiently.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds, eigsh

import compute_h1b_bplus_sl3 as sl3


def main(shifts=None, k_smallest=10, output_file=None):
    if output_file is None:
        output_file = sys.stdout

    def log(msg):
        print(msg, file=output_file)
        output_file.flush()

    log('Building tables...')
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
    log('Inverse tables built.')

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

    if shifts is None:
        shifts = [(s1, s2) for s1 in range(sl3.ELL) for s2 in range(sl3.ELL)]

    total_nullity = 0
    per_shift = []
    for s in shifts:
        log(f'\n=== Shift {s} ===')
        t0 = time.time()
        A, cols_list, n_cols = build_A(s)
        if A is None:
            log(f'  No columns; nullity = 0.')
            per_shift.append((s, 0, 0, 0))
            continue
        log(f'  A built: {time.time()-t0:.1f}s, shape={A.shape}, nnz={A.nnz}')

        # Use sparse SVD to find smallest singular values
        # svds finds k extremal singular values; which='SM' finds smallest
        # But 'SM' may have convergence issues. Try with shift-invert via eigsh on A^* A.
        t0 = time.time()
        # Build G = A^* A as sparse
        G_sparse = (A.conj().T @ A).tocsr()
        log(f'  G_sparse built: {time.time()-t0:.1f}s, nnz={G_sparse.nnz}')

        # Use eigsh with sigma=0 to find eigenvalues near 0 (shift-invert mode)
        t0 = time.time()
        try:
            # Find k_smallest smallest eigenvalues
            # eigsh with sigma=0 uses shift-invert, which is good for finding eigenvalues near 0
            eigvals_small = eigsh(G_sparse, k=k_smallest, sigma=0, which='LM',
                                  return_eigenvectors=False, tol=1e-12, maxiter=10000)
            eigvals_small = np.sort(np.abs(eigvals_small))
            log(f'  eigsh (sigma=0) found {len(eigvals_small)} smallest eigvals: {eigvals_small}')
        except Exception as e:
            log(f'  eigsh failed: {e}')
            # Fallback: find largest eigenvalues to estimate largest
            try:
                eigvals_large = eigsh(G_sparse, k=1, which='LM', return_eigenvectors=False, tol=1e-6)
                largest = float(eigvals_large[0])
            except Exception as e2:
                log(f'  eigsh LM also failed: {e2}')
                largest = 1.0
            eigvals_small = np.array([largest] * k_smallest)  # fallback: assume full rank

        # Find largest eigenvalue for tolerance
        t0b = time.time()
        try:
            eigvals_large = eigsh(G_sparse, k=1, which='LM', return_eigenvectors=False, tol=1e-6)
            largest = float(eigvals_large[0])
            log(f'  Largest eigenvalue: {largest:.6e} (took {time.time()-t0b:.1f}s)')
        except Exception as e:
            log(f'  eigsh LM failed: {e}')
            largest = float(np.max(eigvals_small)) if len(eigvals_small) > 0 else 1.0

        # Count zero eigenvalues among the smallest
        tol = n_cols * largest * 1e-10
        n_zero = int(np.sum(eigvals_small < tol))
        log(f'  tol={tol:.3e}, n_zero among smallest {k_smallest}: {n_zero}')

        # If all k_smallest are zero, the nullity is at least k_smallest.
        # Otherwise, n_zero is the nullity.
        nullity = n_zero
        # But if n_zero == k_smallest, we might have missed more; report this.
        if n_zero == k_smallest:
            log(f'  WARNING: All k_smallest eigvals are below tol. Need larger k_smallest to determine true nullity.')

        log(f'  nullity = {nullity}')
        log(f'  Total time: {time.time()-t0:.1f}s')

        total_nullity += nullity
        per_shift.append((s, n_cols, n_cols - nullity, nullity))

        # Free memory
        del A, G_sparse

    log(f'\n\n=== TOTAL nullity (dim H̃¹_b): {total_nullity} ===')
    log(f'Conjecture predicts: 3')
    return total_nullity, per_shift


if __name__ == "__main__":
    # Allow k_smallest override
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    main(k_smallest=k)
