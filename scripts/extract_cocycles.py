#!/usr/bin/env python3
"""
Extract explicit 2-cocycles from HH^2(u_q(sl_2), C) at ell=3, weight 0.
This block has dim HH^2 = 3, and we identify each cocycle.
"""
import sys
sys.path.insert(0, '/home/z/my-project/scripts')
import cmath, math
import numpy as np
from scipy import sparse
from verify_sl2_hh2_fast import (
    idx, from_idx, weight, build_mult_sparse
)


def main():
    ell = 3
    Q = cmath.exp(2j * math.pi / ell)
    D = Q - Q**(-1)
    dim = ell ** 3

    print("=== Extracting 2-cocycles from HH^2(u_q(sl_2), C) at ell=3, weight 0 ===\n")

    ms = build_mult_sparse(ell, Q, D, dim)
    mult = np.zeros((dim, dim, dim), dtype=complex)
    for i in range(dim):
        for j in range(dim):
            for (l, v) in ms[i][j]:
                mult[l, i, j] = v

    epsilon = np.zeros(dim, dtype=complex)
    for a in range(ell):
        epsilon[idx(a, 0, 0, ell)] = 1.0
    wts = np.array([weight(i, ell) for i in range(dim)])

    # Weight 0 only
    c = 0
    wt0_idx = list(np.where(wts == 0)[0])  # C^1_0 basis: 9 elements
    c2_pairs = [i*dim+j for i in range(dim) for j in range(dim) if (wts[i]+wts[j])%ell == 0]
    c2_map = {ij: k for k, ij in enumerate(c2_pairs)}

    n_c1 = len(wt0_idx)   # 9
    n_c2 = len(c2_pairs)  # 243

    # Build d^1 for weight 0
    d1 = np.zeros((n_c2, n_c1), dtype=complex)
    for col, i_f in enumerate(wt0_idx):
        for row, ij in enumerate(c2_pairs):
            i, j = ij // dim, ij % dim
            mult_if = sum(v for (l, v) in ms[i][j] if l == i_f)
            d1[row, col] = (epsilon[i]*(1.0 if j==i_f else 0.0) - mult_if
                           + (1.0 if i==i_f else 0.0)*epsilon[j])

    # Build d^2 for weight 0
    triples = []
    for i in range(dim):
        for j in range(dim):
            k_wt = (c - wts[i] - wts[j]) % ell
            for k in range(dim):
                if wts[k] == k_wt:
                    triples.append((i, j, k))
    n_c3 = len(triples)  # 6561

    rows_l, cols_l, vals_l = [], [], []
    for row, (i, j, k) in enumerate(triples):
        ei = epsilon[i]
        if abs(ei) > 1e-13:
            jk = j*dim+k
            if jk in c2_map:
                rows_l.append(row); cols_l.append(c2_map[jk]); vals_l.append(ei)
        for (l, v) in ms[i][j]:
            lk = l*dim+k
            if lk in c2_map:
                rows_l.append(row); cols_l.append(c2_map[lk]); vals_l.append(-v)
        for (l, v) in ms[j][k]:
            il = i*dim+l
            if il in c2_map:
                rows_l.append(row); cols_l.append(c2_map[il]); vals_l.append(v)
        ek = epsilon[k]
        if abs(ek) > 1e-13:
            ij = i*dim+j
            if ij in c2_map:
                rows_l.append(row); cols_l.append(c2_map[ij]); vals_l.append(-ek)

    d2 = sparse.csr_matrix((vals_l, (rows_l, cols_l)), shape=(n_c3, n_c2), dtype=complex)
    print(f"d^1: {d1.shape}, d^2: {d2.shape} (nnz={d2.nnz})")

    # Verify d^2 o d^1 = 0
    prod = d2 @ d1
    print(f"||d^2 o d^1|| = {np.max(np.abs(prod.toarray() if hasattr(prod,'toarray') else prod)):.2e}")

    # ker(d^2)
    gram = (d2.conj().T @ d2).toarray()
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(np.abs(eigvals))
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    tol = max(gram.shape) * max(abs(eigvals[-1]), 1) * 1e-10
    null_mask = np.abs(eigvals) < tol
    ker_basis = eigvecs[:, null_mask]
    print(f"dim ker(d^2) = {ker_basis.shape[1]}")
    print(f"eigenvalue gap: {eigvals[ker_basis.shape[1]-1]:.2e} vs {eigvals[ker_basis.shape[1]]:.2e}")

    # im(d^1)
    U1, s1, _ = np.linalg.svd(d1, full_matrices=False)
    tol1 = max(d1.shape) * s1[0] * 1e-14
    rank1 = int(np.sum(s1 > tol1))
    im_basis = U1[:, :rank1]
    print(f"rank(d^1) = {rank1}")

    # HH^2 = ker(d^2) / im(d^1)
    # Project ker_basis onto complement of im_basis
    Q_proj = np.eye(n_c2) - im_basis @ im_basis.conj().T
    ker_proj = Q_proj @ ker_basis
    # Use SVD to find the rank-3 component (dim HH^2 = 3)
    U_hh2, s_hh2, _ = np.linalg.svd(ker_proj, full_matrices=False)
    tol_hh2 = max(ker_proj.shape) * (s_hh2[0] if len(s_hh2) > 0 else 0) * 1e-10
    rank_hh2 = int(np.sum(s_hh2 > tol_hh2))
    Q_hh2 = U_hh2[:, :rank_hh2]
    print(f"dim HH^2 = {Q_hh2.shape[1]} (singular values: {s_hh2[:rank_hh2+2]})")

    # Verify each HH^2 representative is in ker(d^2) and not in im(d^1)
    for n in range(Q_hh2.shape[1]):
        v = Q_hh2[:, n]
        d2v = d2 @ v
        print(f"  cocycle {n+1}: ||d^2(v)|| = {np.linalg.norm(d2v):.2e}, "
              f"||v - proj_im(v)|| = {np.linalg.norm(v - im_basis @ (im_basis.conj().T @ v)):.4f}")

    # Classify each cocycle
    print("\n=== Cocycle classification ===\n")
    for n in range(Q_hh2.shape[1]):
        cocycle = np.zeros((dim, dim), dtype=complex)
        for ij, col in c2_map.items():
            i, j = ij // dim, ij % dim
            cocycle[i, j] = Q_hh2[col, n]

        # Compute scores
        e2 = idx(0, 2, 0, ell); e1 = idx(0, 1, 0, ell)
        f2 = idx(0, 0, 2, ell); f1 = idx(0, 0, 1, ell)
        k0 = idx(0, 0, 0, ell); k1 = idx(1, 0, 0, ell); k2 = idx(2, 0, 0, ell)

        e_score = abs(cocycle[e2, e1]) + abs(cocycle[e1, e2])
        f_score = abs(cocycle[f2, f1]) + abs(cocycle[f1, f2])
        k_score = abs(cocycle[k0, k0]) + abs(cocycle[k1, k0]) + abs(cocycle[k0, k1]) + \
                  abs(cocycle[k1, k1]) + abs(cocycle[k2, k2]) + abs(cocycle[k1, k2]) + abs(cocycle[k2, k1])

        max_val = np.max(np.abs(cocycle))
        # Top entries
        flat = np.argsort(np.abs(cocycle).ravel())[::-1][:5]
        print(f"Cocycle {n+1} (max = {max_val:.4f}):")
        for fi in flat:
            i, j = fi // dim, fi % dim
            a1,b1,c1 = from_idx(i, ell); a2,b2,c2 = from_idx(j, ell)
            print(f"  ({i:2d},{j:2d}) = K^{a1}E^{b1}F^{c1} × K^{a2}E^{b2}F^{c2}: {cocycle[i,j]:.4f}")

        if e_score > 0.05*max_val and e_score > f_score and e_score > k_score:
            label = "[E^3] class"
        elif f_score > 0.05*max_val and f_score > e_score and f_score > k_score:
            label = "[F^3] class"
        elif k_score > 0.05*max_val:
            label = "Cartan-type (xi_K^cross)"
        else:
            label = "unknown"
        print(f"  → {label} (E={e_score:.3f}, F={f_score:.3f}, K={k_score:.3f})")
        print()

    # Write the Cartan-type cocycle explicitly
    print("=== Explicit Cartan-type cocycle (from cocycle with highest K-score) ===\n")
    best_n = max(range(Q_hh2.shape[1]),
                 key=lambda n: abs(Q_hh2[c2_map.get(idx(0,0,0,ell)*dim+idx(0,0,0,ell), 0)
                                   if idx(0,0,0,ell)*dim+idx(0,0,0,ell) in c2_map else 0, n]) +
                              abs(Q_hh2[c2_map.get(idx(1,0,0,ell)*dim+idx(0,0,0,ell), 0)
                                   if idx(1,0,0,ell)*dim+idx(0,0,0,ell) in c2_map else 0, n]))

    cocycle = np.zeros((dim, dim), dtype=complex)
    for ij, col in c2_map.items():
        i, j = ij // dim, ij % dim
        cocycle[i, j] = Q_hh2[col, best_n]
    # Normalize
    max_idx = np.unravel_index(np.argmax(np.abs(cocycle)), cocycle.shape)
    cocycle = cocycle / cocycle[max_idx]

    print(f"Cocycle {best_n+1} (normalized to max=1), top entries:")
    for i in range(dim):
        for j in range(dim):
            if abs(cocycle[i, j]) > 0.05:
                a1,b1,c1 = from_idx(i, ell); a2,b2,c2 = from_idx(j, ell)
                print(f"  xi(K^{a1}E^{b1}F^{c1}, K^{a2}E^{b2}F^{c2}) = {cocycle[i,j]:.4f}")

    # Verify
    xi_flat = np.zeros(n_c2, dtype=complex)
    for ij, col in c2_map.items():
        i, j = ij // dim, ij % dim
        xi_flat[col] = cocycle[i, j]
    print(f"\n  ||d^2(xi)|| = {np.linalg.norm(d2 @ xi_flat):.2e}")


if __name__ == "__main__":
    main()
