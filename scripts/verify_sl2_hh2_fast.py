#!/usr/bin/env python3
"""
Optimized bar-complex for HH^2(u_q(sl_2), C) — uses sparse multiplication table.
"""
import cmath, math, time
import numpy as np
from scipy import sparse


def idx(a, b, c, ell): return a * ell * ell + b * ell + c
def from_idx(i, ell): return i // (ell*ell), (i // ell) % ell, i % ell
def weight(i, ell):
    a, b, c = from_idx(i, ell)
    return (2*b - 2*c) % ell


def multiply_by_F_left(elem, ell, Q, D):
    new = {}
    for (a, b, c), v in elem.items():
        coeff_a = v * Q ** (2*a)
        if c + 1 < ell:
            key = (a, b, c+1)
            new[key] = new.get(key, 0) + coeff_a
        for k in range(b):
            key1 = ((a+1) % ell, b-1, c)
            new[key1] = new.get(key1, 0) + (-coeff_a * Q**(-2*k) / D)
            key2 = ((a + ell - 1) % ell, b-1, c)
            new[key2] = new.get(key2, 0) + (coeff_a * Q**(2*k) / D)
    return {k: v for k, v in new.items() if abs(v) > 1e-13}


def multiply_by_E_left(elem, ell, Q):
    new = {}
    for (a, b, c), v in elem.items():
        if b + 1 < ell:
            coeff = v * Q ** (-2*a)
            key = (a, b+1, c)
            new[key] = new.get(key, 0) + coeff
    return new


def multiply_monomials(a, b, c, ap, bp, cp, ell, Q, D):
    factor = Q ** (2*ap*(c-b))
    a1 = (a + ap) % ell
    middle = {(0, bp, 0): 1}
    for _ in range(c):
        middle = multiply_by_F_left(middle, ell, Q, D)
    for _ in range(b):
        middle = multiply_by_E_left(middle, ell, Q)
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


def build_mult_sparse(ell, Q, D, dim):
    """Build sparse multiplication table: mult_sparse[i][j] = [(l, mult[l,i,j]), ...]."""
    mult_sparse = [[[] for _ in range(dim)] for _ in range(dim)]
    for i in range(dim):
        a, b, c = from_idx(i, ell)
        for j in range(dim):
            ap, bp, cp = from_idx(j, ell)
            prod = multiply_monomials(a, b, c, ap, bp, cp, ell, Q, D)
            for (a2, b2, c2), v in prod.items():
                l = idx(a2, b2, c2, ell)
                mult_sparse[i][j].append((l, v))
    return mult_sparse


def compute_hh2(ell):
    Q = cmath.exp(2j * math.pi / ell)
    D = Q - Q**(-1)
    dim = ell ** 3
    print(f"\n=== ℓ = {ell}, dim = {dim} ===")
    t0 = time.time()
    ms = build_mult_sparse(ell, Q, D, dim)
    print(f"  Mult table (sparse): {time.time()-t0:.1f}s")

    epsilon = np.zeros(dim, dtype=complex)
    for a in range(ell):
        epsilon[idx(a, 0, 0, ell)] = 1.0
    wts = np.array([weight(i, ell) for i in range(dim)])
    wt_idx = {c: np.where(wts == c)[0] for c in range(ell)}

    # Build C^2 index maps
    print(f"  Building C^2 maps...")
    t0 = time.time()
    c2_maps = {}
    c2_lists = {}
    for c in range(ell):
        pairs = [i*dim+j for i in range(dim) for j in range(dim) if (wts[i]+wts[j])%ell == c]
        c2_maps[c] = {ij: k for k, ij in enumerate(pairs)}
        c2_lists[c] = pairs
    print(f"  C^2: {time.time()-t0:.1f}s, sizes: {[len(c2_lists[c]) for c in range(ell)]}")

    # d^1
    print(f"  Building d^1...")
    t0 = time.time()
    d1_blocks = {}
    total_rank_d1 = 0
    for c in range(ell):
        cols_c1 = wt_idx[c]
        rows_c2 = c2_lists[c]
        n_cols, n_rows = len(cols_c1), len(rows_c2)
        if n_cols == 0 or n_rows == 0:
            d1_blocks[c] = None; continue
        d1_c = np.zeros((n_rows, n_cols), dtype=complex)
        for col, i_f in enumerate(cols_c1):
            for row, ij in enumerate(rows_c2):
                i, j = ij // dim, ij % dim
                mult_if_ij = sum(v for (l, v) in ms[i][j] if l == i_f)
                d1_c[row, col] = (epsilon[i] * (1.0 if j == i_f else 0.0)
                                  - mult_if_ij
                                  + (1.0 if i == i_f else 0.0) * epsilon[j])
        d1_blocks[c] = d1_c
        s = np.linalg.svd(d1_c, compute_uv=False)
        tol = max(d1_c.shape) * (s[0] if len(s)>0 else 0) * 1e-14
        rank_c = int(np.sum(s > tol)) if len(s) > 0 else 0
        total_rank_d1 += rank_c
    print(f"  d^1: {time.time()-t0:.1f}s, total rank = {total_rank_d1}")

    # d^2
    print(f"  Building d^2 (sparse mult)...")
    t0 = time.time()
    total_rank_d2 = 0
    total_dim_ker_d2 = 0
    hh2_total = 0

    for c in range(ell):
        triples = []
        for i in range(dim):
            for j in range(dim):
                k_wt = (c - wts[i] - wts[j]) % ell
                for k in wt_idx[k_wt]:
                    triples.append((i, j, k))
        n_rows = len(triples)
        n_cols = len(c2_lists[c])
        if n_rows == 0 or n_cols == 0: continue

        cmap = c2_maps[c]
        rows_l, cols_l, vals_l = [], [], []

        for row, (i, j, k) in enumerate(triples):
            ei = epsilon[i]
            if abs(ei) > 1e-13:
                jk = j*dim+k
                if jk in cmap:
                    rows_l.append(row); cols_l.append(cmap[jk]); vals_l.append(ei)
            # -mult[l,i,j] for each (l,v) in ms[i][j]
            for (l, v) in ms[i][j]:
                lk = l*dim+k
                if lk in cmap:
                    rows_l.append(row); cols_l.append(cmap[lk]); vals_l.append(-v)
            # +mult[l,j,k] for each (l,v) in ms[j][k]
            for (l, v) in ms[j][k]:
                il = i*dim+l
                if il in cmap:
                    rows_l.append(row); cols_l.append(cmap[il]); vals_l.append(v)
            ek = epsilon[k]
            if abs(ek) > 1e-13:
                ij = i*dim+j
                if ij in cmap:
                    rows_l.append(row); cols_l.append(cmap[ij]); vals_l.append(-ek)

        d2_c = sparse.csr_matrix((vals_l, (rows_l, cols_l)), shape=(n_rows, n_cols), dtype=complex)
        del rows_l, cols_l, vals_l

        gram = (d2_c.conj().T @ d2_c).toarray()
        eigvals = np.sort(np.abs(np.linalg.eigvalsh(gram)))[::-1]
        if len(eigvals) == 0 or eigvals[0] == 0:
            rank_c = 0
        else:
            tol = max(d2_c.shape) * eigvals[0] * 1e-10
            rank_c = int(np.sum(eigvals > tol))
        dim_ker_c = n_cols - rank_c

        s1 = np.linalg.svd(d1_blocks[c], compute_uv=False) if d1_blocks[c] is not None else np.array([])
        tol1 = max(d1_blocks[c].shape)*(s1[0] if len(s1)>0 else 0)*1e-14 if d1_blocks[c] is not None else 0
        rank1_c = int(np.sum(s1 > tol1)) if len(s1) > 0 else 0
        hh2_c = dim_ker_c - rank1_c

        total_rank_d2 += rank_c
        total_dim_ker_d2 += dim_ker_c
        hh2_total += hh2_c
        print(f"    wt {c}: C²={n_cols} C³={n_rows} rank(d²)={rank_c} ker(d²)={dim_ker_c} rank(d¹)={rank1_c} HH²={hh2_c}  [{time.time()-t0:.1f}s]")

    print(f"  TOTAL: dim HH² = {hh2_total}")
    print(f"  Predicted = 3, MATCH = {hh2_total == 3}")
    return hh2_total


if __name__ == "__main__":
    for ell in [3, 5]:
        compute_hh2(ell)
