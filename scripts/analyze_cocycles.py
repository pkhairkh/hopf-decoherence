"""
Analyze the 3 cocycles of HH^2(u_q(sl_2), C) at ell=3, weight 0.
"""
import os, sys, cmath, math
import numpy as np
# Make sibling modules importable when run from anywhere
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

ELL = 3
Q = cmath.exp(2j * math.pi / ELL)
Q_INV = Q ** (-1)
D = Q - Q_INV
DIM = ELL ** 3

def idx(a, b, c):
    return a * ELL * ELL + b * ELL + c

def from_idx(i):
    return i // (ELL*ELL), (i // ELL) % ELL, i % ELL

def weight(i):
    a, b, c = from_idx(i)
    return (2*b - 2*c) % ELL


def build_d1_d2_weight0(mult):
    eps = np.zeros(DIM, dtype=complex)
    for a in range(ELL):
        eps[idx(a,0,0)] = 1.0

    wts = np.array([weight(i) for i in range(DIM)])
    wt0 = list(np.where(wts == 0)[0])
    n1 = len(wt0)

    pairs2 = [(i,j) for i in range(DIM) for j in range(DIM) if (wts[i]+wts[j])%ELL == 0]
    n2 = len(pairs2)
    p2idx = {p: k for k, p in enumerate(pairs2)}

    triples3 = [(i,j,k) for i in range(DIM) for j in range(DIM) for k in range(DIM) if (wts[i]+wts[j]+wts[k])%ELL == 0]
    n3 = len(triples3)

    print(f"  Weight-0 dims: C^1 = {n1}, C^2 = {n2}, C^3 = {n3}")

    d1 = np.zeros((n2, n1), dtype=complex)
    for col, i_f in enumerate(wt0):
        for row, (a, b) in enumerate(pairs2):
            mult_if = mult[i_f, a, b]
            d1[row, col] = (eps[a]*(1.0 if b == i_f else 0.0)
                            - mult_if
                            + (1.0 if a == i_f else 0.0)*eps[b])

    from scipy import sparse
    rows, cols, vals = [], [], []
    for r, (a,b,c) in enumerate(triples3):
        ea = eps[a]; ec = eps[c]
        if abs(ea) > 1e-14:
            rows.append(r); cols.append(p2idx[(b,c)]); vals.append(ea)
        for k in range(DIM):
            v = mult[k, a, b]
            if abs(v) > 1e-14:
                if (k, c) in p2idx:
                    rows.append(r); cols.append(p2idx[(k,c)]); vals.append(-v)
        for k in range(DIM):
            v = mult[k, b, c]
            if abs(v) > 1e-14:
                if (a, k) in p2idx:
                    rows.append(r); cols.append(p2idx[(a,k)]); vals.append(v)
        if abs(ec) > 1e-14:
            rows.append(r); cols.append(p2idx[(a,b)]); vals.append(-ec)
    d2 = sparse.csr_matrix((vals, (rows, cols)), shape=(n3, n2), dtype=complex)

    return d1, d2, pairs2, wt0


def classify_pair(i, j):
    a,b,c = from_idx(i)
    ap,bp,cp = from_idx(j)
    def cat(x, y):
        if x == 0 and y == 0: return "K"
        if y == 0: return "E"
        if x == 0: return "F"
        return "EF"
    return cat(b,c) + "-" + cat(bp,cp)


def extract_cocycles(d2, d1, n2, n1):
    """Extract HH^2 = ker(d^2) / im(d^1) representatives.

    HH^2 = ker(d^2) ∩ im(d^1)^⊥ = ker( [d^2 ; im(d^1)^T] ).
    """
    d2_dense = d2.toarray()
    U, s, Vh = np.linalg.svd(d2_dense, full_matrices=True)
    rank_d2 = int(np.sum(s > 1e-9 * s[0]))
    print(f"  rank(d^2) = {rank_d2}, dim ker(d^2) = {n2 - rank_d2}")

    U1, s1, _ = np.linalg.svd(d1, full_matrices=True)
    rank_d1 = int(np.sum(s1 > 1e-9 * s1[0]))
    print(f"  rank(d^1) = {rank_d1}, dim im(d^1) = {rank_d1}")
    im_d1_basis = U1[:, :rank_d1]  # shape (n2, rank_d1), orthonormal columns

    # Build stacked matrix A = [d^2 ; im_d1^T] of shape (n3 + rank_d1, n2)
    A = np.vstack([d2_dense, im_d1_basis.conj().T])
    UA, sA, VhA = np.linalg.svd(A, full_matrices=True)
    rank_A = int(np.sum(sA > 1e-9 * sA[0]))
    dim_hh2 = n2 - rank_A
    print(f"  rank([d^2 ; im(d^1)^T]) = {rank_A}")
    print(f"  dim HH^2 = {dim_hh2}")
    # HH^2 representatives = last (n2 - rank_A) rows of VhA (conjugated)
    cocycles = VhA[rank_A:].conj()  # shape (dim_hh2, n2)
    # Sanity: each should satisfy d^2 xi = 0 and be orthogonal to im(d^1)
    if dim_hh2 > 0:
        for k, xi in enumerate(cocycles):
            d2_norm = np.linalg.norm(d2_dense @ xi)
            im_norm = np.linalg.norm(im_d1_basis.conj().T @ xi)
            xi_norm = np.linalg.norm(xi)
            if k < 5:
                print(f"    xi_{k+1}: ||d^2 xi||/||xi|| = {d2_norm/xi_norm:.2e}, "
                      f"||proj_im(d^1) xi||/||xi|| = {im_norm/xi_norm:.2e}")
    return cocycles, im_d1_basis


def analyze_cocycle_support(cocycle, pairs2):
    cats = ['K-K','K-E','K-F','K-EF','E-K','E-E','E-F','E-EF',
            'F-K','F-E','F-F','F-EF','EF-K','EF-E','EF-F','EF-EF']
    masses = {c: 0.0 for c in cats}
    for k, (i, j) in enumerate(pairs2):
        v = abs(cocycle[k])
        c = classify_pair(i, j)
        masses[c] += v
    total = sum(masses.values())
    return masses, total


def main():
    print(f"=== HH^2(u_q(sl_2), C) at ell={ELL}, weight 0 ===")
    print(f"  q = {Q:.4f}, dim = {DIM}")
    print()
    print("Building multiplication table...")
    from verify_sl2_hh2 import build_multiplication_table
    mult = build_multiplication_table()
    print("Building d^1, d^2 for weight 0...")
    d1, d2, pairs2, wt0 = build_d1_d2_weight0(mult)
    n1, n2 = len(wt0), len(pairs2)

    print("\nExtracting 3 cocycles...")
    cocycles, im_d1_basis = extract_cocycles(d2, d1, n2, n1)

    print("\n=== Cocycle support analysis ===")
    print("(L1-mass fraction on each pair-category)")
    header_cats = ['K-K','K-E','K-F','E-E','E-F','F-F']
    print(f"{'cocycle':>10}  " + "  ".join(f"{c:>6}" for c in header_cats) + "    " + "  ".join(f"{c:>6}" for c in ['E-K','F-K','rest']))
    for k, xi in enumerate(cocycles):
        m, total = analyze_cocycle_support(xi, pairs2)
        rest = total - sum(m[c] for c in m if c in ['K-K','K-E','K-F','E-E','E-F','F-F','E-K','F-K'])
        fracs = {c: m[c]/total if total > 0 else 0 for c in m}
        print(f"  xi_{k+1}:    "
              + "  ".join(f"{fracs[c]:.3f}" for c in header_cats)
              + "    "
              + f"{fracs['E-K']:.3f}  {fracs['F-K']:.3f}  {rest/total if total > 0 else 0:.3f}")

    print()
    print("=== Test 1: Is any cocycle supported on K-K pairs (Cartan-type)? ===")
    for k, xi in enumerate(cocycles):
        m, total = analyze_cocycle_support(xi, pairs2)
        kk_frac = m['K-K'] / total if total > 0 else 0
        print(f"  xi_{k+1}: K-K fraction = {kk_frac:.4f}  ->  {'CARTAN-TYPE' if kk_frac > 0.3 else 'no'}")

    print()
    print("=== Test 2: Direct K-K candidate (uniform on K-K pairs) ===")
    xi_cartan = np.zeros(n2, dtype=complex)
    for k, (i, j) in enumerate(pairs2):
        a,b,c = from_idx(i)
        ap,bp,cp = from_idx(j)
        if b == 0 and c == 0 and bp == 0 and cp == 0:
            xi_cartan[k] = 1.0
    d2_xi = d2 @ xi_cartan
    norm_d2_xi = np.linalg.norm(d2_xi)
    norm_xi = np.linalg.norm(xi_cartan)
    print(f"  Candidate: uniform on K-K pairs (||xi|| = {norm_xi:.4f})")
    print(f"  ||d^2 xi||   = {norm_d2_xi:.4e}  (relative: {norm_d2_xi/(norm_xi+1e-15):.2e})")
    print(f"  Is cocycle?  {'YES' if norm_d2_xi < 1e-9 * norm_xi else 'NO'}")
    if norm_d2_xi < 1e-9 * norm_xi:
        # Project away im(d^1)
        xi_proj = xi_cartan - im_d1_basis @ (im_d1_basis.conj().T @ xi_cartan)
        norm_proj = np.linalg.norm(xi_proj)
        # Re-check cocycle
        d2_xi_proj = d2 @ xi_proj
        norm_d2_proj = np.linalg.norm(d2_xi_proj)
        print(f"  After projecting away im(d^1): ||xi|| = {norm_proj:.4f}, ||d^2 xi|| = {norm_d2_proj:.4e}")
        print(f"  Survives as HH^2 class? {'YES' if norm_proj > 0.1 and norm_d2_proj < 1e-9 * norm_proj else 'no (in im(d^1))'}")

    print()
    print("=== Test 3: Search all K-K supported cochains ===")
    n_KK = sum(1 for (i,j) in pairs2 if from_idx(i)[1:]==(0,0) and from_idx(j)[1:]==(0,0))
    print(f"  Number of K-K pairs in C^2: {n_KK}")
    # Need rank_d1 for the im(d^1) computation
    _, s1_check, _ = np.linalg.svd(d1, full_matrices=True)
    rank_d1_local = int(np.sum(s1_check > 1e-9 * s1_check[0]))
    if n_KK > 0:
        kk_indices = [k for k, (i,j) in enumerate(pairs2) if from_idx(i)[1:]==(0,0) and from_idx(j)[1:]==(0,0)]
        kk_cols = np.zeros((d2.shape[0], n_KK), dtype=complex)
        for j, k in enumerate(kk_indices):
            kk_cols[:, j] = d2[:, k].toarray().flatten()
        s_c = np.linalg.svd(kk_cols, compute_uv=False)
        rank_c = int(np.sum(s_c > 1e-9 * (s_c[0] if len(s_c) else 1)))
        dim_ker = n_KK - rank_c
        # How many of these cocycles lie in im(d^1)?
        e_KK = np.zeros((n2, n_KK), dtype=complex)
        for j, k in enumerate(kk_indices):
            e_KK[k, j] = 1.0
        M = np.hstack([d1, -e_KK])
        s_M = np.linalg.svd(M, compute_uv=False)
        rank_M = int(np.sum(s_M > 1e-9 * (s_M[0] if len(s_M) else 1)))
        dim_null_M = M.shape[1] - rank_M
        dim_ker_d1 = n1 - rank_d1_local
        n_in_im = dim_null_M - dim_ker_d1
        n_KK_survive = dim_ker - n_in_im
        print(f"  K-K cocycles in ker(d^2): {dim_ker}")
        print(f"  K-K cocycles that lie in im(d^1): {n_in_im}")
        print(f"  K-K cocycles surviving in HH^2 (Cartan-type classes): {n_KK_survive}")

    print()
    print("=== Test 4: Apply same analysis to other categories ===")
    for cat_target in ['K-K', 'E-E', 'F-F', 'E-F', 'K-E', 'K-F']:
        idx_cat = [k for k, (i,j) in enumerate(pairs2) if classify_pair(i,j) == cat_target]
        n_cat = len(idx_cat)
        if n_cat == 0:
            print(f"  {cat_target}: 0 pairs")
            continue
        cat_cols = np.zeros((d2.shape[0], n_cat), dtype=complex)
        for j, k in enumerate(idx_cat):
            cat_cols[:, j] = d2[:, k].toarray().flatten()
        s_c = np.linalg.svd(cat_cols, compute_uv=False)
        rank_c = int(np.sum(s_c > 1e-9 * (s_c[0] if len(s_c) else 1)))
        dim_ker = n_cat - rank_c
        e_cat = np.zeros((n2, n_cat), dtype=complex)
        for j, k in enumerate(idx_cat):
            e_cat[k, j] = 1.0
        M = np.hstack([d1, -e_cat])
        s_M = np.linalg.svd(M, compute_uv=False)
        rank_M = int(np.sum(s_M > 1e-9 * (s_M[0] if len(s_M) else 1)))
        dim_null_M = M.shape[1] - rank_M
        dim_ker_d1 = n1 - rank_d1_local
        n_in_im = dim_null_M - dim_ker_d1
        n_survive = dim_ker - n_in_im
        print(f"  {cat_target}: {n_cat} pairs, ker(d^2)={dim_ker}, in im(d^1)={n_in_im}, surviving HH^2={n_survive}")

    print()
    print("=== Test 5: Summary — predicted vs observed ===")
    print("  Cotangent-complex / Tor picture prediction for A_1:")
    print("    Direct: 1 (K^ell-1)  + 1 (E^ell)  + 1 (F^ell)  = 3")
    print("    Tor:    C(1,2) = 0")
    print("    Total:  3")
    print(f"  Observed: {cocycles.shape[0]}")
    print()
    print("  If the picture is correct, the K-K test should give 1 surviving cocycle.")
    print("  If the paper's §6.3 observation (no Cartan-type class at A_1) is correct, K-K should give 0.")

if __name__ == "__main__":
    main()
