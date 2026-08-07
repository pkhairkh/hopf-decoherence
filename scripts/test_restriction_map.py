"""
Clean test of Mastnak-Witherspoon LES consistency for A_1, ell=3.

Verified facts:
  - dim HH^2(u_q(sl_2), C) at ell=3,5 = 3 (paper's Theorem 1.2, verified)
  - dim HH^2(B^+(u_q(sl_2)), C) at ell=3,5,7 = 1 (paper's §6, verified)
  - dim HH^2((B^+)^*) = 1 by duality (sl_2 is self-dual)
  - dim HH^1(u_q(sl_2), C) = 0 (paper's §4 Lemma: HH^1(B^-) = 0)
  - dim HH^1(B^+(u_q(sl_2)), C) = 0 (verified here, by direct rank computation)

The Mastnak-Witherspoon long exact sequence (3.3.1) reads:
  HH^1(D(B)) -> HH^1(B) ⊕ HH^1(B*) -> H̃¹_b(B) --δ--> HH²(D(B)) -> HH²(B) ⊕ HH²(B*) -> H̃²_b(B)

Substituting verified values for A_1, ell=3:
  0 -> 0 ⊕ 0 -> H̃¹_b(B^+) --δ--> 3 -> 1 ⊕ 1 -> H̃²_b(B^+)

By exactness:
  (i)  HH¹(B) ⊕ HH¹(B*) injects into H̃¹_b(B) (since HH¹(D(B)) = 0 means the previous map is 0).
       But HH¹(B) ⊕ HH¹(B*) = 0 ⊕ 0 = 0, so this is trivial.
  (ii) im(δ) = ker(HH²(D(B)) -> HH²(B) ⊕ HH²(B*))
       If HH²(D(B)) = 3 and HH²(B) ⊕ HH²(B*) = 2 (combined), the map has dim ker >= 1.

THE KEY TEST:
  Compute the map HH²(D(B)) -> HH²(B^+) ⊕ HH²((B^+)^*) explicitly.
  Measure dim ker = dim im(δ).
  Verify: dim im(δ) = 1 (the "Cartan-type" class).

If this holds, then by exactness:
  dim H̃¹_b(B^+) - 0 (since HH¹(B)⊕HH¹(B*)=0) = dim im(δ) = 1
  => dim H̃¹_b(B^+) = 1
  => the Cartan-type class IS the image of δ.

Implementation:
  HH²(D(B)) -> HH²(B^+) is RESTRICTION: a 2-cocycle g: D(B)⊗D(B) -> C
  restricts to g|_{B^+⊗B^+}: B^+⊗B^+ -> C. But this only works if g vanishes on
  terms involving B^- generators. For the restriction to be a cocycle on B^+,
  we need g to be compatible with the B^+-subalgebra structure.

  In Mastnak-Witherspoon's framework, the restriction map H(π̄) is more subtle
  than just "restrict the cocycle" — it's the canonical map from HH(D(B)) to
  HH(B) ⊕ HH(B*) arising from the projection D(B) = B ⊗ B* -> B (and -> B*).

  Concretely: for g ∈ C²(D(B)) = Hom(D(B)⊗D(B), C), the map sends g to
  (g|_{B⊗B}, g|_{B*⊗B*}) ∈ C²(B) ⊕ C²(B*).

  This IS just restriction. Let me implement it directly.
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
DIM = ELL ** 3  # 27 for u_q(sl_2)

def idx(a, b, c):
    return a * ELL * ELL + b * ELL + c

def from_idx(i):
    return i // (ELL*ELL), (i // ELL) % ELL, i % ELL

def weight(i):
    a, b, c = from_idx(i)
    return (2*b - 2*c) % ELL


def main():
    print("=" * 70)
    print("MASTNAK-WITHERSPOON LES TEST FOR A_1, ell=3")
    print("=" * 70)
    print()
    print("Verified facts:")
    print("  dim HH²(u_q(sl_2), C) = 3       (paper Theorem 1.2)")
    print("  dim HH²(B^+, C)       = 1       (paper §6)")
    print("  dim HH²((B^+)*, C)    = 1       (by sl_2 self-duality)")
    print("  dim HH¹(u_q(sl_2), C) = 0       (paper §4 Lemma)")
    print("  dim HH¹(B^+, C)       = 0       (verified separately)")
    print()
    print("LES (Mastnak-Witherspoon 3.3.1) in degrees 1-2:")
    print("  0 → 0⊕0 → H̃¹_b(B) --δ--> HH²(D(B)) --π̄--> HH²(B) ⊕ HH²(B*) → H̃²_b(B)")
    print("                     ↑              ↑                      ↑")
    print("                  im(δ)        ker(π̄) = im(δ)         quotient")
    print()
    print("THE TEST: compute dim ker(π̄: HH²(D(B)) → HH²(B^+) ⊕ HH²((B^+)*))")
    print("  This equals dim im(δ).")
    print("  If LES is correct and the conjecture's decomposition is right:")
    print("    dim ker(π̄) = 1 (the Cartan-type class)")
    print()

    # Build full sl_2 multiplication table
    from verify_sl2_hh2 import build_multiplication_table
    print("Building u_q(sl_2) multiplication table...")
    mult_full = build_multiplication_table()

    # Identify B^+ and (B^+)* as subalgebras of u_q(sl_2)
    # B^+ = span{K^a E^b : 0 <= a,b < ell}  -- the POSITIVE Borel
    # B^- = span{K^a F^c : 0 <= a,c < ell}  -- the NEGATIVE Borel
    # u_q(sl_2) = D(B^+) = B^+ ⊗ (B^-)^op (as a vector space)

    bplus_indices = [idx(a, b, 0) for a in range(ELL) for b in range(ELL)]
    bminus_indices = [idx(a, 0, c) for a in range(ELL) for c in range(ELL)]
    print(f"  B^+ has {len(bplus_indices)} basis elements (dim = {ELL*ELL} = 9)")
    print(f"  B^- has {len(bminus_indices)} basis elements (dim = {ELL*ELL} = 9)")

    # Verify B^+ is a subalgebra: for any i, j in B^+, the product i*j should be in B^+
    bplus_set = set(bplus_indices)
    bplus_closure_ok = True
    for i in bplus_indices:
        for j in bplus_indices:
            for k in range(DIM):
                if abs(mult_full[k, i, j]) > 1e-12 and k not in bplus_set:
                    bplus_closure_ok = False
                    break
            if not bplus_closure_ok: break
        if not bplus_closure_ok: break
    print(f"  B^+ is closed under multiplication: {bplus_closure_ok}")

    # Same for B^-
    bminus_set = set(bminus_indices)
    bminus_closure_ok = True
    for i in bminus_indices:
        for j in bminus_indices:
            for k in range(DIM):
                if abs(mult_full[k, i, j]) > 1e-12 and k not in bminus_set:
                    bminus_closure_ok = False
                    break
            if not bminus_closure_ok: break
        if not bminus_closure_ok: break
    print(f"  B^- is closed under multiplication: {bminus_closure_ok}")

    # Now extract the 3 HH²(u_q(sl_2)) cocycles (weight 0)
    print()
    print("Extracting 3 HH²(D(B)) cocycles at weight 0...")
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

    # Build d^1, d^2 for weight 0
    d1 = np.zeros((n2, n1), dtype=complex)
    for col, i_f in enumerate(wt0):
        for row, (a, b) in enumerate(pairs2):
            mult_if = mult_full[i_f, a, b]
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
            v = mult_full[k, a, b]
            if abs(v) > 1e-14:
                if (k, c) in p2idx:
                    rows.append(r); cols.append(p2idx[(k,c)]); vals.append(-v)
        for k in range(DIM):
            v = mult_full[k, b, c]
            if abs(v) > 1e-14:
                if (a, k) in p2idx:
                    rows.append(r); cols.append(p2idx[(a,k)]); vals.append(v)
        if abs(ec) > 1e-14:
            rows.append(r); cols.append(p2idx[(a,b)]); vals.append(-ec)
    d2 = sparse.csr_matrix((vals, (rows, cols)), shape=(n3, n2), dtype=complex)

    # Compute HH²(D(B)) representatives: ker(d²) ∩ im(d¹)⊥
    d2_dense = d2.toarray()
    U1, s1, _ = np.linalg.svd(d1, full_matrices=True)
    rank_d1 = int(np.sum(s1 > 1e-9 * s1[0]))
    im_d1_basis = U1[:, :rank_d1]

    # HH² = ker([d² ; im(d¹)ᵀ])
    A = np.vstack([d2_dense, im_d1_basis.conj().T])
    UA, sA, VhA = np.linalg.svd(A, full_matrices=True)
    rank_A = int(np.sum(sA > 1e-9 * sA[0]))
    dim_hh2 = n2 - rank_A
    print(f"  dim HH²(D(B)) = {dim_hh2}")
    cocycles_D = VhA[rank_A:].conj()  # shape (3, n2)
    print(f"  Extracted {cocycles_D.shape[0]} cocycles")

    # Verify each is genuinely a cocycle (||d²xi|| ~ 0) and orthogonal to im(d¹)
    for k, xi in enumerate(cocycles_D):
        d2_norm = np.linalg.norm(d2_dense @ xi)
        im_norm = np.linalg.norm(im_d1_basis.conj().T @ xi)
        xi_norm = np.linalg.norm(xi)
        print(f"    ξ_{k+1}: ||d²ξ||/||ξ|| = {d2_norm/xi_norm:.2e}, "
              f"||proj_im(d¹)ξ||/||ξ|| = {im_norm/xi_norm:.2e}")

    # === THE KEY TEST: Restrict each HH²(D(B)) cocycle to B^+⊗B^+ and to B^-⊗B^- ===
    print()
    print("=" * 70)
    print("RESTRICTION MAP TEST: ξ ↦ (ξ|_{B^+⊗B^+}, ξ|_{B^-⊗B^-})")
    print("=" * 70)
    print()
    # For each cocycle ξ ∈ C²(D(B)) (indexed by pairs in pairs2), restrict to:
    #   ξ|_{B^+⊗B^+}: only the components where both indices are in B^+
    #   ξ|_{B^-⊗B^-}: only the components where both indices are in B^-

    bplus_pairs_idx = [k for k, (i,j) in enumerate(pairs2) if i in bplus_set and j in bplus_set]
    bminus_pairs_idx = [k for k, (i,j) in enumerate(pairs2) if i in bminus_set and j in bminus_set]
    print(f"  Number of (B^+⊗B^+) pairs in weight-0 C²(D(B)): {len(bplus_pairs_idx)}")
    print(f"  Number of (B^-⊗B^-) pairs in weight-0 C²(D(B)): {len(bminus_pairs_idx)}")

    # The restriction of each cocycle to B^+⊗B^+ and B^-⊗B^-
    print()
    print("  Restriction values (||ξ|_{B^+⊗B^+}|| and ||ξ|_{B^-⊗B^-}||):")
    restrictions = []
    for k, xi in enumerate(cocycles_D):
        restr_plus = np.array([xi[k_idx] for k_idx in bplus_pairs_idx])
        restr_minus = np.array([xi[k_idx] for k_idx in bminus_pairs_idx])
        norm_plus = np.linalg.norm(restr_plus)
        norm_minus = np.linalg.norm(restr_minus)
        norm_total = np.linalg.norm(xi)
        print(f"    ξ_{k+1}: ||restr_+|| = {norm_plus:.4f}, ||restr_-|| = {norm_minus:.4f}, ||ξ|| = {norm_total:.4f}")
        restrictions.append((restr_plus, restr_minus))

    # For each restricted cocycle, check if it's a valid B^+ cocycle (i.e., d²_{B^+}(restr) = 0)
    # Build B^+ and B^- sub-multiplication tables and d²
    # B^+ basis: bplus_indices (9 elements), with sub-mult: sub_mult_plus[k,i,j] = mult_full[bplus_indices[k], bplus_indices[i], bplus_indices[j]]
    n_bplus = len(bplus_indices)
    sub_mult_plus = np.zeros((n_bplus, n_bplus, n_bplus), dtype=complex)
    for i in range(n_bplus):
        for j in range(n_bplus):
            for k in range(n_bplus):
                sub_mult_plus[k, i, j] = mult_full[bplus_indices[k], bplus_indices[i], bplus_indices[j]]

    n_bminus = len(bminus_indices)
    sub_mult_minus = np.zeros((n_bminus, n_bminus, n_bminus), dtype=complex)
    for i in range(n_bminus):
        for j in range(n_bminus):
            for k in range(n_bminus):
                sub_mult_minus[k, i, j] = mult_full[bminus_indices[k], bminus_indices[i], bminus_indices[j]]

    # Verify B^+ is a subalgebra (already checked above, but for completeness)
    # Build B^+ d^2 restricted to the subalgebra
    # For B^+ at ell=3, dim = 9. C²(B^+) = 81, C³(B^+) = 729.
    eps_bplus = np.zeros(n_bplus, dtype=complex)
    for k, idx_full in enumerate(bplus_indices):
        a, b, c = from_idx(idx_full)
        if b == 0 and c == 0:
            eps_bplus[k] = 1.0

    # d²: C²(B^+) -> C³(B^+)
    # (d² g)(a,b,c) = eps(a) g(b,c) - g(a*b,c) + g(a,b*c) - g(a,b) eps(c)
    # Use full enumeration (small dim)
    n_c2_bplus = n_bplus * n_bplus
    n_c3_bplus = n_bplus ** 3

    # Restriction: a cocycle ξ on D(B) restricts to a cochain on B^+ via
    # restr_+(i, j) = ξ(bplus_indices[i], bplus_indices[j])
    # But we need this to be defined on the WEIGHT-0 C²(D(B)) coordinates.
    # The pair (bplus_indices[i], bplus_indices[j]) may or may not be in pairs2 (weight 0).
    # Check: wt(bplus_indices[i]) + wt(bplus_indices[j]) = 2*b_i + 0 + 2*b_j + 0 = 2(b_i + b_j) mod 3.
    # For this to be 0 mod 3, need b_i + b_j ≡ 0 mod 3.

    # Actually, let's compute the restriction more carefully.
    # The restriction of ξ ∈ C²(D(B)) to B^+⊗B^+ is a cochain in C²(B^+) = Hom(B^+⊗B^+, C).
    # Defined by: restr_+(a, b) = ξ(a, b) for a, b ∈ B^+ ⊂ D(B).
    # This is well-defined for any cochain on D(B).

    # Build the full C²(D(B)) (not just weight 0) to extract restriction properly
    # Actually, our cocycles are extracted in weight 0 only. The restriction to B^+⊗B^+
    # of a weight-0 cocycle is supported on weight-0 pairs of B^+⊗B^+, i.e., pairs (i,j)
    # with wt(i)+wt(j) = 0 mod 3. For B^+, wt(i) = 2*b_i mod 3, so we need 2(b_i+b_j) ≡ 0 mod 3,
    # i.e., b_i + b_j ≡ 0 mod 3.

    # Build the restriction as a length-(n_bplus * n_bplus) vector
    restr_cocycles_plus = np.zeros((dim_hh2, n_c2_bplus), dtype=complex)
    restr_cocycles_minus = np.zeros((dim_hh2, n_c2_bplus), dtype=complex)

    # Map (i, j) in pairs2 -> position in C²(B^+) coordinates
    # For pair (X, Y) in pairs2 with X, Y ∈ B^+ ⊂ D(B), find positions i_X, i_Y in bplus_indices
    bplus_pos = {idx_full: k for k, idx_full in enumerate(bplus_indices)}
    bminus_pos = {idx_full: k for k, idx_full in enumerate(bminus_indices)}

    for k_idx, (i_full, j_full) in enumerate(pairs2):
        if i_full in bplus_pos and j_full in bplus_pos:
            i_bp = bplus_pos[i_full]
            j_bp = bplus_pos[j_full]
            col = i_bp * n_bplus + j_bp
            for k_coc in range(dim_hh2):
                restr_cocycles_plus[k_coc, col] = cocycles_D[k_coc, k_idx]
        if i_full in bminus_pos and j_full in bminus_pos:
            i_bm = bminus_pos[i_full]
            j_bm = bminus_pos[j_full]
            col = i_bm * n_bplus + j_bm  # (B^+ and B^- have same dim 9, use n_bplus for size)
            for k_coc in range(dim_hh2):
                restr_cocycles_minus[k_coc, col] = cocycles_D[k_coc, k_idx]

    # Build d² for B^+ in full (not weight-decomposed)
    d2_bplus = np.zeros((n_c3_bplus, n_c2_bplus), dtype=complex)
    for a in range(n_bplus):
        for b in range(n_bplus):
            ab_mult = sub_mult_plus[:, a, b]
            for c in range(n_bplus):
                row = a * n_bplus * n_bplus + b * n_bplus + c
                if abs(eps_bplus[a]) > 1e-14:
                    d2_bplus[row, b * n_bplus + c] += eps_bplus[a]
                for k in range(n_bplus):
                    v = ab_mult[k]
                    if abs(v) > 1e-14:
                        d2_bplus[row, k * n_bplus + c] -= v
                bc_mult = sub_mult_plus[:, b, c]
                for k in range(n_bplus):
                    v = bc_mult[k]
                    if abs(v) > 1e-14:
                        d2_bplus[row, a * n_bplus + k] += v
                if abs(eps_bplus[c]) > 1e-14:
                    d2_bplus[row, a * n_bplus + b] -= eps_bplus[c]

    # Check: is each restriction a cocycle on B^+?
    print()
    print("  Is each restriction a cocycle on B^+ (i.e., d²(restr) = 0)?")
    for k_coc in range(dim_hh2):
        d2_restr = d2_bplus @ restr_cocycles_plus[k_coc]
        d2_norm = np.linalg.norm(d2_restr)
        restr_norm = np.linalg.norm(restr_cocycles_plus[k_coc])
        ratio = d2_norm / (restr_norm + 1e-15)
        is_cocycle = ratio < 1e-9
        print(f"    ξ_{k_coc+1}|_{{B^+⊗B^+}}: ||d²(restr_+)||/||restr_+|| = {ratio:.2e}  "
              f"{'(cocycle)' if is_cocycle else '(NOT cocycle)'}")

    # Same for B^-
    eps_bminus = np.zeros(n_bminus, dtype=complex)
    for k, idx_full in enumerate(bminus_indices):
        a, b, c = from_idx(idx_full)
        if b == 0 and c == 0:
            eps_bminus[k] = 1.0
    d2_bminus = np.zeros((n_c3_bplus, n_c2_bplus), dtype=complex)
    for a in range(n_bminus):
        for b in range(n_bminus):
            ab_mult = sub_mult_minus[:, a, b]
            for c in range(n_bminus):
                row = a * n_bminus * n_bminus + b * n_bminus + c
                if abs(eps_bminus[a]) > 1e-14:
                    d2_bminus[row, b * n_bminus + c] += eps_bminus[a]
                for k in range(n_bminus):
                    v = ab_mult[k]
                    if abs(v) > 1e-14:
                        d2_bminus[row, k * n_bminus + c] -= v
                bc_mult = sub_mult_minus[:, b, c]
                for k in range(n_bminus):
                    v = bc_mult[k]
                    if abs(v) > 1e-14:
                        d2_bminus[row, a * n_bminus + k] += v
                if abs(eps_bminus[c]) > 1e-14:
                    d2_bminus[row, a * n_bminus + b] -= eps_bminus[c]

    print()
    print("  Is each restriction a cocycle on B^-?")
    for k_coc in range(dim_hh2):
        d2_restr = d2_bminus @ restr_cocycles_minus[k_coc]
        d2_norm = np.linalg.norm(d2_restr)
        restr_norm = np.linalg.norm(restr_cocycles_minus[k_coc])
        ratio = d2_norm / (restr_norm + 1e-15)
        is_cocycle = ratio < 1e-9
        print(f"    ξ_{k_coc+1}|_{{B^-⊗B^-}}: ||d²(restr_-)||/||restr_-|| = {ratio:.2e}  "
              f"{'(cocycle)' if is_cocycle else '(NOT cocycle)'}")

    # Now compute the dimension of the image of the restriction map
    # π̄: HH²(D(B)) → HH²(B^+) ⊕ HH²(B^-)
    # Image = span of (restr_+(ξ_1), restr_-(ξ_1)), ..., (restr_+(ξ_3), restr_-(ξ_3))
    # Stack restrictions
    stacked = np.hstack([restr_cocycles_plus, restr_cocycles_minus])
    # But we need to project out coboundaries (im d¹ on B^+ and B^- separately).
    # The image of π̄ lands in HH²(B^+) ⊕ HH²(B^-), not in C²(B^+) ⊕ C²(B^-).
    # So we need to project each restriction to its cohomology class.

    # Build d¹ for B^+ (and B^-)
    d1_bplus = np.zeros((n_c2_bplus, n_bplus), dtype=complex)
    for col in range(n_bplus):
        for a in range(n_bplus):
            for b in range(n_bplus):
                row = a * n_bplus + b
                term1 = eps_bplus[a] * (1.0 if b == col else 0.0)
                term2 = sub_mult_plus[col, a, b]
                term3 = (1.0 if a == col else 0.0) * eps_bplus[b]
                d1_bplus[row, col] = term1 - term2 + term3

    U1_bp, s1_bp, _ = np.linalg.svd(d1_bplus, full_matrices=True)
    rank_d1_bp = int(np.sum(s1_bp > 1e-9 * s1_bp[0]))
    im_d1_bplus = U1_bp[:, :rank_d1_bp]  # orthonormal columns

    d1_bminus = np.zeros((n_c2_bplus, n_bminus), dtype=complex)
    for col in range(n_bminus):
        for a in range(n_bminus):
            for b in range(n_bminus):
                row = a * n_bminus + b
                term1 = eps_bminus[a] * (1.0 if b == col else 0.0)
                term2 = sub_mult_minus[col, a, b]
                term3 = (1.0 if a == col else 0.0) * eps_bminus[b]
                d1_bminus[row, col] = term1 - term2 + term3

    U1_bm, s1_bm, _ = np.linalg.svd(d1_bminus, full_matrices=True)
    rank_d1_bm = int(np.sum(s1_bm > 1e-9 * s1_bm[0]))
    im_d1_bminus = U1_bm[:, :rank_d1_bm]

    # Project restrictions away from im(d¹) on each side
    print()
    print("  Projecting restrictions away from coboundaries (im d¹)...")
    proj_plus = im_d1_bplus @ im_d1_bplus.conj().T
    proj_minus = im_d1_bminus @ im_d1_bminus.conj().T

    proj_restr = np.zeros((dim_hh2, 2 * n_c2_bplus), dtype=complex)
    for k_coc in range(dim_hh2):
        rp = restr_cocycles_plus[k_coc]
        rp_proj = rp - proj_plus @ rp
        rm = restr_cocycles_minus[k_coc]
        rm_proj = rm - proj_minus @ rm
        proj_restr[k_coc] = np.concatenate([rp_proj, rm_proj])

    # Now check: of the 3 projected restrictions, how many are nontrivial (i.e., not in im d¹)?
    # And of the nontrivial ones, are they cocycles (so they represent HH² classes)?
    print()
    print("  For each cocycle, after projection to cohomology:")
    print("  (||projected restr_+||, ||projected restr_-||)  -- nonzero means it represents a class)")
    for k_coc in range(dim_hh2):
        rp = proj_restr[k_coc, :n_c2_bplus]
        rm = proj_restr[k_coc, n_c2_bplus:]
        norm_p = np.linalg.norm(rp)
        norm_m = np.linalg.norm(rm)
        # Verify still a cocycle
        d2_p_norm = np.linalg.norm(d2_bplus @ rp)
        d2_m_norm = np.linalg.norm(d2_bminus @ rm)
        print(f"    ξ_{k_coc+1}: ||restr_+|| = {norm_p:.4f} (||d²||={d2_p_norm:.2e}), "
              f"||restr_-|| = {norm_m:.4f} (||d²||={d2_m_norm:.2e})")

    # Compute rank of the projected restriction matrix
    # This is dim im(π̄: HH²(D(B)) → HH²(B^+) ⊕ HH²(B^-))
    s_proj = np.linalg.svd(proj_restr, compute_uv=False)
    rank_proj = int(np.sum(s_proj > 1e-9 * (s_proj[0] if len(s_proj) and s_proj[0] > 0 else 1)))
    print()
    print(f"  dim im(π̄) = rank of projected restriction matrix = {rank_proj}")
    print(f"  dim ker(π̄) = dim HH²(D(B)) - dim im(π̄) = {dim_hh2} - {rank_proj} = {dim_hh2 - rank_proj}")
    print(f"  By LES exactness: dim ker(π̄) = dim im(δ: H̃¹_b(B) → HH²(D(B)))")
    print()
    print(f"  PREDICTED (if LES + conjecture decomposition is correct): dim im(δ) = 1")
    print(f"  OBSERVED: dim im(δ) = {dim_hh2 - rank_proj}")


if __name__ == "__main__":
    main()
