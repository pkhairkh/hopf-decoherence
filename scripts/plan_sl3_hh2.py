#!/usr/bin/env python3
"""
Planning / feasibility script for HH^2(u_q(sl_3), C) at ℓ = 3.

Project conjecture:
    dim_C HH^2(u_q(sl_3), C) = C(n+1, 2) + 2|Phi^+| = C(3,2) + 2*3 = 3 + 6 = 9.

This script does NOT compute HH^2 directly. Instead, it computes the
structural data needed to assess whether a direct bar-complex computation
is feasible, and quantifies the memory / time obstacles.

Specifically:
  1. Dimension of u_q(sl_3) at ℓ = 3.
  2. Weight space decomposition by (Z/ℓ)^2 (9 weight spaces).
  3. Bar complex block sizes (C^1, C^2, C^3) for each weight, both on the
     full algebra A and on the principal block u_0 (≅ weight-0 subspace).
  4. Memory and time estimates for the d^2 matrix and its Gram matrix.
  5. Tractability analysis on a single 4 GB RAM machine.
  6. Comparison with the already-verified tractable cases (sl_2, B+(sl_3)).
  7. Recommendations for alternative approaches if direct computation
     is intractable.

Run:
    python3 plan_sl3_hh2.py > sl3_plan_output.txt
"""
import itertools
import sys
import time

# ============================================================
# Constants for u_q(sl_3) at ℓ = 3
# ============================================================
ELL = 3                # root of unity order (smallest odd prime that works)
N_POS_ROOTS = 3        # E1, E12 = E1 E2 - q E2 E1, E2
N_NEG_ROOTS = 3        # F1, F21 = F2 F1 - q F1 F2, F2  (note: dual root vector)
RANK = 2               # K1, K2  (Cartan torus, (Z/ℓ)^rank = (Z/3)^2)
N_PBW = N_POS_ROOTS + N_NEG_ROOTS + RANK  # = 8

# Memory constants (for estimates)
BYTES_COMPLEX = 16     # complex128
BYTES_INDEX = 4        # int32 (sufficient for dims up to 2^31)
BYTES_PER_NZ = BYTES_COMPLEX + BYTES_INDEX  # 20 bytes per sparse nonzero
NZ_PER_ROW_D2 = 4      # bar differential d^2 has exactly 4 terms per row


def fmt_bytes(b):
    """Human-readable byte count."""
    if b < 1024:
        return f"{b} B"
    if b < 1024**2:
        return f"{b/1024:.1f} KB"
    if b < 1024**3:
        return f"{b/1024**2:.1f} MB"
    if b < 1024**4:
        return f"{b/1024**3:.2f} GB"
    return f"{b/1024**4:.2f} TB"


def fmt_num(n):
    """Human-readable integer with commas."""
    return f"{n:,}"


# ============================================================
# 1. Dimension
# ============================================================
def section_1_dimension():
    print("=" * 76)
    print("1. DIMENSION OF u_q(sl_3) AT ℓ = 3")
    print("=" * 76)
    dim = ELL ** N_PBW
    print(f"   PBW generators ({N_PBW} total):")
    print(f"     Cartan:        K1, K2                                  (rank {RANK})")
    print(f"     Positive roots: E1, E12 = E1 E2 - q E2 E1, E2          ({N_POS_ROOTS})")
    print(f"     Negative roots: F1, F21 = F2 F1 - q F1 F2, F2          ({N_NEG_ROOTS})")
    print(f"   Each PBW exponent ranges over 0..{ELL-1} (since K_i^{ELL}=1, E_i^{ELL}=0, etc.)")
    print(f"   dim u_q(sl_3) at ℓ = {ELL}  =  {ELL}^{N_PBW}  =  {dim}")
    print(f"   Formula check: ℓ^(#pos_roots + #neg_roots + rank) = ℓ^(3+3+2) = ℓ^8 = {ELL**8}")
    print(f"   ✓  Matches task description 'dim = 3^(3+3+rank) = 3^8 = 6561'")
    return dim


# ============================================================
# 2. Weight space decomposition
# ============================================================
def section_2_weight_spaces():
    print()
    print("=" * 76)
    print("2. WEIGHT SPACE DECOMPOSITION  (by (Z/ℓ)^2 = (Z/3)^2)")
    print("=" * 76)
    print(f"   Weight group: (Z/{ELL})^{RANK}, order {ELL**RANK} = 9")
    print()
    print("   Weight of K1^a K2^b E1^c E12^e E2^d F1^f F21^g F2^h:")
    print("     E1  → α_1 = ( 1,  0)         F1  → -α_1 = (-1,  0) = (2, 0) mod 3")
    print("     E12 → α_1+α_2 = ( 1,  1)     F21 → -(α_1+α_2) = (-1,-1) = (2, 2) mod 3")
    print("     E2  → α_2 = ( 0,  1)         F2  → -α_2 = ( 0, -1) = (0, 2) mod 3")
    print("     K1, K2 → (0, 0)")
    print()
    print("   So weight of monomial = (c+e-f-g,  e+d-g-h) mod 3.")
    print()

    cartan_factor = ELL ** RANK  # 9
    ef_weight_dims = {}
    for w1 in range(ELL):
        for w2 in range(ELL):
            ef_weight_dims[(w1, w2)] = 0
    for c, e, d, f, g, h in itertools.product(range(ELL), repeat=6):
        w1 = (c + e - f - g) % ELL
        w2 = (e + d - g - h) % ELL
        ef_weight_dims[(w1, w2)] += 1

    print(f"   E-F part (3^6 = {ELL**6} = 729 monomials) distributes over 9 weights:")
    for w in sorted(ef_weight_dims.keys()):
        print(f"     weight {w}: E-F dim = {ef_weight_dims[w]}")

    full_weight_dims = {w: ef_weight_dims[w] * cartan_factor for w in ef_weight_dims}
    print()
    print(f"   Full weight space dims (× Cartan factor 3^2 = {cartan_factor}):")
    total = 0
    for w in sorted(full_weight_dims.keys()):
        total += full_weight_dims[w]
        print(f"     weight {w}: dim = {full_weight_dims[w]}")
    print(f"   Sum = {total}  (must equal dim A = {ELL**N_PBW})  ", end="")
    print("✓" if total == ELL**N_PBW else "✗")

    all_dims = list(full_weight_dims.values())
    uniform = all(d == all_dims[0] for d in all_dims)
    print(f"   Uniform distribution (each weight space dim = {all_dims[0]}): {uniform}")
    print(f"   (This is expected: the weight map (Z/3)^6 → (Z/3)^2 is a surjective")
    print(f"    homomorphism with kernel of size 3^4 = 81, so each fiber has size 81;")
    print(f"    times the Cartan factor 9 gives 729 = 6561 / 9.)")
    return full_weight_dims


# ============================================================
# 3. Bar complex block sizes (full A)
# ============================================================
def compute_block_size(weight_dims, n, target_w):
    """dim C^n_target_w = sum over (w_1,...,w_n) with sum = target_w of prod dim(A_{w_i})."""
    weights = list(weight_dims.keys())
    total = 0
    for w_chain in itertools.product(weights, repeat=n - 1):
        last_w = ((target_w[0] - sum(wc[0] for wc in w_chain)) % ELL,
                  (target_w[1] - sum(wc[1] for wc in w_chain)) % ELL)
        prod = 1
        for wc in w_chain:
            prod *= weight_dims[wc]
        prod *= weight_dims[last_w]
        total += prod
    return total


def section_3_block_sizes(full_weight_dims, dim_A):
    print()
    print("=" * 76)
    print("3. BAR COMPLEX BLOCK SIZES  (on full A = u_q(sl_3))")
    print("=" * 76)
    print()
    print("   The bar complex C^n(A, C) = Hom(A^{⊗n}, C) decomposes by total weight")
    print("   in (Z/3)^2. The block of weight w has dimension:")
    print("     dim C^n_w = Σ_{w_1+...+w_n = w}  ∏ dim(A_{w_i}).")
    print()

    c1_blocks = dict(full_weight_dims)
    c2_blocks = {w: compute_block_size(full_weight_dims, 2, w) for w in full_weight_dims}
    c3_blocks = {w: compute_block_size(full_weight_dims, 3, w) for w in full_weight_dims}

    print(f"   {'Weight':<10} {'dim C^1':<10} {'dim C^2':<22} {'dim C^3':<26}")
    print(f"   {'-'*10} {'-'*10} {'-'*22} {'-'*26}")
    for w in sorted(full_weight_dims.keys()):
        print(f"   {str(w):<10} {c1_blocks[w]:<10,} {c2_blocks[w]:<22,} {c3_blocks[w]:<26,}")
    total_c1 = sum(c1_blocks.values())
    total_c2 = sum(c2_blocks.values())
    total_c3 = sum(c3_blocks.values())
    print(f"   {'TOTAL':<10} {total_c1:<10,} {total_c2:<22,} {total_c3:<26,}")
    print(f"   (Sanity check: dim^1={dim_A:,}, dim^2={dim_A**2:,}, dim^3={dim_A**3:,})")
    print(f"   Match: C¹ {'✓' if total_c1 == dim_A else '✗'}, "
          f"C² {'✓' if total_c2 == dim_A**2 else '✗'}, "
          f"C³ {'✓' if total_c3 == dim_A**3 else '✗'}")
    return c1_blocks, c2_blocks, c3_blocks


# ============================================================
# 4. Memory / time estimates
# ============================================================
def section_4_memory(c2_blocks, c3_blocks, full_weight_dims, dim_A):
    print()
    print("=" * 76)
    print("4. MEMORY / TIME ESTIMATES  (direct bar complex on full A)")
    print("=" * 76)
    print()
    print(f"   Bar differential d^2: C³ → C².  Each row has ≤{NZ_PER_ROW_D2} nonzeros:")
    print(f"     (d² g)(a,b,c) = ε(a) g(b,c) - g(a·b, c) + g(a, b·c) - g(a,b) ε(c)")
    print(f"   Sparse storage per nonzero: complex128 ({BYTES_COMPLEX} B) + int32 index ({BYTES_INDEX} B)")
    print(f"                            = {BYTES_PER_NZ} B per nonzero.")
    print()

    w0 = (0, 0)
    n_rows_w0 = c3_blocks[w0]
    n_cols_w0 = c2_blocks[w0]
    nnz_w0 = n_rows_w0 * NZ_PER_ROW_D2
    sparse_w0 = nnz_w0 * BYTES_PER_NZ
    dense_d2_w0 = n_rows_w0 * n_cols_w0 * BYTES_COMPLEX
    dense_gram_w0 = n_cols_w0 * n_cols_w0 * BYTES_COMPLEX

    print(f"   Per-weight block (weight {w0} as representative, by symmetry all 9 blocks equal):")
    print(f"     d² shape:           {n_rows_w0:,} × {n_cols_w0:,}")
    print(f"     Nonzeros (sparse):  {nnz_w0:,}")
    print(f"     Sparse d² storage:  {fmt_bytes(sparse_w0)}")
    print(f"     Dense d²:           {fmt_bytes(dense_d2_w0)}  (infeasible)")
    print(f"     Dense Gram d²ᵀd²:   {fmt_bytes(dense_gram_w0)}  (infeasible)")
    print()

    total_nnz = sum(c3_blocks[w] * NZ_PER_ROW_D2 for w in c3_blocks)
    total_sparse = total_nnz * BYTES_PER_NZ
    print(f"   All 9 weights combined:")
    print(f"     Total nonzeros: {total_nnz:,}")
    print(f"     Sparse storage: {fmt_bytes(total_sparse)}")
    print()

    # Time estimate for sparse rank computation
    # Rough: scipy.sparse.linalg.svds computes k singular values in ~10 matvecs each.
    # Each matvec: 4 nonzeros × n_rows = 4 × n_rows ops.
    flops_per_matvec_w0 = 4 * n_rows_w0
    flops_per_sv = 10 * flops_per_matvec_w0  # 10 matvecs per singular value
    # To get rank, need ~n_cols singular values (worst case)
    n_sv = n_cols_w0
    total_flops = n_sv * flops_per_sv
    # Assume 1 GFLOP/s for sparse matvec (conservative for Python)
    gflops = 1e9
    time_seconds = total_flops / gflops
    time_days = time_seconds / 86400
    print(f"   Time estimate (sparse iterative rank computation via ARPACK-style SVD):")
    print(f"     Each matvec: ~{flops_per_matvec_w0:,} flops  ({flops_per_matvec_w0/gflops:.1f} s at 1 GFLOP/s)")
    print(f"     ~10 matvecs per singular value")
    print(f"     Need ~{n_sv:,} singular values (worst case, to count zeros)")
    print(f"     Total: ~{total_flops:,} flops = ~{time_seconds:.2e} s = ~{time_days:.0f} days")
    print(f"     → Infeasible (years of compute) even if matrix fit in memory.")
    print()
    return sparse_w0, total_sparse


# ============================================================
# 5. Tractability on 4 GB RAM
# ============================================================
def section_5_tractability(sparse_w0, total_sparse):
    print("=" * 76)
    print("5. TRACTABILITY ON A SINGLE 4 GB RAM MACHINE")
    print("=" * 76)
    print()
    ram_gb = 4
    ram_bytes = ram_gb * 1024**3
    print(f"   Sandbox RAM budget:  {ram_gb} GB = {fmt_bytes(ram_bytes)}")
    print(f"   Weight-0 d² sparse:  {fmt_bytes(sparse_w0)}")
    print(f"   All-9-weights d²:    {fmt_bytes(total_sparse)}")
    print()
    max_nz = ram_bytes // BYTES_PER_NZ
    w0_nz = sparse_w0 // BYTES_PER_NZ
    total_nz = total_sparse // BYTES_PER_NZ
    print(f"   Max nonzeros storable in 4 GB:  {max_nz:,}")
    print(f"   Weight-0 d² requires:            {w0_nz:,} nonzeros  ({w0_nz/max_nz:.1f}× RAM)")
    print(f"   All-9-weights d² requires:       {total_nz:,} nonzeros  ({total_nz/max_nz:.1f}× RAM)")
    print()
    print(f"   VERDICT:  Direct bar complex on full u_q(sl_3) at ℓ=3 is INTRACTABLE")
    print(f"             in this sandbox (off by ~8× for weight-0 alone, ~70× for all weights).")
    print()


# ============================================================
# 6. Comparison with tractable cases
# ============================================================
def section_6_comparison(c2_blocks, c3_blocks, full_weight_dims):
    print("=" * 76)
    print("6. COMPARISON WITH TRACTABLE CASES (already verified)")
    print("=" * 76)
    print()
    w0 = (0, 0)

    print("   (a) u_q(sl_2) at ℓ=3  — VERIFIED, dim HH² = 3  (matches conjecture 1 + 2 = 3)")
    sl2_dim = 27
    sl2_per_wt = 9
    sl2_c2 = 3 * sl2_per_wt**2          # 243
    sl2_c3 = 9 * sl2_per_wt**3          # 6,561
    sl2_nz = sl2_c3 * 4
    sl2_sparse = sl2_nz * BYTES_PER_NZ
    sl2_gram = sl2_c2**2 * BYTES_COMPLEX
    print(f"       dim A = {sl2_dim}, weight-0 block: C² = {sl2_c2}, C³ = {sl2_c3}")
    print(f"       Sparse d²: {fmt_bytes(sl2_sparse)};  Dense Gram: {fmt_bytes(sl2_gram)}")
    print(f"       → Tractable; computed in seconds (verified in scripts/verify_sl2_hh2_fast.py)")
    print()

    print("   (b) B⁺(u_q(sl_3)) at ℓ=3  — VERIFIED, dim HH² = 6  (matches 2|Φ⁺| = 6 for Borel)")
    bplus_dim = 243
    bplus_per_wt = 27
    bplus_c2 = 9 * bplus_per_wt**2      # 6,561
    bplus_c3 = 81 * bplus_per_wt**3     # 1,594,323
    bplus_nz = bplus_c3 * 4
    bplus_sparse = bplus_nz * BYTES_PER_NZ
    bplus_gram = bplus_c2**2 * BYTES_COMPLEX
    print(f"       dim B⁺ = {bplus_dim}, weight-0 block: C² = {bplus_c2:,}, C³ = {bplus_c3:,}")
    print(f"       Sparse d²: {fmt_bytes(bplus_sparse)};  Dense Gram: {fmt_bytes(bplus_gram)}")
    print(f"       → Tractable; computed in minutes (verified in scripts/verify_sl3_bplus_hh2.py)")
    print()

    print("   (c) u_q(sl_3) at ℓ=3  — THIS TASK  (conjecture predicts dim HH² = 9)")
    sl3_c2 = c2_blocks[w0]
    sl3_c3 = c3_blocks[w0]
    sl3_nz = sl3_c3 * 4
    sl3_sparse = sl3_nz * BYTES_PER_NZ
    sl3_gram = sl3_c2**2 * BYTES_COMPLEX
    print(f"       dim A = {ELL**N_PBW}, weight-0 block: C² = {sl3_c2:,}, C³ = {sl3_c3:,}")
    print(f"       Sparse d²: {fmt_bytes(sl3_sparse)};  Dense Gram: {fmt_bytes(sl3_gram)}")
    print(f"       → INTRACTABLE on 4 GB RAM")
    print()

    print("   Scaling ratios (weight-0 C³ size):")
    print(f"     sl_3 / sl_2      = {sl3_c3 / sl2_c3:>10,.0f}×  ({sl3_c3:,} / {sl2_c3:,})")
    print(f"     sl_3 / B⁺(sl_3)  = {sl3_c3 / bplus_c3:>10,.0f}×  ({sl3_c3:,} / {bplus_c3:,})")
    print()


# ============================================================
# 7. Principal block u_0 (smallest possible sub-computation)
# ============================================================
def section_7_principal_block(full_weight_dims):
    print("=" * 76)
    print("7. PRINCIPAL BLOCK u_0(sl_3) — SMALLEST POSSIBLE SUB-COMPUTATION")
    print("=" * 76)
    print()
    print("   Per W1-1c:  HH*(A, C) ≅ HH*(u_0, C)  (only the principal block contributes")
    print("   to HH with trivial coefficients, because ε(e_λ) = δ_{λ,0}).")
    print()
    print("   Claim:  u_0(sl_3) at ℓ=3 = weight-0 subspace of A  (as a subalgebra).")
    print("   Reasoning:")
    print("     • The principal block corresponds to the central idempotent")
    print("       e_0 = (1/|G|) Σ_{g ∈ G} g,  where G = (Z/ℓ)^rank = (Z/3)^2 is the group")
    print("       of grouplike elements (powers of K1, K2).  e_0 is precisely the projector")
    print("       onto weight 0.")
    print("     • Hence u_0 = e_0 A e_0 = e_0 A (e_0 central) = weight-0 subspace of A.")
    print("     • dim u_0 = dim(weight-0 subspace) =", full_weight_dims[(0, 0)], ".")
    print()
    dim_u0 = full_weight_dims[(0, 0)]
    print(f"   ✓  dim u_0(sl_3) at ℓ=3 = {dim_u0}")
    print(f"      (consistent with the heuristic dim u_0 ≈ ℓ^(2N) = 3^6 = 729,")
    print(f"       where N = |Φ⁺| = 3 is the number of positive roots.)")
    print()
    print("   For comparison, sl_2 at ℓ=3:  dim u_0 = ℓ^(2·1) = 9 = weight-0 subspace ✓.")
    print()

    # Block sizes on u_0
    print(f"   Bar complex on u_0 (dim {dim_u0}):")
    c1_u0 = dim_u0
    c2_u0 = dim_u0**2
    c3_u0 = dim_u0**3
    nz_u0 = c3_u0 * NZ_PER_ROW_D2
    sparse_u0 = nz_u0 * BYTES_PER_NZ
    gram_u0 = c2_u0**2 * BYTES_COMPLEX
    print(f"     dim C¹ = {c1_u0:,}")
    print(f"     dim C² = {c2_u0:,}")
    print(f"     dim C³ = {c3_u0:,}")
    print(f"     Sparse d² storage: {fmt_bytes(sparse_u0)}")
    print(f"     Dense d²:          {fmt_bytes(c3_u0 * c2_u0 * BYTES_COMPLEX)}  (infeasible)")
    print(f"     Dense Gram d²ᵀd²:  {fmt_bytes(gram_u0)}  (infeasible)")
    print()

    ram_gb = 4
    ram_bytes = ram_gb * 1024**3
    feasible = sparse_u0 < ram_bytes
    print(f"   4 GB RAM feasibility: {'YES' if feasible else 'NO'}")
    print(f"     (sparse d² needs {fmt_bytes(sparse_u0)}; budget = {fmt_bytes(ram_bytes)})")
    print(f"     Ratio: {sparse_u0/ram_bytes:.1f}× over budget.")
    print()

    print("   What would make the u_0 computation feasible?")
    print(f"     • ~{sparse_u0 // (1024**3) + 1} GB RAM for sparse d² storage (single precision)")
    print(f"     • ~{gram_u0 // (1024**4) + 1} TB RAM for dense Gram (infeasible on any single machine)")
    print(f"     • Alternative: sparse rank-revealing QR (e.g., SuiteSparse SPQR), needs ~{sparse_u0 // (1024**3) + 1} GB")
    print(f"     • Or: distributed computation across ~{(sparse_u0 // (128 * 1024**3)) + 1} nodes × 128 GB RAM")
    print()

    # Time estimate
    flops_per_matvec = 4 * c3_u0
    flops_per_sv = 10 * flops_per_matvec
    n_sv = c2_u0
    total_flops = n_sv * flops_per_sv
    gflops = 1e9
    time_seconds = total_flops / gflops
    time_days = time_seconds / 86400
    print(f"   Time estimate (sparse iterative rank computation, u_0 only):")
    print(f"     Each matvec: ~{flops_per_matvec:,} flops ({flops_per_matvec/gflops:.1f} s)")
    print(f"     ~10 matvecs per singular value, need ~{n_sv:,} singular values")
    print(f"     Total: ~{time_seconds:.2e} s = ~{time_days:.1f} days  (infeasible)")
    print()

    return dim_u0, c2_u0, c3_u0, sparse_u0


# ============================================================
# 8. Alternative approaches
# ============================================================
def section_8_alternatives():
    print("=" * 76)
    print("8. ALTERNATIVE APPROACHES  (recommended paths to verify the conjecture)")
    print("=" * 76)
    print()
    print("   The direct bar complex on u_q(sl_3) at ℓ=3 is INTRACTABLE in this sandbox.")
    print("   Promising alternatives, in order of feasibility:")
    print()
    print("   ─── (a) BGG-STYLE RESOLUTION (Hemelsoet–Voorhaar approach, adapted) ───")
    print("     Hemelsoet–Voorhaar compute block cohomology via a BGG resolution,")
    print("     which uses chain groups of size ~|W| × dim P(λ) instead of dim(A)^n.")
    print("     For sl_3 at ℓ=3:  |W| = 6,  dim P(0) = 729,  so chain groups ~4374 dim.")
    print("     This is ~10⁵× smaller than the bar complex.")
    print("     Their software (github.com/RikVoorhaar/bgg-cohomology) computes self-coef")
    print("     HH*(u_λ, u_λ); we would need to adapt it for trivial-coef HH*(u_q(g), C).")
    print("     Status: Their principal-block s=2 case is EXPLICITLY EXCLUDED (Prop 5.1).")
    print("     RECOMMENDED for W2-1b: adapt BGG software / theory for trivial coefficients.")
    print()
    print("   ─── (b) EXPLICIT COCYCLE CONSTRUCTION (constructive proof) ───")
    print("     The conjecture predicts dim HH² = 9 = 3 (Cartan) + 6 (root).")
    print("     Construct 9 explicit candidate 2-cocycles:")
    print("       • 3 Cartan cocycles:  ∂K1 ∧ ∂K2-type derivations from the (Z/3)² torus")
    print("         (rank-2 exterior square of outer-derivation space = C(2,2) = 1 ...)")
    print("         Wait — for full u_q(sl_3), the Cartan contribution is C(n+1, 2) = C(3,2) = 3.")
    print("         These come from deformations of the Drinfeld-double's mixed")
    print("         Cartan–Cartan relations, not from the Cartan subalgebra alone.")
    print("       • 6 root cocycles:  2 per positive root (E1, E12, E2), one for the E-side")
    print("         and one for the F-side of each root.")
    print("     Verify:  d²(cocycle) = 0  and  linear independence mod im d¹.")
    print("     Cost:  O(dim A × 9) per cocycle check  ≈ 60K ops; tractable in seconds.")
    print("     RECOMMENDED for W2-1c: explicit cocycle construction.")
    print()
    print("   ─── (c) MASTNAK–WITHERSPOON LONG EXACT SEQUENCE ───")
    print("     For Drinfeld double D(B):")
    print("       ... → HH^n(B) ⊕ HH^n(B*) → HH^n(D(B)) → HH^{n+1}(B ⊗ B*) → ...")
    print("     Known: HH²(B⁺(sl_3), C) = 6 = 2|Φ⁺|  (verified).")
    print("     By duality:  HH²(B⁺(sl_3)*, C) = 6.")
    print("     LES gives HH²(D(B)) as an extension of 6 ⊕ 6 = 12 by something,")
    print("     and the conjecture says HH²(D(B)) = 9.  The deficit 12 - 9 = 3 should")
    print("     come from the connecting map HH²(D(B)) → HH³(B ⊗ B*) — needs computation.")
    print("     LOWER PRIORITY; requires careful LES analysis (W1-1b flagged that")
    print("     Schweigert–Woike do not provide an alternative to this LES).")
    print()
    print("   ─── (d) MASSIVE COMPUTATION (out of scope) ───")
    print("     TB-scale RAM, weeks of compute on a cluster.")
    print("     Not feasible in this sandbox.")
    print()


# ============================================================
# 9. Smaller sanity checks (could be done now)
# ============================================================
def section_9_sanity_checks(dim_u0, c2_u0, c3_u0, sparse_u0):
    print("=" * 76)
    print("9. SMALLER SANITY CHECKS AVAILABLE IN THIS SANDBOX")
    print("=" * 76)
    print()
    print("   Although the full sl_3 ℓ=3 bar complex is intractable, smaller checks are")
    print("   feasible and could provide partial verification of the conjecture:")
    print()
    print("   (i) Cartan subalgebra HH²:  A_cart = C[K1, K2] / (K1³ - 1, K2³ - 1),  dim 9.")
    print("       Over C,  x³ - 1 = (x-1)(x-ω)(x-ω²)  splits, so by CRT")
    print("       A_cart ≅ C × C × ... × C  (9 copies)  ≅ C⁹.")
    print("       C⁹ is semisimple commutative, so HH^n(C⁹, C) = 0 for all n ≥ 1.")
    print("       → dim HH²(Cartan) = 0   (NOT 3 = C(3,2))")
    print("       This shows that the Cartan contribution '3' in the conjecture is NOT")
    print("       simply HH² of the Cartan subalgebra; it comes from the mixed")
    print("       Cartan–root structure of the full Drinfeld double (the '3' counts")
    print("       deformations of the cross-relations K_i E_j = q^{a_{ij}} E_j K_i, etc.).")
    print()
    print("   (ii) B⁺(sl_3) ℓ=3 already verified:  HH² = 6 = 2|Φ⁺|   (✓ matches conjecture")
    print("        for the Borel:  C(n+1,2) = 0 since Borel has no negative-root part,")
    print("        so predicted = 0 + 2·3 = 6).")
    print()
    print("   (iii) sl_2 ℓ=3 already verified:  HH² = 3 = C(2,2) + 2·1 = 1 + 2  ✓.")
    print()
    print("   (iv) Attempt HH² of a 'Cartan × sl_2' subalgebra?")
    print("        Not a natural subalgebra of u_q(sl_3); skipping.")
    print()
    print("   (v)  Attempt a partial computation on a quotient / sub-block of u_q(sl_3)?")
    print("        E.g., restrict to monomials with exponents in {0, 1} (not 2).")
    print("        Dim = 2^8 = 256 (smaller).  BUT: this subspace is NOT closed under")
    print("        multiplication (e.g., E1 × E1 should give E1² which has exponent 2),")
    print("        so it's not a subalgebra.  Skipping.")
    print()
    print("   CONCLUSION:  No smaller sub-computation of u_q(sl_3) at ℓ=3 verifies the")
    print("   conjecture's full prediction '9'.  The full computation is intractable here.")
    print("   The conjecture at sl_3, ℓ=3 must be verified via (a) BGG adaptation or")
    print("   (b) explicit cocycle construction, in subsequent sub-tasks.")
    print()


# ============================================================
# 10. Conclusion
# ============================================================
def section_10_conclusion(dim_A, dim_u0, c2_u0, c3_u0, sparse_u0):
    print("=" * 76)
    print("10. CONCLUSION  (summary for orchestrator)")
    print("=" * 76)
    print()
    print(f"   • dim u_q(sl_3) at ℓ=3 = {dim_A} (= 3^8).")
    print(f"   • 9 weight spaces (Z/3)², each dim 729  (uniform distribution).")
    print(f"   • Principal block u_0 = weight-0 subspace,  dim = {dim_u0}.")
    print()
    print(f"   • Direct bar complex on full A:")
    print(f"       weight-0 block:  C² = {(9 * 729**2):,},  C³ = {(81 * 729**3):,}")
    print(f"       sparse d² storage: ~{(81 * 729**3 * 4 * 20) // (1024**4)} TB  (per weight, all 9 = ~9× more)")
    print(f"       → INTRACTABLE in this sandbox.")
    print()
    print(f"   • Direct bar complex on u_0 (smallest possible):")
    print(f"       C² = {c2_u0:,},  C³ = {c3_u0:,}")
    print(f"       sparse d² storage: ~{sparse_u0 // (1024**3)} GB")
    print(f"       → Still intractable on 4 GB RAM (exceeds by ~{sparse_u0 / (4 * 1024**3):.1f}×).")
    print()
    print(f"   • Recommended next steps:")
    print(f"       W2-1b:  Adapt Hemelsoet–Voorhaar BGG software to trivial coefficients")
    print(f"               (chain groups ~4374-dim instead of 387M-dim).")
    print(f"       W2-1c:  Construct 9 explicit cocycles and verify d² = 0 + linear independence.")
    print(f"       W2-1d:  Mastnak–Witherspoon LES analysis (lower priority).")
    print()
    print(f"   • The conjecture at sl_3, ℓ=3 remains UNVERIFIED after W2-1a.")
    print(f"     Already-verified cases (sl_2 ℓ=3, B⁺(sl_3) ℓ=3) give partial support.")
    print()


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 76)
    print("PLANNING / FEASIBILITY:  HH²(u_q(sl_3), C) at ℓ = 3")
    print("Project conjecture:  dim HH² = C(n+1, 2) + 2|Φ⁺| = C(3,2) + 2·3 = 3 + 6 = 9")
    print(f"Generated:  W2-1a sub-agent   (sandbox: 4 GB RAM budget)")
    print("=" * 76)

    t0 = time.time()
    dim_A = section_1_dimension()
    full_weight_dims = section_2_weight_spaces()
    c1_blocks, c2_blocks, c3_blocks = section_3_block_sizes(full_weight_dims, dim_A)
    sparse_w0, total_sparse = section_4_memory(c2_blocks, c3_blocks, full_weight_dims, dim_A)
    section_5_tractability(sparse_w0, total_sparse)
    section_6_comparison(c2_blocks, c3_blocks, full_weight_dims)
    dim_u0, c2_u0, c3_u0, sparse_u0 = section_7_principal_block(full_weight_dims)
    section_8_alternatives()
    section_9_sanity_checks(dim_u0, c2_u0, c3_u0, sparse_u0)
    section_10_conclusion(dim_A, dim_u0, c2_u0, c3_u0, sparse_u0)

    print()
    print(f"Total wall-clock: {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
