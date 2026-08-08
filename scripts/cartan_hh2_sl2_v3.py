#!/usr/bin/env python3
"""
Compute dim HH²(u_q(sl_2), C) at ℓ=3 via representation theory.

Strategy: compute J = rad(A) = ∩_j ker(ρ_j), then compute the
Cartan matrix from the block structure, then use the formula
HH² = Σ_j (C_{j0} - δ_{0j}) * (C_{0j} - δ_{0j}).

Key: for u_q(sl_2) at ℓ=3, the simple representations are:
  S(0): dim 1, S(1): dim 2, S(2): dim 3 (Steinberg, projective)
The combined map ρ = (ρ_0, ρ_1, ρ_2): A → M_1 ⊕ M_2 ⊕ M_3
has kernel J, and A/J ≅ im(ρ) ⊆ M_1 ⊕ M_2 ⊕ M_3.
"""
import sys, cmath, math, time
import numpy as np
sys.path.insert(0, '/home/z/my-project/hopf-decoherence/scripts')

ELL = 3; Q = cmath.exp(2j*math.pi/ELL); QI = Q**(-1); D = Q-QI; DIM = ELL**3

def idx(a,b,c): return a*ELL*ELL+b*ELL+c
def from_idx(i): return i//(ELL*ELL),(i//ELL)%ELL,i%ELL

def build_mult():
    from verify_sl2_hh2 import build_multiplication_table
    return build_mult_table() if 'build_mult_table' in dir() else build_multiplication_table()

# Representation matrices
def K_mat(j):
    if j == 0: return np.array([[1.0]])
    if j == 1: return np.diag([Q, QI])
    if j == 2: return np.diag([Q**2, 1.0, Q**(-2)])

def E_mat(j):
    if j == 0: return np.array([[0.0]])
    if j == 1: return np.array([[0,1],[0,0]],dtype=complex)
    if j == 2: return np.array([[0,1,0],[0,0,1],[0,0,0]],dtype=complex)

def F_mat(j):
    if j == 0: return np.array([[0.0]])
    if j == 1: return np.array([[0,0],[1,0]],dtype=complex)
    if j == 2: return np.array([[0,0,0],[1,0,0],[0,1,0]],dtype=complex)

def rho_combined(mult):
    """Build the combined representation matrix ρ: A → C^{1+4+9}=C^{14}.
    ρ(basis[m]) = (ρ_0(m), ρ_1(m), ρ_2(m)) flattened.
    Returns a 14 × DIM matrix."""
    dim_target = 1 + 4 + 9  # = 14
    R = np.zeros((dim_target, DIM), dtype=complex)
    for m in range(DIM):
        a, b, c = from_idx(m)
        col = []
        for j in range(3):
            d = j + 1  # dim S(j)
            K = K_mat(j); E = E_mat(j); F = F_mat(j)
            mat = np.eye(d, dtype=complex)
            mat = mat @ np.linalg.matrix_power(K, a)
            if b > 0: mat = mat @ np.linalg.matrix_power(E, b)
            if c > 0: mat = mat @ np.linalg.matrix_power(F, c)
            col.extend(mat.flatten())
        R[:, m] = col
    return R

def compute_J_and_blocks(mult):
    """Compute J = ker(ρ) and the block structure."""
    R = rho_combined(mult)  # 14 × 27
    U, s, Vh = np.linalg.svd(R, full_matrices=True)
    tol = max(R.shape) * (s[0] if len(s) > 0 else 1) * 1e-10
    rank = int(np.sum(s > tol))
    dim_J = DIM - rank
    
    print(f"  rank(ρ) = {rank} (expected 14)")
    print(f"  dim J = {dim_J} (expected 13)")
    print(f"  Singular values: {s}")
    
    # J = null space of R = last (DIM - rank) rows of Vh
    J_basis = Vh[rank:].conj()  # shape (dim_J, DIM)
    
    # A/J basis = first `rank` rows of Vh (the row space of R)
    AJ_basis = Vh[:rank].conj()  # shape (rank, DIM)
    
    return J_basis, AJ_basis, rank

def compute_cartan_via_blocks(mult, R, rank):
    """Compute the Cartan matrix from the representation theory.
    
    For each simple S(j), the projective indecomposable P(j) is the
    projective cover of S(j). The Cartan matrix entry:
    
    C_{ij} = [P(j) : S(i)] = dim Hom_A(P(i), P(j)) = dim(e_i * A * e_j)
    
    Alternative: C_{ij} = dim Hom_A(P(i), P(j)) = dim Hom_{A/J}(S(i), P(j)/JP(j))
    ... but this requires knowing P(j).
    
    Direct approach: C_{ij} = (P(j) has [P(j):S(i)] copies of S(i))
    = dim(e_i * P(j)) / dim(S(i))
    
    For the minimal projective resolution of k = S(0):
    HH^0 = 1
    HH^1 = [rad(P(0)) : S(0)] = C_{00} - 1
    HH^2 = Σ_j [rad(P(0)) : S(j)] * [rad(P(j)) : S(0)]
          = Σ_j (C_{j0} - δ_{j0}) * (C_{0j} - δ_{0j})
    
    For the Steinberg module S(2) (projective):
    P(2) = S(2), so C_{i2} = δ_{i2} and C_{2j} = [P(j) : S(2)].
    
    The key: we need C_{00}, C_{10}, C_{20}, C_{01}, C_{02}.
    
    For a symmetric Cartan matrix: C_{01} = C_{10}, C_{02} = C_{20}.
    
    C_{02} = [P(2) : S(0)] = 0 (Steinberg block is semisimple).
    C_{22} = [P(2) : S(2)] = 1.
    
    So we need: C_{00}, C_{01} = C_{10}, C_{11}, C_{20} = C_{02} = 0, C_{12} = C_{21}, C_{22} = 1.
    
    Also: dim P(j) = Σ_i C_{ij} * dim S(i)
    dim P(0) = C_{00} + 2*C_{10}
    dim P(1) = C_{01} + 2*C_{11} + 3*C_{21}
    dim P(2) = 3 (Steinberg)
    
    And: Σ_j dim S(j) * dim P(j) = 27
    1*(C_{00}+2*C_{10}) + 2*(C_{01}+2*C_{11}+3*C_{21}) + 3*3 = 27
    C_{00} + 2*C_{10} + 2*C_{01} + 4*C_{11} + 6*C_{21} + 9 = 27
    C_{00} + 4*C_{01} + 4*C_{11} + 6*C_{21} = 18  (using C_{01}=C_{10})
    
    We need additional constraints. Let me compute the Cartan matrix
    numerically from the representations.
    """
    # Compute dim(e_i * A * e_j) using the lifted idempotents
    # But idempotent lifting is hard. Instead, compute directly:
    # C_{ij} = dim Hom_A(P(i), P(j)) = dim(e_i * A * e_j)
    
    # Alternative: compute the composition factors directly from the
    # regular representation and the radical.
    
    # The regular representation A decomposes as ⊕_j P(j)^{⊕ dim S(j)}.
    # So dim A = Σ_j dim S(j) * dim P(j) = 27.
    
    # The radical J = rad(A) satisfies:
    # dim(A/J) = Σ_j (dim S(j))² = 1 + 4 + 9 = 14.
    
    # The Cartan matrix: C_{ij} = [P(j) : S(i)].
    # dim P(j) = Σ_i C_{ij} * dim S(i).
    # Σ_j dim S(j) * dim P(j) = Σ_j dim S(j) * Σ_i C_{ij} * dim S(i) = Σ_{i,j} C_{ij} * dim S(i) * dim S(j) = 27.
    
    # Also: tr(C) = Σ_j C_{jj} = Σ_j [P(j) : S(j)] = dim(A/J) = 14.
    
    # For u_q(sl_2) at ℓ=3 with S(2) projective (C_{i2}=δ_{i2}):
    # C = [[a, b, 0], [b, d, e], [0, e, 1]]
    # (symmetric, with Steinberg block semisimple)
    
    # Constraints:
    # 1. a + 4b + 4d + 6e = 18  (from dim A = 27)
    # 2. a + d + 1 = 14  (trace = dim A/J)
    # 3. dim P(0) = a + 2b, dim P(1) = b + 2d + 3e, dim P(2) = 3
    # 4. C_{ij} ≥ 0 (non-negative)
    # 5. C_{jj} ≥ 1 (each P(j) has S(j) at top)
    
    # From 2: a = 13 - d.
    # Sub into 1: (13-d) + 4b + 4d + 6e = 18 → 4b + 3d + 6e = 5.
    # With b, d, e ≥ 0 and d ≥ 1 (C_{11} ≥ 1):
    # 4b + 3d + 6e = 5, d ≥ 1.
    # If d=1: 4b + 6e = 2. So b=0, e=1/3 (not integer) or b=1/2 (not integer).
    #   Actually b and e must be non-negative INTEGERS.
    #   4b + 6e = 2 with b,e ≥ 0 integers: b=0, e=1/3 (no). No solution.
    
    # Hmm, no integer solution with d=1. Try d=0 (but C_{11} ≥ 1, so d ≥ 1).
    # Actually, is the trace formula correct? tr(C) = dim(A/J)?
    
    # tr(C) = Σ_j [P(j) : S(j)]. This is NOT dim(A/J).
    # dim(A/J) = Σ_j (dim S(j))² = 1 + 4 + 9 = 14.
    # tr(C) is the sum of the diagonal entries of the Cartan matrix.
    
    # Actually, for a finite-dimensional algebra:
    # The Cartan matrix C has the property that
    # Σ_{i,j} C_{ij} * dim S(i) * dim S(j) = dim A
    # and dim A/J = Σ_j (dim S(j))² (the dimension of the semisimple quotient).
    
    # tr(C) is NOT dim A/J in general. Let me remove that constraint.
    
    # Without the trace constraint:
    # a + 4b + 4d + 6e = 18 (from dim A)
    # a ≥ 1, d ≥ 1, e ≥ 0, b ≥ 0 (all integers)
    # C_{22} = 1, C_{02} = C_{20} = 0 (Steinberg block semisimple)
    
    # Also, for the block structure: if S(0), S(1), S(2) are in separate blocks,
    # then C is block diagonal: b = 0, e = 0.
    # a + 4d = 18, a ≥ 1, d ≥ 1.
    # If a = 2, d = 4: 2+16 = 18. ✓
    # If a = 6, d = 3: 6+12 = 18. ✓
    # If a = 10, d = 2: 10+8 = 18. ✓
    # If a = 14, d = 1: 14+4 = 18. ✓
    
    # dim P(0) = a, dim P(1) = 2d, dim P(2) = 3.
    # a + 2*2d + 3*3 = a + 4d + 9 = 18 + 9 = 27. ✓
    
    # For the small quantum group u_q(sl_2) at ℓ=3:
    # The blocks are determined by the central idempotents.
    # For u_q(sl_2), the center is C[K^3, E^3, F^3, ...] = C[K^3, ...]
    # Actually, the center of u_q(sl_2) at ℓ=3 is larger.
    
    # The key question: what are a and d?
    
    # From the representation theory: P(0) is the projective cover of S(0).
    # S(0) is the trivial module (dim 1).
    # The projective cover P(0) has dim = dim S(0) * dim(Block 0) / dim S(0) = dim(Block 0).
    # Wait, that's not right either.
    
    # For a block with one simple S(j), the projective P(j) = Block (the entire block
    # is one projective indecomposable). So dim P(j) = dim(Block j).
    # And C_{jj} = dim P(j) / dim S(j) = dim(Block j) / dim S(j).
    
    # For block 0: dim P(0) = dim(Block 0) = a, C_{00} = a / 1 = a.
    # For block 1: dim P(1) = dim(Block 1) = 2d, C_{11} = 2d / 2 = d.
    # For block 2: dim P(2) = dim(Block 2) = 3, C_{22} = 3 / 3 = 1.
    
    # And a + 4d = 18.
    
    # The block dimensions for u_q(sl_2) at ℓ=3:
    # The Steinberg block (S(2)) has dim = (dim S(2))² = 9.
    # Wait, that's only true if the block is semisimple. For the Steinberg (which is
    # simple and projective), the block IS semisimple: Block 2 = S(2) = M_3(C), dim 9.
    
    # But dim P(2) = dim(Block 2) / dim S(2) = 9/3 = 3. And C_{22} = 3/3 = 1.
    # dim(Block 2) = (dim S(2))² = 9. ✓
    
    # Similarly, if each block is semisimple (which it's NOT for non-semisimple A):
    # dim(Block 0) = (dim S(0))² = 1, dim(Block 1) = (dim S(1))² = 4, dim(Block 2) = 9.
    # Total = 1 + 4 + 9 = 14 ≠ 27. So NOT all blocks are semisimple.
    
    # The non-semisimple blocks have dim > (dim S(j))².
    # dim(Block 0) + dim(Block 1) + dim(Block 2) = 27.
    # dim(Block 2) = 9 (Steinberg, semisimple).
    # dim(Block 0) + dim(Block 1) = 18.
    
    # For the small quantum group, the block dimensions are known:
    # dim(Block j) = dim S(j) * dim P(j) = dim S(j) * (ℓ * dim S(j)) for non-Steinberg.
    # At ℓ=3: dim(Block 0) = 1 * 3 * 1 = 3, dim(Block 1) = 2 * 3 * 2 = 12.
    # Check: 3 + 12 + 9 = 24 ≠ 27. WRONG.
    
    # Hmm. Let me try dim(Block j) = ℓ * (dim S(j))² for non-Steinberg:
    # dim(Block 0) = 3 * 1 = 3, dim(Block 1) = 3 * 4 = 12.
    # Check: 3 + 12 + 9 = 24 ≠ 27. Still wrong.
    
    # Try dim(Block j) = ℓ * dim P(j) for non-Steinberg:
    # dim(Block 0) = 3 * dim P(0), dim(Block 1) = 3 * dim P(1).
    # 3*dim P(0) + 3*dim P(1) + 9 = 27 → dim P(0) + dim P(1) = 6.
    # With dim P(0) = a, dim P(1) = 2d: a + 2d = 6.
    # And a + 4d = 18. Subtracting: 2d = 12, d = 6, a = -6. NEGATIVE!
    
    # None of these formulas work. The small quantum group has a different structure.
    
    # Let me just compute it numerically. The key: compute dim(e_j * A * e_j)
    # for the correct primitive idempotents. But I can't lift the idempotents.
    
    # ALTERNATIVE: compute the Cartan matrix from the composition factors
    # of the regular representation.
    
    # The regular representation A (as a left module) decomposes as:
    # A = ⊕_j P(j)^{⊕ dim S(j)}
    # So [A : S(i)] = Σ_j dim S(j) * [P(j) : S(i)] = Σ_j dim S(j) * C_{ij}
    
    # Also, [A : S(i)] = dim(e_i * A) / dim S(i)
    # And dim(e_i * A) = dim S(i) * dim P(i)
    # So [A : S(i)] = dim P(i) = Σ_j C_{ij} * dim S(j) (consistent).
    
    # But [A : S(i)] can also be computed directly: it's the number of times
    # S(i) appears in a composition series of A (as a left module).
    # This equals dim(A * e_i) / dim S(i) where e_i is the primitive idempotent.
    
    # Since I can't compute the idempotents, let me compute [A : S(i)] differently.
    
    # [A : S(i)] = dim Hom_A(A, S(i)) / dim End(S(i)) * dim S(i)
    #            = dim S(i) / dim S(i) * dim S(i) = dim S(i).
    # Wait, that gives [A : S(i)] = dim S(i), which would mean dim A = Σ dim S(i)² = 14 ≠ 27.
    # That's the semisimple case. For non-semisimple, [A : S(i)] > dim S(i).
    
    # Actually, Hom_A(A, S(i)) = S(i) (as a vector space), so dim Hom_A(A, S(i)) = dim S(i).
    # And [A : S(i)] = dim Hom_A(A, S(i)) = dim S(i)... no, that's wrong.
    # [A : S(i)] = dim Hom_A(P(i), A) / dim End(S(i))... I'm confusing things.
    
    # The correct formula: [A : S(i)] = dim(e_i * A) / dim S(i).
    # And dim(e_i * A) = dim S(i) * dim P(i).
    # So [A : S(i)] = dim P(i) = Σ_j C_{ij} * dim S(j).
    
    # Also: Σ_i [A : S(i)] * dim S(i) = dim A = 27.
    # Σ_i dim P(i) * dim S(i) = 27. (Same as before.)
    
    # I can compute [A : S(i)] directly from the representation theory:
    # [A : S(i)] = dim Hom_A(A, S(i)) = dim S(i) (since A is free of rank 1).
    # Wait, Hom_A(A, M) ≅ M for any module M. So Hom_A(A, S(i)) ≅ S(i), dim = dim S(i).
    # But [A : S(i)] ≠ dim Hom_A(A, S(i)).
    # [A : S(i)] = (number of times S(i) appears in a composition series of A).
    
    # For the regular representation: [A : S(i)] = dim P(i) (this is a standard result).
    
    # OK, I'm stuck. Let me just try all possible Cartan matrices and check which one
    # gives the correct answer HH² = 3.
    
    # C = [[a, 0, 0], [0, d, 0], [0, 0, 1]] (block diagonal, 3 blocks)
    # a + 4d = 18
    # HH² = (a-1)²  (since C_{01}=C_{10}=0, only the j=0 term contributes)
    # (a-1)² = 3 → a-1 = √3 → NOT INTEGER.
    
    # So block-diagonal doesn't work. The blocks must be coupled.
    
    # C = [[a, b, 0], [b, d, 0], [0, 0, 1]] (Steinberg separate, S(0) and S(1) coupled)
    # a + 4b + 4d = 18
    # HH² = (a-1)² + b*(b-0) + 0 = (a-1)² + b²  (using C_{01}=b, C_{10}=b)
    # Wait: HH² = Σ_j (C_{j0}-δ_{0j})*(C_{0j}-δ_{0j})
    #            = (a-1)*(a-1) + (b-0)*(b-0) + (0-0)*(0-0)
    #            = (a-1)² + b²
    # Set (a-1)² + b² = 3.
    # Integer solutions: (a-1)² + b² = 3.
    # a-1=±1, b=±√2 — no.
    # a-1=0, b=±√3 — no.
    # a-1=±√3, b=0 — no.
    # No integer solutions!
    
    # Try: C = [[a, b, c], [b, d, e], [c, e, 1]] (all coupled)
    # HH² = (a-1)² + b² + c²  (if symmetric)
    # (a-1)² + b² + c² = 3.
    # Integer solutions:
    # (a-1)²=1, b²=1, c²=1: a=0 or 2, b=±1, c=±1. a≥1 so a=2, b=1, c=1 (or -1, but C≥0).
    # Check: a=2, b=1, c=1.
    # dim A = a + 4b + 4d + 6c + 6e + 9... wait, let me recompute.
    # Σ C_{ij} * dim S(i) * dim S(j) = a*1 + b*2 + c*3 + b*2 + d*4 + e*6 + c*3 + e*6 + 1*9
    # = a + 4b + 6c + 4d + 12e + 9 = 27
    # a + 4b + 6c + 4d + 12e = 18
    # With a=2, b=1, c=1: 2 + 4 + 6 + 4d + 12e = 18 → 4d + 12e = 6 → d=0, e=0.5 (no) or d=1.5 (no).
    # Not integer.
    
    # (a-1)²=1, b²=1, c²=1: tried above, doesn't work.
    # (a-1)²=0, b²=1, c²=2: c not integer.
    # (a-1)²=1, b²=0, c²=2: c not integer.
    # (a-1)²=0, b²=0, c²=3: c not integer.
    # (a-1)²=3, b²=0, c²=0: a-1 not integer.
    
    # No integer solutions with all entries non-negative!
    
    # This means either:
    # 1. The Cartan matrix is NOT symmetric (but it should be for f.d. algebra over C)
    # 2. Some entries are negative (impossible for Cartan matrix)
    # 3. HH² ≠ 3 (but we VERIFIED it by bar complex and exact certification!)
    # 4. My formula HH² = Σ_j (C_{j0}-δ_{0j})*(C_{0j}-δ_{0j}) is WRONG.
    
    # Let me re-derive the formula.
    # HH^n = dim Hom_A(P_n, k) for minimal resolution.
    # P_0 = P(0), Hom(P(0), k) = Hom(P(0), S(0)) = C (1-dim).
    # HH^0 = 1. ✓
    # K_1 = rad(P(0)). [K_1 : S(j)] = C_{j0} - δ_{j0}.
    # P_1 = ⊕_j P(j)^{[K_1:S(j)]}. Hom(P_1, k) = ⊕_j Hom(P(j), k)^{[K_1:S(j)]}.
    # Hom(P(j), k) = Hom(P(j), S(0)) = δ_{j0} * C (1-dim if j=0, 0 if j≠0).
    # So dim Hom(P_1, k) = [K_1 : S(0)] = C_{00} - 1.
    # HH^1 = C_{00} - 1 (for minimal resolution, im(d_0*) = 0). ✓
    
    # K_2 = rad(P_1). [K_2 : S(i)] = Σ_j [K_1 : S(j)] * ([P(j) : S(i)] - δ_{ij})
    #       = Σ_j (C_{j0} - δ_{j0}) * (C_{ij} - δ_{ij}).
    # dim Hom(P_2, k) = [K_2 : S(0)] = Σ_j (C_{j0} - δ_{j0}) * (C_{0j} - δ_{0j}).
    # HH^2 = [K_2 : S(0)] (for minimal resolution). ✓
    
    # So the formula IS correct. But there's no integer Cartan matrix giving HH²=3?
    
    # Wait — I assumed the Cartan matrix is symmetric. Is this always true?
    # For a finite-dimensional algebra over an algebraically closed field, the Cartan
    # matrix IS symmetric. This is a theorem (Cartan matrix = matrix of the Cartan
    # pairing, which is symmetric).
    
    # Actually, the Cartan matrix C_{ij} = [P(j) : S(i)] is NOT necessarily symmetric!
    # It IS symmetric if the algebra is symmetric (i.e., A ≅ A* as A-bimodules),
    # which is the case for Hopf algebras. u_q(sl_2) is a Hopf algebra, so C IS symmetric.
    
    # But we showed there's no non-negative integer symmetric Cartan matrix with
    # C_{22}=1, C_{02}=C_{20}=0, and HH²=3. So something else is wrong.
    
    # AH WAIT: I assumed C_{02} = C_{20} = 0 (Steinberg block semisimple). But if
    # S(2) is in the SAME block as S(0) or S(1), then C_{02} or C_{12} ≠ 0!
    
    # For u_q(sl_2) at ℓ=3, the block structure is NOT obvious. Let me check:
    # S(2) is projective (Steinberg), so it's in its own block? Actually, a projective
    # simple can be in a block with non-projective simples. The block is determined by
    # the central idempotents, not by projectivity.
    
    # For u_q(sl_2), the center contains the Casimir element C = EF + FE + (qK+q^{-1}K^{-1})/(q+q^{-1})².
    # The central idempotents are determined by the eigenvalues of C on the simples.
    # On S(j): C acts as j(j+2)/(q+q^{-1})²... this is getting complicated.
    
    # Let me just try: ALL THREE simples in one block.
    # C = [[a, b, c], [b, d, e], [c, e, f]] with f = C_{22}.
    # S(2) is projective: P(2) = S(2), so C_{i2} = δ_{i2} and C_{2j} = [P(j) : S(2)].
    # Wait, if S(2) is projective AND simple, then P(2) = S(2) and [P(2) : S(i)] = δ_{i2}.
    # So C_{i2} = δ_{i2}: c = 0, e = 0, f = 1.
    # But C_{2j} = [P(j) : S(2)] can be nonzero!
    
    # So C = [[a, b, 0], [b, d, 0], [c', e', 1]] where c' = C_{20} = [P(0) : S(2)]
    # and e' = C_{21} = [P(1) : S(2)].
    
    # If C is symmetric: c' = 0 and e' = 0 (since C_{02} = 0 implies C_{20} = 0).
    # But that gives the block-diagonal case we already tried.
    
    # UNLESS C is NOT symmetric! But for a Hopf algebra, C IS symmetric.
    
    # Hmm, maybe S(2) being projective doesn't mean C_{i2} = δ_{i2}.
    # If S(2) is projective, then P(2) = S(2), and [P(2) : S(i)] = δ_{i2}.
    # So C_{i2} = δ_{i2}, meaning column 2 is (0, 0, 1).
    # By symmetry, row 2 is also (0, 0, 1), so C_{2j} = δ_{2j}.
    # This means S(2) is in its own block.
    
    # So we're back to: C = [[a, b, 0], [b, d, 0], [0, 0, 1]]
    # with a + 4b + 4d = 18 and (a-1)² + b² = 3.
    
    # Since there's no integer solution, maybe the Cartan matrix formula
    # for HH² is wrong, or the Cartan matrix is different from what I think.
    
    # Let me try a different approach: compute the Cartan matrix NUMERICALLY
    # by computing the composition factors of the regular representation.
    
    # The regular representation A has composition factors:
    # [A : S(i)] = dim P(i) (standard result for f.d. algebras).
    # dim P(i) = Σ_j C_{ij} * dim S(j).
    
    # I can compute [A : S(i)] = dim(e_i * A) / dim S(i) if I have the idempotents.
    # But I can also compute it as:
    # [A : S(i)] = dim Hom_D(A)(ρ_i, A) where D(A) is the enveloping algebra.
    # This is getting circular.
    
    # Actually, there's a simpler formula:
    # [A : S(i)] = trace of the central idempotent e_i acting on A / dim S(i)
    # But I need the central idempotents...
    
    # Let me just compute dim P(i) directly from the block decomposition.
    # The blocks are determined by the central idempotents of A.
    # For u_q(sl_2), the center is generated by the Casimir and the q-Casimir.
    
    # Actually, for u_q(sl_2) at ℓ=3, the center is known:
    # Z(u_q(sl_2)) = C[K^3, E^3, F^3, EF+FE+...] / (relations)
    # At ℓ=3: K^3 = 1, E^3 = F^3 = 0, so the center contains K (which generates C^3)
    # and the q-Casimir C_q = EF + (qK + q^{-1}K^{-1})/(q+q^{-1})^2.
    
    # The central idempotents are polynomials in C_q.
    # On S(j): C_q acts as the eigenvalue c_j = (q^{j+1} + q^{-(j+1)}) / (q + q^{-1})^2
    # = [j+1]_q^2 / [1]_q^2 where [n]_q = (q^n - q^{-n})/(q - q^{-1}).
    
    # For ℓ=3, q = e^{2πi/3}:
    # [1]_q = (q - q^{-1})/(q - q^{-1}) = 1
    # [2]_q = (q^2 - q^{-2})/(q - q^{-1}) = (q^2 - q)/(q - q^{-1}) = q(q-1)/(q-q^{-1})
    # [3]_q = (q^3 - q^{-3})/(q - q^{-1}) = (1 - 1)/(q - q^{-1}) = 0
    
    # On S(0): C_q acts as [1]² = 1.
    # On S(1): C_q acts as [2]².
    # On S(2): C_q acts as [3]² = 0.
    
    # So the eigenvalues are distinct (1, [2]², 0), and each simple is in its own block.
    # This confirms: 3 blocks, C is block diagonal.
    
    # But we showed there's no integer solution with HH² = 3 for block-diagonal C.
    # Something is fundamentally wrong.
    
    # WAIT: maybe dim P(0) ≠ a (i.e., C_{00} ≠ dim P(0) / dim S(0)).
    # For a block with one simple S(j) of dim > 1, the block is NOT just P(j).
    # The block decomposes as P(j)^{⊕ dim S(j)}.
    # So dim(Block j) = dim S(j) * dim P(j).
    # And C_{jj} = [P(j) : S(j)] = dim P(j) / dim S(j)... no, that's wrong.
    # C_{jj} = [P(j) : S(j)] (composition factor multiplicity), which is NOT dim P(j)/dim S(j).
    
    # For a block with one simple S(j), P(j) is the unique indecomposable projective.
    # The regular representation restricted to this block is P(j)^{⊕ dim S(j)}.
    # So dim(Block j) = dim S(j) * dim P(j).
    # And C_{jj} = [P(j) : S(j)] = dim P(j) / dim S(j)... ONLY if all composition factors are S(j).
    
    # For a LOCAL algebra (one simple, one projective), P(j) has only S(j) as composition factor.
    # So C_{jj} = dim P(j) / dim S(j), and dim(Block j) = dim S(j) * dim P(j) = C_{jj} * (dim S(j))².
    
    # For block 0 (S(0) dim 1): C_{00} = dim P(0) / 1 = dim P(0).
    # dim(Block 0) = 1 * dim P(0) = C_{00}.
    # For block 1 (S(1) dim 2): C_{11} = dim P(1) / 2.
    # dim(Block 1) = 2 * dim P(1) = 2 * 2 * C_{11} = 4 * C_{11}.
    # For block 2 (S(2) dim 3): C_{22} = 1 (Steinberg = simple = projective).
    # dim(Block 2) = 3 * 3 = 9.
    
    # dim(Block 0) + dim(Block 1) + dim(Block 2) = C_{00} + 4*C_{11} + 9 = 27.
    # C_{00} + 4*C_{11} = 18.
    
    # HH² = (C_{00} - 1)² (since C_{01} = C_{10} = 0 for separate blocks).
    # (C_{00} - 1)² = 3.
    # C_{00} - 1 = √3. NOT INTEGER.
    
    # This is impossible. HH² = 3 is verified. So either:
    # 1. The blocks are NOT separate (C_{01} ≠ 0), or
    # 2. The Cartan matrix is NOT symmetric, or
    # 3. The formula HH² = (C_{00}-1)² + C_{01}² is wrong.
    
    # Let me re-derive more carefully.
    # K_1 = rad(P(0)). [K_1 : S(0)] = C_{00} - 1 (subtract the top S(0)).
    # K_1 has NO S(1) or S(2) factors (since blocks are separate).
    # P_1 = P(0)^{C_{00}-1} (copies of P(0)).
    # K_2 = rad(P_1) = rad(P(0))^{C_{00}-1} = K_1^{C_{00}-1}.
    # [K_2 : S(0)] = (C_{00}-1) * [K_1 : S(0)] = (C_{00}-1) * (C_{00}-1) = (C_{00}-1)².
    # HH² = (C_{00}-1)².
    
    # So HH² = (C_{00}-1)² = 3 → C_{00} = 1 + √3. NOT INTEGER.
    
    # CONTRADICTION. This means either:
    # (a) The blocks are NOT separate (S(0) and S(1) are in the same block), or
    # (b) P(0) has composition factors other than S(0) (i.e., C_{10} ≠ 0 or C_{20} ≠ 0).
    
    # If S(0) and S(1) are in the same block, the Casimir eigenvalues must be equal
    # on S(0) and S(1). But we computed: C_q(S(0)) = 1, C_q(S(1)) = [2]².
    # [2]_q at q = e^{2πi/3}: [2] = (q²-q^{-2})/(q-q^{-1}) = (q²-q)/(q-q^{-1}) (since q^{-2}=q)
    #   = q(q-1)/(q-q^{-1}) = q(q-1)/(q-q²) (since q^{-1}=q²) = q(q-1)/(q(1-q)) = -(q-1)/(1-q) = 1.
    # Wait, that gives [2]_q = 1. Let me recalculate.
    # q = e^{2πi/3}, q² = e^{4πi/3} = e^{-2πi/3} = q^{-1}.
    # [2]_q = (q² - q^{-2})/(q - q^{-1}) = (q^{-1} - q)/(q - q^{-1}) = -(q - q^{-1})/(q - q^{-1}) = -1.
    # So [2]_q = -1, and [2]_q² = 1.
    
    # So C_q(S(0)) = [1]² = 1 and C_q(S(1)) = [2]² = 1. SAME EIGENVALUE!
    # This means S(0) and S(1) ARE IN THE SAME BLOCK!
    
    # And C_q(S(2)) = [3]² = 0. Different eigenvalue, so S(2) is in a separate block.
    
    # So the blocks are: {S(0), S(1)} and {S(2)}.
    # C = [[a, b, 0], [b, d, 0], [0, 0, 1]] with a + 4b + 4d = 18.
    # HH² = (a-1)² + b².
    # Need: (a-1)² + b² = 3 with a, b non-negative integers and a + 4b + 4d = 18.
    
    # (a-1)² + b² = 3:
    # a-1=±1, b=±√2 — no.
    # a-1=0, b=±√3 — no.
    # NO INTEGER SOLUTIONS!
    
    # Hmm. Wait, maybe the q-Casimir eigenvalue computation is wrong.
    # Let me recalculate.
    
    # The q-Casimir for u_q(sl_2): C_q = EF + FE + (qK + q^{-1}K^{-1})/(q+q^{-1})²
    # Actually, the standard q-Casimir is: C_q = EF + (qK + q^{-1}K^{-1})/(q - q^{-1})²
    # Or sometimes: C_q = FE + (q^{-1}K + qK^{-1})/(q - q^{-1})²
    
    # On S(j) with highest weight j:
    # EF acts as 0 on the highest weight vector.
    # FE acts as [j]_q * [j+1]_q (or something like that).
    # K acts as q^j on the highest weight vector.
    
    # Actually, the eigenvalue of the q-Casimir on S(j) is:
    # c_j = [j]_q * [j+1]_q + (q^{j+1} + q^{-(j+1)}) / (q + q^{-1})
    # or some variation. The exact formula depends on the normalization.
    
    # Let me compute the eigenvalues directly.
    # For S(0) (trivial): C_q = EF + FE + ... acts as 0 + 0 + (q + q^{-1})/(q + q^{-1}) = 1.
    # Actually, on S(0): E = F = 0, K = 1. So C_q = 0 + 0 + (q + q^{-1})/(q + q^{-1}) = 1.
    
    # For S(1) (dim 2): need to compute EF + FE + (qK + q^{-1}K^{-1})/(q + q^{-1})².
    # K = diag(q, q^{-1}), E = [[0,1],[0,0]], F = [[0,0],[1,0]].
    # EF = [[1,0],[0,0]], FE = [[0,0],[0,1]].
    # EF + FE = I (identity).
    # (qK + q^{-1}K^{-1}) = diag(q² + q^{-2}, q^{-2} + q²) = (q² + q^{-1}) * I (since q^{-2} = q, q² = q^{-1}).
    # Wait: q² = e^{4πi/3} = e^{-2πi/3} = q^{-1}. And q^{-2} = q.
    # So qK + q^{-1}K^{-1} = diag(q*q + q^{-1}*q^{-1}, q*q^{-1} + q^{-1}*q) = diag(q² + q^{-2}, 1 + 1) = diag(q^{-1} + q, 2).
    # (q + q^{-1})² = q² + 2 + q^{-2} = q^{-1} + 2 + q = (q + q^{-1}) + 2.
    # At q = e^{2πi/3}: q + q^{-1} = -1 (since q + q² = -1).
    # So (q + q^{-1})² = 1.
    # C_q on S(1) = I + diag(q^{-1}+q, 2)/1 = I + diag(-1, 2) = diag(0, 3).
    
    # Hmm, that's not a scalar. The q-Casimir should be central, so it should act as a scalar.
    # Let me recalculate.
    
    # Actually, the q-Casimir is: C_q = EF + (qK + q^{-1}K^{-1}) / (q - q^{-1})².
    # Note: (q - q^{-1})², not (q + q^{-1})².
    # (q - q^{-1})² = q² - 2 + q^{-2} = q^{-1} - 2 + q = (q + q^{-1}) - 2 = -1 - 2 = -3.
    
    # On S(0): EF = 0, K = 1. C_q = 0 + (q + q^{-1}) / (-3) = (-1)/(-3) = 1/3.
    # On S(1): EF = [[1,0],[0,0]]. qK + q^{-1}K^{-1} = diag(q²+q^{-2}, q+q^{-1}) = diag(-1, -1) (using q²+q^{-2} = q^{-1}+q = -1).
    # Wait: q² = q^{-1}, so q² + q^{-2} = q^{-1} + q = -1.
    # And q + q^{-1} = -1.
    # So qK + q^{-1}K^{-1} = diag(-1, -1) = -I.
    # C_q = [[1,0],[0,0]] + (-I)/(-3) = [[1,0],[0,0]] + [[1/3,0],[0,1/3]] = [[4/3,0],[0,1/3]].
    
    # That's NOT a scalar! The q-Casimir is NOT central for the small quantum group?
    
    # Actually, the q-Casimir IS central for U_q(sl_2) but may NOT be central for u_q(sl_2)
    # (the small quantum group). The center of u_q(sl_2) is different.
    
    # For u_q(sl_2) at ℓ=3, the center is generated by:
    # K^3 = 1 (central, acts as 1 on all simples — not useful for distinguishing)
    # E^3 = 0 (central, acts as 0 on all simples — not useful)
    # F^3 = 0 (central, acts as 0 on all simples — not useful)
    # The element C = EF + FE + (K + K^{-1})/(q - q^{-1})²... let me check if it's central.
    
    # Actually, for the small quantum group, the center is generated by the "Frobenius"
    # center: elements that are fixed by the quantum Frobenius. The center of u_q(sl_2)
    # at ℓ=3 is the image of the center of U_q(sl_2) under the quotient map.
    
    # The center of U_q(sl_2) is generated by the q-Casimir C_q. The image of C_q in
    # u_q(sl_2) may or may not be central.
    
    # OK, I'm spending too much time on the block structure. Let me just compute
    # the Cartan matrix numerically by a different method.
    
    # Method: compute the composition factors of the regular representation A.
    # [A : S(i)] = dim P(i).
    # I can compute [A : S(i)] by counting the dimension of the S(i)-isotypic component
    # of A (as a left module).
    
    # The S(i)-isotypic component of A is e_i * A where e_i is the CENTRAL idempotent
    # for the block containing S(i). But I can't compute the central idempotents either.
    
    # Actually, there's a trick: the character of the regular representation is
    # χ_A = Σ_i dim S(i) * χ_{S(i)} * dim P(i).
    # And χ_A(g) = trace of left-multiplication by g on A = dim A * ε(g) = 27 * ε(g).
    # So Σ_i dim S(i) * dim P(i) * χ_{S(i)}(g) = 27 * ε(g) for all g.
    
    # For g = K^a: χ_{S(i)}(K^a) = Σ_{k=0}^{i} q^{a(i-2k)} (the character of S(i)).
    # ε(K^a) = 1.
    # So Σ_i dim S(i) * dim P(i) * χ_{S(i)}(K^a) = 27 for all a.
    
    # For g = E: χ_{S(i)}(E) = 0 (E is nilpotent, trace = 0).
    # ε(E) = 0. ✓
    
    # For g = E^b F^c with (b,c) ≠ (0,0): χ_{S(i)}(g) = trace of ρ_i(g).
    # ε(g) = 0. So Σ_i dim S(i) * dim P(i) * trace(ρ_i(g)) = 0.
    
    # This gives us: Σ_i dim S(i) * dim P(i) * trace(ρ_i(K^a)) = 27 for a = 0, 1, 2.
    # dim S(i) * trace(ρ_i(K^a)) is the character of S(i) evaluated at K^a times dim S(i).
    
    # For a = 0: Σ_i dim S(i) * dim P(i) * dim S(i) = 27.
    #   1*dim P(0)*1 + 2*dim P(1)*2 + 3*dim P(2)*3 = 27.
    #   dim P(0) + 4*dim P(1) + 9*dim P(2) = 27.
    #   With dim P(2) = 3: dim P(0) + 4*dim P(1) = 0. IMPOSSIBLE (dims ≥ 1).
    
    # Wait, that can't be right. The character of the regular representation at K^0 = 1
    # is dim A = 27. And Σ_i dim S(i) * dim P(i) * χ_{S(i)}(1) = Σ_i dim S(i) * dim P(i) * dim S(i) = 27.
    # 1*dim P(0)*1 + 2*dim P(1)*2 + 3*dim P(2)*3 = dim P(0) + 4*dim P(1) + 9*dim P(2) = 27.
    # With dim P(2) = 3: dim P(0) + 4*dim P(1) + 27 = 27 → dim P(0) + 4*dim P(1) = 0.
    # IMPOSSIBLE!
    
    # The issue: the formula is χ_A = Σ_i [A : S(i)] * χ_{S(i)}, not Σ_i dim S(i) * dim P(i) * χ_{S(i)}.
    # [A : S(i)] = dim P(i) (NOT dim S(i) * dim P(i)).
    # Wait, [A : S(i)] = dim S(i) * dim P(i) (since A = ⊕ P(j)^{⊕ dim S(j)} and [P(j) : S(i)] = C_{ij}).
    # Actually: [A : S(i)] = Σ_j dim S(j) * [P(j) : S(i)] = Σ_j dim S(j) * C_{ij}.
    # And dim A = Σ_i [A : S(i)] * dim S(i) = Σ_i (Σ_j dim S(j) * C_{ij}) * dim S(i) = Σ_{i,j} C_{ij} * dim S(i) * dim S(j).
    
    # The character formula: χ_A(g) = Σ_i [A : S(i)] * χ_{S(i)}(g).
    # At g = 1: χ_A(1) = dim A = Σ_i [A : S(i)] * dim S(i) = 27.
    
    # For g = K: χ_A(K) = trace of left-mult by K on A.
    # K * (K^a E^b F^c) = q^{2(b-c)} K^{a+1} E^b F^c (up to the KE = q^2 EK relation).
    # Actually, K * K^a = K^{a+1}. K * E^b = q^{2b} E^b * K. So K * K^a E^b F^c = q^{2b-2c} K^{a+1} E^b F^c.
    # The trace of left-mult by K: Σ_m <basis[m], K * basis[m]> = Σ_{a,b,c} <K^a E^b F^c, q^{2(b-c)} K^{a+1} E^b F^c>.
    # This is nonzero only when K^{a+1} = K^a (mod K^3 = 1), i.e., a+1 ≡ a (mod 3), which is never true.
    # So trace = 0.
    
    # χ_{S(i)}(K) = Σ_{k=0}^{i} q^{i-2k} (the trace of K on S(i)).
    # S(0): χ = 1.
    # S(1): χ = q + q^{-1} = -1.
    # S(2): χ = q^2 + 1 + q^{-2} = q^{-1} + 1 + q = 0.
    
    # So χ_A(K) = Σ_i [A : S(i)] * χ_{S(i)}(K) = [A:S(0)]*1 + [A:S(1)]*(-1) + [A:S(2)]*0 = [A:S(0)] - [A:S(1)] = 0.
    # So [A:S(0)] = [A:S(1)].
    
    # Similarly, χ_A(K^2) = trace of left-mult by K^2.
    # K^2 * K^a = K^{a+2}. Trace nonzero when a+2 ≡ a (mod 3), i.e., 2 ≡ 0 (mod 3). Never.
    # So χ_A(K^2) = 0.
    # χ_{S(0)}(K^2) = 1, χ_{S(1)}(K^2) = q^2 + q^{-2} = q^{-1} + q = -1, χ_{S(2)}(K^2) = q^4 + 1 + q^{-4} = q + 1 + q^{-1} = 0.
    # 0 = [A:S(0)]*1 + [A:S(1)]*(-1) + [A:S(2)]*0 → [A:S(0)] = [A:S(1)]. (Same as before.)
    
    # At g = 1: [A:S(0)]*1 + [A:S(1)]*2 + [A:S(2)]*3 = 27.
    # With [A:S(0)] = [A:S(1)] = x and [A:S(2)] = y:
    # x + 2x + 3y = 27 → 3x + 3y = 27 → x + y = 9.
    
    # Also, [A:S(2)] = dim P(2) = 3 (Steinberg is projective). So y = 3, x = 6.
    # [A:S(0)] = [A:S(1)] = 6, [A:S(2)] = 3.
    
    # And [A:S(i)] = Σ_j dim S(j) * C_{ij}:
    # [A:S(0)] = C_{00} + 2*C_{01} + 3*C_{02} = 6
    # [A:S(1)] = C_{10} + 2*C_{11} + 3*C_{12} = 6
    # [A:S(2)] = C_{20} + 2*C_{21} + 3*C_{22} = 3
    
    # With C_{22} = 1, C_{02} = C_{20} = 0, C_{12} = C_{21} = 0 (Steinberg block):
    # C_{00} + 2*C_{01} = 6
    # C_{10} + 2*C_{11} = 6
    # 3*1 = 3 ✓
    
    # Symmetric: C_{01} = C_{10} = b.
    # C_{00} + 2b = 6 → C_{00} = 6 - 2b.
    # b + 2*C_{11} = 6 → C_{11} = (6-b)/2.
    
    # For C_{11} to be integer: b must be even. b = 0, 2, 4, 6.
    # b = 0: C_{00} = 6, C_{11} = 3. HH² = (6-1)² + 0 = 25. WRONG.
    # b = 2: C_{00} = 2, C_{11} = 2. HH² = (2-1)² + 4 = 5. WRONG.
    # b = 4: C_{00} = -2. NEGATIVE.
    
    # Hmm, b = 0 gives HH² = 25, b = 2 gives HH² = 5. Neither is 3.
    
    # Wait, maybe I need to also include C_{02} ≠ 0 (S(0) and S(2) in the same block).
    # Let me drop the assumption that C_{02} = 0.
    
    # C = [[a, b, c], [b, d, e], [c, e, 1]] (symmetric, S(2) projective so C_{22}=1)
    # [A:S(0)] = a + 2b + 3c = 6
    # [A:S(1)] = b + 2d + 3e = 6
    # [A:S(2)] = c + 2e + 3 = 3 → c + 2e = 0 → c = 0, e = 0 (non-negative integers).
    
    # So C_{02} = C_{20} = 0 and C_{12} = C_{21} = 0 after all. Back to:
    # C_{00} + 2b = 6, b + 2d = 6.
    # b = 0: a=6, d=3, HH² = 25.
    # b = 2: a=2, d=2, HH² = 5.
    # b = 4: a=-2. Invalid.
    
    # NEITHER GIVES HH² = 3!
    
    # So my formula for HH² must be wrong. Let me re-examine.
    
    # The formula HH² = Σ_j (C_{j0} - δ_{0j}) * (C_{0j} - δ_{0j}) assumes
    # that rad(P(j)) has composition factors [rad(P(j)) : S(i)] = C_{ij} - δ_{ij}.
    # This is only true if P(j) has S(j) appearing ONLY at the top (the Loewy length is 1
    # for the top layer). But for non-semisimple algebras, P(j) may have S(j) in multiple
    # Loewy layers.
    
    # Actually, [rad(P(j)) : S(i)] = [P(j) : S(i)] - [top(P(j)) : S(i)] = C_{ij} - δ_{ij}.
    # The top of P(j) is P(j)/rad(P(j)) = S(j), so [top : S(i)] = δ_{ij}. ✓
    # So [rad(P(j)) : S(i)] = C_{ij} - δ_{ij}. ✓
    
    # Then K_1 = rad(P(0)):
    # [K_1 : S(j)] = C_{j0} - δ_{j0}.
    # P_1 = ⊕_j P(j)^{[K_1:S(j)]} = ⊕_j P(j)^{C_{j0}-δ_{j0}}.
    # K_2 = rad(P_1) = ⊕_j rad(P(j))^{C_{j0}-δ_{j0}}.
    # [K_2 : S(0)] = Σ_j (C_{j0}-δ_{j0}) * [rad(P(j)) : S(0)] = Σ_j (C_{j0}-δ_{j0}) * (C_{0j}-δ_{0j}).
    
    # This is what I had. So the formula IS correct.
    
    # With b=0: C = [[6,0,0],[0,3,0],[0,0,1]]. HH² = (6-1)² + 0 + 0 = 25. WRONG.
    # With b=2: C = [[2,2,0],[2,2,0],[0,0,1]]. HH² = (2-1)² + 2² + 0 = 1+4 = 5. WRONG.
    
    # But we KNOW HH² = 3 (verified by bar complex and exact certification).
    # So either:
    # 1. The Cartan matrix is wrong (my computation of [A:S(i)] is wrong), or
    # 2. The minimal projective resolution is NOT given by the simple formula I'm using.
    
    # Actually, wait. The formula HH^n = [K_n : S(0)] for the MINIMAL resolution is:
    # HH^n = dim Hom_A(P_n, S(0)) = number of copies of P(0) in P_n = [K_n : S(0)].
    
    # But [K_n : S(0)] is NOT the same as Σ_j (C_{j0}-δ_{j0}) * (C_{0j}-δ_{0j}).
    # The latter assumes that K_2 = rad(P_1), which is true for the minimal resolution.
    # But [K_2 : S(0)] = [rad(P_1) : S(0)], and this IS:
    # [rad(P_1) : S(0)] = Σ_j [P_1 : P(j)] * [rad(P(j)) : S(0)]
    #                   = Σ_j [K_1 : S(j)] * (C_{0j} - δ_{0j})
    #                   = Σ_j (C_{j0} - δ_{j0}) * (C_{0j} - δ_{0j}).
    
    # So the formula should be correct.
    
    # Let me check with b=0: C = [[6,0,0],[0,3,0],[0,0,1]].
    # [K_1 : S(0)] = C_{00} - 1 = 5. [K_1 : S(1)] = 0. [K_1 : S(2)] = 0.
    # P_1 = P(0)^5. K_2 = rad(P(0))^5. [K_2 : S(0)] = 5 * (C_{00}-1) = 5*5 = 25. HH² = 25.
    
    # With b=2: C = [[2,2,0],[2,2,0],[0,0,1]].
    # [K_1 : S(0)] = 1. [K_1 : S(1)] = 2. [K_1 : S(2)] = 0.
    # P_1 = P(0)^1 ⊕ P(1)^2.
    # K_2 = rad(P(0))^1 ⊕ rad(P(1))^2.
    # [K_2 : S(0)] = 1*(C_{00}-1) + 2*(C_{01}-0) = 1*1 + 2*2 = 5. HH² = 5.
    
    # Neither gives 3. The issue must be in my computation of [A:S(i)].
    
    # Let me double-check [A:S(0)] = [A:S(1)] = 6, [A:S(2)] = 3.
    # [A:S(2)] = dim P(2) = 3 (Steinberg is simple projective). ✓
    # [A:S(0)] = [A:S(1)] (from the character at K). ✓
    # x + y = 9 with y = 3 gives x = 6. ✓
    
    # But wait, [A:S(i)] = dim P(i) is the STANDARD result only for BASIC algebras.
    # For non-basic algebras (like u_q(sl_2) with dim S(1) = 2), the formula is:
    # A = ⊕_j P(j)^{⊕ dim S(j)} (as left modules).
    # [A : S(i)] = Σ_j dim S(j) * [P(j) : S(i)] = Σ_j dim S(j) * C_{ij}.
    
    # This is what I used. So the computation should be correct.
    
    # UNLESS the blocks are different from what I computed. Let me re-examine the
    # q-Casimir eigenvalue.
    
    # For u_q(sl_2), the q-Casimir is:
    # C_q = EF + FE + (qK + q^{-1}K^{-1}) / (q - q^{-1})²
    # At q = e^{2πi/3}: (q - q^{-1})² = (q - q²)² = q²(1-q)².
    # q - q^{-1} = q - q² = q(1-q). (q-q^{-1})² = q²(1-q)².
    # q² = q^{-1}, so (q-q^{-1})² = q^{-1}(1-q)².
    
    # On S(0): C_q = 0 + 0 + (q + q^{-1}) / (q-q^{-1})² = (-1) / (q²(1-q)²).
    # q(1-q) = q - q² = q - q^{-1}. So (q-q^{-1})² = q²(1-q)² = (q-q^{-1})².
    # C_q(S(0)) = (q+q^{-1})/(q-q^{-1})² = (-1)/(q-q^{-1})².
    # (q-q^{-1})² = q² - 2 + q^{-2} = q^{-1} - 2 + q = (q+q^{-1}) - 2 = -1-2 = -3.
    # C_q(S(0)) = -1/(-3) = 1/3.
    
    # On S(1): EF = [[1,0],[0,0]], FE = [[0,0],[0,1]].
    # EF + FE = I.
    # qK + q^{-1}K^{-1} = diag(q² + q^{-2}, q^{-1} + q) = diag(q^{-1}+q, q+q^{-1}) = diag(-1, -1) = -I.
    # C_q = I + (-I)/(-3) = I + I/3 = (4/3)I. Scalar! Eigenvalue = 4/3.
    
    # On S(2): EF + FE = ? Let me compute.
    # E = [[0,1,0],[0,0,1],[0,0,0]], F = [[0,0,0],[1,0,0],[0,1,0]].
    # EF = [[1,0,0],[0,1,0],[0,0,0]], FE = [[0,0,0],[0,0,0],[0,1,0]]... wait.
    # EF: row i of E times column j of F.
    # E[0] = [0,1,0], F[:,0] = [0,1,0]^T → EF[0,0] = 1.
    # E[0] = [0,1,0], F[:,1] = [0,0,1]^T → EF[0,1] = 0.
    # E[1] = [0,0,1], F[:,0] = [0,1,0]^T → EF[1,0] = 0.
    # E[1] = [0,0,1], F[:,1] = [0,0,1]^T → EF[1,1] = 1.
    # E[2] = [0,0,0] → EF[2,:] = 0.
    # EF = [[1,0,0],[0,1,0],[0,0,0]].
    
    # FE: F[0] = [0,0,0] → FE[0,:] = 0.
    # F[1] = [1,0,0], E[:,0] = [0,0,0] → FE[1,0] = 0.
    # F[1] = [1,0,0], E[:,1] = [1,0,0] → FE[1,1] = 0.
    # F[2] = [0,1,0], E[:,0] = [0,0,0] → FE[2,0] = 0.
    # F[2] = [0,1,0], E[:,1] = [1,0,0] → FE[2,1] = 1.
    # F[2] = [0,1,0], E[:,2] = [0,1,0] → FE[2,2] = 0.
    # FE = [[0,0,0],[0,0,0],[0,1,0]].
    
    # EF + FE = [[1,0,0],[0,1,0],[0,1,0]].
    
    # qK + q^{-1}K^{-1}: K = diag(q², 1, q^{-2}), K^{-1} = diag(q^{-2}, 1, q²).
    # qK = diag(q³, q, q^{-1}) = diag(1, q, q^{-1}).
    # q^{-1}K^{-1} = diag(q^{-3}, q^{-1}, q) = diag(1, q^{-1}, q).
    # qK + q^{-1}K^{-1} = diag(2, q+q^{-1}, q+q^{-1}) = diag(2, -1, -1).
    
    # C_q = [[1,0,0],[0,1,0],[0,1,0]] + diag(2,-1,-1)/(-3) = [[1,0,0],[0,1,0],[0,1,0]] + [[-2/3,0,0],[0,1/3,0],[0,0,1/3]]
    # = [[1/3,0,0],[0,4/3,0],[0,1,1/3]].
    
    # This is NOT a scalar! So the q-Casimir is NOT central for u_q(sl_2) at ℓ=3.
    
    # Actually, I think the issue is that the q-Casimir is central for U_q(sl_2) (the
    # infinite-dimensional quantum group) but NOT for u_q(sl_2) (the small quantum group).
    # The center of u_q(sl_2) is different.
    
    # For u_q(sl_2) at ℓ=3, the center is:
    # Z(u_q) = C[K^ℓ, E^ℓ, F^ℓ, C_q^ℓ] / (relations) = C[1, 0, 0, C_q^3] = C[C_q^3]
    # At ℓ=3: K^3=1, E^3=F^3=0, so the center is generated by C_q^3 (the ℓ-th power of the Casimir).
    
    # But C_q is not central, so C_q^3 might not be either. Let me check.
    # Actually, for the small quantum group, the center is generated by the image of the
    # center of U_q(sl_2) under the quotient map. The center of U_q(sl_2) is C[C_q].
    # The image of C[C_q] in u_q(sl_2) is C[C_q] / (C_q^ℓ - c_0^ℓ) where c_0 is the
    # eigenvalue on the trivial module.
    
    # This is getting too complicated. Let me just try the COMPUTATION.
    
    # The key insight: I should compute the Cartan matrix NUMERICALLY, not guess it.
    # I need to compute dim(e_i * A * e_j) for the correct primitive idempotents.
    # The idempotent lifting failed because of numerical issues with least-squares.
    
    # Let me try Newton's method for idempotent lifting:
    # Start with a lift a of the idempotent ē in A/J.
    # Iterate: a → 3a² - 2a³ (this converges to an idempotent).
    
    # The lift: for block j, the idempotent in A/J is the identity in M_{dim S(j)}(C)
    # and 0 elsewhere. I need to find a ∈ A such that ρ(a) = (0, ..., I, ..., 0).
    
    # The least-squares solution gives an approximate lift. Newton's method will refine it.
    
    print("\n  Computing Cartan matrix from character theory...")
    # From the character computation:
    # [A:S(0)] = [A:S(1)] = 6, [A:S(2)] = 3
    # [A:S(i)] = Σ_j dim S(j) * C_{ij}
    # 6 = C_{00} + 2*C_{01} + 3*C_{02}
    # 6 = C_{10} + 2*C_{11} + 3*C_{12}
    # 3 = C_{20} + 2*C_{21} + 3*C_{22}
    # C_{22} = 1 (Steinberg projective)
    # 3 = C_{20} + 2*C_{21} + 3 → C_{20} + 2*C_{21} = 0 → C_{20} = C_{21} = 0
    # Symmetric: C_{02} = C_{20} = 0, C_{12} = C_{21} = 0
    # 6 = C_{00} + 2*C_{01}
    # 6 = C_{01} + 2*C_{11}
    # With C_{01} = C_{10} = b:
    # C_{00} = 6 - 2b, C_{11} = (6-b)/2
    
    # For C_{11} integer: b even. b = 0, 2, 4, 6.
    # b=0: C = [[6,0,0],[0,3,0],[0,0,1]], HH² = 25
    # b=2: C = [[2,2,0],[2,2,0],[0,0,1]], HH² = 5
    # b=4: C = [[-2,...]], invalid
    # b=6: C = [[-6,...]], invalid
    
    # Neither gives HH²=3. But we KNOW HH²=3.
    
    # The resolution: the formula [A:S(i)] = Σ_j dim S(j) * C_{ij} might be wrong
    # for non-basic algebras. Let me check.
    
    # For A = ⊕_j P(j)^{⊕ dim S(j)} (as left modules):
    # [A : S(i)] = Σ_j dim S(j) * [P(j) : S(i)] = Σ_j dim S(j) * C_{ij}
    # This IS correct for any finite-dimensional algebra.
    
    # So the issue must be in the character computation. Let me verify.
    # χ_A(K) = trace of left-mult by K on A.
    # K * (K^a E^b F^c) = q^{2(b-c)} K^{a+1} E^b F^c (after moving K past E and F).
    # Wait, K * E^b = q^{2b} E^b K (from KE = q² EK).
    # So K * K^a E^b F^c = K * K^a * E^b * F^c = K^{a+1} * E^b * F^c.
    # But K * E^b = q^{2b} E^b * K, so K^{a+1} * E^b = q^{2b} E^b * K^{a+1}.
    # And K * F^c = q^{-2c} F^c * K, so E^b * K^{a+1} * F^c = ...
    # Actually, K * (K^a E^b F^c) is NOT simply K^{a+1} E^b F^c. The product
    # K * K^a E^b F^c = K^{a+1} E^b F^c (since K commutes with K^a). But the
    # NORMAL FORM of K^{a+1} E^b F^c may differ from K^{a+1} E^b F^c by q-phases.
    
    # Actually, in the PBW basis, K^{a+1} E^b F^c is already in normal form
    # (K's come first, then E's, then F's). So K * basis[a,b,c] = basis[(a+1)%3, b, c].
    
    # The trace of left-mult by K: Σ_{a,b,c} <basis[a,b,c], K * basis[a,b,c]>
    # = Σ_{a,b,c} <basis[a,b,c], basis[(a+1)%3, b, c]>
    # = Σ_{a,b,c} δ_{a, (a+1)%3} = 0 (since a ≠ (a+1)%3 for all a ∈ {0,1,2}).
    
    # So χ_A(K) = 0. ✓ (consistent with [A:S(0)] = [A:S(1)])
    
    # Let me also check χ_A(E):
    # E * K^a = q^{-2a} K^a E. So E * K^a E^b F^c = q^{-2a} K^a E^{b+1} F^c.
    # If b+1 < ℓ (= 3), this is a basis element. If b+1 = ℓ, E^ℓ = 0.
    # trace = Σ_{a,b,c} <basis[a,b,c], E * basis[a,b,c]>
    # = Σ_{a,b,c: b<2} q^{-2a} <basis[a,b,c], basis[a,b+1,c]>
    # = Σ_{a,b,c: b<2} q^{-2a} δ_{b,b+1} = 0 (since b ≠ b+1).
    
    # So χ_A(E) = 0. ✓
    
    # Now let me check χ_A(EF):
    # EF * K^a E^b F^c = E * (F * K^a) * E^b F^c = E * q^{2a} K^a F * E^b F^c
    # = q^{2a} E K^a F E^b F^c = q^{2a} q^{-2a} K^a E F E^b F^c = K^a (EF) E^b F^c
    # = K^a (FE + (K-K^{-1})/(q-q^{-1})) E^b F^c
    # = K^a F E^{b+1} F^c + K^a (K-K^{-1})/(q-q^{-1}) E^b F^c
    # 
    # This is getting complicated. Let me just compute it numerically.
    
    # Compute χ_A(g) = trace of left-mult by g on A
    def char_A(g_vector):
        """Character of the regular representation at g."""
        # Left-mult by g: L_g[i, j] = coeff of basis[i] in (g * basis[j])
        # = Σ_k g[k] * mult[i, k, j]
        L_g = np.zeros((DIM, DIM), dtype=complex)
        for k in range(DIM):
            if abs(g_vector[k]) > 1e-14:
                L_g += g_vector[k] * mult[:, k, :]
        return np.trace(L_g)
    
    # Compute characters of simples at K, K^2, E, F, EF, etc.
    K_vec = np.zeros(DIM, dtype=complex); K_vec[idx(1,0,0)] = 1
    K2_vec = np.zeros(DIM, dtype=complex); K2_vec[idx(2,0,0)] = 1
    E_vec = np.zeros(DIM, dtype=complex); E_vec[idx(0,1,0)] = 1
    F_vec = np.zeros(DIM, dtype=complex); F_vec[idx(0,0,1)] = 1
    
    print(f"\n  Characters of regular representation:")
    print(f"    χ_A(1) = {char_A(np.eye(DIM)[0])} (expected {DIM})")
    print(f"    χ_A(K) = {char_A(K_vec):.6f} (expected 0)")
    print(f"    χ_A(K²) = {char_A(K2_vec):.6f} (expected 0)")
    print(f"    χ_A(E) = {char_A(E_vec):.6f} (expected 0)")
    print(f"    χ_A(F) = {char_A(F_vec):.6f} (expected 0)")
    
    # Characters of simples at K:
    # χ_{S(j)}(K) = Σ_{k=0}^{j} q^{j-2k}
    for j in range(3):
        chi_K = sum(Q**(j-2*k) for k in range(j+1))
        print(f"    χ_S({j})(K) = {chi_K:.6f}")
    
    # From χ_A(K) = [A:S(0)]*χ_S(0)(K) + [A:S(1)]*χ_S(1)(K) + [A:S(2)]*χ_S(2)(K)
    # 0 = [A:S(0)]*1 + [A:S(1)]*(-1) + [A:S(2)]*0
    # [A:S(0)] = [A:S(1)] ✓
    
    # From χ_A(1) = [A:S(0)]*1 + [A:S(1)]*2 + [A:S(2)]*3 = 27
    # And [A:S(2)] = dim P(2) = 3 (Steinberg projective)
    # [A:S(0)] + 2*[A:S(0)] + 9 = 27 → 3*[A:S(0)] = 18 → [A:S(0)] = 6.
    # So [A:S(0)] = [A:S(1)] = 6, [A:S(2)] = 3. ✓
    
    # Now: [A:S(i)] = Σ_j dim S(j) * C_{ij}
    # 6 = C_{00} + 2*C_{01} + 3*C_{02}    ... (I)
    # 6 = C_{10} + 2*C_{11} + 3*C_{12}    ... (II)
    # 3 = C_{20} + 2*C_{21} + 3*C_{22}    ... (III)
    # C_{22} = 1 (Steinberg)
    # Symmetric: C_{ij} = C_{ji}
    
    # From (III): C_{20} + 2*C_{21} + 3 = 3 → C_{20} = C_{21} = 0
    # By symmetry: C_{02} = C_{20} = 0, C_{12} = C_{21} = 0
    # From (I): C_{00} + 2*C_{01} = 6
    # From (II): C_{01} + 2*C_{11} = 6
    # With b = C_{01}: C_{00} = 6-2b, C_{11} = (6-b)/2
    # b even: b=0 (C_{00}=6, C_{11}=3, HH²=25), b=2 (C_{00}=2, C_{11}=2, HH²=5)
    
    # Neither gives HH²=3. THE CHARACTER COMPUTATION IS CORRECT BUT THE ANSWER IS WRONG.
    
    # This means: the formula HH² = Σ_j (C_{j0}-δ_{0j})*(C_{0j}-δ_{0j}) is WRONG.
    
    # Let me re-examine. The issue might be that the minimal projective resolution
    # does NOT satisfy K_{n+1} = rad(P_n) for ALL n. This is only true when the
    # algebra is BASIC. For non-basic algebras, the minimal resolution is different.
    
    # For a non-basic algebra, the projective indecomposable modules P(j) have
    # dim P(j) > dim S(j) (in general), and the top of P(j) is S(j)^{⊕ dim S(j)}
    # (not just S(j)). So [top(P(j)) : S(i)] = dim S(j) * δ_{ij}, not δ_{ij}.
    
    # AH HA! THIS IS THE BUG!
    
    # For a non-basic algebra:
    # top(P(j)) = P(j)/rad(P(j)) = S(j)^{⊕ dim S(j)}
    # [top(P(j)) : S(i)] = dim S(j) * δ_{ij}
    
    # So [rad(P(j)) : S(i)] = [P(j) : S(i)] - dim S(j) * δ_{ij} = C_{ij} - dim S(j) * δ_{ij}
    
    # And:
    # [K_1 : S(j)] = [rad(P(0)) : S(j)] = C_{j0} - dim S(0) * δ_{j0} = C_{j0} - δ_{j0}
    #   (since dim S(0) = 1, this is the same as before)
    
    # P_1 = ⊕_j P(j)^{[K_1:S(j)] / dim S(j)}  (divide by dim S(j) because each P(j)
    # covers dim S(j) copies of S(j))
    
    # Wait, the projective cover of M is P → M where the top of P maps onto the top of M.
    # If M has top S(j)^{⊕ m_j}, then the projective cover is P(j)^{⊕ m_j}.
    
    # For K_1 = rad(P(0)): [top(K_1) : S(j)] = [K_1 : S(j)] - [rad(K_1) : S(j)]
    # But I need [top(K_1) : S(j)], not [K_1 : S(j)].
    
    # Actually, for the minimal projective resolution:
    # P_0 = P(0) (projective cover of k = S(0)^1, so 1 copy of P(0))
    # d_0: P(0) → S(0) (surjective, top of P(0) = S(0)^{dim S(0)} maps onto S(0))
    # K_1 = ker(d_0). [K_1 : S(j)] = [P(0) : S(j)] - [im(d_0) : S(j)]
    # im(d_0) = S(0) (the trivial module). [S(0) : S(j)] = δ_{0j}.
    # [K_1 : S(j)] = C_{j0} - δ_{0j}. ✓ (same as before)
    
    # But P_1 = projective cover of K_1. The top of K_1 is:
    # top(K_1) = K_1 / rad(K_1) = K_1 / J*K_1.
    # [top(K_1) : S(j)] = [K_1 : S(j)] - [rad(K_1) : S(j)]
    # rad(K_1) = J * K_1 (the radical of K_1 as a module).
    # [rad(K_1) : S(j)] = ??? (this is NOT simply [K_1 : S(j)] minus the top)
    
    # Actually, [top(K_1) : S(j)] = dim Hom_A(S(j), K_1) = dim Hom_A(S(j), K_1)
    # And Hom_A(S(j), M) = e_j * M (the e_j-isotypic component of the top of M).
    
    # For a module M, [top(M) : S(j)] = dim(e_j * M / e_j * J * M) = dim(e_j * M) - dim(e_j * J * M).
    
    # This is getting very involved. The key issue is that for non-basic algebras,
    # the formula P_1 = ⊕ P(j)^{[K_1:S(j)]} is WRONG. The correct formula is
    # P_1 = ⊕ P(j)^{[top(K_1):S(j)] / dim S(j)}.
    
    # So I need to compute [top(K_1) : S(j)], not [K_1 : S(j)].
    
    # For the minimal resolution:
    # dim Hom(P_n, S(0)) = number of copies of P(0) in P_n
    # = [top(K_n) : S(0)] / dim S(0) = [top(K_n) : S(0)] (since dim S(0) = 1)
    
    # And HH^n = [top(K_n) : S(0)] (for minimal resolution, im(d*) = 0).
    
    # The computation is:
    # [top(K_1) : S(j)] = [K_1 : S(j)] - [rad(K_1) : S(j)]
    # But [rad(K_1) : S(j)] = [J * K_1 : S(j)] = ???
    
    # This requires computing J * K_1 explicitly, which requires knowing J as a matrix.
    
    # OK, I think the correct approach is to compute everything numerically from the
    # multiplication table, without using the Cartan matrix formula. Let me do that.
    
    print("\n  === Direct computation of HH² ===")
    print("  (Computing projective resolution directly from multiplication table)")
    
    # Step 1: Compute J = rad(A) = ker(ρ) where ρ = (ρ_0, ρ_1, ρ_2)
    R = rho_combined(mult)
    U, s, Vh = np.linalg.svd(R, full_matrices=True)
    tol = max(R.shape) * s[0] * 1e-10
    rank = int(np.sum(s > tol))
    J_null = Vh[rank:].conj()  # basis for J as a subspace of A (dim_J × DIM)
    dim_J = DIM - rank
    print(f"  dim J = {dim_J}")
    
    # Step 2: Compute the projective indecomposable P(0) = A * e_0
    # We need the primitive idempotent e_0. Use Newton's method.
    
    # First, find a lift of the idempotent from A/J.
    # In A/J ≅ M_1 ⊕ M_2 ⊕ M_3, the idempotent for block 0 is (1, 0, 0).
    # Target: ρ(a) = (1, 0_{2×2}, 0_{3×3}) as a 14-vector.
    target_e0 = np.zeros(14, dtype=complex)
    target_e0[0] = 1.0  # identity in M_1 = C
    
    # Least-squares lift
    a, _, _, _ = np.linalg.lstsq(R, target_e0, rcond=None)
    print(f"  Initial lift: ||ρ(a) - target|| = {np.linalg.norm(R @ a - target_e0):.2e}")
    
    # Newton iteration: a → 3a² - 2a³ (converges to idempotent)
    for iteration in range(20):
        # Compute a² and a³ in A (using multiplication table)
        a_sq = np.zeros(DIM, dtype=complex)
        a_cu = np.zeros(DIM, dtype=complex)
        for k in range(DIM):
            if abs(a[k]) > 1e-14:
                # a * a = Σ_k a[k] * (basis[k] * a) = Σ_k a[k] * Σ_i a[i] * mult[:, k, i]
                for i in range(DIM):
                    if abs(a[i]) > 1e-14:
                        a_sq += a[k] * a[i] * mult[:, k, i]
        
        for k in range(DIM):
            if abs(a_sq[k]) > 1e-14:
                for i in range(DIM):
                    if abs(a[i]) > 1e-14:
                        a_cu += a_sq[k] * a[i] * mult[:, k, i]
        
        a_new = 3 * a_sq - 2 * a_cu
        diff = np.linalg.norm(a_new - a)
        a = a_new
        if diff < 1e-12:
            break
    
    # Check idempotency
    a_sq = np.zeros(DIM, dtype=complex)
    for k in range(DIM):
        if abs(a[k]) > 1e-14:
            for i in range(DIM):
                if abs(a[i]) > 1e-14:
                    a_sq += a[k] * a[i] * mult[:, k, i]
    idem_err = np.linalg.norm(a_sq - a)
    print(f"  After Newton: idempotency error = {idem_err:.2e}")
    print(f"  ρ(e_0) = {R @ a}")
    
    # P(0) = A * e_0 (right multiplication by e_0)
    R_e0 = np.zeros((DIM, DIM), dtype=complex)
    for l in range(DIM):
        if abs(a[l]) > 1e-14:
            R_e0 += a[l] * mult[:, :, l]
    s_P = np.linalg.svd(R_e0, compute_uv=False)
    dim_P0 = int(np.sum(s_P > 1e-10 * (s_P[0] if len(s_P) > 0 else 1)))
    print(f"  dim P(0) = {dim_P0}")
    
    # K_1 = rad(P(0)) = J * P(0)
    # Compute J * P(0): for each j in J_basis and p in P(0), compute j * p
    P0_basis = R_e0  # columns span P(0)
    # Project P0_basis to its column space
    U_P0, s_P0, _ = np.linalg.svd(P0_basis, full_matrices=False)
    rank_P0 = int(np.sum(s_P0 > 1e-10 * s_P0[0]))
    P0_cols = U_P0[:, :rank_P0]  # orthonormal basis for P(0) as subspace of C^DIM
    
    # J * P(0): multiply each J basis vector (in A) with each P(0) basis vector
    JP0_products = []
    for j_row in J_null:  # j_row is a vector in C^DIM (element of J)
        for col in range(rank_P0):
            p = P0_cols[:, col]
            # j * p = Σ_k j[k] * Σ_l p[l] * mult[:, k, l]
            prod = np.zeros(DIM, dtype=complex)
            for k in range(DIM):
                if abs(j_row[k]) > 1e-14:
                    prod += j_row[k] * (mult[:, k, :] @ p)
            if np.linalg.norm(prod) > 1e-14:
                JP0_products.append(prod)
    
    M_JP0 = np.array(JP0_products).T if JP0_products else np.zeros((DIM, 0))
    U_K1, s_K1, _ = np.linalg.svd(M_JP0, full_matrices=False)
    dim_K1 = int(np.sum(s_K1 > 1e-10 * (s_K1[0] if len(s_K1) > 0 else 1)))
    print(f"  dim K_1 = rad(P(0)) = {dim_K1}")
    
    # [K_1 : S(0)] = dim Hom_A(S(0), K_1) = dim(e_0 * K_1) (the S(0)-top multiplicity)
    # = dim(top(K_1) : S(0))
    # top(K_1) = K_1 / J*K_1. [top(K_1) : S(0)] = dim(e_0 * K_1) - dim(e_0 * J * K_1)
    
    # e_0 * K_1: left-multiply K_1 by e_0
    K1_cols = U_K1[:, :dim_K1]
    e0_K1 = np.zeros((DIM, dim_K1), dtype=complex)
    for col in range(dim_K1):
        k = K1_cols[:, col]
        # e_0 * k = Σ_i a[i] * (basis[i] * k) = Σ_i a[i] * (mult[:, i, :] @ k)
        result = np.zeros(DIM, dtype=complex)
        for i in range(DIM):
            if abs(a[i]) > 1e-14:
                result += a[i] * (mult[:, i, :] @ k)
        e0_K1[:, col] = result
    
    s_e0K1 = np.linalg.svd(e0_K1, compute_uv=False)
    dim_e0K1 = int(np.sum(s_e0K1 > 1e-10 * (s_e0K1[0] if len(s_e0K1) > 0 else 1)))
    
    # J * K_1: left-multiply K_1 by each J element
    JK1_products = []
    for j_row in J_null:
        for col in range(dim_K1):
            k = K1_cols[:, col]
            prod = np.zeros(DIM, dtype=complex)
            for i in range(DIM):
                if abs(j_row[i]) > 1e-14:
                    prod += j_row[i] * (mult[:, i, :] @ k)
            if np.linalg.norm(prod) > 1e-14:
                JK1_products.append(prod)
    
    M_JK1 = np.array(JK1_products).T if JK1_products else np.zeros((DIM, 0))
    
    # e_0 * J * K_1
    if M_JK1.shape[1] > 0:
        U_JK1, s_JK1, _ = np.linalg.svd(M_JK1, full_matrices=False)
        dim_JK1 = int(np.sum(s_JK1 > 1e-10 * s_JK1[0]))
        JK1_cols = U_JK1[:, :dim_JK1]
        
        e0_JK1 = np.zeros((DIM, dim_JK1), dtype=complex)
        for col in range(dim_JK1):
            k = JK1_cols[:, col]
            result = np.zeros(DIM, dtype=complex)
            for i in range(DIM):
                if abs(a[i]) > 1e-14:
                    result += a[i] * (mult[:, i, :] @ k)
            e0_JK1[:, col] = result
        
        s_e0JK1 = np.linalg.svd(e0_JK1, compute_uv=False)
        dim_e0JK1 = int(np.sum(s_e0JK1 > 1e-10 * (s_e0JK1[0] if len(s_e0JK1) > 0 else 1)))
    else:
        dim_e0JK1 = 0
    
    top_K1_S0 = dim_e0K1 - dim_e0JK1
    print(f"  [top(K_1) : S(0)] = {top_K1_S0}")
    print(f"  HH^1 = {top_K1_S0}")
    
    # For HH²: need [top(K_2) : S(0)] where K_2 = rad(P_1)
    # P_1 = projective cover of K_1 = ⊕_j P(j)^{[top(K_1):S(j)]/dim S(j)}
    # This requires computing [top(K_1) : S(j)] for all j, which requires all idempotents.
    
    # For now, just report HH^1
    print(f"\n  HH^0 = 1")
    print(f"  HH^1 = {top_K1_S0}")
    print(f"  (HH^2 requires computing top(K_2), which needs all idempotents)")
    
    return top_K1_S0

if __name__ == "__main__":
    mult = build_mult()
    main()
