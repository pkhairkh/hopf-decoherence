#!/usr/bin/env python3
"""
Minimal projective resolution of k over A = u_q(sl_2) at ℓ=3.

Uses the Cartan matrix C_{ij} = dim(e_i * A * e_j) to compute
composition factor multiplicities, then builds the minimal resolution.

Key formulas (for minimal projective resolution of k = S(0)):
  HH^0 = 1
  HH^1 = C_{00} - 1  (number of S(0) in rad(P(0)))
  HH^2 = Σ_j (C_{j0} - δ_{0j}) * (C_{0j} - δ_{0j})
         (number of S(0) in rad(P_1), where P_1 covers rad(P(0)))
"""
import sys, cmath, math, time
import numpy as np
sys.path.insert(0, '/home/z/my-project/hopf-decoherence/scripts')

ELL = 3
Q = cmath.exp(2j * math.pi / ELL)
DIM = ELL ** 3

def idx(a, b, c): return a * ELL * ELL + b * ELL + c
def from_idx(i): return i // (ELL*ELL), (i // ELL) % ELL, i % ELL

def build_mult_table():
    from verify_sl2_hh2 import build_multiplication_table
    return build_multiplication_table()

def compute_idempotents():
    """e_j = (1/ℓ) Σ_{a=0}^{ℓ-1} q^{-ja} K^a."""
    e = []
    for j in range(ELL):
        v = np.zeros(DIM, dtype=complex)
        for a in range(ELL):
            v[idx(a, 0, 0)] += Q ** (-j * a) / ELL
        e.append(v)
    return e

def compute_cartan_matrix(mult, idempotents):
    """C_{ij} = dim(e_i * A * e_j) = rank(L_{e_i} @ R_{e_j}).
    
    L_{e_i}[m,n] = Σ_k e_i[k] * mult[m, k, n]  (left mult by e_i)
    R_{e_j}[m,n] = Σ_l e_j[l] * mult[m, n, l]  (right mult by e_j)
    M_{ij} = L_{e_i} @ R_{e_j}  (DIM × DIM matrix)
    C_{ij} = rank(M_{ij})
    """
    n = len(idempotents)
    C = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        # Build L_{e_i}: left multiplication by e_i
        L_ei = np.zeros((DIM, DIM), dtype=complex)
        for k in range(DIM):
            if abs(idempotents[i][k]) > 1e-14:
                # L_g[k][m, n] = mult[m, k, n]
                L_ei += idempotents[i][k] * mult[:, k, :]
        
        for j in range(n):
            # Build R_{e_j}: right multiplication by e_j
            R_ej = np.zeros((DIM, DIM), dtype=complex)
            for l in range(DIM):
                if abs(idempotents[j][l]) > 1e-14:
                    # R_g[l][m, n] = mult[m, n, l]
                    R_ej += idempotents[j][l] * mult[:, :, l]
            
            # M = L_{e_i} @ R_{e_j}
            M = L_ei @ R_ej
            # Rank of M = dim(e_i * A * e_j)
            s = np.linalg.svd(M, compute_uv=False)
            tol = max(M.shape) * (s[0] if len(s) > 0 and s[0] > 0 else 1) * 1e-10
            C[i, j] = int(np.sum(s > tol))
    
    return C

def main():
    print("=== Cartan Matrix + Minimal Projective Resolution ===")
    print(f"u_q(sl_2) at ℓ={ELL}, dim A = {DIM}")
    
    t0 = time.time()
    print("Building multiplication table...")
    mult = build_mult_table()
    print(f"  {time.time()-t0:.2f}s")
    
    print("Computing idempotents...")
    idem = compute_idempotents()
    for j in range(ELL):
        print(f"  e_{j}: {np.sum(np.abs(idem[j])>1e-14)} nonzero, ||e_j||={np.linalg.norm(idem[j]):.4f}")
    
    # Verify orthogonality
    for i in range(ELL):
        for j in range(ELL):
            prod = np.zeros(DIM, dtype=complex)
            for k in range(DIM):
                if abs(idem[i][k]) > 1e-14:
                    for l in range(DIM):
                        if abs(idem[j][l]) > 1e-14:
                            prod += idem[i][k] * idem[j][l] * mult[:, k, l]
            expected = idem[j] if i == j else 0
            err = np.linalg.norm(prod - expected)
            if err > 1e-10:
                print(f"  WARNING: e_{i}*e_{j} err = {err:.2e}")
    print("  Idempotents verified.")
    
    # Compute dim(A * e_j) = dim of left ideal
    print("\nProjective indecomposable dimensions:")
    for j in range(ELL):
        R_ej = np.zeros((DIM, DIM), dtype=complex)
        for l in range(DIM):
            if abs(idem[j][l]) > 1e-14:
                R_ej += idem[j][l] * mult[:, :, l]
        s = np.linalg.svd(R_ej, compute_uv=False)
        rank = int(np.sum(s > 1e-10 * (s[0] if len(s) > 0 else 1)))
        print(f"  dim(A * e_{j}) = {rank}")
    
    # Compute Cartan matrix
    print("\nComputing Cartan matrix C_{ij} = dim(e_i * A * e_j)...")
    t0 = time.time()
    C = compute_cartan_matrix(mult, idem)
    print(f"  {time.time()-t0:.2f}s")
    print(f"  Cartan matrix:")
    for i in range(ELL):
        print(f"    {C[i]}")
    
    # Known simple dimensions: dim S(j) = j+1 for u_q(sl_2)
    dim_S = [j + 1 for j in range(ELL)]
    print(f"\n  dim S(j) = {dim_S}")
    
    # Verify: dim(A * e_j) = Σ_i C_{ij} * dim P(i)
    # But we don't know dim P(i) yet. Instead, verify:
    # dim A = Σ_j Σ_i C_{ij} * dim S(i) (since dim(A*e_j) = Σ_i C_{ij} * dim P(i)
    # and dim P(i) = Σ_k C_{ki} * dim S(k), so
    # dim(A*e_j) = Σ_i C_{ij} * Σ_k C_{ki} * dim S(k) = Σ_k dim S(k) * (C^T @ C)_{kj}
    # Actually this is getting circular. Let me just compute dim P(j) from the Cartan matrix.
    
    # dim P(j) = Σ_i C_{ij} * dim S(i)
    dim_P = [sum(C[i, j] * dim_S[i] for i in range(ELL)) for j in range(ELL)]
    print(f"  dim P(j) = {dim_P}")
    
    # Verify: Σ_j dim S(j) * dim P(j) = dim A
    total = sum(dim_S[j] * dim_P[j] for j in range(ELL))
    print(f"  Σ dim S(j) * dim P(j) = {total} (expected {DIM})")
    
    # Also verify: Σ_j dim(A*e_j) = dim A (since A = ⊕ A*e_j)
    # dim(A*e_j) = Σ_i C_{ij} * dim P(i)
    dim_Ae = [sum(C[i, j] * dim_P[i] for i in range(ELL)) for j in range(ELL)]
    print(f"  dim(A*e_j) = {dim_Ae}, sum = {sum(dim_Ae)} (expected {DIM})")
    
    # Compute HH dimensions
    print("\n=== Hochschild Cohomology ===")
    
    # HH^0 = 1 (always, for connected algebra)
    hh0 = 1
    print(f"HH^0 = {hh0}")
    
    # HH^1 = C_{00} - 1 (number of S(0) in rad(P(0)))
    hh1 = C[0, 0] - 1
    print(f"HH^1 = C_00 - 1 = {C[0,0]} - 1 = {hh1}")
    
    # HH^2 = Σ_j (C_{j0} - δ_{0j}) * (C_{0j} - δ_{0j})
    # [K_1 : S(j)] = C_{j0} - δ_{0j}  (composition factors of rad(P(0)))
    # [K_2 : S(0)] = Σ_j [K_1 : S(j)] * (C_{0j} - δ_{0j})
    #              = Σ_j (C_{j0} - δ_{0j}) * (C_{0j} - δ_{0j})
    mults_K1 = [C[j, 0] - (1 if j == 0 else 0) for j in range(ELL)]
    print(f"\n  [K_1 : S(j)] = {mults_K1}")
    
    hh2 = sum(mults_K1[j] * (C[0, j] - (1 if j == 0 else 0)) for j in range(ELL))
    print(f"HH^2 = Σ_j [K_1:S(j)] * [rad(P(j)):S(0)] = {hh2}")
    
    # For symmetric Cartan matrix: HH^2 = Σ_j (C_{j0} - δ_{0j})^2
    if np.allclose(C, C.T):
        hh2_sym = sum((C[j, 0] - (1 if j == 0 else 0))**2 for j in range(ELL))
        print(f"  (symmetric check: Σ (C_j0 - δ)² = {hh2_sym})")
    
    print(f"\n=== RESULT ===")
    print(f"HH^0 = {hh0}")
    print(f"HH^1 = {hh1}")
    print(f"HH^2 = {hh2}")
    print(f"Expected: HH^0=1, HH^1=0, HH^2=3")
    print(f"MATCH HH^2: {hh2 == 3}")
    
    return hh2

if __name__ == "__main__":
    result = main()
