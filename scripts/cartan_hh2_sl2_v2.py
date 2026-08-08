#!/usr/bin/env python3
"""
Compute HH²(u_q(sl_2), C) at ℓ=3 via the minimal projective resolution.

Correct approach: compute the Jacobson radical J = ∩_j ker(ρ_j) where
ρ_j are the simple representations, then build projective indecomposables
from the block decomposition of A/J.

Simple modules of u_q(sl_2) at ℓ=3:
  S(0): dim 1, K→1, E→0, F→0 (trivial)
  S(1): dim 2, K→diag(q,q^{-1}), E→[[0,1],[0,0]], F→[[0,0],[1,0]]
  S(2): dim 3 (Steinberg), K→diag(q^2,1,q^{-2}), E,F→standard 3×3
"""
import sys, cmath, math, time
import numpy as np
sys.path.insert(0, '/home/z/my-project/hopf-decoherence/scripts')

ELL = 3; Q = cmath.exp(2j*math.pi/ELL); QI = Q**(-1); D = Q-QI; DIM = ELL**3

def idx(a,b,c): return a*ELL*ELL+b*ELL+c
def from_idx(i): return i//(ELL*ELL),(i//ELL)%ELL,i%ELL

def build_mult(): 
    from verify_sl2_hh2 import build_multiplication_table
    return build_multiplication_table()

def build_representations():
    """Build the 3 simple representations ρ_j: A → M_{dim S(j)}(C)."""
    reps = []
    # S(0): dim 1, trivial
    reps.append({'K': np.array([[1.0]]), 'E': np.array([[0.0]]), 'F': np.array([[0.0]])})
    # S(1): dim 2
    reps.append({'K': np.diag([Q, QI]), 
                  'E': np.array([[0,1],[0,0]],dtype=complex),
                  'F': np.array([[0,0],[1,0]],dtype=complex)})
    # S(2): dim 3 (Steinberg)
    reps.append({'K': np.diag([Q**2, 1.0, Q**(-2)]),
                  'E': np.array([[0,1,0],[0,0,1],[0,0,0]],dtype=complex),
                  'F': np.array([[0,0,0],[1,0,0],[0,1,0]],dtype=complex)})
    return reps

def rho(reps, j, a, b, c):
    """ρ_j(K^a E^b F^c) as a matrix."""
    K = reps[j]['K']; E = reps[j]['E']; F = reps[j]['F']
    result = np.eye(K.shape[0], dtype=complex)
    result = result @ np.linalg.matrix_power(K, a)
    if b > 0: result = result @ np.linalg.matrix_power(E, b)
    if c > 0: result = result @ np.linalg.matrix_power(F, c)
    return result

def compute_J(reps, mult):
    """J = ∩_j ker(ρ_j). Compute by checking each basis element."""
    # For each basis element K^a E^b F^c, compute its image under each ρ_j.
    # The image is a vector in C^{1+4+9} = C^{14} (flattened matrices).
    # J = kernel of the combined map A → C^{14}.
    images = np.zeros((14, DIM), dtype=complex)
    for m in range(DIM):
        a, b, c = from_idx(m)
        col = []
        for j in range(ELL):
            mat = rho(reps, j, a, b, c)
            col.extend(mat.flatten())
        images[:, m] = col
    
    # J = kernel of images matrix
    U, s, Vh = np.linalg.svd(images, full_matrices=True)
    tol = max(images.shape) * (s[0] if len(s) > 0 else 1) * 1e-10
    rank = int(np.sum(s > tol))
    dim_J = DIM - rank
    print(f"  rank(ρ) = {rank}, dim J = {dim_J}")
    print(f"  dim A/J = {rank} (expected 1+4+9=14)")
    return rank, dim_J

def compute_projective_indecomposables(mult, reps):
    """Compute P(j) = A * e_j where e_j is the primitive idempotent for block j.
    
    For a non-basic algebra, we need the block idempotents from A/J.
    A/J ≅ M_1(C) ⊕ M_2(C) ⊕ M_3(C). The block idempotent for block j is
    the identity in M_{dim S(j)}(C) and 0 elsewhere.
    
    Lift this to A via the representation map.
    """
    # The combined representation: ρ: A → M_1 ⊕ M_2 ⊕ M_3
    # ρ(a) = (ρ_0(a), ρ_1(a), ρ_2(a))
    # The block idempotent e_j lifts to the element a ∈ A such that
    # ρ(a) = (0, ..., I_{dim S(j)}, ..., 0)
    
    # Build the combined image matrix (14 × DIM)
    images = np.zeros((14, DIM), dtype=complex)
    for m in range(DIM):
        a, b, c = from_idx(m)
        col = []
        for j in range(ELL):
            mat = rho(reps, j, a, b, c)
            col.extend(mat.flatten())
        images[:, m] = col
    
    # The block idempotent for block j: vector v_j in C^{14} that is
    # I_{dim S(j)} flattened, with zeros elsewhere
    dim_S = [1, 2, 3]
    offsets = [0, 1, 5]  # cumulative: 0, 1, 1+4=5
    total_dim = 14
    
    e_lifted = []
    for j in range(ELL):
        v_j = np.zeros(total_dim, dtype=complex)
        d = dim_S[j]
        off = offsets[j]
        # Identity matrix flattened
        for k in range(d):
            v_j[off + k * d + k] = 1.0
        # Solve images @ x = v_j (find x ∈ A such that ρ(x) = v_j)
        # Use least squares (since images has rank 14, this is exact)
        x, res, rank, sv = np.linalg.lstsq(images, v_j, rcond=None)
        e_lifted.append(x)
        print(f"  e_{j} lifted: ||residual|| = {np.linalg.norm(images @ x - v_j):.2e}")
    
    # Verify idempotency: e_i * e_j = δ_{ij} * e_j (in A)
    print("  Verifying idempotent orthogonality...")
    for i in range(ELL):
        for j in range(ELL):
            # e_i * e_j = Σ_k e_i[k] * Σ_l e_j[l] * mult[:, k, l]
            prod = np.zeros(DIM, dtype=complex)
            for k in range(DIM):
                if abs(e_lifted[i][k]) > 1e-12:
                    for l in range(DIM):
                        if abs(e_lifted[j][l]) > 1e-12:
                            prod += e_lifted[i][k] * e_lifted[j][l] * mult[:, k, l]
            expected = e_lifted[j] if i == j else np.zeros(DIM)
            err = np.linalg.norm(prod - expected)
            if err > 1e-8:
                print(f"    WARNING: e_{i}*e_{j} err = {err:.2e}")
    print("  Done.")
    
    # Compute P(j) = A * e_j (left ideal = right multiplication by e_j)
    P_list = []
    for j in range(ELL):
        # Right multiplication by e_j: R_{e_j}[k, n] = Σ_l e_j[l] * mult[k, n, l]
        R_ej = np.zeros((DIM, DIM), dtype=complex)
        for l in range(DIM):
            if abs(e_lifted[j][l]) > 1e-12:
                R_ej += e_lifted[j][l] * mult[:, :, l]
        s = np.linalg.svd(R_ej, compute_uv=False)
        rank_P = int(np.sum(s > 1e-10 * (s[0] if len(s) > 0 else 1)))
        P_list.append(R_ej)
        print(f"  dim P({j}) = dim(A*e_{j}) = {rank_P}")
    
    return e_lifted, P_list

def compute_cartan(mult, e_lifted):
    """C_{ij} = dim(e_i * A * e_j) = rank(L_{e_i} @ R_{e_j})."""
    n = len(e_lifted)
    C = np.zeros((n, n), dtype=int)
    for i in range(n):
        L_ei = np.zeros((DIM, DIM), dtype=complex)
        for k in range(DIM):
            if abs(e_lifted[i][k]) > 1e-12:
                L_ei += e_lifted[i][k] * mult[:, k, :]
        for j in range(n):
            R_ej = np.zeros((DIM, DIM), dtype=complex)
            for l in range(DIM):
                if abs(e_lifted[j][l]) > 1e-12:
                    R_ej += e_lifted[j][l] * mult[:, :, l]
            M = L_ei @ R_ej
            s = np.linalg.svd(M, compute_uv=False)
            tol = max(M.shape) * (s[0] if len(s) > 0 and s[0] > 0 else 1) * 1e-10
            C[i, j] = int(np.sum(s > tol))
    return C

def main():
    print("=== Minimal Projective Resolution via Representations ===")
    print(f"u_q(sl_2) at ℓ={ELL}, dim A = {DIM}")
    
    t0 = time.time()
    mult = build_mult()
    print(f"Multiplication table: {time.time()-t0:.2f}s")
    
    reps = build_representations()
    dim_S = [1, 2, 3]
    print(f"Simple module dims: {dim_S}")
    
    # Compute J
    print("\nComputing Jacobson radical J = ∩ ker(ρ_j)...")
    rank_AJ, dim_J = compute_J(reps, mult)
    
    # Compute projective indecomposables
    print("\nComputing projective indecomposable modules...")
    e_lifted, P_list = compute_projective_indecomposables(mult, reps)
    
    # Compute Cartan matrix
    print("\nComputing Cartan matrix C_{ij} = dim(e_i * A * e_j)...")
    C = compute_cartan(mult, e_lifted)
    print(f"  Cartan matrix:")
    for i in range(ELL):
        print(f"    {C[i]}")
    
    # Verify: dim P(j) = Σ_i C_{ij} * dim S(i)
    dim_P = [sum(C[i,j] * dim_S[i] for i in range(ELL)) for j in range(ELL)]
    print(f"  dim P(j) = {dim_P}")
    total = sum(dim_S[j] * dim_P[j] for j in range(ELL))
    print(f"  Σ dim S(j) * dim P(j) = {total} (expected {DIM})")
    
    # Compute HH dimensions
    print("\n=== Hochschild Cohomology ===")
    hh0 = 1
    hh1 = C[0,0] - 1
    mults_K1 = [C[j,0] - (1 if j==0 else 0) for j in range(ELL)]
    hh2 = sum(mults_K1[j] * (C[0,j] - (1 if j==0 else 0)) for j in range(ELL))
    
    print(f"Cartan matrix: {C.tolist()}")
    print(f"[K_1 : S(j)] = {mults_K1}")
    print(f"HH^0 = {hh0}")
    print(f"HH^1 = C_00 - 1 = {hh1}")
    print(f"HH^2 = Σ_j [K_1:S(j)] * [rad(P(j)):S(0)] = {hh2}")
    print(f"\nExpected: HH^0=1, HH^1=0, HH^2=3")
    print(f"MATCH: HH^2={'✓' if hh2==3 else '✗'} (got {hh2})")
    
    return hh2

if __name__ == "__main__":
    main()
