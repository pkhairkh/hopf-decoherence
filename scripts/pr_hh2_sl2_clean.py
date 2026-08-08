#!/usr/bin/env python3
"""Compute dim HH²(u_q(sl_2), C) at ℓ=3 via minimal projective resolution.

Uses: J = rad(A) = ker(ρ), Newton idempotent lifting, and direct
radical computation. No Gröbner bases, no letterplace, no Anick.
"""
import sys, cmath, math, time
import numpy as np
sys.path.insert(0, '/home/z/my-project/hopf-decoherence/scripts')

ELL=3; Q=cmath.exp(2j*math.pi/ELL); QI=Q**(-1); DIM=ELL**3
def idx(a,b,c): return a*9+b*3+c
def from_idx(i): return i//9,(i//3)%3,i%3

def build_mult():
    from verify_sl2_hh2 import build_multiplication_table
    return build_multiplication_table()

def K_mat(j):
    return [np.array([[1.0]]), np.diag([Q,QI]), np.diag([Q**2,1.0,Q**(-2)])][j]
def E_mat(j):
    return [np.array([[0.0]]), np.array([[0,1],[0,0]],dtype=complex),
            np.array([[0,1,0],[0,0,1],[0,0,0]],dtype=complex)][j]
def F_mat(j):
    return [np.array([[0.0]]), np.array([[0,0],[1,0]],dtype=complex),
            np.array([[0,0,0],[1,0,0],[0,1,0]],dtype=complex)][j]

def rho_vec(a,b,c):
    """ρ(K^a E^b F^c) as flattened vector in C^14."""
    col=[]
    for j in range(3):
        d=j+1; K=K_mat(j);E=E_mat(j);F=F_mat(j)
        m=np.eye(d,dtype=complex)@np.linalg.matrix_power(K,a)
        if b>0: m=m@np.linalg.matrix_power(E,b)
        if c>0: m=m@np.linalg.matrix_power(F,c)
        col.extend(m.flatten())
    return np.array(col)

def left_mult(mult, v):
    """Left-multiply by vector v: returns matrix L where L[:,j] = v * basis[j]."""
    L=np.zeros((DIM,DIM),dtype=complex)
    for k in range(DIM):
        if abs(v[k])>1e-14:
            L+=v[k]*mult[:,k,:]
    return L

def right_mult(mult, v):
    """Right-multiply by vector v: returns matrix R where R[:,j] = basis[j] * v."""
    R=np.zeros((DIM,DIM),dtype=complex)
    for l in range(DIM):
        if abs(v[l])>1e-14:
            R+=v[l]*mult[:,:,l]
    return R

def mult_vec(mult, v1, v2):
    """Multiply two vectors in A: v1 * v2."""
    return left_mult(mult, v1) @ v2

def rank_mat(M, tol=1e-10):
    s=np.linalg.svd(M,compute_uv=False)
    if len(s)==0 or s[0]==0: return 0
    return int(np.sum(s>tol*s[0]))

def col_space(M, tol=1e-10):
    """Return orthonormal basis for column space of M."""
    U,s,Vh=np.linalg.svd(M,full_matrices=False)
    r=int(np.sum(s>tol*(s[0] if len(s)>0 else 1)))
    return U[:,:r]

def main():
    print(f"=== HH²(u_q(sl_2), C) at ℓ={ELL} via projective resolution ===")
    t0=time.time()
    mult=build_mult()
    print(f"Mult table: {time.time()-t0:.2f}s")

    # Step 1: Build combined representation ρ: A → C^14
    print("\nStep 1: Compute J = rad(A) = ker(ρ)")
    R=np.zeros((14,DIM),dtype=complex)
    for m in range(DIM):
        a,b,c=from_idx(m)
        R[:,m]=rho_vec(a,b,c)
    rank_R=rank_mat(R)
    dim_J=DIM-rank_R
    U,s,Vh=np.linalg.svd(R,full_matrices=True)
    J_basis=Vh[rank_R:].conj()  # (dim_J, DIM)
    print(f"  rank(ρ)={rank_R}, dim J={dim_J}")

    # Step 2: Lift primitive idempotent e_0 via Newton's method
    print("\nStep 2: Lift idempotent e_0 (Newton's method)")
    target=np.zeros(14,dtype=complex); target[0]=1.0  # (1, 0, 0) in M_1⊕M_2⊕M_3
    a,_,_,_=np.linalg.lstsq(R,target,rcond=None)
    for it in range(30):
        a_sq=mult_vec(mult,a,a)
        a_cu=mult_vec(mult,a_sq,a)
        a_new=3*a_sq-2*a_cu
        diff=np.linalg.norm(a_new-a)
        a=a_new
        if diff<1e-14: break
    a_sq=mult_vec(mult,a,a)
    print(f"  idempotency error: {np.linalg.norm(a_sq-a):.2e}")
    print(f"  ρ(e_0) residual: {np.linalg.norm(R@a-target):.2e}")

    # Step 3: Compute P(0) = A * e_0
    print("\nStep 3: Compute P(0) = A * e_0")
    R_e0=right_mult(mult,a)
    P0=col_space(R_e0)
    dim_P0=P0.shape[1]
    print(f"  dim P(0) = {dim_P0}")

    # Step 4: Compute K_1 = rad(P(0)) = J * P(0)
    print("\nStep 4: Compute K_1 = J * P(0)")
    prods=[]
    for j in range(J_basis.shape[0]):
        for col in range(dim_P0):
            p=P0[:,col]
            prod=left_mult(mult,J_basis[j])@p
            if np.linalg.norm(prod)>1e-14:
                prods.append(prod)
    K1=col_space(np.array(prods).T) if prods else np.zeros((DIM,0))
    dim_K1=K1.shape[1]
    print(f"  dim K_1 = {dim_K1}")

    # Step 5: Compute [top(K_1) : S(0)] = dim(e_0 * K_1) - dim(e_0 * J * K_1)
    print("\nStep 5: Compute HH^1 = [top(K_1) : S(0)]")
    # e_0 * K_1
    e0_K1=np.zeros((DIM,dim_K1),dtype=complex)
    L_e0=left_mult(mult,a)
    for col in range(dim_K1):
        e0_K1[:,col]=L_e0@K1[:,col]
    dim_e0K1=rank_mat(e0_K1)
    print(f"  dim(e_0 * K_1) = {dim_e0K1}")

    # J * K_1
    prods2=[]
    for j in range(J_basis.shape[0]):
        for col in range(dim_K1):
            prod=left_mult(mult,J_basis[j])@K1[:,col]
            if np.linalg.norm(prod)>1e-14:
                prods2.append(prod)
    JK1=col_space(np.array(prods2).T) if prods2 else np.zeros((DIM,0))
    dim_JK1=JK1.shape[1]

    # e_0 * J * K_1
    e0_JK1=np.zeros((DIM,dim_JK1),dtype=complex)
    for col in range(dim_JK1):
        e0_JK1[:,col]=L_e0@JK1[:,col]
    dim_e0JK1=rank_mat(e0_JK1)
    print(f"  dim(e_0 * J * K_1) = {dim_e0JK1}")

    hh1=dim_e0K1-dim_e0JK1
    print(f"  HH^1 = {hh1}")

    # Step 6: For HH², need [top(K_2) : S(0)] where K_2 = rad(P_1)
    # P_1 = projective cover of K_1
    # Need [top(K_1) : S(j)] for ALL j to build P_1
    # Lift e_1 and e_2 similarly
    print("\nStep 6: Lift e_1, e_2 and compute HH²")
    idems=[a]  # e_0 already lifted
    
    for blk in [1, 2]:
        d=blk+1  # dim S(blk)
        tgt=np.zeros(14,dtype=complex)
        off=sum((i+1)**2 for i in range(blk))  # offset in C^14
        for k in range(d):
            tgt[off+k*d+k]=1.0  # identity in M_d
        e,_,_,_=np.linalg.lstsq(R,tgt,rcond=None)
        for it in range(30):
            e_sq=mult_vec(mult,e,e)
            e_cu=mult_vec(mult,e_sq,e)
            e_new=3*e_sq-2*e_cu
            if np.linalg.norm(e_new-e)<1e-14: break
            e=e_new
        idems.append(e)
        e_sq=mult_vec(mult,e,e)
        print(f"  e_{blk}: idempotency={np.linalg.norm(e_sq-e):.2e}, ρ residual={np.linalg.norm(R@e-tgt):.2e}")

    # Verify orthogonality
    for i in range(3):
        for j in range(3):
            prod=mult_vec(mult,idems[i],idems[j])
            expected=idems[j] if i==j else np.zeros(DIM)
            err=np.linalg.norm(prod-expected)
            if err>1e-8:
                print(f"  WARNING: e_{i}*e_{j} err={err:.2e}")

    # Compute [top(K_1) : S(j)] for all j
    print("\n  Computing [top(K_1) : S(j)] for all j:")
    tops_K1=[]
    for j in range(3):
        L_ej=left_mult(mult,idems[j])
        ej_K1=L_ej@K1  # e_j * K_1
        dim_ejK1=rank_mat(ej_K1)
        ej_JK1=L_ej@JK1  # e_j * J * K_1
        dim_ejJK1=rank_mat(ej_JK1)
        top_j=dim_ejK1-dim_ejJK1
        tops_K1.append(top_j)
        print(f"    [top(K_1) : S({j})] = {top_j}")

    # P_1 = ⊕_j P(j)^{tops_K1[j] / dim S(j)}
    # dim Hom(P_1, S(0)) = tops_K1[0] / dim S(0) = tops_K1[0] (since dim S(0)=1)
    # But we also need tops_K1[j] / dim S(j) to be integer!
    dim_S=[1,2,3]
    P1_mults=[tops_K1[j]//dim_S[j] for j in range(3)]
    print(f"\n  P_1 = P(0)^{P1_mults[0]} ⊕ P(1)^{P1_mults[1]} ⊕ P(2)^{P1_mults[2]}")

    # K_2 = rad(P_1) = ⊕_j rad(P(j))^{P1_mults[j]}
    # [top(K_2) : S(0)] = Σ_j P1_mults[j] * [top(rad(P(j))) : S(0)]
    # [top(rad(P(j))) : S(0)] = [rad(P(j)) : S(0)] - [rad²(P(j)) : S(0)]
    # This requires computing rad(P(j)) and rad²(P(j)) for each j.

    # Simpler: compute K_2 directly.
    # P_1 is the direct sum of P(j)^{P1_mults[j]}.
    # rad(P_1) = ⊕_j J*P(j)^{P1_mults[j]}.
    # K_2 = rad(P_1) = J * P_1.

    # Build P_1 as a subspace: ⊕_j P(j)^{P1_mults[j]}
    P1_cols=[]
    for j in range(3):
        R_ej=right_mult(mult,idems[j])
        Pj=col_space(R_ej)
        for copy in range(P1_mults[j]):
            for col in range(Pj.shape[1]):
                P1_cols.append(Pj[:,col])
    P1=np.array(P1_cols).T if P1_cols else np.zeros((DIM,0))
    dim_P1=P1.shape[1]
    print(f"  dim P_1 = {dim_P1}")

    # K_2 = J * P_1
    prods3=[]
    for j in range(J_basis.shape[0]):
        L_j=left_mult(mult,J_basis[j])
        for col in range(dim_P1):
            prod=L_j@P1[:,col]
            if np.linalg.norm(prod)>1e-14:
                prods3.append(prod)
    K2=col_space(np.array(prods3).T) if prods3 else np.zeros((DIM,0))
    dim_K2=K2.shape[1]
    print(f"  dim K_2 = rad(P_1) = {dim_K2}")

    # [top(K_2) : S(0)] = dim(e_0 * K_2) - dim(e_0 * J * K_2)
    e0_K2=L_e0@K2  # reuse L_e0 from step 5
    dim_e0K2=rank_mat(e0_K2)

    # J * K_2
    prods4=[]
    for j in range(J_basis.shape[0]):
        L_j=left_mult(mult,J_basis[j])
        for col in range(dim_K2):
            prod=L_j@K2[:,col]
            if np.linalg.norm(prod)>1e-14:
                prods4.append(prod)
    JK2=col_space(np.array(prods4).T) if prods4 else np.zeros((DIM,0))
    e0_JK2=L_e0@JK2
    dim_e0JK2=rank_mat(e0_JK2)

    hh2=dim_e0K2-dim_e0JK2
    print(f"\n  dim(e_0 * K_2) = {dim_e0K2}")
    print(f"  dim(e_0 * J * K_2) = {dim_e0JK2}")
    print(f"  HH^2 = [top(K_2) : S(0)] = {hh2}")

    print(f"\n=== RESULT ===")
    print(f"HH^0 = 1")
    print(f"HH^1 = {hh1}")
    print(f"HH^2 = {hh2}")
    print(f"Expected: HH^0=1, HH^1=0, HH^2=3")
    print(f"MATCH HH^2: {'✓' if hh2==3 else '✗'}")

if __name__=="__main__":
    main()
