#!/usr/bin/env python3
"""
Compute δ(Θ) ∈ HH²(D(B⁺(sl_2)), C) at ℓ=3 explicitly using MW formula (3.4.4).

The Mastnak-Witherspoon connecting homomorphism
    δ: H̃¹_b(B⁺) → HH²(D(B⁺), C)
at the chain level for a bialgebra 1-cocycle h: B → B (identified with h̄ ∈ (X ⊗ B)^*) is:

    δ(h) = h̄ ∂^X_2 ∂^B_0 φ_c̃

where for n=1, i=1:
  - φ_c̃: D(B)^2 → X^2 ⊗ B^2 is the isomorphism (MW Thm 3.2.1)
  - ∂^B_0: X^2 ⊗ B^2 → X^2 ⊗ B^1
  - ∂^X_2: X^2 ⊗ B^1 → X^1 ⊗ B^1
  - h̄: X^1 ⊗ B^1 → C

We compute δ(Θ) for sl_2 at ℓ=3 with Θ(E) = 2E, Θ(K) = 0.

D(B⁺(sl_2)) at ℓ=3 has dimension 9×9 = 81 (since dim B⁺ = 9 and D = X ⊗ B as coalgebra).
Note: D(B⁺) ≠ u_q(sl_2) — the latter has dim 27 (one Cartan generator, not two).
"""
import cmath, math
import numpy as np

ELL = 3
Q = cmath.exp(2j * math.pi / ELL)
QI = Q ** (-1)
D = Q - QI

# ---------------------------------------------------------------------------
# B⁺(sl_2) at ℓ=3, basis e_{a,c} = K^a E^c, 0 ≤ a, c ≤ 2. dim = 9.
# ---------------------------------------------------------------------------
def bidx(a, c):
    return a * ELL + c

def bfrom(i):
    return i // ELL, i % ELL

DIM_B = ELL * ELL  # 9

# q-binomial [n,k]_{q^2}
def q_binom(n, k, qq):
    if k < 0 or k > n:
        return 0
    def poch(m):
        r = 1
        for i in range(m):
            r *= (1 - qq ** (i + 1))
        return r
    return poch(n) / (poch(k) * poch(n - k))

QB2 = np.zeros((ELL, ELL), dtype=complex)
for n in range(ELL):
    for k in range(ELL):
        QB2[n, k] = q_binom(n, k, Q ** 2)

# Multiplication tensor: bmult_t[k, i, j] = coeff of e_k in e_i * e_j
bmult_t = np.zeros((DIM_B, DIM_B, DIM_B), dtype=complex)
for i in range(DIM_B):
    a_i, c_i = bfrom(i)
    for j in range(DIM_B):
        a_j, c_j = bfrom(j)
        if c_i + c_j >= ELL:
            continue
        phase = Q ** (-2 * a_j * c_i)
        k = bidx((a_i + a_j) % ELL, c_i + c_j)
        bmult_t[k, i, j] = phase

# Comultiplication Δ(e_i): bDelta[k, l, i] = coeff of e_k ⊗ e_l in Δ(e_i)
bDelta = np.zeros((DIM_B, DIM_B, DIM_B), dtype=complex)
for i in range(DIM_B):
    a, c = bfrom(i)
    # Δ(K^a E^c) = sum_k [c,k]_{q^2} e_{a, c-k} ⊗ e_{(a+c-k)%ELL, k}
    for k in range(c + 1):
        coeff = QB2[c, k]
        j = bidx(a, c - k)
        m = bidx((a + c - k) % ELL, k)
        bDelta[j, m, i] += coeff

# Antipode S: bS[j, i] = coeff of e_j in S(e_i)
# S(K^a E^c) = (-1)^c q^{2(a+c)c} K^{-(a+c) mod ELL} E^c
bS = np.zeros((DIM_B, DIM_B), dtype=complex)
for i in range(DIM_B):
    a, c = bfrom(i)
    a_new = (-a - c) % ELL
    sign = (-1) ** c
    phase = Q ** (2 * (a + c) * c)
    j = bidx(a_new, c)
    bS[j, i] = sign * phase

# S^{-1} via matrix inverse
bS_inv = np.linalg.inv(bS)

# Δ²(a) = (Δ ⊗ id) Δ(a) = a_1 ⊗ a_2 ⊗ a_3
# bDelta2[a1, a2, a3, i] = coeff of e_{a1} ⊗ e_{a2} ⊗ e_{a3} in Δ²(e_i)
bDelta2 = np.einsum('jkl,ilm->ijkl', bDelta, bDelta)
# Wait, that's not quite right. Let me redo.
# Δ(e_i) = sum_{j,k} bDelta[j,k,i] e_j ⊗ e_k
# Δ²(e_i) = (Δ ⊗ id)(Δ(e_i)) = sum_{j,k} bDelta[j,k,i] (Δ(e_j) ⊗ e_k)
#         = sum_{j,k} bDelta[j,k,i] (sum_{l,m} bDelta[l,m,j] e_l ⊗ e_m) ⊗ e_k
#         = sum_{j,k,l,m} bDelta[j,k,i] bDelta[l,m,j] e_l ⊗ e_m ⊗ e_k
# So bDelta2[l, m, k, i] = sum_j bDelta[j,k,i] bDelta[l,m,j]
bDelta2 = np.einsum('jki,lmj->lmki', bDelta, bDelta)

# Test: Δ²(K) should be K ⊗ K ⊗ K. K = e_3 (bidx(1,0)).
# Δ(K) = K ⊗ K. Δ²(K) = Δ(K) ⊗ K = K ⊗ K ⊗ K. So bDelta2[3, 3, 3, 3] = 1.
print(f"Δ²(K) check: bDelta2[3,3,3,3] = {bDelta2[3,3,3,3]:.4f} (should be 1)")

# Test: Δ²(E) should be E⊗K⊗K + 1⊗E⊗K + 1⊗1⊗E. E = e_1 (bidx(0,1)).
# E_1 = (bidx(0,1), bidx(0,0), bidx(0,0)) = (1, 0, 0) -- wait, e_1 = K^0 E^1, e_0 = K^0 E^0 = 1.
# So Δ²(E) = e_1 ⊗ e_3 ⊗ e_3 + e_0 ⊗ e_1 ⊗ e_3 + e_0 ⊗ e_0 ⊗ e_1.
# (using e_3 = K, e_1 = E, e_0 = 1)
print(f"Δ²(E) check:")
print(f"  bDelta2[1,3,3,1] = {bDelta2[1,3,3,1]:.4f} (should be 1)")
print(f"  bDelta2[0,1,3,1] = {bDelta2[0,1,3,1]:.4f} (should be 1)")
print(f"  bDelta2[0,0,1,1] = {bDelta2[0,0,1,1]:.4f} (should be 1)")

# ---------------------------------------------------------------------------
# X = (B^*)^{cop} = B⁻(sl_2), basis {x_i} dual to {e_i}.
# Pairing <x_i, e_j> = δ_{i,j}.
# Δ_X(x_i) = sum_{k,l} xDelta[k,l,i] x_k ⊗ x_l, where
# xDelta[k,l,i] = bmult_t[i, l, k]  (dual to multiplication in B, with flip for cop)
# ---------------------------------------------------------------------------
DIM_X = DIM_B
xDelta = np.zeros((DIM_X, DIM_X, DIM_X), dtype=complex)
for i in range(DIM_X):
    for k in range(DIM_X):
        for l in range(DIM_X):
            xDelta[k, l, i] = bmult_t[i, l, k]

# ---------------------------------------------------------------------------
# Cross-action decomposed:
# cross_dec[b, j, a1, a3] = coeff of e_j in S^{-1}(a3) · b · a1
# (so that a1 · x_j · S^{-1}(a3) = sum_b cross_dec[b, j, a1, a3] x_b)
# ---------------------------------------------------------------------------
# S^{-1}(a3) · b · a1 = sum_{p,q,r} bS_inv[p, a3] bmult_t[q, b, a1] bmult_t[r, p, q] e_r
# So coeff of e_j in S^{-1}(a3) · b · a1 = sum_{p,q} bS_inv[p, a3] bmult_t[q, b, a1] bmult_t[j, p, q]
cross_dec = np.einsum('pa,qba,jpq->bjpa', bS_inv, bmult_t, bmult_t)

# ---------------------------------------------------------------------------
# Euler derivation Θ: B → B, Θ(K^a E^c) = 2c · K^a E^c
# ---------------------------------------------------------------------------
Theta_mat = np.zeros((DIM_B, DIM_B), dtype=complex)
for i in range(DIM_B):
    a, c = bfrom(i)
    Theta_mat[i, i] = 2 * c  # Θ(e_i) = 2c · e_i

# Verify Θ is a derivation: Θ(e_i · e_j) = Θ(e_i) · e_j + e_i · Θ(e_j)
# Test on K, E: Θ(K · E) = Θ(K E) = Θ(e_3) · ... wait, e_{bidx(1,1)} = K E.
# Θ(K E) = 2 · K E (since c=1). 
# Θ(K) · E + K · Θ(E) = 0 · E + K · 2E = 2 K E. ✓
print(f"\nΘ derivation check (K, E):")
i, j = bidx(1,0), bidx(0,1)
# K · E = q^0 K^1 E^1 = e_{bidx(1,1)} = e_4
k = bidx(1, 1)
phase = bmult_t[k, i, j]
lhs = phase * Theta_mat[k, k]  # Θ(K·E) = phase * 2c_K·E * e_K·E
# Actually Θ(e_k) = 2 c_k e_k where c_k = (k % ELL). For k = bidx(1,1), c_k = 1, so Θ(e_4) = 2 e_4.
# Θ(K)·E + K·Θ(E) = (0·e_3)·e_1 + e_3·(2·e_1) = 0 + 2·(e_3·e_1) = 2·phase·e_4
rhs = 0 + 2 * phase  # Θ(K) = 0, so first term is 0; second term: K · Θ(E) = e_3 · 2 e_1 = 2 phase e_4
print(f"  Θ(K·E) = {lhs:.4f}, Θ(K)·E + K·Θ(E) = {rhs:.4f} (should be equal)")

# ---------------------------------------------------------------------------
# Compute δ(Θ) as a 2-cochain on D(B) = X ⊗ B (dim 81).
# δ(Θ)(u, v) = Θ̄(∂^X_2 ∂^B_0 φ_c̃(u, v))
# 
# Using einsum with the simplified contraction (after using the δ-pairings).
# ---------------------------------------------------------------------------
# Formula:
# δ(Θ)[xu, au, xv, av] = sum over many indices of:
#   B2[a1, a2, a3, au]                    # Δ²(a_u)
#   * CD[bmid, xv, a1, a3]                # cross-action per (a1, a3)
#   * DX[xu1, xu2, xu]                    # Δ_X(x_u) for φ_1(x_u, a_2)
#   * DB[a21, a22, a2]                    # Δ_B(a_2) for φ_1(x_u, a_2)
#   * Sinv[xu1, a21]                      # <xu_1 | S^{-1}(a2_1)> in φ_1
#   * DX[bv1, bv2, bmid]                  # Δ_X(b_mid) for φ_1(b_mid, a_v)
#   * DB[av1, av2, av]                    # Δ_B(a_v) for φ_1(b_mid, a_v)
#   * Sinv[bv1, av1]                      # <bv_1 | S^{-1}(av_1)> in φ_1
#   * DX[xu21, a22, xu2]                  # ∂^B_0 step (xu_2_2 = a_2_2)
#   * DX[bv21, av21, bv2]                 # ∂^X_2 step (bv_2_2 = av_2_1)
#   * DB[av21, av22, av2]                 # ∂^X_2 step
#   * TH[xu21, av22]                      # Θ̄(xu_2_1, av_2_2)
# 
# einsum indices (single chars):
# au=u, xu=x, av=v, xv=y, a1=b, a2=c, a3=d, bmid=e, xu1=f, xu2=g, 
# a21=h, a22=i, bv1=j, bv2=k, av1=l, av2=m, xu21=n, av21=o, av22=p
# 
# operands:
# 1. B2[b, c, d, u]
# 2. CD[e, y, b, d]
# 3. DX[f, g, x]
# 4. DB[h, i, c]
# 5. Sinv[f, h]
# 6. DX[j, k, e]
# 7. DB[l, m, v]
# 8. Sinv[j, l]
# 9. DX[n, i, g]   (xu_2_2 = a22 = i)
# 10. DX[o, o, k]  -- wait, this is wrong. bv_2_2 = av_2_1, so DX[bv21, av21, bv2] = DX[o, o, k]?
#     No, bv21 and av21 are different indices. Let me re-check.
#     ∂^X_2(x_1 ⊗ x_2, a) = <(x_2)_2 | a_1> (x_1, a_2)
#     Here x_1 = bv_2_1 = o, x_2 = bv_2 = k, a = av_2 = m
#     Δ(x_2) = Δ(bv_2) = bv_2_1 ⊗ bv_2_2 = o ⊗ ?
#     Δ(a) = Δ(av_2) = av_2_1 ⊗ av_2_2 = ? ⊗ p
#     <bv_2_2 | av_2_1> = δ_{bv_2_2, av_2_1}
#     
#     So DX[bv21, bv22, bv2] = DX[o, ?, k] and DB[av21, av22, av2] = DB[?, p, m]
#     with the pairing δ_{bv22, av21} setting bv22 = av21.
#     
#     So DX[o, bv22, k] with bv22 = av21, and DB[av21, p, m].
#     Let's call av21 = bv22 = q (single shared index).
#     Then DX[o, q, k] and DB[q, p, m].

# Let me redo the einsum with proper indices:
# 1. B2[b, c, d, u]
# 2. CD[e, y, b, d]
# 3. DX[f, g, x]
# 4. DB[h, i, c]
# 5. Sinv[f, h]
# 6. DX[j, k, e]
# 7. DB[l, m, v]
# 8. Sinv[j, l]
# 9. DX[n, i, g]   (xu_2_2 = a22 = i)
# 10. DX[o, q, k]  (bv_2_1 = o, bv_2_2 = av_2_1 = q, bv_2 = k)
# 11. DB[q, p, m]  (av_2_1 = q, av_2_2 = p, av_2 = m)
# 12. TH[n, p]     (Θ̄(xu_2_1, av_2_2) = Θ̄(n, p))

# Free: x, u, y, v
# Summed: b, c, d, e, f, g, h, i, j, k, l, m, n, o, q, p

# einsum: 'bcdu,eybd,fgx,hic,fh,jke,lmv,jl,nig,oqk,qpm,np->xuyv'

print("\nComputing δ(Θ) via einsum (this may take a moment)...")
result = np.einsum(
    'bcdu,eybd,fgx,hic,fh,jke,lmv,jl,nig,oqk,qpm,np->xuyv',
    bDelta2, cross_dec, xDelta, bDelta, bS_inv,
    xDelta, bDelta, bS_inv,
    xDelta, xDelta, bDelta, Theta_mat,
    optimize='optimal'
)
print(f"δ(Θ) shape: {result.shape}, max |δ(Θ)| = {np.max(np.abs(result)):.6f}")

# Save as a 2D matrix on D(B)^2
DIM_D = DIM_X * DIM_B  # 81
delta_Theta = result.reshape(DIM_D, DIM_D)

# Print some specific values
def didx(x, b):
    return x * DIM_B + b

# Identify key elements
# 1_X = ε_B (counit of B, as element of X). In dual basis, ε_B = sum_a x_{bidx(a,0)} (since ε(K^a E^c) = δ_{c,0}).
# Actually ε_B(e_{a,c}) = δ_{c,0}, so ε_B = sum_a x_{bidx(a,0)} = x_0 + x_3 + x_6.
# 1_B = e_0 (since e_0 = K^0 E^0 = 1).
# So 1_D = (1_X, 1_B) = didx(idx_1X, 0) where idx_1X is the index of ε_B in X.
# But ε_B is not a basis element; it's a sum. So 1_D is not a single basis index.

# Let me just print values on basis pairs.
# Generators: K = e_3, E = e_1, F = ?
# F corresponds to x_? in X. Since X = B⁻ with K ↔ x_3 dual to e_3 = K, 
# but F is not simply dual to E. Let me think.
# In the standard identification X = (B^+)^{*cop} ≅ B^-(sl_2), the generator F of B⁻ 
# corresponds to... hmm, it depends on the chosen isomorphism.
# 
# In the dual basis: x_i ↔ (some element of B⁻). The natural identification is 
# x_{bidx(a,c)} ↔ (1/Z_{a,c}) K^{-a} F^c for some normalization Z_{a,c}.
# 
# For simplicity, let's identify x_{bidx(0,1)} = x_1 with F (up to normalization).
# Then x_{bidx(a,c)} = (something) · K^{-a} F^c.
# 
# In the dual basis: <x_{bidx(a,c)}, K^{a'} E^{c'}> = δ_{a,a'} δ_{c,c'}.
# 
# In B⁻ = C[K, F] / (K^3 - 1, F^3), the natural pairing with B⁺ would be:
# <K^a F^c, K^{a'} E^{c'}> = (some pairing).
# 
# Let me not worry about the identification and just report values on basis pairs.

# Find the largest entries in delta_Theta
abs_delta = np.abs(delta_Theta)
top_idx = np.argsort(abs_delta.ravel())[::-1][:30]
print(f"\nTop 30 entries of δ(Θ) (indices into D(B)^2 = 81^2):")
print(f"  (rank by |δ(Θ)|)")
for flat_idx in top_idx:
    i = flat_idx // DIM_D
    j = flat_idx % DIM_D
    x_u, a_u = i // DIM_B, i % DIM_B
    x_v, a_v = j // DIM_B, j % DIM_B
    val = delta_Theta[i, j]
    if abs(val) < 1e-10:
        break
    a_u_, c_u_ = bfrom(a_u)
    a_v_, c_v_ = bfrom(a_v)
    print(f"  δ(Θ)[x_({x_u//ELL},{x_u%ELL}), K^{a_u_}E^{c_u_}; x_({x_v//ELL},{x_v%ELL}), K^{a_v_}E^{c_v_}] = {val:.4f}")

# Save the full result
np.save('/tmp/delta_Theta_sl2.npy', delta_Theta)
print(f"\nSaved δ(Θ) to /tmp/delta_Theta_sl2.npy")

# Verify δ(Θ) is a Hochschild 2-cocycle: ∂δ = 0, i.e., 
# ε(u)δ(v,w) - δ(uv, w) + δ(u, vw) - δ(u, v)ε(w) = 0 for all u, v, w.
# This requires the multiplication in D(B), which is more complex.
# For now, just check it's normalized: δ(1, u) = δ(u, 1) = 0.
# Compute 1_D = (1_X, 1_B) where 1_X = ε_B = sum_a x_{bidx(a,0)} and 1_B = e_0.
eps_B = np.zeros(DIM_X, dtype=complex)
for a in range(ELL):
    eps_B[bidx(a, 0)] = 1.0
# 1_D in the basis expansion: sum_a eps_B[a] · (x_a, e_0) = sum_a (x_a, e_0) for a = 0, 3, 6.
one_D = np.zeros(DIM_D, dtype=complex)
for a in range(DIM_X):
    one_D[didx(a, 0)] = eps_B[a]

# δ(Θ)(1, u) = sum_a eps_B[a] · δ(Θ)(didx(a, 0), u)
delta_1_u = one_D @ delta_Theta  # shape (DIM_D,)
delta_u_1 = delta_Theta @ one_D  # shape (DIM_D,)
print(f"\nNormalization check (δ(Θ) should vanish when any arg is 1):")
print(f"  ||δ(Θ)(1, ·)|| = {np.linalg.norm(delta_1_u):.2e}")
print(f"  ||δ(Θ)(·, 1)|| = {np.linalg.norm(delta_u_1):.2e}")
