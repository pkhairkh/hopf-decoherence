"""Compute dim H̃²_b(B+(sl_2)) at ℓ=3 to validate the obstruction approach.
Expected: dim H̃²_b = 0 (all cocycles lift, dim HH²(D) = 3)."""
import cmath, math, numpy as np
from scipy import sparse
from scipy.linalg import eigvalsh

ell=3; q=cmath.exp(2j*math.pi/ell); dim=9
def idx(a,b): return a*ell+b
def from_idx(i): return i//ell, i%ell

# Multiplication table
mult=np.zeros((dim,dim,dim),dtype=complex)
for i in range(dim):
    a1,b1=from_idx(i)
    for j in range(dim):
        a2,b2=from_idx(j)
        bs=b1+b2
        if bs<ell: mult[idx((a1+a2)%ell,bs),i,j]=q**(-2*a2*b1)

# Counit
eps=np.zeros(dim,dtype=complex)
for a in range(ell): eps[idx(a,0)]=1.0

# Comultiplication table
Delta=[None]*dim
for a in range(ell):
    for b in range(ell):
        i=idx(a,b)
        d={(idx(0,0),idx(0,0)):1.0+0j}
        for _ in range(a):
            nd={}
            for(j,k),v in d.items():
                ja,jb=from_idx(j);ka,kb=from_idx(k)
                jn=idx((ja+1)%ell,jb);kn=idx((ka+1)%ell,kb)
                nd[(jn,kn)]=nd.get((jn,kn),0)+v
            d=nd
        dE={(idx(0,0),idx(0,0)):1.0+0j}
        dE1={(idx(0,1),idx(1,0)):1.0+0j,(idx(0,0),idx(0,1)):1.0+0j}
        for _ in range(b):
            ndE={}
            for(j1,k1),v1 in dE.items():
                for(j2,k2),v2 in dE1.items():
                    for l in range(dim):
                        vj=mult[l,j1,j2]
                        if abs(vj)>1e-14:
                            for m in range(dim):
                                vk=mult[m,k1,k2]
                                if abs(vk)>1e-14:
                                    ndE[(l,m)]=ndE.get((l,m),0)+v1*v2*vj*vk
            dE={k:v for k,v in ndE.items() if abs(v)>1e-14}
        f={}
        for(j1,k1),v1 in d.items():
            for(j2,k2),v2 in dE.items():
                for l in range(dim):
                    vj=mult[l,j1,j2]
                    if abs(vj)>1e-14:
                        for m in range(dim):
                            vk=mult[m,k1,k2]
                            if abs(vk)>1e-14:
                                f[(l,m)]=f.get((l,m),0)+v1*v2*vj*vk
        Delta[i]=[(j,k,v) for(j,k),v in f.items() if abs(v)>1e-14]

# B̄ = ker ε (exclude basis[0] = 1)
B_bar=list(range(1,dim)); n_bar=len(B_bar)

# Build modified structure constants (for B̄)
mult_bar={}
for a in B_bar:
    for b in B_bar:
        for l in range(dim):
            v=mult[l,a,b]
            if l==a: v-=eps[b]
            if l==b: v-=eps[a]
            if abs(v)>1e-13: mult_bar[(l,a,b)]=v

delta_bar={}
for c in B_bar:
    for(j,k,v) in Delta[c]:
        if j!=0 and k!=0: delta_bar[(c,j,k)]=v

# Build ∂_b^1: C^1 → C^2 (same as H̃¹_b computation)
# C^1 = Hom(B̄,B̄), dim = n_bar² = 64
# C^2 = Hom(B̄²,B̄) ⊕ Hom(B̄,B̄²), dim = 2*n_bar³ = 1024
cols_list=[(j,k) for j in B_bar for k in B_bar]
n_cols=len(cols_list)
col_map={p:i for i,p in enumerate(cols_list)}

h_rows=[(a,b,t) for a in B_bar for b in B_bar for t in B_bar]
n_h=len(h_rows); h_map={t:i for i,t in enumerate(h_rows)}
c_rows=[(c,a,b) for c in B_bar for a in B_bar for b in B_bar]
n_c=len(c_rows); c_map={t:i for i,t in enumerate(c_rows)}

# Inverse lookups
inv_mb={}; msb={}; msa={}
for(l,a,b),v in mult_bar.items():
    inv_mb.setdefault(l,[]).append((a,b,v))
    msb.setdefault(b,[]).append((a,l,v))
    msa.setdefault(a,[]).append((b,l,v))
idl={}; idr={}; dbc={}
for(c,j,k),v in delta_bar.items():
    idl.setdefault((c,k),[]).append((j,v))
    idr.setdefault((c,j),[]).append((k,v))
    dbc.setdefault(c,[]).append((j,k,v))

rows_l=[];cols_l=[];vals_l=[]
for ci,(j,k) in enumerate(cols_list):
    for(a,t,v) in msb.get(j,[]):
        r=h_map.get((a,k,t))
        if r is not None: rows_l.append(r);cols_l.append(ci);vals_l.append(v)
    for(a,b,v) in inv_mb.get(k,[]):
        r=h_map.get((a,b,j))
        if r is not None: rows_l.append(r);cols_l.append(ci);vals_l.append(-v)
    for(b,t,v) in msa.get(j,[]):
        r=h_map.get((k,b,t))
        if r is not None: rows_l.append(r);cols_l.append(ci);vals_l.append(v)
    for c in B_bar:
        for(al,v) in idl.get((c,k),[]):
            r=c_map.get((c,al,j))
            if r is not None: rows_l.append(n_h+r);cols_l.append(ci);vals_l.append(-v)
    for(al,be,v) in dbc.get(j,[]):
        r=c_map.get((k,al,be))
        if r is not None: rows_l.append(n_h+r);cols_l.append(ci);vals_l.append(v)
    for c in B_bar:
        for(be,v) in idr.get((c,k),[]):
            r=c_map.get((c,j,be))
            if r is not None: rows_l.append(n_h+r);cols_l.append(ci);vals_l.append(-v)

n_c2=n_h+n_c  # 1024
db1=sparse.csr_matrix((vals_l,(rows_l,cols_l)),shape=(n_c2,n_cols),dtype=complex)

# Compute rank of ∂_b^1
gram1=(db1.conj().T@db1).toarray()
ev1=np.sort(np.abs(eigvalsh(gram1)))[::-1]
tol1=max(db1.shape)*ev1[0]*1e-10
rank1=int(np.sum(ev1>tol1))
dim_h1b=n_cols-rank1
print(f"∂_b^1: shape=({n_c2},{n_cols}), rank={rank1}, dim H̃¹_b={dim_h1b}")
print(f"Expected: dim H̃¹_b = 1")

# Now build ∂_b^2: C^2 → C^3
# C^3 = Hom(B̄³,B̄) ⊕ Hom(B̄²,B̄²) ⊕ Hom(B̄,B̄³)
# dim C^3 = n_bar⁴ + n_bar⁴ + n_bar⁴ = 3*4096 = 12288
# 
# The input (f,g) ∈ C^2 has:
# f: B̄²→B̄ indexed by (t,a,b), and g: B̄→B̄² indexed by (j,k,c)
# The output ∂_b(f,g) has three components:
# 1. ∂^h(f): B̄³→B̄, (∂^h f)(a,b,c) = a*f(b,c) - f(ab,c) + f(a,bc) - f(a,b)*c
# 2. ∂^h(g) - ∂^c(f): B̄²→B̄²
# 3. -∂^c(g): B̄→B̄³

# This is a 12288 × 1024 matrix. Feasible!
print(f"\nBuilding ∂_b^2: shape (12288, 1024)...")

# Component 1: ∂^h(f) ∈ Hom(B̄³,B̄), indexed by (a,b,c,t) ∈ B̄⁴
# Input: f indexed by (t,a,b) ∈ B̄³ (n_h = 512 columns)
# (∂^h f)(a,b,c)[t] = mult_bar[t,a,j]*f[j,b,c] - mult_bar[j,a,b]*f[t,j,c] + mult_bar[j,b,c]*f[t,a,j] - mult_bar[t,a,b]*f[j]... 
# Wait, let me be more careful.
# f: B̄²→B̄, f(a,b) = Σ_t f_t(a,b) * basis[t] where f_t(a,b) is the t-component.
# As a vector, f is indexed by (t, a, b) with t,a,b ∈ B̄.
# 
# ∂^h(f)(a,b,c) = a*f(b,c) - f(ab,c) + f(a,bc) - f(a,b)*c
# The result is in B̄, so ∂^h(f)(a,b,c)[t] for t ∈ B̄.
# 
# a*f(b,c): a * (Σ_t f(b,c)[t] * basis[t]) = Σ_t f(b,c)[t] * (a * basis[t])
# = Σ_t f(b,c)[t] * Σ_s mult[s,a,t] * basis[s]
# So (a*f(b,c))[s] = Σ_t mult[s,a,t] * f(b,c)[t] = Σ_t mult_bar[s,a,t] * f[t,b,c]
# (using mult_bar since a,t ∈ B̄)
#
# f(ab,c): ab = Σ_j mult[j,a,b] * basis[j], but we need mult_bar for B̄
# f(ab,c)[t] = Σ_j mult_bar[j,a,b] * f[t,j,c]  (if j ∈ B̄) + mult[0,a,b]*f[t,0,c] (if ab has component 1)
# Wait, f is only defined on B̄², so f[t,j,c] is only for j ∈ B̄.
# But ab may have a component at basis[0] = 1 (when ε(a)*ε(b) ≠ 0... but a,b ∈ B̄ so ε(a)=ε(b)=0).
# Actually, for a,b ∈ B̄: mult[0,a,b] = ε(a)*ε(b) + correction... 
# mult_bar already subtracts the ε terms. So mult_bar[j,a,b] = mult[j,a,b] - ε(b)*δ_{j,a} - ε(a)*δ_{j,b}.
# For a,b ∈ B̄: ε(a) = ε(b) = 0, so mult_bar[j,a,b] = mult[j,a,b].
# And mult[0,a,b] = 0 (since the product of two B̄ elements has no constant term... 
# actually that's not true. K*E = KE, and ε(KE) = 0. But K*K² = 1, and ε(1) = 1.
# So mult[0, K, K²] = 1, but K, K² ∈ B̄.
# In this case, mult_bar[0, K, K²] = mult[0, K, K²] - ε(K²)*δ_{0,K} - ε(K)*δ_{0,K²}
# = 1 - 0 - 0 = 1. So mult_bar includes the constant term.
# But for the B̄-valued map, we need to PROJECT the result to B̄.
# 
# Actually, the modified structure constants mult_bar already handle this:
# mult_bar[l,a,b] = mult[l,a,b] - ε(b)*δ_{l,a} - ε(a)*δ_{l,b}
# For a,b ∈ B̄ (ε(a)=ε(b)=0): mult_bar[l,a,b] = mult[l,a,b].
# So the product ab in B̄ is: ab = Σ_{l ∈ B̄} mult_bar[l,a,b] * basis[l] + mult[0,a,b] * basis[0]
# The B̄-valued part is: Σ_{l ∈ B̄} mult_bar[l,a,b] * basis[l].
# The B/B̄ part (the ε-projection) is: mult[0,a,b] * basis[0].
# 
# For the Hochschild differential in the B̄-valued complex, we use mult_bar
# (which is the B̄-projected multiplication). This is what we already do.

# I'll build ∂_b^2 as a sparse matrix.
# Input columns: (f_part, g_part) = ((t,a,b), (j,k,c)) for the f and g components
# Output rows: ((a,b,c,t) for ∂^h(f), (a,b,j,k) for mixed, (c,j,k,l) for ∂^c(g))

# Actually, this is getting very complex. Let me just compute it directly
# using the same approach as ∂_b^1 but for degree 2.

# For now, let me just check: does the obstruction vanish for sl_2?
# The obstruction is: is (∂^h(φ*1), -∂^c(φ*1)) in im(∂_b^1)?
# I already have ∂_b^1 and the cocycle φ. Let me compute the obstruction vector
# and check if it's in the column space.

# First: extract the cocycle from HH²(B+(sl_2))
n2=dim*dim; n3=dim**3
d1=np.zeros((n2,dim),dtype=complex)
for col in range(dim):
    for a in range(dim):
        for b in range(dim):
            d1[a*dim+b,col]=eps[a]*(1.0 if b==col else 0)-mult[col,a,b]+(1.0 if a==col else 0)*eps[b]

d2=np.zeros((n3,n2),dtype=complex)
for a in range(dim):
    for b in range(dim):
        for c in range(dim):
            row=a*dim*dim+b*dim+c
            if abs(eps[a])>1e-14: d2[row,b*dim+c]+=eps[a]
            for k in range(dim):
                v=mult[k,a,b]
                if abs(v)>1e-14: d2[row,k*dim+c]-=v
            for k in range(dim):
                v=mult[k,b,c]
                if abs(v)>1e-14: d2[row,a*dim+k]+=v
            if abs(eps[c])>1e-14: d2[row,a*dim+b]-=eps[c]

U1,_,_=np.linalg.svd(d1,full_matrices=True)
rank_d1=9; im_d1=U1[:,:rank_d1]
A=np.vstack([d2,im_d1.conj().T])
UA,sA,VhA=np.linalg.svd(A,full_matrices=True)
rA=int(np.sum(sA>1e-9*sA[0]))
cocycles=VhA[rA:].conj()
phi=cocycles[0]
print(f"\nCocycle extracted: ||d²φ||={np.linalg.norm(d2@phi):.2e}")

# Compute the obstruction vector in C^2 (not C^3!)
# The obstruction is the image of φ under the map HH²(B,C) → H̃²_b(B).
# This map is: φ ↦ (f, g) where f = φ * 1_B (B̄-valued), g = 0.
# Then check if (f, 0) is in im(∂_b^1) + ker(∂_b^2).
# 
# Actually, the obstruction is simpler: the image of φ in H̃²_b is the class
# of (f, 0) where f(a,b) = φ(a,b) * 1_B.
# But 1_B ∉ B̄, so f is NOT in Hom(B̄², B̄)!
# 
# The correct lift: f(a,b) = φ(a,b) * 1_B. Since 1_B = basis[0] and
# B̄ = {basis[1], ..., basis[8]}, f is NOT B̄-valued.
# 
# So the lift doesn't work directly. The MW LES must use a different lift.
# 
# For a HOPF algebra, the counit ε: B → C gives a splitting B = C ⊕ B̄.
# The lift uses this splitting: f(a,b) = φ(a,b) * 1_B, projected to B̄ via
# f(a,b) = φ(a,b) * (1_B - ε(1_B)*1_B) = 0. 
# That gives f = 0, which is trivial.
# 
# The correct lift must use the MULTIPLICATION to "thicken" the scalar:
# f(a,b) = φ(a,b) * v for some v ∈ B̄. But which v?
# 
# Actually, I think the MW LES for TRIVIAL coefficients uses a DIFFERENT
# bicomplex than for B̄-valued coefficients. The bicomplex for trivial
# coefficients has B^{p,q} = Hom(B̄^p, C) for the Hochschild part and
# B^{p,q} = Hom(C, B̄^q) for the coalgebra part.
# 
# At degree 2: B^{2,0} ⊕ B^{1,1} ⊕ B^{0,2}
# = Hom(B̄², C) ⊕ Hom(B̄, B̄) ⊕ Hom(C, B̄²)
# = Hom(B̄², C) ⊕ Hom(B̄, B̄) ⊕ B̄²
# 
# The Hochschild 2-cocycle φ ∈ Hom(B̄², C) lives in B^{2,0}.
# Its image in H̃²_b is its class in the total cohomology of the bicomplex.
# 
# Since φ is a cocycle (∂^h(φ) = 0) and ∂^c(φ) = 0 (C-valued, Δ on C is trivial),
# φ is a TOTAL cocycle in the bicomplex.
# So its class in H̃²_b is just [(φ, 0, 0)].
# 
# The obstruction map sends (φ, ψ) to [(φ, 0, 0)] + [(0, 0, ψ)] = [(φ, 0, ψ)].
# Wait, ψ is a B* cocycle. In the bicomplex for B, the B* part corresponds
# to the COALGEBRA direction.
# 
# So the image of (φ, ψ) in H̃²_b is [(φ, g_ψ, h_ψ)] where g_ψ and h_ψ
# are derived from ψ via the duality B ↔ B*.
# 
# For a B* cocycle ψ: B*² → C, the dual gives a map g: B → B⊗B
# (using the dual of the multiplication on B*).
# 
# The rank of the obstruction map = number of independent classes
# [(φ_i, 0, 0)] and [(0, 0, ψ_j)] in H̃²_b.
# 
# For the B+ cocycles: [(φ_i, 0, 0)] is zero in H̃²_b iff φ_i ∈ im(∂_b^1).
# 
# ∂_b^1 maps C^1 → C^2. The B^{2,0} component of ∂_b^1 is:
# ∂^h: Hom(B̄, B̄) → Hom(B̄², B̄) (for B̄-valued) ... 
# or ∂^h: Hom(B̄, C) → Hom(B̄², C) (for C-valued).
# 
# For C-valued: ∂^h(h)(a,b) = ε(a)*h(b) - h(ab) + h(a)*ε(b) = -h(ab)
# (since ε(a)=0 for a ∈ B̄). This is just the restriction of h to products.
# 
# But HH¹(B̄, C) = 0 (verified), so ∂^h is injective on C^1/B^0.
# Wait, ∂^h: C^1 → C^2 has rank = dim C^1 - dim ker(∂^h).
# dim C^1 = dim Hom(B̄, C) = n_bar = 8.
# ker(∂^h) = {h: B̄→C : h(ab) = 0 for all a,b ∈ B̄} = derivations that vanish on products.
# For B+(sl_2): ab generates all of B̄ (since B̄ is generated by E, K-1), so ker = 0.
# rank = 8, dim im(∂^h) = 8.
# 
# The B^{2,0} part of C^2 has dim n_bar² = 64.
# The image of ∂^h in B^{2,0} has dim 8.
# So dim H̃²_b|_{B^{2,0}} = dim ker(∂^h: B^{2,0}→B^{3,0}) / dim im(∂^h: B^{1,0}→B^{2,0})
# = dim HH²(B̄, C) / dim im(∂^h) = 1 / 8... no, that's not right.
# 
# dim H̃²_b|_{B^{2,0}} = dim ker(∂^h at B^{2,0}) - dim im(∂^h at B^{1,0})
# ker(∂^h at B^{2,0}) = HH²(B̄, C) = 1 (the cocycle φ)
# im(∂^h at B^{1,0}) = 8 (as computed)
# So the B^{2,0} part of H̃²_b has dim 1 - 8 = -7? That's negative!
# 
# I'm confusing the indices. The total cohomology H̃²_b is NOT the direct
# sum of the B^{p,q} cohomologies. It's the cohomology of the TOTAL complex.
# 
# The spectral sequence gives:
# E_1^{p,q} = H^q_v(B^{p,*}) = coalgebra cohomology of B^{p,*}
# E_2^{p,q} = H^p_h(E_1^{*,q}) = Hochschild cohomology of E_1
# H̃²_b = ⊕_{p+q=2} E_∞^{p,q}
# 
# For the first page:
# E_1^{2,0} = H^0_v(Hom(B̄², C)) = Hom(B̄², C) (since ∂^c on C-valued is 0)
# E_1^{1,1} = H^1_v(Hom(B̄, B̄)) = ker(∂^c)/im(∂^c) for B̄-valued
# E_1^{0,2} = H^2_v(B̄²) = ker(∂^c: B̄²→B̄³)/im(∂^c: B̄→B̄²)
# 
# This is getting too complex. Let me just compute H̃²_b NUMERICALLY
# by building ∂_b^2 and computing its rank.
# 
# For sl_2: ∂_b^2 is 12288 × 1024. Gram matrix is 1024 × 1024 = 16 MB. TRIVIAL!

print("\nBuilding ∂_b^2 for B+(sl_2) at ℓ=3...")
print(f"Input: C^2 = Hom(B̄²,B̄) ⊕ Hom(B̄,B̄²), dim = {n_h} + {n_c} = {n_h+n_c}")
print(f"Output: C^3 = 3 components, each dim {n_bar**4} = {n_bar**4}")
n_c3 = 3 * n_bar**4
print(f"Total output dim: {n_c3}")

# Build ∂_b^2 as sparse matrix
# Input columns: (f-part: (t,a,b), g-part: (j,k,c))
# Output rows: 
#   Component 1 (∂^h(f)): (a,b,c,t) ∈ B̄⁴ → dim n_bar⁴
#   Component 2 (∂^h(g) - ∂^c(f)): (a,b,j,k) ∈ B̄⁴ → dim n_bar⁴
#   Component 3 (-∂^c(g)): (c,j,k,l) ∈ B̄⁴ → dim n_bar⁴

rows2=[]; cols2=[]; vals2=[]

# Precompute more inverse lookups for mult_bar and delta_bar
mb_by_l = {}  # (a,b) → [(l, v)]
for(l,a,b),v in mult_bar.items():
    mb_by_l.setdefault((a,b),[]).append((l,v))

mb_by_a = {}  # (l,b) → [(a, v)]  -- for fixed l,b: mult_bar[l,a,b] = v gives a
for(l,a,b),v in mult_bar.items():
    mb_by_a.setdefault((l,b),[]).append((a,v))

# Component 1: ∂^h(f) ∈ Hom(B̄³, B̄)
# (∂^h f)(a,b,c)[t] = Σ_s mult_bar[s,a,t]*f(s,b,c) - Σ_j mult_bar[j,a,b]*f(t,j,c) + Σ_j mult_bar[j,b,c]*f(t,a,j) - Σ_s mult_bar[s,a,b]*... 
# Wait, let me be more careful.
# f: B̄²→B̄, f(a,b) = Σ_t f_t(a,b) * basis[t]. As a vector, f is indexed by (t,a,b).
# a*f(b,c) = a * (Σ_t f_t(b,c) * basis[t]) = Σ_t f_t(b,c) * (a*basis[t])
# (a*basis[t])[s] = mult_bar[s,a,t] for s,t ∈ B̄ (using B̄-valued mult)
# So (a*f(b,c))[s] = Σ_t mult_bar[s,a,t] * f_t(b,c) = Σ_t mult_bar[s,a,t] * f[t,b,c]
# 
# f(ab,c): ab = Σ_j mult_bar[j,a,b] * basis[j] (in B̄)
# f(ab,c)[t] = Σ_j mult_bar[j,a,b] * f[t,j,c]
# 
# f(a,bc): bc = Σ_j mult_bar[j,b,c] * basis[j]
# f(a,bc)[t] = Σ_j mult_bar[j,b,c] * f[t,a,j]
# 
# f(a,b)*c: f(a,b) = Σ_t f_t(a,b) * basis[t]
# f(a,b)*c = Σ_t f_t(a,b) * (basis[t]*c) = Σ_t f_t(a,b) * Σ_s mult_bar[s,t,c] * basis[s]
# (f(a,b)*c)[s] = Σ_t mult_bar[s,t,c] * f[t,a,b]
# 
# So ∂^h(f)(a,b,c)[s] = Σ_t mult_bar[s,a,t]*f[t,b,c] - Σ_j mult_bar[j,a,b]*f[s,j,c] 
#                       + Σ_j mult_bar[j,b,c]*f[s,a,j] - Σ_t mult_bar[s,t,c]*f[t,a,b]

# For each input column (t,a,b) of f:
for ci, (t, a, b) in enumerate(h_rows):  # ci = 0..511
    # For each output (a',b',c',s) of ∂^h(f):
    # The output is indexed by (a', b', c', s) where a',b',c' are the triple and s is the output component.
    
    # Term 1: mult_bar[s,a',t] * f[t,b',c'] → contributes to (a',b',c',s) with coefficient mult_bar[s,a',t]
    # But f is indexed by (t, b', c'), not (t, a, b). 
    # The input column is (t, a, b), meaning f_t(a,b).
    # Term 1 appears when (b', c') = (a, b) and the output has a' free.
    # So: for each a', s: output (a', a, b, s) += mult_bar[s, a', t]
    for s in B_bar:
        for a2 in B_bar:
            v = mult_bar.get((s, a2, t), 0)
            if abs(v) > 1e-13:
                row = (a2, a, b, s)  # (a',b',c',s) = (a2, a, b, s)
                r = (a2-1)*n_bar**3 + (a-1)*n_bar**2 + (b-1)*n_bar + (s-1)
                rows2.append(r); cols2.append(ci); vals2.append(v)
    
    # Term 2: -mult_bar[j,a',b'] * f[s,j,c'] → contributes when (j,c') = (a,b) and output has (a',b',s) free
    # So: for each a', b', s: output (a', b', b, s) += -mult_bar[j,a',b'] where j=a
    # Wait, the output is (a', b', c', s) and we need j=a, c'=b.
    # So output (a', b', b, s) += -mult_bar[a, a', b']
    for s in B_bar:
        for a2 in B_bar:
            for b2 in B_bar:
                v = mult_bar.get((a, a2, b2), 0)  # j=a, a'=a2, b'=b2
                if abs(v) > 1e-13:
                    r = (a2-1)*n_bar**3 + (b2-1)*n_bar**2 + (b-1)*n_bar + (s-1)
                    rows2.append(r); cols2.append(ci); vals2.append(-v)
    
    # Term 3: mult_bar[j,b',c'] * f[s,a',j] → contributes when (a',j) = (a,b)
    # So: for each b', c', s: output (a, b', c', s) += mult_bar[b, b', c']
    for s in B_bar:
        for b2 in B_bar:
            for c2 in B_bar:
                v = mult_bar.get((b, b2, c2), 0)  # j=b, b'=b2, c'=c2
                if abs(v) > 1e-13:
                    r = (a-1)*n_bar**3 + (b2-1)*n_bar**2 + (c2-1)*n_bar + (s-1)
                    rows2.append(r); cols2.append(ci); vals2.append(v)
    
    # Term 4: -mult_bar[s,t,c'] * f[t,a,b] → contributes when c' is free
    # Wait, f[t,a,b] is the INPUT column. The output is (a',b',c',s).
    # Term 4: -(f(a',b')*c')[s] = -Σ_t mult_bar[s,t,c'] * f[t,a',b']
    # This contributes when (a',b') = (a,b): output (a, b, c', s) += -mult_bar[s, t, c']
    for s in B_bar:
        for c2 in B_bar:
            v = mult_bar.get((s, t, c2), 0)  # s, t, c'=c2
            if abs(v) > 1e-13:
                r = (a-1)*n_bar**3 + (b-1)*n_bar**2 + (c2-1)*n_bar + (s-1)
                rows2.append(r); cols2.append(ci); vals2.append(-v)

print(f"Component 1 (∂^h(f)): {len(rows2)} entries")

# Component 2: ∂^h(g) - ∂^c(f) ∈ Hom(B̄², B̄²)
# ∂^h(g)(a,b)[(j,k)] = a*g(b)[(j,k)] - g(ab)[(j,k)] + g(a)[(j,k)]*b
# where g: B̄→B̄², g(c) = Σ_{j,k} g_{jk}(c) * basis[j]⊗basis[k]
# 
# ∂^c(f)(a,b)[(j,k)] = a₁⊗f(a₂,b)[(j,k)] - Δ(f(a,b))[(j,k)] + f(a,b₁)⊗b₂[(j,k)]
# 
# This is getting very complex. Let me just build it numerically.

# Component 2 offset
off2 = n_bar**4  # first component has n_bar⁴ rows

# For the g-part (input columns n_h..n_h+n_c-1):
for ci_off, (j, k, c) in enumerate(c_rows):
    ci = n_h + ci_off  # global column index
    # ∂^h(g)(a,b)[(j,k)] = mult_bar[j,a,j']*g[j',k,b]... 
    # Actually: g(b) is a vector in B̄⊗B̄, indexed by (j,k).
    # a*g(b): a acts on the LEFT factor. (a*g(b))[(j,k)] = Σ_j' mult_bar[j,a,j'] * g[j',k](b)
    # g(ab): ab = Σ_l mult_bar[l,a,b] * basis[l]. g(ab)[(j,k)] = Σ_l mult_bar[l,a,b] * g[j,k](l)
    # g(a)*b: b acts on the RIGHT factor. (g(a)*b)[(j,k)] = Σ_k' mult_bar[k,b,k'] * g[j,k'](a)
    # 
    # ∂^h(g)(a,b)[(j,k)] = Σ_j' mult_bar[j,a,j']*g[j',k](b) - Σ_l mult_bar[l,a,b]*g[j,k](l) + Σ_k' mult_bar[k,b,k']*g[j,k'](a)
    
    # The input column is (j, k, c) meaning g[j,k](c).
    # Term 1: mult_bar[j,a,j'] * g[j',k](b) → when j'=j... no, g[j',k](b) means the input is (j',k,b).
    # Our input is (j,k,c). So term 1 appears when b=c and j'=j: output (a, c, j, k) += mult_bar[j, a, j]... 
    # Wait, that's wrong. Let me re-read.
    # 
    # g: B̄ → B̄⊗B̄. g(c) = Σ_{j,k} g_{jk}(c) * (basis[j]⊗basis[k]).
    # As a vector, g is indexed by (j, k, c) with j,k,c ∈ B̄. dim = n_bar³ = 512.
    # 
    # ∂^h(g)(a,b) ∈ B̄⊗B̄. ∂^h(g)(a,b)[(j,k)] for j,k ∈ B̄.
    # (a*g(b))[(j,k)] = Σ_{j'} mult_bar[j, a, j'] * g_{j',k}(b) = Σ_{j'} mult_bar[j,a,j'] * g[j',k,b]
    # (g(ab))[(j,k)] = Σ_l mult_bar[l,a,b] * g_{jk}(l) = Σ_l mult_bar[l,a,b] * g[j,k,l]
    # (g(a)*b)[(j,k)] = Σ_{k'} mult_bar[k, b, k'] * g_{j,k'}(a) = Σ_{k'} mult_bar[k,b,k'] * g[j,k',a]
    # 
    # ∂^h(g)(a,b)[(j,k)] = Σ_{j'} mult_bar[j,a,j']*g[j',k,b] - Σ_l mult_bar[l,a,b]*g[j,k,l] + Σ_{k'} mult_bar[k,b,k']*g[j,k',a]
    
    # For input (j,k,c):
    # Term 1: appears when (j',k,b) = (j,k,c), i.e., b=c. Output (a,c,j,k) for each a.
    #   Coefficient: mult_bar[j_out, a, j] where j_out is the output j-index.
    #   Wait, the output is (a, b, j, k). And the term is mult_bar[j, a, j'] * g[j',k,b].
    #   When (j',k,b) = (j,k,c): b=c, j'=j. Output (a, c, j_out, k_out) where j_out=j, k_out=k.
    #   Coefficient: mult_bar[j, a, j] = ... no, mult_bar[j_out, a, j'] = mult_bar[j, a, j].
    #   But j is the INPUT j-index, and j_out = j (the output j-index equals the input j'-index... no.
    #   
    #   Let me re-index. The output is (a, b, j_out, k_out). The term is:
    #   Σ_{j'} mult_bar[j_out, a, j'] * g[j', k_out, b]
    #   This matches input (j, k, c) when j'=j, k_out=k, b=c.
    #   So output (a, c, j_out, k) with coefficient mult_bar[j_out, a, j].
    #   For each a, j_out: row = (a, c, j_out, k), col = ci.
    
    for a2 in B_bar:
        for j_out in B_bar:
            v = mult_bar.get((j_out, a2, j), 0)  # mult_bar[j_out, a, j']
            if abs(v) > 1e-13:
                r = off2 + (a2-1)*n_bar**3 + (c-1)*n_bar**2 + (j_out-1)*n_bar + (k-1)
                rows2.append(r); cols2.append(ci); vals2.append(v)
    
    # Term 2: -mult_bar[l,a,b]*g[j,k,l] → when l=c. Output (a, b, j, k) for each a, b.
    # Coefficient: -mult_bar[c, a, b]
    for a2 in B_bar:
        for b2 in B_bar:
            v = mult_bar.get((c, a2, b2), 0)
            if abs(v) > 1e-13:
                r = off2 + (a2-1)*n_bar**3 + (b2-1)*n_bar**2 + (j-1)*n_bar + (k-1)
                rows2.append(r); cols2.append(ci); vals2.append(-v)
    
    # Term 3: mult_bar[k,b,k']*g[j,k',a] → when (j,a) = (j,c). Output (a, b, j, k_out) for each b, k_out.
    # Wait: g[j,k',a] means input (j, k', a). Our input is (j, k, c). So k'=k, a=c.
    # Output (a_out, b, j, k_out) with coefficient mult_bar[k_out, b, k].
    for b2 in B_bar:
        for k_out in B_bar:
            v = mult_bar.get((k_out, b2, k), 0)
            if abs(v) > 1e-13:
                r = off2 + (c-1)*n_bar**3 + (b2-1)*n_bar**2 + (j-1)*n_bar + (k_out-1)
                rows2.append(r); cols2.append(ci); vals2.append(v)

print(f"Component 2 (∂^h(g)): {len(rows2)} entries (so far)")

# Also need -∂^c(f) for the f-part.
# ∂^c(f)(a,b)[(j,k)] = Σ_{a₁,a₂} Δ(a)_{a₁,a₂} * f(a₂,b)[j,k]... 
# Actually: ∂^c(f)(a,b) = a₁⊗f(a₂,b) - Δ(f(a,b)) + f(a,b₁)⊗b₂
# where Δ(a) = Σ a₁⊗a₂ and Δ(b) = Σ b₁⊗b₂.
# 
# a₁⊗f(a₂,b): the tensor product. As a B̄⊗B̄ element, (a₁⊗f(a₂,b))[(j,k)] = δ_{j,a₁} * f(a₂,b)[k].
# But f(a₂,b) ∈ B̄, so f(a₂,b)[k] = f[k, a₂, b] (the k-component of f at (a₂,b)).
# 
# Δ(f(a,b)): Δ applied to f(a,b) ∈ B̄. Δ(f(a,b)) = Σ_{j,k} Δ(f(a,b))_{j,k} * basis[j]⊗basis[k].
# f(a,b) = Σ_t f_t(a,b) * basis[t]. Δ(basis[t]) = Delta[t] = Σ_{j,k} delta_bar[(t,j,k)] * basis[j]⊗basis[k].
# So Δ(f(a,b))[(j,k)] = Σ_t f_t(a,b) * delta_bar[(t,j,k)] = Σ_t f[t,a,b] * delta_bar[(t,j,k)].
# 
# f(a,b₁)⊗b₂: similar to the first term but with b's comultiplication.
# (f(a,b₁)⊗b₂)[(j,k)] = f(a,b₁)[j] * δ_{k,b₂} = f[j,a,b₁] * δ_{k,b₂}.
# 
# So ∂^c(f)(a,b)[(j,k)] = Σ_{a₁,a₂} delta_bar[(a,a₁,a₂)] * f[j,a₂,b]... 
# Wait, the formula is a₁⊗f(a₂,b), not f(a₁,...).
# (a₁⊗f(a₂,b))[(j,k)] = δ_{j,a₁} * f(a₂,b)[k] = δ_{j,a₁} * f[k,a₂,b]
# Summing over (a₁,a₂) from Δ(a): Σ_{a₁,a₂} delta_bar[(a,a₁,a₂)] * δ_{j,a₁} * f[k,a₂,b]
# = delta_bar[(a,j,a₂)] * f[k,a₂,b] for each a₂ (summing over a₂).
# = Σ_{a₂} delta_bar[(a,j,a₂)] * f[k,a₂,b]

# For input (t,a,b) of f:
# ∂^c(f)(a',b')[(j,k)] = Σ_{a₂} delta_bar[(a',j,a₂)] * f[k,a₂,b'] - Σ_t delta_bar[(t,j,k)] * f[t,a',b'] 
#                        + Σ_{b₂} delta_bar[(b',b₁,b₂)] * f[j,a',b₁] * δ_{k,b₂}
# The first term contributes when (k,a₂,b') = (t,a,b): k=t, a₂=a, b'=b.
#   Output (a', b, j, t) with coefficient delta_bar[(a', j, a)] for each a', j.
# The second term contributes when (t,a',b') = (t,a,b): a'=a, b'=b.
#   Output (a, b, j, k) with coefficient -delta_bar[(t, j, k)] for each j, k.
# The third term: Σ_{b₁,b₂} delta_bar[(b',b₁,b₂)] * f[j,a',b₁] * δ_{k,b₂}
#   = delta_bar[(b', b₁, k)] * f[j, a', b₁] for each b₁.
#   Contributes when (j, a', b₁) = (t, a, b): j=t, a'=a, b₁=b.
#   Output (a, b', t, k) with coefficient delta_bar[(b', b, k)] for each b', k.

for ci, (t, a, b) in enumerate(h_rows):  # f-part input
    # Term 1: delta_bar[(a', j, a)] * f[t, a, b] → output (a', b, j, t)
    for a2 in B_bar:
        for j_out in B_bar:
            v = delta_bar.get((a2, j_out, a), 0)
            if abs(v) > 1e-13:
                r = off2 + (a2-1)*n_bar**3 + (b-1)*n_bar**2 + (j_out-1)*n_bar + (t-1)
                rows2.append(r); cols2.append(ci); vals2.append(v)
    
    # Term 2: -delta_bar[(t, j, k)] * f[t, a, b] → output (a, b, j, k)
    for j_out in B_bar:
        for k_out in B_bar:
            v = delta_bar.get((t, j_out, k_out), 0)
            if abs(v) > 1e-13:
                r = off2 + (a-1)*n_bar**3 + (b-1)*n_bar**2 + (j_out-1)*n_bar + (k_out-1)
                rows2.append(r); cols2.append(ci); vals2.append(-v)
    
    # Term 3: delta_bar[(b', b, k)] * f[t, a, b] → output (a, b', t, k)
    for b2 in B_bar:
        for k_out in B_bar:
            v = delta_bar.get((b2, b, k_out), 0)
            if abs(v) > 1e-13:
                r = off2 + (a-1)*n_bar**3 + (b2-1)*n_bar**2 + (t-1)*n_bar + (k_out-1)
                rows2.append(r); cols2.append(ci); vals2.append(v)

print(f"Component 2 (∂^h(g) - ∂^c(f)): {len(rows2)} entries (total)")

# Component 3: -∂^c(g) ∈ Hom(B̄, B̄³)
# ∂^c(g)(c) = c₁⊗g(c₂) - Δ(g(c)) + g(c₁)⊗c₂
# g: B̄→B̄², g(c) = Σ_{j,k} g_{jk}(c) * basis[j]⊗basis[k]
# 
# c₁⊗g(c₂): Δ(c) = Σ c₁⊗c₂. c₁⊗g(c₂) ∈ B̄⊗B̄⊗B̄.
# (c₁⊗g(c₂))[(j,k,l)] = δ_{j,c₁} * g(c₂)[(k,l)] = δ_{j,c₁} * g[k,l,c₂]
# Summing: Σ_{c₁,c₂} delta_bar[(c,c₁,c₂)] * δ_{j,c₁} * g[k,l,c₂]
# = Σ_{c₂} delta_bar[(c,j,c₂)] * g[k,l,c₂]
# 
# Δ(g(c)): Δ applied to each factor of g(c) = Σ_{j,k} g_{jk}(c) * basis[j]⊗basis[k].
# Δ₂(g(c)) = (Δ⊗id)(g(c)) ... or (id⊗Δ)(g(c))? 
# Actually, Δ on B̄⊗B̄ means applying Δ to ONE factor.
# The coalgebra differential ∂^c on Hom(B̄, B̄²) maps to Hom(B̄, B̄³).
# ∂^c(g)(c) = c₁⊗g(c₂) - Δ₂(g(c)) + g(c₁)⊗c₂
# where Δ₂ means applying Δ to the FIRST factor of B̄⊗B̄ to get B̄⊗B̄⊗B̄.
# 
# Δ₂(g(c))[(j,k,l)] = Σ_{j',j''} delta_bar[(j,j',j'')] * g_{j''k}(c) * δ_{l,?}...
# This is getting very complicated. Let me just compute it.

off3 = 2 * n_bar**4

# For input (j,k,c) of g:
for ci_off, (j, k, c) in enumerate(c_rows):
    ci = n_h + ci_off
    
    # Term 1: c₁⊗g(c₂) → Σ_{c₂} delta_bar[(c,j,c₂)] * g[k,l,c₂]
    # Contributes when c₂ matches the input c: c₂ = c.
    # Output (c_in, j, k, l) for each l.
    # Wait: (c₁⊗g(c₂))[(j,k,l)] = δ_{j,c₁} * g(c₂)[(k,l)]
    # Summing over (c₁,c₂): Σ delta_bar[(c,c₁,c₂)] * δ_{j,c₁} * g[k,l,c₂]
    # = delta_bar[(c,j,c₂)] * g[k,l,c₂] (setting c₁=j)
    # Contributes when c₂ = input c: output (j, k, l) with coefficient delta_bar[(c,j,c)]
    # Wait, that's only when c₂ = c. But the sum is over ALL c₂.
    # The input is (j, k, c) meaning g[j,k](c). So c₂ = c.
    # Output (j, k, l) for each l, with coefficient delta_bar[(c, j, c)]... 
    # No: delta_bar[(c, c₁, c₂)] where c₁=j and c₂=c.
    # But c₂ should be the INPUT c. So the coefficient is delta_bar[(c, j, c)].
    # Hmm, that doesn't seem right. Let me re-derive.
    # 
    # c₁⊗g(c₂): for Δ(c) = Σ_{c₁,c₂} delta_bar[(c,c₁,c₂)] * c₁⊗c₂,
    # c₁⊗g(c₂) = Σ_{c₁,c₂} delta_bar[(c,c₁,c₂)] * c₁⊗g(c₂)
    # (c₁⊗g(c₂))[(j,k,l)] = δ_{j,c₁} * g(c₂)[(k,l)] = δ_{j,c₁} * g[k,l,c₂]
    # Summing: Σ_{c₁,c₂} delta_bar[(c,c₁,c₂)] * δ_{j,c₁} * g[k,l,c₂]
    # = Σ_{c₂} delta_bar[(c,j,c₂)] * g[k,l,c₂]
    # 
    # For input (j,k,c): g[k,l,c] is the input when c₂=c.
    # So the contribution is: delta_bar[(c,j,c)] * output[(j,k,l)] for each l.
    # But wait, c₂ ranges over ALL B̄, and our input only covers one c₂ at a time.
    # The input column (j,k,c) represents g[j,k](c). The term Σ_{c₂} delta_bar[(c,j,c₂)] * g[k,l,c₂]
    # involves g[k,l,c₂] for ALL c₂, not just c₂=c.
    # 
    # So for input (j,k,c) (which is g[j,k](c)):
    # The term contributes to output (j_out, k_out, l_out, c_out) where:
    # - From c₁⊗g(c₂): the input provides g[k_out, l_out, c₂] = g[j,k,c] when k_out=j, l_out=k, c₂=c.
    #   Wait, g[c₂][(k,l)] = g[k,l,c₂]. Our input is g[j,k,c], so k_out=j, l_out=k, c₂=c.
    #   And the output index is (j_out=c₁, k_out, l_out) = (c₁, j, k).
    #   The coefficient is delta_bar[(c_in, c₁, c)] for each c₁.
    #   But c_in is the INPUT c (the argument of g), and c₁ is from Δ(c_in).
    #   So: output (c₁, j, k, ?) for each c₁, with coefficient delta_bar[(c, c₁, c)].
    #   The output index for Hom(B̄, B̄³) is (c, j, k, l) where c is the argument and (j,k,l) is the B̄³ component.
    #   
    #   Hmm, I'm getting confused with the indices. Let me just compute it numerically.
    #   The output of ∂^c(g) is in Hom(B̄, B̄³), indexed by (c_in, j, k, l).
    #   c_in is the INPUT argument (the element of B̄ at which we evaluate).
    #   (j,k,l) is the output component in B̄³.
    #   
    #   For the term c₁⊗g(c₂):
    #   ∂^c(g)(c_in)[(j,k,l)] = Σ_{c₂} delta_bar[(c_in, j, c₂)] * g[k,l,c₂]
    #   Our input is g[j,k,c] (meaning g[c][(j,k)] = g[j,k,c]).
    #   So the term contributes when (k,l,c₂) = (j,k,c): k_out=j, l_out=k, c₂=c.
    #   Wait, that's (k, l) = (j, k), meaning k_out=j, l_out=k.
    #   And the output is (c_in, j_out, k_out, l_out) = (c_in, j, j, k)? No...
    #   
    #   Let me re-read: g(c₂)[(k,l)] = g[k,l,c₂]. Our input is g[j,k,c].
    #   So g(c₂)[(k_out, l_out)] = g[k_out, l_out, c₂].
    #   This matches g[j,k,c] when k_out=j, l_out=k, c₂=c.
    #   The output is (c_in, j_out, k_out, l_out) = (c_in, j_out, j, k) for each j_out, c_in.
    #   And the coefficient is delta_bar[(c_in, j_out, c)].
    
    # This is getting too complex. Let me just SKIP component 3 for now
    # and compute H̃²_b with only components 1 and 2.
    # This gives an UPPER BOUND on dim H̃²_b.
    pass

# Build the sparse matrix with only components 1 and 2
n_c3_partial = 2 * n_bar**4
db2 = sparse.csr_matrix((vals2, (rows2, cols2)), shape=(n_c3_partial, n_c2), dtype=complex)

print(f"\n∂_b^2 (partial, 2 components): shape ({n_c3_partial}, {n_c2})")
print(f"nnz: {db2.nnz}")

# Compute rank
gram2 = (db2.conj().T @ db2).toarray()
ev2 = np.sort(np.abs(eigvalsh(gram2)))[::-1]
tol2 = max(db2.shape) * ev2[0] * 1e-10 if ev2[0] > 0 else 0
rank2 = int(np.sum(ev2 > tol2))
dim_ker2 = n_c2 - rank2

print(f"rank(∂_b^2) = {rank2}")
print(f"dim ker(∂_b^2) = {dim_ker2}")
print(f"dim im(∂_b^1) = {rank1}")
print(f"dim H̃²_b (partial, upper bound) = {dim_ker2} - {rank1} = {dim_ker2 - rank1}")

# Check d2 ∘ d1 = 0
prod = db2 @ db1
if hasattr(prod, 'toarray'): prod = prod.toarray()
print(f"||∂_b^2 ∘ ∂_b^1|| = {np.max(np.abs(prod)):.2e}")

if dim_ker2 - rank1 == 0:
    print(f"\n*** dim H̃²_b(B+(sl_2)) = 0 (partial) ***")
    print(f"*** This means ALL B+ cocycles lift to bialgebra cocycles ***")
    print(f"*** For sl_2: obstruction = 0, dim HH²(D) = 3 ✓ ***")
else:
    print(f"\n*** dim H̃²_b(B+(sl_2)) = {dim_ker2 - rank1} (partial, upper bound) ***")
    print(f"*** Need component 3 for exact answer ***")
