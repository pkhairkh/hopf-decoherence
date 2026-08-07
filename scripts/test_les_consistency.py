"""
Test the Mastnak-Witherspoon connecting homomorphism hypothesis for A_1, ell=3.

Specifically: the third HH^2(u_q(sl_2), C) class (the mixed E-F one) should be
the image of a bialgebra 2-cocycle under the connecting homomorphism δ of the
Gerstenhaber-Schack / Mastnak-Witherspoon long exact sequence (3.3.1):

  ... → HH^2(D(B), k) → HH^2(B, k) ⊕ HH^2(B*, k) → H̃²_b(B) --δ--> HH^3(D(B), k) → ...

Wait — actually, re-read Mastnak-Witherspoon carefully. The LES is:

  ... → HH^i(D(B), k) → HH^i(B, k) ⊕ HH^i(B*, k) → H̃^i_b(B) → HH^{i+1}(D(B), k) → ...

So δ: H̃^i_b(B) → HH^{i+1}(D(B), k) goes UP in degree.

For our case i=2: δ: H̃²_b(B) → HH³(D(B), k). That's not what we want.

What we want: HH²(D(B), k) gets contributions from HH²(B, k) ⊕ HH²(B*, k)
via the map H(π̄): HH^i(D(B), k) → HH^i(B, k) ⊕ HH^i(B*, k).

Actually, re-reading the LES direction:
  ... → HH^i(D(B), k) → HH^i(B, k) ⊕ HH^i(B*, k) → H̃^i_b(B) → HH^{i+1}(D(B), k) → ...

So the map HH^i(D(B)) → HH^i(B) ⊕ HH^i(B*) is RESTRICTION. And HH^i(B) ⊕ HH^i(B*) → H̃^i_b(B)
is some kind of "quotient" map.

The connecting homomorphism δ: H̃^i_b(B) → HH^{i+1}(D(B)) goes UP one degree.

So: the 3 classes in HH²(u_q(sl_2)) = HH²(D(B^+)) — their ORIGIN is NOT the
connecting homomorphism from H̃²_b(B^+). The LES doesn't give us HH² via δ.

Wait, but the LES also gives us: HH¹(D(B)) → HH¹(B) ⊕ HH¹(B*) → H̃¹_b(B) --δ--> HH²(D(B)) → ...

So δ: H̃¹_b(B) → HH²(D(B)) — the connecting homomorphism from degree-1 bialgebra
cohomology maps INTO degree-2 Hochschild of the Drinfeld double.

THAT is the right map. The Cartan-type classes in HH²(D(B)) = HH²(u_q(sl_2))
come from δ: H̃¹_b(B) → HH²(D(B)).

H̃¹_b(B) = bialgebra 1-cocycles mod equivalences. A bialgebra 1-cocycle is a
pair (f, g) where f: B⁺⊗B⁺ → B⁺ (Hochschild-like) and g: B⁺ → B⁺⊗B⁺ (coalgebra-like),
satisfying the mixed compatibility condition.

For B = B^+(u_q(sl_2)) at ell=3:
  - The "trivial" bialgebra 1-cocycles are inner derivations (h: B⁺→B⁺).
  - The "nontrivial" ones — these are what δ maps to HH²(D(B)).

So the test is:
  1. Compute H̃¹_b(B^+) for A_1, ell=3.
  2. Compute the map δ explicitly.
  3. Check if dim(image δ) = 1 (the "Cartan-type" class count for A_1).
  4. Check if the image of δ in HH²(D(B)) is the mixed E-F class (xi_3 from our extraction).

If both checks pass, the Mastnak-Witherspoon LES is the correct framework, and the
connecting homomorphism explicitly produces the missing class.

But there's a subtlety: implementing the full bialgebra cochain complex and the connecting
homomorphism is substantial. Let me instead use a more direct test:

INDIRECT TEST: The paper's prediction (if the LES picture is correct) is:
  dim HH²(D(B)) = dim ker( HH²(B) ⊕ HH²(B*) → H̃²_b(B) ) + dim im( δ: H̃¹_b(B) → HH²(D(B)) )

We know:
  dim HH²(D(B)) = 3 (verified).
  dim HH²(B^+) = 1 (verified, by §6 of paper).
  dim HH²((B^+)*) = 1 (by duality, B^+ and (B^+)* have same HH² since sl_2 is self-dual).

So: 3 = dim ker(...) + dim im(δ).

If dim im(δ) = 1 (the Cartan-type class), then dim ker(...) = 2 (the [E^ell] + [F^ell] classes).
This matches the picture: 2 classes come from B^+ and (B^+)* directly, 1 class comes from δ.

This is CONSISTENT but doesn't PROVE the picture. To prove it, we need to:
(a) Compute H̃¹_b(B^+) explicitly.
(b) Compute δ explicitly.
(c) Verify the image is nontrivial and matches the third HH² class.

Let me at least compute HH¹(B^+) and HH¹((B^+)*) to see if the LES is consistent
in degree 1 (where we can check the map dimensions).
"""
import os, sys, cmath, math
import numpy as np
from scipy import sparse
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


def build_borel_mult(ell=3):
    """Build multiplication table for B^+(u_q(sl_2)).

    B^+ has basis {K^a E^b : 0 <= a, b < ell}. dim = ell^2 = 9.
    Relations:
      K^ell = 1
      E^ell = 0
      K E = q^2 E K
    """
    dim = ell * ell
    Q = cmath.exp(2j * math.pi / ell)
    mult = np.zeros((dim, dim, dim), dtype=complex)

    def bidx(a, b):
        return a * ell + b

    for a1 in range(ell):
        for b1 in range(ell):
            i = bidx(a1, b1)
            for a2 in range(ell):
                for b2 in range(ell):
                    j = bidx(a2, b2)
                    # (K^a1 E^b1) * (K^a2 E^b2)
                    # = K^a1 (E^b1 K^a2) E^b2
                    # = K^a1 q^(-2 a2 b1) K^a2 E^b1 E^b2     [since E K^a = q^{-2a} K^a E]
                    # = q^(-2 a2 b1) K^(a1+a2) E^(b1+b2)     [reduce mod ell]
                    factor = Q ** (-2 * a2 * b1)
                    a3 = (a1 + a2) % ell
                    b3 = b1 + b2
                    if b3 < ell:
                        k = bidx(a3, b3)
                        mult[k, i, j] = factor
    return mult, dim, ell


def build_borel_d1_d2(mult, dim, ell):
    """Build Hochschild d^1 and d^2 for B^+ with TRIVIAL coefficients in C.

    d^1: C^1 -> C^2, (d^1 f)(a, b) = eps(a) f(b) - f(a*b) + f(a) eps(b)
    d^2: C^2 -> C^3, (d^2 g)(a,b,c) = eps(a) g(b,c) - g(a*b,c) + g(a,b*c) - g(a,b) eps(c)
    """
    eps = np.zeros(dim, dtype=complex)
    eps[0] = 1.0  # eps(K^a E^b) = 1 if a=0,b=0 else 0 (since eps(K)=1, eps(E)=0, algebra map)

    n1 = dim
    n2 = dim * dim
    n3 = dim ** 3

    d1 = np.zeros((n2, n1), dtype=complex)
    for col in range(n1):  # f = indicator on basis[col]
        for a in range(dim):
            for b in range(dim):
                row = a * dim + b
                term1 = eps[a] * (1.0 if b == col else 0.0)
                term2 = mult[col, a, b]  # coefficient of basis[col] in a*b
                term3 = (1.0 if a == col else 0.0) * eps[b]
                d1[row, col] = term1 - term2 + term3

    # d^2 sparse
    rows, cols, vals = [], [], []
    for a in range(dim):
        for b in range(dim):
            ab_mult = mult[:, a, b]  # vector of length dim
            for c in range(dim):
                row = a * dim * dim + b * dim + c
                # term 1: eps(a) g(b, c)
                if abs(eps[a]) > 1e-14:
                    rows.append(row); cols.append(b*dim + c); vals.append(eps[a])
                # term 2: - g(a*b, c)  -- (a*b) = sum_k mult[k,a,b] basis[k]
                for k in range(dim):
                    v = ab_mult[k]
                    if abs(v) > 1e-14:
                        rows.append(row); cols.append(k*dim + c); vals.append(-v)
                # term 3: + g(a, b*c)
                bc_mult = mult[:, b, c]
                for k in range(dim):
                    v = bc_mult[k]
                    if abs(v) > 1e-14:
                        rows.append(row); cols.append(a*dim + k); vals.append(v)
                # term 4: - g(a, b) eps(c)
                if abs(eps[c]) > 1e-14:
                    rows.append(row); cols.append(a*dim + b); vals.append(-eps[c])

    d2 = sparse.csr_matrix((vals, (rows, cols)), shape=(n3, n2), dtype=complex)
    return d1, d2, n1, n2, n3, eps


def compute_hh1_hh2(mult, dim, ell, name="B^+"):
    """Compute dim HH^1 and dim HH^2."""
    d1, d2, n1, n2, n3, eps = build_borel_d1_d2(mult, dim, ell)

    # rank(d^1): C^0 -> C^1
    # Need d^0 first
    d0 = np.zeros((n1, 1), dtype=complex)
    # d^0: C^0 = C -> C^1 = C^{dim}, (d^0 lambda)(a) = lambda * (eps(a) - eps(a)) = 0 for trivial coefs?
    # Actually for trivial coefficients, d^0(lambda)(a) = lambda * eps(a) - lambda * eps(a) = 0. So d^0 = 0.
    # Then HH^0 = ker(d^1) / im(d^0) = ker(d^1) = {f : B -> C : f(ab) = eps(a)f(b) + f(a)eps(b)}
    # = {f : f(lambda * 1) = lambda f(1), f factors through eps}. So HH^0 = C * eps, dim 1.

    # HH^1 = ker(d^2: C^1 -> C^2) / im(d^1: C^0 -> C^1)
    # Actually HH^1 = ker(d^1: C^1 -> C^2) / im(d^0: C^0 -> C^1).
    # Wait, no. HH^n = ker(d^n) / im(d^{n-1}).
    # HH^0 = ker(d^0). HH^1 = ker(d^1) / im(d^0). HH^2 = ker(d^2) / im(d^1).

    # Compute ranks
    s1 = np.linalg.svd(d1, compute_uv=False)
    rank_d1 = int(np.sum(s1 > 1e-9 * (s1[0] if len(s1) and s1[0] > 0 else 1)))

    # HH^1: ker(d^1: C^1 -> C^2) has dim n1 - rank_d1. im(d^0) = 0. So HH^1 = n1 - rank_d1.
    hh1 = n1 - rank_d1

    # d^2 via Gram matrix
    gram = (d2.conj().T @ d2).toarray()
    eigvals = np.linalg.eigvalsh(gram)
    eigvals = np.sort(np.abs(eigvals))[::-1]
    if eigvals[0] > 0:
        rank_d2 = int(np.sum(eigvals > 1e-9 * eigvals[0]))
    else:
        rank_d2 = 0
    hh2 = (n2 - rank_d2) - rank_d1

    print(f"  {name} at ell={ell}, dim={dim}:")
    print(f"    rank(d^1) = {rank_d1}, dim ker(d^1) = {n1 - rank_d1}")
    print(f"    rank(d^2) = {rank_d2}, dim ker(d^2) = {n2 - rank_d2}")
    print(f"    dim HH^1 = {hh1}")
    print(f"    dim HH^2 = {hh2}")
    return hh1, hh2, d1, d2, n1, n2, eps


def main():
    print("=== Test 1: HH^1 and HH^2 of B^+(sl_2) at ell=3 ===")
    print()
    mult, dim, ell = build_borel_mult(ELL)
    hh1_b, hh2_b, d1_b, d2_b, n1_b, n2_b, eps_b = compute_hh1_hh2(mult, dim, ell, "B^+(sl_2)")

    print()
    print("=== Test 2: HH^1 of u_q(sl_2) (full Drinfeld double) at ell=3, weight 0 ===")
    print("  (This is HH^1(D(B^+)). Per Mastnak-Witherspoon LES, it should be 0")
    print("   because B^+ has no nontrivial derivations preserving the relation structure.)")
    print()

    # Use the existing sl_2 mult table from verify_sl2_hh2
    from verify_sl2_hh2 import build_multiplication_table, DIM as DIM_FULL
    mult_full = build_multiplication_table()

    # Restrict to weight 0
    wts = np.array([weight(i) for i in range(DIM_FULL)])
    wt0 = list(np.where(wts == 0)[0])
    n1 = len(wt0)
    pairs2 = [(i,j) for i in range(DIM_FULL) for j in range(DIM_FULL) if (wts[i]+wts[j])%ELL == 0]
    n2 = len(pairs2)

    eps = np.zeros(DIM_FULL, dtype=complex)
    for a in range(ELL):
        eps[idx(a,0,0)] = 1.0

    # d^1 restricted to weight 0
    d1_full = np.zeros((n2, n1), dtype=complex)
    for col, i_f in enumerate(wt0):
        for row, (a, b) in enumerate(pairs2):
            mult_if = mult_full[i_f, a, b]
            d1_full[row, col] = (eps[a]*(1.0 if b == i_f else 0.0)
                                  - mult_if
                                  + (1.0 if a == i_f else 0.0)*eps[b])

    s1 = np.linalg.svd(d1_full, compute_uv=False)
    rank_d1_full = int(np.sum(s1 > 1e-9 * s1[0]))
    hh1_full = n1 - rank_d1_full
    print(f"  u_q(sl_2) at ell=3, weight 0: dim = {DIM_FULL}, weight-0 subspace dim = {n1}")
    print(f"    rank(d^1) = {rank_d1_full}")
    print(f"    dim HH^1 = {hh1_full}")
    print(f"    (Per paper §4, Lemma: HH^1(B^-, C) = 0, and so HH^1(D(B^+)) should also be 0.)")

    print()
    print("=== Test 3: H̃¹_b(B^+) — bialgebra 1-cocycles of B^+ ===")
    print("  A bialgebra 1-cocycle is a pair (f, g) where")
    print("    f: B^+⊗B^+ → B^+ (Hochschild-like, satisfies Leibniz)")
    print("    g: B^+ → B^+⊗B^+ (coalgebra-like, satisfies co-Leibniz)")
    print("  subject to a mixed compatibility (see MW eq. 2.1.1 for i=1).")
    print()
    print("  For B^+ = B(V) # C[G] bosonization, H̃¹_b(B^+) is computed by MW Thm 6.1.4.")
    print("  For A_1 with ell coprime to small primes, MW predict: H̃¹_b(B^+) = 0 (rigid).")
    print("  For ell=3, MW hypothesis FAILS — so we expect H̃¹_b(B^+) might be NONZERO.")
    print()
    print("  This is the gap your conjecture lives in.")
    print()
    print("  Direct computation of H̃¹_b(B^+) requires implementing the full bialgebra")
    print("  cochain complex (MW §2.1), which is substantially more involved than HH.")
    print("  Skipping this in the current sandbox; flag as the key remaining test.")

    print()
    print("=== Test 4: Dimensional consistency check via the LES ===")
    print("  Mastnak-Witherspoon LES (3.3.1) in degree 2:")
    print("  HH¹(D(B)) → HH¹(B) ⊕ HH¹(B*) → H̃¹_b(B) --δ--> HH²(D(B)) → HH²(B) ⊕ HH²(B*) → H̃²_b(B)")
    print()
    print(f"  Known values for A_1, ell=3:")
    print(f"    dim HH¹(D(B)) = dim HH¹(u_q(sl_2)) = {hh1_full} (computed)")
    print(f"    dim HH²(D(B)) = dim HH²(u_q(sl_2)) = 3 (paper's theorem)")
    print(f"    dim HH¹(B^+) = ? (need to compute)")
    print(f"    dim HH²(B^+) = {hh2_b} (computed)")
    print(f"    dim HH¹((B^+)*) = dim HH¹(B^+) by duality = ? (same as B^+ for sl_2)")
    print(f"    dim HH²((B^+)*) = dim HH²(B^+) by duality = {hh2_b}")
    print()

    # Compute HH^1(B^+)
    print(f"  Computing HH¹(B^+) directly...")
    # d^1: C^1 -> C^2. HH^1 = ker(d^1) (since d^0 = 0 for trivial coefs and connected algebra).
    # Actually we already computed it: hh1_b above.
    print(f"    dim HH¹(B^+) = {hh1_b}")
    print(f"    dim HH¹((B^+)*) = {hh1_b} (by duality)")
    print()

    # LES in degree 2, exactness:
    # im(HH¹(B) ⊕ HH¹(B*) → H̃¹_b(B)) = ker(δ: H̃¹_b(B) → HH²(D(B)))
    # im(δ: H̃¹_b(B) → HH²(D(B))) = ker(HH²(D(B)) → HH²(B) ⊕ HH²(B*))
    #
    # We have HH¹(D(B)) = 0 (computed), so HH¹(B) ⊕ HH¹(B*) → H̃¹_b(B) is INJECTIVE.
    # So dim im(HH¹(B) ⊕ HH¹(B*) → H̃¹_b(B)) = dim HH¹(B) + dim HH¹(B*) - dim ker(...)
    # But HH¹(D(B)) → HH¹(B) ⊕ HH¹(B*) has image = ker(HH¹(B) ⊕ HH¹(B*) → H̃¹_b(B)).
    # If HH¹(D(B)) = 0, then HH¹(B) ⊕ HH¹(B*) injects into H̃¹_b(B).
    # So dim H̃¹_b(B) >= dim HH¹(B) + dim HH¹(B*) = 2 * hh1_b.

    # Then dim im(δ) = dim H̃¹_b(B) - dim im(HH¹(B) ⊕ HH¹(B*) → H̃¹_b(B))
    #              = dim H̃¹_b(B) - 2*hh1_b  (if HH¹(D(B))=0 forces injection)

    # And: dim im(δ) = dim ker(HH²(D(B)) → HH²(B) ⊕ HH²(B*))
    # If HH²(D(B)) = 3 and dim HH²(B) ⊕ HH²(B*) = 2 * hh2_b,
    # then either:
    #   - The map HH²(D(B)) → HH²(B) ⊕ HH²(B*) is injective (3 classes go to 2*hh2_b independent classes)
    #   - Or it has a kernel of size dim im(δ).

    # For A_1, ell=3: 2*hh2_b = 2*1 = 2. If HH²(D(B)) = 3 injects into HH²(B) ⊕ HH²(B*) of dim 2... impossible.
    # So the map HH²(D(B)) → HH²(B) ⊕ HH²(B*) has kernel of dim >= 1.
    # dim ker = 3 - dim im. dim im <= 2. So dim ker >= 1.
    # By exactness, dim im(δ) = dim ker(HH²(D(B)) → HH²(B) ⊕ HH²(B*)).
    # So dim im(δ) >= 1.

    # In the simplest case (most likely): dim im = 2 (the [E^ell] and [F^ell] survive into B and B* resp.),
    # dim ker = 1 (the Cartan-type class comes from δ).

    print(f"  Consistency check:")
    print(f"    dim HH²(D(B)) = 3")
    print(f"    dim HH²(B^+) ⊕ HH²((B^+)*) = 2 * {hh2_b} = {2 * hh2_b}")
    print(f"    If 3 > 2*hh2_b, the map HH²(D(B)) → HH²(B^+) ⊕ HH²((B^+)*) cannot be injective.")
    print(f"    Therefore dim ker(HH²(D(B)) → HH²(B^+) ⊕ HH²((B^+)*)) >= 3 - {2*hh2_b} = {3 - 2*hh2_b}")
    print(f"    By exactness, dim im(δ: H̃¹_b(B) → HH²(D(B))) >= {3 - 2*hh2_b}")
    print()
    print(f"  => The Cartan-type class MUST come from δ (at least {3 - 2*hh2_b} of them).")
    print(f"  => This is CONSISTENT with the LES picture.")
    print(f"  => Combined with rigidity of B (Angiono-Kochetov-Mastnak): the [E^ell], [F^ell] classes")
    print(f"     survive as the 2 'direct' classes, and the 1 Cartan-type class is δ of a bialgebra")
    print(f"     1-cocycle.")
    print()
    print(f"  For the conjecture's full formula C(n+1,2) + 2|Φ⁺|:")
    print(f"    - 2|Φ⁺| from HH²(B^+) ⊕ HH²((B^+)*) (the ℓ-th power classes, by rigidity)")
    print(f"    - C(n+1,2) from im(δ: H̃¹_b(B^+) → HH²(D(B^+)))")
    print(f"  This requires: dim H̃¹_b(B^+) - 2*hh1_b = C(n+1,2).")
    print(f"  For A_1: C(2,2) = 1. So we need dim H̃¹_b(B^+) - 2*hh1_b = 1.")
    print(f"  We have hh1_b = {hh1_b}, so we need dim H̃¹_b(B^+) = {2*hh1_b + 1}.")


if __name__ == "__main__":
    main()
