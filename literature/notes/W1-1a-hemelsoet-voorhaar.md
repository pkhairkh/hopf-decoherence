# W1-1a — Deep read of Hemelsoet–Voorhaar (arXiv:2104.05113)

- **Task ID**: W1-1a
- **Agent**: Sub-agent 1a (Wave 1, general-purpose)
- **Date**: 2025-08-07
- **Source**: `/home/z/my-project/hopf-decoherence/literature/texts/Hemelsoet-Voorhaar-HH-small-quantum.txt`
- **Paper**: N. Hemelsoet, R. Voorhaar, *On certain Hochschild cohomology groups for the small quantum group*, arXiv:2104.05113v1 [math.QA], 11 Apr 2021.

---

## 1. Scope of computation

**The paper computes Hochschild cohomology of BLOCKS of `u_q(g)`, not of the full algebra.**

The abstract is explicit:

> "We apply the sheaf cohomology BGG method developed by the authors and Lachowska-Qi to the computation of Hochschild cohomology groups of various **blocks** of the small quantum group."

The relevant block decomposition is recalled in Proposition 2.3:

> "There is a decomposition (as two-sided ideals) `u_q(g) ≅ ⊕_{λ∈S} u_λ(g)`"

so that `z(u_q(g)) ≅ ⊕_λ z(u_λ(g))`, and similarly for higher `HH^•`. The paper never sums over blocks to obtain `HH^•(u_q(g))` for the full algebra.

**Specifically for `g = sl_3` (= type A_2):**

| Section | Block | Degrees computed | Reference |
|---|---|---|---|
| §3 (Thm 3.2) | `u_1(sl_3)`, the non-trivial singular block (geometrically `T*P^2`, parabolic `P` with `SL_3/P ≅ P^2`) | **all `s ≥ 0`** (full ring, including `s = 0, 1, 2, 3, …`) | Thm 3.2 + Prop 3.14 |
| §5 (Prop 5.1) | `u_0(sl_3)`, the **principal block** | `s ∈ {3, 5, 7, 9, 11, 13, 15}` (odd) and `s ∈ {4, 6, 8, 10, 12, 14}` (even) | Prop 5.1 |

**Critical gap for our conjecture**: Proposition 5.1 *explicitly excludes* `s = 2` (the formula is stated for `4 ≤ s ≤ 14` even and `3 ≤ s ≤ 15` odd). The principal-block case `HH^2(u_0(sl_3))` is therefore **not in the paper's tables**.

For the singular block `u_1(sl_3)`, Theorem 3.2 *does* cover `s = 2` (the formula is stated for all `s ≥ 2`, plus separate `s = 0, 1` cases).

---

## 2. Method: BGG / sheaf-cohomology

The method (due to Lachowska–Qi [LQ16, LQ17]; algorithmic implementation by Hemelsoet–Voorhaar [HV21]) proceeds in three steps. It is **fundamentally different** from the bar complex used in our `scripts/verify_sl2_hh2.py`.

**Step 1 — Geometric reduction (Bezrukavnikov–Lachowska, Thm 2.5, [BL07]):**
For the principal block `u_0`,
> `HH^s_{C*}(Ñ) ≅ HH^s(u_0)`
where `Ñ = T*(G/B)` is the Springer resolution and the LHS is `C*`-equivariant Hochschild cohomology. For singular blocks one replaces `Ñ` by `Ñ_P = T*(G/P)` for an appropriate parabolic `P`.

**Step 2 — Sheaf-cohomology decomposition (Eq. (1)):**
> `HH^s_{C*}(Ñ) ≅ ⊕_{i+j+k=s} H^i(G/B, V_{j,k})`
where `V_{j,k} = G ×_B V_{j,k}` are explicit `G`-equivariant vector bundles on `G/B` (built from `Sym^•(u_p) ⊗ ∧^• u_p ⊗ ∧^• n_p`), and `k` is the `C*`-weight (an even integer). This gives `HH^s` a natural **bigrading** by `(i, j)`.

**Step 3 — BGG resolution (Prop 2.12, [LQ16]):**
> `Hom_G(L(λ), H^•(G/B, E)) ≅ H^•(BGG(E, λ))`
so sheaf cohomology is computed as the cohomology of the BGG complex `BGG^•(E, λ) := Hom_n(BGG^•(λ), E)_h`, whose terms are direct sums of weight spaces `E[w·λ]` and whose differentials are explicit monomials in `U(n)` (see Corollary 2.13 for the `sl_3` formula).

**Step 4 — Computational improvement (§2.4):** They realize `V_{j,k}` as `coker Δ` of an explicit map `Δ: T_{j,k} → M_{j,k}`, and compute the BGG differential on `M_{j,k}` (whose `p`-module structure is simpler) using a chosen section `σ` of `π: M_{j,k} → V_{j,k}` (Prop 2.20). This is what makes the algorithm practical for higher rank.

**Why this is infeasible with the bar complex** (which is what we use in `verify_sl2_hh2.py`): the bar complex for `u_q(sl_3)` at `ℓ = 3` would have `dim u_q(sl_3) = ℓ^{dim g} = 3^8 = 6561`, so `C^2(u_q(sl_3), C) ≅ Hom(k, u_q(sl_3)^{⊗ 2})` has dimension `6561^2 ≈ 4.3 × 10^7`. The BGG method instead works with small weight spaces of `g`-modules, so each computation involves matrices of size at most a few hundred.

---

## 3. Result for `sl_3` at `ℓ = 3`

**The paper does NOT directly state `dim_C HH^2(u_q(sl_3), C)` for the full small quantum group.** It does not even state `dim HH^2(u_0(sl_3))` for the principal block — that case is conspicuously absent from Proposition 5.1's range.

What IS in the paper for `s = 2`:

### `HH^2(u_1(sl_3))` (non-trivial singular block; Thm 3.2 with `s = 2m`, `m = 1`)

The table reads (cells listed as `(i+j, j-i) → representation`):

| cell | representation | dimension |
|---|---|---|
| `(0, 0)` `k = 2` | `L(1, 1)` | 8 |
| `(2, 2)` `k = 0` | `L(1, 0) ⊕ L(0, 1) ⊕ L(2, 1) ⊕ L(1, 2) ⊕ L(2, 1)` | 3+3+15+15+15 = 51 |
| `(4, 4)` `k = -2` | *invalid* (`k < 0`) | — |

plus their sl_2-translates (the table shows only the top-left half; the bottom-right half is filled by the sl_2-action `τ ∧ −` of Theorem 2.6).

So the visible primitives already give at least `dim HH^2(u_1(sl_3)) ≥ 8 + 51 = 59`, before counting sl_2-images.

### `HH^2(u_0(sl_3))` (principal block)

**Not in the paper.** Proposition 5.1's formula is stated only for `4 ≤ s ≤ 14` even. A naive extrapolation of the formula `L(m, m)` at `(i+j=0, j-i=0)` and the big middle expression `L(m, m−1) ⊕ L(m−1, m) ⊕ L(2m, m) ⊕ L(m, m+1) ⊕ L(m+1, m)` at `(i+j=2, j-i=2)` to `m = 1` (i.e. `s = 2`) would give `L(1, 1) ⊕ [L(1, 0) ⊕ L(0, 1) ⊕ L(2, 1) ⊕ L(1, 2) ⊕ L(2, 1)]` ≈ `8 + 51 = 59`-dim — but the authors explicitly restrict the range, so the formula may simply not extend to `s = 2`. **Honest reading: the case `s = 2` for the principal block is unknown from this paper.**

### Verdict on the conjecture `dim_C HH^2(u_q(sl_3), C) = 9`

**The paper is silent on this conjecture.** It does not state, verify, or refute it. The block-by-block data in the paper, if summed over the blocks of `u_q(sl_3)` at `ℓ = 3`, would need to be supplemented by:

1. An explicit computation of `HH^2(u_0(sl_3))` (not in the paper);
2. A determination of the block decomposition of `u_q(sl_3)` at `ℓ = 3` (the alcove `C = {λ ∈ P^+ : 0 ≤ (λ+ρ, θ∨) ≤ 3}` contains only `λ ∈ {0, ω_1, ω_2}`, and the `ℓ`-extended affine Weyl group orbits on `C` need to be computed);
3. An explicit sum `dim HH^2(u_q(sl_3)) = Σ_λ dim HH^2(u_λ)`.

The paper provides none of (1)–(3).

> **Caveat / red flag for the conjecture:** if (i) the principal-block formula in Prop 5.1 *does* extend to `s = 2`, (ii) the alcove analysis gives a block decomposition with at least one singular block `u_1(sl_3)` for which Thm 3.2 already yields `dim HH^2(u_1) ≥ 59`, and (iii) `HH^•(u_q(g)) = ⊕_λ HH^•(u_λ)` (which the paper does confirm), then `dim HH^2(u_q(sl_3)) ≥ 59 ≫ 9`, which would **falsify** the conjecture at `A_2, ℓ = 3`. This requires careful verification, but it is the most honest read of what the paper's data would imply.

---

## 4. Module structure: `H^•(u, C) = C[N]`-module

**`N` is the nilpotent cone of `g`** — the closed subvariety of nilpotent elements in `g`. The relevant result is the Ginzburg–Kumar theorem:

> **Theorem 3.10 ([GK93])**. There is an isomorphism of algebras `H^{2•}(u_q(g), C) ≅ C[N]`. Moreover `H^{2•+1}(u_q(g), C) = 0`.

This is the **Hopf-algebra cohomology** `H^•(u_q(g), C) = Ext_{u_q(g)}(C, C)` (trivial coefficients), **not** the Hochschild cohomology `HH^•(u_q(g)) = Ext_{u_q(g) ⊗ u_q(g)^op}(u_q(g), u_q(g))`. The two are distinct invariants.

Because `H^•(u_q(g), C) ≅ C[N]` acts on `HH^•(u_q(g))`, each block's `HH^•(u_λ)` is naturally a `C[N]`-module. The paper's §3.2 computes this `C[N]`-module structure explicitly for `u_1(sl_3)`:

> **Proposition 3.14.** The following tables give the even (resp. odd) Hochschild cohomology groups of `u_1(sl_3)`, as `C[Y]`-modules: [tables with entries `C[Y], C_0, C[Y] ⊕ N_1 ⊕ N_2 ⊕ C[Y]^+ ⊕ M_1 ⊕ M_2`, etc.]

Here `Y = O_p ⊂ N` is the Richardson orbit closure associated to the parabolic `P` (for `sl_3`, `Y` is the closure of the minimal nilpotent orbit), and `C[Ñ_P] ≅ C[Y]` lets one reduce the `C[N]`-module structure on `HH^•(u_λ)` to a `C[Y]`-module structure on the corresponding `H^•(Ñ_P, …)`. Theorem 3.11 ([LQ19]) gives the analogous `C[N]`-module structure for `u_0(sl_2)`:

> `HH^{2•+1}(u_0(sl_2)) ≅ C[N]^+[1] ⊕ C[N][-1]`

with `C[N]^+` the augmentation ideal, `[1]` the `s`-grading shift.

---

## 5. Connection to Lachowska–Qi conjectures

The paper confirms **three** conjectures of Lachowska–Qi, all of which concern **`HH^0`** (= the center `z`) of the principal block, **not** `HH^2` of the full algebra:

> **Conjecture 2.7 ([LQ16]).** As a bigraded vector space, there is an isomorphism `z(u_0(sl_m)) ≅ DC_m`. In particular, `dim z(u_0(sl_m)) = (m + 1)^{m−1}`.

> **Conjecture 2.8 ([LQ17]).** Let `g` be a semisimple Lie algebra with Weyl group `W`. Then `z(u_0(g))^g ≅ DC(W)`.

> **Conjecture 2.9 ([LQ16]).** Let `g = sl_n`. Then `z(u_q(sl_n)) = z(u_q(sl_n))^{sln}`.

(Here `DC_m = C[x_1, …, x_m, y_1, …, y_m] / I` is the double coinvariant algebra for the diagonal `S_m`-action, and `DC(W)` is its generalisation to any Weyl group.)

The paper's confirmations:
- **A_2 (= sl_3)**: already done in [LQ16] — confirmed `z(u_0(sl_3)) ≅ DC_3` (dimension `(3+1)^{3−1} = 16`).
- **G_2** (Thm 4.1): `dim z(u_0(G_2)) = 91`; confirms Conj 2.8 — "for all complex simple Lie algebras of rank 2".
- **B_3, C_3** (Thm 4.3): tables given; confirm Conj 2.8; `z(u_0(so_7))^{so_7} ≅ z(u_0(sp_6))^{sp_6}` (isomorphic Weyl groups).
- **A_4** (Thm 4.5): `z(u_0(sl_5)) ≅ DC_5`; confirms Conj 2.9.

The paper also proposes a new conjecture (Conjecture 2.10) on the dimension divisibility of non-trivial irreducible summands of `z(u_q(g))` by `h + 1` (Coxeter number + 1).

**Relation to our conjecture**: The LQ conjectures are about `HH^0` of the principal block (i.e. the center), expressed as a bigraded algebra isomorphism with the double coinvariant algebra. Our conjecture is about `dim HH^2` of the **full** algebra. The two are **structurally related but distinct**: LQ gives a full bigraded description of `HH^0(u_0)` (which determines `dim HH^0(u_0)` immediately), but says nothing about `HH^2(u_0)` or about the singular blocks. The paper's results on `HH^s` for `s ≥ 3` in type A_2 (Prop 5.1) and on `HH^s` for the singular block `u_1(sl_3)` (Thm 3.2) are the first such higher-Hochschild data, but they do not combine to give a verification of the `dim = 9` formula.

---

## 6. Direct relevance to our conjecture

Honest assessment: **(c) is the closest fit, with a partial contribution toward (b).**

### (a) Does the paper verify our conjecture at A_2? **No.**

The paper does not compute `dim_C HH^2(u_q(sl_3), C)`. It does not even state the conjecture. The case `s = 2` of the principal block is missing from Prop 5.1's explicit range, and the block-by-block data are not summed to give the full algebra's `HH^2`.

### (b) Provides a partial result that combined with other work would verify A_2? **Partially yes — one ingredient is present, two are missing.**

**Present in the paper**:
- `HH^2(u_1(sl_3))` is given (Thm 3.2, `m = 1`). This is the singular block of `sl_3` (corresponding to the maximal parabolic `P` with `G/P ≅ P^2`). The result is a `g`-module decomposition, but it can be summed to give a vector-space dimension.
- The block-decomposition principle `HH^•(u_q(g)) ≅ ⊕_λ HH^•(u_λ)` (Prop 2.3 + the discussion under "decomposition of the center") is explicit in the paper, so the *strategy* of summing over blocks is endorsed.

**Missing** (would need to be supplied by other work):
1. `HH^2(u_0(sl_3))` for the principal block. This is the single biggest gap — Prop 5.1 explicitly starts at `s = 4`. One would need to either (i) run their BGG-cohomology software (https://github.com/RikVoorhaar/bgg-cohomology) at `s = 2` for `g = sl_3`; or (ii) check whether the formula `L(m, m)` at `(i+j=0, j-i=0)` and the middle expression `L(m, m−1) ⊕ L(m−1, m) ⊕ L(2m, m) ⊕ L(m, m+1) ⊕ L(m+1, m)` at `(i+j=2, j-i=2)` extends to `m = 1` (the authors' explicit range-bound `4 ≤ s` suggests it might not); or (iii) perform a direct bar-complex computation restricted to the principal block (much smaller than the full algebra).
2. The block decomposition of `u_q(sl_3)` at `ℓ = 3`: enumerate the orbits of the `ℓ`-extended affine Weyl group on the alcove `C = {0, ω_1, ω_2}`. For `sl_3` at `ℓ = 3 = h` (Coxeter number), this requires a careful linkage-principle computation; the Steinberg weight `(ℓ−1)ρ = 2ρ` has `(λ+ρ, θ∨) = 6 > 3`, so there is **no Steinberg block** at `ℓ = 3`. It is plausible (but not stated in the paper) that there is just one block, in which case `HH^2(u_q(sl_3)) = HH^2(u_0)` — but this needs verification.
3. If there are multiple blocks, account for the `ℤ/2`-Dynkin symmetry `ω_1 ↔ ω_2` to deduce `u_{ω_1} ≅ u_{ω_2}` and hence `dim HH^2(u_q(sl_3)) = dim HH^2(u_0) + 2 · dim HH^2(u_1)`.

### (c) Is it about a different object? **Yes, in scope.**

The paper is about **block cohomology** (`HH^•(u_λ)` for fixed `λ`), not about **full-algebra cohomology** (`HH^•(u_q(g))`). These are related by direct sum (`HH^•(u_q(g)) = ⊕_λ HH^•(u_λ)`), but the paper never performs the summation. Additionally:

- The paper's main novelty for `sl_3` (Thm 3.2) concerns the **singular block** `u_1`, not the principal block `u_0` (whose center was already treated in [LQ16]).
- The paper's principal-block higher-Hochschild results (Prop 5.1) start at `s = 3`/`s = 4`, leaving `HH^2(u_0(sl_3))` untouched.
- The `C[N]`-module structure (Prop 3.14) is additional structural data; it does not directly yield a number `dim HH^2(u_q(sl_3))`.

---

## 7. Citable lemma/theorem

Two results are worth citing verbatim. The first is the foundational Ginzburg–Kumar theorem (used heavily in §3.2); the second is the paper's main new result for `sl_3`.

### Theorem 3.10 (Ginzburg–Kumar, [GK93]) — verbatim from arXiv:2104.05113

> **Theorem 3.10 ([GK93])**. There is an isomorphism of algebras `H^{2•}(u_q(g), C) ≅ C[N]`. Moreover we have that `H^{2•+1}(u_q(g), C) = 0`.

(Cited as Theorem 3.10 in arXiv:2104.05113v1 [math.QA], p. 14; original source: V. Ginzburg and S. Kumar, *Cohomology of quantum groups at roots of unity*, Duke Math. J. **69** (1993), 179–198.)

### Theorem 3.2 (Hemelsoet–Voorhaar) — main sl_3 result, verbatim

> **Theorem 3.2.** Let `u^p_λ(sl_3)` be the block of the small quantum group corresponding to `p`. For `s ≥ 2`, `HH^s(u^p_λ(sl_3))` is given by the following tables:
>
> For `s = 2m + 1` odd:
> ```
>           i+j=1                          i+j=3
> j−i=1   L_{m,m} ⊕ L_{m+1,m} ⊕           0
>         L_{m,m+1} ⊕ L_{m+1,m+1}
> j−i=3   L_{m,m} ⊕ L_{m+1,m} ⊕
>         L_{m,m+1} ⊕ L_{m+1,m+1}
> ```
>
> For `s = 2m` even:
> ```
>           i+j=0            i+j=2                                                    i+j=4
> j−i=0    L_{m,m}          0                                                        0
> j−i=2    0                L_{m,m−1} ⊕ L_{m−1,m} ⊕ L_{2m,m} ⊕                     0
>                           L_{m,m+1} ⊕ L_{m+1,m}
> j−i=4    0                0                                                        L_{m,m}
> ```
>
> For `s = 1`:
> ```
>           i+j=1            i+j=3
> j−i=1    L_{0,0} ⊕ L_{1,1}    L_{0,0}
> j−i=3    L_{0,0} ⊕ L_{1,1}
> ```

(Cited as Theorem 3.2 in arXiv:2104.05113v1, p. 10. For `s = 0` they refer to [LQ17].)

### Also citable — Conjectures 2.7–2.9 (Lachowska–Qi)

If we want to position our `HH^2` conjecture relative to the existing LQ conjectures on `HH^0`:

> **Conjecture 2.7 ([LQ16]).** As a bigraded vector space, there is an isomorphism `z(u_0(sl_m)) ≅ DC_m`. In particular, `dim z(u_0(sl_m)) = (m + 1)^{m−1}`.
>
> **Conjecture 2.8 ([LQ17]).** Let `g` be a semisimple Lie algebra with Weyl group `W`. Then `z(u_0(g))^g ≅ DC(W)`.
>
> **Conjecture 2.9 ([LQ16]).** Let `g = sl_n`. Then `z(u_q(sl_n)) = z(u_q(sl_n))^{sln}`.

---

## Bottom line

**The Hemelsoet–Voorhaar paper does NOT verify (and does not refute) our conjecture `dim_C HH^2(u_q(sl_3), C) = 9`.** It computes `HH^•` of *individual blocks* of `u_q(sl_3)`:

- `HH^s(u_0(sl_3))` for `s ≥ 3` (Prop 5.1) — **`s = 2` is explicitly excluded**;
- `HH^s(u_1(sl_3))` for all `s ≥ 0` (Thm 3.2) — this *does* include `s = 2`, and the answer is a large `g`-module (`≥ 59`-dimensional, before sl_2-symmetry).

The paper provides one of the three ingredients needed to verify our conjecture at `A_2, ℓ = 3` (namely the singular-block answer); the principal-block `s = 2` answer and the block-decomposition enumeration are both missing. The conjecture predicts a number (9) that is **much smaller** than the singular-block dimension alone (`≥ 59`), which suggests either (i) the principal-block formula in Prop 5.1 does *not* extend to `s = 2` (cancellations at low degree), (ii) the block decomposition at `ℓ = 3 = h` is degenerate in a way that removes the singular block, or (iii) the conjecture as stated may not hold at `ℓ = h`. The paper itself is silent on all of this.

---

## Recommended action for the orchestrator

1. **Trigger a follow-up task (W1-1a-α) to compute `HH^2(u_0(sl_3))` directly.** Two routes:
   - **Bar-complex route (smaller than full algebra)**: restrict the bar complex to the principal block `u_0(sl_3) ⊂ u_q(sl_3)`. The principal block is much smaller than the full algebra (its dimension is roughly the size of the projective cover of the trivial module, not `3^8 = 6561`), so this is computationally tractable. Adapt `scripts/verify_sl2_hh2.py` to `sl_3` at `ℓ = 3` restricted to the principal block.
   - **BGG-software route**: clone and run https://github.com/RikVoorhaar/bgg-cohomology at `s = 2`, `g = sl_3`, principal block. The authors state in §5.1 that "in type A_2 we could go up to `s ≤ 13`" — so `s = 2` is within reach of their code (the explicit exclusion in Prop 5.1 is a statement about the *formula's* uniformity, not about the algorithm's capability).

2. **Trigger a separate task (W1-1a-β) to determine the block decomposition of `u_q(sl_3)` at `ℓ = 3`.** Compute the orbits of the `ℓ`-extended affine Weyl group on the alcove `C = {0, ω_1, ω_2}`. If the three weights form a single orbit (i.e. only the principal block exists at `ℓ = 3 = h`), then `HH^2(u_q(sl_3)) = HH^2(u_0)` and the singular-block answer from Thm 3.2 is irrelevant to the conjecture at this `ℓ`. If there are multiple blocks, the `ℤ/2`-symmetry gives `u_{ω_1} ≅ u_{ω_2}`.

3. **Independently, also test the conjecture at `ℓ = 5` or `ℓ = 7` (i.e. `ℓ > h = 3`).** This avoids the degenerate `ℓ = h` regime and gives the block decomposition more room. The Hemelsoet–Voorhaar results are stated for general (admissible) `ℓ`, so they apply at any odd `ℓ > h`.

4. **Cite Theorem 3.10 (Ginzburg–Kumar) and Theorem 3.2 (Hemelsoet–Voorhaar) in our paper.** The first is the canonical reference for the `H^•(u, C) = C[N]` algebra structure (which we will need when discussing the `HH^•(u_λ)`-as-`C[N]`-module picture); the second is the only existing computation of `HH^2` for a singular block of `u_q(sl_3)`.

5. **Flag the apparent tension.** If the principal-block formula does extend to `s = 2` (giving `dim HH^2(u_0) ≈ 59`), our conjecture is in serious trouble at `A_2, ℓ = 3`. The orchestrator should commission the direct verification in (1) before declaring the conjecture "verified at A_2" in the paper.
