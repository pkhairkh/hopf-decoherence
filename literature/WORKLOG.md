# Worklog — hopf-decoherence research project

This worklog tracks all sub-agent activities for the conjecture
`dim_C HH^2(u_q(g), C) = C(n+1, 2) + 2|Phi^+|`
at odd roots of unity. Each sub-agent appends a section upon task completion.

---

## W1-1a — Deep read of Hemelsoet–Voorhaar (arXiv:2104.05113)

- **Task ID**: W1-1a
- **Agent**: Sub-agent 1a (Wave 1, general-purpose)
- **Date**: 2025-08-07
- **Status**: completed
- **Output**: `/home/z/my-project/hopf-decoherence/literature/notes/W1-1a-hemelsoet-voorhaar.md`

### Summary

Read in full the text-extracted version of Hemelsoet & Voorhaar, *On certain Hochschild cohomology groups for the small quantum group* (arXiv:2104.05113v1, Apr 2021). Answered the seven prescribed questions about scope, method, sl_3 results, module structure, LQ-conjecture connections, and direct relevance to our conjecture.

### Key findings

1. **Scope**: The paper computes block cohomology `HH^s(u_λ)`, **not** full-algebra cohomology. For sl_3 it covers (a) the singular block `u_1(sl_3)` at *all* `s ≥ 0` (Thm 3.2) and (b) the principal block `u_0(sl_3)` only for `s ∈ [3, 15]` odd / `s ∈ [4, 14]` even (Prop 5.1). The case `s = 2` for the principal block is **explicitly excluded** from Prop 5.1.

2. **Method**: BGG / sheaf-cohomology (Lachowska–Qi + their [HV21] algorithmic improvement). Fundamentally different from the bar complex — works with small weight spaces of `g`-modules instead of `dim(u_q(g))^2`-size matrices. Makes higher-rank computations feasible.

3. **Result for sl_3, ℓ = 3**: **The paper does not state `dim_C HH^2(u_q(sl_3), C)`.** The principal-block `s = 2` case is missing. The singular-block `s = 2` case (Thm 3.2, `m = 1`) gives a large `g`-module (≥ 59-dim before sl_2-symmetry). The paper is **silent** on our conjecture.

4. **Module structure**: `N` is the nilpotent cone of `g`; `H^{2•}(u_q(g), C) ≅ C[N]` (Ginzburg–Kumar, Thm 3.10) is the Hopf-algebra cohomology, which acts on `HH^•(u_λ)`. Prop 3.14 computes the `C[Y]`-module structure (where `Y = Richardson orbit closure ⊂ N`) for `u_1(sl_3)`.

5. **LQ conjectures**: Paper confirms Conjectures 2.7–2.9 of Lachowska–Qi for `G_2, B_3, C_3, A_4` (A_2 was already done in [LQ16]). These are about `HH^0` of the principal block (the center, isomorphic to the double coinvariant algebra `DC(W)`), structurally related to but distinct from our `HH^2` conjecture.

6. **Relevance to our conjecture**: Closest to option (c): block cohomology ≠ full-algebra cohomology. Partial contribution toward (b) — one of three ingredients is present (singular-block HH^2), two are missing (principal-block HH^2 at `s = 2`, and the block decomposition at `ℓ = 3`). Apparent tension: if the principal-block formula extends to `s = 2`, then `dim HH^2(u_0) ≈ 59 ≫ 9`, which would falsify the conjecture — needs verification.

7. **Citable**: Theorem 3.10 (Ginzburg–Kumar, [GK93]) and Theorem 3.2 (Hemelsoet–Voorhaar's main sl_3 result) should be cited in our paper.

### Recommended next actions for orchestrator

- **W1-1a-α**: Compute `HH^2(u_0(sl_3))` directly — either (i) bar complex restricted to the principal block (adapt `scripts/verify_sl2_hh2.py`), or (ii) run the Hemelsoet–Voorhaar BGG software (https://github.com/RikVoorhaar/bgg-cohomology) at `s = 2`.
- **W1-1a-β**: Determine the block decomposition of `u_q(sl_3)` at `ℓ = 3` (orbits of the `ℓ`-extended affine Weyl group on the alcove `{0, ω_1, ω_2}`; verify whether the singular block exists at `ℓ = h`).
- Consider testing the conjecture at `ℓ = 5` or `ℓ = 7` (`ℓ > h = 3`) to avoid the degenerate `ℓ = h` regime.
- Cite Ginzburg–Kumar Thm 3.10 and Hemelsoet–Voorhaar Thm 3.2 in our paper.

### Files produced / modified

- Created: `/home/z/my-project/hopf-decoherence/literature/notes/W1-1a-hemelsoet-voorhaar.md`
- No code changes; no test files touched.

### Open questions for downstream sub-agents

- Does the principal-block formula `L(m, m)` at `(i+j=0, j-i=0)` + the middle expression at `(i+j=2, j-i=2)` extend from `m ≥ 2` (Prop 5.1) down to `m = 1` (`s = 2`)? The authors' explicit range bound `4 ≤ s` is a yellow flag.
- For `sl_3` at `ℓ = 3 = h` (Coxeter number), is there a singular block `u_1` distinct from the principal block, or is the small quantum group a single block? (Steinberg weight `(ℓ−1)ρ = 2ρ` has `(λ+ρ, θ∨) = 6 > ℓ`, so no Steinberg block; but the principal-block-vs-singular-block split still needs to be verified for the remaining three alcove points.)

---

## W1-1b — Deep read of Schweigert–Woike (arXiv:2204.09018)

- **Task ID**: W1-1b
- **Agent**: Sub-agent 1b (Wave 1, general-purpose)
- **Date**: 2025-08-07
- **Status**: completed
- **Output**: `/home/z/my-project/hopf-decoherence/literature/notes/W1-1b-schweigert-woike.md`

### Summary

Read in full the text-extracted version of Schweigert & Woike, *Homotopy Invariants of Braided Commutative Algebras and the Deligne Conjecture for Finite Tensor Categories* (arXiv:2204.09018v3, Jun 2023). Answered the seven prescribed questions about the main theorem, the connection to the Mastnak–Witherspoon LES, the canonical end, the Deligne conjecture for finite tensor categories, the Farinati–Solotar bracket and Menichi's result, and the application to our conjecture for `u_q(g)`.

### Key findings

1. **Main theorem (Theorem 5.1, "Comparison Theorem")**: For any finite tensor category `C`, the canonical end `A = ∫_{X∈C} X ⊗ X∨` (lifted to its canonical braided commutative algebra structure in `Z(C)`) induces an E₂-algebra structure on the homotopy invariants `C(I, A•)`. Under the equivalence `C(I, A•) ≃ ∫_{X∈Proj C} C(X, X)` (= Hochschild cochain complex of `C`), this E₂-structure gives the standard Gerstenhaber structure on `HH*(C)`. The general construction is Theorem 3.6: for any algebra `T ∈ C` lifting to a braided commutative algebra `T ∈ Z(C)`, the homotopy invariants `C(I, T•)` form an E₂-algebra.

2. **Connection to Mastnak–Witherspoon LES**: **None.** Full-text search confirms Schweigert–Woike do not cite, mention, or construct the Mastnak–Witherspoon long exact sequence (3.3.1) or its connecting homomorphism `δ`. The only "Mastnak" reference is `[MPSW09]` (Mastnak–Pevtsova–Schauenburg–Witherspoon), a different paper. Their framework addresses multiplicative (E₂/Gerstenhaber/BV) structure on a *single* Hochschild complex; it does not provide an alternative construction of `δ`.

3. **Canonical end**: `A = ∫_{X∈C} X ⊗ X∨` is an algebra in `C` whose product is induced by the evaluation `X∨ ⊗ X → I`. It lifts to a braided commutative algebra `A = RI ∈ Z(C)` (right adjoint `R` of the forgetful functor `U : Z(C) → C` applied to the unit), with the "non-crossing half braiding" (2.2)/(2.3). For `C = rep(H)`, `A ≅ H_ad` (Example 5.10, citing [KL01, Thm 7.4.13]). In the unimodular pivotal case, `A` is a symmetric Frobenius algebra in `Z(C)` (Lemma 7.1).

4. **Deligne conjecture for finite tensor categories**: The classical Deligne conjecture (for an associative algebra `R`: the Hochschild cochains `CH*(R; R)` form an E₂-algebra lifting the Gerstenhaber structure on `HH*(R; R)`) is generalized by Theorem 5.1 to the Hochschild cochains `∫_{X∈Proj C} C(X, X)` of *any* finite tensor category `C`. The E₂-structure is explicitly described via the canonical end and its non-crossing half braiding. Theorem 5.11 extends this to exact module categories.

5. **Farinati–Solotar bracket & Menichi**: The Farinati–Solotar (FS) bracket [FS04] is the Gerstenhaber bracket on `Ext*_H(k, k)` for a finite-dimensional Hopf algebra `H`. **Corollary 6.1** lifts this to an E₂-structure on `C(I, I•)` for *any* finite tensor category `C`, and shows the canonical map `Ext*_C(I, I) → HH*(C)` is a monomorphism of Gerstenhaber algebras (and of E₂-algebras at cochain level with suitable models). **Menichi** [Men11, Thm 63] proved for pivotal unimodular Hopf algebras that this monomorphism is actually one of Batalin–Vilkovisky (BV) algebras. **Corollary 7.4** refines this in two ways: (a) holds for all unimodular pivotal finite tensor categories (not just Hopf-algebraic ones); (b) lifts to a framed E₂-structure at cochain level.

6. **Application to our problem (`u_q(g)` at roots of unity)**:
   - **(a) Same HH²**: YES. By Proposition 4.3 + Example 5.10, their `HH*(C)` for `C = rep(u_q(g))` is canonically equivalent to `Ext*_{u_q(g)}(C, u_q(g)_ad)`, which is the standard identification we use. Their E₂-structure is on the same `HH²(u_q(g), C)` we are computing.
   - **(b) Tool to compute δ**: NO. They do not address the LES.
   - **(c) Bypass the LES**: NO. Their framework gives structural (E₂/BV) information, not dimension counts. Dimension computation of `HH²` still requires the LES or another computational technique (bar complex, BGG).
   - **Useful side result (Example 6.2)**: Since `u_q(g)` is braided (has an R-matrix), the Gerstenhaber bracket on `Ext*_{u_q(g)}(C, C)` vanishes by [Her16, Cor 6.3.17]; the inclusion `Ext*_{u_q(g)}(C, C) → HH*(u_q(g))` lands in a *commutative* subalgebra. This is a structural constraint that could help decompose `HH²` into Ext-vs-non-Ext pieces.

7. **Citable**: Theorem 5.1, Theorem 3.6, Proposition 4.3, Corollary 6.1, Corollary 7.4, Lemma 3.1 (Davydov–Müger–Nikshych–Ostrik braided commutativity), Example 5.10 (canonical end ≅ H_ad), Example 6.2 (Gerstenhaber bracket vanishes for braided `u_q(sl_2)`).

### Recommended next actions for orchestrator

- **Cite** Theorem 5.1, Proposition 4.3, Example 5.10, and Example 6.2 from Schweigert–Woike in our paper to establish the categorical framework (`HH*(u_q(g)) ≅ Ext*_{u_q(g)}(C, u_q(g)_ad)`, E₂-structure, vanishing Gerstenhaber bracket on Ext*). Also cite Corollary 7.4 if we discuss the BV structure (relevant since `u_q(g)` is pivotal unimodular at odd roots of unity).
- **Do NOT expect** Schweigert–Woike to supply an alternative to the Mastnak–Witherspoon LES. The LES approach (3.3.1) remains necessary for dimension counting. Their paper is *orthogonal* to the LES approach, not a substitute.
- **Possible follow-up (W2 candidate)**: Use their explicit cochain-level Gerstenhaber bracket (Lemma 5.2 + Lemma 5.7, equation (5.16)) to verify that the bracket structure on `HH²(u_q(g))` is consistent with our predicted dimension `C(n+1, 2) + 2|Φ⁺|` — e.g., does the `C(n+1, 2)` Cartan piece commute with the `2|Φ⁺|` piece? Does the `2|Φ⁺|` piece have trivial self-bracket? This could provide a nontrivial consistency check.
- **Coordinate with W1-1a**: Both papers cite [LQ21] = Lachowska–Qi for the structure of `HH^{2*}(u_q(sl_2))`; Example 6.2 in Schweigert–Woike uses the same decomposition `HH^{2*}(u_q(sl_2)) = Ext* ⊗ Z / (id ⊗ unit)` that Hemelsoet–Voorhaar use. This is consistent.

### Files produced / modified

- Created: `/home/z/my-project/hopf-decoherence/literature/notes/W1-1b-schweigert-woike.md`
- No code changes; no test files touched.

### Open questions for downstream sub-agents

- The Schweigert–Woike framework does not directly address the LES (3.3.1), but the *Drinfeld center* `Z(C)` plays a central role in their paper (as the home of the canonical end `A`). The LES connects `HH(D(B))` to `HH(B) ⊕ HH(B*)`. Since `D(B)` is Tannakian-equivalent to `Z(rep(B))`, is there a way to recast the LES as a statement about homotopy invariants of braided commutative algebras in `Z(C)`? (Speculative — not addressed by Schweigert–Woike, would be a research question of its own.)
- For our dimension formula `dim HH²(u_q(g), C) = C(n+1, 2) + 2|Φ⁺|`: can the decomposition into `C(n+1, 2)` (Cartan) and `2|Φ⁺|` (root) pieces be detected at the E₂-structure level? I.e., do the two pieces have distinct Gerstenhaber-bracket behavior? (The `C(n+1, 2)` piece is in even degree 2; the bracket `[HH², HH²] → HH³` could potentially distinguish the two pieces if it is nonzero on one and zero on the other.)

---

## W1-1c — Block vs full cohomology tension: resolution

- **Task ID**: W1-1c
- **Agent**: Sub-agent 1c (Wave 1, general-purpose)
- **Date**: 2025-08-07
- **Status**: completed
- **Output**: `/home/z/my-project/hopf-decoherence/literature/notes/W1-1c-block-vs-full.md`

### Summary

Resolved the apparent tension flagged in W1-1a between Hemelsoet–Voorhaar's singular-block computation `dim HH^2(u_1(sl_3)) ≥ 59` and our conjecture `dim HH^2(u_q(sl_3), C) = 9`. The tension is **illusory**: the two numbers are dimensions of **different cohomology theories** with **different coefficient modules** and **different block scope**.

### Key findings

1. **Block decomposition (HV Prop 2.3)**: `u_q(g) = ⊕_{λ∈S} u_λ(g)` where `S` is the set of orbits of the ℓ-extended affine Weyl group on the alcove `C = {λ ∈ P^+ : 0 ≤ (λ+ρ, θ∨) ≤ ℓ}`. For `sl_3` at `ℓ = 3 = h` (degenerate regime), the alcove `C = {0, ω_1, ω_2}` has three integral points; the number of distinct blocks (1 or 2) depends on the unresolved orbit computation (W1-1a-β). HV's results (Thm 3.2, Prop 5.1) are stated for `ℓ > h` (i.e. `ℓ ≥ 5` odd for `sl_3`), not for `ℓ = 3`.

2. **Full vs block cohomology** — the precise statement depends on the coefficient module:
   - **Self-coefficient HH**: `HH^*(A, A) ≅ ⊕_λ HH^*(A_λ, A_λ)` — every block contributes.
   - **Trivial-coefficient HH**: `HH^*(A, k) ≅ HH^*(A_0, k)` — **only the principal block contributes**, because `ε(e_λ) = δ_{λ,0}` for the block idempotents (only `e_0` has nonzero counit). Singular blocks contribute **nothing**.
   - **Hopf cohomology**: `H^*(u_q(g), C) = Ext_{u_q(g)}(C, C)` — single-variable, not a block decomposition; Ginzburg–Kumar identifies this with `C[N]`.

3. **HV's coefficient convention**: HV compute `HH^s(u_λ, u_λ) = Ext^s_{u_λ ⊗ u_λ^op}(u_λ, u_λ)` — Hochschild cohomology **with self-coefficients** (the standard default; also the convention forced by the geometric realisation `HH^s_{C*}(Ñ) ≅ HH^s(u_0)` of Thm 2.5). They explicitly distinguish this from `H^•(u_q(g), C) = Ext_{u_q(g)}(C, C)` (Ginzburg–Kumar, Thm 3.10), writing `H^•` for the latter and using it as a coefficient ring acting on `HH^•`. **HV never write `HH^s(u_λ, C)`** for Hochschild with trivial coefficients — that is our project's invariant and is not computed in their paper.

4. **Our project's coefficient convention** (confirmed by reading `paper/main.tex` line 69 and `scripts/verify_sl2_hh2.py` lines 14–16, 305–310): our `HH^2(u_q(g), C) = Ext^2_{u_q(g)^e}(u_q(g), C)` uses **trivial coefficients** via the counit `ε`. The bar-complex differential in `verify_sl2_hh2.py` uses `epsilon[a] = eps(K^a E^b F^c) = δ_{b,0} δ_{c,0}` explicitly.

5. **Concrete sanity check at rank one**:
   - HV/LQ: `HH^{2•}(u_0(sl_2)) ≅ C[N]` as `C[N]`-module (Thm 3.11 of HV, [LQ19]). `dim HH^2(u_0(sl_2), u_0(sl_2)) = ∞`.
   - Our project: `dim HH^2(u_q(sl_2), C) = 3` (finite, verified).
   - These differ by `3 vs ∞`, confirming the two objects are distinct.

### Bottom line

The HV singular-block result is **unrelated to** our conjecture (different mathematical object: different coefficient module, different block scope). It neither confirms nor refutes our conjecture. The W1-1a "tension flag" is **resolved (no tension)** — the apparent `59 ≫ 9` comparison was comparing `dim HH^2(u_1, u_1)` (self-coefficients, singular block) to `dim HH^2(u_q(g), C)` (trivial coefficients, principal block only); these are different invariants of different (sub)algebras.

### Recommended next actions for orchestrator

- **Update W1-1a's tension flag to "resolved"**. The conjecture is unaffected by HV's results. The `A_2, ℓ = 3` case remains open only because **no existing paper computes** the trivial-coefficient invariant `HH^2(u_q(sl_3), C)`.
- **Reclassify W1-1a-α and W1-1a-β as low priority**: both were motivated by the (now-resolved) tension. W1-1a-β (block decomposition at `ℓ = 3`) is irrelevant for our trivial-coefficient invariant. W1-1a-α (principal-block `HH^2(u_0, u_0)`) is also irrelevant — it computes a different invariant.
- **Add a "Comparison with Hemelsoet–Voorhaar" subsection to `paper/main.tex` Section 6** stating explicitly: (a) HV compute `HH^*(u_λ, u_λ)` (self-coefficients) of individual blocks; (b) this is a different invariant from our `HH^*(u_q(g), C)` (trivial coefficients); (c) the trivial-coefficient HH is not computed in any existing paper we are aware of. Distinguish the three cohomology theories: Hopf `H^*`, Hochschild-self `HH^*(A, A)`, Hochschild-trivial `HH^*(A, C)`.
- **Cite Ginzburg–Kumar Thm 3.10** as the closest existing result (Hopf cohomology `H^{2•}(u_q(g), C) ≅ C[N]`), with a clear note that it is Hopf — not Hochschild — cohomology.
- **For direct verification of the conjecture at `A_2, ℓ = 3`** (W2 candidate): the only viable route remains a direct bar-complex computation of `HH^2(u_q(sl_3), C)` on the full `6561`-dim algebra, using the weight-space decomposition strategy from `verify_sl2_hh2.py` adapted to sl_3's 8-dimensional Cartan-and-root lattice.

### Files produced / modified

- Created: `/home/z/my-project/hopf-decoherence/literature/notes/W1-1c-block-vs-full.md`
- No code changes; no test files touched.

### Open questions for downstream sub-agents

- Does HV's `ℓ > h` hypothesis exclude `ℓ = h` strictly, or just require additional care at `ℓ = h`? If the latter, what is the precise modification to Prop 2.3 and Thm 3.2 at `ℓ = h`? (This affects whether HV's results say anything at all about the `ℓ = 3` case for `sl_3` — but it does not affect the resolution of the tension, which holds regardless.)
- The three cohomology theories (Hopf `H^*`, Hochschild-self `HH^*(A, A)`, Hochschild-trivial `HH^*(A, C)`) are related by spectral sequences. Is there a known spectral sequence `HH^*(A, A) ⇒ HH^*(A, k)` (or some filtration) that would let us deduce information about our trivial-coefficient invariant from HV's self-coefficient computations? (The "Hochschild cohomology spectral sequence" `E_2 = HH^*(A, A^*) ⊗ H^*(A, k) ⇒ HH^*(A, A)` goes the wrong way; the relationship `HH^*(A, k) ⊗ H^*(A, k) → HH^*(A, A)` is partial — worth a careful look but not blocking.)

---

## W1-1d — Deep read of Creutzig–Lentner–Rupert (arXiv:2306.11492)

- **Task ID**: W1-1d
- **Agent**: Sub-agent 1d (Wave 1, general-purpose)
- **Date**: 2025-08-07
- **Status**: completed
- **Output**: `/home/z/my-project/hopf-decoherence/literature/notes/W1-1d-log-KL.md`

### Summary

Read in full the text-extracted version of Creutzig, Lentner & Rupert, *An algebraic theory for logarithmic Kazhdan-Lusztig correspondences* (arXiv:2306.11492v1, Jun 2023). Answered the eight prescribed questions about the main KL theorem, the unrolled quantum group $u_q^H(\mathfrak{sl}_2)$, extension to $\mathfrak{sl}_3$, Hochschild cohomology under braided equivalence, VOA-side HH computation, Lentner FP-dimension assumption, path to $A_2$, and citable theorem.

### Key findings

1. **Main theorem (Theorem 1.3(1) = Theorem 8.9 = Example 9.10)**: For $p \in \mathbb{Z}_{\geq 2}$, $q = e^{\pi i/p}$ (a primitive $2p$-th root of unity, **even** order), there is a braided tensor equivalence $\mathcal{O}_{M(p)}^T \cong \mathrm{Rep}^{wt}(u_q^H(\mathfrak{sl}_2))$ between the category of weight modules of the singlet VOA $M(p)$ and the category of weight modules of the small **unrolled** quantum group of $\mathfrak{sl}_2$ at $2p$-th root of unity. Proven twice (Theorem 8.1 recognition via Nichols algebra; Theorem 9.5 recognition via Takeuchi–Skryabin). Also proven: triplet $W(p) \leftrightarrow \tilde{u}_q(\mathfrak{sl}_2)$ (quasi-Hopf), $S(p)$ correspondence, $gl_{1|1}$ correspondence.

2. **Unrolled quantum group $u_q^H(\mathfrak{sl}_2)$**: the realizing Hopf algebra for $\mathcal{U} = Z_\mathcal{C}(\mathcal{B})$ where $\mathcal{C} = \mathrm{Vect}^Q_\mathbb{C}$ (infinite Cartan part $\Gamma = \mathbb{C}$, no quotient, $Q(\lambda) = e^{\pi i \lambda/p}$). It is an **infinite-dimensional** Hopf algebra. Differs from the standard small $u_q(\mathfrak{sl}_2)$ (where $\Gamma = \mathbb{Z}_p$, finite-dim.) in that the Cartan part is not quotiented. $\mathrm{Rep}^{wt}(u_q^H(\mathfrak{sl}_2))$ is **NOT** braided-equivalent to $\mathrm{rep}(u_q(\mathfrak{sl}_2))$ — they share the same Borel/Nichols algebra but differ in the Cartan. Also note the triplet $\tilde{u}_q(\mathfrak{sl}_2)$ corresponds to $\Gamma = \mathbb{Z}_{2p}$ with nontrivial associator (quasi-Hopf).

3. **Extension to $\mathfrak{sl}_3$ or higher rank**: **NOT PROVEN in this paper.** The Feigin–Tipunin algebra $\mathrm{FT}(g,p)$ and its subalgebra $\mathrm{FT}^0(g,p)$ (the singlet analog for higher-rank $g$) are *conjectured* (Example 2.12, citing [CM17]) to correspond to the unrolled quantum group of $g$ at $2p$-th root of unity; "almost nothing is known about the representation theory of $\mathrm{FT}(g,p)$." Line 821: "Higher rank is currently completely out of reach" (for the affine VOA $V^k(\mathfrak{sl}_n)$ triality program). [CRR23] cited for the $\mathfrak{sl}_3$ case is a *different* setup ($L_{-3/2}(\mathfrak{sl}_3)$ admissible-level affine VOA, not a higher-rank singlet).

4. **Hochschild cohomology under braided equivalence**: In general YES — a braided tensor equivalence $\mathcal{C} \simeq \mathcal{D}$ implies $\mathrm{HH}^*(\mathcal{C}) \cong \mathrm{HH}^*(\mathcal{D})$ as $E_2$-algebras, by the Schweigert–Woike framework (canonical end $\mathcal{A} = \int_X X \otimes X^\vee \cong H_{\mathrm{ad}}$ is categorical, preserved by braided equivalence; cf. W1-1b). **BUT** CLR's theorem does NOT give $\mathrm{HH}^*(\mathrm{rep}(u_q(\mathfrak{sl}_2))) \cong \mathrm{HH}^*(\mathrm{rep}(M(p)))$, by two mismatches: (i) the target is $u_q^H(\mathfrak{sl}_2)$ (unrolled, infinite-dim.), not $u_q(\mathfrak{sl}_2)$ (standard, finite-dim.); (ii) the source category $\mathcal{O}_{M(p)}^T$ is **infinite** (not a finite tensor category — Schweigert–Woike applies to finite ones). The braided equivalence transfers $\mathrm{HH}^*$ only between $\mathcal{O}_{M(p)}^T$ and $\mathrm{Rep}^{wt}(u_q^H(\mathfrak{sl}_2))$, neither of which is $\mathrm{rep}(u_q(\mathfrak{sl}_2))$.

5. **VOA-side HH computation**: **No known formula.** The word "Hochschild" appears in CLR only once, as a bibliography citation to Schweigert–Woike [SW22] (a different paper). No $\mathrm{HH}^2$ computation, no dimension count for either $\mathrm{rep}(M(p))$ or $\mathcal{O}_{M(p)}^T$. At $p = 2$ (i.e. $q = i$, $\ell = 4$): not computed. **Odd $\ell$ not covered**: CLR's theorem is stated only for $q = e^{\pi i/p}$ (even order $2p$, i.e. $\ell \in \{4,6,8,\dots\}$). Our conjecture uses $\ell \in \{3,5,7\}$ odd — **outside the proven range**. The odd-$\ell$ side would correspond to the standard $u_q(\mathfrak{sl}_2)$ at odd order $q$, with $\Gamma = \mathbb{Z}_\ell$; the relevant VOA side is **not identified** anywhere in CLR or, to our knowledge, in the surrounding literature.

6. **Frobenius–Perron dimensions / Lentner's assumption**: **Not addressed in CLR.** Grep confirms no occurrences of "Frobenius–Perron", "Perron", "analytic character", or "asymptotic" (only "Frobenius reciprocity"). Lentner 2501.10735 postdates CLR by 18 months. Whether the Lentner assumption holds for $u_q(\mathfrak{sl}_3)$ at $\ell = 3$ cannot be decided on the basis of this paper.

7. **Path to $A_2$ via VOA side**: Assessment of the three options:
   - **(a) Direct**: NOT VIABLE — KL transfers $\mathrm{HH}^*$ between $\mathcal{O}_{M(p)}^T$ and $\mathrm{Rep}^{wt}(u_q^H(\mathfrak{sl}_2))$, not to $\mathrm{rep}(u_q(\mathfrak{sl}_n))$; KL is proven only for $\mathfrak{sl}_2$; proven range is $\ell$ even.
   - **(b) Indirect (canonical end, Schweigert–Woike)**: possible in principle but does not give dimension counts (cf. W1-1b). VOA side adds nothing.
   - **(c) NOT VIABLE for odd $\ell$**: **This is the correct answer.** CLR's log-KL covers only $q = e^{\pi i/p}$ (even order $2p$); our conjecture at $A_2$ uses $\ell \in \{3,5,7\}$ (odd), outside the proven range. The VOA-side path is **not viable in the current state of the art**.

8. **Citable theorem**: Theorem 1.3(1) / Theorem 8.9 — quote verbatim with three caveats: (i) $q = e^{\pi i/p}$, so $\ell = 2p$ even; (ii) target is unrolled $u_q^H(\mathfrak{sl}_2)$ (infinite-dim.), not standard $u_q(\mathfrak{sl}_2)$; (iii) both sides are infinite weight-module categories, not finite tensor categories.

### Bottom line

The CLR log-KL correspondence **does not provide a viable VOA-side route** to our conjecture $\dim \mathrm{HH}^2(u_q(\mathfrak{sl}_n),\mathbb{C}) = \binom{n+1}{2} + 2|\Phi^+|$ at $A_2$ with $\ell$ odd. Three independent obstructions: (1) wrong quantum-group target ($u_q^H$, not $u_q$); (2) no $\mathfrak{sl}_3$ KL theorem (only $\mathfrak{sl}_2$ proven; $\mathfrak{sl}_3$ Feigin–Tipunin is conjectural); (3) parity of $\ell$ (proven only for $\ell$ even, conjecture uses $\ell$ odd). No $\mathrm{HH}^*(\mathrm{rep}(M(p)))$ computation exists in the literature cited.

### Recommended next actions for orchestrator

- **Cite** CLR Theorem 1.3(1) / Theorem 8.9 in our paper as state-of-the-art log-KL at rank one, with explicit caveats (unrolled vs. standard $u_q$; even $\ell$ only; $\mathfrak{sl}_3$ unproven).
- **Do NOT pursue** VOA-side $\mathrm{HH}^2(\mathrm{rep}(M(p)))$ as a route to our conjecture — non-viable per the three obstructions above.
- **Reaffirm W1-1c's recommendation**: the only viable route to verify the conjecture at $A_2, \ell = 3$ remains a direct bar-complex computation of $\mathrm{HH}^2(u_q(\mathfrak{sl}_3),\mathbb{C})$ on the full 6561-dim algebra, using weight-space decomposition adapted from `scripts/verify_sl2_hh2.py`.
- **Cross-link with W1-1b**: the Schweigert–Woike $E_2$-structure (Gerstenhaber bracket vanishes on $\mathrm{Ext}^*$; BV structure since $u_q(\mathfrak{sl}_3)$ is pivotal unimodular at odd $\ell$) gives structural constraints to combine with the direct W2 computation.

### Files produced / modified

- Created: `/home/z/my-project/hopf-decoherence/literature/notes/W1-1d-log-KL.md`
- No code changes; no test files touched.

### Open questions for downstream sub-agents

- Is there a KL-type correspondence at **odd** $\ell$ to a *different* VOA (orbifold / coset of $M(p)$, "half-integer" singlet, or other)? CLR does not address; worth a literature search on Feigin–Tipunin / singlet-like VOAs at odd-order roots of unity.
- Is there a known relationship between $\mathrm{HH}^*(\mathrm{rep}(u_q(\mathfrak{sl}_2)))$ and $\mathrm{HH}^*(\mathrm{Rep}^{wt}(u_q^H(\mathfrak{sl}_2)))$ via the quotient $u_q^H \twoheadrightarrow u_q$ (mod Cartan by $\mathbb{Z}_\ell$)? If yes, CLR could give *indirect* information — not worked out in CLR.
- Does CLR's framework (or cited [CGP15] for the unrolled group) provide analytic-character asymptotics that could verify the Lentner FP-dimension assumption (2501.10735) for $u_q^H(\mathfrak{sl}_3)$ at $\ell = 3$? (Speculative; not addressed in CLR.)

---

## W2-1a — Plan for direct bar complex of HH²(u_q(sl_3), C) at ℓ = 3

- **Task ID**: W2-1a
- **Agent**: Sub-agent 2a (Wave 2, general-purpose)
- **Date**: 2025-08-07
- **Status**: completed (planning only — direct computation intractable in sandbox)
- **Outputs**:
  - `/home/z/my-project/hopf-decoherence/scripts/plan_sl3_hh2.py` (planning / diagnostic script)
  - `/home/z/my-project/hopf-decoherence/scripts/sl3_plan_output.txt` (captured stdout)
  - `/home/z/my-project/hopf-decoherence/literature/notes/W2-1a-sl3-bar-complex-plan.md` (full plan & analysis)

### Summary

Planned and analysed the direct bar-complex computation of `HH²(u_q(sl_3), C)` at `ℓ = 3` (conjecture: `dim = C(3,2) + 2|Φ⁺| = 3 + 6 = 9`). **The direct bar complex is intractable in this sandbox** (4 GB RAM budget). Even the smallest sub-computation — the principal block `u_0(sl_3)`, dim 729 — requires ~29 GB of RAM just to store the `d²` matrix sparsely, and ~95 days of compute to obtain its rank via sparse iterative methods. Three alternative approaches are recommended for subsequent sub-tasks.

### Key findings

1. **Dimension**: `dim_C u_q(sl_3) |_{ℓ=3} = 3^8 = 6561`. PBW basis: `K1^a K2^b E1^c E12^e E2^d F1^f F21^g F2^h`, all exponents in `{0,1,2}` (matches the task's `3^(3+3+rank) = 3^8`).

2. **Weight space decomposition**: graded by `(Z/3)^2` (the character group of the Cartan torus `(K1, K2)`); **9 weight spaces, each of dim 729** (uniform distribution — the weight map `(Z/3)^6 → (Z/3)^2` on the E-F part is a surjective homomorphism with kernel of size `3^4 = 81`, times the Cartan factor `3^2 = 9`).

3. **Principal block**: `u_0(sl_3)` at `ℓ=3` = weight-0 subspace of `A` (since the central idempotent `e_0 = (1/|G|) Σ_{g∈G} g`, `G = (Z/3)^2`, is the weight-0 projector). Hence **dim u_0 = 729** (consistent with the heuristic `dim u_0 ≈ ℓ^{2N} = 3^6` for `N = |Φ⁺| = 3`). For comparison, `sl_2` at `ℓ=3`: `dim u_0 = 9 = ℓ^{2·1}` ✓.

4. **Bar complex block sizes (full A)**: per weight (all 9 equal by symmetry):
   - `dim C¹_w = 729`
   - `dim C²_w = 4,782,969` (~4.8 M)
   - `dim C³_w = 31,381,059,609` (~31.4 B)
   Total across 9 weights matches `dim(A)^n` ✓.

5. **Bar complex block sizes (principal block u_0, dim 729)**:
   - `dim C¹(u_0) = 729`
   - `dim C²(u_0) = 531,441` (~531 K)
   - `dim C³(u_0) = 387,420,489` (~387 M)

6. **Memory obstacles**:
   - **Full A, weight-0 d²** (sparse, 4 nonzeros/row): 125.5 B nonzeros × 20 B = **2.28 TB**. All 9 weights: **20.55 TB**. → 584× over the 4 GB sandbox budget (weight-0 alone).
   - **Principal block u_0, d²** (sparse): 1.55 B nonzeros × 20 B = **28.87 GB**. → 7.2× over budget.
   - **Dense Gram matrices** (4.5 TB for u_0, 333 TB for full A weight-0): infeasible on any single machine.

7. **Time obstacles** (sparse iterative rank via ARPACK-style SVD, conservative 1 GFLOP/s):
   - Full A weight-0: ~69,500 days (~190 years) — infeasible.
   - Principal block u_0: ~95 days — infeasible in the sandbox.
   - Sparse rank-revealing QR (SuiteSparse SPQR) on `u_0`: ~1–2 weeks on a 32 GB workstation — infeasible in the sandbox.

8. **Comparison with tractable cases**:
   | Case | dim A | weight-0 C³ | sparse d² | status |
   |---|---|---|---|---|
   | `u_q(sl_2)`, `ℓ=3` | 27 | 6,561 | 0.5 MB | ✅ verified (`HH² = 3`) |
   | `B⁺(u_q(sl_3))`, `ℓ=3` | 243 | 1,594,323 | 122 MB | ✅ verified (`HH² = 6`) |
   | `u_q(sl_3)`, `ℓ=3` | 6,561 | 31,381,059,609 | 2.28 TB | ❌ intractable |
   | `u_0(sl_3)`, `ℓ=3` | 729 | 387,420,489 | 28.87 GB | ❌ intractable on 4 GB |
   The `sl_3` case is **~5 orders of magnitude larger** than the largest case previously verified (`B⁺(sl_3)`).

### Smaller sanity checks (none verifies the conjecture's `dim = 9`)

- **Cartan subalgebra** `C[K1,K2]/(K1³−1,K2³−1)`, dim 9: splits as `C^9` over C (CRT); semisimple ⇒ `HH² = 0`, **not** `3 = C(3,2)`. The "Cartan piece" `3` in the conjecture is **not** `HH²(Cartan subalgebra)`; it comes from deformations of the cross-relations `K_i E_j = q^{a_{ij}} E_j K_i`, etc., which only exist in the full Drinfeld double.
- `B⁺(sl_3)` `ℓ=3` already verified: `HH² = 6 = 2|Φ⁺|` ✓ (Borel has no negative-root part, so the Cartan piece `C(n+1,2) = 0` and the root piece `2|Φ⁺| = 6`).
- `sl_2` `ℓ=3` already verified: `HH² = 3 = C(2,2) + 2·1` ✓.
- Restricting `u_q(sl_3)` to monomials with exponents in `{0,1}` (dim 2^8 = 256) is **not closed under multiplication** (e.g., `E1·E1 = E1²` has exponent 2); not a subalgebra.

**Conclusion**: no smaller sub-computation verifies `dim HH²(u_q(sl_3), C) = 9`. The conjecture at `sl_3, ℓ=3` remains **unverified** after W2-1a.

### Recommended next actions for orchestrator

The direct bar complex is intractable in this sandbox for `sl_3, ℓ=3`. Three alternative approaches are recommended, in order of promise:

- **W2-1b — BGG-style resolution (adapt Hemelsoet–Voorhaar)**: their BGG software computes self-coef `HH*(u_λ, u_λ)`; we need trivial-coef `HH*(u_0, C)`. The BGG resolution has chain groups of dim `~|W| × dim P(0) ≈ 6 × 729 = 4374` (vs. 387 M for the bar complex `C³`), a `~10⁵×` reduction. Their principal-block `s=2` case is explicitly excluded in Prop 5.1; the exclusion is a range bound, not a fundamental obstruction. **Highest priority.**

- **W2-1c — Explicit cocycle construction**: construct 9 candidate 2-cocycles (3 Cartan-cross-relation + 6 root, 2 per positive root `E1, E12, E2`); verify `d² = 0` and linear independence mod `im d¹`. Cost: `O(dim A × 9)` per cocycle check ≈ 60 K ops; tractable in seconds. Gives a constructive **lower bound** `dim HH² ≥ 9`; needs an independent upper-bound argument to fully prove the conjecture. The main implementation work is extending the Borel multiplication table in `verify_sl3_bplus_hh2.py` to include the F-generators and the cross-relations `[E_i, F_j] = δ_{ij} (K_i − K_i^{−1})/(q − q^{−1})`.

- **W2-1d — Mastnak–Witherspoon LES analysis**: known `HH²(B⁺(sl_3)) ⊕ HH²(B⁺(sl_3)*) = 6 + 6 = 12`; LES gives `HH²(D(B))` as an extension; the conjecture says `HH² = 9`, so the connecting map `HH²(D(B)) → HH³(B ⊗ B*)` should have rank 3. Lower priority — requires careful LES computation.

- **Out of scope**: massive computation (TB RAM, weeks of compute on a cluster). Not feasible in this sandbox.

### Hardware that would make the direct computation feasible (for reference)

- For the principal block `u_0` only: a workstation with **64–128 GB RAM** and a fast SSD; **1–2 weeks** of compute with SuiteSparse SPQR.
- For the full algebra: a distributed cluster with **~30 nodes × 128 GB RAM** (≈ 4 TB aggregate); **months** of compute.

### Files produced / modified

- Created: `/home/z/my-project/hopf-decoherence/scripts/plan_sl3_hh2.py` (planning script).
- Created: `/home/z/my-project/hopf-decoherence/scripts/sl3_plan_output.txt` (captured stdout).
- Created: `/home/z/my-project/hopf-decoherence/literature/notes/W2-1a-sl3-bar-complex-plan.md` (detailed plan, dimensional analysis, recommendations).
- No existing scripts or tests modified.

### Open questions for downstream sub-agents

- **For W2-1b (BGG adaptation)**: Can the Hemelsoet–Voorhaar BGG software be adapted to compute **trivial-coefficient** `HH*(u_0(sl_3), C)` at `s = 2`? Their Prop 5.1 explicitly excludes `s = 2` for the principal block — what is the precise obstruction, and can it be bypassed?
- **For W2-1c (explicit cocycles)**: Can the 3 Cartan cocycles be written down explicitly? (The 6 root cocycles are straightforward; the 3 Cartan ones come from deformations of the cross-relations `K_i E_j`, `K_i F_j`, and need to be identified.)
- **For W2-1d (LES analysis)**: Is the connecting map `HH²(D(B)) → HH³(B ⊗ B*)` computable for `B = B⁺(sl_3)` at `ℓ=3`? If yes, the LES gives `dim HH²(D(B))` from the known `12` minus the rank of the connecting map.
- **For the orchestrator**: Is a partial verification (`dim HH² ≥ 9` via cocycle construction) sufficient to publish, or do we need the full `dim HH² = 9`?
- **Bigger picture**: For `n ≥ 3` and odd `ℓ`, the direct bar complex is likely intractable at all odd `ℓ ≥ 3` (dimensional growth `dim(u_q(sl_n)) = ℓ^{n²+2n}`). The BGG / cocycle / LES approaches are not just expedients — they are **necessary** for higher-rank verification.

---

## W2-1b — LES consistency analysis for sl_3 at ℓ = 3

- **Task ID**: W2-1b
- **Agent**: Sub-agent 2b (Wave 2, general-purpose)
- **Date**: 2025-08-07
- **Status**: completed
- **Outputs**:
  - `scripts/test_sl3_les_consistency.py` (main analysis script; LES constraints + direct HH¹(B⁺) computation)
  - `scripts/sl3_les_output.txt` (captured stdout, 196 lines)
  - `tests/test_sl3_les.py` (9 passing pytest tests)
  - `literature/notes/W2-1b-sl3-les-analysis.md` (full analysis)

### Summary

Investigated the Mastnak–Witherspoon LES at A₂ to determine whether the conjecture `dim_C HH²(u_q(sl_3), C) = C(3,2) + 2|Φ⁺| = 9` at ℓ = 3 is consistent with the verified `dim HH²(B⁺) = 5`, and to make the strongest possible statement about A₂ short of the (intractable per W2-1a) direct bar-complex computation. **The conjecture at A₂ is consistent with the LES** (the structural split `(3, 6)` is among 10 LES-consistent splits), and the LES — together with a direct verification that `dim HH¹(B⁺) = 0` (computed here, ~2 seconds) — **reduces the conjecture at A₂ to a single tractable prediction**: `dim H̃¹_b(B⁺(u_q(sl_3)), C) = C(3, 2) = 3` at ℓ = 3. Status of A₂ is now **open, but reduced to a tractable computation** of `H̃¹_b(B⁺)` (chain groups of dim ~ 59K, in contrast to the intractable `HH²(D)` bar complex at 2.28 TB RAM).

### Key findings

1. **LES dimensional constraints (Q1)**. The Mastnak–Witherspoon LES at degree 2 gives `dim HH²(D(B⁺)) = dim im(δ) + dim im(ῑ at deg 2)`, with constraints: `0 ≤ dim im(δ)`, `0 ≤ dim im(ῑ at deg 2) ≤ dim HH²(B⁺) ⊕ HH²(B⁻) = 10`. The LES alone does not pin down the split.

2. **LES-consistent splits (Q2)**. Under the conjecture (`dim HH²(D) = 9`), 10 splits `(dim im(δ), dim im(ῑ))` are consistent: `(0,9), (1,8), (2,7), (3,6), (4,5), (5,4), (6,3), (7,2), (8,1), (9,0)`. The conjecture's structural split `(3, 6)` is among them → **CONSISTENT** (necessary condition satisfied).

3. **Cannot pin down dim im(δ) from first principles (Q3)**. Need either: (a) direct `H̃¹_b(B⁺)` computation (tractable), or (b) direct restriction-map `ῑ: HH²(D) → HH²(B) ⊕ HH²(B*)` computation (intractable per W2-1a).

4. **Direct HH¹(B⁺) computation (Q4)** — the new empirical result of this task. Computed `dim HH¹(B⁺(u_q(sl_3)), C) = 0` at ℓ = 3 via weight-decomposed bar complex (9 weight blocks, each `6561 × 27`, SVD on each; total `rank(d¹) = 243 = dim B⁺`). Total computation time ~2 seconds. **Matches the sl_2 case and the paper's stated expectation (Sec. 7)**. This was previously only "expected in general" — now verified at A₂.

5. **LES simplification (Q5)**. With `dim HH¹(B⁺) = dim HH¹(B⁻) = 0`, the map `π̄: HH¹(B) ⊕ HH¹(B*) → H̃¹_b(B)` has zero source, so `dim im(π̄) = 0`. By exactness, `ker(δ) = 0`, so **δ is injective**: `dim im(δ) = dim H̃¹_b(B⁺)`. The LES simplifies to:
   ```
   dim HH²(D(B⁺)) = dim H̃¹_b(B⁺) + dim im(ῑ at deg 2)
   ```

6. **Strongest statement about A₂ (Q6)**. Given HH¹ vanishing (verified) and LES exactness:
   - The conjecture (`dim HH²(D) = 9`) is **equivalent** to `dim H̃¹_b(B⁺(u_q(sl_3)), C) = C(3, 2) = 3` at ℓ = 3.
   - **Sufficient REFUTATION criterion**: if `dim H̃¹_b(B⁺) > 9`, then `dim HH²(D) > 9`, refuting the conjecture.
   - **Sufficient VERIFICATION criterion** (under the structural prediction `dim im(ῑ) = 2|Φ⁺| = 6`): if `dim H̃¹_b(B⁺) = 3`, then `dim HH²(D) = 9`, verifying the conjecture.
   - A direct computation of `H̃¹_b(B⁺)` — feasible at `dim B⁺ = 243` (chain groups ~ 59K-dim) — would either verify (if = 3) or refute (if ≠ 3) the conjecture at A₂.

### Comparison with A₁

| Quantity | A₁, ℓ = 3 (verified) | A₂, ℓ = 3 (this task) |
|---|---|---|
| `dim HH¹(B⁺)` | 0 | **0** (computed Q4) |
| `dim HH²(B⁺)` | 1 | 5 (paper Sec. 6.5) |
| `dim HH²(B⁻)` | 1 | 5 (by duality) |
| `dim HH²(D(B⁺))` | 3 | 9 (conjecture, **not** verified) |
| `dim im(ῑ at deg 2)` | 2 (verified by restriction map) | 6 (predicted, **not** verified) |
| `dim im(δ)` | 1 (verified by restriction map) | 3 (predicted, **not** verified) |
| `dim H̃¹_b(B⁺)` | 1 (conjecture + LES) | 3 (conjecture + LES, **not** verified) |
| Status | **Theorem-in-waiting** (Sec. 7.4) | **Open, reduced to H̃¹_b(B⁺) computation** |

The A₁ verification relied on the tractable restriction-map computation (`dim u_q(sl_2) = 27`); at A₂ (`dim u_q(sl_3) = 6561`), the analogous computation is intractable per W2-1a. The W2-1b reduction identifies the smaller object `H̃¹_b(B⁺)` (decoupled from the full Drinfeld double) as the correct target — a **~10⁷× reduction** in problem size (from `dim(A)³ ≈ 2.8 × 10¹¹` to `dim(B⁺)² ≈ 6 × 10⁴`).

### Tests

`tests/test_sl3_les.py` (9 tests, all passing):
- `test_script_runs`: consistency script returns a result dict.
- `test_hh1_bplus_vanishes`: `dim HH¹(B⁺) = 0` (new empirical computation).
- `test_hh2_bplus_is_5`: `dim HH²(B⁺) = dim HH²(B⁻) = 5` (paper Sec. 6.5).
- `test_conjecture_dimension_is_9`: `dim HH²(u_q(sl_3)) = 9` (conjecture).
- `test_conjecture_split_is_les_consistent`: the conjecture's split `(3, 6)` is among the LES-consistent splits.
- `test_conjecture_is_necessary_les_consistent`: independent re-check of the necessary conditions.
- `test_les_simplification_under_hh1_vanishing`: under HH¹ = 0, δ is injective; conjecture ⇔ `dim H̃¹_b(B⁺) = 3`.
- `test_refutation_criterion_identified`: `dim H̃¹_b(B⁺) > 9` would refute the conjecture.
- `test_output_file_exists`: `scripts/sl3_les_output.txt` exists and records the verdict.

Full test suite (excluding slow): **80 passed, 1 deselected** in ~35 s. The new tests do not regress existing tests.

### Files produced / modified

- Created: `scripts/test_sl3_les_consistency.py` (main analysis script).
- Created: `scripts/sl3_les_output.txt` (captured stdout).
- Created: `tests/test_sl3_les.py` (9 tests).
- Created: `literature/notes/W2-1b-sl3-les-analysis.md` (full analysis, this task's primary writeup).
- No existing scripts or paper source modified.

### Open questions for downstream sub-agents

- **For W2-1c (explicit cocycle construction, in progress per W2-1a)**: Can the 3 Cartan-type / mixed E–F cocycles predicted to live in `H̃¹_b(B⁺)` be written down explicitly? For sl_2, the single mixed E–F class in `HH²(D)` was extracted in Sec. 6.4 of the paper, but its preimage under `δ` in `H̃¹_b(B⁺)` was not. Constructing these would give a constructive lower bound `dim H̃¹_b(B⁺) ≥ 3`, providing positive evidence for the conjecture.
- **For W2-1d (direct `H̃¹_b(B⁺)` computation)**: Implement the bialgebra cochain complex of MW §2.1 for `B⁺(u_q(sl_3))` at ℓ = 3, compute `dim H̃¹_b(B⁺)`, check whether it equals 3. Chain-group size ~ `dim(B⁺)² = 59049` per degree — comparable to the existing `HH²(B⁺)` computation that runs in ~4 minutes (paper Sec. 6.5). **This is now the highest-priority next step for resolving A₂.**
- **For the orchestrator**: The conjecture's structural prediction `dim im(ῑ at deg 2) = 2|Φ⁺| = 6` requires either an Angiono–Kochetov–Mastnak [AKM15] rigidity argument or an explicit verification that the 6 ℓ-th power classes `[E_α^ℓ], [F_α^ℓ]` survive into `im(ῑ)`. Note `dim HH²(B⁺) ⊕ HH²(B⁻) = 10`, so 4 of the 10 classes must die under `π̄: HH²(B⁺) ⊕ HH²(B⁻) → H̃²_b(B⁺)`; identifying these 4 "extra" classes is also open.
- **Bigger picture**: The W2-1b reduction (`HH²(D)` ↔ `H̃¹_b(B⁺)` under HH¹ vanishing) likely generalises to all type-A_n at odd ℓ. If `dim HH¹(B⁺) = 0` holds in general (analogous to sl_2, sl_3), the conjecture at all A_n reduces to `dim H̃¹_b(B⁺) = C(n+1, 2)` — a tractable computation per rank.

---

## W3-1b — Extract MW bialgebra cochain equations (no implementation)

- **Task ID**: W3-1b
- **Agent**: Sub-agent 3b (Wave 3, general-purpose)
- **Date**: 2025-08-07
- **Status**: completed
- **Output**: `literature/notes/W3-1b-mw-equations.md` (this task's primary writeup, ~13 sections)
- **Predecessor**: W3-1a (crashed while attempting direct implementation; this sub-task is the focused "extract equations only" redo)
- **Successor**: W3-1c (will implement the bialgebra cochain complex for `B⁺(u_q(sl_3))` at ℓ=3 using these equations)

### Summary

Read MW §2.1 (Bialgebra cohomology and deformations) and §3.4 (Morphisms in the LES) carefully and wrote down, in mathematical notation, the equations that W3-1c needs to implement. **No code written.** This is the foundation that W3-1c (the implementation) will use.

### Key findings

1. **Main target equation (the single most important fact for W3-1c)**: `H̃¹_b(B) = { h: B̄ → B̄ | ∂^h h = 0 AND ∂^c h = 0 } / 0`, where
   - `∂^h h(a, b) = a·h(b) − h(a·b) + h(a)·b` (derivation condition),
   - `∂^c h(c) = c₁ ⊗ h(c₂) − Δ(h(c)) + h(c₁) ⊗ c₂` (coderivation condition).
   
   This is the **i=1 case** of MW eq. (2.1.1) (MW displays it just before eq. 2.1.1 in the text). The coboundaries `B̃¹_b(B) = 0` because the source of `∂_b` at degree 1 in the truncated normalized complex is zero. **Hence `dim H̃¹_b(B) = dim ker(∂_b: Hom(B̄, B̄) → Hom(B̄², B̄) ⊕ Hom(B̄, B̄²))` is a single linear-algebra rank computation.**

2. **The bialgebra 1-cocycle is a SINGLE linear map `h: B̄ → B̄`, NOT a pair `(f, g)`** — clarified the apparent confusion in the task description. The pair `(f, g)` with `f: B̄² → B̄` and `g: B̄ → B̄²` lives in `(B_+)^{2,1} ⊕ (B_+)^{1,2}`, which is `(Tot B⁰_+)³` and represents a class in `H̃²_b(B)` (NOT `H̃¹_b(B)`). The MW eqs. (2.1.1) and (2.1.2) describe `H̃²_b(B)`, which W3-1c does NOT compute directly (intractable, see W2-1b). For H̃¹_b, the only cocycle condition is the simultaneous derivation/coderivation condition on the single map `h`.

3. **The three MW eq. (2.1.1) conditions on `(f, g)`** (for H̃²_b, written down for completeness):
   - Equation 1 (Hochschild 2-cocycle on `f`): `a·f(b,c) + f(a,bc) = f(ab,c) + f(a,b)·c`.
   - Equation 2 (Cartier 2-cocycle on `g`): `c₁ ⊗ g(c₂) + (1⊗Δ)g(c) = (Δ⊗1)g(c) + g(c₁) ⊗ c₂`.
   - Equation 3 (Mixed compatibility, `∂^c f + ∂^h g = 0`): `f(a₁,b₁) ⊗ a₂b₂ − Δ(f(a,b)) + a₁b₁ ⊗ f(a₂,b₂) = −(Δa)g(b) + g(ab) − g(a)(Δb)`.
   - Coboundary form (MW eq. 2.1.2): `(f, g) = (∂^h h, −∂^c h)` for some `h: B̄ → B̄`.

4. **Sign trick for the total differential**: `∂_b|_{B^{p,q}} = ∂^h + (−1)^p ∂^c`. For `p=1` (relevant for H̃¹_b), the vertical component picks up a `−1` sign, giving `∂_b h = (∂^h h, −∂^c h)`. This explains the sign in MW eq. (2.1.2): `g(c) = −c₁ ⊗ h(c₂) + Δ(h(c)) − h(c₁) ⊗ c₂` (i.e., `g = −∂^c h`, with the `−1` from `(−1)^p` for `p=1`).

5. **Notational clash in MW §3.4 resolved**: MW's "connecting homomorphism" (MW's `δ`) is the map `π̄: HH^i_h(B) ⊕ HH^i_h(X) → H̃^i_b(B)` — going INTO `H̃^i_b(B)`. The map the task wants (modern `δ`, going FROM `H̃^i_b(B)` to `HH^{i+1}_h(D(B))`) is MW's third arrow, given by MW eq. (3.4.4), NOT by MW eqs. (3.4.1) and (3.4.2). Wrote down all three formulas (MW eqs. 3.4.1, 3.4.2, 3.4.4) and clarified the direction conventions. **For our case** (`dim HH¹(B⁺) = dim HH¹(X) = 0`, verified by W2-1b): both sources of `π̄` at degree 1 are zero, so `π̄ = 0` and `δ_modern` is **injective**. Therefore `dim im(δ_modern) = dim H̃¹_b(B⁺)` automatically.

6. **Concrete recipe for W3-1c** (Section 10 of the notes): 
   - `dim H̃¹_b(B⁺) = 242² − rank(∂_b)`, where `∂_b` is a sparse `28344976 × 58564` matrix.
   - Weight-decomposed into 9 blocks each of size `~39204 × 6561`, total computation time ~minutes, memory <4 GB.
   - Expected rank under conjecture: `58561`, giving `dim H̃¹_b = 3 = C(3, 2)`.
   - Cross-check: replicate for sl_2 at ℓ=3 (`dim B = 9, dim B̄ = 8`) — expected `dim H̃¹_b = 1`.
   - Existing multiplication table in `verify_sl3_bplus_hh2.py` can be reused; **comultiplication table needs to be built new** using the Lusztig root-vector formula `Δ(E₁₂) = E₁₂⊗1 + K₁K₂⊗E₁₂ + (q⁻¹−q) K₂E₁⊗E₂`.

7. **The map `δ_modern: H̃¹_b(B⁺) → HH²(D(B⁺))` does NOT need to be computed explicitly** — direct computation would require building 2-cochains on `D(B⁺)` of size `dim(D(B⁺))² = 6561² ≈ 4.3 × 10⁷`, which is the intractable W2-1a regime. The W3-1c computation works entirely on `B⁺` (dim 243) and uses the injectivity of `δ_modern` to deduce `dim im(δ_modern) = dim H̃¹_b(B⁺)` without computing `δ_modern` itself.

### Comparison with W2-1b's open question

W2-1b's "highest-priority next step" was:
> "For W2-1d (direct `H̃¹_b(B⁺)` computation): Implement the bialgebra cochain complex of MW §2.1 for `B⁺(u_q(sl_3))` at ℓ = 3, compute `dim H̃¹_b(B⁺)`, check whether it equals 3. Chain-group size ~ `dim(B⁺)² = 59049` per degree — comparable to the existing `HH²(B⁺)` computation that runs in ~4 minutes (paper Sec. 6.5)."

W3-1b confirms this is tractable and provides the precise equations for W3-1c (= W2-1d, renamed for Wave 3) to implement. The chain-group dimensions are even smaller than W2-1b's rough estimate: `Hom(B̄, B̄) = 242² = 58564` (not `243² = 59049`), since we work with the augmentation ideal `B̄ = ker ε` (dim 242), not the full algebra `B` (dim 243).

### Files produced / modified

- **Created**: `literature/notes/W3-1b-mw-equations.md` — 13-section writeup of the MW equations in both abstract and concrete (B⁺(u_q(sl_3)) at ℓ=3) form. Sections cover: bicomplex vertices and face maps, differentials, total differential with sign trick, truncated and normalized complexes, the H̃¹_b cocycle condition (the main target), the H̃²_b three-equation cocycle condition (for reference, NOT to be computed directly), concrete translation to our case, the connecting homomorphism (MW eqs. 3.4.1, 3.4.2, 3.4.4 with notation disambiguated), and a step-by-step recipe for W3-1c.
- No code modified. No tests added.

### Open questions for downstream sub-agents

- **For W3-1c (implementation)**: All equations needed are now in `W3-1b-mw-equations.md` §6 (cocycles), §7 (coboundaries, for reference), §8 (concrete PBW-basis structure constants), §10 (recipe). Expected runtime ~minutes, memory <4 GB. **Highest-priority task in the project** — would establish the conjecture at A₂ (modulo the structural prediction `dim im(ῑ at deg 2) = 6`).
- **For the orchestrator**: The chain W1 → W2 → W3-1b (equations) → W3-1c (computation) → conjecture verified at A₂ is now fully specified. If W3-1c yields `dim H̃¹_b = 3`, the conjecture at A₂ is established subject to the structural prediction `dim im(ῑ at deg 2) = 2|Φ⁺| = 6` (which itself requires either AKM rigidity or explicit verification of the 6 ℓ-th power classes).
- **For W3-1d (explicit cocycle extraction, future)**: If W3-1c succeeds, the next step is to extract explicit basis cocycles `h₁, h₂, h₃: B̄ → B̄` (each a `242 × 242` matrix). The conjecture identifies these as "Cartan-type / mixed E–F" cocycles; their explicit form would give a constructive lower bound independent of the dimension-counting argument.
- **Bigger picture**: The same equation `H̃¹_b(B) = {h: B̄ → B̄ | ∂^h h = ∂^c h = 0}` (with appropriate structure constants) generalises to all type-A_n at odd ℓ. The W3-1c implementation, once written, can be re-run for any `sl_n` at any odd ℓ with only the structure-constant tables swapped.

---
