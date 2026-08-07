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
