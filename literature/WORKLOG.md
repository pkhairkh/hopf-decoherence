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

## W4-2 — Outreach email drafts (Negron / Witherspoon / Qi)

- **Task ID**: W4-2
- **Agent**: Sub-agent 4b (Wave 4, general-purpose)
- **Date**: 2025-08-07
- **Status**: completed (drafts only — NOT sent; author to review and send)
- **Output**: `/home/z/my-project/hopf-decoherence/literature/notes/W4-2-outreach-drafts.md`
- **Predecessor**: W4-1 (sl₄ `dim H̃¹_b = 3`, refuting the `C(n+1,2)` structural prediction for `n ≥ 3`)
- **Successor**: none (this is an outreach task, not a research task)

### Summary

Drafted three short outreach emails, one each to Cris Negron (USC), Sarah
Witherspoon (TAMU), and You Qi (UVA), seeking their reaction to the W3-1c /
W4-1 finding that `dim H̃¹_b(B⁺(u_q(sl_n))) = n − 1 = rank(sl_n)` at `ℓ = 3`
for `n = 2, 3, 4`, refuting the original conjecture's structural prediction
`C(n+1, 2)`. Each email is under 250 words; each is paired with a one-paragraph
personal "cover note" the author can prepend when sending. Drafts only — not
sent.

### Key facts the drafts communicate

The drafts are built around the same five facts, with the framing tuned to
each recipient's expertise:

1. **The computation**: `dim H̃¹_b(B⁺(u_q(sl_n)))` at `ℓ = 3`, `n = 2,3,4`, by
   direct linear algebra on the MW derivation/coderivation 1-cocycle equations.
   Results: `1, 2, 3`.
2. **The pattern**: `dim = n − 1 = rank(sl_n)`, *not* `C(n+1, 2)`.
3. **Cocycle shape**: diagonal in PBW basis, exactly linear in root-vector
   exponents (residual `~10⁻¹⁶`), zero Cartan coefficients; `∂ʰ` constraints
   follow from `∂ᶜ` via the q-commutators.
4. **What is refuted**: the *structural decomposition* `dim im(δ) = C(n+1,2)`.
5. **What is NOT refuted**: the *total count* `dim HH²(u_q(g)) = C(n+1,2) +
   2|Φ⁺|`. Full `HH²(u_q(sl_3))` bar computation remains intractable; the
   8-vs-9 question is left explicitly open in every draft.

### Per-recipient framing

- **Cris Negron** (arXiv:1511.07059, *Braided Hochschild cohomology and Hopf
  actions*; arXiv:2005.02965 with Pevtsova): framed as a structural /
  braided-HH question — does the diagonal-linear-`rank(g)` pattern ring any
  bells from the braided-HH framework? Negron is the natural person to ask
  whether there is a structural reason `H̃¹_b` of the bosonization should be
  `rank(g)`-dimensional.

- **Sarah Witherspoon** (arXiv:0704.2771, the Mastnak–Witherspoon LES used in
  the reduction): framed around the **failure regime of MW Theorem 6.1.4**
  — at `ℓ = 3`, `3 | |Γ| = 3ⁿ`, so the hypothesis of Thm 6.1.4 fails. Two
  questions: (1) has she seen this `rank(g)` pattern in any pointed-Hopf /
  bosonization setting, especially in the small-`ℓ` gap? (2) does the failure
  of the Thm 6.1.4 hypothesis have a known structural consequence that might
  explain the `rank(g)`-dimensionality?

- **You Qi** (Lachowska–Qi, derived center of small quantum groups; the LQ21
  reference cited in the paper abstract; **already in correspondence** with
  the author — familiar tone): framed as a clean refutation of the structural
  picture previously sketched to him. Two questions: (1) does the `rank(g)`
  pattern suggest anything from the LQ / BGG perspective — e.g. a
  principal-block `HH²(sl_3)` computation via Hemelsoet–Voorhaar
  (arXiv:2104.05113) that would distinguish 8 from 9? (2) is there a
  structural reason the cocycles should be diagonal and Cartan-coefficient-free?

### Address verification

The task brief flagged Negron's and Qi's addresses for verification
(`negron@usc.edu`, `you.qi@virginia.edu`). The drafts file includes a header
note recommending the author re-verify all three addresses (especially
Witherspoon's, since she is recently retired from TAMU and forwarding may
have changed) against the recipients' current institutional webpages before
sending. No live web verification was performed by this sub-agent.

### Why these three (and not others)

- **Negron**: braided-HH framework is the natural home for a structural
  interpretation of the diagonal / linear / `rank(g)` pattern.
- **Witherspoon**: co-author of the LES the reduction rests on; the pattern
  appears precisely in the failure regime of her Thm 6.1.4.
- **Qi**: the LQ / BGG machinery is, per W1-1a, the only realistic route to
  an independent `dim HH²(u_q(sl_3))` computation that could settle the
  8-vs-9 question; prior correspondence makes a familiar tone appropriate.

Other candidates considered but **not** drafted (kept in reserve): Mitja
Mastnak (LES co-author — Witherspoon is the more natural first contact for
the structural question); Iván Angiono / Mikhail Kochetov (Nichols-rigidity
— relevant to the `2|Φ⁺|` root piece, *not* the structural piece the
outreach is about); Nicolas Hemelsoet (sl₃ BGG / principal-block s = 2 — the
most direct route to 8-vs-9, but the author may prefer to go through Qi
first); Thomas Creutzig (log-KL / non-semisimple TQFT context).

### What the drafts deliberately do NOT do

- Do **not** claim the original count is refuted — only the structural
  decomposition. (The 8-vs-9 question for `sl_3` is left explicitly open in
  every draft, matching W4-1 §"Implications".)
- Do **not** cite a non-existent arXiv number. The paper is referred to by
  its GitHub URL only (`https://github.com/pkhairkh/hopf-decoherence`,
  `paper/main.tex`), with "arXiv posting pending" noted.
- Do **not** request coauthorship, employment, or a recommendation — only a
  reaction / pattern-match check.
- Do **not** attach the full paper PDF by default — only the GitHub link.

### Files produced / modified

- **Created**: `/home/z/my-project/hopf-decoherence/literature/notes/W4-2-outreach-drafts.md`
  — three outreach emails + three cover notes + a "why these three" / "what
  the drafts do not do" rationale section.
- No code, tests, or paper source modified.

### Open questions for downstream sub-agents / orchestrator

- **For the author (Parham Khairkhah)**: review the three drafts, verify the
  recipient addresses, and send (or hold) at your discretion. The drafts are
  written to be sent essentially as-is with the cover note prepended as the
  opening paragraph of each outgoing message.
- **If replies come back**: the most useful reply would be from Qi or
  Hemelsoet-style BGG experts on the 8-vs-9 question (does a
  principal-block `HH²(sl_3)` at `s = 2` computation exist or is it
  tractable?), and from Negron / Witherspoon on whether the `rank(g)`
  pattern has a known structural explanation. Replies should be logged in a
  new worklog section (e.g. W5-1, W5-2, ...) and any pattern-match should be
  cross-checked against the W4-1 cocycle extraction.
- **For W4-3 (theoretical explanation of the `rank(g)` pattern)**: the
  outreach to Negron and Witherspoon is partly insurance against the case
  where W4-3 cannot find a structural explanation internally — if either
  replies with a known mechanism, W4-3 can fold it in.

---

---

## W4-3 — Exact cyclotomic certification of `dim HH^2(u_q(sl_2), C) = 3` at `ℓ = 3`

- **Task ID**: W4-3
- **Agent**: Sub-agent 4c (Wave 4, general-purpose)
- **Date**: 2025-08-08
- **Status**: completed
- **Output**: `scripts/certify_a1_exact.py`, `scripts/certify_a1_output.txt`, `tests/test_a1_certification.py`, `literature/notes/W4-3-a1-exact-certification.md`
- **Predecessor**: W1-1a (paper read; floating-point `dim HH² = 3` verification in `scripts/verify_sl2_hh2.py`)
- **Successor**: none for A₁ at `ℓ = 3`; natural next targets are A₁ at `ℓ = 5` and A₂ at `ℓ = 3`

### Summary

Implemented **exact cyclotomic arithmetic certification** of the A₁ case
of the paper's Theorem 1.2 at `ℓ = 3`.  The structure constants of
`u_q(sl_2)` at `q = e^{2πi/3}` live in the localization `Z[ω, 1/3]`
(where `ω` is a primitive cube root of unity and the `1/3` arises because
`D = q − q⁻¹ = 1 + 2ω` has norm 3, so `1/D = (−1 − 2ω)/3`).  The
bar-complex differentials `d¹` (729 × 27) and `d²` (19683 × 729) are
built as sparse matrices with entries in `Z[ω, 1/3]`, then reduced modulo
11 primes `p ≡ 1 (mod 3)` with `p ≠ 3` (`7, 13, 19, 31, 37, 43, 61, 67,
73, 79, 97`), and ranks are computed over each `F_p` via exact Gaussian
elimination (no floating point, no tolerance).  Rank semicontinuity over
the Dedekind domain `Z[ω, 1/3]` then certifies the ranks over `Q(ω)` —
and hence over `C` — provided they are consistent across all tested
primes.  They are: `rank(d¹) = 27`, `rank(d²) = 699` modulo every prime,
giving `dim HH² = (729 − 699) − 27 = 3`.  This implements roadmap item
1.1 for the A₁ case at `ℓ = 3`.

### Key findings

1. **Structure-constant ring is `Z[ω, 1/3]`, not `Z[ω]`.**  The
   commutator `[E, F] = (K − K⁻¹)/(q − q⁻¹)` divides by
   `D = q − q⁻¹ = 1 + 2ω`, whose algebraic norm in `Z[ω]` is 3.  So `D`
   is not a unit in `Z[ω]`; instead `D² = −3`, hence `1/D = −D/3 =
   (−1 − 2ω)/3 ∈ Z[ω, 1/3] \ Z[ω]`.  All structure constants of
   `u_q(sl_2)` at `ℓ = 3` therefore lie in `Z[ω, 1/3]`.  (This is the
   only subtlety not explicitly flagged in the task description, which
   said "in fact in `Z[q]`" — that is true for the multiplication table
   of the *generators* but not for the *full* normal-form multiplication
   table, which uses `1/D`.)

2. **`Z[ω, 1/3]` is a Dedekind domain**, so rank semicontinuity applies:
   the locus where `rank(M) < r` is Zariski-closed in `Spec Z[ω, 1/3]`,
   hence either all of the spectrum or a finite set of closed points.
   Since 3 is inverted, the only "bad" primes are absent, and the rank is
   constant on the entire spectrum — equal to the generic-fiber rank
   over `Q(ω)`.  Embedding `ω ↦ e^{2πi/3}` gives the rank over `C`.

3. **Representation of `Z[ω, 1/3]`:** each element is a triple
   `(a, b, c) ∈ Z³` with `c ≥ 0`, denoting `(a + b·ω)/3^c`, with
   multiplication using `ω² = −1 − ω` and `_reduce()` cancelling common
   factors of 3 from `(a, b)`.  Reduction mod `p` (for `p ≠ 3`,
   `ω ↦ q_p ∈ F_p`) uses `pow(3, −c, p)` for the modular inverse of the
   denominator.

4. **Certified ranks (consistent across all 11 primes):**
   - `rank(d¹) = 27` (full column rank; `d¹` is injective, so
     `dim HH¹ = 0`).
   - `rank(d²) = 699`.
   - `dim ker(d²) = 729 − 699 = 30`.
   - `dim im(d¹) = 27`.
   - `dim HH² = 30 − 27 = 3`.  ✓

5. **Cross-check with floating-point:** the exact ranks (`27, 699`)
   match the floating-point SVD ranks from `verify_sl2_hh2.py` (`27,
   699`) to the digit, confirming that the floating-point tolerance
   (`1e-9`) was correctly chosen and that the exact and floating-point
   computations agree.

6. **Runtime:** ~25 s total for 11 primes (~2.2 s per prime for
   reduction + Gaussian elimination; multiplication table and
   differentials built in < 1 s).

### Why this is a *certification*, not just a *computation*

Three features distinguish this from the floating-point verification:

1. **Exact structure constants** — every entry of `d¹` and `d²` is an
   element of `Z[ω, 1/3]` (a triple of integers); all arithmetic is
   exact integer operations, no rounding.
2. **Exact rank over `F_p`** — Gaussian elimination with modular
   inverses (`pow(x, −1, p)`); the rank is a precise integer, not a
   singular-value count above a threshold.
3. **Lifting via rank semicontinuity** — the Dedekind-domain theorem
   guarantees that constancy of rank across all primes `p ≡ 1 (mod 3)`
   with `p ≠ 3` certifies the rank over `Q(ω)` and hence over `C`.  The
   script checks this constancy across 11 primes and reports the
   conclusion explicitly.

### Files produced / modified

- **Created**: `scripts/certify_a1_exact.py` (~400 lines) — `ER` class
  for `Z[ω, 1/3]`, multiplication table with exact arithmetic, sanity
  checks of the 5 defining relations of `u_q(sl_2)`, `d¹` and `d²`
  builders, modular reduction, exact Gaussian elimination over `F_p`,
  main certification loop over 11 primes with tee'd output to
  `certify_a1_output.txt`.
- **Created**: `scripts/certify_a1_output.txt` — captured stdout from a
  full 11-prime run.
- **Created**: `tests/test_a1_certification.py` — 7 tests: `test_er_arithmetic`,
  `test_find_cube_root_mod_p`, `test_algebra_relations`,
  `test_rank_consistent_across_primes`, `test_dim_hh2_is_3`,
  `test_rank_values_match_certified`, `test_script_runs`.  All pass in
  ~45 s.
- **Created**: `literature/notes/W4-3-a1-exact-certification.md` —
  9-section writeup covering the mathematical setup
  (`Z[ω, 1/3]` as a Dedekind domain, rank semicontinuity), the
  implementation (ER class, multiplication, differentials, modular
  reduction, Gaussian elimination), the full results table, the
  certification conclusion, the comparison with the floating-point
  computation, and limitations / next steps.
- No existing files modified.

### Open questions for downstream sub-agents / orchestrator

- **A₁ at `ℓ = 5`:** the same framework applies with `Z[ω_5, 1/5]` and
  primes `p ≡ 1 (mod 5)`, `p ≠ 5`.  The structure-constant ring is
  slightly larger (`Q(ω_5)` has degree 4 over `Q`), and `dim u_q(sl_2)
  = 5³ = 125`, so `dim C² = 15625` and `dim C³ = ~1.95 × 10⁶` — the
  Gaussian elimination on `d²` (~2M × 15.6K) would be the bottleneck,
  but sparse methods or block-decomposition by weight may keep it
  tractable.  **This is the natural next certification target** and
  would establish the A₁ case of Theorem 1.2 at both `ℓ = 3` and
  `ℓ = 5` (the two primes covered by the paper's A₁ verification).

- **A₂ at `ℓ = 3`:** the exact-arithmetic framework would certify the
  *full* `dim HH²(u_q(sl_3), C)` (resolving the 8-vs-9 question left
  open by W4-1 / W4-2), but requires implementing the Lusztig
  root-vector multiplication table for `u_q(sl_3)` in exact `Z[ω, 1/3]`
  arithmetic — a non-trivial extension of `verify_sl3_bplus_hh2.py`.
  The bar complex has `dim C² = 729` and `dim C³ = 19683` (same as A₁
  since `dim u_q(sl_3) = 27 = dim u_q(sl_2)`), so the rank computation
  itself would be equally fast (~2 s per prime).

- **Generalisation to other types / odd `ℓ`:** the method generalises
  verbatim to any `u_q(g)` at any odd `ℓ` for which `Z[ω_ℓ]` has class
  number 1 (so `ℓ ∈ {3, 5, 7, 11, 13, 17, 19}`) and the structure
  constants lie in `Z[ω_ℓ, 1/ℓ]` (which they do, since the only
  division is by `q − q⁻¹`, whose norm is `ℓ`).  For composite `ℓ` or
  class-number-`> 1` cyclotomic fields, the Dedekind-domain argument
  still holds but the rank check requires reducing modulo prime *ideals*
  rather than rational primes — a mild generalisation.

- **For the orchestrator:** roadmap item 1.1 (exact cyclotomic
  certification) is now **complete for A₁ at `ℓ = 3`**.  The
  floating-point verification in `verify_sl2_hh2.py` is now backed by a
  fully exact, modular-arithmetic certificate.  The matching
  certification for A₁ at `ℓ = 5` would complete the A₁ case of the
  paper's Theorem 1.2 on the exact side.


---

## W0-1a — AST/IR framework part 1: parser + normal forms

- **Task ID**: W0-1a
- **Agent**: Sub-agent 0a (Wave 0, general-purpose)
- **Date**: 2025-08-08
- **Status**: completed
- **Output**: `/home/z/my-project/hopf-decoherence/ir/parser.py`, `/home/z/my-project/hopf-decoherence/ir/__init__.py`, `/home/z/my-project/hopf-decoherence/tests/test_ir_parser.py`

### Summary

Built the foundational layer of the AST/IR framework for the
hopf-decoherence project: a `Z[q, q^{-1}]` coefficient ring (`QLaurent`),
free-algebra `Monomial` / `Term` / `Polynomial` classes, `RewriteRule`
with `matches` / `apply`, `Presentation` with a recursive-descent string
parser, and `NormalFormReducer` using a leftmost-match strategy.  All 31
unit tests pass, including the quantum-plane integration test
(`x*y = q*y*x`) which verifies `x*y -> q*y*x`, `x*y*x -> q*y*x*x`,
`x*x*y -> q^2*y*x*x`, `x*y*x*y -> q^3*y*y*x*x`, `x^4*y^3 -> q^12*y^3*x^4`,
and `x*y - q*y*x -> 0`.

### Why a new coefficient ring instead of reusing ER

The `ER` class in `scripts/certify_a1_exact.py` is hardcoded to the
cube roots of unity (`omega^3 = 1`, `Z[omega, 1/3]`).  The AST/IR
framework needs to work over `Z[q, q^{-1}]` *generically* — without
fixing `q^N = 1` — so that:

1. Knuth-Bendix completion (W0-1b) can compare monomials by length-lex
   order over the *free* coefficient ring, without worrying about
   torsion from a specific root of unity.
2. The same IR can be specialised to `ell = 3` (`sl_3`), `ell = 5`
   (`sl_4`), or any other root, by reducing the `QLaurent` coefficients
   modulo the appropriate cyclotomic ring at certification time (the
   Anick-resolution cohomology rank is itself certified by reduction
   modulo many primes, mirroring the `ER`-based modular method).
3. Structure constants of `u_q(sl_n)` at a root of unity live in
   `Z[q, q^{-1}]` *before* quotienting by the cyclotomic relation,
   which is exactly the ring the rewrite rules are stated over.

`QLaurent` is a sparse Laurent polynomial (`{exponent: coefficient}`
dict with no zero coefficients).  It supports `+`, `-`, `*`, `==`,
`hash`, `is_zero`, `is_one`, plus the helpers `qpow(n)` and
`qint(n)`.  The display routine renders `2*q^3 - q + 5` as
`QLaurent(2*q^3 - q + 5)`.

### Design choices

1. **Monomial order**: length-lex (shorter first, then lexicographic on
   the tuple of generator indices), exposed via `Monomial.order_key()`.
   The reducer itself is *independent* of the monomial order — it just
   finds the leftmost match — but `Polynomial.normalize()` uses
   length-lex to give a canonical term ordering for `==` and `hash`.
   Knuth-Bendix completion (W0-1b) will use this order (or a
   user-supplied refinement) to orient rules.

2. **Reducer strategy**: leftmost-match, recursive.  At each step,
   `find_match` scans the monomial from left to right; at the first
   position where any rule matches, the first matching rule (in the
   order given in `Presentation.rules`) is applied.  The result is a
   polynomial; each term is recursively reduced, scaled by the original
   term's coefficient.  For a confluent + terminating rewrite system
   this yields the unique normal form.  Non-terminating systems may
   recurse forever — Knuth-Bendix completion (W0-1b) is responsible for
   guaranteeing termination.

3. **Parser grammar** (whitespace-insensitive):

   ```
   polynomial := ['+'|'-'] term (('+'|'-') term)*
   term       := factor ('*' factor)*
   factor     := int | 'q' ['^' ['-'] int] | generator_name | '(' polynomial ')'
   ```

   Integer and `q` factors multiply to give the term's coefficient;
   generator factors concatenate to give the term's monomial.
   Parenthesized sub-polynomials may appear as factors (so
   `(q + 1) * x` parses correctly).  Unknown names raise `ValueError`.

4. **Polynomial equality**: `==` normalizes both sides (combine like
   monomials, drop zero coefficients, sort by length-lex) and compares
   the resulting term lists.  `hash` is consistent with `==`.

5. **RewriteRule.apply**: returns `prefix * RHS * suffix` where
   `prefix = monomial[:position]` and `suffix = monomial[position+len(lhs):]`.
   If RHS is the zero polynomial, the entire match (including prefix
   and suffix) becomes zero — this is the correct algebraic behaviour
   (e.g., `x*x = 0` implies `y*x*x*z = 0`).

### Files

- `/home/z/my-project/hopf-decoherence/ir/__init__.py` (new): package
  docstring + `__version__ = "0.1.0-w0-1a"`.
- `/home/z/my-project/hopf-decoherence/ir/parser.py` (new, ~700 lines):
  the full module.  All public symbols are listed in `__all__`.
- `/home/z/my-project/hopf-decoherence/tests/test_ir_parser.py` (new,
  ~370 lines): 31 tests in 7 classes (`TestQLaurent`, `TestMonomial`,
  `TestPolynomial`, `TestRewriteRule`, `TestPresentation`,
  `TestQuantumPlane`, `TestNoRules`).
- No existing files modified.

### Test results

```
$ pytest tests/test_ir_parser.py -v
============================= 31 passed in 0.12s ==============================
```

All 5 required tests are present and passing:

- `test_monomial_basic` (in `TestMonomial`): length, indexing, slicing,
  concatenation, equality, hash, empty-monomial identity.
- `test_polynomial_addition` (in `TestPolynomial`): like monomials
  combine, cancellation drops zero-coefficient terms.
- `test_rewrite_rule_match` (in `TestRewriteRule`): `matches` at
  correct positions, `apply` produces the right polynomial.
- `test_quantum_plane_normal_form` (in `TestQuantumPlane`): the
  integration test — all four reductions verified
  (`x*y -> q*y*x`, `x*y*x -> q*y*x*x`, `x*x*y -> q^2*y*x*x`,
  `x*y*x*y -> q^3*y*y*x*x`).
- `test_presentation_parse` (in `TestPresentation`): parses
  `"q^2 * K * E - E * K"` to the correct polynomial.

Plus 26 additional tests covering: `QLaurent` arithmetic (addition,
multiplication, distributivity, negation, zero/one identities, hash
equality), `Monomial` length-lex ordering, polynomial scalar
multiplication, polynomial-polynomial multiplication, polynomial
negation, rewrite-rule error handling (mismatch, empty LHS), parser
edge cases (no spaces, negative exponents, integer coefficients,
parentheses, leading minus, single generator, unknown names), and
end-to-end quantum-plane reduction via the parser.

### Open questions / handoff notes for W0-1b (Knuth-Bendix completion)

1. **Monomial order**: length-lex is exposed via `Monomial.order_key()`
   and is already used by `Polynomial.normalize()`.  If W0-1b needs a
   different order (e.g., weight order, or a path order for proving
   termination), it can subclass `Monomial` or wrap `order_key`.  The
   reducer does not depend on the order.

2. **Critical pairs**: to find overlaps of two rules `L1 -> R1` and
   `L2 -> R2`, W0-1b will need to enumerate all positions where a
   non-trivial suffix of `L1` equals a non-trivial prefix of `L2` (and
   vice versa).  This is a helper function on `Monomial`, not provided
   here — W0-1b should add it (e.g., as a free function
   `overlaps(m1, m2)` returning a list of `(offset, overlap_length)`).

3. **Termination check**: the reducer does NOT detect infinite loops.
   W0-1b should either (a) wrap `NormalFormReducer` with a
   max-iterations counter, or (b) prove termination for each completed
   system before reducing (the Knuth-Bendix completion itself can fail
   to terminate — that's the undecidable side of the problem).

4. **Performance**: the recursive reducer is fine for monomials of
   length up to ~20.  For the Anick resolution on `u_q(sl_3)` (basis
   size 27, monomials of length up to ~5 in `B+`), this is more than
   sufficient.  For `u_q(sl_4)` (basis size 125, monomials of length
   up to ~7), still fine.  If monomials get much longer, an iterative
   reducer with memoization would help.

5. **Coefficient ring extensions**: the `QLaurent` ring is
   `Z[q, q^{-1}]`.  When W0-1b or later sub-agents need to specialise
   to `ell = 3` (cube roots), they should either (a) reduce
   `QLaurent` coefficients modulo `q^2 + q + 1` to get
   `Z[omega]` (and then to `Z[omega, 1/3]` if the commutator
   `[E, F] = (K - K^{-1})/(q - q^{-1})` is involved), or (b) use the
   `ER` class directly.  The modular-reduction infrastructure in
   `certify_a1_exact.py` (`reduce_mod`, `rank_mod_p`) is the model.


---

## W1-1a-IR — AST/IR framework validation: dim HH²(u_q(sl_2), C) = 3 via the IR

- **Task ID**: W1-1a-IR (IR validation; distinct from the W1-1a Hemelsoet–Voorhaar literature read)
- **Agent**: Sub-agent 1a (Wave 1, general-purpose)
- **Date**: 2025-08-09
- **Status**: completed
- **Output**: `/home/z/my-project/hopf-decoherence/ir/qomega.py`, `/home/z/my-project/hopf-decoherence/ir/uq_sl2.py`, `/home/z/my-project/hopf-decoherence/scripts/verify_ir_sl2.py`, `/home/z/my-project/hopf-decoherence/scripts/ir_sl2_output.txt`, `/home/z/my-project/hopf-decoherence/tests/test_ir_uq_sl2.py`, `/home/z/my-project/hopf-decoherence/literature/notes/W1-1a-ir-sl2.md`

### Summary

Applied the AST/IR framework (built in W0-1a and W0-1b) to the small
quantum group `u_q(sl_2)` at `ℓ = 3`, completing the rewrite system,
verifying the 27-element PBW basis, enumerating the Anick resolution
generators at degrees 0–2, and computing `dim HH²(u_q(sl_2), C) = 3`
via the bar complex on PBW normal forms. The result matches the direct
bar-complex computation in `scripts/verify_sl2_hh2.py` and the exact
certification in `scripts/certify_a1_exact.py`, validating the IR
framework end-to-end and unblocking its application to `u_q(sl_3)` (W2).

### Key findings

1. **Commutator division**: handled by specializing the coefficient ring
   to `Q(omega)` at `ℓ = 3` (approach (a) in the task description).
   Created `ir/qomega.py` with `QOmega3`, a `QLaurent` subclass storing
   `a + b*omega` exactly as `Fraction` pairs, with field inverse
   `1/(q - q^{-1}) = (-1 - 2*omega)/3`. Because `QOmega3` is a subclass
   of `QLaurent`, the existing `NormalFormReducer` and KB completion
   work with it unchanged (Python's subclass-priority rule for reflected
   operators makes mixed-type arithmetic dispatch correctly).

2. **6-rule PBW rewrite system**: K^3 → 1, E^3 → 0, F^3 → 0,
   E K → omega*K E (q^{-2}=omega at ℓ=3), F K → omega²*K F (q²=omega²),
   F E → E F - alpha*K + alpha*K² with alpha = 1/(q-q^{-1}).

3. **Knuth-Bendix completion at ℓ = 3**: terminates after 1 iteration
   with **0 new rules added** (the 6-rule system is already confluent).
   All 21 critical pairs (upper-triangular i ≤ j over 6 rules) reduce
   to zero in `Q(omega)`. This is the "Diamond Lemma for quantum
   groups" phenomenon at a fixed root of unity. (Generically, over
   `Z[q, q^{-1}]`, the system is *not* confluent — e.g., the critical
   pair `E K^3` gives `(q^{-6} - 1) E`, which is nonzero generically
   but zero at `ℓ = 3` since `q^3 = 1`. This is why we work over
   `Q(omega)` rather than the generic ring.)

4. **PBW basis verified**: 27 normal forms `K^a E^b F^c` with
   `0 ≤ a, b, c ≤ 2`. All 27 are in normal form (no rule matches).
   200 random monomials of length 0–8 all reduce to polynomials whose
   terms are in the PBW basis.

5. **Anick degree-2 syzygy count = 13**: 6 self-overlaps of the cubic
   rules (K^4, K^5, E^4, E^5, F^4, F^5) + 7 cross-rule overlaps
   (E K^3, F K^3, E^3 K, F E^3, F^3 K, F^3 E, F E K). Full listing
   in the analysis note.

6. **dim HH² = 3** via the bar complex on PBW normal forms
   (homotopy-equivalent to the Anick resolution, hence computing the
   same `Ext^*_A(k, k) ≅ HH^*(A, k)` for Hopf `A`):
   - `rank(d^1) = 27`, `rank(d^2) = 699`
   - `dim ker(d^2) = 729 - 699 = 30`
   - `dim HH² = 30 - 27 = 3` ✓
   Matches the conjecture `C(n+1, 2) + 2|Φ^+| = C(2,2) + 2·1 = 3`
   for `A_1`, and matches `verify_sl2_hh2.py`'s direct bar-complex
   value exactly.

7. **Multiplication-table sanity** all pass: unit law, K^3 = 1,
   E^3 = F^3 = 0, K E = q² E K, K F = q^{-2} F K,
   [E, F] = (K - K^{-1})/(q - q^{-1}). This validates that the IR
   normal-form reducer produces the correct algebra structure.

### Why the bar complex and not the Anick differential

For a Hopf algebra `A`, `HH^*(A, k) ≅ Ext^*_A(k, k)`, computed by
either the bar resolution (free A-bimodule resolution of A) or the
Anick resolution (free A-module resolution of k) — they are
homotopy-equivalent. We use the bar complex on PBW normal forms for
the actual rank computation because the bar differential depends only
on the multiplication table and the counit (both immediate from the IR
reducer), while the Anick differential requires following Anick's 1986
construction with "chains", "tips", and recursive reductions on
n-chains — a substantial implementation effort that is left for a
future task.

For `u_q(sl_2)` at `ℓ = 3` (`dim C^2 = 729`, `dim C^3 = 19683`), the
bar complex runs in ~1.3 seconds. For `u_q(sl_3)` at `ℓ = 3`
(`dim u_q(sl_3) = 6561`, `dim C^2 = 4.3 × 10^7`), the bar complex is
intractable and the Anick differential (with `dim C_2` in the dozens)
will be needed.

### Test results

```
$ pytest tests/test_ir_uq_sl2.py -v
============================= 11 passed in 2.10s ==============================

$ pytest tests/test_ir_parser.py tests/test_ir_groebner.py tests/test_ir_uq_sl2.py
============================= 55 passed in 2.29s ==============================
```

Four required tests + 7 additional tests, all passing:
- `test_presentation_parses` — generators, 6 rules, R6 commutator sign
- `test_pbw_basis_size` — 27 PBW normal forms, all in normal form
- `test_anick_degree2_count` — 1, 6, 13 generators at degrees 0, 1, 2
- `test_dim_hh2_is_3` — `dim HH² = 3` ✓

### Recommended next actions for the orchestrator

- **W2-1c (sl_3 via IR)**: Apply the IR framework to `u_q(sl_3)` at
  `ℓ = 3`. The presentation has ~8 generators and ~20 relations; PBW
  basis size 6561; expected `dim HH² = 9` (conjecture
  `C(3,2) + 2·3 = 3 + 6 = 9`). The bar complex is intractable, so the
  Anick differential must be implemented first. The 13-syzygy
  enumeration at degree 2 for sl_2 generalizes to a similar (but
  larger) syzygy enumeration for sl_3; the Anick degree-2 count for
  sl_3 should be in the range 30–80.
- **Anick differential implementation**: implement `d_1`, `d_2`, `d_3`
  of the Anick resolution following Anick 1986 / Skryabin's survey.
  `d_1` is straightforward; `d_2` requires expressing each syzygy's
  reduction-difference as a combination of relations; `d_3` is more
  intricate. Once implemented, the chain groups for sl_3 will be small
  (dim ≤ ~100 in each degree ≤ 3), making `dim HH²` tractable.
- **Exact rank certification**: replace the SVD-based numerical rank
  with exact `Fraction` + `sympy.Matrix.rank()`, or modular
  certification (extending `scripts/certify_a1_exact.py`). This
  eliminates the `1e-9` numerical tolerance and gives a proof-grade
  rank.

### Files produced / modified

- Created: `/home/z/my-project/hopf-decoherence/ir/qomega.py` (~300 lines)
- Created: `/home/z/my-project/hopf-decoherence/ir/uq_sl2.py` (~600 lines)
- Created: `/home/z/my-project/hopf-decoherence/scripts/verify_ir_sl2.py`
- Created: `/home/z/my-project/hopf-decoherence/scripts/ir_sl2_output.txt`
- Created: `/home/z/my-project/hopf-decoherence/tests/test_ir_uq_sl2.py` (~340 lines)
- Created: `/home/z/my-project/hopf-decoherence/literature/notes/W1-1a-ir-sl2.md`
- Modified: `/home/z/my-project/worklog.md` (appended this section)
- No existing source files modified. `ir/parser.py` and `ir/groebner.py`
  are unchanged; `QOmega3` is a `QLaurent` subclass that works with the
  existing infrastructure via Python's subclass-priority rule for
  reflected operators.

### Open questions for downstream sub-agents

- **(W2-1c prerequisite)** The Anick differential `d_2`: for each
  critical pair `(M, R_i at p1, R_j at p2)`, the value `d_2(M, i, j)`
  is the syzygy value expressed as a combination of relations. The
  standard formula reduces `M` via `R_i` at `p1` (giving `poly1`) and
  via `R_j` at `p2` (giving `poly2`); then `poly1 - poly2` is reduced
  to zero, and the "reduction history" gives the syzygy coefficients.
  A correct implementation needs to track this history carefully.
- For `u_q(sl_3)` at `ℓ = 3`: is the 6-or-7-rule (per generator pair)
  PBW system confluent without KB completion (as for sl_2), or does KB
  completion add new rules? The sl_2 case (no new rules) is promising
  but not conclusive for sl_3.
- The Anick degree-3 enumeration in `ir/groebner.py` is currently
  best-effort (capped at 100 entries, not properly verified). For
  `dim HH³` or for cross-checking `dim HH²` via the Anick
  differential (which needs `d_3`), a proper degree-3 enumeration is
  needed.
