# W1-1d — Deep read of Creutzig–Lentner–Rupert (arXiv:2306.11492)

- **Task ID**: W1-1d
- **Agent**: Sub-agent 1d (Wave 1, general-purpose)
- **Date**: 2025-08-07
- **Source**: `/home/z/my-project/hopf-decoherence/literature/texts/Creutzig-Lentner-Rupert-log-KL.txt`
- **Paper**: T. Creutzig, S. Lentner, M. Rupert, *An algebraic theory for logarithmic Kazhdan-Lusztig correspondences*, arXiv:2306.11492v1, 20 Jun 2023.

---

## 1. Main theorem — the logarithmic KL correspondence they prove

The main KL correspondence is **Theorem 1.3 (1)**, restated and proven as **Theorem 8.9** (with an independent proof in **Example 9.10** via Theorem 9.5). Verbatim from the text (lines 693–707):

> **Theorem 1.3.** The following are equivalent as braided tensor categories
>
> 1. Let $\mathcal{O}_{M(p)}^T$ the category of weight modules of the singlet algebra of [CMY23a] and $\mathrm{Rep}^{wt} u_q^H(\mathfrak{sl}_2)$ the category of weight modules of the small unrolled quantum group of $\mathfrak{sl}_2$ at $2p$-th root of unity of [CGP15]. Then
>
> $$\mathcal{O}_{M(p)}^T \;\cong\; \mathrm{Rep}^{wt}(u_q^H(\mathfrak{sl}_2))$$
>
> as braided tensor categories (Theorem 8.9 as well as Example 9.10).

Parameters: $p \in \mathbb{Z}_{\geq 2}$, $q = e^{\pi i/p}$ (a primitive $2p$-th root of unity — always **even** order). Other parts of Theorem 1.3 give: (2) the triplet VOA $W(p)$ is braided equivalent to $\mathrm{Rep}(\tilde{u}_q(\mathfrak{sl}_2))$ (quasi-Hopf, [CGR20; GLO18]) — obtained by *uprolling* (Corollary 10.6); (3) the $S(p)$ VOA correspondence (Lemma 10.7); (4) the affine $gl_{1|1}$ VOA correspondence (Corollary 10.8). The structural engine is **Theorem 8.1** (recognition via Nichols algebra in $\mathrm{Vect}^Q_\Gamma$) and **Theorem 9.5** (recognition via Takeuchi–Skryabin relative Hopf modules).

---

## 2. The unrolled quantum group $u_q^H(\mathfrak{sl}_2)$

**Definition.** The unrolled quantum group $u_q^H(\mathfrak{sl}_2)$ is the realizing Hopf algebra for the relative Drinfeld center $\mathcal{U} = Z_{\mathcal{C}}(\mathcal{B})$ where $\mathcal{C} = \mathrm{Vect}^Q_\mathbb{C}$ is the category of $\mathbb{C}$-graded vector spaces with quadratic form $Q(\lambda) = e^{\pi i \lambda/p}$ (Example 6.9, lines 3713–3726; construction in Example 6.7). The Cartan part is $u_q^H(\mathfrak{sl}_2)_0 = \mathbb{C}[\mathbb{C}] = \mathbb{C}[K, K^{-1}]$ with **no quotient** to a finite group, so the algebra $u_q^H(\mathfrak{sl}_2)$ is **infinite-dimensional** (line 3216: "$u_q^H_q(g)_0$ of the unrolled quantum group"). The Borel part (Nichols algebra $\mathcal{B}(X) = \mathbb{C}[x]/x^p$) and the rest of the structure match those of $u_q(\mathfrak{sl}_2)$.

**Difference from standard $u_q(\mathfrak{sl}_2)$.** Three variants are obtained from the same Nichols algebra $\mathcal{B}(X) = \mathbb{C}[x]/x^p$ by different choices of $\Gamma$ (lines 466–470, 3709–3712):

| Variant | $\Gamma$ | Order of $q$ | Realizing algebra |
|---|---|---|---|
| small quantum group $u_q(\mathfrak{sl}_2)$ | $\mathbb{Z}_p$ | odd, $p$ | finite-dim. Hopf |
| quasi-Hopf $\tilde{u}_q(\mathfrak{sl}_2)$ | $\mathbb{Z}_{2p}$ | even, $2p$ | finite-dim. quasi-Hopf, nontrivial associator |
| unrolled $u_q^H(\mathfrak{sl}_2)$ | $\mathbb{C}$ | any $2p$-th | infinite-dim. Hopf |

So $u_q^H(\mathfrak{sl}_2)$ differs from $u_q(\mathfrak{sl}_2)$ in two ways: (i) the Cartan part is **infinite** (not quotiented to $\mathbb{Z}_p$ or $\mathbb{Z}_{2p}$); (ii) it is a Hopf algebra (no associator), even when $q$ has even order — at the cost of being infinite-dimensional.

**Category equivalence.** $\mathrm{Rep}^{wt}(u_q^H(\mathfrak{sl}_2))$ is **NOT** braided-equivalent to $\mathrm{rep}(u_q(\mathfrak{sl}_2))$. The two braided tensor categories share the same Nichols-algebra/Borel part but differ in the Cartan part (infinite vs. finite). This is the central obstruction to using this paper's KL correspondence to compute $\mathrm{HH}^2(u_q(\mathfrak{sl}_2),\mathbb{C})$ via the VOA side (see §4 below). The KL equivalence transfers structure between $\mathcal{O}_{M(p)}^T$ and $\mathrm{Rep}^{wt}(u_q^H(\mathfrak{sl}_2))$ — both of which are **infinite** braided tensor categories (the singlet has infinitely many simples $M_{r,s}$ indexed by $r \in \mathbb{Z}_{\geq 1}$, $s \in \{1,\dots,p-1\}$).

---

## 3. Extension to $\mathfrak{sl}_3$ or higher rank

**Not proven in this paper.** Three places discuss higher rank:

- **Example 2.12 (1)(a)–(b)** (lines 1820–1830): the Feigin–Tipunin algebra $\mathrm{FT}(g,p)$ for simply-laced $g$ is conjectured (but not proven) to have representation category equivalent to a quasi-Hopf modification of the small quantum group of $g$ at $2p$-th root of unity; "almost nothing is known about the representation theory of $\mathrm{FT}(g,p)$". The subalgebra $\mathrm{FT}^0(g,p)$ (the singlet analog for higher rank) is *expected* (conjectured, [CM17]) to correspond to the **unrolled** quantum group of $g$ at $2p$-th root of unity. **No higher-rank KL equivalence is proven here.**

- **Outlook, line 777**: "see [AM22, CRR23] for some results in the $\mathfrak{sl}_3$-case." [CRR23] = Creutzig–Ridout–Rupert, *A Kazhdan–Lusztig Correspondence for $L_{-3/2}(\mathfrak{sl}_3)$*, CMP 400 (2023) 639–682 — a **different** setup (admissible-level affine VOA $L_{-3/2}(\mathfrak{sl}_3)$, Kazhdan–Lusztig category, not the singlet analog for $\mathfrak{sl}_3$). Not a higher-rank singlet KL.

- **Outlook, line 821**: "Higher rank is currently completely out of reach" — explicitly stated for the affine VOA $V^k(\mathfrak{sl}_n)$ triality program.

**Status of $\mathfrak{sl}_3$ extension**: conjectural / open. The CLR 2306.11492 paper proves only rank-one ($\mathfrak{sl}_2$) cases.

---

## 4. Hochschild cohomology under braided equivalence

**General principle — YES.** If $\mathcal{C} \simeq \mathcal{D}$ as braided tensor categories, then $\mathrm{HH}^*(\mathcal{C}) \cong \mathrm{HH}^*(\mathcal{D})$ as algebras (in fact as $E_2$-algebras / Gerstenhaber algebras). This follows from the Schweigert–Woike framework (W1-1b): the canonical end $\mathcal{A} = \int_{X \in \mathcal{C}} X \otimes X^\vee \cong H_{\mathrm{ad}}$ is built from categorical data (unit, duality, tensor product, braiding) that is preserved by braided equivalence; the Hochschild cochain complex $\int_{X \in \mathrm{Proj}\mathcal{C}} \mathcal{C}(X,X)$ and its $E_2$-structure are therefore invariant. So in principle a braided equivalence $\mathcal{C} \simeq \mathcal{D}$ transfers $\mathrm{HH}^*$.

**Application of CLR's result to our setting — NO, by two mismatches:**

1. **Wrong algebra.** CLR's equivalence is $\mathcal{O}_{M(p)}^T \cong \mathrm{Rep}^{wt}(u_q^H(\mathfrak{sl}_2))$. The target is the unrolled quantum group $u_q^H(\mathfrak{sl}_2)$ — an **infinite-dimensional** Hopf algebra, NOT the standard small quantum group $u_q(\mathfrak{sl}_2)$ (finite-dimensional) that our conjecture is about. So CLR's theorem would give
   $$\mathrm{HH}^*(\mathcal{O}_{M(p)}^T) \;\cong\; \mathrm{HH}^*(\mathrm{Rep}^{wt}(u_q^H(\mathfrak{sl}_2)))$$
   but **does NOT give** $\mathrm{HH}^*(\mathrm{rep}(u_q(\mathfrak{sl}_2))) \cong \mathrm{HH}^*(\mathcal{O}_{M(p)}^T)$.

2. **Infinite categories.** Both $\mathcal{O}_{M(p)}^T$ and $\mathrm{Rep}^{wt}(u_q^H(\mathfrak{sl}_2))$ are **infinite** tensor categories (infinitely many simple objects, indexed by $\mathbb{Z}$). Schweigert–Woike's framework (W1-1b) is for **finite** tensor categories. Extension of $\mathrm{HH}^*$ to infinite categories exists (e.g. via ind-completions or the locally finite subcategory), but the dimension counts and finite-dimensionality of $\mathrm{HH}^2$ that our conjecture uses may fail in the infinite setting.

**Conclusion**: CLR's result does NOT imply $\mathrm{HH}^*(\mathrm{rep}(u_q(\mathfrak{sl}_2))) \cong \mathrm{HH}^*(\mathrm{rep}(M(p)))$ — neither as plain algebras nor as $E_2$-algebras — because the braided equivalence CLR proves involves a different category ($\mathrm{Rep}^{wt}(u_q^H)$, not $\mathrm{rep}(u_q)$) and a different VOA category ($\mathcal{O}_{M(p)}^T$, not a finite subcategory).

---

## 5. VOA-side HH computation

**No known formula.** The CLR paper does **not** compute $\mathrm{HH}^*(\mathrm{rep}(M(p)))$ or $\mathrm{HH}^*(\mathcal{O}_{M(p)}^T)$ in any form. The word "Hochschild" appears in the paper only once — in the bibliography, as a citation to Schweigert–Woike [SW22] (line 6928) for *homotopy coherent mapping class group actions and excision for Hochschild complexes of modular categories* (a different framework). No dimension counts, no $\mathrm{HH}^2$ calculation.

**At $p = 2$ (i.e. $q = e^{\pi i / 2} = i$, $\ell = 4$).** Not computed in CLR, and to our knowledge not computed in the surrounding literature (CMY21, CMY23a, CGP15) either. The CLR theorem at $p = 2$ equates $\mathcal{O}_{M(2)}^T \simeq \mathrm{Rep}^{wt}(u_i^H(\mathfrak{sl}_2))$ but does not give $\mathrm{HH}^2$ of either side.

**Does log-KL apply at odd $\ell$?** **NO.** The CLR theorem is stated for $p \in \mathbb{Z}_{\geq 2}$ with $q = e^{\pi i / p}$, a primitive $2p$-th root of unity. Therefore $\ell = 2p \in \{4, 6, 8, 10, \dots\}$ is **always even**. Our conjecture is at $\ell \in \{3, 5, 7\}$ — **odd**. The log-KL correspondence as proven by CLR does **not** cover odd $\ell$ at all. Even the broader KL framework (triplet $W(p)$, Feigin–Tipunin $\mathrm{FT}(g,p)$) always has $q = e^{\pi i / p}$ (even order $2p$).

For odd $\ell = 2p+1$, the corresponding quantum-group side would be the *standard* (non-unrolled, non-quasi) small quantum group $u_q(\mathfrak{sl}_2)$ at $q$ of odd order $2p+1$, with $\Gamma = \mathbb{Z}_{2p+1}$. The relevant VOA side is **not identified** in this paper or, to our knowledge, anywhere in the literature.

---

## 6. Frobenius–Perron dimensions

**Not addressed in this paper.** Grep confirms the paper contains no occurrences of "Frobenius–Perron", "Perron", "analytic character", or "asymptotic" (the only "Frobenius" hits are *Frobenius reciprocity* in categorical arguments, unrelated). Lentner's conditional proof (2501.10735) postdates this paper (June 2023 vs. Jan 2025) by 18 months; CLR does not cite it and could not.

**For $u_q(\mathfrak{sl}_3)$ at $\ell = 3$ specifically**: CLR provides **no information**. Whether the Lentner assumption (FP dimensions = asymptotics of analytic characters) holds for $u_q(\mathfrak{sl}_3)$ at $\ell = 3$ cannot be decided on the basis of this paper.

---

## 7. Path to $A_2$ via the VOA side

Assessment of the three options:

- **(a) Direct — compute $\mathrm{HH}^2(\mathrm{rep}(M(p)))$ at appropriate $p$, use KL to transfer.**
  **NOT VIABLE**, for three independent reasons:
  (i) The KL correspondence proven by CLR transfers $\mathrm{HH}^*$ between $\mathcal{O}_{M(p)}^T$ and $\mathrm{Rep}^{wt}(u_q^H(\mathfrak{sl}_2))$, **not** to $\mathrm{rep}(u_q(\mathfrak{sl}_n))$ — so the target side does not match our conjecture's algebra.
  (ii) The KL correspondence is proven only for $\mathfrak{sl}_2$, **not** for $\mathfrak{sl}_3$; the higher-rank Feigin–Tipunin program is conjectural.
  (iii) The KL parameter $p$ forces $\ell = 2p$ even; our conjecture at $A_2$ uses $\ell \in \{3,5,7\}$ odd, **outside** the proven range.

- **(b) Indirect — use the categorical characterization via canonical end (Schweigert–Woike).**
  **Possible in principle but does not give dimension counts.** As established in W1-1b, the Schweigert–Woike framework gives an $E_2$-structure on $\mathrm{HH}^*(\mathcal{C})$ and identifies $\mathrm{HH}^*(\mathrm{rep}(u_q(g))) \cong \mathrm{Ext}^*_{u_q(g)}(\mathbb{C}, u_q(g)_{\mathrm{ad}})$, but does not provide a dimension-counting tool. It would still need to be combined with the Mastnak–Witherspoon LES or another computational technique (bar complex / BGG). The VOA side adds nothing here.

- **(c) Not viable for odd $\ell$ — the log-KL correspondence only covers even roots of unity.**
  **This is the correct answer.** The CLR log-KL correspondence, as proven, is restricted to $q = e^{\pi i/p}$ (even order $2p$). For our conjecture at $A_2$, $\ell \in \{3,5,7\}$ (odd), the log-KL framework provides **no direct entry point** to a VOA-side computation.

**Realistic path to verifying our conjecture at $A_2$ via VOA methods: NOT VIABLE in the current state of the art.** The W2 candidate path (direct bar-complex computation of $\mathrm{HH}^2(u_q(\mathfrak{sl}_3),\mathbb{C})$ on the full 6561-dim algebra at $\ell = 3$, identified in W1-1c) remains the only viable route.

---

## 8. Citable theorem (verbatim)

> **Theorem 1.3** (Creutzig–Lentner–Rupert, arXiv:2306.11492, lines 693–707).
>
> The following are equivalent as braided tensor categories
>
> 1. Let $\mathcal{O}_{M(p)}^T$ the category of weight modules of the singlet algebra of [CMY23a] and $\mathrm{Rep}^{wt} u_q^H(\mathfrak{sl}_2)$ the category of weight modules of the small unrolled quantum group of $\mathfrak{sl}_2$ at $2p$-th root of unity of [CGP15]. Then
>
> $$\mathcal{O}_{M(p)}^T \;\cong\; \mathrm{Rep}^{wt}(u_q^H(\mathfrak{sl}_2))$$
>
> as braided tensor categories (Theorem 8.9 as well as Example 9.10).
>
> 2. The analogous result for triplet VOA and quasi Hopf modification of the small quantum group (Remark 8.10 and Corollary 10.6): $\mathcal{O}_{W(p)}^T \cong \mathrm{Rep}(\tilde{u}_q(\mathfrak{sl}_2))$.
>
> 3. The Hopf algebra $\mathcal{U}(S(p))$ is described in (42). The vertex tensor category of representations of the vertex superalgebra $S(p)$ studied in [CMY23b] is braided equivalent to the category of weight modules of $\mathcal{U}(S(p))$ (Lemma 10.7).
>
> 4. The braided tensor category of weight representations of $V^k(gl_{1|1})$ is equivalent to ${}^{\tilde{N}}_{\tilde{N}}\mathrm{YD}(\mathcal{C})$ (Corollary 10.8).

For our purposes only **Theorem 1.3 (1) / Theorem 8.9** is relevant. When citing, also mention:
- The restriction $p \in \mathbb{Z}_{\geq 2}$, $q = e^{\pi i/p}$ (so $\ell = 2p$ is **even**);
- The target is the **unrolled** $u_q^H(\mathfrak{sl}_2)$ (infinite-dimensional), not the standard $u_q(\mathfrak{sl}_2)$;
- The category on both sides is the **infinite** category of weight modules, not a finite tensor category.

---

## Bottom line

The CLR log-KL correspondence **does not provide a viable VOA-side route** to verifying our conjecture $\dim \mathrm{HH}^2(u_q(\mathfrak{sl}_n),\mathbb{C}) = \binom{n+1}{2} + 2|\Phi^+|$ at $A_2$ with $\ell$ odd. Three independent obstructions:

1. **Wrong algebra on the quantum-group side.** CLR's KL correspondence targets $u_q^H(\mathfrak{sl}_2)$ (the *unrolled* quantum group, infinite-dimensional), **not** the standard small quantum group $u_q(\mathfrak{sl}_2)$ that our conjecture is about. A braided equivalence transfers $\mathrm{HH}^*$ only between the two sides of the equivalence — i.e. between $\mathcal{O}_{M(p)}^T$ and $\mathrm{Rep}^{wt}(u_q^H(\mathfrak{sl}_2))$ — not to $\mathrm{rep}(u_q(\mathfrak{sl}_2))$.

2. **No higher-rank KL theorem.** Only the rank-one ($\mathfrak{sl}_2$) case is proven; the $\mathfrak{sl}_3$ Feigin–Tipunin / unrolled-quantum-group correspondence is conjectural (Example 2.12, [CM17]).

3. **Parity of $\ell$.** CLR's KL correspondence is proven only for $q = e^{\pi i/p}$ (even order $\ell = 2p$). Our conjecture uses $\ell \in \{3,5,7\}$ (odd) — **outside the proven range of log-KL**.

No $\mathrm{HH}^*(\mathcal{O}_{M(p)}^T)$ or $\mathrm{HH}^*(\mathrm{rep}(M(p)))$ computation exists in the literature cited by CLR. Frobenius–Perron dimension questions (relevant to Lentner's conditional proof, 2501.10735) are not addressed.

## Recommended action

- **Cite** CLR Theorem 1.3(1) / Theorem 8.9 in our paper as the state of the art for log-KL at rank one — but with a clear note that (a) the target is $u_q^H(\mathfrak{sl}_2)$ (unrolled, infinite-dim.), not $u_q(\mathfrak{sl}_2)$; (b) the proven range is $\ell = 2p$ even; (c) the higher-rank generalization is conjectural.
- **Do NOT pursue** the VOA-side computation of $\mathrm{HH}^2(\mathrm{rep}(M(p)))$ as a route to our conjecture. The mismatches in (1)–(3) above make it non-viable.
- **Reaffirm W1-1c's recommendation**: the only viable route to verifying our conjecture at $A_2, \ell = 3$ remains a direct bar-complex computation of $\mathrm{HH}^2(u_q(\mathfrak{sl}_3), \mathbb{C})$ on the full $\ell^{3(n^2-1)} = 6561$-dim algebra, using the weight-space decomposition strategy from `scripts/verify_sl2_hh2.py` adapted to $\mathfrak{sl}_3$'s 8-dimensional Cartan-and-root lattice.
- **Cross-link with W1-1b (Schweigert–Woike)**: the Schweigert–Woike $E_2$-structure on $\mathrm{HH}^*$ gives a *structural* constraint on $\mathrm{HH}^2(u_q(\mathfrak{sl}_3))$ (Gerstenhaber bracket vanishes on $\mathrm{Ext}^*$; BV-structure since $u_q(\mathfrak{sl}_3)$ is pivotal unimodular at odd $\ell$), but no dimension count. Combine with the direct computation in W2.
- **Optional follow-up (W3 candidate)**: if Lentner's conditional proof (2501.10735) becomes relevant, investigate whether the FP-dimension assumption can be checked for $u_q(\mathfrak{sl}_3)$ at $\ell = 3$ using the *unrolled* quantum group $u_q^H(\mathfrak{sl}_3)$ (where the analytic character theory is more developed, [CGP15]) as a proxy — but this is a research question in its own right, not a direct application of CLR 2306.11492.

## Open questions for downstream sub-agents

- Is there a separate KL-type correspondence at **odd** $\ell$ to a *different* VOA (perhaps an orbifold or coset of $M(p)$, or a "half-integer" singlet)? CLR does not address this. Worth a literature search on Feigin–Tipunin / singlet-like VOAs at odd-order roots of unity.
- Is there a known relationship between $\mathrm{HH}^*(\mathrm{rep}(u_q(\mathfrak{sl}_2)))$ and $\mathrm{HH}^*(\mathrm{Rep}^{wt}(u_q^H(\mathfrak{sl}_2)))$ — e.g. via a restriction functor along the quotient $u_q^H(\mathfrak{sl}_2) \twoheadrightarrow u_q(\mathfrak{sl}_2)$ (modding out the Cartan by $\mathbb{Z}_\ell$)? If yes, then CLR's theorem would give *indirect* information about $\mathrm{HH}^*(\mathrm{rep}(u_q(\mathfrak{sl}_2)))$ — but this is not worked out in CLR.
- The Lentner 2501.10735 conditional proof assumes FP-dim = asymptotic of analytic characters. Does CLR's framework (or its cited [CGP15] for the unrolled group) provide analytic-character asymptotics that could verify the Lentner assumption for $u_q^H(\mathfrak{sl}_3)$ at $\ell = 3$? (Speculative; not addressed in CLR.)
