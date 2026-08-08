# W3-1b — Mastnak–Witherspoon bialgebra cochain equations (extracted)

- **Task ID**: W3-1b
- **Agent**: Sub-agent 3b (Wave 3, general-purpose)
- **Date**: 2025-08-07
- **Status**: completed
- **Output**: this file
- **Predecessor**: W3-1a (crashed while attempting direct implementation; this sub-task is the focused "extract equations only" redo)
- **Successor**: W3-1c (will implement the bialgebra cochain complex for B⁺(u_q(sl_3)) at ℓ=3 using these equations)

## Scope and intent

The predecessor W3-1a crashed trying to implement the MW bialgebra cochain complex directly. This sub-task is the focused "extract the equations only" redo: read MW §2.1 (Definitions) and §3.4 (Morphisms in the LES) carefully, and write down — in mathematical (LaTeX-style) notation — the equations that W3-1c will need to implement. **No code is written here.**

The downstream goal (per W2-1b): compute `dim H̃¹_b(B⁺(u_q(sl_3)), C)` directly. The conjecture predicts it equals `C(3, 2) = 3` at ℓ=3. Combined with the LES + verified `dim HH¹(B⁺) = 0`, this would verify the conjecture at A₂.

Source: `literature/texts/Mastnak-Witherspoon-LES.txt` (arXiv:0704.2771v2, May 2008). Quotations from MW are verbatim; all deviations and amplifications are flagged.

## Notation

Throughout, `k` is the ground field (we take `k = C`), `B` is a bialgebra (in our case `B = B⁺(u_q(sl_3))` at `ℓ = 3`), `B⁺ = ker ε` is the augmentation ideal (NOT to be confused with the *positive Borel* — both happen to be denoted `B⁺` in the literature; here we follow MW and use `B⁺` for the augmentation ideal of whatever bialgebra we are working with). To minimise confusion in the concrete sections we will write `B = B⁺_Borel(u_q(sl_3))` for the algebra and `B̄ = ker ε ⊂ B` for its augmentation ideal; `B̄` is what MW calls `B⁺`.

Other MW conventions:
- `m`: multiplication, `η`: unit, `Δ`: comultiplication, `ε`: counit.
- Sweedler: `Δ(b) = b₁ ⊗ b₂`, `(1⊗Δ)Δ(b) = b₁ ⊗ b₂ ⊗ b₃`, etc.
- `Vⁿ` = n-fold tensor power; tensor factors indexed by superscripts `v¹ ⊗ ... ⊗ vⁿ ∈ Vⁿ`.
- `λ_l, λ_r`: left/right diagonal *actions* of B on `Bⁿ`. `ρ_l, ρ_r`: left/right diagonal *coactions* of B on `Bⁿ`.
- Indices in sums are 0-based when MW writes them as such (the face maps go from `i=0` to `i=p+1` for horizontal, `j=0` to `j=q+1` for vertical).

For our concrete case (Section 8):
- `q = e^{2πi/3}`, so `q³ = 1`, `q + q⁻¹ = -1`, `1 + q + q² = 0`.
- `B = B⁺_Borel(u_q(sl_3))` is generated (as an algebra) by `K₁, K₂, E₁, E₂`, with the derived root vector `E₁₂ := E₁E₂ − q E₂E₁` for the root `α₁+α₂`. The task statement lists `K₁, K₂, E₁, E₂, E₁₂` as the "generators" — we follow this convention (treating `E₁₂` as a 5th generator subject to its defining relation).
- `dim_C B = 3⁵ = 243`; `dim_C B̄ = ker ε = 242`.

---

## §1. The bicomplex `B^{p,q}` (MW §2.1)

### 1a. Vertices

> "The standard complex for computing bialgebra cohomology is the following cosimplicial bicomplex `B^{p,q}`. The vertices are `B^{p,q} = Hom_k(B^p, B^q)`."

**Abstract**:
```
B^{p,q} := Hom_k(B^⊗p, B^⊗q),    p, q ≥ 0.
```

**Concrete** (`B = B⁺_Borel(u_q(sl_3))`, ℓ=3): with `B̄ = ker ε` and the normalized complex (see §5 below), the relevant chain groups are
```
(B_+)^{p,q} = Hom_C(B̄^⊗p, B̄^⊗q),    p, q ≥ 1.
```
For `p, q ≥ 2`, the dimensions grow as `242^{p+q}` (e.g. `(B_+)^{2,1}` has dimension `242² · 242 = 242³ = 14,172,488`, `(B_+)^{1,2}` similarly; the relevant matrices for H̃¹_b are `242 × 242` and `59049 × 242`, `242 × 59049`).

### 1b. Horizontal (Hochschild) face maps

> "The horizontal faces `∂_i^h: Hom_k(B^p, B^q) → Hom_k(B^{p+1}, B^q)` and degeneracies `σ_i^h: Hom_k(B^{p+1}, B^q) → Hom_k(B^p, B^q)` are those for computing Hochschild cohomology"

MW eqs. (verbatim, lines 116–121):
```
∂₀^h f = λ_l (1 ⊗ f),
∂_i^h f = f (1 ⊗ ... ⊗ m ⊗ ... ⊗ 1),    1 ≤ i ≤ p,
∂_{p+1}^h f = λ_r (f ⊗ 1),
σ_i^h f = f (1 ⊗ ... ⊗ η ⊗ ... ⊗ 1).
```

**Abstract, expanded for `f: B^⊗p → B^⊗q` (with the diagonal actions `λ_l, λ_r` of `B` on `B^⊗q`)**: for `b¹ ⊗ ... ⊗ b^{p+1} ∈ B^{⊗(p+1)}`,
```
(∂₀^h f)(b¹,...,b^{p+1}) = (b¹)_1 · f(b²,...,b^{p+1})_1 ⊗ ... ⊗ (b¹)_q · f(b²,...,b^{p+1})_q
                          = λ_l(b¹ ⊗ f(b²,...,b^{p+1})),
(∂_i^h f)(b¹,...,b^{p+1}) = f(b¹,...,b^i · b^{i+1},...,b^{p+1}),    1 ≤ i ≤ p,
(∂_{p+1}^h f)(b¹,...,b^{p+1}) = λ_r(f(b¹,...,b^p) ⊗ b^{p+1})
                              = f(b¹,...,b^p)_1 · (b^{p+1})_1 ⊗ ... ⊗ f(b¹,...,b^p)_q · (b^{p+1})_q.
```

Here `λ_l(a ⊗ (x₁⊗...⊗x_q)) = a₁ x₁ ⊗ ... ⊗ a_q x_q` (using `Δ_q(a) = a₁⊗...⊗a_q`), and similarly `λ_r((x₁⊗...⊗x_q) ⊗ a) = x₁ a₁ ⊗ ... ⊗ x_q a_q`.

### 1c. Vertical (Cartier / coalgebra) face maps

> "the vertical faces `∂_j^c: Hom_k(B^p, B^q) → Hom_k(B^p, B^{q+1})` and degeneracies `σ_j^c: Hom_k(B^p, B^{q+1}) → Hom_k(B^p, B^q)` are those for computing coalgebra (Cartier) cohomology"

MW eqs. (verbatim, lines 127–132):
```
∂₀^c f = (1 ⊗ f) ρ_l,
∂_j^c f = (1 ⊗ ... ⊗ Δ ⊗ ... ⊗ 1) f,    1 ≤ j ≤ q,
∂_{q+1}^c f = (f ⊗ 1) ρ_r,
σ_j^c f = (1 ⊗ ... ⊗ ε ⊗ ... ⊗ 1) f.
```

**Abstract, expanded for `f: B^⊗p → B^⊗q`** (with diagonal coactions `ρ_l: B^⊗p → B ⊗ B^⊗p`, `ρ_r: B^⊗p → B^⊗p ⊗ B`):
```
(∂₀^c f)(b) = (1 ⊗ f) ρ_l (b)        [apply ρ_l on the input side, then f]
(∂_j^c f)(b) = (id ⊗ ... ⊗ Δ ⊗ ... ⊗ id) f(b)   [apply Δ in slot j of the output, 1 ≤ j ≤ q]
(∂_{q+1}^c f)(b) = (f ⊗ 1) ρ_r (b)   [apply ρ_r on the input side, then f]
```

For `p = 1` (`b ∈ B`): `ρ_l(b) = Δ(b) = b₁ ⊗ b₂` and `ρ_r(b) = b₁ ⊗ b₂` (the coactions reduce to the coproduct for a single tensor factor).

For `p = 2` (`b = a ⊗ b' ∈ B²`): `ρ_l(a ⊗ b') = (a · b')₁ ⊗ (a · b')_{2,1} ⊗ (a · b')_{2,2}` where `Δ²(a b') = (a b')₁ ⊗ (a b')_{2,1} ⊗ (a b')_{2,2}`, using the algebra-map property of `Δ`. The "outer" factor (first slot of `B ⊗ B²`) is the first Sweedler component of the product; the "inner" factors (slots 2,3 of `B ⊗ B²`) are the iterated coproduct of the second Sweedler component.

**Sign convention (MW eq. 2.1.1 vs MW eq. 2.1.2):** in the cocycle equation for `f: B² → B` (MW eq. 2.1.1, condition 3), the term `f(a₁, b₁) ⊗ a₂ b₂` comes from `∂₀^c f` and the term `a₁ b₁ ⊗ f(a₂, b₂)` comes from `∂_{q+1}^c f`. The middle term `Δ(f(a,b))` comes from `∂_1^c f` (with `q = 1`, so `j = 1` is the only middle face). The full expansion for `p = 2, q = 1`:
```
(∂^c f)(a, b) = f(a₁, b₁) ⊗ a₂ b₂  −  Δ(f(a, b))  +  a₁ b₁ ⊗ f(a₂, b₂).
```
(The signs `+, −, +` come from the alternating sum `Σ_j (−1)^j ∂_j^c`.)

---

## §2. Horizontal and vertical differentials (MW §2.1)

> "The vertical and horizontal differentials are given by the usual alternating sums `∂^h = Σ (−1)^i ∂_i^h`, `∂^c = Σ (−1)^j ∂_j^c`."

**Abstract**:
```
∂^h : B^{p,q} → B^{p+1,q},    ∂^h = Σ_{i=0}^{p+1} (−1)^i ∂_i^h,
∂^c : B^{p,q} → B^{p,q+1},    ∂^c = Σ_{j=0}^{q+1} (−1)^j ∂_j^c.
```

**Properties** (standard, MW relies on these):
- `∂^h ∂^h = 0`, `∂^c ∂^c = 0` (each is a differential).
- `∂^h ∂^c = ∂^c ∂^h` (the bicomplex commutes).

### Concrete expansion for the cases W3-1c needs

**Case `h: B̄ → B̄` (p=1, q=1)** — relevant for H̃¹_b:
```
(∂^h h)(a, b) = a · h(b) − h(a · b) + h(a) · b,           a, b ∈ B̄,
(∂^c h)(c)    = c₁ ⊗ h(c₂) − Δ(h(c)) + h(c₁) ⊗ c₂,        c ∈ B̄.
```
These are the standard Hochschild differential (with values in `B̄` under left/right regular action) and the Cartier differential (with values in `B̄^⊗2` under left/right regular coaction). The first says `h` is a *derivation*; the second says `h` is a *coderivation*.

**Case `f: B̄² → B̄` (p=2, q=1)** — relevant for H̃²_b (Hochschild part):
```
(∂^h f)(a, b, c) = a · f(b, c) − f(a·b, c) + f(a, b·c) − f(a, b) · c,    a, b, c ∈ B̄,
(∂^c f)(a, b)    = f(a₁, b₁) ⊗ a₂ b₂ − Δ(f(a, b)) + a₁ b₁ ⊗ f(a₂, b₂).
```

**Case `g: B̄ → B̄²` (p=1, q=2)** — relevant for H̃²_b (coalgebra part):
```
(∂^h g)(a, b)   = (Δ a) · g(b) − g(a·b) + g(a) · (Δ b),                     a, b ∈ B̄,
(∂^c g)(c)      = c₁ ⊗ g(c₂) − (Δ ⊗ 1) g(c) + (1 ⊗ Δ) g(c) − g(c₁) ⊗ c₂,   c ∈ B̄.
```
In `∂^h g`, the diagonal action of `B` on `B²` is `a · (x ⊗ y) = a₁ x ⊗ a₂ y`, so `(Δ a) · g(b)` is shorthand for `a₁ · g(b)₁ ⊗ a₂ · g(b)₂`.

---

## §3. Total differential `∂_b` on `Tot(B)` (MW §2.1)

> "The bialgebra cohomology of `B` is then defined as `H^*_b(B) = H_*(Tot B)`, where `Tot B = B^{0,0} → B^{1,0} ⊕ B^{0,1} → ... → ⊕_{p+q=n} B^{p,q} → ...` and `∂` is given by the sign trick (i.e., `∂|_{B^{p,q}} = ∂^h ⊕ (−1)^p ∂^c: B^{p,q} → B^{p+1,q} ⊕ B^{p,q+1}`)."

**Abstract**:
```
(Tot B)^n = ⊕_{p+q=n} B^{p,q},
∂_b |_{B^{p,q}}  =  ∂^h  +  (−1)^p ∂^c   :    B^{p,q}  →  B^{p+1,q} ⊕ B^{p,q+1} ⊆ (Tot B)^{n+1}.
```

The `(−1)^p` sign on `∂^c` is the **sign trick**: vertical differential picks up `(−1)^p` where `p` is the horizontal degree.

**Properties**:
- `∂_b ∂_b = 0` (follows from `∂^h ∂^h = ∂^c ∂^c = 0` and `∂^h ∂^c = ∂^c ∂^h`).
- Bialgebra cohomology: `H^n_b(B) = H^n(Tot B) = ker(∂_b: (Tot B)^n → (Tot B)^{n+1}) / im(∂_b: (Tot B)^{n−1} → (Tot B)^n)`.

### Concrete: `∂_b` on `B̄ → B̄` (h ∈ (B_+)^{1,1})

For `h ∈ B^{1,1} = Hom(B̄, B̄)`, `p = 1`, so the sign on `∂^c` is `(−1)^1 = −1`:
```
∂_b h  =  (∂^h h,  −∂^c h)  ∈  B^{2,1} ⊕ B^{1,2}  =  Hom(B̄², B̄) ⊕ Hom(B̄, B̄²).
```
Concretely:
```
(∂_b h)(a, b)  =  ( a·h(b) − h(ab) + h(a)·b,  −[c₁ ⊗ h(c₂) − Δ(h(c)) + h(c₁) ⊗ c₂] )
                 =  ( a·h(b) − h(ab) + h(a)·b,  −c₁ ⊗ h(c₂) + Δ(h(c)) − h(c₁) ⊗ c₂ ).
```
(Here the first component is the "Hochschild part" evaluated at `(a, b)` and the second is the "coalgebra part" evaluated at `c`.)

### Concrete: `∂_b` on `(f, g) ∈ B^{2,1} ⊕ B^{1,2}`

For `f ∈ B^{2,1}` (p=2), `∂_b f = (∂^h f, (−1)^2 ∂^c f) = (∂^h f, ∂^c f)`.
For `g ∈ B^{1,2}` (p=1), `∂_b g = (∂^h g, (−1)^1 ∂^c g) = (∂^h g, −∂^c g)`.

Adding (both `f` and `g` live in `(Tot B)^3`):
```
∂_b (f, g)  =  ( ∂^h f,  ∂^c f + ∂^h g,  −∂^c g )  ∈  B^{3,1} ⊕ B^{2,2} ⊕ B^{1,3} = (Tot B)^4.
```

So `(f, g)` is a **degree-2 cocycle** (i.e., in `Z̃²_b(B)`) iff
```
∂^h f = 0   (in B^{3,1}),
∂^c g = 0   (in B^{1,3}),
∂^c f + ∂^h g = 0   (in B^{2,2}).
```
These are precisely the three MW eq. (2.1.1) conditions — see §7 below.

---

## §4. The truncated complex `B⁰` (MW §2.1)

> "Let `B⁰` denote the bicomplex obtained from `B` by replacing the edges by zeroes, that is `(B⁰)^{p,0} = 0 = (B⁰)^{0,q}` for all `p, q`."

**Abstract**:
```
(B⁰)^{p,q} := { 0,                  if p = 0 or q = 0,
              { B^{p,q},            if p ≥ 1 and q ≥ 1.
```
The horizontal edge (`q = 0` row) and the vertical edge (`p = 0` column) are zeroed out.

**Truncated bialgebra cohomology**:
```
H̃^i_b(B)  :=  H^{i+1}(Tot B⁰).     (Note the degree shift by +1.)
```

So `H̃¹_b(B) = H²(Tot B⁰)`, `H̃²_b(B) = H³(Tot B⁰)`, etc.

### Concrete: chain groups of `Tot B⁰` for small total degree

| total degree `n` | `(Tot B⁰)^n = ⊕_{p+q=n} (B⁰)^{p,q}` |
|---|---|
| 0 | `(B⁰)^{0,0} = 0` |
| 1 | `(B⁰)^{1,0} ⊕ (B⁰)^{0,1} = 0 ⊕ 0 = 0` |
| 2 | `(B⁰)^{2,0} ⊕ (B⁰)^{1,1} ⊕ (B⁰)^{0,2} = 0 ⊕ B^{1,1} ⊕ 0 = Hom(B, B)` |
| 3 | `(B⁰)^{3,0} ⊕ (B⁰)^{2,1} ⊕ (B⁰)^{1,2} ⊕ (B⁰)^{0,3} = 0 ⊕ B^{2,1} ⊕ B^{1,2} ⊕ 0 = Hom(B², B) ⊕ Hom(B, B²)` |
| 4 | `(B⁰)^{4,0} ⊕ (B⁰)^{3,1} ⊕ (B⁰)^{2,2} ⊕ (B⁰)^{1,3} ⊕ (B⁰)^{0,4} = 0 ⊕ B^{3,1} ⊕ B^{2,2} ⊕ B^{1,3} ⊕ 0` |

In the **normalized** complex `B⁰_+` (see §5 below), replace each `B^{p,q}` with `Hom(B̄^⊗p, B̄^⊗q)` for `p, q ≥ 1` (and zero on the edges as above).

For `B = B⁺_Borel(u_q(sl_3))` at ℓ=3 (`dim B̄ = 242`):
- `(Tot B⁰_+)^2 = Hom(B̄, B̄)`, dimension `242² = 58564`.
- `(Tot B⁰_+)^3 = Hom(B̄², B̄) ⊕ Hom(B̄, B̄²)`, dimension `2 × 242³ = 2 × 14172488 = 28344976`.
- `(Tot B⁰_+)^4 = Hom(B̄³, B̄) ⊕ Hom(B̄², B̄²) ⊕ Hom(B̄, B̄³)`, dimension `242⁴ + 242⁴ + 242⁴ = 3 × 3429659024 ≈ 1.03 × 10¹⁰` — **too big to build explicitly**.

**Implication for W3-1c**: when computing `H̃¹_b(B)`, we only need `(Tot B⁰_+)^2 → (Tot B⁰_+)^3`. We do NOT need `(Tot B⁰_+)^4` — that would be needed only for `H̃²_b(B)`, which we are NOT computing directly (per W2-1b: the conjecture at A₂ reduces to `H̃¹_b`, not `H̃²_b`).

---

## §5. The normalized complex `B_+` (MW §2.1)

> "For computations usually the normalized subcomplex `B_+` is used. The normalized complex `B_+` is obtained from the cochain complex `B` by replacing `B^{p,q} = Hom_k(B^p, B^q)` with the intersection of degeneracies:
> `(B_+)^{p,q} = (∩_i ker σ_i^h) ∩ (∩_j ker σ_j^c) ≃ Hom_k((B⁺)^p, (B⁺)^q)`."

**Abstract**:
```
(B_+)^{p,q}  =  (∩_{i=0}^{p} ker σ_i^h)  ∩  (∩_{j=0}^{q} ker σ_j^c)  ≃  Hom_k((B̄)^⊗p, (B̄)^⊗q),
```
where the isomorphism `≃` is the canonical one: a normalized cochain is uniquely determined by its values on `(B̄)^⊗p` (i.e., its values vanish whenever any tensor factor is a scalar multiple of `1`).

**Quasi-isomorphism**: `B_+ ↪ B` is a quasi-isomorphism (standard; the normalized subcomplex computes the same cohomology as the unnormalized one). Similarly `(B⁰)_+ ↪ B⁰` is a quasi-isomorphism. Hence
```
H̃^i_b(B)  =  H^{i+1}(Tot B⁰)  =  H^{i+1}(Tot (B⁰)_+).
```

**W3-1c will work with `(B⁰)_+` throughout** (the "truncated normalized complex"), with chain groups `Hom(B̄^⊗p, B̄^⊗q)` for `p, q ≥ 1` and zero for `p = 0` or `q = 0`.

---

## §6. H̃¹_b(B) — the i=1 case (MW §2.1, displayed identification above eq. 2.1.1) — OUR MAIN TARGET

MW displays the following identification (lines 175–184 of the text, BEFORE the numbered eq. 2.1.1; the task calls this "MW eq. 2.1.1 for i=1"):

> "Note that we can identify
> `H̃¹_b(B) = { f: B⁺ → B⁺ | f(ab) = a f(b) + f(a) b,  Δ f(a) = a₁ ⊗ f(a₂) + f(a₁) ⊗ a₂ }`"

This is the **i=1** analog of MW eq. (2.1.1) (which is stated for i=2). It says: a 1-cocycle in the truncated normalized bialgebra complex is a single linear map `h: B̄ → B̄` that is simultaneously a *derivation* (Hochschild 1-cocycle) and a *coderivation* (Cartier 1-cocycle). There are no nontrivial 1-coboundaries because `(Tot B⁰_+)^1 = 0`.

### 6a. Cocycle condition `Z̃¹_b(B)` (MW eq. 2.1.1, i=1 case)

**Abstract**:
```
Z̃¹_b(B)  =  { h: B̄ → B̄  |  ∂^h h = 0  AND  ∂^c h = 0 }
         =  { h: B̄ → B̄  |  ∀ a, b ∈ B̄:   a·h(b) − h(a·b) + h(a)·b = 0,
                                  ∀ c ∈ B̄:    c₁ ⊗ h(c₂) − Δ(h(c)) + h(c₁) ⊗ c₂ = 0 }.
```
Equivalently (MW's display, with `f = h`):
```
h(a·b)  =  a·h(b) + h(a)·b                      (derivation condition),
Δ(h(c))  =  c₁ ⊗ h(c₂) + h(c₁) ⊗ c₂           (coderivation condition).
```

### 6b. Coboundary condition `B̃¹_b(B)` (MW eq. 2.1.2, i=1 case)

In the truncated normalized complex, the source of `∂_b: (Tot B⁰_+)^1 → (Tot B⁰_+)^2` is zero (since `(Tot B⁰_+)^1 = 0`). Therefore:
```
B̃¹_b(B)  =  im(∂_b: (Tot B⁰_+)^1 → (Tot B⁰_+)^2)  =  0.
```

So **every 1-cocycle is automatically nontrivial** (no two distinct 1-cocycles are cohomologous):
```
H̃¹_b(B)  =  Z̃¹_b(B) / B̃¹_b(B)  =  Z̃¹_b(B) / 0  =  Z̃¹_b(B).
```

This is the **key simplification** that makes `H̃¹_b(B)` directly computable: it is exactly the kernel of the linear map
```
∂_b : Hom(B̄, B̄) → Hom(B̄², B̄) ⊕ Hom(B̄, B̄²),
h ↦ (∂^h h, −∂^c h),
```
and the kernel can be computed by a single linear-algebra rank/nullity computation.

### 6c. Concrete for `B = B⁺_Borel(u_q(sl_3))` at ℓ=3

- Chain group: `Hom(B̄, B̄)` is a vector space of dimension `242² = 58564` (each `h` is encoded as a `58564`-dim vector, equivalently a `242 × 242` matrix).
- Target: `Hom(B̄², B̄) ⊕ Hom(B̄, B̄²)`, dimension `242³ + 242³ = 2 · 14172488 = 28344976`.
- So `∂_b` is a sparse complex matrix of shape `28344976 × 58564`.
- `dim H̃¹_b(B) = 58564 − rank(∂_b)`.
- **Conjecture (A₂)**: `dim H̃¹_b(B) = C(3, 2) = 3`, i.e. `rank(∂_b) = 58564 − 3 = 58561`.

The W3-1c computation is: build `∂_b` as a sparse `28344976 × 58564` matrix, compute its rank (via SVD on a weight-decomposed block structure analogous to W2-1b's HH¹ computation, where each block is at most `~39204 × 6561`), and read off `dim H̃¹_b = 58564 − rank`.

---

## §7. H̃²_b(B) — the i=2 case (MW eqs. 2.1.1 and 2.1.2) — NEEDED FOR TASK §4

This section addresses **task part 4**: write down the conditions on a pair `(f, g)` where
- `f: (B̄)^⊗2 → B̄` (the "Hochschild part", element of `(B_+)^{2,1}`)
- `g: B̄ → (B̄)^⊗2` (the "coalgebra part", element of `(B_+)^{1,2}`)

for `(f, g)` to be a degree-2 cocycle and for it to be a coboundary. **Note on terminology**: the task calls this a "degree-1 bialgebra cocycle" — this is the *bialgebra cohomology degree* (the cocycle represents a class in `H̃²_b(B)`, but it can be viewed as the image `∂_b h` of a degree-1 element `h ∈ (B_+)^{1,1}`). In MW's bicomplex convention, `(f, g)` lives in `(Tot B⁰_+)^3` and represents a class in `H̃²_b(B) = H³(Tot B⁰_+)`. We follow the task's terminology in this section to avoid confusion with §6.

### 7a. Cocycle condition (MW eq. 2.1.1, i=2 case) — three equations

**Verbatim from MW** (lines 194–201):
> `(f, g) | f: B⁺ ⊗ B⁺ → B⁺, g: B⁺ → B⁺ ⊗ B⁺,
> a f(b, c) + f(a, bc) = f(ab, c) + f(a, b)c,
> c₁ ⊗ g(c₂) + (1 ⊗ Δ) g(c) = (Δ ⊗ 1) g(c) + g(c₁) ⊗ c₂,
> f(a₁, b₁) ⊗ a₂ b₂ − Δ f(a, b) + a₁ b₁ ⊗ f(a₂, b₂) = −(Δ a) g(b) + g(ab) − g(a) (Δ b)`

So `Z̃²_b(B)` is the set of `(f, g)` satisfying three equations, **abstractly**:

**Equation 1 — Hochschild 2-cocycle condition on `f` (= `∂^h f = 0` in `B^{3,1}`)**:
```
∀ a, b, c ∈ B̄:
    a · f(b, c)  +  f(a, b·c)  =  f(a·b, c)  +  f(a, b) · c.
```

**Equation 2 — Cartier 2-cocycle condition on `g` (= `∂^c g = 0` in `B^{1,3}`)**:
```
∀ c ∈ B̄:
    c₁ ⊗ g(c₂)  +  (1 ⊗ Δ) g(c)  =  (Δ ⊗ 1) g(c)  +  g(c₁) ⊗ c₂.
```

**Equation 3 — Mixed compatibility (= `∂^c f + ∂^h g = 0` in `B^{2,2}`)**:
```
∀ a, b ∈ B̄:
    f(a₁, b₁) ⊗ a₂ b₂  −  Δ(f(a, b))  +  a₁ b₁ ⊗ f(a₂, b₂)
        =  −(Δ a) · g(b)  +  g(a·b)  −  g(a) · (Δ b).
```

**Verification of the derivation from `∂_b (f, g) = 0`**: as computed in §3, `∂_b (f, g) = (∂^h f, ∂^c f + ∂^h g, −∂^c g) ∈ B^{3,1} ⊕ B^{2,2} ⊕ B^{1,3}`. Setting each component to zero gives:
- `∂^h f = 0` ↔ Equation 1.
- `−∂^c g = 0`, i.e. `∂^c g = 0` ↔ Equation 2.
- `∂^c f + ∂^h g = 0` ↔ Equation 3 (the LHS of Equation 3 is `(∂^c f)(a, b)` and the RHS is `−(∂^h g)(a, b)`).

### 7b. Coboundary condition (MW eq. 2.1.2, i=2 case)

**Verbatim from MW** (lines 206–211):
> `(f, g) | ∃ h: B⁺ → B⁺,
> f(a, b) = a h(b) − h(ab) + h(a) b
> g(c) = −c₁ ⊗ h(c₂) + Δ h(c) − h(c₁) ⊗ c₂`

So `B̃²_b(B)` is the set of `(f, g)` for which there exists a linear map `h: B̄ → B̄` with:

**Coboundary Equations**:
```
f(a, b)  =  a · h(b)  −  h(a·b)  +  h(a) · b        [i.e., f = ∂^h h],

g(c)     =  −c₁ ⊗ h(c₂)  +  Δ(h(c))  −  h(c₁) ⊗ c₂  [i.e., g = −∂^c h, with the (−1)^p sign trick for p=1].
```

**Verification**: `∂_b h = (∂^h h, (−1)^1 ∂^c h) = (∂^h h, −∂^c h)`. So `(f, g) = ∂_b h` iff `f = ∂^h h` and `g = −∂^c h`, which is exactly what MW writes.

### 7c. The cocycle quotient

```
H̃²_b(B)  =  Z̃²_b(B) / B̃²_b(B)
         =  { (f, g) satisfying Equations 1, 2, 3 } / { (f, g) = (∂^h h, −∂^c h) for some h: B̄ → B̄ }.
```

**Concrete for `B = B⁺_Borel(u_q(sl_3))` at ℓ=3** (for reference; W3-1c will NOT compute this directly, per W2-1b):
- `f: B̄² → B̄`: a `59049 × 242` matrix (14172488 complex entries).
- `g: B̄ → B̄²`: a `242 × 59049` matrix (14172488 complex entries).
- Together: ~28M complex entries per `(f, g)` pair, total `(B_+)^{2,1} ⊕ (B_+)^{1,2}` ≅ `C^{28344976}`.
- Equations 1, 2, 3 are linear in the entries of `(f, g)` and translate to a (large) linear system. The matrix of Equation 1 has size `(dim B̄³ × dim B̄) × (dim B̄² × dim B̄) = (242³) × (242³) ≈ 2 × 10¹⁴` — **utterly intractable** by direct linear algebra. Hence W3-1c does NOT compute `H̃²_b(B)`; the conjecture does not require it.

---

## §8. Concrete translation for `B = B⁺_Borel(u_q(sl_3))` at ℓ=3

### 8a. Algebra structure (from `scripts/verify_sl3_bplus_hh2.py` header comment)

Generators: `K₁, K₂, E₁, E₂, E₁₂` (we treat `E₁₂` as a 5th generator subject to its defining relation).

**Relations** (with `q = e^{2πi/3}`, so `q³ = 1` and `q + q⁻¹ = −1`):
```
K₁³ = K₂³ = 1,                          (Cartan, ℓ-torsion)
E₁³ = E₂³ = E₁₂³ = 0,                   (root vectors, ℓ-nilpotent)
K₁ E₁ K₁⁻¹ = q² E₁,                     K₂ E₁ K₂⁻¹ = q⁻¹ E₁,
K₁ E₂ K₁⁻¹ = q⁻¹ E₂,                    K₂ E₂ K₂⁻¹ = q² E₂,
K₁ E₁₂ K₁⁻¹ = q E₁₂,                    K₂ E₁₂ K₂⁻¹ = q E₁₂,
E₁₂ = E₁ E₂ − q E₂ E₁,                  (definition of E₁₂)
Quantum Serre:
    E₁² E₂ − (q + q⁻¹) E₁ E₂ E₁ + E₂ E₁² = 0    ⟹    E₁² E₂ + E₁ E₂ E₁ + E₂ E₁² = 0,
    E₂² E₁ − (q + q⁻¹) E₂ E₁ E₂ + E₁ E₂² = 0    ⟹    E₂² E₁ + E₂ E₁ E₂ + E₁ E₂² = 0.
```

**Derived PBW commutation relations** (verified in `verify_sl3_bplus_hh2.py` lines 19–67):
```
E₂ E₁   = q⁻¹ E₁ E₂  −  q⁻¹ E₁₂,
E₁₂ E₁  = q E₁ E₁₂,
E₂ E₁₂  = q E₁₂ E₂.
```

**PBW basis** (standard Lusztig order, used by `verify_sl3_bplus_hh2.py`):
```
{ K₁^a K₂^b E₁^c E₁₂^e E₂^d : 0 ≤ a, b, c, e, d ≤ 2 }   (243 elements)
```
excluding `(a, b, c, e, d) = (0, 0, 0, 0, 0)` (the unit) when working in `B̄ = ker ε` (242 elements).

The PBW index function is `idx(a, b, c, e, d) = a·3⁴ + b·3³ + c·3² + e·3 + d`, and the weight of a basis element (used for block-decomposition) is `(c+e mod 3, e+d mod 3)`.

### 8b. Coalgebra structure (comultiplication) — needs to be built by W3-1c

The Lusztig coproduct on `u_q(sl_3)` (restricted to the positive Borel):
```
Δ(K_i)     = K_i ⊗ K_i,                                              i = 1, 2,
Δ(E_i)     = E_i ⊗ 1  +  K_i ⊗ E_i,                                  i = 1, 2,
Δ(E₁₂)     = E₁₂ ⊗ 1  +  K₁ K₂ ⊗ E₁₂  +  (q⁻¹ − q) · K₂ E₁ ⊗ E₂,
```
where the last formula is the Lusztig root-vector coproduct for the root `α₁ + α₂` (with our normalization `E₁₂ = E₁ E₂ − q E₂ E₁`). The cross-term coefficient `(q⁻¹ − q) = (q² − q) = −i√3` for `q = e^{2πi/3}` — W3-1c should verify this coefficient explicitly by expanding `Δ(E₁ E₂ − q E₂ E₁) = Δ(E₁) Δ(E₂) − q Δ(E₂) Δ(E₁)` and using the K-E commutation relations; see derivation sketch in W2-1a header.

The full comultiplication table on the 243-element PBW basis is determined by:
1. The above formulas on generators.
2. `Δ` being an algebra homomorphism: `Δ(xy) = Δ(x) Δ(y)`.
3. Reduction of the resulting tensors to PBW normal form using the commutation relations.

**Action item for W3-1c**: build the sparse `243 × (243 × 243)` matrix `Delta_matrix` whose `(i, (j, k))`-entry is the coefficient of `basis[i]` in `Δ(basis[j]) · basis[k]` — or, equivalently, the coefficient of `basis[j] ⊗ basis[k]` in `Δ(basis[i])`. The construction is fully analogous to the multiplication table builder in `verify_sl3_bplus_hh2.py:build_mult_sparse` (lines 446–459), but with `Δ` replacing `m`. Note that `Δ` is coassociative and counital; both should be sanity-checked numerically.

### 8c. Concrete equation: `∂^h h = 0` for `h: B̄ → B̄`

For each PBW basis pair `(a_idx, b_idx) ∈ {0, ..., 241}²`, the equation `a · h(b) − h(a·b) + h(a) · b = 0` is a vector equation in `B̄` (242 components). The map `(a, b) ↦ a·b` is given by the multiplication table `mult[a_idx, b_idx]` (a 242-vector with structure constants). The map `h` is a `242 × 242` matrix `H`. So:
```
(∂^h H)[a_idx, b_idx, target_idx]  =  Σ_j (mult[a_idx, j] · H[j, b_idx] · δ[j ≠ 0? no, B̄ basis])
                                     −  Σ_j H[j, mult[a_idx, b_idx]_nonzero_components]
                                     +  Σ_j (H[j, a_idx] · mult[j, b_idx]).
```
(Indices adjusted for the 242-element `B̄` basis vs. the 243-element `B` basis; the unit `1` is excluded from `B̄` but `a · b` may have a `1`-component which is then dropped when projecting to `B̄`.)

The matrix of `∂^h` has shape `(242³ = 14172488) × (242² = 58564)`. After weight-block decomposition (each `B̄^⊗2 → B̄` decomposes as `⊕_{weight} B̄_{w1} ⊗ B̄_{w2} → B̄_{w1+w2}`), this becomes 9 blocks of size at most `(242·81) × (81·81) = 19602 × 6561` (per the W2-1b weight analysis). Each block is small enough for sparse SVD in seconds.

### 8d. Concrete equation: `∂^c h = 0` for `h: B̄ → B̄`

For each PBW basis element `c_idx ∈ {0, ..., 241}`, the equation `c₁ ⊗ h(c₂) − Δ(h(c)) + h(c₁) ⊗ c₂ = 0` is a vector equation in `B̄^⊗2` (59049 components, of which 242² = 58564 are in `B̄^⊗2 ⊂ B^⊗2`). The structure constants are:
- `Δ(c)` = `Delta_matrix[c_idx, :, :]` — a `243 × 243` sparse array of coefficients (with the unit row/column contributing the "1 ⊗ c" and "c ⊗ 1" parts that vanish on `B̄`).
- `Δ(h(c))` = `Delta_matrix[H[c_idx, :], :, :]` — composition: first apply `h` to `c`, then apply `Δ` to the result.
- `c₁ ⊗ h(c₂) = (Δ(c)) acted by (id ⊗ h)`: take `Delta_matrix[c_idx, j, k]` and contract with `H` on the second index to get `Σ_{j,k} Delta_matrix[c_idx, j, k] · basis[j] ⊗ H[k, :]`. Symmetrically for `h(c₁) ⊗ c₂`.

The matrix of `∂^c` has shape `(242³ = 14172488) × (242² = 58564)`, same as `∂^h`. Weight-decomposition applies similarly (the coalgebra is weight-graded).

### 8e. Concrete equation: stacked `∂_b`

Stack `∂^h` and `−∂^c` (with the sign `(−1)^p = −1` for `p = 1`) vertically:
```
∂_b : Hom(B̄, B̄)  →  Hom(B̄², B̄) ⊕ Hom(B̄, B̄²),
∂_b(h)  =  (∂^h h,  −∂^c h).
```
Matrix shape: `(2 · 14172488) × 58564 = 28344976 × 58564`. After weight decomposition, 9 blocks each of shape `~2 × 19602 × 6561` — still tractable.

**Computation**:
```
dim H̃¹_b(B⁺_Borel(u_q(sl_3)))  =  dim ker(∂_b)  =  58564 − rank(∂_b).
```
Equivalently, restricting to `B̄ = ker ε` (242-dim, not the full `B`):
```
dim H̃¹_b  =  242² − rank(∂_b restricted to B̄)  =  58564 − rank(∂_b).
```
**Conjecture**: this equals `C(3, 2) = 3`, i.e. `rank(∂_b) = 58561`.

---

## §9. The connecting homomorphism — MW §3.4

### 9a. Notational warning: MW's `δ` vs. modern `δ`

There is a **notational clash** between MW and the modern literature (and the W2-1b analysis) on what the "connecting homomorphism" is. We disentangle this carefully.

**MW's LES (eq. 3.3.1)**, with MW's own labels:
```
... → H^i_h(D(B), k)  --ῑ-->  H^i_h(X, k) ⊕ H^i_h(B, k)  --δ_MW-->  H̃^i_b(B)  --?-->  H^{i+1}_h(D(B), k) → ...
```
where `X = (B^op)^* = (B^*)^cop` (so for our `B = B⁺_Borel(u_q(sl_3))`, `X ≅ B⁻_Borel(u_q(sl_3))` by the Chevalley involution).

**MW's labels**:
- `ῑ: H^i_h(D(B), k) → H^i_h(X, k) ⊕ H^i_h(B, k)` — given by MW eq. (3.4.4).
- `δ_MW: H^i_h(X, k) ⊕ H^i_h(B, k) → H̃^i_b(B)` — MW calls this the "connecting homomorphism"; given by MW eqs. (3.4.1) and (3.4.2).
- The third arrow `H̃^i_b(B) → H^{i+1}_h(D(B), k)` is NOT separately labeled by MW; in the standard LES convention it would be the connecting homomorphism `δ_modern`.

**Modern / W2-1b convention** (the one the task uses):
- `ῑ: H^i_h(D(B), k) → H^i_h(X, k) ⊕ H^i_h(B, k)` — same as MW's `ῑ`.
- `π̄: H^i_h(X, k) ⊕ H^i_h(B, k) → H̃^i_b(B)` — this is **MW's `δ_MW`**. The map is the *projection* from the edge complex onto the truncated complex; in MW's terminology this is the "connecting homomorphism" because in the original Gerstenhaber–Schack LES it appears as such.
- `δ_modern: H̃^i_b(B) → H^{i+1}_h(D(B), k)` — this is the actual connecting homomorphism in the modern LES convention, the one W2-1b labels `δ` and the one the task part 5 calls "the connecting homomorphism `δ: H̃¹_b(B⁺) → HH²(D(B⁺))`".

So **MW eqs. (3.4.1) and (3.4.2) describe `π̄ = δ_MW`, NOT `δ_modern`**. The map `δ_modern` is described by MW eq. (3.4.4) (which MW labels as `ῑ`).

We will write down both, since the task explicitly asks for (3.4.1) and (3.4.2) AND for the map `δ_modern: H̃¹_b → HH²(D(B))`.

### 9b. MW eq. (3.4.1): the `f ↦ F` component of `δ_MW = π̄`

**Verbatim from MW** (lines 974–985):
> "if `f: B^i → k` and `g: X^i → k` are cocycles, then `δ(f, g) = (∂^X f, (−1)^i ∂^B g)`. More precisely
> `F := ∂^X f ∈ Hom_k(B^i, B) ⊆ ⊕_{m+n=i+1} Hom_k(B^m, B^n)`,
> is given by
> `(3.4.1)    F(b) = f(b₁) b₂^c − f(b₂) b₁^c`."

The superscript `^c` denotes (in MW's notation) the **coaction**: `b^c_1` is the first tensor factor of `b` under the right coaction `ρ_r: B → B ⊗ B`. Since `ρ_r(b) = Δ(b) = b₁ ⊗ b₂` for `b ∈ B` (single tensor factor), the formula simplifies to:

**Abstract (cleaned up)**:
```
F(b) = f(b₁) · b₂  −  f(b₂) · b₁,    b ∈ B,
```
where `Δ(b) = b₁ ⊗ b₂` (Sweedler) and `f: B → k` is a Hochschild 1-cocycle (i.e. `f ∈ Z¹_h(B, k)`).

Wait — re-reading MW more carefully, the formula `F(b) = f(b₁) b₂^c − f(b₂) b₁^c` actually uses `b₁^c` and `b₂^c` to denote the *X-action* on B (action #3 or #4 in MW §3.1's four-action notation). The X-action of `x ∈ X` on `b ∈ B` is `x · b = ⟨x, b₂⟩ b₁` (left) or `b · x = ⟨x, b₁⟩ b₂` (right). With `f ∈ X^* = B` (under the canonical identification `X^* ≅ B`), the formula becomes:
```
F(b) = f(b₁) · b₂  −  f(b₂) · b₁,
```
where `f(-)` is the pairing `⟨f, -⟩: X ⊗ B → k` (treating `f ∈ X^* = B`, so `f(b_i) ∈ k`), and the multiplications `f(b_1) · b_2` and `f(b_2) · b_1` are scalar-multiplications in `B`.

**Concrete for our case** (`B = B⁺_Borel(u_q(sl_3))`, `i = 1`): the source is `Z¹_h(B, k) ⊕ Z¹_h(X, k)`. By the W2-1b computation, `dim HH¹(B) = dim HH¹(X) = 0`, so the source is **zero**. Therefore `F = 0` identically; MW eq. (3.4.1) contributes nothing nontrivially.

**For reference (if `dim HH¹(B) > 0`)**: a Hochschild 1-cocycle `f: B → k` is a *trace-like functional* satisfying `f(ab) = ε(a) f(b) + f(a) ε(b)` (Hochschild cocycle condition with trivial coefficients). The formula `F(b) = f(b₁) b₂ − f(b₂) b₁` then produces an element `F ∈ Hom(B, B) = B^{1,1}` (a candidate 1-cocycle in the bialgebra complex). One can verify directly that `∂^h F = 0` and `∂^c F = 0` (so `F ∈ Z̃¹_b(B)`), confirming that MW eq. (3.4.1) indeed maps Hochschild cocycles to bialgebra cocycles.

### 9c. MW eq. (3.4.2): the `g ↦ G` component of `δ_MW = π̄`

**Verbatim from MW** (lines 987–999):
> "If we identify `g` with an element of `B^i` (`g ∈ (X^i)^* ≃ (B^i)^** ≃ B^i`), then
> `G := (−1)^i ∂^B g ∈ Hom_k(B, B^i) ⊆ ⊕_{m+n=i+1} Hom_k(B^m, B^n)`
> is given by
> `(3.4.2)    G(b) = (−1)^i ( (Δ^i b) g − g(Δ^i b) )`.
> Recall that `Δ^i b = b₁ ⊗ ... ⊗ b_i`."

**Abstract (cleaned up)**: for `g ∈ Z¹_h(X, k) ≅ B` (via the canonical identification `(X^*)^* ≅ B` — using `X = (B^op)^*` so `X^* = B^op`, and then identifying `g ∈ X^* = B^op` with an element of `B` by reversing the multiplication order, OR more cleanly, treating `g ∈ (X^i)^*` and using finite-dimensionality to identify `(X^i)^** ≅ X^i`-dual-of-dual which is `B^i`):
```
G(b)  =  (−1)^i [ (Δ^i b) · g  −  g · (Δ^i b) ],    b ∈ B,
```
where `Δ^i b = b₁ ⊗ ... ⊗ b_i ∈ B^⊗i` is the i-fold coproduct, and the multiplications are in `B^⊗i` (treating `g ∈ B^⊗i` via the identification above; for `i = 1`, `g ∈ B` and the multiplication is just the regular product in `B`).

**For `i = 1`** (our case): `G(b) = (−1)^1 [ b · g − g · b ] = −[b, g]` (the negative commutator with `g ∈ B`). So `G: B → B` is (up to sign) the inner derivation `[−, g]` of `B`.

**Concrete for our case**: source is `Z¹_h(X, k) ≅ B^⊗1 = B` (243-dim) — but as noted, `dim HH¹(X) = 0`, so the source is zero and `G = 0` identically.

### 9d. The map the task actually wants: `δ_modern: H̃¹_b(B) → HH²(D(B))` — given by MW eq. (3.4.4)

The task part 5 asks for the connecting homomorphism `δ: H̃¹_b(B⁺) → HH²(D(B⁺))` "that maps our bialgebra 1-cocycles to the HH² classes we extracted at A₁". This is `δ_modern`, which in MW's notation is the THIRD arrow of the LES (the unlabeled arrow from `H̃^i_b(B)` to `H^{i+1}_h(D(B), k)`), and is given by MW eq. (3.4.4).

**Verbatim from MW** (lines 1057–1081):
> "The map `ῑ: H̃^n_b(B) → H^{n+1}_h(D(B), k)` is given by the composite
> `H̃^n_b(B) ≃ H^{n+1}(Tot C⁰) → H^{n+1}(Tot C) → H^{n+1}(Diag C) → H^{n+1}_h(D(B), k).`
> `Φ (φc̃)^*`
> More precisely, if `f̄: X^i ⊗ B^{n+1−i} → k` corresponds to `f: B^{n+1−i} → B^i`, then
> `(3.4.4)    ῑ f = f̄ · ∂^X_{i+1} ... ∂^X_{n+1} · ∂^B_0 ... ∂^B_0 · φ · c̃`."

(Note MW labels this map `ῑ`, but in the W2-1b / modern convention this is `δ_modern`. The double-use of `ῑ` between MW eq. (3.4.4) and MW §3.3 is itself confusing: in §3.3, `ῑ` is the FIRST arrow `H^i_h(D(B)) → H^i_h(X) ⊕ H^i_h(B)`, while in (3.4.4), `ῑ` is the THIRD arrow `H̃^i_b(B) → H^{i+1}_h(D(B))`. The two `ῑ`'s are different maps; this is a notational inconsistency in MW that we just have to live with.)

**Abstract (cleaned up)**: For `n = 1` (so `i ∈ {0, 1}` and `n+1 = 2`):
- If `i = 0`: `f: B² → B⁰ = k` (Hochschild-type 2-cochain with trivial coefficients), `f̄: X⁰ ⊗ B² → k`.
- If `i = 1`: `f: B¹ → B¹ = B` (a 1-cochain in the bialgebra complex, i.e. our `h: B̄ → B̄`), `f̄: X¹ ⊗ B¹ → k`.

The formula for `i = 1, n = 1` (the case for `H̃¹_b(B) → HH²(D(B))`):
```
δ_modern(h)  =  h̄  ·  ∂^X_2  ·  ∂^B_0  ·  φ  ·  c̃   ∈  Hom(D(B)^⊗2, k) = C²(D(B), k),
```
where:
- `c̃: D(B)^⊗2 → X^⊗2 ⊗ B^⊗2` is the cosimplicial Alexander–Whitney / flip map (built from the crosstalk `c: B ⊗ X → X ⊗ B` of the Drinfeld double).
- `φ: X^⊗2 ⊗ B^⊗2 → X^⊗2 ⊗ B^⊗2` is the Lusztig / MW `φ`-map (`φ(x, a) = ⟨x₁, S⁻¹(a₁)⟩ x₂ ⊗ a₂`).
- `∂^X_2: X² ⊗ B² → X¹ ⊗ B²` is the 2nd face map on the X-side.
- `∂^B_0: X¹ ⊗ B² → X¹ ⊗ B¹` is the 0th face map on the B-side.
- `h̄: X¹ ⊗ B¹ → k` corresponds to `h: B¹ → B¹` under `Hom(B¹, B¹) ≅ Hom(X¹ ⊗ B¹, k)`.

The composition produces a 2-cochain on `D(B) = X ⋊ B`. For `h` to be a cocycle (`∂_b h = 0`), this 2-cochain is a Hochschild 2-cocycle (i.e. `δ_modern(h) ∈ Z²_h(D(B), k)`), and its class in `HH²(D(B)) = H²_h(D(B), k)` is `δ_modern([h])`.

**Concrete for our case** (`B = B⁺_Borel(u_q(sl_3))`, `i = 1`):
- The source is `H̃¹_b(B⁺)` (conjecturally 3-dim).
- The target is `HH²(D(B⁺)) = HH²(u_q(sl_3))` (conjecturally 9-dim).
- The map is **injective** by W2-1b's verification that `dim HH¹(B) = dim HH¹(X) = 0` (so `ker(δ_modern) = im(π̄) = 0` by LES exactness).
- The image `im(δ_modern) = ker(π̄ at degree 2)` is the kernel of the double restriction `HH²(D(B⁺)) → HH²(B⁺) ⊕ HH²(X)`, which is conjecturally 3-dim (the "Cartan-type / mixed E–F" classes).

**Important note for W3-1c**: We do **NOT** need to compute `δ_modern` explicitly. The W3-1c task is to compute `dim H̃¹_b(B⁺)` **directly** via the bialgebra cochain complex (Section 6 above), and the injectivity of `δ_modern` then gives `dim im(δ_modern) = dim H̃¹_b(B⁺)` automatically. Computing `δ_modern` itself would require building 2-cochains on `D(B⁺)` — i.e. `dim(D(B⁺))² = 6561² ≈ 4.3 × 10⁷`-dim cochains — which is the intractable W2-1a regime. Avoid this.

### 9e. Summary of the LES maps (W2-1b / modern convention)

```
... → HH^i(D(B))  --ῑ-->  HH^i(X) ⊕ HH^i(B)  --π̄-->  H̃^i_b(B)  --δ-->  HH^{i+1}(D(B)) → ...
                       [MW (3.4.4)?]            [MW (3.4.1)+(3.4.2)]   [MW (3.4.4)]
```

For `i = 1` and `B = B⁺_Borel(u_q(sl_3))` at ℓ=3:
- `dim HH¹(D(B)) = ?` (unknown directly; conjecturally related to `dim HH¹(u_q(sl_3))`).
- `dim HH¹(X) ⊕ HH¹(B) = 0 ⊕ 0 = 0` (W2-1b verified).
- `dim im(π̄ at deg 1) = 0` (zero source).
- `dim ker(δ) = dim im(π̄ at deg 1) = 0`, so **δ is injective**: `dim im(δ) = dim H̃¹_b(B⁺)`.
- `dim HH²(D(B)) = dim im(δ) + dim im(ῑ at deg 2) = dim H̃¹_b(B⁺) + dim im(ῑ at deg 2)`.

Under the conjecture: `dim HH²(D(B)) = 9`, `dim im(ῑ at deg 2) = 6` (the `2|Φ⁺|` ℓ-th-power classes), so `dim H̃¹_b(B⁺) = 9 − 6 = 3 = C(3, 2)`. ✓

---

## §10. Summary and recipe for W3-1c

### 10a. What W3-1c needs to compute

```
dim_C H̃¹_b(B⁺_Borel(u_q(sl_3)), C)   at   ℓ = 3.
```

**Expected answer** (conjecture): `3 = C(3, 2)`.

### 10b. The computation in one line

```
dim H̃¹_b(B⁺)  =  dim ker( ∂_b : Hom(B̄, B̄) → Hom(B̄², B̄) ⊕ Hom(B̄, B̄²) )
             =  242²  −  rank(∂_b)
             =  58564  −  rank(∂_b).
```

### 10c. The matrix `∂_b` to build

For each `h ∈ Hom(B̄, B̄)` (a `242 × 242` matrix `H`), `∂_b H` is the pair `(∂^h H, −∂^c H) ∈ Hom(B̄², B̄) ⊕ Hom(B̄, B̄²)`. The full matrix of `∂_b` has shape `(2 · 242³) × 242² = 28344976 × 58564`.

**Block decomposition** (by weight, as in W2-1b): `B̄` decomposes as `⊕_{w ∈ (Z/3)²} B̄_w` where `B̄_w` is the 27-dim (or so) subspace of weight `w`. Then:
- `Hom(B̄, B̄) = ⊕_{w} Hom(B̄_w, B̄_w) ⊕ ⊕_{w ≠ w'} Hom(B̄_w, B̄_{w'})` — but actually `∂_b` preserves the "diagonal" weight, so we can split `∂_b` into 9 diagonal blocks `Hom(B̄_w, B̄_w) → Hom(B̄_w², B̄_w) ⊕ Hom(B̄_w, B̄_w²)` plus possibly off-diagonal terms. Need to verify weight-preservation carefully — `∂^h` preserves weight (since `a · h(b)`, `h(a b)`, `h(a) · b` all have weight `wt(a) + wt(b)` if `h` is weight-preserving, but `h` need not be weight-preserving a priori). The block structure is more subtle than for HH¹; W3-1c should compute the full matrix or use weight-decomposition carefully.

### 10d. Algorithmic steps for W3-1c

1. **Build the multiplication table** of `B` on the 243-element PBW basis — already done in `verify_sl3_bplus_hh2.py:build_mult_sparse` (lines 446–459). Reuse this.

2. **Build the comultiplication table** `Δ_matrix` of `B` on the 243-element PBW basis — **new code needed**. Use the Lusztig formulas (Section 8b above) and the algebra-homomorphism property. Sanity check: coassociativity `(Δ ⊗ 1) Δ = (1 ⊗ Δ) Δ` on a sample of basis elements; counitality `(ε ⊗ 1) Δ = id = (1 ⊗ ε) Δ`.

3. **Build the augmentation** `ε: B → C` on the PBW basis: `ε(K_i) = 1`, `ε(E_i) = 0`, `ε(E₁₂) = 0`. Then `B̄ = ker ε` is the 242-dim subspace spanned by all PBW basis elements except the unit. Choose a basis of `B̄` (e.g. the 242 non-unit PBW elements).

4. **Build `∂^h` matrix**: for each pair `(a_idx, b_idx) ∈ {0, ..., 241}²` and each output basis element `t_idx ∈ {0, ..., 241}`, compute the coefficient of `basis[t_idx]` in `basis[a_idx] · h(basis[b_idx]) − h(basis[a_idx] · basis[b_idx]) + h(basis[a_idx]) · basis[b_idx]`, where `h` is encoded as `H[j, k] = "coefficient of basis[j] in h(basis[k])"`. This is a `242³ × 242²` matrix; use sparse representation.

5. **Build `∂^c` matrix**: for each `c_idx ∈ {0, ..., 241}` and each output basis pair `(s_idx, t_idx) ∈ {0, ..., 241}²`, compute the coefficient of `basis[s_idx] ⊗ basis[t_idx]` in `basis[c_idx]_1 ⊗ h(basis[c_idx]_2) − Δ(h(basis[c_idx])) + h(basis[c_idx]_1) ⊗ basis[c_idx]_2`. This is a `242³ × 242²` matrix.

6. **Stack**: `∂_b = [∂^h ; −∂^c]`, shape `2 · 242³ × 242²`.

7. **Compute rank**: use sparse SVD (e.g. `scipy.sparse.linalg.svds`) on `∂_b`. The expected rank is `58564 − 3 = 58561` (under the conjecture). Tolerance: `1e-12 × σ_max` (consistent with W2-1b).

8. **Output**: `dim H̃¹_b(B⁺) = 58564 − rank(∂_b)`. If this equals 3, the conjecture at A₂ is verified (modulo the structural prediction `dim im(ῑ at deg 2) = 6`, which itself requires either the AKM rigidity argument or an explicit verification that the 6 ℓ-th-power classes survive into `im(ῑ)`).

### 10e. Cross-checks W3-1c should perform

- **Coherence of `∂^h ∂^c = ∂^c ∂^h`**: build `∂^h` and `∂^c` as operators on `Hom(B̄, B̄)` and verify `∂^h ∂^c h − ∂^c ∂^h h = 0` on a random sample of `h`'s.
- **`∂_b² = 0`**: verify `∂_b ∘ ∂_b = 0` (on a sample).
- **Symmetry with W2-1b's HH¹ computation**: `∂^h: Hom(B̄, B̄) → Hom(B̄², B̄)` is the same as the dual of the Hochschild differential `d¹: C¹(B, B̄) → C²(B, B̄)` used in W2-1b. Sanity-check by comparing the `∂^h` matrix to the (transposed) W2-1b `d¹` matrix.
- **At A₁ (sl_2)**: replicate the computation for `B = B⁺_Borel(u_q(sl_2))` at ℓ=3 (`dim B = 9`, `dim B̄ = 8`) and verify `dim H̃¹_b = 1` (the W2-1b-verified sl_2 value). This is the most important sanity check.

### 10f. Estimated resources

- Building `Δ_matrix`: ~seconds (analogous to `build_mult_sparse`, which takes <1 s).
- Building `∂^h` matrix: ~seconds (242³ ≈ 14M entries, sparse).
- Building `∂^c` matrix: ~seconds (similar size).
- Sparse SVD on `∂_b` (size `28344976 × 58564`): if weight-decomposed into 9 blocks of size `~2·242·81 × 81·81 = ~39204 × 6561` (per W2-1b weight-blocking pattern), each block SVD takes ~seconds. Total: **minutes**, comparable to W2-1b's HH¹ computation (~2 s) and W2-1a's HH² computation (~4 min).

**Total expected time**: under 10 minutes. **Memory**: under 4 GB (sparse representation).

---

## §11. Cross-references and open questions for downstream sub-agents

### 11a. For W3-1c (implementation)

- Use this document as the spec. Implement steps 10d(1)–(8).
- The multiplication table is already in `verify_sl3_bplus_hh2.py`; reuse it.
- The comultiplication table is **new** — see Section 8b for the formulas.
- Sanity-check with the sl_2 case first (`dim H̃¹_b(B⁺(u_q(sl_2))) = 1`).
- Output: `dim H̃¹_b(B⁺(u_q(sl_3)))` at ℓ=3, expected = 3.

### 11b. For W3-1d (or W2-1c, in progress per W2-1a): explicit cocycles

If W3-1c finds `dim H̃¹_b = 3`, the next step is to extract explicit basis cocycles `h_1, h_2, h_3: B̄ → B̄` (each a `242 × 242` matrix) representing the 3 generators. The conjecture identifies these as "Cartan-type / mixed E–F" cocycles; their explicit form would give a constructive lower bound `dim H̃¹_b ≥ 3` independent of the dimension-counting argument.

### 11c. For the orchestrator

- The W3-1b → W3-1c → (conjecture verified at A₂) path is now fully specified.
- If W3-1c yields `dim H̃¹_b = 3`: under the structural prediction `dim im(ῑ at deg 2) = 6` (which requires either AKM rigidity or explicit verification of the 6 ℓ-th-power classes), the conjecture at A₂ is **established**.
- If W3-1c yields `dim H̃¹_b ≠ 3`: the conjecture at A₂ is either **refuted** (if `> 9`) or **modified** (if between 4 and 9, requiring a refined structural picture).
- The `dim H̃¹_b = 3` computation is the **single highest-priority next step** for the project.

### 11d. Cross-link with W1-1b (Schweigert–Woike)

The 3 conjectured generators of `H̃¹_b(B⁺)` map (via `δ_modern = MW's ῑ`) to 3 of the 9 classes in `HH²(D(B⁺))`. The BV / Gerstenhaber bracket structure on `HH*(rep(D(B⁺)))` constrains the bracket relations among these 3 classes. If the bracket can be computed (W1-1b framework), this gives an independent verification channel: the 3 Cartan-type classes should form a 3-dim abelian Lie subalgebra under the bracket (matching `sl_3`'s Cartan).

---

## §12. Files produced / modified

- **Created**: `/home/z/my-project/hopf-decoherence/literature/notes/W3-1b-mw-equations.md` (this file).
- **No code modified** (this sub-task is purely expository).
- **No tests added** (no new code to test).

## §13. Status

- W3-1b: **completed**. All MW equations extracted and translated to our concrete case.
- W3-1c (implementation): **pending**, blocked on this document. With this document in hand, W3-1c can proceed directly to implementation.
- Conjecture at A₂: **open, reduced to W3-1c** (a ~10-minute computation).
