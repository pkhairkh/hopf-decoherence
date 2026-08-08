# W2-1a — Plan for direct bar complex computation of HH²(u_q(sl_3), C) at ℓ = 3

- **Task ID**: W2-1a
- **Agent**: Sub-agent 2a (Wave 2, general-purpose)
- **Date**: 2025-08-07
- **Status**: completed (planning only — computation intractable in sandbox)
- **Outputs**:
  - `/home/z/my-project/hopf-decoherence/scripts/plan_sl3_hh2.py` (planning / diagnostic script)
  - `/home/z/my-project/hopf-decoherence/scripts/sl3_plan_output.txt` (captured stdout)
  - This note

## Summary

Planned and analysed the direct bar-complex computation of `HH²(u_q(sl_3), C)` at `ℓ = 3`. The conjecture predicts `dim HH² = C(3,2) + 2|Φ⁺| = 3 + 6 = 9`. The direct bar complex turns out to be **intractable in the present sandbox** (4 GB RAM budget): even the smallest sub-computation (the principal block `u_0(sl_3)`, dim 729) requires ~29 GB of RAM just to store the `d²` matrix sparsely, and weeks of compute to obtain its rank. This note documents the dimensional analysis, quantifies the obstacles, and recommends three alternative approaches for subsequent sub-tasks (W2-1b BGG adaptation, W2-1c explicit cocycle construction, W2-1d Mastnak–Witherspoon LES analysis).

## 1. Dimensional structure of u_q(sl_3) at ℓ = 3

### 1.1 Algebra generators and dimension

| Block | Generators | Count |
|---|---|---|
| Cartan torus | `K1, K2` | 2 |
| Positive root vectors | `E1, E12 = E1 E2 − q E2 E1, E2` | 3 |
| Negative root vectors | `F1, F21 = F2 F1 − q F1 F2, F2` | 3 |
| **Total PBW generators** | | **8** |

Each PBW exponent ranges over `{0, 1, 2}` (since `K_i³ = 1`, `E_i³ = F_i³ = 0`, etc.). Therefore

$$\dim_\mathbb{C} u_q(\mathfrak{sl}_3)\big|_{\ell=3} \;=\; 3^8 \;=\; 6561.$$

This matches the task's heuristic `dim = 3^(3+3+rank) = 3^8`.

### 1.2 Weight space decomposition

The algebra is graded by the group `(Z/ℓ)^rank = (Z/3)^2` of weights (the character group of the Cartan torus `{K1^a K2^b}`). The weight of a PBW monomial `K1^a K2^b E1^c E12^e E2^d F1^f F21^g F2^h` is

$$(w_1, w_2) \;=\; (c + e - f - g,\;\; e + d - g - h) \pmod{3}.$$

The weight map `(Z/3)^6 → (Z/3)^2` (on the E-F part; the Cartan part is weight 0) is a surjective group homomorphism with kernel of size `3^4 = 81`. Hence each of the 9 weight spaces has E-F dimension 81, and full dimension `81 × 9 = 729` (the Cartan factor 9 = `3^rank`):

| Weight `(w_1, w_2)` | E-F dim | Cartan dim | Full dim |
|---|---|---|---|
| (0,0), (0,1), (0,2), (1,0), …, (2,2) | 81 | 9 | **729** |
| **Sum (9 weights)** | 729 | 81 | **6561** ✓ |

The distribution is **uniform**: every weight space has dim 729. This is a structural symmetry that simplifies the block-size analysis below.

### 1.3 Principal block

Per W1-1c, only the principal block `u_0` contributes to trivial-coefficient Hochschild cohomology `HH*(A, C)`, because the counit satisfies `ε(e_λ) = δ_{λ, 0}` for the block idempotents.

**Claim**: `u_0(sl_3)` at `ℓ = 3` coincides with the weight-0 subspace of `A = u_q(sl_3)` (as a subalgebra).

*Proof sketch*: The principal block corresponds to the central idempotent
$$e_0 \;=\; \frac{1}{|G|} \sum_{g \in G} g, \qquad G = \langle K_1, K_2 \rangle \cong (\mathbb{Z}/3)^2,$$
i.e., the projector onto weight 0. Since `e_0` is central, `u_0 = e_0 A e_0 = e_0 A`, which is precisely the weight-0 subspace of `A`.

Hence **dim u_0 = 729** (consistent with the heuristic `dim u_0 ≈ ℓ^{2N} = 3^6 = 729` for `N = |Φ⁺| = 3`).

For comparison: at `sl_2, ℓ=3`, dim u_0 = `ℓ^{2·1} = 9` = weight-0 subspace ✓.

## 2. Bar complex block sizes

### 2.1 On the full algebra A (dim 6561)

For each weight `w ∈ (Z/3)^2`, the bar complex `C^n(A, C) = Hom(A^{⊗n}, C)` has a weight-`w` block of dimension
$$\dim C^n_w \;=\; \sum_{w_1 + \dots + w_n = w} \prod_{i=1}^n \dim(A_{w_i}).$$

Computed values (verified by the planning script, summing to `dim(A)^n`):

| Weight | dim C¹ | dim C² | dim C³ |
|---|---|---|---|
| any of 9 weights | 729 | 4,782,969 | 31,381,059,609 |
| **Total (9 weights)** | 6,561 | 43,046,721 | 282,429,536,481 |
| Sanity: `dim(A)^n` | 6,561 | 43,046,721 | 282,429,536,481 ✓ |

The 9 weight blocks are all of equal size by the uniform-distribution symmetry.

### 2.2 On the principal block u_0 (dim 729)

The bar complex on `u_0` (equivalently: the weight-0 sub-complex of the bar complex on `A`) has chain groups

| | dim |
|---|---|
| `C¹(u_0)` | 729 |
| `C²(u_0)` | 531,441 |
| `C³(u_0)` | 387,420,489 |

This is the smallest direct sub-computation that could in principle yield `dim HH²(u_0, C) = dim HH²(u_q(sl_3), C)` (the full trivial-coef Hochschild cohomology, per W1-1c).

## 3. Memory and time obstacles

### 3.1 Direct bar complex on full A

Each row of the bar differential `d²` has exactly 4 nonzeros (from the 4-term formula `ε(a)g(b,c) − g(a·b,c) + g(a,b·c) − g(a,b)ε(c)`).

For the weight-0 block (representative; all 9 are equal):
- `d²` shape: `31,381,059,609 × 4,782,969` (≈ 31.4 billion rows × 4.8 million columns)
- Nonzeros (sparse): ~125.5 billion
- Sparse storage (complex128 + int32 index = 20 B per nonzero): **2.28 TB**
- Dense `d²`: ~2.2 PB (infeasible)
- Dense Gram `d²ᵀd²`: ~333 TB (infeasible)

**All 9 weights combined**: sparse storage **20.55 TB**.

On 4 GB RAM: weight-0 alone is **584× over budget**; all 9 weights is 5260× over budget.

**Time estimate** (sparse iterative rank via ARPACK-style SVD, 1 GFLOP/s):
- ~125 s per matvec
- ~10 matvecs per singular value
- Need ~4.8 M singular values to count zeros (worst case)
- Total: ~69,500 days (~190 years). **Infeasible.**

### 3.2 Direct bar complex on principal block u_0 (best case)

For `u_0` alone:
- `d²` shape: `387,420,489 × 531,441` (≈ 387 M rows × 531 K columns)
- Nonzeros: ~1.55 billion
- Sparse storage: **28.87 GB**
- Dense `d²`: ~3000 TB (infeasible)
- Dense Gram: ~4.1 TB (infeasible)

On 4 GB RAM: **7.2× over budget**. Even on a 32 GB machine, this is borderline (would require careful sparse rank-revealing QR, e.g., SuiteSparse SPQR).

**Time estimate** (sparse iterative rank):
- ~1.5 s per matvec
- ~10 matvecs per singular value
- ~531 K singular values needed
- Total: ~95 days. **Infeasible** in the sandbox; borderline on a 32 GB workstation (1–2 weeks with sparse QR).

### 3.3 Comparison with the tractable verified cases

| Case | dim A | weight-0 C² | weight-0 C³ | sparse d² | status |
|---|---|---|---|---|---|
| `u_q(sl_2)`, `ℓ=3` | 27 | 243 | 6,561 | 0.5 MB | ✅ verified (HH² = 3) |
| `B⁺(u_q(sl_3))`, `ℓ=3` | 243 | 6,561 | 1,594,323 | 122 MB | ✅ verified (HH² = 6) |
| `u_q(sl_3)`, `ℓ=3` (this task) | 6,561 | 4,782,969 | 31,381,059,609 | 2.28 TB | ❌ intractable |
| `u_0(sl_3)`, `ℓ=3` (principal block) | 729 | 531,441 | 387,420,489 | 28.87 GB | ❌ intractable on 4 GB |

Scaling ratios (weight-0 C³):
- `sl_3 / sl_2 = 4,782,969×` (from 6,561 to 31.4 billion)
- `sl_3 / B⁺(sl_3) = 19,683×` (from 1.6 M to 31.4 billion)

The `sl_3` case is ~5 orders of magnitude larger than the largest case we have verified.

## 4. Smaller sanity checks (what *could* be done)

The planning script considers and rejects several smaller sanity-check computations:

| Option | Verdict |
|---|---|
| (i) Cartan subalgebra `C[K1,K2]/(K1³−1,K2³−1)`, dim 9 | Splits as `C^9` over C (CRT); semisimple commutative ⇒ `HH² = 0`, not `3 = C(3,2)`. The "Cartan piece" `3` in the conjecture is **not** `HH²(Cartan subalgebra)`; it comes from deformations of the cross-relations `K_i E_j = q^{a_{ij}} E_j K_i`, etc., which only make sense in the full Drinfeld double. |
| (ii) `B⁺(sl_3)` at `ℓ=3` | Already verified: `HH² = 6 = 2|Φ⁺|` ✓ (matches conjecture restricted to the Borel, where the Cartan piece `C(n+1,2) = 0` because there is no negative-root part). |
| (iii) `sl_2` at `ℓ=3` | Already verified: `HH² = 3 = C(2,2) + 2·1` ✓. |
| (iv) "Cartan × sl_2"-type subalgebra | Not a natural subalgebra of `u_q(sl_3)`. |
| (v) Restrict to monomials with exponents in `{0,1}` (dim 2^8 = 256) | Not closed under multiplication (`E1·E1 = E1²` has exponent 2). Not a subalgebra. |

**Conclusion**: no smaller sub-computation of `u_q(sl_3)` at `ℓ=3` verifies the conjecture's full prediction `dim = 9`. The conjecture at `sl_3, ℓ=3` remains **unverified** after W2-1a.

## 5. Recommended alternative approaches

The direct bar complex on `u_q(sl_3)` at `ℓ=3` is intractable in this sandbox. Three alternative approaches, in order of promise, are recommended for subsequent sub-tasks:

### 5.1 W2-1b — BGG-style resolution (Hemelsoet–Voorhaar approach, adapted)

Hemelsoet–Voorhaar compute block cohomology via a BGG resolution, which replaces the bar complex's `dim(A)^n`-size chain groups with `~|W| × dim P(λ)`-size chain groups. For `sl_3` at `ℓ=3`:
- `|W| = 6` (Weyl group `S_3`)
- `dim P(0) = 729` (projective cover of the trivial module)
- ⇒ chain groups of dim ~4,374

This is **~10⁵× smaller** than the bar complex (387 M for `C³(u_0)`).

Their software (https://github.com/RikVoorhaar/bgg-cohomology) computes **self-coefficient** `HH*(u_λ, u_λ)`. Per W1-1c, our project's invariant is **trivial-coefficient** `HH*(u_q(g), C)` — a different invariant that is not directly computed by their code. Adapting their BGG machinery to trivial coefficients is the main work item.

**Status check**: Hemelsoet–Voorhaar Prop 5.1 explicitly excludes `s = 2` for the principal block. The `s = 2` case requires either extending their range or finding an independent BGG-style computation.

### 5.2 W2-1c — Explicit cocycle construction (constructive proof)

The conjecture predicts `dim HH² = 9 = 3 (Cartan) + 6 (root)`. Construct 9 explicit candidate 2-cocycles `f_k ∈ C²(A, C) = Hom(A ⊗ A, C)`:

- **3 Cartan cocycles**: deformations of the Drinfeld-double's Cartan-cross-relations (`K_i E_j`, `K_i F_j`); these are not captured by `HH²` of the Cartan subalgebra alone (which is 0; see §4(i)).
- **6 root cocycles**: 2 per positive root (`E1, E12, E2`), one for the E-side and one for the F-side.

Verify for each candidate `f_k`:
1. Cocycle condition: `d² f_k = 0`, i.e., `f_k(a·b, c) − f_k(a, b·c) + ε(a) f_k(b, c) − ε(c) f_k(a, b) = 0` for all `a, b, c ∈ A`.
2. Linear independence modulo `im d¹` (i.e., not a coboundary).

**Cost**: `O(dim A × 9)` per cocycle check ≈ 60 K operations; tractable in seconds once the multiplication table is built. The multiplication table for the full `u_q(sl_3)` (extending the Borel `verify_sl3_bplus_hh2.py` to include the F-generator commutators `[E_i, F_j] = δ_{ij} (K_i − K_i^{−1})/(q − q^{−1})`) is the main implementation work.

This approach gives a **constructive upper bound** `dim HH² ≥ 9`. To prove the conjecture, one also needs `dim HH² ≤ 9`, which requires either:
- An independent argument (e.g., the Mastnak–Witherspoon LES, see §5.3), or
- A complete computation of `ker d² / im d¹` (which is the intractable part).

### 5.3 W2-1d — Mastnak–Witherspoon long exact sequence (lower priority)

For the Drinfeld double `D(B)`:
$$\dots \to HH^n(B) \oplus HH^n(B^*) \to HH^n(D(B)) \to HH^{n+1}(B \otimes B^*) \to \dots$$

Known inputs:
- `HH²(B⁺(sl_3), C) = 6 = 2|Φ⁺|` (verified)
- `HH²(B⁺(sl_3)^*, C) = 6` (by duality)

LES gives `HH²(D(B))` as an extension of `6 ⊕ 6 = 12` by something, and the conjecture says `HH²(D(B)) = 9`. The deficit `12 − 9 = 3` should come from the connecting map `HH²(D(B)) → HH³(B ⊗ B*)`. Computing this connecting map is the main work item.

**Lower priority** — requires careful LES analysis. W1-1b noted that Schweigert–Woike do not provide an alternative to this LES, so the LES route remains necessary.

## 6. Hardware requirements (if direct computation were attempted)

For the **smallest possible** direct computation, `HH²(u_0(sl_3), C)` on the principal block (dim 729):

| Resource | Required | Sandbox budget | Verdict |
|---|---|---|---|
| RAM (sparse `d²` storage) | ~29 GB | 4 GB | ❌ 7.2× over |
| RAM (dense Gram `d²ᵀd²`) | ~4.1 TB | 4 GB | ❌ infeasible |
| Time (sparse iterative rank) | ~95 days | hours | ❌ infeasible |
| Time (sparse QR rank-revealing) | ~1–2 weeks | hours | ❌ infeasible |

For the **full** computation `HH²(u_q(sl_3), C)` (all 9 weights):

| Resource | Required | Sandbox budget | Verdict |
|---|---|---|---|
| RAM (sparse `d²` storage) | ~20 TB | 4 GB | ❌ 5000× over |
| Time (sparse iterative rank) | ~190 years | hours | ❌ infeasible |

**Hardware that would make the `u_0` computation feasible**:
- A workstation with **64–128 GB RAM** and a fast SSD (for sparse QR out-of-core).
- Time budget: **1–2 weeks** of compute.
- Software: SuiteSparse SPQR (rank-revealing sparse QR) or a custom distributed sparse rank algorithm.

**Hardware for the full computation**:
- A distributed cluster with **~30 nodes × 128 GB RAM each** (≈ 4 TB aggregate).
- Time: **months** of compute.

Neither is available in the present sandbox.

## 7. What was verified vs. what remains open

| Verification target | Status |
|---|---|
| `dim HH²(u_q(sl_2), C) = 3` at `ℓ=3` | ✅ Verified (Wave 0, `verify_sl2_hh2_fast.py`) |
| `dim HH²(B⁺(u_q(sl_3)), C) = 6` at `ℓ=3` | ✅ Verified (Wave 0, `verify_sl3_bplus_hh2.py`) |
| `dim HH²(u_q(sl_3), C) = 9` at `ℓ=3` (this task) | ❌ **Unverified** (direct bar complex intractable; alternatives deferred to W2-1b/c/d) |
| `dim HH²(u_q(sl_3), C) = 9` at `ℓ=5,7` | ❌ Unverified (would be even larger: `dim = 5^8 = 390625` at `ℓ=5`) |
| `dim HH²(u_q(sl_n), C) = C(n+1,2) + 2|Φ⁺|` general | ❌ Unverified for `n ≥ 3` |

## 8. Files produced / modified

- **Created**: `/home/z/my-project/hopf-decoherence/scripts/plan_sl3_hh2.py` — the planning / diagnostic script.
- **Created**: `/home/z/my-project/hopf-decoherence/scripts/sl3_plan_output.txt` — captured stdout of the planning script.
- **Created**: `/home/z/my-project/hopf-decoherence/literature/notes/W2-1a-sl3-bar-complex-plan.md` — this note.
- **No existing scripts or tests modified.**

## 9. Open questions for downstream sub-agents

1. **For W2-1b (BGG adaptation)**: Can the Hemelsoet–Voorhaar BGG software be adapted to compute **trivial-coefficient** `HH*(u_0(sl_3), C)` at `s = 2`? Their Prop 5.1 explicitly excludes `s = 2` for the principal block; the exclusion is technical (range bound on `s`), not a fundamental obstruction. What is the precise obstruction, and can it be bypassed?

2. **For W2-1c (explicit cocycles)**: Can we write down the 9 candidate cocycles explicitly? The 6 root cocycles are likely straightforward (analogous to the 2 root cocycles in `u_q(sl_2)`); the 3 Cartan cocycles are subtler — they correspond to deformations of the cross-relations `K_i E_j`, `K_i F_j`, and `K_i K_j` (the latter being trivial since `K_i K_j = K_j K_i`). Need to identify which 3 mixed Cartan-root deformations give the cocycles.

3. **For W2-1d (LES analysis)**: The Mastnak–Witherspoon LES has a connecting map `HH²(D(B)) → HH³(B ⊗ B*)`. Is this connecting map computable for `B = B⁺(sl_3)` at `ℓ=3`? If yes, the LES gives `dim HH²(D(B))` from the known `HH²(B) ⊕ HH²(B*) = 12` minus the rank of the connecting map (which should be 3 to give `HH² = 9`).

4. **For the orchestrator**: Is a partial verification sufficient to publish? E.g., if W2-1c constructs 9 cocycles (giving `dim HH² ≥ 9`) but no upper bound, is that a publishable result? Or do we need the full `dim HH² = 9` (matching the conjecture)?

5. **Bigger picture**: For `n ≥ 3` and odd `ℓ`, is there any case where the conjecture can be **fully verified** by direct bar complex? The dimensional growth `dim(u_q(sl_n)) = ℓ^{n²+2n}` is prohibitive: for `sl_3` at `ℓ=3`, dim 6561 (intractable); for `sl_2` at `ℓ=5`, dim 125 (tractable); for `sl_2` at `ℓ=7`, dim 343 (tractable). For `sl_n` with `n ≥ 3`, direct bar complex is likely intractable at all odd `ℓ ≥ 3`. The BGG / cocycle / LES approaches are therefore not just expedients — they are **necessary** for higher-rank verification.
