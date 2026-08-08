# W3 — Bialgebra 1-cocycle computation for B⁺(u_q(sl_3)) at ℓ=3

**Date:** 2026-08-08
**Status:** COMPLETED with unexpected result

## Executive summary

We computed `dim H̃¹_b(B⁺(u_q(sl_3))), C)` at ℓ = 3 directly via the
Mastnak-Witherspoon bialgebra cochain complex. The result:

| Quantity | Prediction (conjecture's structural split) | Computed |
|---|---|---|
| `dim H̃¹_b(B⁺(u_q(sl_2))), C)` at ℓ=3 | `C(2,2) = 1` | **1** ✓ |
| `dim H̃¹_b(B⁺(u_q(sl_3))), C)` at ℓ=3 | `C(3,2) = 3` | **2** ✗ |

The sl_2 cross-check matches the conjecture. The sl_3 computation
**refutes the conjecture's structural prediction** that
`dim im(δ) = C(n+1,2)`.

## What this means

The Mastnak-Witherspoon LES (3.3.1) at degree 2 reads:

```
HH¹(D(B⁺)) → HH¹(B⁺) ⊕ HH¹(B⁻) → H̃¹_b(B⁺) --δ--> HH²(D(B⁺)) --π̄--> HH²(B⁺) ⊕ HH²(B⁻) → H̃²_b(B⁺)
```

Verified facts at sl_3, ℓ=3:
- `dim HH¹(B⁺) = dim HH¹(B⁻) = 0` (verified W2-1b)
- `dim HH¹(D(B⁺)) = 0` (verified, paper §4)
- `dim HH²(B⁺) = dim HH²(B⁻) = 5` (verified, paper §6)
- `dim H̃¹_b(B⁺) = 2` (THIS WAVE — VERIFIED)
- Since HH¹ vanishes, δ is injective, so `dim im(δ) = dim H̃¹_b(B⁺) = 2`

Conjecture's structural decomposition was: `dim HH²(D(B⁺)) = dim im(δ) + dim im(π̄) = C(n+1,2) + 2|Φ⁺| = 3 + 6 = 9`.

With `dim im(δ) = 2` instead of 3, the LES gives:
- If `dim HH²(D(B⁺)) = 9` (full conjecture holds): `dim im(π̄) = 7` (not 6 as predicted)
- If `dim im(π̄) = 6` (structural prediction holds): `dim HH²(D(B⁺)) = 8` (conjecture refuted)

So the full conjecture (`dim HH² = 9`) is NOT yet verified nor refuted at A_2. But the structural prediction `dim im(δ) = C(n+1,2)` is **refuted** at A_2.

## The bug we caught during this wave

The first run of the sl_3 script gave `dim = 0`, contradicting both the
conjecture and the sl_2 cross-check. Investigation showed a subtle but
critical bug in the **weight function**:

- **Wrong (original):** `wt(K1^a K2^b E1^c E12^e E2^d) = (c+e, e+d) mod 3`
  This is the decomposition by simple-root *exponents*, giving 9 weight
  spaces of 27 elements each.
- **Correct (fixed):** `wt(...) = (2c+e-d, -c+e+2d) mod 3`
  This is the true K-eigenvalue decomposition (using the Cartan matrix),
  giving 3 weight spaces of 81 elements each.

The difference matters because the Cartan matrix of sl_3 has determinant 3,
which is 0 mod 3. So the simple-root basis and the fundamental-weight
basis give *different* decompositions mod 3, and only the K-eigenvalue
decomposition is preserved by multiplication and comultiplication.

After the fix:
- sl_2 cross-check: still `dim = 1` (matches)
- sl_3 computation: `dim = 2` (refutes structural prediction of 3)

## Methodology (verified)

1. **Multiplication table**: lifted from `verify_sl3_bplus_hh2.py`, sanity-checked.
2. **Comultiplication table**: built from generator coproducts
   (Δ(K_i) = K_i ⊗ K_i, Δ(E_i) = E_i ⊗ K_i + 1 ⊗ E_i,
   Δ(E_12) derived from Δ(E_12) = Δ(E_1)Δ(E_2) - q Δ(E_2)Δ(E_1)).
3. **Algebraic sanity checks** (all pass to 10⁻¹⁴):
   - Δ(E_12) formula consistent: max error 1.11e-16
   - Coassociativity (Δ⊗1)Δ = (1⊗Δ)Δ on 30 random samples: max error 4.31e-15
   - Counitality: max error 0.00e+00
4. **Bialgebra coboundary ∂_b = (∂^h, -∂^c)**:
   - `∂^h h(a, b) = a h(b) - h(ab) + h(a) b` (Hochschild)
   - `∂^c h(c) = c_1 ⊗ h(c_2) - Δ(h(c)) + h(c_1) ⊗ c_2` (coalgebra)
5. **Weight decomposition**: 3 weight spaces (Cartan matrix basis), each ≤ 19522 cols.
6. **Rank computation**: Gram matrix `A^* A` + `eigvalsh`, tolerance 1e-10 relative.
7. **Spectral gap**: clean (smallest nonzero eigenvalue > 1, largest "zero" < 1e-10).

## Files produced

- `scripts/compute_h1b_bplus_sl3.py` — main computation (corrected weight)
- `scripts/compute_h1b_bplus_sl2.py` — sl_2 cross-check
- `scripts/h1b_sl3_output.txt` — captured output
- `scripts/h1b_sl2_output.txt` — captured output
- `tests/test_h1b_computation.py` — tests

## Implications for the conjecture

The conjecture's count `dim HH²(u_q(sl_3)) = 9` is **not directly verifiable**
in this sandbox (the direct bar complex on the 6561-dim algebra is
intractable per W2-1a). The structural prediction `dim im(δ) = C(n+1,2)` is
**refuted** at A_2 (computed = 2, predicted = 3).

Three possibilities remain for the full conjecture at A_2:

1. **Conjecture count is correct, structural split is wrong**:
   `dim HH²(u_q(sl_3)) = 9 = 2 + 7`, with `dim im(δ) = 2` (verified)
   and `dim im(π̄) = 7` (would need direct verification).
   This means the "2|Φ⁺| = 6" prediction for `dim im(π̄)` is also wrong;
   the actual ℓ-th-power + cross-relation count is 7, not 6.

2. **Conjecture count is wrong**: maybe `dim HH²(u_q(sl_3)) = 8 = 2 + 6`,
   with structural prediction `dim im(π̄) = 6` correct but the
   C(n+1,2) term wrong. This would refute the conjecture at A_2.

3. **Both wrong**: some other split entirely.

The way to distinguish: compute `dim im(π̄)` directly. This requires the
restriction map from HH²(D(B⁺)) to HH²(B⁺) ⊕ HH²(B⁻), which needs the
full HH²(D(B⁺)) — intractable in this sandbox.

## Recommended next steps

1. **Communicate the result**: contact Cris Negron, Sarah Witherspoon, and
   You Qi. The structural prediction `dim im(δ) = C(n+1,2)` was a natural
   guess based on sl_2; the sl_3 refutation is significant new information.

2. **Compute at sl_2, ℓ=5** to check whether `dim H̃¹_b = 1` holds at
   other odd ℓ (sanity check that sl_2 is not also a fluke).

3. **Compute at sl_3, ℓ=5** to see whether `dim H̃¹_b` is 2 there too,
   or whether the value depends on ℓ.

4. **Investigate the sl_2 vs sl_3 difference**: at sl_2 we get 1 = C(2,2).
   At sl_3 we get 2, not 3 = C(3,2). Is the pattern `dim H̃¹_b = n-1`?
   If so, at sl_4 we'd expect 3, at sl_5 we'd expect 4, etc. This would
   be a different conjecture entirely.

5. **Re-examine the structural decomposition**: the C(n+1,2) guess was
   based on counting "Cartan-type" classes from K_i^ℓ = 1 relations.
   Maybe the actual count is n-1, corresponding to a smaller set of
   "diagonal" Cartan classes (one per simple root, minus one for the
   product relation).
