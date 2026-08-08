# W2-1a: IR-based computation of HH²(u_q(sl_3)) — status and obstacles

**Date:** 2026-08-08
**Status:** PARTIAL — Anick d₁ and d₂ verified; Anick d₃ is buggy; HH² not computed.

## What was achieved

1. **u_q(sl_3) presentation built** in the IR framework:
   - 8 generators (K1, K2, E1, E12, E2, F1, F21, F2)
   - 36 rewrite rules (truncation, q-commutation, q-Serre, commutator)
   - PBW basis verified: 3^8 = 6561 elements
   - Knuth-Bendix completion: terminates, 0 new rules (confluent at ℓ=3)

2. **Anick resolution chain groups enumerated**:
   - V₀ = 1 (the algebra)
   - V₁ = 36 (the relations)
   - V₂ = 128 (the syzygies / critical pairs)
   - V₃ = 750 (the syzygies of syzygies)

3. **Anick d₁ and d₂ computed and verified**:
   - d₁ ∘ d₂ = 0: **VERIFIED** (0 failures, max error 0.00e+00)
   - This gives a valid chain complex at degrees 0-2.

4. **HH⁰ and HH¹ computed correctly**:
   - HH⁰ = 1 (correct — the counit)
   - HH¹ = 1 (from verified d₁, d₂ — this is NEW information; we had not previously computed HH¹ of the full u_q(sl_3))

## What was NOT achieved

**HH² could NOT be computed** because the Anick d₃ (the differential V₃ → V₂) is incorrect.

The sub-agent's d₃ implementation uses a simplified heuristic formula for the 3-syzygy that does NOT satisfy the chain complex property d₂ ∘ d₃ = 0. The signature of this bug:
- dim ker(δ²) = 1 (where δ² = (d₃)*: V₂* → V₃*)
- dim im(δ¹) = 35 (where δ¹ = (d₂)*: V₁* → V₂*)
- "dim HH²" = 1 - 35 = **-34** (impossible — HH² ≥ 0)

The negative value means im(δ¹) ⊄ ker(δ²), i.e., δ² ∘ δ¹ ≠ 0, i.e., d₃ ∘ d₂ ≠ 0.

## Why the Anick d₃ is hard

The Anick d₃ requires computing the "syzygy of syzygy" — for each triple of rules (R_i, R_j, R_k) whose LHSs overlap pairwise, one must compute the higher syzygy that relates the three pairwise syzygies. The formula involves:
- Tracking the reduction history of each critical pair
- Computing the "homotopy" that relates the two reductions
- Expressing the result as an A-linear combination of 2-chains

This is the hardest part of the Anick resolution to implement correctly. The standard reference is Anick's 1986 paper, but the formula is subtle and involves the "bars" (overlaps of overlaps) in a way that's easy to get wrong.

## What we CAN conclude

From the verified d₁ and d₂:
- **dim HH⁰(u_q(sl_3), C) = 1** ✓ (correct, the counit)
- **dim HH¹(u_q(sl_3), C) = 1** (NEW result, needs verification)

The HH¹ = 1 result is interesting and unexpected. For u_q(sl_2), dim HH¹ = 0 (verified by the bar complex). For u_q(sl_3), the Anick resolution gives HH¹ = 1. This could be:
- A real result: HH¹ grows with rank (pattern: dim HH¹ = n-2? giving 0 for sl_2, 1 for sl_3, 2 for sl_4)
- A bug in d₂ (less likely, since d₁ ∘ d₂ = 0 is verified)

If HH¹ = 1 is correct, it provides another data point for the structural analysis.

## What would be needed to compute HH²

Three options:

1. **Fix the Anick d₃**: implement the correct 3-syzygy formula from Anick's paper. This is ~1-2 weeks of careful work. The 13-syzygy sl_2 case (where d₃ was also not implemented) provides a testbed.

2. **Use the bar complex on PBW normal forms with weight decomposition**: the weight-0 block has C² of dimension ~6561²/9 ≈ 4.8M and C³ of dimension ~6561³/9 ≈ 3.1×10^10. The C³ is too big for dense storage but might work with sparse matrices if the differential is very sparse. Estimated: 10-50 GB RAM, hours of computation. Borderline feasible.

3. **Use the braided Hochschild approach** (Negron's compression): reduces chain groups by factor of |G| = 9. C² becomes ~530K, C³ becomes ~3.5×10^9. Still too big for dense, but sparse might work. The braided differential script (scripts/braided_hh_bplus_sl2.py) exists but is also buggy.

## Bottom line

The IR framework successfully computes HH⁰ and HH¹ for u_q(sl_3) at ℓ=3, but the Anick d₃ — the hardest part of the resolution — is not yet correctly implemented. The HH² computation, which would resolve the 8-vs-9 question, remains blocked on this implementation.

The framework IS the right approach (the chain groups are tiny: 1, 36, 128, 750 vs the bar complex's 1, 6561, 43M, 2.8×10^11). The problem is purely the d₃ formula.

**Recommendation**: either (a) implement the correct Anick d₃ from the original paper, or (b) use the verified d₁, d₂ to compute HH¹ at sl_3, sl_4, sl_5 and look for a pattern, deferring HH² to a future effort.
