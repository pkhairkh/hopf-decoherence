# W1-1a-IR — `u_q(sl_2)` at `ℓ = 3` in the AST/IR framework: validation of `dim HH² = 3`

**Task ID**: W1-1a (IR validation; distinct from the W1-1a Hemelsoet–Voorhaar literature read)
**Agent**: Sub-agent 1a (Wave 1, general-purpose)
**Date**: 2025-08-09
**Status**: completed
**Output**:
- `/home/z/my-project/hopf-decoherence/ir/qomega.py` — specialized coefficient ring `Q(omega)`
- `/home/z/my-project/hopf-decoherence/ir/uq_sl2.py` — `u_q(sl_2)` presentation + HH² computation
- `/home/z/my-project/hopf-decoherence/scripts/verify_ir_sl2.py` — verification script
- `/home/z/my-project/hopf-decoherence/scripts/ir_sl2_output.txt` — captured output
- `/home/z/my-project/hopf-decoherence/tests/test_ir_uq_sl2.py` — 11 tests
- This note.

---

## 1. Goal

The AST/IR framework built in W0-1a (parser + normal-form reducer) and
W0-1b (Knuth–Bendix completion + Anick-resolution generator enumeration)
had not yet been applied to an actual small quantum group. This task
applies it to `u_q(sl_2)` at `ℓ = 3` and verifies that the resulting
Hochschild cohomology dimension matches the value computed by the direct
bar complex in `scripts/verify_sl2_hh2.py` and certified exactly in
`scripts/certify_a1_exact.py`:

```
dim_C HH^2(u_q(sl_2), C) = 3 = C(2,2) + 2*|Φ^+(A_1)| = 1 + 2.
```

This is the conjecture `dim_C HH^2(u_q(g), C) = C(n+1, 2) + 2|Φ^+|` at
rank one. Validating it via the IR framework (rather than the hand-coded
bar complex) is the prerequisite for applying the IR framework to
`u_q(sl_3)` at `ℓ = 3`, where the bar complex is intractable
(`dim u_q(sl_3) = 6561`, so `dim C^2 = 4.3 × 10^7`).

## 2. The `Q(omega)` coefficient ring (approach (a) to the commutator division)

The commutator relation

```
[E, F] = (K - K^{-1}) / (q - q^{-1})
```

involves division by `q - q^{-1}`, which is **not** a unit in the
generic ring `Z[q, q^{-1}]` used by `QLaurent`. The task description
suggested three approaches: (a) localize `Z[q, q^{-1}]` at
`(q - q^{-1})`; (b) multiply through by `(q - q^{-1})`; (c) introduce a
new generator for the inverse.

We use a specialization variant of (a): at `ℓ = 3` the relevant
coefficient field is `Q(omega)` where `omega = e^{2πi/3}` is a primitive
3rd root of unity. This field contains `Z[q, q^{-1}]` (via `q ↦ omega`)
and the inverse `1/(q - q^{-1}) = 1/(omega - omega^2) = (-1 - 2*omega)/3`,
so all structure constants of `u_q(sl_2)` at `ℓ = 3` live in `Q(omega)`.

The new module `ir/qomega.py` defines `QOmega3`, a subclass of
`QLaurent` that stores elements as `{0: a, 1: b}` (a dict with exponents
in `{0, 1}` and Fraction coefficients) representing `a + b*omega` with
`omega^2 = -1 - omega` and `omega^3 = 1`. Every arithmetic operation
reduces the result modulo `(q^3 - 1, q^2 + q + 1)` to keep the canonical
form.

Key features:
- **Exact arithmetic** via `fractions.Fraction` (no floating-point error
  in the algebra operations).
- **Field structure**: every nonzero element has a multiplicative
  inverse, so `try_divide` always succeeds for nonzero divisors. This is
  essential for Knuth–Bendix completion, which orients critical pairs by
  dividing the leading coefficient into the rest.
- **Subclass of `QLaurent`**: passes the existing `isinstance(x,
  QLaurent)` checks in `Polynomial.__mul__` and `Term.__mul__`.
  Python's "subclass reflected operator first" rule (`QOmega3.__radd__`
  is called before `QLaurent.__add__` when the right operand is a
  `QOmega3`) makes the existing `NormalFormReducer` work with
  `QOmega3` coefficients without modification.
- **Conversion to complex**: `QOmega3.to_complex()` returns the Python
  complex number, used only at the final rank-computation step (which
  uses SVD on a 729×729 complex matrix).

Three useful constants are pre-computed:
- `OMEGA = omega`
- `OMEGA2 = omega^2 = -1 - omega`
- `Q_MINUS_Q_INV_INV = 1/(q - q^{-1}) = (-1 - 2*omega)/3` — the
  commutator scalar.

Sanity check: `Q_MINUS_Q_INV * Q_MINUS_Q_INV_INV == 1` ✓ (verified in
the test suite).

## 3. The `u_q(sl_2)` presentation

Generators: `K, E, F` (indices 0, 1, 2). Six rewrite rules, oriented by
length-lex to the PBW form `K^a E^b F^c` with `0 ≤ a, b, c ≤ 2`:

| Rule | LHS | RHS (at ℓ = 3) | Source |
|------|-----|----------------|--------|
| R1 | `K^3` | `1` (empty monomial) | `K^ℓ = 1` |
| R2 | `E^3` | `0` | `E^ℓ = 0` |
| R3 | `F^3` | `0` | `F^ℓ = 0` |
| R4 | `E K` | `omega * K E` | from `K E = q^2 E K` ⟹ `E K = q^{-2} K E = omega * K E` |
| R5 | `F K` | `omega^2 * K F` | from `K F = q^{-2} F K` ⟹ `F K = q^2 K F = omega^2 * K F` |
| R6 | `F E` | `E F - alpha*K + alpha*K^2` | from `[E,F] = (K-K^{-1})/(q-q^{-1})` with `K^{-1}=K^2`, `alpha = 1/(q-q^{-1})` |

**Orientation rationale.** The PBW form `K^a E^b F^c` has K on the left,
then E, then F. So the rules must:
- Eliminate high powers: `K^3 → 1`, `E^3 → 0`, `F^3 → 0`.
- Move K to the left of E and F: `E K → ... K E` and `F K → ... K F`.
- Move E to the left of F: `F E → ... E F + ...`.

The RHS of R6 contains the "constant" `alpha*K - alpha*K^2` (no E or F
factors), which is shorter in E–F degree than the LHS. This guarantees
termination of the rewrite system on E–F degree grounds.

**Sign convention.** The standard relation is `[E, F] = EF - FE =
(K - K^{-1})/(q - q^{-1})`. Solving for `FE`:
`FE = EF - (K - K^{-1})/(q - q^{-1}) = EF - alpha*K + alpha*K^2`
(since `K^{-1} = K^2` at `ℓ = 3`). This is exactly R6.

## 4. Knuth–Bendix completion

At `ℓ = 3`, the 6-rule PBW system is **already confluent**:
`knuth_bendix_complete` terminates after 1 iteration, checking 21
critical pairs (the `i ≤ j` upper-triangular pairs of 6 rules), and
**adds zero new rules**. All critical pairs reduce to zero in `Q(omega)`.

This is the "Diamond Lemma for quantum groups" phenomenon: at a fixed
root of unity, the PBW rewrite system for `u_q(sl_n)` is confluent
without further completion. (Generically, over `Z[q, q^{-1}]`, the
system is *not* confluent — the critical pair `E K^3` gives
`(q^{-6} - 1) E`, which is nonzero in `Z[q, q^{-1}]` but zero at `ℓ = 3`
since `q^3 = 1`. This is why we work over `Q(omega)` rather than the
generic ring.)

**Statistics**:
- Initial rules: 6
- Final rules: 6
- Critical pairs checked: 21
- New rules added: 0
- Iterations: 1
- Terminated (confluent): True
- Failed pairs: 0

## 5. PBW basis verification

The 27 monomials `K^a E^b F^c` with `0 ≤ a, b, c ≤ 2` are exactly the
normal forms:

- **All 27 are in normal form**: no rule's LHS matches any PBW monomial.
- **Random reductions land in the PBW basis**: 200 random monomials of
  length 0–8, each reduced to a polynomial whose terms are all PBW
  monomials.
- **Sanity reductions** all give the expected results:
  - `K^3 → 1`, `E^3 → 0`, `F^3 → 0`
  - `E K → omega * K E`, `F K → omega^2 * K F`
  - `F E → E F - alpha*K + alpha*K^2`
  - `K F E K → alpha - alpha*K + K^2 E F` (uses all 6 rules)

This matches `dim_C u_q(sl_2) = 27`, confirming the presentation is
correct and the rewrite system is confluent.

## 6. Anick resolution: degree-0, 1, 2 generators

The Anick resolution `C_• → k → 0` is a free `u_q(sl_2)`-resolution of
the trivial module `k`. Its chain groups are spanned by "chains"
(specific overlap monomials of rule LHSs).

| Degree | Count | Description |
|--------|-------|-------------|
| 0 | 1 | the unit `1` |
| 1 | 6 | one generator per rewrite rule (the "relations") |
| 2 | 13 | one generator per critical pair (the "syzygies") |

**The 13 syzygies** (full listing):

| # | Overlap monomial M | Rules (Ri, Rj) | Positions (p1, p2) |
|---|--------------------|----------------|--------------------|
| 1 | `K^5` | (R1, R1) | (0, 2) |
| 2 | `K^4` | (R1, R1) | (0, 1) |
| 3 | `E K^3` | (R1, R4) | (1, 0) |
| 4 | `F K^3` | (R1, R5) | (1, 0) |
| 5 | `E^5` | (R2, R2) | (0, 2) |
| 6 | `E^4` | (R2, R2) | (0, 1) |
| 7 | `E^3 K` | (R2, R4) | (0, 2) |
| 8 | `F E^3` | (R2, R6) | (1, 0) |
| 9 | `F^5` | (R3, R3) | (0, 2) |
| 10 | `F^4` | (R3, R3) | (0, 1) |
| 11 | `F^3 K` | (R3, R5) | (0, 2) |
| 12 | `F^3 E` | (R3, R6) | (0, 2) |
| 13 | `F E K` | (R4, R6) | (1, 0) |

**Breakdown**:
- 6 self-overlaps of the cubic rules (R1, R2, R3 each contribute 2:
  lengths 4 and 5).
- 7 cross-rule overlaps involving one cubic and one quadratic rule, or
  two quadratic rules whose suffix–prefix matches.

Each syzygy represents an ambiguity: the overlap monomial `M` can be
reduced by applying rule `Ri` at position `p1` or rule `Rj` at position
`p2`, and the two reductions agree (in `Q(omega)` at `ℓ = 3`).

## 7. dim HH² via the bar complex on PBW normal forms

For a Hopf algebra `A` there is a canonical isomorphism
`HH^*(A, k) ≅ Ext^*_A(k, k)` (Cartan–Eilenberg). Both the bar complex
on PBW normal forms and the Anick resolution compute `Ext^*_A(k, k)`,
hence compute the same `HH^*`. The two resolutions are
homotopy-equivalent.

We use the **bar complex on PBW normal forms** for the actual
differential-rank computation, because the bar differential is
straightforward to implement (it depends only on the multiplication
table and the counit, both of which we have via the IR normal-form
reducer). The Anick differential is more subtle (it depends on the
chains and the rewriting structure in a deeper way); implementing it is
left for a future task — see §9 below.

### 7.1 Multiplication table

The 27 × 27 × 27 multiplication table `mult[k, i, j]` (coefficient of
`basis[k]` in `basis[i] * basis[j]`) is built by reducing each product
of PBW monomials via the IR `NormalFormReducer`. This is the central
validation of the IR framework: if the reducer is correct, the
multiplication table matches the one built by hand in
`verify_sl2_hh2.py`.

**Sanity checks** (all pass):
- `1 * x = x * 1 = x` ✓
- `K^3 = 1` ✓
- `E^3 = 0`, `F^3 = 0` ✓
- `K E = q^2 E K` (i.e. `E K = q^{-2} K E = omega * K E`) ✓
- `K F = q^{-2} F K` (i.e. `F K = q^2 K F = omega^2 * K F`) ✓
- `[E, F] = (K - K^{-1})/(q - q^{-1})` ✓

### 7.2 Differentials and rank

The Hochschild differentials with trivial coefficients
(`ε(K^a E^b F^c) = δ_{b,0} δ_{c,0}`):

- `d^1: C^1 → C^2`, `(d^1 f)(a, b) = ε(a) f(b) - f(a*b) + f(a) ε(b)`
  — shape `(729, 27)`, dense complex matrix.
- `d^2: C^2 → C^3`, `(d^2 g)(a, b, c) = ε(a) g(b,c) - g(a*b, c) + g(a, b*c) - g(a, b) ε(c)`
  — shape `(19683, 729)`, SciPy CSR sparse.

**Ranks** (computed via SVD for `d^1`, Gram-matrix eigenvalues for `d^2`):
- `rank(d^1) = 27`
- `rank(d^2) = 699`
- `dim ker(d^2) = 729 - 699 = 30`
- `dim im(d^1) = rank(d^1) = 27`
- **`dim HH^2 = 30 - 27 = 3`** ✓

This matches `verify_sl2_hh2.py`'s value (`rank(d^1) = 27`,
`rank(d^2) = 699`, `dim HH^2 = 3`) exactly.

### 7.3 Why the bar complex and not the Anick differential?

The bar complex on PBW normal forms is mathematically equivalent to the
Anick resolution (both compute `Ext^*_A(k, k) ≅ HH^*(A, k)` for a Hopf
algebra `A`). The choice is purely computational:

- The bar differential depends only on the multiplication table
  (`mult[k, i, j]`) and the counit (`ε`). Both are immediate from the
  IR framework.
- The Anick differential depends on the chain structure (the specific
  overlap monomials and the rule applications) in a more intricate way.
  A correct implementation requires following Anick's original 1986
  construction carefully, including the recursive definition of the
  differential on `n`-chains via "tips" and "reductions of
  ambiguities". This is a substantial implementation effort.

For `u_q(sl_2)` at `ℓ = 3`, the bar complex is small enough
(`dim C^2 = 729`, `dim C^3 = 19683`) that the rank computation takes
~1 second. The bar complex is the right tool here.

For `u_q(sl_3)` at `ℓ = 3`, the bar complex is intractable
(`dim u_q(sl_3) = 6561`, `dim C^2 = 4.3 × 10^7`). The Anick resolution
— whose chain groups are spanned by the syzygies, numbering in the
dozens, not the millions — is the right tool there. Implementing the
Anick differential is the natural next step (§9).

## 8. Test results

```
$ pytest tests/test_ir_uq_sl2.py -v
============================= 11 passed in 2.10s ==============================
```

The four required tests:

- `test_presentation_parses`: presentation has the expected generators
  `['K', 'E', 'F']` and 6 rules with the correct LHS monomials and
  RHS polynomials (including the commutator R6 with the correct
  `alpha = 1/(q - q^{-1})` coefficient).
- `test_pbw_basis_size`: PBW basis has exactly 27 normal forms; all are
  in normal form; 100 random monomials all reduce into the PBW basis.
- `test_anick_degree2_count`: Anick resolution has 1, 6, 13 generators
  at degrees 0, 1, 2 respectively.
- `test_dim_hh2_is_3`: `dim HH^2(u_q(sl_2), C) = 3` via the IR
  framework's bar complex on PBW normal forms.

Plus 7 additional tests covering: PBW index roundtrip, specific
reductions (`K^3 → 1`, `E^3 → 0`, `E K → omega*K E`, `F E → ...`),
KB completion terminates with no new rules, Anick self-overlap counts,
multiplication-table sanity checks, the HH² rank-nullity decomposition,
and a randomised confluence check.

Combined with the existing IR tests:
```
$ pytest tests/test_ir_parser.py tests/test_ir_groebner.py tests/test_ir_uq_sl2.py
============================= 55 passed in 2.29s ==============================
```

The IR framework (W0-1a + W0-1b + W1-1a-IR) is now end-to-end
validated.

## 9. Open questions and next actions

1. **Implement the Anick differential directly**. The current
   computation uses the bar complex on PBW normal forms, which is
   equivalent but does not exploit the smallness of the Anick chain
   groups (`dim C_2 = 13` vs `dim C^2 = 729`). For `u_q(sl_3)` at
   `ℓ = 3`, the bar complex is intractable, so the Anick differential
   is needed. The relevant reference is Anick's 1986 paper
   (*On the homology of associative algebras*, Trans. AMS) and
   Skryabin's survey. The degree-1 differential `d_1` is
   straightforward (sends a rule `R_i: L_i → r_i` to the polynomial
   `L_i - r_i` in `A`); the degree-2 differential `d_2` requires
   expressing each syzygy's "reduction difference" as a combination of
   relations; `d_3` is more intricate.

2. **Apply to `u_q(sl_3)` at `ℓ = 3`**. The presentation has 8
   generators (`K_1, K_2, E_1, E_2, E_{12}, F_1, F_2, F_{12}` — or
   similar, depending on convention) and ~20 relations (q-commutators
   + nilpotence + commutators). Expected PBW basis size:
   `dim u_q(sl_3) = 6561 = 3^8`. Expected `dim HH^2 = 9` (the
   conjecture: `C(4,2) + 2*3 = 6 + 6 = 12`... wait, the conjecture is
   `C(n+1, 2) + 2|Φ^+|`, for `n = 2` (sl_3): `C(3, 2) + 2*3 = 3 + 6 = 9`).
   The Anick degree-2 syzygy count should be ~50–100; the bar complex
   will be intractable.

3. **Exact rank certification**. The current rank computation uses
   floating-point SVD with a tolerance of `1e-9 * sigma_max`. For
   `u_q(sl_2)` at `ℓ = 3` this is reliable (the eigenvalue gap is
   clean). For larger algebras, exact rank via
   `fractions.Fraction` + Gaussian elimination, or modular
   certification (as in `scripts/certify_a1_exact.py`), would be safer.
   `QOmega3` already uses `Fraction` internally; only the final
   complex-number conversion and SVD step loses exactness. A future
   version could keep `QOmega3` throughout and use `sympy.Matrix.rank()`
   for exact rank.

4. **Generic vs. specialized**. The current implementation specializes
   to `ℓ = 3` from the start (via `QOmega3`). For `u_q(sl_n)` at
   arbitrary `ℓ`, one would want a generic `QLoc` class (localization of
   `Z[q, q^{-1}]` at `(q - q^{-1})`) that specializes to
   `Q(omega_ell)` for any `ℓ`. This is approach (a) in its full
   generality. The current `QOmega3` is the `ℓ = 3` specialization of
   this hypothetical `QLoc`.

5. **Anick degree-3 syzygies**. The current `anick_resolution_generators`
   has only a best-effort degree-3 enumeration. For a complete `dim HH^3`
   computation (or to verify `dim HH^2` via the Anick differential
   directly, which needs `d_3`), a proper degree-3 chain enumeration is
   needed. This is also left for a future task.

## 10. Files produced / modified

- **Created** `/home/z/my-project/hopf-decoherence/ir/qomega.py` (~300 lines):
  the `QOmega3` coefficient class and constants.
- **Created** `/home/z/my-project/hopf-decoherence/ir/uq_sl2.py` (~600 lines):
  the `u_q(sl_2)` presentation, PBW verification, KB completion runner,
  Anick degree-2 counter, multiplication-table builder, bar-complex
  differentials, rank computation, and `full_computation` orchestrator.
- **Created** `/home/z/my-project/hopf-decoherence/scripts/verify_ir_sl2.py`:
  standalone verification script (exit code 0 if `dim HH^2 = 3`).
- **Created** `/home/z/my-project/hopf-decoherence/scripts/ir_sl2_output.txt`:
  captured output (125 lines).
- **Created** `/home/z/my-project/hopf-decoherence/tests/test_ir_uq_sl2.py`
  (~340 lines): 11 tests in 5 classes.
- **Created** this note.
- **Modified** `/home/z/my-project/worklog.md`: appended the W1-1a-IR
  section.
- **No existing files modified** except the worklog. The IR framework
  modules (`ir/parser.py`, `ir/groebner.py`) are unchanged; `QOmega3`
  is a *subclass* of `QLaurent` and works with the existing
  infrastructure via Python's subclass-priority rule for reflected
  operators.

## 11. Reproducibility

```bash
$ cd /home/z/my-project/hopf-decoherence
$ python scripts/verify_ir_sl2.py
# ... prints full report, exits 0 if dim HH^2 = 3 ...
$ pytest tests/test_ir_uq_sl2.py -v
# 11 passed in ~2 seconds
```

Total wall time for the full computation (build presentation, KB
completion, PBW check, Anick count, multiplication table, d^1 + d^2
rank computation): ~1.3 seconds.
