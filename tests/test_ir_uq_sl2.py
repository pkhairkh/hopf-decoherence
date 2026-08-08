"""Tests for the u_q(sl_2) IR presentation and HH^2 computation (task W1-1a).

These tests verify that the AST/IR framework correctly computes
``dim_C HH^2(u_q(sl_2), C) = 3`` at ``ell = 3``, validating the framework
against the existing bar-complex computation in
``scripts/verify_sl2_hh2.py`` and the conjecture
``dim HH^2 = C(n+1, 2) + 2*|Phi^+| = 1 + 2`` for ``A_1``.

Tests:

  - :class:`TestPresentation` -- ``test_presentation_parses``:
    presentation has the expected generators and 6 rules; each rule has
    the expected LHS monomial and the commutator R6 has the correct
    coefficient ``(K - K^2)/(q - q^{-1})``.
  - :class:`TestPBWBasis` -- ``test_pbw_basis_size``:
    exactly 27 PBW normal forms ``K^a E^b F^c`` with ``0 <= a, b, c <= 2``;
    every PBW monomial is in normal form; random monomials all reduce
    into the PBW basis.
  - :class:`TestAnickResolution` -- ``test_anick_degree2_count``:
    the Anick resolution has 1 generator at degree 0, 6 at degree 1
    (one per rule), and 13 at degree 2 (critical pairs / syzygies).
  - :class:`TestHH2` -- ``test_dim_hh2_is_3``:
    the bar complex on PBW normal forms gives ``dim HH^2 = 3``,
    matching the conjecture.

Run::

    pytest tests/test_ir_uq_sl2.py -v
"""
import os
import sys

import pytest

# Make the ir/ package importable (it lives at the repo root, not under src/).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ir.parser import (  # noqa: E402
    Monomial, Polynomial, Term, RewriteRule, Presentation, NormalFormReducer,
)
from ir.qomega import (  # noqa: E402
    QOmega3, OMEGA, OMEGA2, OMEGA3_ONE, OMEGA3_ZERO,
    Q_MINUS_Q_INV, Q_MINUS_Q_INV_INV,
)
from ir.uq_sl2 import (  # noqa: E402
    ELL, DIM, ALPHA,
    build_uq_sl2_presentation,
    pbw_basis, pbw_index, pbw_from_index,
    verify_pbw_basis,
    run_kb_completion,
    anick_degree0_count, anick_degree1_count, anick_degree2_count,
    anick_degree2_listing,
    compute_hh2, build_multiplication_table, sanity_check_multiplication,
)
from ir.groebner import (  # noqa: E402
    knuth_bendix_complete, anick_resolution_generators, check_confluence,
)


# ----------------------------------------------------------------------------
# TestPresentation
# ----------------------------------------------------------------------------


class TestPresentation:
    """Tests for the u_q(sl_2) presentation (Step 1 of W1-1a)."""

    def test_presentation_parses(self):
        """The presentation has the expected generators and 6 rules.

        Verifies:
          - Generators are ['K', 'E', 'F'] (indices 0, 1, 2).
          - 6 rules, one per defining relation.
          - Each rule's LHS is the expected monomial:
              R1: K^3, R2: E^3, R3: F^3, R4: E K, R5: F K, R6: F E.
          - R1 RHS is the unit monomial.
          - R2, R3 RHS is the zero polynomial.
          - R4 RHS is ``omega * K E``  (q^{-2} = omega at ell = 3).
          - R5 RHS is ``omega^2 * K F`` (q^2 = omega^2 at ell = 3).
          - R6 RHS is ``E F - alpha*K + alpha*K^2`` with
            ``alpha = 1/(q - q^{-1})``  (the standard commutator).
          - All RHS coefficients are QOmega3 (specialized to ell = 3).
        """
        pres = build_uq_sl2_presentation()

        # Generators.
        assert pres.generators == ["K", "E", "F"], \
            f"Expected ['K', 'E', 'F'], got {pres.generators}"

        # 6 rules.
        assert len(pres.rules) == 6, \
            f"Expected 6 rules, got {len(pres.rules)}"

        # Each rule's LHS.
        expected_lhs = [
            Monomial((0, 0, 0)),  # R1: K^3
            Monomial((1, 1, 1)),  # R2: E^3
            Monomial((2, 2, 2)),  # R3: F^3
            Monomial((1, 0)),     # R4: E K
            Monomial((2, 0)),     # R5: F K
            Monomial((2, 1)),     # R6: F E
        ]
        for i, (rule, exp) in enumerate(zip(pres.rules, expected_lhs)):
            assert rule.lhs == exp, \
                f"R{i+1}: expected LHS {exp}, got {rule.lhs}"

        # R1: K^3 -> 1 (empty monomial with coefficient 1).
        r1 = pres.rules[0]
        assert len(r1.rhs.terms) == 1, f"R1 RHS should have 1 term, got {len(r1.rhs.terms)}"
        assert r1.rhs.terms[0].monomial == Monomial(()), \
            f"R1 RHS monomial should be empty (1), got {r1.rhs.terms[0].monomial}"
        assert r1.rhs.terms[0].coeff == OMEGA3_ONE, \
            f"R1 RHS coeff should be OMEGA3_ONE, got {r1.rhs.terms[0].coeff}"

        # R2, R3: nilpotence -> zero.
        assert pres.rules[1].rhs == Polynomial.zero(), \
            f"R2 (E^3 -> 0) RHS should be zero, got {pres.rules[1].rhs}"
        assert pres.rules[2].rhs == Polynomial.zero(), \
            f"R3 (F^3 -> 0) RHS should be zero, got {pres.rules[2].rhs}"

        # R4: E K -> q^{-2} K E = omega * K E.
        r4 = pres.rules[3]
        assert len(r4.rhs.terms) == 1, f"R4 RHS should have 1 term"
        assert r4.rhs.terms[0].monomial == Monomial((0, 1)), \
            f"R4 RHS monomial should be (0, 1) = K E, got {r4.rhs.terms[0].monomial}"
        assert r4.rhs.terms[0].coeff == OMEGA, \
            f"R4 RHS coeff should be omega (= q^{{-2}}), got {r4.rhs.terms[0].coeff}"

        # R5: F K -> q^2 K F = omega^2 * K F.
        r5 = pres.rules[4]
        assert len(r5.rhs.terms) == 1, f"R5 RHS should have 1 term"
        assert r5.rhs.terms[0].monomial == Monomial((0, 2)), \
            f"R5 RHS monomial should be (0, 2) = K F, got {r5.rhs.terms[0].monomial}"
        assert r5.rhs.terms[0].coeff == OMEGA2, \
            f"R5 RHS coeff should be omega^2 (= q^2), got {r5.rhs.terms[0].coeff}"

        # R6: F E -> E F - alpha*K + alpha*K^2 (alpha = 1/(q - q^{-1})).
        r6 = pres.rules[5]
        assert len(r6.rhs.terms) == 3, \
            f"R6 RHS should have 3 terms (E F + alpha*K + alpha*K^2), got {len(r6.rhs.terms)}"
        # Collect terms by monomial.
        r6_dict = {t.monomial: t.coeff for t in r6.rhs.terms}
        assert Monomial((1, 2)) in r6_dict, "R6 RHS missing E F term"
        assert Monomial((0,)) in r6_dict, "R6 RHS missing K term"
        assert Monomial((0, 0)) in r6_dict, "R6 RHS missing K^2 term"
        # Coefficients.
        assert r6_dict[Monomial((1, 2))] == OMEGA3_ONE, \
            f"E F coeff should be 1, got {r6_dict[Monomial((1, 2))]}"
        assert r6_dict[Monomial((0,))] == -ALPHA, \
            f"K coeff should be -alpha, got {r6_dict[Monomial((0,))]}"
        assert r6_dict[Monomial((0, 0))] == ALPHA, \
            f"K^2 coeff should be +alpha, got {r6_dict[Monomial((0, 0))]}"
        # Verify alpha = 1/(q - q^{-1}).
        assert ALPHA == Q_MINUS_Q_INV_INV, \
            f"ALPHA should be 1/(q - q^{{-1}}), got {ALPHA}"
        # Verify alpha * (q - q^{-1}) = 1.
        assert (ALPHA * Q_MINUS_Q_INV).is_one(), \
            f"alpha * (q - q^{{-1}}) should be 1, got {ALPHA * Q_MINUS_Q_INV}"

        # All RHS coefficients are QOmega3.
        for i, rule in enumerate(pres.rules):
            for t in rule.rhs.terms:
                assert isinstance(t.coeff, QOmega3), \
                    f"R{i+1} RHS term coefficient should be QOmega3, " \
                    f"got {type(t.coeff).__name__}"


# ----------------------------------------------------------------------------
# TestPBWBasis
# ----------------------------------------------------------------------------


class TestPBWBasis:
    """Tests for the PBW basis property (Step 2 of W1-1a)."""

    def test_pbw_basis_size(self):
        """The PBW basis has 27 normal forms K^a E^b F^c.

        Verifies:
          - ``pbw_basis()`` returns exactly 27 monomials.
          - All 27 monomials are in normal form (no rule matches).
          - 100 random monomials all reduce to polynomials whose terms
            are in the PBW basis.
          - The PBW monomials are exactly ``{K^a E^b F^c : 0 <= a,b,c <= 2}``.
        """
        pres = build_uq_sl2_presentation()
        basis = pbw_basis()

        # 27 elements.
        assert len(basis) == 27, f"PBW basis size: expected 27, got {len(basis)}"

        # Each is K^a E^b F^c with 0 <= a, b, c <= 2.
        expected = set()
        for a in range(ELL):
            for b in range(ELL):
                for c in range(ELL):
                    expected.add(Monomial((0,) * a + (1,) * b + (2,) * c))
        assert set(basis) == expected, "PBW basis doesn't match K^a E^b F^c"

        # Each is in normal form.
        reducer = NormalFormReducer(pres)
        for m in basis:
            assert reducer.find_match(m) is None, \
                f"PBW monomial {m} is not in normal form (a rule matches)"

        # Random reductions land in the PBW basis.
        check = verify_pbw_basis(pres, n_random=100, seed=42)
        assert check["pbw_all_normal"], \
            "Not all PBW monomials are in normal form"
        assert check["random_reductions_in_pbw"], \
            "Some random reductions produced monomials outside the PBW basis"

    def test_pbw_index_roundtrip(self):
        """``pbw_index`` and ``pbw_from_index`` are inverse."""
        for a in range(ELL):
            for b in range(ELL):
                for c in range(ELL):
                    i = pbw_index(a, b, c)
                    assert 0 <= i < DIM
                    a2, b2, c2 = pbw_from_index(i)
                    assert (a, b, c) == (a2, b2, c2), \
                        f"Roundtrip failed: ({a},{b},{c}) -> {i} -> ({a2},{b2},{c2})"

    def test_specific_reductions(self):
        """Spot-check that specific products reduce to expected PBW forms."""
        pres = build_uq_sl2_presentation()
        reducer = NormalFormReducer(pres)

        # K^3 -> 1.
        nf = reducer.normal_form(Monomial((0, 0, 0)))
        assert nf == Polynomial([Term(OMEGA3_ONE, Monomial(()))]), \
            f"K^3 -> {nf}, expected 1"

        # E^3 -> 0.
        nf = reducer.normal_form(Monomial((1, 1, 1)))
        assert nf.is_zero(), f"E^3 -> {nf}, expected 0"

        # F^3 -> 0.
        nf = reducer.normal_form(Monomial((2, 2, 2)))
        assert nf.is_zero(), f"F^3 -> {nf}, expected 0"

        # E K -> omega * K E.
        nf = reducer.normal_form(Monomial((1, 0)))
        assert nf == Polynomial([Term(OMEGA, Monomial((0, 1)))]), \
            f"E K -> {nf}, expected omega * K E"

        # F K -> omega^2 * K F.
        nf = reducer.normal_form(Monomial((2, 0)))
        assert nf == Polynomial([Term(OMEGA2, Monomial((0, 2)))]), \
            f"F K -> {nf}, expected omega^2 * K F"

        # F E -> E F - alpha*K + alpha*K^2.
        nf = reducer.normal_form(Monomial((2, 1)))
        expected = Polynomial([
            Term(OMEGA3_ONE, Monomial((1, 2))),
            Term(-ALPHA, Monomial((0,))),
            Term(ALPHA, Monomial((0, 0))),
        ])
        assert nf == expected, f"F E -> {nf}, expected {expected}"


# ----------------------------------------------------------------------------
# TestKnuthBendix
# ----------------------------------------------------------------------------


class TestKnuthBendix:
    """Tests for Knuth-Bendix completion at ell = 3."""

    def test_kb_completion_terminates_with_no_new_rules(self):
        """KB completion terminates without adding any new rules.

        At ell = 3 the 6-rule PBW system is already confluent (all
        critical pairs reduce to zero in Q(omega)).
        """
        pres = build_uq_sl2_presentation()
        completed, stats = run_kb_completion(pres, max_iterations=50)
        assert stats.terminated, \
            f"KB did not terminate: stats = {stats}"
        assert stats.new_rules_added == 0, \
            f"KB added {stats.new_rules_added} new rules (expected 0)"
        assert stats.final_rules == 6, \
            f"Final rule count: {stats.final_rules} (expected 6)"
        assert len(stats.failed_pairs) == 0, \
            f"KB failed pairs: {stats.failed_pairs}"


# ----------------------------------------------------------------------------
# TestAnickResolution
# ----------------------------------------------------------------------------


class TestAnickResolution:
    """Tests for the Anick resolution generator counts (Step 4 of W1-1a)."""

    def test_anick_degree2_count(self):
        """Anick resolution: degree 0 = 1, degree 1 = 6, degree 2 = 13.

        The degree-2 count is the number of critical pairs (syzygies).
        Breakdown of the 13 syzygies:

          - 6 self-overlaps of the cubic rules (R1, R2, R3 each give 2):
              K^4, K^5, E^4, E^5, F^4, F^5.
          - 7 cross-rule overlaps:
              R1<->R4 (E K^3), R1<->R5 (F K^3),
              R2<->R4 (E^3 K), R2<->R6 (F E^3),
              R3<->R5 (F^3 K), R3<->R6 (F^3 E),
              R4<->R6 (F E K).
        """
        pres = build_uq_sl2_presentation()

        n0 = anick_degree0_count(pres)
        n1 = anick_degree1_count(pres)
        n2 = anick_degree2_count(pres)

        assert n0 == 1, f"Anick degree 0: expected 1, got {n0}"
        assert n1 == 6, f"Anick degree 1: expected 6, got {n1}"
        assert n2 == 13, f"Anick degree 2: expected 13, got {n2}"

        # Verify by listing the syzygies explicitly.
        listing = anick_degree2_listing(pres)
        assert len(listing) == 13, \
            f"Syzygy listing length: {len(listing)} (expected 13)"

        # All overlap monomials should be distinct (each syzygy is
        # identified by its overlap monomial + rule pair).
        # (Different rule pairs can share an overlap monomial, so we
        # check (M, rule_pair) tuples instead.)
        seen = set()
        for M, (i, j), (p1, p2) in listing:
            key = (M, i, j, p1, p2)
            assert key not in seen, f"Duplicate syzygy: {key}"
            seen.add(key)

    def test_anick_self_overlap_counts(self):
        """The cubic rules R1, R2, R3 each have 2 self-overlaps (K^4, K^5).

        Verifies that the syzygy listing contains the expected
        self-overlap monomials K^4 = (0,0,0,0), K^5 = (0,0,0,0,0),
        E^4, E^5, F^4, F^5.
        """
        pres = build_uq_sl2_presentation()
        listing = anick_degree2_listing(pres)
        overlap_monomials = {M for (M, _, _) in listing}

        expected_self_overlaps = {
            Monomial((0, 0, 0, 0)),  # K^4
            Monomial((0, 0, 0, 0, 0)),  # K^5
            Monomial((1, 1, 1, 1)),  # E^4
            Monomial((1, 1, 1, 1, 1)),  # E^5
            Monomial((2, 2, 2, 2)),  # F^4
            Monomial((2, 2, 2, 2, 2)),  # F^5
        }
        assert expected_self_overlaps.issubset(overlap_monomials), \
            f"Missing self-overlap monomials: " \
            f"{expected_self_overlaps - overlap_monomials}"


# ----------------------------------------------------------------------------
# TestHH2
# ----------------------------------------------------------------------------


class TestHH2:
    """Tests for the dim HH^2 computation (Step 5 of W1-1a)."""

    def test_dim_hh2_is_3(self):
        """``dim HH^2(u_q(sl_2), C) = 3`` via the IR framework.

        This is the central validation of W1-1a: the IR framework's
        normal-form reducer + bar complex reproduces the value obtained
        by the direct bar-complex computation in
        ``scripts/verify_sl2_hh2.py``, matching the conjecture
        ``dim HH^2 = C(n+1, 2) + 2*|Phi^+| = 1 + 2 = 3`` for ``A_1``.
        """
        pres = build_uq_sl2_presentation()
        hh2 = compute_hh2(pres, verbose=False)

        # Sanity: dimensions of the bar complex.
        assert hh2["dim_uq_sl2"] == 27, \
            f"dim u_q(sl_2): expected 27, got {hh2['dim_uq_sl2']}"
        assert hh2["dim_c1"] == 27, \
            f"dim C^1: expected 27, got {hh2['dim_c1']}"
        assert hh2["dim_c2"] == 27 * 27, \
            f"dim C^2: expected 729, got {hh2['dim_c2']}"
        assert hh2["dim_c3"] == 27 ** 3, \
            f"dim C^3: expected 19683, got {hh2['dim_c3']}"

        # The decisive check.
        assert hh2["dim_hh2"] == 3, \
            f"dim HH^2: expected 3 (conjecture), got {hh2['dim_hh2']}"

    def test_multiplication_table_sanity(self):
        """The IR-reduced multiplication table satisfies the relations.

        Verifies the multiplication table (built via the IR normal-form
        reducer) satisfies:
          - Unit: ``1 * x = x * 1 = x``
          - ``K^3 = 1``
          - ``E^3 = F^3 = 0``
          - ``K E = q^2 E K`` (i.e. ``E K = q^{-2} K E``)
          - ``K F = q^{-2} F K`` (i.e. ``F K = q^2 K F``)
          - ``[E, F] = (K - K^{-1})/(q - q^{-1})``
        """
        pres = build_uq_sl2_presentation()
        mult = build_multiplication_table(pres)
        checks = sanity_check_multiplication(mult)
        for name, ok in checks.items():
            assert ok, f"Multiplication sanity check failed: {name}"

    def test_hh2_decomposition(self):
        """``dim HH^2 = dim ker(d^2) - dim im(d^1)``.

        Verifies the rank-nullity decomposition:
          ``dim ker(d^2) = dim C^2 - rank(d^2) = 729 - rank(d^2)``
          ``dim im(d^1) = rank(d^1)``
          ``dim HH^2 = dim ker(d^2) - dim im(d^1)``
        """
        pres = build_uq_sl2_presentation()
        hh2 = compute_hh2(pres, verbose=False)

        expected_ker = hh2["dim_c2"] - hh2["rank_d2"]
        assert hh2["dim_ker_d2"] == expected_ker, \
            f"dim ker(d^2) = {hh2['dim_ker_d2']}, expected {expected_ker}"

        expected_hh2 = hh2["dim_ker_d2"] - hh2["rank_d1"]
        assert hh2["dim_hh2"] == expected_hh2, \
            f"dim HH^2 = {hh2['dim_hh2']}, expected {expected_hh2}"


# ----------------------------------------------------------------------------
# TestConfluence (randomised)
# ----------------------------------------------------------------------------


class TestConfluence:
    """Randomised confluence check on the completed rewrite system."""

    def test_random_confluence(self):
        """Randomised confluence check: reducing random monomials twice
        gives the same normal form."""
        pres = build_uq_sl2_presentation()
        # The system is already confluent (KB added no rules); a
        # randomised check should pass.
        assert check_confluence(pres, n_tests=100, seed=123), \
            "Randomised confluence check failed"
