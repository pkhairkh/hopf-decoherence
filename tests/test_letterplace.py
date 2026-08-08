"""Tests for the letterplace encoder module (task LP-0a).

Verifies the letterplace correspondence (Cohen 1987; La Scala
arXiv:1605.06944) implementation in :mod:`ir.letterplace`:

  - :class:`LetterplaceEncoder.encode_word` — NC word -> commutative monomial
  - :class:`LetterplaceEncoder.encode_polynomial` — NC poly -> commutative poly
  - :class:`LetterplaceEncoder.encode_relation` — LHS-RHS at a position
  - :class:`LetterplaceEncoder.encode_presentation` — all relations, all positions
  - :class:`LetterplaceEncoder.compute_groebner_basis` — quantum plane basis
  - :class:`LetterplaceEncoder.decode_monomial` — commutative monomial -> (NC word, start)
  - :class:`LetterplaceEncoder.decode_polynomial` — commutative poly -> dict of NC polys

Test algebra: the *quantum plane* ``x * y = q * y * x`` (generators
x=0, y=1; one rewrite rule).  This is the canonical example from La
Scala–Levandovskyy.

Run::

    pytest tests/test_letterplace.py -v
"""
import os
import sys

import pytest
import sympy as sp

# Make the ir/ package importable (it lives at the repo root, not under src/).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ir.parser import (  # noqa: E402
    Monomial,
    Polynomial,
    RewriteRule,
    Presentation,
    QLaurent,
    Term,
    qpow,
    qint,
    ONE,
    Q,
)
from ir.letterplace import LetterplaceEncoder  # noqa: E402


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _quantum_plane_presentation() -> Presentation:
    """Build the quantum plane presentation: generators x, y; rule (x,y) -> q*(y,x)."""
    rule = RewriteRule(
        Monomial((0, 1)),
        Polynomial([Term(qpow(1), Monomial((1, 0)))]),
    )
    return Presentation(["x", "y"], [rule])


def _rule_xy() -> RewriteRule:
    """The single quantum-plane rule: (x, y) -> q * (y, x)."""
    return RewriteRule(
        Monomial((0, 1)),
        Polynomial([Term(qpow(1), Monomial((1, 0)))]),
    )


# ----------------------------------------------------------------------------
# 1. encode_word
# ----------------------------------------------------------------------------


class TestEncodeWord:
    """Tests for LetterplaceEncoder.encode_word."""

    def test_encode_word_basic(self):
        """encode_word((0, 1), 0) = x0_s0 * x1_s1 (as a sympy Mul)."""
        enc = LetterplaceEncoder(["x", "y"], max_degree=4)
        result = enc.encode_word((0, 1), 0)
        # Expected: x0_s0 * x1_s1
        x0_s0 = sp.Symbol("x0_s0")
        x1_s1 = sp.Symbol("x1_s1")
        expected = x0_s0 * x1_s1
        assert sp.simplify(result - expected) == 0, (
            f"encode_word((0,1), 0) = {result}, expected {expected}"
        )

    def test_encode_word_three_factors(self):
        """encode_word((0, 1, 0), 2) = x0_s2 * x1_s3 * x0_s4."""
        enc = LetterplaceEncoder(["x", "y"], max_degree=4)
        result = enc.encode_word((0, 1, 0), 2)
        x0_s2 = sp.Symbol("x0_s2")
        x1_s3 = sp.Symbol("x1_s3")
        x0_s4 = sp.Symbol("x0_s4")
        expected = x0_s2 * x1_s3 * x0_s4
        assert sp.simplify(result - expected) == 0, (
            f"encode_word((0,1,0), 2) = {result}, expected {expected}"
        )

    def test_encode_word_yx_at_zero(self):
        """encode_word((1, 0), 0) = x1_s0 * x0_s1."""
        enc = LetterplaceEncoder(["x", "y"], max_degree=4)
        result = enc.encode_word((1, 0), 0)
        x1_s0 = sp.Symbol("x1_s0")
        x0_s1 = sp.Symbol("x0_s1")
        expected = x1_s0 * x0_s1
        assert sp.simplify(result - expected) == 0, (
            f"encode_word((1,0), 0) = {result}, expected {expected}"
        )

    def test_encode_word_empty_is_one(self):
        """encode_word((), 0) = 1 (the empty monomial)."""
        enc = LetterplaceEncoder(["x", "y"], max_degree=4)
        result = enc.encode_word((), 0)
        assert result == 1, f"encode_word((), 0) = {result}, expected 1"
        assert isinstance(result, sp.Integer), (
            f"encode_word((), 0) should be sympy.Integer(1), got {type(result).__name__}"
        )

    def test_encode_word_out_of_range_raises(self):
        """encode_word raises ValueError if the word exceeds max_degree."""
        enc = LetterplaceEncoder(["x", "y"], max_degree=2)
        # Word of length 3 starting at 0 needs positions 0,1,2 — OK.
        enc.encode_word((0, 1, 0), 0)
        # Word of length 3 starting at 1 needs positions 1,2,3 — exceeds max_degree=2.
        with pytest.raises(ValueError, match="exceeding"):
            enc.encode_word((0, 1, 0), 1)

    def test_encode_word_accepts_monomial(self):
        """encode_word also accepts a Monomial instance (not just a tuple)."""
        enc = LetterplaceEncoder(["x", "y"], max_degree=4)
        result = enc.encode_word(Monomial((0, 1)), 0)
        x0_s0 = sp.Symbol("x0_s0")
        x1_s1 = sp.Symbol("x1_s1")
        expected = x0_s0 * x1_s1
        assert sp.simplify(result - expected) == 0


# ----------------------------------------------------------------------------
# 2. encode_polynomial
# ----------------------------------------------------------------------------


class TestEncodePolynomial:
    """Tests for LetterplaceEncoder.encode_polynomial."""

    def test_encode_polynomial_single_term(self):
        """encode_polynomial of a single-term polynomial: q*(x*y) at s=0."""
        enc = LetterplaceEncoder(["x", "y"], max_degree=4)
        # Polynomial: q * (x, y)
        poly = Polynomial([Term(qpow(1), Monomial((0, 1)))])
        result = enc.encode_polynomial(poly, start=0)
        expected = enc.q_sym * sp.Symbol("x0_s0") * sp.Symbol("x1_s1")
        assert sp.simplify(result - expected) == 0, (
            f"encode_polynomial(q*xy, 0) = {result}, expected {expected}"
        )

    def test_encode_polynomial_two_terms(self):
        """encode_polynomial of (x*y - q*y*x) at s=0.

        This is the quantum plane relation; the encoded form is
        ``x0_s0 * x1_s1 - q * x1_s0 * x0_s1``.
        """
        enc = LetterplaceEncoder(["x", "y"], max_degree=4)
        poly = Polynomial([
            Term(ONE, Monomial((0, 1))),
            Term(-qpow(1), Monomial((1, 0))),
        ])
        result = enc.encode_polynomial(poly, start=0)
        x0_s0, x1_s0, x0_s1, x1_s1 = sp.symbols("x0_s0 x1_s0 x0_s1 x1_s1")
        expected = x0_s0 * x1_s1 - enc.q_sym * x1_s0 * x0_s1
        assert sp.simplify(result - expected) == 0, (
            f"encode_polynomial(xy - q*yx, 0) = {result}, expected {expected}"
        )

    def test_encode_polynomial_at_nonzero_start(self):
        """encode_polynomial at start=1 shifts all positions by 1."""
        enc = LetterplaceEncoder(["x", "y"], max_degree=4)
        poly = Polynomial([Term(ONE, Monomial((0, 1)))])
        result = enc.encode_polynomial(poly, start=1)
        x0_s1 = sp.Symbol("x0_s1")
        x1_s2 = sp.Symbol("x1_s2")
        expected = x0_s1 * x1_s2
        assert sp.simplify(result - expected) == 0, (
            f"encode_polynomial(xy, 1) = {result}, expected {expected}"
        )

    def test_encode_polynomial_zero(self):
        """The zero polynomial encodes to sympy 0."""
        enc = LetterplaceEncoder(["x", "y"], max_degree=4)
        result = enc.encode_polynomial(Polynomial.zero(), start=0)
        assert result == 0

    def test_encode_polynomial_integer_coefficient(self):
        """Integer coefficients are preserved: 3*(x) at s=0 -> 3*x0_s0."""
        enc = LetterplaceEncoder(["x", "y"], max_degree=4)
        poly = Polynomial([Term(qint(3), Monomial((0,)))])
        result = enc.encode_polynomial(poly, start=0)
        expected = sp.Integer(3) * sp.Symbol("x0_s0")
        assert sp.simplify(result - expected) == 0


# ----------------------------------------------------------------------------
# 3. encode_relation
# ----------------------------------------------------------------------------


class TestEncodeRelation:
    """Tests for LetterplaceEncoder.encode_relation."""

    def test_encode_relation_at_zero(self):
        """encode_relation(rule_xy, 0) = x0_s0*x1_s1 - q*x1_s0*x0_s1."""
        enc = LetterplaceEncoder(["x", "y"], max_degree=4)
        rule = _rule_xy()
        result = enc.encode_relation(rule, start=0)
        x0_s0, x1_s0, x0_s1, x1_s1 = sp.symbols("x0_s0 x1_s0 x0_s1 x1_s1")
        expected = x0_s0 * x1_s1 - enc.q_sym * x1_s0 * x0_s1
        assert sp.simplify(result - expected) == 0, (
            f"encode_relation(rule_xy, 0) = {result}, expected {expected}"
        )

    def test_encode_relation_at_one(self):
        """encode_relation(rule_xy, 1) = x0_s1*x1_s2 - q*x1_s1*x0_s2."""
        enc = LetterplaceEncoder(["x", "y"], max_degree=4)
        rule = _rule_xy()
        result = enc.encode_relation(rule, start=1)
        x0_s1, x1_s1, x0_s2, x1_s2 = sp.symbols("x0_s1 x1_s1 x0_s2 x1_s2")
        expected = x0_s1 * x1_s2 - enc.q_sym * x1_s1 * x0_s2
        assert sp.simplify(result - expected) == 0, (
            f"encode_relation(rule_xy, 1) = {result}, expected {expected}"
        )

    def test_encode_relation_is_difference(self):
        """encode_relation(rule, s) = encode_word(lhs, s) - encode_polynomial(rhs, s)."""
        enc = LetterplaceEncoder(["x", "y"], max_degree=4)
        rule = _rule_xy()
        result = enc.encode_relation(rule, start=0)
        lhs_enc = enc.encode_word(rule.lhs, 0)
        rhs_enc = enc.encode_polynomial(rule.rhs, 0)
        assert sp.simplify(result - (lhs_enc - rhs_enc)) == 0


# ----------------------------------------------------------------------------
# 4. encode_presentation
# ----------------------------------------------------------------------------


class TestEncodePresentation:
    """Tests for LetterplaceEncoder.encode_presentation."""

    def test_encode_presentation_quantum_plane(self):
        """For max_degree=3, the quantum plane gives generators at s=0,1,2."""
        enc = LetterplaceEncoder(["x", "y"], max_degree=3)
        pres = _quantum_plane_presentation()
        gens = enc.encode_presentation(pres)
        # LHS length is 2, so s ranges over 0 .. max_degree - 2 = 1. Wait:
        # the spec says s in 0 .. max_degree - LHS_len, so for max_degree=3
        # and LHS_len=2: s in 0, 1.  (s=2 would need positions 2,3 — but
        # max_degree=3 allows position 3, so s=2 IS valid: 0..max_degree-LHS_len = 0..1.
        # Hmm, let me re-check: max_start = max_degree - lhs_len = 3 - 2 = 1.
        # So s in {0, 1}.  That's 2 generators.
        # Actually wait: s=2 would put the word at positions 2,3, both <= 3.
        # So s=2 should be valid.  Let me re-read the spec.
        # Spec: "positions 0, 1, ..., max_degree - max_LHS_length"
        # For max_degree=3, LHS_len=2: max_degree - LHS_len = 1.
        # So s in {0, 1}.  But s=2 also fits (positions 2,3 <= 3).
        # The spec is conservative: it uses max_degree - LHS_len as the max s.
        # We follow the spec.
        assert len(gens) == 2, (
            f"Expected 2 generators (s=0,1) for quantum plane at max_degree=3, "
            f"got {len(gens)}: {gens}"
        )

    def test_encode_presentation_at_various_max_degree(self):
        """The number of generators scales as (max_degree - LHS_len + 1)."""
        rule = _rule_xy()
        pres = Presentation(["x", "y"], [rule])
        for max_deg in [2, 3, 4, 5]:
            enc = LetterplaceEncoder(["x", "y"], max_degree=max_deg)
            gens = enc.encode_presentation(pres)
            expected_count = max_deg - 2 + 1  # s in 0..max_deg-2 inclusive
            assert len(gens) == expected_count, (
                f"max_degree={max_deg}: expected {expected_count} generators, "
                f"got {len(gens)}"
            )

    def test_encode_presentation_multiple_rules(self):
        """A presentation with two rules produces generators for each."""
        # Two rules: (x,y) -> q*(y,x) and (x,x) -> 0 (nilpotency, say).
        rule1 = RewriteRule(
            Monomial((0, 1)),
            Polynomial([Term(qpow(1), Monomial((1, 0)))]),
        )
        rule2 = RewriteRule(
            Monomial((0, 0)),
            Polynomial.zero(),  # x*x = 0
        )
        pres = Presentation(["x", "y"], [rule1, rule2])
        enc = LetterplaceEncoder(["x", "y"], max_degree=3)
        gens = enc.encode_presentation(pres)
        # rule1 (LHS len 2): s in 0..1 -> 2 generators
        # rule2 (LHS len 2): s in 0..1 -> 2 generators
        # Total: 4
        assert len(gens) == 4, (
            f"Expected 4 generators (2 rules x 2 positions), got {len(gens)}"
        )

    def test_encode_presentation_lhs_too_long(self):
        """A rule whose LHS is longer than max_degree+1 produces no generators."""
        rule = RewriteRule(
            Monomial((0, 1, 0, 1, 0)),  # length 5
            Polynomial([Term(ONE, Monomial((1,)))]),
        )
        pres = Presentation(["x", "y"], [rule])
        enc = LetterplaceEncoder(["x", "y"], max_degree=3)
        gens = enc.encode_presentation(pres)
        assert gens == [], (
            f"Rule with LHS length 5 > max_degree+1=4 should produce no "
            f"generators, got {gens}"
        )


# ----------------------------------------------------------------------------
# 5. compute_groebner_basis (quantum plane)
# ----------------------------------------------------------------------------


class TestComputeGroebnerBasis:
    """Tests for LetterplaceEncoder.compute_groebner_basis on the quantum plane."""

    def test_compute_groebner_basis_nonempty(self):
        """The Groebner basis of the quantum plane letterplace ideal is non-empty."""
        pres = _quantum_plane_presentation()
        enc = LetterplaceEncoder(["x", "y"], max_degree=3)
        gb = enc.compute_groebner_basis(pres, order="grevlex")
        assert len(gb) >= 1, f"Groebner basis should be non-empty, got {gb}"

    def test_compute_groebner_basis_reduces_encoded_relations(self):
        """Each encoded relation reduces to 0 modulo the Groebner basis.

        This is the defining property of a Groebner basis: every element
        of the ideal reduces to 0.  We check the encoded relations at
        every valid starting position.
        """
        pres = _quantum_plane_presentation()
        enc = LetterplaceEncoder(["x", "y"], max_degree=3)
        order = "grevlex"
        gb = enc.compute_groebner_basis(pres, order=order)
        all_vars = enc.all_variables()  # x-vars only (q stays in coeffs)
        rule = pres.rules[0]
        lhs_len = len(rule.lhs)
        for s in range(0, enc.max_degree - lhs_len + 1):
            rel = enc.encode_relation(rule, s)
            # sp.reduced returns (quotient_list, remainder).
            remainder = sp.reduced(rel, gb, *all_vars, order=order)[1]
            assert sp.expand(remainder) == 0, (
                f"Encoded relation at s={s} should reduce to 0 mod the "
                f"Groebner basis, got remainder {remainder}"
            )

    def test_compute_groebner_basis_lex_order(self):
        """With lex order (q included last), leading terms are NC LHS monomials.

        For the quantum plane, the Groebner basis with ``order='lex'``
        and ``include_q=True`` should have leading monomials containing
        ``x0_*`` variables (the NC LHS orientation: ``x`` before ``y``),
        not just ``x1_*`` and ``q`` (the NC RHS scaled by q).
        """
        pres = _quantum_plane_presentation()
        enc = LetterplaceEncoder(["x", "y"], max_degree=2)
        gb = enc.compute_groebner_basis(pres, order="lex", include_q=True)
        assert len(gb) >= 1
        all_vars_with_q = enc.full_variable_list()
        for p in gb:
            if p == 0:
                continue
            # sp.LM returns the leading monomial (as a sympy expression)
            # under the specified order.  (sp.Poly does not accept the
            # 'order' keyword, so we use the global sp.LM instead.)
            lm = sp.LM(p, *all_vars_with_q, order="lex")
            lm_names = {s.name for s in lm.free_symbols}
            has_x0 = any(n.startswith("x0_") for n in lm_names)
            assert has_x0, (
                f"With lex+include_q, leading monomial {lm} of basis "
                f"element {p} should contain an x0_* variable "
                f"(NC LHS orientation: x before y)"
            )

    def test_compute_groebner_basis_no_relations(self):
        """A presentation with no relations has an empty letterplace ideal."""
        pres = Presentation(["x", "y"], [])
        enc = LetterplaceEncoder(["x", "y"], max_degree=3)
        gb = enc.compute_groebner_basis(pres)
        assert gb == []

    def test_compute_groebner_basis_bigger_max_degree(self):
        """A larger max_degree gives a larger Groebner basis (more positions)."""
        pres = _quantum_plane_presentation()
        enc_small = LetterplaceEncoder(["x", "y"], max_degree=2)
        enc_large = LetterplaceEncoder(["x", "y"], max_degree=4)
        gb_small = enc_small.compute_groebner_basis(pres, order="grevlex")
        gb_large = enc_large.compute_groebner_basis(pres, order="grevlex")
        assert len(gb_large) >= len(gb_small), (
            f"Larger max_degree should give >= Groebner basis size, "
            f"got {len(gb_large)} < {len(gb_small)}"
        )


# ----------------------------------------------------------------------------
# 6. decode_monomial
# ----------------------------------------------------------------------------


class TestDecodeMonomial:
    """Tests for LetterplaceEncoder.decode_monomial."""

    def test_decode_monomial_basic(self):
        """x0_s2 * x1_s3 * x0_s4 -> ((0, 1, 0), 2)."""
        enc = LetterplaceEncoder(["x", "y"], max_degree=4)
        mono = enc.encode_word((0, 1, 0), 2)
        word, start = enc.decode_monomial(mono)
        assert word == (0, 1, 0), f"word = {word}, expected (0, 1, 0)"
        assert start == 2, f"start = {start}, expected 2"

    def test_decode_monomial_single_symbol(self):
        """A single Symbol x1_s3 -> ((1,), 3)."""
        enc = LetterplaceEncoder(["x", "y"], max_degree=4)
        mono = sp.Symbol("x1_s3")
        word, start = enc.decode_monomial(mono)
        assert word == (1,)
        assert start == 3

    def test_decode_monomial_roundtrip(self):
        """encode_word then decode_monomial is a left inverse (for valid words)."""
        enc = LetterplaceEncoder(["x", "y"], max_degree=5)
        test_cases = [
            ((0,), 0),
            ((1,), 3),
            ((0, 1), 0),
            ((1, 0), 1),
            ((0, 1, 0), 2),
            ((1, 0, 1, 0), 0),
            ((0, 0, 1, 1), 1),
        ]
        for word, start in test_cases:
            mono = enc.encode_word(word, start)
            decoded_word, decoded_start = enc.decode_monomial(mono)
            assert decoded_word == word, (
                f"Roundtrip failed for word={word}, start={start}: "
                f"decoded_word={decoded_word}"
            )
            assert decoded_start == start, (
                f"Roundtrip failed for word={word}, start={start}: "
                f"decoded_start={decoded_start}"
            )

    def test_decode_monomial_empty(self):
        """The integer 1 decodes to ((), 0) by convention."""
        enc = LetterplaceEncoder(["x", "y"], max_degree=4)
        word, start = enc.decode_monomial(sp.Integer(1))
        assert word == ()
        assert start == 0

    def test_decode_monomial_rejects_nonconsecutive(self):
        """A monomial with non-consecutive positions raises ValueError."""
        enc = LetterplaceEncoder(["x", "y"], max_degree=4)
        # x0_s0 * x1_s2 — positions 0 and 2 are not consecutive.
        bad_mono = sp.Symbol("x0_s0") * sp.Symbol("x1_s2")
        with pytest.raises(ValueError, match="not consecutive"):
            enc.decode_monomial(bad_mono)

    def test_decode_monomial_rejects_power(self):
        """A monomial with a variable raised to a power > 1 raises ValueError."""
        enc = LetterplaceEncoder(["x", "y"], max_degree=4)
        x0_s0 = sp.Symbol("x0_s0")
        bad_mono = x0_s0 * x0_s0  # x0_s0^2
        with pytest.raises(ValueError, match="power.* > 1"):
            enc.decode_monomial(bad_mono)


# ----------------------------------------------------------------------------
# 7. decode_polynomial (bonus — not required but useful)
# ----------------------------------------------------------------------------


class TestDecodePolynomial:
    """Tests for LetterplaceEncoder.decode_polynomial."""

    def test_decode_polynomial_single_position(self):
        """A polynomial at a single position decodes to a one-entry dict."""
        enc = LetterplaceEncoder(["x", "y"], max_degree=4)
        rule = _rule_xy()
        comm_poly = enc.encode_relation(rule, start=0)
        decoded = enc.decode_polynomial(comm_poly)
        # The relation xy - q*yx at s=0 should decode to a single entry
        # at start=0: the NC polynomial (0,1) - q*(1,0).
        assert 0 in decoded, f"Expected start=0 in decoded dict, got {decoded.keys()}"
        nc_poly = decoded[0]
        expected = Polynomial([
            Term(ONE, Monomial((0, 1))),
            Term(-qpow(1), Monomial((1, 0))),
        ])
        assert nc_poly == expected, (
            f"Decoded NC poly at s=0 = {nc_poly}, expected {expected}"
        )

    def test_decode_polynomial_multiple_positions(self):
        """A sum of relations at different positions decodes to multiple entries."""
        enc = LetterplaceEncoder(["x", "y"], max_degree=4)
        rule = _rule_xy()
        # Sum the encoded relations at s=0 and s=1.
        comm_poly = enc.encode_relation(rule, 0) + enc.encode_relation(rule, 1)
        decoded = enc.decode_polynomial(comm_poly)
        assert 0 in decoded, f"Expected start=0 in decoded, got {decoded.keys()}"
        assert 1 in decoded, f"Expected start=1 in decoded, got {decoded.keys()}"
        # Each entry should be the NC polynomial (0,1) - q*(1,0).
        expected = Polynomial([
            Term(ONE, Monomial((0, 1))),
            Term(-qpow(1), Monomial((1, 0))),
        ])
        assert decoded[0] == expected
        assert decoded[1] == expected

    def test_decode_polynomial_zero(self):
        """The zero polynomial decodes to the empty dict."""
        enc = LetterplaceEncoder(["x", "y"], max_degree=4)
        assert enc.decode_polynomial(sp.Integer(0)) == {}

    def test_decode_polynomial_roundtrip(self):
        """encode_polynomial followed by decode_polynomial recovers the NC poly.

        For a polynomial whose terms all start at the same position,
        decode should recover it exactly.
        """
        enc = LetterplaceEncoder(["x", "y"], max_degree=4)
        original = Polynomial([
            Term(ONE, Monomial((0, 1))),
            Term(-qpow(1), Monomial((1, 0))),
            Term(qint(2), Monomial((0, 0))),
        ])
        comm = enc.encode_polynomial(original, start=0)
        decoded = enc.decode_polynomial(comm)
        assert 0 in decoded
        # The decoded NC polynomial should equal the original (after normalize).
        assert decoded[0] == original.normalize(), (
            f"Roundtrip failed: original={original.normalize()}, "
            f"decoded={decoded[0]}"
        )


# ----------------------------------------------------------------------------
# 8. Integration: letterplace ideal correctness
# ----------------------------------------------------------------------------


class TestLetterplaceIdeal:
    """Integration tests for the letterplace ideal of the quantum plane."""

    def test_quantum_plane_normal_form_via_letterplace(self):
        """The letterplace Groebner basis reduces x0_s0*x1_s1 to q*x1_s0*x0_s1.

        This mirrors the NC rewriting ``x*y -> q*y*x``: the leading
        monomial ``x0_s0 * x1_s1`` (the NC LHS at s=0) should reduce to
        ``q * x1_s0 * x0_s1`` (the NC RHS at s=0) modulo the Groebner
        basis with lex order.
        """
        pres = _quantum_plane_presentation()
        enc = LetterplaceEncoder(["x", "y"], max_degree=2)
        # Use lex order with q included so the leading term is the NC LHS.
        gb = enc.compute_groebner_basis(pres, order="lex", include_q=True)
        all_vars = enc.full_variable_list()
        # The NC LHS monomial at s=0:
        lhs_mono = enc.encode_word((0, 1), 0)  # x0_s0 * x1_s1
        remainder = sp.reduced(lhs_mono, gb, *all_vars, order="lex")[1]
        # The remainder should be q * x1_s0 * x0_s1 (the NC RHS at s=0).
        expected_rhs = enc.q_sym * enc.encode_word((1, 0), 0)
        assert sp.simplify(remainder - expected_rhs) == 0, (
            f"Letterplace reduction of x0_s0*x1_s1 should give "
            f"q*x1_s0*x0_s1, got {remainder}"
        )

    def test_quantum_plane_double_swap(self):
        """x0_s0 * x0_s1 * x1_s2 (i.e. NC x*x*y at s=0) reduces to q^2 * x1_s0 * x0_s1 * x0_s2.

        This mirrors the NC rewriting ``x*x*y -> q^2 * y*x*x``.
        """
        pres = _quantum_plane_presentation()
        enc = LetterplaceEncoder(["x", "y"], max_degree=3)
        gb = enc.compute_groebner_basis(pres, order="lex", include_q=True)
        all_vars = enc.full_variable_list()
        lhs_mono = enc.encode_word((0, 0, 1), 0)  # x0_s0 * x0_s1 * x1_s2
        remainder = sp.reduced(lhs_mono, gb, *all_vars, order="lex")[1]
        # NC normal form: x*x*y -> q^2 * y*x*x.
        expected_rhs = enc.q_sym ** 2 * enc.encode_word((1, 0, 0), 0)
        assert sp.simplify(remainder - expected_rhs) == 0, (
            f"Letterplace reduction of x0_s0*x0_s1*x1_s2 should give "
            f"q^2*x1_s0*x0_s1*x0_s2, got {remainder}"
        )

    def test_quantum_plane_already_normal(self):
        """y*x (already normal in the NC algebra) is fixed by the letterplace basis.

        ``x1_s0 * x0_s1`` (NC y*x at s=0) should reduce to itself (it's
        already in normal form).
        """
        pres = _quantum_plane_presentation()
        enc = LetterplaceEncoder(["x", "y"], max_degree=2)
        gb = enc.compute_groebner_basis(pres, order="lex", include_q=True)
        all_vars = enc.full_variable_list()
        normal_mono = enc.encode_word((1, 0), 0)  # x1_s0 * x0_s1
        remainder = sp.reduced(normal_mono, gb, *all_vars, order="lex")[1]
        assert sp.simplify(remainder - normal_mono) == 0, (
            f"y*x is already normal; letterplace reduction should leave it "
            f"unchanged, got {remainder}"
        )


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------


if __name__ == "__main__":
    import unittest

    unittest.main()
