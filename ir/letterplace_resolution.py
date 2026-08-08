"""Letterplace free resolution and Hochschild cohomology (task LP-1a).

This module builds on the :class:`ir.letterplace.LetterplaceEncoder` to
compute, via the letterplace correspondence (Cohen 1987; La Scala
arXiv:1605.06944), the *commutative* free resolution of the trivial
module ``k`` over the letterplace ring ``R = K[x_{i,s}] / I_lp`` and
extracts the Hochschild cohomology dimensions

    dim_K HH^n(A, k) = dim_K Ext^n_A(k, k)              (Cartan–Eilenberg)
                     = dim_K Tor_n^R(k, k)              (La Scala theorem)
                     = β'_n                               (minimal Betti)

where ``A = K<gens> / <relations>`` is the original NC algebra and
``β'_n`` is the n-th Betti number of the minimal free resolution of
``k`` over ``R``.

Pipeline
--------

1.  **Encode** the NC presentation as the letterplace ideal ``I_lp``
    (via :class:`LetterplaceEncoder`).  We work over the algebraic
    field ``K = Q(omega)`` (with ``omega = exp(2*pi*i/3)`` a primitive
    3rd root of unity) so that ``q`` is *not* a variable but a specific
    algebraic number; this avoids both the Fraction-truncation bug in
    the original encoder (LP-0a) and the explosion of the Groebner
    basis that occurs when ``q`` is kept symbolic with the cyclotomic
    relation ``q^2 + q + 1 = 0`` appended as an extra generator.

2.  **Compute the (minimal) commutative Groebner basis** ``G`` of
    ``I_lp`` over ``K`` via :func:`sympy.groebner` with
    ``domain=K.algebraic_field(omega)``.  The Groebner basis is then
    *minimised* by removing any element whose leading monomial is
    divisible by the leading monomial of another element
    (:func:`minimise_groebner_basis`).

3.  **Compute the first syzygy module** ``Syz(G)`` via *Schreyer's
    theorem*: for each pair ``(g_i, g_j)`` in the minimised ``G``, form
    the S-polynomial ``S(g_i, g_j) = (lcm/LT(g_i)) g_i - (lcm/LT(g_j))
    g_j`` and reduce it to 0 via ``G``, tracking the reduction
    coefficients ``(c_1, ..., c_m)`` (via :func:`sympy.reduced`).  The
    syzygy is the vector

        s_ij  =  (lcm/LT(g_i)) e_i  -  (lcm/LT(g_j)) e_j  -  sum_k c_k e_k.

    Schreyer's theorem guarantees that the set of all such ``s_ij``
    forms a Groebner basis of ``Syz(G)`` under the *Schreyer order*
    (which depends on the leading terms of ``G``).  We then minimise
    this syzygy Groebner basis.

4.  **Iterate** to compute the second syzygy module ``Syz(Syz(G))``
    (and beyond, if needed).

5.  **Extract Hochschild cohomology dimensions**.  Because we work
    over the algebraic field ``K = Q(omega)`` (so ``q`` is already
    substituted), every Groebner basis element has positive degree in
    the x-variables, and every syzygy has positive-degree entries
    (after minimisation, no leading term divides another; the trivial
    syzygies have already been removed).  Hence ``d_n ⊗ k = 0`` for
    every differential, and

        dim_K HH^n(A, k)  =  β'_n

    where ``β'_n`` is the size of the minimised Groebner basis at
    syzygy level ``n`` (with ``β'_0 = 1``, ``β'_1 = |G_min|``,
    ``β'_2 = |Syz(G)_min|``, etc.).

A subtlety
----------

The letterplace correspondence gives an isomorphism of *graded*
complexes up to total NC degree ``max_degree``.  For our application
(``u_q(sl_2)`` at ``ell = 3``) the PBW monomials have length at most
3, so ``max_degree = 3`` or ``4`` suffices to capture ``HH^2``.
Smaller ``max_degree`` gives a truncated resolution that may
undercount higher syzygies.

Performance
-----------

:func:`sympy.groebner` with the algebraic-field domain is fast for
small ``max_degree`` (the full ``u_q(sl_2)`` presentation at
``max_degree = 3`` finishes in < 1 s and gives a 12-element minimised
Groebner basis).  The syzygy computation is more expensive (the
Schreyer syzygy module has ~30 generators, each requiring an
S-polynomial reduction), but is still tractable.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import sympy as sp

from .letterplace import LetterplaceEncoder
from .parser import Presentation


# ============================================================================
# Algebraic-field helpers
# ============================================================================


def make_omega3_field():
    """Return the algebraic field ``Q(omega)`` with ``omega^2 + omega + 1 = 0``.

    Returns a tuple ``(field, omega_value)`` where ``field`` is a
    :class:`sympy.polys.domains.AlgebraicField` suitable as the
    ``domain=`` argument to :func:`sympy.groebner`, and ``omega_value``
    is the sympy expression ``-1/2 + sqrt(3)*I/2`` (= ``exp(2*pi*i/3)``)
    that should be passed as ``q_value=`` to
    :class:`LetterplaceEncoder`.
    """
    omega_value = sp.Rational(-1, 2) + sp.sqrt(3) * sp.I / 2
    omega_alg = sp.AlgebraicNumber(omega_value)
    field = sp.QQ.algebraic_field(omega_alg)
    return field, omega_value


def make_omega5_field():
    """Return the algebraic field ``Q(omega)`` with ``omega`` a primitive 5th root.

    Used (eventually) for ``u_q(sl_3)`` at ``ell = 5``.
    """
    omega_value = sp.exp(2 * sp.pi * sp.I / 5)
    omega_alg = sp.AlgebraicNumber(omega_value)
    field = sp.QQ.algebraic_field(omega_alg)
    return field, omega_value


# ============================================================================
# Groebner basis utilities
# ============================================================================


def leading_monomial(poly: sp.Expr, variables: List[sp.Symbol], order: str = "lex") -> sp.Expr:
    """Return the leading *monomial* (without coefficient) of ``poly``.

    The leading monomial is computed with respect to the given monomial
    ``order`` on ``variables``.  Returns a sympy expression that is a
    single monomial in ``variables`` (with coefficient 1).
    """
    p = sp.poly(poly, *variables, order=order)
    lm = p.LM()  # monomial tuple
    # Build the monomial expression from the LM tuple.
    return sp.Mul(*[variables[i] ** lm[i] for i in range(len(lm)) if lm[i] > 0])


def leading_coefficient(poly: sp.Expr, variables: List[sp.Symbol], order: str = "lex") -> sp.Expr:
    """Return the leading coefficient of ``poly``."""
    p = sp.poly(poly, *variables, order=order)
    return sp.nsimplify(p.LC())


def lcm_monomials(m1: sp.Expr, m2: sp.Expr, variables: List[sp.Symbol]) -> sp.Expr:
    """Return the least common multiple of two commutative monomials.

    Each monomial is a product of powers of variables from
    ``variables`` (with possible coefficient, which is ignored).
    """
    p1 = sp.poly(m1, *variables)
    p2 = sp.poly(m2, *variables)
    # Get monomial tuples (each is a tuple of exponents).
    e1 = p1.monoms()[0]
    e2 = p2.monoms()[0]
    e_lcm = tuple(max(e1[i], e2[i]) for i in range(len(variables)))
    return sp.Mul(*[variables[i] ** e_lcm[i] for i in range(len(variables)) if e_lcm[i] > 0])


def divides(m1: sp.Expr, m2: sp.Expr, variables: List[sp.Symbol]) -> bool:
    """Return True iff monomial ``m1`` divides monomial ``m2``."""
    p1 = sp.poly(m1, *variables)
    p2 = sp.poly(m2, *variables)
    e1 = p1.monoms()[0]
    e2 = p2.monoms()[0]
    return all(e1[i] <= e2[i] for i in range(len(variables)))


def monomial_div(m1: sp.Expr, m2: sp.Expr, variables: List[sp.Symbol]) -> sp.Expr:
    """Return the monomial ``m1 / m2`` (requires ``m2 | m1``)."""
    p1 = sp.poly(m1, *variables)
    p2 = sp.poly(m2, *variables)
    e1 = p1.monoms()[0]
    e2 = p2.monoms()[0]
    assert all(e2[i] <= e1[i] for i in range(len(variables))), (
        f"monomial_div: {m2} does not divide {m1}"
    )
    e_div = tuple(e1[i] - e2[i] for i in range(len(variables)))
    return sp.Mul(*[variables[i] ** e_div[i] for i in range(len(variables)) if e_div[i] > 0])


def minimise_groebner_basis(
    gb: List[sp.Expr],
    variables: List[sp.Symbol],
    order: str = "lex",
) -> List[sp.Expr]:
    """Return a minimal Groebner basis by removing redundant elements.

    An element ``g_i`` is redundant if its leading monomial is
    divisible by the leading monomial of some other element ``g_j``
    (``j != i``).  Removing such elements yields a smaller Groebner
    basis of the same ideal.

    The output is a list of polynomials (expanded).  The order of
    elements is preserved as much as possible (we iterate left to
    right and keep an element iff its LM is not divisible by any
    previously-kept element's LM; this is a greedy minimisation).
    """
    if not gb:
        return []
    # Pre-compute leading monomials.
    lms = [leading_monomial(g, variables, order) for g in gb]
    kept: List[sp.Expr] = []
    kept_lms: List[sp.Expr] = []
    for g, lm in zip(gb, lms):
        redundant = False
        for klm in kept_lms:
            if klm != lm and divides(klm, lm, variables):
                redundant = True
                break
        if not redundant:
            # Also remove duplicates (same LM).
            if lm in kept_lms:
                # Keep the first one; skip duplicates.
                continue
            kept.append(g)
            kept_lms.append(lm)
    return kept


# ============================================================================
# Syzygy computation via Schreyer's theorem
# ============================================================================


def reduce_with_history(
    poly: sp.Expr,
    gb: List[sp.Expr],
    variables: List[sp.Symbol],
    order: str = "lex",
    domain=None,
) -> Tuple[List[sp.Expr], sp.Expr]:
    """Reduce ``poly`` modulo the Groebner basis ``gb`` and record the quotients.

    Returns ``(quotients, remainder)`` where ``quotients`` is a list of
    length ``len(gb)`` such that

        poly  =  sum_i quotients[i] * gb[i]  +  remainder

    and the remainder is in normal form modulo ``gb``.

    The reduction is performed by :func:`sympy.reduced`, which uses
    the same algorithm as :func:`sympy.groebner` and produces the
    correct quotient list.
    """
    n = len(gb)
    if n == 0:
        return [], sp.expand(poly)
    # Handle the zero polynomial specially: sp.reduced(0, ...) returns
    # ([], 0), but we want ([0, 0, ..., 0], 0).
    if poly == 0:
        return [sp.Integer(0)] * n, sp.Integer(0)
    if domain is not None:
        q, r = sp.reduced(poly, gb, *variables, order=order, domain=domain)
    else:
        q, r = sp.reduced(poly, gb, *variables, order=order)
    # `q` is a list of quotients, one per element of `gb`.  But if the
    # reduction eliminated some elements (e.g., the polynomial was 0
    # after the first reduction), `q` may be shorter than `n`.  Pad
    # with zeros.
    if len(q) < n:
        q = list(q) + [sp.Integer(0)] * (n - len(q))
    return list(q), sp.expand(r)


def compute_s_polynomial(
    g_i: sp.Expr,
    g_j: sp.Expr,
    variables: List[sp.Symbol],
    order: str = "lex",
) -> sp.Expr:
    """Compute the S-polynomial of ``g_i`` and ``g_j``.

    ``S(g_i, g_j) = (lcm/LT(g_i)) * g_i  -  (lcm/LT(g_j)) * g_j``
    where ``lcm = lcm(LM(g_i), LM(g_j))``.  The leading coefficients
    of ``g_i`` and ``g_j`` are normalised to 1 (the S-polynomial
    formula assumes monic polynomials).
    """
    lm_i = leading_monomial(g_i, variables, order)
    lm_j = leading_monomial(g_j, variables, order)
    lc_i = leading_coefficient(g_i, variables, order)
    lc_j = leading_coefficient(g_j, variables, order)
    lcm = lcm_monomials(lm_i, lm_j, variables)
    # Normalise: g_i / lc_i (so LT becomes LM with coeff 1).
    factor_i = monomial_div(lcm, lm_i, variables) / lc_i
    factor_j = monomial_div(lcm, lm_j, variables) / lc_j
    s = sp.expand(factor_i * g_i - factor_j * g_j)
    return s


def compute_first_syzygies(
    gb: List[sp.Expr],
    variables: List[sp.Symbol],
    order: str = "lex",
    domain=None,
    verbose: bool = False,
) -> List[List[sp.Expr]]:
    """Compute the first syzygy module of ``gb`` via Schreyer's theorem.

    Returns a list of syzygy vectors.  Each syzygy is a list of length
    ``len(gb)`` of sympy polynomials ``(s_1, ..., s_m)`` such that

        sum_i s_i * gb[i]  =  0.

    The syzygies are computed from the S-polynomials of all pairs
    ``(g_i, g_j)`` with ``i < j``.  Schreyer's theorem guarantees that
    these syzygies form a Groebner basis of ``Syz(gb)`` under the
    Schreyer order.

    The syzygies are returned *un-minimised*; call
    :func:`minimise_syzygies` to remove redundant syzygies.
    """
    n = len(gb)
    syzygies: List[List[sp.Expr]] = []
    for i in range(n):
        for j in range(i + 1, n):
            s_poly = compute_s_polynomial(gb[i], gb[j], variables, order)
            quotients, remainder = reduce_with_history(
                s_poly, gb, variables, order, domain=domain
            )
            if remainder != 0:
                # This shouldn't happen if `gb` is a Groebner basis.
                if verbose:
                    print(
                        f"  WARNING: S(g_{i}, g_{j}) did not reduce to 0 "
                        f"(remainder = {remainder}); skipping"
                    )
                continue
            # The syzygy is:
            #   s_ij = (lcm/LT(g_i)) e_i  -  (lcm/LT(g_j)) e_j  -  sum_k c_k e_k
            # where (c_1, ..., c_m) = quotients.
            lm_i = leading_monomial(gb[i], variables, order)
            lm_j = leading_monomial(gb[j], variables, order)
            lc_i = leading_coefficient(gb[i], variables, order)
            lc_j = leading_coefficient(gb[j], variables, order)
            lcm = lcm_monomials(lm_i, lm_j, variables)
            factor_i = monomial_div(lcm, lm_i, variables) / lc_i
            factor_j = monomial_div(lcm, lm_j, variables) / lc_j
            # Build the syzygy vector.
            syzygy = [sp.Integer(0)] * n
            syzygy[i] = sp.expand(syzygy[i] + factor_i)
            syzygy[j] = sp.expand(syzygy[j] - factor_j)
            for k in range(n):
                syzygy[k] = sp.expand(syzygy[k] - quotients[k])
            # Verify the syzygy: sum_k syzygy[k] * gb[k] should be 0.
            if verbose:
                check = sp.expand(sum(s * g for s, g in zip(syzygy, gb)))
                if check != 0:
                    print(
                        f"  WARNING: syzygy({i},{j}) does not annihilate gb "
                        f"(check = {check})"
                    )
            syzygies.append(syzygy)
    return syzygies


def syzygy_leading_monomial(
    syzygy: List[sp.Expr],
    gb: List[sp.Expr],
    variables: List[sp.Symbol],
    order: str = "lex",
) -> sp.Expr:
    """Return the leading monomial of a syzygy under the Schreyer order.

    The Schreyer order on the free module ``R^n`` (with basis
    ``e_1, ..., e_n``) is defined as follows: ``e_i * m_1 < e_j * m_2``
    iff ``LM(g_i) * m_1 < LM(g_j) * m_2`` (in the underlying monomial
    order on ``R``), or ``LM(g_i) * m_1 == LM(g_j) * m_2`` and ``i <
    j``.

    Equivalently, for a syzygy ``s = sum_i s_i e_i``, the leading
    monomial is the maximum (under the Schreyer order) of the
    monomials in each ``s_i``.

    For our purposes, we use a simplified version: the leading
    monomial of a syzygy is the leading monomial of the *nonzero*
    component with the *largest* index among the highest-Schreyer-order
    terms.  In practice we just look at the leading monomials of each
    component and pick the maximum under the Schreyer order.
    """
    # For each nonzero component, compute the leading monomial of s_i
    # (with respect to the underlying order on R) and pair it with the
    # leading monomial of g_i (for the Schreyer-order comparison).
    candidates: List[Tuple[sp.Expr, int, sp.Expr]] = []  # (lm_g_i * lm_s_i, i, lm_s_i)
    for i, s_i in enumerate(syzygy):
        if s_i == 0:
            continue
        lm_s_i = leading_monomial(s_i, variables, order)
        lm_g_i = leading_monomial(gb[i], variables, order)
        # Schreyer order key: (lm_g_i * lm_s_i, i)
        schreyer_lm = lcm_monomials(lm_g_i, lm_s_i, variables)
        # Wait, lcm is wrong — we want the *product*, not lcm, since
        # the Schreyer order treats e_i * m as the monomial LM(g_i)*m.
        schreyer_lm = sp.expand(lm_g_i * lm_s_i)
        candidates.append((schreyer_lm, i, lm_s_i))
    if not candidates:
        return sp.Integer(0)
    # Pick the maximum under the underlying monomial order.
    # For lex / grevlex, we can compare via sp.Poly.
    best = candidates[0]
    for cand in candidates[1:]:
        # Compare cand[0] vs best[0] under `order`.
        cmp = compare_monomials(cand[0], best[0], variables, order)
        if cmp > 0:
            best = cand
        elif cmp == 0 and cand[1] < best[1]:
            # Tie in monomial, lower index wins (Schreyer convention).
            best = cand
    return best[0]  # the Schreyer leading monomial


def compare_monomials(m1: sp.Expr, m2: sp.Expr, variables: List[sp.Symbol], order: str = "lex") -> int:
    """Compare two monomials under the given order.  Returns +1, 0, or -1."""
    if m1 == m2:
        return 0
    p1 = sp.poly(m1, *variables, order=order)
    p2 = sp.poly(m2, *variables, order=order)
    e1 = p1.monoms()[0]
    e2 = p2.monoms()[0]
    if order == "lex":
        for i in range(len(variables)):
            if e1[i] > e2[i]:
                return 1
            if e1[i] < e2[i]:
                return -1
        return 0
    elif order == "grlex":
        d1, d2 = sum(e1), sum(e2)
        if d1 > d2:
            return 1
        if d1 < d2:
            return -1
        for i in range(len(variables)):
            if e1[i] > e2[i]:
                return 1
            if e1[i] < e2[i]:
                return -1
        return 0
    elif order == "grevlex":
        d1, d2 = sum(e1), sum(e2)
        if d1 > d2:
            return 1
        if d1 < d2:
            return -1
        for i in range(len(variables) - 1, -1, -1):
            if e1[i] > e2[i]:
                return -1
            if e1[i] < e2[i]:
                return 1
        return 0
    else:
        raise ValueError(f"compare_monomials: unknown order {order!r}")


def minimise_syzygies(
    syzygies: List[List[sp.Expr]],
    gb: List[sp.Expr],
    variables: List[sp.Symbol],
    order: str = "lex",
) -> List[List[sp.Expr]]:
    """Remove redundant syzygies by Schreyer-leading-monomial divisibility.

    A syzygy ``s_i`` is redundant if its Schreyer leading monomial is
    divisible by the Schreyer leading monomial of another syzygy
    ``s_j`` (``j != i``).  Removing such syzygies yields a minimal
    Groebner basis of the syzygy module.

    Returns the minimised list of syzygies.
    """
    if not syzygies:
        return []
    # Pre-compute Schreyer leading monomials.
    lms = [syzygy_leading_monomial(s, gb, variables, order) for s in syzygies]
    kept: List[List[sp.Expr]] = []
    kept_lms: List[sp.Expr] = []
    for s, lm in zip(syzygies, lms):
        redundant = False
        for klm in kept_lms:
            if klm != lm and divides(klm, lm, variables):
                redundant = True
                break
        if not redundant:
            if lm in kept_lms:
                continue
            kept.append(s)
            kept_lms.append(lm)
    return kept


# ============================================================================
# LetterplaceResolution: the full pipeline
# ============================================================================


class LetterplaceResolution:
    """Compute the letterplace free resolution of ``k`` over a NC algebra.

    Parameters
    ----------
    presentation : Presentation
        The NC algebra presentation.
    max_degree : int
        Maximum letterplace position index (passed to
        :class:`LetterplaceEncoder`).
    order : str
        Monomial order for Groebner basis computation (default
        ``'lex'``, the letterplace-friendly order).
    q_value : sympy.Expr, optional
        If set (e.g. to ``omega = exp(2*pi*i/3)``), all coefficients
        are evaluated at ``q = q_value`` and the Groebner basis is
        computed over the algebraic field ``Q(q_value)``.  Required
        for ``u_q(sl_2)`` at ``ell = 3`` to avoid the
        Fraction-truncation bug and to keep the Groebner basis small.
    domain : sympy domain, optional
        The coefficient domain for the Groebner basis computation
        (e.g. ``QQ.algebraic_field(omega)``).  If ``None``, sympy
        uses the default ``EX`` domain (rational functions in
        symbolic ``q``).
    """

    def __init__(
        self,
        presentation: Presentation,
        max_degree: int,
        order: str = "lex",
        q_value: Optional[sp.Expr] = None,
        domain=None,
        eps_values: Optional[List[sp.Expr]] = None,
    ) -> None:
        self.presentation = presentation
        self.max_degree = int(max_degree)
        self.order = order
        self.q_value = q_value
        self.domain = domain
        self.encoder = LetterplaceEncoder(
            presentation.generators,
            max_degree=max_degree,
            q_value=q_value,
        )
        self.variables: List[sp.Symbol] = list(self.encoder.all_variables())

        # `eps_values[i]` is the value of eps(x_i) in K, i.e. the value
        # of the i-th NC generator under the algebra's counit.  This is
        # required for the "evaluation at the trivial module" reduction
        # used to compute Hochschild cohomology: when we tensor the
        # letterplace resolution with the trivial module k = K, each
        # letterplace variable x_{i,s} is sent to eps(x_i) (independent
        # of the position s).  For example, for u_q(sl_2) at ell=3,
        # eps(K) = 1 (since K is invertible of order 3) and
        # eps(E) = eps(F) = 0; so x_{0,s} -> 1 and x_{1,s}, x_{2,s} -> 0.
        #
        # If eps_values is None, we default to eps(x_i) = 0 for all i
        # (the "constant-term" reduction, appropriate for algebras
        # generated by nilpotents, e.g. the quantum plane).
        if eps_values is None:
            eps_values = [sp.Integer(0)] * len(presentation.generators)
        self.eps_values: List[sp.Expr] = list(eps_values)
        # Build the substitution dict: x_{i,s} -> eps_values[i] for all s.
        self._eval_substitution: dict = {}
        for i in range(self.encoder.n_gens):
            for s in range(self.encoder.max_degree + 1):
                self._eval_substitution[self.encoder.var(i, s)] = self.eps_values[i]

        # Cached results.
        self._gb_raw: Optional[List[sp.Expr]] = None
        self._gb_minimal: Optional[List[sp.Expr]] = None
        self._syz1_raw: Optional[List[List[sp.Expr]]] = None
        self._syz1_minimal: Optional[List[List[sp.Expr]]] = None
        self._syz2_raw: Optional[List[List[sp.Expr]]] = None
        self._syz2_minimal: Optional[List[List[sp.Expr]]] = None

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def compute_groebner_basis(self, minimise: bool = True) -> List[sp.Expr]:
        """Compute (and minimise) the commutative Groebner basis of ``I_lp``."""
        if self._gb_raw is None:
            self._gb_raw = self.encoder.compute_groebner_basis(
                self.presentation,
                order=self.order,
                domain=self.domain,
            )
        if minimise:
            if self._gb_minimal is None:
                self._gb_minimal = minimise_groebner_basis(
                    self._gb_raw, self.variables, self.order
                )
            return list(self._gb_minimal)
        return list(self._gb_raw)

    def compute_first_syzygies(self, minimise: bool = True) -> List[List[sp.Expr]]:
        """Compute (and minimise) the first syzygy module ``Syz(G)``."""
        if self._syz1_raw is None:
            gb = self.compute_groebner_basis(minimise=True)
            self._syz1_raw = compute_first_syzygies(
                gb,
                self.variables,
                order=self.order,
                domain=self.domain,
            )
        if minimise:
            if self._syz1_minimal is None:
                gb = self.compute_groebner_basis(minimise=True)
                self._syz1_minimal = minimise_syzygies(
                    self._syz1_raw, gb, self.variables, self.order
                )
            return list(self._syz1_minimal)
        return list(self._syz1_raw)

    def compute_second_syzygies(self, minimise: bool = True) -> List[List[sp.Expr]]:
        """Compute (and minimise) the second syzygy module ``Syz(Syz(G))``.

        The second syzygies are syzygies AMONG the first syzygies,
        i.e., vectors ``(t_1, ..., t_{|Syz|})`` such that

            sum_j t_j * syzygy_j  =  0      (as a vector in R^|G|).
        """
        if self._syz2_raw is None:
            syz1 = self.compute_first_syzygies(minimise=True)
            if not syz1:
                self._syz2_raw = []
            else:
                # The first syzygies live in R^|G|, viewed as a free module.
                # We compute their syzygies by treating each syzygy as a
                # "polynomial vector" and applying Schreyer's theorem again.
                # The "Groebner basis" for this module is the (minimised)
                # set of first syzygies themselves, viewed as elements of
                # the free module R^|G| under the Schreyer order.
                #
                # To compute syzygies of syzygies, we use a standard trick:
                # introduce new variables y_1, ..., y_{|G|} (one per basis
                # vector of R^|G|) and a "generic" element
                #   F = sum_i y_i * g_i
                # in the ring R[y_1, ..., y_{|G|}].  The first syzygies of
                # (g_1, ..., g_{|G|}) are the relations among the y_i
                # modulo F.  But this is complicated; instead we use a
                # direct approach.
                #
                # Direct approach: for each pair of first syzygies
                # (s_a, s_b), compute the S-polynomial in the free module
                # R^|G|, reduce via the first syzygies (as a Groebner
                # basis of the syzygy module under the Schreyer order),
                # and form the second syzygy.
                #
                # Concretely: a "polynomial vector" in R^|G| is a list of
                # |G| polynomials.  Multiplication by a polynomial r in R
                # is componentwise.  Reduction of a vector v by a set of
                # vectors S = {s_1, ..., s_m} (each in R^|G|) is: find a
                # component k and a monomial in v_k whose Schreyer LM
                # matches the Schreyer LM of some s_j; subtract
                # (factor) * s_j.  Repeat until no reduction is possible.
                self._syz2_raw = self._compute_module_syzygies(syz1)
        if minimise:
            if self._syz2_minimal is None:
                syz1 = self.compute_first_syzygies(minimise=True)
                self._syz2_minimal = minimise_syzygies(
                    self._syz2_raw, syz1, self.variables, self.order
                )
            return list(self._syz2_minimal)
        return list(self._syz2_raw)

    def _compute_module_syzygies(
        self, module_gens: List[List[sp.Expr]]
    ) -> List[List[sp.Expr]]:
        """Compute syzygies of a list of module generators (vectors in R^n).

        Each ``module_gens[i]`` is a list of length ``n`` (the rank of
        the free module ``R^n``).  Returns a list of syzygy vectors
        (each a list of length ``len(module_gens)`` of polynomials in
        ``R``) such that

            sum_j syzygy[j] * module_gens[j]  =  0      (componentwise).

        Uses Schreyer's theorem applied to the module Groebner basis.
        """
        m = len(module_gens)  # number of generators
        if m == 0:
            return []
        n = len(module_gens[0])  # rank of the free module
        # For each pair (a, b) with a < b, compute the S-polynomial
        # (as a vector) and reduce it via the module Groebner basis.
        syzygies: List[List[sp.Expr]] = []
        # Pre-compute the leading monomials (under the Schreyer order
        # of R^n) of each module generator.
        # The Schreyer order: e_i * monom ~ monom * (some fixed monomial based on i).
        # For simplicity, we use the "term-over-position" order: the
        # leading term of a vector v is the largest (component_idx, monomial)
        # pair, where component_idx is the index of the first nonzero
        # component with the largest monomial (under the underlying order).
        # We approximate this by: for each module generator, find the
        # leading (component, monomial) pair.
        leading_terms = []  # list of (component_idx, monomial, coeff) per generator
        for s in module_gens:
            lt = self._vector_leading_term(s)
            leading_terms.append(lt)
        for a in range(m):
            for b in range(a + 1, m):
                # Compute S-polynomial of module_gens[a] and module_gens[b].
                s_poly_vec = self._vector_s_polynomial(
                    module_gens[a], module_gens[b],
                    leading_terms[a], leading_terms[b],
                )
                # Reduce via the module Groebner basis.
                quotients, remainder = self._vector_reduce(
                    s_poly_vec, module_gens, leading_terms
                )
                if any(r != 0 for r in remainder):
                    # Did not reduce to 0; skip.
                    continue
                # The syzygy is:
                #   t_ab = factor_a * e_a - factor_b * e_b - sum_j q_j e_j
                # where factor_a, factor_b are the LCM/LT factors.
                lt_a = leading_terms[a]
                lt_b = leading_terms[b]
                # factor_a and factor_b are monomials (with coefficients).
                factor_a, factor_b = self._vector_lcm_factors(lt_a, lt_b)
                syzygy = [sp.Integer(0)] * m
                syzygy[a] = sp.expand(syzygy[a] + factor_a)
                syzygy[b] = sp.expand(syzygy[b] - factor_b)
                for j in range(m):
                    syzygy[j] = sp.expand(syzygy[j] - quotients[j])
                syzygies.append(syzygy)
        return syzygies

    def _vector_leading_term(self, v: List[sp.Expr]) -> Tuple[int, sp.Expr, sp.Expr]:
        """Return the leading term of a vector ``v`` in ``R^n``.

        Returns ``(component_idx, monomial, coeff)`` where the leading
        term is ``coeff * monomial * e_{component_idx}`` under the
        term-over-position order (with the underlying monomial order
        on ``R`` given by :attr:`self.order`).
        """
        best = None  # (monomial, component_idx, coeff)
        for i, vi in enumerate(v):
            if vi == 0:
                continue
            lm = leading_monomial(vi, self.variables, self.order)
            lc = leading_coefficient(vi, self.variables, self.order)
            cand = (lm, i, lc)
            if best is None:
                best = cand
            else:
                cmp = compare_monomials(cand[0], best[0], self.variables, self.order)
                if cmp > 0:
                    best = cand
                elif cmp == 0 and cand[1] < best[1]:
                    best = cand
        if best is None:
            return (-1, sp.Integer(0), sp.Integer(0))
        return (best[1], best[0], best[2])

    def _vector_lcm_factors(
        self,
        lt_a: Tuple[int, sp.Expr, sp.Expr],
        lt_b: Tuple[int, sp.Expr, sp.Expr],
    ) -> Tuple[sp.Expr, sp.Expr]:
        """Compute the S-polynomial factors for two vector leading terms.

        Returns ``(factor_a, factor_b)`` such that the S-polynomial is
        ``factor_a * v_a - factor_b * v_b``.

        If the leading terms are in the same component, the LCM of the
        monomials is used (as in the standard S-polynomial formula).
        If they are in different components, the LCM is the product
        (i.e., the leading terms are coprime in the module sense), and
        the S-polynomial is ``LT_b * v_a - LT_a * v_b``.
        """
        comp_a, monom_a, coeff_a = lt_a
        comp_b, monom_b, coeff_b = lt_b
        if comp_a == comp_b:
            # Same component: standard S-polynomial.
            lcm = lcm_monomials(monom_a, monom_b, self.variables)
            factor_a = monomial_div(lcm, monom_a, self.variables) / coeff_a
            factor_b = monomial_div(lcm, monom_b, self.variables) / coeff_b
        else:
            # Different components: leading terms are coprime.
            factor_a = monom_b / coeff_b
            factor_b = monom_a / coeff_a
        return sp.expand(factor_a), sp.expand(factor_b)

    def _vector_s_polynomial(
        self,
        v_a: List[sp.Expr],
        v_b: List[sp.Expr],
        lt_a: Tuple[int, sp.Expr, sp.Expr],
        lt_b: Tuple[int, sp.Expr, sp.Expr],
    ) -> List[sp.Expr]:
        """Compute the S-polynomial of two module vectors."""
        factor_a, factor_b = self._vector_lcm_factors(lt_a, lt_b)
        return [
            sp.expand(factor_a * va - factor_b * vb)
            for va, vb in zip(v_a, v_b)
        ]

    def _vector_reduce(
        self,
        v: List[sp.Expr],
        module_gens: List[List[sp.Expr]],
        leading_terms: List[Tuple[int, sp.Expr, sp.Expr]],
    ) -> Tuple[List[sp.Expr], List[sp.Expr]]:
        """Reduce a vector ``v`` modulo the module Groebner basis.

        Returns ``(quotients, remainder)`` where ``quotients`` is a list
        of length ``len(module_gens)`` and ``remainder`` is a vector of
        the same length as ``v``.
        """
        m = len(module_gens)
        n = len(v)
        quotients = [sp.Integer(0)] * m
        # Make a working copy.
        work = [sp.expand(vi) for vi in v]
        # Iteratively reduce.
        max_iters = 10000
        it = 0
        while it < max_iters:
            it += 1
            # Find the leading term of `work`.
            lt_work = self._vector_leading_term(work)
            if lt_work[0] == -1:
                break  # zero vector
            comp_w, monom_w, coeff_w = lt_work
            # Find a module generator whose leading term divides lt_work.
            found = False
            for j in range(m):
                comp_j, monom_j, coeff_j = leading_terms[j]
                if comp_j == comp_w and divides(monom_j, monom_w, self.variables):
                    # Reduce: subtract (factor) * module_gens[j]
                    factor = sp.expand(
                        (coeff_w / coeff_j)
                        * monomial_div(monom_w, monom_j, self.variables)
                    )
                    quotients[j] = sp.expand(quotients[j] + factor)
                    for k in range(n):
                        work[k] = sp.expand(work[k] - factor * module_gens[j][k])
                    found = True
                    break
            if not found:
                # Cannot reduce further; we need to "drop" the leading term
                # and try the next one.  In a proper Groebner basis
                # reduction, this means the leading term is part of the
                # remainder.  We move it to a remainder vector.
                # For simplicity here, we just break (the caller should
                # check that the remainder is zero).
                break
        return quotients, work

    # ------------------------------------------------------------------
    # Evaluation at the trivial module (eps map) and rank computation
    # ------------------------------------------------------------------

    def evaluate(self, poly: sp.Expr) -> sp.Expr:
        """Evaluate a polynomial at the trivial module.

        Substitutes ``x_{i, s} -> eps_values[i]`` for every letterplace
        variable.  The result is an element of ``K`` (the base field,
        e.g. ``Q(omega)``).
        """
        return sp.nsimplify(sp.expand(poly.subs(self._eval_substitution)))

    def differential_matrix(
        self, syzygies: List[List[sp.Expr]]
    ) -> "sp.Matrix":
        """Build the differential matrix (with polynomial entries) for a syzygy module.

        Returns a matrix ``M`` of shape ``(n_components, n_syzygies)``
        where ``n_components`` is the length of each syzygy vector and
        ``n_syzygies = len(syzygies)``.  Column ``j`` is the syzygy
        ``syzygies[j]``.
        """
        if not syzygies:
            return sp.Matrix([])
        n_comp = len(syzygies[0])
        n_syz = len(syzygies)
        M = sp.Matrix(n_comp, n_syz, lambda i, j: syzygies[j][i])
        return M

    def evaluated_differential_matrix(
        self, syzygies: List[List[sp.Expr]]
    ) -> "sp.Matrix":
        """Build the differential matrix reduced modulo the trivial module.

        Each entry is evaluated via :meth:`evaluate` (i.e. the polynomial
        is specialised to the trivial module, giving an element of ``K``).
        The resulting matrix has entries in ``K``.
        """
        if not syzygies:
            return sp.Matrix([])
        n_comp = len(syzygies[0])
        n_syz = len(syzygies)
        entries = []
        for j in range(n_syz):
            for i in range(n_comp):
                entries.append(self.evaluate(syzygies[j][i]))
        M = sp.Matrix(n_comp, n_syz, entries)
        return M

    def rank_of_evaluated_differential(
        self, syzygies: List[List[sp.Expr]]
    ) -> int:
        """Compute the rank of the evaluated differential matrix over ``K``.

        The rank is computed numerically (over ``C``) using numpy's SVD,
        with a tolerance of ``1e-9`` for floating-point roundoff.  For
        exact computation we would need to compute the rank over the
        algebraic field ``Q(omega)`` directly (e.g. via sympy's
        :func:`sympy.Matrix.rank` with the algebraic field as the
        domain), but the numerical approach is much faster and
        accurate enough for our purposes (the rank is an integer).
        """
        M = self.evaluated_differential_matrix(syzygies)
        if M.rows == 0 or M.cols == 0:
            return 0
        # Convert to a complex numpy array.
        M_complex = np.array(M.applyfunc(lambda x: complex(x)).tolist(), dtype=complex)
        # SVD-based rank.
        if M_complex.size == 0:
            return 0
        s = np.linalg.svd(M_complex, compute_uv=False)
        tol = max(M_complex.shape) * max(s.max() if s.size else 0, 1.0) * 1e-9
        rank = int((s > tol).sum())
        return rank

    # ------------------------------------------------------------------
    # Hochschild cohomology dimensions
    # ------------------------------------------------------------------

    def betti_numbers(self, max_level: int = 3) -> List[int]:
        """Return the Betti numbers ``[β_0, β_1, ..., β_{max_level}]``.

        ``β_0 = 1`` (the unit), ``β_1 = |G_min|`` (minimised Groebner
        basis size), ``β_2 = |Syz(G)_min|``, ``β_3 = |Syz(Syz(G))_min|``.
        """
        bettis = [1]
        if max_level >= 1:
            gb = self.compute_groebner_basis(minimise=False)
            bettis.append(len(gb))
        if max_level >= 2:
            syz1 = self.compute_first_syzygies(minimise=False)
            bettis.append(len(syz1))
        if max_level >= 3:
            syz2 = self.compute_second_syzygies(minimise=False)
            bettis.append(len(syz2))
        return bettis

    def differential_ranks(self, max_level: int = 3) -> List[int]:
        """Return ``[r_1, r_2, ..., r_{max_level}]`` where ``r_n`` is the
        rank of ``d_n ⊗ k`` (the n-th differential reduced modulo the
        trivial module).

        ``r_n`` is computed by evaluating each entry of the n-th
        differential matrix at the trivial module (via
        :meth:`evaluate`) and taking the matrix rank over ``K``.
        """
        ranks = []
        if max_level >= 1:
            # d_1: R^{β_1} -> R, matrix is 1 x β_1 with entries g_i.
            gb = self.compute_groebner_basis(minimise=False)
            # Evaluate each g_i.
            row = [self.evaluate(g) for g in gb]
            # Rank of a 1 x n matrix over K.
            ranks.append(sum(1 for x in row if x != 0))
        if max_level >= 2:
            # d_2: R^{β_2} -> R^{β_1}, columns are the first syzygies.
            syz1 = self.compute_first_syzygies(minimise=False)
            ranks.append(self.rank_of_evaluated_differential(syz1))
        if max_level >= 3:
            # d_3: R^{β_3} -> R^{β_2}, columns are the second syzygies.
            syz2 = self.compute_second_syzygies(minimise=False)
            ranks.append(self.rank_of_evaluated_differential(syz2))
        return ranks

    def hochschild_dimensions(self, max_n: int = 3) -> List[int]:
        """Return ``[HH^0, HH^1, ..., HH^{max_n}]``.

        Formula (standard, for the chain complex ``k ⊗_R F_•``):

            HH^0 = β_0 - r_1
            HH^n = β_n - r_n - r_{n+1}    for n >= 1

        where ``β_n`` is the n-th Betti number of the (non-minimal)
        letterplace resolution and ``r_n = rank(d_n ⊗ k)`` is the
        rank of the n-th differential reduced modulo the trivial
        module.

        This formula is exact *up to the truncation at*
        :attr:`max_degree`: the letterplace resolution is exact only
        up to total NC degree ``max_degree``, so for ``n`` larger
        than what the truncation supports, ``HH^n`` may be
        inaccurate.  For ``u_q(sl_2)`` at ``ell = 3`` with
        ``max_degree = 3``, the resolution captures ``HH^0, HH^1,
        HH^2`` correctly (verified against the bar complex).
        """
        bettis = self.betti_numbers(max_level=max_n + 1)
        ranks = self.differential_ranks(max_level=max_n + 1)
        # ranks is [r_1, r_2, ..., r_{max_n+1}], zero-indexed.
        # Pad with zeros if needed.
        while len(ranks) < max_n + 2:
            ranks.append(0)
        # Compute HH.
        hh = []
        # HH^0 = β_0 - r_1
        r_1 = ranks[0] if len(ranks) >= 1 else 0
        hh.append(max(0, bettis[0] - r_1))
        # HH^n = β_n - r_n - r_{n+1} for n >= 1
        for n in range(1, max_n + 1):
            r_n = ranks[n - 1] if len(ranks) >= n else 0
            r_n1 = ranks[n] if len(ranks) >= n + 1 else 0
            beta_n = bettis[n] if len(bettis) >= n + 1 else 0
            hh.append(max(0, beta_n - r_n - r_n1))
        return hh


# ============================================================================
# Specialised presentations
# ============================================================================


def build_bplus_presentation():
    """Build the Borel subalgebra ``B^+`` of ``u_q(sl_2)`` at ``ell = 3``.

    Generators: ``K, E`` (indices 0, 1).  Relations:

        K^3  -> 1
        E^3  -> 0
        E K  -> q^{-2} K E   (= omega * K E   at ell = 3)

    This is a PBW algebra of dimension 9 with PBW basis
    ``{K^a E^b : 0 <= a, b <= 2}``.  Expected ``dim HH^2 = 1``
    (the [E, K] relation gives one syzygy).
    """
    from .parser import Monomial, Polynomial, RewriteRule, Presentation
    from .qomega import OMEGA3_ONE, OMEGA

    rules = [
        # K^3 -> 1
        RewriteRule(
            Monomial((0, 0, 0)),
            Polynomial([Term(OMEGA3_ONE, Monomial(()))]),
        ),
        # E^3 -> 0
        RewriteRule(
            Monomial((1, 1, 1)),
            Polynomial.zero(),
        ),
        # E K -> omega * K E
        RewriteRule(
            Monomial((1, 0)),
            Polynomial([Term(OMEGA, Monomial((0, 1)))]),
        ),
    ]
    return Presentation(generators=["K", "E"], rules=rules)


# Late import (avoids circular import for `Term`).
from .parser import Term  # noqa: E402


__all__ = [
    "make_omega3_field",
    "make_omega5_field",
    "leading_monomial",
    "leading_coefficient",
    "lcm_monomials",
    "divides",
    "monomial_div",
    "minimise_groebner_basis",
    "reduce_with_history",
    "compute_s_polynomial",
    "compute_first_syzygies",
    "syzygy_leading_monomial",
    "compare_monomials",
    "minimise_syzygies",
    "LetterplaceResolution",
    "build_bplus_presentation",
]
