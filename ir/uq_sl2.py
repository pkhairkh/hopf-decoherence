"""``u_q(sl_2)`` at ``ell = 3`` in the AST/IR framework (task W1-1a).

This module builds the small quantum group ``u_q(sl_2)`` at ``ell = 3`` as
a noncommutative rewrite presentation in the IR framework, runs
Knuth-Bendix completion, verifies the PBW basis, counts the Anick
resolution generators at degree 2, and computes ``dim HH^2`` via the bar
complex on PBW normal forms (homotopy-equivalent to the Anick
resolution, hence computing the same Hochschild cohomology).

Presentation
------------
Generators: ``K, E, F`` (indices 0, 1, 2).  Relations, oriented by
length-lex to the PBW form ``K^a E^b F^c`` with ``0 <= a, b, c <= 2``::

    R1: K^3  -> 1                                    (K has order 3)
    R2: E^3  -> 0                                    (E nilpotent)
    R3: F^3  -> 0                                    (F nilpotent)
    R4: E K  -> q^{-2} K E       (= omega   * K E   at ell = 3)
    R5: F K  -> q^2   K F       (= omega^2 * K F   at ell = 3)
    R6: F E  -> E F + (K - K^2) / (q - q^{-1})
                              (= E F + alpha*K - alpha*K^2
                                 with alpha = 1/(q - q^{-1})
                                          = (-1 - 2*omega)/3  at ell = 3)

The rewrite system is confluent at ``ell = 3``: every critical pair
reduces to zero in ``Q(omega)`` (verified by Knuth-Bendix completion,
which terminates without adding new rules).  The PBW basis
``{K^a E^b F^c : 0 <= a, b, c <= 2}`` gives 27 normal forms, matching
``dim_C u_q(sl_2) = 27``.

HH^2 computation
-----------------
For a Hopf algebra ``A`` there is a canonical isomorphism
``HH^*(A, k) ~= Ext^*_A(k, k)`` (Cartan-Eilenberg).  Both the bar
complex on PBW normal forms and the Anick resolution compute
``Ext^*_A(k, k)``, hence compute the same ``HH^*``.  This module uses
the bar complex on PBW normal forms because the bar differential is
straightforward to implement; the Anick resolution's chain groups are
reported (and the degree-2 syzygy count is verified) but the Anick
*differential* is left for a future task.  The two resolutions are
homotopy-equivalent, so the resulting ``dim HH^2`` is the same either
way.

Expected result: ``dim_C HH^2(u_q(sl_2), C) = 3``, matching the bar
complex in ``scripts/verify_sl2_hh2.py`` and the conjecture
``dim HH^2 = C(n+1, 2) + 2*|Phi^+| = 1 + 2 = 3`` for ``A_1``.
"""
from __future__ import annotations

import cmath
import math
import random
from fractions import Fraction
from typing import Optional

import numpy as np
from scipy import sparse

from .parser import (
    Monomial,
    Polynomial,
    Term,
    RewriteRule,
    Presentation,
    NormalFormReducer,
    ONE,
    ZERO,
    Q,
    Q_INV,
    QLaurent,
    qpow,
    qint,
)
from .groebner import (
    all_critical_pairs,
    critical_pairs,
    knuth_bendix_complete,
    anick_resolution_generators,
    check_confluence,
    KBStats,
)
from .qomega import (
    QOmega3,
    FRACT_ZERO,
    FRACT_ONE,
    OMEGA3_ZERO,
    OMEGA3_ONE,
    OMEGA,
    OMEGA2,
    Q_MINUS_Q_INV,
    Q_MINUS_Q_INV_INV,
)


# ============================================================================
# Constants at ell = 3
# ============================================================================

ELL: int = 3
DIM: int = ELL ** 3  # = 27, dimension of u_q(sl_2) at ell = 3.

# At ell = 3 with q = omega = e^{2*pi*i/3}:
#   q     = omega
#   q^2   = omega^2 = -1 - omega
#   q^3   = 1
#   q^{-1} = q^2 = omega^2
#   q^{-2} = q   = omega
#   q - q^{-1} = omega - omega^2 = 1 + 2*omega
Q_VAL: QOmega3 = OMEGA
Q_INV_VAL: QOmega3 = OMEGA2
Q2_VAL: QOmega3 = OMEGA2  # q^2
Q_INV2_VAL: QOmega3 = OMEGA  # q^{-2}

# alpha = 1/(q - q^{-1}) = (-1 - 2*omega)/3
ALPHA: QOmega3 = Q_MINUS_Q_INV_INV


# ============================================================================
# Presentation
# ============================================================================


def build_uq_sl2_presentation() -> Presentation:
    """Build the ``u_q(sl_2)`` presentation at ``ell = 3``.

    Returns a :class:`Presentation` with generators ``["K", "E", "F"]``
    (indices 0, 1, 2) and 6 rewrite rules oriented to PBW form
    ``K^a E^b F^c``.  Coefficients are :class:`QOmega3` (elements of
    ``Q(omega)``).
    """
    # Generator indices: K = 0, E = 1, F = 2.
    rules = [
        # R1: K^3 -> 1
        RewriteRule(
            Monomial((0, 0, 0)),
            Polynomial([Term(OMEGA3_ONE, Monomial(()))]),
        ),
        # R2: E^3 -> 0
        RewriteRule(
            Monomial((1, 1, 1)),
            Polynomial.zero(),
        ),
        # R3: F^3 -> 0
        RewriteRule(
            Monomial((2, 2, 2)),
            Polynomial.zero(),
        ),
        # R4: E K -> q^{-2} K E = omega * K E
        RewriteRule(
            Monomial((1, 0)),
            Polynomial([Term(Q_INV2_VAL, Monomial((0, 1)))]),
        ),
        # R5: F K -> q^2 K F = omega^2 * K F
        RewriteRule(
            Monomial((2, 0)),
            Polynomial([Term(Q2_VAL, Monomial((0, 2)))]),
        ),
        # R6: F E -> E F - alpha * K + alpha * K^2
        #     where alpha = 1/(q - q^{-1}) = (-1 - 2*omega)/3.
        #
        # Derivation: the standard relation is [E, F] = (K - K^{-1})/(q - q^{-1}).
        # At ell = 3, K^{-1} = K^2, so EF - FE = (K - K^2) * alpha.
        # Solving for FE: FE = EF - alpha*(K - K^2) = EF - alpha*K + alpha*K^2.
        # The rewrite rule orients FE -> (PBW normal form), which is the RHS above.
        RewriteRule(
            Monomial((2, 1)),
            Polynomial([
                Term(OMEGA3_ONE, Monomial((1, 2))),       # E F
                Term(-ALPHA, Monomial((0,))),              # - alpha * K
                Term(ALPHA, Monomial((0, 0))),             # + alpha * K^2
            ]),
        ),
    ]
    return Presentation(generators=["K", "E", "F"], rules=rules)


# ============================================================================
# PBW basis
# ============================================================================


def pbw_basis() -> list:
    """Return the 27 PBW normal-form monomials ``K^a E^b F^c``.

    The list is ordered by ``a`` (outer), then ``b``, then ``c``, so the
    index of ``K^a E^b F^c`` in the list is ``a * 9 + b * 3 + c``.
    """
    basis = []
    for a in range(ELL):
        for b in range(ELL):
            for c in range(ELL):
                gens = (0,) * a + (1,) * b + (2,) * c
                basis.append(Monomial(gens))
    return basis


def pbw_index(a: int, b: int, c: int) -> int:
    """Linear index of ``K^a E^b F^c`` in :func:`pbw_basis`."""
    return a * ELL * ELL + b * ELL + c


def pbw_from_index(i: int) -> tuple:
    """Inverse of :func:`pbw_index`."""
    return i // (ELL * ELL), (i // ELL) % ELL, i % ELL


def verify_pbw_basis(pres: Presentation, n_random: int = 200,
                     seed: int = 42) -> dict:
    """Verify the PBW basis property of the rewrite system.

    Returns a dict with keys:

    - ``'pbw_size'``: 27 (the expected number of PBW normal forms).
    - ``'pbw_all_normal'``: True iff every PBW monomial is in normal form
      (no rule matches it).
    - ``'random_reductions_in_pbw'``: True iff every monomial in every
      normal form of ``n_random`` random monomials is in the PBW basis.
    - ``'random_sample'``: a list of ``(input, normal_form)`` pairs for
      inspection.
    - ``'unique_normal_forms'``: True iff distinct PBW monomials have
      distinct normal forms (trivially true since they ARE their own
      normal forms, but verifies the count).
    """
    reducer = NormalFormReducer(pres)
    basis = pbw_basis()
    pbw_set = set(basis)

    # (1) Every PBW monomial must be in normal form.
    pbw_all_normal = True
    for m in basis:
        if reducer.find_match(m) is not None:
            pbw_all_normal = False
            break

    # (2) Random monomials should reduce to a polynomial whose monomials
    #     are all in the PBW basis.
    rng = random.Random(seed)
    random_sample = []
    random_reductions_in_pbw = True
    for _ in range(n_random):
        length = rng.randint(0, 8)
        gens = tuple(rng.choice([0, 1, 2]) for _ in range(length))
        m = Monomial(gens)
        nf = reducer.normal_form(m)
        for t in nf.terms:
            if t.monomial not in pbw_set:
                random_reductions_in_pbw = False
                break
        random_sample.append((m, nf))

    return {
        "pbw_size": len(basis),
        "pbw_all_normal": pbw_all_normal,
        "random_reductions_in_pbw": random_reductions_in_pbw,
        "random_sample": random_sample,
        "unique_normal_forms": pbw_all_normal,  # proxy
    }


# ============================================================================
# Knuth-Bendix completion
# ============================================================================


def run_kb_completion(pres: Presentation,
                      max_iterations: int = 50,
                      max_rules: int = 100,
                      verbose: bool = False) -> tuple:
    """Run Knuth-Bendix completion on the presentation.

    Returns ``(completed_pres, stats)``.  For ``u_q(sl_2)`` at ``ell = 3``
    we expect completion to terminate with no new rules added (the
    initial 6-rule system is already confluent).
    """
    return knuth_bendix_complete(
        pres,
        max_iterations=max_iterations,
        max_rules=max_rules,
        verbose=verbose,
    )


# ============================================================================
# Anick resolution: count degree-2 generators
# ============================================================================


def anick_degree0_count(pres: Presentation) -> int:
    """Count Anick degree-0 generators (always 1: the unit)."""
    return 1


def anick_degree1_count(pres: Presentation) -> int:
    """Count Anick degree-1 generators (one per relation)."""
    return len(pres.rules)


def anick_degree2_count(pres: Presentation) -> int:
    """Count Anick degree-2 generators (critical pairs / syzygies).

    For ``u_q(sl_2)`` at ``ell = 3`` with the 6-rule PBW rewrite system,
    the expected count is 13 (6 self-overlaps of the cubic rules +
    7 cross-rule overlaps; see the analysis note).
    """
    gens = anick_resolution_generators(pres, max_degree=2)
    return len(gens[2])


def anick_degree2_listing(pres: Presentation) -> list:
    """Return the list of Anick degree-2 generators (syzygies).

    Each entry is ``(M, (i, j), (p1, p2))`` where ``M`` is the overlap
    monomial, ``i, j`` are the rule indices, and ``p1, p2`` are the
    positions where the rules match inside ``M``.
    """
    gens = anick_resolution_generators(pres, max_degree=2)
    return list(gens[2])


# ============================================================================
# Multiplication table in the PBW basis (via the IR normal-form reducer)
# ============================================================================


def build_multiplication_table(pres: Presentation) -> np.ndarray:
    """Build the 27 x 27 x 27 multiplication table in the PBW basis.

    ``mult[k, i, j]`` is the coefficient of ``basis[k]`` in the product
    ``basis[i] * basis[j]`` (after reduction to normal form).
    Coefficients are stored as complex numbers (QOmega3 -> complex via
    :meth:`QOmega3.to_complex`).

    Raises :class:`ValueError` if any product reduces to a monomial that
    is not in the PBW basis (i.e. the rewrite system is not confluent
    with the PBW basis as normal forms).
    """
    reducer = NormalFormReducer(pres)
    basis = pbw_basis()
    basis_idx = {m: i for i, m in enumerate(basis)}

    mult = np.zeros((DIM, DIM, DIM), dtype=complex)
    for i in range(DIM):
        for j in range(DIM):
            # The product basis[i] * basis[j] in the free algebra is just
            # the concatenation basis[i].gens + basis[j].gens.
            m = Monomial(basis[i].gens + basis[j].gens)
            nf = reducer.normal_form(m)
            for t in nf.terms:
                k = basis_idx.get(t.monomial)
                if k is None:
                    raise ValueError(
                        f"Product basis[{i}] * basis[{j}] reduced to "
                        f"monomial {t.monomial} which is not in the PBW "
                        f"basis.  Rewrite system is not PBW-confluent."
                    )
                # Coerce coefficient to QOmega3 (handles the case where the
                # reducer's "no-match" branch returns a generic QLaurent ONE).
                coeff = t.coeff
                if not isinstance(coeff, QOmega3):
                    coeff = QOmega3(dict(coeff.terms))
                mult[k, i, j] = coeff.to_complex()
    return mult


def sanity_check_multiplication(mult: np.ndarray) -> dict:
    """Verify the multiplication table against the defining relations.

    Checks:
      1. ``1 * x = x * 1 = x`` (unit).
      2. ``K^3 = 1``.
      3. ``E^3 = 0``, ``F^3 = 0`` (nilpotence).
      4. ``K E = q^2 E K`` (i.e. ``K E K^{-1} = q^2 E``;
         equivalently ``mult[K*E, i, j]`` matches).
      5. ``K F = q^{-2} F K``.
      6. ``[E, F] = (K - K^{-1})/(q - q^{-1})``.

    Returns a dict of check-name -> bool.
    """
    omega = cmath.exp(2j * cmath.pi / ELL)
    # Sanity constants
    q = omega
    q_inv = omega ** (-1)
    q2 = omega ** 2
    q_inv2 = omega ** (-2)
    d = q - q_inv  # = 1 + 2*omega (purely imaginary for ell = 3)
    tol = 1e-9

    e0 = pbw_index(0, 0, 0)  # 1
    K = pbw_index(1, 0, 0)
    K2 = pbw_index(2, 0, 0)
    E = pbw_index(0, 1, 0)
    E2 = pbw_index(0, 2, 0)
    F = pbw_index(0, 0, 1)
    F2 = pbw_index(0, 0, 2)
    KE = pbw_index(1, 1, 0)  # K E
    KF = pbw_index(1, 0, 1)  # K F

    checks = {}

    # 1) Unit: 1 * x = x * 1 = x.
    checks["unit"] = True
    for i in range(DIM):
        if abs(mult[i, e0, i] - 1.0) > tol or abs(mult[i, i, e0] - 1.0) > tol:
            checks["unit"] = False
            break

    # 2) K^3 = 1 (K * K^2 = 1, K^2 * K = 1).
    checks["K^3 = 1"] = (
        abs(mult[e0, K, K2] - 1.0) < tol and
        abs(mult[e0, K2, K] - 1.0) < tol
    )

    # 3) E^3 = 0 (E^2 * E = 0).
    checks["E^3 = 0"] = all(
        abs(mult[i, E2, E]) < tol for i in range(DIM)
    )
    # F^3 = 0
    checks["F^3 = 0"] = all(
        abs(mult[i, F2, F]) < tol for i in range(DIM)
    )

    # 4) K E = q^2 E K.  Here K E is basis[KE] = K^1 E^1, and E K reduces
    #    to q^{-2} K E (rule R4 with our orientation), so
    #    mult[KE, K, E] = 1 and mult[KE, E, K] = q^{-2}.
    checks["K E = q^2 E K"] = (
        abs(mult[KE, K, E] - 1.0) < tol and
        abs(mult[KE, E, K] - q_inv2) < tol
    )

    # 5) K F = q^{-2} F K.  Rule R5 orients as F K -> q^2 K F, so
    #    mult[KF, K, F] = 1 and mult[KF, F, K] = q^2.
    checks["K F = q^{-2} F K"] = (
        abs(mult[KF, K, F] - 1.0) < tol and
        abs(mult[KF, F, K] - q2) < tol
    )

    # 6) [E, F] = (K - K^{-1})/(q - q^{-1}) = (K - K^2)/d.
    EF_minus_FE = mult[:, E, F] - mult[:, F, E]
    expected = np.zeros(DIM, dtype=complex)
    expected[K] = 1.0 / d
    expected[K2] = -1.0 / d
    checks["[E, F] = (K - K^{-1})/(q - q^{-1})"] = bool(
        np.max(np.abs(EF_minus_FE - expected)) < tol
    )

    return checks


# ============================================================================
# Bar complex on PBW normal forms: compute dim HH^2
# ============================================================================


def build_epsilon() -> np.ndarray:
    """Counit ``eps: u_q(sl_2) -> C``.

    ``eps(K^a E^b F^c) = 1`` if ``b = c = 0``, else 0.  (Equivalently,
    ``eps(K) = 1``, ``eps(E) = eps(F) = 0``, extended as an algebra map.)
    """
    epsilon = np.zeros(DIM, dtype=complex)
    for a in range(ELL):
        epsilon[pbw_index(a, 0, 0)] = 1.0
    return epsilon


def build_d1(mult: np.ndarray, epsilon: np.ndarray) -> np.ndarray:
    """Hochschild differential ``d^1: C^1 -> C^2``.

    ``(d^1 f)(a, b) = eps(a) f(b) - f(a*b) + f(a) eps(b)``.

    Shape: ``(DIM^2, DIM)``.
    """
    n2 = DIM * DIM
    d1 = np.zeros((n2, DIM), dtype=complex)
    for i in range(DIM):
        # f = indicator on basis[i]
        for a in range(DIM):
            for b in range(DIM):
                row = a * DIM + b
                term1 = epsilon[a] * (1.0 if b == i else 0.0)
                term2 = mult[i, a, b]  # coefficient of basis[i] in a*b
                term3 = (1.0 if a == i else 0.0) * epsilon[b]
                d1[row, i] = term1 - term2 + term3
    return d1


def build_d2_sparse(mult: np.ndarray, epsilon: np.ndarray):
    """Hochschild differential ``d^2: C^2 -> C^3``.

    ``(d^2 g)(a, b, c) = eps(a) g(b, c) - g(a*b, c) + g(a, b*c) - g(a, b) eps(c)``.

    Shape: ``(DIM^3, DIM^2)``.  Returned as a SciPy CSR sparse matrix.
    """
    n2 = DIM * DIM
    n3 = DIM ** 3
    rows, cols, vals = [], [], []
    for a in range(DIM):
        for b in range(DIM):
            for c in range(DIM):
                row = a * DIM * DIM + b * DIM + c
                # Term 1: eps(a) * g(b, c)
                ea = epsilon[a]
                if abs(ea) > 1e-13:
                    rows.append(row)
                    cols.append(b * DIM + c)
                    vals.append(ea)
                # Term 2: - g(a*b, c)  -- (a*b) = sum_k mult[k, a, b] basis[k]
                for k in range(DIM):
                    v = mult[k, a, b]
                    if abs(v) > 1e-13:
                        rows.append(row)
                        cols.append(k * DIM + c)
                        vals.append(-v)
                # Term 3: + g(a, b*c)  -- (b*c) = sum_k mult[k, b, c] basis[k]
                for k in range(DIM):
                    v = mult[k, b, c]
                    if abs(v) > 1e-13:
                        rows.append(row)
                        cols.append(a * DIM + k)
                        vals.append(v)
                # Term 4: - g(a, b) * eps(c)
                ec = epsilon[c]
                if abs(ec) > 1e-13:
                    rows.append(row)
                    cols.append(a * DIM + b)
                    vals.append(-ec)
    return sparse.csr_matrix(
        (vals, (rows, cols)), shape=(n3, n2), dtype=complex
    )


def matrix_rank_complex(mat: np.ndarray, tol: float = 1e-9) -> tuple:
    """Rank of a dense complex matrix via SVD.

    Returns ``(rank, singular_values)``.
    """
    s = np.linalg.svd(mat, compute_uv=False)
    if len(s) == 0:
        return 0, np.array([])
    s_max = s[0]
    return int(np.sum(s > tol * s_max)), s


def sparse_rank_via_gram(mat_csr, tol: float = 1e-9) -> tuple:
    """Rank of a tall sparse matrix via the (small) Gram matrix.

    ``rank(A) = rank(A^* A)``.  ``A^* A`` is ``n_cols x n_cols`` and
    Hermitian, so we use ``np.linalg.eigvalsh`` on its dense form.

    Returns ``(rank, eigenvalues_descending)``.
    """
    n = mat_csr.shape[1]
    gram = (mat_csr.conj().T @ mat_csr).toarray()
    eigvals = np.linalg.eigvalsh(gram)  # ascending real eigenvalues
    eigvals = np.sort(np.abs(eigvals))[::-1]  # descending abs
    if eigvals[0] == 0:
        return 0, eigvals
    return int(np.sum(eigvals > tol * eigvals[0])), eigvals


def compute_hh2(pres: Presentation,
                verbose: bool = False) -> dict:
    """Compute ``dim HH^2(u_q(sl_2), C)`` via the bar complex on PBW forms.

    Pipeline:

    1. Build the 27 x 27 x 27 multiplication table by reducing products
       of PBW monomials via the IR normal-form reducer.
    2. Build the Hochschild differentials ``d^1`` and ``d^2``.
    3. Compute ``rank(d^1)`` (dense SVD) and ``rank(d^2)`` (Gram matrix).
    4. ``dim HH^2 = dim ker(d^2) - dim im(d^1) = (DIM^2 - rank(d^2)) - rank(d^1)``.

    Returns a dict with all intermediate quantities.
    """
    # (1) Multiplication table.
    mult = build_multiplication_table(pres)
    if verbose:
        sanity = sanity_check_multiplication(mult)
        print("  Multiplication table sanity checks:")
        for k, v in sanity.items():
            print(f"    {k}: {v}")

    # (2) Counit and differentials.
    epsilon = build_epsilon()
    d1 = build_d1(mult, epsilon)
    d2 = build_d2_sparse(mult, epsilon)

    # (3) Ranks.
    rank_d1, sv_d1 = matrix_rank_complex(d1)
    rank_d2, ev_d2 = sparse_rank_via_gram(d2)

    # (4) dim HH^2.
    dim_c2 = DIM * DIM
    dim_ker_d2 = dim_c2 - rank_d2
    dim_hh2 = dim_ker_d2 - rank_d1

    return {
        "dim_uq_sl2": DIM,
        "dim_c1": DIM,
        "dim_c2": dim_c2,
        "dim_c3": DIM ** 3,
        "rank_d1": rank_d1,
        "rank_d2": rank_d2,
        "dim_ker_d2": dim_ker_d2,
        "dim_im_d1": rank_d1,
        "dim_hh2": dim_hh2,
        "sv_d1": sv_d1,
        "ev_d2": ev_d2,
    }


# ============================================================================
# Top-level entry point
# ============================================================================


def full_computation(verbose: bool = True) -> dict:
    """Run the full W1-1a computation: build presentation, verify PBW basis,
    run KB completion, count Anick degree-2 syzygies, and compute dim HH^2.

    Returns a dict with all results, and prints progress if ``verbose``.
    """
    results = {}

    # --- Build presentation ---
    if verbose:
        print("=" * 70)
        print("W1-1a: u_q(sl_2) at ell = 3 in the IR framework")
        print("=" * 70)
        print()
        print("Step 1: build presentation")
    pres = build_uq_sl2_presentation()
    results["presentation"] = pres
    if verbose:
        print(f"  generators: {pres.generators}")
        print(f"  rules ({len(pres.rules)}):")
        for i, r in enumerate(pres.rules):
            print(f"    R{i+1}: {r.lhs} -> {r.rhs}")
        print()

    # --- Verify PBW basis ---
    if verbose:
        print("Step 2: verify PBW basis")
    pbw_check = verify_pbw_basis(pres)
    results["pbw"] = pbw_check
    if verbose:
        print(f"  PBW basis size: {pbw_check['pbw_size']}")
        print(f"  All PBW monomials in normal form: {pbw_check['pbw_all_normal']}")
        print(f"  Random reductions all land in PBW: {pbw_check['random_reductions_in_pbw']}")
        print()

    # --- KB completion ---
    if verbose:
        print("Step 3: Knuth-Bendix completion")
    completed, kb_stats = run_kb_completion(pres, verbose=verbose)
    results["kb_stats"] = kb_stats
    results["completed_presentation"] = completed
    if verbose:
        print(f"  Initial rules: {kb_stats.initial_rules}")
        print(f"  Final rules:   {kb_stats.final_rules}")
        print(f"  New rules added: {kb_stats.new_rules_added}")
        print(f"  Critical pairs checked: {kb_stats.critical_pairs_checked}")
        print(f"  Terminated (confluent): {kb_stats.terminated}")
        print(f"  Failed pairs: {len(kb_stats.failed_pairs)}")
        print()

    # Use the completed presentation for subsequent steps (it's the same
    # as `pres` if completion added no rules).
    pres_eff = completed

    # --- Anick degree-2 count ---
    if verbose:
        print("Step 4: Anick resolution generators")
    n0 = anick_degree0_count(pres_eff)
    n1 = anick_degree1_count(pres_eff)
    n2 = anick_degree2_count(pres_eff)
    results["anick_degree0"] = n0
    results["anick_degree1"] = n1
    results["anick_degree2"] = n2
    if verbose:
        print(f"  Degree 0 (algebra):       {n0}")
        print(f"  Degree 1 (relations):     {n1}")
        print(f"  Degree 2 (syzygies):      {n2}")
        print()

    # --- dim HH^2 ---
    if verbose:
        print("Step 5: dim HH^2 via bar complex on PBW normal forms")
    hh2 = compute_hh2(pres_eff, verbose=verbose)
    results["hh2"] = hh2
    if verbose:
        print(f"  dim u_q(sl_2): {hh2['dim_uq_sl2']}")
        print(f"  dim C^1:       {hh2['dim_c1']}")
        print(f"  dim C^2:       {hh2['dim_c2']}")
        print(f"  dim C^3:       {hh2['dim_c3']}")
        print(f"  rank(d^1):     {hh2['rank_d1']}")
        print(f"  rank(d^2):     {hh2['rank_d2']}")
        print(f"  dim ker(d^2):  {hh2['dim_ker_d2']}")
        print(f"  dim im(d^1):   {hh2['dim_im_d1']}")
        print(f"  dim HH^2:      {hh2['dim_hh2']}")
        print()

    # --- Conclusion ---
    expected = 3  # C(2,2) + 2*|Phi^+(A_1)| = 1 + 2
    match = hh2["dim_hh2"] == expected
    results["expected_hh2"] = expected
    results["match"] = match
    if verbose:
        print("=" * 70)
        print(f"  dim HH^2 = {hh2['dim_hh2']}, expected = {expected}")
        print(f"  MATCH: {match}")
        print("=" * 70)

    return results


__all__ = [
    # Constants
    "ELL",
    "DIM",
    "ALPHA",
    # Presentation
    "build_uq_sl2_presentation",
    # PBW basis
    "pbw_basis",
    "pbw_index",
    "pbw_from_index",
    "verify_pbw_basis",
    # KB completion
    "run_kb_completion",
    # Anick
    "anick_degree0_count",
    "anick_degree1_count",
    "anick_degree2_count",
    "anick_degree2_listing",
    # Bar complex / HH^2
    "build_multiplication_table",
    "sanity_check_multiplication",
    "build_epsilon",
    "build_d1",
    "build_d2_sparse",
    "matrix_rank_complex",
    "sparse_rank_via_gram",
    "compute_hh2",
    # Top-level
    "full_computation",
]
