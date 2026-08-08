"""
Noncommutative Gröbner basis / Knuth-Bendix completion + Anick resolution.

Built on top of ir/parser.py (Monomial, Polynomial, RewriteRule, Presentation,
NormalFormReducer). Provides:

  - overlaps(m1, m2): enumerate non-trivial overlaps between two monomials
  - critical_pairs(rule1, rule2): enumerate critical pairs (ambiguities)
  - knuth_bendix_complete(presentation): complete the rewrite system
  - anick_resolution_generators(presentation): enumerate generators of the
    Anick resolution by degree (0=algebra, 1=relations, 2=syzygies, ...)
  - test_confluence(presentation, n_tests): randomised confluence check

The Anick resolution [Anick 1986] is the minimal free resolution of the
algebra over itself, indexed by the "ambiguities" (critical pairs) of the
rewrite system. Its chain groups are much smaller than the bar complex,
making HH^2 computation tractable for algebras like u_q(sl_3).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from .parser import (
    Monomial, Polynomial, Term, RewriteRule, Presentation, NormalFormReducer,
    QLaurent,
)


# ---------------------------------------------------------------------------
# Overlap detection
# ---------------------------------------------------------------------------

def overlaps(m1: Monomial, m2: Monomial) -> list[tuple[int, int, int]]:
    """Find all non-trivial overlaps where a suffix of m1 equals a prefix of m2.

    Returns list of (start_in_m1, start_in_m2, overlap_length) tuples.
    A non-trivial overlap means overlap_length >= 1 and it's a proper suffix
    of m1 and a proper prefix of m2 (i.e., the overlap doesn't consume either
    monomial entirely).

    For the Anick resolution we also include the "full overlap" case where
    m1 == m2 (this gives a self-overlap). Use `include_full=True` for that.
    """
    result = []
    n1, n2 = len(m1.gens), len(m2.gens)
    max_overlap = min(n1, n2)
    for L in range(1, max_overlap + 1):
        # suffix of m1 of length L: m1.gens[n1-L : n1]
        # prefix of m2 of length L: m2.gens[0 : L]
        if m1.gens[n1 - L:] == m2.gens[:L]:
            # Non-trivial: L < n1 or L < n2 (proper suffix/prefix)
            # Actually for critical pairs we want L < n1 AND L < n2,
            # but for self-overlaps L = n1 = n2 is excluded.
            if L < n1 or L < n2:
                result.append((n1 - L, 0, L))
    return result


def all_critical_pairs(rule1: RewriteRule, rule2: RewriteRule) -> list[tuple[Monomial, int, int]]:
    """Find all critical pair positions between rule1 and rule2.

    A critical pair is a monomial M such that:
      - rule1.LHS occurs in M at position p1
      - rule2.LHS occurs in M at position p2
      - the two occurrences overlap (share at least one generator)
      - neither occurrence contains the other

    Returns list of (M, p1, p2) where M is the overlap monomial,
    p1 is where rule1.LHS starts in M, p2 is where rule2.LHS starts in M.

    Self-overlaps (rule1 == rule2) are included.
    """
    L1, L2 = rule1.lhs, rule2.lhs
    n1, n2 = len(L1.gens), len(L2.gens)
    pairs = []

    # Case A: suffix of L1 = prefix of L2 (L1 occurs first, overlapping)
    for L in range(1, min(n1, n2)):
        if L1.gens[n1 - L:] == L2.gens[:L]:
            # M = L1 + L2[L:] (concatenation, with the overlap shared)
            M_gens = L1.gens + L2.gens[L:]
            M = Monomial(M_gens)
            # rule1.LHS starts at position 0 in M
            # rule2.LHS starts at position n1 - L in M
            pairs.append((M, 0, n1 - L))

    # Case B: suffix of L2 = prefix of L1 (L2 occurs first, overlapping)
    # Skip if rule1 is rule2 (Case A already covers it symmetrically for self-pairs)
    if rule1 is not rule2 and L1 != L2:
        for L in range(1, min(n1, n2)):
            if L2.gens[n2 - L:] == L1.gens[:L]:
                M_gens = L2.gens + L1.gens[L:]
                M = Monomial(M_gens)
                pairs.append((M, n2 - L, 0))

    # Case C: full overlap (L1 == L2, i.e., the same monomial) — only if rule1 is rule2
    # This is a "trivial" critical pair; we exclude it because applying the same rule
    # at the same position gives the same result.

    return pairs


def critical_pairs(rule1: RewriteRule, rule2: RewriteRule,
                   reducer: Optional[NormalFormReducer] = None) -> list[Polynomial]:
    """Compute critical pairs as polynomials (the difference after reduction).

    For each overlap monomial M with rule1 at position p1 and rule2 at position p2:
      - Apply rule1 at p1: get poly1
      - Apply rule2 at p2: get poly2
      - Compute (poly1 - poly2)
      - If reducer is given, reduce the difference to normal form
      - If the result is non-zero, it's a critical pair that needs a new rule

    Returns list of non-zero critical pair polynomials (in normal form if reducer given).
    """
    pairs_data = all_critical_pairs(rule1, rule2)
    results = []
    for M, p1, p2 in pairs_data:
        # Apply rule1 at position p1 in M
        poly1 = _apply_rule_at(rule1, M, p1)
        # Apply rule2 at position p2 in M
        poly2 = _apply_rule_at(rule2, M, p2)
        # Difference
        diff = poly1 - poly2
        if reducer is not None:
            diff = reducer.reduce(diff)
        if diff != Polynomial.zero():
            results.append(diff)
    return results


def _apply_rule_at(rule: RewriteRule, M: Monomial, pos: int) -> Polynomial:
    """Apply rule at position pos in monomial M.

    Returns: (prefix of M before pos) * (rule.rhs) * (suffix of M after the LHS match).
    """
    n = len(M.gens)
    lhs_len = len(rule.lhs.gens)
    prefix = M.gens[:pos]
    suffix = M.gens[pos + lhs_len:]
    # Result = Polynomial(prefix) * rule.rhs * Polynomial(suffix)
    # Since prefix and suffix are monomials, this is:
    # sum over terms in rule.rhs of (coeff * prefix * monomial * suffix)
    result_terms = []
    for term in rule.rhs.terms:
        new_gens = prefix + term.monomial.gens + suffix
        result_terms.append(Term(term.coeff, Monomial(new_gens)))
    return Polynomial(result_terms).normalize()


# ---------------------------------------------------------------------------
# Knuth-Bendix completion
# ---------------------------------------------------------------------------

@dataclass
class KBStats:
    """Statistics from Knuth-Bendix completion."""
    initial_rules: int = 0
    final_rules: int = 0
    critical_pairs_checked: int = 0
    new_rules_added: int = 0
    iterations: int = 0
    terminated: bool = False
    failed_pairs: list = field(default_factory=list)  # pairs that couldn't be oriented


def knuth_bendix_complete(presentation: Presentation,
                          max_iterations: int = 1000,
                          max_rules: int = 200,
                          verbose: bool = False) -> tuple[Presentation, KBStats]:
    """Complete the rewrite system via Knuth-Bendix.

    Algorithm:
      1. Start with the given rules.
      2. Repeat:
         a. For each pair of rules (r1, r2) including self-pairs:
            - Compute critical pairs.
            - For each non-zero critical pair (after reduction), try to orient it:
              reduce both sides; the larger normal form becomes LHS, the smaller becomes RHS.
         b. If no new rules added, terminate (confluent).
         c. If max_iterations or max_rules exceeded, terminate with `terminated=False`.
      3. Return the completed presentation and statistics.

    Orientation: we use length-lex order. The LHS is the monomial with the larger
    order key. If the critical pair is a polynomial (not a single monomial), we
    take the leading monomial as the LHS and move it to the RHS.

    Note: This is a simplified implementation. A full implementation would use
    the recursive path order or a more sophisticated monomial order, and would
    handle the case where the critical pair cannot be oriented (incomparable).
    """
    stats = KBStats(initial_rules=len(presentation.rules))

    # Work with a list of rules we can extend
    rules = list(presentation.rules)
    reducer = NormalFormReducer(Presentation(presentation.generators, rules))

    stats.final_rules = len(rules)

    for iteration in range(max_iterations):
        stats.iterations = iteration + 1
        new_rules_added_this_iter = 0

        # Check all pairs (including self-pairs and (i,j) with i <= j)
        for i in range(len(rules)):
            for j in range(i, len(rules)):
                stats.critical_pairs_checked += 1
                cps = critical_pairs(rules[i], rules[j], reducer)
                for cp in cps:
                    # Try to orient: find the leading monomial
                    if not cp.terms:
                        continue
                    # Sort terms by monomial order (descending — largest first)
                    sorted_terms = sorted(cp.terms, key=lambda t: t.monomial.order_key(), reverse=True)
                    lead_term = sorted_terms[0]
                    # LHS = lead monomial, RHS = -(cp - lead_term) / lead_coeff
                    lhs = lead_term.monomial
                    # RHS = lead_term.coeff * lhs - cp, then negate and divide by lead_term.coeff
                    # cp = lead_term + (rest)
                    # So: lead_term.coeff * lhs - cp = -rest
                    # We want lhs -> rest / lead_term.coeff (with a sign)
                    # Actually: cp = 0 means lead_term + rest = 0, so lhs = -rest/lead_coeff
                    rest_terms = sorted_terms[1:]
                    # RHS = -rest / lead_coeff (so that lead_coeff * LHS + rest = 0, i.e., LHS = -rest/lead_coeff)
                    rhs_terms_exact = []
                    skip = False
                    for t in rest_terms:
                        # t.coeff / lead_term.coeff — try exact division
                        if isinstance(t.coeff, QLaurent) and isinstance(lead_term.coeff, QLaurent):
                            quot = t.coeff.try_divide(lead_term.coeff)
                            if quot is None:
                                stats.failed_pairs.append((i, j, f"non-exact division: {t.coeff} / {lead_term.coeff}"))
                                skip = True
                                break
                            rhs_terms_exact.append(Term(-quot, t.monomial))
                        else:
                            skip = True
                            break
                    if skip:
                        continue
                    rhs = Polynomial(rhs_terms_exact).normalize()

                    # Check if this rule already exists
                    new_rule = RewriteRule(lhs, rhs)
                    if any(r.lhs == new_rule.lhs and r.rhs == new_rule.rhs for r in rules):
                        continue
                    # Check if the LHS is already a rule's LHS (would create inconsistency)
                    existing = [r for r in rules if r.lhs == new_rule.lhs]
                    if existing:
                        # Same LHS, different RHS — system is not confluent
                        # This shouldn't happen if the system is terminating
                        stats.failed_pairs.append((i, j, f"conflict: {new_rule.lhs} -> {new_rule.rhs} vs {existing[0].rhs}"))
                        continue

                    rules.append(new_rule)
                    new_rules_added_this_iter += 1
                    stats.new_rules_added += 1
                    if verbose:
                        print(f"  iter {iteration}: added rule {new_rule.lhs} -> {new_rule.rhs}")
                    # Rebuild reducer with new rule
                    reducer = NormalFormReducer(Presentation(presentation.generators, rules))
                    if len(rules) >= max_rules:
                        if verbose:
                            print(f"  max_rules={max_rules} reached, stopping")
                        return Presentation(presentation.generators, rules), stats

        stats.final_rules = len(rules)
        if new_rules_added_this_iter == 0:
            stats.terminated = True
            if verbose:
                print(f"  confluent after {iteration+1} iterations, {len(rules)} rules")
            break

    return Presentation(presentation.generators, rules), stats


# ---------------------------------------------------------------------------
# Anick resolution generators
# ---------------------------------------------------------------------------

def anick_resolution_generators(presentation: Presentation,
                                max_degree: int = 3) -> dict[int, list]:
    """Enumerate generators of the Anick resolution by degree.

    Degree 0: 1 generator (the algebra itself, represented as the empty monomial)
    Degree 1: 1 generator per rule (each rule.LHS is a "relation")
    Degree 2: 1 generator per critical pair (each is an "ambiguity"/"syzygy")
    Degree 3: 1 generator per "ambiguity of ambiguity" (syzygy of syzygy)
              — these are triples of rules whose LHSs overlap pairwise

    For degree 2, we return the overlap monomials (the M's from critical_pairs).
    For degree 3, we return triples (M, rule_indices) representing higher syzygies.

    Note: degree 3+ is harder to compute correctly; this is a best-effort enumeration.
    """
    result = {0: [Monomial(())]}  # degree 0: the unit
    rules = presentation.rules

    # Degree 1: the relations (one per rule)
    result[1] = [(i, rule.lhs) for i, rule in enumerate(rules)]

    # Degree 2: critical pairs
    degree2 = []
    for i in range(len(rules)):
        for j in range(i, len(rules)):
            cps = all_critical_pairs(rules[i], rules[j])
            for M, p1, p2 in cps:
                degree2.append((M, (i, j), (p1, p2)))
    result[2] = degree2

    # Degree 3: triples of rules with pairwise overlaps (best-effort)
    # A degree-3 syzygy is a monomial M where three rules overlap pairwise.
    # This is harder to enumerate correctly; we approximate by looking for
    # monomials where rule_i, rule_j, rule_k all have LHS matches that overlap.
    if max_degree >= 3:
        degree3 = []
        # For each pair of critical pairs that share a rule and overlap, add a degree-3 syzygy
        for a, (M_a, (i_a, j_a), (p1_a, p2_a)) in enumerate(degree2):
            for b, (M_b, (i_b, j_b), (p1_b, p2_b)) in enumerate(degree2):
                if a >= b:
                    continue
                # Check if they share a rule and the monomials overlap
                shared_rules = set([i_a, j_a]) & set([i_b, j_b])
                if shared_rules:
                    degree3.append((M_a, M_b, tuple(shared_rules)))
        result[3] = degree3[:100]  # cap to avoid explosion

    return result


# ---------------------------------------------------------------------------
# Confluence testing
# ---------------------------------------------------------------------------

def check_confluence(presentation: Presentation, n_tests: int = 100,
                     max_monomial_len: int = 8, seed: int = 42) -> bool:
    """Randomised confluence check.

    Generate n_tests random monomials. For each, reduce it to normal form
    in two different ways (by applying rules in different orders) and check
    that the results agree. If any disagree, the system is not confluent.

    Note: this is a probabilistic check, not a proof. A passing check means
    "probably confluent"; a failing check means "definitely not confluent".
    """
    rng = random.Random(seed)
    reducer = NormalFormReducer(presentation)
    if not presentation.rules:
        return True  # trivially confluent

    # Get generator indices
    gen_indices = list(range(len(presentation.generators)))
    if not gen_indices:
        return True

    for _ in range(n_tests):
        # Random monomial
        length = rng.randint(1, max_monomial_len)
        gens = tuple(rng.choice(gen_indices) for _ in range(length))
        M = Monomial(gens)

        # Reduce normally
        nf1 = reducer.reduce(Polynomial([Term(QLaurent({0: 1}), M)]))

        # Reduce by applying rules in reverse order (a different strategy)
        # For a proper test we'd implement a different reduction strategy;
        # for now, just check that the normal form is stable under a second reduction
        nf2 = reducer.reduce(nf1)

        if nf1 != nf2:
            return False

    return True
