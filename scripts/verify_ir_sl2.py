#!/usr/bin/env python3
"""Verification script for W1-1a: u_q(sl_2) at ell = 3 in the IR framework.

Runs the full computation -- builds the u_q(sl_2) presentation, runs
Knuth-Bendix completion, verifies the PBW basis (27 normal forms),
counts the Anick resolution generators at degree 2, and computes
``dim HH^2(u_q(sl_2), C)`` via the bar complex on PBW normal forms.

Expected result: ``dim HH^2 = 3``, matching the bar complex in
``verify_sl2_hh2.py`` and the conjecture
``dim HH^2 = C(n+1, 2) + 2*|Phi^+| = 1 + 2 = 3`` for ``A_1``.

Run::

    python scripts/verify_ir_sl2.py
"""
from __future__ import annotations

import os
import sys

# Add the repo root to sys.path so we can import the ir/ package.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ir.uq_sl2 import (
    build_uq_sl2_presentation,
    verify_pbw_basis,
    run_kb_completion,
    anick_degree0_count,
    anick_degree1_count,
    anick_degree2_count,
    anick_degree2_listing,
    compute_hh2,
    sanity_check_multiplication,
    build_multiplication_table,
    ELL,
    DIM,
)
from ir.parser import NormalFormReducer
from ir.qomega import QOmega3, OMEGA, OMEGA2


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main() -> int:
    print("=" * 72)
    print(f"W1-1a: u_q(sl_2) at ell = {ELL} in the AST/IR framework")
    print("=" * 72)
    print()
    print(f"Conjecture: dim_C HH^2(u_q(sl_2), C) = C(2,2) + 2*|Phi^+(A_1)|")
    print(f"                                        = 1 + 2 = 3")
    print()

    # --- Step 1: build presentation ---
    print("-" * 72)
    print("Step 1: build u_q(sl_2) presentation")
    print("-" * 72)
    pres = build_uq_sl2_presentation()
    print(f"  Generators: {pres.generators}  (indices 0, 1, 2)")
    print(f"  Rules ({len(pres.rules)}):")
    for i, r in enumerate(pres.rules):
        print(f"    R{i+1}: {r.lhs} -> {r.rhs}")
    print()
    print(f"  Coefficient field: Q(omega) with omega = e^(2*pi*i/{ELL})")
    print(f"  omega = {OMEGA.to_complex()}")
    print(f"  omega^2 = {OMEGA2.to_complex()}")
    from ir.qomega import Q_MINUS_Q_INV_INV
    print(f"  1/(q - q^{{-1}}) = {Q_MINUS_Q_INV_INV} = {Q_MINUS_Q_INV_INV.to_complex()}")
    print()

    # --- Step 2: verify PBW basis ---
    print("-" * 72)
    print("Step 2: verify PBW basis")
    print("-" * 72)
    pbw_check = verify_pbw_basis(pres, n_random=200, seed=42)
    print(f"  PBW basis size:                 {pbw_check['pbw_size']}  (expected: {DIM})")
    print(f"  All PBW monomials in normal form: {pbw_check['pbw_all_normal']}")
    print(f"  Random reductions all in PBW:     {pbw_check['random_reductions_in_pbw']}")
    print(f"  (200 random monomials of length 0-8 reduced; all results in PBW basis)")
    print()

    # Sanity: show a few reductions
    reducer = NormalFormReducer(pres)
    from ir.parser import Monomial
    print("  Sample reductions:")
    samples = [
        ("E K",       Monomial((1, 0))),
        ("F K",       Monomial((2, 0))),
        ("F E",       Monomial((2, 1))),
        ("K K K",     Monomial((0, 0, 0))),
        ("E E E",     Monomial((1, 1, 1))),
        ("F F F",     Monomial((2, 2, 2))),
        ("E K K",     Monomial((1, 0, 0))),
        ("F E K",     Monomial((2, 1, 0))),
        ("K F E K",   Monomial((0, 2, 1, 0))),
        ("E F K E",   Monomial((1, 2, 0, 1))),
    ]
    for name, mon in samples:
        nf = reducer.normal_form(mon)
        print(f"    {name:10s} -> {nf}")
    print()

    # --- Step 3: KB completion ---
    print("-" * 72)
    print("Step 3: Knuth-Bendix completion")
    print("-" * 72)
    completed, stats = run_kb_completion(pres, max_iterations=50, verbose=False)
    print(f"  Initial rules:           {stats.initial_rules}")
    print(f"  Final rules:             {stats.final_rules}")
    print(f"  New rules added:         {stats.new_rules_added}")
    print(f"  Critical pairs checked:  {stats.critical_pairs_checked}")
    print(f"  Iterations:              {stats.iterations}")
    print(f"  Terminated (confluent):  {stats.terminated}")
    print(f"  Failed pairs:            {len(stats.failed_pairs)}")
    if stats.new_rules_added == 0 and stats.terminated:
        print(f"  -> The 6-rule PBW system is already confluent at ell = 3.")
        print(f"     (All critical pairs reduce to zero in Q(omega).)")
    print()

    # --- Step 4: Anick resolution generators ---
    print("-" * 72)
    print("Step 4: Anick resolution generators")
    print("-" * 72)
    n0 = anick_degree0_count(completed)
    n1 = anick_degree1_count(completed)
    n2 = anick_degree2_count(completed)
    print(f"  Degree 0 (algebra):       {n0}   (the unit 1)")
    print(f"  Degree 1 (relations):     {n1}   (one per rewrite rule)")
    print(f"  Degree 2 (syzygies):      {n2}   (critical pairs)")
    print()
    print(f"  Syzygy listing (overlap monomial, rule pair, positions):")
    listing = anick_degree2_listing(completed)
    for i, (M, (ri, rj), (p1, p2)) in enumerate(listing):
        print(f"    {i+1:2d}. M = {M}, rules (R{ri+1}, R{rj+1}), positions ({p1}, {p2})")
    print()

    # --- Step 5: dim HH^2 ---
    print("-" * 72)
    print("Step 5: dim HH^2 via bar complex on PBW normal forms")
    print("-" * 72)
    print("  (The bar complex is homotopy-equivalent to the Anick resolution,")
    print("   so both compute the same dim HH^2; the bar complex is used here")
    print("   because its differential is straightforward to implement.)")
    print()
    print("  Building 27 x 27 x 27 multiplication table via IR reducer...")
    mult = build_multiplication_table(completed)
    print(f"  Multiplication table built. Sanity checks:")
    sanity = sanity_check_multiplication(mult)
    for k, v in sanity.items():
        print(f"    {k}: {v}")
    print()
    hh2 = compute_hh2(completed, verbose=False)
    print(f"  dim u_q(sl_2):       {hh2['dim_uq_sl2']}")
    print(f"  dim C^1:             {hh2['dim_c1']}")
    print(f"  dim C^2:             {hh2['dim_c2']}")
    print(f"  dim C^3:             {hh2['dim_c3']}")
    print(f"  rank(d^1):           {hh2['rank_d1']}")
    print(f"  rank(d^2):           {hh2['rank_d2']}")
    print(f"  dim ker(d^2):        {hh2['dim_ker_d2']}   (= dim C^2 - rank(d^2))")
    print(f"  dim im(d^1):         {hh2['dim_im_d1']}   (= rank(d^1))")
    print(f"  dim HH^2:            {hh2['dim_hh2']}   (= ker(d^2) - im(d^1))")
    print()

    # --- Conclusion ---
    expected = 3
    match = hh2["dim_hh2"] == expected
    print("=" * 72)
    print(f"  dim HH^2(u_q(sl_2), C) = {hh2['dim_hh2']}")
    print(f"  Expected (conjecture):   {expected}  = C(2,2) + 2*|Phi^+(A_1)| = 1 + 2")
    print(f"  MATCH: {match}")
    print("=" * 72)
    print()

    # --- Final summary ---
    print("Summary of W1-1a validation:")
    print(f"  - Presentation parses:           True  (6 rules, QOmega3 coefficients)")
    print(f"  - PBW basis size:                {pbw_check['pbw_size']}  (expected: {DIM})")
    print(f"  - PBW all normal:                {pbw_check['pbw_all_normal']}")
    print(f"  - KB completion adds no rules:   {stats.new_rules_added == 0}")
    print(f"  - KB terminated (confluent):     {stats.terminated}")
    print(f"  - Anick degree-0 count:          {n0}  (the unit)")
    print(f"  - Anick degree-1 count:          {n1}  (one per rule)")
    print(f"  - Anick degree-2 count:          {n2}  (syzygies / critical pairs)")
    print(f"  - All multiplication checks:     {all(sanity.values())}")
    print(f"  - dim HH^2:                      {hh2['dim_hh2']}  (expected: {expected})")
    print(f"  - dim HH^2 MATCHES conjecture:   {match}")
    print()

    return 0 if match else 1


if __name__ == "__main__":
    sys.exit(main())
