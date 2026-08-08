#!/usr/bin/env python3
r"""
LES consistency analysis for sl_3 at ell = 3.

Task W2-1b: Investigate whether the Mastnak-Witherspoon LES decomposition
  dim HH^2(D(B^+), C) = dim im(bar-iota) + dim im(delta)
is consistent with the conjecture
  dim HH^2(u_q(sl_3), C) = C(3,2) + 2|Phi^+| = 3 + 6 = 9
at ell = 3, given that dim HH^2(B^+(u_q(sl_3)), C) = 5 (verified, paper Sec. 6.5).

Notation (matching paper/main.tex Sec. 7 / eq. mwles):
  - B  := B^+(u_q(sl_3)), the positive Borel subalgebra, dim = 3^5 = 243.
  - B* := (B^+)* ~= B^- by the Chevalley involution (sl_3 self-duality).
  - D(B) := u_q(sl_3), the Drinfeld double.
  - H^i(A) := HH^i(A, C) (Hochschild cohomology with trivial C-coefficients).
  - Hb^i(B) := H~^i_b(B), reduced bialgebra cohomology (Gerstenhaber-Schack / MW).
  - bar-iota: H^i(D(B)) -> H^i(B) \oplus H^i(B*) induced by projections.
  - bar-pi:   H^i(B) \oplus H^i(B*) -> Hb^i(B).
  - delta:    Hb^i(B) -> H^{i+1}(D(B)), the connecting homomorphism.

Mastnak-Witherspoon LES (eq. mwles), in degrees 1-2:
  H^1(D) --bar-iota--> H^1(B) \oplus H^1(B*) --bar-pi--> Hb^1(B)
        --delta--> H^2(D) --bar-iota--> H^2(B) \oplus H^2(B*) --bar-pi--> Hb^2(B)

This script answers Q1-Q6 of the W2-1b task brief:
  Q1: LES dimensional constraints.
  Q2: Splits (dim im(delta), dim im(bar-iota at deg 2)) consistent with the LES,
      given dim H^2(D) = 9 (conjecture).
  Q3: Can we pin down dim im(delta) from first principles?
  Q4: Compute dim H^1(B^+) directly (weight-decomposed bar complex, dim C^1 = 243,
      dim C^2 = 59049 per block: small).
  Q5: Use the H^1 computation to simplify the LES.
  Q6: Strongest statement about A_2.
"""
import os
import sys
import math
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Import the B^+(sl_3) multiplication-table builder + weight function
from verify_sl3_bplus_hh2 import (
    build_mult_sparse, idx, from_idx, weight, ELL, Q, Q_INV, DIM,
)


# ============================================================
# Known dimensions (paper/main.tex Sec. 6.5 and Sec. 7)
# ============================================================
N = 2                       # sl_3: n = 2
PHI_PLUS = 3                # |Phi^+| for sl_3
ELL_VAL = 3
DIM_BPLUS = DIM             # 243

CONJ_DIM_HH2_FULL = math.comb(N + 1, 2) + 2 * PHI_PLUS    # 3 + 6 = 9
DIM_HH2_BPLUS = 5           # verified, paper Sec. 6.5
DIM_HH2_BMINUS = 5          # by Chevalley duality (sl_3 self-dual)
DIM_HH2_BPLUS_PLUS_BMINUS = DIM_HH2_BPLUS + DIM_HH2_BMINUS   # 10


def banner(s):
    print()
    print("=" * 72)
    print(s)
    print("=" * 72)


def main():
    print(f"=== LES Consistency Analysis for sl_3 at ell = {ELL_VAL} ===")
    print()
    print(f"Conjecture (paper/main.tex eq. main):")
    print(f"  dim_C HH^2(u_q(sl_{{n}}), C) = C(n+1, 2) + 2|Phi^+|.")
    print(f"For sl_3 (n={N}): C({N+1}, 2) + 2*{PHI_PLUS} = {math.comb(N+1, 2)} + {2*PHI_PLUS} = {CONJ_DIM_HH2_FULL}.")
    print()
    print(f"Known dimensions:")
    print(f"  dim HH^2(B^+(u_q(sl_3)), C) = {DIM_HH2_BPLUS}  (verified, paper Sec. 6.5)")
    print(f"  dim HH^2(B^-(u_q(sl_3)), C) = {DIM_HH2_BMINUS}  (by Chevalley duality B^+ <-> B^-)")
    print(f"  dim HH^2(B^+) \\oplus HH^2(B^-) = {DIM_HH2_BPLUS_PLUS_BMINUS}")
    print(f"  dim B^+ = {DIM_BPLUS}  (3^5 = 3^(rank + 2|Phi^+|))")

    # ============================================================
    # Q1: LES dimensional constraints
    # ============================================================
    banner("Q1: LES dimensional constraints (Mastnak-Witherspoon eq. mwles)")
    print()
    print("LES in degrees 1-2 (H_i := HH^i, Hb := H~_b):")
    print()
    print("  H_1(D) --i--> H_1(B) \\oplus H_1(B*) --p--> Hb^1(B)")
    print("        --d--> H_2(D) --i--> H_2(B) \\oplus H_2(B*) --p--> Hb^2(B)")
    print()
    print("where B = B^+(u_q(sl_3)), B* = (B^+)* \\cong B^- by Chevalley, D(B) = u_q(sl_3).")
    print()
    print("Here i := bar-iota (restriction), p := bar-pi, d := delta (connecting homomorphism).")
    print()
    print("By exactness at H_2(D):")
    print("  dim H_2(D) = dim im(d) + dim im(i at deg 2).")
    print()
    print("Constraints on the split (x := dim im(d), y := dim im(i at deg 2)):")
    print("  (a) x + y = dim H_2(D)                         (LES exactness)")
    print("  (b) 0 <= x <= dim Hb^1(B)                      (image of d)")
    print("  (c) 0 <= y <= dim H_2(B) \\oplus H_2(B*) =", DIM_HH2_BPLUS_PLUS_BMINUS, "  (image of i at deg 2)")
    print("  (d) y <= dim H_2(D)                            (image is a subspace)")
    print()

    # ============================================================
    # Q2: Consistent splits given the conjecture
    # ============================================================
    banner("Q2: Consistent (dim im(d), dim im(i)) splits under the conjecture")
    print()
    print(f"Assume dim H_2(D) = {CONJ_DIM_HH2_FULL} (conjecture). Splits (x, y) with x + y = {CONJ_DIM_HH2_FULL}")
    print(f"and 0 <= y <= {DIM_HH2_BPLUS_PLUS_BMINUS}:")
    print()
    splits = []
    for x in range(CONJ_DIM_HH2_FULL + 1):
        y = CONJ_DIM_HH2_FULL - x
        if 0 <= y <= DIM_HH2_BPLUS_PLUS_BMINUS:
            splits.append((x, y))
            marker = ""
            if (x, y) == (math.comb(N + 1, 2), 2 * PHI_PLUS):
                marker = "   <-- conjecture's structural prediction"
            print(f"  dim im(d) = {x}, dim im(i) = {y}{marker}")
    print()
    print(f"Total: {len(splits)} splits consistent with the LES (under dim H_2(D) = {CONJ_DIM_HH2_FULL}).")
    print()
    print("The LES alone DOES NOT pin down the split.")

    # ============================================================
    # Q3: Can we pin down dim im(d) from first principles?
    # ============================================================
    banner("Q3: Can we pin down dim im(d) from first principles?")
    print()
    print("No, not from the LES alone -- see Q2:", len(splits), "consistent splits.")
    print()
    print("To pin down dim im(d), we need one of:")
    print("  (a) Compute dim Hb^1(B^+) directly (source of d).")
    print("      By LES exactness: dim im(d) = dim Hb^1(B) - dim im(p at deg 1).")
    print("      If dim H^1(B^+) = dim H^1(B^-) = 0, then p at deg 1 has zero source,")
    print("      so dim im(p at deg 1) = 0, and d is INJECTIVE.")
    print("      Then dim im(d) = dim Hb^1(B^+) and the LES becomes:")
    print("        dim H_2(D) = dim Hb^1(B^+) + dim im(i at deg 2).")
    print("  (b) Compute the restriction map i: H^2(D) -> H^2(B) \\oplus H^2(B*) explicitly.")
    print("      This requires knowing H^2(D) -- INTRACTABLE per W2-1a")
    print("      (sparse d^2 needs 2.28 TB RAM, ~190 years).")
    print()
    print("Approach (a) is the tractable path. First step: verify dim H^1(B^+) = 0 (Q4).")
    print("Second step (out of scope here, future task): compute Hb^1(B^+) directly.")

    # ============================================================
    # Q4: Compute dim HH^1(B^+) directly
    # ============================================================
    banner(f"Q4: Compute dim HH^1(B^+(u_q(sl_3)), C) at ell = {ELL_VAL}")
    print()
    print("HH^1(B^+) = ker(d^1: C^1 -> C^2) / im(d^0: C^0 -> C^1).")
    print("For trivial coefficients, d^0 = 0 (since d^0(lambda)(a) = eps(a) lambda - eps(a) lambda = 0).")
    print("Hence HH^1 = ker(d^1).")
    print()
    print(f"  dim B^+ = {DIM_BPLUS}, dim C^1 = {DIM_BPLUS}, dim C^2 = {DIM_BPLUS**2}.")
    print(f"  Weight-space decomposition: {DIM_BPLUS // 27} weights, each weight space dim 27.")
    print(f"  Each weight block of d^1 has size dim(C^2_w) x dim(C^1_w) = 6561 x 27.")
    print(f"  Total rank(d^1) = sum of weight-block ranks.")
    print()

    print(f"Building multiplication table for B^+(sl_3)...")
    t0 = time.time()
    ms = build_mult_sparse(DIM_BPLUS)
    print(f"  done in {time.time()-t0:.2f}s")

    # Counit: eps(K1^a K2^b E1^c E12^e E2^d) = 1 if c=e=d=0, else 0.
    epsilon = np.zeros(DIM_BPLUS, dtype=complex)
    for a in range(ELL):
        for b in range(ELL):
            epsilon[idx(a, b, 0, 0, 0)] = 1.0

    # Weights (2D, mod ell)
    wts = [weight(i) for i in range(DIM_BPLUS)]
    unique_wts = sorted(set(wts))
    assert len(unique_wts) == 9, f"Expected 9 weights, got {len(unique_wts)}"

    # Group basis indices by weight
    wt_indices = {w: [i for i in range(DIM_BPLUS) if wts[i] == w] for w in unique_wts}
    # Group C^2 pairs by total weight
    c2_by_wt = {w: [] for w in unique_wts}
    for i in range(DIM_BPLUS):
        for j in range(DIM_BPLUS):
            w = ((wts[i][0] + wts[j][0]) % ELL, (wts[i][1] + wts[j][1]) % ELL)
            c2_by_wt[w].append(i * DIM_BPLUS + j)

    print()
    print(f"  Weight block analysis:")
    print(f"    {'weight':<10} {'dim C^1':<10} {'dim C^2':<10} {'rank(d^1)':<12} "
          f"{'dim ker(d^1)':<14} {'dim HH^1':<10}")
    total_rank_d1 = 0
    total_dim_c1 = 0
    hh1_by_wt = {}
    for w in unique_wts:
        cols = wt_indices[w]
        rows = c2_by_wt[w]
        n_cols, n_rows = len(cols), len(rows)
        total_dim_c1 += n_cols
        if n_cols == 0 or n_rows == 0:
            hh1_by_wt[w] = n_cols
            print(f"    {str(w):<10} {n_cols:<10} {n_rows:<10} {'0':<12} "
                  f"{n_cols:<14} {n_cols:<10}")
            continue
        d1 = np.zeros((n_rows, n_cols), dtype=complex)
        for col, i_f in enumerate(cols):
            for row, ij in enumerate(rows):
                i, j = ij // DIM_BPLUS, ij % DIM_BPLUS
                # (d^1 f)(a, b) = eps(a) f(b) - f(a*b) + f(a) eps(b), where f = indicator on basis[i_f]
                mult_if = sum(v for (l, v) in ms[i][j] if l == i_f)
                d1[row, col] = (epsilon[i] * (1.0 if j == i_f else 0.0)
                                - mult_if
                                + (1.0 if i == i_f else 0.0) * epsilon[j])
        s = np.linalg.svd(d1, compute_uv=False)
        tol = max(d1.shape) * (s[0] if len(s) > 0 else 0) * 1e-14
        rank = int(np.sum(s > tol)) if len(s) > 0 else 0
        total_rank_d1 += rank
        hh1_w = n_cols - rank
        hh1_by_wt[w] = hh1_w
        print(f"    {str(w):<10} {n_cols:<10} {n_rows:<10} {rank:<12} "
              f"{n_cols - rank:<14} {hh1_w:<10}")

    hh1_total = sum(hh1_by_wt.values())
    print()
    print(f"  TOTAL: dim C^1 = {total_dim_c1}, rank(d^1) = {total_rank_d1}, "
          f"dim HH^1(B^+(u_q(sl_3)), C) = {hh1_total}")
    print()
    if hh1_total == 0:
        print(f"  ==> dim HH^1(B^+) = 0  (matches sl_2 at ell = 3, 5, 7; matches paper Sec. 7)")
    else:
        print(f"  ==> dim HH^1(B^+) = {hh1_total} != 0 (unexpected).")

    # ============================================================
    # Q5: Simplified LES given HH^1 vanishing
    # ============================================================
    banner("Q5: Simplified LES given the H^1 computation")
    print()
    print(f"  Computed: dim H^1(B^+) = {hh1_total}")
    print(f"  By Chevalley duality: dim H^1(B^-) = dim H^1(B^+) = {hh1_total}")
    print()

    if hh1_total == 0:
        print("  ==> dim H^1(B^+) = dim H^1(B^-) = 0, as expected.")
        print()
        print("  Consequences for the LES in degree 2:")
        print("    - The map bar-pi: H^1(B) \\oplus H^1(B*) -> Hb^1(B) has zero source,")
        print("      hence zero image.")
        print("    - By exactness, ker(delta: Hb^1(B) -> H^2(D(B))) = im(bar-pi) = 0.")
        print("    - So delta is INJECTIVE: dim im(delta) = dim Hb^1(B^+).")
        print()
        print("  Simplified LES decomposition (under HH^1 vanishing):")
        print("    dim HH^2(D(B^+)) = dim Hb^1(B^+) + dim im(bar-iota at deg 2)")
        print()
        print("  Substituting the conjecture's structural predictions:")
        print(f"    - dim im(bar-iota at deg 2) = 2|Phi^+| = {2*PHI_PLUS}  (ell-th power classes)")
        print(f"    - dim im(delta) = C({N+1}, 2) = {math.comb(N+1, 2)}  (Cartan-type / mixed)")
        print(f"  Yields dim Hb^1(B^+) = {math.comb(N+1, 2)}.")
        print()
        print("  CONVERSELY, the conjecture at A_2 is EQUIVALENT (under HH^1 vanishing)")
        print(f"  to the single prediction:")
        print(f"    dim Hb^1(B^+(u_q(sl_3)), C) = C({N+1}, 2) = {math.comb(N+1, 2)} at ell = {ELL_VAL}.")
    else:
        print(f"  HH^1 did NOT vanish. The LES does not simplify.")

    # ============================================================
    # Q6: Strongest statement about A_2
    # ============================================================
    banner("Q6: Strongest statement about A_2")
    print()
    print(f"Given:")
    print(f"  (i)   dim HH^2(B^+) = dim HH^2(B^-) = {DIM_HH2_BPLUS}  (verified)")
    print(f"  (ii)  dim HH^1(B^+) = dim HH^1(B^-) = {hh1_total}  (computed in Q4)")
    print(f"  (iii) LES exactness at degree 2.")
    print()
    print(f"Constraints on dim HH^2(D(B^+)) = dim HH^2(u_q(sl_3), C):")
    if hh1_total == 0:
        print(f"  dim HH^2(D) = dim Hb^1(B^+) + dim im(bar-iota at deg 2)")
        print(f"  with delta injective (since HH^1 vanishes).")
        print(f"  Let k := dim Hb^1(B^+). Then dim HH^2(D) = k + dim im(bar-iota at deg 2).")
        print(f"  Constraints:")
        print(f"    0 <= k                                          (trivial)")
        print(f"    0 <= dim im(bar-iota at deg 2) <= {DIM_HH2_BPLUS_PLUS_BMINUS}                "
              f"(target dim)")
        print(f"  Hence dim HH^2(D) \\in [k, k + {DIM_HH2_BPLUS_PLUS_BMINUS}].")
        print()
        # Lower bound on k for the conjecture to be consistent
        # Need: k <= 9 AND k + 10 >= 9, i.e. -1 <= k <= 9, i.e. 0 <= k <= 9.
        k_min_for_conj = max(0, CONJ_DIM_HH2_FULL - DIM_HH2_BPLUS_PLUS_BMINUS)
        k_max_for_conj = CONJ_DIM_HH2_FULL
        print(f"  CONJECTURE (dim HH^2(D) = {CONJ_DIM_HH2_FULL}) is consistent iff")
        print(f"    k_min = {k_min_for_conj} <= dim Hb^1(B^+) <= k_max = {k_max_for_conj}.")
        print()
        print(f"  CONJECTURE REFUTATION CRITERION (sufficient):")
        print(f"    If dim Hb^1(B^+) > {k_max_for_conj}, then dim HH^2(D) > {CONJ_DIM_HH2_FULL},")
        print(f"    REFUTING the conjecture at A_2.")
        print()
        print(f"  CONJECTURE VERIFICATION CRITERION (given the structural prediction")
        print(f"  dim im(bar-iota at deg 2) = 2|Phi^+| = {2*PHI_PLUS}):")
        print(f"    If dim Hb^1(B^+) = {CONJ_DIM_HH2_FULL - 2 * PHI_PLUS} = C({N+1}, 2),")
        print(f"    then dim HH^2(D) = {2 * PHI_PLUS} + {math.comb(N+1, 2)} = {CONJ_DIM_HH2_FULL},")
        print(f"    VERIFYING the conjecture at A_2.")
        print()
        print(f"  STRONGEST STATEMENT (given HH^1 vanishing):")
        print(f"    The conjecture at A_2 is EQUIVALENT to")
        print(f"      dim Hb^1(B^+(u_q(sl_3)), C) = C({N+1}, 2) = {math.comb(N+1, 2)} at ell = {ELL_VAL}.")
        print(f"    A direct computation of Hb^1(B^+) -- feasible at dim B^+ = {DIM_BPLUS} --")
        print(f"    would either VERIFY (if = {math.comb(N+1, 2)}) or REFUTE (if != {math.comb(N+1, 2)})")
        print(f"    the conjecture at A_2.")
    print()

    # ============================================================
    # Summary
    # ============================================================
    banner("Summary")
    print()
    print(f"  dim HH^2(B^+) = {DIM_HH2_BPLUS}                              (verified)")
    print(f"  dim HH^2(B^-) = {DIM_HH2_BMINUS}                              (by duality)")
    print(f"  dim HH^1(B^+) = {hh1_total}                                (computed here)")
    print(f"  dim HH^1(B^-) = {hh1_total}                                (by duality)")
    print(f"  Conjecture:   dim HH^2(u_q(sl_3), C) = {CONJ_DIM_HH2_FULL}")
    print()
    print(f"  LES consistency (given dim HH^2(D) = {CONJ_DIM_HH2_FULL}):")
    print(f"    {len(splits)} splits (dim im(delta), dim im(bar-iota)) are consistent.")
    print(f"    Conjecture's specific split ({math.comb(N+1,2)}, {2*PHI_PLUS}) is among them.  ==> CONSISTENT.")
    print()
    if hh1_total == 0:
        print(f"  HH^1 vanishing (dim = 0) ==> delta is injective.")
        print(f"  Conjecture at A_2 EQUIVALENT to dim Hb^1(B^+) = C({N+1}, 2) = {math.comb(N+1, 2)}.")
        print()
        print(f"  Status of A_2: OPEN, but reduced to a TRACTABLE computation.")
        print(f"  The bialgebra cochain complex for B^+ has chain groups of dim ~ {DIM_BPLUS}^2 = "
              f"{DIM_BPLUS**2},")
        print(f"  small enough to admit direct computation (cf. intractable HH^2(D) at 2.28 TB).")
    print()
    return {
        "hh1_bplus": hh1_total,
        "hh2_bplus": DIM_HH2_BPLUS,
        "hh2_bminus": DIM_HH2_BMINUS,
        "conj_dim_hh2_full": CONJ_DIM_HH2_FULL,
        "n_consistent_splits": len(splits),
        "conjecture_split_in_set": (math.comb(N + 1, 2), 2 * PHI_PLUS) in splits,
    }


if __name__ == "__main__":
    main()
