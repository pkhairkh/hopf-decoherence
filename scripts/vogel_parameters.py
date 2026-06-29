#!/usr/bin/env python3
"""
Vogel Parameter Table for ALL Simple Lie Algebras
==================================================

Computes and verifies the complete Vogel parameter table for every simple
Lie algebra, including classical series (A_n, B_n, C_n, D_n) and
exceptional groups (G_2, F_4, E_6, E_7, E_8).

References:
  - Vogel, P. (1999), "The Universal Lie algebra", hep-th/9912204
  - Mkrtchyan, R. (2012), arXiv:1109.1668 [math-ph] (Casimir eigenvalues)
  - Landsberg, Manivel (2006), "A universal dimension formula", math/0401296
  - Bishler, Mironov, Morozov (2023), arXiv:2311.05358 (Vogel's Table 1)

Vogel's Universal Lie Algebra:
  Every simple Lie algebra g is parameterized by three projective
  parameters (alpha, beta, gamma) on P^2/S_3 with S_3 permutation symmetry.

  The normalized parameters satisfy alpha_hat + beta_hat + gamma_hat = -2
  (in one common convention) or = 1/2 (in the t=1/2 convention).

  Universal dimension formula:
    dim(g) = (alpha_hat - 1)(beta_hat - 1)(gamma_hat - 1) / (alpha_hat * beta_hat * gamma_hat)
    [in the t = 1/2 normalization where sum = 1/2]

  Equivalently, using unnormalized parameters (alpha, beta, gamma) with
  alpha + beta + gamma = h^vee (dual Coxeter number) and alpha = -2:
    dim(g) = (alpha - 2*h^vee)(beta - 2*h^vee)(gamma - 2*h^vee) / (alpha * beta * gamma)
"""

from fractions import Fraction
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import json
import os


# ============================================================================
# Fraction arithmetic for exact computation
# ============================================================================

def frac(n, d=1):
    """Convenience constructor for Fraction."""
    return Fraction(n, d)


# ============================================================================
# Core data structures
# ============================================================================

@dataclass
class VogelParameters:
    """Normalized Vogel parameters in the t=1/2 convention (sum = 1/2)."""
    alpha_hat: Fraction
    beta_hat: Fraction
    gamma_hat: Fraction


@dataclass
class UnnormVogelParameters:
    """Unnormalized Vogel parameters in the alpha=-2, sum=h^vee convention."""
    alpha: Fraction
    beta: Fraction
    gamma: Fraction
    h_vee: Fraction  # = alpha + beta + gamma


@dataclass
class NormSumMinus2:
    """Normalized Vogel parameters with alpha_hat + beta_hat + gamma_hat = -2."""
    alpha_hat: Fraction
    beta_hat: Fraction
    gamma_hat: Fraction


@dataclass
class LieAlgebraData:
    """Complete data for a simple Lie algebra."""
    name: str
    cartan_type: str  # A, B, C, D, G2, F4, E6, E7, E8
    rank: int
    dim: int
    h_vee: int  # dual Coxeter number
    pos_roots: int  # |Phi^+|
    pos_roots_ns: int  # |Phi^+_ns| = |Phi^+| - rank
    unnorm_vogel: UnnormVogelParameters
    norm_vogel_t_half: VogelParameters  # t=1/2 normalization (sum=1/2)
    norm_vogel_sum_minus2: NormSumMinus2  # sum=-2 normalization

    # Derived quantities for the D-tilde^2 analysis
    d_tilde_sq_power: int = 0  # = 2*dim - 3*rank
    z_raw_power: Fraction = frac(0)  # = 3*rank/2
    norm_exponent: int = 0  # = 3*|Phi^+_ns| = 3*(dim - 3*rank)/2

    def compute_derived(self):
        """Compute derived quantities from primary data."""
        self.pos_roots_ns = self.pos_roots - self.rank
        self.d_tilde_sq_power = 2 * self.dim - 3 * self.rank
        self.z_raw_power = Fraction(3 * self.rank, 2)
        self.norm_exponent = 3 * self.pos_roots_ns
        return self


# ============================================================================
# Dimension formula verification
# ============================================================================

def vogel_dim_unnorm(alpha, beta, gamma, h_vee):
    """
    Compute dim(g) from unnormalized Vogel parameters.

    Formula: dim(g) = (alpha - 2*h^vee)(beta - 2*h^vee)(gamma - 2*h^vee) / (alpha*beta*gamma)

    where alpha + beta + gamma = h_vee and alpha = -2.
    """
    alpha, beta, gamma, h_vee = Fraction(alpha), Fraction(beta), Fraction(gamma), Fraction(h_vee)
    numerator = (alpha - 2 * h_vee) * (beta - 2 * h_vee) * (gamma - 2 * h_vee)
    denominator = alpha * beta * gamma
    return numerator / denominator


def vogel_dim_norm_t_half(alpha_hat, beta_hat, gamma_hat):
    """
    Compute dim(g) from normalized Vogel parameters in the t=1/2 convention.

    Formula: dim(g) = (alpha_hat - 1)(beta_hat - 1)(gamma_hat - 1) / (alpha_hat * beta_hat * gamma_hat)

    where alpha_hat + beta_hat + gamma_hat = 1/2.
    """
    a, b, g = Fraction(alpha_hat), Fraction(beta_hat), Fraction(gamma_hat)
    numerator = (a - 1) * (b - 1) * (g - 1)
    denominator = a * b * g
    return numerator / denominator


def vogel_dim_norm_sum_minus2(alpha_hat, beta_hat, gamma_hat):
    """
    Compute dim(g) from normalized Vogel parameters with sum = -2.

    Formula: dim(g) = (alpha_hat + 4)(beta_hat + 4)(gamma_hat + 4) / (alpha_hat * beta_hat * gamma_hat)

    where alpha_hat + beta_hat + gamma_hat = -2.
    """
    a, b, g = Fraction(alpha_hat), Fraction(beta_hat), Fraction(gamma_hat)
    numerator = (a + 4) * (b + 4) * (g + 4)
    denominator = a * b * g
    return numerator / denominator


# ============================================================================
# Convert between normalizations
# ============================================================================

def unnorm_to_norm_t_half(alpha, beta, gamma):
    """Convert unnormalized (alpha=-2, sum=h^vee) to t=1/2 normalized (sum=1/2)."""
    alpha, beta, gamma = Fraction(alpha), Fraction(beta), Fraction(gamma)
    h_vee = alpha + beta + gamma
    s = 2 * h_vee  # scaling factor
    return VogelParameters(alpha / s, beta / s, gamma / s)


def unnorm_to_norm_sum_minus2(alpha, beta, gamma):
    """Convert unnormalized (alpha=-2, sum=h^vee) to sum=-2 normalized."""
    alpha, beta, gamma = Fraction(alpha), Fraction(beta), Fraction(gamma)
    h_vee = alpha + beta + gamma
    s = -h_vee / 2  # scaling factor
    return NormSumMinus2(alpha / s, beta / s, gamma / s)


# ============================================================================
# Build the complete table
# ============================================================================

def build_classical_algebras():
    """Build data for classical Lie algebra series A_n, B_n, C_n, D_n."""
    algebras = []

    # ---- A_n (sl_{n+1}) ----
    # dim = n(n+2), rank = n, |Phi^+| = n(n+1)/2, h^vee = n+1
    # Unnormalized: (alpha, beta, gamma) = (-2, 2, n+1)
    for n in range(1, 9):  # A_1 through A_8
        h_vee = n + 1
        alpha, beta, gamma = frac(-2), frac(2), frac(n + 1)
        assert alpha + beta + gamma == h_vee, f"A_{n}: sum = {alpha+beta+gamma}, h^vee = {h_vee}"

        alg = LieAlgebraData(
            name=f"A_{n} (sl_{n+1})",
            cartan_type="A",
            rank=n,
            dim=n * (n + 2),
            h_vee=h_vee,
            pos_roots=n * (n + 1) // 2,
            pos_roots_ns=0,
            unnorm_vogel=UnnormVogelParameters(alpha, beta, gamma, h_vee),
            norm_vogel_t_half=unnorm_to_norm_t_half(alpha, beta, gamma),
            norm_vogel_sum_minus2=unnorm_to_norm_sum_minus2(alpha, beta, gamma),
        )
        alg.compute_derived()
        algebras.append(alg)

    # ---- B_n (so_{2n+1}) ----
    # dim = n(2n+1), rank = n, |Phi^+| = n^2, h^vee = 2n-1
    # Unnormalized: (alpha, beta, gamma) = (-2, 4, 2n-3)
    for n in range(2, 9):  # B_2 through B_8
        h_vee = 2 * n - 1
        alpha, beta, gamma = frac(-2), frac(4), frac(2 * n - 3)
        assert alpha + beta + gamma == h_vee, f"B_{n}: sum = {alpha+beta+gamma}, h^vee = {h_vee}"

        alg = LieAlgebraData(
            name=f"B_{n} (so_{2*n+1})",
            cartan_type="B",
            rank=n,
            dim=n * (2 * n + 1),
            h_vee=h_vee,
            pos_roots=n * n,
            pos_roots_ns=0,
            unnorm_vogel=UnnormVogelParameters(alpha, beta, gamma, h_vee),
            norm_vogel_t_half=unnorm_to_norm_t_half(alpha, beta, gamma),
            norm_vogel_sum_minus2=unnorm_to_norm_sum_minus2(alpha, beta, gamma),
        )
        alg.compute_derived()
        algebras.append(alg)

    # ---- C_n (sp_{2n}) ----
    # dim = n(2n+1), rank = n, |Phi^+| = n^2, h^vee = n+1
    # Unnormalized: (alpha, beta, gamma) = (-2, n+2, 1)
    # Note: C_n parameters involve a permutation of the sp(N) parameters
    # from the paper. The set {1, -2, n+2} permutes to (-2, n+2, 1).
    for n in range(2, 9):  # C_2 through C_8
        h_vee = n + 1
        alpha, beta, gamma = frac(-2), frac(n + 2), frac(1)
        assert alpha + beta + gamma == h_vee, f"C_{n}: sum = {alpha+beta+gamma}, h^vee = {h_vee}"

        alg = LieAlgebraData(
            name=f"C_{n} (sp_{2*n})",
            cartan_type="C",
            rank=n,
            dim=n * (2 * n + 1),
            h_vee=h_vee,
            pos_roots=n * n,
            pos_roots_ns=0,
            unnorm_vogel=UnnormVogelParameters(alpha, beta, gamma, h_vee),
            norm_vogel_t_half=unnorm_to_norm_t_half(alpha, beta, gamma),
            norm_vogel_sum_minus2=unnorm_to_norm_sum_minus2(alpha, beta, gamma),
        )
        alg.compute_derived()
        algebras.append(alg)

    # ---- D_n (so_{2n}) ----
    # dim = n(2n-1), rank = n, |Phi^+| = n(n-1), h^vee = 2n-2
    # Unnormalized: (alpha, beta, gamma) = (-2, 4, 2n-4)
    for n in range(3, 9):  # D_3 through D_8 (D_2 is not simple, D_3 = A_3)
        h_vee = 2 * n - 2
        alpha, beta, gamma = frac(-2), frac(4), frac(2 * n - 4)
        assert alpha + beta + gamma == h_vee, f"D_{n}: sum = {alpha+beta+gamma}, h^vee = {h_vee}"

        alg = LieAlgebraData(
            name=f"D_{n} (so_{2*n})",
            cartan_type="D",
            rank=n,
            dim=n * (2 * n - 1),
            h_vee=h_vee,
            pos_roots=n * (n - 1),
            pos_roots_ns=0,
            unnorm_vogel=UnnormVogelParameters(alpha, beta, gamma, h_vee),
            norm_vogel_t_half=unnorm_to_norm_t_half(alpha, beta, gamma),
            norm_vogel_sum_minus2=unnorm_to_norm_sum_minus2(alpha, beta, gamma),
        )
        alg.compute_derived()
        algebras.append(alg)

    return algebras


def build_exceptional_algebras():
    """Build data for exceptional Lie algebras G2, F4, E6, E7, E8."""
    algebras = []

    # Unnormalized Vogel parameters (alpha = -2, sum = h^vee):
    # G_2: (-2, 10/3, 8/3), h^vee = 4
    # F_4: (-2, 5, 6), h^vee = 9
    # E_6: (-2, 6, 8), h^vee = 12
    # E_7: (-2, 8, 12), h^vee = 18
    # E_8: (-2, 12, 20), h^vee = 30

    exceptional_data = [
        ("G_2", "G2", 2, 14, 4, 6, frac(-2), frac(10, 3), frac(8, 3)),
        ("F_4", "F4", 4, 52, 9, 24, frac(-2), frac(5), frac(6)),
        ("E_6", "E6", 6, 78, 12, 36, frac(-2), frac(6), frac(8)),
        ("E_7", "E7", 7, 133, 18, 63, frac(-2), frac(8), frac(12)),
        ("E_8", "E8", 8, 248, 30, 120, frac(-2), frac(12), frac(20)),
    ]

    for name, ctype, rank, dim, h_vee, pos_roots, alpha, beta, gamma in exceptional_data:
        assert alpha + beta + gamma == h_vee, f"{name}: sum = {alpha+beta+gamma}, h^vee = {h_vee}"

        alg = LieAlgebraData(
            name=name,
            cartan_type=ctype,
            rank=rank,
            dim=dim,
            h_vee=h_vee,
            pos_roots=pos_roots,
            pos_roots_ns=0,
            unnorm_vogel=UnnormVogelParameters(alpha, beta, gamma, h_vee),
            norm_vogel_t_half=unnorm_to_norm_t_half(alpha, beta, gamma),
            norm_vogel_sum_minus2=unnorm_to_norm_sum_minus2(alpha, beta, gamma),
        )
        alg.compute_derived()
        algebras.append(alg)

    return algebras


# ============================================================================
# Verification
# ============================================================================

def verify_all(algebras: List[LieAlgebraData]) -> dict:
    """Run all verification checks and return results."""
    results = {
        "identity_checks": [],
        "dim_formula_checks_unnorm": [],
        "dim_formula_checks_norm_t_half": [],
        "dim_formula_checks_norm_sum_minus2": [],
        "all_passed": True,
    }

    for alg in algebras:
        # ---- Identity 1: |Phi^+_ns| = |Phi^+| - rank ----
        check1 = alg.pos_roots_ns == alg.pos_roots - alg.rank
        results["identity_checks"].append({
            "algebra": alg.name,
            "check": "|Phi^+_ns| = |Phi^+| - rank",
            "LHS": str(alg.pos_roots_ns),
            "RHS": str(alg.pos_roots - alg.rank),
            "passed": check1,
        })
        if not check1:
            results["all_passed"] = False

        # ---- Identity 2: |Phi^+_ns| = (dim - 3*rank)/2 ----
        expected_ns = Fraction(alg.dim - 3 * alg.rank, 2)
        check2 = Fraction(alg.pos_roots_ns) == expected_ns
        results["identity_checks"].append({
            "algebra": alg.name,
            "check": "|Phi^+_ns| = (dim - 3*rank)/2",
            "LHS": str(alg.pos_roots_ns),
            "RHS": str(expected_ns),
            "passed": check2,
        })
        if not check2:
            results["all_passed"] = False

        # ---- Identity 3: D_tilde^2 power = 2*dim - 3*rank ----
        check3 = alg.d_tilde_sq_power == 2 * alg.dim - 3 * alg.rank
        results["identity_checks"].append({
            "algebra": alg.name,
            "check": "D~^2 power = 2*dim - 3*rank",
            "LHS": str(alg.d_tilde_sq_power),
            "RHS": str(2 * alg.dim - 3 * alg.rank),
            "passed": check3,
        })
        if not check3:
            results["all_passed"] = False

        # ---- Identity 4: D~^2 = dim + 2*|Phi^+_ns| ----
        check4 = alg.d_tilde_sq_power == alg.dim + 2 * alg.pos_roots_ns
        results["identity_checks"].append({
            "algebra": alg.name,
            "check": "D~^2 = dim + 2*|Phi^+_ns|",
            "LHS": str(alg.d_tilde_sq_power),
            "RHS": str(alg.dim + 2 * alg.pos_roots_ns),
            "passed": check4,
        })
        if not check4:
            results["all_passed"] = False

        # ---- Identity 5: norm_exponent = 3*|Phi^+_ns| ----
        check5 = alg.norm_exponent == 3 * alg.pos_roots_ns
        results["identity_checks"].append({
            "algebra": alg.name,
            "check": "norm_exponent = 3*|Phi^+_ns|",
            "LHS": str(alg.norm_exponent),
            "RHS": str(3 * alg.pos_roots_ns),
            "passed": check5,
        })
        if not check5:
            results["all_passed"] = False

        # ---- Identity 6: norm_exponent = D~^2_excess + Z_raw_deficit ----
        # D~^2 excess over dim = 2*|Phi^+_ns| = dim - 3*rank
        # Z_raw deficit from dim/2 = dim/2 - 3*rank/2 = (dim-3*rank)/2 = |Phi^+_ns|
        # So D~^2_excess + Z_raw_deficit = 2*|Phi^+_ns| + |Phi^+_ns| = 3*|Phi^+_ns|
        d2_excess = alg.d_tilde_sq_power - alg.dim  # = 2*|Phi^+_ns|
        z_deficit = Fraction(alg.dim, 2) - alg.z_raw_power  # = |Phi^+_ns|
        check6 = Fraction(alg.norm_exponent) == d2_excess + z_deficit
        results["identity_checks"].append({
            "algebra": alg.name,
            "check": "3|Phi^+_ns| = D~^2_excess + Z_raw_deficit",
            "D~^2_excess": str(d2_excess),
            "Z_raw_deficit": str(z_deficit),
            "sum": str(d2_excess + z_deficit),
            "norm_exponent": str(alg.norm_exponent),
            "passed": check6,
        })
        if not check6:
            results["all_passed"] = False

        # ---- Dimension formula check: unnormalized ----
        u = alg.unnorm_vogel
        dim_computed = vogel_dim_unnorm(u.alpha, u.beta, u.gamma, u.h_vee)
        check_d1 = dim_computed == alg.dim
        results["dim_formula_checks_unnorm"].append({
            "algebra": alg.name,
            "formula": "(alpha-2h∨)(beta-2h∨)(gamma-2h∨)/(alpha*beta*gamma)",
            "computed": str(dim_computed),
            "expected": str(alg.dim),
            "passed": check_d1,
        })
        if not check_d1:
            results["all_passed"] = False

        # ---- Dimension formula check: normalized t=1/2 ----
        nv = alg.norm_vogel_t_half
        dim_computed2 = vogel_dim_norm_t_half(nv.alpha_hat, nv.beta_hat, nv.gamma_hat)
        check_d2 = dim_computed2 == alg.dim
        results["dim_formula_checks_norm_t_half"].append({
            "algebra": alg.name,
            "formula": "(α̂-1)(β̂-1)(γ̂-1)/(α̂β̂γ̂)  [sum=1/2]",
            "computed": str(dim_computed2),
            "expected": str(alg.dim),
            "passed": check_d2,
        })
        if not check_d2:
            results["all_passed"] = False

        # ---- Dimension formula check: normalized sum=-2 ----
        nv2 = alg.norm_vogel_sum_minus2
        dim_computed3 = vogel_dim_norm_sum_minus2(nv2.alpha_hat, nv2.beta_hat, nv2.gamma_hat)
        check_d3 = dim_computed3 == alg.dim
        results["dim_formula_checks_norm_sum_minus2"].append({
            "algebra": alg.name,
            "formula": "(α̂+4)(β̂+4)(γ̂+4)/(α̂β̂γ̂)  [sum=-2]",
            "computed": str(dim_computed3),
            "expected": str(alg.dim),
            "passed": check_d3,
        })
        if not check_d3:
            results["all_passed"] = False

    return results


# ============================================================================
# Pretty printing
# ============================================================================

def fmt_frac(f):
    """Format a Fraction nicely."""
    if isinstance(f, Fraction):
        if f.denominator == 1:
            return str(f.numerator)
        return f"{f.numerator}/{f.denominator}"
    return str(f)


def print_main_table(algebras: List[LieAlgebraData]):
    """Print the main Vogel parameter table."""
    print("\n" + "=" * 140)
    print("VOGEL PARAMETER TABLE FOR ALL SIMPLE LIE ALGEBRAS")
    print("=" * 140)

    header = (
        f"{'Algebra':<18} {'rank':>4} {'dim':>6} {'h∨':>4} {'|Φ⁺|':>5} "
        f"{'|Φ⁺_ns|':>7} {'3|Φ⁺_ns|':>8} "
        f"{'α':>8} {'β':>8} {'γ':>8} "
        f"{'α̂(½)':>8} {'β̂(½)':>8} {'γ̂(½)':>8} "
        f"{'D̃²':>6} {'Z_raw':>6} {'Norm':>6}"
    )
    print(header)
    print("-" * 140)

    for alg in algebras:
        u = alg.unnorm_vogel
        nv = alg.norm_vogel_t_half
        row = (
            f"{alg.name:<18} {alg.rank:>4} {alg.dim:>6} {alg.h_vee:>4} {alg.pos_roots:>5} "
            f"{alg.pos_roots_ns:>7} {alg.norm_exponent:>8} "
            f"{fmt_frac(u.alpha):>8} {fmt_frac(u.beta):>8} {fmt_frac(u.gamma):>8} "
            f"{fmt_frac(nv.alpha_hat):>8} {fmt_frac(nv.beta_hat):>8} {fmt_frac(nv.gamma_hat):>8} "
            f"{alg.d_tilde_sq_power:>6} {fmt_frac(alg.z_raw_power):>6} {alg.norm_exponent:>6}"
        )
        print(row)

    print("=" * 140)


def print_normalized_sum_minus2_table(algebras: List[LieAlgebraData]):
    """Print the normalized parameters in the sum=-2 convention."""
    print("\n" + "=" * 100)
    print("NORMALIZED VOGEL PARAMETERS (α̂ + β̂ + γ̂ = -2 CONVENTION)")
    print("Dimension formula: dim(g) = (α̂+4)(β̂+4)(γ̂+4) / (α̂β̂γ̂)")
    print("=" * 100)

    header = f"{'Algebra':<18} {'α̂':>10} {'β̂':>10} {'γ̂':>10} {'sum':>8} {'dim_check':>10}"
    print(header)
    print("-" * 100)

    for alg in algebras:
        nv = alg.norm_vogel_sum_minus2
        s = nv.alpha_hat + nv.beta_hat + nv.gamma_hat
        dim_c = vogel_dim_norm_sum_minus2(nv.alpha_hat, nv.beta_hat, nv.gamma_hat)
        row = (
            f"{alg.name:<18} {fmt_frac(nv.alpha_hat):>10} {fmt_frac(nv.beta_hat):>10} "
            f"{fmt_frac(nv.gamma_hat):>10} {fmt_frac(s):>8} {fmt_frac(dim_c):>10}"
        )
        print(row)

    print("=" * 100)


def print_paper_normalization_table(algebras: List[LieAlgebraData]):
    """
    Print the normalized parameters in the t=1/2 convention
    (matching arXiv:2311.05358 Table 1 format).
    """
    print("\n" + "=" * 100)
    print("NORMALIZED VOGEL PARAMETERS (t = 1/2 CONVENTION, α̂ + β̂ + γ̂ = 1/2)")
    print("Dimension formula: dim(g) = (α̂-1)(β̂-1)(γ̂-1) / (α̂β̂γ̂)")
    print("Source: arXiv:2311.05358 Table 1 (Bishler, Mironov, Morozov)")
    print("=" * 100)

    header = f"{'Algebra':<18} {'α̂':>10} {'β̂':>10} {'γ̂':>10} {'sum':>8} {'dim_check':>10}"
    print(header)
    print("-" * 100)

    for alg in algebras:
        nv = alg.norm_vogel_t_half
        s = nv.alpha_hat + nv.beta_hat + nv.gamma_hat
        dim_c = vogel_dim_norm_t_half(nv.alpha_hat, nv.beta_hat, nv.gamma_hat)
        row = (
            f"{alg.name:<18} {fmt_frac(nv.alpha_hat):>10} {fmt_frac(nv.beta_hat):>10} "
            f"{fmt_frac(nv.gamma_hat):>10} {fmt_frac(s):>8} {fmt_frac(dim_c):>10}"
        )
        print(row)

    print("=" * 100)


def print_derived_quantities(algebras: List[LieAlgebraData]):
    """Print derived quantities for D-tilde^2 analysis."""
    print("\n" + "=" * 110)
    print("DERIVED QUANTITIES FOR D̃² AND NORMALIZATION ANALYSIS")
    print("=" * 110)

    header = (
        f"{'Algebra':<18} {'dim':>6} {'rank':>4} "
        f"{'|Φ⁺_ns|':>7} {'2dim-3r':>8} {'dim+2|ns|':>9} "
        f"{'3|Φ⁺_ns|':>8} {'3r/2':>6} "
        f"{'D²_excess':>9} {'Z_deficit':>9} {'sum':>6}"
    )
    print(header)
    print("-" * 110)

    for alg in algebras:
        d2_excess = alg.d_tilde_sq_power - alg.dim
        z_deficit = Fraction(alg.dim, 2) - alg.z_raw_power
        row = (
            f"{alg.name:<18} {alg.dim:>6} {alg.rank:>4} "
            f"{alg.pos_roots_ns:>7} {alg.d_tilde_sq_power:>8} "
            f"{alg.dim + 2 * alg.pos_roots_ns:>9} "
            f"{alg.norm_exponent:>8} {fmt_frac(alg.z_raw_power):>6} "
            f"{d2_excess:>9} {fmt_frac(z_deficit):>9} "
            f"{fmt_frac(d2_excess + z_deficit):>6}"
        )
        print(row)

    print("=" * 110)


def print_verification_summary(results: dict):
    """Print a summary of verification results."""
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)

    # Count checks
    total = 0
    passed = 0

    for category in ["identity_checks", "dim_formula_checks_unnorm",
                     "dim_formula_checks_norm_t_half", "dim_formula_checks_norm_sum_minus2"]:
        checks = results[category]
        cat_passed = sum(1 for c in checks if c["passed"])
        cat_total = len(checks)
        total += cat_total
        passed += cat_passed
        status = "✓ ALL PASSED" if cat_passed == cat_total else "✗ SOME FAILED"
        print(f"\n  {category}: {cat_passed}/{cat_total} {status}")

        # Print failures
        for c in checks:
            if not c["passed"]:
                print(f"    FAILED: {c['algebra']}: {c['check']}")
                print(f"      LHS = {c.get('LHS', c.get('computed', '?'))}, "
                      f"RHS = {c.get('RHS', c.get('expected', '?'))}")

    overall = "✓ ALL VERIFICATIONS PASSED" if results["all_passed"] else "✗ SOME VERIFICATIONS FAILED"
    print(f"\n  OVERALL: {passed}/{total} {overall}")
    print("=" * 80)


def print_exceptional_detail(algebras: List[LieAlgebraData]):
    """Print detailed data for exceptional algebras."""
    print("\n" + "=" * 100)
    print("DETAILED VOGEL PARAMETERS FOR EXCEPTIONAL LIE ALGEBRAS")
    print("=" * 100)

    exceptional = [a for a in algebras if a.cartan_type in ("G2", "F4", "E6", "E7", "E8")]

    for alg in exceptional:
        u = alg.unnorm_vogel
        nv = alg.norm_vogel_t_half
        nv2 = alg.norm_vogel_sum_minus2

        print(f"\n  {alg.name}:")
        print(f"    rank = {alg.rank}, dim = {alg.dim}, h∨ = {alg.h_vee}")
        print(f"    |Φ⁺| = {alg.pos_roots}, |Φ⁺_ns| = {alg.pos_roots_ns}, 3|Φ⁺_ns| = {alg.norm_exponent}")
        print(f"    Unnormalized (α=-2, sum=h∨): α={fmt_frac(u.alpha)}, β={fmt_frac(u.beta)}, γ={fmt_frac(u.gamma)}")
        print(f"    Normalized (sum=1/2): α̂={fmt_frac(nv.alpha_hat)}, β̂={fmt_frac(nv.beta_hat)}, γ̂={fmt_frac(nv.gamma_hat)}")
        print(f"    Normalized (sum=-2): α̂={fmt_frac(nv2.alpha_hat)}, β̂={fmt_frac(nv2.beta_hat)}, γ̂={fmt_frac(nv2.gamma_hat)}")
        print(f"    D̃² power = {alg.d_tilde_sq_power}, Z_raw = {fmt_frac(alg.z_raw_power)}, Norm = {alg.norm_exponent}")


def print_vogel_plane_lines(algebras: List[LieAlgebraData]):
    """Print the Vogel plane line structure."""
    print("\n" + "=" * 100)
    print("VOGEL PLANE LINE STRUCTURE")
    print("=" * 100)
    print("""
The Vogel plane P²/S₃ parameterizes all simple Lie algebras by three
homogeneous coordinates (α:β:γ). The algebras lie on four lines:

  1. SL-line (A-series):  γ varies with α=-2, β=2 fixed
  2. SO-line (B/D-series): γ varies with α=-2, β=4 fixed
  3. SP-line (C-series):  β varies with α=-2, γ=1 fixed
  4. Exceptional points:   isolated points off the classical lines

Key observation: For all exceptional algebras, γ̂ = 1/3 (t=1/2 convention).
This means all exceptional algebras lie on a line in the Vogel plane!
""")

    # Verify the line structure
    print("  SL-line (A_n): α = -2, β = 2, γ = n+1")
    a_algs = [a for a in algebras if a.cartan_type == "A"]
    for a in a_algs[:5]:
        u = a.unnorm_vogel
        print(f"    {a.name}: γ = {fmt_frac(u.gamma)}, γ̂(½) = {fmt_frac(a.norm_vogel_t_half.gamma_hat)}")

    print("\n  SO-line (B_n/D_n): α = -2, β = 4, γ varies")
    bd_algs = [a for a in algebras if a.cartan_type in ("B", "D")]
    for a in bd_algs[:6]:
        u = a.unnorm_vogel
        print(f"    {a.name}: γ = {fmt_frac(u.gamma)}, γ̂(½) = {fmt_frac(a.norm_vogel_t_half.gamma_hat)}")

    print("\n  SP-line (C_n): α = -2, γ = 1, β varies")
    c_algs = [a for a in algebras if a.cartan_type == "C"]
    for a in c_algs[:5]:
        u = a.unnorm_vogel
        print(f"    {a.name}: β = {fmt_frac(u.beta)}, β̂(½) = {fmt_frac(a.norm_vogel_t_half.beta_hat)}")

    print("\n  Exceptional line: γ̂(½) = 1/3 for ALL exceptional algebras")
    exc_algs = [a for a in algebras if a.cartan_type in ("G2", "F4", "E6", "E7", "E8")]
    for a in exc_algs:
        nv = a.norm_vogel_t_half
        print(f"    {a.name}: γ̂(½) = {fmt_frac(nv.gamma_hat)} ✓" if nv.gamma_hat == Fraction(1, 3)
              else f"    {a.name}: γ̂(½) = {fmt_frac(nv.gamma_hat)} ✗")


# ============================================================================
# JSON export
# ============================================================================

def to_json_serializable(obj):
    """Convert Fraction and dataclass objects to JSON-serializable format."""
    if isinstance(obj, Fraction):
        return {"numerator": obj.numerator, "denominator": obj.denominator, "value": str(obj)}
    if isinstance(obj, bool):
        return obj
    if hasattr(obj, '__dataclass_fields__'):
        result = {}
        for key, value in obj.__dict__.items():
            result[key] = to_json_serializable(value)
        return result
    if isinstance(obj, list):
        return [to_json_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    return obj


def export_json(algebras: List[LieAlgebraData], results: dict, filepath: str):
    """Export all data to JSON."""
    data = {
        "metadata": {
            "description": "Vogel parameter table for all simple Lie algebras",
            "references": [
                "Vogel (1999) hep-th/9912204",
                "Mkrtchyan arXiv:1109.1668",
                "Landsberg-Manivel math/0401296",
                "Bishler-Mironov-Morozov arXiv:2311.05358",
            ],
            "conventions": {
                "unnorm": "α=-2, α+β+γ=h∨, dim=(α-2h∨)(β-2h∨)(γ-2h∨)/(αβγ)",
                "norm_t_half": "α̂+β̂+γ̂=1/2, dim=(α̂-1)(β̂-1)(γ̂-1)/(α̂β̂γ̂)",
                "norm_sum_minus2": "α̂+β̂+γ̂=-2, dim=(α̂+4)(β̂+4)(γ̂+4)/(α̂β̂γ̂)",
            },
        },
        "algebras": to_json_serializable(algebras),
        "verification": to_json_serializable(results),
    }

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)

    print(f"\n  Data exported to: {filepath}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("VOGEL PARAMETER TABLE COMPUTATION FOR ALL SIMPLE LIE ALGEBRAS")
    print("=" * 80)

    # Build data
    classical = build_classical_algebras()
    exceptional = build_exceptional_algebras()
    all_algebras = classical + exceptional

    # Sort by type then rank
    type_order = {"A": 0, "B": 1, "C": 2, "D": 3, "G2": 4, "F4": 5, "E6": 6, "E7": 7, "E8": 8}
    all_algebras.sort(key=lambda a: (type_order.get(a.cartan_type, 9), a.rank))

    # Print tables
    print_main_table(all_algebras)
    print_paper_normalization_table(all_algebras)
    print_normalized_sum_minus2_table(all_algebras)
    print_derived_quantities(all_algebras)
    print_exceptional_detail(all_algebras)
    print_vogel_plane_lines(all_algebras)

    # Verify
    results = verify_all(all_algebras)
    print_verification_summary(results)

    # Export
    output_path = "/home/z/my-project/hopf-decoherence/scripts/vogel_data.json"
    export_json(all_algebras, results, output_path)

    # Print key numerical results
    print("\n" + "=" * 80)
    print("KEY NUMERICAL RESULTS")
    print("=" * 80)

    print("\n  Unnormalized Vogel parameters (α = -2, α + β + γ = h∨):")
    print("  " + "-" * 75)
    for alg in all_algebras:
        u = alg.unnorm_vogel
        print(f"  {alg.name:<18} α={fmt_frac(u.alpha):>6}, β={fmt_frac(u.beta):>6}, "
              f"γ={fmt_frac(u.gamma):>6}, h∨={alg.h_vee}")

    print("\n  Normalized Vogel parameters (α̂ + β̂ + γ̂ = -2):")
    print("  " + "-" * 75)
    for alg in all_algebras:
        nv = alg.norm_vogel_sum_minus2
        print(f"  {alg.name:<18} α̂={fmt_frac(nv.alpha_hat):>8}, β̂={fmt_frac(nv.beta_hat):>8}, "
              f"γ̂={fmt_frac(nv.gamma_hat):>8}")

    # Highlight: exceptional line γ̂ = -4/3 in sum=-2 convention
    print("\n  Exceptional line observation (sum=-2 convention):")
    print("  All exceptional algebras have γ̂ = -4/3:")
    for alg in all_algebras:
        if alg.cartan_type in ("G2", "F4", "E6", "E7", "E8"):
            nv = alg.norm_vogel_sum_minus2
            print(f"    {alg.name}: γ̂ = {fmt_frac(nv.gamma_hat)}")

    return all_algebras, results


if __name__ == "__main__":
    algebras, results = main()
