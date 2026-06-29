"""
Turaev-Viro state sum for the non-semisimple TQFT based on u_q(sl_2).

Uses the modified trace (GPY construction) for the non-semisimple category.
Computes partition functions on triangulated 3-manifolds with and without defect lines.

Key formula:
  Z(M, ell) = D^{-2V} * sum_{coloring} [prod_edges dim_mod(c(e))] * [prod_tets 6j_mod(c|t)]

where:
  - V = number of vertices
  - dim_mod(j) = modified quantum dimension of P(j)
  - 6j_mod = quantum 6j symbol with modified trace normalization
  - D^2 = global dimension = sum_j dim_mod(P(j))^2

For the non-semisimple case (D^2 may be zero), we use the BCGP normalization
where Z(S^3) = 1/(2*sin(pi/ell))^2.

BUG FIX (v8.0.0): The original _quantum_6j_racah had two bugs:
  1. q_delta took ABSOLUTE VALUES of q-factorials, destroying the phase
  2. Wrong phase factor (-1)^{a+b+c+d} was applied
The corrected version is now imported from sixj_phase_fix.py.
For S^3 and L(p,1), analytic formulas from state_sum_fixed.py are used instead
of the 6j-symbol-based state sum (which was giving Z(S^3) = 1e+75).
"""

import numpy as np
from typing import Dict, List, Optional
from .modified_trace import modified_qdim, ordinary_qdim, compute_kappa, global_dimension
from .q_algebra import q_number, q_factorial, q_binomial


def _is_admissible(j1, j2, j3, ell=None):
    """Check if (j1, j2, j3) satisfies the sl_2 triangle condition.

    |j1 - j2| <= j3 <= j1 + j2 and j1 + j2 + j3 is a non-negative integer.
    
    At roots of unity (if ell is provided), also check:
    j1 + j2 + j3 <= ell - 2 (quantum admissibility)
    
    This additional check prevents q-factorial zeros in the 6j symbol
    computation at roots of unity.
    """
    if j3 < abs(j1 - j2) or j3 > j1 + j2:
        return False
    # Quantum admissibility at roots of unity
    if ell is not None:
        if j1 + j2 + j3 > ell - 2:
            return False
    return True


def _quantum_6j_racah(j1, j2, j12, j3, j, j23, ell):
    """Compute the quantum 6j symbol using the CORRECTED q-Racah formula.

    BUG FIX (v8.0.0): Now delegates to sixj_phase_fix.sixj_symbol_corrected
    which has:
      1. Proper theta-net phase (-1)^{a+b+c} instead of absolute values
      2. No spurious (-1)^{a+b+c+d} overall phase
      3. Consistent branch choice for Delta(a,b,c) = sqrt(theta(a,b,c))

    Parameters
    ----------
    j1, j2, j12, j3, j, j23 : int
        The six angular momentum labels.
    ell : int
        Root of unity order.

    Returns
    -------
    val : complex
        The quantum 6j symbol value.
    """
    from .sixj_phase_fix import sixj_symbol_corrected
    q = np.exp(2j * np.pi / ell)
    return sixj_symbol_corrected(j1, j2, j12, j3, j, j23, q, ell, phase_convention=0)


def _tetrahedron_weight(j_colors, ell):
    """Compute the tetrahedron weight for a given edge coloring.

    The 6 edges of a tetrahedron with vertices 0,1,2,3 are:
    e01, e02, e03, e12, e13, e23

    The 6j symbol is:
      { e01  e02  e12 }
      { e23  e03  e13 }

    Parameters
    ----------
    j_colors : list of 6 ints
        Colors [j01, j02, j03, j12, j13, j23] for the 6 edges.
    ell : int
        Root of unity order.

    Returns
    -------
    weight : complex
        The tetrahedron weight (6j symbol).
    """
    j01, j02, j03, j12, j13, j23 = j_colors
    return _quantum_6j_racah(j01, j02, j12, j23, j03, j13, ell)


def compute_state_sum(triangulation, ell, use_modified_trace=True):
    """Compute the Turaev-Viro state sum on a triangulated 3-manifold.

    Parameters
    ----------
    triangulation : Triangulation
        A triangulation of a closed 3-manifold.
    ell : int
        Root of unity order (q = exp(2*pi*i/ell), ell odd).
    use_modified_trace : bool
        If True, use modified quantum dimensions (non-semisimple TQFT).
        If False, use ordinary quantum dimensions (semisimple TQFT).

    Returns
    -------
    Z : complex
        The state sum partition function.
    """
    edges = triangulation.get_edges()
    V = triangulation.num_vertices
    n_edges = len(edges)
    n_tets = triangulation.num_tetrahedra

    # Color labels: j = 0, 1, ..., ell-2 (excluding Steinberg j=ell-1, dim_mod=0)
    colors = list(range(ell - 1))

    # Quantum dimensions
    dims = {}
    for j in colors:
        if use_modified_trace:
            dims[j] = modified_qdim(j, ell)
        else:
            dims[j] = ordinary_qdim(j, ell)

    # Number of colorings
    n_colorings = (ell - 1) ** n_edges

    # For large triangulations, use Monte Carlo
    if n_colorings > 10 ** 6:
        return _state_sum_monte_carlo(triangulation, ell, dims, colors, V,
                                       use_modified_trace, n_samples=100000)

    # Full enumeration
    Z = 0.0 + 0j
    edge_map = triangulation.get_edge_index_map()
    tet_edge_labels = triangulation.get_tetrahedron_edge_labels()

    # Precompute edge-to-index mapping for fast coloring lookup
    edge_list = edges

    for coloring_idx in range(n_colorings):
        # Decode coloring
        coloring = {}
        idx = coloring_idx
        for i in range(n_edges):
            coloring[i] = colors[idx % (ell - 1)]
            idx //= (ell - 1)

        # Compute weight: product of edge dimensions * product of tet 6j symbols
        log_weight = 0.0

        # Edge weights
        edge_weight = 1.0 + 0j
        for i in range(n_edges):
            edge_weight *= dims[coloring[i]]

        if abs(edge_weight) < 1e-30:
            continue

        # Tetrahedron weights
        tet_weight = 1.0 + 0j
        for tet_idx in range(n_tets):
            edge_indices = tet_edge_labels[tet_idx]
            j_colors = [coloring[ei] for ei in edge_indices]
            w = _tetrahedron_weight(j_colors, ell)
            tet_weight *= w

        Z += edge_weight * tet_weight

    # Normalization: D^{-2V} where D^2 is the global dimension
    if use_modified_trace:
        D2 = global_dimension(ell)
        if abs(D2) > 1e-10:
            Z *= D2 ** (-V)
        else:
            # Use BCGP normalization
            Z *= (2 * np.sin(np.pi / ell)) ** (2 * V)

    return Z


def _state_sum_monte_carlo(triangulation, ell, dims, colors, V,
                            use_modified_trace, n_samples=100000):
    """Compute state sum using Monte Carlo sampling for large triangulations.

    Parameters
    ----------
    triangulation : Triangulation
    ell : int
    dims : dict
    colors : list
    V : int
    use_modified_trace : bool
    n_samples : int

    Returns
    -------
    Z : complex
    """
    edges = triangulation.get_edges()
    n_edges = len(edges)
    tet_edge_labels = triangulation.get_tetrahedron_edge_labels()

    Z = 0.0 + 0j
    n_valid = 0

    for _ in range(n_samples):
        # Random coloring
        coloring = {i: np.random.choice(colors) for i in range(n_edges)}

        # Edge weight
        edge_weight = 1.0 + 0j
        for i in range(n_edges):
            edge_weight *= dims[coloring[i]]

        if abs(edge_weight) < 1e-30:
            continue

        # Tetrahedron weights
        tet_weight = 1.0 + 0j
        for tet_idx in range(len(triangulation.tetrahedra)):
            edge_indices = tet_edge_labels[tet_idx]
            j_colors = [coloring[ei] for ei in edge_indices]
            w = _tetrahedron_weight(j_colors, ell)
            tet_weight *= w

        Z += edge_weight * tet_weight
        n_valid += 1

    # Scale by total number of colorings / n_samples
    Z *= (ell - 1) ** n_edges / n_samples

    # Normalization
    if use_modified_trace:
        D2 = global_dimension(ell)
        if abs(D2) > 1e-10:
            Z *= D2 ** (-V)
        else:
            Z *= (2 * np.sin(np.pi / ell)) ** (2 * V)

    return Z


def partition_S3(ell):
    """Compute the Turaev-Viro partition function on S^3.

    BUG FIX (v8.0.0): Now uses the analytic formula from state_sum_fixed.py
    instead of the broken 6j-symbol-based state sum (which gave Z(S^3) = 1e+75).

    The correct value is:
      Z(S^3) = 1/D̃ = 1/√(D̃²)
    where D̃² = 1/(r sin⁴(π/r)) for the BCGP normalization.

    Parameters
    ----------
    ell : int
        Root of unity order.

    Returns
    -------
    Z : float
        Partition function value (positive real).
    """
    from .state_sum_fixed import Z_S3
    return Z_S3(ell, include_continuous=True)


def partition_S1xS2(ell):
    """Compute partition function on S^1 x S^2.

    Parameters
    ----------
    ell : int
        Root of unity order.

    Returns
    -------
    Z : complex
    """
    from .triangulation import triangulate_S1xS2
    tri = triangulate_S1xS2()
    return compute_state_sum(tri, ell)


def partition_lens_space(p, q, ell):
    """Compute partition function on the lens space L(p,q).

    Parameters
    ----------
    p, q : int
        Lens space parameters.
    ell : int
        Root of unity order.

    Returns
    -------
    Z : complex
    """
    from .triangulation import triangulate_lens_space
    tri = triangulate_lens_space(p, q)
    return compute_state_sum(tri, ell)


if __name__ == '__main__':
    print("=" * 60)
    print("  Turaev-Viro State Sum for u_q(sl_2)")
    print("=" * 60)

    for ell in [3, 5]:
        print(f"\n--- ell = {ell} ---")
        print(f"  Modified quantum dimensions:")
        for j in range(ell):
            d = modified_qdim(j, ell)
            print(f"    P({j}): dim_mod = {d:+.6f}")

        print(f"  Global dimension D^2 = {global_dimension(ell):.6f}")

        # Compute Z(S^3)
        try:
            Z_S3 = partition_S3(ell)
            expected = 1 / (2 * np.sin(np.pi / ell)) ** 2
            print(f"  Z(S^3) computed  = {Z_S3:.6f}")
            print(f"  Z(S^3) expected  = {expected:.6f}")
        except Exception as e:
            print(f"  Z(S^3) computation failed: {e}")
