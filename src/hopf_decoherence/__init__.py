"""
hopf-decoherence
================

Companion code for the paper *Hochschild Cohomology of Small Quantum Groups
at Roots of Unity*.

The paper proposes the conjecture
``dim_C HH^2(u_q(g), C) = binom(n+1, 2) + 2|Phi^+|`` for any finite-dimensional
complex simple Lie algebra ``g`` of rank ``n`` at an odd root of unity
``q = e^(2*pi*i/ell)``, and verifies it by direct bar-complex computation in
the ``A_1`` case (see ``scripts/verify_sl2_hh2.py`` and
``tests/test_sl2_hh2_bar_complex.py``).

This package also contains an auxiliary computational framework for the BCGP
non-semisimple TQFT and BTZ black-hole entropy programme that the paper
references speculatively in Section 7. The auxiliary modules are:

  * ``q_algebra`` — q-numbers, Weyl modules.
  * ``coproduct`` — coproduct matrices and rank computations.
  * ``rank_deficiency`` — ``D_2(ell) = (ell^3 - ell) / 6`` (sl_2 coproduct
    rank deficiency).
  * ``modified_trace`` — GPY modified trace.
  * ``projective_modules`` — projective indecomposable P(j).
  * ``defect_tqft`` — BCGP + ker(Delta) defect TQFT machinery.

The main bar-complex verification (``scripts/verify_sl2_hh2.py``) is a
standalone script and does not import from this package.

Author
------
Parham Khairkhah <pkhairkh@icloud.com>
"""

__version__ = "1.0.0"
__author__ = "Parham Khairkhah"
__email__ = "pkhairkh@icloud.com"
__license__ = "MIT"

from .q_algebra import (
    q_number, q_factorial, q_binomial,
    build_weyl_module, verify_algebra_relations, verify_sl3_relations,
    compute_rank, RANK_TOL,
)
from .coproduct import (
    coproduct_matrices_sl2, coproduct_matrices_sl3,
    compute_phi_rank, compute_cross_rep_coproduct_rank,
)
from .rank_deficiency import (
    D_ell, D_k1k2, expected_rank,
    verlinde_rank_sl3, deficiency_fraction_sl2,
    deficiency_fraction_sl3,
)
from .modified_trace import (
    modified_qdim, ordinary_qdim, global_dimension,
    compute_kappa, modified_trace_on_morphism,
    verify_modified_trace_properties,
)
from .projective_modules import (
    build_regular_representation, build_projective_module,
    build_all_projectives_sl2,
)
from .triangulation import (
    Triangulation, Tetrahedron,
    triangulate_S3, triangulate_S1xS2, triangulate_lens_space,
)
from .state_sum import (
    compute_state_sum, partition_S3, partition_S1xS2, partition_lens_space,
)
from .defect_networks import (
    DefectLine, DefectNetwork,
    compute_defect_state_sum, compute_defect_entropy_correction,
    compute_entropy_deficit_limit, verify_defect_tqft_consistency,
)
from .rt_comparison import (
    rt_quantum_dimension, rt_conformal_weight, rt_conformal_weight_wzw,
    rt_global_dimension_squared, rt_integrable_spins,
    rt_solid_torus_partition, rt_solid_torus_partition_components,
    rt_entropy, rt_extract_log_correction,
    rt_entropy_scaled_beta, rt_extract_log_correction_scaled,
    rt_twist_factor, rt_lens_space_partition, rt_s3_partition,
    compute_all_entropies, extract_all_log_corrections,
)
from .deficiency_log_connection import (
    D_2 as D_2_deficiency,
    expected_rank_sl2,
    deficiency_fraction_sl2 as def_frac_sl2,
    verlinde_rank_sl3 as verlinde_sl3,
    D_min_sl3,
    deficiency_fraction_sl3 as def_frac_sl3,
    compute_sl3_coproduct_rank,
    Z_cont_full_trace_sl2,
    Z_BTZ_full_trace_sl2,
    Z_BTZ_modified_sl2,
    extract_log_correction as extract_log_corr,
    analyze_scaling,
    analytical_asymptotics_sl2,
    sl_N_log_correction,
    sl_N_deficiency_summary,
    deficiency_log_derivation,
)
from .information_radical import (
    radical_dim_simplified, radical_dim_actual,
    total_radical_dim_simplified, total_radical_dim_actual,
    total_semisimple_dim_simplified, total_semisimple_dim_actual,
    radical_fraction_simplified, radical_fraction_actual, radical_fraction_regular_rep,
    page_time_equality, page_time_regular_rep,
    page_curve_entanglement, page_curve_evolution,
    channel_capacity_scaled_beta,
    radical_channel_capacity_analytical, radical_channel_capacity_weighted_log,
    verify_channel_capacity_growth,
)
from .wormhole_partition import (
    verlinde_s_matrix, s_matrix_element, s_matrix_vacuum_row,
    s_matrix_continuous_0, s_matrix_continuous_discrete,
    s_matrix_continuous_continuous,
    wormhole_Z_full_smatrix, wormhole_Z_diagonal,
    wormhole_Z_diagonal_continuous,
    wormhole_partition_complete,
    compute_wormhole_entropy, compute_wormhole_entropy_fast,
    extract_log_correction as extract_wormhole_log_correction,
    verify_verlinde_unitarity, verify_triple_product,
    analytical_asymptotics_wormhole, analytical_asymptotics_wormhole_detailed,
    compute_boundary_entanglement, er_epr_analysis,
    compare_solid_torus_wormhole,
)
from .modular_invariance_proof import (
    wzw_s_matrix, lcft_s_matrix_standard, lcft_s_matrix_projective,
    nonsemisimple_s_matrix, verify_s_matrix_properties,
    full_thermal_character_discrete, full_thermal_character_continuous,
    modified_trace_character_discrete, modified_trace_character_continuous,
    Z_full, Z_modified, D_tilde_squared,
    verify_full_character_modular_covariance,
    verify_modified_character_breaks_modular,
    compute_partition_function_modular_invariance,
    steinberg_analysis, compare_verlinde_bcgp,
    verify_alternating_sum_identity,
    prove_modular_invariance, s_transform_analysis,
    verify_D_tilde_asymptotics,
)
from .radical_zero_mode_map import (
    KillingZeroMode, enumerate_zero_modes, zero_mode_commutation_relations,
    zero_mode_entropy_contribution,
    radical_structure, radical_dimension_total,
    radical_channel_capacity_analytical, radical_channel_capacity_numerical,
    radical_zero_mode_map,
    log_correction_decomposition,
    modified_trace_over_suppression_mechanism,
    verify_correspondence,
    create_visualization,
)
from .information_theorem import (
    Z_mod, Z_full, compute_entropy,
    verify_information_loss_modified_trace,
    verify_sign_cancellation_information_loss,
    verify_full_trace_preserves_information,
    radical_channel_capacity,
    radical_channel_capacity_scaled,
    channel_capacity_growth_table,
    page_curve_bh_analog,
    page_time_analysis,
    verify_modified_trace_zero_capacity,
    compute_half_shift,
    prove_radical_equals_bh_interior,
    detailed_entropy_comparison,
    detailed_channel_capacity_table,
    information_paradox_resolution_table,
)
from .positivity_theorem import (
    modified_qdim as positivity_modified_qdim,
    full_dim, conformal_weight as positivity_conformal_weight,
    z_bcgp_discrete, z_bcgp_continuous, z_bcgp,
    z_bcgp_discrete_terms,
    z_full_discrete, z_full_continuous, z_full,
    z_full_discrete_terms,
    modified_global_dimension as positivity_global_dimension,
    demonstrate_negative_probabilities,
    demonstrate_zero_dimension,
    analytical_proof,
    prove_alternating_sum_identity,
    positivity_theorem,
    phase_diagram,
    deficit_analysis,
    physical_interpretation,
    run_comprehensive_test,
)
from .continuity_theorem import (
    q_number as ct_q_number,
    generic_quantum_dimension,
    semisimple_raw_trace,
    full_trace_raw,
    modified_trace_raw,
    D_tilde_squared as ct_D_tilde_squared,
    continuity_analysis_raw,
    sign_alternation_analysis,
    compute_log_coefficients,
    formal_continuity_proof,
    categorical_vs_physical_analysis,
    verify_continuity_theorem,
)
from .master_theorem import (
    modified_qdim as mt_modified_qdim,
    conformal_weight as mt_conformal_weight,
    typical_weight as mt_typical_weight,
    D_tilde_squared as mt_D_tilde_squared,
    full_projective_dim,
    Z_full_disc, Z_full_cont, Z_full,
    Z_bcgp_disc, Z_bcgp_cont, Z_bcgp,
    proof_1_hilbert_space,
    proof_2_positivity,
    proof_3_modular_invariance,
    proof_4_chern_simons,
    proof_5_radical_zero_mode,
    proof_6_information_theorem,
    proof_7_wormhole,
    proof_8_continuity,
    corollary_1_log_entropy,
    corollary_2_radical_bh_interior,
    corollary_3_correction_factor,
    corollary_4_n_boundary,
    verify_all,
    master_identity,
    summary_table,
)
from .formal_proof_positivity import (
    modified_qdim as fp_modified_qdim,
    full_dim as fp_full_dim,
    conformal_weight as fp_conformal_weight,
    z_bcgp_discrete as fp_z_bcgp_discrete,
    z_full_discrete as fp_z_full_discrete,
    theorem_1_verify,
    lemma_1_verify,
    theorem_2_verify,
    theorem_3_verify,
    numerical_verification_table,
    generate_latex_proof,
    run_formal_proof,
    THEOREM_1_LATEX,
    LEMMA_1_LATEX,
    THEOREM_2_LATEX,
    THEOREM_3_LATEX,
    PHYSICAL_INTERPRETATION_LATEX,
)
from .formal_proof_modular_invariance import (
    theorem_A_full_characters_modular_covariant,
    theorem_B_modified_characters_not_modular,
    theorem_C_s_matrix_rank,
    corollary_ads_cft_consistency,
    compute_rank_table,
    print_rank_table_latex,
    formal_proof,
    prove_alternating_sum_identity,
    prove_outer_product_rank_1,
)
from .formal_proof_continuity import (
    theorem_continuity_physical_Z,
    theorem_discontinuity_modified_trace,
    lemma_dimension_counting,
    physical_argument,
    verify_sign_continuity,
    verify_partition_function_ratio,
    verify_log_coefficients,
    latex_theorem_statements,
    run_comprehensive_verification as run_formal_continuity_verification,
)
from .formal_proof_hilbert_space import (
    simple_module_dim,
    projective_module_dim,
    conformal_weight as fps_conformal_weight,
    typical_conformal_weight as fps_typical_conformal_weight,
    loewy_structure,
    verify_theorem1,
    verify_theorem2,
    verify_theorem3,
    verify_theorem4,
    full_thermal_trace,
    full_thermal_trace_discrete,
    full_thermal_trace_continuous,
    bcgp_partition,
    compute_correction_factor,
    verify_corollary,
    verify_all as formal_proof_verify_all,
    verify_alternating_sum_identity,
    latex_proof_summary,
)
from .sl3_radical_structure import (
    enumerate_alcove, weyl_dim_sl3, alcove_summary,
    modified_qdim_sl3, modified_qdim_sl3_absolute,
    modified_global_dimension_sl3,
    verify_steinberg_vanishing,
    alternating_sum_sl3, alternating_sum_sl2,
    compare_alternating_sums,
    loewy_structure_sl3,
    radical_fraction_sl3, radical_fraction_table,
    modified_trace_projector_ratio_sl3,
    modified_trace_vs_full_trace_sl3,
    coproduct_rank_sl3_on_R,
    radical_channel_capacity_sl3,
    channel_capacity_growth_sl3,
    compare_sl2_sl3,
    verify_sl3_radical_structure,
)
from .holographic_dictionary import (
    HolographicDictionaryEntry,
    build_holographic_dictionary,
    print_dictionary_table,
    BulkReconstructionMap,
    verify_bulk_reconstruction,
    entanglement_wedge_analysis,
    verify_entanglement_wedge,
    QuantumErrorCode,
    verify_error_correction,
    compute_tqft_entanglement_entropy,
    page_curve_comparison,
    verify_holographic_dictionary,
)
from .multi_boundary_predictions import (
    Z_solid_torus_disc, Z_solid_torus_cont,
    Z_wormhole_disc, Z_wormhole_disc_diagonal,
    Z_triple_boundary_disc, Z_triple_boundary_disc_alternative,
    Z_triple_boundary_diagonal,
    Z_n_boundary_disc,
    compute_entropy as mb_compute_entropy,
    extract_log_coefficient as mb_extract_log_coefficient,
    zero_mode_counting, generate_prediction_table,
    er_epr_entanglement_entropy,
    er_epr_multi_boundary_analysis,
    er_epr_scaling_analysis,
    radical_contribution_analysis,
    verify_n1_solid_torus, verify_n2_wormhole, verify_n2_wormhole_diagonal,
    predict_n3_triple, predict_n3_triple_diagonal,
    run_multi_boundary_verification,
)
from .bulk_reconstruction import (
    HKLLKernel,
    InteriorOperator,
    interior_operator_summary,
    compute_coproduct_pbw_rank_on_projective,
    coproduct_cartan_rank_on_projective,
    analyze_coproduct_all_projectives,
    compare_coproduct_rank_weyl_vs_projective,
    verify_bulk_reconstruction as bulk_reconstruction_verify,
    coproduct_rank_table,
)
from .zero_mode_analysis import (
    D2_coproduct,
    total_radical_dimension,
    radical_structure_per_j,
    compute_effective_zero_mode_count,
    boltzmann_weighted_zero_mode_count,
    zero_mode_partition_function_contribution,
    radical_dominance_analysis,
    derive_half_shift_analytical,
    verify_zero_mode_count,
    verify_r_scaling,
    three_special_radical_points,
    generate_results_json,
)
from .sl4_verification import (
    S00_SU4,
    S00_SUN,
    enumerate_alcove_sl4,
    modified_qdim_disc_sl4,
    D2_disc_sl4,
    verify_sl4,
)
from .sl4_full_numerical import (
    casimir_sl4,
    conformal_weight_sl4,
    weyl_dim_sl4,
    modified_qdim_disc_sl4 as sl4_mod_qdim_disc,
    modified_qdim_cont_sl4,
    D2_disc_sl4 as sl4_D2_disc,
    D2_cont_sl4,
    Z_full_disc_sl4,
    Z_mod_disc_sl4,
    Z_full_cont_sl4,
    Z_mod_cont_sl4,
    S00_SU4 as sl4_S00_SU4,
    S00_SUN as sl4_S00_SUN,
    compute_single_r as sl4_compute_single_r,
    verify_sl4_full,
)
