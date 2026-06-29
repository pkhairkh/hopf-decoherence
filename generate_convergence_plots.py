"""
Generate publication-quality convergence plots for the Master Identity verification.

Master Identity: -3/2 = -2 (modified trace) + (+1/2) (radical)

5 plots:
  1. Power-law convergence (master plot)
  2. Finite-difference exponent convergence
  3. Radical shift convergence
  4. D₂/r³ convergence to 1/6
  5. Summary/dashboard (multi-panel)
"""

import numpy as np
from scipy import integrate, special
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import warnings

warnings.filterwarnings("ignore", category=integrate.IntegrationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================================
# FONT SETUP
# ============================================================================
fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/SarasaMonoSC-Regular.ttf')
fm.fontManager.addfont('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Sarasa Mono SC']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# COLOR SCHEME (professional)
# ============================================================================
COLOR_FULL = '#2166AC'    # deep blue
COLOR_BCGP = '#B2182B'    # deep red
COLOR_SHIFT = '#1B7837'   # deep green
COLOR_D2 = '#762A83'      # purple
COLOR_REF = '#666666'     # gray for reference lines
COLOR_MARKER_FULL = '#4393C3'
COLOR_MARKER_BCGP = '#D6604D'
COLOR_MARKER_SHIFT = '#5AAE61'
COLOR_MARKER_D2 = '#9970AB'

# ============================================================================
# CORE FORMULAS (self-contained)
# ============================================================================

def modified_qdim(j, r):
    if j == r - 1 or j < 0 or j >= r:
        return 0.0
    return ((-1)**j * np.sin(np.pi*(j+1)/r)) / (r * np.sin(np.pi/r)**2)

def conformal_weight(j, r):
    return j*(j+2) / (4.0*r)

def typical_weight(alpha, r):
    return (alpha**2 - 1) / (4.0*r)

def D_tilde_squared(r, include_continuous=True):
    sin4 = np.sin(np.pi/r)**4
    if include_continuous:
        return 1.0 / (r * sin4)
    else:
        return 1.0 / (2.0 * r * sin4)

def full_projective_dim(j, r):
    if j < 0 or j >= r:
        return 0.0
    return r if j == r - 1 else 2 * r

# ============================================================================
# PARTITION FUNCTIONS (efficient)
# ============================================================================

def Z_full_disc(beta, r):
    """Full discrete sector: sum_j dim(P_j) * exp(-beta * h_j)"""
    Z = 0.0
    for j in range(r):
        Z += full_projective_dim(j, r) * np.exp(-beta * conformal_weight(j, r))
    return Z

def Z_full_cont(beta, r):
    """Full continuous sector using analytical error function formula."""
    if beta <= 1e-15:
        return float(r * r)
    sqrt_br = np.sqrt(beta * r)
    erf_val = special.erf(sqrt_br / 2.0)
    return r * np.exp(beta / (4.0 * r)) * np.sqrt(np.pi * r / beta) * erf_val

def Z_full_unnorm(beta, r):
    return Z_full_disc(beta, r) + Z_full_cont(beta, r)

def Z_full_norm(beta, r):
    return Z_full_unnorm(beta, r) / D_tilde_squared(r)

def Z_bcgp_disc(beta, r):
    """BCGP discrete sector: sum_j d_tilde(P_j) * exp(-beta * h_j)"""
    Z = 0.0
    for j in range(r):
        d = modified_qdim(j, r)
        h = conformal_weight(j, r)
        Z += d * np.exp(-beta * h)
    return Z

def Z_bcgp_cont(beta, r):
    """BCGP continuous sector with numerical integration."""
    if beta <= 1e-15:
        return 0.0
    sin_pi_r = np.sin(np.pi / r)
    prefactor = 1.0 / (r * sin_pi_r**2)
    
    def integrand(alpha):
        d = prefactor * np.sin(np.pi * alpha / r)
        return d * np.exp(-beta * typical_weight(alpha, r))
    
    # Effective integration range
    sigma = np.sqrt(4.0 * r / beta)
    alpha_max = min(float(r) - 1e-8, 8.0 * sigma)
    
    Z, _ = integrate.quad(integrand, 1e-8, alpha_max, limit=200)
    return Z

def Z_bcgp_unnorm(beta, r):
    return Z_bcgp_disc(beta, r) + Z_bcgp_cont(beta, r)

def Z_bcgp_norm(beta, r):
    return Z_bcgp_unnorm(beta, r) / D_tilde_squared(r)

def D_ell(r):
    """Coproduct deficiency D_2(r) = (r^3 - r) / 6"""
    return (r**3 - r) // 6

# ============================================================================
# PRECOMPUTE ALL DATA
# ============================================================================

def precompute_data(r_max=501, beta=1.0):
    """Precompute all Z values exactly for r = 3, 5, ..., r_max."""
    print(f"Precomputing data: r = 3, 5, ..., {r_max}, beta = {beta}")
    t0 = time.time()
    
    r_values = list(range(3, r_max + 1, 2))
    data = {}
    
    for i, r in enumerate(r_values):
        zfn = Z_full_norm(beta, r)
        zbn = Z_bcgp_norm(beta, r)
        data[r] = {
            'Z_full_norm': zfn,
            'Z_bcgp_norm': zbn,
            'ln_Z_full': np.log(zfn) if zfn > 0 else float('nan'),
            'ln_Z_bcgp': np.log(zbn) if zbn > 0 else float('nan'),
            'exact': True,
        }
        if (i+1) % 50 == 0:
            print(f"  r = {r} ({i+1}/{len(r_values)})")
    
    elapsed = time.time() - t0
    print(f"  Precomputation done in {elapsed:.1f}s ({len(data)} r values)")
    return data

# ============================================================================
# PLOT 1: Power-Law Convergence
# ============================================================================

def plot1_power_law(data, beta=1.0, output_path='/home/z/my-project/download/plot1_power_law_convergence.png'):
    """Plot ln(Z_norm) vs ln(r) with reference slopes."""
    print("\nGenerating Plot 1: Power-law convergence...")
    
    r_vals = sorted([r for r in data.keys()])
    ln_r = np.array([np.log(r) for r in r_vals])
    ln_Zf = np.array([data[r]['ln_Z_full'] for r in r_vals])
    ln_Zb = np.array([data[r]['ln_Z_bcgp'] for r in r_vals])
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    fig.set_dpi(150)
    
    # Data curves
    ax.plot(ln_r, ln_Zf, '-', color=COLOR_FULL, linewidth=2.0, label=r'$\ln Z_{\mathrm{full}}^{\mathrm{norm}}(r)$', zorder=3)
    ax.plot(ln_r, ln_Zb, '-', color=COLOR_BCGP, linewidth=2.0, label=r'$\ln Z_{\mathrm{BCGP}}^{\mathrm{norm}}(r)$', zorder=3)
    
    # Mark every 25th data point
    marker_idx = list(range(0, len(r_vals), 25))
    ax.plot(ln_r[marker_idx], ln_Zf[marker_idx], 'o', color=COLOR_MARKER_FULL, markersize=5, zorder=4)
    ax.plot(ln_r[marker_idx], ln_Zb[marker_idx], 's', color=COLOR_MARKER_BCGP, markersize=5, zorder=4)
    
    # Reference lines with slopes -3/2 and -2
    # Choose reference point at center of data
    r_ref = 101
    ln_r_ref = np.log(r_ref)
    ln_Zf_ref = data[r_ref]['ln_Z_full']
    ln_Zb_ref = data[r_ref]['ln_Z_bcgp']
    
    # Slope -3/2 reference through full trace reference point
    ref_r = np.linspace(np.log(3), np.log(501), 100)
    ref_full = ln_Zf_ref + (-1.5) * (ref_r - ln_r_ref)
    ref_bcgp = ln_Zb_ref + (-2.0) * (ref_r - ln_r_ref)
    
    ax.plot(ref_r, ref_full, '--', color=COLOR_FULL, linewidth=1.2, alpha=0.6,
            label=r'Reference slope $-3/2$', zorder=2)
    ax.plot(ref_r, ref_bcgp, '--', color=COLOR_BCGP, linewidth=1.2, alpha=0.6,
            label=r'Reference slope $-2$', zorder=2)
    
    ax.set_xlabel(r'$\ln r$', fontsize=14)
    ax.set_ylabel(r'$\ln Z^{\mathrm{norm}}(r)$', fontsize=14)
    ax.set_title(r'Master Identity: Power-Law Convergence ($\beta = 1.0$)', fontsize=15)
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='-')
    
    # Add annotation
    ax.annotate(r'$Z_{\mathrm{full}}^{\mathrm{norm}} \sim r^{-3/2}$' + '\n' + r'$Z_{\mathrm{BCGP}}^{\mathrm{norm}} \sim r^{-2}$',
                xy=(0.02, 0.02), xycoords='axes fraction', fontsize=13,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='gray', alpha=0.9),
                va='bottom', ha='left')
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")
    return output_path

# ============================================================================
# PLOT 2: Finite-Difference Exponent Convergence
# ============================================================================

def plot2_fd_convergence(data, output_path='/home/z/my-project/download/plot2_fd_convergence.png'):
    """Plot finite-difference exponent vs 1/r."""
    print("\nGenerating Plot 2: Finite-difference exponent convergence...")
    
    r_vals = sorted(data.keys())
    fd_data = []
    
    for r in r_vals:
        rp = r + 2
        rm = r - 2
        if rm < 3 or rp not in data or rm not in data:
            continue
        dp = data[rp]
        dm = data[rm]
        if (np.isfinite(dp['ln_Z_full']) and np.isfinite(dm['ln_Z_full']) and
                np.isfinite(dp['ln_Z_bcgp']) and np.isfinite(dm['ln_Z_bcgp'])):
            dlnr = np.log(rp) - np.log(rm)
            exp_full = (dp['ln_Z_full'] - dm['ln_Z_full']) / dlnr
            exp_bcgp = (dp['ln_Z_bcgp'] - dm['ln_Z_bcgp']) / dlnr
            fd_data.append({
                'r': r,
                'inv_r': 1.0 / r,
                'exp_full': exp_full,
                'exp_bcgp': exp_bcgp,
                'shift': exp_full - exp_bcgp,
            })
    
    inv_r = np.array([d['inv_r'] for d in fd_data])
    exp_full = np.array([d['exp_full'] for d in fd_data])
    exp_bcgp = np.array([d['exp_bcgp'] for d in fd_data])
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    fig.set_dpi(150)
    
    ax.plot(inv_r, exp_full, '-', color=COLOR_FULL, linewidth=2.0,
            label=r'$d\ln Z_{\mathrm{full}} / d\ln r$', zorder=3)
    ax.plot(inv_r, exp_bcgp, '-', color=COLOR_BCGP, linewidth=2.0,
            label=r'$d\ln Z_{\mathrm{BCGP}} / d\ln r$', zorder=3)
    
    # Markers
    marker_idx = list(range(0, len(fd_data), 20))
    ax.plot(inv_r[marker_idx], exp_full[marker_idx], 'o', color=COLOR_MARKER_FULL, markersize=5, zorder=4)
    ax.plot(inv_r[marker_idx], exp_bcgp[marker_idx], 's', color=COLOR_MARKER_BCGP, markersize=5, zorder=4)
    
    # Reference lines
    ax.axhline(y=-1.5, color=COLOR_FULL, linestyle='--', linewidth=1.5, alpha=0.5,
               label=r'Target: $-3/2$')
    ax.axhline(y=-2.0, color=COLOR_BCGP, linestyle='--', linewidth=1.5, alpha=0.5,
               label=r'Target: $-2$')
    
    # Annotate convergence values at smallest 1/r
    if len(fd_data) > 0:
        last = fd_data[-1]
        ax.annotate(f"r={last['r']}: {last['exp_full']:.4f}",
                    xy=(last['inv_r'], last['exp_full']),
                    xytext=(last['inv_r'] + 0.02, last['exp_full'] + 0.15),
                    fontsize=10, color=COLOR_FULL,
                    arrowprops=dict(arrowstyle='->', color=COLOR_FULL, lw=1))
        ax.annotate(f"r={last['r']}: {last['exp_bcgp']:.4f}",
                    xy=(last['inv_r'], last['exp_bcgp']),
                    xytext=(last['inv_r'] + 0.02, last['exp_bcgp'] - 0.2),
                    fontsize=10, color=COLOR_BCGP,
                    arrowprops=dict(arrowstyle='->', color=COLOR_BCGP, lw=1))
    
    ax.set_xlabel(r'$1/r$', fontsize=14)
    ax.set_ylabel(r'Finite-difference exponent', fontsize=14)
    ax.set_title(r'Finite-Difference Exponent Convergence', fontsize=15)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='-')
    ax.invert_xaxis()  # So that r increases to the right
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")
    return output_path

# ============================================================================
# PLOT 3: Radical Shift Convergence
# ============================================================================

def plot3_radical_shift(data, output_path='/home/z/my-project/download/plot3_radical_shift.png'):
    """Plot ln(CF) vs ln(r) with reference slope +1/2."""
    print("\nGenerating Plot 3: Radical shift convergence...")
    
    r_min_fit = 51  # Minimum r for asymptotic fit
    r_vals_all = sorted(data.keys())  # All r values (exact computation)
    r_vals_fit = sorted([r for r in data.keys() if r >= r_min_fit])  # For fit
    
    # Data for plotting (all r)
    ln_r_plot = []
    ln_CF_plot = []
    r_plot = []
    for r in r_vals_all:
        zfn = data[r]['Z_full_norm']
        zbn = data[r]['Z_bcgp_norm']
        if zfn > 0 and zbn > 0:
            ln_r_plot.append(np.log(r))
            ln_CF_plot.append(np.log(zfn / zbn))
            r_plot.append(r)
    ln_r_plot = np.array(ln_r_plot)
    ln_CF_plot = np.array(ln_CF_plot)
    r_plot_arr = np.array(r_plot, dtype=float)
    
    # Data for fit (r >= r_min_fit)
    ln_r = []
    ln_CF = []
    r_list = []
    for r in r_vals_fit:
        zfn = data[r]['Z_full_norm']
        zbn = data[r]['Z_bcgp_norm']
        if zfn > 0 and zbn > 0:
            ln_r.append(np.log(r))
            ln_CF.append(np.log(zfn / zbn))
            r_list.append(r)
    ln_r = np.array(ln_r)
    ln_CF = np.array(ln_CF)
    r_arr = np.array(r_list, dtype=float)
    
    # Fit ln(CF) = a * ln(r) + b + c/r  (only for r >= r_min_fit)
    A = np.column_stack([ln_r, np.ones_like(ln_r), 1.0 / r_arr])
    coeffs, _, _, _ = np.linalg.lstsq(A, ln_CF, rcond=None)
    shift_exp = coeffs[0]
    shift_int = coeffs[1]
    shift_1r = coeffs[2]
    
    # Compute per-r shift from finite difference
    fd_shift = []
    fd_inv_r = []
    for r in r_vals_all:
        rp = r + 2
        rm = r - 2
        if rm < 3 or rp not in data or rm not in data:
            continue
        dp = data[rp]
        dm = data[rm]
        if (dp['Z_full_norm'] > 0 and dp['Z_bcgp_norm'] > 0 and
                dm['Z_full_norm'] > 0 and dm['Z_bcgp_norm'] > 0):
            CF_p = dp['Z_full_norm'] / dp['Z_bcgp_norm']
            CF_m = dm['Z_full_norm'] / dm['Z_bcgp_norm']
            dlnr = np.log(rp) - np.log(rm)
            dlnCF = np.log(CF_p) - np.log(CF_m)
            fd_shift.append(dlnCF / dlnr)
            fd_inv_r.append(1.0 / r)
    
    fd_shift = np.array(fd_shift)
    fd_inv_r = np.array(fd_inv_r)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.set_dpi(150)
    
    # Left panel: ln(CF) vs ln(r)
    ax1.plot(ln_r_plot, ln_CF_plot, '-', color=COLOR_SHIFT, linewidth=2.0,
             label=r'$\ln(\mathrm{CF}) = \ln(Z_{\mathrm{full}}/Z_{\mathrm{BCGP}})$', zorder=3)
    
    # Markers
    marker_idx = list(range(0, len(r_plot), 10))
    ax1.plot(ln_r_plot[marker_idx], ln_CF_plot[marker_idx], 'o', color=COLOR_MARKER_SHIFT, markersize=5, zorder=4)
    
    # Reference line with slope +1/2
    r_ref = 101
    ln_r_ref = np.log(r_ref)
    CF_ref = data[r_ref]['Z_full_norm'] / data[r_ref]['Z_bcgp_norm']
    ln_CF_ref = np.log(CF_ref)
    ref_line = ln_CF_ref + 0.5 * (ln_r_plot - ln_r_ref)
    ax1.plot(ln_r_plot, ref_line, '--', color=COLOR_REF, linewidth=1.5, alpha=0.7,
             label=r'Reference slope $+1/2$', zorder=2)
    
    # Fit line (only over fit range, but extended for visualization)
    fit_line = coeffs[0] * ln_r_plot + coeffs[1] + coeffs[2] / r_plot_arr
    ax1.plot(ln_r_plot, fit_line, ':', color='orange', linewidth=1.5, alpha=0.8,
             label=fr'3-param fit (r$\geq${r_min_fit}): slope = {shift_exp:.4f}', zorder=2)
    
    ax1.set_xlabel(r'$\ln r$', fontsize=14)
    ax1.set_ylabel(r'$\ln(\mathrm{CF})$', fontsize=14)
    ax1.set_title(r'Radical Shift: $\mathrm{CF} \sim r^{+1/2}$', fontsize=14)
    ax1.legend(loc='best', fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='-')
    
    # Annotate
    ax1.annotate(fr'Fit slope = {shift_exp:.4f}' + '\n' + fr'Target = +1/2' + '\n' +
                 fr'Deviation = {abs(shift_exp - 0.5):.4f}',
                 xy=(0.02, 0.02), xycoords='axes fraction', fontsize=11,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='gray', alpha=0.9),
                 va='bottom', ha='left')
    
    # Right panel: FD shift vs 1/r
    ax2.plot(fd_inv_r, fd_shift, '-', color=COLOR_SHIFT, linewidth=2.0,
             label=r'$d\ln(\mathrm{CF}) / d\ln r$ (finite diff.)', zorder=3)
    marker_idx2 = list(range(0, len(fd_shift), 10))
    ax2.plot(fd_inv_r[marker_idx2], fd_shift[marker_idx2], 'o',
             color=COLOR_MARKER_SHIFT, markersize=5, zorder=4)
    
    ax2.axhline(y=0.5, color=COLOR_REF, linestyle='--', linewidth=1.5, alpha=0.7,
                label=r'Target: $+1/2$')
    
    ax2.set_xlabel(r'$1/r$', fontsize=14)
    ax2.set_ylabel(r'Shift exponent (FD)', fontsize=14)
    ax2.set_title(r'Per-$r$ Shift from Finite Difference', fontsize=14)
    ax2.legend(loc='best', fontsize=11, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='-')
    ax2.invert_xaxis()
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")
    print(f"  Shift exponent from fit: {shift_exp:.6f} (target: 0.5)")
    return output_path, shift_exp

# ============================================================================
# PLOT 4: D₂/r³ Convergence to 1/6
# ============================================================================

def plot4_D2_convergence(output_path='/home/z/my-project/download/plot4_D2_convergence.png'):
    """Plot D₂/r³ vs 1/r showing convergence to 1/6."""
    print("\nGenerating Plot 4: D₂/r³ convergence...")
    
    r_vals = list(range(3, 502, 2))
    inv_r = np.array([1.0/r for r in r_vals])
    D2_over_r3 = np.array([D_ell(r) / r**3 for r in r_vals])
    error = np.abs(D2_over_r3 - 1.0/6.0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.set_dpi(150)
    
    # Left: D₂/r³ vs 1/r
    ax1.plot(inv_r, D2_over_r3, '-', color=COLOR_D2, linewidth=2.0,
             label=r'$D_2(r)/r^3$', zorder=3)
    marker_idx = list(range(0, len(r_vals), 25))
    ax1.plot(inv_r[marker_idx], D2_over_r3[marker_idx], 'o',
             color=COLOR_MARKER_D2, markersize=5, zorder=4)
    ax1.axhline(y=1.0/6.0, color=COLOR_REF, linestyle='--', linewidth=1.5, alpha=0.7,
                label=r'Target: $1/6 \approx 0.1\overline{6}$')
    
    ax1.set_xlabel(r'$1/r$', fontsize=14)
    ax1.set_ylabel(r'$D_2(r)/r^3$', fontsize=14)
    ax1.set_title(r'Coproduct Deficiency: $D_2/r^3 \to 1/6$', fontsize=14)
    ax1.legend(loc='best', fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='-')
    ax1.invert_xaxis()
    
    # Right: |error| vs 1/r (log scale)
    ax2.semilogy(inv_r, error, '-', color=COLOR_D2, linewidth=2.0,
                 label=r'$|D_2/r^3 - 1/6|$', zorder=3)
    ax2.plot(inv_r[marker_idx], error[marker_idx], 'o',
             color=COLOR_MARKER_D2, markersize=5, zorder=4)
    
    # Reference line: error ~ 1/r^2 (since D_2/r^3 = (1 - 1/r^2)/6)
    ref_error = 1.0 / (6.0 * np.array(r_vals, dtype=float)**2)
    ax2.semilogy(inv_r, ref_error, '--', color=COLOR_REF, linewidth=1.5, alpha=0.7,
                 label=r'$\sim r^{-2}$ reference')
    
    ax2.set_xlabel(r'$1/r$', fontsize=14)
    ax2.set_ylabel(r'$|D_2/r^3 - 1/6|$', fontsize=14)
    ax2.set_title(r'Convergence Error (log scale)', fontsize=14)
    ax2.legend(loc='best', fontsize=11, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='-')
    ax2.invert_xaxis()
    
    # Annotate key values
    for r in [3, 11, 51, 201, 501]:
        val = D_ell(r) / r**3
        ax1.annotate(f'r={r}: {val:.6f}',
                     xy=(1.0/r, val), fontsize=8, color=COLOR_D2,
                     xytext=(1.0/r + 0.01, val + 0.002))
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")
    return output_path

# ============================================================================
# PLOT 5: Summary Dashboard
# ============================================================================

def plot5_dashboard(data, shift_exp, output_path='/home/z/my-project/download/plot5_master_identity_dashboard.png'):
    """Multi-panel dashboard summarizing all convergence results."""
    print("\nGenerating Plot 5: Master identity dashboard...")
    
    fig = plt.figure(figsize=(18, 11))
    fig.set_dpi(150)
    
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
    
    # ---- Panel (0,0): Power-law convergence ----
    ax00 = fig.add_subplot(gs[0, 0])
    r_vals = sorted(data.keys())
    ln_r = np.array([np.log(r) for r in r_vals])
    ln_Zf = np.array([data[r]['ln_Z_full'] for r in r_vals])
    ln_Zb = np.array([data[r]['ln_Z_bcgp'] for r in r_vals])
    
    ax00.plot(ln_r, ln_Zf, '-', color=COLOR_FULL, linewidth=1.8,
              label=r'$Z_{\mathrm{full}}^{\mathrm{norm}}$')
    ax00.plot(ln_r, ln_Zb, '-', color=COLOR_BCGP, linewidth=1.8,
              label=r'$Z_{\mathrm{BCGP}}^{\mathrm{norm}}$')
    
    # Reference slopes
    r_ref = 101
    if r_ref in data:
        ln_r_ref = np.log(r_ref)
        ref_r = np.linspace(np.log(3), np.log(501), 50)
        ax00.plot(ref_r, data[r_ref]['ln_Z_full'] + (-1.5)*(ref_r - ln_r_ref),
                  '--', color=COLOR_FULL, linewidth=1.0, alpha=0.5, label=r'slope $-3/2$')
        ax00.plot(ref_r, data[r_ref]['ln_Z_bcgp'] + (-2.0)*(ref_r - ln_r_ref),
                  '--', color=COLOR_BCGP, linewidth=1.0, alpha=0.5, label=r'slope $-2$')
    
    ax00.set_xlabel(r'$\ln r$', fontsize=11)
    ax00.set_ylabel(r'$\ln Z^{\mathrm{norm}}$', fontsize=11)
    ax00.set_title('(a) Power-Law Convergence', fontsize=12, fontweight='bold')
    ax00.legend(loc='best', fontsize=9, framealpha=0.9)
    ax00.grid(True, alpha=0.3)
    
    # ---- Panel (0,1): FD convergence ----
    ax01 = fig.add_subplot(gs[0, 1])
    fd_data = []
    for r in r_vals:
        rp, rm = r + 2, r - 2
        if rm < 3 or rp not in data or rm not in data:
            continue
        dp, dm = data[rp], data[rm]
        if (np.isfinite(dp['ln_Z_full']) and np.isfinite(dm['ln_Z_full']) and
                np.isfinite(dp['ln_Z_bcgp']) and np.isfinite(dm['ln_Z_bcgp'])):
            dlnr = np.log(rp) - np.log(rm)
            fd_data.append({
                'inv_r': 1.0/r,
                'exp_full': (dp['ln_Z_full'] - dm['ln_Z_full']) / dlnr,
                'exp_bcgp': (dp['ln_Z_bcgp'] - dm['ln_Z_bcgp']) / dlnr,
            })
    
    inv_r_fd = np.array([d['inv_r'] for d in fd_data])
    exp_full_fd = np.array([d['exp_full'] for d in fd_data])
    exp_bcgp_fd = np.array([d['exp_bcgp'] for d in fd_data])
    
    ax01.plot(inv_r_fd, exp_full_fd, '-', color=COLOR_FULL, linewidth=1.8,
              label=r'$Z_{\mathrm{full}}$')
    ax01.plot(inv_r_fd, exp_bcgp_fd, '-', color=COLOR_BCGP, linewidth=1.8,
              label=r'$Z_{\mathrm{BCGP}}$')
    ax01.axhline(y=-1.5, color=COLOR_FULL, linestyle='--', linewidth=1.2, alpha=0.5)
    ax01.axhline(y=-2.0, color=COLOR_BCGP, linestyle='--', linewidth=1.2, alpha=0.5)
    
    ax01.set_xlabel(r'$1/r$', fontsize=11)
    ax01.set_ylabel('FD exponent', fontsize=11)
    ax01.set_title('(b) Finite-Difference Convergence', fontsize=12, fontweight='bold')
    ax01.legend(loc='best', fontsize=9, framealpha=0.9)
    ax01.grid(True, alpha=0.3)
    ax01.invert_xaxis()
    
    # ---- Panel (0,2): Radical shift ----
    ax02 = fig.add_subplot(gs[0, 2])
    r_exact = sorted(data.keys())  # All r values (exact computation)
    ln_r_cf = []
    ln_CF = []
    for r in r_exact:
        zfn = data[r]['Z_full_norm']
        zbn = data[r]['Z_bcgp_norm']
        if zfn > 0 and zbn > 0:
            ln_r_cf.append(np.log(r))
            ln_CF.append(np.log(zfn / zbn))
    
    ln_r_cf = np.array(ln_r_cf)
    ln_CF = np.array(ln_CF)
    
    ax02.plot(ln_r_cf, ln_CF, '-', color=COLOR_SHIFT, linewidth=1.8,
              label=r'$\ln(Z_{\mathrm{full}}/Z_{\mathrm{BCGP}})$')
    
    # Reference line slope +1/2
    if len(ln_r_cf) > 0:
        mid = len(ln_r_cf) // 2
        ref_line = ln_CF[mid] + 0.5 * (ln_r_cf - ln_r_cf[mid])
        ax02.plot(ln_r_cf, ref_line, '--', color=COLOR_REF, linewidth=1.2, alpha=0.7,
                  label=r'slope $+1/2$')
    
    ax02.set_xlabel(r'$\ln r$', fontsize=11)
    ax02.set_ylabel(r'$\ln(\mathrm{CF})$', fontsize=11)
    ax02.set_title(r'(c) Radical Shift: CF $\sim r^{+1/2}$', fontsize=12, fontweight='bold')
    ax02.legend(loc='best', fontsize=9, framealpha=0.9)
    ax02.grid(True, alpha=0.3)
    
    # ---- Panel (1,0): D₂/r³ ----
    ax10 = fig.add_subplot(gs[1, 0])
    r_d2 = list(range(3, 502, 2))
    inv_r_d2 = np.array([1.0/r for r in r_d2])
    D2_r3 = np.array([D_ell(r) / r**3 for r in r_d2])
    
    ax10.plot(inv_r_d2, D2_r3, '-', color=COLOR_D2, linewidth=1.8,
              label=r'$D_2(r)/r^3$')
    ax10.axhline(y=1.0/6.0, color=COLOR_REF, linestyle='--', linewidth=1.2, alpha=0.7,
                 label=r'$1/6$')
    
    ax10.set_xlabel(r'$1/r$', fontsize=11)
    ax10.set_ylabel(r'$D_2/r^3$', fontsize=11)
    ax10.set_title(r'(d) $D_2/r^3 \to 1/6$', fontsize=12, fontweight='bold')
    ax10.legend(loc='best', fontsize=9, framealpha=0.9)
    ax10.grid(True, alpha=0.3)
    ax10.invert_xaxis()
    
    # ---- Panel (1,1): Verification table ----
    ax11 = fig.add_subplot(gs[1, 1])
    ax11.axis('off')
    
    # Compute key numbers
    # Method 1: power-law fit
    r_fit = sorted([r for r in data.keys() if r >= 51])
    ln_r_fit = np.array([np.log(r) for r in r_fit])
    ln_Zf_fit = np.array([data[r]['ln_Z_full'] for r in r_fit])
    ln_Zb_fit = np.array([data[r]['ln_Z_bcgp'] for r in r_fit])
    r_fit_arr = np.array(r_fit, dtype=float)
    
    A_fit = np.column_stack([ln_r_fit, np.ones_like(ln_r_fit), 1.0/r_fit_arr])
    cf_full, _, _, _ = np.linalg.lstsq(A_fit, ln_Zf_fit, rcond=None)
    cf_bcgp, _, _, _ = np.linalg.lstsq(A_fit, ln_Zb_fit, rcond=None)
    
    m1_full = cf_full[0]
    m1_bcgp = cf_bcgp[0]
    m1_shift = m1_full - m1_bcgp
    
    # Method 2: FD average for large r
    fd_large = [d for d in fd_data if d['inv_r'] <= 1.0/301]
    m2_full = np.mean([d['exp_full'] for d in fd_large]) if fd_large else float('nan')
    m2_bcgp = np.mean([d['exp_bcgp'] for d in fd_large]) if fd_large else float('nan')
    m2_shift = m2_full - m2_bcgp
    
    # D2 at r=501
    d2_last = D_ell(501) / 501**3
    
    table_text = (
        "Verification Table\n\n"
        "Method 1: Power-Law Fit (3-param, r>=51)\n"
        f"  Full trace exponent:  {m1_full:+.4f}  (target: -3/2)\n"
        f"  BCGP trace exponent:  {m1_bcgp:+.4f}  (target: -2)\n"
        f"  Radical shift:        {m1_shift:+.4f}  (target: +1/2)\n\n"
        "Method 2: Finite-Difference (r>=301)\n"
        f"  Full trace exponent:  {m2_full:+.4f}  (target: -3/2)\n"
        f"  BCGP trace exponent:  {m2_bcgp:+.4f}  (target: -2)\n"
        f"  Radical shift:        {m2_shift:+.4f}  (target: +1/2)\n\n"
        "Correction Factor\n"
        f"  Shift exponent:       {shift_exp:+.4f}  (target: +1/2)\n\n"
        "Coproduct Deficiency\n"
        f"  D2(501)/501^3 = {d2_last:.8f}\n"
        f"  Target 1/6    = {1/6:.8f}\n"
        f"  Error = {abs(d2_last - 1/6):.2e}"
    )
    
    ax11.text(0.05, 0.95, table_text, transform=ax11.transAxes,
              fontsize=10, verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='gray', alpha=0.9))
    ax11.set_title('(e) Verification Results', fontsize=12, fontweight='bold')
    
    # ---- Panel (1,2): Master identity equation ----
    ax12 = fig.add_subplot(gs[1, 2])
    ax12.axis('off')
    
    identity_text = (
        "Master Identity\n\n"
        r"$-\frac{3}{2} = -2 + \frac{1}{2}$" + "\n\n"
        "  -3/2  (full trace)\n"
        "   =  -2  (modified trace)\n"
        "   + +1/2 (radical)\n\n"
        "Physical Interpretation\n\n"
        r"$Z_{full}^{norm} \sim r^{-3/2}$" + "  (gravity: 3 zero modes)\n\n"
        r"$Z_{BCGP}^{norm} \sim r^{-2}$" + "  (categorical trace)\n\n"
        r"$CF = Z_{full}/Z_{BCGP} \sim r^{+1/2}$" + "  (radical = BH interior)\n\n"
        r"$D_2 / r^3 \to 1/6$" + "  (coproduct deficiency)"
    )
    
    ax12.text(0.05, 0.95, identity_text, transform=ax12.transAxes,
              fontsize=11, verticalalignment='top',
              bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0ff', edgecolor='gray', alpha=0.9))
    ax12.set_title('(f) Master Identity', fontsize=12, fontweight='bold')
    
    fig.suptitle(r"Master Identity: $-3/2 = -2 + 1/2$ — Definitive Verification",
                 fontsize=16, fontweight='bold', y=0.98)
    
    fig.subplots_adjust(top=0.93, bottom=0.06, left=0.05, right=0.97, hspace=0.35, wspace=0.35)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")
    
    return output_path, m1_full, m1_bcgp, m1_shift, m2_full, m2_bcgp, m2_shift

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    t0 = time.time()
    beta = 1.0
    
    # Precompute all data
    data = precompute_data(r_max=501, beta=beta)
    
    # Generate plots
    p1 = plot1_power_law(data, beta=beta)
    p2 = plot2_fd_convergence(data)
    p3, shift_exp = plot3_radical_shift(data)
    p4 = plot4_D2_convergence()
    p5, m1_full, m1_bcgp, m1_shift, m2_full, m2_bcgp, m2_shift = plot5_dashboard(data, shift_exp)
    
    elapsed = time.time() - t0
    
    print("\n" + "=" * 80)
    print("CONVERGENCE PLOT GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nGenerated plots:")
    print(f"  1. {p1}")
    print(f"  2. {p2}")
    print(f"  3. {p3}")
    print(f"  4. {p4}")
    print(f"  5. {p5}")
    print(f"\nKey results:")
    print(f"  Method 1 (power-law fit):  full={m1_full:+.4f}, BCGP={m1_bcgp:+.4f}, shift={m1_shift:+.4f}")
    print(f"  Method 2 (finite-diff):    full={m2_full:+.4f}, BCGP={m2_bcgp:+.4f}, shift={m2_shift:+.4f}")
    print(f"  Correction factor exponent: {shift_exp:+.4f} (target: +0.5)")
    print(f"\nTotal time: {elapsed:.1f}s")
