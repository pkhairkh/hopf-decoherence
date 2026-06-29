"""Generate spectral sequence E2 page diagram for the paper."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(6, 5))

# Grid
ax.set_xlim(-0.5, 4.5)
ax.set_ylim(-0.5, 3.5)
ax.set_aspect('equal')

# Draw grid lines
for i in range(6):
    ax.axhline(y=i-0.5, color='gray', linewidth=0.3, alpha=0.5)
    ax.axvline(x=i-0.5, color='gray', linewidth=0.3, alpha=0.5)

# Fill the E2 page cells
# E2^{2,0} = C^{binom(n+1,2) + |Phi+|}  (B+ cohomology)
# E2^{1,1} = 0
# E2^{0,2} = C^{|Phi+|}  (B-invariant B- cohomology)

cells = {
    (2, 0): {'text': r'$E_2^{2,0}$' + '\n' + r'$\mathrm{HH}^2(B^+,\mathbb{C})$' + '\n' + r'$\mathbb{C}^{\binom{n+1}{2}+|\Phi^+|}$',
              'color': '#D4E6F1', 'border': '#2980B9'},
    (1, 1): {'text': r'$E_2^{1,1}$' + '\n' + r'$= 0$', 
              'color': '#FADBD8', 'border': '#E74C3C'},
    (0, 2): {'text': r'$E_2^{0,2}$' + '\n' + r'$\mathrm{HH}^0(B^+, \mathrm{HH}^2(B^-))$' + '\n' + r'$\mathbb{C}^{|\Phi^+|}$',
              'color': '#D5F5E3', 'border': '#27AE60'},
}

# Also show the zero regions
for p in range(5):
    for q in range(4):
        if (p,q) not in cells and p+q <= 4:
            rect = mpatches.FancyBboxPatch((p-0.4, q-0.4), 0.8, 0.8, 
                                            boxstyle="round,pad=0.05",
                                            facecolor='#F8F9F9', edgecolor='#BDC3C7',
                                            linewidth=0.5)
            ax.add_patch(rect)

for (p, q), info in cells.items():
    rect = mpatches.FancyBboxPatch((p-0.4, q-0.4), 0.8, 0.8,
                                    boxstyle="round,pad=0.05",
                                    facecolor=info['color'], edgecolor=info['border'],
                                    linewidth=2.0)
    ax.add_patch(rect)
    ax.text(p, q, info['text'], ha='center', va='center', fontsize=7,
            fontweight='bold', color='#2C3E50')

# Draw d2 differential arrows
# d2: E2^{0,2} -> E2^{2,1} (vanishes)
ax.annotate('', xy=(2.3, 1.3), xytext=(0.3, 2.3),
            arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=1.5, ls='--'))
ax.text(1.1, 2.0, r'$d_2 = 0$', fontsize=8, color='#E74C3C', style='italic', rotation=-30)

# d2: E2^{2,0} -> E2^{4,-1} (vanishes by first quadrant)
ax.annotate('', xy=(4.0, -0.3), xytext=(2.3, 0.3),
            arrowprops=dict(arrowstyle='->', color='#95A5A6', lw=1.0, ls=':'))
ax.text(3.3, -0.1, r'$d_2 = 0$', fontsize=7, color='#95A5A6', style='italic')

# Labels
ax.set_xlabel(r'$p$ (filtration degree)', fontsize=11, fontweight='bold')
ax.set_ylabel(r'$q$ (complementary degree)', fontsize=11, fontweight='bold')
ax.set_title(r'Drinfeld Double Spectral Sequence: $E_2^{p,q}$ Page', fontsize=12, fontweight='bold')

# Tick labels
ax.set_xticks(range(5))
ax.set_xticklabels(['0', '1', '2', '3', '4'])
ax.set_yticks(range(4))
ax.set_yticklabels(['0', '1', '2', '3'])

# Highlight total degree 2
ax.text(4.2, 2.5, r'Total degree $p+q=2$', fontsize=9, color='#8E44AD',
        bbox=dict(boxstyle='round', facecolor='#F5EEF8', edgecolor='#8E44AD', alpha=0.8))

plt.tight_layout()
plt.savefig('/home/z/my-project/download/hopf-paper/figures/spectral_sequence.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/home/z/my-project/download/hopf-paper/figures/spectral_sequence.png', dpi=300, bbox_inches='tight')
print("Spectral sequence diagram saved")
