"""Generate the three-fold decomposition figure for sl3."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(10, 4))

# Color scheme - Oxford blue palette
colors = {
    'cartan': ('#002147', '#E8EFF7'),  # Oxford blue
    'pos_lth': ('#7B2D26', '#FAEAE8'), # Dark red
    'neg_lth': ('#1E6F5C', '#E8F5F0'), # Dark green
}

# Panel 1: Cartan-type classes
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title(r'Cartan-type' + '\n' + r'$\binom{n+1}{2} = 3$', fontsize=12, fontweight='bold',
             color=colors['cartan'][0])

classes_cartan = [
    (r'$\xi_{K_1 K_2}$', 'Type I: pairwise\n$K_1^3=K_2^3=1$'),
    (r'$\xi_{K_1}^{\mathrm{cross}}$', 'Type II: $K_1 E_j$\ncross relation'),
    (r'$\xi_{K_2}^{\mathrm{cross}}$', 'Type II: $K_2 E_j$\ncross relation'),
]
for i, (sym, desc) in enumerate(classes_cartan):
    y = 7 - i * 2.8
    rect = mpatches.FancyBboxPatch((0.5, y-0.8), 9, 2.2,
                                    boxstyle="round,pad=0.2",
                                    facecolor=colors['cartan'][1], edgecolor=colors['cartan'][0],
                                    linewidth=1.5)
    ax.add_patch(rect)
    ax.text(2.5, y+0.2, sym, fontsize=12, fontweight='bold', va='center', color=colors['cartan'][0])
    ax.text(5.5, y+0.0, desc, fontsize=8, va='center', color='#4A4A4A')

# Panel 2: Positive l-th power
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title(r'Positive $\ell$-th power' + '\n' + r'$|\Phi^+| = 3$', fontsize=12, fontweight='bold',
             color=colors['pos_lth'][0])

classes_pos = [
    (r'$[E_1^3]$', r'Simple root $\alpha_1$'),
    (r'$[E_2^3]$', r'Simple root $\alpha_2$'),
    (r'$[E_{12}^3]$', r'Composite root $\alpha_1+\alpha_2$'),
]
for i, (sym, desc) in enumerate(classes_pos):
    y = 7 - i * 2.8
    rect = mpatches.FancyBboxPatch((0.5, y-0.8), 9, 2.2,
                                    boxstyle="round,pad=0.2",
                                    facecolor=colors['pos_lth'][1], edgecolor=colors['pos_lth'][0],
                                    linewidth=1.5)
    ax.add_patch(rect)
    ax.text(2.5, y+0.2, sym, fontsize=12, fontweight='bold', va='center', color=colors['pos_lth'][0])
    ax.text(5.5, y+0.0, desc, fontsize=8, va='center', color='#4A4A4A')

# Panel 3: Negative l-th power
ax = axes[2]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title(r'Negative $\ell$-th power' + '\n' + r'$|\Phi^+| = 3$', fontsize=12, fontweight='bold',
             color=colors['neg_lth'][0])

classes_neg = [
    (r'$[F_1^3]$', r'Simple root $\alpha_1$'),
    (r'$[F_2^3]$', r'Simple root $\alpha_2$'),
    (r'$[F_{12}^3]$', r'Composite root $\alpha_1+\alpha_2$'),
]
for i, (sym, desc) in enumerate(classes_neg):
    y = 7 - i * 2.8
    rect = mpatches.FancyBboxPatch((0.5, y-0.8), 9, 2.2,
                                    boxstyle="round,pad=0.2",
                                    facecolor=colors['neg_lth'][1], edgecolor=colors['neg_lth'][0],
                                    linewidth=1.5)
    ax.add_patch(rect)
    ax.text(2.5, y+0.2, sym, fontsize=12, fontweight='bold', va='center', color=colors['neg_lth'][0])
    ax.text(5.5, y+0.0, desc, fontsize=8, va='center', color='#4A4A4A')

fig.suptitle(r'Decomposition of $\mathrm{HH}^2(u_q(\mathfrak{sl}_3), \mathbb{C})$: $\dim = 3 + 3 + 3 = 9$',
             fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('/home/z/my-project/download/hopf-paper/figures/decomposition.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/home/z/my-project/download/hopf-paper/figures/decomposition.png', dpi=300, bbox_inches='tight')
print("Decomposition figure saved")
