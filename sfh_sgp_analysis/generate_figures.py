"""
SGP-Tribe3: SFH-SGP Topological Field Analysis (FINAL)
=======================================================
Generates all SFH-SGP visualizations and results.
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

OUTPUT_DIR = "results/full_battery_1000/sfh_sgp"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/figures", exist_ok=True)

NODE_ORDER = ['G1_broca', 'G2_wernicke', 'G3_tpj', 'G4_pfc', 'G5_dmn',
              'G6_limbic', 'G7_sensory', 'G8_atl', 'G9_premotor']

NODE_SHORT = ['Broca', 'Wern', 'TPJ', 'PFC', 'DMN', 'Limb', 'Sens', 'ATL', 'Prem']

NODE_NAMES = {
    'G1_broca': 'Broca (Dorsal)',
    'G2_wernicke': 'Wernicke (Ventral)',
    'G3_tpj': 'TPJ (Convergence)',
    'G4_pfc': 'PFC (Executive)',
    'G5_dmn': 'DMN (Fertility)',
    'G6_limbic': 'Limbic (Torsion)',
    'G7_sensory': 'Sensory (Input)',
    'G8_atl': 'ATL (Semantic)',
    'G9_premotor': 'Premotor (Output)'
}

ALPHA, BETA = 1.0, 1.0
K_MAX = 50
CONVERGENCE_THRESHOLD = 0.001

print("=" * 70)
print("SGP-Tribe3: SFH-SGP Topological Field Analysis (FINAL)")
print("=" * 70)

# ─── Load Data ────────────────────────────────────────────────────────────────

results_files = sorted(glob.glob("results/full_battery_1000/checkpoint_*.json"))
all_results = []
for f in results_files:
    with open(f) as fp:
        data = json.load(fp)
    all_results.extend(data.get('results', []))

seen = set()
results = []
for r in all_results:
    sid = r.get('stimulus_id')
    if sid and sid not in seen:
        seen.add(sid)
        results.append(r)

df = pd.DataFrame(results)
for node in NODE_ORDER:
    df[node] = df['sgp_nodes'].apply(lambda x: x.get(node, 0) if isinstance(x, dict) else 0)

with open("results/full_battery_1000/statistical_analysis.json") as f:
    stats_data = json.load(f)

category_differentials = stats_data['category_differentials']
categories = sorted(category_differentials.keys())

# ─── Compute SFH-SGP Primitives ─────────────────────────────────────────────

delta_matrix = np.array([[category_differentials[cat][node] for node in NODE_ORDER] 
                         for cat in categories])

Q_per_category = np.sum(np.abs(delta_matrix), axis=1)
F_per_category = np.array([category_differentials[cat]['G5_dmn'] for cat in categories])

delta_coactivation = np.corrcoef(delta_matrix.T)
eigenvalues_coact, eigenvectors_coact = np.linalg.eigh(delta_coactivation)
λ_leading = eigenvalues_coact[-1]
leading_eigenvector = eigenvectors_coact[:, -1]

C_per_category = np.array([np.dot(delta_matrix[i], leading_eigenvector) for i in range(len(categories))])
C_min, C_max = C_per_category.min(), C_per_category.max()
if C_max > C_min:
    C_per_category = (C_per_category - C_min) / (C_max - C_min)
else:
    C_per_category = np.zeros_like(C_per_category)

χ_per_category = ALPHA * C_per_category + BETA * np.abs(F_per_category)

# ─── Figure 1: χ Topographic Ranking ────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Panel A: χ ranking
ax1 = axes[0]
colors = plt.cm.RdYlBu(np.linspace(0, 1, len(categories)))
chi_ranked = sorted(zip(categories, χ_per_category), key=lambda x: x[1], reverse=True)
cats_sorted = [x[0] for x in chi_ranked]
chi_sorted = [x[1] for x in chi_ranked]

bars = ax1.barh(range(len(cats_sorted)), chi_sorted, color=colors[::-1])
ax1.set_yticks(range(len(cats_sorted)))
ax1.set_yticklabels([c.capitalize() for c in cats_sorted])
ax1.set_xlabel('χ (Sentient Potential)', fontsize=12)
ax1.set_title('A. χ Topographic Ranking', fontsize=14, fontweight='bold')
ax1.invert_yaxis()

for i, (cat, chi) in enumerate(chi_ranked[:3]):
    ax1.annotate(f'χ={chi:.3f}', xy=(chi, i), xytext=(chi+0.02, i), fontsize=9)

# Panel B: χ clusters
ax2 = axes[1]
chi_mean = np.mean(χ_per_category)
chi_std = np.std(χ_per_category)
colors_cluster = ['#2ecc71' if χ < chi_mean - chi_std else '#f39c12' if χ > chi_mean + chi_std else '#3498db' 
                  for χ in χ_per_category]

bars = ax2.bar(range(len(categories)), χ_per_category, color=colors_cluster)
ax2.axhline(y=chi_mean, color='black', linestyle='--', label=f'Mean = {chi_mean:.3f}')
ax2.axhline(y=chi_mean + chi_std, color='red', linestyle=':', alpha=0.5)
ax2.axhline(y=chi_mean - chi_std, color='red', linestyle=':', alpha=0.5)
ax2.set_xticks(range(len(categories)))
ax2.set_xticklabels([c[:4].upper() for c in categories], rotation=45)
ax2.set_ylabel('χ (Sentient Potential)', fontsize=12)
ax2.set_title('B. χ Clusters (High/Mid/Low)', fontsize=14, fontweight='bold')
ax2.legend()

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/fig1_chi_topography.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig1_chi_topography.png")

# ─── Figure 2: Differential Activation Heatmap ─────────────────────────────────

fig, ax = plt.subplots(figsize=(14, 10))

im = ax.imshow(delta_matrix, cmap='RdBu_r', aspect='auto', vmin=-0.025, vmax=0.025)
ax.set_xticks(range(len(NODE_ORDER)))
ax.set_xticklabels(NODE_SHORT, rotation=45, ha='right')
ax.set_yticks(range(len(categories)))
ax.set_yticklabels([c.capitalize() for c in categories])
ax.set_xlabel('SGP Node', fontsize=12)
ax.set_ylabel('Semantic Category', fontsize=12)
ax.set_title('Differential Activation (δ) Matrix\n(Red = Above Mean, Blue = Below Mean)', 
             fontsize=14, fontweight='bold')

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('δ (Deviation from Mean)', fontsize=11)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/fig2_delta_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig2_delta_heatmap.png")

# ─── Figure 3: Category Clustering Dendrogram ─────────────────────────────────

distances = pdist(delta_matrix, metric='euclidean')
linkage_matrix = linkage(distances, method='ward')
cluster_labels = fcluster(linkage_matrix, t=2, criterion='maxclust')

fig, ax = plt.subplots(figsize=(12, 6))
from scipy.cluster.hierarchy import dendrogram
dendrogram(linkage_matrix, labels=[c[:6].upper() for c in categories], ax=ax)
ax.set_title('Hierarchical Clustering of Semantic Categories\n(Based on Differential Activation Similarity)', 
             fontsize=14, fontweight='bold')
ax.set_ylabel('Distance (Ward)', fontsize=12)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/fig3_category_clustering.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig3_category_clustering.png")

# ─── Figure 4: Q-F-C Space ───────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 5))

# Panel A: Q (Total Flux)
ax1 = fig.add_subplot(131)
q_colors = plt.cm.viridis((Q_per_category - Q_per_category.min()) / (Q_per_category.max() - Q_per_category.min() + 1e-10))
ax1.bar(range(len(categories)), Q_per_category, color=q_colors)
ax1.set_xticks(range(len(categories)))
ax1.set_xticklabels([c[:4].upper() for c in categories], rotation=45)
ax1.set_ylabel('Q (Total Differential Flux)')
ax1.set_title('A. Q per Category')

# Panel B: C (Coherence)
ax2 = fig.add_subplot(132)
ax2.bar(range(len(categories)), C_per_category, color='steelblue')
ax2.set_xticks(range(len(categories)))
ax2.set_xticklabels([c[:4].upper() for c in categories], rotation=45)
ax2.set_ylabel('C (Coherence)')
ax2.set_title('B. C per Category')

# Panel C: χ (Sentient Potential)
ax3 = fig.add_subplot(133)
ax3.bar(range(len(categories)), χ_per_category, color='coral')
ax3.set_xticks(range(len(categories)))
ax3.set_xticklabels([c[:4].upper() for c in categories], rotation=45)
ax3.set_ylabel('χ (Sentient Potential)')
ax3.set_title('C. χ per Category')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/fig4_qfc_space.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig4_qfc_space.png")

# ─── Figure 5: Node Profiles per Category ─────────────────────────────────────

fig, axes = plt.subplots(2, 5, figsize=(20, 10))
axes = axes.flatten()

for i, cat in enumerate(categories):
    ax = axes[i]
    deltas = [category_differentials[cat][node] for node in NODE_ORDER]
    colors = ['#e74c3c' if d > 0 else '#3498db' for d in deltas]
    ax.bar(range(len(NODE_ORDER)), deltas, color=colors)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xticks(range(len(NODE_ORDER)))
    ax.set_xticklabels(NODE_SHORT, fontsize=7, rotation=45)
    ax.set_title(cat.capitalize(), fontweight='bold')
    ax.set_ylim(-0.025, 0.05)

for ax in axes[10:]:
    ax.axis('off')

fig.suptitle('Differential Activation (δ) Profiles by Category\n(Red = Above Mean, Blue = Below Mean)', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/fig5_node_profiles.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig5_node_profiles.png")

# ─── Figure 6: Coherence Structure ────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: Co-activation matrix
ax1 = axes[0]
im1 = ax1.imshow(delta_coactivation, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
ax1.set_xticks(range(len(NODE_ORDER)))
ax1.set_xticklabels(NODE_SHORT, rotation=45, ha='right')
ax1.set_yticks(range(len(NODE_ORDER)))
ax1.set_yticklabels(NODE_SHORT)
ax1.set_title('A. Differential Co-activation Matrix', fontweight='bold')
plt.colorbar(im1, ax=ax1, shrink=0.8)

# Panel B: Eigenvalue spectrum
ax2 = axes[1]
ev_sorted = sorted(enumerate(eigenvalues_coact), key=lambda x: x[1], reverse=True)
ev_indices = [x[0] for x in ev_sorted]
ev_values = [x[1] for x in ev_sorted]
bars = ax2.bar(range(len(ev_values)), ev_values, color='steelblue')
ax2.axhline(y=1.0, color='red', linestyle='--', label='λ=1 threshold')
ax2.set_xticks(range(len(NODE_ORDER)))
ax2.set_xticklabels([NODE_SHORT[i] for i in ev_indices], rotation=45)
ax2.set_ylabel('Eigenvalue', fontsize=12)
ax2.set_title('B. Eigenvalue Spectrum (Leading λ={:.2f})'.format(λ_leading), fontweight='bold')
ax2.legend()

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/fig6_coherence_structure.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig6_coherence_structure.png")

# ─── Figure 7: 2D χ Landscape ─────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 8))

# Compute 2D projection using PCA on delta_matrix
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
delta_2d = pca.fit_transform(delta_matrix)

scatter = ax.scatter(delta_2d[:, 0], delta_2d[:, 1], c=χ_per_category, 
                     cmap='plasma', s=200, edgecolors='black', linewidths=2)

for i, cat in enumerate(categories):
    ax.annotate(cat.capitalize(), (delta_2d[i, 0], delta_2d[i, 1]), 
                xytext=(5, 5), textcoords='offset points', fontsize=9)

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)', fontsize=12)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)', fontsize=12)
ax.set_title('χ Topographic Landscape (2D Projection)\nColor = χ Sentient Potential', 
              fontsize=14, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
cbar.set_label('χ', fontsize=11)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/fig7_chi_landscape_2d.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig7_chi_landscape_2d.png")

# ─── Save Results ─────────────────────────────────────────────────────────────

results = {
    'date': datetime.now().isoformat(),
    'categories': categories,
    'nodes': NODE_ORDER,
    'differential_matrix': [[float(x) for x in row] for row in delta_matrix],
    'Q_per_category': {cat: float(Q_per_category[i]) for i, cat in enumerate(categories)},
    'F_per_category': {cat: float(F_per_category[i]) for i, cat in enumerate(categories)},
    'C_per_category': {cat: float(C_per_category[i]) for i, cat in enumerate(categories)},
    'chi_per_category': {cat: float(χ_per_category[i]) for i, cat in enumerate(categories)},
    'chi_ranked': [[cat, float(chi)] for cat, chi in chi_ranked],
    'leading_eigenvalue': float(λ_leading),
    'cluster_labels': {cat: int(cluster_labels[i]) for i, cat in enumerate(categories)},
    'summary': {
        'Q_mean': float(np.mean(Q_per_category)),
        'chi_mean': float(np.mean(χ_per_category)),
        'chi_std': float(np.std(χ_per_category)),
        'chi_range': [float(χ_per_category.min()), float(χ_per_category.max())]
    }
}

with open(f'{OUTPUT_DIR}/sfh_sgp_final_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {OUTPUT_DIR}/sfh_sgp_final_results.json")

# Save numpy arrays
np.save(f'{OUTPUT_DIR}/delta_matrix.npy', delta_matrix)
np.save(f'{OUTPUT_DIR}/chi_per_category.npy', χ_per_category)
print(f"Saved numpy arrays to: {OUTPUT_DIR}/")

print("\n" + "=" * 70)
print("SFH-SGP ANALYSIS COMPLETE")
print("=" * 70)
print("\nKEY FINDINGS:")
print(f"  1. χ varies from {χ_per_category.min():.4f} to {χ_per_category.max():.4f}")
print(f"  2. Categories cluster into {len(set(cluster_labels))} basins")
print(f"  3. Leading eigenvalue λ = {λ_leading:.4f}")
print(f"  4. Generated 7 figures in {OUTPUT_DIR}/figures/")
print()
