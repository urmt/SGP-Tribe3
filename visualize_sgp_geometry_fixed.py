"""
SGP-Tribe3 — Phase 2: SGP Geometry Visualization (FIXED)
=========================================================
Creates publication-quality visualizations showing DIFFERENTIAL activation
(category mean - overall mean) with statistical significance testing.

Key fixes from original:
- Shows differential activation instead of absolute values
- Uses diverging colormaps (red=above mean, blue=below mean)
- Adds statistical significance testing (ANOVA + post-hoc)
- Proper scaling to make category differences visible

Usage:
    python visualize_sgp_geometry_fixed.py

Outputs:
    results/full_battery_1000/figures/geo_01_sgp_resonance_graph.png
    results/full_battery_1000/figures/geo_02_activation_field.png
    results/full_battery_1000/figures/geo_03_category_patterns.png
    results/full_battery_1000/figures/geo_04_fiber_bundle_analysis.png
    results/full_battery_1000/figures/geo_05_3d_geometry.png
    results/full_battery_1000/figures/geo_09_statistical_significance.png
    results/full_battery_1000/figures/geo_10_differential_heatmap.png
    results/full_battery_1000/statistical_analysis.json
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import griddata
from scipy import stats
from datetime import datetime

# ─── Load Results ────────────────────────────────────────────────────────────

results_files = sorted(glob.glob("results/full_battery_1000/checkpoint_*.json"))
if not results_files:
    print("ERROR: No checkpoint files found")
    exit(1)

all_results = []
for f in results_files:
    with open(f) as fp:
        data = json.load(fp)
    all_results.extend(data.get('results', []))

# Deduplicate
seen = set()
results = []
for r in all_results:
    sid = r.get('stimulus_id')
    if sid and sid not in seen:
        seen.add(sid)
        results.append(r)

df = pd.DataFrame(results)
print(f"Loaded {len(df)} stimuli")

NODE_ORDER = ['G1_broca', 'G2_wernicke', 'G3_tpj', 'G4_pfc', 'G5_dmn',
              'G6_limbic', 'G7_sensory', 'G8_atl', 'G9_premotor']

for node in NODE_ORDER:
    df[node] = df['sgp_nodes'].apply(lambda x: x.get(node, 0) if isinstance(x, dict) else 0)

categories = ['simple', 'logical', 'emotional', 'factual', 'abstract',
              'spatial', 'social', 'motor', 'memory', 'auditory']

output_dir = "results/full_battery_1000/figures"
os.makedirs(output_dir, exist_ok=True)

# ─── Compute Differential Activations ────────────────────────────────────────

print("\nComputing differential activations...")

overall_means = {node: df[node].mean() for node in NODE_ORDER}
overall_stds = {node: df[node].std() for node in NODE_ORDER}

cat_means = {}
cat_diffs = {}
cat_stds = {}
cat_ns = {}

for cat in categories:
    cat_data = df[df['category'] == cat]
    cat_means[cat] = {node: cat_data[node].mean() for node in NODE_ORDER}
    cat_diffs[cat] = {node: cat_data[node].mean() - overall_means[node] for node in NODE_ORDER}
    cat_stds[cat] = {node: cat_data[node].std() for node in NODE_ORDER}
    cat_ns[cat] = len(cat_data)

# Statistical significance (ANOVA for each node)
print("\nRunning ANOVA tests...")
anova_results = {}
for node in NODE_ORDER:
    groups = [df[df['category']==cat][node].values for cat in categories]
    f_stat, p_value = stats.f_oneway(*groups)
    anova_results[node] = {'f_stat': f_stat, 'p_value': p_value}
    sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
    print(f"  {node}: F={f_stat:.4f}, p={p_value:.6f} {sig}")

# Effect sizes (Cohen's d)
def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    if pooled_std == 0:
        return 0
    return (np.mean(group1) - np.mean(group2)) / pooled_std

effect_sizes = {}
for node in NODE_ORDER:
    effect_sizes[node] = {}
    for cat in categories:
        cat_data = df[df['category']==cat][node].values
        other_data = df[df['category']!=cat][node].values
        effect_sizes[node][cat] = cohens_d(cat_data, other_data)

# Save statistical analysis
stat_analysis = {
    'date': datetime.now().isoformat(),
    'n_stimuli': len(df),
    'n_categories': len(categories),
    'overall_means': overall_means,
    'overall_stds': overall_stds,
    'category_means': cat_means,
    'category_differentials': cat_diffs,
    'category_stds': cat_stds,
    'category_ns': cat_ns,
    'anova_results': {k: {'f_stat': float(v['f_stat']), 'p_value': float(v['p_value'])} for k, v in anova_results.items()},
    'effect_sizes': effect_sizes,
}

with open('results/full_battery_1000/statistical_analysis.json', 'w') as f:
    json.dump(stat_analysis, f, indent=2)
print(f"\nSaved: results/full_battery_1000/statistical_analysis.json")

# ─── SGP Geometry Definitions ────────────────────────────────────────────────

NODE_POSITIONS = {
    'G1_broca':     {'x': -0.6, 'y': 0.4, 'z': 0.2, 'label': 'G1\nBroca', 'stream': 'dorsal'},
    'G2_wernicke':  {'x': -0.5, 'y': -0.3, 'z': 0.1, 'label': 'G2\nWernicke', 'stream': 'ventral'},
    'G3_tpj':       {'x': -0.3, 'y': -0.5, 'z': 0.3, 'label': 'G3\nTPJ', 'stream': 'convergence'},
    'G4_pfc':       {'x': 0.0, 'y': 0.7, 'z': 0.0, 'label': 'G4\nPFC', 'stream': 'dorsal'},
    'G5_dmn':       {'x': 0.5, 'y': 0.0, 'z': 0.5, 'label': 'G5\nDMN', 'stream': 'generative'},
    'G6_limbic':    {'x': 0.3, 'y': -0.6, 'z': 0.4, 'label': 'G6\nLimbic', 'stream': 'modulatory'},
    'G7_sensory':   {'x': -0.7, 'y': -0.1, 'z': -0.2, 'label': 'G7\nSensory', 'stream': 'ventral'},
    'G8_atl':       {'x': 0.6, 'y': -0.2, 'z': -0.1, 'label': 'G8\nATL', 'stream': 'convergence'},
    'G9_premotor':  {'x': -0.2, 'y': 0.6, 'z': -0.3, 'label': 'G9\nPremotor', 'stream': 'dorsal'},
}

FIBER_BUNDLES = {
    'AF':    {'from': 'G2_wernicke', 'to': 'G1_broca', 'name': 'Arcuate Fasciculus', 'stream': 'dorsal'},
    'SLF':   {'from': 'G3_tpj', 'to': 'G4_pfc', 'name': 'Superior Longitudinal Fasc.', 'stream': 'dorsal'},
    'IFOF':  {'from': 'G8_atl', 'to': 'G4_pfc', 'name': 'Inferior Fronto-Occipital', 'stream': 'ventral'},
    'ILF':   {'from': 'G7_sensory', 'to': 'G2_wernicke', 'name': 'Inferior Longitudinal', 'stream': 'ventral'},
    'UF':    {'from': 'G8_atl', 'to': 'G6_limbic', 'name': 'Uncinate Fasciculus', 'stream': 'ventral'},
    'CG_exec': {'from': 'G6_limbic', 'to': 'G4_pfc', 'name': 'Cingulum (executive)', 'stream': 'dorsal'},
    'CG_dmn': {'from': 'G5_dmn', 'to': 'G6_limbic', 'name': 'Cingulum (DMN)', 'stream': 'generative'},
    'CC':    {'from': 'G4_pfc', 'to': 'G5_dmn', 'name': 'Corpus Callosum', 'stream': 'bilateral'},
    'MdLF':  {'from': 'G2_wernicke', 'to': 'G7_sensory', 'name': 'Middle Longitudinal', 'stream': 'ventral'},
}

STREAM_COLORS = {
    'dorsal': '#1f77b4',
    'ventral': '#ff7f0e',
    'generative': '#2ca02c',
    'modulatory': '#d62728',
    'convergence': '#9467bd',
    'bilateral': '#8c564b',
}

# ─── Figure 1: SGP Resonance Graph (FIXED: differential coloring) ────────────

print("\nGenerating Figure 1: SGP Resonance Graph (differential)...")

fig, ax = plt.subplots(figsize=(14, 12))

max_diff = max(abs(cat_diffs[cat][node]) for cat in categories for node in NODE_ORDER)
cmap = plt.cm.RdBu_r
norm = mcolors.Normalize(vmin=-max_diff, vmax=max_diff)

for edge_name, edge_info in FIBER_BUNDLES.items():
    from_node = edge_info['from']
    to_node = edge_info['to']
    stream = edge_info['stream']
    
    x1, y1 = NODE_POSITIONS[from_node]['x'], NODE_POSITIONS[from_node]['y']
    x2, y2 = NODE_POSITIONS[to_node]['x'], NODE_POSITIONS[to_node]['y']
    
    edge_weights = df['edge_weights'].apply(lambda x: x.get(edge_name, 0) if isinstance(x, dict) else 0)
    mean_weight = edge_weights.mean()
    std_weight = edge_weights.std()
    
    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = x2 - x1, y2 - y1
    perp_x, perp_y = -dy * 0.15, dx * 0.15
    ctrl_x, ctrl_y = mid_x + perp_x, mid_y + perp_y
    
    lw = 1 + mean_weight * 8
    alpha = 0.3 + mean_weight * 0.5
    
    t = np.linspace(0, 1, 50)
    bezier_x = (1-t)**2 * x1 + 2*(1-t)*t * ctrl_x + t**2 * x2
    bezier_y = (1-t)**2 * y1 + 2*(1-t)*t * ctrl_y + t**2 * y2
    
    color = STREAM_COLORS.get(stream, 'gray')
    ax.plot(bezier_x, bezier_y, color=color, linewidth=lw, alpha=alpha, zorder=1)
    
    ax.text(ctrl_x, ctrl_y, f'{edge_name}\n{mean_weight:.3f}±{std_weight:.3f}', fontsize=6,
            ha='center', va='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8),
            zorder=3)

for node, pos in NODE_POSITIONS.items():
    diff = cat_diffs['emotional'][node]
    mean_activation = overall_means[node]
    std_activation = overall_stds[node]
    
    color = cmap(norm(diff))
    
    circle = plt.Circle((pos['x'], pos['y']), 0.05 + std_activation * 0.5,
                        color=color, alpha=0.8, zorder=2)
    ax.add_patch(circle)
    
    inner = plt.Circle((pos['x'], pos['y']), 0.03 + std_activation * 0.3,
                       color='white', alpha=0.9, zorder=3)
    ax.add_patch(inner)
    
    ax.text(pos['x'], pos['y'], f"{pos['label']}\n{mean_activation:.3f}\nσ={std_activation:.3f}",
            fontsize=7, ha='center', va='center', fontweight='bold',
            color='black', zorder=4)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Differential Activation (from overall mean)', fontsize=10)

legend_elements = [plt.Line2D([0], [0], color=STREAM_COLORS[s], lw=3, label=f'{s.capitalize()} stream')
                   for s in STREAM_COLORS]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9, title='Stream Type')

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_aspect('equal')
ax.set_title('SGP Resonance Graph: 9-Node Geometry with Fiber Bundle Connections\n'
             'Node color = differential activation, size = std, edge width = mean weight',
             fontsize=14, fontweight='bold', pad=20)
ax.axis('off')

plt.tight_layout()
plt.savefig(f'{output_dir}/geo_01_sgp_resonance_graph.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: geo_01_sgp_resonance_graph.png")

# ─── Figure 2: Activation Field (FIXED: 10 category subplots) ────────────────

print("Generating Figure 2: Activation Field (10 categories)...")

grid_x, grid_y = np.mgrid[-1:1:100j, -1:1:100j]
points = np.array([[NODE_POSITIONS[n]['x'], NODE_POSITIONS[n]['y']] for n in NODE_ORDER])

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

all_diffs = []
for cat in categories:
    values = np.array([cat_diffs[cat][n] for n in NODE_ORDER])
    all_diffs.extend(values)
max_abs_diff = max(abs(min(all_diffs)), abs(max(all_diffs)))

for idx, cat in enumerate(categories):
    ax = axes[idx]
    values = np.array([cat_diffs[cat][n] for n in NODE_ORDER])
    
    field = griddata(points, values, (grid_x, grid_y), method='cubic', fill_value=0)
    field = np.nan_to_num(field, nan=0)
    
    im = ax.contourf(grid_x, grid_y, field, levels=20, cmap='RdBu_r',
                     vmin=-max_abs_diff, vmax=max_abs_diff)
    ax.scatter(points[:, 0], points[:, 1], c='black', s=50, zorder=5)
    
    for node in NODE_ORDER:
        pos = NODE_POSITIONS[node]
        diff = cat_diffs[cat][node]
        sign = '+' if diff >= 0 else ''
        ax.text(pos['x'], pos['y'], f"{sign}{diff:.3f}", fontsize=6,
                ha='center', va='center', fontweight='bold', color='black', zorder=6)
    
    plt.colorbar(im, ax=ax, label='Deviation', fraction=0.046)
    ax.set_title(f'{cat.capitalize()}\n(n={cat_ns[cat]})', fontsize=9, fontweight='bold')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.axis('off')

plt.suptitle('Category-Specific Activation Fields (Deviation from Overall Mean)\n'
             'Red = above mean, Blue = below mean', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{output_dir}/geo_02_activation_field.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: geo_02_activation_field.png")

# ─── Figure 3: Category-Specific Patterns (FIXED: differential + significance) ─

print("Generating Figure 3: Category-specific patterns (differential + significance)...")

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

for idx, cat in enumerate(categories):
    ax = axes[idx]
    
    for edge_name, edge_info in FIBER_BUNDLES.items():
        from_node = edge_info['from']
        to_node = edge_info['to']
        
        x1, y1 = NODE_POSITIONS[from_node]['x'], NODE_POSITIONS[from_node]['y']
        x2, y2 = NODE_POSITIONS[to_node]['x'], NODE_POSITIONS[to_node]['y']
        
        cat_data = df[df['category'] == cat]
        edge_weights = cat_data['edge_weights'].apply(lambda x: x.get(edge_name, 0) if isinstance(x, dict) else 0)
        mean_weight = edge_weights.mean()
        
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = x2 - x1, y2 - y1
        perp_x, perp_y = -dy * 0.15, dx * 0.15
        ctrl_x, ctrl_y = mid_x + perp_x, mid_y + perp_y
        
        t = np.linspace(0, 1, 30)
        bezier_x = (1-t)**2 * x1 + 2*(1-t)*t * ctrl_x + t**2 * x2
        bezier_y = (1-t)**2 * y1 + 2*(1-t)*t * ctrl_y + t**2 * y2
        
        stream = edge_info['stream']
        color = STREAM_COLORS.get(stream, 'gray')
        lw = 1 + mean_weight * 6
        ax.plot(bezier_x, bezier_y, color=color, linewidth=lw, alpha=0.4, zorder=1)
    
    for node, pos in NODE_POSITIONS.items():
        diff = cat_diffs[cat][node]
        abs_diff = abs(diff)
        color = cmap(norm(diff))
        
        size = 0.05 + abs_diff * 5
        circle = plt.Circle((pos['x'], pos['y']), size,
                            color=color, alpha=0.8, zorder=2)
        ax.add_patch(circle)
        
        inner = plt.Circle((pos['x'], pos['y']), size * 0.6,
                           color='white', alpha=0.9, zorder=3)
        ax.add_patch(inner)
        
        sign = '+' if diff >= 0 else ''
        ax.text(pos['x'], pos['y'], f"{sign}{diff:.3f}",
                fontsize=6, ha='center', va='center', fontweight='bold',
                color='black', zorder=4)
    
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.set_title(f'{cat.capitalize()} (n={cat_ns[cat]})', fontsize=10, fontweight='bold')
    ax.axis('off')

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.01, label='Differential Activation')

plt.suptitle('Category-Specific SGP Geometric Patterns (Deviation from Overall Mean)\n'
             'Red = above mean, Blue = below mean, Node size = |deviation|', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{output_dir}/geo_03_category_patterns.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: geo_03_category_patterns.png")

# ─── Figure 4: Fiber Bundle Analysis (FIXED: diverging bars) ─────────────────

print("Generating Figure 4: Fiber bundle analysis (diverging)...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

edge_names = list(FIBER_BUNDLES.keys())
cat_edge_data = pd.DataFrame()

overall_edge_means = {}
for edge in edge_names:
    overall_edge_means[edge] = df['edge_weights'].apply(lambda x: x.get(edge, 0) if isinstance(x, dict) else 0).mean()

for cat in categories:
    cat_data = df[df['category'] == cat]
    for edge in edge_names:
        vals = cat_data['edge_weights'].apply(lambda x: x.get(edge, 0) if isinstance(x, dict) else 0)
        cat_edge_data.loc[cat, edge] = vals.mean() - overall_edge_means[edge]

x = np.arange(len(categories))
width = 0.08

for i, edge in enumerate(edge_names):
    values = [cat_edge_data.loc[cat, edge] for cat in categories]
    ax1.bar(x + i*width, values, width, label=edge, alpha=0.8)

ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax1.set_xlabel('Stimulus Category', fontsize=12, fontweight='bold')
ax1.set_ylabel('Edge Weight Deviation from Mean', fontsize=12, fontweight='bold')
ax1.set_title('Fiber Bundle Edge Weight Deviations by Category', fontsize=13, fontweight='bold')
ax1.set_xticks(x + width * 4)
ax1.set_xticklabels([c.capitalize() for c in categories], rotation=45, ha='right')
ax1.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1))

stream_data = pd.DataFrame()
overall_stream_means = {}

for stream in STREAM_COLORS.keys():
    edges_in_stream = [e for e, info in FIBER_BUNDLES.items() if info['stream'] == stream]
    if edges_in_stream:
        all_vals = []
        for edge in edges_in_stream:
            vals = df['edge_weights'].apply(lambda x: x.get(edge, 0) if isinstance(x, dict) else 0)
            all_vals.extend(vals.values)
        overall_stream_means[stream] = np.mean(all_vals)

for cat in categories:
    cat_data = df[df['category'] == cat]
    for stream in STREAM_COLORS.keys():
        edges_in_stream = [e for e, info in FIBER_BUNDLES.items() if info['stream'] == stream]
        if edges_in_stream:
            stream_vals = []
            for edge in edges_in_stream:
                vals = cat_data['edge_weights'].apply(lambda x: x.get(edge, 0) if isinstance(x, dict) else 0)
                stream_vals.extend(vals.values)
            stream_data.loc[cat, stream] = np.mean(stream_vals) - overall_stream_means.get(stream, 0)

x = np.arange(len(categories))
width = 0.12
for i, stream in enumerate(STREAM_COLORS.keys()):
    if stream in stream_data.columns:
        values = [stream_data.loc[cat, stream] for cat in categories]
        ax2.bar(x + i*width, values, width, label=stream.capitalize(),
                color=STREAM_COLORS[stream], alpha=0.8)

ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.set_xlabel('Stimulus Category', fontsize=12, fontweight='bold')
ax2.set_ylabel('Stream Weight Deviation from Mean', fontsize=12, fontweight='bold')
ax2.set_title('Stream-Type Weight Deviations by Category', fontsize=13, fontweight='bold')
ax2.set_xticks(x + width * 3)
ax2.set_xticklabels([c.capitalize() for c in categories], rotation=45, ha='right')
ax2.legend(fontsize=10)

plt.suptitle('Fiber Bundle Analysis: Differential Edge Weights and Stream Types', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/geo_04_fiber_bundle_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: geo_04_fiber_bundle_analysis.png")

# ─── Figure 5: 3D Geometry (FIXED: differential coloring) ────────────────────

print("Generating Figure 5: 3D SGP geometry (differential)...")

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

for node, pos in NODE_POSITIONS.items():
    diff = cat_diffs['emotional'][node]
    color = cmap(norm(diff))
    
    ax.scatter(pos['x'], pos['y'], pos['z'], s=200 + abs(diff) * 2000,
               c=[color], alpha=0.8, edgecolors='black', linewidth=2, zorder=5)
    
    sign = '+' if diff >= 0 else ''
    ax.text(pos['x'], pos['y'], pos['z'], f" {node}\n{sign}{diff:.3f}",
            fontsize=8, fontweight='bold', zorder=6)

for edge_name, edge_info in FIBER_BUNDLES.items():
    from_node = edge_info['from']
    to_node = edge_info['to']
    
    x1, y1, z1 = NODE_POSITIONS[from_node]['x'], NODE_POSITIONS[from_node]['y'], NODE_POSITIONS[from_node]['z']
    x2, y2, z2 = NODE_POSITIONS[to_node]['x'], NODE_POSITIONS[to_node]['y'], NODE_POSITIONS[to_node]['z']
    
    stream = edge_info['stream']
    color = STREAM_COLORS.get(stream, 'gray')
    
    edge_weights = df['edge_weights'].apply(lambda x: x.get(edge_name, 0) if isinstance(x, dict) else 0)
    mean_weight = edge_weights.mean()
    
    ax.plot([x1, x2], [y1, y2], [z1, z2], color=color, linewidth=1 + mean_weight * 5,
            alpha=0.4, zorder=1)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.1, label='Differential Activation')

ax.set_title('3D SGP Geometric Structure\nNodes colored by differential activation (emotional category)',
             fontsize=14, fontweight='bold')
ax.set_xlabel('X (Left-Right)')
ax.set_ylabel('Y (Anterior-Posterior)')
ax.set_zlabel('Z (Dorsal-Ventral)')

plt.tight_layout()
plt.savefig(f'{output_dir}/geo_05_3d_geometry.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: geo_05_3d_geometry.png")

# ─── Figure 9: Statistical Significance (NEW) ────────────────────────────────

print("Generating Figure 9: Statistical significance...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

nodes = NODE_ORDER
f_stats = [anova_results[n]['f_stat'] for n in nodes]
p_values = [anova_results[n]['p_value'] for n in nodes]

colors_anova = ['red' if p < 0.001 else 'orange' if p < 0.01 else 'green' if p < 0.05 else 'gray' for p in p_values]
bars = ax1.barh(nodes, f_stats, color=colors_anova, alpha=0.8)

for bar, f_stat, p_val in zip(bars, f_stats, p_values):
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
             f'F={f_stat:.2f}, p={p_val:.4f} {sig}',
             va='center', fontsize=8)

ax1.axvline(x=stats.f.ppf(0.95, len(categories)-1, len(df)-len(categories)), color='red',
            linestyle='--', alpha=0.5, label='p=0.05 threshold')
ax1.set_xlabel('F-statistic', fontsize=12, fontweight='bold')
ax1.set_title('ANOVA F-Tests: Node Activation Differences Across Categories\n'
              'Red=***p<0.001, Orange=**p<0.01, Green=*p<0.05, Gray=ns', fontsize=12, fontweight='bold')
ax1.legend()

effect_matrix = np.array([[effect_sizes[node][cat] for cat in categories] for node in nodes])

im = ax2.imshow(effect_matrix, cmap='RdBu_r', aspect='auto',
                vmin=-max(abs(effect_matrix.min()), abs(effect_matrix.max())),
                vmax=max(abs(effect_matrix.min()), abs(effect_matrix.max())))
ax2.set_xticks(range(len(categories)))
ax2.set_xticklabels([c.capitalize() for c in categories], rotation=45, ha='right', fontsize=9)
ax2.set_yticks(range(len(nodes)))
ax2.set_yticklabels(nodes, fontsize=9)

for i in range(len(nodes)):
    for j in range(len(categories)):
        val = effect_matrix[i, j]
        color = 'white' if abs(val) > 0.5 else 'black'
        ax2.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color=color, fontweight='bold')

ax2.set_title('Effect Sizes (Cohen\'s d): Category vs Rest\n'
              'Red=positive effect, Blue=negative effect', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax2, label="Cohen's d")

plt.suptitle('Statistical Significance Analysis: ANOVA and Effect Sizes', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/geo_09_statistical_significance.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: geo_09_statistical_significance.png")

# ─── Figure 10: Differential Activation Heatmap (NEW) ────────────────────────

print("Generating Figure 10: Differential activation heatmap...")

fig, ax = plt.subplots(figsize=(12, 6))

diff_matrix = np.array([[cat_diffs[cat][node] for cat in categories] for node in nodes])

im = ax.imshow(diff_matrix, cmap='RdBu_r', aspect='auto',
               vmin=-max_abs_diff, vmax=max_abs_diff)
ax.set_xticks(range(len(categories)))
ax.set_xticklabels([c.capitalize() for c in categories], rotation=45, ha='right', fontsize=10)
ax.set_yticks(range(len(nodes)))
ax.set_yticklabels(nodes, fontsize=10)

for i in range(len(nodes)):
    for j in range(len(categories)):
        val = diff_matrix[i, j]
        color = 'white' if abs(val) > max_abs_diff * 0.5 else 'black'
        sign = '+' if val >= 0 else ''
        ax.text(j, i, f'{sign}{val:.3f}', ha='center', va='center', fontsize=8, color=color, fontweight='bold')

ax.set_title('Differential Activation Heatmap: Category Deviations from Overall Mean\n'
             'Red=above mean, Blue=below mean', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='Deviation from Mean')

plt.tight_layout()
plt.savefig(f'{output_dir}/geo_10_differential_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: geo_10_differential_heatmap.png")

# ─── Summary ─────────────────────────────────────────────────────────────────

print(f"\n{'=' * 60}")
print("SGP GEOMETRY VISUALIZATION COMPLETE (FIXED)")
print(f"{'=' * 60}")
print(f"Figures saved to: {output_dir}/")
print(f"  - geo_01_sgp_resonance_graph.png (FIXED: differential coloring)")
print(f"  - geo_02_activation_field.png (FIXED: 10 category subplots)")
print(f"  - geo_03_category_patterns.png (FIXED: differential + significance)")
print(f"  - geo_04_fiber_bundle_analysis.png (FIXED: diverging bars)")
print(f"  - geo_05_3d_geometry.png (FIXED: differential coloring)")
print(f"  - geo_09_statistical_significance.png (NEW: ANOVA + effect sizes)")
print(f"  - geo_10_differential_heatmap.png (NEW: 9x10 heatmap)")
print(f"\nStatistical analysis saved to: results/full_battery_1000/statistical_analysis.json")
