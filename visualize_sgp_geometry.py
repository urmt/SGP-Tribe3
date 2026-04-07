"""
SGP-Tribe3 — Phase 2: SGP Geometry Visualization
===================================================
Creates publication-quality visualizations of the SGP geometric structure:
1. SGP Resonance Graph with fiber bundle connections
2. Activation field visualization over the 9-node geometry
3. Category-specific geometric patterns
4. Fiber bundle edge weight analysis

Usage:
    python visualize_sgp_geometry.py

Outputs:
    results/full_battery_1000/figures/geo_*.png
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
from matplotlib.patches import FancyArrowPatch
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
from scipy.interpolate import griddata
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

output_dir = "results/full_battery_1000/figures"
os.makedirs(output_dir, exist_ok=True)

# ─── SGP Geometry Definitions ────────────────────────────────────────────────

# Node positions in geometric space (based on MNI coordinates, normalized)
# These represent the "base space" of the fiber bundle
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

# White matter tract connections (fiber bundles)
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
    'dorsal': '#1f77b4',      # Blue
    'ventral': '#ff7f0e',      # Orange
    'generative': '#2ca02c',   # Green
    'modulatory': '#d62728',   # Red
    'convergence': '#9467bd',  # Purple
    'bilateral': '#8c564b',    # Brown
}

# ─── Figure 1: SGP Resonance Graph ──────────────────────────────────────────

print("Generating Figure: SGP Resonance Graph (fiber bundles)...")

fig, ax = plt.subplots(figsize=(14, 12))

# Draw fiber bundle connections
for edge_name, edge_info in FIBER_BUNDLES.items():
    from_node = edge_info['from']
    to_node = edge_info['to']
    stream = edge_info['stream']
    
    x1, y1 = NODE_POSITIONS[from_node]['x'], NODE_POSITIONS[from_node]['y']
    x2, y2 = NODE_POSITIONS[to_node]['x'], NODE_POSITIONS[to_node]['y']
    
    # Calculate mean edge weight from results
    edge_weights = df['edge_weights'].apply(lambda x: x.get(edge_name, 0) if isinstance(x, dict) else 0)
    mean_weight = edge_weights.mean()
    
    # Draw curved fiber bundle
    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
    # Add slight curvature
    dx, dy = x2 - x1, y2 - y1
    perp_x, perp_y = -dy * 0.15, dx * 0.15
    ctrl_x, ctrl_y = mid_x + perp_x, mid_y + perp_y
    
    # Line width proportional to edge weight
    lw = 1 + mean_weight * 8
    alpha = 0.3 + mean_weight * 0.5
    
    t = np.linspace(0, 1, 50)
    bezier_x = (1-t)**2 * x1 + 2*(1-t)*t * ctrl_x + t**2 * x2
    bezier_y = (1-t)**2 * y1 + 2*(1-t)*t * ctrl_y + t**2 * y2
    
    color = STREAM_COLORS.get(stream, 'gray')
    ax.plot(bezier_x, bezier_y, color=color, linewidth=lw, alpha=alpha, zorder=1)
    
    # Add edge label
    ax.text(ctrl_x, ctrl_y, f'{edge_name}\n{mean_weight:.2f}', fontsize=7,
            ha='center', va='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8),
            zorder=3)

# Draw nodes
for node, pos in NODE_POSITIONS.items():
    # Node size based on mean activation
    node_activations = df[node]
    mean_activation = node_activations.mean()
    std_activation = node_activations.std()
    
    # Node color based on stream
    stream = pos['stream']
    color = STREAM_COLORS.get(stream, 'gray')
    
    # Draw node as circle with size proportional to activation
    circle = plt.Circle((pos['x'], pos['y']), 0.05 + mean_activation * 0.08,
                        color=color, alpha=0.7, zorder=2)
    ax.add_patch(circle)
    
    # Inner circle showing activation
    inner = plt.Circle((pos['x'], pos['y']), 0.03 + mean_activation * 0.05,
                       color='white', alpha=0.9, zorder=3)
    ax.add_patch(inner)
    
    # Label
    ax.text(pos['x'], pos['y'], f"{pos['label']}\n{mean_activation:.2f}",
            fontsize=9, ha='center', va='center', fontweight='bold',
            color='black', zorder=4)
    
    # Add std as error ring
    ring = plt.Circle((pos['x'], pos['y']), 0.05 + mean_activation * 0.08 + std_activation * 0.05,
                      fill=False, edgecolor=color, linewidth=1, linestyle='--', alpha=0.5, zorder=2)
    ax.add_patch(ring)

# Add legend
legend_elements = [plt.Line2D([0], [0], color=STREAM_COLORS[s], lw=3, label=f'{s.capitalize()} stream')
                   for s in STREAM_COLORS]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9, title='Stream Type')

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_aspect('equal')
ax.set_title('SGP Resonance Graph: 9-Node Geometry with Fiber Bundle Connections\n'
             'Node size = mean activation, line width = edge weight, dashed ring = std',
             fontsize=14, fontweight='bold', pad=20)
ax.axis('off')

plt.tight_layout()
plt.savefig(f'{output_dir}/geo_01_sgp_resonance_graph.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: geo_01_sgp_resonance_graph.png")

# ─── Figure 2: Activation Field Visualization ────────────────────────────────

print("Generating Figure: Activation Field (continuous field over SGP space)...")

# Create a grid for field interpolation
grid_x, grid_y = np.mgrid[-1:1:200j, -1:1:200j]

# Get mean activations for all nodes
mean_activations = {node: df[node].mean() for node in NODE_ORDER}

# Interpolate to create continuous field
points = np.array([[NODE_POSITIONS[n]['x'], NODE_POSITIONS[n]['y']] for n in NODE_ORDER])
values = np.array([mean_activations[n] for n in NODE_ORDER])

# Use inverse distance weighting for smooth field
field = griddata(points, values, (grid_x, grid_y), method='cubic', fill_value=0)
field = np.nan_to_num(field, nan=0.5)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Left: Continuous activation field
im1 = ax1.contourf(grid_x, grid_y, field, levels=20, cmap='RdYlBu_r', vmin=0.85, vmax=1.0)
ax1.scatter(points[:, 0], points[:, 1], c='black', s=100, zorder=5)
for node in NODE_ORDER:
    pos = NODE_POSITIONS[node]
    ax1.text(pos['x'], pos['y'], node.replace('_', '\n'), fontsize=7,
             ha='center', va='center', fontweight='bold', color='white', zorder=6)
plt.colorbar(im1, ax=ax1, label='Mean Activation')
ax1.set_title('Continuous Activation Field\nover SGP Geometric Space', fontsize=12, fontweight='bold')
ax1.set_xlim(-1, 1)
ax1.set_ylim(-1, 1)
ax1.set_aspect('equal')
ax1.axis('off')

# Right: Field gradient (showing "flow" of activation)
grad_y, grad_x = np.gradient(field)
magnitude = np.sqrt(grad_x**2 + grad_y**2)

im2 = ax2.contourf(grid_x, grid_y, magnitude, levels=20, cmap='viridis')
# Add streamlines using regular grid
x_reg = np.linspace(-1, 1, 200)
y_reg = np.linspace(-1, 1, 200)
X_reg, Y_reg = np.meshgrid(x_reg, y_reg)
try:
    ax2.streamplot(X_reg, Y_reg, grad_x, grad_y, color='white', linewidth=0.5,
                   density=1.5, arrowstyle='->', arrowsize=1)
except Exception:
    pass  # Skip streamlines if grid issues
ax2.scatter(points[:, 0], points[:, 1], c='black', s=100, zorder=5)
plt.colorbar(im2, ax=ax2, label='Activation Gradient')
ax2.set_title('Activation Gradient Field\nwith Flow Streamlines', fontsize=12, fontweight='bold')
ax2.set_xlim(-1, 1)
ax2.set_ylim(-1, 1)
ax2.set_aspect('equal')
ax2.axis('off')

plt.suptitle('SGP Activation Field: Continuous Geometric Representation\n'
             'Left: Activation magnitude, Right: Gradient flow', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/geo_02_activation_field.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: geo_02_activation_field.png")

# ─── Figure 3: Category-Specific Geometric Patterns ─────────────────────────

print("Generating Figure: Category-specific geometric patterns...")

categories = ['simple', 'logical', 'emotional', 'factual', 'abstract',
              'spatial', 'social', 'motor', 'memory', 'auditory']

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

for idx, cat in enumerate(categories):
    cat_data = df[df['category'] == cat]
    cat_activations = {node: cat_data[node].mean() for node in NODE_ORDER}
    
    ax = axes[idx]
    
    # Draw fiber bundles
    for edge_name, edge_info in FIBER_BUNDLES.items():
        from_node = edge_info['from']
        to_node = edge_info['to']
        
        x1, y1 = NODE_POSITIONS[from_node]['x'], NODE_POSITIONS[from_node]['y']
        x2, y2 = NODE_POSITIONS[to_node]['x'], NODE_POSITIONS[to_node]['y']
        
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
    
    # Draw nodes
    for node, pos in NODE_POSITIONS.items():
        activation = cat_activations[node]
        color = STREAM_COLORS.get(pos['stream'], 'gray')
        
        circle = plt.Circle((pos['x'], pos['y']), 0.05 + activation * 0.08,
                            color=color, alpha=0.7, zorder=2)
        ax.add_patch(circle)
        
        inner = plt.Circle((pos['x'], pos['y']), 0.03 + activation * 0.05,
                           color='white', alpha=0.9, zorder=3)
        ax.add_patch(inner)
        
        ax.text(pos['x'], pos['y'], f"{activation:.2f}",
                fontsize=7, ha='center', va='center', fontweight='bold',
                color='black', zorder=4)
    
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.set_title(f'{cat.capitalize()} (n={len(cat_data)})', fontsize=10, fontweight='bold')
    ax.axis('off')

plt.suptitle('Category-Specific SGP Geometric Patterns\n'
             'Node values = mean activation, line width = edge weight', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{output_dir}/geo_03_category_patterns.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: geo_03_category_patterns.png")

# ─── Figure 4: Fiber Bundle Analysis ─────────────────────────────────────────

print("Generating Figure: Fiber bundle edge weight analysis...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left: Edge weights by category
edge_names = list(FIBER_BUNDLES.keys())
cat_edge_data = pd.DataFrame()
for cat in categories:
    cat_data = df[df['category'] == cat]
    for edge in edge_names:
        cat_edge_data.loc[cat, edge] = cat_data['edge_weights'].apply(
            lambda x: x.get(edge, 0) if isinstance(x, dict) else 0).mean()

x = np.arange(len(categories))
width = 0.08
colors_edge = plt.cm.tab10(np.linspace(0, 1, len(edge_names)))

for i, edge in enumerate(edge_names):
    values = [cat_edge_data.loc[cat, edge] for cat in categories]
    ax1.bar(x + i*width, values, width, label=edge, alpha=0.8)

ax1.set_xlabel('Stimulus Category', fontsize=12, fontweight='bold')
ax1.set_ylabel('Mean Edge Weight', fontsize=12, fontweight='bold')
ax1.set_title('Fiber Bundle Edge Weights by Category', fontsize=13, fontweight='bold')
ax1.set_xticks(x + width * 4)
ax1.set_xticklabels([c.capitalize() for c in categories], rotation=45, ha='right')
ax1.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1, 1))
ax1.set_ylim(0.85, 1.02)

# Right: Stream-type aggregation
stream_data = pd.DataFrame()
for cat in categories:
    cat_data = df[df['category'] == cat]
    for stream in STREAM_COLORS.keys():
        edges_in_stream = [e for e, info in FIBER_BUNDLES.items() if info['stream'] == stream]
        if edges_in_stream:
            stream_vals = []
            for edge in edges_in_stream:
                vals = cat_data['edge_weights'].apply(lambda x: x.get(edge, 0) if isinstance(x, dict) else 0)
                stream_vals.extend(vals.values)
            stream_data.loc[cat, stream] = np.mean(stream_vals)

x = np.arange(len(categories))
width = 0.12
for i, stream in enumerate(STREAM_COLORS.keys()):
    if stream in stream_data.columns:
        values = [stream_data.loc[cat, stream] for cat in categories]
        ax2.bar(x + i*width, values, width, label=stream.capitalize(),
                color=STREAM_COLORS[stream], alpha=0.8)

ax2.set_xlabel('Stimulus Category', fontsize=12, fontweight='bold')
ax2.set_ylabel('Mean Stream Activation', fontsize=12, fontweight='bold')
ax2.set_title('Stream-Type Aggregated Edge Weights', fontsize=13, fontweight='bold')
ax2.set_xticks(x + width * 3)
ax2.set_xticklabels([c.capitalize() for c in categories], rotation=45, ha='right')
ax2.legend(fontsize=10)
ax2.set_ylim(0.85, 1.02)

plt.suptitle('Fiber Bundle Analysis: Edge Weights and Stream Types', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/geo_04_fiber_bundle_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: geo_04_fiber_bundle_analysis.png")

# ─── Figure 5: 3D Geometric Projection ──────────────────────────────────────

print("Generating Figure: 3D SGP geometry...")

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot nodes in 3D
for node, pos in NODE_POSITIONS.items():
    activation = df[node].mean()
    color = STREAM_COLORS.get(pos['stream'], 'gray')
    
    ax.scatter(pos['x'], pos['y'], pos['z'], s=200 + activation * 500,
               c=color, alpha=0.7, edgecolors='black', linewidth=2, zorder=5)
    
    ax.text(pos['x'], pos['y'], pos['z'], f" {node}\n{activation:.2f}",
            fontsize=8, fontweight='bold', zorder=6)

# Plot fiber bundles in 3D
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

ax.set_title('3D SGP Geometric Structure\nNodes in MNI-derived space with fiber bundle connections',
             fontsize=14, fontweight='bold')
ax.set_xlabel('X (Left-Right)')
ax.set_ylabel('Y (Anterior-Posterior)')
ax.set_zlabel('Z (Dorsal-Ventral)')

plt.tight_layout()
plt.savefig(f'{output_dir}/geo_05_3d_geometry.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: geo_05_3d_geometry.png")

# ─── Summary ─────────────────────────────────────────────────────────────────

print(f"\n{'=' * 60}")
print("SGP GEOMETRY VISUALIZATION COMPLETE")
print(f"{'=' * 60}")
print(f"Figures saved to: {output_dir}/")
print(f"  - geo_01_sgp_resonance_graph.png")
print(f"  - geo_02_activation_field.png")
print(f"  - geo_03_category_patterns.png")
print(f"  - geo_04_fiber_bundle_analysis.png")
print(f"  - geo_05_3d_geometry.png")
