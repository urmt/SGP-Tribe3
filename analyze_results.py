"""
SGP-Tribe3 — Statistical Analysis & Visualization Pipeline
==========================================================
Analyzes research battery results and generates publication-quality figures.

Usage:
    python analyze_results.py
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from datetime import datetime

# Find latest results file
results_files = sorted(glob.glob("results/research_battery_*.json"))
if not results_files:
    print("ERROR: No research battery results found in results/")
    sys.exit(1)

RESULTS_FILE = results_files[-1]
print(f"Loading results from: {RESULTS_FILE}")

with open(RESULTS_FILE, 'r') as f:
    data = json.load(f)

results = data['results']
metadata = data['metadata']

# Convert to DataFrame
df = pd.DataFrame(results)

# Node ordering for consistent display
NODE_ORDER = ['G1_broca', 'G2_wernicke', 'G3_tpj', 'G4_pfc', 'G5_dmn', 'G6_limbic', 'G7_sensory', 'G8_atl', 'G9_premotor']
STREAM_ORDER = ['dorsal', 'ventral', 'generative', 'modulatory', 'convergence']
CATEGORY_ORDER = ['simple', 'logical', 'emotional', 'factual', 'abstract', 'spatial', 'social', 'motor', 'memory', 'auditory']

# Expand sgp_nodes into separate columns
for node in NODE_ORDER:
    df[node] = df['sgp_nodes'].apply(lambda x: x.get(node, 0))

for stream in STREAM_ORDER:
    df[stream] = df['streams'].apply(lambda x: x.get(stream, 0))

print(f"\nLoaded {len(df)} stimuli across {df['category'].nunique()} categories")
print(f"Text encoder: {metadata.get('text_encoder', 'unknown')}")
print(f"Total time: {metadata.get('total_time_minutes', 0):.1f} minutes")

# Create output directory
output_dir = "results/figures"
os.makedirs(output_dir, exist_ok=True)

# ─── Figure 1: Node Activation Heatmap by Category ───────────────────────────
print("\nGenerating Figure 1: Node Activation Heatmap...")

heatmap_data = pd.DataFrame()
for cat in CATEGORY_ORDER:
    cat_data = df[df['category'] == cat]
    for node in NODE_ORDER:
        heatmap_data.loc[cat, node] = cat_data[node].mean()

fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(heatmap_data.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(len(NODE_ORDER)))
ax.set_xticklabels(NODE_ORDER, rotation=45, ha='right', fontsize=10)
ax.set_yticks(range(len(CATEGORY_ORDER)))
ax.set_yticklabels([c.capitalize() for c in CATEGORY_ORDER], fontsize=10)

# Add text annotations
for i in range(len(CATEGORY_ORDER)):
    for j in range(len(NODE_ORDER)):
        val = heatmap_data.iloc[i, j]
        ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=8,
                color='black' if val < 0.7 else 'white')

ax.set_xlabel('SGP Node', fontsize=12, fontweight='bold')
ax.set_ylabel('Stimulus Category', fontsize=12, fontweight='bold')
ax.set_title('SGP Node Activation Profiles by Stimulus Category\n(Mean Activation, 30 Stimuli)', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='Mean Activation')
plt.tight_layout()
plt.savefig(f'{output_dir}/fig1_node_activation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/fig1_node_activation_heatmap.png")

# ─── Figure 2: Category-wise Node Activation Bar Chart ───────────────────────
print("Generating Figure 2: Category-wise Node Activation...")

fig, axes = plt.subplots(2, 5, figsize=(22, 8))
axes = axes.flatten()

for idx, cat in enumerate(CATEGORY_ORDER):
    cat_data = df[df['category'] == cat]
    means = [cat_data[node].mean() for node in NODE_ORDER]
    stds = [cat_data[node].std() for node in NODE_ORDER]
    
    ax = axes[idx]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(NODE_ORDER)))
    bars = ax.bar(NODE_ORDER, means, yerr=stds, capsize=3, color=colors, alpha=0.8)
    ax.set_title(f'{cat.capitalize()} (n={len(cat_data)})', fontsize=11, fontweight='bold')
    ax.set_ylim(0.85, 1.02)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    
    # Highlight top 2 nodes
    top2_idx = np.argsort(means)[-2:]
    for i in top2_idx:
        bars[i].set_edgecolor('red')
        bars[i].set_linewidth(2)

plt.suptitle('SGP Node Activation by Category with Standard Deviation\nRed borders indicate top 2 dominant nodes', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{output_dir}/fig2_category_node_activations.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/fig2_category_node_activations.png")

# ─── Figure 3: Stream Activation Comparison ──────────────────────────────────
print("Generating Figure 3: Stream Activation Comparison...")

stream_data = pd.DataFrame()
for cat in CATEGORY_ORDER:
    cat_data = df[df['category'] == cat]
    for stream in STREAM_ORDER:
        stream_data.loc[cat, stream] = cat_data[stream].mean()

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(CATEGORY_ORDER))
width = 0.15

for i, stream in enumerate(STREAM_ORDER):
    values = [stream_data.loc[cat, stream] for cat in CATEGORY_ORDER]
    ax.bar(x + i*width, values, width, label=stream.capitalize(), alpha=0.8)

ax.set_xlabel('Stimulus Category', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean Stream Activation', fontsize=12, fontweight='bold')
ax.set_title('Stream Activation Profiles Across Stimulus Categories', fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 2)
ax.set_xticklabels([c.capitalize() for c in CATEGORY_ORDER], rotation=45, ha='right')
ax.legend(fontsize=10)
ax.set_ylim(0.85, 1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/fig3_stream_activations.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/fig3_stream_activations.png")

# ─── Figure 4: Hypothesis Support Rate ───────────────────────────────────────
print("Generating Figure 4: Hypothesis Support Rate...")

hypothesis_support = []
for cat in CATEGORY_ORDER:
    cat_data = df[df['category'] == cat]
    support_rate = cat_data['hypothesis_support'].mean() * 100
    hypothesis_support.append(support_rate)

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['green' if x >= 50 else 'orange' if x >= 25 else 'red' for x in hypothesis_support]
bars = ax.bar([c.capitalize() for c in CATEGORY_ORDER], hypothesis_support, color=colors, alpha=0.8, edgecolor='black')

for bar, val in zip(bars, hypothesis_support):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.0f}%', 
            ha='center', va='bottom', fontweight='bold', fontsize=10)

ax.axhline(y=37, color='gray', linestyle='--', alpha=0.7, label=f'Overall: 37%')
ax.set_xlabel('Stimulus Category', fontsize=12, fontweight='bold')
ax.set_ylabel('Hypothesis Support Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Hickok-Poeppel Dual-Stream Hypothesis Support Rate by Category\n(Green ≥50%, Orange 25-50%, Red <25%)', fontsize=14, fontweight='bold')
ax.set_ylim(0, 120)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f'{output_dir}/fig4_hypothesis_support.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/fig4_hypothesis_support.png")

# ─── Figure 5: Dominant Node Distribution ────────────────────────────────────
print("Generating Figure 5: Dominant Node Distribution...")

# Count how often each node appears as top-1 or top-2
top1_counts = {node: 0 for node in NODE_ORDER}
top2_counts = {node: 0 for node in NODE_ORDER}

for _, row in df.iterrows():
    sorted_nodes = sorted([(node, row[node]) for node in NODE_ORDER], key=lambda x: x[1], reverse=True)
    top1_counts[sorted_nodes[0][0]] += 1
    top2_counts[sorted_nodes[1][0]] += 1

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

colors1 = plt.cm.Set3(np.linspace(0, 1, len(NODE_ORDER)))
ax1.bar(NODE_ORDER, [top1_counts[n] for n in NODE_ORDER], color=colors1, alpha=0.8, edgecolor='black')
ax1.set_title('Most Dominant Node (Rank 1)', fontsize=13, fontweight='bold')
ax1.tick_params(axis='x', rotation=45)
ax1.set_ylabel('Count (out of 30 stimuli)', fontsize=11)

ax2.bar(NODE_ORDER, [top2_counts[n] for n in NODE_ORDER], color=colors1, alpha=0.8, edgecolor='black')
ax2.set_title('Second Most Dominant Node (Rank 2)', fontsize=13, fontweight='bold')
ax2.tick_params(axis='x', rotation=45)
ax2.set_ylabel('Count (out of 30 stimuli)', fontsize=11)

plt.suptitle('Distribution of Dominant SGP Nodes Across All Stimuli', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/fig5_dominant_node_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/fig5_dominant_node_distribution.png")

# ─── Figure 6: Inference Time Analysis ───────────────────────────────────────
print("Generating Figure 6: Inference Time Analysis...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Time by category
cat_times = df.groupby('category')['inference_time_seconds'].agg(['mean', 'std'])
cat_times = cat_times.reindex(CATEGORY_ORDER)

ax1.bar([c.capitalize() for c in cat_times.index], cat_times['mean'], 
        yerr=cat_times['std'], capsize=5, color='steelblue', alpha=0.8, edgecolor='black')
ax1.set_title('Mean Inference Time by Category', fontsize=13, fontweight='bold')
ax1.set_ylabel('Time (seconds)', fontsize=11)
ax1.tick_params(axis='x', rotation=45)

# Time vs text length
ax2.scatter(df['text_length'], df['inference_time_seconds'], alpha=0.7, s=50, c='coral', edgecolors='black')
z = np.polyfit(df['text_length'], df['inference_time_seconds'], 1)
p = np.poly1d(z)
x_line = np.linspace(df['text_length'].min(), df['text_length'].max(), 100)
ax2.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
corr, p_value = stats.pearsonr(df['text_length'], df['inference_time_seconds'])
ax2.set_title(f'Inference Time vs Text Length\n(r={corr:.3f}, p={p_value:.4f})', fontsize=13, fontweight='bold')
ax2.set_xlabel('Text Length (characters)', fontsize=11)
ax2.set_ylabel('Inference Time (seconds)', fontsize=11)

plt.suptitle('Inference Time Analysis (Ollama + Random Projection)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/fig6_inference_time.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/fig6_inference_time.png")

# ─── Figure 7: Co-activation Matrix ─────────────────────────────────────────
print("Generating Figure 7: Co-activation Matrix...")

# Compute correlation matrix across stimuli
activation_matrix = df[NODE_ORDER].values
corr_matrix = np.corrcoef(activation_matrix.T)

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(NODE_ORDER)))
ax.set_xticklabels(NODE_ORDER, rotation=45, ha='right', fontsize=10)
ax.set_yticks(range(len(NODE_ORDER)))
ax.set_yticklabels(NODE_ORDER, fontsize=10)

for i in range(len(NODE_ORDER)):
    for j in range(len(NODE_ORDER)):
        val = corr_matrix[i, j]
        color = 'white' if abs(val) > 0.5 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=9, color=color)

ax.set_title('SGP Node Co-activation Matrix\n(Pearson Correlation Across 30 Stimuli)', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='Pearson Correlation')
plt.tight_layout()
plt.savefig(f'{output_dir}/fig7_coactivation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/fig7_coactivation_matrix.png")

# ─── Figure 8: Edge Weight Analysis ─────────────────────────────────────────
print("Generating Figure 8: Edge Weight Analysis...")

edge_names = ['AF', 'SLF', 'IFOF', 'ILF', 'UF', 'CG_exec', 'CG_dmn', 'CC', 'MdLF']
edge_data = pd.DataFrame()
for cat in CATEGORY_ORDER:
    cat_data = df[df['category'] == cat]
    for edge in edge_names:
        edge_data.loc[cat, edge] = cat_data['edge_weights'].apply(lambda x: x.get(edge, 0)).mean()

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(CATEGORY_ORDER))
width = 0.08

colors = plt.cm.tab10(np.linspace(0, 1, len(edge_names)))
for i, edge in enumerate(edge_names):
    values = [edge_data.loc[cat, edge] for cat in CATEGORY_ORDER]
    ax.bar(x + i*width, values, width, label=edge, alpha=0.8)

ax.set_xlabel('Stimulus Category', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean Edge Weight', fontsize=12, fontweight='bold')
ax.set_title('White Matter Tract Edge Weights by Stimulus Category', fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 4)
ax.set_xticklabels([c.capitalize() for c in CATEGORY_ORDER], rotation=45, ha='right')
ax.legend(fontsize=9, loc='upper left', bbox_to_anchor=(1, 1))
ax.set_ylim(0.85, 1.02)
plt.tight_layout()
plt.savefig(f'{output_dir}/fig8_edge_weights.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_dir}/fig8_edge_weights.png")

# ─── Statistical Summary Table ───────────────────────────────────────────────
print("\nGenerating Statistical Summary Table...")

summary_rows = []
for cat in CATEGORY_ORDER:
    cat_data = df[df['category'] == cat]
    support_rate = cat_data['hypothesis_support'].mean() * 100
    
    row = {
        'Category': cat.capitalize(),
        'N_Stimuli': len(cat_data),
        'Hypothesis_Support_%': round(support_rate, 1),
        'Mean_Inference_Time_s': round(cat_data['inference_time_seconds'].mean(), 2),
        'Std_Inference_Time_s': round(cat_data['inference_time_seconds'].std(), 2),
        'Mean_Word_Count': round(cat_data['word_count'].mean(), 1),
        'Mean_Text_Length': round(cat_data['text_length'].mean(), 1),
    }
    
    # Add top 2 nodes
    sorted_nodes = sorted([(node, cat_data[node].mean()) for node in NODE_ORDER], key=lambda x: x[1], reverse=True)
    row['Top_Node_1'] = sorted_nodes[0][0]
    row['Top_Node_1_Activation'] = round(sorted_nodes[0][1], 4)
    row['Top_Node_2'] = sorted_nodes[1][0]
    row['Top_Node_2_Activation'] = round(sorted_nodes[1][1], 4)
    
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(f'{output_dir}/statistical_summary.csv', index=False)
print(f"  Saved: {output_dir}/statistical_summary.csv")

# ─── Generate Summary Report ─────────────────────────────────────────────────
print("\nGenerating Summary Report...")

successful = df[~df.get('error', pd.Series([False]*len(df))).astype(bool)]
hypothesis_supports = successful[successful['hypothesis_support']]

report = f"""
SGP-Tribe3 Research Battery — Statistical Summary
===================================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Results File: {RESULTS_FILE}
Text Encoder: {metadata.get('text_encoder', 'unknown')}
Total Stimuli: {metadata.get('total_stimuli', 0)}
Successful: {metadata.get('successful', 0)}
Failed: {metadata.get('failed', 0)}
Total Time: {metadata.get('total_time_minutes', 0):.2f} minutes
Average Time per Stimulus: {metadata.get('total_time_minutes', 0)/metadata.get('total_stimuli', 1)*60:.1f} seconds

HYPOTHESIS SUPPORT
==================
Overall Support Rate: {len(hypothesis_supports)}/{len(successful)} ({len(hypothesis_supports)/len(successful)*100:.0f}%)

By Category:
"""

for cat in CATEGORY_ORDER:
    cat_data = df[df['category'] == cat]
    support = cat_data['hypothesis_support'].sum()
    total = len(cat_data)
    report += f"  {cat.capitalize():12s}: {int(support)}/{total} ({int(support)/total*100:.0f}%)\n"

report += f"""
NODE ACTIVATION SUMMARY (Mean ± SD)
====================================
"""

for cat in CATEGORY_ORDER:
    cat_data = df[df['category'] == cat]
    report += f"\n{cat.upper()}:\n"
    for node in NODE_ORDER:
        mean = cat_data[node].mean()
        std = cat_data[node].std()
        report += f"  {node:15s}: {mean:.4f} ± {std:.4f}\n"

with open(f'{output_dir}/summary_report.txt', 'w') as f:
    f.write(report)

print(f"  Saved: {output_dir}/summary_report.txt")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print(f"\nFigures generated in: {output_dir}/")
print(f"  - fig1_node_activation_heatmap.png")
print(f"  - fig2_category_node_activations.png")
print(f"  - fig3_stream_activations.png")
print(f"  - fig4_hypothesis_support.png")
print(f"  - fig5_dominant_node_distribution.png")
print(f"  - fig6_inference_time.png")
print(f"  - fig7_coactivation_matrix.png")
print(f"  - fig8_edge_weights.png")
print(f"  - statistical_summary.csv")
print(f"  - summary_report.txt")
print(f"\nReady for journal article production!")
