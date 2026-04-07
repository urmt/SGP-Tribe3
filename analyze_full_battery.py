"""
SGP-Tribe3 - Full 1000-Stimulus Analysis
==========================================
Analyzes the 1000-stimulus battery results and generates publication figures.

Usage:
    python analyze_full_battery.py

Outputs (all NEW):
    - results/full_battery_1000/figures/*.png
    - results/full_battery_1000/statistical_summary.csv
    - results/full_battery_1000/summary_report.txt
    - manuscript_v2.md
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
from scipy import stats
from datetime import datetime

# Find results
results_files = sorted(glob.glob("results/full_battery_1000/checkpoint_*.json"))
if not results_files:
    print("ERROR: No checkpoint files found in results/full_battery_1000/")
    sys.exit(1)

# Load all checkpoints and merge
all_results = []
for f in results_files:
    with open(f, 'r') as fp:
        data = json.load(fp)
    all_results.extend(data.get('results', []))

# Deduplicate by stimulus_id
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
CATEGORY_ORDER = ['simple', 'logical', 'emotional', 'factual', 'abstract',
                  'spatial', 'social', 'motor', 'memory', 'auditory']

for node in NODE_ORDER:
    df[node] = df['sgp_nodes'].apply(lambda x: x.get(node, 0) if isinstance(x, dict) else 0)

output_dir = "results/full_battery_1000/figures"
os.makedirs(output_dir, exist_ok=True)

# Figure 1: Heatmap
print("Generating Figure 1: Node Activation Heatmap...")
heatmap_data = pd.DataFrame()
for cat in CATEGORY_ORDER:
    cat_data = df[df['category'] == cat]
    for node in NODE_ORDER:
        heatmap_data.loc[cat, node] = cat_data[node].mean()

fig, ax = plt.subplots(figsize=(14, 7))
im = ax.imshow(heatmap_data.values, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(len(NODE_ORDER)))
ax.set_xticklabels(NODE_ORDER, rotation=45, ha='right', fontsize=10)
ax.set_yticks(range(len(CATEGORY_ORDER)))
ax.set_yticklabels([c.capitalize() for c in CATEGORY_ORDER], fontsize=10)
for i in range(len(CATEGORY_ORDER)):
    for j in range(len(NODE_ORDER)):
        val = heatmap_data.iloc[i, j]
        ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=8,
                color='black' if val < 0.7 else 'white')
ax.set_xlabel('SGP Node', fontsize=12, fontweight='bold')
ax.set_ylabel('Stimulus Category', fontsize=12, fontweight='bold')
ax.set_title(f'SGP Node Activation by Category (n={len(df)})', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='Mean Activation')
plt.tight_layout()
plt.savefig(f'{output_dir}/fig1_node_activation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 2: Hypothesis support
print("Generating Figure 2: Hypothesis Support...")
hypothesis_support = []
for cat in CATEGORY_ORDER:
    cat_data = df[df['category'] == cat]
    support_rate = cat_data['hypothesis_support'].mean() * 100
    hypothesis_support.append(support_rate)

fig, ax = plt.subplots(figsize=(12, 6))
colors = ['green' if x >= 50 else 'orange' if x >= 25 else 'red' for x in hypothesis_support]
bars = ax.bar([c.capitalize() for c in CATEGORY_ORDER], hypothesis_support, color=colors, alpha=0.8, edgecolor='black')
for bar, val in zip(bars, hypothesis_support):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.0f}%',
            ha='center', va='bottom', fontweight='bold', fontsize=10)
overall = df['hypothesis_support'].mean() * 100
ax.axhline(y=overall, color='gray', linestyle='--', alpha=0.7, label=f'Overall: {overall:.0f}%')
ax.set_xlabel('Stimulus Category', fontsize=12, fontweight='bold')
ax.set_ylabel('Hypothesis Support Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('Hickok-Poeppel Dual-Stream Hypothesis Support Rate (1000 Stimuli)', fontsize=14, fontweight='bold')
ax.set_ylim(0, 120)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f'{output_dir}/fig2_hypothesis_support.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 3: Co-activation
print("Generating Figure 3: Co-activation Matrix...")
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
ax.set_title('SGP Node Co-activation Matrix (1000 Stimuli)', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='Pearson Correlation')
plt.tight_layout()
plt.savefig(f'{output_dir}/fig3_coactivation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 4: Dominant node distribution
print("Generating Figure 4: Dominant Node Distribution...")
top1_counts = {node: 0 for node in NODE_ORDER}
for _, row in df.iterrows():
    if 'sgp_nodes' in row and isinstance(row['sgp_nodes'], dict):
        sorted_nodes = sorted([(n, row[n]) for n in NODE_ORDER], key=lambda x: x[1], reverse=True)
        top1_counts[sorted_nodes[0][0]] += 1

fig, ax = plt.subplots(figsize=(12, 6))
colors = plt.cm.Set3(np.linspace(0, 1, len(NODE_ORDER)))
ax.bar(NODE_ORDER, [top1_counts[n] for n in NODE_ORDER], color=colors, alpha=0.8, edgecolor='black')
ax.set_title('Most Dominant SGP Node (1000 Stimuli)', fontsize=14, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
ax.set_ylabel('Count', fontsize=11)
plt.tight_layout()
plt.savefig(f'{output_dir}/fig4_dominant_node_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 5: Category-wise bar charts
print("Generating Figure 5: Category-wise Node Activations...")
fig, axes = plt.subplots(2, 5, figsize=(22, 8))
axes = axes.flatten()
for idx, cat in enumerate(CATEGORY_ORDER):
    cat_data = df[df['category'] == cat]
    means = [cat_data[node].mean() for node in NODE_ORDER]
    stds = [cat_data[node].std() for node in NODE_ORDER]
    ax = axes[idx]
    colors_bar = plt.cm.viridis(np.linspace(0.2, 0.8, len(NODE_ORDER)))
    bars = ax.bar(NODE_ORDER, means, yerr=stds, capsize=3, color=colors_bar, alpha=0.8)
    ax.set_title(f'{cat.capitalize()} (n={len(cat_data)})', fontsize=11, fontweight='bold')
    ax.set_ylim(0.85, 1.02)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    top2_idx = np.argsort(means)[-2:]
    for ti in top2_idx:
        bars[ti].set_edgecolor('red')
        bars[ti].set_linewidth(2)
plt.suptitle('SGP Node Activation by Category with Standard Deviation\nRed borders indicate top 2 dominant nodes', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{output_dir}/fig5_category_node_activations.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 6: Inference time
print("Generating Figure 6: Inference Time...")
fig, ax = plt.subplots(figsize=(10, 6))
cat_times = df.groupby('category')['inference_time_seconds'].agg(['mean', 'std'])
cat_times = cat_times.reindex(CATEGORY_ORDER)
ax.bar([c.capitalize() for c in cat_times.index], cat_times['mean'],
       yerr=cat_times['std'], capsize=5, color='steelblue', alpha=0.8, edgecolor='black')
ax.set_title('Mean Inference Time by Category', fontsize=13, fontweight='bold')
ax.set_ylabel('Time (seconds)', fontsize=11)
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig(f'{output_dir}/fig6_inference_time.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 7: Stream activation
print("Generating Figure 7: Stream Activation...")
STREAM_ORDER = ['dorsal', 'ventral', 'generative', 'modulatory', 'convergence']
stream_data = pd.DataFrame()
for cat in CATEGORY_ORDER:
    cat_data = df[df['category'] == cat]
    for stream in STREAM_ORDER:
        stream_data.loc[cat, stream] = cat_data['streams'].apply(lambda x: x.get(stream, 0) if isinstance(x, dict) else 0).mean()

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
plt.savefig(f'{output_dir}/fig7_stream_activations.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 8: Edge weights
print("Generating Figure 8: Edge Weights...")
edge_names = ['AF', 'SLF', 'IFOF', 'ILF', 'UF', 'CG_exec', 'CG_dmn', 'CC', 'MdLF']
edge_data = pd.DataFrame()
for cat in CATEGORY_ORDER:
    cat_data = df[df['category'] == cat]
    for edge in edge_names:
        edge_data.loc[cat, edge] = cat_data['edge_weights'].apply(lambda x: x.get(edge, 0) if isinstance(x, dict) else 0).mean()

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(CATEGORY_ORDER))
width = 0.08
colors_edge = plt.cm.tab10(np.linspace(0, 1, len(edge_names)))
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

# Statistical summary
print("Generating statistical summary...")
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
    sorted_nodes = sorted([(node, cat_data[node].mean()) for node in NODE_ORDER], key=lambda x: x[1], reverse=True)
    row['Top_Node_1'] = sorted_nodes[0][0]
    row['Top_Node_1_Activation'] = round(sorted_nodes[0][1], 4)
    row['Top_Node_2'] = sorted_nodes[1][0]
    row['Top_Node_2_Activation'] = round(sorted_nodes[1][1], 4)
    summary_rows.append(row)

pd.DataFrame(summary_rows).to_csv(f'{output_dir}/statistical_summary.csv', index=False)

# Summary report
successful = df[~df.get('error', pd.Series([False]*len(df))).astype(bool)]
hypothesis_supports = successful[successful['hypothesis_support']]

report = f"""
SGP-Tribe3 Full Research Battery - Statistical Summary
=======================================================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Stimuli: {len(df)}
Successful: {len(successful)}
Failed: {len(df) - len(successful)}
Overall Hypothesis Support: {len(hypothesis_supports)}/{len(successful)} ({len(hypothesis_supports)/len(successful)*100:.0f}%)

By Category:
"""
for cat in CATEGORY_ORDER:
    cat_data = df[df['category'] == cat]
    support = cat_data['hypothesis_support'].sum()
    total = len(cat_data)
    report += f"  {cat.capitalize():12s}: {int(support)}/{total} ({int(support)/total*100:.0f}%)\n"

report += "\n\nNODE ACTIVATION SUMMARY (Mean +/- SD)\n====================================\n"
for cat in CATEGORY_ORDER:
    cat_data = df[df['category'] == cat]
    report += f"\n{cat.upper()}:\n"
    for node in NODE_ORDER:
        mean = cat_data[node].mean()
        std = cat_data[node].std()
        report += f"  {node:15s}: {mean:.4f} +/- {std:.4f}\n"

with open(f'{output_dir}/summary_report.txt', 'w') as f:
    f.write(report)

print(f"\nAnalysis complete. Figures saved to: {output_dir}/")
print(f"  - fig1_node_activation_heatmap.png")
print(f"  - fig2_hypothesis_support.png")
print(f"  - fig3_coactivation_matrix.png")
print(f"  - fig4_dominant_node_distribution.png")
print(f"  - fig5_category_node_activations.png")
print(f"  - fig6_inference_time.png")
print(f"  - fig7_stream_activations.png")
print(f"  - fig8_edge_weights.png")
print(f"  - statistical_summary.csv")
print(f"  - summary_report.txt")
