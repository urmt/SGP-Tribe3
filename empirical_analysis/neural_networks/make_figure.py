#!/usr/bin/env python3
"""
SFH-SGP_FIGURE_01_ALPHA_DET_SCATTER
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("figure_data.csv")

plt.figure(figsize=(7, 6))

operators = ['identity', 'tanh']
colors = {'identity': 'blue', 'tanh': 'red'}
markers = {'identity': 'o', 'tanh': 's'}

for op in operators:
    subset = df[df['operator'] == op]
    layer_order = ['input', 'hidden_0', 'hidden_1', 'output']
    subset['layer_cat'] = pd.Categorical(subset['layer'], categories=layer_order, ordered=True)
    subset_sorted = subset.sort_values('layer_cat')
    
    plt.scatter(
        subset_sorted['alpha'],
        subset_sorted['det'],
        c=colors[op],
        marker=markers[op],
        s=120,
        label=op,
        alpha=0.85,
        edgecolors='black',
        linewidths=0.8
    )
    
    plt.plot(
        subset_sorted['alpha'],
        subset_sorted['det'],
        linestyle='--',
        alpha=0.5,
        color=colors[op],
        linewidth=1.5
    )
    
    for i, row in subset_sorted.iterrows():
        offset_x = 0.01
        offset_y = 0.02
        plt.annotate(
            row['layer'],
            (row['alpha'], row['det']),
            textcoords="offset points",
            xytext=(offset_x * 100, offset_y * 100),
            fontsize=9,
            fontweight='bold'
        )

plt.xlabel(r'$\alpha$ (Recurrence Scaling)', fontsize=12)
plt.ylabel('DET (Determinism)', fontsize=12)
plt.title(r'Operator-Dependent Geometry in $(\alpha, DET)$ Space', fontsize=14)
plt.legend(loc='upper right', fontsize=11)
plt.grid(True, alpha=0.3, linestyle=':')
plt.xlim(0.35, 1.05)
plt.ylim(-0.02, 0.92)
plt.tight_layout()

plt.savefig("figure1_alpha_det.png", dpi=300)
print("Figure saved to figure1_alpha_det.png")