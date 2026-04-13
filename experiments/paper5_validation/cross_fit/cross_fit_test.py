#!/usr/bin/env python3
"""
Cross-Fit Falsification Test
Determines whether sigmoid parameters are system-specific or generic
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

OUTPUT_DIR = '/home/student/sgp-tribe3/experiments/paper5_validation/cross_fit/'

systems = ['TRIBE', 'Hierarchical', 'Correlated', 'Sparse', 'CurvedManifold']
n_systems = len(systems)

# Load parameters from previous test
params_df = pd.read_csv('/home/student/sgp-tribe3/experiments/paper5_validation/parameter_test/parameters.csv')
print(params_df)

# Extract parameters
X = params_df[['A', 'k0', 'beta']].values

# True labels (Sparse is different)
true_labels = np.array([0, 0, 0, 1, 0])

print("="*70)
print("CROSS-FIT FALSIFICATION TEST")
print("="*70)

# ============================================================
# STEP 1: PARAMETER SWAP GENERATION
# ============================================================
print("\n" + "="*70)
print("STEP 1: PARAMETER SWAP GENERATION")
print("="*70)

swap_data = []
for i, source in enumerate(systems):
    for j, target in enumerate(systems):
        if i != j:  # Only mismatched pairs
            swap_data.append({
                'source_system': source,
                'target_system': target,
                'target_label': true_labels[j],
                'A': X[i, 0],
                'k0': X[i, 1],
                'beta': X[i, 2]
            })

swap_df = pd.DataFrame(swap_data)
swap_df.to_csv(OUTPUT_DIR + 'parameter_swaps.csv', index=False)
print(f"\nGenerated {len(swap_df)} cross-fit pairs")
print(f"Sample:")
print(swap_df.head(10).to_string(index=False))
print(f"\nSaved: parameter_swaps.csv")

# ============================================================
# STEP 2: CLASSIFICATION TEST
# ============================================================
print("\n" + "="*70)
print("STEP 2: CROSS-FIT CLASSIFICATION TEST")
print("="*70)

# Standardize training data
scaler = StandardScaler()
X_train = scaler.fit_transform(X)
X_test_swapped = scaler.transform(swap_df[['A', 'k0', 'beta']].values)

# Train classifier on original data
clf = KNeighborsClassifier(n_neighbors=2)
clf.fit(X_train, true_labels)

# Test on original (correct) pairs
preds_correct, probs_correct = [], []
for i in range(n_systems):
    pred = clf.predict(X_train[i].reshape(1, -1))[0]
    preds_correct.append(pred)
correct_accuracy = np.mean([p == t for p, t in zip(preds_correct, true_labels)])

# Test on swapped pairs
preds_swapped = clf.predict(X_test_swapped)
swap_df['predicted_label'] = preds_swapped
swap_df['correct'] = (preds_swapped == swap_df['target_label']).astype(int)
swapped_accuracy = swap_df['correct'].mean()

print(f"\nOriginal classification accuracy: {correct_accuracy:.1%}")
print(f"Cross-fit classification accuracy: {swapped_accuracy:.1%}")

# Confusion matrix for cross-fit
from collections import Counter
confusion = swap_df.groupby(['target_system', 'predicted_label']).size().unstack(fill_value=0)
print(f"\nCross-fit confusion:")
print(confusion)

# Save classification results
cross_fit_results = swap_df[['source_system', 'target_system', 'target_label', 'predicted_label', 'correct']]
cross_fit_results.to_csv(OUTPUT_DIR + 'cross_fit_classification.csv', index=False)
print(f"\nSaved: cross_fit_classification.csv")

# ============================================================
# STEP 3: PERFORMANCE COMPARISON
# ============================================================
print("\n" + "="*70)
print("STEP 3: PERFORMANCE COMPARISON")
print("="*70)

comparison_df = pd.DataFrame({
    'Condition': ['Original (correct pairs)', 'Cross-fit (swapped pairs)', 'Chance'],
    'Accuracy': [correct_accuracy, swapped_accuracy, 0.2]
})
comparison_df.to_csv(OUTPUT_DIR + 'accuracy_comparison.csv', index=False)
print(comparison_df.to_string(index=False))
print(f"\nSaved: accuracy_comparison.csv")

# ============================================================
# STEP 4: DISTANCE ANALYSIS
# ============================================================
print("\n" + "="*70)
print("STEP 4: DISTANCE ANALYSIS")
print("="*70)

# Compute distances
distances = cdist(X_train, X_train, metric='euclidean')

# Within-system (diagonal) vs between-system distances
within_dists = np.diag(distances)
between_dists = distances[np.triu_indices(n_systems, k=1)]

print(f"\nWithin-system distances (diagonal):")
for i, s in enumerate(systems):
    print(f"  {s}: {within_dists[i]:.3f}")

print(f"\nBetween-system distances:")
print(f"  Mean: {np.mean(between_dists):.3f}")
print(f"  Std: {np.std(between_dists):.3f}")
print(f"  Min: {np.min(between_dists):.3f}")
print(f"  Max: {np.max(between_dists):.3f}")

# Analyze where swapped parameters fall
print(f"\nCross-fit analysis:")
for i, source in enumerate(systems):
    source_point = X_train[i]
    distances_to_all = np.linalg.norm(X_train - source_point, axis=1)
    nearest = np.argmin(distances_to_all)
    print(f"  {source}: nearest neighbor = {systems[nearest]} (dist={distances_to_all[nearest]:.3f})")

# Save distance analysis
distance_data = []
for i, s in enumerate(systems):
    distance_data.append({
        'system': s,
        'within_distance': within_dists[i],
        'mean_to_others': np.mean(distances[i, :])
    })
for i in range(n_systems):
    for j in range(i+1, n_systems):
        distance_data.append({
            'system': f'{systems[i]}-{systems[j]}',
            'within_distance': np.nan,
            'mean_to_others': distances[i, j]
        })

dist_df = pd.DataFrame(distance_data)
dist_df.to_csv(OUTPUT_DIR + 'distance_analysis.csv', index=False)
print(f"\nSaved: distance_analysis.csv")

# ============================================================
# STEP 5: DECISION BOUNDARY VISUALIZATION
# ============================================================
print("\n" + "="*70)
print("STEP 5: DECISION BOUNDARY VISUALIZATION")
print("="*70)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Color mapping
colors = ['steelblue', 'coral', 'green', 'purple', 'orange']
system_labels = ['TRIBE', 'Hier.', 'Corr.', 'Sparse', 'Curve']

# A vs k0
ax = axes[0]
# Plot decision regions
h = 0.05
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel(), X_train[:, 2].mean() * np.ones_like(xx.ravel())])
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, alpha=0.3)

# Plot original points
for i, s in enumerate(systems):
    ax.scatter(X_train[i, 0], X_train[i, 1], c=colors[i], s=200, label=system_labels[i], 
               edgecolors='black', linewidths=2, marker='o')
    ax.annotate(system_labels[i], (X_train[i, 0], X_train[i, 1]), 
                xytext=(5, 5), textcoords='offset points', fontsize=8)

ax.set_xlabel('A (amplitude)')
ax.set_ylabel('k0 (midpoint)')
ax.set_title('A vs k0')
ax.legend(fontsize=7)

# A vs beta
ax = axes[1]
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(-1, 3, h))
Z = clf.predict(np.c_[xx.ravel(), X_train[:, 1].mean() * np.ones_like(xx.ravel()), yy.ravel()])
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, alpha=0.3)

for i, s in enumerate(systems):
    ax.scatter(X_train[i, 0], X_train[i, 2], c=colors[i], s=200, label=system_labels[i],
               edgecolors='black', linewidths=2, marker='o')

ax.set_xlabel('A (amplitude)')
ax.set_ylabel('beta (slope)')
ax.set_title('A vs beta')
ax.legend(fontsize=7)

# k0 vs beta
ax = axes[2]
xx, yy = np.meshgrid(np.arange(-1, 3, h), np.arange(-1, 3, h))
Z = clf.predict(np.c_[X_train[:, 0].mean() * np.ones_like(xx.ravel()), xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, alpha=0.3)

for i, s in enumerate(systems):
    ax.scatter(X_train[i, 1], X_train[i, 2], c=colors[i], s=200, label=system_labels[i],
               edgecolors='black', linewidths=2, marker='o')

ax.set_xlabel('k0 (midpoint)')
ax.set_ylabel('beta (slope)')
ax.set_title('k0 vs beta')
ax.legend(fontsize=7)

plt.suptitle('Decision Boundaries in Parameter Space\n(Cross-Fit Test)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'plots/decision_boundary.pdf', dpi=150)
plt.close()
print("Saved: plots/decision_boundary.pdf")

# ============================================================
# STEP 6: CONFUSION STRUCTURE
# ============================================================
print("\n" + "="*70)
print("STEP 6: CONFUSION STRUCTURE")
print("="*70)

# Analyze misclassification patterns
misclassified = swap_df[swap_df['correct'] == 0]
classified = swap_df[swap_df['correct'] == 1]

print(f"\nMisclassification rate: {1-swapped_accuracy:.1%}")
print(f"\nMisclassified pairs: {len(misclassified)}")

if len(misclassified) > 0:
    print(f"\nMisclassification patterns:")
    confusion_pairs = misclassified.groupby(['target_system', 'predicted_label']).size()
    print(confusion_pairs)
    
    # Which system is Sparse confused with?
    sparse_confused = misclassified[misclassified['target_system'] == 'Sparse']
    print(f"\nSparse misclassifications:")
    print(sparse_confused[['source_system', 'predicted_label']])
    
    # Which system do others confuse Sparse with?
    sparse_as_target = misclassified[misclassified['predicted_label'] == true_labels[systems.index('Sparse')]]
    print(f"\nSystems confused as Sparse:")
    print(sparse_as_target[['source_system', 'target_system']])

# Confusion matrix
confusion_matrix = np.zeros((n_systems, n_systems))
for _, row in swap_df.iterrows():
    target_idx = systems.index(row['target_system'])
    pred_idx = int(row['predicted_label'])
    confusion_matrix[target_idx, pred_idx] += 1

conf_df = pd.DataFrame(confusion_matrix, index=systems, columns=systems)
conf_df.to_csv(OUTPUT_DIR + 'confusion_analysis.csv')
print(f"\nSaved: confusion_analysis.csv")

# ============================================================
# STEP 7: SUMMARY TABLE
# ============================================================
print("\n" + "="*70)
print("STEP 7: SUMMARY TABLE")
print("="*70)

summary_df = pd.DataFrame({
    'Test': ['Original classification', 'Cross-fit classification', 'Chance'],
    'Accuracy': [correct_accuracy, swapped_accuracy, 0.2],
    'Description': [
        'Parameters matched to correct system',
        'Parameters from one system used for another',
        'Random guess'
    ]
})
summary_df.to_csv(OUTPUT_DIR + 'summary.csv', index=False)
print(summary_df.to_string(index=False))
print(f"\nSaved: summary.csv")

# ============================================================
# STEP 8: INTERPRETATION
# ============================================================
print("\n" + "="*70)
print("STEP 8: INTERPRETATION")
print("="*70)

# Determine result
if swapped_accuracy < 0.3:
    if correct_accuracy > 0.6:
        classification = "Parameters are system-specific (non-transferable)"
    else:
        classification = "Parameters are generic (transferable)"
elif swapped_accuracy > 0.6:
    classification = "Parameters are generic (transferable)"
else:
    classification = "Parameters partially transferable"

# Error analysis
error_rate = 1 - swapped_accuracy
structured_errors = misclassified.groupby('target_system').size().std() if len(misclassified) > 0 else 0

print(f"\nKey Findings:")
print(f"  Original accuracy: {correct_accuracy:.1%}")
print(f"  Cross-fit accuracy: {swapped_accuracy:.1%}")
print(f"  Error rate: {error_rate:.1%}")
print(f"  Structured vs random errors: {structured_errors:.3f}")

print(f"\nFinal Classification: {classification}")

# Write interpretation
with open(OUTPUT_DIR + 'interpretation.txt', 'w') as f:
    f.write("CROSS-FIT FALSIFICATION TEST - INTERPRETATION\n")
    f.write("="*60 + "\n\n")
    
    f.write("1. DOES CROSS-FIT DESTROY CLASSIFICATION?\n")
    f.write("-"*40 + "\n")
    f.write(f"   Original (correct pairs): {correct_accuracy:.1%}\n")
    f.write(f"   Cross-fit (swapped pairs): {swapped_accuracy:.1%}\n")
    f.write(f"   Drop: {(correct_accuracy - swapped_accuracy)*100:.0f}%\n")
    if swapped_accuracy < 0.3:
        f.write("   → YES, cross-fit destroys classification\n\n")
    else:
        f.write("   → NO, cross-fit maintains classification\n\n")
    
    f.write("2. ARE SIGMOID PARAMETERS SYSTEM-SPECIFIC?\n")
    f.write("-"*40 + "\n")
    if swapped_accuracy < 0.3:
        f.write("   Parameters learned from one system CANNOT classify another.\n")
        f.write("   → Parameters are SYSTEM-SPECIFIC\n\n")
    else:
        f.write("   Parameters learned from one system CAN classify another.\n")
        f.write("   → Parameters are GENERIC / TRANSFERABLE\n\n")
    
    f.write("3. ARE ERRORS RANDOM OR STRUCTURED?\n")
    f.write("-"*40 + "\n")
    if len(misclassified) > 0:
        f.write(f"   Misclassification rate: {error_rate:.1%}\n")
        f.write(f"   Error structure std: {structured_errors:.3f}\n")
        if structured_errors < 1:
            f.write("   → Errors appear RANDOM\n")
        else:
            f.write("   → Errors are STRUCTURED\n")
    else:
        f.write("   No misclassifications.\n")
    f.write("\n")
    
    f.write("4. WHAT DOES DISTANCE ANALYSIS SHOW?\n")
    f.write("-"*40 + "\n")
    f.write(f"   Mean within-system distance: {np.mean(within_dists):.3f}\n")
    f.write(f"   Mean between-system distance: {np.mean(between_dists):.3f}\n")
    if np.mean(between_dists) > np.mean(within_dists) * 2:
        f.write("   → Systems are well-separated in parameter space\n")
    else:
        f.write("   → Systems overlap in parameter space\n")
    f.write("\n")
    
    f.write("5. FINAL CLASSIFICATION\n")
    f.write("-"*40 + "\n")
    f.write(f"   {classification}\n\n")
    
    f.write("6. SUMMARY\n")
    f.write("-"*40 + "\n")
    if swapped_accuracy < 0.3:
        f.write("   Cross-fit accuracy drops to near chance level.\n")
        f.write("   This indicates that sigmoid parameters encode\n")
        f.write("   SYSTEM-SPECIFIC information that cannot be\n")
        f.write("   transferred between systems.\n\n")
        f.write("   The classification depends on WHERE the parameters\n")
        f.write("   came from, not just their VALUES.\n")
    else:
        f.write("   Cross-fit accuracy remains high.\n")
        f.write("   This indicates parameters are GENERIC and can\n")
        f.write("   be transferred between systems.\n")

print(f"\nSaved: interpretation.txt")
print("\nANALYSIS COMPLETE")
