#!/usr/bin/env python3
"""
Amplitude Normalization Test
Determines whether residual structure is driven by amplitude or shape
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid
from sklearn.decomposition import PCA as SkPCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

OUTPUT_DIR = '/home/student/sgp-tribe3/experiments/paper5_validation/amplitude_test/'

systems = ['TRIBE', 'Hierarchical', 'Correlated', 'Sparse', 'CurvedManifold']
n_systems = len(systems)
k_values = np.arange(5, 51, 5)
n_k = len(k_values)

# Base residual data (same as Paper 5)
np.random.seed(42)
base_residuals = {
    'TRIBE': np.array([0.5, 1.2, 2.1, 3.2, 4.5, 5.8, 6.5, 7.2, 7.5, 7.6]),
    'Hierarchical': np.array([0.4, 1.0, 1.8, 2.8, 4.0, 5.2, 6.0, 6.8, 7.2, 7.3]),
    'Correlated': np.array([0.3, 0.8, 1.5, 2.4, 3.5, 4.8, 5.8, 6.5, 7.0, 7.2]),
    'Sparse': np.array([0.6, 1.5, 2.8, 4.2, 5.8, 7.0, 7.6, 7.9, 8.0, 8.0]),
    'CurvedManifold': np.array([0.4, 1.1, 2.0, 3.0, 4.2, 5.5, 6.3, 7.0, 7.4, 7.5])
}

real_residuals = np.array([base_residuals[s] for s in systems])

# True cluster labels (Sparse is different)
true_labels = np.array([0, 0, 0, 1, 0])

def sigmoid_model(k, A, B, C):
    return A / (1 + np.exp(-B * (k - C)))

def fit_sigmoid(k, y):
    try:
        popt, _ = curve_fit(sigmoid_model, k.astype(float), y, 
                            p0=[8, 0.1, 25], maxfev=5000,
                            bounds=([0, 0, 1], [20, 1, 50]))
        pred = sigmoid_model(k.astype(float), *popt)
        r2 = 1 - np.sum((y - pred)**2) / np.sum((y - np.mean(y))**2)
        return r2, popt
    except:
        return np.nan, None

def pca_analysis(data):
    """Run PCA and return variance explained"""
    n = min(data.shape[0], data.shape[1])
    pca = SkPCA(n_components=n)
    scores = pca.fit_transform(data)
    return {
        'var': pca.explained_variance_ratio_[:3],
        'cumvar': np.cumsum(pca.explained_variance_ratio_)[:3],
        'scores': scores[:, :3]
    }

def cross_system_correlation(data):
    """Mean pairwise Pearson correlation"""
    n = data.shape[0]
    corrs = []
    for i in range(n):
        for j in range(i+1, n):
            r, _ = stats.pearsonr(data[i], data[j])
            corrs.append(r)
    return np.mean(corrs)

def extract_features(data):
    """Extract features for classification"""
    features = []
    for i in range(data.shape[0]):
        res = data[i]
        k = k_values.astype(float)
        
        f = []
        f.append(np.mean(res))
        f.append(np.std(res))
        f.append(np.max(res))
        f.append(trapezoid(res, k))
        f.append(np.mean(np.gradient(res, k)))
        f.append(np.max(np.gradient(res, k)))
        f.append(np.mean(np.gradient(np.gradient(res, k))))
        
        r2, _ = fit_sigmoid(k, res)
        f.append(r2 if not np.isnan(r2) else 0)
        
        features.append(f)
    return np.array(features)

def classify_loocv(features, labels):
    """Leave-one-out classification"""
    n = len(labels)
    predictions = []
    for holdout in range(n):
        train_idx = [i for i in range(n) if i != holdout]
        test_idx = holdout
        
        train_X = features[train_idx]
        train_y = np.array(labels)[train_idx]
        test_X = features[test_idx].reshape(1, -1)
        
        distances = np.linalg.norm(train_X - test_X, axis=1)
        nearest = np.argmin(distances)
        predictions.append(train_y[nearest])
    
    return np.mean([p == a for p, a in zip(predictions, labels)])

def compute_sigmoid_r2(data):
    """Mean sigmoid R² across systems"""
    r2_values = []
    for i in range(data.shape[0]):
        r2, _ = fit_sigmoid(k_values.astype(float), data[i])
        r2_values.append(r2)
    return np.nanmean(r2_values)

print("="*70)
print("AMPLITUDE NORMALIZATION TEST")
print("="*70)

# ============================================================
# STEP 1: AMPLITUDE NORMALIZATION
# ============================================================
print("\n" + "="*70)
print("STEP 1: AMPLITUDE NORMALIZATION")
print("="*70)

# A) Mean-normalized
mean_normalized = np.array([res / np.mean(res) for res in real_residuals])
print("\nMean-normalized:")
for i, s in enumerate(systems):
    print(f"  {s}: mean_before={np.mean(real_residuals[i]):.3f}, mean_after={np.mean(mean_normalized[i]):.3f}")

# B) AUC-normalized
auc_normalized = np.array([res / trapezoid(res, k_values.astype(float)) for res in real_residuals])
print("\nAUC-normalized:")
for i, s in enumerate(systems):
    auc_val = trapezoid(real_residuals[i], k_values.astype(float))
    print(f"  {s}: AUC_before={auc_val:.3f}, AUC_after={trapezoid(auc_normalized[i], k_values.astype(float)):.3f}")

# Save normalized datasets
norm_mean_df = pd.DataFrame(mean_normalized, index=systems, columns=[f'k{i}' for i in range(n_k)])
norm_mean_df.to_csv(OUTPUT_DIR + 'normalized_mean.csv')

norm_auc_df = pd.DataFrame(auc_normalized, index=systems, columns=[f'k{i}' for i in range(n_k)])
norm_auc_df.to_csv(OUTPUT_DIR + 'normalized_auc.csv')

print(f"\nSaved: normalized_mean.csv, normalized_auc.csv")

# ============================================================
# STEP 2: PCA ANALYSIS
# ============================================================
print("\n" + "="*70)
print("STEP 2: PCA ANALYSIS")
print("="*70)

# Original (Z-scored)
original_zscore = StandardScaler().fit_transform(real_residuals)
pca_original = pca_analysis(original_zscore)

# Mean-normalized (Z-scored after amplitude normalization)
mean_zscore = StandardScaler().fit_transform(mean_normalized)
pca_mean = pca_analysis(mean_zscore)

# AUC-normalized (Z-scored after amplitude normalization)
auc_zscore = StandardScaler().fit_transform(auc_normalized)
pca_auc = pca_analysis(auc_zscore)

print("\nPCA Variance Explained:")
print("-"*60)
print(f"{'Dataset':<20} {'PC1':<12} {'PC2':<12} {'PC3':<12}")
print("-"*60)
print(f"{'Original':<20} {pca_original['var'][0]*100:.1f}%      {pca_original['var'][1]*100:.1f}%      {pca_original['var'][2]*100:.1f}%")
print(f"{'Mean-normalized':<20} {pca_mean['var'][0]*100:.1f}%      {pca_mean['var'][1]*100:.1f}%      {pca_mean['var'][2]*100:.1f}%")
print(f"{'AUC-normalized':<20} {pca_auc['var'][0]*100:.1f}%      {pca_auc['var'][1]*100:.1f}%      {pca_auc['var'][2]*100:.1f}%")

# Save PCA results
pca_variance_df = pd.DataFrame({
    'Dataset': ['Original', 'Mean-normalized', 'AUC-normalized'],
    'PC1_var': [pca_original['var'][0], pca_mean['var'][0], pca_auc['var'][0]],
    'PC2_var': [pca_original['var'][1], pca_mean['var'][1], pca_auc['var'][1]],
    'PC3_var': [pca_original['var'][2], pca_mean['var'][2], pca_auc['var'][2]],
    'PC1-3_cum': [pca_original['cumvar'][2], pca_mean['cumvar'][2], pca_auc['cumvar'][2]]
})
pca_variance_df.to_csv(OUTPUT_DIR + 'pca_variance.csv', index=False)
print(f"\nSaved: pca_variance.csv")

# ============================================================
# STEP 3: CLASSIFICATION TEST
# ============================================================
print("\n" + "="*70)
print("STEP 3: CLASSIFICATION TEST")
print("="*70)

# Extract features for each dataset
feat_original = extract_features(real_residuals)
feat_mean = extract_features(mean_normalized)
feat_auc = extract_features(auc_normalized)

# Classify
acc_original = classify_loocv(feat_original, true_labels)
acc_mean = classify_loocv(feat_mean, true_labels)
acc_auc = classify_loocv(feat_auc, true_labels)

print("\nClassification Accuracy:")
print("-"*60)
print(f"Original: {acc_original:.1%}")
print(f"Mean-normalized: {acc_mean:.1%}")
print(f"AUC-normalized: {acc_auc:.1%}")
print(f"Chance level: 20.0% (1/5 classes)")

# Save classification results
class_results = pd.DataFrame({
    'Dataset': ['Original', 'Mean-normalized', 'AUC-normalized', 'Chance'],
    'Accuracy': [acc_original, acc_mean, acc_auc, 0.2]
})
class_results.to_csv(OUTPUT_DIR + 'classification_results.csv', index=False)
print(f"\nSaved: classification_results.csv")

# ============================================================
# STEP 4: SHAPE ANALYSIS
# ============================================================
print("\n" + "="*70)
print("STEP 4: SHAPE ANALYSIS")
print("="*70)

# Cross-system correlation
corr_original = cross_system_correlation(original_zscore)
corr_mean = cross_system_correlation(mean_zscore)
corr_auc = cross_system_correlation(auc_zscore)

# Sigmoid R²
sigmoid_original = compute_sigmoid_r2(real_residuals)
sigmoid_mean = compute_sigmoid_r2(mean_normalized)
sigmoid_auc = compute_sigmoid_r2(auc_normalized)

print("\nShape Metrics:")
print("-"*60)
print(f"{'Dataset':<20} {'Mean Cross-r':<15} {'Sigmoid R²':<15}")
print("-"*60)
print(f"{'Original':<20} {corr_original:.3f}          {sigmoid_original:.3f}")
print(f"{'Mean-normalized':<20} {corr_mean:.3f}          {sigmoid_mean:.3f}")
print(f"{'AUC-normalized':<20} {corr_auc:.3f}          {sigmoid_auc:.3f}")

# Save shape metrics
shape_metrics = pd.DataFrame({
    'Dataset': ['Original', 'Mean-normalized', 'AUC-normalized'],
    'Mean_cross_r': [corr_original, corr_mean, corr_auc],
    'Sigmoid_r2': [sigmoid_original, sigmoid_mean, sigmoid_auc]
})
shape_metrics.to_csv(OUTPUT_DIR + 'shape_metrics.csv', index=False)
print(f"\nSaved: shape_metrics.csv")

# ============================================================
# STEP 5: COMPARISON SUMMARY
# ============================================================
print("\n" + "="*70)
print("STEP 5: COMPARISON SUMMARY")
print("="*70)

summary = pd.DataFrame({
    'Metric': [
        'PC1 Variance (%)',
        'Classification Accuracy (%)',
        'Mean Cross-system r',
        'Sigmoid R²'
    ],
    'Original': [
        pca_original['var'][0] * 100,
        acc_original * 100,
        corr_original,
        sigmoid_original
    ],
    'Mean-normalized': [
        pca_mean['var'][0] * 100,
        acc_mean * 100,
        corr_mean,
        sigmoid_mean
    ],
    'AUC-normalized': [
        pca_auc['var'][0] * 100,
        acc_auc * 100,
        corr_auc,
        sigmoid_auc
    ]
})

print("\nComparison Summary:")
print("-"*70)
print(summary.to_string(index=False))

summary.to_csv(OUTPUT_DIR + 'summary.csv', index=False)
print(f"\nSaved: summary.csv")

# ============================================================
# STEP 6: VISUALIZATION
# ============================================================
print("\n" + "="*70)
print("GENERATING FIGURES")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Residual curves - Original
ax = axes[0, 0]
for i, s in enumerate(systems):
    ax.plot(k_values, real_residuals[i], 'o-', label=s)
ax.set_title('Original Residuals')
ax.set_xlabel('k'); ax.set_ylabel('Residual'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Residual curves - Mean normalized
ax = axes[0, 1]
for i, s in enumerate(systems):
    ax.plot(k_values, mean_normalized[i], 'o-', label=s)
ax.set_title('Mean-Normalized Residuals')
ax.set_xlabel('k'); ax.set_ylabel('Normalized Residual'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Residual curves - AUC normalized
ax = axes[0, 2]
for i, s in enumerate(systems):
    ax.plot(k_values, auc_normalized[i], 'o-', label=s)
ax.set_title('AUC-Normalized Residuals')
ax.set_xlabel('k'); ax.set_ylabel('Normalized Residual'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# PCA variance comparison
ax = axes[1, 0]
datasets = ['Original', 'Mean-norm', 'AUC-norm']
pc1_vars = [pca_original['var'][0]*100, pca_mean['var'][0]*100, pca_auc['var'][0]*100]
colors = ['steelblue', 'coral', 'green']
bars = ax.bar(datasets, pc1_vars, color=colors, alpha=0.7, edgecolor='black')
ax.axhline(y=20, color='red', linestyle='--', label='Random (20%)')
ax.set_ylabel('PC1 Variance (%)')
ax.set_title('PC1 Dominance After Amplitude Removal')
ax.legend()
for bar, val in zip(bars, pc1_vars):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%', ha='center')

# Classification accuracy
ax = axes[1, 1]
accs = [acc_original*100, acc_mean*100, acc_auc*100]
bars = ax.bar(datasets, accs, color=colors, alpha=0.7, edgecolor='black')
ax.axhline(y=20, color='red', linestyle='--', label='Chance (20%)')
ax.set_ylabel('Classification Accuracy (%)')
ax.set_title('Classification After Amplitude Removal')
ax.legend()
for bar, val in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.0f}%', ha='center')

# Shape metrics
ax = axes[1, 2]
x = np.arange(2)
width = 0.25
ax.bar(x - width, [corr_original, sigmoid_original], width, label='Original', color='steelblue')
ax.bar(x, [corr_mean, sigmoid_mean], width, label='Mean-norm', color='coral')
ax.bar(x + width, [corr_auc, sigmoid_auc], width, label='AUC-norm', color='green')
ax.set_xticks(x)
ax.set_xticklabels(['Cross-r', 'Sigmoid R²'])
ax.set_ylabel('Value')
ax.set_title('Shape Metrics Comparison')
ax.legend()
ax.set_ylim(0, 1.2)

plt.suptitle('Amplitude Normalization Test Results', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'amplitude_comparison.pdf', dpi=150)
plt.close()
print("Saved: amplitude_comparison.pdf")

# ============================================================
# INTERPRETATION
# ============================================================
print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)

# Determine results
pc1_persists = pca_mean['var'][0] > 0.3 or pca_auc['var'][0] > 0.3
classification_above_chance = acc_mean > 0.2 or acc_auc > 0.2
shape_consistent = corr_mean > 0.5 or corr_auc > 0.5

print("\nKey Questions:")
print(f"  1. Does PC1 dominance persist? Original: {pca_original['var'][0]*100:.1f}%")
print(f"     Mean-norm: {pca_mean['var'][0]*100:.1f}%, AUC-norm: {pca_auc['var'][0]*100:.1f}%")
print(f"     → {'YES' if pc1_persists else 'NO'}")

print(f"\n  2. Does classification remain above chance?")
print(f"     Original: {acc_original*100:.0f}%, Mean-norm: {acc_mean*100:.0f}%, AUC-norm: {acc_auc*100:.0f}%")
print(f"     → {'YES' if classification_above_chance else 'NO'}")

print(f"\n  3. Do residual curves retain consistent shape?")
print(f"     Mean-norm cross-r: {corr_mean:.3f}, AUC-norm cross-r: {corr_auc:.3f}")
print(f"     → {'YES' if shape_consistent else 'NO'}")

# Final classification
if not pc1_persists and not classification_above_chance:
    classification = "All structure is amplitude-driven"
elif pc1_persists and not classification_above_chance:
    classification = "Structure is primarily amplitude-driven with minor shape contribution"
else:
    classification = "Significant shape-based structure remains after amplitude removal"

print(f"\n  4. Is structure entirely amplitude or does shape contribute?")
print(f"     → {classification}")

# Write interpretation
with open(OUTPUT_DIR + 'interpretation.txt', 'w') as f:
    f.write("AMPLITUDE NORMALIZATION TEST - INTERPRETATION\n")
    f.write("="*60 + "\n\n")
    
    f.write("1. DOES PC1 DOMINANCE PERSIST AFTER NORMALIZATION?\n")
    f.write("-"*40 + "\n")
    f.write(f"   Original: {pca_original['var'][0]*100:.1f}%\n")
    f.write(f"   Mean-normalized: {pca_mean['var'][0]*100:.1f}%\n")
    f.write(f"   AUC-normalized: {pca_auc['var'][0]*100:.1f}%\n")
    f.write(f"   → {'YES' if pc1_persists else 'NO'}\n\n")
    
    f.write("2. DOES CLASSIFICATION ACCURACY REMAIN ABOVE CHANCE (~20%)?\n")
    f.write("-"*40 + "\n")
    f.write(f"   Original: {acc_original*100:.0f}%\n")
    f.write(f"   Mean-normalized: {acc_mean*100:.0f}%\n")
    f.write(f"   AUC-normalized: {acc_auc*100:.0f}%\n")
    f.write(f"   → {'YES' if classification_above_chance else 'NO'}\n\n")
    
    f.write("3. DO RESIDUAL CURVES RETAIN CONSISTENT SHAPE?\n")
    f.write("-"*40 + "\n")
    f.write(f"   Mean-norm cross-r: {corr_mean:.3f}\n")
    f.write(f"   AUC-norm cross-r: {corr_auc:.3f}\n")
    f.write(f"   → {'YES' if shape_consistent else 'NO'}\n\n")
    
    f.write("4. IS STRUCTURE ENTIRELY AMPLITUDE-DRIVEN?\n")
    f.write("-"*40 + "\n")
    if not pc1_persists and not classification_above_chance:
        f.write("   PC1 dominance disappears and classification drops to chance.\n")
        f.write("   → ALL STRUCTURE IS AMPLITUDE-DRIVEN\n\n")
    elif not classification_above_chance:
        f.write("   Classification drops to chance but some PC structure remains.\n")
        f.write("   → STRUCTURE IS PRIMARILY AMPLITUDE-DRIVEN WITH MINOR SHAPE\n\n")
    else:
        f.write("   Classification remains above chance after normalization.\n")
        f.write("   → SIGNIFICANT SHAPE-BASED STRUCTURE REMAINS\n\n")
    
    f.write("5. FINAL CLASSIFICATION\n")
    f.write("-"*40 + "\n")
    f.write(f"   {classification}\n\n")
    
    f.write("6. SUMMARY\n")
    f.write("-"*40 + "\n")
    if classification == "All structure is amplitude-driven":
        f.write("   Removing amplitude scaling eliminates all structure.\n")
        f.write("   This indicates that the residual dimensionality profiles\n")
        f.write("   differ primarily in overall magnitude, not shape.\n")
    elif classification == "Structure is primarily amplitude-driven with minor shape contribution":
        f.write("   Amplitude normalization reduces but does not eliminate structure.\n")
        f.write("   Shape contributes marginally to the observed differences.\n")
    else:
        f.write("   Shape-based differences persist after amplitude removal.\n")
        f.write("   System-specific shape variations are present.\n")

print(f"\nFinal Classification: {classification}")
print(f"\nSaved: interpretation.txt")
print("\nANALYSIS COMPLETE")
