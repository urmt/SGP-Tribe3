#!/usr/bin/env python3
"""
PC1 Interpretation Analysis
Determines what geometric properties PC1 of residual dimensionality profiles represents
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid
from sklearn.decomposition import PCA as SkPCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

OUTPUT_DIR = '/home/student/sgp-tribe3/experiments/paper5_validation/pc1_analysis/'

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
real_normalized = StandardScaler().fit_transform(real_residuals)

# Sigmoid model
def sigmoid_model(k, A, B, C):
    return A / (1 + np.exp(-B * (k - C)))

def fit_sigmoid_full(k, y):
    try:
        popt, _ = curve_fit(sigmoid_model, k.astype(float), y, 
                            p0=[8, 0.1, 25], maxfev=5000,
                            bounds=([0, 0, 1], [20, 1, 50]))
        pred = sigmoid_model(k.astype(float), *popt)
        r2 = 1 - np.sum((y - pred)**2) / np.sum((y - np.mean(y))**2)
        return popt, r2
    except:
        return None, None

print("="*70)
print("PC1 INTERPRETATION ANALYSIS")
print("="*70)

# ============================================================
# STEP 1: PCA RECOMPUTATION
# ============================================================
print("\n" + "="*70)
print("STEP 1: PCA RECOMPUTATION")
print("="*70)

pca = SkPCA(n_components=3)
scores = pca.fit_transform(real_normalized)
pc1_scores = scores[:, 0]
pc1_loadings = pca.components_[0]

print("\nPC1 Analysis:")
print(f"  PC1 variance explained: {pca.explained_variance_ratio_[0]*100:.1f}%")
print(f"  PC1-3 cumulative: {np.sum(pca.explained_variance_ratio_[:3])*100:.1f}%")

print("\nPC1 Scores (per system):")
for i, s in enumerate(systems):
    print(f"  {s}: {pc1_scores[i]:.3f}")

# Save PCA results
pca_df = pd.DataFrame({
    'system': systems,
    'PC1_score': pc1_scores,
    'PC2_score': scores[:, 1],
    'PC3_score': scores[:, 2]
})
pca_df.to_csv(OUTPUT_DIR + 'pca_results.csv', index=False)
print(f"\nSaved: pca_results.csv")

# ============================================================
# STEP 2: FEATURE EXTRACTION
# ============================================================
print("\n" + "="*70)
print("STEP 2: FEATURE EXTRACTION")
print("="*70)

features = []
for i, system in enumerate(systems):
    res = real_residuals[i]
    k = k_values.astype(float)
    
    f = {}
    f['system'] = system
    
    # Basic Features
    f['mean_residual'] = np.mean(res)
    f['std_residual'] = np.std(res)
    f['max_residual'] = np.max(res)
    f['min_residual'] = np.min(res)
    f['range_residual'] = np.max(res) - np.min(res)
    f['auc'] = trapezoid(res, k)  # Area under curve
    f['auc_normalized'] = trapezoid(res, k) / k[-1]  # Normalized AUC
    
    # Derivative Features
    first_deriv = np.gradient(res, k)
    second_deriv = np.gradient(first_deriv, k)
    
    f['mean_first_deriv'] = np.mean(first_deriv)
    f['max_first_deriv'] = np.max(first_deriv)
    f['min_first_deriv'] = np.min(first_deriv)
    f['slope_at_midpoint'] = first_deriv[len(first_deriv)//2]
    
    f['mean_second_deriv'] = np.mean(second_deriv)
    f['max_second_deriv'] = np.max(second_deriv)
    f['abs_mean_curvature'] = np.mean(np.abs(second_deriv))
    
    # Sigmoid Fit Parameters
    sigmoid_params, sigmoid_r2 = fit_sigmoid_full(k, res)
    if sigmoid_params is not None:
        f['sigmoid_A'] = sigmoid_params[0]  # Amplitude
        f['sigmoid_B'] = sigmoid_params[1]  # Steepness
        f['sigmoid_C'] = sigmoid_params[2]  # Midpoint
        f['sigmoid_r2'] = sigmoid_r2
    else:
        f['sigmoid_A'] = np.nan
        f['sigmoid_B'] = np.nan
        f['sigmoid_C'] = np.nan
        f['sigmoid_r2'] = np.nan
    
    # Additional metrics
    f['final_value'] = res[-1]
    f['initial_value'] = res[0]
    f['growth_amount'] = res[-1] - res[0]
    f['normalized_growth'] = (res[-1] - res[0]) / np.mean(res) if np.mean(res) > 0 else np.nan
    
    features.append(f)

feature_df = pd.DataFrame(features)
print("\nExtracted Features:")
print(feature_df.columns.tolist())
feature_df.to_csv(OUTPUT_DIR + 'features.csv', index=False)
print(f"\nSaved: features.csv")

print("\nFeature Summary:")
for col in feature_df.columns[1:]:
    print(f"  {col}: mean={feature_df[col].mean():.3f}, std={feature_df[col].std():.3f}")

# ============================================================
# STEP 3: CORRELATION ANALYSIS
# ============================================================
print("\n" + "="*70)
print("STEP 3: CORRELATION ANALYSIS")
print("="*70)

correlation_results = []
for col in feature_df.columns[1:]:
    vals = feature_df[col].values
    if not np.any(np.isnan(vals)) and np.std(vals) > 0:
        r, p = stats.pearsonr(pc1_scores, vals)
        correlation_results.append({
            'feature': col,
            'correlation': r,
            'abs_correlation': abs(r),
            'p_value': p,
            'significant': p < 0.05
        })

corr_df = pd.DataFrame(correlation_results).sort_values('abs_correlation', ascending=False)
print("\nCorrelation with PC1 (sorted by strength):")
print("-"*60)
print(f"{'Feature':<25} {'r':<10} {'p-value':<12} {'Significant':<12}")
print("-"*60)
for _, row in corr_df.iterrows():
    sig = "Yes*" if row['significant'] else "No"
    print(f"{row['feature']:<25} {row['correlation']:.3f}      {row['p_value']:.4f}      {sig}")

corr_df.to_csv(OUTPUT_DIR + 'correlations.csv', index=False)
print(f"\nSaved: correlations.csv")

# Top correlations
top5 = corr_df.head(5)
print("\nTop 5 Correlations:")
for _, row in top5.iterrows():
    print(f"  {row['feature']}: r={row['correlation']:.3f} (p={row['p_value']:.4f})")

# ============================================================
# STEP 4: VISUALIZATION
# ============================================================
print("\n" + "="*70)
print("STEP 4: VISUALIZATION")
print("="*70)

# Top 5 scatter plots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (_, row) in enumerate(top5.iterrows()):
    if idx < 6:
        feat = row['feature']
        vals = feature_df[feat].values
        
        ax = axes[idx]
        ax.scatter(pc1_scores, vals, s=100, c='steelblue', edgecolors='black')
        
        # Regression line
        slope, intercept, r_val, p_val, std_err = stats.linregress(pc1_scores, vals)
        x_line = np.linspace(pc1_scores.min(), pc1_scores.max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'r--', linewidth=2, label=f'r={r_val:.3f}')
        
        ax.set_xlabel('PC1 Score', fontsize=10)
        ax.set_ylabel(feat, fontsize=10)
        ax.set_title(f'{feat}\nr = {row["correlation"]:.3f}, p = {row["p_value"]:.4f}', fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3)

# Hide unused subplot
if len(top5) < 6:
    axes[-1].axis('off')

plt.suptitle('PC1 Score vs Top Correlated Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'plots/top_correlations.pdf', dpi=150)
plt.close()
print("Saved: plots/top_correlations.pdf")

# All correlations bar plot
fig, ax = plt.subplots(figsize=(12, 8))
sorted_df = corr_df.sort_values('correlation')
colors = ['coral' if v < 0 else 'steelblue' for v in sorted_df['correlation']]
bars = ax.barh(sorted_df['feature'], sorted_df['correlation'], color=colors, edgecolor='black')
ax.axvline(x=0, color='black', linewidth=1)
ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=-0.5, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Pearson Correlation with PC1', fontsize=12)
ax.set_title('All Feature Correlations with PC1', fontsize=14)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'plots/all_correlations.pdf', dpi=150)
plt.close()
print("Saved: plots/all_correlations.pdf")

# ============================================================
# STEP 5: LOADINGS INTERPRETATION
# ============================================================
print("\n" + "="*70)
print("STEP 5: LOADINGS INTERPRETATION")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# PC1 loadings over k
ax = axes[0, 0]
ax.plot(k_values, pc1_loadings, 'bo-', markersize=8, linewidth=2)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Neighborhood Scale (k)', fontsize=11)
ax.set_ylabel('PC1 Loading', fontsize=11)
ax.set_title('PC1 Loadings as Function of k', fontsize=12)
ax.grid(True, alpha=0.3)

# Uniformity test
loading_std = np.std(pc1_loadings)
loading_range = np.max(pc1_loadings) - np.min(pc1_loadings)
is_uniform = loading_range < 0.1  # Threshold for "uniform"
print(f"\nPC1 Loadings Analysis:")
print(f"  Loading range: {loading_range:.4f}")
print(f"  Loading std: {loading_std:.4f}")
print(f"  Uniform (range < 0.1): {'YES' if is_uniform else 'NO'}")

# Check if increasing
is_increasing = pc1_loadings[-1] > pc1_loadings[0]
is_centered = abs(np.mean(pc1_loadings)) < 0.05
print(f"  Increasing (end > start): {'YES' if is_increasing else 'NO'}")
print(f"  Centered at zero: {'YES' if is_centered else 'NO'}")

# Actual residual curves
ax = axes[0, 1]
for i, s in enumerate(systems):
    ax.plot(k_values, real_residuals[i], 'o-', label=s, markersize=6)
ax.set_xlabel('Neighborhood Scale (k)', fontsize=11)
ax.set_ylabel('Residual Dimensionality', fontsize=11)
ax.set_title('Actual Residual Curves', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# PC1 loadings interpretation summary
interpretation = []
if is_uniform:
    interpretation.append("UNIFORM: Suggests amplitude scaling")
if is_increasing:
    interpretation.append("INCREASING: Suggests growth/slope encoding")
if is_centered:
    interpretation.append("CENTERED: May indicate orthonormal basis")

ax = axes[1, 0]
ax.bar(range(len(interpretation)), [1]*len(interpretation), color='steelblue')
ax.set_xticks(range(len(interpretation)))
ax.set_xticklabels(interpretation, rotation=45, ha='right')
ax.set_ylabel('Evidence', fontsize=11)
ax.set_title('PC1 Loadings Interpretation', fontsize=12)
ax.set_ylim(0, 1.5)

# Correlation heatmap of features
ax = axes[1, 1]
feature_cols = ['mean_residual', 'std_residual', 'max_residual', 'auc', 
                'mean_first_deriv', 'sigmoid_A', 'sigmoid_B', 'sigmoid_C']
available_cols = [c for c in feature_cols if c in feature_df.columns]
sub_df = feature_df[available_cols].copy()
sub_df['PC1'] = pc1_scores
corr_matrix = sub_df.corr()

im = ax.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_matrix.columns)))
ax.set_yticks(range(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(corr_matrix.columns, fontsize=9)
plt.colorbar(im, ax=ax, label='Correlation')
ax.set_title('Feature Correlation Matrix', fontsize=12)

plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'pc1_loadings.pdf', dpi=150)
plt.close()
print(f"\nSaved: pc1_loadings.pdf")

# ============================================================
# STEP 6: CONTROL COMPARISON
# ============================================================
print("\n" + "="*70)
print("STEP 6: CONTROL COMPARISON")
print("="*70)

# Null-of-null residuals
np.random.seed(123)
null1 = np.random.randn(n_systems, n_k) * 2 + 3
np.random.seed(456)
null2 = np.random.randn(n_systems, n_k) * 2 + 3
null_residuals = null1 - null2
null_normalized = StandardScaler().fit_transform(null_residuals)

pca_null = SkPCA(n_components=3)
scores_null = pca_null.fit_transform(null_normalized)
pc1_null = scores_null[:, 0]

# Shuffled residuals
np.random.seed(42)
shuffled_residuals = np.array([np.random.permutation(r) for r in real_residuals])
shuffled_normalized = StandardScaler().fit_transform(shuffled_residuals)

pca_shuffled = SkPCA(n_components=3)
scores_shuffled = pca_shuffled.fit_transform(shuffled_normalized)
pc1_shuffled = scores_shuffled[:, 0]

# Compute correlations for controls
def compute_feature_correlations(pc_scores, residual_data, feature_names):
    """Compute correlations between PC1 and features"""
    corrs = []
    for name in feature_names:
        if name in ['system', 'sigmoid_A', 'sigmoid_B', 'sigmoid_C', 'sigmoid_r2']:
            continue
        vals = []
        for i in range(residual_data.shape[0]):
            res = residual_data[i]
            k = k_values.astype(float)
            
            if name == 'mean_residual':
                v = np.mean(res)
            elif name == 'std_residual':
                v = np.std(res)
            elif name == 'max_residual':
                v = np.max(res)
            elif name == 'auc':
                v = trapezoid(res, k)
            elif name == 'mean_first_deriv':
                v = np.mean(np.gradient(res, k))
            elif name == 'auc_normalized':
                v = trapezoid(res, k) / k[-1]
            else:
                continue
            vals.append(v)
        
        vals = np.array(vals)
        if np.std(vals) > 0 and not np.any(np.isnan(vals)):
            r, p = stats.pearsonr(pc_scores, vals)
            corrs.append({'feature': name, 'correlation': r, 'abs_correlation': abs(r)})
    return pd.DataFrame(corrs)

feature_names = ['mean_residual', 'std_residual', 'max_residual', 'auc', 'mean_first_deriv', 'auc_normalized']

corr_real = corr_df[corr_df['feature'].isin(feature_names)][['feature', 'correlation', 'abs_correlation']]
corr_null = compute_feature_correlations(pc1_null, null_residuals, feature_names)
corr_shuffled = compute_feature_correlations(pc1_shuffled, shuffled_residuals, feature_names)

print("\nControl Comparison:")
print("-"*70)
print(f"{'Feature':<25} {'Real r':<12} {'Null r':<12} {'Shuffled r':<12}")
print("-"*70)

comparison_data = []
for feat in feature_names:
    r_real = corr_real[corr_real['feature'] == feat]['correlation'].values
    r_null = corr_null[corr_null['feature'] == feat]['correlation'].values
    r_shuf = corr_shuffled[corr_shuffled['feature'] == feat]['correlation'].values
    
    r_real = r_real[0] if len(r_real) > 0 else np.nan
    r_null = r_null[0] if len(r_null) > 0 else np.nan
    r_shuf = r_shuf[0] if len(r_shuf) > 0 else np.nan
    
    print(f"{feat:<25} {r_real:.3f}       {r_null:.3f}       {r_shuf:.3f}")
    comparison_data.append({
        'feature': feat, 'real': r_real, 'null': r_null, 'shuffled': r_shuf
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv(OUTPUT_DIR + 'control_comparison.csv', index=False)
print(f"\nSaved: control_comparison.csv")

# Figure
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(feature_names))
width = 0.25

bars_real = ax.bar(x - width, comparison_df['real'].fillna(0), width, label='Real', color='steelblue')
bars_null = ax.bar(x, comparison_df['null'].fillna(0), width, label='Null', color='coral')
bars_shuf = ax.bar(x + width, comparison_df['shuffled'].fillna(0), width, label='Shuffled', color='green')

ax.set_xlabel('Feature', fontsize=11)
ax.set_ylabel('Correlation with PC1', fontsize=11)
ax.set_title('PC1 Correlation: Real vs Null vs Shuffled', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(feature_names, rotation=45, ha='right')
ax.legend()
ax.axhline(y=0, color='black', linewidth=0.5)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'control_comparison.pdf', dpi=150)
plt.close()
print("Saved: control_comparison.pdf")

# ============================================================
# FINAL INTERPRETATION
# ============================================================
print("\n" + "="*70)
print("FINAL INTERPRETATION")
print("="*70)

# Determine what PC1 best represents
top_feature = corr_df.iloc[0]['feature']
top_r = corr_df.iloc[0]['correlation']
top_p = corr_df.iloc[0]['p_value']

# Check feature categories
amplitude_features = ['mean_residual', 'max_residual', 'sigmoid_A', 'auc', 'range_residual']
slope_features = ['mean_first_deriv', 'max_first_deriv', 'growth_amount', 'slope_at_midpoint']
curvature_features = ['mean_second_deriv', 'abs_mean_curvature', 'std_residual']
sigmoid_features = ['sigmoid_A', 'sigmoid_B', 'sigmoid_C']

amplitude_corrs = corr_df[corr_df['feature'].isin(amplitude_features)]['abs_correlation']
slope_corrs = corr_df[corr_df['feature'].isin(slope_features)]['abs_correlation']
curvature_corrs = corr_df[corr_df['feature'].isin(curvature_features)]['abs_correlation']
sigmoid_corrs = corr_df[corr_df['feature'].isin(sigmoid_features)]['abs_correlation']

print("\nFeature Category Correlations:")
print(f"  Amplitude features: max r = {amplitude_corrs.max():.3f}" if len(amplitude_corrs) > 0 else "  Amplitude features: N/A")
print(f"  Slope features: max r = {slope_corrs.max():.3f}" if len(slope_corrs) > 0 else "  Slope features: N/A")
print(f"  Curvature features: max r = {curvature_corrs.max():.3f}" if len(curvature_corrs) > 0 else "  Curvature features: N/A")
print(f"  Sigmoid features: max r = {sigmoid_corrs.max():.3f}" if len(sigmoid_corrs) > 0 else "  Sigmoid features: N/A")

# Determine classification
categories = {
    'amplitude': amplitude_corrs.max() if len(amplitude_corrs) > 0 else 0,
    'slope': slope_corrs.max() if len(slope_corrs) > 0 else 0,
    'curvature': curvature_corrs.max() if len(curvature_corrs) > 0 else 0,
    'sigmoid': sigmoid_corrs.max() if len(sigmoid_corrs) > 0 else 0
}

best_category = max(categories, key=categories.get)
best_r = categories[best_category]

# Check if null/shuffle show similar patterns
null_similar = abs(comparison_df['null'].mean()) > 0.3
shuffled_similar = abs(comparison_df['shuffled'].mean()) > 0.3

# Classification
if best_r > 0.7 and not null_similar and not shuffled_similar:
    if best_category == 'amplitude':
        classification = "PC1 = amplitude scaling"
    elif best_category == 'sigmoid':
        sigmoid_best = corr_df[corr_df['feature'].str.contains('sigmoid')].iloc[0]['feature']
        classification = f"PC1 = sigmoid parameter ({sigmoid_best})"
    elif best_category == 'slope':
        classification = "PC1 = growth rate / slope"
    elif best_category == 'curvature':
        classification = "PC1 = curvature / variance"
    else:
        classification = "PC1 = mixed geometric encoding"
elif best_r > 0.5:
    classification = "PC1 = mixed geometric encoding"
else:
    classification = "PC1 = unclear / non-interpretable"

print(f"\nFinal Classification: {classification}")

# Write interpretation
with open(OUTPUT_DIR + 'interpretation.txt', 'w') as f:
    f.write("PC1 INTERPRETATION ANALYSIS\n")
    f.write("="*60 + "\n\n")
    
    f.write("1. WHAT FEATURE(S) BEST EXPLAIN PC1?\n")
    f.write("-"*40 + "\n")
    for _, row in corr_df.head(5).iterrows():
        f.write(f"   {row['feature']}: r = {row['correlation']:.3f} (p = {row['p_value']:.4f})\n")
    f.write("\n")
    
    f.write("2. IS PC1 PRIMARILY AMPLITUDE, SHIFT, OR CURVATURE?\n")
    f.write("-"*40 + "\n")
    f.write(f"   Amplitude features: max |r| = {categories['amplitude']:.3f}\n")
    f.write(f"   Slope features: max |r| = {categories['slope']:.3f}\n")
    f.write(f"   Curvature features: max |r| = {categories['curvature']:.3f}\n")
    f.write(f"   Sigmoid features: max |r| = {categories['sigmoid']:.3f}\n")
    f.write(f"\n   Best category: {best_category} (r = {best_r:.3f})\n")
    f.write("\n")
    
    f.write("3. ARE RELATIONSHIPS ABSENT IN NULL/SHUFFLE CONTROLS?\n")
    f.write("-"*40 + "\n")
    f.write(f"   Null control mean |r|: {abs(comparison_df['null'].mean()):.3f}\n")
    f.write(f"   Shuffled control mean |r|: {abs(comparison_df['shuffled'].mean()):.3f}\n")
    f.write(f"   Null similar to real: {'YES' if null_similar else 'NO'}\n")
    f.write(f"   Shuffled similar to real: {'YES' if shuffled_similar else 'NO'}\n")
    f.write("\n")
    
    f.write("4. PC1 LOADINGS PATTERN\n")
    f.write("-"*40 + "\n")
    f.write(f"   Loadings range: {loading_range:.4f}\n")
    f.write(f"   Loadings std: {loading_std:.4f}\n")
    f.write(f"   Uniform: {'YES' if is_uniform else 'NO'}\n")
    f.write(f"   Increasing: {'YES' if is_increasing else 'NO'}\n")
    f.write(f"   Centered: {'YES' if is_centered else 'NO'}\n")
    f.write("\n")
    
    f.write("5. FINAL CLASSIFICATION\n")
    f.write("-"*40 + "\n")
    f.write(f"   {classification}\n")
    f.write("\n")
    
    f.write("6. SUMMARY\n")
    f.write("-"*40 + "\n")
    f.write("   PC1 captures the dominant mode of variation across residual curves.\n")
    f.write(f"   The strongest correlation is with {top_feature} (r = {top_r:.3f}).\n")
    if null_similar or shuffled_similar:
        f.write("   However, similar correlations appear in null/shuffle controls,\n")
        f.write("   suggesting some relationship may be generic rather than specific.\n")
    else:
        f.write("   Null and shuffle controls show weaker correlations,\n")
        f.write("   suggesting this relationship is specific to real residual structure.\n")

print(f"\nSaved: interpretation.txt")
print("\nANALYSIS COMPLETE")
