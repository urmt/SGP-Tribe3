#!/usr/bin/env python3
"""
Shape Randomization Falsification Test
Determines whether classification is driven by true shape structure or generic properties
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid
from scipy.fft import fft, ifft
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA as SkPCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

OUTPUT_DIR = '/home/student/sgp-tribe3/experiments/paper5_validation/shape_test/'

systems = ['TRIBE', 'Hierarchical', 'Correlated', 'Sparse', 'CurvedManifold']
n_systems = len(systems)
k_values = np.arange(5, 51, 5)
n_k = len(k_values)

# Base residual data
np.random.seed(42)
base_residuals = {
    'TRIBE': np.array([0.5, 1.2, 2.1, 3.2, 4.5, 5.8, 6.5, 7.2, 7.5, 7.6]),
    'Hierarchical': np.array([0.4, 1.0, 1.8, 2.8, 4.0, 5.2, 6.0, 6.8, 7.2, 7.3]),
    'Correlated': np.array([0.3, 0.8, 1.5, 2.4, 3.5, 4.8, 5.8, 6.5, 7.0, 7.2]),
    'Sparse': np.array([0.6, 1.5, 2.8, 4.2, 5.8, 7.0, 7.6, 7.9, 8.0, 8.0]),
    'CurvedManifold': np.array([0.4, 1.1, 2.0, 3.0, 4.2, 5.5, 6.3, 7.0, 7.4, 7.5])
}

real_residuals = np.array([base_residuals[s] for s in systems])

# True cluster labels
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

def phase_randomize(signal):
    """Phase randomization while preserving magnitude spectrum"""
    fft_vals = fft(signal)
    magnitude = np.abs(fft_vals)
    phase = np.random.uniform(-np.pi, np.pi, len(signal))
    randomized = magnitude * np.exp(1j * phase)
    # Ensure real output
    result = np.real(ifft(randomized))
    # Preserve mean
    result = result - np.mean(result) + np.mean(signal)
    return result

def permute_and_smooth(signal, sigma=2.0):
    """Random permutation followed by Gaussian smoothing"""
    permuted = np.random.permutation(signal)
    smoothed = gaussian_filter1d(permuted, sigma=sigma)
    return smoothed

def sigmoid_replace(signal, k):
    """Replace with pure sigmoid fit"""
    _, popt = fit_sigmoid(k.astype(float), signal)
    if popt is not None:
        return sigmoid_model(k.astype(float), *popt)
    return signal

print("="*70)
print("SHAPE RANDOMIZATION FALSIFICATION TEST")
print("="*70)

# ============================================================
# STEP 1: SHAPE RANDOMIZATION (CONTROL DATASETS)
# ============================================================
print("\n" + "="*70)
print("STEP 1: GENERATING CONTROL DATASETS")
print("="*70)

# Method A: Phase Randomized
phase_randomized = np.array([phase_randomize(res) for res in real_residuals])
print("\nPhase Randomized:")
for i, s in enumerate(systems):
    print(f"  {s}: mean={phase_randomized[i].mean():.3f}, std={phase_randomized[i].std():.3f}")

# Method B: Permuted + Smoothed
np.random.seed(42)
permuted_smooth = np.array([permute_and_smooth(res, sigma=1.5) for res in real_residuals])
print("\nPermuted + Smoothed:")
for i, s in enumerate(systems):
    print(f"  {s}: mean={permuted_smooth[i].mean():.3f}, std={permuted_smooth[i].std():.3f}")

# Method C: Sigmoid-only
sigmoid_only = np.array([sigmoid_replace(res, k_values) for res in real_residuals])
print("\nSigmoid Only:")
for i, s in enumerate(systems):
    print(f"  {s}: mean={sigmoid_only[i].mean():.3f}, std={sigmoid_only[i].std():.3f}")

# Save control datasets
datasets = {
    'original': real_residuals,
    'phase_randomized': phase_randomized,
    'permuted_smoothed': permuted_smooth,
    'sigmoid_only': sigmoid_only
}

for name, data in datasets.items():
    df = pd.DataFrame(data, index=systems, columns=[f'k{i}' for i in range(n_k)])
    df.to_csv(OUTPUT_DIR + f'controls/{name}.csv')

# Save individual controls
pd.DataFrame(phase_randomized, index=systems).to_csv(OUTPUT_DIR + 'controls/phase_randomized.csv')
pd.DataFrame(permuted_smooth, index=systems).to_csv(OUTPUT_DIR + 'controls/permuted_smoothed.csv')
pd.DataFrame(sigmoid_only, index=systems).to_csv(OUTPUT_DIR + 'controls/sigmoid_only.csv')

print(f"\nSaved control datasets to: {OUTPUT_DIR}controls/")

# ============================================================
# STEP 2: NORMALIZATION
# ============================================================
print("\n" + "="*70)
print("STEP 2: NORMALIZATION")
print("="*70)

def mean_normalize(data):
    return np.array([res / np.mean(res) for res in data])

def auc_normalize(data):
    k = k_values.astype(float)
    return np.array([res / trapezoid(res, k) for res in data])

# For classification, we'll use the original residuals and compare
# controls directly against original

# ============================================================
# STEP 3: FEATURE EXTRACTION
# ============================================================
print("\n" + "="*70)
print("STEP 3: FEATURE EXTRACTION")
print("="*70)

def extract_features(data):
    """Extract shape-invariant features"""
    features = []
    for i in range(data.shape[0]):
        res = data[i]
        k = k_values.astype(float)
        
        f = []
        # Normalized amplitude features (to be invariant to scale)
        f.append(np.mean(res) / np.mean(real_residuals[i]))  # Relative amplitude
        f.append(np.std(res) / np.std(real_residuals[i]))   # Relative std
        
        # Shape features (scale-invariant)
        res_norm = res / np.mean(res) if np.mean(res) != 0 else res
        f.append(np.max(res_norm) - np.min(res_norm))  # Normalized range
        f.append(trapezoid(res_norm, k) / k[-1])  # Normalized AUC
        
        # Derivative features
        deriv = np.gradient(res_norm, k)
        f.append(np.mean(deriv))
        f.append(np.max(deriv))
        f.append(np.mean(np.gradient(deriv, k)))  # Curvature
        
        # Sigmoid fit quality (shape consistency)
        r2, _ = fit_sigmoid(k, res)
        f.append(r2 if not np.isnan(r2) else 0)
        
        features.append(f)
    return np.array(features)

# ============================================================
# STEP 4: CLASSIFICATION TEST
# ============================================================
print("\n" + "="*70)
print("STEP 4: CLASSIFICATION TEST")
print("="*70)

def classify_loocv(features, labels):
    """Leave-one-out classification"""
    n = len(labels)
    predictions = []
    for holdout in range(n):
        train_idx = [i for i in range(n) if i != holdout]
        train_X = features[train_idx]
        train_y = np.array(labels)[train_idx]
        test_X = features[holdout].reshape(1, -1)
        
        distances = np.linalg.norm(train_X - test_X, axis=1)
        nearest = np.argmin(distances)
        predictions.append(train_y[nearest])
    
    return np.mean([p == a for p, a in zip(predictions, labels)])

# Classify each dataset
classification_results = {}
for name, data in datasets.items():
    feat = extract_features(data)
    acc = classify_loocv(feat, true_labels)
    classification_results[name] = acc
    print(f"  {name}: {acc:.1%}")

# Save classification results
class_df = pd.DataFrame({
    'Dataset': list(classification_results.keys()),
    'Accuracy': list(classification_results.values())
})
class_df.to_csv(OUTPUT_DIR + 'classification_results.csv', index=False)
print(f"\nSaved: classification_results.csv")

# ============================================================
# STEP 5: PCA ANALYSIS
# ============================================================
print("\n" + "="*70)
print("STEP 5: PCA ANALYSIS")
print("="*70)

pca_results = {}
for name, data in datasets.items():
    # Z-score for PCA
    zscore = StandardScaler().fit_transform(data)
    pca = SkPCA(n_components=min(data.shape[0], data.shape[1]))
    scores = pca.fit_transform(zscore)
    pc1_var = pca.explained_variance_ratio_[0]
    pca_results[name] = pc1_var
    print(f"  {name}: PC1 = {pc1_var*100:.1f}%")

# Save PCA results
pca_df = pd.DataFrame({
    'Dataset': list(pca_results.keys()),
    'PC1_Variance': list(pca_results.values())
})
pca_df.to_csv(OUTPUT_DIR + 'pca_results.csv', index=False)
print(f"\nSaved: pca_results.csv")

# ============================================================
# STEP 6: CROSS-SYSTEM SIMILARITY
# ============================================================
print("\n" + "="*70)
print("STEP 6: CROSS-SYSTEM SIMILARITY")
print("="*70)

def mean_pairwise_correlation(data):
    """Mean pairwise Pearson correlation"""
    n = data.shape[0]
    corrs = []
    for i in range(n):
        for j in range(i+1, n):
            r, _ = stats.pearsonr(data[i], data[j])
            corrs.append(r)
    return np.mean(corrs)

similarity_results = {}
for name, data in datasets.items():
    # Use z-scored for correlation
    zscore = StandardScaler().fit_transform(data)
    mean_r = mean_pairwise_correlation(zscore)
    similarity_results[name] = mean_r
    print(f"  {name}: mean r = {mean_r:.3f}")

# Save similarity
sim_df = pd.DataFrame({
    'Dataset': list(similarity_results.keys()),
    'Mean_Correlation': list(similarity_results.values())
})
sim_df.to_csv(OUTPUT_DIR + 'similarity.csv', index=False)
print(f"\nSaved: similarity.csv")

# ============================================================
# STEP 7: SUMMARY TABLE
# ============================================================
print("\n" + "="*70)
print("STEP 7: SUMMARY TABLE")
print("="*70)

summary_df = pd.DataFrame({
    'Dataset': list(datasets.keys()),
    'PC1_Variance': [pca_results[k] for k in datasets.keys()],
    'Classification_Accuracy': [classification_results[k] for k in datasets.keys()],
    'Mean_Cross_r': [similarity_results[k] for k in datasets.keys()]
})
summary_df.to_csv(OUTPUT_DIR + 'summary.csv', index=False)
print(summary_df.to_string(index=False))
print(f"\nSaved: summary.csv")

# ============================================================
# VISUALIZATION
# ============================================================
print("\n" + "="*70)
print("GENERATING FIGURES")
print("="*70)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Plot curves for each dataset
for idx, (name, data) in enumerate(datasets.items()):
    ax = axes[0, idx]
    for i, s in enumerate(systems):
        ax.plot(k_values, data[i], 'o-', label=s, markersize=4)
    ax.set_title(f'{name}\nAcc: {classification_results[name]*100:.0f}%')
    ax.set_xlabel('k')
    ax.set_ylabel('Residual')
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)

# Metrics comparison
ax = axes[1, 0]
datasets_names = list(datasets.keys())
accs = [classification_results[k] for k in datasets_names]
colors = ['steelblue', 'coral', 'green', 'purple', 'orange']
bars = ax.bar(datasets_names, [a*100 for a in accs], color=colors, alpha=0.7, edgecolor='black')
ax.axhline(y=20, color='red', linestyle='--', label='Chance (20%)')
ax.set_ylabel('Classification Accuracy (%)')
ax.set_title('Classification Accuracy by Dataset')
ax.legend()
for bar, val in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{val*100:.0f}%', ha='center')

# PC1 variance
ax = axes[1, 1]
pc1s = [pca_results[k] for k in datasets_names]
bars = ax.bar(datasets_names, [p*100 for p in pc1s], color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('PC1 Variance (%)')
ax.set_title('PC1 Variance by Dataset')
for bar, val in zip(bars, pc1s):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val*100:.1f}%', ha='center')

# Mean cross-r
ax = axes[1, 2]
rs = [similarity_results[k] for k in datasets_names]
bars = ax.bar(datasets_names, rs, color=colors, alpha=0.7, edgecolor='black')
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_ylabel('Mean Cross-system r')
ax.set_title('Cross-system Similarity')
for bar, val in zip(bars, rs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02 if val > 0 else bar.get_height() - 0.1, f'{val:.2f}', ha='center')

# Normalized comparison (relative to original)
ax = axes[1, 3]
ax.axis('off')

plt.suptitle('Shape Randomization Falsification Test Results', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'shape_falsification.pdf', dpi=150)
plt.close()
print("Saved: shape_falsification.pdf")

# ============================================================
# STEP 8: INTERPRETATION
# ============================================================
print("\n" + "="*70)
print("STEP 8: INTERPRETATION")
print("="*70)

# Find which control reduces accuracy most
original_acc = classification_results['original']
control_accs = {k: v for k, v in classification_results.items() if k != 'original'}
most_damaging = min(control_accs, key=control_accs.get)
acc_drop = original_acc - classification_results[most_damaging]

# Determine classification driver
if classification_results['sigmoid_only'] > 0.4:
    driver = "coarse functional form"
elif classification_results['permuted_smooth'] > 0.4:
    driver = "smooth shape properties"
elif classification_results['phase_randomized'] > 0.4:
    driver = "spectral properties"
else:
    driver = "fine structure"

# Final classification
if acc_drop > 0.3:
    final_class = "Classification driven by true shape structure"
elif classification_results['sigmoid_only'] > 0.4:
    final_class = "Classification driven by coarse functional form"
else:
    final_class = "Classification driven by artifacts / weak structure"

print(f"\nKey Findings:")
print(f"  Original accuracy: {original_acc*100:.0f}%")
print(f"  Most damaging control: {most_damaging} (acc={classification_results[most_damaging]*100:.0f}%)")
print(f"  Accuracy drop: {acc_drop*100:.0f}%")
print(f"  Sigmoid-only accuracy: {classification_results['sigmoid_only']*100:.0f}%")
print(f"\nClassification driver: {driver}")
print(f"Final classification: {final_class}")

# Write interpretation
with open(OUTPUT_DIR + 'interpretation.txt', 'w') as f:
    f.write("SHAPE RANDOMIZATION FALSIFICATION TEST - INTERPRETATION\n")
    f.write("="*60 + "\n\n")
    
    f.write("1. DOES CLASSIFICATION PERSIST AFTER SHAPE DESTRUCTION?\n")
    f.write("-"*40 + "\n")
    for name, acc in classification_results.items():
        status = "✓" if acc > 0.2 else "✗"
        f.write(f"   {status} {name}: {acc*100:.0f}%\n")
    f.write("\n")
    
    f.write("2. WHICH CONTROL REDUCES ACCURACY THE MOST?\n")
    f.write("-"*40 + "\n")
    f.write(f"   Most damaging: {most_damaging}\n")
    f.write(f"   Accuracy drop: {acc_drop*100:.0f}%\n")
    f.write(f"   Original: {original_acc*100:.0f}% → {classification_results[most_damaging]*100:.0f}%\n")
    f.write("\n")
    
    f.write("3. IS CLASSIFICATION DRIVEN BY FINE STRUCTURE OR COARSE FORM?\n")
    f.write("-"*40 + "\n")
    f.write(f"   Sigmoid-only accuracy: {classification_results['sigmoid_only']*100:.0f}%\n")
    f.write(f"   Permuted+smoothed accuracy: {classification_results['permuted_smoothed']*100:.0f}%\n")
    f.write(f"   Phase-randomized accuracy: {classification_results['phase_randomized']*100:.0f}%\n")
    if classification_results['sigmoid_only'] > 0.4:
        f.write(f"   → Classification survives even with pure functional form\n")
        f.write(f"   → DRIVEN BY COARSE FUNCTIONAL FORM\n")
    elif classification_results['permuted_smooth'] > 0.4:
        f.write(f"   → Smoothing destroys structure\n")
        f.write(f"   → DRIVEN BY SMOOTH SHAPE PROPERTIES\n")
    else:
        f.write(f"   → Fine structure required\n")
        f.write(f"   → DRIVEN BY TRUE SHAPE STRUCTURE\n")
    f.write("\n")
    
    f.write("4. DOES SIGMOID-ONLY RETAIN PREDICTIVE POWER?\n")
    f.write("-"*40 + "\n")
    if classification_results['sigmoid_only'] > 0.6:
        f.write(f"   Yes ({classification_results['sigmoid_only']*100:.0f}%) - strong predictive power\n")
    elif classification_results['sigmoid_only'] > 0.2:
        f.write(f"   Partial ({classification_results['sigmoid_only']*100:.0f}%) - some predictive power\n")
    else:
        f.write(f"   No ({classification_results['sigmoid_only']*100:.0f}%) - no predictive power\n")
    f.write("\n")
    
    f.write("5. FINAL CLASSIFICATION\n")
    f.write("-"*40 + "\n")
    f.write(f"   {final_class}\n")
    f.write("\n")
    
    f.write("6. SUMMARY\n")
    f.write("-"*40 + "\n")
    f.write(f"   Classification results by dataset:\n")
    for name, acc in classification_results.items():
        bar = "█" * int(acc * 20)
        f.write(f"   {name:<20}: {bar:<20} {acc*100:.0f}%\n")
    f.write("\n")
    if acc_drop > 0.3:
        f.write("   Shape randomization DESTROYS classification.\n")
        f.write("   This confirms classification depends on true shape structure.\n")
    else:
        f.write("   Classification PERSISTS after shape randomization.\n")
        f.write("   This indicates classification depends on coarse functional properties.\n")

print(f"\nSaved: interpretation.txt")
print("\nANALYSIS COMPLETE")
