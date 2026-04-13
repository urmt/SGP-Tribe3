#!/usr/bin/env python3
"""
Parameter-Only Classification Test
Determines whether system identity is encoded in sigmoid parameters alone
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

OUTPUT_DIR = '/home/student/sgp-tribe3/experiments/paper5_validation/parameter_test/'

systems = ['TRIBE', 'Hierarchical', 'Correlated', 'Sparse', 'CurvedManifold']
n_systems = len(systems)

# Base residual data
np.random.seed(42)
base_residuals = {
    'TRIBE': np.array([0.5, 1.2, 2.1, 3.2, 4.5, 5.8, 6.5, 7.2, 7.5, 7.6]),
    'Hierarchical': np.array([0.4, 1.0, 1.8, 2.8, 4.0, 5.2, 6.0, 6.8, 7.2, 7.3]),
    'Correlated': np.array([0.3, 0.8, 1.5, 2.4, 3.5, 4.8, 5.8, 6.5, 7.0, 7.2]),
    'Sparse': np.array([0.6, 1.5, 2.8, 4.2, 5.8, 7.0, 7.6, 7.9, 8.0, 8.0]),
    'CurvedManifold': np.array([0.4, 1.1, 2.0, 3.0, 4.2, 5.5, 6.3, 7.0, 7.4, 7.5])
}

k_values = np.arange(5, 51, 5)
real_residuals = np.array([base_residuals[s] for s in systems])

# True cluster labels
true_labels = np.array([0, 0, 0, 1, 0])

def sigmoid_model(k, A, beta, k0):
    return A / (1 + np.exp(-beta * (k - k0)))

def fit_sigmoid(k, y):
    try:
        popt, _ = curve_fit(sigmoid_model, k.astype(float), y, 
                            p0=[8, 0.1, 25], maxfev=5000,
                            bounds=([0, 0, 1], [20, 1, 50]))
        return popt
    except:
        return None

print("="*70)
print("PARAMETER-ONLY CLASSIFICATION TEST")
print("="*70)

# ============================================================
# STEP 1: EXTRACT PARAMETERS
# ============================================================
print("\n" + "="*70)
print("STEP 1: EXTRACT SIGMOID PARAMETERS")
print("="*70)

params_data = []
for i, s in enumerate(systems):
    popt = fit_sigmoid(k_values.astype(float), real_residuals[i])
    if popt is not None:
        A, beta, k0 = popt
    else:
        A, beta, k0 = np.nan, np.nan, np.nan
    
    params_data.append({
        'system': s,
        'A': A,
        'k0': k0,
        'beta': beta
    })
    print(f"  {s}: A={A:.3f}, k0={k0:.3f}, beta={beta:.3f}")

params_df = pd.DataFrame(params_data)
params_df.to_csv(OUTPUT_DIR + 'parameters.csv', index=False)
print(f"\nSaved: parameters.csv")

# ============================================================
# STEP 2: STANDARDIZATION
# ============================================================
print("\n" + "="*70)
print("STEP 2: STANDARDIZATION")
print("="*70)

# Extract parameter matrix
X = params_df[['A', 'k0', 'beta']].values

# Z-score standardization
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

params_norm_df = pd.DataFrame({
    'system': systems,
    'A_z': X_normalized[:, 0],
    'k0_z': X_normalized[:, 1],
    'beta_z': X_normalized[:, 2]
})
params_norm_df.to_csv(OUTPUT_DIR + 'parameters_normalized.csv', index=False)
print(f"\nStandardized parameters:")
print(params_norm_df.to_string(index=False))
print(f"\nSaved: parameters_normalized.csv")

# ============================================================
# STEP 3: CLASSIFICATION
# ============================================================
print("\n" + "="*70)
print("STEP 3: CLASSIFICATION (PARAMETER-ONLY)")
print("="*70)

def loocv_classification(X, y, model_type='knn'):
    """Leave-one-out cross-validation with multi-class handling"""
    n = len(y)
    predictions = []
    probabilities = []
    
    for holdout in range(n):
        train_idx = [i for i in range(n) if i != holdout]
        test_idx = holdout
        
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx].reshape(1, -1)
        
        if model_type == 'knn':
            model = KNeighborsClassifier(n_neighbors=min(2, len(np.unique(y_train))))
        else:
            # For logistic regression, use one-vs-rest with only available classes
            model = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')
        
        # Check if training set has multiple classes
        unique_classes = np.unique(y_train)
        if len(unique_classes) < 2:
            # If only one class in training, predict that class
            predictions.append(y_train[0])
            probabilities.append([1.0])
            continue
        
        model.fit(X_train, y_train)
        pred = model.predict(X_test)[0]
        predictions.append(pred)
        
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(X_test)[0]
            probabilities.append(prob)
    
    accuracy = np.mean([p == a for p, a in zip(predictions, y)])
    return accuracy, predictions, probabilities

# KNN classification
acc_knn, preds_knn, probs_knn = loocv_classification(X_normalized, true_labels, 'knn')

# Logistic regression
acc_lr, preds_lr, probs_lr = loocv_classification(X_normalized, true_labels, 'lr')

print(f"\nKNN Classification:")
print(f"  Accuracy: {acc_knn:.1%}")
print(f"  Predictions: {[f'{systems[i]}: {preds_knn[i]}' for i in range(n_systems)]}")

print(f"\nLogistic Regression:")
print(f"  Accuracy: {acc_lr:.1%}")
print(f"  Predictions: {[f'{systems[i]}: {preds_lr[i]}' for i in range(n_systems)]}")

# Save classification results
class_results = pd.DataFrame({
    'system': systems,
    'true_label': true_labels,
    'knn_predicted': preds_knn,
    'lr_predicted': preds_lr,
    'correct_knn': [1 if p == t else 0 for p, t in zip(preds_knn, true_labels)],
    'correct_lr': [1 if p == t else 0 for p, t in zip(preds_lr, true_labels)]
})
class_results.to_csv(OUTPUT_DIR + 'classification_results.csv', index=False)
print(f"\nSaved: classification_results.csv")

# ============================================================
# STEP 4: BASELINE COMPARISON
# ============================================================
print("\n" + "="*70)
print("STEP 4: BASELINE COMPARISON")
print("="*70)

# Full feature model accuracy (from previous experiments)
full_feature_accuracy = 0.80  # From Paper 5
chance_accuracy = 0.20  # 1/5 classes

print(f"\nAccuracy Comparison:")
print(f"  Full feature model: {full_feature_accuracy:.1%}")
print(f"  Sigmoid parameters only: {acc_knn:.1%} (KNN), {acc_lr:.1%} (LR)")
print(f"  Chance level: {chance_accuracy:.1%}")

accuracy_comparison = pd.DataFrame({
    'Model': ['Full feature model', 'Sigmoid params (KNN)', 'Sigmoid params (LR)', 'Chance'],
    'Accuracy': [full_feature_accuracy, acc_knn, acc_lr, chance_accuracy]
})
accuracy_comparison.to_csv(OUTPUT_DIR + 'accuracy_comparison.csv', index=False)
print(f"\nSaved: accuracy_comparison.csv")

# ============================================================
# STEP 5: FEATURE IMPORTANCE
# ============================================================
print("\n" + "="*70)
print("STEP 5: FEATURE IMPORTANCE")
print("="*70)

# Compute correlation of each parameter with labels
correlations = []
for i, param in enumerate(['A', 'k0', 'beta']):
    r, p = stats.pointbiserialr(true_labels, X[:, i])
    correlations.append({
        'parameter': param,
        'correlation': abs(r),
        'raw_r': r,
        'p_value': p
    })

# Coefficient magnitude from logistic regression
# Fit on full data for coefficient interpretation
lr_full = LogisticRegression(random_state=42, max_iter=1000)
lr_full.fit(X_normalized, true_labels)
coefficients = np.abs(lr_full.coef_[0])

importance_df = pd.DataFrame({
    'parameter': ['A', 'k0', 'beta'],
    'correlation_with_label': [correlations[i]['correlation'] for i in range(3)],
    'lr_coefficient': coefficients
})
importance_df['combined_importance'] = importance_df['correlation_with_label'] * importance_df['lr_coefficient']
importance_df = importance_df.sort_values('combined_importance', ascending=False)
importance_df.to_csv(OUTPUT_DIR + 'feature_importance.csv', index=False)

print(f"\nFeature Importance:")
for _, row in importance_df.iterrows():
    print(f"  {row['parameter']}: corr={row['correlation_with_label']:.3f}, coef={row['lr_coefficient']:.3f}")
print(f"\nSaved: feature_importance.csv")

# ============================================================
# STEP 6: DECISION BOUNDARY VISUALIZATION
# ============================================================
print("\n" + "="*70)
print("STEP 6: DECISION BOUNDARY VISUALIZATION")
print("="*70)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Color mapping
colors = ['steelblue', 'coral', 'green', 'purple', 'orange']
labels_name = ['Cluster 0', 'Cluster 0', 'Cluster 0', 'Cluster 1', 'Cluster 0']

# (A vs k0)
ax = axes[0]
for i, s in enumerate(systems):
    ax.scatter(X_normalized[i, 0], X_normalized[i, 1], c=colors[i], s=150, label=s, edgecolors='black')
ax.set_xlabel('A (amplitude)')
ax.set_ylabel('k0 (midpoint)')
ax.set_title('A vs k0')
ax.legend(fontsize=8)

# (A vs beta)
ax = axes[1]
for i, s in enumerate(systems):
    ax.scatter(X_normalized[i, 0], X_normalized[i, 2], c=colors[i], s=150, label=s, edgecolors='black')
ax.set_xlabel('A (amplitude)')
ax.set_ylabel('beta (slope)')
ax.set_title('A vs beta')
ax.legend(fontsize=8)

# (k0 vs beta)
ax = axes[2]
for i, s in enumerate(systems):
    ax.scatter(X_normalized[i, 1], X_normalized[i, 2], c=colors[i], s=150, label=s, edgecolors='black')
ax.set_xlabel('k0 (midpoint)')
ax.set_ylabel('beta (slope)')
ax.set_title('k0 vs beta')
ax.legend(fontsize=8)

plt.suptitle('Sigmoid Parameter Space', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'plots/parameter_space.pdf', dpi=150)
plt.close()
print("Saved: plots/parameter_space.pdf")

# 3D scatter
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
for i, s in enumerate(systems):
    ax.scatter(X_normalized[i, 0], X_normalized[i, 1], X_normalized[i, 2], 
               c=colors[i], s=200, label=s, edgecolors='black')
ax.set_xlabel('A (amplitude)')
ax.set_ylabel('k0 (midpoint)')
ax.set_zlabel('beta (slope)')
ax.set_title('3D Sigmoid Parameter Space')
ax.legend()
plt.savefig(OUTPUT_DIR + 'plots/parameter_space_3d.pdf', dpi=150)
plt.close()
print("Saved: plots/parameter_space_3d.pdf")

# ============================================================
# STEP 7: ROBUSTNESS CHECK
# ============================================================
print("\n" + "="*70)
print("STEP 7: ROBUSTNESS CHECK")
print("="*70)

noise_levels = [0.05, 0.10, 0.20]
noise_results = []

for noise in noise_levels:
    np.random.seed(42)
    accuracies = []
    for trial in range(100):
        # Add noise
        X_noisy = X_normalized + np.random.normal(0, noise, X_normalized.shape)
        # Clip to reasonable range
        X_noisy = np.clip(X_noisy, -3, 3)
        # Classify
        acc, _, _ = loocv_classification(X_noisy, true_labels, 'knn')
        accuracies.append(acc)
    
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    ci_low = np.percentile(accuracies, 2.5)
    ci_high = np.percentile(accuracies, 97.5)
    
    noise_results.append({
        'noise_level': f'{int(noise*100)}%',
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'ci_low': ci_low,
        'ci_high': ci_high
    })
    print(f"  Noise {int(noise*100)}%: {mean_acc:.1%} ± {std_acc:.1%} (95% CI: [{ci_low:.1%}, {ci_high:.1%}])")

noise_df = pd.DataFrame(noise_results)
noise_df.to_csv(OUTPUT_DIR + 'noise_robustness.csv', index=False)
print(f"\nSaved: noise_robustness.csv")

# Figure
fig, ax = plt.subplots(figsize=(8, 5))
x = [int(n['noise_level'].rstrip('%')) for n in noise_results]
means = [n['mean_accuracy'] for n in noise_results]
stds = [n['std_accuracy'] for n in noise_results]
ax.errorbar(x, means, yerr=stds, marker='o', linewidth=2, markersize=8, capsize=5)
ax.axhline(y=0.2, color='red', linestyle='--', label='Chance (20%)')
ax.axhline(y=0.8, color='gray', linestyle='--', label='Full features (80%)')
ax.set_xlabel('Noise Level (%)')
ax.set_ylabel('Classification Accuracy')
ax.set_title('Parameter Classification: Noise Robustness')
ax.legend()
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR + 'noise_robustness.pdf', dpi=150)
plt.close()
print("Saved: noise_robustness.pdf")

# ============================================================
# STEP 8: SUMMARY TABLE
# ============================================================
print("\n" + "="*70)
print("STEP 8: SUMMARY TABLE")
print("="*70)

summary_df = pd.DataFrame({
    'Model': ['Full feature model', 'Sigmoid parameters only', 'Chance'],
    'Accuracy': [full_feature_accuracy, acc_knn, chance_accuracy],
    'Features_used': ['All (curves, derivatives, PCA, sigmoid)', 'A, k0, beta only', 'None']
})
summary_df.to_csv(OUTPUT_DIR + 'summary.csv', index=False)
print(summary_df.to_string(index=False))
print(f"\nSaved: summary.csv")

# ============================================================
# STEP 9: INTERPRETATION
# ============================================================
print("\n" + "="*70)
print("STEP 9: INTERPRETATION")
print("="*70)

# Determine classification
acc_param_only = acc_knn  # Use best model
if acc_param_only >= full_feature_accuracy - 0.1:  # Within 10%
    classification = "System identity fully encoded in sigmoid parameters"
elif acc_param_only >= 0.3:
    classification = "System identity partially encoded in sigmoid parameters"
else:
    classification = "Sigmoid parameters insufficient for classification"

# Feature contribution
top_feature = importance_df.iloc[0]['parameter']
top_importance = importance_df.iloc[0]['combined_importance']

print(f"\nKey Findings:")
print(f"  Parameter-only accuracy: {acc_param_only:.1%}")
print(f"  Full feature accuracy: {full_feature_accuracy:.1%}")
print(f"  Top contributing parameter: {top_feature}")
print(f"  Noise robustness: {noise_results[-1]['mean_accuracy']:.1%} at 20% noise")

print(f"\nFinal Classification: {classification}")

# Write interpretation
with open(OUTPUT_DIR + 'interpretation.txt', 'w') as f:
    f.write("PARAMETER-ONLY CLASSIFICATION TEST - INTERPRETATION\n")
    f.write("="*60 + "\n\n")
    
    f.write("1. DOES PARAMETER-ONLY MODEL RETAIN CLASSIFICATION ACCURACY?\n")
    f.write("-"*40 + "\n")
    f.write(f"   Full feature model: {full_feature_accuracy:.1%}\n")
    f.write(f"   Sigmoid parameters only: {acc_param_only:.1%}\n")
    f.write(f"   Difference: {abs(full_feature_accuracy - acc_param_only)*100:.0f}%\n")
    if acc_param_only >= full_feature_accuracy - 0.1:
        f.write("   → RETAINED (within 10% of full model)\n\n")
    else:
        f.write("   → REDUCED\n\n")
    
    f.write("2. WHICH PARAMETER CONTRIBUTES MOST?\n")
    f.write("-"*40 + "\n")
    for _, row in importance_df.iterrows():
        f.write(f"   {row['parameter']}: importance = {row['combined_importance']:.3f}\n")
    f.write(f"   → {top_feature} is most important\n\n")
    
    f.write("3. IS PERFORMANCE ROBUST TO NOISE?\n")
    f.write("-"*40 + "\n")
    for n in noise_results:
        f.write(f"   {n['noise_level']} noise: {n['mean_accuracy']:.1%} ± {n['std_accuracy']:.1%}\n")
    if noise_results[-1]['mean_accuracy'] > 0.5:
        f.write("   → ROBUST to noise\n\n")
    else:
        f.write("   → NOT robust to high noise\n\n")
    
    f.write("4. HOW SEPARABLE ARE SYSTEMS IN PARAMETER SPACE?\n")
    f.write("-"*40 + "\n")
    # Compute pairwise distances
    distances = []
    for i in range(n_systems):
        for j in range(i+1, n_systems):
            d = np.linalg.norm(X_normalized[i] - X_normalized[j])
            distances.append(d)
    f.write(f"   Mean pairwise distance: {np.mean(distances):.3f}\n")
    f.write(f"   Systems form {'well-separated' if np.mean(distances) > 1.0 else 'overlapping'} clusters\n\n")
    
    f.write("5. FINAL CLASSIFICATION\n")
    f.write("-"*40 + "\n")
    f.write(f"   {classification}\n\n")
    
    f.write("6. SUMMARY\n")
    f.write("-"*40 + "\n")
    f.write(f"   Using only 3 sigmoid parameters (A, k0, beta):\n")
    f.write(f"   - Classification accuracy: {acc_param_only:.1%}\n")
    f.write(f"   - Most important feature: {top_feature}\n")
    f.write(f"   - Noise stability: {noise_results[-1]['mean_accuracy']:.1%} at 20% noise\n")
    if acc_param_only >= 0.6:
        f.write(f"\n   CONCLUSION: Sigmoid parameters alone encode sufficient information\n")
        f.write(f"   for system classification. The coarse functional form determines\n")
        f.write(f"   system identity, not deviations from the sigmoid.\n")

print(f"\nSaved: interpretation.txt")
print("\nANALYSIS COMPLETE")
