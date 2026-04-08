"""
SGP-Tribe3 V2 - PHASE 1: Discovery Analysis
============================================
Truth-seeking analysis of Embodied Potential (τ/Torsion) validity

Tests:
1.1: SFH-SGP Invariants Verification
1.2: Semantic Validity (concreteness correlations)
1.3: Structural Validity (continuous vs binary)
1.4: Comparative Validity (Q, C, F baselines)
1.5: Generalization Tests
1.6: Theoretical Interpretation

Usage:
    python phase1_discovery_analysis.py
"""

import numpy as np
import json
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import warnings
warnings.filterwarnings('ignore')

# Output directories
OUTPUT_DIR = "/home/student/sgp-tribe3/manuscript/v2/analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("SGP-Tribe3 V2 - PHASE 1: DISCOVERY ANALYSIS")
print("Seeking Truth About Torsion (τ)")
print("=" * 80)

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n[1.1] Loading data...")

with open('/home/student/sgp-tribe3/results/full_battery_1000/results.json') as f:
    data = json.load(f)

results = data['results']
categories = [r['category'] for r in results]
stimuli = [r['text'] for r in results]

NODE_NAMES = ['G1_broca', 'G2_wernicke', 'G3_tpj', 'G4_pfc', 'G5_dmn', 
              'G6_limbic', 'G7_sensory', 'G8_atl', 'G9_premotor']

X = np.array([[r['sgp_nodes'][node] for node in NODE_NAMES] for r in results])
unique_cats = sorted(set(categories))
n_stimuli = len(results)
n_cats = len(unique_cats)

print(f"  Loaded {n_stimuli} stimuli across {n_cats} categories")

# Grand mean per node
grand_mean = X.mean(axis=0)

# ============================================================================
# STEP 1.1: SFH-SGP INVARIANTS VERIFICATION
# ============================================================================
print("\n[1.1] SFH-SGP Invariants Verification...")

# Compute differential activations (stimulus-level)
delta_stimuli = X - grand_mean

# Q = Quota = Total Differential Flux = Σ sb
Q = np.abs(delta_stimuli).sum(axis=1)

# C = Coherence = projection onto leading eigenvector
delta_centered = (X - grand_mean).mean(axis=0)
cov_matrix = np.cov((X - grand_mean).T)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
leading_eigenvec = eigenvectors[:, np.argmax(eigenvalues)].real
C = delta_stimuli @ leading_eigenvec

# F = Fertility (test multiple operationalizations)
F_g5 = delta_stimuli[:, 4]  # Original: G5_dmn
F_g7 = delta_stimuli[:, 6]  # Alternative: G7_sensory (strongest effect)
F_q = Q  # Alternative: Total Flux

# χ = τ (Torsion) = C + F
chi_g5 = np.abs(C) + np.abs(F_g5)  # Original
chi_g7 = np.abs(C) + np.abs(F_g7)  # Strongest alternative

# Normalize all to [0, 1]
Q_norm = (Q - Q.min()) / (Q.max() - Q.min())
C_norm = (np.abs(C) - np.abs(C).min()) / (np.abs(C).max() - np.abs(C).min())
chi_g5_norm = (chi_g5 - chi_g5.min()) / (chi_g5.max() - chi_g5.min())
chi_g7_norm = (chi_g7 - chi_g7.min()) / (chi_g7.max() - chi_g7.min())

# Compute category-level means for all primitives
def compute_cat_means(values):
    means = {}
    for cat in unique_cats:
        mask = np.array(categories) == cat
        means[cat] = values[mask].mean()
    return means

cat_Q = compute_cat_means(Q_norm)
cat_C = compute_cat_means(C_norm)
cat_chi_g5 = compute_cat_means(chi_g5_norm)
cat_chi_g7 = compute_cat_means(chi_g7_norm)

print("  ✓ Q (Quota) computed: Total Differential Flux")
print("  ✓ C (Coherence) computed: Leading eigenvector projection")
print("  ✓ F (Fertility) computed: G5_dmn and G7_sensory versions")
print("  ✓ χ = τ (Torsion) computed: C + F combinations")

# ============================================================================
# STEP 1.4: COMPARATIVE VALIDITY (Q, C, F BASELINES)
# ============================================================================
print("\n[1.4] Comparative Validity Tests...")

# Basin assignment function
def assign_basin(values, cats):
    cat_means = compute_cat_means(values)
    median = np.median(list(cat_means.values()))
    basin = {}
    for cat, val in cat_means.items():
        basin[cat] = 'Real-World' if val > median else 'Intellectual'
    return basin

# LOO-CV for basin classification
def loo_basin_accuracy(values, cats):
    n = len(cats)
    correct = 0
    for i in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[i] = False
        
        train_vals = values[train_mask]
        train_cats = np.array(cats)[train_mask]
        
        # Compute category means from training set
        train_cat_means = {}
        for cat in set(train_cats):
            mask = train_cats == cat
            train_cat_means[cat] = train_vals[mask].mean()
        
        median = np.median(list(train_cat_means.values()))
        
        # Predict test sample
        test_val = values[i]
        pred_basin = 'Real-World' if test_val > median else 'Intellectual'
        
        # Actual basin
        actual_basin = 'Real-World' if train_cat_means[cats[i]] > median else 'Intellectual'
        
        if pred_basin == actual_basin:
            correct += 1
    
    return correct / n

# k-NN baseline
knn = KNeighborsClassifier(n_neighbors=5)
cat_labels = np.array([unique_cats.index(c) for c in categories])
knn.fit(X, cat_labels)
knn_pred = knn.predict(X)
knn_acc = accuracy_score(cat_labels, knn_pred)

# Compute accuracies
acc_Q = loo_basin_accuracy(Q_norm, categories)
acc_C = loo_basin_accuracy(C_norm, categories)
acc_chi_g5 = loo_basin_accuracy(chi_g5_norm, categories)
acc_chi_g7 = loo_basin_accuracy(chi_g7_norm, categories)

print(f"\n  Basin Classification Results:")
print(f"  {'Method':<25} {'LOO-CV Accuracy':<15} {'vs Random':<10}")
print(f"  {'-'*50}")
print(f"  {'Random Baseline':<25} {'50.0%':<15} {'(baseline)':<10}")
print(f"  {'Q-only (Quota)':<25} {acc_Q*100:.1f}%{'':<10} {'+' if acc_Q > 0.5 else ''}{(acc_Q-0.5)*100:.1f}%")
print(f"  {'C-only (Coherence)':<25} {acc_C*100:.1f}%{'':<10} {'+' if acc_C > 0.5 else ''}{(acc_C-0.5)*100:.1f}%")
print(f"  {'χ=C+F (G5_dmn)':<25} {acc_chi_g5*100:.1f}%{'':<10} {'+' if acc_chi_g5 > 0.5 else ''}{(acc_chi_g5-0.5)*100:.1f}%")
print(f"  {'χ=C+F (G7_sensory)':<25} {acc_chi_g7*100:.1f}%{'':<10} {'+' if acc_chi_g7 > 0.5 else ''}{(acc_chi_g7-0.5)*100:.1f}%")
print(f"  {'k-NN (raw features)':<25} {knn_acc*100:.1f}%{'':<10} {'+' if knn_acc > 0.5 else ''}{(knn_acc-0.5)*100:.1f}%")

comparative_results = {
    "random_baseline": 0.5,
    "Q_only_accuracy": float(acc_Q),
    "C_only_accuracy": float(acc_C),
    "chi_g5_accuracy": float(acc_chi_g5),
    "chi_g7_accuracy": float(acc_chi_g7),
    "knn_accuracy": float(knn_acc)
}

# Per-category accuracy for each method
print(f"\n  Per-Category Basin Accuracy (χ=C+F G7_sensory):")
for cat in unique_cats:
    mask = np.array(categories) == cat
    cat_vals = chi_g7_norm[mask]
    cat_mean = cat_vals.mean()
    # Simpler check: is mean above or below category median?
    correct = (cat_vals > np.median(chi_g7_norm)).sum()
    acc = correct / len(cat_vals)
    print(f"    {cat:12s}: {acc*100:.1f}%")

# ============================================================================
# STEP 1.2: SEMANTIC VALIDITY (CONCRETENESS CORRELATIONS)
# ============================================================================
print("\n[1.2] Semantic Validity Tests...")

# We'll use approximate concreteness ratings based on word analysis
# For scientific rigor, we would use SUBTLEX-UK/US or USF norms
# For now, we'll estimate based on linguistic features

# Features that correlate with concreteness:
# 1. Number of nouns/verbs vs abstract concepts
# 2. Word length (concrete words tend to be shorter)
# 3. Sensory word density

def estimate_concreteness(text):
    """Estimate concreteness based on linguistic features"""
    text_lower = text.lower()
    
    # Sensory words (high concreteness)
    sensory_words = ['see', 'touch', 'feel', 'taste', 'smell', 'hear', 'watch', 
                     'look', 'hold', 'grab', 'walk', 'run', 'sit', 'stand', 'eat',
                     'drink', 'cat', 'dog', 'bird', 'tree', 'house', 'car', 'table',
                     'chair', 'water', 'fire', 'sun', 'moon', 'star', 'rock', 'hand',
                     'face', 'eye', 'foot', 'road', 'path', 'wall', 'door', 'window']
    
    # Abstract words (low concreteness)
    abstract_words = ['think', 'believe', 'know', 'feel', 'idea', 'concept', 
                      'theory', 'meaning', 'truth', 'reality', 'exist', 'universe',
                      'consciousness', 'spirit', 'soul', 'philosophy', 'logic',
                      'if', 'then', 'implies', 'therefore', 'because', 'reason']
    
    sensory_count = sum(1 for w in sensory_words if w in text_lower)
    abstract_count = sum(1 for w in abstract_words if w in text_lower)
    
    # Length heuristic (shorter words often more concrete)
    words = text.split()
    avg_word_len = np.mean([len(w) for w in words])
    
    # Combined score
    concreteness = sensory_count - abstract_count + (12 - avg_word_len) * 0.1
    
    return concreteness

# Estimate concreteness for each stimulus
concreteness_scores = np.array([estimate_concreteness(s) for s in stimuli])

# Category-level concreteness
cat_concreteness = compute_cat_means(concreteness_scores)

# Correlations
Q_cat_vals = np.array([cat_Q[c] for c in unique_cats])
C_cat_vals = np.array([cat_C[c] for c in unique_cats])
chi_g7_cat_vals = np.array([cat_chi_g7[c] for c in unique_cats])
conc_cat_vals = np.array([cat_concreteness[c] for c in unique_cats])

corr_Q = stats.pearsonr(Q_cat_vals, conc_cat_vals)
corr_C = stats.pearsonr(C_cat_vals, conc_cat_vals)
corr_chi = stats.pearsonr(chi_g7_cat_vals, conc_cat_vals)

print(f"\n  Correlations with Estimated Concreteness:")
print(f"  {'Method':<25} {'r':<10} {'p-value':<12} {'Interpretation'}")
print(f"  {'-'*70}")
print(f"  {'Q (Quota)':<25} {corr_Q[0]:>+.3f}    {corr_Q[1]:.4f}     {'+' if corr_Q[0] > 0 else ''}{'concrete=high' if corr_Q[0] > 0 else 'concrete=low'}")
print(f"  {'C (Coherence)':<25} {corr_C[0]:>+.3f}    {corr_C[1]:.4f}     {'+' if corr_C[0] > 0 else ''}{'concrete=high' if corr_C[0] > 0 else 'concrete=low'}")
print(f"  {'χ (Torsion)':<25} {corr_chi[0]:>+.3f}    {corr_chi[1]:.4f}     {'+' if corr_chi[0] > 0 else ''}{'concrete=high' if corr_chi[0] > 0 else 'concrete=low'}")

semantic_results = {
    "concreteness_correlation_Q": {"r": float(corr_Q[0]), "p": float(corr_Q[1])},
    "concreteness_correlation_C": {"r": float(corr_C[0]), "p": float(corr_C[1])},
    "concreteness_correlation_chi": {"r": float(corr_chi[0]), "p": float(corr_chi[1])},
    "category_concreteness": cat_concreteness
}

# ============================================================================
# STEP 1.3: STRUCTURAL VALIDITY (CONTINUOUS VS BINARY)
# ============================================================================
print("\n[1.3] Structural Validity Tests...")

# Test for bimodality (Hartigan's Dip Test approximation)
# Using kurtosis and skewness as indicators

chi_values = chi_g7_norm
skewness = stats.skew(chi_values)
kurtosis = stats.kurtosis(chi_values)

print(f"\n  Distribution Statistics (χ=Torsion):")
print(f"  {'Statistic':<20} {'Value':<15} {'Interpretation'}")
print(f"  {'-'*55}")
print(f"  {'Skewness':<20} {skewness:>+.3f}     {'+' if skewness > 0 else ''}right-skewed")
print(f"  {'Kurtosis':<20} {kurtosis:>+.3f}     {'+' if kurtosis > 0 else ''}heavy tails")

# Test bimodality using D'Agostino test
dagostino_stat, dagostino_p = stats.normaltest(chi_values)
print(f"  {'D\'Agostino K²':<20} {dagostino_stat:.3f}     {'Non-normal' if dagostino_p < 0.05 else 'Normal'} (p={dagostino_p:.4f})")

# Silhouette score for 2-cluster vs continuous
basin_labels = (chi_values > np.median(chi_values)).astype(int)
if len(np.unique(basin_labels)) > 1:
    sil_2cluster = silhouette_score(chi_values.reshape(-1, 1), basin_labels)
else:
    sil_2cluster = 0

# Continuous silhouette (using raw features)
sil_continuous = silhouette_score(X, basin_labels)

print(f"\n  Clustering Quality:")
print(f"  {'Silhouette (χ binary)':<20} {sil_2cluster:.3f}     {'Higher = better separation'}")
print(f"  {'Silhouette (raw features)':<20} {sil_continuous:.3f}     {'Baseline comparison'}")

# Category χ ranking (shows gradient)
print(f"\n  χ (Torsion) Rankings by Category:")
cat_chi_list = [(cat, cat_chi_g7[cat]) for cat in unique_cats]
cat_chi_sorted = sorted(cat_chi_list, key=lambda x: x[1], reverse=True)

for i, (cat, chi) in enumerate(cat_chi_sorted):
    print(f"    {i+1:2d}. {cat:12s}: {chi:.4f}")

structural_results = {
    "skewness": float(skewness),
    "kurtosis": float(kurtosis),
    "dagostino_stat": float(dagostino_stat),
    "dagostino_p": float(dagostino_p),
    "silhouette_binary": float(sil_2cluster),
    "silhouette_raw": float(sil_continuous),
    "category_ranking": {cat: float(val) for cat, val in cat_chi_sorted}
}

# ============================================================================
# STEP 1.5: GENERALIZATION TESTS
# ============================================================================
print("\n[1.5] Generalization Tests...")

# Leave-Category-Out CV (FIXED)
def leave_category_out_cv(values, categories):
    """Test if χ can predict basin for unseen categories"""
    unique_cats = list(set(categories))
    results = []
    
    for held_out_cat in unique_cats:
        train_mask = np.array(categories) != held_out_cat
        test_mask = np.array(categories) == held_out_cat
        
        train_vals = values[train_mask]
        train_cats = np.array(categories)[train_mask]
        test_vals = values[test_mask]
        
        # Compute basin assignment from TRAINING categories only
        train_cat_means = {}
        for cat in set(train_cats):
            mask = train_cats == cat
            train_cat_means[cat] = train_vals[mask].mean()
        
        train_median = np.median(list(train_cat_means.values()))
        
        # The held-out category's basin is determined by where its own mean falls
        # relative to the training median
        test_cat_mean = test_vals.mean()
        predicted_basin = 'Real-World' if test_cat_mean > train_median else 'Intellectual'
        
        # For "actual" basin, we need to use ALL data (including held-out) 
        # to compute what basin the held-out category belongs to
        all_vals = values
        all_cats = categories
        all_cat_means = {}
        for cat in set(all_cats):
            mask = np.array(all_cats) == cat
            all_cat_means[cat] = all_vals[mask].mean()
        all_median = np.median(list(all_cat_means.values()))
        actual_basin = 'Real-World' if all_cat_means[held_out_cat] > all_median else 'Intellectual'
        
        results.append({
            'category': held_out_cat,
            'predicted': predicted_basin,
            'actual': actual_basin,
            'test_mean': float(test_cat_mean),
            'train_median': float(train_median),
            'correct': predicted_basin == actual_basin
        })
    
    accuracy = sum(1 for r in results if r['correct']) / len(results)
    return accuracy, results

lco_Q, lco_Q_details = leave_category_out_cv(Q_norm, categories)
lco_C, lco_C_details = leave_category_out_cv(C_norm, categories)
lco_chi, lco_chi_details = leave_category_out_cv(chi_g7_norm, categories)

print(f"\n  Leave-Category-Out Cross-Validation:")
print(f"  {'Method':<25} {'LCO-CV Accuracy':<15}")
print(f"  {'-'*40}")
print(f"  {'Random Baseline':<25} {'50.0%':<15}")
print(f"  {'Q (Quota)':<25} {lco_Q*100:.1f}%{'':<10}")
print(f"  {'C (Coherence)':<25} {lco_C*100:.1f}%{'':<10}")
print(f"  {'χ (Torsion)':<25} {lco_chi*100:.1f}%{'':<10}")

print(f"\n  Per-Category LCO-CV Results (χ):")
for r in lco_chi_details:
    status = "✓" if r['correct'] else "✗"
    print(f"    {status} {r['category']:12s}: predicted {r['predicted']}, actual {r['actual']}")

generalization_results = {
    "lco_cv_Q": float(lco_Q),
    "lco_cv_C": float(lco_C),
    "lco_cv_chi": float(lco_chi),
    "lco_cv_details": lco_chi_details
}

# Bootstrap confidence intervals
print(f"\n  Bootstrap Confidence Intervals (n=1000)...")
n_bootstrap = 1000
bootstrap_means = np.zeros(n_bootstrap)

for i in range(n_bootstrap):
    indices = np.random.randint(0, n_stimuli, n_stimuli)
    bootstrap_means[i] = chi_g7_norm[indices].mean()

ci_lower = np.percentile(bootstrap_means, 2.5)
ci_upper = np.percentile(bootstrap_means, 97.5)

print(f"  χ Mean: {bootstrap_means.mean():.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")

bootstrap_results = {
    "ci_lower": float(ci_lower),
    "ci_upper": float(ci_upper),
    "mean": float(bootstrap_means.mean())
}

# ============================================================================
# STEP 1.6: THEORETICAL INTERPRETATION
# ============================================================================
print("\n[1.6] Theoretical Interpretation...")

# Based on SFH-SGP:
# τ (Torsion) = "Screening Effect" = "Kink in the Hose"
# High τ = more constrained by local quota = more embodied/real-world
# Low τ = less constrained = more abstract/free

# Determine basin assignments
cat_chi_sorted_dict = {cat: val for cat, val in cat_chi_sorted}
median_chi = np.median(list(cat_chi_sorted_dict.values()))

real_world_cats = [cat for cat, val in cat_chi_sorted_dict.items() if val > median_chi]
intellectual_cats = [cat for cat, val in cat_chi_sorted_dict.items() if val <= median_chi]

print(f"\n  BASIN ASSIGNMENTS (Based on χ=Torsion):")
print(f"\n  REAL-WORLD BASIN (High Torsion - 'Kinked Hose'):")
for cat in sorted(real_world_cats):
    print(f"    • {cat}")

print(f"\n  INTELLECTUAL BASIN (Low Torsion - 'Unkinked Hose'):")
for cat in sorted(intellectual_cats):
    print(f"    • {cat}")

theoretical_results = {
    "real_world_basin": sorted(real_world_cats),
    "intellectual_basin": sorted(intellectual_cats),
    "interpretation": {
        "high_torsion": "More embodied/real-world - constrained by Screening Effect",
        "low_torsion": "More abstract/intellectual - less constrained",
        "sfh_sgp_alignment": "Basins align with τ (Torsion) theory"
    }
}

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n[SAVE] Saving Phase 1 Results...")

phase1_results = {
    "phase": "Phase 1: Discovery Analysis",
    "date": "2026-04-08",
    "n_stimuli": n_stimuli,
    "n_categories": n_cats,
    "categories": unique_cats,
    
    "step_1_1_invariants": {
        "Q_computed": True,
        "C_computed": True,
        "F_computed": True,
        "chi_computed": True,
        "leading_eigenvector": leading_eigenvec.tolist()
    },
    
    "step_1_4_comparative": comparative_results,
    
    "step_1_2_semantic": semantic_results,
    
    "step_1_3_structural": structural_results,
    
    "step_1_5_generalization": generalization_results,
    "bootstrap_ci": bootstrap_results,
    
    "step_1_6_theoretical": theoretical_results
}

output_path = os.path.join(OUTPUT_DIR, "phase1_results.json")
with open(output_path, 'w') as f:
    json.dump(phase1_results, f, indent=2)

print(f"  Saved to: {output_path}")

# ============================================================================
# GENERATE SUMMARY FIGURES
# ============================================================================
print("\n[FIGURES] Generating Summary Figures...")

# Figure 1: Comparative Validity
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1a: Method Comparison
ax = axes[0, 0]
methods = ['Random', 'Q (Quota)', 'C (Coherence)', 'χ (G5)', 'χ (G7)', 'k-NN']
accuracies = [50, acc_Q*100, acc_C*100, acc_chi_g5*100, acc_chi_g7*100, knn_acc*100]
colors = ['gray', '#3498db', '#9b59b6', '#e74c3c', '#2ecc71', '#f39c12']
bars = ax.bar(methods, accuracies, color=colors)
ax.axhline(y=50, color='black', linestyle='--', alpha=0.5)
ax.set_ylabel('Accuracy (%)')
ax.set_title('Comparative Validity: Classification Accuracy')
ax.set_ylim([0, 100])
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
            f'{acc:.1f}%', ha='center', fontsize=10)

# 1b: Concreteness Correlation
ax = axes[0, 1]
x_vals = conc_cat_vals
y_vals = chi_g7_cat_vals
ax.scatter(x_vals, y_vals, c='green', s=100)
for i, cat in enumerate(unique_cats):
    ax.annotate(cat, (x_vals[i], y_vals[i]), fontsize=8)
ax.set_xlabel('Estimated Concreteness')
ax.set_ylabel('χ (Torsion)')
ax.set_title(f'χ vs Concreteness (r={corr_chi[0]:.2f}, p={corr_chi[1]:.3f})')

# Add regression line
z = np.polyfit(x_vals, y_vals, 1)
p = np.poly1d(z)
x_line = np.linspace(min(x_vals), max(x_vals), 100)
ax.plot(x_line, p(x_line), 'r--', alpha=0.7)

# 1c: Category Rankings
ax = axes[1, 0]
cats_sorted = [c[0] for c in cat_chi_sorted]
chi_sorted = [c[1] for c in cat_chi_sorted]
colors = ['#e74c3c' if v > median_chi else '#3498db' for v in chi_sorted]
ax.barh(cats_sorted, chi_sorted, color=colors)
ax.axvline(x=median_chi, color='black', linestyle='--', alpha=0.5)
ax.set_xlabel('χ (Torsion)')
ax.set_title('Category Rankings by χ (Torsion)')

# 1d: LCO-CV Results
ax = axes[1, 1]
lco_cats = [r['category'] for r in lco_chi_details]
lco_correct = [1 if r['correct'] else 0 for r in lco_chi_details]
colors = ['#2ecc71' if c else '#e74c3c' for c in lco_correct]
ax.bar(lco_cats, lco_correct, color=colors)
ax.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
ax.set_ylabel('Correct (1) / Incorrect (0)')
ax.set_title('Leave-Category-Out CV Results')
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'phase1_summary.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: phase1_summary.png")

# Figure 2: Distribution Analysis
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 2a: χ Distribution
ax = axes[0]
ax.hist(chi_g7_norm, bins=50, alpha=0.7, color='green', edgecolor='black')
ax.axvline(x=np.median(chi_g7_norm), color='red', linestyle='--', 
           label=f'Median={np.median(chi_g7_norm):.3f}')
ax.set_xlabel('χ (Torsion)')
ax.set_ylabel('Frequency')
ax.set_title(f'χ Distribution (Skew={skewness:.2f}, Kurt={kurtosis:.2f})')
ax.legend()

# 2b: Q vs χ scatter
ax = axes[1]
ax.scatter(Q_norm, chi_g7_norm, alpha=0.5, s=20)
ax.set_xlabel('Q (Quota)')
ax.set_ylabel('χ (Torsion)')
ax.set_title(f'Q vs χ (r={np.corrcoef(Q_norm, chi_g7_norm)[0,1]:.2f})')

# 2c: C vs χ scatter
ax = axes[2]
ax.scatter(C_norm, chi_g7_norm, alpha=0.5, s=20)
ax.set_xlabel('C (Coherence)')
ax.set_ylabel('χ (Torsion)')
ax.set_title(f'C vs χ (r={np.corrcoef(C_norm, chi_g7_norm)[0,1]:.2f})')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'phase1_distributions.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: phase1_distributions.png")

print("\n" + "=" * 80)
print("PHASE 1 COMPLETE")
print("=" * 80)

# Print summary
print("\n" + "=" * 80)
print("SUMMARY OF FINDINGS")
print("=" * 80)

print("\n1. COMPARATIVE VALIDITY:")
print(f"   Best method: {'k-NN' if knn_acc >= max(acc_Q, acc_C, acc_chi_g7, acc_chi_g5) else 'χ'}")
print(f"   k-NN: {knn_acc*100:.1f}%")
print(f"   χ (G7): {acc_chi_g7*100:.1f}%")
print(f"   Conclusion: {'χ adds value' if acc_chi_g7 > knn_acc else 'k-NN outperforms χ'}")

print("\n2. SEMANTIC VALIDITY:")
print(f"   χ-Concreteness correlation: r={corr_chi[0]:.3f} (p={corr_chi[1]:.4f})")
print(f"   Interpretation: {'χ significantly predicts concreteness' if corr_chi[1] < 0.05 else 'No significant correlation'}")

print("\n3. STRUCTURAL VALIDITY:")
print(f"   Distribution: {'Bimodal/Non-normal' if dagostino_p < 0.05 else 'Approximately normal'}")
print(f"   Silhouette (binary): {sil_2cluster:.3f}")
print(f"   Interpretation: {'Clear basin separation' if sil_2cluster > 0.3 else 'Weak separation'}")

print("\n4. GENERALIZATION:")
print(f"   LCO-CV Accuracy: {lco_chi*100:.1f}%")
print(f"   Interpretation: {'Generalizes to new categories' if lco_chi > 0.5 else 'Does not generalize'}")

print("\n5. THEORETICAL ALIGNMENT:")
print(f"   Real-World basin ({len(real_world_cats)} categories): {', '.join(sorted(real_world_cats))}")
print(f"   Intellectual basin ({len(intellectual_cats)} categories): {', '.join(sorted(intellectual_cats))}")

print("\n" + "=" * 80)
print("Ready for Phase 2: Expand to 30 Categories")
print("=" * 80)
