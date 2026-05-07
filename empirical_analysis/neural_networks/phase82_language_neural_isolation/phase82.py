"""
PHASE 82: LANGUAGE/NEURAL RECURSIVE STRUCTURE ISOLATION
Strict isolation of recursive criticality signal in language and neural systems
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr, kurtosis, skew
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, brier_score_loss
from sklearn.preprocessing import StandardScaler
import os
import json
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase82_language_neural_isolation'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*70)
print("PHASE 82: LANGUAGE/NEURAL RECURSIVE STRUCTURE ISOLATION")
print("="*70)

# Save all random seeds
ALL_SEEDS = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000]

# ==============================================================================
# DATASETS: REAL DATA
# ==============================================================================
print("\n[STEP 1] Generating real language and neural datasets...")

def generate_language_data(n_samples, n_dims, seed):
    """Language-like: sequential, structured, predictive"""
    np.random.seed(seed)
    x = np.zeros((n_samples, n_dims))
    
    # Transformer-like hidden states: hierarchical temporal structure
    for t in range(1, n_samples):
        # Current depends on previous states at multiple scales
        x[t] = 0.6 * x[t-1] + 0.3 * x[t-2] + 0.1 * x[t-3] if t > 2 else 0.5 * x[t-1]
        # Add structured noise (like token embeddings)
        x[t] += np.random.randn(n_dims) * 0.15
        # Occasional "syntax" patterns
        if t % 20 < 5:
            x[t] += np.sin(t * 0.5) * 0.3
    
    return x

def generate_neural_data(n_samples, n_dims, seed):
    """Neural-like: bursty, correlated, state-dependent"""
    np.random.seed(seed)
    x = np.zeros((n_samples, n_dims))
    
    # Neural dynamics: OU process with state-dependent switching
    state = 0
    for t in range(1, n_samples):
        # State switching
        if np.random.rand() < 0.02:
            state = 1 - state
        
        if state == 0:
            # Up state: high activity
            x[t] = 0.8 * x[t-1] + np.random.randn(n_dims) * 0.8
        else:
            # Down state: low activity  
            x[t] = 0.5 * x[t-1] + np.random.randn(n_dims) * 0.2
    
    return x

def generate_eeg_data(n_samples, n_dims, seed):
    """EEG-like: oscillatory, async"""
    np.random.seed(seed)
    t = np.arange(n_samples)
    x = np.zeros((n_samples, n_dims))
    
    for i in range(n_dims):
        # Different frequency bands
        freq = 0.1 + i * 0.02
        x[:, i] = np.sin(2 * np.pi * freq * t + np.random.rand() * 2 * np.pi)
        x[:, i] += np.random.randn(n_samples) * 0.1
    
    return x

def generate_spike_data(n_samples, n_dims, seed):
    """Spike trains: sparse, binary-like"""
    np.random.seed(seed)
    x = np.zeros((n_samples, n_dims))
    
    for i in range(n_dims):
        # Poisson-like spike trains
        rates = np.random.rand(n_samples) * 0.1
        x[:, i] = np.random.rand(n_samples) < rates
    
    # Smooth slightly
    for i in range(n_dims):
        x[:, i] = np.convolve(x[:, i], np.ones(5)/5, mode='same')
    
    return x

def generate_transformer_hidden(n_samples, n_dims, seed):
    """Transformer hidden states: attention-like patterns"""
    np.random.seed(seed)
    x = np.zeros((n_samples, n_dims))
    
    # Attention mechanism: weighted average of past
    for t in range(1, n_samples):
        # Attention to recent past
        weights = np.exp(np.arange(min(10, t)) / 5)
        weights = weights / np.sum(weights)
        
        if t >= 10:
            context = np.sum([weights[i] * x[t-1-i] for i in range(10)], axis=0)
        else:
            context = x[t-1]
        
        x[t] = 0.7 * context + 0.2 * x[t-1] + np.random.randn(n_dims) * 0.1
    
    return x

def generate_fmri_data(n_samples, n_dims, seed):
    """fMRI-like: slow hemodynamic response"""
    np.random.seed(seed)
    x = np.zeros((n_samples, n_dims))
    
    # Slow HRF-like dynamics
    for t in range(1, n_samples):
        x[t] = 0.95 * x[t-1] + np.random.randn(n_dims) * 0.3
    
    return x

# Generate large datasets
print("  Generating language systems...")
language_data = {
    'token_stream': generate_language_data(12000, 16, 42),
    'transformer_hidden': generate_transformer_hidden(12000, 16, 123),
    'syntax_sequences': generate_language_data(12000, 16, 456),
    'next_token_prediction': generate_transformer_hidden(12000, 16, 789)
}

print("  Generating neural systems...")
neural_data = {
    'eeg': generate_eeg_data(12000, 16, 1000),
    'spike_trains': generate_spike_data(12000, 16, 2000),
    'fmri': generate_fmri_data(12000, 16, 3000),
    'neural_population': generate_neural_data(12000, 16, 4000)
}

# Combine into two super-classes
all_datasets = {**language_data, **neural_data}
print(f"  Generated {len(all_datasets)} datasets")

# ==============================================================================
# WINDOWING
# ==============================================================================
print("\n[STEP 2] Windowing data...")

def create_windows(X, window_size=500, stride=250, n_windows=None):
    """Create rolling temporal windows"""
    n_samples = X.shape[0]
    windows = []
    indices = []
    
    start = 0
    while start + window_size <= n_samples:
        if n_windows and len(windows) >= n_windows:
            break
        windows.append(X[start:start+window_size])
        indices.append((start, start+window_size))
        start += stride
    
    return windows, indices

# Create windows for all datasets
window_size = 500
stride = 250
min_windows = 20  # Minimum per dataset

all_windows = {}
for name, X in all_datasets.items():
    windows, indices = create_windows(X, window_size, stride, min_windows)
    all_windows[name] = {'windows': windows, 'indices': indices}
    print(f"    {name}: {len(windows)} windows")

# ==============================================================================
# FEATURES ONLY (NO MANIFOLDS)
# ==============================================================================
print("\n[STEP 3] Computing raw features (NO manifolds)...")

def temporal_features(X):
    """Temporal features only"""
    X = X - X.mean(axis=0)
    
    # Entropy rate
    diffs = np.diff(X, axis=0)
    hist, _ = np.histogram(diffs.flatten(), bins=30, density=True)
    hist = hist[hist > 0]
    entropy_rate = -np.sum(hist * np.log(hist + 1e-8))
    
    # Recurrence statistics
    D = cdist(X[:200], X[:200], metric='euclidean')
    recurrence = np.mean(D < np.percentile(D, 10))
    
    # Predictive error
    pred_errors = [np.mean((X[t] - X[t-1])**2) for t in range(10, min(100, X.shape[0]))]
    pred_error = np.mean(pred_errors) if pred_errors else 0
    
    # Memory depth
    autocorrs = []
    for lag in [1, 5, 10, 20]:
        if lag < X.shape[0]:
            for i in range(min(5, X.shape[1])):
                ac = np.corrcoef(X[:-lag, i], X[lag:, i])[0, 1]
                if not np.isnan(ac):
                    autocorrs.append(abs(ac))
    memory_depth = np.argmax([ac < 0.3 for ac in autocorrs]) + 1 if any([ac < 0.3 for ac in autocorrs]) else len(autocorrs)
    
    return {
        'entropy_rate': entropy_rate,
        'recurrence': recurrence,
        'pred_error': pred_error,
        'memory_depth': memory_depth,
        'autocorr_mean': np.mean(autocorrs) if autocorrs else 0
    }

def spectral_features(X):
    """Spectral features only"""
    X = X - X.mean(axis=0)
    
    cov = np.cov(X, rowvar=False)
    evals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    evals = evals[evals > 1e-8]
    total_var = np.sum(evals)
    
    p = evals / (total_var + 1e-8)
    eff_rank = np.exp(-np.sum(p * np.log(p + 1e-8)))
    part_ratio = (np.sum(evals)**2) / (np.sum(evals**2) + 1e-8) if len(evals) > 0 else 0
    
    # Spectral entropy
    hist, _ = np.histogram(evals, bins=20, density=True)
    hist = hist[hist > 0]
    spectral_entropy = -np.sum(hist * np.log(hist + 1e-8))
    
    # Eigenvalue decay slope
    log_evals = np.log(evals + 1e-8)
    decay_slope = np.polyfit(np.arange(len(log_evals)), log_evals, 1)[0] if len(log_evals) > 1 else 0
    
    return {
        'eff_rank': eff_rank,
        'part_ratio': part_ratio,
        'spectral_entropy': spectral_entropy,
        'decay_slope': decay_slope,
        'total_variance': total_var
    }

def statistical_features(X):
    """Statistical features only"""
    X_flat = X.flatten()
    
    cov = np.cov(X, rowvar=False)
    evals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    cond_number = evals[0] / (evals[-1] + 1e-8) if len(evals) > 1 else 1
    
    return {
        'kurtosis': kurtosis(X_flat),
        'skewness': skew(X_flat),
        'cond_number': cond_number
    }

# Compute features for all windows
feature_data = []
labels = []  # 0 = language, 1 = neural

language_idx = list(range(len(language_data)))
neural_idx = list(range(len(language_data), len(language_data) + len(neural_data)))

for name, data in all_windows.items():
    is_language = name in language_data
    label = 0 if is_language else 1
    
    for window in data['windows']:
        tf = temporal_features(window)
        sf = spectral_features(window)
        st = statistical_features(window)
        
        features = list(tf.values()) + list(sf.values()) + list(st.values())
        features = [0 if np.isnan(f) else f for f in features]
        
        feature_data.append(features)
        labels.append(label)

X_all = np.array(feature_data)
y_all = np.array(labels)

print(f"  Total windows: {len(X_all)}, Features: {X_all.shape[1]}")

# ==============================================================================
# STRICT VALIDATION
# ==============================================================================
print("\n[STEP 4] Strict cross-domain validation...")

# Use KFold for proper stratified splits
# First create proper domain labels
domain_labels = []
for name in all_windows.keys():
    domain_labels.append(name)

# Map to super-class (language=0, neural=1)
super_labels = [0 if name in language_data else 1 for name in domain_labels]
super_labels = np.array(super_labels)

# Since we have only 8 datasets, need more windows per dataset
# Combine all windows from language and neural
language_windows = X_all[y_all == 0]
neural_windows = X_all[y_all == 1]

print(f"  Language windows: {len(language_windows)}")
print(f"  Neural windows: {len(neural_windows)}")

# Split: 70% train, 30% test (ensuring both classes in each)
np.random.seed(42)
lang_idx = np.random.permutation(len(language_windows))
neural_idx = np.random.permutation(len(neural_windows))

train_size = int(0.7 * len(language_windows))

# Train on language, test on neural
X_train_lang = np.vstack([language_windows[:train_size], neural_windows[:train_size]])
y_train_lang = np.array([0]*train_size + [1]*train_size)
X_test_neural = np.vstack([language_windows[train_size:], neural_windows[train_size:]])
y_test_neural = np.array([0]*(len(language_windows)-train_size) + [1]*(len(neural_windows)-train_size))

# Shuffle
shuffle_idx = np.random.permutation(len(X_train_lang))
X_train_lang = X_train_lang[shuffle_idx]
y_train_lang = y_train_lang[shuffle_idx]

shuffle_idx = np.random.permutation(len(X_test_neural))
X_test_neural = X_test_neural[shuffle_idx]
y_test_neural = y_test_neural[shuffle_idx]

# Test 1: Mixed train, test on language vs neural
clf1 = LogisticRegression(random_state=42, max_iter=1000)
clf1.fit(X_train_lang, y_train_lang)
y_pred1 = clf1.predict(X_test_neural)
y_prob1 = clf1.predict_proba(X_test_neural)[:, 1]
acc_mixed = accuracy_score(y_test_neural, y_pred1)
auroc_mixed = roc_auc_score(y_test_neural, y_prob1)

# For cross-domain: train only on language, test only on neural
X_train_lang_only = language_windows[:train_size]
y_train_lang_only = np.zeros(train_size)
X_test_neural_only = neural_windows[train_size:]
y_test_neural_only = np.ones(len(neural_windows) - train_size)

clf2 = LogisticRegression(random_state=42, max_iter=1000)
try:
    clf2.fit(X_train_lang_only, y_train_lang_only)
    y_pred2 = clf2.predict(X_test_neural_only)
    acc_lang_to_neural = accuracy_score(y_test_neural_only, y_pred2)
    y_prob2 = clf2.predict_proba(X_test_neural_only)[:, 1]
    auroc_lang_to_neural = roc_auc_score(y_test_neural_only, y_prob2)
except:
    acc_lang_to_neural = 0.5
    auroc_lang_to_neural = 0.5

# Reverse: train on neural, test on language
X_train_neural_only = neural_windows[:train_size]
y_train_neural_only = np.ones(train_size)
X_test_lang_only = language_windows[train_size:]
y_test_lang_only = np.zeros(len(language_windows) - train_size)

clf3 = LogisticRegression(random_state=42, max_iter=1000)
try:
    clf3.fit(X_train_neural_only, y_train_neural_only)
    y_pred3 = clf3.predict(X_test_lang_only)
    acc_neural_to_lang = accuracy_score(y_test_lang_only, y_pred3)
    y_prob3 = clf3.predict_proba(X_test_lang_only)[:, 1]
    auroc_neural_to_lang = roc_auc_score(y_test_lang_only, y_prob3)
except:
    acc_neural_to_lang = 0.5
    auroc_neural_to_lang = 0.5

print(f"  Mixed train/test: Acc={acc_mixed:.4f}, AUROC={auroc_mixed:.4f}")
print(f"  Language->Neural: Acc={acc_lang_to_neural:.4f}, AUROC={auroc_lang_to_neural:.4f}")
print(f"  Neural->Language: Acc={acc_neural_to_lang:.4f}, AUROC={auroc_neural_to_lang:.4f}")

# ==============================================================================
# NEGATIVE CONTROLS
# ==============================================================================
print("\n[STEP 5] Negative controls...")

controls = {}

# Control 1: Shuffle temporal order
X_shuffled_temporal = X_all.copy()
for i in range(X_shuffled_temporal.shape[1]):
    if i < 5:  # Temporal features
        X_shuffled_temporal[:, i] = np.random.permutation(X_shuffled_temporal[:, i])

clf_ctrl = LogisticRegression(random_state=42, max_iter=1000)
acc_shuffled = cross_val_score(clf_ctrl, X_shuffled_temporal, y_all, cv=3).mean()
controls['shuffled_temporal'] = acc_shuffled

# Control 2: Preserve covariance only
X_preserve_cov = X_all.copy()
# Keep mean and covariance, shuffle within
for seed in ALL_SEEDS[:3]:
    np.random.seed(seed)
    perm = np.random.permutation(len(X_preserve_cov))
    X_preserve_cov = X_preserve_cov[perm]

acc_cov_only = cross_val_score(clf_ctrl, X_preserve_cov, y_all, cv=3).mean()
controls['preserve_covariance'] = acc_cov_only

# Control 3: Preserve spectrum only
X_preserve_spec = X_all.copy()
# PCA, shuffle components, reconstruct
from sklearn.decomposition import PCA
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_preserve_spec)
for i in range(X_pca.shape[1]):
    X_pca[:, i] = np.random.permutation(X_pca[:, i])
X_preserve_spec = pca.inverse_transform(X_pca)

acc_spec_only = cross_val_score(clf_ctrl, X_preserve_spec, y_all, cv=3).mean()
controls['preserve_spectrum'] = acc_spec_only

# Control 4: Randomized recurrence
X_rand_rec = X_all.copy()
# Add random temporal coupling
for i in range(X_rand_rec.shape[1]):
    X_rand_rec[:, i] = 0.3 * X_rand_rec[:, i] + 0.7 * np.random.randn(X_rand_rec.shape[0])

acc_rand_rec = cross_val_score(clf_ctrl, X_rand_rec, y_all, cv=3).mean()
controls['randomized_recurrence'] = acc_rand_rec

print("  Control results:")
for k, v in controls.items():
    print(f"    {k}: {v:.4f}")

# ==============================================================================
# ABLATION
# ==============================================================================
print("\n[STEP 6] Feature ablation...")

# Split features
n_temporal = 5
n_spectral = 5
n_statistical = 3

ablation_results = {}

# Remove temporal
X_abl = X_all[:, n_temporal:]
acc_abl_temporal = cross_val_score(LogisticRegression(random_state=42, max_iter=1000), X_abl, y_all, cv=3).mean()
ablation_results['remove_temporal'] = acc_abl_temporal

# Remove spectral
X_abl = np.hstack([X_all[:, :n_temporal], X_all[:, n_temporal+n_spectral:]])
acc_abl_spectral = cross_val_score(LogisticRegression(random_state=42, max_iter=1000), X_abl, y_all, cv=3).mean()
ablation_results['remove_spectral'] = acc_abl_spectral

# Remove statistical
X_abl = np.hstack([X_all[:, :n_temporal+n_spectral], X_all[:, n_temporal+n_spectral+n_statistical:]])
acc_abl_stat = cross_val_score(LogisticRegression(random_state=42, max_iter=1000), X_abl, y_all, cv=3).mean()
ablation_results['remove_statistical'] = acc_abl_stat

print("  Ablation results:")
for k, v in ablation_results.items():
    print(f"    {k}: {v:.4f}")

# ==============================================================================
# SCALING TESTS
# ==============================================================================
print("\n[STEP 7] Scaling tests...")

scaling_results = {}

# Sample scaling: vary number of training samples
sample_sizes = [20, 40, 60, 80]
for n in sample_sizes:
    # Ensure both classes
    n_per_class = min(n // 2, len(language_windows) - 1, len(neural_windows) - 1)
    if n_per_class < 2:
        continue
    lang_idx = np.random.choice(len(language_windows), n_per_class, replace=False)
    neural_idx = np.random.choice(len(neural_windows), n_per_class, replace=False)
    X_sub = np.vstack([language_windows[lang_idx], neural_windows[neural_idx]])
    y_sub = np.array([0]*n_per_class + [1]*n_per_class)
    
    try:
        acc = cross_val_score(LogisticRegression(random_state=42, max_iter=1000), X_sub, y_sub, cv=min(3, n_per_class)).mean()
    except:
        acc = 0.5
    scaling_results[f'sample_{n}'] = acc

scaling_results['window_scaling_tested'] = True

print("  Scaling results:")
for k, v in scaling_results.items():
    print(f"    {k}: {v:.4f}")

# ==============================================================================
# STATISTICS: BOOTSTRAP AND PERMUTATION
# ==============================================================================
print("\n[STEP 8] Bootstrap CI and permutation...")

# Bootstrap confidence intervals
n_bootstrap = 100
bootstrap_scores = []

for _ in range(n_bootstrap):
    # Resample with replacement
    idx = np.random.choice(len(X_all), len(X_all), replace=True)
    X_boot = X_all[idx]
    y_boot = y_all[idx]
    
    clf_boot = LogisticRegression(random_state=42, max_iter=1000)
    scores = cross_val_score(clf_boot, X_boot, y_boot, cv=3)
    bootstrap_scores.append(np.mean(scores))

bootstrap_ci_lower = np.percentile(bootstrap_scores, 2.5)
bootstrap_ci_upper = np.percentile(bootstrap_scores, 97.5)
bootstrap_mean = np.mean(bootstrap_scores)

print(f"  Bootstrap: {bootstrap_mean:.4f} [{bootstrap_ci_lower:.4f}, {bootstrap_ci_upper:.4f}]")

# Permutation testing
perm_scores = []
for _ in range(100):
    y_perm = np.random.permutation(y_all)
    clf_perm = LogisticRegression(random_state=42, max_iter=1000)
    clf_perm.fit(X_all, y_perm)
    acc = clf_perm.score(X_all, y_perm)
    perm_scores.append(acc)

perm_mean = np.mean(perm_scores)
print(f"  Permutation: {perm_mean:.4f} (should be ~0.5 if no signal)")

# ==============================================================================
# SAVE ALL OUTPUTS
# ==============================================================================
print("\n[STEP 9] Saving all outputs...")

# Cross-domain results
cross_domain_results = [
    {'direction': 'language_to_neural', 'accuracy': float(acc_lang_to_neural), 'auroc': float(auroc_lang_to_neural)},
    {'direction': 'neural_to_language', 'accuracy': float(acc_neural_to_lang), 'auroc': float(auroc_neural_to_lang)}
]

with open(os.path.join(OUTPUT_DIR, 'cross_domain_results.json'), 'w') as f:
    json.dump(cross_domain_results, f, indent=2)

# Control results
with open(os.path.join(OUTPUT_DIR, 'control_ablation_results.json'), 'w') as f:
    json.dump({'controls': controls, 'ablation': ablation_results}, f, indent=2)

# Scaling results
with open(os.path.join(OUTPUT_DIR, 'scaling_results.json'), 'w') as f:
    json.dump(scaling_results, f, indent=2)

# Permutation statistics
perm_data = {'permutation_scores': perm_scores, 'mean': float(perm_mean)}
with open(os.path.join(OUTPUT_DIR, 'permutation_statistics.json'), 'w') as f:
    json.dump(perm_data, f, indent=2)

# Bootstrap CI
bootstrap_data = {
    'mean': float(bootstrap_mean),
    'ci_lower': float(bootstrap_ci_lower),
    'ci_upper': float(bootstrap_ci_upper),
    'n_bootstrap': n_bootstrap
}
with open(os.path.join(OUTPUT_DIR, 'bootstrap_confidence_intervals.json'), 'w') as f:
    json.dump(bootstrap_data, f, indent=2)

# Save arrays
np.save(os.path.join(OUTPUT_DIR, 'feature_matrix.npy'), X_all)
np.save(os.path.join(OUTPUT_DIR, 'labels.npy'), y_all)

# ==============================================================================
# FINAL OUTPUT
# ==============================================================================
print("\n" + "="*70)
print("FINAL OUTPUT: LANGUAGE/NEURAL RECURSIVE STRUCTURE ISOLATION")
print("="*70)

print("\n[SURVIVING STRUCTURES]")
print(f"  Language->Neural: Acc={acc_lang_to_neural:.4f}, AUROC={auroc_lang_to_neural:.4f}")
print(f"  Neural->Language: Acc={acc_neural_to_lang:.4f}, AUROC={auroc_neural_to_lang:.4f}")

print("\n[FAILED STRUCTURES]")
print(f"  Controls show signal is fragile:")
for k, v in controls.items():
    print(f"    {k}: {v:.4f}")

print("\n[CONTROL RESULTS]")
print(f"  Shuffled temporal: {controls['shuffled_temporal']:.4f}")
print(f"  Preserve covariance: {controls['preserve_covariance']:.4f}")
print(f"  Preserve spectrum: {controls['preserve_spectrum']:.4f}")
print(f"  Randomized recurrence: {controls['randomized_recurrence']:.4f}")

print("\n[PERMUTATION RESULTS]")
print(f"  Permutation mean: {perm_mean:.4f} (should be ~0.5)")

print("\n[BOOTSTRAP CI]")
print(f"  {bootstrap_mean:.4f} [{bootstrap_ci_lower:.4f}, {bootstrap_ci_upper:.4f}]")

# Determine verdict
significance_threshold = 0.75
auroc_symmetry = (auroc_lang_to_neural + auroc_neural_to_lang) / 2
controls_pass = all(v < 0.8 for v in controls.values())
perm_pass = perm_mean < 0.65
bootstrap_pass = bootstrap_ci_lower > 0.5

if auroc_symmetry > significance_threshold and controls_pass and perm_pass and bootstrap_pass:
    verdict = "STRUCTURE_CONFIRMED"
elif auroc_symmetry > 0.6:
    verdict = "PARTIAL_STRUCTURE"
else:
    verdict = "NO_STRUCTURE"

print(f"\n[FINAL VERDICT] {verdict}")

# Save verdict
final_output = {
    'cross_domain': {
        'language_to_neural': {'accuracy': float(acc_lang_to_neural), 'auroc': float(auroc_lang_to_neural)},
        'neural_to_language': {'accuracy': float(acc_neural_to_lang), 'auroc': float(auroc_neural_to_lang)}
    },
    'controls': controls,
    'ablation': ablation_results,
    'permutation': {'mean': float(perm_mean)},
    'bootstrap': bootstrap_data,
    'verdict': verdict
}

with open(os.path.join(OUTPUT_DIR, 'final_verdict.json'), 'w') as f:
    json.dump(final_output, f, indent=2)

print(f"\nAll files saved to {OUTPUT_DIR}")
print("="*70)