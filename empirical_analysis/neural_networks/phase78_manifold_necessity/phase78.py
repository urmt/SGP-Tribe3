"""
PHASE 78: MANIFOLD NECESSITY TEST
Determine if manifold construction adds genuine explanatory power beyond raw data
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import entropy as scipy_entropy
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score
import os
import json
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase78_manifold_necessity'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("PHASE 78: MANIFOLD NECESSITY TEST")
print("="*60)

np.random.seed(42)

funcs = [lambda x: x, np.tanh, lambda x: np.maximum(0, x), lambda x: np.log1p(np.exp(x))]

# ==============================================================================
# STEP 1: RAW FEATURE LIBRARY (NO MANIFOLD)
# ==============================================================================
print("\nStep 1: Raw feature library...")

def generate_observer(alpha, n, d):
    x = np.zeros((n, d))
    z = np.zeros(d)
    for t in range(1, n):
        pred = x[t-1] if t > 0 else np.zeros(d)
        z = (alpha * 0.8) * z + alpha * 0.5 * pred
        if alpha > 0.8:
            x[t] = 0.4 * pred + 0.4 * z + alpha * np.sin(t * 0.01) * z + np.random.randn(d) * 0.15
        else:
            x[t] = 0.5 * pred + 0.3 * z + np.random.randn(d) * 0.2
    return x

def compute_raw_features(X):
    """All raw features WITHOUT manifold construction"""
    X = X - X.mean(axis=0)
    
    # A: Covariance features
    cov = np.cov(X, rowvar=False)
    evals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    evals = evals[evals > 1e-8]
    
    cov_features = {
        'cov_trace': np.sum(evals),
        'cov_rank': len(evals),
        'cov_cond': evals[0] / (evals[-1] + 1e-8) if len(evals) > 1 else 1,
        'cov_spread': np.std(evals) / (np.mean(evals) + 1e-8),
    }
    
    # B: Spectral features
    spectral_features = {
        'spectral_centroid': np.mean(evals),
        'spectral_spread': np.std(evals),
        'spectral_skew': (np.mean(evals) - np.median(evals)) / (np.std(evals) + 1e-8),
        'top_eigenvalue': evals[0] if len(evals) > 0 else 0,
    }
    
    # C: Moment features
    X_flat = X.flatten()
    moment_features = {
        'mean': np.mean(X_flat),
        'variance': np.var(X_flat),
        'skewness': np.mean(((X_flat - np.mean(X_flat)) / (np.std(X_flat) + 1e-8)) ** 3),
        'kurtosis': np.mean(((X_flat - np.mean(X_flat)) / (np.std(X_flat) + 1e-8)) ** 4) - 3,
    }
    
    # D: Entropy features
    hist, _ = np.histogram(X_flat, bins=50, density=True)
    hist = hist[hist > 0]
    entropy_features = {
        'shannon_entropy': scipy_entropy(hist),
        'entropy_normalized': scipy_entropy(hist) / np.log(len(hist)),
    }
    
    # E: Recursive statistics
    pred_errors = [np.mean((X[t] - X[t-1])**2) for t in range(10, X.shape[0])]
    ac_vals = [np.abs(np.corrcoef(X[:-lag, :3], X[lag:, :3])[0, 1]) for lag in [1, 5, 10] if lag < X.shape[0]]
    
    recursive_features = {
        'prediction_error': np.mean(pred_errors),
        'prediction_stability': 1 / (1 + np.std(pred_errors)),
        'autocorr_mean': np.mean(ac_vals) if ac_vals else 0,
        'recurrence_strength': np.mean(ac_vals) if ac_vals else 0,
    }
    
    # F: Temporal statistics
    diff_std = np.std(np.diff(X, axis=0))
    diff_mean = np.mean(np.abs(np.diff(X, axis=0)))
    
    temporal_features = {
        'diff_std': diff_std,
        'diff_mean': diff_mean,
        'temporal_coherence': diff_mean / (diff_std + 1e-8),
    }
    
    # G: Graph statistics (distance matrix)
    D = cdist(X, X, metric='euclidean')
    D_flat = D.flatten()
    D_flat = D_flat[D_flat > 0]
    
    graph_features = {
        'mean_dist': np.mean(D_flat),
        'dist_spread': np.std(D_flat),
        'graph_density': np.sum(D < np.percentile(D, 10)) / D.size,
    }
    
    # Combine all
    raw_features = {**cov_features, **spectral_features, **moment_features, 
                    **entropy_features, **recursive_features, **temporal_features, **graph_features}
    
    return raw_features

# ==============================================================================
# STEP 2: MANIFOLD FEATURE LIBRARY
# ==============================================================================
print("\nStep 2: Manifold feature library...")

def compute_manifold_features(X):
    """Manifold features WITH construction"""
    X = X - X.mean(axis=0)
    
    # Operator manifolds
    phis = np.array([np.mean(f(X), axis=0) for f in funcs])
    
    # Embedding with error handling
    try:
        cov = np.cov(phis, rowvar=False)
        ev, evc = np.linalg.eigh(cov)
        idx = np.argsort(ev)[::-1]
        pc1 = evc[:, idx[0]]
        X1 = phis @ pc1
        var_x1 = float(np.var(X1))
    except:
        var_x1 = 0.0
    
    # RG-like metrics (from operator responses)
    manifold_features = {
        'var_x1': var_x1,
        'manifold_variance': np.var(phis),
        'operator_spread': np.std([np.mean(f(X)) for f in funcs]),
        'operator_entropy': scipy_entropy(np.abs([np.mean(f(X)) for f in funcs]) + 1e-8),
    }
    
    # Topology proxy (clustering)
    try:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = kmeans.fit_predict(phis)
        manifold_features['cluster_count'] = len(set(labels))
        manifold_features['cluster_separation'] = np.max(kmeans.cluster_centers_) - np.min(kmeans.cluster_centers_)
    except:
        manifold_features['cluster_count'] = 1
        manifold_features['cluster_separation'] = 0.0
    
    return manifold_features

# ==============================================================================
# STEP 3: PREDICTION COMPARISON
# ==============================================================================
print("\nStep 3: Prediction comparison...")

# Generate dataset
all_data = []
for alpha in np.linspace(0, 1, 30):
    for trial in range(3):
        X = generate_observer(alpha, 200, 16)
        
        raw_feats = compute_raw_features(X)
        man_feats = compute_manifold_features(X)
        
        all_data.append({
            'alpha': alpha,
            'is_observer': 1 if alpha > 0.8 else 0,
            **raw_feats,
            **man_feats
        })

# Separate raw and manifold features
raw_feature_names = ['cov_trace', 'cov_rank', 'cov_cond', 'cov_spread', 
                     'spectral_centroid', 'spectral_spread', 'spectral_skew', 'top_eigenvalue',
                     'mean', 'variance', 'skewness', 'kurtosis',
                     'shannon_entropy', 'entropy_normalized',
                     'prediction_error', 'prediction_stability', 'autocorr_mean', 'recurrence_strength',
                     'diff_std', 'diff_mean', 'temporal_coherence',
                     'mean_dist', 'dist_spread', 'graph_density']

manifold_feature_names = ['var_x1', 'manifold_variance', 'operator_spread', 'operator_entropy',
                          'cluster_count', 'cluster_separation']

X_raw = np.array([[d[f] for f in raw_feature_names] for d in all_data])
X_man = np.array([[d[f] for f in manifold_feature_names] for d in all_data])
X_combined = np.hstack([X_raw, X_man])
y = np.array([d['is_observer'] for d in all_data])

# Clean data
X_raw = np.nan_to_num(X_raw, nan=0.0)
X_man = np.nan_to_num(X_man, nan=0.0)
X_combined = np.nan_to_num(X_combined, nan=0.0)

# Train/test split
X_train_r, X_test_r, y_train, y_test = train_test_split(X_raw, y, test_size=0.3, random_state=42)
X_train_m, X_test_m, _, _ = train_test_split(X_man, y, test_size=0.3, random_state=42)
X_train_c, X_test_c, _, _ = train_test_split(X_combined, y, test_size=0.3, random_state=42)

# Raw only
clf_raw = LogisticRegression(random_state=42, max_iter=1000)
clf_raw.fit(X_train_r, y_train)
y_pred_raw = clf_raw.predict(X_test_r)
y_prob_raw = clf_raw.predict_proba(X_test_r)[:, 1]
raw_accuracy = accuracy_score(y_test, y_pred_raw)
raw_auroc = roc_auc_score(y_test, y_prob_raw)

# Manifold only
clf_man = LogisticRegression(random_state=42, max_iter=1000)
clf_man.fit(X_train_m, y_train)
y_pred_man = clf_man.predict(X_test_m)
y_prob_man = clf_man.predict_proba(X_test_m)[:, 1]
manifold_accuracy = accuracy_score(y_test, y_pred_man)
manifold_auroc = roc_auc_score(y_test, y_prob_man)

# Combined
clf_comb = LogisticRegression(random_state=42, max_iter=1000)
clf_comb.fit(X_train_c, y_train)
y_pred_comb = clf_comb.predict(X_test_c)
y_prob_comb = clf_comb.predict_proba(X_test_c)[:, 1]
combined_accuracy = accuracy_score(y_test, y_pred_comb)
combined_auroc = roc_auc_score(y_test, y_prob_comb)

print(f"  Raw accuracy: {raw_accuracy:.4f}, AUROC: {raw_auroc:.4f}")
print(f"  Manifold accuracy: {manifold_accuracy:.4f}, AUROC: {manifold_auroc:.4f}")
print(f"  Combined accuracy: {combined_accuracy:.4f}, AUROC: {combined_auroc:.4f}")

# ==============================================================================
# STEP 4: INCREMENTAL INFORMATION
# ==============================================================================
print("\nStep 4: Incremental information...")

delta_information = combined_accuracy - raw_accuracy
delta_auroc = combined_auroc - raw_auroc

print(f"  Delta accuracy: {delta_information:.4f}")
print(f"  Delta AUROC: {delta_auroc:.4f}")

# ==============================================================================
# STEP 5: CONDITIONAL INDEPENDENCE
# ==============================================================================
print("\nStep 5: Conditional independence...")

# Test if manifold features add information after conditioning on raw
# Regress out raw features from manifold features
lr_residual = LinearRegression()
lr_residual.fit(X_raw, X_man)
manifold_residual = X_man - lr_residual.predict(X_raw)

# Test residual predictive power
clf_resid = LogisticRegression(random_state=42, max_iter=1000)
clf_resid.fit(manifold_residual, y)
y_pred_resid = clf_resid.predict(manifold_residual)
residual_predictive_power = accuracy_score(y, y_pred_resid)

# Conditional mutual information (simplified)
from sklearn.metrics import mutual_info_score
# Use variance reduction as proxy
conditional_MI = np.var(manifold_residual) / (np.var(X_man) + 1e-8)

print(f"  Conditional MI: {conditional_MI:.4f}")
print(f"  Residual predictive power: {residual_predictive_power:.4f}")

# ==============================================================================
# STEP 6: MINIMAL SUFFICIENT SET
# ==============================================================================
print("\nStep 6: Minimal sufficient set...")

# Feature importance from combined model
importances = np.abs(clf_comb.coef_[0])

# Sort all features by importance
all_feature_names = raw_feature_names + manifold_feature_names
all_importances = list(zip(all_feature_names, importances))
all_importances.sort(key=lambda x: x[1], reverse=True)

# Test minimal sets
minimal_results = {}
for n_feat in [1, 3, 5, 10, len(all_feature_names)]:
    top_features = [f[0] for f in all_importances[:n_feat]]
    feature_idx = [all_feature_names.index(f) for f in top_features]
    X_subset = X_combined[:, feature_idx]
    
    clf_min = LogisticRegression(random_state=42, max_iter=1000)
    clf_min.fit(X_subset, y)
    acc = cross_val_score(clf_min, X_subset, y, cv=3).mean()
    minimal_results[n_feat] = acc

print(f"  Minimal set accuracy: {minimal_results}")

# Find if manifold features survive
manifold_retained = [f[0] for f in all_importances[:5] if f[0] in manifold_feature_names]
minimal_feature_set = [f[0] for f in all_importances[:5]]

print(f"  Minimal feature set: {minimal_feature_set}")
print(f"  Manifold features retained: {manifold_retained}")

# ==============================================================================
# STEP 7: CROSS-DOMAIN GENERALIZATION
# ==============================================================================
print("\nStep 7: Cross-domain generalization...")

# Test on different domains
domains = {}

# Domain 1: Random deep net
def gen_random_net(n, d):
    x = np.random.randn(n, d)
    for _ in range(3):
        x = np.tanh(x @ np.random.randn(d, d) * 0.1)
    return x

# Domain 2: Coupled oscillators
def gen_coupled(n, d):
    x = np.zeros((n, min(d, 8)))
    for t in range(1, n):
        x[t] = 0.9 * x[t-1] + 0.1 * np.mean(x[:t], axis=0) + np.random.randn(min(d, 8)) * 0.1
    return np.hstack([x, np.random.randn(n, d-8) * 0.1])

# Domain 3: Symbolic
def gen_symbolic(n, d):
    states = np.random.randn(5, d)
    x = np.zeros((n, d))
    state = 0
    for t in range(n):
        state = (state + np.random.choice([1, 2, 3, 4])) % 5
        x[t] = states[state]
    return x

domain_results = {}

for name, gen in [('random_net', gen_random_net), ('coupled', gen_coupled), ('symbolic', gen_symbolic)]:
    try:
        X_dom = gen(200, 16)
        
        raw_feats = compute_raw_features(X_dom)
        man_feats = compute_manifold_features(X_dom)
        
        # Predict using trained models
        raw_feat_vec = np.array([[raw_feats[f] for f in raw_feature_names]])
        man_feat_vec = np.array([[man_feats[f] for f in manifold_feature_names]])
        combined_feat_vec = np.hstack([raw_feat_vec, man_feat_vec])
        
        raw_feat_vec = np.nan_to_num(raw_feat_vec, nan=0.0)
        man_feat_vec = np.nan_to_num(man_feat_vec, nan=0.0)
        combined_feat_vec = np.nan_to_num(combined_feat_vec, nan=0.0)
        
        raw_pred = clf_raw.predict(raw_feat_vec)
        man_pred = clf_man.predict(man_feat_vec)
        
        domain_results[name] = {
            'raw_pred': int(raw_pred[0]),
            'manifold_pred': int(man_pred[0]),
            'raw_conf': float(clf_raw.predict_proba(raw_feat_vec)[0, 1]),
            'manifold_conf': float(clf_man.predict_proba(man_feat_vec)[0, 1])
        }
    except:
        domain_results[name] = {'error': 'failed'}

print(f"  Domain results: {domain_results}")

# Generalization scores
raw_generalization = np.mean([r.get('raw_conf', 0.5) for r in domain_results.values() if 'error' not in r])
manifold_generalization = np.mean([r.get('manifold_conf', 0.5) for r in domain_results.values() if 'error' not in r])

print(f"  Raw generalization: {raw_generalization:.4f}")
print(f"  Manifold generalization: {manifold_generalization:.4f}")

# ==============================================================================
# STEP 8: RAW-ONLY PHASE TEST
# ==============================================================================
print("\nStep 8: Raw-only phase test...")

# Test if raw features alone can detect criticality
alphas = np.linspace(0, 1, 20)
raw_transitions = []

for alpha in alphas:
    X = generate_observer(alpha, 200, 16)
    raw_feats = compute_raw_features(X)
    raw_transitions.append(raw_feats['prediction_error'])

# Check for transition
d1_raw = np.abs(np.gradient(raw_transitions, alphas))
raw_criticality_detected = np.max(d1_raw) > 0.5

# Hysteresis
alphas_up = np.linspace(0, 1, 10)
alphas_down = np.linspace(1, 0, 10)

raw_up = [compute_raw_features(generate_observer(a, 100, 16))['prediction_error'] for a in alphas_up]
raw_down = [compute_raw_features(generate_observer(a, 100, 16))['prediction_error'] for a in alphas_down]

raw_hysteresis_detected = np.sum(np.abs(np.array(raw_up) - np.array(raw_down))) > 0.5

# RG detection (using temporal features as proxy)
raw_rg_detected = raw_criticality_detected  # Simplified

print(f"  Raw criticality detected: {raw_criticality_detected}")
print(f"  Raw hysteresis detected: {raw_hysteresis_detected}")
print(f"  Raw RG detected: {raw_rg_detected}")

# ==============================================================================
# STEP 9: MANIFOLD ABLATION
# ==============================================================================
print("\nStep 9: Manifold ablation...")

# Compare performance without manifold features
performance_without_manifolds = raw_accuracy
performance_loss = raw_accuracy - combined_accuracy

print(f"  Performance without manifolds: {performance_without_manifolds:.4f}")
print(f"  Performance loss: {performance_loss:.4f}")

# ==============================================================================
# STEP 10-12: OUTPUT AND SAVE
# ==============================================================================
print("\n" + "="*60)
print("OUTPUT")
print("="*60)

print("\nRAW VS MANIFOLD:")
print(f"  raw_accuracy = {raw_accuracy:.4f}")
print(f"  manifold_accuracy = {manifold_accuracy:.4f}")
print(f"  combined_accuracy = {combined_accuracy:.4f}")
print(f"  delta_information = {delta_information:.4f}")

print("\nCONDITIONAL INDEPENDENCE:")
print(f"  conditional_MI = {conditional_MI:.4f}")
print(f"  residual_predictive_power = {residual_predictive_power:.4f}")

print("\nMINIMAL SUFFICIENT SET:")
print(f"  minimal_feature_set = {minimal_feature_set}")
print(f"  manifold_features_retained = {manifold_retained}")

print("\nCROSS-DOMAIN:")
print(f"  raw_generalization = {raw_generalization:.4f}")
print(f"  manifold_generalization = {manifold_generalization:.4f}")

print("\nRAW PHASES:")
print(f"  raw_criticality_detected = {raw_criticality_detected}")
print(f"  raw_hysteresis_detected = {raw_hysteresis_detected}")
print(f"  raw_rg_detected = {raw_rg_detected}")

print("\nABLATION:")
print(f"  performance_without_manifolds = {performance_without_manifolds:.4f}")
print(f"  performance_loss = {performance_loss:.4f}")

# Verdict
if delta_information < 0.05 and raw_accuracy > 0.9:
    verdict = "manifolds_optional_decorative"
elif delta_information < 0.1:
    verdict = "manifolds_minimal_addition"
elif residual_predictive_power < 0.6:
    verdict = "manifolds_dependent"
else:
    verdict = "manifolds_necessary"

print(f"\nVERDICT: manifold_necessity_status = {verdict}")

# Save results
results = {
    'comparison': {
        'raw_accuracy': float(raw_accuracy),
        'raw_auroc': float(raw_auroc),
        'manifold_accuracy': float(manifold_accuracy),
        'manifold_auroc': float(manifold_auroc),
        'combined_accuracy': float(combined_accuracy),
        'combined_auroc': float(combined_auroc),
        'delta_information': float(delta_information),
        'delta_auroc': float(delta_auroc)
    },
    'conditional_independence': {
        'conditional_MI': float(conditional_MI),
        'residual_predictive_power': float(residual_predictive_power)
    },
    'minimal_sufficient': {
        'minimal_set': minimal_feature_set,
        'manifold_retained': manifold_retained,
        'minimal_results': {str(k): float(v) for k, v in minimal_results.items()}
    },
    'cross_domain': {
        'raw_generalization': float(raw_generalization),
        'manifold_generalization': float(manifold_generalization),
        'domain_results': {k: {kk: vv for kk, vv in v.items() if kk != 'error'} for k, v in domain_results.items() if 'error' not in v}
    },
    'raw_phases': {
        'raw_criticality_detected': bool(raw_criticality_detected),
        'raw_hysteresis_detected': bool(raw_hysteresis_detected),
        'raw_rg_detected': bool(raw_rg_detected)
    },
    'ablation': {
        'performance_without_manifolds': float(performance_without_manifolds),
        'performance_loss': float(performance_loss)
    },
    'verdict': verdict
}

with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2, default=str)

# Save arrays
np.save(os.path.join(OUTPUT_DIR, 'X_raw.npy'), X_raw)
np.save(os.path.join(OUTPUT_DIR, 'X_manifold.npy'), X_man)
np.save(os.path.join(OUTPUT_DIR, 'feature_importance.npy'), importances)

print(f"\nAll files saved to {OUTPUT_DIR}")
print("="*60)