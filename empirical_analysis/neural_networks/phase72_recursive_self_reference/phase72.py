"""
PHASE 72: RECURSIVE SELF-REFERENCE TEST
Test whether manifold organization is driven by recursive self-reference
"""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import LeaveOneOut
import os
import json
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase72_recursive_self_reference'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("PHASE 72: RECURSIVE SELF-REFERENCE TEST")
print("="*60)

np.random.seed(42)
n_samples = 500
n_dim = 32

funcs = [lambda x: x, np.tanh, lambda x: np.maximum(0, x), lambda x: np.log1p(np.exp(x))]

# ==============================================================================
# STEP 1: BASE SYSTEMS - CONTROLLED RECURSION
# ==============================================================================
print("\nStep 1: Creating systems with controlled recursion...")

def gen_gaussian(n, d):
    return np.random.randn(n, d)

def gen_laplace(n, d):
    return np.random.laplace(0, 1, (n, d))

def gen_shuffled(n, d):
    x = np.random.randn(n, d)
    for i in range(d):
        x[:, i] = np.random.permutation(x[:, i])
    return x

def gen_low_ar(n, d):
    x = np.zeros((n, d))
    x[0] = np.random.randn(d)
    for t in range(1, n):
        x[t] = 0.5 * x[t-1] + np.random.randn(d) * 0.5
    return x

def gen_shallow_rnn(n, d):
    x = np.zeros((n, d))
    h = np.zeros(d)
    W = np.random.randn(d, d) * 0.1
    for t in range(n):
        h = np.tanh(x[t-1] @ W + np.random.randn(d) * 0.1) if t > 0 else np.zeros(d)
        x[t] = h + np.random.randn(d) * 0.5
    return x

def gen_markov_chain(n, d):
    x = np.zeros((n, d))
    probs = np.random.dirichlet(np.ones(5), d)
    state = np.random.randint(0, 5, d)
    for t in range(n):
        state = np.array([np.random.choice(5, p=probs[i]) for i in range(d)])
        x[t] = state + np.random.randn(d) * 0.3
    return x

def gen_deep_recurrence(n, d):
    x = np.zeros((n, d))
    h = np.zeros(d)
    for layer in range(3):
        W = np.random.randn(d, d) * 0.3
        for t in range(n):
            h = np.tanh(x[t-3] @ W if t >= 3 else np.zeros(d))
            x[t] = h + np.random.randn(d) * 0.3
    return x

def gen_recursive_ca(n, d):
    x = np.random.randn(n, d)
    for t in range(2, n):
        x[t] = np.tanh(0.5 * x[t-1] - 0.3 * x[t-2] + np.random.randn(d) * 0.2)
    return x

def gen_self_predictive(n, d):
    x = np.zeros((n, d))
    for t in range(1, n):
        x[t] = x[t-1] + np.random.randn(d) * 0.3
        pred_error = x[t] - x[t-1]
        x[t] = x[t] - 0.1 * pred_error
    return x

def gen_nested_ar(n, d):
    x = np.zeros((n, d))
    for t in range(3, n):
        x[t] = 0.4 * x[t-1] + 0.3 * x[t-2] + 0.2 * x[t-3] + np.random.randn(d) * 0.4
    return x

def gen_transformer_like(n, d):
    x = np.zeros((n, d))
    for t in range(1, n):
        attn = np.mean(x[:t], axis=0) if t > 0 else np.zeros(d)
        x[t] = np.tanh(x[t-1] + 0.3 * attn + np.random.randn(d) * 0.3)
    return x

def gen_self_modeling(n, d):
    x = np.zeros((n, d))
    latent = np.zeros(d)
    for t in range(1, n):
        latent = 0.8 * latent + 0.1 * x[t-1]
        x[t] = latent + np.random.randn(d) * 0.2
    return x

def gen_min_self_pred_error(n, d):
    x = np.random.randn(n, d)
    for t in range(1, n):
        pred = np.mean(x[:t], axis=0) if t > 1 else x[t-1]
        x[t] = x[t] - 0.2 * (x[t] - pred)
    return x

def gen_internal_latent(n, d):
    z = np.zeros((n, d//2))
    x = np.zeros((n, d))
    for t in range(1, n):
        z[t] = np.tanh(z[t-1] + np.random.randn(d//2) * 0.1)
        x[t, :d//2] = z[t]
        x[t, d//2:] = z[t] + np.random.randn(d//2) * 0.3
    return x

# Group definitions
groups = {
    'non_recursive': [('gaussian', gen_gaussian), ('laplace', gen_laplace), ('shuffled', gen_shuffled)],
    'weakly_recursive': [('low_ar', gen_low_ar), ('shallow_rnn', gen_shallow_rnn), ('markov', gen_markov_chain)],
    'strongly_recursive': [('deep_rec', gen_deep_recurrence), ('recursive_ca', gen_recursive_ca), 
                           ('self_pred', gen_self_predictive), ('nested_ar', gen_nested_ar), ('transformer', gen_transformer_like)],
    'self_reference': [('self_model', gen_self_modeling), ('min_error', gen_min_self_pred_error), ('internal_latent', gen_internal_latent)]
}

# Generate systems
all_systems = []
for group_name, generators in groups.items():
    for sys_name, gen_func in generators:
        for trial in range(5):
            try:
                X = gen_func(n_samples, n_dim)
                all_systems.append({
                    'system': sys_name,
                    'group': group_name,
                    'trial': trial,
                    'data': X
                })
            except:
                continue

print(f"  Total systems: {len(all_systems)}")
print(f"  Groups: {[(g, len([s for s in all_systems if s['group']==g])) for g in groups.keys()]}")

# ==============================================================================
# STEP 2: MANIFOLD PIPELINE
# ==============================================================================
print("\nStep 2: Computing manifold metrics...")

def compute_manifold(X):
    X = X - X.mean(axis=0)
    
    phis = []
    for f in funcs:
        t = f(X)
        phis.append(np.mean(t, axis=0))
    phis = np.array(phis)
    
    cov = np.cov(phis, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    idx = np.argsort(evals)[::-1]
    pc1 = evecs[:, idx[0]]
    X1 = phis @ pc1
    
    var_x1 = float(np.var(X1))
    unique_x1 = int(len(np.unique(np.round(X1, 3))))
    
    cov_orig = np.cov(X, rowvar=False)
    try:
        evals_orig = np.sort(np.linalg.eigvalsh(cov_orig))[::-1]
        evals_orig = evals_orig[evals_orig > 1e-8]
        cumvar = np.cumsum(evals_orig / np.sum(evals_orig))
        latent_dim = int(np.argmax(cumvar >= 0.95) + 1) if np.sum(evals_orig) > 0 else 1
    except:
        latent_dim = 1
    
    return var_x1, unique_x1, latent_dim

for s in all_systems:
    X = s['data']
    var_x1, unique_x1, lat_dim = compute_manifold(X)
    s['var_x1'] = var_x1
    s['unique_x1'] = unique_x1
    s['latent_dimension'] = lat_dim

# ==============================================================================
# STEP 3: RECURSION METRICS
# ==============================================================================
print("\nStep 3: Computing recursion metrics...")

def compute_recursion_metrics(X):
    X = X - X.mean(axis=0)
    
    # Recurrence depth: autocorrelation decay
    ac_depth = 5
    for lag in range(1, 30):
        ac_vals = []
        for i in range(min(10, X.shape[1])):
            if lag < X.shape[0]:
                ac = np.corrcoef(X[:-lag, i], X[lag:, i])[0, 1]
                ac_vals.append(abs(ac) if not np.isnan(ac) else 0)
        if np.mean(ac_vals) < 0.3:
            ac_depth = lag
            break
    
    # Feedback depth: how far back needed for prediction
    feedback_depth = int(np.argmax([
        np.mean([np.corrcoef(X[t:, i], X[:-t, i])[0, 1] for i in range(min(10, X.shape[1])) if t < X.shape[0]])
        for t in range(1, 20)
    ]) + 1)
    
    # Self-prediction accuracy
    pred_errors = []
    for t in range(10, X.shape[0]):
        pred = X[t-1]
        error = np.mean((X[t] - pred)**2)
        pred_errors.append(error)
    self_pred_acc = 1 / (1 + np.mean(pred_errors))
    
    # Recursive mutual information
    I_future_past = []
    for lag in [1, 2, 5, 10]:
        mi = 0
        for i in range(min(5, X.shape[1])):
            if lag < X.shape[0]:
                x_curr = X[:-lag, i]
                x_next = X[lag:, i]
                if len(x_curr) > 10:
                    mi += abs(np.corrcoef(x_curr, x_next)[0, 1])
        I_future_past.append(mi / min(5, X.shape[1]))
    recursive_mi = np.mean(I_future_past)
    
    # Hierarchical recursion depth
    nested_dep = 0
    for lag in [1, 2, 3, 5, 10]:
        ac = np.mean([np.corrcoef(X[:-lag, i], X[lag:, i])[0, 1] for i in range(min(5, X.shape[1])) if lag < X.shape[0]])
        if abs(ac) > 0.2:
            nested_dep = max(nested_dep, lag)
    
    # Latent self-model consistency
    X_proj = X @ np.random.randn(X.shape[1], 5)
    consistency = 1 / (1 + np.std(np.diff(X_proj, axis=0)))
    
    return {
        'recursion_depth': ac_depth,
        'feedback_depth': feedback_depth,
        'self_pred_accuracy': self_pred_acc,
        'recursive_mi': recursive_mi,
        'hierarchical_depth': nested_dep,
        'latent_consistency': consistency
    }

for s in all_systems:
    rec_metrics = compute_recursion_metrics(s['data'])
    s.update(rec_metrics)

# ==============================================================================
# STEP 4: RECURSION PREDICTION
# ==============================================================================
print("\Step 4: Recursion prediction...")

rec_features = ['recursion_depth', 'feedback_depth', 'self_pred_accuracy', 
                'recursive_mi', 'hierarchical_depth', 'latent_consistency']

X_rec = np.array([[s[f] for f in rec_features] for s in all_systems])
y_var_x1 = np.array([s['var_x1'] for s in all_systems])

loo = LeaveOneOut()
results = {}

for name, model in [
    ('linear', LinearRegression()),
    ('random_forest', RandomForestRegressor(n_estimators=50, random_state=42)),
    ('gradient_boosting', GradientBoostingRegressor(n_estimators=50, random_state=42)),
    ('ridge', Ridge(alpha=1.0))
]:
    preds = []
    actuals = []
    for train_idx, test_idx in loo.split(X_rec):
        model.fit(X_rec[train_idx], y_var_x1[train_idx])
        preds.append(model.predict(X_rec[test_idx])[0])
        actuals.append(y_var_x1[test_idx][0])
    
    r2 = r2_score(actuals, preds)
    mae = mean_absolute_error(actuals, preds)
    results[name] = {'r2': r2, 'mae': mae}

print("  Recursion prediction results:")
for name, res in results.items():
    print(f"    {name}: R2={res['r2']:.4f}, MAE={res['mae']:.4f}")

best_model = max(results.items(), key=lambda x: x[1]['r2'])
print(f"  Best: {best_model[0]} R2={best_model[1]['r2']:.4f}")

# ==============================================================================
# STEP 5-6: SELF-REFERENCE INTERVENTIONS
# ==============================================================================
print("\nStep 5-6: Self-reference interventions...")

X_base = all_systems[0]['data'].copy()
base_var = all_systems[0]['var_x1']

interventions = {}

# Intervention A: Destroy self-reference, preserve recurrence
def intervention_a(X):
    X_new = X.copy()
    for i in range(X.shape[1]):
        X_new[:, i] = np.roll(X_new[:, i], np.random.randint(1, 10))
    return X_new

# Intervention B: Preserve self-reference, destroy long-range stats
def intervention_b(X):
    X_new = X.copy()
    for i in range(0, X.shape[0], 2):
        X_new[i] = np.random.randn(X.shape[1])
    return X_new

# Intervention C: Inject recursive self-modeling
def intervention_c(X):
    X_new = X.copy()
    for t in range(1, X.shape[0]):
        X_new[t] = 0.8 * X_new[t-1] + 0.1 * np.mean(X_new[:t], axis=0) + np.random.randn(X.shape[1]) * 0.2
    return X_new

# Intervention D: Hierarchical self-reference at multiple scales
def intervention_d(X):
    X_new = X.copy()
    for t in range(5, X.shape[0]):
        X_new[t] = 0.3 * X_new[t-1] + 0.2 * X_new[t-5] + 0.1 * np.mean(X_new[t-10:t], axis=0) + np.random.randn(X.shape[1]) * 0.3
    return X_new

for name, func in [('A', intervention_a), ('B', intervention_b), ('C', intervention_c), ('D', intervention_d)]:
    X_int = func(X_base.copy())
    var_int, _, _ = compute_manifold(X_int)
    delta_var = abs(var_int - base_var)
    rec_int = compute_recursion_metrics(X_int)
    delta_rec = abs(rec_int['self_pred_accuracy'] - all_systems[0]['self_pred_accuracy'])
    
    interventions[name] = {
        'delta_var_x1': float(delta_var) if not np.isnan(delta_var) else 0.0,
        'delta_recursion': float(delta_rec) if not np.isnan(delta_rec) else 0.0
    }
    print(f"  {name}: delta_var={delta_var:.4f}, delta_rec={delta_rec:.4f}")

# ==============================================================================
# STEP 7: RECURSIVE RG FLOW
# ==============================================================================
print("\nStep 7: Recursive RG flow...")

# Use recursion depth and self-prediction as coordinates
rec_coords = np.array([[s['recursion_depth'], s['self_pred_accuracy'], s['recursive_mi']] for s in all_systems])

n_bins = 5
min_vals = np.min(rec_coords, axis=0)
max_vals = np.max(rec_coords, axis=0)

rg_trajectory = []
for i in range(len(all_systems)):
    coord = rec_coords[i]
    bin_idx = tuple(int(np.clip(int((c - min_vals[j]) / (max_vals[j] - min_vals[j] + 1e-8) * (n_bins - 1)), 0, n_bins - 1)) for j, c in enumerate(coord))
    rg_trajectory.append(bin_idx)

from collections import Counter
bin_counts = Counter(rg_trajectory)
fixed_points = len(bin_counts)
attractors = sum(1 for b, c in bin_counts.items() if c >= 3)
collapse_regions = sum(1 for b, c in bin_counts.items() if c == 1)

# Find critical threshold
rec_depths = [s['recursion_depth'] for s in all_systems]
sorted_idx = np.argsort([s['self_pred_accuracy'] for s in all_systems])
var_x1_sorted = np.array([all_systems[i]['var_x1'] for i in sorted_idx])
gradients = np.abs(np.diff(var_x1_sorted))
critical_idx = np.argmax(gradients)
critical_threshold = [s['self_pred_accuracy'] for s in all_systems][sorted_idx[critical_idx]]

print(f"  Fixed points: {fixed_points}, Attractors: {attractors}, Collapse: {collapse_regions}")
print(f"  Critical threshold: {critical_threshold:.4f}")

# ==============================================================================
# STEP 8: META-UNIVERSALITY
# ==============================================================================
print("\nStep 8: Meta-universality...")

# Define U_global
u_global = np.array([1 / (1 + abs(s['var_x1'] - np.mean(y_var_x1)) / np.std(y_var_x1)) for s in all_systems])

# Fit U ~ recursion
lr_meta = LinearRegression()
lr_meta.fit(rec_coords, u_global)
r2_meta = lr_meta.score(rec_coords, u_global)

recursion_thresh = float(np.percentile([s['recursion_depth'] for s in all_systems], 75))
self_ref_thresh = float(np.percentile([s['self_pred_accuracy'] for s in all_systems], 75))

print(f"  Recursion threshold: {recursion_thresh:.4f}")
print(f"  Self-reference threshold: {self_ref_thresh:.4f}")
print(f"  Meta-universality R2: {r2_meta:.4f}")

# ==============================================================================
# STEP 9: RECURSIVE PHASES
# ==============================================================================
print("\nStep 9: Recursive phases...")

kmeans_rec = KMeans(n_clusters=3, random_state=42, n_init=10)
rec_labels = kmeans_rec.fit_predict(X_rec)

# Compare with groups
group_to_num = {'non_recursive': 0, 'weakly_recursive': 1, 'strongly_recursive': 2, 'self_reference': 3}
true_labels = np.array([group_to_num[s['group']] for s in all_systems])

# Agreement with ground truth
agreement_with_group = np.mean([1 if rec_labels[i] == (true_labels[i] % 3) else 0 for i in range(len(all_systems))])

# Compare with var_x1 clustering
var_x1_labels = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(y_var_x1.reshape(-1, 1))
agreement_with_geom = np.mean(rec_labels == var_x1_labels)

print(f"  Agreement with group: {agreement_with_group:.4f}")
print(f"  Agreement with geometry: {agreement_with_geom:.4f}")

# ==============================================================================
# STEP 10-12: OUTPUT AND SAVE
# ==============================================================================
print("\n" + "="*60)
print("OUTPUT")
print("="*60)

print("\nRECURSION PREDICTION:")
print(f"  model = {best_model[0]}")
print(f"  R2 = {best_model[1]['r2']:.4f}")
print(f"  MAE = {best_model[1]['mae']:.4f}")

print("\nSELF-REFERENCE INTERVENTIONS:")
for name, res in interventions.items():
    print(f"  {name}: delta_var={res['delta_var_x1']:.4f}")

print("\nRECURSIVE RG:")
print(f"  fixed_points = {fixed_points}")
print(f"  attractors = {attractors}")
print(f"  collapse_regions = {collapse_regions}")
print(f"  critical_threshold = {critical_threshold:.4f}")

print("\nMETA-UNIVERSALITY:")
print(f"  recursion_threshold = {recursion_thresh:.4f}")
print(f"  self_reference_threshold = {self_ref_thresh:.4f}")

print("\nPHASE AGREEMENT:")
print(f"  agreement_score = {agreement_with_geom:.4f}")

# Verdict
if best_model[1]['r2'] > 0.6 and interventions['D']['delta_var_x1'] > 0.5:
    verdict = 'recursive_self_reference_origin'
elif best_model[1]['r2'] > 0.4:
    verdict = 'recursive_computational_origin'
else:
    verdict = 'non_recursive_origin'

print(f"\nVERDICT: organization_origin = {verdict}")

# Save results
results = {
    'recursion_prediction': {
        'best_model': best_model[0],
        'r2': float(best_model[1]['r2']),
        'mae': float(best_model[1]['mae']),
        'all_results': {k: {'r2': float(v['r2']), 'mae': float(v['mae'])} for k, v in results.items()}
    },
    'interventions': interventions,
    'recursive_rg': {
        'fixed_points': fixed_points,
        'attractors': attractors,
        'collapse_regions': collapse_regions,
        'critical_threshold': float(critical_threshold)
    },
    'meta_universality': {
        'recursion_threshold': recursion_thresh,
        'self_reference_threshold': self_ref_thresh,
        'r2': float(r2_meta)
    },
    'phase_agreement': {
        'with_group': float(agreement_with_group),
        'with_geometry': float(agreement_with_geom)
    },
    'verdict': verdict
}

with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2)

# Save arrays
np.save(os.path.join(OUTPUT_DIR, 'rec_features.npy'), X_rec)
np.save(os.path.join(OUTPUT_DIR, 'rec_labels.npy'), rec_labels)
np.save(os.path.join(OUTPUT_DIR, 'var_x1.npy'), y_var_x1)

print(f"\nAll files saved to {OUTPUT_DIR}")
print("="*60)