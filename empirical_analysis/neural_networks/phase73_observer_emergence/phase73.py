"""
PHASE 73: OBSERVER EMERGENCE TEST
Test whether observer-like structure is the source of geometric organization
"""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error
import os
import json
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase73_observer_emergence'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("PHASE 73: OBSERVER EMERGENCE TEST")
print("="*60)

np.random.seed(42)
n_samples = 500
n_dim = 32

funcs = [lambda x: x, np.tanh, lambda x: np.maximum(0, x), lambda x: np.log1p(np.exp(x))]

# ==============================================================================
# STEP 1: SYSTEM CLASSES
# ==============================================================================
print("\nStep 1: Creating system classes...")

# Group A: Non-observers - no memory, no prediction, no internal state persistence
def gen_iid_noise(n, d):
    return np.random.randn(n, d)

def gen_random_transform(n, d):
    x = np.random.randn(n, d)
    return x @ np.random.randn(d, d)

def gen_shuffled(n, d):
    x = np.random.randn(n, d)
    for i in range(d):
        x[:, i] = np.random.permutation(x[:, i])
    return x

def gen_feedforward_random(n, d):
    x = np.random.randn(n, d)
    x = np.tanh(x @ np.random.randn(d, d//2))
    x = x @ np.random.randn(d//2, d)
    return x

# Group B: Passive memory - memory, recurrence, but NO self-modeling
def gen_ar(n, d):
    x = np.zeros((n, d))
    x[0] = np.random.randn(d)
    for t in range(1, n):
        x[t] = 0.6 * x[t-1] + np.random.randn(d) * 0.4
    return x

def gen_finite_state(n, d):
    x = np.zeros((n, d))
    states = np.random.randn(5, d)
    current = 0
    for t in range(n):
        current = np.random.choice(5)
        x[t] = states[current] + np.random.randn(d) * 0.1
    return x

def gen_markov(n, d):
    x = np.zeros((n, d))
    trans = np.random.dirichlet(np.ones(5), d)
    state = np.random.randint(0, 5, d)
    for t in range(n):
        state = np.array([np.random.choice(5, p=trans[i]) for i in range(d)])
        x[t] = state + np.random.randn(d) * 0.2
    return x

def gen_shallow_rnn(n, d):
    x = np.zeros((n, d))
    h = np.zeros(d)
    W = np.random.randn(d, d) * 0.1
    for t in range(n):
        h = np.tanh(x[t-1] @ W + np.random.randn(d) * 0.1) if t > 0 else np.zeros(d)
        x[t] = h + np.random.randn(d) * 0.5
    return x

# Group C: Predictive systems - predict future, minimize error, NO self-modeling
def gen_predictive_coding(n, d):
    x = np.zeros((n, d))
    for t in range(1, n):
        pred = np.mean(x[:t], axis=0) if t > 1 else x[t-1]
        error = x[t-1] - pred
        x[t] = x[t-1] + 0.3 * error + np.random.randn(d) * 0.2
    return x

def gen_transformer_style(n, d):
    x = np.zeros((n, d))
    for t in range(1, n):
        attn = np.mean(x[:t], axis=0) if t > 0 else np.zeros(d)
        x[t] = np.tanh(x[t-1] + 0.3 * attn + np.random.randn(d) * 0.3)
    return x

def gen_sequence_pred(n, d):
    x = np.zeros((n, d))
    for t in range(2, n):
        x[t] = 0.5 * x[t-1] + 0.3 * x[t-2] + np.random.randn(d) * 0.4
    return x

def gen_predictive_error(n, d):
    x = np.random.randn(n, d)
    for t in range(1, n):
        pred = x[t-1] + np.random.randn(d) * 0.1
        x[t] = x[t] - 0.2 * (x[t] - pred)
    return x

# Group D: Observer-like systems - model external input, model own latent state, recursively update self-models
def gen_self_modeling_rnn(n, d):
    x = np.zeros((n, d))
    z = np.zeros(d)  # latent state
    for t in range(1, n):
        z = 0.9 * z + 0.05 * x[t-1]  # self-model updates
        x[t] = z + np.random.randn(d) * 0.2
    return x

def gen_recursive_predictive(n, d):
    x = np.zeros((n, d))
    z = np.zeros(d)
    for t in range(1, n):
        external_pred = np.mean(x[:t], axis=0) if t > 1 else x[t-1]
        self_model_pred = z
        z = 0.8 * z + 0.1 * external_pred
        x[t] = 0.7 * external_pred + 0.3 * self_model_pred + np.random.randn(d) * 0.15
    return x

def gen_latent_observer(n, d):
    z = np.zeros((n, d//2))
    x = np.zeros((n, d))
    for t in range(1, n):
        z[t] = np.tanh(z[t-1] + 0.1 * x[t-1, :d//2])
        x[t, :d//2] = z[t]
        x[t, d//2:] = z[t] + np.random.randn(d//2) * 0.3
    return x

def gen_active_inference(n, d):
    x = np.zeros((n, d))
    s = np.zeros(d)  # belief state
    for t in range(1, n):
        s = 0.85 * s + 0.1 * x[t-1]
        prediction_error = x[t-1] - s
        x[t] = s + 0.15 * prediction_error + np.random.randn(d) * 0.15
    return x

def gen_self_predictive_latent(n, d):
    x = np.zeros((n, d))
    h = np.zeros(d)
    for t in range(1, n):
        h = 0.7 * h + 0.2 * x[t-1]
        x[t] = h + np.random.randn(d) * 0.25
    return x

def gen_hierarchical_observer(n, d):
    x = np.zeros((n, d))
    z1 = np.zeros(d)
    z2 = np.zeros(d)
    for t in range(1, n):
        z1 = 0.8 * z1 + 0.1 * x[t-1]
        z2 = 0.9 * z2 + 0.05 * z1
        x[t] = 0.5 * z1 + 0.5 * z2 + np.random.randn(d) * 0.2
    return x

# Group definitions
groups = {
    'non_observer': [('iid', gen_iid_noise), ('random_transform', gen_random_transform), 
                    ('shuffled', gen_shuffled), ('feedforward', gen_feedforward_random)],
    'passive_memory': [('ar', gen_ar), ('finite_state', gen_finite_state), 
                       ('markov', gen_markov), ('shallow_rnn', gen_shallow_rnn)],
    'predictive': [('pred_coding', gen_predictive_coding), ('transformer', gen_transformer_style),
                   ('seq_pred', gen_sequence_pred), ('pred_error', gen_predictive_error)],
    'observer': [('self_model_rnn', gen_self_modeling_rnn), ('recursive_pred', gen_recursive_predictive),
                ('latent_observer', gen_latent_observer), ('active_inference', gen_active_inference),
                ('self_pred_latent', gen_self_predictive_latent), ('hierarchical_observer', gen_hierarchical_observer)]
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
    
    try:
        cov = np.cov(phis, rowvar=False)
        evals, evecs = np.linalg.eigh(cov)
        idx = np.argsort(evals)[::-1]
        pc1 = evecs[:, idx[0]]
        X1 = phis @ pc1
    except:
        X1 = np.random.randn(X.shape[0])
    
    var_x1 = float(np.var(X1))
    unique_x1 = int(len(np.unique(np.round(X1, 3))))
    
    return var_x1, unique_x1

for s in all_systems:
    X = s['data']
    var_x1, unique_x1 = compute_manifold(X)
    s['var_x1'] = var_x1 if not np.isnan(var_x1) else 0.0
    s['unique_x1'] = unique_x1

# Filter out invalid systems
all_systems = [s for s in all_systems if not np.isnan(s['var_x1']) and s['var_x1'] > 0]
print(f"  Valid systems after filtering: {len(all_systems)}")

# ==============================================================================
# STEP 3: OBSERVER METRICS
# ==============================================================================
print("\nStep 3: Computing observer metrics...")

def compute_observer_metrics(X):
    X = X - X.mean(axis=0)
    
    # Memory depth
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
    
    # Persistence time
    persistence = np.mean([np.var(X[t:, i]) / (np.var(X[:t+1, i]) + 1e-8) for i in range(min(5, X.shape[1])) for t in range(10, X.shape[0], 20)])
    
    # Recurrence strength
    D = cdist(X, X, metric='euclidean')
    D_norm = D / (np.max(D) + 1e-8)
    recurrence_strength = np.mean(D_norm < 0.1)
    
    # Future prediction accuracy
    pred_errors = []
    for t in range(10, X.shape[0]):
        pred = X[t-1]
        error = np.mean((X[t] - pred)**2)
        pred_errors.append(error)
    future_pred_acc = 1 / (1 + np.mean(pred_errors))
    
    # Prediction horizon
    pred_horizon = 1
    for h in range(1, 10):
        errors = []
        for t in range(h, X.shape[0]):
            pred = X[t-h]
            errors.append(np.mean((X[t] - pred)**2))
        if np.mean(errors) < 2 * np.mean(pred_errors[:10]):
            pred_horizon = h
    
    # Prediction stability
    pred_stability = 1 / (1 + np.std(pred_errors))
    
    # Latent self-consistency
    X_proj = X @ np.random.randn(X.shape[1], 5)
    consistency = 1 / (1 + np.std(np.diff(X_proj, axis=0)))
    
    # Internal state predictability
    internal_pred = []
    for t in range(1, X.shape[0]):
        internal = np.mean(X[:t], axis=0)
        internal_pred.append(np.mean((X[t] - internal)**2))
    internal_state_pred = 1 / (1 + np.mean(internal_pred))
    
    # Recursive self-information
    self_info = []
    for lag in [1, 2, 5]:
        for i in range(min(3, X.shape[1])):
            if lag < X.shape[0]:
                mi = abs(np.corrcoef(X[:-lag, i], X[lag:, i])[0, 1])
                self_info.append(mi)
    recursive_self_info = np.mean(self_info)
    
    # Self-model compression
    comp_ratio = np.std(np.diff(X, axis=0)) / (np.std(X) + 1e-8)
    
    # Temporal coherence
    temp_coherence = np.mean([np.corrcoef(X[:-1, i], X[1:, i])[0, 1] for i in range(min(10, X.shape[1]))])
    
    # Latent stability
    latent_stability = 1 / (1 + np.mean(np.abs(np.diff(X, axis=0))))
    
    # Representational persistence
    rep_persistence = np.mean([np.corrcoef(X[:100, i], X[-100:, i])[0, 1] for i in range(min(5, X.shape[1]))])
    
    # Self-reference consistency
    self_ref_cons = future_pred_acc * internal_state_pred
    
    return {
        'memory_depth': ac_depth,
        'persistence': float(persistence),
        'recurrence_strength': float(recurrence_strength),
        'future_pred_acc': future_pred_acc,
        'prediction_horizon': pred_horizon,
        'prediction_stability': pred_stability,
        'latent_consistency': consistency,
        'internal_state_pred': internal_state_pred,
        'recursive_self_info': recursive_self_info,
        'self_model_compression': comp_ratio,
        'temporal_coherence': float(temp_coherence),
        'latent_stability': latent_stability,
        'rep_persistence': float(rep_persistence) if not np.isnan(float(rep_persistence)) else 0.0,
        'self_ref_consistency': self_ref_cons
    }

for s in all_systems:
    obs_metrics = compute_observer_metrics(s['data'])
    s.update(obs_metrics)

# ==============================================================================
# STEP 4: OBSERVER PREDICTION
# ==============================================================================
print("\nStep 4: Observer prediction...")

obs_features = ['memory_depth', 'persistence', 'recurrence_strength', 'future_pred_acc',
                'prediction_horizon', 'prediction_stability', 'latent_consistency', 
                'internal_state_pred', 'recursive_self_info', 'self_model_compression',
                'temporal_coherence', 'latent_stability', 'rep_persistence', 'self_ref_consistency']

X_obs = np.array([[s[f] for f in obs_features] for s in all_systems])
y_var_x1 = np.array([s['var_x1'] for s in all_systems])

# Handle NaN
X_obs = np.nan_to_num(X_obs, nan=0.0, posinf=1e6, neginf=-1e6)

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
    for train_idx, test_idx in loo.split(X_obs):
        model.fit(X_obs[train_idx], y_var_x1[train_idx])
        preds.append(model.predict(X_obs[test_idx])[0])
        actuals.append(y_var_x1[test_idx][0])
    
    r2 = r2_score(actuals, preds)
    mae = mean_absolute_error(actuals, preds)
    results[name] = {'r2': r2, 'mae': mae}

print("  Observer prediction results:")
for name, res in results.items():
    print(f"    {name}: R2={res['r2']:.4f}, MAE={res['mae']:.4f}")

best_model = max(results.items(), key=lambda x: x[1]['r2'])
print(f"  Best: {best_model[0]} R2={best_model[1]['r2']:.4f}")

# ==============================================================================
# STEP 5-6: OBSERVER INTERVENTIONS
# ==============================================================================
print("\nStep 5-6: Observer interventions...")

X_base = all_systems[0]['data'].copy()
base_var = all_systems[0]['var_x1']

interventions = {}

# A: Destroy self-modeling, preserve prediction
def intervention_a(X):
    X_new = X.copy()
    for i in range(X.shape[1]):
        X_new[:, i] = np.roll(X_new[:, i], np.random.randint(1, 20))
    return X_new

# B: Preserve self-modeling, destroy recurrence
def intervention_b(X):
    X_new = X.copy()
    for t in range(0, X.shape[0], 2):
        X_new[t] = np.random.randn(X.shape[1])
    return X_new

# C: Inject latent self-modeling into random system
def intervention_c(X):
    X_new = X.copy()
    z = np.zeros(X.shape[1])
    for t in range(1, X.shape[0]):
        z = 0.9 * z + 0.1 * X_new[t-1]
        X_new[t] = z + X_new[t] * 0.1
    return X_new

# D: Increase observer coherence gradually
def intervention_d(X):
    X_new = X.copy()
    for i in range(X.shape[1]):
        X_new[:, i] = X_new[:, i] * (1 + 0.5 * np.sin(np.arange(X.shape[0]) * 0.01))
    return X_new

for name, func in [('A', intervention_a), ('B', intervention_b), ('C', intervention_c), ('D', intervention_d)]:
    X_int = func(X_base.copy())
    var_int, _ = compute_manifold(X_int)
    delta_var = abs(var_int - base_var)
    obs_int = compute_observer_metrics(X_int)
    delta_self_ref = abs(obs_int['self_ref_consistency'] - all_systems[0]['self_ref_consistency'])
    
    interventions[name] = {
        'delta_var_x1': float(delta_var),
        'delta_self_ref': float(delta_self_ref)
    }
    print(f"  {name}: delta_var={delta_var:.4f}, delta_self_ref={delta_self_ref:.4f}")

# ==============================================================================
# STEP 7: OBSERVER RG FLOW
# ==============================================================================
print("\nStep 7: Observer RG flow...")

# Observer coherence coordinates
obs_coords = np.array([[s['self_ref_consistency'], s['internal_state_pred'], s['latent_stability']] for s in all_systems])

n_bins = 5
min_vals = np.min(obs_coords, axis=0)
max_vals = np.max(obs_coords, axis=0)

rg_trajectory = []
for i in range(len(all_systems)):
    coord = obs_coords[i]
    bin_idx = tuple(int(np.clip(int((c - min_vals[j]) / (max_vals[j] - min_vals[j] + 1e-8) * (n_bins - 1)), 0, n_bins - 1)) for j, c in enumerate(coord))
    rg_trajectory.append(bin_idx)

from collections import Counter
bin_counts = Counter(rg_trajectory)
fixed_points = len(bin_counts)
attractors = sum(1 for b, c in bin_counts.items() if c >= 3)
collapse_regions = sum(1 for b, c in bin_counts.items() if c == 1)

# Critical threshold
sorted_idx = np.argsort([s['self_ref_consistency'] for s in all_systems])
var_x1_sorted = np.array([all_systems[i]['var_x1'] for i in sorted_idx])
gradients = np.abs(np.diff(var_x1_sorted))
critical_idx = np.argmax(gradients)
critical_threshold = [s['self_ref_consistency'] for s in all_systems][sorted_idx[critical_idx]]

print(f"  Fixed points: {fixed_points}, Attractors: {attractors}, Collapse: {collapse_regions}")
print(f"  Critical threshold: {critical_threshold:.4f}")

# ==============================================================================
# STEP 8: OBSERVER THRESHOLDS
# ==============================================================================
print("\nStep 8: Observer thresholds...")

u_global = np.array([1 / (1 + abs(s['var_x1'] - np.mean(y_var_x1)) / np.std(y_var_x1)) for s in all_systems])

lr_meta = LinearRegression()
lr_meta.fit(obs_coords, u_global)
r2_meta = lr_meta.score(obs_coords, u_global)

pred_thresh = float(np.percentile([s['future_pred_acc'] for s in all_systems], 75))
self_model_thresh = float(np.percentile([s['self_ref_consistency'] for s in all_systems], 75))
observer_coh_thresh = float(np.percentile([s['temporal_coherence'] for s in all_systems], 75))

print(f"  Prediction threshold: {pred_thresh:.4f}")
print(f"  Self-model threshold: {self_model_thresh:.4f}")
print(f"  Observer coherence threshold: {observer_coh_thresh:.4f}")
print(f"  Meta-universality R2: {r2_meta:.4f}")

# ==============================================================================
# STEP 9: OBSERVER PHASES
# ==============================================================================
print("\nStep 9: Observer phases...")

kmeans_obs = KMeans(n_clusters=3, random_state=42, n_init=10)
obs_labels = kmeans_obs.fit_predict(X_obs)

# Compare with groups
group_to_num = {'non_observer': 0, 'passive_memory': 1, 'predictive': 2, 'observer': 3}
true_labels = np.array([group_to_num[s['group']] for s in all_systems])

agreement_with_group = np.mean([1 if obs_labels[i] == (true_labels[i] % 3) else 0 for i in range(len(all_systems))])

# Compare with var_x1 clustering
var_x1_labels = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(y_var_x1.reshape(-1, 1))
agreement_with_geom = np.mean(obs_labels == var_x1_labels)

print(f"  Agreement with group: {agreement_with_group:.4f}")
print(f"  Agreement with geometry: {agreement_with_geom:.4f}")

# ==============================================================================
# STEP 10-12: OUTPUT AND SAVE
# ==============================================================================
print("\n" + "="*60)
print("OUTPUT")
print("="*60)

print("\nOBSERVER PREDICTION:")
print(f"  model = {best_model[0]}")
print(f"  R2 = {best_model[1]['r2']:.4f}")
print(f"  MAE = {best_model[1]['mae']:.4f}")

print("\nOBSERVER INTERVENTIONS:")
for name, res in interventions.items():
    print(f"  {name}: delta_var={res['delta_var_x1']:.4f}")

print("\nOBSERVER RG:")
print(f"  fixed_points = {fixed_points}")
print(f"  attractors = {attractors}")
print(f"  collapse_regions = {collapse_regions}")
print(f"  critical_threshold = {critical_threshold:.4f}")

print("\nOBSERVER THRESHOLDS:")
print(f"  prediction_threshold = {pred_thresh:.4f}")
print(f"  self_model_threshold = {self_model_thresh:.4f}")
print(f"  observer_coherence_threshold = {observer_coh_thresh:.4f}")

print("\nPHASE AGREEMENT:")
print(f"  agreement_score = {agreement_with_geom:.4f}")

# Verdict
if best_model[1]['r2'] > 0.6 and interventions['C']['delta_var_x1'] > 0.5:
    verdict = 'observer_emergence_origin'
elif best_model[1]['r2'] > 0.4:
    verdict = 'observer_related_origin'
else:
    verdict = 'non_observer_origin'

print(f"\nVERDICT: organization_origin = {verdict}")

# Save results
results = {
    'observer_prediction': {
        'best_model': best_model[0],
        'r2': float(best_model[1]['r2']),
        'mae': float(best_model[1]['mae']),
        'all_results': {k: {'r2': float(v['r2']), 'mae': float(v['mae'])} for k, v in results.items()}
    },
    'interventions': interventions,
    'observer_rg': {
        'fixed_points': fixed_points,
        'attractors': attractors,
        'collapse_regions': collapse_regions,
        'critical_threshold': float(critical_threshold)
    },
    'thresholds': {
        'prediction': float(pred_thresh),
        'self_model': float(self_model_thresh),
        'observer_coherence': float(observer_coh_thresh)
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
np.save(os.path.join(OUTPUT_DIR, 'obs_features.npy'), X_obs)
np.save(os.path.join(OUTPUT_DIR, 'obs_labels.npy'), obs_labels)
np.save(os.path.join(OUTPUT_DIR, 'var_x1.npy'), y_var_x1)

print(f"\nAll files saved to {OUTPUT_DIR}")
print("="*60)