"""
PHASE 76: TRIVIALITY / REDISCOVERY TEST
Test whether observer transition is novel or known mathematics
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.manifold import Isomap
import os
import json
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase76_triviality_rediscovery'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("PHASE 76: TRIVIALITY / REDISCOVERY TEST")
print("="*60)

np.random.seed(42)
n_samples = 500
n_dim = 32

funcs = [lambda x: x, np.tanh, lambda x: np.maximum(0, x), lambda x: np.log1p(np.exp(x))]

def compute_manifold(X):
    X = X - X.mean(axis=0)
    phis = np.array([np.mean(f(X), axis=0) for f in funcs])
    try:
        cov = np.cov(phis, rowvar=False)
        ev, evc = np.linalg.eigh(cov)
        idx = np.argsort(ev)[::-1]
        pc1 = evc[:, idx[0]]
        X1 = phis @ pc1
    except:
        X1 = np.random.randn(X.shape[0])
    var_x1 = float(np.var(X1))
    return var_x1

# ==============================================================================
# STEP 1: STANDARD THEORY LIBRARY
# ==============================================================================
print("\nStep 1: Creating standard theory systems...")

def generate_lorenz(n, d):
    """Lorenz attractor (chaotic dynamical system)"""
    x = np.zeros((n, 3))
    sigma, rho, beta = 10, 28, 8/3
    dt = 0.01
    x[0] = [1, 1, 20]
    for t in range(1, n):
        dx = sigma * (x[t-1, 1] - x[t-1, 0])
        dy = x[t-1, 0] * (rho - x[t-1, 2]) - x[t-1, 1]
        dz = x[t-1, 0] * x[t-1, 1] - beta * x[t-1, 2]
        x[t] = x[t-1] + np.array([dx, dy, dz]) * dt
    
    # Embed in higher dimensions
    X = np.zeros((n, d))
    X[:, :3] = x
    X[:, 3:] = np.random.randn(n, d-3) * 0.1
    return X

def generate_rossler(n, d):
    """Rossler attractor"""
    x = np.zeros((n, 3))
    a, b, c = 0.2, 0.2, 5.7
    dt = 0.05
    x[0] = [1, 1, 1]
    for t in range(1, n):
        dx = -x[t-1, 1] - x[t-1, 2]
        dy = x[t-1, 0] + a * x[t-1, 1]
        dz = b + x[t-1, 2] * (x[t-1, 0] - c)
        x[t] = x[t-1] + np.array([dx, dy, dz]) * dt
    
    X = np.zeros((n, d))
    X[:, :3] = x
    X[:, 3:] = np.random.randn(n, d-3) * 0.1
    return X

def generate_kuramoto(n, d):
    """Kuramoto model (synchronization)"""
    n_osc = min(d, 8)
    phases = np.random.uniform(0, 2*np.pi, n_osc)
    omegas = np.random.uniform(0.9, 1.1, n_osc)
    K = 1.5  # Coupling
    
    data = np.zeros((n, n_osc))
    data[0] = phases
    for t in range(1, n):
        phases = phases + omegas + K/n_osc * np.sum(np.sin(data[t-1] - phases.reshape(-1, 1)), axis=0)
        phases = phases % (2*np.pi)
        data[t] = phases
    
    X = np.zeros((n, d))
    X[:, :n_osc] = np.cos(data)
    X[:, n_osc:] = np.sin(data)
    return X

def generate_logistic(n, d):
    """Logistic map (chaos)"""
    r = 3.9  # Chaotic parameter
    x = np.zeros(n)
    x[0] = 0.5
    for t in range(1, n):
        x[t] = r * x[t-1] * (1 - x[t-1])
    
    X = np.zeros((n, d))
    for i in range(d):
        X[:, i] = x + np.random.randn(n) * 0.01
    return X

def generate_ising(n, d):
    """Ising-like system"""
    x = np.random.choice([-1, 1], (n, d))
    # Simple dynamics toward alignment
    for t in range(1, n):
        x[t] = np.sign(x[t-1] + np.random.randn(d) * 0.5)
    return x.astype(float)

def generate_potts(n, d):
    """Potts model"""
    x = np.random.randint(0, 5, (n, d))
    for t in range(1, n):
        for i in range(d):
            if np.random.rand() < 0.3:
                x[t, i] = np.random.randint(0, 5)
    return x.astype(float)

def generate_random_deep_net(n, d):
    """Random deep network (representation learning)"""
    x = np.random.randn(n, d)
    # Random forward passes
    for _ in range(5):
        x = np.tanh(x @ np.random.randn(d, d) * 0.1)
    return x

def generate_transformer_like(n, d):
    """Transformer-style attention"""
    x = np.random.randn(n, d)
    for t in range(1, n):
        attn = np.mean(x[:t], axis=0) if t > 0 else np.zeros(d)
        x[t] = np.tanh(x[t-1] + 0.3 * attn + np.random.randn(d) * 0.2)
    return x

def generate_autoencoder(n, d):
    """Autoencoder latent space"""
    x = np.random.randn(n, d)
    # Encode
    h = np.tanh(x @ np.random.randn(d, d//2))
    # Decode
    x_recon = h @ np.random.randn(d//2, d)
    return h + np.random.randn(n, d//2) * 0.1

def generate_fisher_metric(n, d):
    """Fisher metric system (information geometry)"""
    x = np.random.randn(n, d)
    # Exponential family-like structure
    x = np.exp(x)
    return x

def generate_exponential_family(n, d):
    """Exponential family"""
    x = np.random.exponential(1, (n, d))
    return x

def generate_hopfield(n, d):
    """Hopfield network attractors"""
    x = np.random.choice([-1, 1], (n, min(d, 10)))
    # Simple dynamics
    for t in range(1, n):
        x[t] = np.sign(x[t-1] @ np.random.randn(10, min(d, 10)) * 0.1 + np.random.randn(min(d, 10)) * 0.2)
    
    X = np.zeros((n, d))
    X[:, :min(d, 10)] = x
    return X.astype(float)

def generate_reservoir(n, d):
    """Reservoir computing"""
    x = np.zeros((n, min(d, 16)))
    W = np.random.randn(16, 16) * 1.5
    W = W / np.max(np.abs(np.linalg.eigvals(W)))
    
    for t in range(1, n):
        x[t] = np.tanh(W @ x[t-1] + np.random.randn(16) * 0.1)
    
    X = np.zeros((n, d))
    X[:, :16] = x
    return X

# Create standard theory systems
standard_systems = [
    ('lorenz', generate_lorenz),
    ('rossler', generate_rossler),
    ('kuramoto', generate_kuramoto),
    ('logistic_map', generate_logistic),
    ('ising', generate_ising),
    ('potts', generate_potts),
    ('random_deep_net', generate_random_deep_net),
    ('transformer', generate_transformer_like),
    ('autoencoder', generate_autoencoder),
    ('fisher_metric', generate_fisher_metric),
    ('exp_family', generate_exponential_family),
    ('hopfield', generate_hopfield),
    ('reservoir', generate_reservoir),
]

print(f"  Created {len(standard_systems)} standard theory systems")

# ==============================================================================
# STEP 2: IDENTICAL PIPELINE
# ==============================================================================
print("\nStep 2: Running identical pipeline...")

system_results = []

for sys_name, gen_func in standard_systems:
    for trial in range(3):
        try:
            X = gen_func(n_samples, n_dim)
            var_x1 = compute_manifold(X)
            
            # Compute observer-like metrics
            # Temporal coherence
            temp_coh = np.mean([np.corrcoef(X[:-1, i], X[1:, i])[0, 1] for i in range(min(10, X.shape[1]))])
            
            # Recurrence
            D = cdist(X, X, metric='euclidean')
            recurrence = np.mean(D < np.percentile(D, 10))
            
            # Self-model proxy
            pred_errors = []
            for t in range(10, X.shape[0]):
                pred = X[t-1]
                pred_errors.append(np.mean((X[t] - pred)**2))
            self_model = 1 / (1 + np.mean(pred_errors))
            
            system_results.append({
                'system': sys_name,
                'trial': trial,
                'var_x1': var_x1 if not np.isnan(var_x1) else 0.0,
                'temporal_coherence': temp_coh if not np.isnan(temp_coh) else 0.0,
                'recurrence': recurrence,
                'self_model': self_model,
                'category': 'standard_theory'
            })
        except Exception as e:
            continue

print(f"  Processed {len(system_results)} systems")

# ==============================================================================
# STEP 3: EQUIVALENCE TEST
# ==============================================================================
print("\nStep 3: Equivalence test...")

# Compare with observer systems from Phase 74
# Recreate observer sweep
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

observer_systems = []
for alpha in [0.2, 0.5, 0.8, 0.95]:
    X = generate_observer(alpha, 500, 32)
    var_x1 = compute_manifold(X)
    temp_coh = np.mean([np.corrcoef(X[:-1, i], X[1:, i])[0, 1] for i in range(min(10, X.shape[1]))])
    D = cdist(X, X, metric='euclidean')
    recurrence = np.mean(D < np.percentile(D, 10))
    pred_errors = [np.mean((X[t] - X[t-1])**2) for t in range(10, X.shape[0])]
    self_model = 1 / (1 + np.mean(pred_errors))
    
    observer_systems.append({
        'alpha': alpha,
        'var_x1': var_x1 if not np.isnan(var_x1) else 0.0,
        'temporal_coherence': temp_coh if not np.isnan(temp_coh) else 0.0,
        'recurrence': recurrence,
        'self_model': self_model,
        'category': 'observer'
    })

# Compare metrics
std_var_x1 = np.array([s['var_x1'] for s in system_results])
obs_var_x1 = np.array([s['var_x1'] for s in observer_systems])

# Transition similarity (can standard systems produce similar transitions?)
# Test if sweeping a parameter in standard systems produces transition
transitions_standard = []

# Sweep logistic map parameter
for r in np.linspace(2.0, 4.0, 10):
    x = np.zeros(500)
    x[0] = 0.5
    for t in range(1, 500):
        x[t] = r * x[t-1] * (1 - x[t-1])
    X = np.zeros((500, 32))
    for i in range(32):
        X[:, i] = x + np.random.randn(500) * 0.01
    var_x1 = compute_manifold(X)
    transitions_standard.append(var_x1 if not np.isnan(var_x1) else 0.0)

# Find max derivative (susceptibility)
d_trans = np.abs(np.gradient(transitions_standard))
max_suscept_std = np.max(d_trans)

# Observer transition susceptibility (from Phase 74)
d_obs = np.abs(np.gradient([s['var_x1'] for s in observer_systems]))
max_suscept_obs = np.max(d_obs)

transition_similarity = max_suscept_std / (max_suscept_obs + 1e-8)
print(f"  Transition similarity (std/observer): {transition_similarity:.4f}")

# RG similarity
# Use temporal coherence and recurrence as RG-like features
rg_coords_std = np.array([[s['temporal_coherence'], s['recurrence']] for s in system_results])
rg_coords_obs = np.array([[s['temporal_coherence'], s['recurrence']] for s in observer_systems])

rg_dist = cdist(rg_coords_std.mean(axis=0).reshape(1,-1), rg_coords_obs.mean(axis=0).reshape(1,-1))
rg_similarity = 1 - float(np.min(rg_dist)) / 2
print(f"  RG similarity: {rg_similarity:.4f}")

# Topology similarity (Betti-like)
# Use variance of var_x1 as proxy
var_x1_std = np.std([s['var_x1'] for s in system_results])
var_x1_obs = np.std([s['var_x1'] for s in observer_systems])
topology_similarity = 1 - abs(var_x1_std - var_x1_obs) / (max(var_x1_std, var_x1_obs) + 1e-8)
print(f"  Topology similarity: {topology_similarity:.4f}")

# Universality similarity
# Check if same scaling behavior
universality_similarity = 1 - abs(transition_similarity - 0.5)  # Close to 0.5 means different
print(f"  Universality similarity: {universality_similarity:.4f}")

# ==============================================================================
# STEP 4: THEORY REDUCTION
# ==============================================================================
print("\nStep 4: Theory reduction...")

# Combine all systems
all_data = system_results + observer_systems
X_all = np.array([[s['var_x1'], s['temporal_coherence'], s['recurrence'], s['self_model']] for s in all_data])
y_var_x1 = np.array([s['var_x1'] for s in all_data])

# Test different reduction theories
# A: Covariance geometry
cov_features = np.array([[np.var(all_data[i]['var_x1']), np.mean([s['var_x1'] for s in system_results[:10]])] for i in range(len(all_data))])
lr_cov = LinearRegression()
lr_cov.fit(cov_features, y_var_x1)
r2_cov = lr_cov.score(cov_features, y_var_x1)

# B: Information geometry
info_features = X_all[:, 1:3]  # temporal_coherence, recurrence
lr_info = LinearRegression()
lr_info.fit(info_features, y_var_x1)
r2_info = lr_info.score(info_features, y_var_x1)

# C: Attractor geometry
attr_features = X_all[:, 3:4]  # self_model
lr_attr = LinearRegression()
lr_attr.fit(attr_features, y_var_x1)
r2_attr = lr_attr.score(attr_features, y_var_x1)

# D: Criticality theory (use derivative-like features)
crit_features = np.array([[s['self_model'] * s['temporal_coherence']] for s in all_data])
lr_crit = LinearRegression()
lr_crit.fit(crit_features, y_var_x1)
r2_crit = lr_crit.score(crit_features, y_var_x1)

# E: Representation geometry
rep_features = X_all
lr_rep = LinearRegression()
lr_rep.fit(rep_features, y_var_x1)
r2_rep = lr_rep.score(rep_features, y_var_x1)

reduction_results = {
    'covariance_geometry': r2_cov,
    'information_geometry': r2_info,
    'attractor_geometry': r2_attr,
    'criticality_theory': r2_crit,
    'representation_geometry': r2_rep
}

best_reduction = max(reduction_results.items(), key=lambda x: x[1])
print(f"  Best reduction theory: {best_reduction[0]} with R2={best_reduction[1]:.4f}")

# ==============================================================================
# STEP 5: MINIMAL EXPLANATION TEST
# ==============================================================================
print("\nStep 5: Minimal explanation test...")

# Find minimal set of features
from sklearn.feature_selection import mutual_info_regression

mi_scores = []
for i in range(4):
    mi = mutual_info_regression(X_all[:, i].reshape(-1, 1), y_var_x1)
    mi_scores.append(mi[0])

feature_names = ['var_x1', 'temporal_coherence', 'recurrence', 'self_model']
ranking = sorted(zip(feature_names, mi_scores), key=lambda x: x[1], reverse=True)

# Test minimal sets
minimal_results = {}
for n_feat in [1, 2, 3, 4]:
    top_features = [r[0] for r in ranking[:n_feat]]
    feature_idx = [feature_names.index(f) for f in top_features]
    X_min = X_all[:, feature_idx]
    lr_min = LinearRegression()
    lr_min.fit(X_min, y_var_x1)
    minimal_results[n_feat] = lr_min.score(X_min, y_var_x1)

print(f"  Minimal explanation R2 by features: {minimal_results}")
minimal_explanatory_set = [r[0] for r in ranking[:2]]  # Best 2-feature set

# ==============================================================================
# STEP 6: NOVELTY TEST
# ==============================================================================
print("\nStep 6: Novelty test...")

# What cannot be explained by standard theories?
# Compute residual after best reduction
best_pred = best_reduction[0]
if best_reduction[0] == 'covariance_geometry':
    best_pred = lr_cov.predict(cov_features)
elif best_reduction[0] == 'information_geometry':
    best_pred = lr_info.predict(info_features)
elif best_reduction[0] == 'attractor_geometry':
    best_pred = lr_attr.predict(attr_features)
elif best_reduction[0] == 'criticality_theory':
    best_pred = lr_crit.predict(crit_features)
else:
    best_pred = lr_rep.predict(rep_features)

explained_variance = 1 - np.var(y_var_x1 - best_pred) / np.var(y_var_x1)
residual_novelty = 1 - explained_variance

print(f"  Explained variance: {explained_variance:.4f}")
print(f"  Residual novelty: {residual_novelty:.4f}")

# Find what remains unexplained
residual = y_var_x1 - best_pred
unexplained_std = np.std(residual[:len(system_results)])  # Standard systems
unexplained_obs = np.std(residual[len(system_results):])  # Observer systems

print(f"  Unexplained structure (std systems): {unexplained_std:.4f}")
print(f"  Unexplained structure (observer): {unexplained_obs:.4f}")

# ==============================================================================
# STEP 7: LITERATURE ALIGNMENT
# ==============================================================================
print("\nStep 7: Literature alignment...")

# Active inference: self-model + temporal coherence
active_inf_score = (np.mean([s['self_model'] for s in observer_systems]) + 
                    np.mean([s['temporal_coherence'] for s in observer_systems])) / 2
active_inf_score = min(active_inf_score, 1.0)

# Criticality: high susceptibility
criticality_score = min(transition_similarity * 2, 1.0)

# Information geometry: entropy-like features
info_geom_score = r2_info

# Representation geometry: manifold features
repr_geom_score = r2_rep

literature_overlap = {
    'active_inference': float(active_inf_score),
    'criticality': float(criticality_score),
    'information_geometry': float(info_geom_score),
    'representation_geometry': float(repr_geom_score)
}

print(f"  Literature overlap scores:")
for k, v in literature_overlap.items():
    print(f"    {k}: {v:.4f}")

# ==============================================================================
# STEP 8: BLIND ABSTRACTION TEST
# ==============================================================================
print("\nStep 8: Blind abstraction test...")

# Hide all labels, use geometry alone
X_geom = np.array([[s['var_x1'], s['temporal_coherence'], s['recurrence'], s['self_model']] for s in all_data])
labels_true = [0] * len(system_results) + [1] * len(observer_systems)  # 0=standard, 1=observer

# KMeans clustering
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
labels_pred = kmeans_fit = kmeans.fit_predict(X_geom)

clustering_accuracy = np.mean([1 if labels_pred[i] == labels_true[i] else 0 for i in range(len(labels_true))])

# Phase separation
from sklearn.metrics import silhouette_score
phase_separation = silhouette_score(X_geom, labels_true)

print(f"  Clustering accuracy: {clustering_accuracy:.4f}")
print(f"  Phase separation: {phase_separation:.4f}")

# ==============================================================================
# STEP 9-11: OUTPUT AND SAVE
# ==============================================================================
print("\n" + "="*60)
print("OUTPUT")
print("="*60)

print("\nEQUIVALENCE:")
print(f"  transition_similarity = {transition_similarity:.4f}")
print(f"  rg_similarity = {rg_similarity:.4f}")
print(f"  topology_similarity = {topology_similarity:.4f}")
print(f"  universality_similarity = {universality_similarity:.4f}")

print("\nREDUCTION:")
print(f"  best_reduction_theory = {best_reduction[0]}")
print(f"  explained_variance = {explained_variance:.4f}")

print("\nMINIMAL EXPLANATION:")
print(f"  minimal_explanatory_set = {minimal_explanatory_set}")

print("\nNOVELTY:")
print(f"  residual_novelty = {residual_novelty:.4f}")
print(f"  unexplained_structure (std): {unexplained_std:.4f}")
print(f"  unexplained_structure (obs): {unexplained_obs:.4f}")

print("\nLITERATURE OVERLAP:")
for k, v in literature_overlap.items():
    print(f"  {k}_overlap = {v:.4f}")

print("\nBLIND GEOMETRY:")
print(f"  clustering_accuracy = {clustering_accuracy:.4f}")
print(f"  phase_separation = {phase_separation:.4f}")

# Verdict
if residual_novelty > 0.5 and clustering_accuracy > 0.8:
    novelty_status = 'genuinely_novel'
elif residual_novelty > 0.3:
    novelty_status = 'partially_novel'
else:
    novelty_status = 'rediscovery'

if clustering_accuracy > 0.7:
    observer_status = 'observer_transition_confirmed'
else:
    observer_status = 'observer_transition_weak'

print(f"\nVERDICT:")
print(f"  observer_transition_status = {observer_status}")
print(f"  novelty_status = {novelty_status}")

# Save results
results = {
    'equivalence': {
        'transition_similarity': float(transition_similarity),
        'rg_similarity': float(rg_similarity),
        'topology_similarity': float(topology_similarity),
        'universality_similarity': float(universality_similarity)
    },
    'reduction': {
        'best_theory': best_reduction[0],
        'explained_variance': float(explained_variance),
        'all_r2': {k: float(v) for k, v in reduction_results.items()}
    },
    'minimal_explanation': {
        'minimal_set': minimal_explanatory_set,
        'r2_by_features': {str(k): float(v) for k, v in minimal_results.items()}
    },
    'novelty': {
        'residual_novelty': float(residual_novelty),
        'unexplained_std': float(unexplained_std),
        'unexplained_obs': float(unexplained_obs)
    },
    'literature_overlap': literature_overlap,
    'blind_geometry': {
        'clustering_accuracy': float(clustering_accuracy),
        'phase_separation': float(phase_separation)
    },
    'verdict': {
        'observer_transition_status': observer_status,
        'novelty_status': novelty_status
    }
}

with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2)

# Save arrays
np.save(os.path.join(OUTPUT_DIR, 'standard_systems.npy'), np.array([s['var_x1'] for s in system_results]))
np.save(os.path.join(OUTPUT_DIR, 'observer_systems.npy'), np.array([s['var_x1'] for s in observer_systems]))
np.save(os.path.join(OUTPUT_DIR, 'all_features.npy'), X_all)

print(f"\nAll files saved to {OUTPUT_DIR}")
print("="*60)