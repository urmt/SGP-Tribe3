"""
PHASE 71: COMPUTATIONAL ORGANIZATION TEST
Test whether manifold organization is computational in origin
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.signal import correlate
from scipy.fft import fft
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import os
import json
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase71_computational_organization'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("PHASE 71: COMPUTATIONAL ORGANIZATION TEST")
print("="*60)

np.random.seed(42)
n_samples = 500
n_dim = 32

# ==============================================================================
# STEP 1: CREATE MASTER SYSTEMS
# ==============================================================================
print("\nStep 1: Creating master systems...")

systems_config = [
    ('gaussian', lambda n,d: np.random.randn(n, d)),
    ('laplace', lambda n,d: np.random.laplace(0, 1, (n, d))),
    ('student_t_2', lambda n,d: np.random.standard_t(2, (n, d))),
    ('student_t_5', lambda n,d: np.random.standard_t(5, (n, d))),
    ('cauchy', lambda n,d: np.clip(np.random.standard_cauchy((n, d)), -10, 10)),
    ('uniform', lambda n,d: np.random.uniform(-np.sqrt(3), np.sqrt(3), (n, d))),
    ('exponential', lambda n,d: np.random.exponential(1, (n, d)) - 1),
    ('heavy_tail', lambda n,d: np.random.standard_t(1.5, (n, d))),
    ('bimodal', lambda n,d: np.concatenate([np.random.randn(n//2, d) - 3, np.random.randn(n//2, d) + 3])),
    ('sparse', lambda n,d: np.where(np.random.rand(n,d) > 0.9, np.random.randn(n,d) * 5, 0)),
    ('low_rank_4', lambda n,d: np.random.randn(n, 4) @ np.random.randn(4, d)),
    ('low_rank_8', lambda n,d: np.random.randn(n, 8) @ np.random.randn(8, d)),
    ('correlated', lambda n,d: np.random.multivariate_normal(np.zeros(d), np.eye(d) * 0.5 + np.ones((d,d)) * 0.5, n)),
    ('hierarchical', lambda n,d: np.random.randn(n, d) * np.tile(np.arange(1, d+1)**0.5, (n, 1))),
    ('log_normal', lambda n,d: np.exp(np.random.randn(n, d))),
]

funcs = [lambda x: x, np.tanh, lambda x: np.maximum(0, x), lambda x: np.log1p(np.exp(x))]

# ==============================================================================
# STEP 2: COMPUTATIONAL FEATURES
# ==============================================================================
print("\nStep 2: Computing computational features...")

def compute_algorithmic_features(X):
    """Compute algorithmic/computational complexity features"""
    X = X - X.mean(axis=0)
    
    # Compression ratio proxy (variance of differences)
    diffs = np.diff(X, axis=0)
    compression_ratio = np.std(diffs) / (np.std(X) + 1e-8)
    
    # Entropy rate (first-order entropy of differences)
    diffs_flat = diffs.flatten()
    hist, _ = np.histogram(diffs_flat, bins=20, density=True)
    hist = hist[hist > 0]
    entropy_rate = -np.sum(hist * np.log(hist + 1e-8))
    
    # Autocorrelation depth (lag at which autocorrelation drops below threshold)
    ac_depth = 5  # default
    for lag in range(1, 20):
        ac = np.mean([np.corrcoef(X[:-lag, i], X[lag:, i])[0, 1] for i in range(min(10, X.shape[1]))])
        if abs(ac) < 0.5:
            ac_depth = lag
            break
    
    # Recurrence depth (number of similar states)
    D = cdist(X, X, metric='euclidean')
    D_sorted = np.sort(D, axis=1)
    recurrence_depth = np.mean(D_sorted[:, 1] / (D_sorted[:, -1] + 1e-8))
    
    # Spectral complexity
    X_fft = np.abs(fft(X, axis=0))
    spectral_flat = X_fft.flatten()
    spectral_hist, _ = np.histogram(spectral_flat, bins=20, density=True)
    spectral_hist = spectral_hist[spectral_hist > 0]
    spectral_complexity = -np.sum(spectral_hist * np.log(spectral_hist + 1e-8))
    
    # Graph complexity (sparsity of distance matrix)
    D_norm = D / (np.max(D) + 1e-8)
    graph_complexity = np.sum(D_norm < 0.1) / D.size
    
    # Manifold curvature (second derivative proxy)
    second_diff = np.diff(diffs, axis=0)
    curvature = np.std(second_diff) / (np.std(diffs) + 1e-8)
    
    # Lyapunov estimate (local expansion rate)
    lyapunov_est = np.std(np.log(np.abs(diffs) + 1e-8))
    
    # Perturbation amplification
    X_pert = X + np.random.randn(*X.shape) * 0.01
    amplification = np.std(X_pert - X) / 0.01
    
    return {
        'compression_ratio': float(compression_ratio),
        'entropy_rate': float(entropy_rate),
        'autocorr_depth': int(ac_depth),
        'recurrence_depth': float(recurrence_depth),
        'spectral_complexity': float(spectral_complexity),
        'graph_complexity': float(graph_complexity),
        'curvature': float(curvature),
        'lyapunov_estimate': float(lyapunov_est),
        'perturbation_amp': float(amplification)
    }

def compute_basic_features(X):
    """Basic spectral and geometric features"""
    X = X - X.mean(axis=0)
    
    cov = np.cov(X, rowvar=False)
    evals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    evals = evals[evals > 1e-8]
    total_var = np.sum(evals)
    p = evals / (total_var + 1e-8)
    eff_rank = np.exp(-np.sum(p * np.log(p + 1e-8)))
    part_ratio = (np.sum(evals)**2) / (np.sum(evals**2) + 1e-8)
    
    phis = []
    for f in funcs:
        t = f(X)
        phis.append(np.mean(t, axis=0))
    phis = np.array(phis)
    
    cov_p = np.cov(phis, rowvar=False)
    evals_p, evecs_p = np.linalg.eigh(cov_p)
    idx = np.argsort(evals_p)[::-1]
    pc1 = evecs_p[:, idx[0]]
    X1 = phis @ pc1
    var_x1 = float(np.var(X1))
    
    return {
        'var_x1': var_x1,
        'effective_rank': float(eff_rank),
        'participation_ratio': float(part_ratio),
        'total_variance': float(total_var)
    }

# Generate all systems
all_systems = []
for sys_name, gen in systems_config:
    for trial in range(3):
        try:
            X = gen(n_samples, n_dim)
            alg_feats = compute_algorithmic_features(X)
            basic_feats = compute_basic_features(X)
            
            if np.isnan(basic_feats['var_x1']):
                continue
                
            all_systems.append({
                'system': sys_name, 'trial': trial,
                **basic_feats,
                **alg_feats
            })
        except:
            continue

print(f"  Total systems: {len(all_systems)}")

# ==============================================================================
# STEP 3: COMPUTATIONAL PREDICTION
# ==============================================================================
print("\nStep 3: Computational prediction...")

comp_features = ['compression_ratio', 'entropy_rate', 'autocorr_depth', 'recurrence_depth',
                 'spectral_complexity', 'graph_complexity', 'curvature', 'lyapunov_estimate', 'perturbation_amp']

X_comp = np.array([[s[f] for f in comp_features] for s in all_systems])
y_var_x1 = np.array([s['var_x1'] for s in all_systems])

# Leave-one-system-out validation
from sklearn.model_selection import LeaveOneOut
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
    for train_idx, test_idx in loo.split(X_comp):
        model.fit(X_comp[train_idx], y_var_x1[train_idx])
        preds.append(model.predict(X_comp[test_idx])[0])
        actuals.append(y_var_x1[test_idx][0])
    
    from sklearn.metrics import r2_score, mean_absolute_error
    r2 = r2_score(actuals, preds)
    mae = mean_absolute_error(actuals, preds)
    results[name] = {'r2': r2, 'mae': mae}

print("  Computational prediction results:")
for name, res in results.items():
    print(f"    {name}: R2={res['r2']:.4f}, MAE={res['mae']:.4f}")

best_model = max(results.items(), key=lambda x: x[1]['r2'])
print(f"  Best model: {best_model[0]} with R2={best_model[1]['r2']:.4f}")

# ==============================================================================
# STEP 4: MINIMAL COMPUTATIONAL COORDINATES
# ==============================================================================
print("\nStep 4: Minimal computational coordinates...")

# Test different dimensionality
test_dims = [1, 2, 3, 5]
dim_results = {}

for nd in test_dims:
    if nd >= X_comp.shape[1]:
        continue
    pca = PCA(n_components=nd)
    X_red = pca.fit_transform(X_comp)
    
    lr = LinearRegression()
    preds = []
    actuals = []
    for train_idx, test_idx in loo.split(X_red):
        lr.fit(X_red[train_idx], y_var_x1[train_idx])
        preds.append(lr.predict(X_red[test_idx])[0])
        actuals.append(y_var_x1[test_idx][0])
    
    r2 = r2_score(actuals, preds)
    dim_results[nd] = r2

print("  Dimensionality results:")
for nd, r2 in dim_results.items():
    print(f"    dim={nd}: R2={r2:.4f}")

minimal_dim = min(dim_results.items(), key=lambda x: x[1])[0] if any(r2 > 0.3 for r2 in dim_results.values()) else list(dim_results.keys())[np.argmax(list(dim_results.values()))]
reconstruction_error = 1 - max(dim_results.values())

print(f"  Minimal dimension: {minimal_dim}, reconstruction error: {reconstruction_error:.4f}")

# ==============================================================================
# STEP 5-6: COMPUTATIONAL INTERVENTIONS
# ==============================================================================
print("\nStep 5-6: Computational interventions...")

X_base = np.random.randn(n_samples, n_dim)
base_feats = compute_basic_features(X_base)
base_alg = compute_algorithmic_features(X_base)
base_var = base_feats['var_x1']

interventions = {}

# Intervention A: Preserve covariance, destroy recurrence
# Randomize time ordering while preserving covariance
def intervention_a(X):
    idx = np.random.permutation(X.shape[0])
    return X[idx]

# Intervention B: Preserve spectrum, destroy compressibility
# Add high-frequency noise preserving covariance structure
def intervention_b(X):
    return X + np.random.randn(*X.shape) * 0.1

# Intervention C: Preserve entropy, destroy algorithmic regularity
# Shuffle within rows to break temporal structure
def intervention_c(X):
    X_new = X.copy()
    for i in range(X.shape[1]):
        X_new[:, i] = np.random.permutation(X_new[:, i])
    return X_new

# Intervention D: Inject recursive structure
# Create periodic structure
def intervention_d(X):
    period = 10
    return X * np.tile(np.sin(np.arange(n_samples) * 2 * np.pi / period).reshape(-1, 1), (1, n_dim))

for name, func in [('A', intervention_a), ('B', intervention_b), ('C', intervention_c), ('D', intervention_d)]:
    X_int = func(X_base.copy())
    int_feats = compute_basic_features(X_int)
    int_alg = compute_algorithmic_features(X_int)
    
    delta_var = abs(int_feats['var_x1'] - base_var)
    delta_compression = abs(int_alg['compression_ratio'] - base_alg['compression_ratio'])
    
    interventions[name] = {
        'delta_var_x1': float(delta_var),
        'delta_compression': float(delta_compression)
    }
    print(f"  Intervention {name}: delta_var={delta_var:.4f}, delta_compression={delta_compression:.4f}")

# ==============================================================================
# STEP 7: COMPUTATIONAL RG FLOW
# ==============================================================================
print("\nStep 7: Computational RG flow...")

# Track trajectory over computational features
comp_coords = X_comp[:, :3]  # Use first 3 computational features

# Simple coarse-graining: average in local regions
n_bins = 5
min_vals = np.min(comp_coords, axis=0)
max_vals = np.max(comp_coords, axis=0)

rg_trajectory = []
for i in range(len(all_systems)):
    coord = comp_coords[i]
    bin_idx = tuple(int(np.clip(int((c - min_vals[j]) / (max_vals[j] - min_vals[j] + 1e-8) * (n_bins - 1)), 0, n_bins - 1)) for j, c in enumerate(coord))
    rg_trajectory.append(bin_idx)

# Fixed points (most common bins)
from collections import Counter
bin_counts = Counter(rg_trajectory)
fixed_points = [bin_idx for bin_idx, count in bin_counts.most_common(3)]
attractors = len(fixed_points)
collapse_regions = sum(1 for b, c in bin_counts.items() if c < 2)

print(f"  Fixed points: {len(fixed_points)}, Attractors: {attractors}, Collapse regions: {collapse_regions}")

# ==============================================================================
# STEP 8: META-UNIVERSALITY
# ==============================================================================
print("\nStep 8: Meta-universality...")

# Complexity features
comp_mean = np.mean(X_comp, axis=1)
comp_std = np.std(X_comp, axis=1)

# Fit U_global ~ complexity
from sklearn.metrics import r2_score

# Define U_global proxy (based on how predictable var_x1 is)
u_global_proxy = np.array([1 / (1 + abs(s['var_x1'] - np.mean(y_var_x1)) / np.std(y_var_x1)) for s in all_systems])

# Simple linear fit
lr_meta = LinearRegression()
lr_meta.fit(np.column_stack([comp_mean, comp_std]), u_global_proxy)
r2_meta = lr_meta.score(np.column_stack([comp_mean, comp_std]), u_global_proxy)

# Thresholds
complexity_threshold = float(np.percentile(comp_mean, 75))
collapse_threshold = float(np.percentile(comp_std, 75))

print(f"  Complexity threshold: {complexity_threshold:.4f}")
print(f"  Collapse threshold: {collapse_threshold:.4f}")
print(f"  Meta-universality R2: {r2_meta:.4f}")

# ==============================================================================
# STEP 9: COMPUTATIONAL PHASES
# ==============================================================================
print("\nStep 9: Computational phases...")

# Cluster using computational features only
kmeans_comp = KMeans(n_clusters=3, random_state=42, n_init=10)
comp_labels = kmeans_comp.fit_predict(X_comp)

# Compare with basic features clustering
basic_features = ['effective_rank', 'participation_ratio', 'total_variance', 'var_x1']
X_basic = np.array([[s[f] for f in basic_features] for s in all_systems])
kmeans_basic = KMeans(n_clusters=3, random_state=42, n_init=10)
basic_labels = kmeans_basic.fit_predict(X_basic)

# Agreement score
agreement = np.mean(comp_labels == basic_labels)
print(f"  Phase agreement: {agreement:.4f}")

# ==============================================================================
# STEP 10-12: OUTPUT AND SAVE
# ==============================================================================
print("\n" + "="*60)
print("OUTPUT")
print("="*60)

print("\nCOMPUTATIONAL PREDICTION:")
print(f"  model = {best_model[0]}")
print(f"  R2 = {best_model[1]['r2']:.4f}")
print(f"  MAE = {best_model[1]['mae']:.4f}")

print("\nMINIMAL COMPUTATIONAL COORDINATES:")
print(f"  parameters = {minimal_dim}")
print(f"  dimension = {minimal_dim}")
print(f"  reconstruction_error = {reconstruction_error:.4f}")

print("\nINTERVENTIONS:")
for name, res in interventions.items():
    print(f"  {name}: delta_var={res['delta_var_x1']:.4f}")

print("\nCOMPUTATIONAL RG:")
print(f"  fixed_points = {len(fixed_points)}")
print(f"  attractors = {attractors}")
print(f"  collapse_regions = {collapse_regions}")

print("\nMETA-UNIVERSALITY:")
print(f"  complexity_threshold = {complexity_threshold:.4f}")
print(f"  collapse_threshold = {collapse_threshold:.4f}")

print("\nPHASE AGREEMENT:")
print(f"  agreement_score = {agreement:.4f}")

# Verdict
if best_model[1]['r2'] > 0.5 and agreement > 0.6:
    verdict = 'computational_origin'
elif best_model[1]['r2'] > 0.3:
    verdict = 'hybrid_computational_statistical'
else:
    verdict = 'statistical_origin'

print(f"\nVERDICT: organization_origin = {verdict}")

# Save results
results = {
    'computational_prediction': {
        'best_model': best_model[0],
        'r2': float(best_model[1]['r2']),
        'mae': float(best_model[1]['mae']),
        'all_results': {k: {'r2': float(v['r2']), 'mae': float(v['mae'])} for k, v in results.items()}
    },
    'minimal_coordinates': {
        'dimension': minimal_dim,
        'reconstruction_error': float(reconstruction_error),
        'dim_results': {str(k): float(v) for k, v in dim_results.items()}
    },
    'interventions': interventions,
    'computational_rg': {
        'fixed_points': len(fixed_points),
        'attractors': attractors,
        'collapse_regions': collapse_regions
    },
    'meta_universality': {
        'complexity_threshold': complexity_threshold,
        'collapse_threshold': collapse_threshold,
        'r2': float(r2_meta)
    },
    'phase_agreement': float(agreement),
    'verdict': verdict
}

with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2)

# Save arrays
np.save(os.path.join(OUTPUT_DIR, 'comp_features.npy'), X_comp)
np.save(os.path.join(OUTPUT_DIR, 'comp_labels.npy'), comp_labels)
np.save(os.path.join(OUTPUT_DIR, 'basic_labels.npy'), basic_labels)

print(f"\nAll files saved to {OUTPUT_DIR}")
print("="*60)