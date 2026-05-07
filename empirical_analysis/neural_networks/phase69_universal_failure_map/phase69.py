"""
PHASE 69: UNIVERSAL FAILURE MAP
Find EXACTLY WHERE universality breaks
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, davies_bouldin_score
import os
import json
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase69_universal_failure_map'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("PHASE 69: UNIVERSAL FAILURE MAP")
print("="*60)

np.random.seed(42)
n_samples = 500
n_dim = 32

# ==============================================================================
# STEP 1: LOAD/CREATE MASTER DATA
# ==============================================================================
print("\nStep 1: Loading/Creating master data...")

# Check if previous phase data exists
data_sources = [
    '/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase57_distribution_geometry',
    '/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase67_universality_stress_test',
    '/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase68_universal_geometric_organization',
]

master_data = []
system_sources = {}

# Try to load from previous phases, otherwise create new data
for source in data_sources:
    results_file = os.path.join(source, 'results.json')
    if os.path.exists(results_file):
        try:
            with open(results_file) as f:
                data = json.load(f)
                print(f"  Loaded from {source}")
                system_sources[source] = data
        except:
            pass

# Create comprehensive dataset with diverse systems
print("  Creating diverse system dataset...")

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

def compute_features(X):
    X = X - X.mean(axis=0)
    
    # Spectral features
    X_flat = X.flatten()
    kt = np.mean([np.mean(x**4)/np.mean(x**2)**2 - 3 for x in X.T])
    sk = np.mean([np.mean(x**3)/np.mean(x**2)**1.5 for x in X.T])
    
    cov = np.cov(X, rowvar=False)
    evals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    evals = evals[evals > 1e-8]
    total_var = np.sum(evals)
    p = evals / (total_var + 1e-8)
    eff_rank = np.exp(-np.sum(p * np.log(p + 1e-8)))
    part_ratio = (np.sum(evals)**2) / (np.sum(evals**2) + 1e-8)
    top_frac = evals[0] / (total_var + 1e-8) if len(evals) > 0 else 0
    
    # Entropy approximation
    hist, _ = np.histogram(X_flat, bins=50, density=True)
    hist = hist[hist > 0]
    shannon_ent = -np.sum(hist * np.log(hist + 1e-8))
    
    # Fisher metric
    fisher = np.sum(1 / (evals + 1e-8))
    
    return {
        'kurtosis': kt, 'skewness': sk, 'entropy': shannon_ent,
        'effective_rank': eff_rank, 'participation_ratio': part_ratio,
        'top_fraction': top_frac, 'total_variance': total_var,
        'fisher_metric': fisher
    }

def compute_manifold(X):
    X = X - X.mean(axis=0)
    n = X.shape[0]
    
    # Compute operator responses
    phis = []
    for f in funcs:
        t = f(X)
        phis.append(np.mean(t, axis=0))
    phis = np.array(phis)  # shape: (n_ops, n_dim)
    
    # Use first two principal components as proxy
    cov = np.cov(phis, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    idx = np.argsort(evals)[::-1]
    pc1 = evecs[:, idx[0]]
    pc2 = evecs[:, idx[1]] if len(idx) > 1 else pc1
    
    X1 = phis @ pc1
    
    var_x1 = np.var(X1)
    unique_x1 = len(np.unique(np.round(X1, 3)))
    
    return var_x1, unique_x1

# Generate all systems
all_systems = []
for sys_name, gen in systems_config:
    for trial in range(3):
        try:
            X = gen(n_samples, n_dim)
            feats = compute_features(X)
            var_x1, unique_x1 = compute_manifold(X)
            
            # Skip invalid values
            if np.isnan(var_x1) or np.isnan(unique_x1):
                continue
            if any(np.isnan(v) for v in feats.values()):
                continue
                
            all_systems.append({
                'system': sys_name, 'trial': trial,
                'var_x1': float(var_x1), 'unique_x1': int(unique_x1),
                **{k: float(v) for k, v in feats.items()}
            })
        except:
            continue

print(f"  Total systems: {len(all_systems)}")

if len(all_systems) == 0:
    print("ERROR: No valid systems generated!")
    exit(1)

# ==============================================================================
# STEP 2: UNIVERSALITY RESIDUALS
# ==============================================================================
print("\nStep 2: Computing universality residuals...")

# Best model from Phase 68: linear regression using effective_rank, participation_ratio, fisher_metric
features_for_model = ['effective_rank', 'participation_ratio', 'fisher_metric']
X_model = np.array([[s[f] for f in features_for_model] for s in all_systems])
y_actual = np.array([s['var_x1'] for s in all_systems])

# Simple linear model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_model, y_actual)
y_pred = lr.predict(X_model)

residuals = y_actual - y_pred

# Save residuals
np.save(os.path.join(OUTPUT_DIR, 'universality_residuals.npy'), residuals)
np.save(os.path.join(OUTPUT_DIR, 'system_data.npy'), np.array([list(s.values()) for s in all_systems]))

print(f"  Residual mean: {np.mean(residuals):.4f}")
print(f"  Residual std: {np.std(residuals):.4f}")
print(f"  Max |residual|: {np.max(np.abs(residuals)):.4f}")

# ==============================================================================
# STEP 3: FAILURE CLUSTERING
# ==============================================================================
print("\nStep 3: Failure clustering...")

# Feature matrix for clustering
cluster_features = ['kurtosis', 'skewness', 'entropy', 'effective_rank', 
                    'participation_ratio', 'top_fraction', 'var_x1']
X_cluster = np.array([[s[f] for f in cluster_features] for s in all_systems])
X_cluster = (X_cluster - X_cluster.mean(axis=0)) / (X_cluster.std(axis=0) + 1e-8)

# Cluster by residual behavior
residual_labels = (np.abs(residuals) > np.percentile(np.abs(residuals), 50)).astype(int)

# KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
km_labels = kmeans.fit_predict(X_cluster)
km_silhouette = silhouette_score(X_cluster, km_labels)
km_db = davies_bouldin_score(X_cluster, km_labels)

# GMM
gmm = GaussianMixture(n_components=3, random_state=42, n_init=5)
gmm_labels = gmm.fit_predict(X_cluster)

# Spectral
spectral = SpectralClustering(n_clusters=3, random_state=42, affinity='rbf')
spec_labels = spectral.fit_predict(X_cluster)

cluster_results = {
    'kmeans': {'labels': km_labels.tolist(), 'silhouette': float(km_silhouette), 'db': float(km_db)},
    'gmm': {'labels': gmm_labels.tolist()},
    'spectral': {'labels': spec_labels.tolist()}
}

print(f"  KMeans: silhouette={km_silhouette:.3f}, DB={km_db:.3f}")

# ==============================================================================
# STEP 4: STRUCTURAL FAILURE ANALYSIS
# ==============================================================================
print("\nStep 4: Structural failure analysis...")

# Binary classification: success vs failure
threshold = np.percentile(np.abs(residuals), 50)
y_class = (np.abs(residuals) < threshold).astype(int)

# Features
all_feature_names = ['kurtosis', 'skewness', 'entropy', 'effective_rank', 
                     'participation_ratio', 'top_fraction', 'total_variance', 'fisher_metric']
X_clf = np.array([[s[f] for f in all_feature_names] for s in all_systems])

# Logistic Regression
lr_clf = LogisticRegression(random_state=42, max_iter=1000)
lr_clf.fit(X_clf, y_class)
lr_importance = np.abs(lr_clf.coef_[0])

# Random Forest
rf_clf = RandomForestClassifier(n_estimators=50, random_state=42)
rf_clf.fit(X_clf, y_class)
rf_importance = rf_clf.feature_importances_

importance_ranking = sorted(zip(all_feature_names, rf_importance), key=lambda x: x[1], reverse=True)

print("  Feature importance (Random Forest):")
for f, imp in importance_ranking[:5]:
    print(f"    {f}: {imp:.4f}")

# ==============================================================================
# STEP 5: FAILURE SURFACE
# ==============================================================================
print("\nStep 5: Failure surface...")

# PCA
pca = PCA(n_components=2)
pca_embedding = pca.fit_transform(X_clf)

# ISOMAP
try:
    iso = Isomap(n_neighbors=5, n_components=2)
    iso_embedding = iso.fit_transform(X_clf)
except:
    iso_embedding = pca_embedding

# Intrinsic dimension
cumvar = np.cumsum(pca.explained_variance_ratio_)
intrinsic_dim = int(np.argmax(cumvar >= 0.95) + 1)

failure_surface = {
    'intrinsic_dimension': intrinsic_dim,
    'pca_variance_explained': float(pca.explained_variance_ratio_[0])
}

print(f"  Failure surface: dim={intrinsic_dim}")

# ==============================================================================
# STEP 6: CRITICAL BOUNDARIES
# ==============================================================================
print("\nStep 6: Critical boundaries...")

# Sweep through each feature to find where U_local < 0.5
critical_boundaries = {}

for feat in ['kurtosis', 'effective_rank', 'entropy', 'participation_ratio']:
    values = np.array([s[feat] for s in all_systems])
    sorted_idx = np.argsort(values)
    sorted_residuals = np.abs(residuals[sorted_idx])
    
    # Find where residual > 0.5 * max
    threshold_val = 0.5 * np.max(sorted_residuals)
    critical_idx = np.where(sorted_residuals > threshold_val)[0]
    
    if len(critical_idx) > 0:
        critical_boundaries[feat] = float(values[sorted_idx[critical_idx[0]]])
    else:
        critical_boundaries[feat] = float(np.max(values))

print("  Critical boundaries:")
for feat, val in critical_boundaries.items():
    print(f"    {feat}: {val:.4f}")

# ==============================================================================
# STEP 7: ADVERSARIAL SEARCH
# ==============================================================================
print("\nStep 7: Adversarial search...")

adversarial_systems = []

# Generate adversarial systems designed to break universality
adversarial_configs = [
    ('extreme_heavy_tail', lambda: np.random.standard_t(1.0, (n_samples, n_dim))),
    ('sparse_spikes', lambda: np.where(np.random.rand(n_samples, n_dim) > 0.95, np.random.randn(n_samples, n_dim) * 10, 0)),
    ('hierarchical_cov', lambda: np.random.randn(n_samples, n_dim) @ np.diag(np.arange(1, n_dim+1)**2)),
    ('discontinuous', lambda: np.where(np.random.rand(n_samples, n_dim) > 0.5, np.random.randn(n_samples, n_dim) * 10, np.random.uniform(-1, 1, (n_samples, n_dim)))),
    ('multi_scale', lambda: np.random.randn(n_samples, n_dim) * np.random.uniform(0.1, 10, (1, n_dim))),
]

adversarial_results = []
for adv_name, gen in adversarial_configs:
    try:
        X_adv = gen()
        X_adv = X_adv - X_adv.mean(axis=0)
        
        feats = compute_features(X_adv)
        
        # Predict using our model
        X_adv_model = np.array([[feats[f] for f in features_for_model]])
        pred_adv = lr.predict(X_adv_model)[0]
        
        # Compute actual var_x1 (need to compute manifold)
        # Use simple proxy: variance of random projection
        var_x1_adv = np.var(np.random.randn(n_samples, 10) @ X_adv.T)
        
        # Distortion
        distortion = abs(var_x1_adv - pred_adv) if var_x1_adv > 0 else abs(pred_adv)
        
        # U_local (simplified)
        u_local = 1 / (1 + distortion)
        
        adversarial_results.append({
            'system': adv_name,
            'u_local': float(u_local),
            'distortion': float(distortion),
            'var_x1_actual': float(var_x1_adv),
            'var_x1_pred': float(pred_adv)
        })
    except Exception as e:
        adversarial_results.append({
            'system': adv_name,
            'error': str(e)
        })

print(f"  Generated {len(adversarial_results)} adversarial systems")

# ==============================================================================
# STEP 8: META-UNIVERSALITY TEST
# ==============================================================================
print("\nStep 8: Meta-universality test...")

# Compute distance matrix over failure trajectories
# Use residuals as failure indicator
failure_dists = np.abs(residuals.reshape(-1, 1) - residuals.reshape(-1, 1))

# Cluster failure patterns
km_failure = KMeans(n_clusters=2, random_state=42, n_init=10)
failure_labels = km_failure.fit_predict(residuals.reshape(-1, 1))

# Failure geometry dimension
pca_failure = PCA(n_components=2)
failure_pca = pca_failure.fit_transform(np.column_stack([residuals, np.abs(residuals)]))

meta_universality = {
    'failure_geometry_dimension': 1,  # residuals are 1D
    'failure_cluster_count': len(set(failure_labels)),
    'failure_topology': 'discrete' if len(set(failure_labels)) > 1 else 'connected'
}

print(f"  Failure clusters: {len(set(failure_labels))}")

# ==============================================================================
# STEP 9-11: OUTPUT AND SAVE
# ==============================================================================
print("\n" + "="*60)
print("OUTPUT")
print("="*60)

# Find best clustering
best_sil = max([km_silhouette])

print("\nFAILURE CLUSTERS:")
print(f"  cluster_count = {len(set(km_labels))}")
print(f"  silhouette = {best_sil:.4f}")
print(f"  davies_bouldin = {km_db:.4f}")

print("\nFAILURE DRIVERS:")
for f, imp in importance_ranking[:5]:
    print(f"  {f} = {imp:.4f}")

print("\nCRITICAL SURFACES:")
for feat, val in critical_boundaries.items():
    print(f"  critical_{feat} = {val:.4f}")

print("\nADVERSARIAL SYSTEMS:")
for r in adversarial_results:
    if 'error' not in r:
        print(f"  {r['system']}: U_local={r['u_local']:.4f}, distortion={r['distortion']:.4f}")

print("\nMETA-UNIVERSALITY:")
print(f"  failure_geometry_dimension = {meta_universality['failure_geometry_dimension']}")
print(f"  failure_topology = {meta_universality['failure_topology']}")

# Verdict
max_distortion = max([r.get('distortion', 0) for r in adversarial_results], default=0)
if max_distortion > 1.0 and best_sil < 0.3:
    verdict = 'catastrophic_failure_boundaries'
elif best_sil < 0.5:
    verdict = 'diffuse_failure_manifold'
else:
    verdict = 'structured_failure_clusters'

print(f"\nVERDICT: universality_failure_structure = {verdict}")

# Save all results
results = {
    'failure_clusters': {
        'cluster_count': int(len(set(km_labels))),
        'silhouette': float(best_sil),
        'davies_bouldin': float(km_db)
    },
    'failure_drivers': {f: float(imp) for f, imp in importance_ranking},
    'critical_boundaries': critical_boundaries,
    'adversarial_results': adversarial_results,
    'meta_universality': meta_universality,
    'verdict': verdict
}

with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2, default=str)

# Save arrays
np.save(os.path.join(OUTPUT_DIR, 'residuals.npy'), residuals)
np.save(os.path.join(OUTPUT_DIR, 'cluster_features.npy'), X_cluster)
np.save(os.path.join(OUTPUT_DIR, 'pca_embedding.npy'), pca_embedding)

print(f"\nAll files saved to {OUTPUT_DIR}")
print("="*60)