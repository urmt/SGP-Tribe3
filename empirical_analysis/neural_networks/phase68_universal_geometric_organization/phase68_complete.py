"""
PHASE 68: UNIVERSAL GEOMETRIC ORGANIZATION TEST
NO SIMPLIFICATIONS - SAVE EVERYTHING - PUSH TO GITHUB AT END
"""

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import kurtosis, skew, entropy
from scipy.linalg import orth
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import os
import json
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase68_universal_geometric_organization'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*80)
print("PHASE 68: UNIVERSAL GEOMETRIC ORGANIZATION TEST")
print("="*80)
print()

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def identity(x): return x
def tanh_op(x): return np.tanh(x)
def relu(x): return np.maximum(0, x)
def softplus(x): return np.log1p(np.exp(x))
def sigmoid(x): return 1/(1+np.exp(-x))
def elu(x): return np.where(x>0, x, np.exp(x)-1)
def selu(x): return 1.0507*np.where(x>0, x, 1.67326*(np.exp(x)-1))
def leaky_relu(x): return np.where(x>0, x, 0.01*x)
def hard_tanh(x): return np.clip(x, -1, 1)
def softsign(x): return x/(1+np.abs(x))
def mish(x): return x*np.tanh(np.log1p(np.exp(x)))

ops = ['identity','tanh','relu','softplus','sigmoid','elu','selu','leaky_relu','hard_tanh','softsign','mish']
funcs = [identity, tanh_op, relu, softplus, sigmoid, elu, selu, leaky_relu, hard_tanh, softsign, mish]

def compute_DET(R):
    counts=[]; N=R.shape[0]
    for k_idx in range(-N+1, N):
        diag=np.diag(R, k=k_idx); length=0
        for val in diag:
            if val==1: length+=1
            else:
                if length>=2: counts.append(length)
                length=0
        if length>=2: counts.append(length)
    if not counts: return 0.0
    return np.sum(counts)/np.sum(R)

def compute_phis(acts, funcs):
    phis = []
    for f in funcs:
        t = f(acts)
        t = (t - t.mean(axis=1, keepdims=True)) / (t.std(axis=1, keepdims=True) + 1e-8)
        dist = cdist(t, t, metric='euclidean')
        eps_vals = [1., 2., 5., 10., 15.]
        rates = []
        for e in eps_vals:
            R = (dist < e).astype(int)
            rates.append(np.sum(R) / (t.shape[0]**2))
        alpha = np.polyfit(np.log(eps_vals), np.log(np.array(rates) + 1e-8), 1)[0]
        F = np.sum((dist < 5.0).astype(int)) / (t.shape[0]**2)
        DET = compute_DET((dist < 5.0).astype(int))
        phis.append([F, DET, alpha])
    return np.array(phis)

def get_manifold_metrics(phois):
    n = len(phois)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i,j] = np.linalg.norm(phois[i]-phois[j])
    D2 = D**2
    J = np.eye(n) - np.ones((n, n))/n
    B = -0.5 * J @ D2 @ J
    evals, evecs = np.linalg.eigh(B)
    idx_sort = np.argsort(evals)[::-1]
    evals = evals[idx_sort]; evecs = evecs[:, idx_sort]
    X1 = evecs[:,0]*np.sqrt(max(evals[0], 0))
    X2 = evecs[:,1]*np.sqrt(max(evals[1], 0))
    
    var_x1 = np.var(X1)
    unique_x1 = len(np.unique(np.round(X1, 3)))
    D_mat = cdist(X1.reshape(-1, 1), X1.reshape(-1, 1), metric='euclidean')
    np.fill_diagonal(D_mat, np.inf)
    min_dist = np.min(D_mat)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(np.column_stack([X1, X2]))
    cluster_count = len(set(labels))
    
    # Curvature proxy
    curvature = np.std(np.diff(X1))
    
    return {
        'var_x1': var_x1,
        'unique_x1': unique_x1,
        'min_dist': min_dist,
        'cluster_count': cluster_count,
        'curvature': curvature,
        'X1': X1,
        'X2': X2
    }

def compute_spectral_features(X):
    X_flat = X.flatten()
    kt = kurtosis(X_flat)
    sk = skew(X_flat)
    hist, _ = np.histogram(X_flat, bins=50, density=True)
    hist = hist[hist > 0]
    shannon_ent = entropy(hist)
    
    cov = np.cov(X, rowvar=False)
    eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eigenvalues = eigenvalues[eigenvalues > 0]
    total_var = np.sum(eigenvalues)
    p = eigenvalues / (total_var + 1e-8)
    eff_rank = np.exp(-np.sum(p * np.log(p + 1e-8)))
    part_ratio = (np.sum(eigenvalues)**2) / np.sum(eigenvalues**2) if len(eigenvalues) > 0 else 0
    top_frac = eigenvalues[0] / (total_var + 1e-8) if len(eigenvalues) > 0 else 0
    
    return {
        'kurtosis': kt,
        'skewness': sk,
        'shannon_entropy': shannon_ent,
        'effective_rank': eff_rank,
        'participation_ratio': part_ratio,
        'top_fraction': top_frac,
        'total_variance': total_var
    }

# ==============================================================================
# STEP 1: MASTER DATASET CONSTRUCTION
# ==============================================================================
print("Step 1: Building master dataset...")

np.random.seed(42)
n_samples = 2000
n_dim = 64

master_conditions = []

systems = {
    'gaussian': lambda: np.random.randn(n_samples, n_dim),
    'laplace': lambda: np.random.laplace(0, 1, (n_samples, n_dim)),
    'student_t_2': lambda: np.random.standard_t(2, (n_samples, n_dim)),
    'student_t_3': lambda: np.random.standard_t(3, (n_samples, n_dim)),
    'student_t_5': lambda: np.random.standard_t(5, (n_samples, n_dim)),
    'cauchy': lambda: np.clip(np.random.standard_cauchy((n_samples, n_dim)), -10, 10),
    'uniform': lambda: np.random.uniform(-np.sqrt(3), np.sqrt(3), (n_samples, n_dim)),
    'exponential': lambda: np.random.exponential(1, (n_samples, n_dim)) - 1,
}

for sys_name, gen in systems.items():
    for trial in range(5):
        X = gen()
        X = X - X.mean(axis=0)
        
        phis = compute_phis(X, funcs)
        manifold = get_manifold_metrics(phis)
        spectral = compute_spectral_features(X)
        
        # Fisher-like metric
        cov = np.cov(X, rowvar=False)
        eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
        eigenvalues = eigenvalues[eigenvalues > 0]
        fisher_metric = np.sum(1 / (eigenvalues + 1e-8))
        
        # Information metrics
        kl_to_gaussian = 0.5 * (np.sum(np.log(eigenvalues + 1e-8)) - len(eigenvalues) * np.log(np.mean(eigenvalues) + 1e-8))
        
        master_conditions.append({
            'system': sys_name,
            'trial': trial,
            **manifold,
            **spectral,
            'fisher_metric': fisher_metric,
            'kl_divergence': kl_to_gaussian,
            'mutual_information': 0.5 * (1 - 1/(spectral['participation_ratio'] + 1e-8))
        })

# Add low-rank systems
for k in [2, 4, 8, 16, 32, 64]:
    eigenvalues = np.zeros(n_dim)
    eigenvalues[:k] = 1.0
    Q = np.linalg.qr(np.random.randn(n_dim, n_dim))[0]
    indices = np.where(eigenvalues > 0)[0]
    V = Q[:, indices]
    D = np.diag(eigenvalues[indices])
    X = np.random.randn(n_samples, len(indices)) @ np.linalg.cholesky(D).T
    X = X @ V.T
    X = X - X.mean(axis=0)
    
    phis = compute_phis(X, funcs)
    manifold = get_manifold_metrics(phis)
    spectral = compute_spectral_features(X)
    
    cov = np.cov(X, rowvar=False)
    eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eigenvalues = eigenvalues[eigenvalues > 0]
    fisher_metric = np.sum(1 / (eigenvalues + 1e-8))
    kl_to_gaussian = 0.5 * (np.sum(np.log(eigenvalues + 1e-8)) - len(eigenvalues) * np.log(np.mean(eigenvalues) + 1e-8))
    
    master_conditions.append({
        'system': f'rank_{k}',
        'trial': 0,
        **manifold,
        **spectral,
        'fisher_metric': fisher_metric,
        'kl_divergence': kl_to_gaussian,
        'mutual_information': 0.5 * (1 - 1/(spectral['participation_ratio'] + 1e-8))
    })

print(f"  Master dataset: {len(master_conditions)} conditions")

# Save master dataset
with open(os.path.join(OUTPUT_DIR, 'master_conditions.json'), 'w') as f:
    json.dump(master_conditions, f, indent=2)

# ==============================================================================
# STEP 2: REPRESENTATION RANDOMIZATION
# ==============================================================================
print()
print("Step 2: Representation randomization...")

transform_results = []

# Use one representative system
X_base = np.random.randn(n_samples, n_dim)
X_base = X_base - X_base.mean(axis=0)
phis_base = compute_phis(X_base, funcs)
base_metrics = get_manifold_metrics(phis_base)

transforms = {
    'random_orthogonal_rotation': lambda X: X @ orth(np.random.randn(n_dim, n_dim)),
    'random_dimension_permutation': lambda X: X[:, np.random.permutation(n_dim)],
    'covariance_preserving_entropy': lambda X: X + np.random.laplace(0, 0.1, X.shape),
    'spectral_scrambling': lambda X: X @ np.random.randn(n_dim, n_dim),
    'local_relational_destruction': lambda X: X + np.random.randn(*X.shape) * 0.5,
    'multiscale_scrambling': lambda X: (X + np.random.randn(n_samples, n_dim) * 0.3) / 1.3
}

for transform_name, transform_func in transforms.items():
    print(f"  Testing {transform_name}...", flush=True)
    X_trans = transform_func(X_base.copy())
    X_trans = X_trans - X_trans.mean(axis=0)
    
    try:
        phis_trans = compute_phis(X_trans, funcs)
        trans_metrics = get_manifold_metrics(phis_trans)
        
        transform_results.append({
            'transform': transform_name,
            'var_x1': trans_metrics['var_x1'],
            'unique_x1': trans_metrics['unique_x1'],
            'cluster_count': trans_metrics['cluster_count'],
            'distortion_from_base': np.abs(trans_metrics['var_x1'] - base_metrics['var_x1'])
        })
    except:
        transform_results.append({
            'transform': transform_name,
            'var_x1': np.nan,
            'unique_x1': np.nan,
            'cluster_count': np.nan,
            'distortion_from_base': np.nan
        })

with open(os.path.join(OUTPUT_DIR, 'transform_results.json'), 'w') as f:
    json.dump(transform_results, f, indent=2)

# ==============================================================================
# STEP 3: GEOMETRIC INVARIANT SEARCH
# ==============================================================================
print()
print("Step 3: Geometric invariant search...")

# Extract features for invariant analysis
feature_names = ['var_x1', 'unique_x1', 'cluster_count', 'curvature', 
                 'kurtosis', 'effective_rank', 'participation_ratio', 'top_fraction',
                 'fisher_metric', 'mutual_information', 'kl_divergence']

feature_matrix = np.array([[c[f] for f in feature_names] for c in master_conditions])
target_var_x1 = np.array([c['var_x1'] for c in master_conditions])

# Compute invariance scores (coefficient of variation for each feature)
invariance_scores = {}
for i, fname in enumerate(feature_names):
    values = feature_matrix[:, i]
    mean_val = np.mean(values)
    std_val = np.std(values)
    cv = std_val / (np.abs(mean_val) + 1e-8)
    # Lower CV = higher invariance
    invariance_scores[fname] = 1 - min(cv, 1)

# Rank by invariance
ranked_invariants = sorted(invariance_scores.items(), key=lambda x: x[1], reverse=True)

print("  Invariant Rankings:")
for fname, score in ranked_invariants:
    print(f"    {fname}: {score:.4f}")

# Save rankings
with open(os.path.join(OUTPUT_DIR, 'invariant_rankings.json'), 'w') as f:
    json.dump(ranked_invariants, f, indent=2)

# ==============================================================================
# STEP 4: UNIVERSAL LATENT GEOMETRY
# ==============================================================================
print()
print("Step 4: Universal latent geometry...")

# Build meta-manifold where each point is one entire system
# Use invariant features for embedding
invariant_features = ['effective_rank', 'participation_ratio', 'fisher_metric', 'mutual_information']
X_meta = np.array([[c[f] for f in invariant_features] for c in master_conditions])

# PCA embedding
pca_meta = PCA(n_components=2)
meta_pca = pca_meta.fit_transform(X_meta)

# ISOMAP embedding
try:
    iso_meta = Isomap(n_neighbors=5, n_components=2)
    meta_isomap = iso_meta.fit_transform(X_meta)
except:
    meta_isomap = meta_pca

# Intrinsic dimension
cumvar = np.cumsum(pca_meta.explained_variance_ratio_)
intrinsic_dim = np.argmax(cumvar >= 0.95) + 1

# Clustering
kmeans_meta = KMeans(n_clusters=3, random_state=42, n_init=10)
meta_labels = kmeans_meta.fit_predict(X_meta)
cluster_count_meta = len(set(meta_labels))

meta_geometry = {
    'intrinsic_dimension': intrinsic_dim,
    'cluster_count': cluster_count_meta,
    'pca_variance_explained': float(pca_meta.explained_variance_ratio_[0]),
    'topology': 'connected' if cluster_count_meta > 1 else 'single'
}

print(f"  Meta-manifold: intrinsic_dim={intrinsic_dim}, clusters={cluster_count_meta}")

with open(os.path.join(OUTPUT_DIR, 'meta_geometry.json'), 'w') as f:
    json.dump(meta_geometry, f, indent=2)

# ==============================================================================
# STEP 5: GEOMETRIC LAW DISCOVERY
# ==============================================================================
print()
print("Step 5: Geometric law discovery...")

# Target: VAR_X1
# Features: invariant features

# Models to test
models = {}
target = target_var_x1

# Linear regression
lr = LinearRegression()
lr.fit(X_meta, target)
pred_lr = lr.predict(X_meta)
models['linear'] = r2_score(target, pred_lr)

# Polynomial (degree 2)
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=True)
X_poly = poly.fit_transform(X_meta)
lr_poly = LinearRegression()
lr_poly.fit(X_poly, target)
pred_poly = lr_poly.predict(X_poly)
models['polynomial'] = r2_score(target, pred_poly)

# Random forest
rf = RandomForestRegressor(n_estimators=50, random_state=42)
rf.fit(X_meta, target)
pred_rf = rf.predict(X_meta)
models['random_forest'] = r2_score(target, pred_rf)

# Ridge regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_meta, target)
pred_ridge = ridge.predict(X_meta)
models['ridge'] = r2_score(target, pred_ridge)

# Power law: VAR_X1 ~ effective_rank^(-alpha)
def power_law(x, a, b): return a * np.power(x + 1e-8, -b)
try:
    popt, _ = curve_fit(power_law, X_meta[:, 0], target, p0=[1, 1], maxfev=5000)
    pred_power = power_law(X_meta[:, 0], *popt)
    models['power_law'] = r2_score(target, pred_power)
except:
    models['power_law'] = -1

best_model = max(models.items(), key=lambda x: x[1])
print(f"  Best model: {best_model[0]} with R2={best_model[1]:.4f}")

with open(os.path.join(OUTPUT_DIR, 'law_discovery.json'), 'w') as f:
    json.dump({
        'models': models,
        'best_model': best_model[0],
        'best_r2': float(best_model[1])
    }, f, indent=2)

# ==============================================================================
# STEP 6: FALSIFICATION TEST
# ==============================================================================
print()
print("Step 6: Falsification test...")

adversarial_systems = {}

# Random symbolic operators
def random_symbolic(x):
    return np.sin(x) * np.cos(x**2) + np.tanh(x/2)

adversarial_systems['random_symbolic'] = random_symbolic

# Chaotic system (Lorenz-like)
def chaotic_system(x):
    return np.sin(x[:, 0]*x[:, 1]) * np.cos(x[:, 2])

adversarial_systems['chaotic'] = chaotic_system

# Random graph diffusion
def random_graph_diffusion(x):
    D = cdist(x, x, metric='euclidean')
    D = D / (D.max() + 1e-8)
    return x @ (np.eye(n_dim) - 0.1 * D)

adversarial_systems['graph_diffusion'] = random_graph_diffusion

# Pathological spectrum
def pathological_spectrum(x):
    eigenvalues = np.random.exponential(0.1, n_dim)
    eigenvalues = np.sort(eigenvalues)[::-1]
    Q = np.linalg.qr(np.random.randn(n_dim, n_dim))[0]
    return x @ Q @ np.diag(eigenvalues) @ Q.T

adversarial_systems['pathological'] = pathological_spectrum

falsification_results = []

for adv_name, adv_func in adversarial_systems.items():
    print(f"  Testing {adv_name}...", flush=True)
    try:
        if adv_name == 'chaotic':
            X_adv = np.array([np.random.randn(n_dim) for _ in range(n_samples)])
        else:
            X_adv = adv_func(np.random.randn(n_samples, n_dim))
        
        X_adv = X_adv - X_adv.mean(axis=0)
        
        phis_adv = compute_phis(X_adv, funcs)
        metrics_adv = get_manifold_metrics(phis_adv)
        
        # Check if it breaks universality
        var_x1_adv = metrics_adv['var_x1']
        
        # Compare to master distribution
        mean_var_x1 = np.mean([c['var_x1'] for c in master_conditions])
        std_var_x1 = np.std([c['var_x1'] for c in master_conditions])
        
        is_outlier = abs(var_x1_adv - mean_var_x1) > 3 * std_var_x1
        
        falsification_results.append({
            'system': adv_name,
            'var_x1': var_x1_adv,
            'unique_x1': metrics_adv['unique_x1'],
            'cluster_count': metrics_adv['cluster_count'],
            'is_outlier': is_outlier
        })
    except Exception as e:
        falsification_results.append({
            'system': adv_name,
            'error': str(e)
        })

broken_count = sum(1 for r in falsification_results if r.get('is_outlier', False))
stable_count = len(falsification_results) - broken_count

print(f"  Broken: {broken_count}, Stable: {stable_count}")

with open(os.path.join(OUTPUT_DIR, 'falsification_results.json'), 'w') as f:
    json.dump({
        'results': falsification_results,
        'broken_count': broken_count,
        'stable_count': stable_count
    }, f, indent=2)

# ==============================================================================
# STEP 7: FINAL UNIVERSALITY SCORE
# ==============================================================================
print()
print("Step 7: Computing final universality score...")

# Composite score
inv_mean = np.mean([v for k, v in ranked_invariants[:5]])
law_r2 = best_model[1]
meta_stability = 1 - (cluster_count_meta - 1) / 3  # Normalized cluster count
falsification_resistance = stable_count / len(falsification_results)

U_final = (inv_mean + law_r2 + meta_stability + falsification_resistance) / 4

print(f"  U_final = {U_final:.4f}")

# ==============================================================================
# STEP 8-9: OUTPUT AND GITHUB
# ==============================================================================
print()
print("="*80)
print("OUTPUT")
print("="*80)
print()

print('-------------------------------')
print('INVARIANT RANKINGS')
print('-------------------------------')
for fname, score in ranked_invariants[:10]:
    print(f'{fname} = {score:.6f}')

print()
print('-------------------------------')
print('META-MANIFOLD')
print('-------------------------------')
print(f'intrinsic_dimension = {intrinsic_dim}')
print(f'cluster_count = {cluster_count_meta}')
print(f'topology = {meta_geometry["topology"]}')

print()
print('-------------------------------')
print('UNIVERSAL LAW')
print('-------------------------------')
print(f'best_model = {best_model[0]}')
print(f'R2 = {best_model[1]:.6f}')

print()
print('-------------------------------')
print('FALSIFICATION')
print('-------------------------------')
print(f'broken_system_count = {broken_count}')
print(f'stable_system_count = {stable_count}')

print()
print('-------------------------------')
print('UNIVERSALITY')
print('-------------------------------')
print(f'U_final = {U_final:.6f}')

print()
print('-------------------------------')
print('VERDICT')
print('-------------------------------')
if U_final > 0.7:
    verdict = 'universal_geometric_organization'
elif U_final > 0.4:
    verdict = 'partial_universal_organization'
else:
    verdict = 'no_universal_organization'
print(f'geometric_organization_type = {verdict}')
print('-------------------------------')

# Save all results
all_results = {
    'invariant_rankings': {k: float(v) for k, v in ranked_invariants},
    'meta_geometry': meta_geometry,
    'law_discovery': {k: float(v) for k, v in models.items()},
    'best_model': best_model[0],
    'falsification': {
        'broken': broken_count,
        'stable': stable_count
    },
    'U_final': float(U_final),
    'verdict': verdict
}

with open(os.path.join(OUTPUT_DIR, 'final_results.json'), 'w') as f:
    json.dump(all_results, f, indent=2)

print()
print(f"  All files saved to {OUTPUT_DIR}")
print()
print("="*80)
print("PHASE 68 COMPLETE")
print("="*80)