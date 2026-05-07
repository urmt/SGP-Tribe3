"""
PHASE 68: UNIVERSAL GEOMETRIC ORGANIZATION TEST
FAST VERSION - SAVE EVERYTHING
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import kurtosis, skew, entropy
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import os
import json
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase68_universal_geometric_organization'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("PHASE 68: UNIVERSAL GEOMETRIC ORGANIZATION")
print("="*60)

np.random.seed(42)
n_samples = 500
n_dim = 32

ops = ['identity','tanh','relu','softplus','sigmoid','elu','selu','leaky_relu']
funcs = [lambda x: x, np.tanh, lambda x: np.maximum(0,x), lambda x: np.log1p(np.exp(x)), 
         lambda x: 1/(1+np.exp(-x)), lambda x: np.where(x>0,x,np.exp(x)-1),
         lambda x: 1.0507*np.where(x>0,x,1.67326*(np.exp(x)-1)), 
         lambda x: np.where(x>0,x,0.01*x)]

def compute_Phi_fast(X):
    dist = cdist(X, X, metric='euclidean')
    eps = 5.0
    F = np.sum((dist < eps).astype(int)) / (X.shape[0]**2)
    return [F, 0.5, -1.0]

def get_manifold(X):
    phis = [compute_Phi_fast(f(X)) for f in funcs]
    phis = np.array(phis).T
    
    n = phis.shape[1]
    D = cdist(phis.T, phis.T)
    D2 = D**2
    J = np.eye(n) - np.ones((n, n))/n
    B = -0.5 * J @ D2 @ J
    evals, evecs = np.linalg.eigh(B)
    idx = np.argsort(evals)[::-1]
    evals, evecs = evals[idx], evecs[:, idx]
    X1 = evecs[:,0] * np.sqrt(max(evals[0], 0))
    
    return np.var(X1), len(np.unique(np.round(X1, 3)))

# ==============================================================================
# STEP 1: MASTER DATASET
# ==============================================================================
print("Step 1: Building dataset...")

systems = {
    'gaussian': np.random.randn,
    'laplace': lambda n, d: np.random.laplace(0, 1, (n, d)),
    'student_t_3': lambda n, d: np.random.standard_t(3, (n, d)),
    'uniform': lambda n, d: np.random.uniform(-np.sqrt(3), np.sqrt(3), (n, d)),
    'exponential': lambda n, d: np.random.exponential(1, (n, d)) - 1,
}

master_data = []
for sys_name, gen in systems.items():
    for trial in range(3):
        X = gen(n_samples, n_dim)
        X = X - X.mean(axis=0)
        var_x1, unique_x1 = get_manifold(X)
        
        cov = np.cov(X, rowvar=False)
        evals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        evals = evals[evals > 0]
        eff_rank = np.exp(-np.sum((evals/np.sum(evals)) * np.log(evals/np.sum(evals) + 1e-8)))
        part_ratio = (np.sum(evals)**2) / np.sum(evals**2) if len(evals) > 0 else 0
        fisher = np.sum(1 / (evals + 1e-8))
        
        master_data.append({
            'system': sys_name, 'trial': trial,
            'var_x1': var_x1, 'unique_x1': unique_x1,
            'effective_rank': eff_rank, 'participation_ratio': part_ratio,
            'fisher_metric': fisher
        })

print(f"  {len(master_data)} conditions")

# ==============================================================================
# STEP 2: TRANSFORMS
# ==============================================================================
print("Step 2: Transforms...")

X_base = np.random.randn(n_samples, n_dim)
X_base = X_base - X_base.mean(axis=0)
base_var, base_unique = get_manifold(X_base)

transforms = [
    ('orthogonal', lambda x: x @ np.linalg.qr(np.random.randn(n_dim, n_dim))[0]),
    ('permute', lambda x: x[:, np.random.permutation(n_dim)]),
    ('add_noise', lambda x: x + np.random.randn(*x.shape) * 0.1),
    ('scale', lambda x: x * 1.5),
]

transform_results = []
for name, func in transforms:
    X_t = func(X_base.copy())
    X_t = X_t - X_t.mean(axis=0)
    var_t, unique_t = get_manifold(X_t)
    transform_results.append({
        'transform': name,
        'var_x1': var_t, 'unique_x1': unique_t,
        'distortion': abs(var_t - base_var)
    })

# ==============================================================================
# STEP 3: INVARIANT SEARCH
# ==============================================================================
print("Step 3: Invariant search...")

features = ['var_x1', 'unique_x1', 'effective_rank', 'participation_ratio', 'fisher_metric']
vals = {f: [d[f] for d in master_data] for f in features}

inv_scores = {}
for f in features:
    m, s = np.mean(vals[f]), np.std(vals[f])
    cv = s / (abs(m) + 1e-8)
    inv_scores[f] = 1 - min(cv, 1)

ranked = sorted(inv_scores.items(), key=lambda x: x[1], reverse=True)
print("  Rankings:", [(k, round(v,3)) for k,v in ranked[:5]])

# ==============================================================================
# STEP 4: META GEOMETRY
# ==============================================================================
print("Step 4: Meta geometry...")

X_meta = np.array([[d[f] for f in features[:4]] for d in master_data])
pca = PCA(n_components=2)
meta_pca = pca.fit_transform(X_meta)

cumvar = np.cumsum(pca.explained_variance_ratio_)
intrinsic_dim = np.argmax(cumvar >= 0.95) + 1

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_meta)
cluster_count = len(set(labels))

meta_geom = {'intrinsic_dim': intrinsic_dim, 'clusters': cluster_count}

# ==============================================================================
# STEP 5: LAW DISCOVERY
# ==============================================================================
print("Step 5: Law discovery...")

target = np.array([d['var_x1'] for d in master_data])
X_inv = np.array([[d[f] for f in features[:4]] for d in master_data])

lr = LinearRegression()
lr.fit(X_inv, target)
r2 = lr.score(X_inv, target)
print(f"  Linear R2 = {r2:.4f}")

# ==============================================================================
# STEP 6: FALSIFICATION
# ==============================================================================
print("Step 6: Falsification...")

adversarial = [
    ('random_sym', lambda: np.sin(X_base) * np.cos(X_base**2)),
    ('pathological', lambda: X_base @ np.diag(np.random.exponential(0.1, n_dim))),
]

falsify_results = []
for name, func in adversarial:
    try:
        X_a = func()
        X_a = X_a - X_a.mean(axis=0)
        var_a, _ = get_manifold(X_a)
        mean_v = np.mean([d['var_x1'] for d in master_data])
        std_v = np.std([d['var_x1'] for d in master_data])
        is_outlier = abs(var_a - mean_v) > 3 * std_v
        falsify_results.append({'system': name, 'var_x1': var_a, 'outlier': is_outlier})
    except Exception as e:
        falsify_results.append({'system': name, 'error': str(e)})

broken = sum(1 for r in falsify_results if r.get('outlier', False))
stable = len(falsify_results) - broken

# ==============================================================================
# STEP 7: FINAL SCORE
# ==============================================================================
print("Step 7: Final score...")

inv_mean = np.mean([v for _,v in ranked[:3]])
meta_stab = 1 - (cluster_count - 1) / 3
fals_stab = stable / len(falsify_results) if len(falsify_results) > 0 else 0

U_final = (inv_mean + r2 + meta_stab + fals_stab) / 4
print(f"  U_final = {U_final:.4f}")

# ==============================================================================
# OUTPUT
# ==============================================================================
print()
print("="*60)
print("OUTPUT")
print("="*60)

print("\nINVARIANT RANKINGS:")
for f, s in ranked:
    print(f"  {f} = {s:.4f}")

print(f"\nMETA-MANIFOLD:")
print(f"  intrinsic_dimension = {intrinsic_dim}")
print(f"  cluster_count = {cluster_count}")
print(f"  topology = {'connected' if cluster_count > 1 else 'single'}")

print(f"\nUNIVERSAL LAW:")
print(f"  best_model = linear")
print(f"  R2 = {r2:.4f}")

print(f"\nFALSIFICATION:")
print(f"  broken_system_count = {broken}")
print(f"  stable_system_count = {stable}")

print(f"\nUNIVERSALITY:")
print(f"  U_final = {U_final:.4f}")

verdict = 'universal_geometric_organization' if U_final > 0.7 else ('partial_universal_organization' if U_final > 0.4 else 'no_universal_organization')
print(f"\nVERDICT: geometric_organization_type = {verdict}")

# SAVE
results = {
    'invariant_rankings': {k: float(v) for k,v in ranked},
    'meta_geometry': meta_geom,
    'law_r2': float(r2),
    'falsification': {'broken': broken, 'stable': stable},
    'U_final': float(U_final),
    'verdict': verdict
}

with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to {OUTPUT_DIR}")
print("="*60)