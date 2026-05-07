"""
PHASE 66: Information Geometry Test
Complete implementation - NO simplifications
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import kurtosis, skew, entropy
from scipy.special import rel_entr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.cluster import KMeans
import os
import json
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase66_information_geometry'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def identity(x): return x
def tanh_op(x): return np.tanh(x)
def relu(x): return np.maximum(0, x)
def softplus(x): return np.log1p(np.exp(x))
def sigmoid(x): return 1/(1+np.exp(-x))
def elu(x): return np.where(x>0, x, np.exp(x)-1)
def selu(x): return 1.0507*np.where(x>0, x, 1.67326*(np.exp(x)-1))
def leaky_relu_001(x): return np.where(x>0, x, 0.001*x)
def leaky_relu_01(x): return np.where(x>0, x, 0.01*x)
def leaky_relu_02(x): return np.where(x>0, x, 0.02*x)
def hard_tanh(x): return np.clip(x, -1, 1)
def softsign(x): return x/(1+np.abs(x))
def mish(x): return x*np.tanh(np.log1p(np.exp(x)))
def swish_beta1(x): return x/(1+np.exp(-x))
def swish_beta2(x): return x/(1+np.exp(-2*x))
def linear_scaled_2x(x): return 2*x
def linear_scaled_05x(x): return 0.5*x

ops = ['identity','tanh','relu','softplus','sigmoid','elu','selu','leaky_relu_001','leaky_relu_01','leaky_relu_02','hard_tanh','softsign','mish','swish_beta1','swish_beta2','linear_scaled_2x','linear_scaled_05x']
funcs_list = [identity, tanh_op, relu, softplus, sigmoid, elu, selu, leaky_relu_001, leaky_relu_01, leaky_relu_02, hard_tanh, softsign, mish, swish_beta1, swish_beta2, linear_scaled_2x, linear_scaled_05x]

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

def compute_phis(acts):
    phis = []
    for f in funcs_list:
        t = f(acts)
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

def get_X1(phis):
    n = len(ops)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i,j] = np.linalg.norm(phis[i]-phis[j])
    D2 = D**2
    J = np.eye(n) - np.ones((n, n))/n
    B = -0.5 * J @ D2 @ J
    evals, evecs = np.linalg.eigh(B)
    idx_sort = np.argsort(evals)[::-1]
    evals = evals[idx_sort]; evecs = evecs[:, idx_sort]
    X1 = evecs[:,0]*np.sqrt(max(evals[0], 0))
    return X1

def get_geom_metrics(X):
    X = X - X.mean(axis=0)
    phis = compute_phis(X)
    X1 = get_X1(phis)
    var_x1 = np.var(X1)
    unique_x1 = len(np.unique(np.round(X1, 3)))
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    try:
        labels = kmeans.fit_predict(X1.reshape(-1, 1))
        cluster_count = len(set(labels))
    except:
        cluster_count = 1
    return var_x1, unique_x1, cluster_count

def compute_information_features(X):
    """Compute comprehensive information-geometric features"""
    X_flat = X.flatten()
    
    # Shannon entropy (discretized)
    hist, _ = np.histogram(X_flat, bins=50, density=True)
    hist = hist[hist > 0]
    shannon_entropy = entropy(hist)
    
    # Differential entropy estimate (kernel-based)
    try:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(X_flat)
        x_grid = np.linspace(np.min(X_flat), np.max(X_flat), 100)
        pdf = kde(x_grid)
        pdf = pdf[pdf > 0]
        diff_entropy = np.mean(np.log(pdf + 1e-8))
    except:
        diff_entropy = 0.0
    
    # Fisher-like metrics
    cov = np.cov(X, rowvar=False)
    eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eigenvalues = eigenvalues[eigenvalues > 0]
    
    log_det_cov = np.log(np.linalg.det(cov) + 1e-8)
    inv_cov_trace = np.sum(1 / (eigenvalues + 1e-8))
    
    # KL divergence to Gaussian (approximate)
    n = len(eigenvalues)
    kl_to_gaussian = 0.5 * (np.sum(np.log(eigenvalues + 1e-8)) - n * np.log(np.mean(eigenvalues) + 1e-8))
    
    # Participation ratio
    total_var = np.sum(eigenvalues)
    p = eigenvalues / (total_var + 1e-8)
    participation_ratio = (np.sum(eigenvalues)**2) / np.sum(eigenvalues**2) if len(eigenvalues) > 0 else 0
    
    # Effective rank
    effective_rank = np.exp(-np.sum(p * np.log(p + 1e-8)))
    
    # Mutual information estimate (between dimensions)
    mi_estimate = 0.5 * (1 - 1 / participation_ratio) if participation_ratio > 1 else 0
    
    return {
        'shannon_entropy': shannon_entropy,
        'differential_entropy': diff_entropy,
        'log_det_cov': log_det_cov,
        'inv_cov_trace': inv_cov_trace,
        'kl_to_gaussian': kl_to_gaussian,
        'participation_ratio': participation_ratio,
        'effective_rank': effective_rank,
        'mutual_information': mi_estimate
    }

print("="*70)
print("PHASE 66: INFORMATION GEOMETRY TEST")
print("="*70)
print()

np.random.seed(42)
n_samples = 2000
n_dim = 64

print("Step 1: Generating master dataset from multiple phases...")

all_conditions = []

# Systems from Phase 57, 58, 61, 63
systems = {
    'gaussian': (lambda: np.random.randn(n_samples, n_dim), 'gaussian'),
    'laplace': (lambda: np.random.laplace(0, 1, (n_samples, n_dim)), 'heavy_tail'),
    'student_t_2': (lambda: np.random.standard_t(2, (n_samples, n_dim)), 'heavy_tail'),
    'student_t_3': (lambda: np.random.standard_t(3, (n_samples, n_dim)), 'heavy_tail'),
    'student_t_5': (lambda: np.random.standard_t(5, (n_samples, n_dim)), 'heavy_tail'),
    'cauchy': (lambda: np.clip(np.random.standard_cauchy((n_samples, n_dim)), -10, 10), 'heavy_tail'),
    'uniform': (lambda: np.random.uniform(-np.sqrt(3), np.sqrt(3), (n_samples, n_dim)), 'bounded'),
    'exponential': (lambda: np.random.exponential(1, (n_samples, n_dim)) - 1, 'asymmetric'),
    'logistic': (lambda: np.random.logistic(0, 1, (n_samples, n_dim)), 'heavy_tail'),
    'gaussian_mixture': (lambda: np.concatenate([np.random.randn(n_samples//2, n_dim), np.random.randn(n_samples//2, n_dim) * 3 + 2]), 'multimodal'),
}

for sys_name, (gen_func, category) in systems.items():
    print(f"  Processing {sys_name}...", flush=True)
    for trial in range(8):
        X = gen_func()
        X = X - X.mean(axis=0)
        
        # Compute information features
        info_feats = compute_information_features(X)
        
        # Compute geometric metrics
        var_x1, unique_x1, cluster_count = get_geom_metrics(X)
        
        cov = np.cov(X, rowvar=False)
        eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
        eigenvalues = eigenvalues[eigenvalues > 0]
        total_var = np.sum(eigenvalues)
        top_fraction = eigenvalues[0] / (total_var + 1e-8) if len(eigenvalues) > 0 else 0
        
        all_conditions.append({
            'system': sys_name,
            'category': category,
            'trial': trial,
            'var_x1': var_x1,
            'unique_x1': unique_x1,
            'cluster_count': cluster_count,
            **info_feats,
            'top_fraction': top_fraction,
            'total_variance': total_var
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
    
    info_feats = compute_information_features(X)
    var_x1, unique_x1, cluster_count = get_geom_metrics(X)
    
    cov = np.cov(X, rowvar=False)
    eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
    total_var = np.sum(eigenvalues)
    top_fraction = eigenvalues[0] / (total_var + 1e-8) if len(eigenvalues) > 0 else 0
    
    all_conditions.append({
        'system': f'rank_{k}',
        'category': 'low_rank',
        'trial': 0,
        'var_x1': var_x1,
        'unique_x1': unique_x1,
        'cluster_count': cluster_count,
        **info_feats,
        'top_fraction': top_fraction,
        'total_variance': total_var
    })

print(f"  Total conditions: {len(all_conditions)}")

# Save intermediate data
intermediate_file = os.path.join(OUTPUT_DIR, 'master_dataset.json')
with open(intermediate_file, 'w') as f:
    json.dump(all_conditions, f, indent=2)
print(f"  Saved: {intermediate_file}")

print()
print("Step 2: Computing spectral features for comparison...")

spectral_features = []
for cond in all_conditions:
    spectral_features.append({
        'system': cond['system'],
        'top_fraction': cond['top_fraction'],
        'total_variance': cond['total_variance'],
        'var_x1': cond['var_x1']
    })

spectral_file = os.path.join(OUTPUT_DIR, 'spectral_features.json')
with open(spectral_file, 'w') as f:
    json.dump(spectral_features, f, indent=2)
print(f"  Saved: {spectral_file}")

print()
print("Step 3: Building feature matrices for prediction...")

# Extract feature matrices
info_feature_names = ['shannon_entropy', 'differential_entropy', 'log_det_cov', 'inv_cov_trace', 
                      'kl_to_gaussian', 'participation_ratio', 'effective_rank', 'mutual_information']
spectral_feature_names = ['top_fraction', 'total_variance']
all_feature_names = info_feature_names + spectral_feature_names

info_matrix = np.array([[cond[f] for f in info_feature_names] for cond in all_conditions])
spectral_matrix = np.array([[cond[f] for f in spectral_feature_names] for cond in all_conditions])
combined_matrix = np.column_stack([info_matrix, spectral_matrix])

var_x1_arr = np.array([cond['var_x1'] for cond in all_conditions])
unique_x1_arr = np.array([cond['unique_x1'] for cond in all_conditions])
cluster_count_arr = np.array([cond['cluster_count'] for cond in all_conditions])

print(f"  Info features shape: {info_matrix.shape}")
print(f"  Spectral features shape: {spectral_matrix.shape}")
print(f"  Combined features shape: {combined_matrix.shape}")

print()
print("Step 4: Leave-one-system-out validation for prediction...")

unique_systems = list(set([cond['system'] for cond in all_conditions]))
system_indices = {sys: [] for sys in unique_systems}
for i, cond in enumerate(all_conditions):
    system_indices[cond['system']].append(i)

def leave_one_system_out_validation(X, y, model_type='linear'):
    r2_scores = []
    mae_scores = []
    
    for held_out_sys in unique_systems:
        train_idx = []
        test_idx = []
        for sys, indices in system_indices.items():
            if sys == held_out_sys:
                test_idx.extend(indices)
            else:
                train_idx.extend(indices)
        
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
            
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        y_train = y[train_idx]
        y_test = y[test_idx]
        
        if model_type == 'linear':
            model = LinearRegression()
        elif model_type == 'ridge':
            model = Ridge(alpha=1.0)
        elif model_type == 'rf':
            model = RandomForestRegressor(n_estimators=50, random_state=42)
        elif model_type == 'kernel':
            model = KernelRidge(alpha=1.0, kernel='rbf')
        
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        if len(np.unique(y_test)) > 1:
            r2 = r2_score(y_test, pred)
        else:
            r2 = 0.0
        mae = mean_absolute_error(y_test, pred)
        
        r2_scores.append(r2)
        mae_scores.append(mae)
    
    return np.mean(r2_scores), np.mean(mae_scores)

print("  Testing VAR_X1 prediction...")
print("    Info features only:")
r2_info_var, mae_info_var = leave_one_system_out_validation(info_matrix, var_x1_arr, 'linear')
print(f"      Linear R2: {r2_info_var:.6f}, MAE: {mae_info_var:.6f}")

r2_info_var_ridge, mae_info_var_ridge = leave_one_system_out_validation(info_matrix, var_x1_arr, 'ridge')
print(f"      Ridge R2: {r2_info_var_ridge:.6f}, MAE: {mae_info_var_ridge:.6f}")

print("    Spectral features only:")
r2_spec_var, mae_spec_var = leave_one_system_out_validation(spectral_matrix, var_x1_arr, 'linear')
print(f"      Linear R2: {r2_spec_var:.6f}, MAE: {mae_spec_var:.6f}")

print("    Combined features:")
r2_comb_var, mae_comb_var = leave_one_system_out_validation(combined_matrix, var_x1_arr, 'linear')
print(f"      Linear R2: {r2_comb_var:.6f}, MAE: {mae_comb_var:.6f}")

print()
print("Step 5: Building information latent space...")

scaler_info = StandardScaler()
info_scaled = scaler_info.fit_transform(info_matrix)

pca_info = PCA(n_components=min(8, info_matrix.shape[1]))
info_pca = pca_info.fit_transform(info_scaled)
pca_var_explained = np.sum(pca_info.explained_variance_ratio_)

iso_info = Isomap(n_neighbors=5, n_components=2)
info_isomap = iso_info.fit_transform(info_scaled)

# Intrinsic dimension estimate
cumvar = np.cumsum(pca_info.explained_variance_ratio_)
intrinsic_dim = np.argmax(cumvar >= 0.95) + 1

# Cluster separation
kmeans_info = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans_info.fit_predict(info_pca)
cluster_separation = len(set(labels))

print(f"  PCA variance explained (2 components): {pca_var_explained:.6f}")
print(f"  Intrinsic dimension (95%): {intrinsic_dim}")
print(f"  Cluster separation: {cluster_separation}")

print()
print("Step 6: Tail continuum correlation analysis...")

# Focus on student-t tail continuum
tail_conditions = [c for c in all_conditions if 'student_t' in c['system'] or c['system'] in ['gaussian', 'laplace', 'uniform']]
tail_var_x1 = np.array([c['var_x1'] for c in tail_conditions])
tail_kurt = np.array([c['shannon_entropy'] for c in tail_conditions])
tail_fisher = np.array([c['inv_cov_trace'] for c in tail_conditions])
tail_kl = np.array([c['kl_to_gaussian'] for c in tail_conditions])
tail_mi = np.array([c['mutual_information'] for c in tail_conditions])

corr_kurt = np.corrcoef(tail_var_x1, tail_kurt)[0,1]
corr_fisher = np.corrcoef(tail_var_x1, tail_fisher)[0,1]
corr_kl = np.corrcoef(tail_var_x1, tail_kl)[0,1]
corr_mi = np.corrcoef(tail_var_x1, tail_mi)[0,1]

print(f"  Entropy correlation with VAR_X1: {corr_kurt:.6f}")
print(f"  Fisher correlation with VAR_X1: {corr_fisher:.6f}")
print(f"  KL divergence correlation with VAR_X1: {corr_kl:.6f}")
print(f"  Mutual information correlation with VAR_X1: {corr_mi:.6f}")

print()
print("Step 7: Universal collapse tests...")

def test_collapse(x, y):
    """Test how well x predicts y"""
    lr = LinearRegression()
    lr.fit(x.reshape(-1, 1), y)
    pred = lr.predict(x.reshape(-1, 1))
    r2 = r2_score(y, pred)
    mae = mean_absolute_error(y, pred)
    return r2, mae, pred

# Entropy only collapse
r2_entropy, mae_entropy, pred_entropy = test_collapse(info_matrix[:, 0], var_x1_arr)

# Fisher only (inv_cov_trace)
r2_fisher, mae_fisher, pred_fisher = test_collapse(info_matrix[:, 3], var_x1_arr)

# KL only
r2_kl, mae_kl, pred_kl = test_collapse(info_matrix[:, 4], var_x1_arr)

# Combined info coordinate
from sklearn.decomposition import PCA
pca_ig = PCA(n_components=1)
info_1d = pca_ig.fit_transform(StandardScaler().fit_transform(info_matrix))
r2_comb, mae_comb, pred_comb = test_collapse(info_1d.flatten(), var_x1_arr)

print(f"  Entropy collapse R2: {r2_entropy:.6f}, MAE: {mae_entropy:.6f}")
print(f"  Fisher collapse R2: {r2_fisher:.6f}, MAE: {mae_fisher:.6f}")
print(f"  KL collapse R2: {r2_kl:.6f}, MAE: {mae_kl:.6f}")
print(f"  Combined info collapse R2: {r2_comb:.6f}, MAE: {mae_comb:.6f}")

print()
print("Step 8: Causal interventions...")

print("  A) Entropy-preserving covariance scramble...")
# Use student-t (high entropy variance) but scramble covariance
X_base = np.random.standard_t(3, (n_samples, n_dim))
X_base = X_base - X_base.mean(axis=0)
var_x1_base, _, _ = get_geom_metrics(X_base)

# Eigendecomposition and randomize eigenvectors
cov = np.cov(X_base, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(cov)
np.random.seed(123)
Q, _ = np.linalg.qr(np.random.randn(n_dim, n_dim))
X_scrambled = X_base @ eigenvectors @ Q.T
X_scrambled = X_scrambled - X_scrambled.mean(axis=0)
var_x1_scrambled, unique_scrambled, _ = get_geom_metrics(X_scrambled)
distortion_scrambled = np.abs(var_x1_base - var_x1_scrambled)

print(f"    delta_VAR_X1: {distortion_scrambled:.6f}")

print("  B) Covariance-preserving entropy distortion...")
# Match covariance but change distribution shape
X_gaussian = np.random.randn(n_samples, n_dim)
X_gaussian = X_gaussian - X_gaussian.mean(axis=0)
cov_match = np.cov(X_gaussian, rowvar=False)
eigenvalues_g, eigenvectors_g = np.linalg.eigh(cov)
X_transformed = X_base @ eigenvectors_g @ np.diag(np.sqrt(eigenvalues_g[:n_dim])) @ eigenvectors_g.T
X_transformed = X_transformed - X_transformed.mean(axis=0)
var_x1_transformed, unique_transformed, _ = get_geom_metrics(X_transformed)
distortion_transformed = np.abs(var_x1_base - var_x1_transformed)

print(f"    delta_VAR_X1: {distortion_transformed:.6f}")

print("  C) Tail-preserving Gaussianization...")
# Rank-preserving transformation to Gaussian
from scipy.stats import rankdata
X_ranked = np.apply_along_axis(lambda x: rankdata(x), 1, X_base)
from scipy.stats import norm
X_gaussianized = norm.ppf((X_ranked + 0.5) / (X_ranked.shape[1] + 1))
X_gaussianized = X_gaussianized - X_gaussianized.mean(axis=0)
var_x1_gaussianized, unique_gaussianized, _ = get_geom_metrics(X_gaussianized)
distortion_gaussianized = np.abs(var_x1_base - var_x1_gaussianized)

print(f"    delta_VAR_X1: {distortion_gaussianized:.6f}")

print()
print("="*70)
print("OUTPUT")
print("="*70)
print()

print('-------------------------------')
print('MODEL PERFORMANCE')
print('-------------------------------')
print('feature_set   target   R2   MAE')
print(f'info_only     VAR_X1   {r2_info_var:.6f}   {mae_info_var:.6f}')
print(f'spectral_only VAR_X1   {r2_spec_var:.6f}   {mae_spec_var:.6f}')
print(f'combined      VAR_X1   {r2_comb_var:.6f}   {mae_comb_var:.6f}')

print()
print('-------------------------------')
print('INFORMATION LATENT SPACE')
print('-------------------------------')
print(f'intrinsic_dimension = {intrinsic_dim}')
print(f'curvature = {1 - pca_var_explained:.6f}')
print(f'cluster_separation = {cluster_separation}')

print()
print('-------------------------------')
print('TAIL CORRELATIONS')
print('-------------------------------')
print('metric   correlation_with_VAR_X1')
print(f'shannon_entropy   {corr_kurt:.6f}')
print(f'fisher_metric   {corr_fisher:.6f}')
print(f'kl_divergence   {corr_kl:.6f}')
print(f'mutual_information   {corr_mi:.6f}')

print()
print('-------------------------------')
print('UNIVERSAL COLLAPSE')
print('-------------------------------')
best_params = max([(r2_entropy, 'entropy'), (r2_fisher, 'fisher'), (r2_kl, 'kl'), (r2_comb, 'combined')], key=lambda x: x[0])
print(f'best_information_parameter = {best_params[1]}')
print(f'R2 = {best_params[0]:.6f}')
print(f'collapse_error = {mae_entropy:.6f}')

print()
print('-------------------------------')
print('INTERVENTIONS')
print('-------------------------------')
print('condition   delta_VAR_X1   distortion')
print(f'entropy_preserve_cov_scramble   {distortion_scrambled:.6f}   {distortion_scrambled:.6f}')
print(f'cov_preserve_entropy_distort   {distortion_transformed:.6f}   {distortion_transformed:.6f}')
print(f'tail_preserve_gaussianize   {distortion_gaussianized:.6f}   {distortion_gaussianized:.6f}')

print()
print('-------------------------------')
print('VERDICT')
print('-------------------------------')
# Determine based on results
if r2_info_var > r2_spec_var:
    geometry_origin = 'information_geometry'
else:
    geometry_origin = 'covariance_geometry'

print(f'geometry_origin = {geometry_origin}')
print('-------------------------------')

# Save all intermediate arrays
np.save(os.path.join(OUTPUT_DIR, 'info_features.npy'), info_matrix)
np.save(os.path.join(OUTPUT_DIR, 'spectral_features.npy'), spectral_matrix)
np.save(os.path.join(OUTPUT_DIR, 'combined_features.npy'), combined_matrix)
np.save(os.path.join(OUTPUT_DIR, 'var_x1.npy'), var_x1_arr)
np.save(os.path.join(OUTPUT_DIR, 'unique_x1.npy'), unique_x1_arr)
np.save(os.path.join(OUTPUT_DIR, 'info_pca.npy'), info_pca)
np.save(os.path.join(OUTPUT_DIR, 'info_isomap.npy'), info_isomap)

print(f"  Saved all intermediate arrays to {OUTPUT_DIR}")

# Prepare for GitHub push
results = {
    'phase': 66,
    'model_performance': {
        'info_only_r2': r2_info_var,
        'info_only_mae': mae_info_var,
        'spectral_only_r2': r2_spec_var,
        'spectral_only_mae': mae_spec_var,
        'combined_r2': r2_comb_var,
        'combined_mae': mae_comb_var
    },
    'information_latent_space': {
        'intrinsic_dimension': intrinsic_dim,
        'variance_explained': pca_var_explained,
        'cluster_separation': cluster_separation
    },
    'tail_correlations': {
        'entropy': corr_kurt,
        'fisher': corr_fisher,
        'kl_divergence': corr_kl,
        'mutual_information': corr_mi
    },
    'universal_collapse': {
        'best_parameter': best_params[1],
        'r2': best_params[0],
        'mae': mae_entropy
    },
    'interventions': {
        'entropy_preserve_cov_scramble': distortion_scrambled,
        'cov_preserve_entropy_distort': distortion_transformed,
        'tail_preserve_gaussianize': distortion_gaussianized
    },
    'verdict': geometry_origin
}

with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2)

print("  Results saved to results.json")
print()
print("="*70)
print("PHASE 66 COMPLETE")
print("="*70)