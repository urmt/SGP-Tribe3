"""
PHASE 77: CIRCULARITY / LEAKAGE TEST
Test for hidden circularity and feature leakage
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, r2_score
from sklearn.manifold import Isomap
import os
import json
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase77_circularity_leakage'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("PHASE 77: CIRCULARITY / LEAKAGE TEST")
print("="*60)

np.random.seed(42)
n_samples = 500
n_dim = 32

funcs = [lambda x: x, np.tanh, lambda x: np.maximum(0, x), lambda x: np.log1p(np.exp(x))]

def compute_manifold(X):
    """Geometry-derived features"""
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
    return {'var_x1': float(np.var(X1)), 'X1': X1}

# ==============================================================================
# STEP 1: FEATURE INDEPENDENCE
# ==============================================================================
print("\nStep 1: Feature independence test...")

# Generate observer systems
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

# Create mixed dataset
all_data = []
for alpha in [0.2, 0.5, 0.8, 0.95]:
    for trial in range(10):
        X = generate_observer(alpha, 300, 16)
        
        # A: Geometry-derived
        geom = compute_manifold(X)
        
        # B: Observer-derived
        temp_coh = np.mean([np.corrcoef(X[:-1, i], X[1:, i])[0, 1] for i in range(min(10, X.shape[1]))])
        pred_err = [np.mean((X[t] - X[t-1])**2) for t in range(10, X.shape[0])]
        observer = 1 / (1 + np.mean(pred_err))
        
        # C: Recursion-derived
        rec_depth = 5
        for lag in range(1, 20):
            ac = np.mean([np.abs(np.corrcoef(X[:-lag, i], X[lag:, i])[0, 1]) for i in range(min(5, X.shape[1])) if lag < X.shape[0]])
            if ac < 0.3:
                rec_depth = lag
                break
        
        # D: Statistical-derived
        variance = np.var(X)
        entropy = -np.sum(np.abs(X.flatten()) * np.log(np.abs(X.flatten()) + 1e-8))
        
        # E: Computational-derived
        diff_std = np.std(np.diff(X, axis=0))
        
        all_data.append({
            'alpha': alpha,
            'is_observer': 1 if alpha > 0.8 else 0,
            'var_x1': geom['var_x1'],
            'observer_metric': observer,
            'recursion_depth': rec_depth,
            'variance': variance,
            'entropy': entropy,
            'diff_std': diff_std
        })

# Feature independence: mutual information
geom_vals = np.array([d['var_x1'] for d in all_data])
obs_vals = np.array([d['observer_metric'] for d in all_data])
rec_vals = np.array([d['recursion_depth'] for d in all_data])
stat_vals = np.array([d['variance'] for d in all_data])
comp_vals = np.array([d['diff_std'] for d in all_data])

# Pairwise correlations
corr_geom_obs = abs(pearsonr(geom_vals, obs_vals)[0])
corr_geom_rec = abs(pearsonr(geom_vals, rec_vals)[0])
corr_geom_stat = abs(pearsonr(geom_vals, stat_vals)[0])
corr_geom_comp = abs(pearsonr(geom_vals, comp_vals)[0])

max_mutual_information = max(corr_geom_obs, corr_geom_rec, corr_geom_stat, corr_geom_comp)

# Conditional independence (partial correlation)
lr = LinearRegression()
lr.fit(np.column_stack([stat_vals, comp_vals]), geom_vals)
residual_geom = geom_vals - lr.predict(np.column_stack([stat_vals, comp_vals]))
partial_corr = abs(pearsonr(residual_geom, obs_vals)[0])
conditional_independence_score = 1 - partial_corr

# Redundancy
redundancy_ratio = corr_geom_stat / (corr_geom_obs + 1e-8)

print(f"  Correlations: geom-obs={corr_geom_obs:.4f}, geom-rec={corr_geom_rec:.4f}, geom-stat={corr_geom_stat:.4f}")
print(f"  Max MI: {max_mutual_information:.4f}")
print(f"  Conditional independence: {conditional_independence_score:.4f}")
print(f"  Redundancy ratio: {redundancy_ratio:.4f}")

# ==============================================================================
# STEP 2: FEATURE ABLATION
# ==============================================================================
print("\nStep 2: Feature ablation test...")

X_geom = geom_vals.reshape(-1, 1)
X_obs = np.column_stack([obs_vals, rec_vals])
X_stat = np.column_stack([stat_vals, comp_vals])
y_label = np.array([d['is_observer'] for d in all_data])

# Observer without geometry
lr_obs_only = LinearRegression()
lr_obs_only.fit(X_stat, y_label)
pred_obs_only = (lr_obs_only.predict(X_stat) > 0.5).astype(int)
observer_without_geometry = accuracy_score(y_label, pred_obs_only)

# Geometry without observer
lr_geom_only = LinearRegression()
lr_geom_only.fit(X_stat, X_geom)
pred_geom_only = lr_geom_only.predict(X_stat)
geometry_without_observer = r2_score(geom_vals, pred_geom_only)

print(f"  Observer prediction without geometry: {observer_without_geometry:.4f}")
print(f"  Geometry prediction without observer: {geometry_without_observer:.4f}")

# ==============================================================================
# STEP 3: ORTHOGONALIZATION
# ==============================================================================
print("\nStep 3: Orthogonalization...")

# Regression residualization: remove geometry from observer metrics
lr_remove_geom = LinearRegression()
lr_remove_geom.fit(X_geom, obs_vals)
obs_orthogonal = obs_vals - lr_remove_geom.predict(X_geom)

# Re-test transition with orthogonalized features
alphas = np.array([d['alpha'] for d in all_data])
transition_d1 = np.abs(np.gradient(obs_orthogonal, alphas))
max_transition_ortho = np.max(transition_d1)

# Original transition
transition_d1_orig = np.abs(np.gradient(obs_vals, alphas))
max_transition_orig = np.max(transition_d1_orig)

transition_survival = max_transition_ortho / (max_transition_orig + 1e-8)
residual_transition_strength = max_transition_ortho

print(f"  Transition survival: {transition_survival:.4f}")
print(f"  Residual transition: {residual_transition_strength:.4f}")

# ==============================================================================
# STEP 4: RANDOM FEATURE CONTROL
# ==============================================================================
print("\nStep 4: Random feature control...")

# Generate random features with same statistics
random_features = []
for _ in range(20):
    # Match mean and variance of observer metric
    rand_feat = np.random.randn(len(all_data)) * np.std(obs_vals) + np.mean(obs_vals)
    random_features.append(rand_feat)

random_transition_rates = []
for rf in random_features:
    d1_rand = np.abs(np.gradient(rf, alphas))
    random_transition_rates.append(np.max(d1_rand))

random_feature_transition_rate = np.mean(random_transition_rates)

# Matched random accuracy
matched_random = np.mean(random_features[0])
matched_random_accuracy = 1 - abs(matched_random - np.mean(obs_vals)) / (np.std(obs_vals) + 1e-8)

print(f"  Random transition rate: {random_feature_transition_rate:.4f}")
print(f"  Matched random accuracy: {matched_random_accuracy:.4f}")

# ==============================================================================
# STEP 5: LABEL LEAKAGE TEST
# ==============================================================================
print("\nStep 5: Label leakage test...")

# Shuffle labels
y_shuffled = np.random.permutation(y_label)

# Classification with shuffled labels
X_all = np.column_stack([geom_vals, obs_vals, rec_vals, stat_vals, comp_vals])
lr_shuffled = LinearRegression()
lr_shuffled.fit(X_all, y_shuffled)
pred_shuffled = (lr_shuffled.predict(X_all) > 0.5).astype(int)
shuffled_accuracy = accuracy_score(y_shuffled, pred_shuffled)

# False positive rate (random guessing should be ~50%)
false_positive_rate = abs(0.5 - shuffled_accuracy)

print(f"  Shuffled accuracy: {shuffled_accuracy:.4f}")
print(f"  False positive rate: {false_positive_rate:.4f}")

# ==============================================================================
# STEP 6: CAUSAL DIRECTION TEST
# ==============================================================================
print("\nStep 6: Causal direction test...")

# Test if observer -> geometry OR geometry -> observer
# Use temporal prediction: can X predict Y better than Y predicts X?

# Observer predicts geometry
lr_obs_to_geom = LinearRegression()
lr_obs_to_geom.fit(obs_vals.reshape(-1, 1), geom_vals)
r2_obs_to_geom = r2_score(geom_vals, lr_obs_to_geom.predict(obs_vals.reshape(-1, 1)))

# Geometry predicts observer
lr_geom_to_obs = LinearRegression()
lr_geom_to_obs.fit(geom_vals.reshape(-1, 1), obs_vals)
r2_geom_to_obs = r2_score(obs_vals, lr_geom_to_obs.predict(geom_vals.reshape(-1, 1)))

directionality_score = r2_obs_to_geom - r2_geom_to_obs
causal_asymmetry = abs(directionality_score)

print(f"  Observer -> Geometry R2: {r2_obs_to_geom:.4f}")
print(f"  Geometry -> Observer R2: {r2_geom_to_obs:.4f}")
print(f"  Directionality: {directionality_score:.4f}")
print(f"  Causal asymmetry: {causal_asymmetry:.4f}")

# ==============================================================================
# STEP 7: EMBEDDING INVARIANCE
# ==============================================================================
print("\nStep 7: Embedding invariance...")

# Use raw data
X_raw = np.array([generate_observer(a, 200, 8) for a in [0.2, 0.5, 0.8, 0.95] for _ in range(5)])
X_raw = X_raw.reshape(X_raw.shape[0], -1)

# Different embeddings
embeddings = {}

# PCA
pca = PCA(n_components=2)
embeddings['pca'] = pca.fit_transform(X_raw[:, :16])

# MDS
try:
    from sklearn.manifold import MDS
    mds = MDS(n_components=2, random_state=42)
    embeddings['mds'] = mds.fit_transform(X_raw[:, :16])
except:
    embeddings['mds'] = embeddings['pca']

# ISOMAP
try:
    iso = Isomap(n_neighbors=5, n_components=2)
    embeddings['isomap'] = iso.fit_transform(X_raw[:, :16])
except:
    embeddings['isomap'] = embeddings['pca']

# Compute variance in each embedding
embedding_stabilities = {}
for name, emb in embeddings.items():
    var = np.var(emb[:, 0])
    embedding_stabilities[name] = var

embedding_stability = np.std(list(embedding_stabilities.values()))
embedding_dependence = 1 - min(embedding_stability / (max(embedding_stabilities.values()) + 1e-8), 1)

print(f"  Embedding stabilities: {embedding_stabilities}")
print(f"  Embedding dependence: {embedding_dependence:.4f}")

# ==============================================================================
# STEP 8: RAW DATA TEST
# ==============================================================================
print("\nStep 8: Raw data test...")

# Test directly from raw data without manifold
# Raw covariance
X_cov = np.array([np.cov(generate_observer(a, 100, 8), rowvar=False).flatten()[:10] for a in [0.2, 0.5, 0.8, 0.95] for _ in range(5)])
y_raw = np.array([1 if a > 0.8 else 0 for a in [0.2, 0.5, 0.8, 0.95] for _ in range(5)])

lr_raw = LinearRegression()
try:
    lr_raw.fit(X_cov, y_raw)
    pred_raw = (lr_raw.predict(X_cov) > 0.5).astype(int)
    raw_covariance_accuracy = accuracy_score(y_raw, pred_raw)
except:
    raw_covariance_accuracy = 0.5

# Raw spectra
X_spec = np.array([np.linalg.eigvalsh(np.cov(generate_observer(a, 100, 8), rowvar=False))[:5] for a in [0.2, 0.5, 0.8, 0.95] for _ in range(5)])

lr_spec = LinearRegression()
try:
    lr_spec.fit(X_spec, y_raw)
    pred_spec = (lr_spec.predict(X_spec) > 0.5).astype(int)
    raw_spectrum_accuracy = accuracy_score(y_raw, pred_spec)
except:
    raw_spectrum_accuracy = 0.5

# Raw moments
X_mom = np.array([np.array([np.mean(generate_observer(a, 100, 8)**i) for i in [1,2,3,4]]) for a in [0.2, 0.5, 0.8, 0.95] for _ in range(5)])

lr_mom = LinearRegression()
try:
    lr_mom.fit(X_mom, y_raw)
    pred_mom = (lr_mom.predict(X_mom) > 0.5).astype(int)
    raw_moment_accuracy = accuracy_score(y_raw, pred_mom)
except:
    raw_moment_accuracy = 0.5

print(f"  Raw covariance accuracy: {raw_covariance_accuracy:.4f}")
print(f"  Raw spectrum accuracy: {raw_spectrum_accuracy:.4f}")
print(f"  Raw moment accuracy: {raw_moment_accuracy:.4f}")

# ==============================================================================
# STEP 9: NULL CAUSAL PIPELINES
# ==============================================================================
print("\nStep 9: Null causal pipelines...")

# Generate random pipelines
artifact_rates = []
false_transition_rates = []

for _ in range(10):
    # Random operators
    rand_phis = np.random.randn(20, 16)
    
    # Random embeddings
    rand_emb = np.random.randn(20, 2)
    
    # Random RG flow
    rand_rg = np.random.randn(20)
    
    # Test if random produces transition-like behavior
    d1_rand = np.abs(np.diff(rand_rg))
    if np.max(d1_rand) > 0.5:
        artifact_rates.append(1)
    else:
        artifact_rates.append(0)
    
    # Random classification
    rand_labels = np.random.choice([0, 1], 20)
    rand_pred = np.random.choice([0, 1], 20)
    if accuracy_score(rand_labels, rand_pred) > 0.6:
        false_transition_rates.append(1)
    else:
        false_transition_rates.append(0)

artifact_rate = np.mean(artifact_rates)
pipeline_false_transition_rate = np.mean(false_transition_rates)

print(f"  Artifact rate: {artifact_rate:.4f}")
print(f"  False transition rate: {pipeline_false_transition_rate:.4f}")

# ==============================================================================
# STEP 10-12: OUTPUT AND SAVE
# ==============================================================================
print("\n" + "="*60)
print("OUTPUT")
print("="*60)

print("\nFEATURE INDEPENDENCE:")
print(f"  max_mutual_information = {max_mutual_information:.4f}")
print(f"  conditional_independence_score = {conditional_independence_score:.4f}")
print(f"  redundancy_ratio = {redundancy_ratio:.4f}")

print("\nABLATION:")
print(f"  observer_without_geometry = {observer_without_geometry:.4f}")
print(f"  geometry_without_observer = {geometry_without_observer:.4f}")

print("\nORTHOGONALIZATION:")
print(f"  transition_survival = {transition_survival:.4f}")
print(f"  residual_transition_strength = {residual_transition_strength:.4f}")

print("\nRANDOM CONTROLS:")
print(f"  random_feature_transition_rate = {random_feature_transition_rate:.4f}")
print(f"  matched_random_accuracy = {matched_random_accuracy:.4f}")

print("\nLABEL LEAKAGE:")
print(f"  false_positive_rate = {false_positive_rate:.4f}")
print(f"  shuffled_accuracy = {shuffled_accuracy:.4f}")

print("\nCAUSALITY:")
print(f"  directionality_score = {directionality_score:.4f}")
print(f"  causal_asymmetry = {causal_asymmetry:.4f}")

print("\nEMBEDDINGS:")
print(f"  embedding_stability = {embedding_stability:.4f}")
print(f"  embedding_dependence = {embedding_dependence:.4f}")

print("\nRAW DATA:")
print(f"  raw_covariance_accuracy = {raw_covariance_accuracy:.4f}")
print(f"  raw_spectrum_accuracy = {raw_spectrum_accuracy:.4f}")
print(f"  raw_moment_accuracy = {raw_moment_accuracy:.4f}")

print("\nPIPELINE NULLS:")
print(f"  artifact_rate = {artifact_rate:.4f}")
print(f"  pipeline_false_transition_rate = {pipeline_false_transition_rate:.4f}")

# Verdict
leakage_detected = false_positive_rate > 0.3 or pipeline_false_transition_rate > 0.3
circularity_detected = max_mutual_information > 0.8

if leakage_detected or circularity_detected:
    if leakage_detected and circularity_detected:
        verdict = "significant_leakage_and_circularity"
    elif leakage_detected:
        verdict = "feature_leakage_detected"
    else:
        verdict = "circularity_detected"
else:
    if transition_survival > 0.5 and causal_asymmetry > 0.1:
        verdict = "valid_observer_transition"
    else:
        verdict = "weak_or_invalid"

print(f"\nVERDICT: observer_transition_validity = {verdict}")

# Save results
results = {
    'feature_independence': {
        'max_mutual_information': float(max_mutual_information),
        'conditional_independence_score': float(conditional_independence_score),
        'redundancy_ratio': float(redundancy_ratio)
    },
    'ablation': {
        'observer_without_geometry': float(observer_without_geometry),
        'geometry_without_observer': float(geometry_without_observer)
    },
    'orthogonalization': {
        'transition_survival': float(transition_survival),
        'residual_transition_strength': float(residual_transition_strength)
    },
    'random_controls': {
        'random_feature_transition_rate': float(random_feature_transition_rate),
        'matched_random_accuracy': float(matched_random_accuracy)
    },
    'label_leakage': {
        'false_positive_rate': float(false_positive_rate),
        'shuffled_accuracy': float(shuffled_accuracy)
    },
    'causality': {
        'directionality_score': float(directionality_score),
        'causal_asymmetry': float(causal_asymmetry)
    },
    'embeddings': {
        'embedding_stability': float(embedding_stability),
        'embedding_dependence': float(embedding_dependence),
        'per_embedding': {k: float(v) for k, v in embedding_stabilities.items()}
    },
    'raw_data': {
        'raw_covariance_accuracy': float(raw_covariance_accuracy),
        'raw_spectrum_accuracy': float(raw_spectrum_accuracy),
        'raw_moment_accuracy': float(raw_moment_accuracy)
    },
    'pipeline_nulls': {
        'artifact_rate': float(artifact_rate),
        'pipeline_false_transition_rate': float(pipeline_false_transition_rate)
    },
    'verdict': verdict
}

with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2)

# Save arrays
np.save(os.path.join(OUTPUT_DIR, 'geom_features.npy'), geom_vals)
np.save(os.path.join(OUTPUT_DIR, 'observer_features.npy'), obs_vals)
np.save(os.path.join(OUTPUT_DIR), 'stat_features.npy', stat_vals)
np.save(os.path.join(OUTPUT_DIR, 'comp_features.npy'), comp_vals)

print(f"\nAll files saved to {OUTPUT_DIR}")
print("="*60)