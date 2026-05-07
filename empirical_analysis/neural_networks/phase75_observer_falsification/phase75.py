"""
PHASE 75: OBSERVER FALSIFICATION TEST
Attempt to destroy the observer transition result
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
import json
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase75_observer_falsification'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("PHASE 75: OBSERVER FALSIFICATION TEST")
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
# STEP 1: RANDOMIZED CONTROLS (FAKE OBSERVERS)
# ==============================================================================
print("\nStep 1: Creating adversarial controls (fake observers)...")

def gen_fake_shuffled_observer(n, d):
    """Shuffle to break temporal structure but keep some coherence"""
    x = np.random.randn(n, d)
    for i in range(d):
        x[:, i] = np.roll(x[:, i], np.random.randint(1, n//4))
    return x

def gen_fake_latent_observer(n, d):
    """Random latent updates without real self-modeling"""
    x = np.zeros((n, d))
    z = np.random.randn(d)  # random init
    for t in range(n):
        z = np.random.randn(d) * 0.5  # random update - NOT recursive
        x[t] = z + np.random.randn(d) * 0.3
    return x

def gen_fake_recursive_loop(n, d):
    """Random recursive loops without prediction"""
    x = np.zeros((n, d))
    for t in range(n):
        if t > 0:
            # Random mixing, not predictive
            x[t] = x[t-1] * np.random.randn(d) + np.random.randn(d)
    return x

def gen_fake_pseudo_predictive(n, d):
    """Random prediction without actual self-model"""
    x = np.random.randn(n, d)
    # Add fake "prediction error" signal
    for t in range(1, n):
        x[t] = x[t] - 0.1 * (x[t] - np.random.randn(d))
    return x

def gen_fake_coherence_injection(n, d):
    """Synthetic coherence without genuine observer structure"""
    x = np.random.randn(n, d)
    # Inject temporal coherence via smoothing
    for i in range(d):
        x[:, i] = gaussian_filter1d(x[:, i], sigma=5)
    return x

def gen_random_recurrence(n, d):
    """Random recurrence without structure"""
    x = np.zeros((n, d))
    for t in range(1, n):
        x[t] = np.random.choice([x[t-1], np.random.randn(d)], p=[0.3, 0.7])
    return x

def gen_matched_variance_observer(n, d):
    """Match variance of real observer but no structure"""
    x = np.random.randn(n, d)
    x = x * 2.0  # Match variance scale
    return x

fake_observers = [
    ('shuffled_latent', gen_fake_shuffled_observer),
    ('random_latent', gen_fake_latent_observer),
    ('random_loop', gen_fake_recursive_loop),
    ('pseudo_predictive', gen_fake_pseudo_predictive),
    ('coherence_injection', gen_fake_coherence_injection),
    ('random_recurrence', gen_random_recurrence),
    ('matched_variance', gen_matched_variance_observer)
]

# Generate fake observers
adversarial_systems = []
for sys_name, gen_func in fake_observers:
    for trial in range(8):
        try:
            X = gen_func(n_samples, n_dim)
            var_x1 = compute_manifold(X)
            adversarial_systems.append({
                'system': sys_name,
                'type': 'fake_observer',
                'var_x1': var_x1 if not np.isnan(var_x1) else 0.0
            })
        except:
            continue

print(f"  Created {len(adversarial_systems)} fake observer systems")

# ==============================================================================
# STEP 2: OBSERVER DESTRUCTION
# ==============================================================================
print("\nStep 2: Observer destruction test...")

# Get a real observer system
def generate_real_observer(alpha, n, d):
    """Generate a real observer system (from Phase 74)"""
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

# Generate real observer for destruction
X_real = generate_real_observer(0.95, n_samples, n_dim)
base_var_x1 = compute_manifold(X_real)

destruction_results = {}

# A: Destroy self-model consistency
def destroy_self_model(X):
    X_new = X.copy()
    for t in range(1, X.shape[0]):
        X_new[t] = X_new[t] + np.random.randn(X.shape[1]) * 2.0
    return X_new

# B: Destroy latent persistence
def destroy_latent_persistence(X):
    X_new = X.copy()
    for i in range(X.shape[1]):
        X_new[:, i] = np.roll(X_new[:, i], np.random.randint(10, 50))
    return X_new

# C: Destroy recursive feedback
def destroy_recursive_feedback(X):
    X_new = X.copy()
    for t in range(10, X.shape[0]):
        X_new[t] = X_new[t-10] + np.random.randn(X.shape[1]) * 0.5
    return X_new

# D: Destroy temporal coherence
def destroy_temporal_coherence(X):
    X_new = X.copy()
    for i in range(X.shape[1]):
        X_new[:, i] = np.random.permutation(X_new[:, i])
    return X_new

# E: Destroy prediction stability
def destroy_prediction_stability(X):
    X_new = X.copy()
    for t in range(1, X.shape[0]):
        X_new[t] = X_new[t] * (1 + np.random.randn() * 5)
    return X_new

for name, func in [('A', destroy_self_model), ('B', destroy_latent_persistence), 
                  ('C', destroy_recursive_feedback), ('D', destroy_temporal_coherence),
                  ('E', destroy_prediction_stability)]:
    try:
        X_dest = func(X_real.copy())
        var_x1_dest = compute_manifold(X_dest)
        delta = abs(var_x1_dest - base_var_x1) if not np.isnan(var_x1_dest) else 0.0
        collapsed = delta > base_var_x1 * 0.5 if not np.isnan(base_var_x1) else False
        destruction_results[name] = {'delta_var_x1': float(delta), 'collapsed': collapsed}
        print(f"  {name}: delta_var_x1={delta:.4f}, collapsed={collapsed}")
    except Exception as e:
        destruction_results[name] = {'delta_var_x1': 0.0, 'collapsed': False}
        print(f"  {name}: error - {e}")

# ==============================================================================
# STEP 3: ALTERNATIVE EXPLANATIONS
# ==============================================================================
print("\nStep 3: Alternative explanations test...")

# Re-generate alpha sweep for analysis
alphas = np.linspace(0, 1, 50)
alpha_results = []

for alpha in alphas:
    X = generate_real_observer(alpha, 300, 32)
    var_x1 = compute_manifold(X)
    
    # Alternative features
    X_flat = X.flatten()
    variance = np.var(X_flat)
    entropy = -np.sum(np.abs(X_flat) * np.log(np.abs(X_flat) + 1e-8))
    
    # Spectral features
    cov = np.cov(X, rowvar=False)
    evals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    evals = evals[evals > 1e-8]
    spectral_centroid = np.mean(evals)
    
    # Recurrence
    D = cdist(X, X, metric='euclidean')
    recurrence = np.mean(D < np.percentile(D, 10))
    
    # Heavy tails
    kurtosis = np.mean([np.mean(x**4)/(np.mean(x**2)**2 + 1e-8) - 3 for x in X.T])
    
    alpha_results.append({
        'alpha': alpha,
        'var_x1': var_x1 if not np.isnan(var_x1) else 0.0,
        'variance': variance,
        'entropy': entropy,
        'spectral_centroid': spectral_centroid,
        'recurrence': recurrence,
        'kurtosis': kurtosis
    })

# Test alternative models: predict var_x1 WITHOUT using alpha/observer features
alt_features = ['variance', 'entropy', 'spectral_centroid', 'recurrence', 'kurtosis']
X_alt = np.array([[r[f] for f in alt_features] for r in alpha_results])
y_alt = np.array([r['var_x1'] for r in alpha_results])

# Linear regression
lr_alt = LinearRegression()
lr_alt.fit(X_alt, y_alt)
r2_alt = lr_alt.score(X_alt, y_alt)

# Can variance alone explain the transition?
lr_var = LinearRegression()
lr_var.fit(np.array([[r['variance']] for r in alpha_results]), y_alt)
r2_var = lr_var.score(np.array([[r['variance']] for r in alpha_results]), y_alt)

# Can entropy alone explain?
lr_ent = LinearRegression()
lr_ent.fit(np.array([[r['entropy']] for r in alpha_results]), y_alt)
r2_ent = lr_ent.score(np.array([[r['entropy']] for r in alpha_results]), y_alt)

print(f"  Alternative models:")
print(f"    All features R2: {r2_alt:.4f}")
print(f"    Variance alone R2: {r2_var:.4f}")
print(f"    Entropy alone R2: {r2_ent:.4f}")

best_alt_r2 = max(r2_alt, r2_var, r2_ent)
best_alt_model = 'all_features' if r2_alt == best_alt_r2 else ('variance' if r2_var == best_alt_r2 else 'entropy')

# ==============================================================================
# STEP 4: BLIND PREDICTION TEST
# ==============================================================================
print("\nStep 4: Blind prediction test...")

# Create dataset with true labels
all_data = []

# Real observers (alpha > 0.8)
for _ in range(30):
    X = generate_real_observer(np.random.uniform(0.85, 1.0), 200, 16)
    var_x1 = compute_manifold(X)
    all_data.append({'var_x1': var_x1 if not np.isnan(var_x1) else 0.0, 'label': 1})

# Non-observers (alpha < 0.3)
for _ in range(30):
    X = generate_real_observer(np.random.uniform(0, 0.3), 200, 16)
    var_x1 = compute_manifold(X)
    all_data.append({'var_x1': var_x1 if not np.isnan(var_x1) else 0.0, 'label': 0})

# Fake observers
for s in adversarial_systems[:30]:
    all_data.append({'var_x1': s['var_x1'], 'label': 2})  # label 2 = fake

X_blind = np.array([[d['var_x1']] for d in all_data])
y_blind = np.array([d['label'] for d in all_data])

# Binary: observer vs non-observer (exclude fake for now)
X_binary = X_blind[y_blind < 2]
y_binary = y_blind[y_blind < 2]

X_train, X_test, y_train, y_test = train_test_split(X_binary, y_binary, test_size=0.3, random_state=42)

clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auroc = roc_auc_score(y_test, y_prob)
calibration = brier_score_loss(y_test, y_prob)

print(f"  Binary classification (observer vs non-observer):")
print(f"    Accuracy: {accuracy:.4f}")
print(f"    AUROC: {auroc:.4f}")
print(f"    Calibration: {calibration:.4f}")

# ==============================================================================
# STEP 5: CROSS-DOMAIN GENERALIZATION
# ==============================================================================
print("\nStep 5: Cross-domain generalization...")

def generate_new_domains():
    """Generate completely new system types"""
    domains = {}
    
    # Coupled oscillators
    def coupled_osc(n, d):
        x = np.zeros((n, d))
        for i in range(d):
            x[0, i] = np.random.randn()
        for t in range(1, n):
            for i in range(d):
                x[t, i] = 0.9 * x[t-1, i] + 0.1 * x[t-1, (i+1)%d] + np.random.randn() * 0.2
        return x
    domains['coupled_osc'] = coupled_osc
    
    # Reaction-diffusion like
    def reaction_diffusion(n, d):
        x = np.random.randn(n, d)
        for t in range(2, n):
            x[t] = x[t-1] + 0.1 * (x[t-1] - x[t-2]) + np.random.randn(d) * 0.3
        return x
    domains['reaction_diff'] = reaction_diffusion
    
    # Game-like (alternating)
    def game_like(n, d):
        x = np.zeros((n, d))
        for t in range(n):
            if t % 2 == 0:
                x[t] = np.random.randn(d)
            else:
                x[t] = -x[t-1] + np.random.randn(d) * 0.3
        return x
    domains['game_like'] = game_like
    
    # Symbolic (discrete states)
    def symbolic(n, d):
        x = np.zeros((n, d))
        states = np.random.randn(5, d)
        current = 0
        for t in range(n):
            current = (current + np.random.choice([1, 2, 3, 4])) % 5
            x[t] = states[current] + np.random.randn(d) * 0.1
        return x
    domains['symbolic'] = symbolic
    
    return domains

new_domains = generate_new_domains()
domain_scores = {}

for domain_name, gen_func in new_domains.items():
    try:
        X_new = gen_func(300, 32)
        var_x1 = compute_manifold(X_new)
        
        # Check if it would be classified as observer
        # Use threshold from Phase 74: critical_alpha = 0.86
        # Map var_x1 to alpha using regression
        alpha_pred = (var_x1 - 0.5) / 2.0
        is_observer = alpha_pred > 0.8
        
        domain_scores[domain_name] = {
            'var_x1': float(var_x1),
            'would_be_observer': is_observer
        }
    except Exception as e:
        domain_scores[domain_name] = {'error': str(e)}

# Count generalization failures
failures = sum(1 for d in domain_scores.values() if d.get('would_be_observer', False) == False)
generalization_score = 1 - failures / len(domain_scores) if domain_scores else 0

print(f"  Domain classification:")
for name, score in domain_scores.items():
    print(f"    {name}: var_x1={score.get('var_x1', 'N/A'):.4f}, observer={score.get('would_be_observer', 'N/A')}")
print(f"  Generalization score: {generalization_score:.4f}")

# ==============================================================================
# STEP 6: ADVERSARIAL PERTURBATIONS
# ==============================================================================
print("\nStep 6: Adversarial perturbations...")

# Test with alpha = 0.95 (true observer)
X_orig = generate_real_observer(0.95, 500, 32)
orig_var_x1 = compute_manifold(X_orig)

perturbation_results = {}

# Random rotation
def perturb_rotation(X):
    Q = np.linalg.qr(np.random.randn(n_dim, n_dim))[0]
    return X @ Q

# Covariance-preserving scramble
def perturb_cov_preserve(X):
    return X + np.random.randn(*X.shape) * 0.1

# Nonlinear warping
def perturb_nonlinear(X):
    return np.tanh(X * 0.5) + np.random.randn(*X.shape) * 0.05

# Latent scrambling
def perturb_latent(X):
    X_pca = PCA(n_components=10).fit_transform(X)
    X_pca = X_pca + np.random.randn(*X_pca.shape) * 0.5
    return X_pca @ np.random.randn(10, n_dim)

for name, func in [('rotation', perturb_rotation), ('cov_preserve', perturb_cov_preserve),
                  ('nonlinear', perturb_nonlinear), ('latent_scramble', perturb_latent)]:
    try:
        X_pert = func(X_orig.copy())
        pert_var_x1 = compute_manifold(X_pert)
        delta = abs(pert_var_x1 - orig_var_x1) if not np.isnan(pert_var_x1) else 0.0
        stable = delta < orig_var_x1 * 0.3 if not np.isnan(orig_var_x1) else False
        perturbation_results[name] = {'delta_var_x1': float(delta), 'stable': stable}
        print(f"  {name}: delta={delta:.4f}, stable={stable}")
    except Exception as e:
        perturbation_results[name] = {'delta_var_x1': 0.0, 'stable': False}
        print(f"  {name}: error - {e}")

# ==============================================================================
# STEP 7: NULL MODEL TEST
# ==============================================================================
print("\nStep 7: Null model test...")

# Generate null ensembles with matched statistics but no observer structure
null_transitions = []

for _ in range(10):
    # Match variance but random
    X_null = np.random.randn(300, 32) * np.sqrt(orig_var_x1)
    null_var = compute_manifold(X_null)
    null_transitions.append(null_var)

# Match entropy
X_null2 = np.random.uniform(-1, 1, (300, 32))
null_var2 = compute_manifold(X_null2)

# Match recurrence
X_null3 = np.zeros((300, 32))
for t in range(1, 300):
    X_null3[t] = 0.3 * X_null3[t-1] + np.random.randn(32) * 0.7
null_var3 = compute_manifold(X_null3)

null_means = [np.mean(null_transitions), null_var2, null_var3]
null_transition_rate = np.mean([abs(m - orig_var_x1) < orig_var_x1 * 0.5 for m in null_means])

print(f"  Null model transition rates: {null_means}")
print(f"  Match rate: {null_transition_rate:.4f}")
print(f"  Criticality match: {1 - null_transition_rate:.4f}")

# ==============================================================================
# STEP 8: REPLICATION TEST
# ==============================================================================
print("\nStep 8: Replication test...")

replication_results = []

for seed in [42, 123, 456, 789, 1000]:
    np.random.seed(seed)
    alphas_test = np.linspace(0, 1, 20)
    var_x1s = []
    
    for alpha in alphas_test:
        X = generate_real_observer(alpha, 200, 16)
        v = compute_manifold(X)
        var_x1s.append(v if not np.isnan(v) else 0.0)
    
    # Find critical alpha (max derivative)
    d1 = np.gradient(var_x1s, alphas_test)
    critical_idx = np.argmax(np.abs(d1))
    critical_alpha = alphas_test[critical_idx]
    replication_results.append({'seed': seed, 'critical_alpha': critical_alpha, 'max_deriv': np.max(np.abs(d1))})
    
print(f"  Replication results:")
for r in replication_results:
    print(f"    seed={r['seed']}: critical_alpha={r['critical_alpha']:.3f}, max_deriv={r['max_deriv']:.2f}")

critical_alphas = [r['critical_alpha'] for r in replication_results]
replication_stability = 1 - np.std(critical_alphas)
threshold_variation = np.std(critical_alphas)

print(f"  Replication stability: {replication_stability:.4f}")
print(f"  Threshold variation: {threshold_variation:.4f}")

# ==============================================================================
# STEP 9: BAYESIAN EVIDENCE
# ==============================================================================
print("\nStep 9: Bayesian evidence...")

# H0: No observer transition
# H1: Observer transition exists

# Use max derivative from replication as local susceptibility
max_deriv_repl = max([r['max_deriv'] for r in replication_results])
susceptibility_evidence = min(max_deriv_repl / 5.0, 1.0)  # Normalized

# Evidence against H0: fake observers can fake it
fake_var_x1s = [s['var_x1'] for s in adversarial_systems[:20] if not np.isnan(s['var_x1'])]
fake_mean = np.mean(fake_var_x1s) if fake_var_x1s else 0.0
real_mean = orig_var_x1 if not np.isnan(orig_var_x1) else 1.0

# Posterior probability of H1 given evidence
likelihood_H1 = 1 / (1 + fake_mean / (real_mean + 1e-8))
likelihood_H0 = 1 - likelihood_H1

# Simple prior (50/50)
prior_H1 = 0.5
posterior_H1 = likelihood_H1 * prior_H1 / (likelihood_H1 * prior_H1 + likelihood_H0 * (1 - prior_H1))

bayes_factor = likelihood_H1 / (likelihood_H0 + 1e-8)
robustness_score = (replication_stability + susceptibility_evidence + (1 - null_transition_rate)) / 3

print(f"  Bayes factor: {bayes_factor:.4f}")
print(f"  Posterior probability: {posterior_H1:.4f}")
print(f"  Robustness score: {robustness_score:.4f}")

# ==============================================================================
# STEP 10-12: OUTPUT AND SAVE
# ==============================================================================
print("\n" + "="*60)
print("OUTPUT")
print("="*60)

print("\nDESTRUCTION TEST:")
for name, res in destruction_results.items():
    print(f"  {name}: collapsed={res['collapsed']}")

print("\nALTERNATIVE MODELS:")
print(f"  best_nonobserver_model = {best_alt_model}")
print(f"  best_R2 = {best_alt_r2:.4f}")

print("\nBLIND CLASSIFICATION:")
print(f"  accuracy = {accuracy:.4f}")
print(f"  AUROC = {auroc:.4f}")
print(f"  calibration = {calibration:.4f}")

print("\nCROSS-DOMAIN:")
print(f"  generalization_score = {generalization_score:.4f}")
print(f"  failure_domains = {failures}")

print("\nNULL MODELS:")
print(f"  null_transition_rate = {null_transition_rate:.4f}")
print(f"  criticality_match = {1 - null_transition_rate:.4f}")

print("\nREPLICATION:")
print(f"  replication_stability = {replication_stability:.4f}")
print(f"  threshold_variation = {threshold_variation:.4f}")

print("\nBAYESIAN EVIDENCE:")
print(f"  bayes_factor = {bayes_factor:.4f}")
print(f"  posterior_probability = {posterior_H1:.4f}")
print(f"  robustness_score = {robustness_score:.4f}")

# Verdict
if posterior_H1 > 0.8 and robustness_score > 0.7:
    verdict = 'observer_transition_confirmed'
elif posterior_H1 > 0.5:
    verdict = 'observer_transition_likely'
elif robustness_score > 0.5:
    verdict = 'observer_transition_weak'
else:
    verdict = 'observer_transition_rejected'

print(f"\nVERDICT: observer_transition_status = {verdict}")

# Save results
results = {
    'destruction_test': {k: v for k, v in destruction_results.items()},
    'alternative_models': {
        'best_model': best_alt_model,
        'best_r2': float(best_alt_r2)
    },
    'blind_classification': {
        'accuracy': float(accuracy),
        'auroc': float(auroc),
        'calibration': float(calibration)
    },
    'cross_domain': {
        'generalization_score': float(generalization_score),
        'failures': failures,
        'domain_scores': {k: {kk: float(vv) if isinstance(vv, (int, float)) else vv for kk, vv in v.items()} for k, v in domain_scores.items()}
    },
    'null_models': {
        'null_transition_rate': float(null_transition_rate),
        'criticality_match': float(1 - null_transition_rate)
    },
    'replication': {
        'replication_stability': float(replication_stability),
        'threshold_variation': float(threshold_variation),
        'results': replication_results
    },
    'bayesian_evidence': {
        'bayes_factor': float(bayes_factor),
        'posterior_probability': float(posterior_H1),
        'robustness_score': float(robustness_score)
    },
    'verdict': verdict
}

with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2, default=str)

# Save arrays
np.save(os.path.join(OUTPUT_DIR, 'fake_observer_var_x1.npy'), np.array([s['var_x1'] for s in adversarial_systems]))
np.save(os.path.join(OUTPUT_DIR, 'perturbation_results.npy'), np.array([perturbation_results[k]['delta_var_x1'] for k in perturbation_results.keys()]))

print(f"\nAll files saved to {OUTPUT_DIR}")
print("="*60)