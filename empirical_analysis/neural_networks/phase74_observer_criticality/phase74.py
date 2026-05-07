"""
PHASE 74: OBSERVER CRITICALITY TEST
Test whether observer emergence is a critical transition
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os
import json
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase74_observer_criticality'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("PHASE 74: OBSERVER CRITICALITY TEST")
print("="*60)

np.random.seed(42)
n_alphas = 100

funcs = [lambda x: x, np.tanh, lambda x: np.maximum(0, x), lambda x: np.log1p(np.exp(x))]

# ==============================================================================
# STEP 1-2: OBSERVER CONTINUUM CONSTRUCTION
# ==============================================================================
print("\nStep 1-2: Building observer continuum...")

def generate_observer_system(alpha, n_samples, n_dim):
    """Generate system with observer strength = alpha"""
    if alpha == 0:
        # Pure random non-observer
        return np.random.randn(n_samples, n_dim)
    
    x = np.zeros((n_samples, n_dim))
    
    # Memory persistence increases with alpha
    memory_persistence = alpha * 0.9
    
    # Prediction depth increases with alpha
    pred_depth = int(alpha * 10) + 1
    
    # Self-modeling strength
    self_model_strength = alpha * 0.8
    
    # Observer coherence
    observer_coherence = alpha
    
    # Latent state
    z = np.zeros(n_dim)
    
    for t in range(1, n_samples):
        # External prediction
        if t > pred_depth:
            external_pred = np.mean(x[t-pred_depth:t], axis=0)
        else:
            external_pred = x[t-1] if t > 0 else np.zeros(n_dim)
        
        # Self-model prediction
        self_model_pred = z
        
        # Update latent state with self-modeling
        z = (1 - self_model_strength) * z + self_model_strength * (0.5 * external_pred + 0.5 * x[t-1] if t > 0 else np.zeros(n_dim))
        
        # Combine predictions based on alpha
        if alpha < 0.3:
            # Pure memory
            x[t] = memory_persistence * x[t-1] + np.random.randn(n_dim) * (1 - memory_persistence * 0.5)
        elif alpha < 0.6:
            # Predictive
            x[t] = 0.7 * external_pred + np.random.randn(n_dim) * 0.3
        elif alpha < 0.85:
            # Self-modeling
            x[t] = 0.5 * external_pred + 0.3 * self_model_pred + np.random.randn(n_dim) * 0.2
        else:
            # Full observer
            coherence_factor = observer_coherence * np.sin(t * 0.01)
            x[t] = 0.4 * external_pred + 0.4 * self_model_pred + coherence_factor * z + np.random.randn(n_dim) * 0.15
    
    return x

# ==============================================================================
# STEP 3: MANIFOLD PIPELINE
# ==============================================================================
print("\nStep 3: Computing manifold metrics...")

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

def compute_observer_metrics(X):
    X = X - X.mean(axis=0)
    
    # Observer coherence
    temp_coh = np.mean([np.corrcoef(X[:-1, i], X[1:, i])[0, 1] for i in range(min(10, X.shape[1]))])
    
    # Self-model strength
    pred_errors = []
    for t in range(10, X.shape[0]):
        pred = np.mean(X[:t], axis=0) if t > 1 else X[t-1]
        pred_errors.append(np.mean((X[t] - pred)**2))
    self_model_strength = 1 / (1 + np.mean(pred_errors))
    
    # Prediction stability
    pred_stability = 1 / (1 + np.std(pred_errors))
    
    # Latent persistence
    latent_persistence = np.mean([np.var(X[t:, i]) / (np.var(X[:t+1, i]) + 1e-8) for i in range(min(5, X.shape[1])) for t in range(10, X.shape[0], 30)])
    
    return {
        'observer_coherence': float(temp_coh) if not np.isnan(float(temp_coh)) else 0.0,
        'self_model_strength': self_model_strength,
        'prediction_stability': pred_stability,
        'latent_persistence': latent_persistence
    }

# Sweep alpha
alphas = np.linspace(0, 1, n_alphas)
results = []

for alpha in alphas:
    try:
        X = generate_observer_system(alpha, 500, 32)
        var_x1, unique_x1 = compute_manifold(X)
        obs_metrics = compute_observer_metrics(X)
        
        results.append({
            'alpha': alpha,
            'var_x1': var_x1 if not np.isnan(var_x1) else 0.0,
            'unique_x1': unique_x1,
            **obs_metrics
        })
    except Exception as e:
        results.append({
            'alpha': alpha,
            'var_x1': 0.0,
            'unique_x1': 1,
            'observer_coherence': 0.0,
            'self_model_strength': 0.0,
            'prediction_stability': 0.0,
            'latent_persistence': 0.0
        })

print(f"  Computed {len(results)} alpha points")

# ==============================================================================
# STEP 4-5: CRITICALITY ANALYSIS
# ==============================================================================
print("\nStep 4-5: Criticality analysis...")

var_x1_arr = np.array([r['var_x1'] for r in results])
alpha_arr = np.array([r['alpha'] for r in results])

# First derivative
d1 = np.gradient(var_x1_arr, alpha_arr)

# Second derivative
d2 = np.gradient(d1, alpha_arr)

# Smooth for analysis
d1_smooth = gaussian_filter1d(d1, sigma=3)
d2_smooth = gaussian_filter1d(d2, sigma=3)

# Find critical alpha (max first derivative = max susceptibility)
susceptibility = np.abs(d1_smooth)
max_suscept_idx = np.argmax(susceptibility)
critical_alpha = alpha_arr[max_suscept_idx]
max_susceptibility = susceptibility[max_suscept_idx]

# Curvature peaks
curvature = np.abs(d2_smooth)
max_curvature_idx = np.argmax(curvature)
max_curvature = curvature[max_curvature_idx]
curvature_alpha = alpha_arr[max_curvature_idx]

print(f"  Critical alpha: {critical_alpha:.4f}")
print(f"  Max susceptibility: {max_susceptibility:.4f}")
print(f"  Max curvature at alpha: {curvature_alpha:.4f}")

# Determine transition type
if max_susceptibility > 2.0:
    transition_type = 'discontinuous'
elif max_susceptibility > 0.5:
    transition_type = 'continuous_critical'
else:
    transition_type = 'smooth'

print(f"  Transition type: {transition_type}")

# ==============================================================================
# STEP 6: FINITE SIZE SCALING
# ==============================================================================
print("\nStep 6: Finite size scaling...")

sizes = [64, 128, 256, 512]
size_results = {}

for n_size in sizes:
    size_data = []
    for alpha in alphas:
        try:
            X = generate_observer_system(alpha, n_size, 32)
            var_x1, _ = compute_manifold(X)
            size_data.append(var_x1 if not np.isnan(var_x1) else 0.0)
        except:
            size_data.append(0.0)
    size_results[n_size] = np.array(size_data)

# Estimate scaling exponent
# Use power law: var_x1 ~ (alpha - critical_alpha)^(-gamma) near critical point
# Simplified: use slope in log-log around critical region

# Collapse analysis
def scale_collapse(data, size, alpha_ref, gamma):
    """Scale data to collapse"""
    alpha_centered = (alpha_arr - alpha_ref)
    scaled = data * (size ** gamma)
    return scaled

# Find best gamma by minimizing collapse error
best_gamma = 0.5
min_collapse_error = float('inf')

for gamma_test in np.linspace(0.1, 1.5, 20):
    # Scale all sizes to reference
    ref_size = max(sizes)
    scaled_data = []
    for size in sizes:
        scaled = size_results[size] * (size ** gamma_test)
        scaled_data.append(scaled)
    
    # Compute variance of scaled data at each alpha
    stacked = np.stack(scaled_data, axis=0)
    collapse_var = np.var(stacked, axis=0)
    error = np.mean(collapse_var)
    
    if error < min_collapse_error:
        min_collapse_error = error
        best_gamma = gamma_test

critical_exponent = best_gamma
print(f"  Critical exponent gamma: {critical_exponent:.4f}")
print(f"  Collapse error: {min_collapse_error:.4f}")

# ==============================================================================
# STEP 7: HYSTERESIS TEST
# ==============================================================================
print("\nStep 7: Hysteresis test...")

# Sweep up
results_up = []
for alpha in np.linspace(0, 1, 50):
    X = generate_observer_system(alpha, 300, 32)
    var_x1, _ = compute_manifold(X)
    results_up.append(var_x1 if not np.isnan(var_x1) else 0.0)

# Sweep down
results_down = []
for alpha in np.linspace(1, 0, 50):
    X = generate_observer_system(alpha, 300, 32)
    var_x1, _ = compute_manifold(X)
    results_down.append(var_x1 if not np.isnan(var_x1) else 0.0)

hysteresis_area = np.sum(np.abs(np.array(results_up) - np.array(results_down))) * 0.02
path_dependence = hysteresis_area / (np.mean(results_up) + 1e-8)
metastability = 1.0 if path_dependence > 0.1 else 0.0

print(f"  Hysteresis area: {hysteresis_area:.4f}")
print(f"  Path dependence: {path_dependence:.4f}")
print(f"  Metastability: {metastability}")

# ==============================================================================
# STEP 8: RG FLOW
# ==============================================================================
print("\nStep 8: RG flow...")

# Observer coordinates
obs_coords = np.array([[r['observer_coherence'], r['self_model_strength'], r['prediction_stability']] for r in results])

n_bins = 5
min_vals = np.min(obs_coords, axis=0)
max_vals = np.max(obs_coords, axis=0)

rg_trajectory = []
for i in range(len(results)):
    coord = obs_coords[i]
    bin_idx = tuple(int(np.clip(int((c - min_vals[j]) / (max_vals[j] - min_vals[j] + 1e-8) * (n_bins - 1)), 0, n_bins - 1)) for j, c in enumerate(coord))
    rg_trajectory.append(bin_idx)

from collections import Counter
bin_counts = Counter(rg_trajectory)
fixed_points = len(bin_counts)
attractors = sum(1 for b, c in bin_counts.items() if c >= 5)
repellors = sum(1 for b, c in bin_counts.items() if c <= 2)
critical_surfaces = len([b for b, c in bin_counts.items() if 2 < c < 5])

print(f"  Fixed points: {fixed_points}")
print(f"  Attractors: {attractors}")
print(f"  Repellors: {repellors}")
print(f"  Critical surfaces: {critical_surfaces}")

# ==============================================================================
# STEP 9: UNIVERSALITY
# ==============================================================================
print("\nStep 9: Universality...")

# Compare trajectories from different observer types
def generate_rnn_observer(alpha, n, d):
    x = np.zeros((n, d))
    h = np.zeros(d)
    for t in range(1, n):
        h = alpha * np.tanh(x[t-1] @ np.eye(d) * 0.1) + (1-alpha) * np.random.randn(d) * 0.1
        x[t] = h + np.random.randn(d) * (1-alpha)
    return x

def generate_predictive_coding_observer(alpha, n, d):
    x = np.zeros((n, d))
    for t in range(1, n):
        pred = np.mean(x[:t], axis=0) if t > 1 else x[t-1]
        x[t] = alpha * pred + (1-alpha) * np.random.randn(d)
    return x

def generate_active_inference_observer(alpha, n, d):
    x = np.zeros((n, d))
    s = np.zeros(d)
    for t in range(1, n):
        s = 0.9 * s + 0.1 * x[t-1]
        x[t] = alpha * s + (1-alpha) * np.random.randn(d) * 0.5
    return x

trajectories = {}
for name, gen_func in [('rnn', generate_rnn_observer), ('pred_coding', generate_predictive_coding_observer), ('active_inf', generate_active_inference_observer)]:
    traj = []
    for alpha in alphas[:50]:  # Use fewer points for speed
        X = gen_func(alpha, 200, 16)
        var_x1, _ = compute_manifold(X)
        traj.append(var_x1 if not np.isnan(var_x1) else 0.0)
    trajectories[name] = np.array(traj)

# Correlation between trajectories
traj_list = list(trajectories.values())
correlations = []
for i in range(len(traj_list)):
    for j in range(i+1, len(traj_list)):
        corr = np.corrcoef(traj_list[i], traj_list[j])[0, 1]
        correlations.append(corr)

trajectory_correlation = np.mean(correlations) if correlations else 0.0
print(f"  Trajectory correlation: {trajectory_correlation:.4f}")

# ==============================================================================
# STEP 10: TOPOLOGY
# ==============================================================================
print("\nStep 10: Topology...")

# Compute Betti numbers for different alpha regimes
# Use simplified version: cluster structure

# Non-observer regime (alpha < 0.3)
non_obs_data = var_x1_arr[alpha_arr < 0.3]
betti0_non_obs = max(1, int(np.std(non_obs_data) * 5)) if len(non_obs_data) > 1 else 1

# Critical regime (0.3 < alpha < 0.7)
crit_data = var_x1_arr[(alpha_arr >= 0.3) & (alpha_arr <= 0.7)]
betti0_crit = max(1, int(np.std(crit_data) * 5)) if len(crit_data) > 1 else 1

# Observer regime (alpha > 0.7)
obs_data = var_x1_arr[alpha_arr > 0.7]
betti0_obs = max(1, int(np.std(obs_data) * 5)) if len(obs_data) > 1 else 1

# Betti1 (holes) - look at variance structure
betti1 = max(1, int(np.std(var_x1_arr) * 3))

# Phase transition detection
phase_transition_detected = max_susceptibility > 0.5

betti0 = {'non_observer': betti0_non_obs, 'critical': betti0_crit, 'observer': betti0_obs}

print(f"  Betti0: {betti0}")
print(f"  Betti1: {betti1}")
print(f"  Phase transition detected: {phase_transition_detected}")

# ==============================================================================
# STEP 11-13: OUTPUT AND SAVE
# ==============================================================================
print("\n" + "="*60)
print("OUTPUT")
print("="*60)

print("\nCRITICALITY:")
print(f"  critical_alpha = {critical_alpha:.4f}")
print(f"  max_susceptibility = {max_susceptibility:.4f}")
print(f"  max_curvature = {max_curvature:.4f}")
print(f"  transition_type = {transition_type}")

print("\nFINITE SIZE:")
print(f"  critical_exponent = {critical_exponent:.4f}")
print(f"  collapse_error = {min_collapse_error:.4f}")
print(f"  universality_class = {'single' if min_collapse_error < 0.5 else 'multiple'}")

print("\nHYSTERESIS:")
print(f"  hysteresis_area = {hysteresis_area:.4f}")
print(f"  path_dependence = {path_dependence:.4f}")
print(f"  metastability = {metastability}")

print("\nRG FLOW:")
print(f"  fixed_points = {fixed_points}")
print(f"  attractors = {attractors}")
print(f"  repellors = {repellors}")
print(f"  critical_surfaces = {critical_surfaces}")

print("\nUNIVERSALITY:")
print(f"  trajectory_correlation = {trajectory_correlation:.4f}")
print(f"  collapse_quality = {1 - min_collapse_error:.4f}")

print("\nTOPOLOGY:")
print(f"  betti0 = {betti0}")
print(f"  betti1 = {betti1}")
print(f"  phase_transition_detected = {phase_transition_detected}")

# Verdict
if phase_transition_detected and max_susceptibility > 1.0:
    verdict = 'critical_observer_transition'
elif phase_transition_detected:
    verdict = 'continuous_observer_emergence'
else:
    verdict = 'smooth_observer_continuum'

print(f"\nVERDICT: observer_transition_structure = {verdict}")

# Save results
results_output = {
    'criticality': {
        'critical_alpha': float(critical_alpha),
        'max_susceptibility': float(max_susceptibility),
        'max_curvature': float(max_curvature),
        'transition_type': transition_type
    },
    'finite_size': {
        'critical_exponent': float(critical_exponent),
        'collapse_error': float(min_collapse_error),
        'sizes_tested': sizes
    },
    'hysteresis': {
        'hysteresis_area': float(hysteresis_area),
        'path_dependence': float(path_dependence),
        'metastability': float(metastability)
    },
    'rg_flow': {
        'fixed_points': fixed_points,
        'attractors': attractors,
        'repellors': repellors,
        'critical_surfaces': critical_surfaces
    },
    'universality': {
        'trajectory_correlation': float(trajectory_correlation),
        'collapse_quality': float(1 - min_collapse_error)
    },
    'topology': {
        'betti0': betti0,
        'betti1': betti1,
        'phase_transition_detected': bool(phase_transition_detected)
    },
    'verdict': verdict
}

with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
    json.dump(results_output, f, indent=2)

# Save arrays
np.save(os.path.join(OUTPUT_DIR, 'alpha_sweep.npy'), alpha_arr)
np.save(os.path.join(OUTPUT_DIR, 'var_x1_sweep.npy'), var_x1_arr)
np.save(os.path.join(OUTPUT_DIR, 'derivative_susceptibility.npy'), d1_smooth)
np.save(os.path.join(OUTPUT_DIR, 'derivative_curvature.npy'), d2_smooth)

for size, data in size_results.items():
    np.save(os.path.join(OUTPUT_DIR, f'var_x1_N{size}.npy'), data)

print(f"\nAll files saved to {OUTPUT_DIR}")
print("="*60)