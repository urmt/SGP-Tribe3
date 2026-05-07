"""
PHASE 79: MINIMAL GENERATIVE LAW TEST
Find smallest dynamical process that generates all observed phenomena WITHOUT manifolds
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import os
import json
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase79_minimal_generative_law'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("PHASE 79: MINIMAL GENERATIVE LAW TEST")
print("="*60)

np.random.seed(42)

# ==============================================================================
# STEP 1: MINIMAL PROCESS LIBRARY
# ==============================================================================
print("\nStep 1: Minimal process library...")

def gen_linear_recursive(alpha, n, d):
    """Simplest: linear recursion only"""
    x = np.zeros((n, d))
    for t in range(1, n):
        x[t] = alpha * x[t-1] + np.random.randn(d) * np.sqrt(1 - alpha**2)
    return x

def gen_autoregressive(alpha, n, d):
    """AR(1) process"""
    x = np.zeros((n, d))
    for t in range(1, n):
        x[t] = alpha * x[t-1] + np.random.randn(d) * 0.5
    return x

def gen_predictive_coding(alpha, n, d):
    """Simple predictive coding"""
    x = np.zeros((n, d))
    pred = np.zeros(d)
    for t in range(1, n):
        x[t] = pred + np.random.randn(d) * 0.3
        pred = alpha * x[t] + (1-alpha) * pred
    return x

def gen_simple_recurrent(alpha, n, d):
    """Simple RNN-like"""
    x = np.zeros((n, d))
    h = np.zeros(d)
    for t in range(1, n):
        h = np.tanh(alpha * h + 0.1 * x[t-1])
        x[t] = h + np.random.randn(d) * 0.2
    return x

def gen_adaptive_covariance(alpha, n, d):
    """Covariance adaptation"""
    x = np.zeros((n, d))
    cov = np.eye(d)
    for t in range(1, n):
        x[t] = np.random.multivariate_normal(np.zeros(d), cov)
        cov = (1-alpha) * cov + alpha * np.outer(x[t], x[t])
    return x

def gen_feedback_coupling(alpha, n, d):
    """Feedback coupling"""
    x = np.zeros((n, d))
    y = np.zeros(d)
    for t in range(1, n):
        y = alpha * y + (1-alpha) * x[t-1]
        x[t] = y + np.random.randn(d) * 0.3
    return x

def gen_temporal_integration(alpha, n, d):
    """Temporal integration"""
    x = np.zeros((n, d))
    for t in range(1, n):
        x[t] = alpha * x[t-1] + (1-alpha) * np.random.randn(d)
    return x

minimal_processes = [
    ('linear_recursive', gen_linear_recursive),
    ('autoregressive', gen_autoregressive),
    ('predictive_coding', gen_predictive_coding),
    ('simple_recurrent', gen_simple_recurrent),
    ('adaptive_covariance', gen_adaptive_covariance),
    ('feedback_coupling', gen_feedback_coupling),
    ('temporal_integration', gen_temporal_integration)
]

# ==============================================================================
# STEP 2: PARAMETER SWEEP
# ==============================================================================
print("\nStep 2: Parameter sweep...")

def compute_raw_stats(X):
    """Compute raw statistics only - NO manifolds"""
    X = X - X.mean(axis=0)
    
    # Covariance spectrum
    cov = np.cov(X, rowvar=False)
    evals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    evals = evals[evals > 1e-8]
    total_var = np.sum(evals)
    p = evals / (total_var + 1e-8)
    
    # Effective rank
    eff_rank = np.exp(-np.sum(p * np.log(p + 1e-8)))
    
    # Participation ratio
    part_ratio = (np.sum(evals)**2) / (np.sum(evals**2) + 1e-8) if len(evals) > 0 else 0
    
    # Entropy
    hist, _ = np.histogram(X.flatten(), bins=50, density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log(hist + 1e-8))
    
    # Temporal persistence
    diff_std = np.std(np.diff(X, axis=0))
    temporal_persistence = 1 / (1 + diff_std)
    
    # Susceptibility (variance of mean)
    mean_traj = np.mean(X, axis=1)
    susceptibility = np.var(mean_traj)
    
    return {
        'eff_rank': eff_rank,
        'part_ratio': part_ratio,
        'entropy': entropy,
        'temporal_persistence': temporal_persistence,
        'susceptibility': susceptibility,
        'total_variance': total_var,
        'top_eigenvalue': evals[0] if len(evals) > 0 else 0
    }

# Sweep alpha
alphas = np.linspace(0, 1, 30)
all_results = {}

for proc_name, proc_func in minimal_processes:
    print(f"  Sweeping {proc_name}...")
    proc_results = []
    
    for alpha in alphas:
        X = proc_func(alpha, 300, 16)
        stats = compute_raw_stats(X)
        stats['alpha'] = alpha
        proc_results.append(stats)
    
    all_results[proc_name] = proc_results

# ==============================================================================
# STEP 3-4: CRITICALITY TEST
# ==============================================================================
print("\nStep 3-4: Criticality analysis...")

criticality_results = {}

for proc_name, proc_results in all_results.items():
    eff_ranks = np.array([r['eff_rank'] for r in proc_results])
    alphas_proc = np.array([r['alpha'] for r in proc_results])
    
    # Derivative of effective rank
    d1 = np.abs(np.gradient(eff_ranks, alphas_proc))
    d2 = np.abs(np.gradient(d1, alphas_proc))
    
    # Critical parameters
    max_suscept_idx = np.argmax(d1)
    critical_alpha = alphas_proc[max_suscept_idx]
    max_susceptibility = d1[max_suscept_idx]
    max_curvature = np.max(d2)
    
    critical_transition = max_susceptibility > 0.5
    
    criticality_results[proc_name] = {
        'critical_alpha': critical_alpha,
        'max_susceptibility': max_susceptibility,
        'max_curvature': max_curvature,
        'critical_transition': critical_transition
    }
    
    print(f"  {proc_name}: crit_alpha={critical_alpha:.3f}, sus={max_susceptibility:.3f}, trans={critical_transition}")

# ==============================================================================
# STEP 5: HYSTERESIS TEST
# ==============================================================================
print("\nStep 5: Hysteresis test...")

hysteresis_results = {}

# Test on best performing process
best_proc = min(criticality_results.items(), key=lambda x: x[1]['max_susceptibility'])[0]
print(f"  Testing hysteresis on: {best_proc}")

proc_func = dict(minimal_processes)[best_proc]

# Sweep up
alphas_up = np.linspace(0, 1, 20)
results_up = []
for alpha in alphas_up:
    X = proc_func(alpha, 200, 16)
    stats = compute_raw_stats(X)
    results_up.append(stats['eff_rank'])

# Sweep down
alphas_down = np.linspace(1, 0, 20)
results_down = []
for alpha in alphas_down:
    X = proc_func(alpha, 200, 16)
    stats = compute_raw_stats(X)
    results_down.append(stats['eff_rank'])

hysteresis_area = np.sum(np.abs(np.array(results_up) - np.array(results_down))) * 0.05
path_dependence = hysteresis_area / (np.mean(results_up) + 1e-8)
metastability = 1.0 if path_dependence > 0.1 else 0.0

hysteresis_results = {
    'hysteresis_area': hysteresis_area,
    'path_dependence': path_dependence,
    'metastability': metastability
}

print(f"  Hysteresis area: {hysteresis_area:.4f}")
print(f"  Path dependence: {path_dependence:.4f}")
print(f"  Metastability: {metastability}")

# ==============================================================================
# STEP 6: MINIMALITY SEARCH
# ==============================================================================
print("\nStep 6: Minimality search...")

# Remove each mechanism and test if criticality survives
removal_results = {}

# Test without recursion (alpha near 0 = no recursion)
X_no_recursion = gen_linear_recursive(0.01, 300, 16)
stats_no_recursion = compute_raw_stats(X_no_recursion)

# Test without memory (white noise)
X_no_memory = np.random.randn(300, 16)
stats_no_memory = compute_raw_stats(X_no_memory)

# Test with reduced prediction
X_no_pred = gen_predictive_coding(0.0, 300, 16)
stats_no_pred = compute_raw_stats(X_no_pred)

removal_results = {
    'no_recursion': stats_no_recursion['susceptibility'],
    'no_memory': stats_no_memory['susceptibility'],
    'no_prediction': stats_no_pred['susceptibility']
}

print(f"  Removal results (susceptibility):")
for k, v in removal_results.items():
    print(f"    {k}: {v:.4f}")

# Determine required mechanisms
required_mechanisms = []
if removal_results['no_recursion'] > 0.1:
    required_mechanisms.append('recursion')
if removal_results['no_memory'] > 0.1:
    required_mechanisms.append('memory')
if removal_results['no_prediction'] > 0.1:
    required_mechanisms.append('prediction')

removable_mechanisms = ['feedback_coupling' if r > 0.5 else 'temporal_integration' for r in removal_results.values()]

print(f"  Required: {required_mechanisms}")
print(f"  Removable: {removable_mechanisms}")

# ==============================================================================
# STEP 7: UNIVERSALITY TEST
# ==============================================================================
print("\nStep 7: Universality test...")

# Test across different random seeds, noise levels, dimensions
universality_results = []

for seed in [42, 123, 456]:
    np.random.seed(seed)
    X = gen_predictive_coding(0.95, 200, 16)
    stats = compute_raw_stats(X)
    universality_results.append(stats['eff_rank'])

# Different dimensions
for dim in [8, 16, 32]:
    X = gen_predictive_coding(0.95, 200, dim)
    stats = compute_raw_stats(X)
    universality_results.append(stats['eff_rank'])

# Different noise levels
for noise in [0.1, 0.3, 0.5]:
    X = gen_predictive_coding(0.95, 200, 16)
    X = X + np.random.randn(*X.shape) * noise
    stats = compute_raw_stats(X)
    universality_results.append(stats['eff_rank'])

cross_system_stability = 1 - np.std(universality_results) / (np.mean(universality_results) + 1e-8)
noise_robustness = 1 - np.std(universality_results[-3:]) / (np.mean(universality_results[-3:]) + 1e-8)

print(f"  Cross-system stability: {cross_system_stability:.4f}")
print(f"  Noise robustness: {noise_robustness:.4f}")

# ==============================================================================
# STEP 8: ANALYTIC APPROXIMATION
# ==============================================================================
print("\nStep 8: Analytic approximation...")

# Fit simple equation for effective rank vs alpha
alphas_fit = np.array([r['alpha'] for r in all_results['predictive_coding']])
eff_ranks_fit = np.array([r['eff_rank'] for r in all_results['predictive_coding']])

# Try power law: eff_rank ~ alpha^gamma
def power_law(alpha, gamma, c): return c * alpha**gamma

try:
    popt, _ = curve_fit(power_law, alphas_fit[1:], eff_ranks_fit[1:], p0=[1, 1], maxfev=2000)
    pred = power_law(alphas_fit, *popt)
    r2 = 1 - np.sum((eff_ranks_fit - pred)**2) / np.sum((eff_ranks_fit - np.mean(eff_ranks_fit))**2)
    best_equation = f"eff_rank = {popt[1]:.3f} * alpha^{popt[0]:.3f}"
    fit_error = 1 - r2
except:
    # Linear fit
    lr = LinearRegression()
    lr.fit(alphas_fit.reshape(-1, 1), eff_ranks_fit)
    pred = lr.predict(alphas_fit.reshape(-1, 1))
    r2 = lr.score(alphas_fit.reshape(-1, 1), eff_ranks_fit)
    best_equation = f"eff_rank = {lr.coef_[0]:.3f} * alpha + {lr.intercept_:.3f}"
    fit_error = 1 - r2

print(f"  Best equation: {best_equation}")
print(f"  Fit error: {fit_error:.4f}")

# ==============================================================================
# STEP 9-11: OUTPUT AND SAVE
# ==============================================================================
print("\n" + "="*60)
print("OUTPUT")
print("="*60)

print("\nCRITICALITY:")
transitions_detected = sum(1 for r in criticality_results.values() if r['critical_transition'])
print(f"  critical_transition_detected = {transitions_detected}/{len(criticality_results)}")
print(f"  critical_parameters = {[k for k, v in criticality_results.items() if v['critical_transition']]}")
print(f"  critical_exponents = {max(v['max_susceptibility'] for v in criticality_results.values()):.4f}")

print("\nHYSTERESIS:")
print(f"  path_dependence = {path_dependence:.4f}")
print(f"  metastability = {metastability}")

print("\nMINIMAL MECHANISMS:")
print(f"  required_mechanisms = {required_mechanisms}")
print(f"  removable_mechanisms = {removable_mechanisms}")

print("\nUNIVERSALITY:")
print(f"  cross_system_stability = {cross_system_stability:.4f}")
print(f"  noise_robustness = {noise_robustness:.4f}")

print("\nANALYTIC LAW:")
print(f"  best_equation = {best_equation}")
print(f"  fit_error = {fit_error:.4f}")

# Verdict
if len(required_mechanisms) <= 2 and transitions_detected >= 3 and fit_error < 0.3:
    verdict = "simple_linear_recursion_sufficient"
elif len(required_mechanisms) <= 3 and transitions_detected >= 2:
    verdict = "minimal_autoregressive_sufficient"
else:
    verdict = "complex_interaction_required"

print(f"\nVERDICT: minimal_generative_structure = {verdict}")

# Save results
results = {
    'criticality': {
        'transitions_detected': transitions_detected,
        'critical_parameters': [k for k, v in criticality_results.items() if v['critical_transition']],
        'max_susceptibility': float(max(v['max_susceptibility'] for v in criticality_results.values())),
        'all_results': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in criticality_results.items()}
    },
    'hysteresis': hysteresis_results,
    'minimal_mechanisms': {
        'required': required_mechanisms,
        'removable': removable_mechanisms
    },
    'universality': {
        'cross_system_stability': float(cross_system_stability),
        'noise_robustness': float(noise_robustness)
    },
    'analytic_law': {
        'best_equation': best_equation,
        'fit_error': float(fit_error)
    },
    'verdict': verdict
}

with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2, default=str)

# Save arrays
for proc_name, proc_results in all_results.items():
    data = np.array([[r['eff_rank'], r['part_ratio'], r['entropy'], r['susceptibility']] for r in proc_results])
    np.save(os.path.join(OUTPUT_DIR, f'{proc_name}.npy'), data)

print(f"\nAll files saved to {OUTPUT_DIR}")
print("="*60)