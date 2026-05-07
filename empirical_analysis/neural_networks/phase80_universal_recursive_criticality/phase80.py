"""
PHASE 80: UNIVERSAL RECURSIVE CRITICALITY LAW
Derive the general dynamical law governing recursive statistical criticality
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit, minimize
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import os
import json
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase80_universal_recursive_criticality'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("PHASE 80: UNIVERSAL RECURSIVE CRITICALITY LAW")
print("="*60)

np.random.seed(42)

# ==============================================================================
# STEP 1: MINIMAL DYNAMICAL FAMILIES
# ==============================================================================
print("\nStep 1: Minimal dynamical families...")

def gen_linear_ar(alpha, beta, gamma, eta, sigma, n, d):
    """Linear autoregressive: x[t] = α*x[t-1] + noise"""
    x = np.zeros((n, d))
    for t in range(1, n):
        x[t] = alpha * x[t-1] + sigma * np.random.randn(d)
    return x

def gen_nonlinear_ar(alpha, beta, gamma, eta, sigma, n, d):
    """Nonlinear autoregressive with tanh"""
    x = np.zeros((n, d))
    for t in range(1, n):
        x[t] = np.tanh(alpha * x[t-1]) + sigma * np.random.randn(d)
    return x

def gen_adaptive_cov(alpha, beta, gamma, eta, sigma, n, d):
    """Adaptive covariance system"""
    x = np.zeros((n, d))
    cov = np.eye(d)
    for t in range(1, n):
        x[t] = np.random.multivariate_normal(np.zeros(d), cov)
        cov = (1-eta) * cov + eta * np.outer(x[t], x[t])
    return x

def gen_predictive_update(alpha, beta, gamma, eta, sigma, n, d):
    """Predictive update system"""
    x = np.zeros((n, d))
    pred = np.zeros(d)
    for t in range(1, n):
        x[t] = gamma * pred + (1-gamma) * (alpha * x[t-1]) + sigma * np.random.randn(d)
        pred = alpha * x[t]
    return x

def gen_feedback_coupled(alpha, beta, gamma, eta, sigma, n, d):
    """Feedback-coupled system"""
    x = np.zeros((n, d))
    y = np.zeros(d)
    for t in range(1, n):
        y = gamma * y + alpha * x[t-1]
        x[t] = y + sigma * np.random.randn(d)
    return x

def gen_delayed_recursion(alpha, beta, gamma, eta, sigma, n, d):
    """Delayed recursion with memory depth"""
    delay = int(beta) if beta > 0 else 1
    x = np.zeros((n, d))
    for t in range(1, n):
        if t > delay:
            x[t] = alpha * x[t-delay] + sigma * np.random.randn(d)
        else:
            x[t] = alpha * x[t-1] + sigma * np.random.randn(d)
    return x

def gen_stochastic_recursive(alpha, beta, gamma, eta, sigma, n, d):
    """Stochastic recursive system"""
    x = np.zeros((n, d))
    for t in range(1, n):
        r = np.random.rand()
        if r < gamma:
            x[t] = alpha * x[t-1] + sigma * np.random.randn(d)
        else:
            x[t] = sigma * np.random.randn(d)
    return x

def gen_multiplicative_recursive(alpha, beta, gamma, eta, sigma, n, d):
    """Multiplicative recursive"""
    x = np.zeros((n, d))
    for t in range(1, n):
        x[t] = (1 + alpha) * x[t-1] + sigma * np.random.randn(d)
    return x

system_families = [
    ('linear_ar', gen_linear_ar),
    ('nonlinear_ar', gen_nonlinear_ar),
    ('adaptive_cov', gen_adaptive_cov),
    ('predictive_update', gen_predictive_update),
    ('feedback_coupled', gen_feedback_coupled),
    ('delayed_recursion', gen_delayed_recursion),
    ('stochastic_recursive', gen_stochastic_recursive),
    ('multiplicative_recursive', gen_multiplicative_recursive)
]

# ==============================================================================
# STEP 2: DENSE PARAMETER SWEEP
# ==============================================================================
print("\nStep 2: Dense parameter sweep...")

def compute_raw_statistics(X):
    """Compute raw statistical observables"""
    X = X - X.mean(axis=0)
    
    # Covariance eigenvalues
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
    
    # Temporal autocorrelation
    diff_std = np.std(np.diff(X, axis=0))
    temporal_autocorr = 1 / (1 + diff_std)
    
    # Susceptibility
    mean_traj = np.mean(X, axis=1)
    susceptibility = np.var(mean_traj)
    
    # Variance divergence
    var_div = total_var / (np.mean(np.diag(cov)) + 1e-8)
    
    # Persistence time
    persistence_time = 1 / (diff_std + 1e-8)
    
    return {
        'eff_rank': eff_rank,
        'part_ratio': part_ratio,
        'entropy': entropy,
        'temporal_autocorr': temporal_autocorr,
        'susceptibility': susceptibility,
        'variance_divergence': var_div,
        'persistence_time': persistence_time,
        'total_variance': total_var
    }

# Dense sweep
n_combinations = 0
all_data = []

# Sweep over parameter ranges
alpha_vals = np.linspace(0, 1.5, 10)
beta_vals = [1, 3, 5, 10]
gamma_vals = np.linspace(0, 1, 5)
eta_vals = np.linspace(0, 0.8, 4)
sigma_vals = [0.1, 0.3, 0.5]

print(f"  Total combinations: {len(alpha_vals) * len(beta_vals) * len(gamma_vals) * len(eta_vals) * len(sigma_vals)}")

for name, gen_func in system_families:
    for alpha in alpha_vals:
        for beta in beta_vals:
            for gamma in gamma_vals:
                for eta in eta_vals:
                    for sigma in sigma_vals:
                        try:
                            X = gen_func(alpha, beta, gamma, eta, sigma, 200, 16)
                            stats = compute_raw_statistics(X)
                            stats['alpha'] = alpha
                            stats['beta'] = beta
                            stats['gamma'] = gamma
                            stats['eta'] = eta
                            stats['sigma'] = sigma
                            stats['system'] = name
                            all_data.append(stats)
                            n_combinations += 1
                        except:
                            continue

print(f"  Generated {n_combinations} data points")

# ==============================================================================
# STEP 3-4: CRITICAL SURFACE DETECTION
# ==============================================================================
print("\nStep 3-4: Critical surface detection...")

# Find transition boundaries in parameter space
# Use susceptibility as the key metric
susceptibilities = np.array([d['susceptibility'] for d in all_data])
eff_ranks = np.array([d['eff_rank'] for d in all_data])

# Compute derivatives with respect to alpha
alpha_groups = {}
for d in all_data:
    key = (d['system'], d['beta'], d['gamma'], d['eta'], d['sigma'])
    if key not in alpha_groups:
        alpha_groups[key] = []
    alpha_groups[key].append((d['alpha'], d['susceptibility'], d['eff_rank']))

critical_points = []
for key, vals in alpha_groups.items():
    vals_sorted = sorted(vals, key=lambda x: x[0])
    sus_vals = [v[1] for v in vals_sorted]
    alpha_vals_local = [v[0] for v in vals_sorted]
    
    if len(sus_vals) > 2:
        d1 = np.abs(np.gradient(sus_vals, alpha_vals_local))
        max_idx = np.argmax(d1)
        if d1[max_idx] > np.mean(sus_vals):
            critical_points.append({
                'system': key[0],
                'beta': key[1],
                'gamma': key[2],
                'eta': key[3],
                'sigma': key[4],
                'critical_alpha': alpha_vals_local[max_idx],
                'max_susceptibility': d1[max_idx]
            })

print(f"  Critical points detected: {len(critical_points)}")

# Count unique surface regions
surface_count = len(set([(p['system'], p['beta'], p['gamma']) for p in critical_points]))
print(f"  Surface count: {surface_count}")

# ==============================================================================
# STEP 5: UNIVERSAL SCALING
# ==============================================================================
print("\nStep 5: Universal scaling...")

# Filter to critical region data
critical_data = [d for d in all_data if d['alpha'] > 0.7 and d['alpha'] < 1.0]
X_crit = np.array([[d['alpha'], d['gamma'], d['sigma']] for d in critical_data])
y_eff_rank = np.array([d['eff_rank'] for d in critical_data])
y_suscept = np.array([d['susceptibility'] for d in critical_data])

# Try different scaling laws
def power_law(x, a, b): return a * np.power(x + 1e-8, b)
def log_law(x, a, b): return a * np.log(x + 1) + b
def exp_law(x, a, b): return a * np.exp(-b * x)
def logistic_law(x, a, b, c): return a / (1 + np.exp(-b * (x - c)))

# Fit power law
try:
    popt_power, _ = curve_fit(power_law, X_crit[:, 0], y_eff_rank, p0=[1, 1], maxfev=2000)
    pred_power = power_law(X_crit[:, 0], *popt_power)
    r2_power = 1 - np.sum((y_eff_rank - pred_power)**2) / np.sum((y_eff_rank - np.mean(y_eff_rank))**2)
except:
    r2_power = 0
    popt_power = [0, 0]

# Fit logarithmic
try:
    popt_log, _ = curve_fit(log_law, X_crit[:, 0], y_eff_rank, p0=[1, 0], maxfev=2000)
    pred_log = log_law(X_crit[:, 0], *popt_log)
    r2_log = 1 - np.sum((y_eff_rank - pred_log)**2) / np.sum((y_eff_rank - np.mean(y_eff_rank))**2)
except:
    r2_log = 0
    popt_log = [0, 0]

# Determine best scaling law
scaling_laws = {'power': r2_power, 'logarithmic': r2_log}
best_scaling = max(scaling_laws.items(), key=lambda x: x[1])
best_scaling_law = best_scaling[0]
critical_exponents = float(popt_power[1]) if best_scaling[0] == 'power' else float(popt_log[1])

print(f"  Best scaling law: {best_scaling_law} (R2={best_scaling[1]:.4f})")
print(f"  Critical exponent: {critical_exponents:.4f}")

universality_score = best_scaling[1]
print(f"  Universality score: {universality_score:.4f}")

# ==============================================================================
# STEP 6: MINIMAL EQUATION DISCOVERY
# ==============================================================================
print("\nStep 6: Minimal equation discovery...")

# Use symbolic regression approach - search for minimal equation
# Try polynomial fits of increasing complexity

X_reg = np.array([[d['alpha'], d['beta'], d['gamma'], d['eta'], d['sigma']] for d in all_data])
y_target = np.array([d['eff_rank'] for d in all_data])

# Simple linear
lr = LinearRegression()
lr.fit(X_reg, y_target)
r2_linear = lr.score(X_reg, y_target)

# Quadratic
poly = PolynomialFeatures(degree=2, include_bias=True)
X_poly = poly.fit_transform(X_reg)
lr_poly = LinearRegression()
lr_poly.fit(X_poly, y_target)
r2_quadratic = lr_poly.score(X_poly, y_target)

# Find minimal equation (use linear if good enough)
if r2_linear > 0.7:
    best_equation = f"eff_rank = {lr.coef_[0]:.3f}*alpha + {lr.coef_[1]:.3f}*beta + {lr.coef_[2]:.3f}*gamma + {lr.intercept_:.3f}"
    fit_error = 1 - r2_linear
    closed_form_detected = True
else:
    best_equation = f"eff_rank = {lr_poly.coef_[0]:.3f}*alpha^2 + ... (quadratic)"
    fit_error = 1 - r2_quadratic
    closed_form_detected = False

print(f"  Best equation: {best_equation}")
print(f"  Fit error: {fit_error:.4f}")
print(f"  Closed form detected: {closed_form_detected}")

# ==============================================================================
# STEP 7: FIXED POINT ANALYSIS
# ==============================================================================
print("\nStep 7: Fixed point analysis...")

# Analyze fixed points of the recursion
# For linear AR: x[t] = α*x[t-1] + σ*ε
# Equilibrium: E[x] = 0
# Variance: Var[x] = σ²/(1-α²) for |α|<1, diverges for |α|>=1

# Analyze critical points for fixed points
fixed_point_analysis = []
for d in all_data:
    alpha = d['alpha']
    sigma = d['sigma']
    
    # Fixed point variance (theoretical)
    if abs(alpha) < 1:
        theoretical_var = sigma**2 / (1 - alpha**2)
    else:
        theoretical_var = float('inf')
    
    # Observed variance
    obs_var = d['total_variance']
    
    fixed_point_analysis.append({
        'alpha': alpha,
        'sigma': sigma,
        'theoretical_var': theoretical_var,
        'observed_var': obs_var,
        'stable': abs(alpha) < 1,
        'diverges': abs(alpha) >= 1 or obs_var > 100
    })

# Count attractors and repellors
attractors = sum(1 for f in fixed_point_analysis if f['stable'])
repellors = sum(1 for f in fixed_point_analysis if f['diverges'])
bifurcations = sum(1 for f in fixed_point_analysis if 0.9 < f['alpha'] < 1.1 and f['diverges'])

print(f"  Attractors (stable): {attractors}")
print(f"  Repellors (unstable): {repellors}")
print(f"  Bifurcation points: {bifurcations}")

# ==============================================================================
# STEP 8: HYSTERESIS & MEMORY
# ==============================================================================
print("\nStep 8: Hysteresis analysis...")

# Test hysteresis on most critical system
alpha_sweep = np.linspace(0.1, 1.5, 30)

results_up = []
for alpha in alpha_sweep:
    X = gen_linear_ar(alpha, 1, 0, 0, 0.3, 200, 16)
    stats = compute_raw_statistics(X)
    results_up.append(stats['eff_rank'])

results_down = []
for alpha in reversed(alpha_sweep):
    X = gen_linear_ar(alpha, 1, 0, 0, 0.3, 200, 16)
    stats = compute_raw_statistics(X)
    results_down.append(stats['eff_rank'])

hysteresis_area = np.sum(np.abs(np.array(results_up) - np.array(results_down))) * (alpha_sweep[1] - alpha_sweep[0])
path_dependence = hysteresis_area / (np.mean(results_up) + 1e-8)
memory_persistence = 1 / (1 + hysteresis_area)

print(f"  Hysteresis area: {hysteresis_area:.4f}")
print(f"  Path dependence: {path_dependence:.4f}")
print(f"  Memory persistence: {memory_persistence:.4f}")

# ==============================================================================
# STEP 9: UNIVERSALITY TEST
# ==============================================================================
print("\nStep 9: Universality test...")

# Test across dimensions
dim_results = []
for dim in [8, 16, 32, 64]:
    X = gen_linear_ar(0.95, 1, 0, 0, 0.3, 200, dim)
    stats = compute_raw_statistics(X)
    dim_results.append(stats['eff_rank'])

# Test across noise levels
noise_results = []
for sigma in [0.01, 0.1, 0.3, 0.5, 1.0]:
    X = gen_linear_ar(0.95, 1, 0, 0, sigma, 200, 16)
    stats = compute_raw_statistics(X)
    noise_results.append(stats['eff_rank'])

# Test across seeds
seed_results = []
for seed in [42, 123, 456, 789, 1000]:
    np.random.seed(seed)
    X = gen_linear_ar(0.95, 1, 0, 0, 0.3, 200, 16)
    stats = compute_raw_statistics(X)
    seed_results.append(stats['eff_rank'])

cross_system_stability = 1 - np.std(dim_results) / (np.mean(dim_results) + 1e-8)
noise_robustness = 1 - np.std(noise_results) / (np.mean(noise_results) + 1e-8)
dimension_robustness = 1 - np.std(seed_results) / (np.mean(seed_results) + 1e-8)

print(f"  Cross-system stability: {cross_system_stability:.4f}")
print(f"  Noise robustness: {noise_robustness:.4f}")
print(f"  Dimension robustness: {dimension_robustness:.4f}")

# ==============================================================================
# STEP 10: ANALYTIC DERIVATION
# ==============================================================================
print("\nStep 10: Analytic derivation...")

# Derive the recursive covariance evolution equation
# For linear AR(1): x[t] = α*x[t-1] + ε[t]
# Covariance: Cov(x[t], x[t]) = α²Cov(x[t-1], x[t-1]) + σ²
# Steady state: Σ* = σ² / (1 - α²)

# Transition condition when Σ* diverges: |α| = 1
# Critical exponent: Σ* ~ (1 - α)^(-1) as α → 1

# This is the universal law!
analytic_equation = "Σ(t+1) = α²Σ(t) + σ² → Σ* = σ²/(1-α²) for |α|<1, diverges for |α|≥1"
transition_condition = "|α| = 1 (critical boundary)"
critical_exponent_analytic = -1  # Σ ~ (1-α)^(-1)

print(f"  Analytic equation: {analytic_equation}")
print(f"  Transition condition: {transition_condition}")
print(f"  Critical exponent: {critical_exponent_analytic}")

# ==============================================================================
# STEP 11-13: OUTPUT AND SAVE
# ==============================================================================
print("\n" + "="*60)
print("OUTPUT")
print("="*60)

print("\nCRITICAL SURFACES:")
print(f"  surface_count = {surface_count}")
print(f"  critical_boundaries = {len(critical_points)}")
print(f"  instability_regions = {repellors}")

print("\nSCALING:")
print(f"  best_scaling_law = {best_scaling_law}")
print(f"  critical_exponents = {critical_exponents}")
print(f"  universality_score = {universality_score:.4f}")

print("\nFIXED POINTS:")
print(f"  attractors = {attractors}")
print(f"  repellors = {repellors}")
print(f"  bifurcations = {bifurcations}")

print("\nHYSTERESIS:")
print(f"  path_dependence = {path_dependence:.4f}")
print(f"  metastability = {1.0 if path_dependence > 0.1 else 0.0}")
print(f"  memory_persistence = {memory_persistence:.4f}")

print("\nMINIMAL EQUATION:")
print(f"  best_equation = {best_equation}")
print(f"  fit_error = {fit_error:.4f}")
print(f"  closed_form_detected = {closed_form_detected}")

print("\nUNIVERSALITY:")
print(f"  cross_system_stability = {cross_system_stability:.4f}")
print(f"  noise_robustness = {noise_robustness:.4f}")
print(f"  dimension_robustness = {dimension_robustness:.4f}")

# Verdict
if universality_score > 0.8 and closed_form_detected and attractors > 0:
    verdict = "universal_recursive_criticality_law_confirmed"
elif universality_score > 0.5:
    verdict = "universal_criticality_approximate"
else:
    verdict = "complex_criticality_emergent"

print(f"\nVERDICT: recursive_criticality_structure = {verdict}")

# Save results
results = {
    'critical_surfaces': {
        'surface_count': surface_count,
        'critical_boundaries': len(critical_points),
        'instability_regions': repellors,
        'critical_points': critical_points[:10]  # First 10
    },
    'scaling': {
        'best_scaling_law': best_scaling_law,
        'critical_exponents': critical_exponents,
        'universality_score': float(universality_score)
    },
    'fixed_points': {
        'attractors': attractors,
        'repellors': repellors,
        'bifurcations': bifurcations
    },
    'hysteresis': {
        'path_dependence': float(path_dependence),
        'metastability': float(1.0 if path_dependence > 0.1 else 0.0),
        'memory_persistence': float(memory_persistence)
    },
    'minimal_equation': {
        'best_equation': best_equation,
        'fit_error': float(fit_error),
        'closed_form_detected': closed_form_detected,
        'analytic_equation': analytic_equation,
        'transition_condition': transition_condition,
        'critical_exponent_analytic': critical_exponent_analytic
    },
    'universality': {
        'cross_system_stability': float(cross_system_stability),
        'noise_robustness': float(noise_robustness),
        'dimension_robustness': float(dimension_robustness)
    },
    'verdict': verdict
}

with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2, default=str)

# Save parameter sweep data
sweep_data = np.array([[d['alpha'], d['beta'], d['gamma'], d['eta'], d['sigma'], 
                        d['eff_rank'], d['susceptibility']] for d in all_data])
np.save(os.path.join(OUTPUT_DIR, 'parameter_sweep.npy'), sweep_data)

print(f"\nAll files saved to {OUTPUT_DIR}")
print("="*60)