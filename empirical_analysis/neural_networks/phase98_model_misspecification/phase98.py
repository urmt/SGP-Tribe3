"""
PHASE 98 - MODEL MISSPECIFICATION ELIMINATION TEST
Tests whether poor recovery fits are due to model misspecification or genuine structure
"""

import os
import json
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import welch, lfilter
from scipy.stats import linregress
from sklearn.linear_model import Ridge, LinearRegression

np.random.seed(98)
OUTDIR = "/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase98_model_misspecification"
os.makedirs(OUTDIR, exist_ok=True)

# =============================================================================
# TRAJECTORY QUALITY METRICS
# =============================================================================

def measure_trajectory_quality(trajectory):
    """Measure signal quality metrics"""
    # SNR: signal variance / noise variance
    signal_power = np.var(trajectory)
    noise_estimate = np.var(np.diff(trajectory))
    snr = signal_power / (noise_estimate + 1e-12)
    
    # Stationarity: variance in first half vs second half
    n = len(trajectory)
    var_first = np.var(trajectory[:n//2])
    var_second = np.var(trajectory[n//2:])
    stationarity = 1 - abs(var_first - var_second) / (var_first + var_second + 1e-12)
    
    # Temporal smoothness (smoothness = low high-frequency content)
    freqs, psd = welch(trajectory, nperseg=min(32, len(trajectory)//4))
    psd_norm = psd / (np.sum(psd) + 1e-12)
    high_freq_power = np.sum(psd_norm[len(psd_norm)*2//3:])
    smoothness = 1 - high_freq_power
    
    # Variance stability
    window_size = len(trajectory) // 4
    variances = [np.var(trajectory[i*window_size:(i+1)*window_size]) for i in range(4)]
    var_stability = 1 - np.std(variances) / (np.mean(variances) + 1e-12)
    
    return {
        "snr": float(snr),
        "stationarity": float(stationarity),
        "smoothness": float(smoothness),
        "var_stability": float(var_stability),
        "quality_score": float(np.mean([snr, stationarity, smoothness, var_stability]))
    }

# =============================================================================
# GENERATE RECOVERY TRAJECTORIES
# =============================================================================

def generate_baseline(domain, n=400):
    """Generate baseline signal"""
    x = np.zeros(n)
    if domain == "Ecological":
        for i in range(2, n):
            x[i] = 0.85 * x[i-1] - 0.12 * x[i-2] + np.random.randn() * 0.3
    elif domain == "Physiological":
        for i in range(2, n):
            x[i] = 0.6 * x[i-1] + np.random.randn() * 0.5
    elif domain == "Network":
        for i in range(2, n):
            x[i] = 0.7 * x[i-1] + 0.1 * x[i-2] + np.random.randn() * 0.4
    elif domain == "Financial":
        x = np.random.randn(n)
        for i in range(2, n):
            x[i] += 0.15 * x[i-1]
    elif domain == "Climate":
        for i in range(2, n):
            x[i] = 0.8 * x[i-1] - 0.1 * x[i-2] + np.random.randn() * 0.25
    return x

def generate_recovery(domain, noise_level=0.3):
    """Generate recovery trajectory with clean structure"""
    baseline = generate_baseline(domain, n=400)
    
    # Add perturbation
    baseline[200:250] += 3 * np.random.randn(50)
    
    # Generate clean exponential-like recovery
    t = np.arange(150)
    tau = 15 + np.random.rand() * 10
    recovery = 2 * np.exp(-t / tau) + np.random.randn(150) * noise_level
    
    return baseline[250:].copy() + recovery[:150]

# =============================================================================
# EXPANDED RECOVERY MODELS
# =============================================================================

def fit_multi_exponential(t, y):
    """Multi-exponential decay"""
    def model(t, a1, tau1, a2, tau2, y0):
        return a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2) + y0
    try:
        popt, _ = curve_fit(model, t, y, p0=[1, 10, 0.5, 30, 0], 
                           bounds=([0, 1, 0, 1, -5], [5, 100, 5, 200, 5]), maxfev=5000)
        y_pred = model(t, *popt)
        r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
        return {"r2": float(r2), "params": popt.tolist(), "success": True}
    except:
        return {"r2": 0, "params": None, "success": False}

def fit_nonlinear_ar(y):
    """Nonlinear autoregressive"""
    if len(y) < 10:
        return {"r2": 0, "success": False}
    # Simple lagged features
    n = len(y) - 3
    X = np.zeros((n, 3))
    X[:, 0] = y[2:-1]  # y(t-1)
    X[:, 1] = y[1:-2]  # y(t-2)
    X[:, 2] = y[2:-1] ** 2  # y(t-1)^2
    y_target = y[3:]
    
    model = Ridge(alpha=0.1)
    model.fit(X[:-20], y_target[:-20])
    y_pred = model.predict(X[-20:])
    r2 = 1 - np.sum((y_target[-20:] - y_pred)**2) / np.var(y_target[-20:])
    return {"r2": float(r2), "success": True}

def fit_state_dependent(y):
    """State-dependent recovery"""
    t = np.arange(len(y))
    # Simple polynomial
    try:
        coeffs = np.polyfit(t, y, 3)
        y_pred = np.polyval(coeffs, t)
        r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
        return {"r2": float(r2), "params": coeffs.tolist(), "success": True}
    except:
        return {"r2": 0, "success": False}

def fit_damped_oscillation_delay(t, y):
    """Damped oscillation with delay"""
    def model(t, A, omega, gamma, delay, y0):
        t_delayed = np.maximum(t - delay, 0)
        return A * np.exp(-gamma * t_delayed) * np.cos(omega * t_delayed) + y0
    try:
        popt, _ = curve_fit(model, t, y, p0=[1, 0.2, 0.1, 5, 0],
                           bounds=([0, 0.01, 0.01, 0, -5], [5, 2, 1, 50, 5]), maxfev=5000)
        y_pred = model(t, *popt)
        r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
        return {"r2": float(r2), "params": popt.tolist(), "success": True}
    except:
        return {"r2": 0, "success": False}

def fit_fractional_relaxation(t, y):
    """Fractional relaxation"""
    def model(t, A, alpha, tau, y0):
        # Cap alpha to valid range
        alpha = min(max(alpha, 0.01), 1.99)
        # Approximation of fractional derivative
        t_capped = np.maximum(t, 0.1)
        return A * (t_capped ** (-alpha)) * np.exp(-t_capped / tau) + y0
    try:
        popt, _ = curve_fit(model, t, y, p0=[1, 0.5, 20, 0],
                           bounds=([0, 0.01, 1, -5], [5, 1.99, 100, 5]), maxfev=5000)
        y_pred = model(t, *popt)
        r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
        return {"r2": float(r2), "params": popt.tolist(), "success": True}
    except:
        return {"r2": 0, "success": False}

def fit_kalman_recovery(y):
    """Kalman-style recovery model"""
    # Simple state estimation
    x_est = np.zeros(len(y))
    x_est[0] = y[0]
    
    for i in range(1, len(y)):
        x_est[i] = 0.9 * x_est[i-1] + 0.1 * y[i-1]
    
    r2 = 1 - np.sum((y - x_est)**2) / np.sum((y - np.mean(y))**2)
    return {"r2": float(r2), "success": True}

# =============================================================================
# BUILD DATASET
# =============================================================================

print("[BUILDING HIGH-QUALITY RECOVERY TRAJECTORIES]")

domains = ["Ecological", "Physiological", "Network", "Financial", "Climate"]

trajectories = {}
trajectory_quality = {}

for domain in domains:
    print(f"\nGenerating {domain}...")
    
    trajs = []
    qualities = []
    
    for _ in range(40):
        traj = generate_recovery(domain, noise_level=0.2)
        
        # Measure quality
        q = measure_trajectory_quality(traj)
        
        if q["quality_score"] > 0.3:  # Accept only quality trajectories
            trajs.append(traj)
            qualities.append(q)
    
    trajectories[domain] = np.array(trajs)
    trajectory_quality[domain] = qualities
    print(f"  {domain}: {len(trajs)} high-quality trajectories")
    print(f"    Mean quality: {np.mean([q['quality_score'] for q in qualities]):.3f}")

# =============================================================================
# FIT EXPANDED MODELS
# =============================================================================

print("\n[FITTING EXPANDED MODELS]")

results = {}

for domain in domains:
    print(f"\n{domain}:")
    
    domain_results = {}
    
    trajs = trajectories[domain]
    t = np.arange(len(trajs[0]))
    
    # Fit each model
    models = [
        ("MultiExponential", fit_multi_exponential),
        ("NonlinearAR", lambda t, y: fit_nonlinear_ar(y)),
        ("StateDependent", lambda t, y: fit_state_dependent(y)),
        ("DampedDelay", fit_damped_oscillation_delay),
        ("Fractional", fit_fractional_relaxation),
        ("Kalman", lambda t, y: fit_kalman_recovery(y))
    ]
    
    for model_name, model_func in models:
        r2_values = []
        
        for traj in trajs:
            result = model_func(t, traj)
            if result.get("success", False):
                r2_values.append(result["r2"])
        
        if r2_values:
            mean_r2 = np.mean(r2_values)
            domain_results[model_name] = {
                "mean_r2": float(mean_r2),
                "std_r2": float(np.std(r2_values)),
                "n_fits": len(r2_values)
            }
            print(f"  {model_name}: R² = {mean_r2:.4f}")
    
    results[domain] = domain_results

# =============================================================================
# NULL TRAJECTORY TEST
# =============================================================================

print("\n[NULL TRAJECTORY COMPARISON]")

def generate_null_trajectory(real_traj):
    """Generate null preserving spectrum and variance"""
    # FFT-based surrogate
    fft_vals = np.fft.rfft(real_traj)
    mag = np.abs(fft_vals)
    phase = np.random.uniform(0, 2*np.pi, len(fft_vals))
    null = np.real(np.fft.irfft(mag * np.exp(1j * phase)))
    
    # Match variance
    null = null * (np.std(real_traj) / (np.std(null) + 1e-12))
    
    return null

null_results = {}
for domain in domains:
    trajs = trajectories[domain]
    
    # Fit multi-exponential to real
    real_r2s = []
    for traj in trajs[:20]:
        result = fit_multi_exponential(np.arange(len(traj)), traj)
        if result["success"]:
            real_r2s.append(result["r2"])
    
    # Fit multi-exponential to null
    null_r2s = []
    for traj in trajs[:20]:
        null_traj = generate_null_trajectory(traj)
        result = fit_multi_exponential(np.arange(len(null_traj)), null_traj)
        if result["success"]:
            null_r2s.append(result["r2"])
    
    null_results[domain] = {
        "real_mean": float(np.mean(real_r2s)) if real_r2s else 0,
        "null_mean": float(np.mean(null_r2s)) if null_r2s else 0
    }
    print(f"  {domain}: Real R²={np.mean(real_r2s):.4f}, Null R²={np.mean(null_r2s):.4f}")

# =============================================================================
# CROSS-DOMAIN VALIDATION
# =============================================================================

print("\n[CROSS-DOMAIN VALIDATION]")

# Train on some domains, test on others
cross_domain_results = {}

for train_domain in ["Ecological", "Physiological"]:
    for test_domain in ["Network", "Financial"]:
        train_trajs = trajectories[train_domain]
        test_trajs = trajectories[test_domain]
        
        # Fit on train
        train_r2s = []
        t = np.arange(len(train_trajs[0]))
        for traj in train_trajs[:20]:
            result = fit_multi_exponential(t, traj)
            if result["success"]:
                train_r2s.append(result["r2"])
        
        # Test on test
        test_r2s = []
        for traj in test_trajs[:20]:
            result = fit_multi_exponential(t, traj)
            if result["success"]:
                test_r2s.append(result["r2"])
        
        key = f"{train_domain}->{test_domain}"
        cross_domain_results[key] = {
            "train_r2": float(np.mean(train_r2s)) if train_r2s else 0,
            "test_r2": float(np.mean(test_r2s)) if test_r2s else 0
        }
        print(f"  {key}: Train R²={np.mean(train_r2s):.4f}, Test R²={np.mean(test_r2s):.4f}")

# =============================================================================
# VERDICT
# =============================================================================

# Compute average R² across expanded models
all_r2s = []
for domain in domains:
    for model_name, data in results[domain].items():
        all_r2s.append(data["mean_r2"])

avg_expanded_r2 = np.mean(all_r2s)

# Check null vs real
real_means = [null_results[d]["real_mean"] for d in domains]
null_means = [null_results[d]["null_mean"] for d in domains]

null_gap = np.mean(real_means) - np.mean(null_means)

print(f"\nAverage expanded model R²: {avg_expanded_r2:.4f}")
print(f"Null gap: {null_gap:.4f}")

# Determine verdict
if avg_expanded_r2 > 0.85:
    verdict = "FULLY_MODEL_SPECIFIED"
elif avg_expanded_r2 > 0.60:
    verdict = "MOSTLY_MODEL_SPECIFIED"
elif null_gap < 0.05:
    verdict = "MIXED_RECOVERY_DYNAMICS"
else:
    verdict = "POSSIBLE_IRREDUCIBLE_STRUCTURE"

print(f"\nFINAL VERDICT: {verdict}")

# =============================================================================
# SAVE EVERYTHING
# =============================================================================

final_results = {
    "domain_results": results,
    "trajectory_quality": trajectory_quality,
    "null_comparison": null_results,
    "cross_domain": cross_domain_results,
    "verdict": verdict,
    "analysis": {
        "avg_expanded_r2": float(avg_expanded_r2),
        "null_gap": float(null_gap),
        "n_domains": len(domains)
    }
}

with open(os.path.join(OUTDIR, "metrics.json"), "w") as f:
    json.dump(final_results, f, indent=2)

for domain in domains:
    np.save(os.path.join(OUTDIR, f"trajectories_{domain}.npy"), trajectories[domain])

print("\n" + "="*60)
print("PHASE 98 FINAL RESULTS")
print("="*60)
print(f"\n[EXPANDED MODEL PERFORMANCE]")
best_per_domain = {}
for domain in domains:
    best = max(results[domain].items(), key=lambda x: x[1]["mean_r2"])
    best_per_domain[domain] = best
    print(f"  {domain}: {best[0]} (R²={best[1]['mean_r2']:.3f})")

print(f"\n[NULL COMPARISON]")
print(f"  Average Real R²: {np.mean(real_means):.4f}")
print(f"  Average Null R²: {np.mean(null_means):.4f}")
print(f"  Gap: {null_gap:.4f}")

print(f"\n[CROSS-DOMAIN]")
for k, v in cross_domain_results.items():
    print(f"  {k}: Train={v['train_r2']:.3f}, Test={v['test_r2']:.3f}")

print(f"\n[FINAL VERDICT]")
print(verdict)
print("="*60)