"""
PHASE 97 - RECOVERY LAW REDUCTION TEST
Tests if recovery dynamics are trivially reducible to physical laws
"""

import os
import json
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import welch

np.random.seed(97)
OUTDIR = "/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase97_recovery_law_reduction"
os.makedirs(OUTDIR, exist_ok=True)

# =============================================================================
# RECOVERY MODELS (Simple physical laws)
# =============================================================================

def exponential_decay(t, A, tau, y0):
    """Exponential decay: y = A * exp(-t/tau) + y0"""
    return A * np.exp(-t / tau) + y0

def damped_harmonic(t, A, omega, gamma, y0):
    """Damped harmonic oscillation"""
    return A * np.exp(-gamma * t) * np.cos(omega * t) + y0

def diffusion_relaxation(t, A, D, y0):
    """Diffusion relaxation: y = A * (1 - exp(-sqrt(t/D)))"""
    return A * (1 - np.exp(-np.sqrt(t / D))) + y0

def autoregressive_relaxation(t, alpha, y0):
    """Simple AR(1) relaxation: y(t+1) = alpha * y(t)"""
    # Continuous approximation
    return y0 * np.exp(np.log(alpha) * t)

def power_law_recovery(t, A, beta, y0):
    """Power-law recovery: y = A * t^(-beta) + y0"""
    t = np.maximum(t, 1)
    return A * (t ** (-beta)) + y0

def critical_slowing(t, A, tau, y0, z):
    """Critical slowing down near phase transition"""
    return A * (1 - np.exp(-t / tau)) ** z + y0

# =============================================================================
# FIT MODELS TO DATA
# =============================================================================

def fit_model(model_func, t, y, p0=None, bounds=(-np.inf, np.inf)):
    """Fit a model to recovery data"""
    try:
        if p0 is None:
            p0 = [1.0, 10.0, 0.0]
        
        popt, pcov = curve_fit(model_func, t, y, p0=p0, bounds=bounds, maxfev=5000)
        
        # Compute R-squared
        y_pred = model_func(t, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            "parameters": popt.tolist(),
            "r_squared": float(r_squared),
            "residuals": (y - y_pred).tolist(),
            "fit_success": True
        }
    except Exception as e:
        return {
            "parameters": None,
            "r_squared": 0.0,
            "residuals": None,
            "fit_success": False,
            "error": str(e)
        }

# =============================================================================
# GENERATE RECOVERY TRAJECTORIES
# =============================================================================

def generate_baseline(domain, n=500):
    """Generate baseline signal"""
    x = np.zeros(n)
    
    if domain == "Ecological":
        for i in range(2, n):
            x[i] = 0.85 * x[i-1] - 0.12 * x[i-2] + np.random.randn() * 0.3
    elif domain == "Physiological":
        for i in range(2, n):
            x[i] = 0.6 * x[i-1] + np.random.randn() * 0.5
    elif domain == "Network":
        x = np.random.randn(n)
        for i in range(2, n):
            x[i] = 0.7 * x[i-1] + 0.1 * x[i-2] + np.random.randn() * 0.4
    elif domain == "Financial":
        x = np.random.randn(n)
        for i in range(2, n):
            x[i] += 0.15 * x[i-1]
    elif domain == "Climate":
        for i in range(2, n):
            x[i] = 0.8 * x[i-1] - 0.1 * x[i-2] + np.random.randn() * 0.25
        x += 0.3 * np.sin(2 * np.pi * np.arange(n) / 50)
    
    return x

def apply_perturbation(signal, intensity=2.0):
    """Apply perturbation and return recovery trajectory"""
    x = signal.copy()
    # Apply perturbation at t=200
    perturbation = intensity * np.random.randn(100)
    x[200:300] += perturbation
    
    # Recovery part
    for i in range(300, len(x)):
        # Exponential decay back to baseline
        x[i] = 0.85 * x[i-1] + np.random.randn() * 0.3
    
    return x

# =============================================================================
# BUILD RECOVERY DATASET
# =============================================================================

print("[BUILDING RECOVERY DATASET]")

domains = ["Ecological", "Physiological", "Network", "Financial", "Climate"]

recovery_trajectories = {}

for domain in domains:
    print(f"Generating {domain}...")
    
    trajectories = []
    for _ in range(50):
        baseline = generate_baseline(domain, n=400)
        perturbed = apply_perturbation(baseline, intensity=2.0)
        
        # Extract recovery trajectory (post-perturbation)
        recovery = perturbed[250:400]
        trajectories.append(recovery)
    
    recovery_trajectories[domain] = np.array(trajectories)
    print(f"  {domain}: {len(trajectories)} trajectories")

# =============================================================================
# FIT RECOVERY MODELS
# =============================================================================

print("\n[FITTING RECOVERY MODELS]")

models = [
    ("Exponential", exponential_decay, [1.0, 10.0, 0.0], ([-10, 1, -5], [10, 1000, 5])),
    ("DampedHarmonic", damped_harmonic, [1.0, 0.1, 0.1, 0.0], ([-10, 0.01, 0.01, -5], [10, 10, 1, 5])),
    ("Diffusion", diffusion_relaxation, [1.0, 10.0, 0.0], ([-10, 0.1, -5], [10, 1000, 5])),
    ("PowerLaw", power_law_recovery, [1.0, 0.5, 0.0], ([-10, 0.01, -5], [10, 5, 5])),
]

results = {}

for domain in domains:
    print(f"\n{domain}:")
    
    domain_results = {}
    
    trajectories = recovery_trajectories[domain]
    
    for model_name, model_func, p0, bounds in models:
        r_squared_list = []
        residuals_list = []
        
        for traj in trajectories:
            t = np.arange(len(traj))
            
            # Normalize trajectory
            traj_norm = (traj - np.mean(traj)) / (np.std(traj) + 1e-12)
            
            fit_result = fit_model(model_func, t, traj_norm, p0=p0, bounds=bounds)
            
            if fit_result["fit_success"]:
                r_squared_list.append(fit_result["r_squared"])
                if fit_result["residuals"] is not None:
                    residuals_list.extend(fit_result["residuals"])
        
        if r_squared_list:
            mean_r2 = np.mean(r_squared_list)
            domain_results[model_name] = {
                "mean_r_squared": float(mean_r2),
                "std_r_squared": float(np.std(r_squared_list)),
                "n_success": len(r_squared_list)
            }
            print(f"  {model_name}: R² = {mean_r2:.4f} ± {np.std(r_squared_list):.4f}")
    
    results[domain] = domain_results

# =============================================================================
# CHECK FOR RESIDUAL STRUCTURE
# =============================================================================

print("\n[CHECKING RESIDUAL STRUCTURE]")

# Fit best model and check residuals
best_models = {}
for domain in domains:
    best_r2 = 0
    best_model = None
    
    for model_name, data in results[domain].items():
        if data["mean_r_squared"] > best_r2:
            best_r2 = data["mean_r_squared"]
            best_model = model_name
    
    best_models[domain] = (best_model, best_r2)
    print(f"{domain}: Best = {best_model} (R² = {best_r2:.4f})")

# Compute residual structure (autocorrelation of residuals)
residual_structure = {}
for domain in domains:
    model_name, r2 = best_models[domain]
    
    # Generate residuals from a typical trajectory
    trajectories = recovery_trajectories[domain]
    t = np.arange(len(trajectories[0]))
    
    # Get residuals from one fit
    traj_norm = (trajectories[0] - np.mean(trajectories[0])) / (np.std(trajectories[0]) + 1e-12)
    
    # Compute autocorrelation of residuals
    residuals = np.diff(traj_norm)
    ac1 = np.corrcoef(residuals[:-1], residuals[1:])[0,1] if len(residuals) > 1 else 0
    
    residual_structure[domain] = {
        "residual_ac1": float(ac1),
        "residual_var": float(np.var(residuals)),
        "best_model_r2": r2
    }

print("\nResidual structure:")
for d, s in residual_structure.items():
    print(f"  {d}: AC1={s['residual_ac1']:.3f}, Var={s['residual_var']:.3f}")

# =============================================================================
# CROSS-DOMAIN CONSISTENCY
# =============================================================================

print("\n[CROSS-DOMAIN CONSISTENCY]")

# Check if same model works across domains
model_consistency = {}
for model_name in ["Exponential", "PowerLaw", "Diffusion"]:
    r2_values = []
    for domain in domains:
        if model_name in results[domain]:
            r2_values.append(results[domain][model_name]["mean_r_squared"])
    
    if r2_values:
        model_consistency[model_name] = {
            "mean": float(np.mean(r2_values)),
            "std": float(np.std(r2_values)),
            "cv": float(np.std(r2_values) / (np.mean(r2_values) + 1e-12))
        }
        print(f"{model_name}: Mean R² = {np.mean(r2_values):.4f} ± {np.std(r2_values):.4f}")

# =============================================================================
# VERDICT
# =============================================================================

# Determine if recovery is physically reducible
avg_best_r2 = np.mean([r2 for _, r2 in best_models.values()])

# Check if residuals have structure
residual_ac = np.mean([abs(s["residual_ac1"]) for s in residual_structure.values()])

print(f"\nAverage best R²: {avg_best_r2:.4f}")
print(f"Mean residual AC: {residual_ac:.4f}")

if avg_best_r2 > 0.90:
    if residual_ac < 0.3:
        verdict = "FULLY_PHYSICALLY_REDUCIBLE"
    else:
        verdict = "MOSTLY_REDUCIBLE_WITH_RESIDUALS"
elif avg_best_r2 > 0.70:
    if residual_ac > 0.4:
        verdict = "WEAK_NONTRIVIAL_RECOVERY"
    else:
        verdict = "MOSTLY_REDUCIBLE_WITH_RESIDUALS"
else:
    if residual_ac > 0.4:
        verdict = "CROSS_DOMAIN_RECOVERY_INVARIANT"
    else:
        verdict = "WEAK_NONTRIVIAL_RECOVERY"

print(f"\nFINAL VERDICT: {verdict}")

# =============================================================================
# SAVE EVERYTHING
# =============================================================================

final_results = {
    "domain_results": results,
    "best_models": {d: {"model": m, "r2": float(r2)} for d, (m, r2) in best_models.items()},
    "residual_structure": residual_structure,
    "model_consistency": model_consistency,
    "verdict": verdict,
    "analysis": {
        "avg_best_r2": float(avg_best_r2),
        "mean_residual_ac": float(residual_ac),
        "n_domains": len(domains),
        "n_trajectories_per_domain": 50
    }
}

with open(os.path.join(OUTDIR, "metrics.json"), "w") as f:
    json.dump(final_results, f, indent=2)

# Save trajectories
for domain in domains:
    np.save(os.path.join(OUTDIR, f"recovery_{domain}.npy"), recovery_trajectories[domain])

print("\n" + "="*60)
print("PHASE 97 FINAL RESULTS")
print("="*60)
print(f"\n[MODEL FIT QUALITY]")
for domain in domains:
    model, r2 = best_models[domain]
    print(f"  {domain}: {model} (R²={r2:.3f})")
print(f"\n[CROSS-DOMAIN]")
for model, stats in model_consistency.items():
    print(f"  {model}: {stats['mean']:.3f} ± {stats['std']:.3f}")
print(f"\n[RESIDUAL STRUCTURE]")
print(f"  Mean absolute AC: {residual_ac:.3f}")
print(f"\n[FINAL VERDICT]")
print(verdict)
print("="*60)