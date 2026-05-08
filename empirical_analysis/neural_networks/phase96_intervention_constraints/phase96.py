"""
PHASE 96 - INTERVENTION-BOUND ORGANIZATIONAL CONSTRAINT TEST
Tests whether any organizational invariants survive REAL causal intervention
"""

import os
import json
import numpy as np
from scipy.signal import welch
from scipy.stats import entropy as scipy_entropy
from scipy.fft import fft, ifft

np.random.seed(96)
OUTDIR = "/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase96_intervention_constraints"
os.makedirs(OUTDIR, exist_ok=True)

# =============================================================================
# REALISTIC SIGNALS (Documented proxies for intervention studies)
# =============================================================================

def generate_signal(domain, n=500):
    """Generate baseline signals"""
    x = np.zeros(n)
    
    if domain == "EEG":
        for i in range(2, n):
            x[i] = 0.7*x[i-1] - 0.2*x[i-2] + np.random.randn()*0.4
        x += 0.5*np.sin(2*np.pi*10*np.arange(n)/100)
    elif domain == "Financial":
        x = np.random.randn(n)
        for i in range(2, n):
            x[i] += 0.15*x[i-1]
    elif domain == "Ecological":
        for i in range(2, n):
            x[i] = 0.85*x[i-1] - 0.12*x[i-2] + np.random.randn()*0.3
    elif domain == "Physiological":
        for i in range(2, n):
            x[i] = 0.6*x[i-1] + np.random.randn()*0.5
    
    return x

# =============================================================================
# INTERVENTIONS
# =============================================================================

def intervention_shock(signal, intensity=3.0):
    """Apply sudden shock intervention"""
    x = signal.copy()
    # Apply shock at t=200
    x[200:] = x[200:] + intensity * np.random.randn(len(x)-200)
    return x

def intervention_block(signal):
    """Block/pause intervention"""
    x = signal.copy()
    # Block signal at t=200-250
    x[200:250] = x[199] * np.ones(50) + np.random.randn(50)*0.1
    return x

def intervention_reset(signal):
    """Reset state intervention"""
    x = signal.copy()
    # Reset to baseline at t=200
    x[200:] = np.random.randn(len(x)-200) * np.std(x[:200])
    return x

def intervention_cascade(signal):
    """Cascade failure intervention"""
    x = signal.copy()
    # Cascading failure from t=200
    for i in range(200, len(x)):
        x[i] = 1.1 * x[i-1] + np.random.randn() * (0.5 + (i-200)/500)
    return x

# Null interventions (no effect)
def null_intervention_random(signal):
    """Random time-shifted intervention (no actual effect)"""
    x = signal.copy()
    # Simply add small random noise - same as control
    x += np.random.randn(len(x)) * 0.05
    return x

def null_intervention_shape(signal):
    """Shape-matched null (preserve temporal pattern)"""
    x = signal.copy()
    # Add time-symmetric perturbation
    t = np.arange(len(x))
    x += 0.1 * np.sin(2*np.pi*t/100) * np.exp(-((t-200)/50)**2)
    return x

# =============================================================================
# MEASURE RECOVERY DYNAMICS
# =============================================================================

def measure_recovery(signal, intervention_point=200):
    """Measure recovery trajectory after intervention"""
    pre = signal[:intervention_point]
    post = signal[intervention_point:]
    
    # Basic recovery metrics
    mean_pre = np.mean(pre)
    std_pre = np.std(pre)
    mean_post = np.mean(post)
    std_post = np.std(post)
    
    # Spectral recovery - use same length
    min_len = min(len(pre), len(post))
    freqs_pre, psd_pre = welch(pre[:min_len], nperseg=min(32, min_len//4))
    freqs_post, psd_post = welch(post[:min_len], nperseg=min(32, min_len//4))
    
    # Normalize PSD
    psd_pre_norm = psd_pre / (np.sum(psd_pre) + 1e-12)
    psd_post_norm = psd_post / (np.sum(psd_post) + 1e-12)
    
    # Pad to same length
    max_len = max(len(psd_pre_norm), len(psd_post_norm))
    psd_pre_pad = np.zeros(max_len)
    psd_post_pad = np.zeros(max_len)
    psd_pre_pad[:len(psd_pre_norm)] = psd_pre_norm
    psd_post_pad[:len(psd_post_norm)] = psd_post_norm
    
    # Spectral divergence
    spectral_div = np.sum(np.abs(psd_pre_pad - psd_post_pad))
    
    # Temporal asymmetry in recovery
    d = np.diff(post)
    if len(d) > 2:
        asym = np.mean(np.abs(d[:len(d)//2])) - np.mean(np.abs(d[len(d)//2:]))
    else:
        asym = 0
    
    # Entropy change
    ent_pre = scipy_entropy(np.histogram(pre, bins=20, density=True)[0] + 1e-12)
    ent_post = scipy_entropy(np.histogram(post, bins=20, density=True)[0] + 1e-12)
    
    # Variance ratio (persistence)
    var_ratio = std_post / (std_pre + 1e-12)
    
    # Recovery time estimate
    if len(post) > 2:
        ac = np.corrcoef(post[:-1], post[1:])[0,1]
        if not np.isnan(ac):
            recovery_est = 1 - abs(ac)
        else:
            recovery_est = 1.0
    else:
        recovery_est = 1.0
    
    return {
        "spectral_divergence": float(spectral_div),
        "temporal_asymmetry": float(asym),
        "entropy_change": float(ent_post - ent_pre),
        "variance_ratio": float(var_ratio),
        "mean_shift": float(abs(mean_post - mean_pre)),
        "recovery_time_estimate": float(recovery_est)
    }

def compute_all_metrics(signal):
    """Compute all organizational metrics"""
    # Basic statistics
    mean = np.mean(signal)
    std = np.std(signal)
    var = np.var(signal)
    
    # Autocorrelation
    ac1 = np.corrcoef(signal[:-1], signal[1:])[0,1] if len(signal) > 1 else 0
    
    # Spectral
    freqs, psd = welch(signal, nperseg=min(64, len(signal)//4))
    psd_norm = psd / (np.sum(psd) + 1e-12)
    spectral_entropy = scipy_entropy(psd_norm + 1e-12)
    
    # Entropy
    hist = np.histogram(signal, bins=20, density=True)[0]
    shannon = scipy_entropy(hist + 1e-12)
    
    return {
        "mean": mean, "std": std, "var": var,
        "ac1": ac1, "spectral_entropy": spectral_entropy,
        "shannon": shannon, "max_psd": float(np.max(psd)),
        "peak_freq": float(freqs[np.argmax(psd)])
    }

# =============================================================================
# BUILD INTERVENTION STUDY
# =============================================================================

print("[BUILDING INTERVENTION STUDY]")

domains = ["EEG", "Financial", "Ecological", "Physiological"]

interventions = [
    ("Shock", intervention_shock),
    ("Block", intervention_block),
    ("Reset", intervention_reset),
    ("Cascade", intervention_cascade)
]

null_interventions = [
    ("Null_Random", null_intervention_random),
    ("Null_Shape", null_intervention_shape)
]

results = {}

for domain in domains:
    print(f"\nTesting {domain}...")
    
    domain_results = {}
    
    # Generate baseline signals
    baselines = [generate_signal(domain, n=500) for _ in range(60)]
    
    # Apply real interventions
    for int_name, int_func in interventions:
        intervened_signals = [int_func(b.copy()) for b in baselines]
        
        # Measure recovery
        recoveries = [measure_recovery(s) for s in intervened_signals]
        
        # Compute metric changes
        pre_metrics = [compute_all_metrics(b) for b in baselines]
        post_metrics = [compute_all_metrics(s[200:]) for s in intervened_signals]
        
        # Changes
        metric_changes = {}
        for key in pre_metrics[0].keys():
            pre_vals = [m[key] for m in pre_metrics]
            post_vals = [m[key] for m in post_metrics]
            metric_changes[key] = {
                "pre_mean": float(np.mean(pre_vals)),
                "post_mean": float(np.mean(post_vals)),
                "change": float(np.mean(post_vals) - np.mean(pre_vals)),
                "change_ratio": float(np.mean(post_vals) / (np.mean(pre_vals) + 1e-12))
            }
        
        # Recovery characteristics
        recovery_metrics = {
            "spectral_div": float(np.mean([r["spectral_divergence"] for r in recoveries])),
            "asymm": float(np.mean([r["temporal_asymmetry"] for r in recoveries])),
            "entropy_change": float(np.mean([r["entropy_change"] for r in recoveries])),
            "var_ratio": float(np.mean([r["variance_ratio"] for r in recoveries]))
        }
        
        domain_results[int_name] = {
            "metric_changes": metric_changes,
            "recovery": recovery_metrics
        }
    
    # Apply null interventions
    for null_name, null_func in null_interventions:
        null_signals = [null_func(b.copy()) for b in baselines]
        
        recoveries = [measure_recovery(s) for s in null_signals]
        
        recovery_metrics = {
            "spectral_div": float(np.mean([r["spectral_divergence"] for r in recoveries])),
            "asymm": float(np.mean([r["temporal_asymmetry"] for r in recoveries])),
            "entropy_change": float(np.mean([r["entropy_change"] for r in recoveries])),
            "var_ratio": float(np.mean([r["variance_ratio"] for r in recoveries]))
        }
        
        domain_results[null_name] = {
            "recovery": recovery_metrics
        }
    
    results[domain] = domain_results

# =============================================================================
# COMPARE REAL VS NULL INTERVENTIONS
# =============================================================================

print("\n[COMPARING REAL VS NULL INTERVENTIONS]")

# Aggregate
real_effects = {}
null_effects = {}

for domain in domains:
    for int_name, int_func in interventions:
        recovery = results[domain][int_name]["recovery"]
        if int_name not in real_effects:
            real_effects[int_name] = []
        real_effects[int_name].append(recovery["spectral_div"])
    
    for null_name, null_func in null_interventions:
        recovery = results[domain][null_name]["recovery"]
        if null_name not in null_effects:
            null_effects[null_name] = []
        null_effects[null_name].append(recovery["spectral_div"])

print("\nSpectral Divergence (lower = more similar to baseline):")
for int_name, vals in real_effects.items():
    print(f"  {int_name}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
for null_name, vals in null_effects.items():
    print(f"  {null_name}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

# =============================================================================
# TEST FOR INVARIANCE
# =============================================================================

# Check if any metrics remain consistent across interventions
print("\n[TESTING FOR INVARIANCE]")

# Calculate coefficient of variation across interventions
invariant_metrics = []

for domain in domains:
    # Get all intervention results for this domain
    all_recoveries = []
    for int_name in [i[0] for i in interventions]:
        all_recoveries.append(results[domain][int_name]["recovery"])
    
    # Check if any metric is invariant across intervention types
    for metric in ["spectral_div", "asymm", "entropy_change", "var_ratio"]:
        vals = [r[metric] for r in all_recoveries]
        cv = np.std(vals) / (np.abs(np.mean(vals)) + 1e-12)
        
        if cv < 0.5:  # Low variation = invariant
            invariant_metrics.append((domain, metric, cv))

print(f"\nPotential invariants found: {len(invariant_metrics)}")
for d, m, cv in invariant_metrics[:5]:
    print(f"  {d} - {m}: CV={cv:.3f}")

# =============================================================================
# VERDICT
# =============================================================================

# Calculate effect size difference between real and null interventions
real_mean = np.mean([np.mean(v) for v in real_effects.values()])
null_mean = np.mean([np.mean(v) for v in null_effects.values()])

effect_size = abs(real_mean - null_mean) / (np.std([v for v in real_effects.values() for v in v]) + np.std([v for v in null_effects.values() for v in v]) + 1e-12)

print(f"\nReal intervention effect: {real_mean:.4f}")
print(f"Null intervention effect: {null_mean:.4f}")
print(f"Effect size: {effect_size:.4f}")

# Determine verdict
if len(invariant_metrics) > 3 and effect_size > 0.5:
    verdict = "ROBUST_INTERVENTION_CONSTRAINTS"
elif len(invariant_metrics) > 0 and effect_size > 0.3:
    verdict = "PARTIAL_INTERVENTION_STABILITY"
elif effect_size > 0.2:
    verdict = "WEAK_DOMAIN_LOCAL_CONSTRAINTS"
else:
    verdict = "NO_INTERVENTION_INVARIANTS"

print(f"\nFINAL VERDICT: {verdict}")

# =============================================================================
# SAVE EVERYTHING
# =============================================================================

final_results = {
    "domain_results": results,
    "real_effects": {k: [float(v) for v in vals] for k, vals in real_effects.items()},
    "null_effects": {k: [float(v) for v in vals] for k, vals in null_effects.items()},
    "invariant_metrics": [(d, m, float(cv)) for d, m, cv in invariant_metrics],
    "verdict": verdict,
    "analysis": {
        "real_mean": float(real_mean),
        "null_mean": float(null_mean),
        "effect_size": float(effect_size),
        "n_domains": len(domains),
        "n_interventions": len(interventions),
        "n_nulls": len(null_interventions)
    }
}

with open(os.path.join(OUTDIR, "metrics.json"), "w") as f:
    json.dump(final_results, f, indent=2)

# Save signals
for domain in domains:
    signals = [generate_signal(domain, n=500) for _ in range(60)]
    np.save(os.path.join(OUTDIR, f"signals_{domain}.npy"), np.array(signals))

print("\n" + "="*60)
print("PHASE 96 FINAL RESULTS")
print("="*60)
print(f"\n[INTERVENTION EFFECTS]")
print(f"Real interventions: {real_mean:.4f}")
print(f"Null interventions: {null_mean:.4f}")
print(f"Effect size: {effect_size:.4f}")
print(f"\n[INVARIANTS FOUND]")
print(f"Count: {len(invariant_metrics)}")
print(f"\n[FINAL VERDICT]")
print(verdict)
print("="*60)