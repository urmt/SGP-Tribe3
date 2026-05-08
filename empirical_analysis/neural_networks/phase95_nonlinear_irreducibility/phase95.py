"""
PHASE 95 - NONLINEAR IRREDUCIBILITY TEST
Tests if nonlinear temporal structure is irreducible or reducible to linear statistics
"""

import os
import json
import numpy as np
from scipy.signal import welch
from scipy.fft import fft, ifft
from scipy.stats import entropy as scipy_entropy

np.random.seed(95)
OUTDIR = "/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase95_nonlinear_irreducibility"
os.makedirs(OUTDIR, exist_ok=True)

# =============================================================================
# REALISTIC SIGNALS (Same as Phase 94)
# =============================================================================

def generate_signal(domain_type, n=500):
    """Generate realistic temporal signals"""
    x = np.zeros(n)
    
    if domain_type == "EEG":
        for i in range(2, n):
            x[i] = 0.7*x[i-1] - 0.2*x[i-2] + np.random.randn()*0.4
        t = np.arange(n)
        x += 0.5*np.sin(2*np.pi*10*t/100)
    elif domain_type == "Financial":
        x = np.random.randn(n)
        for i in range(2, n):
            x[i] += 0.15*x[i-1]
    elif domain_type == "Ecological":
        for i in range(2, n):
            x[i] = 0.85*x[i-1] - 0.12*x[i-2] + np.random.randn()*0.3
    elif domain_type == "Language":
        for i in range(2, n):
            x[i] = 0.75*x[i-1] + np.random.randn()*0.5
        x += 0.3*np.sin(np.arange(n)*0.1)
    elif domain_type == "Physiological":
        for i in range(2, n):
            x[i] = 0.6*x[i-1] + np.random.randn()*0.5
    
    return x

# =============================================================================
# NONLINEAR DESTRUCTION SURROGATES (Preserve linear structure)
# =============================================================================

def surrogate_iaaft(x, n_iter=10):
    """Iterative amplitude adjusted Fourier transform - preserves spectrum"""
    x = np.asarray(x)
    n = len(x)
    
    # Get target amplitude spectrum
    x_fft = fft(x)
    target_amp = np.abs(x_fft)
    
    # Start with random phases
    phases = np.random.uniform(0, 2*np.pi, n)
    surrogate = np.real(ifft(target_amp * np.exp(1j*phases)))
    
    # Iterate
    for _ in range(n_iter):
        # Sort to preserve amplitude distribution
        ranks = np.argsort(np.argsort(surrogate))
        surrogate[ranks] = x[np.argsort(x)]
        
        # Adjust spectrum
        surr_fft = fft(surrogate)
        surr_fft = target_amp * np.exp(1j * np.angle(surr_fft))
        surrogate = np.real(ifft(surr_fft))
    
    return surrogate

def surrogate_time_reverse(x):
    """Time reversal - destroys direction but preserves spectrum"""
    return x[::-1] + np.random.randn(len(x)) * np.std(x) * 0.1

def surrogate_shuffle_blocks(x, block_size=10):
    """Shuffle temporal blocks - preserves local statistics"""
    n = len(x)
    n_blocks = n // block_size
    blocks = [x[i*block_size:(i+1)*block_size] for i in range(n_blocks)]
    np.random.shuffle(blocks)
    return np.concatenate(blocks)

def surrogate_nonlinear_phase(x):
    """Nonlinear phase randomization"""
    x_fft = fft(x)
    amp = np.abs(x_fft)
    
    # Add nonlinear phase
    phases = np.cumsum(np.random.randn(len(x)))  # Random walk for phase
    phases = phases - phases[0]  # Center
    
    return np.real(ifft(amp * np.exp(1j*phases)))

def surrogate_marques(x):
    """Marques surrogates - preserve linear correlations"""
    x = np.asarray(x)
    n = len(x)
    
    if n < 50:
        return np.random.permutation(x)
    
    # Fit AR model
    from sklearn.linear_model import Ridge
    X_train = np.column_stack([x[1:-1], x[:-2]])
    y_train = x[2:]
    
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    # Generate surrogate from AR residuals
    residuals = y_train - model.predict(X_train)
    np.random.shuffle(residuals)  # Shuffle residuals
    
    surrogate = np.zeros(n)
    surrogate[0] = x[0]
    if n > 1:
        surrogate[1] = x[1]
    
    for i in range(2, n):
        pred = model.predict([[surrogate[i-1], surrogate[i-2]]])[0]
        r_idx = i - 2
        if r_idx < len(residuals):
            surrogate[i] = pred + residuals[r_idx]
        else:
            surrogate[i] = pred
    
    return surrogate

# =============================================================================
# NONLINEAR METRICS
# =============================================================================

def permutation_entropy(x, order=3):
    """Compute permutation entropy - measures temporal structure"""
    n = len(x) - order
    patterns = []
    for i in range(n):
        pattern = tuple(np.argsort(x[i:i+order+1]))
        patterns.append(pattern)
    
    # Count patterns
    unique_patterns = set(patterns)
    counts = [patterns.count(p) for p in unique_patterns]
    counts = np.array(counts) / len(patterns)
    
    return -np.sum(counts * np.log(counts + 1e-12))

def lempel_ziv_complexity(x, binary_threshold=None):
    """Lempel-Ziv complexity - measures algorithmic complexity"""
    if binary_threshold is None:
        binary_threshold = np.median(x)
    
    binary = "".join(["1" if v > binary_threshold else "0" for v in x])
    
    n = len(binary)
    complexity = 0
    i = 0
    subseqs = set()
    
    while i < n:
        found = False
        for j in range(i+1, n+1):
            subseq = binary[i:j]
            if subseq not in subseqs:
                subseqs.add(subseq)
                complexity += 1
                i = j
                found = True
                break
        if not found:
            break
    
    return complexity

def temporal_asymmetry(x):
    """Compute temporal asymmetry - forward vs backward time"""
    d_forward = np.diff(x)
    d_backward = np.diff(x[::-1])
    
    # First and second order asymmetry
    mean_forward = np.mean(np.abs(d_forward))
    mean_backward = np.mean(np.abs(d_backward))
    
    var_forward = np.var(d_forward)
    var_backward = np.var(d_backward)
    
    return abs(mean_forward - mean_backward) / (mean_forward + mean_backward + 1e-12)

def nonlinear_prediction_error(x, horizon=1):
    """Simple nonlinear prediction via k-nearest neighbors"""
    from sklearn.neighbors import KNeighborsRegressor
    
    n = len(x)
    if n < 100:
        return 1.0
    
    # Simple embedding
    X = []
    y = []
    for i in range(horizon, n-1):
        X.append([x[i-1], x[i-2]])
        y.append(x[i])
    
    X = np.array(X)
    y = np.array(y)
    
    if len(X) < 30:
        return 1.0
    
    try:
        knn = KNeighborsRegressor(n_neighbors=5)
        knn.fit(X[:-10], y[:-10])
        pred = knn.predict(X[-10:])
        error = np.mean((pred - y[-10:])**2)
        return error / (np.var(y) + 1e-12)
    except:
        return 1.0

def transfer_entropy(x, y, lag=1):
    """Simplified transfer entropy from X to Y"""
    if lag >= len(x) or lag >= len(y):
        return 0
    
    # Very simplified version
    x_lag = x[:-lag-1]
    y_past = y[:-lag-1]
    y_future = y[lag:-1]
    
    # Conditional entropy approach (simplified)
    return np.abs(np.corrcoef(x_lag, y_future)[0,1]) * np.abs(np.corrcoef(y_past, y_future)[0,1])

# =============================================================================
# BUILD DATASET
# =============================================================================

print("[BUILDING DATASETS]")
domains = ["EEG", "Financial", "Ecological", "Language", "Physiological"]
domain_signals = {}

for domain in domains:
    signals = []
    for _ in range(80):  # 80 signals per domain
        signals.append(generate_signal(domain, n=500))
    domain_signals[domain] = np.array(signals)
    print(f"  {domain}: {len(signals)} signals")

# =============================================================================
# TEST NONLINEAR IRREDUCIBILITY
# =============================================================================

print("\n[TESTING NONLINEAR IRREDUCIBILITY]")

surrogates = [
    ("IAAFT", surrogate_iaaft),
    ("TimeReverse", surrogate_time_reverse),
    ("BlockShuffle", surrogate_shuffle_blocks),
    ("NonlinearPhase", surrogate_nonlinear_phase),
    ("AR_Surrogate", surrogate_marques)
]

metrics = [
    ("PermutationEntropy", permutation_entropy),
    ("LempelZiv", lempel_ziv_complexity),
    ("TemporalAsymmetry", temporal_asymmetry),
    ("PredictionError", nonlinear_prediction_error)
]

results = {}

for domain in domains:
    print(f"\nTesting {domain}...")
    signals = domain_signals[domain]
    
    domain_results = {}
    
    # Original metrics
    original_metrics = {}
    for metric_name, metric_func in metrics:
        vals = [metric_func(s) for s in signals]
        original_metrics[metric_name] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals))
        }
    
    domain_results["original"] = original_metrics
    
    # Surrogate metrics
    for surr_name, surr_func in surrogates:
        surr_metrics = {}
        
        for metric_name, metric_func in metrics:
            surr_vals = []
            for s in signals:
                surr = surr_func(s.copy())
                surr_vals.append(metric_func(surr))
            
            # Compute survival ratio
            orig_mean = original_metrics[metric_name]["mean"]
            surr_mean = np.mean(surr_vals)
            
            if abs(orig_mean) > 1e-6:
                survival = surr_mean / orig_mean
            else:
                survival = 1.0
            
            # Bound
            survival = max(0, min(survival, 2))
            
            surr_metrics[metric_name] = {
                "surrogate_mean": float(surr_mean),
                "survival_ratio": float(survival)
            }
        
        domain_results[surr_name] = surr_metrics
    
    results[domain] = domain_results

# =============================================================================
# AGGREGATE RESULTS
# =============================================================================

print("\n[AGGREGATE RESULTS]")

# For each surrogate, compute average survival across metrics and domains
surrogate_survival = {}
for surr_name, _ in surrogates:
    survival_ratios = []
    for domain in domains:
        for metric_name, _ in metrics:
            if "survival_ratio" in results[domain][surr_name].get(metric_name, {}):
                survival_ratios.append(
                    results[domain][surr_name][metric_name]["survival_ratio"]
                )
    if survival_ratios:
        surrogate_survival[surr_name] = {
            "mean": float(np.mean(survival_ratios)),
            "std": float(np.std(survival_ratios)),
            "min": float(np.min(survival_ratios)),
            "max": float(np.max(survival_ratios))
        }

print("\nSurrogate Survival Ratios:")
for surr, stats in surrogate_survival.items():
    print(f"  {surr}: {stats['mean']:.3f} ± {stats['std']:.3f}")

# =============================================================================
# NULL COMPARISON (Permutation test)
# =============================================================================

print("\n[NULL COMPARISON]")

# For one domain, run permutation test
domain_test = "EEG"
signals = domain_signals[domain_test]

null_results = {}
for metric_name, metric_func in metrics:
    null_survival = []
    for _ in range(20):
        # Permute labels and recompute
        perm_signals = [np.random.permutation(s) for s in signals[:20]]
        orig_vals = [metric_func(s) for s in signals[:20]]
        perm_vals = [metric_func(s) for s in perm_signals]
        
        if np.mean(orig_vals) != 0:
            null_survival.append(np.mean(perm_vals) / np.mean(orig_vals))
    
    null_results[metric_name] = float(np.mean(null_survival))

print("Null (permuted) survival ratios:")
for metric, val in null_results.items():
    print(f"  {metric}: {val:.3f}")

# =============================================================================
# VERDICT
# =============================================================================

avg_survival = np.mean([s["mean"] for s in surrogate_survival.values()])
null_avg = np.mean(list(null_results.values()))

print(f"\nAverage surrogate survival: {avg_survival:.3f}")
print(f"Null permutation survival: {null_avg:.3f}")

# If surrogates still show high survival (>0.7), structure is reducible
# If surrogates collapse (<0.5), there's nonlinear structure
# But need to compare to null

if avg_survival > 0.8:
    verdict = "FULLY_LINEARLY_REDUCIBLE"
elif avg_survival > 0.6:
    verdict = "MOSTLY_LINEARLY_REDUCIBLE"
elif avg_survival < 0.4:
    verdict = "ROBUST_NONLINEAR_IRREDUCIBILITY"
elif avg_survival < 0.6:
    verdict = "WEAK_NONLINEAR_SURVIVAL"
else:
    verdict = "POSSIBLE_NONLINEAR_IRREDUCIBILITY"

print(f"\nFINAL VERDICT: {verdict}")

# =============================================================================
# SAVE EVERYTHING
# =============================================================================

final_results = {
    "domain_results": results,
    "surrogate_survival": surrogate_survival,
    "null_results": null_results,
    "verdict": verdict,
    "analysis": {
        "avg_survival": float(avg_survival),
        "null_survival": float(null_avg),
        "n_domains": len(domains),
        "n_signals_per_domain": 80,
        "n_surrogates": len(surrogates),
        "n_metrics": len(metrics)
    }
}

with open(os.path.join(OUTDIR, "metrics.json"), "w") as f:
    json.dump(final_results, f, indent=2)

# Save signals
for domain in domains:
    np.save(os.path.join(OUTDIR, f"signals_{domain}.npy"), domain_signals[domain])

# Save survival matrix
import pandas as pd
survival_df = pd.DataFrame(surrogate_survival).T
survival_df.to_csv(os.path.join(OUTDIR, "surrogate_survival.csv"))

print("\n" + "="*60)
print("PHASE 95 FINAL RESULTS")
print("="*60)
print(f"\n[NONLINEAR SURVIVAL]")
for surr, stats in surrogate_survival.items():
    print(f"  {surr}: {stats['mean']:.3f}")
print(f"\n[NULL COMPARISON]")
for metric, val in null_results.items():
    print(f"  {metric}: {val:.3f}")
print(f"\n[FINAL VERDICT]")
print(verdict)
print("="*60)