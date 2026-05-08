"""
PHASE 99 - SCALE-FREE RESIDUAL STRUCTURE TEST
Tests if residuals show scale-free properties or finite memory only
"""

import os
import json
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import welch

np.random.seed(99)
OUTDIR = "/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase99_scale_free_residuals"
os.makedirs(OUTDIR, exist_ok=True)

# =============================================================================
# SCALING ANALYSIS FUNCTIONS
# =============================================================================

def compute_autocorrelation_decay(residuals, max_lag=50):
    """Compute autocorrelation decay"""
    n = len(residuals)
    acf = []
    for lag in range(1, min(max_lag, n//2)):
        ac = np.corrcoef(residuals[:-lag], residuals[lag:])[0,1]
        if not np.isnan(ac):
            acf.append(ac)
    return np.array(acf)

def fit_exponential_decay(lags, acf):
    """Fit exponential decay: ac(l) = A * exp(-l/tau)"""
    def model(l, A, tau):
        return A * np.exp(-l / tau)
    try:
        popt, _ = curve_fit(model, lags, np.abs(acf), p0=[1, 5], bounds=([0, 1], [2, 100]), maxfev=2000)
        # Compute R²
        pred = model(lags, *popt)
        ss_res = np.sum((np.abs(acf) - pred)**2)
        ss_tot = np.sum((np.abs(acf) - np.mean(np.abs(acf)))**2)
        r2 = 1 - ss_res / ss_tot
        return {"type": "exponential", "tau": float(popt[1]), "r2": float(r2), "success": True}
    except:
        return {"type": "exponential", "tau": 0, "r2": 0, "success": False}

def fit_power_law_decay(lags, acf):
    """Fit power-law decay: ac(l) = A * l^(-alpha)"""
    def model(l, A, alpha):
        l = np.maximum(l, 1)
        return A * (l ** (-alpha))
    try:
        popt, _ = curve_fit(model, lags, np.abs(acf), p0=[1, 0.5], bounds=([0, 0.01], [2, 3]), maxfev=2000)
        pred = model(lags, *popt)
        ss_res = np.sum((np.abs(acf) - pred)**2)
        ss_tot = np.sum((np.abs(acf) - np.mean(np.abs(acf)))**2)
        r2 = 1 - ss_res / ss_tot
        return {"type": "power_law", "alpha": float(popt[1]), "r2": float(r2), "success": True}
    except:
        return {"type": "power_law", "alpha": 0, "r2": 0, "success": False}

def fit_stretched_exponential(lags, acf):
    """Fit stretched exponential: ac(l) = A * exp(-(l/tau)^beta)"""
    def model(l, A, tau, beta):
        return A * np.exp(-((l / tau) ** beta))
    try:
        popt, _ = curve_fit(model, lags, np.abs(acf), p0=[1, 10, 0.5], 
                           bounds=([0, 1, 0.01], [2, 100, 1.5]), maxfev=2000)
        pred = model(lags, *popt)
        ss_res = np.sum((np.abs(acf) - pred)**2)
        ss_tot = np.sum((np.abs(acf) - np.mean(np.abs(acf)))**2)
        r2 = 1 - ss_res / ss_tot
        return {"type": "stretched", "tau": float(popt[1]), "beta": float(popt[2]), "r2": float(r2), "success": True}
    except:
        return {"type": "stretched", "tau": 0, "beta": 0, "r2": 0, "success": False}

def compute_hurst_exponent(residuals):
    """Compute Hurst exponent using R/S analysis"""
    def rs_analysis(series, n):
        series = np.array(series)
        length = len(series)
        segments = length // n
        
        rs_values = []
        for i in range(segments):
            segment = series[i*n:(i+1)*n]
            mean = np.mean(segment)
            deviations = segment - mean
            cumdev = np.cumsum(deviations)
            R = np.max(cumdev) - np.min(cumdev)
            S = np.std(segment) + 1e-12
            rs_values.append(R / S)
        
        return np.mean(rs_values)
    
    sizes = [10, 20, 40, 80]
    rs_values = []
    
    for size in sizes:
        if size <= len(residuals) // 2:
            rs = rs_analysis(residuals, size)
            rs_values.append(rs)
    
    # Fit power law: RS ~ n^H
    try:
        log_sizes = np.log(sizes[:len(rs_values)])
        log_rs = np.log(np.array(rs_values) + 1e-12)
        slope, intercept = np.polyfit(log_sizes, log_rs, 1)
        return float(slope)
    except:
        return 0.5

def compute_dfa_scaling(residuals):
    """Compute DFA scaling exponent"""
    def dfa_profile(series):
        return np.cumsum(series - np.mean(series))
    
    def dfa_fluc(series, window_size):
        profile = dfa_profile(series)
        n = len(profile)
        segments = n // window_size
        
        if segments < 2:
            return 0
        
        f2 = []
        for i in range(segments):
            segment = profile[i*window_size:(i+1)*window_size]
            # Fit line
            x = np.arange(len(segment))
            y = segment
            coeffs = np.polyfit(x, y, 1)
            fit = np.polyval(coeffs, x)
            f2.append(np.mean((segment - fit)**2))
        
        return np.sqrt(np.mean(f2))
    
    sizes = [4, 8, 16, 32, 64]
    f_values = []
    
    for size in sizes:
        if size < len(residuals) // 2:
            f = dfa_fluc(residuals, size)
            f_values.append(f)
    
    if len(f_values) < 2:
        return 0.5
    
    # Fit power law: F ~ n^alpha
    log_sizes = np.log(sizes[:len(f_values)])
    log_f = np.log(np.array(f_values) + 1e-12)
    
    try:
        slope, _ = np.polyfit(log_sizes, log_f, 1)
        return float(slope)
    except:
        return 0.5

def compute_spectral_slope(residuals):
    """Compute power spectral density slope"""
    freqs, psd = welch(residuals, nperseg=min(64, len(residuals)//4))
    
    # Only positive frequencies
    freqs = freqs[1:]
    psd = psd[1:]
    
    # Remove zero frequencies
    mask = freqs > 0
    freqs = freqs[mask]
    psd = psd[mask]
    
    if len(freqs) < 3:
        return 0
    
    # Fit power law: PSD ~ f^(-beta)
    log_freqs = np.log(freqs)
    log_psd = np.log(psd + 1e-12)
    
    try:
        slope, _ = np.polyfit(log_freqs, log_psd, 1)
        return float(-slope)  # Negative because slope is negative
    except:
        return 0

# =============================================================================
# GENERATE RESIDUALS
# =============================================================================

def generate_recovery_with_residuals(domain, n=300):
    """Generate recovery trajectory with complex residuals"""
    x = np.zeros(n)
    
    # Base recovery (exponential)
    t = np.arange(n)
    tau = 15 + np.random.rand() * 10
    base = 2 * np.exp(-t / tau)
    
    # Add complex residuals
    if domain == "Ecological":
        for i in range(2, n):
            x[i] = 0.1 * x[i-1] + 0.05 * x[i-2] + np.random.randn() * 0.2
        x += base + 0.3 * np.sin(t / 10)
    elif domain == "Physiological":
        for i in range(2, n):
            x[i] = 0.2 * x[i-1] + np.random.randn() * 0.3
        x += base + 0.2 * np.sin(t / 15)
    elif domain == "Network":
        # Multi-scale
        for i in range(2, n):
            x[i] = 0.15 * x[i-1] + 0.08 * x[i-2] + np.random.randn() * 0.25
        x += base
    elif domain == "Financial":
        # Rough
        for i in range(2, n):
            x[i] = 0.05 * x[i-1] + np.random.randn() * 0.4
        x += base
    elif domain == "Climate":
        # Oscillatory with memory
        for i in range(2, n):
            x[i] = 0.12 * x[i-1] - 0.08 * x[i-2] + np.random.randn() * 0.15
        x += base + 0.4 * np.sin(t / 20)
    
    return x

# =============================================================================
# COMPUTE SCALING
# =============================================================================

print("[COMPUTING SCALING ANALYSIS]")

domains = ["Ecological", "Physiological", "Network", "Financial", "Climate"]

scaling_results = {}

for domain in domains:
    print(f"\n{domain}:")
    
    domain_results = []
    
    for _ in range(40):
        residuals = generate_recovery_with_residuals(domain, n=300)
        
        # Compute ACF decay
        acf = compute_autocorrelation_decay(residuals, max_lag=40)
        
        if len(acf) < 5:
            continue
        
        lags = np.arange(1, len(acf) + 1)
        
        # Fit different decays
        exp_fit = fit_exponential_decay(lags, acf)
        pl_fit = fit_power_law_decay(lags, acf)
        stretch_fit = fit_stretched_exponential(lags, acf)
        
        # Compute scaling exponents
        hurst = compute_hurst_exponent(residuals)
        dfa = compute_dfa_scaling(residuals)
        spectral = compute_spectral_slope(residuals)
        
        domain_results.append({
            "exp_tau": exp_fit.get("tau", 0),
            "exp_r2": exp_fit.get("r2", 0),
            "pl_alpha": pl_fit.get("alpha", 0),
            "pl_r2": pl_fit.get("r2", 0),
            "stretch_tau": stretch_fit.get("tau", 0),
            "stretch_beta": stretch_fit.get("beta", 0),
            "stretch_r2": stretch_fit.get("r2", 0),
            "hurst": hurst,
            "dfa": dfa,
            "spectral_slope": spectral
        })
    
    # Average results
    if domain_results:
        avg_results = {}
        for key in domain_results[0].keys():
            vals = [r[key] for r in domain_results]
            avg_results[key] = float(np.mean(vals))
        
        scaling_results[domain] = avg_results
        
        # Determine best decay type
        best_exp_r2 = np.mean([r["exp_r2"] for r in domain_results])
        best_pl_r2 = np.mean([r["pl_r2"] for r in domain_results])
        best_stretch_r2 = np.mean([r["stretch_r2"] for r in domain_results])
        
        best_decay = "exponential" if best_exp_r2 > max(best_pl_r2, best_stretch_r2) else \
                    "power_law" if best_pl_r2 > best_stretch_r2 else "stretched"
        
        print(f"  Best decay: {best_decay}")
        print(f"  Hurst: {avg_results['hurst']:.3f}")
        print(f"  DFA: {avg_results['dfa']:.3f}")
        print(f"  Spectral: {avg_results['spectral_slope']:.3f}")

# =============================================================================
# NULL CONTROLS
# =============================================================================

print("\n[NULL CONTROLS]")

def generate_surrogate(residuals):
    """Generate surrogate preserving spectrum"""
    fft_vals = np.fft.rfft(residuals)
    mag = np.abs(fft_vals)
    phase = np.random.uniform(0, 2*np.pi, len(fft_vals))
    return np.real(np.fft.irfft(mag * np.exp(1j * phase)))

null_results = {}
for domain in domains:
    # Generate surrogate residuals
    sample_residuals = generate_recovery_with_residuals(domain, n=300)
    surrogate = generate_surrogate(sample_residuals)
    
    # Compute same metrics
    acf = compute_autocorrelation_decay(surrogate, max_lag=40)
    if len(acf) > 5:
        lags = np.arange(1, len(acf) + 1)
        pl_fit = fit_power_law_decay(lags, acf)
        hurst = compute_hurst_exponent(surrogate)
        dfa = compute_dfa_scaling(surrogate)
        
        null_results[domain] = {
            "pl_alpha": pl_fit.get("alpha", 0),
            "hurst": hurst,
            "dfa": dfa
        }
        
        print(f"  {domain}: α={pl_fit.get('alpha', 0):.3f}, H={hurst:.3f}, DFA={dfa:.3f}")

# =============================================================================
# CROSS-DOMAIN SCALING TRANSFER
# =============================================================================

print("\n[CROSS-DOMAIN SCALING]")

# Compare exponents across domains
exponent_comparison = {}
for key in ["hurst", "dfa", "spectral_slope"]:
    values = [scaling_results[d][key] for d in domains]
    null_values = [null_results[d][key] if key in null_results[d] else 0 for d in domains]
    
    exponent_comparison[key] = {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "cv": float(np.std(values) / (np.mean(values) + 1e-12)),
        "null_mean": float(np.mean(null_values)),
        "real_null_diff": float(np.mean(values) - np.mean(null_values))
    }
    
    print(f"  {key}: Mean={np.mean(values):.3f}±{np.std(values):.3f}, CV={np.std(values)/(np.mean(values)+1e-12):.2f}")

# =============================================================================
# VERDICT
# =============================================================================

# Determine verdict
avg_cv = np.mean([exponent_comparison[k]["cv"] for k in exponent_comparison])
avg_null_gap = np.mean([exponent_comparison[k]["real_null_diff"] for k in exponent_comparison])

# Check if scaling is stable across domains (low CV)
# Check if scaling is different from null (significant gap)
# Check if power-law fits well (compare R² values)

power_law_better = []
for domain in domains:
    pl_r2 = scaling_results[domain]["pl_r2"]
    exp_r2 = scaling_results[domain]["exp_r2"]
    if pl_r2 > exp_r2:
        power_law_better.append(domain)

print(f"\nPower-law better in {len(power_law_better)}/{len(domains)} domains")

# Determine verdict
if avg_cv < 0.2 and avg_null_gap > 0.1 and len(power_law_better) >= 3:
    verdict = "POSSIBLE_SCALE_FREE_STRUCTURE"
elif avg_cv < 0.3 and avg_null_gap > 0.05:
    verdict = "DOMAIN_SPECIFIC_SCALING"
elif avg_cv > 0.5:
    verdict = "FINITE_MEMORY_ONLY"
else:
    verdict = "WEAK_SCALING"

print(f"\nFINAL VERDICT: {verdict}")

# =============================================================================
# SAVE EVERYTHING
# =============================================================================

final_results = {
    "domain_results": scaling_results,
    "null_results": null_results,
    "exponent_comparison": exponent_comparison,
    "verdict": verdict,
    "analysis": {
        "avg_cv": float(avg_cv),
        "avg_null_gap": float(avg_null_gap),
        "power_law_better_count": len(power_law_better),
        "n_domains": len(domains)
    }
}

with open(os.path.join(OUTDIR, "metrics.json"), "w") as f:
    json.dump(final_results, f, indent=2)

# Save sample residuals
for domain in domains:
    residuals = generate_recovery_with_residuals(domain, n=300)
    np.save(os.path.join(OUTDIR, f"residuals_{domain}.npy"), residuals)

print("\n" + "="*60)
print("PHASE 99 FINAL RESULTS")
print("="*60)
print(f"\n[SCALING EXPONENTS]")
for domain in domains:
    print(f"  {domain}: H={scaling_results[domain]['hurst']:.3f}, DFA={scaling_results[domain]['dfa']:.3f}")
print(f"\n[NULL COMPARISON]")
for key, vals in exponent_comparison.items():
    print(f"  {key}: Real={vals['mean']:.3f}, Null={vals['null_mean']:.3f}, Gap={vals['real_null_diff']:.3f}")
print(f"\n[FINAL VERDICT]")
print(verdict)
print("="*60)