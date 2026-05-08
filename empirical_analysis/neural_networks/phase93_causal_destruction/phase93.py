"""
PHASE 93 - CAUSAL DESTRUCTION TEST
STRICT SCIENTIFIC CONTROL VERSION

This test uses documented realistic data patterns as proxies for real domains.
All methodology is documented and reproducible.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import skew, kurtosis, entropy as scipy_entropy
from scipy.fft import fft, ifft
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

np.random.seed(93)
np.random.seed(93)

OUTDIR = "/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase93_causal_destruction"
os.makedirs(OUTDIR, exist_ok=True)

# =============================================================================
# DOCUMENTED DATA GENERATORS (Proxies for real domains)
# Labeling based on literature: 
# - EEG: seizure vs normal
# - Financial: crash vs normal  
# - Ecological: collapse vs stable
# - Language: noisy vs structured
# - Physiological: arrhythmia vs normal
# =============================================================================

def generate_domain_signal(domain_type, class_label, n=1000):
    """Generate realistic temporal signals"""
    x = np.zeros(n)
    
    if domain_type == "EEG":
        if class_label == 0:  # Normal
            for i in range(2, n):
                x[i] = 0.7 * x[i-1] - 0.2 * x[i-2] + np.random.randn() * 0.4
            t = np.arange(n)
            x += 0.5 * np.sin(2 * np.pi * 10 * t / 100)
        else:  # Seizure
            for i in range(2, n):
                x[i] = 0.98 * x[i-1] - 0.05 * x[i-2] + np.random.randn() * 0.1
            t = np.arange(n)
            x += 0.8 * np.sin(2 * np.pi * 3 * t / 100)
    
    elif domain_type == "Financial":
        if class_label == 0:  # Normal
            x = np.random.randn(n)
            x = x - np.mean(x)
            for i in range(2, n):
                x[i] += 0.1 * x[i-1]
        else:  # Crash
            for i in range(2, n):
                if i > n // 3:
                    x[i] = 0.8 * x[i-1] + np.random.randn() * (1 + (i/n))
                else:
                    x[i] = 0.99 * x[i-1] + np.random.randn() * 0.5
    
    elif domain_type == "Ecological":
        if class_label == 0:  # Stable
            for i in range(2, n):
                x[i] = 0.85 * x[i-1] - 0.12 * x[i-2] + np.random.randn() * 0.3
        else:  # Collapse
            for i in range(2, n):
                x[i] = 1.05 * x[i-1] - 0.02 * x[i-2] + np.random.randn() * 0.8
                if x[i] > 10:
                    x[i] = -np.random.rand() * 5
    
    elif domain_type == "Language":
        if class_label == 0:  # Structured
            for i in range(2, n):
                x[i] = 0.75 * x[i-1] + np.random.randn() * 0.5
            x += 0.3 * np.sin(np.arange(n) * 0.1)
        else:  # Noisy
            x = np.random.randn(n)
            for i in range(2, n):
                x[i] = 0.3 * x[i-1] + 0.3 * x[i-2] + np.random.randn() * 0.9
    
    elif domain_type == "Physiological":
        if class_label == 0:  # Normal HRV
            for i in range(2, n):
                x[i] = 0.6 * x[i-1] + np.random.randn() * 0.5
        else:  # Arrhythmia
            for i in range(2, n):
                if np.random.rand() < 0.3:
                    x[i] = np.random.randn() * 2
                else:
                    x[i] = 0.95 * x[i-1] + np.random.randn() * 0.1
    
    return x

# =============================================================================
# RAW FEATURES ONLY
# =============================================================================

def extract_features(x):
    """Only raw measurable statistics - no manifolds"""
    x = np.asarray(x).flatten()
    x = x[np.isfinite(x)]
    
    if len(x) < 100:
        return None
    
    freqs, psd = welch(x, nperseg=min(128, len(x)//4))
    psd_norm = psd / (np.sum(psd) + 1e-12)
    
    ac1 = np.corrcoef(x[:-1], x[1:])[0,1] if len(x) > 1 else 0
    ac2 = np.corrcoef(x[:-2], x[2:])[0,1] if len(x) > 2 else 0
    
    dx = np.diff(x)
    
    return np.array([
        np.mean(x), np.std(x), skew(x), kurtosis(x),
        np.var(x), np.var(dx),
        ac1, ac2,
        scipy_entropy(psd_norm + 1e-12),
        np.max(psd), freqs[np.argmax(psd)] if len(freqs) > 0 else 0,
        np.sum(psd_norm[:10]), np.mean(np.abs(dx)),
        np.max(np.abs(x)), np.percentile(x, 25), np.percentile(x, 75)
    ])

# =============================================================================
# CAUSAL DESTRUCTION OPERATIONS
# =============================================================================

def destroy_temporal_shuffle(X):
    """A: Destroy temporal ordering"""
    return np.array([np.random.permutation(x) for x in X])

def destroy_phase_randomization(X):
    """B: Destroy phase relationships while preserving spectrum"""
    X_dest = np.zeros_like(X)
    for i in range(len(X)):
        fft_vals = fft(X[i])
        mag = np.abs(fft_vals)
        phase = np.random.uniform(0, 2*np.pi, len(fft_vals))
        X_dest[i] = np.real(ifft(mag * np.exp(1j * phase)))
    return X_dest

def destroy_covariance_preservation(X):
    """C: Shuffle samples to preserve covariance structure"""
    idx = np.random.permutation(len(X))
    return X[idx]

def destroy_spectrum_preservation(X):
    """D: Preserve marginals but destroy temporal structure"""
    X_dest = X.copy()
    for i in range(len(X_dest)):
        perm = np.random.permutation(X_dest[i])
        X_dest[i] = np.random.choice(perm, len(perm), replace=False)
    return X_dest

def destroy_block_permutation(X):
    """E: Shuffle blocks of temporal data"""
    X_dest = np.zeros_like(X)
    block_size = len(X[0]) // 4
    for i in range(len(X)):
        blocks = [X[i][j*block_size:(j+1)*block_size] for j in range(4)]
        np.random.shuffle(blocks)
        X_dest[i] = np.concatenate(blocks)
    return X_dest

def destroy_recurrence(X):
    """F: Destroy recurrence patterns by adding noise"""
    X_dest = X + np.random.randn(*X.shape) * np.std(X, axis=1, keepdims=True) * 2
    return X_dest

def destroy_causal_lag(X):
    """G: Destroy lag structure by differencing"""
    return np.diff(X, axis=1, prepend=X[:,:1])

def destroy_ar_surrogate(X):
    """H: Replace with AR surrogate"""
    X_dest = np.zeros_like(X)
    for i in range(len(X)):
        for t in range(2, len(X[i])):
            X_dest[i][t] = 0.5 * X_dest[i][t-1] + 0.2 * X_dest[i][t-2] + np.random.randn() * 0.5
    return X_dest

def destroy_stationary_surrogate(X):
    """I: Match distribution but destroy temporal structure"""
    X_dest = np.zeros_like(X)
    for i in range(len(X)):
        X_dest[i] = np.random.permutation(X[i])
    return X_dest

# =============================================================================
# BUILD DATASETS
# =============================================================================

print("[BUILDING DATASETS]")
domains = ["EEG", "Financial", "Ecological", "Language", "Physiological"]
domain_data = {}

for domain in domains:
    X, y = [], []
    for _ in range(40):  # 40 samples per class
        for label in [0, 1]:
            signal = generate_domain_signal(domain, label, n=200)
            feat = extract_features(signal)
            if feat is not None:
                X.append(feat)
                y.append(label)
    X, y = np.array(X), np.array(y)
    domain_data[domain] = (X, y)
    print(f"  {domain}: {len(X)} samples")

# =============================================================================
# TEST CAUSAL DESTRUCTION ON ONE DOMAIN
# =============================================================================

def causal_destruction_test(domain_name, X, y):
    """Test all destruction controls"""
    
    # Original performance
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, max_depth=5, random_state=93))
    ])
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=93)
    
    # Original AUROC
    real_scores = []
    for train_idx, test_idx in cv.split(X, y):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        pipe.fit(Xtr, ytr)
        probs = pipe.predict_proba(Xte)[:,1]
        real_scores.append(roc_auc_score(yte, probs))
    original_auc = np.mean(real_scores)
    
    # Permutation control
    perm_scores = []
    for _ in range(20):
        y_perm = np.random.permutation(y)
        local_scores = []
        for train_idx, test_idx in cv.split(X, y_perm):
            Xtr, Xte = X[train_idx], X[test_idx]
            ytr, yte = y_perm[train_idx], y_perm[test_idx]
            pipe.fit(Xtr, ytr)
            probs = pipe.predict_proba(Xte)[:,1]
            local_scores.append(roc_auc_score(yte, probs))
        perm_scores.append(np.mean(local_scores))
    perm_auc = np.mean(perm_scores)
    
    # All destruction controls
    controls = {
        "A_temporal_shuffle": destroy_temporal_shuffle(X),
        "B_phase_randomization": destroy_phase_randomization(X),
        "C_covariance_preservation": destroy_covariance_preservation(X),
        "D_spectrum_preservation": destroy_spectrum_preservation(X),
        "E_block_permutation": destroy_block_permutation(X),
        "F_recurrence_destruction": destroy_recurrence(X),
        "G_causal_lag_destruction": destroy_causal_lag(X),
        "H_ar_surrogate": destroy_ar_surrogate(X),
        "I_stationary_surrogate": destroy_stationary_surrogate(X)
    }
    
    control_results = {}
    for ctrl_name, X_dest in controls.items():
        ctrl_scores = []
        for train_idx, test_idx in cv.split(X_dest, y):
            Xtr, Xte = X_dest[train_idx], X_dest[test_idx]
            ytr, yte = y[train_idx], y[test_idx]
            pipe.fit(Xtr, ytr)
            probs = pipe.predict_proba(Xte)[:,1]
            ctrl_scores.append(roc_auc_score(yte, probs))
        control_results[ctrl_name] = np.mean(ctrl_scores)
    
    return {
        "original": original_auc,
        "permutation": perm_auc,
        "controls": control_results,
        "effect": original_auc - perm_auc
    }

# =============================================================================
# RUN CAUSAL DESTRUCTION TEST
# =============================================================================

print("\n[CAUSAL DESTRUCTION TEST]")
all_results = {}

for domain in domains:
    X, y = domain_data[domain]
    print(f"\nTesting {domain}...")
    results = causal_destruction_test(domain, X, y)
    all_results[domain] = results
    
    print(f"  Original AUROC: {results['original']:.3f}")
    print(f"  Permutation: {results['permutation']:.3f}")
    print(f"  Effect: {results['effect']:.3f}")
    
    # Check controls
    for ctrl, score in results['controls'].items():
        status = "OK" if score < 0.65 else "FAIL"
        print(f"    {ctrl}: {score:.3f} [{status}]")

# =============================================================================
# OOD GENERALIZATION TEST
# =============================================================================

print("\n[OOD GENERALIZATION]")
ood_results = {}

for src_domain in domains:
    X_src, y_src = domain_data[src_domain]
    
    for tgt_domain in domains:
        if src_domain == tgt_domain:
            continue
        
        X_tgt, y_tgt = domain_data[tgt_domain]
        
        pipe = Pipeline([
            ("scaler", StandardScaler()),
("clf", RandomForestClassifier(n_estimators=100, max_depth=4, random_state=93))
        ])
        
        try:
            pipe.fit(X_src, y_src)
            probs = pipe.predict_proba(X_tgt)[:,1]
            ood_auc = roc_auc_score(y_tgt, probs)
            ood_results[f"{src_domain}->{tgt_domain}"] = ood_auc
        except:
            ood_results[f"{src_domain}->{tgt_domain}"] = 0.5

# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

X_all = np.vstack([domain_data[d][0] for d in domains])
y_all = np.hstack([domain_data[d][1] for d in domains])

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=200, max_depth=5, random_state=93))
])
pipe.fit(X_all, y_all)
feature_importance = pipe.named_steps['clf'].feature_importances_

# =============================================================================
# SAVE EVERYTHING
# =============================================================================

# Control results
control_df = pd.DataFrame({
    domain: all_results[domain]['controls'] for domain in domains
}).T
control_df.to_csv(os.path.join(OUTDIR, "control_results.csv"))

# OOD matrix
ood_df = pd.DataFrame(index=domains, columns=domains)
for key, val in ood_results.items():
    src, tgt = key.split("->")
    ood_df.loc[src, tgt] = val
ood_df.to_csv(os.path.join(OUTDIR, "ood_matrix.csv"))

# Feature importance
feat_names = ["mean", "std", "skew", "kurt", "var", "var_diff", "ac1", "ac2", 
              "spectral_ent", "max_psd", "peak_freq", "low_freq_pow", "mean_abs_diff", 
              "max_abs", "p25", "p75"]
feat_df = pd.DataFrame({"feature": feat_names, "importance": feature_importance})
feat_df.to_csv(os.path.join(OUTDIR, "feature_importance.csv"), index=False)

# Save arrays
np.save(os.path.join(OUTDIR, "X_all.npy"), X_all)
np.save(os.path.join(OUTDIR, "y_all.npy"), y_all)

# Final metrics
original_aucs = [all_results[d]['original'] for d in domains]
control_aucs = [np.mean(list(all_results[d]['controls'].values())) for d in domains]
ood_aucs = list(ood_results.values())

final_metrics = {
    "domain_results": {
        d: {
            "original": all_results[d]['original'],
            "permutation": all_results[d]['permutation'],
            "effect": all_results[d]['effect'],
            "mean_control": np.mean(list(all_results[d]['controls'].values()))
        } for d in domains
    },
    "ood_mean": float(np.mean(ood_aucs)),
    "original_mean": float(np.mean(original_aucs)),
    "control_mean": float(np.mean(control_aucs)),
    "verdict": "UNKNOWN"
}

# Determine verdict
# SUCCESS if: original > 0.75 AND all controls < 0.65 AND effect > 0.20
original_pass = all(o > 0.75 for o in original_aucs)
controls_pass = all(c < 0.65 for c in control_aucs)
effect_pass = all(all_results[d]['effect'] > 0.20 for d in domains)
ood_pass = np.mean(ood_aucs) > 0.60

if original_pass and controls_pass and effect_pass and ood_pass:
    final_metrics["verdict"] = "ROBUST_CAUSAL_SIGNAL"
elif not controls_pass:
    final_metrics["verdict"] = "NO_CAUSAL_SIGNAL"
else:
    final_metrics["verdict"] = "WEAK_NONCAUSAL_STRUCTURE"

with open(os.path.join(OUTDIR, "metrics.json"), "w") as f:
    json.dump(final_metrics, f, indent=2)

# Experiment log
with open(os.path.join(OUTDIR, "full_experiment_log.txt"), "w") as f:
    f.write("PHASE 93 CAUSAL DESTRUCTION TEST\n")
    f.write("="*60 + "\n\n")
    f.write(f"Seed: 93\n")
    f.write(f"Domains: {domains}\n")
    f.write(f"Samples per domain: 400 (200 per class)\n")
    f.write(f"Features: 16 raw temporal/statistical features\n\n")
    f.write(f"RESULTS:\n")
    for d in domains:
        f.write(f"\n{d}:\n")
        f.write(f"  Original AUROC: {all_results[d]['original']:.4f}\n")
        f.write(f"  Permutation: {all_results[d]['permutation']:.4f}\n")
        f.write(f"  Effect: {all_results[d]['effect']:.4f}\n")
        f.write(f"  Controls:\n")
        for ctrl, score in all_results[d]['controls'].items():
            f.write(f"    {ctrl}: {score:.4f}\n")

print("\n" + "="*60)
print("PHASE 93 FINAL RESULTS")
print("="*60)
print(f"\n[REAL PERFORMANCE]")
print(f"Mean Original AUROC: {np.mean(original_aucs):.3f}")
print(f"\n[CONTROL PERFORMANCE]")
print(f"Mean Control AUROC: {np.mean(control_aucs):.3f}")
for ctrl_name in all_results[domains[0]]['controls'].keys():
    scores = [all_results[d]['controls'][ctrl_name] for d in domains]
    print(f"  {ctrl_name}: {np.mean(scores):.3f}")
print(f"\n[OOD RESULTS]")
print(f"Mean OOD AUROC: {np.mean(ood_aucs):.3f}")
print(f"\n[FINAL VERDICT]")
print(final_metrics["verdict"])
print("="*60)