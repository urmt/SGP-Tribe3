"""
PHASE 93 - CAUSAL DESTRUCTION (SIMPLIFIED)
"""
import os
import json
import numpy as np
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from scipy.fft import fft, ifft
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

np.random.seed(93)
OUTDIR = "/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase93_causal_destruction"
os.makedirs(OUTDIR, exist_ok=True)

# Simple generators
def gen_signal(domain, label, n=200):
    x = np.zeros(n)
    if label == 0:
        for i in range(2, n):
            x[i] = 0.7*x[i-1] + np.random.randn()*0.4
    else:
        for i in range(2, n):
            x[i] = 0.95*x[i-1] + np.random.randn()*0.1
    return x

def feats(x):
    freqs, psd = welch(x, nperseg=64)
    ac1 = np.corrcoef(x[:-1], x[1:])[0,1]
    return np.array([np.mean(x), np.std(x), skew(x), kurtosis(x), 
                     np.var(x), ac1, np.max(psd), freqs[np.argmax(psd)]])

# Build data
X, y = [], []
for _ in range(50):
    for label in [0, 1]:
        X.append(feats(gen_signal("test", label)))
        y.append(label)
X, y = np.array(X), np.array(y)
print(f"Data: {X.shape}")

# Controls
def destroy_phase(X):
    Xd = np.zeros_like(X)
    for i in range(len(X)):
        f = fft(X[i]); m = np.abs(f); p = np.random.uniform(0, 2*np.pi, len(f))
        Xd[i] = np.real(ifft(m*np.exp(1j*p)))
    return Xd

def destroy_temporal(X):
    return np.array([np.random.permutation(x) for x in X])

# Test
pipe = Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=50, max_depth=3, random_state=93))])
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=93)

# Original
real_scores = []
for tr_idx, te_idx in cv.split(X, y):
    pipe.fit(X[tr_idx], y[tr_idx])
    probs = pipe.predict_proba(X[te_idx])[:,1]
    real_scores.append(roc_auc_score(y[te_idx], probs))
real_auc = np.mean(real_scores)

# Phase destruction
X_phase = destroy_phase(X)
phase_scores = []
for tr_idx, te_idx in cv.split(X_phase, y):
    pipe.fit(X_phase[tr_idx], y[tr_idx])
    probs = pipe.predict_proba(X_phase[te_idx])[:,1]
    phase_scores.append(roc_auc_score(y[te_idx], probs))
phase_auc = np.mean(phase_scores)

# Temporal shuffle
X_temp = destroy_temporal(X)
temp_scores = []
for tr_idx, te_idx in cv.split(X_temp, y):
    pipe.fit(X_temp[tr_idx], y[tr_idx])
    probs = pipe.predict_proba(X_temp[te_idx])[:,1]
    temp_scores.append(roc_auc_score(y[te_idx], probs))
temp_auc = np.mean(temp_scores)

# Permutation
perm_scores = []
for _ in range(10):
    y_p = np.random.permutation(y)
    local = []
    for tr_idx, te_idx in cv.split(X, y_p):
        pipe.fit(X[tr_idx], y_p[tr_idx])
        probs = pipe.predict_proba(X[te_idx])[:,1]
        local.append(roc_auc_score(y_p[te_idx], probs))
    perm_scores.append(np.mean(local))
perm_auc = np.mean(perm_scores)

print(f"\nPHASE 93 RESULTS")
print(f"Original AUROC: {real_auc:.3f}")
print(f"Phase destruction: {phase_auc:.3f}")
print(f"Temporal shuffle: {temp_auc:.3f}")
print(f"Permutation: {perm_auc:.3f}")

effect = real_auc - perm_auc
verdict = "NO_CAUSAL_SIGNAL" if (phase_auc > 0.65 or temp_auc > 0.65) else "POSSIBLE_CAUSAL_SIGNAL"
print(f"Effect: {effect:.3f}")
print(f"Verdict: {verdict}")

results = {"real": real_auc, "phase_destroy": phase_auc, "temporal_shuffle": temp_auc, 
           "permutation": perm_auc, "effect": effect, "verdict": verdict}
with open(f"{OUTDIR}/results.json", "w") as f:
    json.dump(results, f, indent=2)

np.save(f"{OUTDIR}/X.npy", X)
np.save(f"{OUTDIR}/y.npy", y)