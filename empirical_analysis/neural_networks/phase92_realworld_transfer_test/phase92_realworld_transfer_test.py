import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import entropy, skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import os
import json

np.random.seed(92)

OUTDIR = "empirical_analysis/neural_networks/phase92_realworld_transfer_test"
os.makedirs(OUTDIR, exist_ok=True)

def self_model_system(n=4000):
    x = np.zeros(n)
    pred = 0.0
    memory = 0.0

    for t in range(2, n):
        err = x[t-1] - pred
        memory = 0.97 * memory + 0.25 * err
        pred = 0.7 * x[t-1] + 0.2 * memory + 0.1 * np.sin(x[t-2])
        x[t] = 0.82 * x[t-1] - 0.08 * x[t-2] + 0.3 * pred + np.random.normal(0, 0.25)
    return x

def control_system(n=4000):
    x = np.zeros(n)
    for t in range(2, n):
        x[t] = 0.92 * x[t-1] - 0.15 * x[t-2] + np.random.normal(0, 0.25)
    return x

def extract_features(x):
    freqs, psd = welch(x, nperseg=256)
    ac1 = np.corrcoef(x[:-1], x[1:])[0,1]
    ac2 = np.corrcoef(x[:-2], x[2:])[0,1]
    dx = np.diff(x)
    return np.array([
        np.mean(x), np.std(x), skew(x), kurtosis(x),
        np.mean(np.abs(dx)), np.std(dx),
        ac1, ac2,
        entropy(np.histogram(x, bins=40)[0] + 1),
        np.mean(psd), np.std(psd),
        np.max(psd), freqs[np.argmax(psd)],
        np.sum(psd[:10]), np.sum(psd[10:30]), np.sum(psd[30:]),
        np.var(x), np.mean(np.abs(x))
    ])

X, y = [], []
for _ in range(50):
    s = self_model_system()
    c = control_system()
    X.append(extract_features(s)); y.append(1)
    X.append(extract_features(c)); y.append(0)

X, y = np.array(X), np.array(y)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=100, max_depth=4, random_state=92))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=92)

real_scores = []
for train_idx, test_idx in cv.split(X, y):
    Xtr, Xte = X[train_idx], X[test_idx]
    ytr, yte = y[train_idx], y[test_idx]
    pipe.fit(Xtr, ytr)
    probs = pipe.predict_proba(Xte)[:,1]
    auc = roc_auc_score(yte, probs)
    real_scores.append(auc)

real_auc = float(np.mean(real_scores))

perm_scores = []
for _ in range(50):
    yp = np.random.permutation(y)
    local = []
    for train_idx, test_idx in cv.split(X, yp):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = yp[train_idx], yp[test_idx]
        pipe.fit(Xtr, ytr)
        probs = pipe.predict_proba(Xte)[:,1]
        auc = roc_auc_score(yte, probs)
        local.append(auc)
    perm_scores.append(np.mean(local))

perm_auc = float(np.mean(perm_scores))
effect = real_auc - perm_auc

pipe.fit(X, y)
importances = pipe.named_steps["clf"].feature_importances_

results = {
    "real_auc": real_auc,
    "perm_auc": perm_auc,
    "effect_size": effect,
    "top_features": importances.tolist(),
    "verdict": "REAL_TRANSFERABLE_SIGNAL" if effect > 0.25 else "NO_REAL_SIGNAL"
}

with open(f"{OUTDIR}/results.json", "w") as f:
    json.dump(results, f, indent=2)

np.save(f"{OUTDIR}/X.npy", X)
np.save(f"{OUTDIR}/y.npy", y)

print("\nPHASE 92 RESULTS\n")
print("REAL AUROC:", round(real_auc,3))
print("PERM AUROC:", round(perm_auc,3))
print("EFFECT SIZE:", round(effect,3))
print("VERDICT:", results["verdict"])