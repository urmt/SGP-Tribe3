import os
import json
import random
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

np.random.seed(42)
random.seed(42)

BASE_DIR = "/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase91_adversarial_selfmodel_falsification"
os.makedirs(BASE_DIR, exist_ok=True)

def extract_features(x):
    x = np.asarray(x)
    mean = np.mean(x)
    std = np.std(x)
    sk = skew(x)
    kurt = kurtosis(x)
    ac1 = np.corrcoef(x[:-1], x[1:])[0,1]
    ac5 = np.corrcoef(x[:-5], x[5:])[0,1]
    freqs, psd = welch(x, nperseg=min(128, len(x)))
    spectral_entropy = -np.sum((psd / np.sum(psd)) * np.log((psd / np.sum(psd)) + 1e-12))
    peak_freq = freqs[np.argmax(psd)]
    return np.array([mean, std, sk, kurt, ac1, ac5, spectral_entropy, peak_freq])

def generic_ar(length=2000):
    x = np.zeros(length)
    for t in range(1, length):
        x[t] = 0.92 * x[t-1] + np.random.normal(0, 0.5)
    return x

def self_model_A(length=2000):
    x = np.zeros(length)
    pred = 0
    for t in range(2, length):
        pred = 0.7 * x[t-1] + 0.2 * (x[t-1] - x[t-2])
        error = x[t-1] - pred
        x[t] = pred + 0.6 * error + np.random.normal(0, 0.2)
    return x

def self_model_B(length=2000):
    x = np.zeros(length)
    memory = 0
    for t in range(2, length):
        uncertainty = abs(x[t-1] - x[t-2])
        gate = 1 / (1 + uncertainty)
        memory = gate * memory + (1 - gate) * x[t-1]
        x[t] = 0.75 * memory + 0.2 * x[t-1] + np.random.normal(0, 0.25)
    return x

def adversarial_control(reference):
    ref = np.asarray(reference)
    shuffled = np.random.permutation(ref)
    fft = np.fft.rfft(shuffled)
    mag = np.abs(np.fft.rfft(ref))
    randomized_phase = np.exp(1j * np.random.uniform(0, 2*np.pi, len(fft)))
    synth = np.fft.irfft(mag * randomized_phase)
    synth = (synth - np.mean(synth)) / (np.std(synth) + 1e-8)
    return synth

X, y, labels = [], [], []
N = 200

for _ in range(N):
    g = generic_ar()
    a = self_model_A()
    b = self_model_B()
    ac = adversarial_control(a)
    X.append(extract_features(g)); y.append(0); labels.append("generic")
    X.append(extract_features(a)); y.append(1); labels.append("selfA")
    X.append(extract_features(b)); y.append(1); labels.append("selfB")
    X.append(extract_features(ac)); y.append(0); labels.append("adv_control")

X = np.array(X)
y = np.array(y)
labels = np.array(labels)

train_mask = np.isin(labels, ["generic", "selfA"])
test_mask = np.isin(labels, ["selfB", "adv_control"])

X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

clf = RandomForestClassifier(n_estimators=300, random_state=42)
clf.fit(X_train, y_train)
probs = clf.predict_proba(X_test)[:,1]
auroc = roc_auc_score(y_test, probs)

perm_scores = []
for _ in range(100):
    y_perm = np.random.permutation(y_train)
    clf_perm = RandomForestClassifier(n_estimators=300, random_state=42)
    clf_perm.fit(X_train, y_perm)
    p = clf_perm.predict_proba(X_test)[:,1]
    perm_scores.append(roc_auc_score(y_test, p))

perm_mean = float(np.mean(perm_scores))

feature_names = ["mean", "std", "skew", "kurtosis", "ac1", "ac5", "spectral_entropy", "peak_freq"]
importance = permutation_importance(clf, X_test, y_test, n_repeats=20, random_state=42)
feature_importance = {feature_names[i]: float(v) for i, v in enumerate(importance.importances_mean)}

effect_size = auroc - perm_mean
verdict = "ROBUST_SELF_MODEL_SIGNAL" if (auroc > 0.85 and effect_size > 0.25) else "FAILED_UNDER_ADVERSARIAL_CONTROL"

results = {"auroc": float(auroc), "perm_mean": perm_mean, "effect_size": float(effect_size), "feature_importance": feature_importance, "verdict": verdict}
with open(os.path.join(BASE_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

np.save(os.path.join(BASE_DIR, "X.npy"), X)
np.save(os.path.join(BASE_DIR, "y.npy"), y)

print("\n==============================")
print("PHASE 91 RESULTS")
print("==============================")
print(f"AUROC: {auroc:.3f}")
print(f"Permutation Mean: {perm_mean:.3f}")
print(f"Effect Size: {effect_size:.3f}")
print(f"Verdict: {verdict}")
print("==============================\n")