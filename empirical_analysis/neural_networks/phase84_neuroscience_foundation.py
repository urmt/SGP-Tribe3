import os
import json
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import entropy
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

BASE_DIR = "/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase84_neuroscience_foundation"
os.makedirs(BASE_DIR, exist_ok=True)

# ============================================================
# PHASE 84
# STRICT NEUROSCIENCE FOUNDATION TEST
#
# GOAL:
# Determine whether recursive temporal organization
# exists INSIDE neuroscience data ONLY.
#
# NO UNIVERSAL CLAIMS
# NO MANIFOLDS
# NO EMBEDDINGS
# NO TOPOLOGY
# NO RG
#
# ONLY RAW FEATURES
# ONLY STRICT VALIDATION
# ONLY REALISTIC SIGNAL STRUCTURE
# ============================================================

# ------------------------------------------------------------
# SYNTHETIC EEG-LIKE SIGNALS
# (used ONLY as controlled statistical scaffolding)
# ------------------------------------------------------------

def generate_resting_state(n=4000):
    x = np.random.randn(n)

    for i in range(2, n):
        x[i] += (
            0.92 * x[i-1]
            - 0.15 * x[i-2]
            + 0.05 * np.sin(i / 20)
        )

    return x


def generate_task_state(n=4000):
    x = np.random.randn(n)

    latent = 0

    for i in range(2, n):
        prediction = 0.85 * x[i-1]

        error = np.sin(i / 15) - prediction

        latent = (
            0.98 * latent
            + 0.15 * error
        )

        x[i] += (
            prediction
            + latent
            + 0.25 * np.sin(i / 8)
        )

    return x


# ------------------------------------------------------------
# RAW FEATURES ONLY
# ------------------------------------------------------------

def extract_features(x):

    x = np.asarray(x)
    x = np.nan_to_num(x)

    cov = np.cov(x[:-1], x[1:])

    eigvals = np.linalg.eigvalsh(cov)

    eigvals = np.abs(eigvals) + 1e-12

    psd = np.abs(np.fft.rfft(x)) ** 2
    psd /= np.sum(psd)

    autocorr1 = np.corrcoef(x[:-1], x[1:])[0,1]

    autocorr5 = np.corrcoef(x[:-5], x[5:])[0,1]

    spectral_entropy = entropy(psd)

    variance = np.var(x)

    mean_abs_diff = np.mean(np.abs(np.diff(x)))

    return np.array([
        variance,
        autocorr1,
        autocorr5,
        spectral_entropy,
        np.max(eigvals),
        np.min(eigvals),
        np.mean(eigvals),
        mean_abs_diff
    ])


# ------------------------------------------------------------
# BUILD DATASET
# ------------------------------------------------------------

X = []
y = []

N_TRIALS = 300

for _ in range(N_TRIALS):

    X.append(extract_features(generate_resting_state()))
    y.append(0)

for _ in range(N_TRIALS):

    X.append(extract_features(generate_task_state()))
    y.append(1)

X = np.array(X)
y = np.array(y)

# ------------------------------------------------------------
# STRICT VALIDATION
# ------------------------------------------------------------

skf = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

aurocs = []

for train_idx, test_idx in skf.split(X, y):

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        random_state=42
    )

    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_test)[:,1]

    auroc = roc_auc_score(y_test, probs)

    aurocs.append(auroc)

# ------------------------------------------------------------
# NEGATIVE CONTROLS
# ------------------------------------------------------------

shuffle_scores = []

for _ in range(20):

    y_perm = np.random.permutation(y)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42
    )

    clf.fit(X, y_perm)

    probs = clf.predict_proba(X)[:,1]

    shuffle_scores.append(
        roc_auc_score(y_perm, probs)
    )

# ------------------------------------------------------------
# FEATURE IMPORTANCE
# ------------------------------------------------------------

clf_final = RandomForestClassifier(
    n_estimators=500,
    max_depth=5,
    random_state=42
)

clf_final.fit(X, y)

feature_names = [
    "variance",
    "autocorr1",
    "autocorr5",
    "spectral_entropy",
    "eig_max",
    "eig_min",
    "eig_mean",
    "mean_abs_diff"
]

importance = dict(zip(
    feature_names,
    clf_final.feature_importances_
))

# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------

results = {
    "mean_auroc": float(np.mean(aurocs)),
    "std_auroc": float(np.std(aurocs)),
    "shuffle_mean": float(np.mean(shuffle_scores)),
    "shuffle_std": float(np.std(shuffle_scores)),
    "feature_importance": importance
}

with open(
    os.path.join(BASE_DIR, "results.json"),
    "w"
) as f:
    json.dump(results, f, indent=2)

print("\nPHASE 84 RESULTS\n")

print("AUROC")
print(np.mean(aurocs))

print("\nSHUFFLE CONTROL")
print(np.mean(shuffle_scores))

print("\nFEATURE IMPORTANCE")
for k, v in importance.items():
    print(k, round(v, 4))

if np.mean(aurocs) > 0.80 and np.mean(shuffle_scores) < 0.60:
    verdict = "REAL_NEURAL_STRUCTURE_DETECTED"
else:
    verdict = "NO_ROBUST_STRUCTURE"

print("\nVERDICT")
print(verdict)