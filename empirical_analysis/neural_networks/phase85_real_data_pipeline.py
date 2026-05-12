import os
import json
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

BASE_DIR = "/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase85_real_data_pipeline"
os.makedirs(BASE_DIR, exist_ok=True)

np.random.seed(42)

def load_real_timeseries_csv(path):
    df = pd.read_csv(path)
    values = df.select_dtypes(include=[np.number]).values.flatten()
    values = values[np.isfinite(values)]
    return values

def extract_features(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) < 100:
        return None

    psd = np.abs(np.fft.rfft(x))**2
    psd = psd / (np.sum(psd) + 1e-12)

    autocorr1 = np.corrcoef(x[:-1], x[1:])[0,1]
    autocorr5 = np.corrcoef(x[:-5], x[5:])[0,1]
    variance = np.var(x)
    mean_abs_diff = np.mean(np.abs(np.diff(x)))
    spectral_entropy = entropy(psd)

    return np.array([variance, autocorr1, autocorr5, mean_abs_diff, spectral_entropy])

DATA_DIR = os.path.join(BASE_DIR, "data")
class0_dir = os.path.join(DATA_DIR, "class0")
class1_dir = os.path.join(DATA_DIR, "class1")

X = []
y = []

for fname in os.listdir(class0_dir):
    if not fname.endswith(".csv"):
        continue
    path = os.path.join(class0_dir, fname)
    try:
        ts = load_real_timeseries_csv(path)
        feat = extract_features(ts)
        if feat is not None:
            X.append(feat)
            y.append(0)
    except:
        pass

for fname in os.listdir(class1_dir):
    if not fname.endswith(".csv"):
        continue
    path = os.path.join(class1_dir, fname)
    try:
        ts = load_real_timeseries_csv(path)
        feat = extract_features(ts)
        if feat is not None:
            X.append(feat)
            y.append(1)
    except:
        pass

X = np.array(X)
y = np.array(y)

print("Loaded samples:", len(X))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
aurocs = []

for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42)
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:,1]
    auroc = roc_auc_score(y_test, probs)
    aurocs.append(auroc)

perm_scores = []
for _ in range(50):
    y_perm = np.random.permutation(y)
    clf = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42)
    clf.fit(X, y_perm)
    probs = clf.predict_proba(X)[:,1]
    perm_scores.append(roc_auc_score(y_perm, probs))

results = {
    "mean_auroc": float(np.mean(aurocs)),
    "std_auroc": float(np.std(aurocs)),
    "perm_mean": float(np.mean(perm_scores)),
    "perm_std": float(np.std(perm_scores)),
    "n_samples": int(len(X))
}

with open(os.path.join(BASE_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\nPHASE 85 RESULTS\n")
for k,v in results.items():
    print(k, v)

verdict = "REAL_SIGNAL_PRESENT" if (results["mean_auroc"] > 0.75 and results["perm_mean"] < 0.60) else "NO_ROBUST_SIGNAL"
print("\nVERDICT:", verdict)