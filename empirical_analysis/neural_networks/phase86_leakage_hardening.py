import os
import json
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

BASE_DIR = "/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase86_leakage_hardening"
os.makedirs(BASE_DIR, exist_ok=True)

np.random.seed(42)

def load_csv(path):
    df = pd.read_csv(path)
    vals = df.select_dtypes(include=[np.number]).values.flatten()
    vals = vals[np.isfinite(vals)]
    return vals

def extract_features(x):
    if len(x) < 100:
        return None
    x = np.asarray(x)
    psd = np.abs(np.fft.rfft(x))**2
    psd = psd / (np.sum(psd) + 1e-12)
    feats = [
        np.var(x),
        np.mean(np.abs(np.diff(x))),
        np.corrcoef(x[:-1], x[1:])[0,1],
        np.corrcoef(x[:-5], x[5:])[0,1],
        entropy(psd)
    ]
    feats = np.nan_to_num(feats)
    return np.array(feats)

DATA_DIR = "/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase85_real_data_pipeline/data"
X = []
y = []

for label, cname in enumerate(["class0", "class1"]):
    cdir = os.path.join(DATA_DIR, cname)
    if not os.path.exists(cdir):
        continue
    for fname in os.listdir(cdir):
        if not fname.endswith(".csv"):
            continue
        try:
            path = os.path.join(cdir, fname)
            ts = load_csv(path)
            feat = extract_features(ts)
            if feat is not None:
                X.append(feat)
                y.append(label)
        except:
            pass

X = np.array(X)
y = np.array(y)
print("Samples:", len(X))

idx0 = np.where(y == 0)[0]
idx1 = np.where(y == 1)[0]
n = min(len(idx0), len(idx1))
idx = np.concatenate([idx0[:n], idx1[:n]])
X = X[idx]
y = y[idx]

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")

perm_scores = []
for i in range(100):
    yp = np.random.permutation(y)
    s = cross_val_score(pipeline, X, yp, cv=cv, scoring="roc_auc")
    perm_scores.append(np.mean(s))

perm_scores = np.array(perm_scores)

results = {
    "real_mean_auc": float(np.mean(scores)),
    "real_std_auc": float(np.std(scores)),
    "perm_mean_auc": float(np.mean(perm_scores)),
    "perm_std_auc": float(np.std(perm_scores)),
    "effect_size": float(np.mean(scores) - np.mean(perm_scores))
}

with open(os.path.join(BASE_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\nPHASE 86 RESULTS\n")
for k,v in results.items():
    print(k, v)

verdict = "POSSIBLE_REAL_SIGNAL" if (results["real_mean_auc"] > 0.75 and results["effect_size"] > 0.20) else "LIKELY_LEAKAGE_OR_TRIVIALITY"
print("\nVERDICT:", verdict)