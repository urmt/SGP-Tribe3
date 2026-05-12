import os
import json
import numpy as np
from scipy.stats import entropy
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

BASE_DIR = "/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase89_self_model_coupling"
os.makedirs(BASE_DIR, exist_ok=True)

np.random.seed(42)

def generic_recursive(n=2000):
    x = np.zeros(n)
    for i in range(1,n):
        x[i] = 0.85*x[i-1] + np.random.randn()*0.5
    return x

def coupled_self_model(n=2000):
    x = np.zeros(n)
    internal = 0
    prediction = 0
    env = np.random.randn(n)
    for i in range(1,n):
        error = env[i-1] - prediction
        internal = 0.97*internal + 0.25*error
        prediction = 0.9*prediction + 0.5*internal
        x[i] = prediction + np.random.randn()*0.1
    return x

def features(x):
    psd = np.abs(np.fft.rfft(x))**2
    psd = psd / (np.sum(psd)+1e-12)
    dx = np.diff(x)
    feats = [
        np.var(x),
        np.var(dx),
        np.mean(np.abs(dx)),
        np.corrcoef(x[:-1],x[1:])[0,1],
        np.corrcoef(x[:-10],x[10:])[0,1],
        entropy(psd),
        np.max(np.abs(x)),
        np.std(x)
    ]
    return np.nan_to_num(feats)

X, y = [], []
for _ in range(300):
    X.append(features(generic_recursive()))
    y.append(0)
    X.append(features(coupled_self_model()))
    y.append(1)

X, y = np.array(X), np.array(y)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42))
])
pipe.fit(Xtr, ytr)
probs = pipe.predict_proba(Xte)[:,1]
auc = roc_auc_score(yte, probs)

perm_scores = []
for _ in range(100):
    yp = np.random.permutation(y)
    Xtr, Xte, ytr, yte = train_test_split(X, yp, test_size=0.3, stratify=yp, random_state=42)
    pipe2 = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42))
    ])
    pipe2.fit(Xtr, ytr)
    p = pipe2.predict_proba(Xte)[:,1]
    perm_scores.append(roc_auc_score(yte, p))

perm_scores = np.array(perm_scores)

results = {
    "real_auc": float(auc),
    "perm_mean": float(np.mean(perm_scores)),
    "effect_size": float(auc - np.mean(perm_scores))
}

with open(os.path.join(BASE_DIR,"results.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\nPHASE 89 RESULTS\n")
for k,v in results.items():
    print(k,v)

verdict = "SELF_MODEL_SIGNAL_PRESENT" if (results["real_auc"] > 0.80 and results["effect_size"] > 0.25) else "NO_SELF_MODEL_SIGNAL"
print("\nVERDICT:", verdict)