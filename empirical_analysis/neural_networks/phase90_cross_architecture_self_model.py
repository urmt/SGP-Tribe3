import os
import json
import numpy as np
from scipy.stats import entropy
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

BASE_DIR = "/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase90_cross_architecture_self_model"
os.makedirs(BASE_DIR, exist_ok=True)

np.random.seed(42)

def archA_self_model(n=2000):
    x = np.zeros(n)
    internal = 0
    pred = 0
    env = np.random.randn(n)
    for i in range(1,n):
        err = env[i-1] - pred
        internal = 0.96*internal + 0.30*err
        pred = 0.85*pred + 0.50*internal
        x[i] = pred + np.random.randn()*0.1
    return x

def archB_self_model(n=2000):
    x = np.zeros(n)
    memory = 0
    gate = 0.5
    env = np.random.randn(n)
    for i in range(1,n):
        mismatch = np.abs(env[i-1] - memory)
        gate = 1/(1 + np.exp(-mismatch))
        memory = gate*memory + (1-gate)*env[i-1]
        x[i] = memory + np.random.randn()*0.1
    return x

def generic_recursive(n=2000):
    x = np.zeros(n)
    for i in range(1,n):
        x[i] = 0.85*x[i-1] + np.random.randn()*0.5
    return x

def feats(x):
    psd = np.abs(np.fft.rfft(x))**2
    psd = psd / (np.sum(psd)+1e-12)
    dx = np.diff(x)
    f = [
        np.var(x), np.var(dx), np.mean(np.abs(dx)),
        np.corrcoef(x[:-1],x[1:])[0,1],
        np.corrcoef(x[:-10],x[10:])[0,1],
        entropy(psd), np.max(np.abs(x)), np.std(x)
    ]
    return np.nan_to_num(f)

XA, yA = [], []
for _ in range(300):
    XA.append(feats(generic_recursive()))
    yA.append(0)
    XA.append(feats(archA_self_model()))
    yA.append(1)
XA, yA = np.array(XA), np.array(yA)

XB, yB = [], []
for _ in range(300):
    XB.append(feats(generic_recursive()))
    yB.append(0)
    XB.append(feats(archB_self_model()))
    yB.append(1)
XB, yB = np.array(XB), np.array(yB)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42))
])
pipe.fit(XA, yA)
pB = pipe.predict_proba(XB)[:,1]
auc_A_B = roc_auc_score(yB, pB)

pipe2 = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42))
])
pipe2.fit(XB, yB)
pA = pipe2.predict_proba(XA)[:,1]
auc_B_A = roc_auc_score(yA, pA)

results = {"A_to_B": float(auc_A_B), "B_to_A": float(auc_B_A), "mean_cross_architecture": float((auc_A_B + auc_B_A)/2)}
with open(os.path.join(BASE_DIR,"results.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\nPHASE 90 RESULTS\n")
for k,v in results.items():
    print(k,v)

verdict = "TRANSFERABLE_SELF_MODEL_STRUCTURE" if results["mean_cross_architecture"] > 0.75 else "ARCHITECTURE_SPECIFIC_ONLY"
print("\nVERDICT:", verdict)