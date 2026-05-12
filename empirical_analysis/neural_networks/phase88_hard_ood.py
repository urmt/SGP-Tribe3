import os
import json
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

BASE_DIR = "/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase88_hard_ood"
os.makedirs(BASE_DIR, exist_ok=True)

np.random.seed(42)

def make_sourceA_class0(n=2000):
    x = np.zeros(n)
    for i in range(1,n):
        x[i] = 0.95*x[i-1] + np.random.randn()*0.2
    x += np.sin(np.linspace(0,20,n))
    return x

def make_sourceA_class1(n=2000):
    x = np.zeros(n)
    for i in range(1,n):
        x[i] = 0.6*x[i-1] + np.random.randn()*0.8
    x += np.sin(np.linspace(0,60,n))
    return x

def make_sourceB_class0(n=2000):
    x = np.random.laplace(size=n)
    mask = np.random.rand(n) < 0.03
    x[mask] += np.random.randn(np.sum(mask))*8
    return x

def make_sourceB_class1(n=2000):
    x = np.zeros(n)
    state = 0
    for i in range(n):
        if np.random.rand() < 0.02:
            state = 1 - state
        if state == 0:
            x[i] = np.random.randn()*0.2
        else:
            x[i] = np.random.randn()*3
    return x

def feats(x):
    psd = np.abs(np.fft.rfft(x))**2
    psd = psd / (np.sum(psd)+1e-12)
    f = [
        np.var(x),
        np.mean(np.abs(np.diff(x))),
        np.corrcoef(x[:-1],x[1:])[0,1],
        np.corrcoef(x[:-5],x[5:])[0,1],
        entropy(psd),
        np.max(np.abs(x)),
        np.std(np.diff(x))
    ]
    return np.nan_to_num(f)

XA, yA = [], []
XB, yB = [], []

for _ in range(100):
    XA.append(feats(make_sourceA_class0()))
    yA.append(0)
    XA.append(feats(make_sourceA_class1()))
    yA.append(1)
    XB.append(feats(make_sourceB_class0()))
    yB.append(0)
    XB.append(feats(make_sourceB_class1()))
    yB.append(1)

XA, XB = np.array(XA), np.array(XB)
yA, yB = np.array(yA), np.array(yB)

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

results = {"A_to_B": float(auc_A_B), "B_to_A": float(auc_B_A), "mean_hard_OOD": float((auc_A_B + auc_B_A)/2)}
with open(os.path.join(BASE_DIR,"results.json"), "w") as f:
    json.dump(results,f,indent=2)

print("\nPHASE 88 RESULTS\n")
for k,v in results.items():
    print(k,v)

verdict = "SIGNAL_SURVIVES_HARD_OOD" if results["mean_hard_OOD"] > 0.75 else "NO_REAL_GENERALIZATION"
print("\nVERDICT:", verdict)