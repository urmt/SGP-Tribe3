import os
import json
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

BASE_DIR = "/home/student/sgp-tribe3/empirical_analysis/neural_networks/phase87_out_of_distribution"
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
    psd = np.abs(np.fft.rfft(x))**2
    psd = psd / (np.sum(psd) + 1e-12)
    feats = [
        np.var(x),
        np.mean(np.abs(np.diff(x))),
        np.corrcoef(x[:-1], x[1:])[0,1],
        np.corrcoef(x[:-5], x[5:])[0,1],
        entropy(psd)
    ]
    return np.nan_to_num(feats)

DATA_DIR = os.path.join(BASE_DIR, "data")

def load_source(source_name):
    X = []
    y = []
    for label, cname in enumerate(["class0", "class1"]):
        cdir = os.path.join(DATA_DIR, source_name, cname)
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
    return np.array(X), np.array(y)

XA, yA = load_source("sourceA")
XB, yB = load_source("sourceB")

print("SourceA:", len(XA))
print("SourceB:", len(XB))

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42))
])
pipeline.fit(XA, yA)
probs_B = pipeline.predict_proba(XB)[:,1]
auc_A_to_B = roc_auc_score(yB, probs_B)

pipeline2 = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42))
])
pipeline2.fit(XB, yB)
probs_A = pipeline2.predict_proba(XA)[:,1]
auc_B_to_A = roc_auc_score(yA, probs_A)

results = {
    "A_to_B_AUROC": float(auc_A_to_B),
    "B_to_A_AUROC": float(auc_B_to_A),
    "mean_OOD_AUROC": float((auc_A_to_B + auc_B_to_A)/2)
}

with open(os.path.join(BASE_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\nPHASE 87 RESULTS\n")
for k,v in results.items():
    print(k, v)

verdict = "OOD_SIGNAL_SURVIVES" if results["mean_OOD_AUROC"] > 0.75 else "NO_TRUE_GENERALIZATION"
print("\nVERDICT:", verdict)