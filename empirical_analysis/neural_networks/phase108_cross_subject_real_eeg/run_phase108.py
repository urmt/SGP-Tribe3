import os
import json
import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
import mne

BASE = "/home/student/sgp-tribe3"
DATA_DIR = os.path.join(
    BASE,
    "empirical_analysis/neural_networks/phase105_real_eeg_download/raw"
)

OUTDIR = os.path.join(
    BASE,
    "empirical_analysis/neural_networks/phase108_cross_subject_real_eeg"
)

os.makedirs(OUTDIR, exist_ok=True)

WINDOW_SEC = 4
SFREQ_TARGET = 128
MIN_WINDOWS = 20
THRESHOLD = 0.10

def extract_features(x, sf):
    f, p = welch(x, sf, nperseg=min(len(x), 256))
    p = np.nan_to_num(p)

    bands = {"delta": (1,4), "theta": (4,8), "alpha": (8,12), "beta": (12,30)}
    feats = []
    for lo, hi in bands.values():
        mask = (f >= lo) & (f < hi)
        feats.append(float(np.mean(p[mask])))

    feats.extend([
        float(np.mean(x)),
        float(np.std(x)),
        float(np.mean(np.abs(np.diff(x)))),
        float(np.corrcoef(x[:-1], x[1:])[0,1]) if len(x) > 2 else 0.0
    ])
    return np.nan_to_num(feats)

rows, groups, labels = [], [], []

edf_files = [os.path.join(root, f) for root, _, files in os.walk(DATA_DIR) for f in files if f.lower().endswith(".edf")]
print(f"Found {len(edf_files)} EDF files")

if len(edf_files) == 0:
    result = {"status": "NO_REAL_EEG_FOUND", "verdict": "NO_EFFECT"}
    with open(os.path.join(OUTDIR, "results.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(result)
    raise SystemExit

subject_id = 0

for path in edf_files:
    print(f"Processing: {os.path.basename(path)}")
    try:
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        raw.resample(SFREQ_TARGET)
        data = raw.get_data()

        if data.shape[0] == 0:
            print(f"  No data in {os.path.basename(path)}")
            continue

        sig = data[0]
        sig = np.nan_to_num(sig)
        win = WINDOW_SEC * SFREQ_TARGET

        local_count = 0
        local_rows, local_labels = [], []

        for start in range(0, len(sig)-win, win):
            seg = sig[start:start+win]
            if np.std(seg) < 1e-8:
                continue
            feat = extract_features(seg, SFREQ_TARGET)
            local_rows.append(feat)
            midpoint = (len(sig)-win)//2
            local_labels.append(0 if start < midpoint else 1)
            local_count += 1

        print(f"  Windows: {local_count}")

        if local_count >= MIN_WINDOWS:
            rows.extend(local_rows)
            groups.extend([subject_id] * local_count)
            labels.extend(local_labels)
            print(f"  Added as subject {subject_id}")
            subject_id += 1
        else:
            print(f"  Skipped - only {local_count} windows (< {MIN_WINDOWS})")

    except Exception as e:
        print(f"  FAILED: {e}")
        continue

X = np.array(rows)
y = np.array(labels)
g = np.array(groups)

valid_subjects = len(np.unique(g))
print(f"Total subjects: {valid_subjects}, Total windows: {len(X)}")

if valid_subjects < 2:
    result = {"status": "INSUFFICIENT_SUBJECTS", "subjects": int(valid_subjects), "verdict": "NO_EFFECT"}
    with open(os.path.join(OUTDIR, "results.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(result)
    raise SystemExit

logo = LeaveOneGroupOut()
aucs = []

for train_idx, test_idx in logo.split(X, y, g):
    Xtr, Xte = X[train_idx], X[test_idx]
    ytr, yte = y[train_idx], y[test_idx]
    if len(np.unique(ytr)) < 2:
        continue
    clf = RandomForestClassifier(n_estimators=200, random_state=0)
    clf.fit(Xtr, ytr)
    prob = clf.predict_proba(Xte)[:,1]
    auc = roc_auc_score(yte, prob)
    aucs.append(float(auc))

if len(aucs) == 0:
    result = {"status": "CV_FAILED", "verdict": "NO_EFFECT"}
    with open(os.path.join(OUTDIR, "results.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(result)
    raise SystemExit

real_auc = float(np.mean(aucs))

rng = np.random.default_rng(0)
perm_aucs = []

for _ in range(25):
    yp = rng.permutation(y)
    local = []
    for train_idx, test_idx in logo.split(X, yp, g):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = yp[train_idx], yp[test_idx]
        if len(np.unique(ytr)) < 2:
            continue
        clf = RandomForestClassifier(n_estimators=200, random_state=0)
        clf.fit(Xtr, ytr)
        prob = clf.predict_proba(Xte)[:,1]
        local.append(roc_auc_score(yte, prob))
    if len(local) > 0:
        perm_aucs.append(np.mean(local))

perm_auc = float(np.mean(perm_aucs))
effect = float(real_auc - perm_auc)
verdict = "REAL_TEMPORAL_SIGNAL" if effect >= THRESHOLD else "NO_ROBUST_SIGNAL"

result = {
    "subjects": int(valid_subjects),
    "windows": int(len(X)),
    "real_auc": real_auc,
    "perm_auc": perm_auc,
    "effect_size": effect,
    "threshold": THRESHOLD,
    "verdict": verdict
}

with open(os.path.join(OUTDIR, "results.json"), "w") as f:
    json.dump(result, f, indent=2)

pd.DataFrame(result, index=[0]).to_csv(os.path.join(OUTDIR, "summary.csv"), index=False)
print(json.dumps(result, indent=2))