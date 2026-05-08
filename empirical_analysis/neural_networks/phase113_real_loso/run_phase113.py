import os
import json
import numpy as np
import pandas as pd
from scipy import stats, signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
import mne

BASE = "/home/student/sgp-tribe3"
OUTDIR = os.path.join(BASE, "empirical_analysis/neural_networks/phase113_real_loso")

DATA_DIRS = [
    os.path.join(BASE, "empirical_analysis/neural_networks/phase105_real_eeg_download/raw/CHBMIT.edf"),
    os.path.join(BASE, "empirical_analysis/neural_networks/phase111_long_duration_real_eeg/downloaded/chb01_03.edf"),
    os.path.join(BASE, "empirical_analysis/neural_networks/phase112_persistent_acquisition/downloaded/chb01_04.edf"),
    os.path.join(BASE, "empirical_analysis/neural_networks/phase112_persistent_acquisition/downloaded/chb02_01.edf"),
    os.path.join(BASE, "empirical_analysis/neural_networks/phase112_persistent_acquisition/downloaded/chb03_01.edf"),
    os.path.join(BASE, "empirical_analysis/neural_networks/phase112_persistent_acquisition/downloaded/chb04_01.edf"),
]

SUBJECT_MAP = {
    "CHBMIT.edf": "chb00",
    "chb01_03.edf": "chb01",
    "chb01_04.edf": "chb01",
    "chb02_01.edf": "chb02",
    "chb03_01.edf": "chb03",
    "chb04_01.edf": "chb04",
}

WINDOW_SEC = 10
SFREQ = 128

def extract_features(x):
    x = np.nan_to_num(x, nan=0, posinf=0, neginf=0)
    if np.std(x) < 1e-10:
        return None

    f, p = signal.welch(x, fs=SFREQ, nperseg=min(256, len(x)))
    p = np.abs(p) + 1e-12

    bands = {"delta": (1,4), "theta": (4,8), "alpha": (8,12), "beta": (12,30)}
    band_pow = {name: np.mean(p[(f>=lo) & (f<hi)]) for name, (lo,hi) in bands.items()}

    p_norm = p / (np.sum(p) + 1e-12)
    spec_ent = -np.sum(p_norm * np.log(p_norm + 1e-12))

    dx = np.diff(x)
    act = np.var(x)
    mob = np.sqrt(np.var(dx) / (act + 1e-12))
    comp = np.sqrt(np.var(np.diff(dx)) / (np.var(dx) + 1e-12)) / (mob + 1e-12)

    try:
        lag1 = np.corrcoef(x[:-1], x[1:])[0,1]
    except:
        lag1 = 0

    zcr = np.sum(np.diff(np.sign(x)) != 0) / len(x)

    return {
        'mean': float(np.mean(x)),
        'variance': float(np.var(x)),
        'RMS': float(np.sqrt(np.mean(x**2))),
        'hjorth_activity': float(act),
        'hjorth_mobility': float(mob),
        'hjorth_complexity': float(comp),
        'spectral_entropy': float(spec_ent),
        'bandpower_delta': float(band_pow['delta']),
        'bandpower_theta': float(band_pow['theta']),
        'bandpower_alpha': float(band_pow['alpha']),
        'bandpower_beta': float(band_pow['beta']),
        'lag1_autocorr': float(lag1),
        'zero_crossing_rate': float(zcr),
    }

print("="*60)
print("PHASE 113 - STRICT REAL EEG LOSO VALIDATION")
print("="*60)

real_windows = []
real_subjects = []
feature_names = None

for fpath in DATA_DIRS:
    if not os.path.exists(fpath):
        continue

    fname = os.path.basename(fpath)
    subject = SUBJECT_MAP.get(fname, fname.split('_')[0])

    raw = mne.io.read_raw_edf(fpath, preload=False, verbose=False)
    raw.resample(SFREQ)
    sig = raw.get_data()[0]

    win_len = WINDOW_SEC * SFREQ
    for start in range(0, len(sig) - win_len, win_len):
        seg = sig[start:start+win_len]
        feat = extract_features(seg)
        if feat is not None:
            if feature_names is None:
                feature_names = sorted(feat.keys())
            real_windows.append([feat[k] for k in feature_names])
            real_subjects.append(subject)

print(f"Real windows: {len(real_windows)}")
print(f"Unique subjects: {len(set(real_subjects))}")

real_windows = np.array(real_windows)
real_subjects = np.array(real_subjects)

rng = np.random.default_rng(42)
control_windows = []
for w in real_windows:
    ctrl = w.copy()
    ctrl = rng.permutation(ctrl)
    control_windows.append(ctrl)
control_windows = np.array(control_windows)

X = np.vstack([real_windows, control_windows])
y = np.array([1] * len(real_windows) + [0] * len(control_windows))
g = np.concatenate([real_subjects, real_subjects])

print(f"Total samples: {len(X)} (real={np.sum(y==1)}, ctrl={np.sum(y==0)})")

logo = LeaveOneGroupOut()
real_aucs = []
perm_aucs = []
shuffle_aucs = []
phase_aucs = []
fold_results = []

subjects = sorted(set(g))
print(f"Running LOSO CV with {len(subjects)} subjects...")

for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, g)):
    Xtr, Xte = X[train_idx], X[test_idx]
    ytr, yte = y[train_idx], y[test_idx]

    held_out = g[test_idx[0]]
    print(f"Fold {fold_idx+1}: hold out subject {held_out}")

    if len(np.unique(ytr)) < 2:
        print(f"  Skipping: only one class in training")
        continue

    clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    clf.fit(Xtr, ytr)
    prob = clf.predict_proba(Xte)[:,1]
    auc = roc_auc_score(yte, prob)
    real_aucs.append(auc)
    print(f"  Real AUC: {auc:.3f}")

    rng = np.random.default_rng(42 + fold_idx)

    yp = rng.permutation(ytr)
    clf_p = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    clf_p.fit(Xtr, yp)
    auc_p = roc_auc_score(yte, clf_p.predict_proba(Xte)[:,1])
    perm_aucs.append(auc_p)
    print(f"  Perm AUC: {auc_p:.3f}")

    Xshuf = Xtr.copy()
    for i in range(Xshuf.shape[1]):
        Xshuf[:, i] = rng.permutation(Xshuf[:, i])
    clf_s = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    clf_s.fit(Xshuf, ytr)
    auc_s = roc_auc_score(yte, clf_s.predict_proba(Xte)[:,1])
    shuffle_aucs.append(auc_s)
    print(f"  Shuffle AUC: {auc_s:.3f}")

    Xph = Xtr.copy()
    for i in range(Xph.shape[1]):
        x = Xph[:, i]
        fft = np.fft.fft(x)
        phase = rng.uniform(-np.pi, np.pi, len(x))
        Xph[:, i] = np.real(np.fft.ifft(fft * np.exp(1j * phase)))
    clf_ph = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    clf_ph.fit(Xph, ytr)
    auc_ph = roc_auc_score(yte, clf_ph.predict_proba(Xte)[:,1])
    phase_aucs.append(auc_ph)
    print(f"  Phase AUC: {auc_ph:.3f}")

    fold_results.append({
        'fold': fold_idx,
        'held_out_subject': held_out,
        'real_auc': auc,
        'perm_auc': auc_p,
        'shuffle_auc': auc_s,
        'phase_auc': auc_ph
    })

real_auc = float(np.mean(real_aucs))
perm_auc = float(np.mean(perm_aucs))
shuffle_auc = float(np.mean(shuffle_aucs))
phase_auc = float(np.mean(phase_aucs))
effect = real_auc - perm_auc

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"Real AUC: {real_auc:.3f}")
print(f"Permutation AUC: {perm_auc:.3f}")
print(f"Shuffle AUC: {shuffle_auc:.3f}")
print(f"Phase AUC: {phase_auc:.3f}")
print(f"Effect size: {effect:.3f}")

verdict = "ROBUST_REAL_STRUCTURE" if (real_auc > 0.75 and perm_auc < 0.60 and effect > 0.15) else "NO_ROBUST_REAL_STRUCTURE"
print(f"VERDICT: {verdict}")

pd.DataFrame(fold_results).to_csv(os.path.join(OUTDIR, "fold_metrics.csv"), index=False)

result = {
    "real_auc": real_auc,
    "perm_auc": perm_auc,
    "shuffle_auc": shuffle_auc,
    "phase_auc": phase_auc,
    "effect_size": effect,
    "verdict": verdict,
    "n_subjects": len(set(real_subjects)),
    "n_windows": len(real_windows)
}

with open(os.path.join(OUTDIR, "final_summary.json"), "w") as f:
    json.dump(result, f, indent=2)

print("\nPHASE 113 RESULTS")
print(json.dumps(result, indent=2))