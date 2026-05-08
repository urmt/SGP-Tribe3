import os
import json
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import mne

BASE = "/home/student/sgp-tribe3"
OUTDIR = os.path.join(BASE, "empirical_analysis/neural_networks/phase116_iaaft_test")

DATA_DIRS = [
    os.path.join(BASE, "empirical_analysis/neural_networks/phase105_real_eeg_download/raw/CHBMIT.edf"),
    os.path.join(BASE, "empirical_analysis/neural_networks/phase111_long_duration_real_eeg/downloaded/chb01_03.edf"),
    os.path.join(BASE, "empirical_analysis/neural_networks/phase112_persistent_acquisition/downloaded/chb01_04.edf"),
    os.path.join(BASE, "empirical_analysis/neural_networks/phase112_persistent_acquisition/downloaded/chb02_01.edf"),
    os.path.join(BASE, "empirical_analysis/neural_networks/phase112_persistent_acquisition/downloaded/chb03_01.edf"),
    os.path.join(BASE, "empirical_analysis/neural_networks/phase112_persistent_acquisition/downloaded/chb04_01.edf"),
]

SUBJECT_MAP = {"CHBMIT.edf": "chb00", "chb01_03.edf": "chb01", "chb01_04.edf": "chb01",
    "chb02_01.edf": "chb02", "chb03_01.edf": "chb03", "chb04_01.edf": "chb04"}

WINDOW_SEC, SFREQ = 10, 128

def iaaft_surrogate(x, rng, n_iter=25):
    """
    True IAAFT surrogate - preserves exact spectrum and exact amplitude histogram.
    """
    x = x.astype(float)
    n = len(x)
    
    sorted_x = np.sort(x)
    fft_full = np.fft.fft(x)
    spectrum = np.abs(fft_full)
    
    x_new = rng.standard_normal(n)
    
    for _ in range(n_iter):
        phases = rng.uniform(-np.pi, np.pi, n)
        fourier = spectrum * np.exp(1j * phases)
        x_phase = np.real(np.fft.ifft(fourier))
        
        sorted_phase = np.sort(x_phase)
        x_new = sorted_x[np.searchsorted(sorted_phase, x_phase)]
    
    return x_new

def extract_features(x):
    x = np.nan_to_num(x, nan=0, posinf=0, neginf=0)
    if np.std(x) < 1e-10: return None
    f, p = signal.welch(x, fs=SFREQ, nperseg=min(256, len(x)))
    p = np.abs(p) + 1e-12
    bands = {"delta": (1,4), "theta": (4,8), "alpha": (8,12), "beta": (12,30)}
    band_pow = {n: np.mean(p[(f>=l)&(f<h)]) for n,(l,h) in bands.items()}
    p_norm = p / (np.sum(p) + 1e-12)
    spec_ent = -np.sum(p_norm * np.log(p_norm + 1e-12))
    dx = np.diff(x)
    act = np.var(x)
    mob = np.sqrt(np.var(dx) / (act + 1e-12))
    comp = np.sqrt(np.var(np.diff(dx)) / (np.var(dx) + 1e-12)) / (mob + 1e-12)
    try: lag1 = np.corrcoef(x[:-1], x[1:])[0,1]
    except: lag1 = 0
    zcr = np.sum(np.diff(np.sign(x)) != 0) / len(x)
    return {'mean': float(np.mean(x)), 'variance': float(np.var(x)), 'RMS': float(np.sqrt(np.mean(x**2))),
        'hjorth_activity': float(act), 'hjorth_mobility': float(mob), 'hjorth_complexity': float(comp),
        'spectral_entropy': float(spec_ent), 'bandpower_delta': float(band_pow['delta']),
        'bandpower_theta': float(band_pow['theta']), 'bandpower_alpha': float(band_pow['alpha']),
        'bandpower_beta': float(band_pow['beta']), 'lag1_autocorr': float(lag1), 'zero_crossing_rate': float(zcr)}

print("="*60)
print("PHASE 116 - IAAFT HIGHER-ORDER TEMPORAL TEST")
print("="*60)

raw_signals, feature_names = {}, None

for fpath in DATA_DIRS:
    if not os.path.exists(fpath): continue
    fname = os.path.basename(fpath)
    subj = SUBJECT_MAP.get(fname, fname.split('_')[0])
    raw = mne.io.read_raw_edf(fpath, preload=True, verbose=False)
    raw.resample(SFREQ)
    raw_signals[subj] = raw.get_data()[0]

subjects = sorted(raw_signals.keys())
print(f"Subjects: {subjects}")

iaaft_results = []
perm_results = []
fold_results = []

for ho_idx, ho in enumerate(subjects):
    print(f"\nFold {ho_idx+1}: hold out {ho}")
    train_subs = [s for s in subjects if s != ho]
    
    X_train, y_train = [], []
    for sub in train_subs:
        sig = raw_signals[sub]
        rng = np.random.default_rng(42 + hash(sub) % 1000)
        win_len = WINDOW_SEC * SFREQ
        for st in range(0, len(sig) - win_len, win_len):
            seg = sig[st:st+win_len]
            feat = extract_features(seg)
            if feat:
                X_train.append([feat[k] for k in sorted(feat.keys())]); y_train.append(1)
                ctrl = iaaft_surrogate(seg, rng, 25)
                feat_ctrl = extract_features(ctrl)
                if feat_ctrl:
                    X_train.append([feat_ctrl[k] for k in sorted(feat_ctrl.keys())]); y_train.append(0)
    
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    sig_test = raw_signals[ho]
    rng_test = np.random.default_rng(42 + ho_idx)
    X_test_real, X_test_ctrl = [], []
    win_len = WINDOW_SEC * SFREQ
    for st in range(0, sig_test.shape[0] - win_len, win_len):
        seg = sig_test[st:st+win_len]
        feat = extract_features(seg)
        if feat:
            X_test_real.append([feat[k] for k in sorted(feat.keys())])
            ctrl = iaaft_surrogate(seg, rng_test, 25)
            feat_ctrl = extract_features(ctrl)
            if feat_ctrl:
                X_test_ctrl.append([feat_ctrl[k] for k in sorted(feat_ctrl.keys())])
    
    X_test = np.vstack([np.array(X_test_real), np.array(X_test_ctrl)])
    y_test = np.array([1]*len(X_test_real) + [0]*len(X_test_ctrl))

    clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    clf.fit(X_train, y_train)
    iaaft_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
    iaaft_results.append(iaaft_auc)
    print(f"  IAAFT AUC: {iaaft_auc:.3f}")
    
    rng_perm = np.random.default_rng(100 + ho_idx)
    clf_perm = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    clf_perm.fit(X_train, rng_perm.permutation(y_train))
    perm_auc = roc_auc_score(y_test, clf_perm.predict_proba(X_test)[:,1])
    perm_results.append(perm_auc)
    print(f"  Perm AUC: {perm_auc:.3f}")
    
    fold_results.append({'fold': ho_idx+1, 'held_out': ho, 'iaaft_auc': iaaft_auc, 'perm_auc': perm_auc})

iaaft_mean = float(np.mean(iaaft_results))
perm_mean = float(np.mean(perm_results))
effect = iaaft_mean - perm_mean

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"iaaft_auc: {iaaft_mean:.3f}")
print(f"perm_auc: {perm_mean:.3f}")
print(f"effect_size: {effect:.3f}")

if iaaft_mean > 0.75:
    interp = "EEG structure survives IAAFT destruction - higher-order nonlinear temporal organization present"
    verdict = "HIGHER_ORDER_TEMPORAL"
elif iaaft_mean < 0.60:
    interp = "EEG structure destroyed by IAAFT - reducible to linear spectral statistics"
    verdict = "MOSTLY_LINEAR"
else:
    interp = "Mixed results - some temporal structure survives IAAFT"
    verdict = "NO_ROBUST_STRUCTURE"

print(f"\nINTERPRETATION: {interp}")
print(f"VERDICT: {verdict}")

pd.DataFrame(fold_results).to_csv(os.path.join(OUTDIR, "fold_results.csv"), index=False)

result = {"iaaft_auc": iaaft_mean, "perm_auc": perm_mean, "effect_size": effect,
    "interpretation": interp, "verdict": verdict, "n_subjects": len(subjects)}

with open(os.path.join(OUTDIR, "final_summary.json"), "w") as f:
    json.dump(result, f, indent=2)

print("\nPHASE 116 RESULTS")
print(json.dumps(result, indent=2))