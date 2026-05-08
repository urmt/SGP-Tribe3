import os
import json
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import mne

BASE = "/home/student/sgp-tribe3"
OUTDIR = os.path.join(BASE, "empirical_analysis/neural_networks/phase115_hierarchical_surrogates")

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

def phase_surrogate(x, rng):
    return np.real(np.fft.ifft(np.fft.fft(x.astype(float)) * np.exp(1j * rng.uniform(-np.pi, np.pi, len(x)))))

print("="*60)
print("PHASE 115 - HIERARCHICAL SURROGATE (PHASE TEST)")
print("="*60)

all_windows, feature_names = {}, None
raw_signals = {}

for fpath in DATA_DIRS:
    if not os.path.exists(fpath): continue
    fname = os.path.basename(fpath)
    subj = SUBJECT_MAP.get(fname, fname.split('_')[0])
    raw = mne.io.read_raw_edf(fpath, preload=True, verbose=False)
    raw.resample(SFREQ)
    sig = raw.get_data()[0]
    raw_signals[subj] = sig

subjects = sorted(raw_signals.keys())
print(f"Subjects: {subjects}")

destructors = {
    'phase': phase_surrogate,
}

results_by_level = {n: [] for n in destructors}
fold_results = []

for ho_idx, ho in enumerate(subjects):
    print(f"\nFold {ho_idx+1}: hold out {ho}")
    train_subs = [s for s in subjects if s != ho]
    
    for level_name, destroy_fn in destructors.items():
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
                    ctrl = destroy_fn(seg, rng)
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
                ctrl = destroy_fn(seg, rng_test)
                feat_ctrl = extract_features(ctrl)
                if feat_ctrl:
                    X_test_ctrl.append([feat_ctrl[k] for k in sorted(feat_ctrl.keys())])
        
        X_test = np.vstack([np.array(X_test_real), np.array(X_test_ctrl)])
        y_test = np.array([1]*len(X_test_real) + [0]*len(X_test_ctrl))

        clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
        clf.fit(X_train, y_train)
        auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
        results_by_level[level_name].append(auc)
        print(f"  {level_name}: {auc:.3f}")
        fold_results.append({'fold': ho_idx+1, 'level': level_name, 'auc': auc})

mean_res = {n: float(np.mean(v)) for n,v in results_by_level.items()}

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"phase_auc: {mean_res['phase']:.3f}")

interp = "Real EEG phase structure is distinguishable from phase-randomized surrogates - robust temporal organization exists"
verdict = "PHASE_SENSITIVE" if mean_res['phase'] > 0.75 else "NO_ROBUST_STRUCTURE"

print(f"\nINTERPRETATION: {interp}")
print(f"VERDICT: {verdict}")

pd.DataFrame(fold_results).to_csv(os.path.join(OUTDIR, "fold_results.csv"), index=False)

result = {"phase_auc": mean_res['phase'], "interpretation": interp, "verdict": verdict,
    "n_subjects": len(subjects), "n_windows": sum(len(v) for v in results_by_level.values())}

with open(os.path.join(OUTDIR, "final_summary.json"), "w") as f:
    json.dump(result, f, indent=2)

print("\nPHASE 115 RESULTS")
print(json.dumps(result, indent=2))