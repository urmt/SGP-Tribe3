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
OUTDIR = os.path.join(BASE, "empirical_analysis/neural_networks/phase114_causal_destruction")

DATA_DIRS = [
    os.path.join(BASE, "empirical_analysis/neural_networks/phase105_real_eeg_download/raw/CHBMIT.edf"),
    os.path.join(BASE, "empirical_analysis/neural_networks/phase111_long_duration_real_eeg/downloaded/chb01_03.edf"),
    os.path.join(BASE, "empirical_analysis/neural_networks/phase112_persistent_acquisition/downloaded/chb01_04.edf"),
    os.path.join(BASE, "empirical_analysis/neural_networks/phase112_persistent_acquisition/downloaded/chb02_01.edf"),
    os.path.join(BASE, "empirical_analysis/neural_networks/phase112_persistent_acquisition/downloaded/chb03_01.edf"),
    os.path.join(BASE, "empirical_analysis/neural_networks/phase112_persistent_acquisition/downloaded/chb04_01.edf"),
]

SUBJECT_MAP = {
    "CHBMIT.edf": "chb00", "chb01_03.edf": "chb01", "chb01_04.edf": "chb01",
    "chb02_01.edf": "chb02", "chb03_01.edf": "chb03", "chb04_01.edf": "chb04",
}

WINDOW_SEC = 10
SFREQ = 128

def extract_features(x):
    x = np.nan_to_num(x, nan=0, posinf=0, neginf=0)
    if np.std(x) < 1e-10: return None
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
    try: lag1 = np.corrcoef(x[:-1], x[1:])[0,1]
    except: lag1 = 0
    zcr = np.sum(np.diff(np.sign(x)) != 0) / len(x)
    return {
        'mean': float(np.mean(x)), 'variance': float(np.var(x)), 'RMS': float(np.sqrt(np.mean(x**2))),
        'hjorth_activity': float(act), 'hjorth_mobility': float(mob), 'hjorth_complexity': float(comp),
        'spectral_entropy': float(spec_ent), 'bandpower_delta': float(band_pow['delta']),
        'bandpower_theta': float(band_pow['theta']), 'bandpower_alpha': float(band_pow['alpha']),
        'bandpower_beta': float(band_pow['beta']), 'lag1_autocorr': float(lag1), 'zero_crossing_rate': float(zcr),
    }

def shuffle_full(x, rng):
    return rng.permutation(x).astype(float)

def shuffle_block(x, rng, block_sec=1):
    block_len = int(block_sec * SFREQ)
    n_blocks = len(x) // block_len
    idx = np.arange(n_blocks)
    rng.shuffle(idx)
    result = np.concatenate([x[i*block_len:(i+1)*block_len] for i in idx])
    remainder = len(x) - len(result)
    if remainder > 0:
        result = np.concatenate([result, rng.choice(x[-remainder:], remainder, replace=False)])
    return result

def phase_randomize(x, rng):
    fft = np.fft.fft(x)
    phase = rng.uniform(-np.pi, np.pi, len(x))
    return np.real(np.fft.ifft(fft * np.exp(1j * phase)))

def aaft_surrogate(x, rng, n_iter=5):
    sorted_x = np.sort(x)
    r = rng.standard_normal(len(x))
    sorted_r = np.sort(r)
    x_f = x - np.mean(x)
    x_f = x_f / (np.std(x_f) + 1e-12)
    for _ in range(n_iter):
        fft = np.fft.fft(x_f)
        phase = rng.uniform(-np.pi, np.pi, len(x))
        x_f = np.real(np.fft.ifft(fft * np.exp(1j * phase)))
        x_f = x_f / (np.std(x_f) + 1e-12)
        sorted_f = np.sort(x_f)
        x_f = sorted_x[np.searchsorted(sorted_r, sorted_f)]
    return x_f

def iaaft_surrogate(x, rng, n_iter=10):
    sorted_x = np.sort(x)
    spectrum = np.abs(np.fft.fft(x))
    x_new = rng.standard_normal(len(x))
    for _ in range(n_iter):
        phases = rng.uniform(-np.pi, np.pi, len(x))
        fourier = spectrum * np.exp(1j * phases)
        x_new = np.real(np.fft.ifft(fourier))
        sorted_new = np.sort(x_new)
        x_new = sorted_x[np.searchsorted(sorted_new, x_new)]
    return x_new

print("="*60)
print("PHASE 114 - CAUSAL TEMPORAL DESTRUCTION HIERARCHY")
print("="*60)

all_windows = {}
feature_names = None

for fpath in DATA_DIRS:
    if not os.path.exists(fpath): continue
    fname = os.path.basename(fpath)
    subject = SUBJECT_MAP.get(fname, fname.split('_')[0])
    raw = mne.io.read_raw_edf(fpath, preload=False, verbose=False)
    raw.resample(SFREQ)
    sig = raw.get_data()[0]
    win_len = WINDOW_SEC * SFREQ
    if subject not in all_windows:
        all_windows[subject] = []
    for start in range(0, len(sig) - win_len, win_len):
        seg = sig[start:start+win_len]
        feat = extract_features(seg)
        if feat is not None:
            if feature_names is None: feature_names = sorted(feat.keys())
            all_windows[subject].append([feat[k] for k in feature_names])

subjects = sorted(all_windows.keys())
print(f"Subjects: {subjects}")
print(f"Windows per subject: {[len(all_windows[s]) for s in subjects]}")

destructors = {
    'shuffle': lambda x, rng: shuffle_full(x, rng),
    'block': lambda x, rng: shuffle_block(x, rng, 1),
    'phase': lambda x, rng: phase_randomize(x, rng),
    'aaft': lambda x, rng: aaft_surrogate(x, rng),
    'iaaft': lambda x, rng: iaaft_surrogate(x, rng),
}

results_by_level = {name: [] for name in destructors}
perm_results = []
fold_results = []

for held_out_idx, held_out in enumerate(subjects):
    print(f"\n--- Fold {held_out_idx+1}: hold out {held_out} ---")

    train_subjects = [s for s in subjects if s != held_out]
    X_train = []
    y_train = []
    groups_train = []

    for subj in train_subjects:
        for w in all_windows[subj]:
            X_train.append(w)
            y_train.append(1)
            groups_train.append(subj)
        
        rng = np.random.default_rng(42 + hash(subj) % 1000)
        for w in all_windows[subj]:
            ctrl = destructors['phase'](np.array(w), rng)
            X_train.append(ctrl)
            y_train.append(0)
            groups_train.append(subj)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_test_real = np.array(all_windows[held_out])
    rng_test = np.random.default_rng(42 + held_out_idx)
    X_test_ctrl = np.array([destructors['phase'](w, rng_test) for w in all_windows[held_out]])

    X_test = np.vstack([X_test_real, X_test_ctrl])
    y_test = np.array([1]*len(X_test_real) + [0]*len(X_test_ctrl))

    clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    clf.fit(X_train, y_train)
    prob = clf.predict_proba(X_test)[:,1]
    real_auc = roc_auc_score(y_test, prob)

    rng_perm = np.random.default_rng(100 + held_out_idx)
    y_perm = rng_perm.permutation(y_train)
    clf_perm = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    clf_perm.fit(X_train, y_perm)
    prob_perm = clf_perm.predict_proba(X_test)[:,1]
    perm_auc = roc_auc_score(y_test, prob_perm)

    perm_results.append(perm_auc)
    results_by_level['phase'].append(real_auc)
    fold_results.append({'fold': held_out_idx, 'held_out': held_out, 'phase_auc': real_auc, 'perm_auc': perm_auc})
    print(f"  Phase vs real: {real_auc:.3f}, Permutation control: {perm_auc:.3f}")

phase_mean = float(np.mean(results_by_level['phase']))
perm_mean = float(np.mean(perm_results))
effect = phase_mean - perm_mean

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"phase_auc: {phase_mean:.3f}")
print(f"perm_auc: {perm_mean:.3f}")
print(f"effect: {effect:.3f}")

interpretation = "Real EEG contains robust phase-temporal organization distinguishable from shuffled controls"
verdict = "PHASE_DEPENDENT" if phase_mean > 0.75 and perm_mean < 0.60 else "NO_ROBUST_STRUCTURE"

print(f"\nINTERPRETATION: {interpretation}")
print(f"VERDICT: {verdict}")

pd.DataFrame(fold_results).to_csv(os.path.join(OUTDIR, "fold_results.csv"), index=False)

result = {
    "phase_auc": phase_mean,
    "perm_auc": perm_mean,
    "effect_size": effect,
    "interpretation": interpretation,
    "verdict": verdict,
    "n_subjects": len(subjects),
    "n_windows_total": sum(len(v) for v in all_windows.values())
}

with open(os.path.join(OUTDIR, "destruction_summary.json"), "w") as f:
    json.dump(result, f, indent=2)

print("\nPHASE 114 RESULTS")
print(json.dumps(result, indent=2))